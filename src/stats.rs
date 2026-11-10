//! Diagnostics behind the per-request timings: where a prefill's wall time
//! went, how the paged n-gram table's gathers fared, and what the system's
//! virtual memory did meanwhile.
//!
//! The engine counters are cumulative and per thread: the code that does the
//! work adds to them ([`record`]), and the server reads them before and after
//! a phase and subtracts ([`Counters::since`]). Only the engine thread runs a
//! request, so its counters are exactly the request's, with no locking and no
//! plumbing through the model API. Updating them costs a thread-local read
//! and write.
//!
//! The system side ([`vm_counters`], [`pressure_level`], [`task_memory`]) is
//! a handful of Mach and sysctl calls, a few microseconds each, sampled at
//! phase boundaries only. The dashboard's machine figures ([`cpu_ticks`],
//! [`memory_use`], [`swap_used`]) are more of the same, read on the HTTP
//! thread that answers `/v1/lily/status` and on the dashboard recorder's
//! thread, which samples them (with [`pressure_level`] and [`task_memory`])
//! for the 7-day history (`serve::recorder`). They are declared here rather than
//! taken from the `libc` crate, whose Mach bindings are deprecated.
//! [`own_energy_nj`] is lily's own CPU energy, which the machine sampler
//! (`serve::machine`) turns into lily's CPU power; it needs a newer
//! `rusage_info` revision than `libc` declares.
//!
//! On Linux the same figures come from `/proc` (`procfs` below): the paging
//! counters from `/proc/vmstat`, the pressure level from the kernel's
//! pressure stall information, the machine's memory and swap from
//! `/proc/meminfo`, the CPU ticks from `/proc/stat`, the process's footprint
//! from `/proc/self/status` and the lock limit from `RLIMIT_MEMLOCK`. Linux
//! keeps no per-process energy account, so [`own_energy_nj`] is `None` there.

use std::cell::Cell;

/// What the paged n-gram table's gathers did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gather {
    /// Staged batches (one per decode step, verify step or prefill chunk).
    pub batches: u64,
    pub rows: u64,
    /// Rows whose pages were checked with `mincore` right before the copy:
    /// all of them in decode and verify batches, every 16th in prefill
    /// batches, where checking every page would cost more than the copy.
    pub checked_rows: u64,
    /// Pages the checked rows' bytes lie on, counted per row (a row spans
    /// three to six pages across its codes, scales and biases; a page two
    /// rows share counts twice).
    pub pages: u64,
    /// Of those, the ones that were not resident: each is a read from the
    /// SSD that the copy then waits for.
    pub cold_pages: u64,
    /// Prefill rows outside the 1-in-16 sample whose pages were checked and
    /// hinted as well, because the sample found their batch cold (at least
    /// 1 page in 32); their pages are not in `pages` or `cold_pages`.
    pub hinted_rows: u64,
    /// Wall time inside `PagedTable::gather`, residency checks included.
    pub secs: f64,
    /// Of the prefill's staging time (hashing plus the gathers), the part
    /// that ran while the GPU was still busy with the previous chunk, so it
    /// cost the prefill nothing; the rest is `Prefill::ngram_secs`.
    pub hidden_secs: f64,
}

/// Where the prefill loop's time went, summed over its chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prefill {
    pub chunks: u64,
    /// Growing the state's caches and the prefill scratch (allocation and
    /// zeroing, plus the copy of the old caches when the state grows).
    pub alloc_secs: f64,
    /// Hashing the chunks' n-gram ids and staging their rows, as far as the
    /// GPU waited for it: the first chunk's whole staging, and whatever of a
    /// later chunk's (staged while the GPU ran the chunk before it) outlasted
    /// that chunk. The overlapped rest is `Gather::hidden_secs`.
    pub ngram_secs: f64,
    /// Encoding the chunk's pass on the host.
    pub encode_secs: f64,
    /// The passes' GPU execution, from the commit feedback's timestamps.
    pub gpu_secs: f64,
    /// Commit to completion minus the GPU span: submission latency, the GPU
    /// busy with other work, residency being established. A pass without
    /// usable timestamps counts here whole.
    pub wait_secs: f64,
}

/// The engine thread's cumulative counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Counters {
    pub gather: Gather,
    pub prefill: Prefill,
}

const ZERO: Counters = Counters {
    gather: Gather {
        batches: 0,
        rows: 0,
        checked_rows: 0,
        pages: 0,
        cold_pages: 0,
        hinted_rows: 0,
        secs: 0.0,
        hidden_secs: 0.0,
    },
    prefill: Prefill {
        chunks: 0,
        alloc_secs: 0.0,
        ngram_secs: 0.0,
        encode_secs: 0.0,
        gpu_secs: 0.0,
        wait_secs: 0.0,
    },
};

thread_local! {
    static COUNTERS: Cell<Counters> = const { Cell::new(ZERO) };
}

/// This thread's counters so far.
pub fn counters() -> Counters {
    COUNTERS.with(Cell::get)
}

/// Adds to this thread's counters.
pub fn record(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|cell| {
        let mut counters = cell.get();
        f(&mut counters);
        cell.set(counters);
    });
}

impl Counters {
    /// What was added since `earlier` (an earlier [`counters`] of the same
    /// thread).
    pub fn since(self, earlier: Counters) -> Counters {
        let (g, e) = (self.gather, earlier.gather);
        let (p, q) = (self.prefill, earlier.prefill);
        Counters {
            gather: Gather {
                batches: g.batches.saturating_sub(e.batches),
                rows: g.rows.saturating_sub(e.rows),
                checked_rows: g.checked_rows.saturating_sub(e.checked_rows),
                pages: g.pages.saturating_sub(e.pages),
                cold_pages: g.cold_pages.saturating_sub(e.cold_pages),
                hinted_rows: g.hinted_rows.saturating_sub(e.hinted_rows),
                secs: (g.secs - e.secs).max(0.0),
                hidden_secs: (g.hidden_secs - e.hidden_secs).max(0.0),
            },
            prefill: Prefill {
                chunks: p.chunks.saturating_sub(q.chunks),
                alloc_secs: (p.alloc_secs - q.alloc_secs).max(0.0),
                ngram_secs: (p.ngram_secs - q.ngram_secs).max(0.0),
                encode_secs: (p.encode_secs - q.encode_secs).max(0.0),
                gpu_secs: (p.gpu_secs - q.gpu_secs).max(0.0),
                wait_secs: (p.wait_secs - q.wait_secs).max(0.0),
            },
        }
    }
}

/// System-wide paging counters (`host_statistics64`, `HOST_VM_INFO64`), in
/// pages of the kernel's page size (16 KB on Apple Silicon). Cumulative since
/// boot; the difference of two samples is what happened in between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct VmCounters {
    /// Pages read in from files (a cold n-gram row is one of these).
    pub pageins: u64,
    /// Dirty file pages written out to make room.
    pub pageouts: u64,
    /// Compressed pages read back from the swap files.
    pub swapins: u64,
    /// Compressed pages written to the swap files.
    pub swapouts: u64,
    /// Pages the compressor took in to free memory.
    pub compressions: u64,
    /// Compressed pages that were touched again and expanded.
    pub decompressions: u64,
}

impl VmCounters {
    pub fn since(self, earlier: VmCounters) -> VmCounters {
        VmCounters {
            pageins: self.pageins.saturating_sub(earlier.pageins),
            pageouts: self.pageouts.saturating_sub(earlier.pageouts),
            swapins: self.swapins.saturating_sub(earlier.swapins),
            swapouts: self.swapouts.saturating_sub(earlier.swapouts),
            compressions: self.compressions.saturating_sub(earlier.compressions),
            decompressions: self.decompressions.saturating_sub(earlier.decompressions),
        }
    }
}

/// The system's paging counters now, `None` when the call fails.
#[cfg(target_os = "macos")]
pub fn vm_counters() -> Option<VmCounters> {
    let mut info = sys::VmStatistics64::default();
    let mut count = sys::VmStatistics64::COUNT;
    // SAFETY: `info` is a correctly laid out `vm_statistics64` of `count`
    // 32-bit words; the kernel writes at most that many.
    let rc = unsafe {
        sys::host_statistics64(
            sys::host(),
            sys::HOST_VM_INFO64,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    (rc == 0).then_some(VmCounters {
        pageins: info.pageins,
        pageouts: info.pageouts,
        swapins: info.swapins,
        swapouts: info.swapouts,
        compressions: info.compressions,
        decompressions: info.decompressions,
    })
}

/// The memorystatus pressure level (`kern.memorystatus_vm_pressure_level`):
/// 1 normal, 2 warning, 4 critical. `None` when the sysctl fails.
#[cfg(target_os = "macos")]
pub fn pressure_level() -> Option<u32> {
    let mut level: std::ffi::c_int = 0;
    let mut len = std::mem::size_of::<std::ffi::c_int>();
    // SAFETY: a NUL-terminated name and an int-sized output buffer.
    let rc = unsafe {
        sys::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && level >= 0).then_some(level as u32)
}

/// How much memory this process may lock with `mlock`: the smaller of
/// `vm.user_wire_limit` (per process) and `vm.global_user_wire_limit` (every
/// process's user wirings plus what the system has wired already, e.g.
/// 116.8 GB each on a 128 GB machine). `None` when neither can be read.
#[cfg(target_os = "macos")]
pub fn user_wire_limit() -> Option<u64> {
    let per_process = sysctl_u64(c"vm.user_wire_limit");
    let global = sysctl_u64(c"vm.global_user_wire_limit");
    match (per_process, global) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// A 32- or 64-bit unsigned sysctl, `None` when it cannot be read.
#[cfg(target_os = "macos")]
fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: a NUL-terminated name and an 8-byte output buffer of `len`.
    let rc = unsafe {
        sys::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (rc, len) {
        (0, 8) => Some(value),
        // A 32-bit value lands in the low half (little-endian).
        (0, 4) => Some(value & 0xffff_ffff),
        _ => None,
    }
}

/// The host's CPU time so far, in scheduler ticks summed over every core
/// (`host_statistics`, `HOST_CPU_LOAD_INFO`). Cumulative since boot; two
/// samples give the load between them ([`CpuTicks::load_since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuTicks {
    pub user: u64,
    pub system: u64,
    pub idle: u64,
    pub nice: u64,
}

impl CpuTicks {
    /// The share of the ticks since `earlier` that were not idle, in
    /// `0.0..=1.0`; `None` when no tick passed (the counters are 32-bit in
    /// the kernel and wrap, which reads as no time passing as well).
    pub fn load_since(self, earlier: CpuTicks) -> Option<f64> {
        let busy = (self.user + self.system + self.nice)
            .checked_sub(earlier.user + earlier.system + earlier.nice)?;
        let idle = self.idle.checked_sub(earlier.idle)?;
        let total = busy + idle;
        (total > 0).then(|| busy as f64 / total as f64)
    }
}

/// The host's CPU ticks now, `None` when the call fails.
#[cfg(target_os = "macos")]
pub fn cpu_ticks() -> Option<CpuTicks> {
    let mut ticks = [0u32; 4];
    let mut count = 4u32;
    // SAFETY: `ticks` is `host_cpu_load_info` (four `natural_t`), `count`
    // words long; the kernel writes at most that many.
    let rc = unsafe {
        sys::host_statistics(
            sys::host(),
            sys::HOST_CPU_LOAD_INFO,
            ticks.as_mut_ptr().cast(),
            &mut count,
        )
    };
    // `CPU_STATE_USER`, `_SYSTEM`, `_IDLE`, `_NICE`.
    (rc == 0 && count == 4).then(|| CpuTicks {
        user: ticks[0].into(),
        system: ticks[1].into(),
        idle: ticks[2].into(),
        nice: ticks[3].into(),
    })
}

/// The machine's physical memory and how much of it is used, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUse {
    /// `hw.memsize`.
    pub total: u64,
    /// What Activity Monitor calls "Memory Used": app memory (anonymous
    /// pages that are not purgeable), wired memory and the compressor's
    /// pages. The page cache is not in it.
    pub used: u64,
}

/// The machine's memory use now, `None` when a call fails.
#[cfg(target_os = "macos")]
pub fn memory_use() -> Option<MemoryUse> {
    let total = sysctl_u64(c"hw.memsize")?;
    let page = sysctl_u64(c"hw.pagesize")?;
    let mut info = sys::VmStatistics64::default();
    let mut count = sys::VmStatistics64::COUNT;
    // SAFETY: as in `vm_counters`.
    let rc = unsafe {
        sys::host_statistics64(
            sys::host(),
            sys::HOST_VM_INFO64,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    (rc == 0).then(|| {
        let app = u64::from(info.internal_page_count)
            .saturating_sub(info.purgeable_count.into());
        let pages =
            app + u64::from(info.wire_count) + u64::from(info.compressor_page_count);
        MemoryUse { total, used: (pages * page).min(total) }
    })
}

/// Bytes of swap in use (`vm.swapusage`), `None` when the sysctl fails.
#[cfg(target_os = "macos")]
pub fn swap_used() -> Option<u64> {
    /// `struct xsw_usage`.
    #[repr(C)]
    #[derive(Default)]
    struct XswUsage {
        total: u64,
        avail: u64,
        used: u64,
        pagesize: u32,
        encrypted: u32,
    }
    let mut usage = XswUsage::default();
    let mut len = std::mem::size_of::<XswUsage>();
    // SAFETY: a NUL-terminated name and an output buffer of `len` bytes laid
    // out as the kernel's `xsw_usage`.
    let rc = unsafe {
        sys::sysctlbyname(
            c"vm.swapusage".as_ptr(),
            (&raw mut usage).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<XswUsage>()).then_some(usage.used)
}

/// The name of a [`pressure_level`].
pub fn pressure_name(level: u32) -> &'static str {
    match level {
        1 => "normal",
        2 => "warning",
        4 => "critical",
        _ => "unknown",
    }
}

/// This process's memory as the system accounts it (`task_info`,
/// `TASK_VM_INFO`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskMemory {
    /// What memory pressure and the jetsam limits charge the process with
    /// (Activity Monitor's "Memory"): its anonymous memory, its GPU buffers
    /// and its wired pages, not the page cache it maps.
    pub phys_footprint: u64,
    /// Bytes of the process's memory currently held by the compressor.
    pub compressed: u64,
}

/// This process's footprint now, `None` when the call fails.
#[cfg(target_os = "macos")]
pub fn task_memory() -> Option<TaskMemory> {
    let mut info = sys::TaskVmInfo::default();
    let mut count = sys::TaskVmInfo::COUNT;
    // SAFETY: `info` is the revision-1 `task_vm_info` layout, `count` words
    // long; the kernel fills at most that many.
    let rc = unsafe {
        sys::task_info(
            sys::mach_task_self_,
            sys::TASK_VM_INFO,
            (&raw mut info).cast(),
            &mut count,
        )
    };
    (rc == 0 && count >= sys::TaskVmInfo::COUNT).then_some(TaskMemory {
        phys_footprint: info.phys_footprint,
        compressed: info.compressed,
    })
}

/// The CPU energy this process has used since it started, in nanojoules
/// (`proc_pid_rusage`, `RUSAGE_INFO_V6`, `ri_energy_nj`): the kernel's
/// estimate summed over the efficiency and the performance cores (the field
/// next to it, `ri_penergy_nj`, is the performance cores' share). `None` when
/// the call fails. Two readings give lily's CPU power between them; the
/// dashboard labels it an estimate (docs/dashboard.md § 6).
#[cfg(target_os = "macos")]
pub fn own_energy_nj() -> Option<u64> {
    let mut info = sys::RusageInfoV6::default();
    // SAFETY: `info` is laid out as the SDK's `struct rusage_info_v6` (its
    // size is asserted below), the flavor asks for exactly that revision, and
    // `rusage_info_t` is a `void *` the kernel writes that struct through.
    let rc = unsafe {
        sys::proc_pid_rusage(
            std::process::id() as i32,
            sys::RUSAGE_INFO_V6,
            (&raw mut info).cast(),
        )
    };
    (rc == 0).then_some(info.ri_energy_nj)
}

/// The Mach and sysctl calls above. Layouts are the macOS SDK's
/// `vm_statistics64` and `task_vm_info` (checked against the headers with
/// `offsetof`: `pageins` at 32, `swapouts` at 120; `compressed` at 120,
/// `phys_footprint` at 144).
#[cfg(target_os = "macos")]
mod sys {
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    pub const HOST_CPU_LOAD_INFO: c_int = 3;
    /// `RUSAGE_INFO_V6` (`sys/resource.h`).
    pub const RUSAGE_INFO_V6: c_int = 6;
    pub const HOST_VM_INFO64: c_int = 4;
    pub const TASK_VM_INFO: c_int = 22;

    #[repr(C)]
    #[derive(Default)]
    pub struct VmStatistics64 {
        pub free_count: u32,
        pub active_count: u32,
        pub inactive_count: u32,
        pub wire_count: u32,
        pub zero_fill_count: u64,
        pub reactivations: u64,
        pub pageins: u64,
        pub pageouts: u64,
        pub faults: u64,
        pub cow_faults: u64,
        pub lookups: u64,
        pub hits: u64,
        pub purges: u64,
        pub purgeable_count: u32,
        pub speculative_count: u32,
        pub decompressions: u64,
        pub compressions: u64,
        pub swapins: u64,
        pub swapouts: u64,
        pub compressor_page_count: u32,
        pub throttled_count: u32,
        pub external_page_count: u32,
        pub internal_page_count: u32,
        pub total_uncompressed_pages_in_compressor: u64,
    }

    impl VmStatistics64 {
        /// `HOST_VM_INFO64_COUNT` of this revision: 152 bytes in 32-bit words.
        pub const COUNT: u32 = (std::mem::size_of::<Self>() / 4) as u32;
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct TaskVmInfo {
        pub virtual_size: u64,
        pub region_count: i32,
        pub page_size: i32,
        pub resident_size: u64,
        pub resident_size_peak: u64,
        pub device: u64,
        pub device_peak: u64,
        pub internal: u64,
        pub internal_peak: u64,
        pub external: u64,
        pub external_peak: u64,
        pub reusable: u64,
        pub reusable_peak: u64,
        pub purgeable_volatile_pmap: u64,
        pub purgeable_volatile_resident: u64,
        pub purgeable_volatile_virtual: u64,
        pub compressed: u64,
        pub compressed_peak: u64,
        pub compressed_lifetime: u64,
        pub phys_footprint: u64,
    }

    impl TaskVmInfo {
        /// `TASK_VM_INFO_REV1_COUNT`: through `phys_footprint`, 152 bytes.
        pub const COUNT: u32 = (std::mem::size_of::<Self>() / 4) as u32;
    }

    /// `struct rusage_info_v6` of the macOS SDK's `sys/resource.h`, field for
    /// field (`libc` 0.2.190 stops at `rusage_info_v4`): 16 bytes of UUID, 50
    /// counters and 6 reserved words, 464 bytes. Only `ri_energy_nj` is read.
    #[repr(C)]
    #[derive(Default)]
    pub struct RusageInfoV6 {
        pub ri_uuid: [u8; 16],
        pub ri_user_time: u64,
        pub ri_system_time: u64,
        pub ri_pkg_idle_wkups: u64,
        pub ri_interrupt_wkups: u64,
        pub ri_pageins: u64,
        pub ri_wired_size: u64,
        pub ri_resident_size: u64,
        pub ri_phys_footprint: u64,
        pub ri_proc_start_abstime: u64,
        pub ri_proc_exit_abstime: u64,
        pub ri_child_user_time: u64,
        pub ri_child_system_time: u64,
        pub ri_child_pkg_idle_wkups: u64,
        pub ri_child_interrupt_wkups: u64,
        pub ri_child_pageins: u64,
        pub ri_child_elapsed_abstime: u64,
        pub ri_diskio_bytesread: u64,
        pub ri_diskio_byteswritten: u64,
        pub ri_cpu_time_qos_default: u64,
        pub ri_cpu_time_qos_maintenance: u64,
        pub ri_cpu_time_qos_background: u64,
        pub ri_cpu_time_qos_utility: u64,
        pub ri_cpu_time_qos_legacy: u64,
        pub ri_cpu_time_qos_user_initiated: u64,
        pub ri_cpu_time_qos_user_interactive: u64,
        pub ri_billed_system_time: u64,
        pub ri_serviced_system_time: u64,
        pub ri_logical_writes: u64,
        pub ri_lifetime_max_phys_footprint: u64,
        pub ri_instructions: u64,
        pub ri_cycles: u64,
        pub ri_billed_energy: u64,
        pub ri_serviced_energy: u64,
        pub ri_interval_max_phys_footprint: u64,
        pub ri_runnable_time: u64,
        pub ri_flags: u64,
        pub ri_user_ptime: u64,
        pub ri_system_ptime: u64,
        pub ri_pinstructions: u64,
        pub ri_pcycles: u64,
        pub ri_energy_nj: u64,
        pub ri_penergy_nj: u64,
        pub ri_secure_time_in_system: u64,
        pub ri_secure_ptime_in_system: u64,
        pub ri_neural_footprint: u64,
        pub ri_lifetime_max_neural_footprint: u64,
        pub ri_interval_max_neural_footprint: u64,
        pub ri_conclave_footprint: u64,
        pub ri_page_wait_time_mach: u64,
        pub ri_page_cache_hits: u64,
        pub ri_reserved: [u64; 6],
    }

    const _: () = assert!(std::mem::size_of::<RusageInfoV6>() == 464);

    unsafe extern "C" {
        pub static mach_task_self_: u32;
        fn mach_host_self() -> u32;
        pub fn host_statistics64(
            host: u32,
            flavor: c_int,
            info: *mut c_int,
            count: *mut u32,
        ) -> c_int;
        pub fn host_statistics(
            host: u32,
            flavor: c_int,
            info: *mut c_int,
            count: *mut u32,
        ) -> c_int;
        pub fn task_info(
            task: u32,
            flavor: c_int,
            info: *mut c_int,
            count: *mut u32,
        ) -> c_int;
        /// `libproc.h`; `buffer` is a `rusage_info_t *`, which is `void **`
        /// in the header but is written through as the struct itself.
        pub fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void)
        -> c_int;
        pub fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// The host port, taken once: every `mach_host_self` call adds a send
    /// right to it.
    pub fn host() -> u32 {
        static HOST: OnceLock<u32> = OnceLock::new();
        // SAFETY: no preconditions.
        *HOST.get_or_init(|| unsafe { mach_host_self() })
    }
}

#[cfg(target_os = "linux")]
pub use procfs::{
    cpu_ticks, memory_use, own_energy_nj, pressure_level, swap_used, task_memory,
    user_wire_limit, vm_counters,
};

/// The Linux sources of the figures above, read from `/proc`. Each reading
/// is a read of one small text file, tens of microseconds, at the same
/// phase boundaries and status requests as on macOS. The figures keep the
/// meanings the callers were written for where Linux has the same thing, and
/// say below where they are the nearest Linux has.
#[cfg(target_os = "linux")]
mod procfs {
    use super::{CpuTicks, MemoryUse, TaskMemory, VmCounters};

    /// The pressure stall share (percent of the last 10 s) from which
    /// [`pressure_level`] reports warning: some task waited for memory a
    /// tenth of the time.
    pub(super) const PSI_WARNING_PCT: f64 = 10.0;
    /// ... and from which it reports critical, read from the `full` line:
    /// every runnable task waited for memory at once a tenth of the time.
    pub(super) const PSI_CRITICAL_PCT: f64 = 10.0;
    /// `CAP_IPC_LOCK` (`linux/capability.h`): `mlock` without the
    /// `RLIMIT_MEMLOCK` limit.
    const CAP_IPC_LOCK: u32 = 14;

    /// The values of `keys` in a `/proc` file of `name value [kB]` lines
    /// (`/proc/vmstat`, `/proc/meminfo`, `/proc/self/status`; a colon after
    /// the name is ignored), in bytes where the line gives `kB`; `None` for a
    /// key the text does not have.
    pub(super) fn proc_values<const N: usize>(
        text: &str,
        keys: [&str; N],
    ) -> [Option<u64>; N] {
        let mut out = [None; N];
        for line in text.lines() {
            let mut words = line.split_whitespace();
            let (Some(name), Some(value)) = (words.next(), words.next()) else {
                continue;
            };
            let name = name.strip_suffix(':').unwrap_or(name);
            if let Some(i) = keys.iter().position(|&k| k == name) {
                let scale = if words.next() == Some("kB") { 1024 } else { 1 };
                out[i] = value.parse::<u64>().ok().and_then(|v| v.checked_mul(scale));
            }
        }
        out
    }

    /// [`proc_values`] of the file at `path`, `None` when it cannot be read.
    fn read_values<const N: usize>(
        path: &str,
        keys: [&str; N],
    ) -> Option<[Option<u64>; N]> {
        std::fs::read_to_string(path).ok().map(|text| proc_values(&text, keys))
    }

    fn page_size() -> Option<u64> {
        // SAFETY: sysconf has no preconditions.
        u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|&p| p > 0)
    }

    /// The system's paging counters from `/proc/vmstat`, in pages of the
    /// kernel's page size. `pageins` is `pgpgin` (kilobytes read from block
    /// devices, files and swap alike, converted to pages): Linux counts
    /// reads, not file page-ins. `pageouts` is `nr_vmscan_write`, the pages
    /// reclaim wrote out to make room (to files or to swap); plain write-back
    /// of written files is not in it. Swap is `pswpin`/`pswpout`. The
    /// compressor is zswap: `zswpout` pages compressed into it, `zswpin`
    /// pages read back out (0 on a kernel without zswap or swap).
    pub fn vm_counters() -> Option<VmCounters> {
        let [pgpgin, vmscan_write, pswpin, pswpout, zswpout, zswpin] = read_values(
            "/proc/vmstat",
            ["pgpgin", "nr_vmscan_write", "pswpin", "pswpout", "zswpout", "zswpin"],
        )?;
        let page = page_size()?;
        Some(VmCounters {
            pageins: pgpgin?.saturating_mul(1024) / page,
            pageouts: vmscan_write?,
            swapins: pswpin.unwrap_or(0),
            swapouts: pswpout.unwrap_or(0),
            compressions: zswpout.unwrap_or(0),
            decompressions: zswpin.unwrap_or(0),
        })
    }

    /// The memory pressure level on macOS's scale (1 normal, 2 warning,
    /// 4 critical) from the kernel's pressure stall information
    /// (`/proc/pressure/memory`), which is what Linux has instead of a
    /// level: the share of the last 10 s that tasks waited for memory
    /// (reclaim, refaults, swap-ins). `None` without PSI (`CONFIG_PSI` off
    /// or `psi=0`).
    pub fn pressure_level() -> Option<u32> {
        psi_level(&std::fs::read_to_string("/proc/pressure/memory").ok()?)
    }

    /// [`pressure_level`] of the file's text: critical from
    /// [`PSI_CRITICAL_PCT`] on the `full` line, warning from
    /// [`PSI_WARNING_PCT`] on the `some` line, normal otherwise.
    pub(super) fn psi_level(text: &str) -> Option<u32> {
        let avg10 = |kind: &str| {
            text.lines()
                .find(|line| line.split_whitespace().next() == Some(kind))?
                .split_whitespace()
                .find_map(|field| field.strip_prefix("avg10="))?
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
        };
        let some = avg10("some")?;
        // The `full` line is missing on kernels before 5.13 for some
        // resources; for memory it has been there from the start, but a
        // missing one reads as no stall.
        let full = avg10("full").unwrap_or(0.0);
        Some(if full >= PSI_CRITICAL_PCT {
            4
        } else if some >= PSI_WARNING_PCT {
            2
        } else {
            1
        })
    }

    /// How much memory this process may lock with `mlock`: the soft
    /// `RLIMIT_MEMLOCK` (8 MB on a stock Arch system, raised with `ulimit -l`
    /// or `LimitMEMLOCK=` in a unit). Without a limit (`unlimited`, or
    /// `CAP_IPC_LOCK`, which lifts it) only the physical memory bounds it,
    /// and that is what is returned. `None` when neither can be read.
    pub fn user_wire_limit() -> Option<u64> {
        let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        // SAFETY: a valid out-parameter for a resource the kernel knows.
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) };
        let unlimited = rc == 0 && limit.rlim_cur == libc::RLIM_INFINITY;
        if rc != 0 || unlimited || has_ipc_lock() {
            return crate::qwen4exp::weights::physical_memory();
        }
        Some(limit.rlim_cur)
    }

    /// Whether this process holds `CAP_IPC_LOCK` (`CapEff` in
    /// `/proc/self/status`, a hexadecimal mask).
    fn has_ipc_lock() -> bool {
        let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
            return false;
        };
        text.lines()
            .find_map(|line| line.strip_prefix("CapEff:"))
            .and_then(|mask| u64::from_str_radix(mask.trim(), 16).ok())
            .is_some_and(|mask| mask & (1 << CAP_IPC_LOCK) != 0)
    }

    /// The host's CPU ticks (`USER_HZ`) from the `cpu` line of `/proc/stat`:
    /// `user` and `nice` as they are, `system` with the interrupt and
    /// steal time (busy, not idle), `idle` with the time waiting for I/O.
    pub fn cpu_ticks() -> Option<CpuTicks> {
        let text = std::fs::read_to_string("/proc/stat").ok()?;
        cpu_line(text.lines().next()?)
    }

    /// [`cpu_ticks`] of the `cpu` line. Fields after `steal` (guest time,
    /// already in `user` and `nice`) are not added.
    pub(super) fn cpu_line(line: &str) -> Option<CpuTicks> {
        let mut words = line.split_whitespace();
        (words.next()? == "cpu").then_some(())?;
        let mut v = [0u64; 8];
        for (slot, word) in v.iter_mut().zip(&mut words) {
            *slot = word.parse().ok()?;
        }
        let [user, nice, system, idle, iowait, irq, softirq, steal] = v;
        Some(CpuTicks {
            user,
            system: system + irq + softirq + steal,
            idle: idle + iowait,
            nice,
        })
    }

    /// The machine's memory from `/proc/meminfo`: `MemTotal`, and as used
    /// what the kernel cannot hand out without swapping, `MemTotal` minus
    /// `MemAvailable` (the page cache and reclaimable slabs are available;
    /// GPU buffers, which are pinned shared memory, are not).
    pub fn memory_use() -> Option<MemoryUse> {
        let [total, available] =
            read_values("/proc/meminfo", ["MemTotal", "MemAvailable"])?;
        let total = total.filter(|&t| t > 0)?;
        Some(MemoryUse { total, used: total.saturating_sub(available?) })
    }

    /// Bytes of swap in use, `SwapTotal` minus `SwapFree` (0 without swap).
    /// Pages zswap holds count: each keeps its swap slot.
    pub fn swap_used() -> Option<u64> {
        let [total, free] = read_values("/proc/meminfo", ["SwapTotal", "SwapFree"])?;
        Some(total?.saturating_sub(free?))
    }

    /// This process's memory from `/proc/self/status`. The footprint is its
    /// anonymous and shared-memory resident pages (`RssAnon` + `RssShmem`),
    /// not the file pages it maps (the n-gram table's page cache), as on
    /// macOS. The GPU buffers are in `RssShmem` only on a kernel whose GPU
    /// driver charges the pages it holds to the process that created them
    /// (the Asahi driver of the aurora kernels does); on another, the
    /// buffers' pages are mapped as device memory and this figure leaves them
    /// out. `compressed` is `VmSwap`: the process's pages in swap, zswap's
    /// compressed pool included, the nearest to macOS's compressor.
    pub fn task_memory() -> Option<TaskMemory> {
        let [anon, shmem, swap] =
            read_values("/proc/self/status", ["RssAnon", "RssShmem", "VmSwap"])?;
        Some(TaskMemory {
            phys_footprint: anon?.saturating_add(shmem.unwrap_or(0)),
            compressed: swap.unwrap_or(0),
        })
    }

    /// Linux keeps no per-process energy estimate (RAPL-style counters are
    /// per package, and this machine exposes none), so lily's CPU power is
    /// unknown: always `None`.
    pub fn own_energy_nj() -> Option<u64> {
        None
    }
}

#[cfg(test)]
#[path = "../tests/unit/stats.rs"]
mod tests;
