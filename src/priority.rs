//! Lowering a thread's priority on Linux, where macOS's quality-of-service
//! classes (`pthread_set_qos_class_self_np`) and per-thread disk I/O
//! policies (`setiopolicy_np`) do not exist. The nearest Linux has for each
//! is two per-thread settings: the nice value (`setpriority`, which under
//! NPTL is a property of the thread, not of the process) for the CPU, and
//! the I/O priority (`ioprio_set`) for the disk. Neither moves a thread to
//! the efficiency cores the way macOS's background class does; the
//! scheduler only gives it a smaller share wherever it runs. The I/O
//! priority is honoured by the BFQ and mq-deadline schedulers and ignored
//! under `none`, which NVMe drives often use: there the disk is not lowered.

use std::ffi::c_int;

/// What a lowered thread should yield to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// macOS's utility class with utility I/O: work that should finish in
    /// about its foreground time on an idle machine but yield to requests.
    /// Nice 10 and the lowest best-effort I/O priority.
    Utility,
    /// macOS's background class with throttled I/O: work that may take as
    /// long as it needs. Nice 19 and the idle I/O class, which gets the
    /// disk only when nothing else asks for it.
    Background,
}

/// `IOPRIO_WHO_PROCESS`, `IOPRIO_CLASS_SHIFT`, `IOPRIO_CLASS_BE` and
/// `IOPRIO_CLASS_IDLE` (`linux/ioprio.h`); the `libc` crate does not
/// declare them. With `IOPRIO_WHO_PROCESS` and `who` 0 the call applies to
/// the calling thread.
const IOPRIO_WHO_PROCESS: c_int = 1;
const IOPRIO_CLASS_SHIFT: c_int = 13;
const IOPRIO_CLASS_BE: c_int = 2;
const IOPRIO_CLASS_IDLE: c_int = 3;

/// Lowers the calling thread to `class`. Best effort: returns whether both
/// the nice value and the I/O priority took. Lowering either needs no
/// privilege; a thread that already runs nicer (lily started under `nice`)
/// keeps its nice value, since raising it back would need privilege and is
/// not what was asked.
pub fn lower_current_thread(class: Class) -> bool {
    let (nice, ioprio) = match class {
        Class::Utility => (10, (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | 7),
        Class::Background => (19, IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT),
    };
    // SAFETY: `who` 0 is the calling thread for all three calls (Linux
    // resolves it to the current task); they read or change only its own
    // scheduling. `getpriority` cannot fail for the calling thread.
    unsafe {
        let nice = nice.max(libc::getpriority(libc::PRIO_PROCESS, 0));
        let cpu = libc::setpriority(libc::PRIO_PROCESS, 0, nice) == 0;
        let io =
            libc::syscall(libc::SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0, ioprio) == 0;
        cpu && io
    }
}

#[cfg(test)]
#[path = "../tests/unit/priority.rs"]
mod tests;
