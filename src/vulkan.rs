//! Vulkan device context for Linux (Honeykrisp on Apple GPUs), with the API of
//! `metal.rs`: `lib.rs` mounts this file as `crate::metal` there, so the
//! kernels, the model and the server use one set of names on both systems.
//!
//! How the Metal 4 model of `metal.rs` maps to Vulkan:
//!
//! - **Kernels** are GLSL compute shaders in `src/kernels/vulkan/`, compiled
//!   to SPIR-V by `build.rs` and embedded, one per Metal kernel function and
//!   under its name ([`MetalContext::pipeline`] ignores the MSL source it is
//!   given). A dispatch's threadgroup size is the pipeline's local size, set
//!   through specialization constants 0 to 2, so there is one pipeline per
//!   kernel and threadgroup size.
//! - **Arguments** bind by GPU address, as on Metal 4: a dispatch's buffer
//!   and param addresses are a table in the pass's arena, whose address and
//!   the thread grid are the push constants (`common.glsl`).
//! - **Ordering.** Dispatches of one command buffer overlap unless a barrier
//!   separates them (Honeykrisp encodes a barrier in the compute stream), so
//!   serial passes place one after every dispatch and concurrent passes at
//!   their levels, as on Metal. Command buffers run in submission order.
//! - **Events** ([`SharedEvent`]) are host objects. One thread per context
//!   submits every pass's program in commit order: a wait blocks it until the
//!   host has signalled, so the queue never holds a wait, as a gated Metal
//!   pass's feeder does ([`ComputePass::gate_host_waits`] is how every pass
//!   is submitted here). A second thread watches the queue's completion
//!   counter (a timeline semaphore each submission signals) and raises the
//!   events a program signals, the fence among them.
//! - **Jobs.** The kernel driver gives a GPU job about two seconds from its
//!   submission, so a pass is submitted as command buffers of a bounded
//!   number of dispatches, a few in flight at a time.
//! - **GPU time** is the host's: a pass's span runs from its first
//!   submission, or the end of the pass before it, to the completion the
//!   watcher saw. The per-kernel profile submits and waits for each dispatch
//!   alone, which adds a job's fixed cost to each.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::CStr;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use ash::vk;
use core::ffi::c_void;
use core::ptr::NonNull;

use crate::tensor::Tensor;

include!(concat!(env!("OUT_DIR"), "/vulkan_kernels.rs"));

pub type Pipeline = vk::Pipeline;

/// A device buffer handle. Cloning shares the allocation, which is freed
/// when the last handle drops; the owner drops it only once every pass that
/// used it has completed. Like the context, a buffer stays on its thread.
pub type Buffer = Rc<GpuBuffer>;

pub struct Kernel {
    /// The kernel function's name (the pipeline's and the profile's key).
    pub name: &'static str,
}

impl Kernel {
    /// The subgroup size, 32 on every Apple GPU.
    pub fn thread_execution_width(&self) -> usize {
        32
    }

    /// Metal reports the threadgroup memory a pipeline reserves; SPIR-V
    /// pipelines do not, and nothing sizes anything by it.
    pub fn static_threadgroup_memory_length(&self) -> usize {
        0
    }
}

/// The MSL language version a kernel source requires. The shaders here are
/// built ahead of time, so every version names the same one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MslVersion {
    V3_1,
    V4_0,
    V4_1,
}

/// Largest `params` slice any dispatch passes.
const MAX_KERNEL_PARAMS: usize = 16;

/// Metal's buffer-argument limit, kept: the kernels' binding indices are
/// Metal's.
const MAX_BUFFER_BINDINGS: usize = 31;

/// Alignment of every inline param and address table in the arena.
const PARAM_ALIGN: usize = 16;

/// First arena chunk; a pass that outgrows it gets a larger one appended.
const ARENA_CHUNK: usize = 1 << 20;

/// The push constants of every kernel (`common.glsl`): the address of the
/// dispatch's address table and its thread grid.
const PUSH_BYTES: u32 = 32;

/// Dispatches per command buffer of a serial and of a concurrent pass
/// (`LILY_VK_BATCH` sets both), and command buffers in flight
/// (`LILY_VK_WINDOW`). A prefill dispatch can take milliseconds and a decode
/// one tens of microseconds; the kernel times a job from its submission.
const SERIAL_BATCH: usize = 32;
const CONCURRENT_BATCH: usize = 128;
const WINDOW: u64 = 3;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok().filter(|&v| v > 0)
}

// The per-kernel profile (`profile`) and the busy time (`busy`), shared with
// the Metal backend.
include!("gpu_stats.rs");

/// `VkPhysicalDeviceShaderBfloat16FeaturesKHR`, which the `ash` release in
/// use predates.
#[repr(C)]
struct Bfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}
const BFLOAT16_FEATURES: vk::StructureType = vk::StructureType::from_raw(1000141000);
const BFLOAT16_EXTENSION: &CStr = c"VK_KHR_shader_bfloat16";
const COOPERATIVE_MATRIX_EXTENSION: &CStr = c"VK_KHR_cooperative_matrix";

/// The device and what lives as long as it: shared by the context, its
/// buffers and its two threads, destroyed with the last of them.
struct Dev {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    layout: vk::PipelineLayout,
    /// Signalled to `n` by the `n`th command buffer submitted.
    completed: vk::Semaphore,
    memory_type: u32,
    heap_bytes: u64,
    modules: Mutex<HashMap<&'static str, vk::ShaderModule>>,
    pipelines: Mutex<HashMap<(&'static str, [u32; 3]), vk::Pipeline>>,
    allocated: AtomicUsize,
    allocations: AtomicUsize,
}

// SAFETY: Vulkan handles are plain values; the queue is used by the
// submitter thread alone and everything else here is internally
// synchronized by Vulkan or by the mutexes.
unsafe impl Send for Dev {}
unsafe impl Sync for Dev {}

impl Drop for Dev {
    fn drop(&mut self) {
        // SAFETY: the last owner; nothing uses the device any more.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(pipelines) = self.pipelines.lock() {
                for &pipeline in pipelines.values() {
                    self.device.destroy_pipeline(pipeline, None);
                }
            }
            if let Ok(modules) = self.modules.lock() {
                for &module in modules.values() {
                    self.device.destroy_shader_module(module, None);
                }
            }
            self.device.destroy_semaphore(self.completed, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Dev {
    fn new() -> Result<Arc<Self>> {
        // SAFETY: plain Vulkan calls on valid handles throughout.
        unsafe {
            let entry =
                ash::Entry::load().context("loading the Vulkan loader (libvulkan)")?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"lily")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .context("creating a Vulkan instance")?;
            let physical = *instance
                .enumerate_physical_devices()?
                .first()
                .ok_or_else(|| anyhow!("no Vulkan device found"))?;
            let extensions = instance.enumerate_device_extension_properties(physical)?;
            let has = |name: &CStr| {
                extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
            };
            ensure!(
                has(COOPERATIVE_MATRIX_EXTENSION) && has(BFLOAT16_EXTENSION),
                "lily needs VK_KHR_cooperative_matrix and VK_KHR_shader_bfloat16 \
                 (Honeykrisp on an M5 with ASAHI_M5_COOPERATIVE_MATRIX=1)"
            );
            let queue_family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| anyhow!("no compute queue"))?
                as u32;

            // Everything the device supports of the feature sets the shaders
            // draw on, enabled as queried.
            let mut bfloat = Bfloat16Features {
                s_type: BFLOAT16_FEATURES,
                p_next: std::ptr::null_mut(),
                shader_bfloat16_type: 0,
                shader_bfloat16_dot_product: 0,
                shader_bfloat16_cooperative_matrix: 0,
            };
            let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            cooperative.p_next = (&raw mut bfloat).cast();
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13)
                .push_next(&mut cooperative);
            instance.get_physical_device_features2(physical, &mut features);
            // Robust buffer access costs a bounds check per access and the
            // kernels address memory by pointer anyway.
            features.features.robust_buffer_access = vk::FALSE;
            let core = features.features;
            ensure!(
                core.shader_int64 == vk::TRUE && core.shader_int16 == vk::TRUE,
                "the Vulkan device lacks 64-bit or 16-bit integers"
            );

            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&[1.0])];
            let names = [COOPERATIVE_MATRIX_EXTENSION.as_ptr(), BFLOAT16_EXTENSION.as_ptr()];
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&names);
            info.p_next = (&raw const features).cast();
            let device = instance
                .create_device(physical, &info, None)
                .context("creating the Vulkan device")?;
            let queue = device.get_device_queue(queue_family, 0);

            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(PUSH_BYTES)];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )?;
            let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let completed = device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                None,
            )?;

            let memory = instance.get_physical_device_memory_properties(physical);
            let wanted = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory_type = (0..memory.memory_type_count)
                .find(|&i| memory.memory_types[i as usize].property_flags.contains(wanted))
                .ok_or_else(|| anyhow!("no host-visible device memory type"))?;
            let heap = memory.memory_types[memory_type as usize].heap_index;
            let heap_bytes = memory.memory_heaps[heap as usize].size;

            Ok(Arc::new(Self {
                _entry: entry,
                instance,
                device,
                queue,
                queue_family,
                layout,
                completed,
                memory_type,
                heap_bytes,
                modules: Mutex::new(HashMap::new()),
                pipelines: Mutex::new(HashMap::new()),
                allocated: AtomicUsize::new(0),
                allocations: AtomicUsize::new(0),
            }))
        }
    }

    /// The shader module of kernel `name`: its embedded SPIR-V, or
    /// `$LILY_VK_SHADER_DIR/<name>.spv` when that file exists (a shader under
    /// work, without a rebuild).
    fn module(&self, name: &'static str) -> Result<vk::ShaderModule> {
        let mut modules =
            self.modules.lock().map_err(|e| anyhow!("module cache poisoned: {e}"))?;
        if let Some(&module) = modules.get(name) {
            return Ok(module);
        }
        let external = std::env::var_os("LILY_VK_SHADER_DIR")
            .map(|dir| std::path::PathBuf::from(dir).join(format!("{name}.spv")))
            .filter(|path| path.is_file());
        let bytes = match &external {
            Some(path) => std::fs::read(path)
                .with_context(|| format!("reading {}", path.display()))?,
            None => KERNELS
                .iter()
                .find(|(kernel, _)| *kernel == name)
                .map(|(_, bytes)| bytes.to_vec())
                .ok_or_else(|| {
                    anyhow!("kernel '{name}' has no Vulkan shader (src/kernels/vulkan)")
                })?,
        };
        let words = ash::util::read_spv(&mut std::io::Cursor::new(&bytes))
            .with_context(|| format!("SPIR-V of {name}"))?;
        // SAFETY: `words` is a SPIR-V module.
        let module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )
        }
        .with_context(|| format!("creating the shader module of {name}"))?;
        modules.insert(name, module);
        Ok(module)
    }

    /// The pipeline of kernel `name` at threadgroup size `local`.
    fn pipeline(&self, name: &'static str, local: [u32; 3]) -> Result<vk::Pipeline> {
        {
            let pipelines = self
                .pipelines
                .lock()
                .map_err(|e| anyhow!("pipeline cache poisoned: {e}"))?;
            if let Some(&pipeline) = pipelines.get(&(name, local)) {
                return Ok(pipeline);
            }
        }
        let module = self.module(name)?;
        let entries = [0u32, 1, 2].map(|i| {
            vk::SpecializationMapEntry::default().constant_id(i).offset(4 * i).size(4)
        });
        let data: Vec<u8> = local.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let specialization =
            vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&specialization);
        let info =
            [vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout)];
        // SAFETY: valid module and layout.
        let pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        }
        .map_err(|(_, e)| anyhow!("creating the pipeline of {name} at {local:?}: {e}"))?[0];
        self.pipelines
            .lock()
            .map_err(|e| anyhow!("pipeline cache poisoned: {e}"))?
            .insert((name, local), pipeline);
        Ok(pipeline)
    }

    fn new_buffer(self: &Arc<Self>, len: usize, zero: bool) -> Result<Buffer> {
        let size = len.max(1).next_multiple_of(16) as u64;
        // SAFETY: plain Vulkan calls; every failure path frees what it made.
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = match self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(self.memory_type)
                    .push_next(&mut flags),
                None,
            ) {
                Ok(memory) => memory,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    bail!("failed to allocate {len}-byte buffer: {e}");
                }
            };
            let mapped = self
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|()| {
                    self.device.map_memory(
                        memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                });
            let ptr = match mapped {
                Ok(ptr) => NonNull::new(ptr).expect("mapped memory is not null"),
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                    bail!("failed to map {len}-byte buffer: {e}");
                }
            };
            if zero {
                // The driver recycles freed memory without clearing it.
                core::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0, len);
            }
            let address = self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer),
            );
            self.allocated.fetch_add(requirements.size as usize, Ordering::Relaxed);
            self.allocations.fetch_add(1, Ordering::Relaxed);
            Ok(Rc::new(GpuBuffer {
                dev: self.clone(),
                buffer,
                memory,
                ptr,
                len,
                reserved: requirements.size as usize,
                address,
            }))
        }
    }
}

pub struct GpuBuffer {
    dev: Arc<Dev>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: NonNull<c_void>,
    len: usize,
    reserved: usize,
    address: u64,
}

impl GpuBuffer {
    /// The buffer's address on the GPU.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The buffer's memory, mapped for as long as it lives.
    pub fn contents(&self) -> NonNull<c_void> {
        self.ptr
    }

    /// The buffer's size in bytes, as requested.
    pub fn length(&self) -> usize {
        self.len
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: the owner dropped the last handle after every pass that
        // used the buffer completed.
        unsafe {
            self.dev.device.destroy_buffer(self.buffer, None);
            self.dev.device.free_memory(self.memory, None);
        }
        self.dev.allocated.fetch_sub(self.reserved, Ordering::Relaxed);
        self.dev.allocations.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A host/GPU synchronization counter, as `MTLSharedEvent` is on Metal: the
/// host raises it ([`Self::signal`]) and so does a pass at a point of its
/// program ([`ComputePass::signal_event`]); a pass waits for it at a point
/// of its program ([`ComputePass::wait_event`]) and so does the host.
#[derive(Clone)]
pub struct SharedEvent {
    inner: Arc<EventInner>,
}

struct EventInner {
    value: AtomicU64,
    lock: Mutex<()>,
    raised: Condvar,
}

impl SharedEvent {
    fn new() -> Self {
        Self {
            inner: Arc::new(EventInner {
                value: AtomicU64::new(0),
                lock: Mutex::new(()),
                raised: Condvar::new(),
            }),
        }
    }

    /// Raises the counter to `value`; a lower value is ignored.
    pub fn signal(&self, value: u64) {
        let _guard = self.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.inner.value.fetch_max(value, Ordering::AcqRel) < value {
            self.inner.raised.notify_all();
        }
    }

    pub fn signaled_value(&self) -> u64 {
        self.inner.value.load(Ordering::Acquire)
    }

    /// Blocks until the counter reaches `value` or `timeout_ms` passes;
    /// true when it did.
    pub fn wait_until(&self, value: u64, timeout_ms: u64) -> bool {
        if self.signaled_value() >= value {
            return true;
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let mut guard = self.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        while self.signaled_value() < value {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            guard = self
                .inner
                .raised
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }

    /// Releases every pending and future wait, for error paths that would
    /// otherwise leave a committed pass waiting forever.
    pub fn release_all(&self) {
        self.signal(u64::MAX);
    }
}

/// First GPU error, context-wide.
#[derive(Default)]
struct Fault {
    message: Mutex<Option<String>>,
    set: AtomicBool,
}

impl Fault {
    fn record(&self, message: String) {
        if let Ok(mut slot) = self.message.lock() {
            slot.get_or_insert(message);
        }
        self.set.store(true, Ordering::Release);
    }

    fn is_set(&self) -> bool {
        self.set.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if !self.is_set() {
            return Ok(());
        }
        match self.message.lock() {
            Ok(slot) => match &*slot {
                Some(message) => bail!("Vulkan command buffer failed: {message}"),
                None => Ok(()),
            },
            Err(e) => bail!("fault lock poisoned: {e}"),
        }
    }

    fn message(&self) -> Option<String> {
        self.message.lock().ok().and_then(|slot| slot.clone())
    }
}

/// One step of a pass's program.
enum Item {
    /// A command buffer, and in profile mode the kernel it holds.
    Cmd(vk::CommandBuffer, &'static str),
    Wait(SharedEvent, u64),
    Signal(SharedEvent, u64),
}

/// What the submitter learns of a pass as it runs, for
/// [`CompletedPass::timing`] and the profile.
#[derive(Default)]
struct FeedbackState {
    done: bool,
    gpu_start_secs: f64,
    gpu_end_secs: f64,
}

#[derive(Default)]
struct Feedback {
    state: Mutex<FeedbackState>,
    changed: Condvar,
}

impl Feedback {
    fn finish(&self, start: f64, end: f64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.done = true;
        state.gpu_start_secs = start;
        state.gpu_end_secs = end;
        self.changed.notify_all();
    }

    fn complete(&self, fault: &Fault) -> Result<(f64, f64)> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.done {
            fault.check()?;
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok((state.gpu_start_secs, state.gpu_end_secs))
    }
}

/// A committed pass on its way to the submitter thread.
struct Submission {
    items: Vec<Item>,
    fence_value: u64,
    feedback: Arc<Feedback>,
    /// Profile mode: the pass's label.
    profile: Option<&'static str>,
}

/// What the watcher does once the queue has completed `serial` command
/// buffers.
struct Completion {
    serial: u64,
    then: Then,
}

enum Then {
    Signal(SharedEvent, u64),
    /// A pass's end: its feedback, the host time of its first submission
    /// (none for a pass without command buffers) and its fence value.
    Fence(Arc<Feedback>, Option<f64>, u64),
}

/// How long the submitter polls a host event before it blocks on it, as a
/// gated Metal pass's feeder does.
const GATE_SPIN: Duration = Duration::from_micros(200);

/// After this long at one wait the submitter gives up and faults the
/// context (`metal.rs`, `GATE_PATIENCE`).
const GATE_PATIENCE: Duration = Duration::from_secs(30);

struct Submitter {
    dev: Arc<Dev>,
    fault: Arc<Fault>,
    work: Receiver<Submission>,
    completions: Sender<Completion>,
    window: u64,
    profile: bool,
    serial: u64,
}

impl Submitter {
    fn run(mut self) {
        while let Ok(submission) = self.work.recv() {
            let mut first = None;
            let mut kernels = Vec::new();
            let began = host_secs();
            for item in &submission.items {
                if self.fault.is_set() {
                    break;
                }
                match item {
                    Item::Cmd(cmd, name) => {
                        let at = host_secs();
                        first.get_or_insert(at);
                        if let Err(e) = self.submit(*cmd) {
                            self.fault.record(format!("{e:#}"));
                        } else if self.profile {
                            self.wait_completed(self.serial);
                            kernels.push(profile::KernelSample {
                                name,
                                gpu_secs: host_secs() - at,
                            });
                        }
                    }
                    Item::Wait(event, value) => self.gate(event, *value),
                    Item::Signal(event, value) => {
                        let _ = self.completions.send(Completion {
                            serial: self.serial,
                            then: Then::Signal(event.clone(), *value),
                        });
                    }
                }
            }
            if let Some(label) = submission.profile {
                profile::record(profile::PassProfile {
                    label,
                    span_secs: host_secs() - began,
                    kernels,
                });
            }
            let _ = self.completions.send(Completion {
                serial: self.serial,
                then: Then::Fence(submission.feedback, first, submission.fence_value),
            });
        }
    }

    /// Submits one command buffer, which signals the completion counter,
    /// once fewer than the window are in flight.
    fn submit(&mut self, cmd: vk::CommandBuffer) -> Result<()> {
        if self.serial >= self.window {
            self.wait_completed(self.serial + 1 - self.window);
        }
        self.fault.check()?;
        self.serial += 1;
        let values = [self.serial];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let cmds = [cmd];
        let semaphores = [self.dev.completed];
        let info = [vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline)];
        // SAFETY: this thread alone uses the queue; the command buffer is
        // complete and stays alive until its pass was waited for.
        unsafe { self.dev.device.queue_submit(self.dev.queue, &info, vk::Fence::null()) }
            .map_err(|e| anyhow!("queue submission failed: {e}"))
    }

    /// Blocks until the queue has completed `serial` command buffers, or a
    /// fault.
    fn wait_completed(&self, serial: u64) {
        wait_completed(&self.dev, &self.fault, serial);
    }

    /// Blocks until the host has raised `event` to `value`.
    fn gate(&self, event: &SharedEvent, value: u64) {
        let began = Instant::now();
        while event.signaled_value() < value {
            if began.elapsed() < GATE_SPIN {
                std::hint::spin_loop();
                continue;
            }
            if event.wait_until(value, 50) || self.fault.is_set() {
                return;
            }
            if began.elapsed() > GATE_PATIENCE {
                let message = format!(
                    "a pass gave up after {:.1} s waiting for a host event to reach {value}",
                    began.elapsed().as_secs_f64()
                );
                eprintln!("vulkan: {message}");
                self.fault.record(message);
                return;
            }
        }
    }
}

/// Blocks until the queue has completed `serial` command buffers; a device
/// error becomes the context's fault, which also ends the wait.
fn wait_completed(dev: &Dev, fault: &Fault, serial: u64) {
    let semaphores = [dev.completed];
    let values = [serial];
    let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
    while !fault.is_set() {
        // SAFETY: a valid timeline semaphore.
        match unsafe { dev.device.wait_semaphores(&info, 1_000_000_000) } {
            Ok(()) => return,
            Err(vk::Result::TIMEOUT) => {}
            Err(e) => {
                let message = format!("waiting for the GPU failed: {e}");
                eprintln!("vulkan: {message}");
                fault.record(message);
            }
        }
    }
}

/// The watcher thread: raises each completion's event once the queue is
/// that far, in order.
fn watch(
    dev: Arc<Dev>,
    fault: Arc<Fault>,
    completions: Receiver<Completion>,
    fence: SharedEvent,
) {
    let mut last_end = 0.0f64;
    while let Ok(Completion { serial, then }) = completions.recv() {
        if serial > 0 {
            wait_completed(&dev, &fault, serial);
        }
        let now = host_secs();
        match then {
            Then::Signal(event, value) => event.signal(value),
            Then::Fence(feedback, first, value) => {
                let (start, end) = match first {
                    Some(first) => (first.max(last_end), now),
                    None => (now, now),
                };
                if first.is_some() && end > start {
                    busy::record(&[(start, end)]);
                }
                last_end = now;
                feedback.finish(start, end);
                fence.signal(value);
            }
        }
    }
}

pub struct MetalContext {
    dev: Arc<Dev>,
    pool: Rc<Pool>,
    /// Completion counter: the watcher raises it to a pass's value once the
    /// pass is done.
    fence: SharedEvent,
    fence_value: AtomicU64,
    fault: Arc<Fault>,
    work: Option<Sender<Submission>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    built: AtomicU64,
    build_nanos: AtomicU64,
    profile: bool,
    batch: (usize, usize),
    recorded: RefCell<Option<Vec<Weak<GpuBuffer>>>>,
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // Closing the channel ends the submitter, whose end closes the
        // watcher's.
        self.work = None;
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl MetalContext {
    /// A context on the first Vulkan device; the per-kernel profile mode is
    /// on when `LILY_KERNEL_PROFILE` is set to anything but `0`.
    pub fn new() -> Result<Self> {
        let profile = std::env::var_os("LILY_KERNEL_PROFILE")
            .is_some_and(|v| !v.is_empty() && v != "0");
        Self::new_with_profile(profile)
    }

    /// Like [`Self::new`], with the per-kernel profile mode chosen explicitly.
    pub fn new_with_profile(profile: bool) -> Result<Self> {
        let dev = Dev::new()?;
        let fault = Arc::new(Fault::default());
        let fence = SharedEvent::new();
        let (work, submissions) = channel();
        let (completions, completed) = channel();
        let submitter = Submitter {
            dev: dev.clone(),
            fault: fault.clone(),
            work: submissions,
            completions,
            window: env_usize("LILY_VK_WINDOW").map_or(WINDOW, |v| v as u64),
            profile,
            serial: 0,
        };
        let threads = vec![
            std::thread::Builder::new()
                .name("lily-vk-submit".into())
                .spawn(move || submitter.run())?,
            {
                let (dev, fault, fence) = (dev.clone(), fault.clone(), fence.clone());
                std::thread::Builder::new()
                    .name("lily-vk-watch".into())
                    .spawn(move || watch(dev, fault, completed, fence))?
            },
        ];
        let batch = env_usize("LILY_VK_BATCH")
            .map_or((SERIAL_BATCH, CONCURRENT_BATCH), |v| (v, v));
        Ok(Self {
            pool: Rc::new(Pool { dev: dev.clone(), free: RefCell::new(Vec::new()) }),
            dev,
            fence,
            fence_value: AtomicU64::new(0),
            fault,
            work: Some(work),
            threads,
            built: AtomicU64::new(0),
            build_nanos: AtomicU64::new(0),
            profile,
            batch,
            recorded: RefCell::new(None),
        })
    }

    /// Whether the per-kernel profile mode is on; see [`profile`].
    pub fn profiling(&self) -> bool {
        self.profile
    }

    /// The kernel `fn_name`. `source` and `version` are the Metal backend's:
    /// the shader was built with the crate (`src/kernels/vulkan`). Its
    /// pipelines are created per threadgroup size at their first dispatch;
    /// this checks that the shader exists.
    pub fn pipeline(
        &self,
        fn_name: &'static str,
        _source: &str,
        _version: MslVersion,
    ) -> Result<Kernel> {
        self.dev.module(fn_name)?;
        Ok(Kernel { name: fn_name })
    }

    /// Pipelines built so far and the seconds that took.
    pub fn build_stats(&self) -> (u64, f64) {
        (
            self.built.load(Ordering::Relaxed),
            self.build_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        )
    }

    /// Allocates a zero-initialized buffer the host and the GPU share.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let buf = self.dev.new_buffer(len, true)?;
        if let Some(recorded) = self.recorded.borrow_mut().as_mut() {
            recorded.push(Rc::downgrade(&buf));
        }
        Ok(buf)
    }

    /// Starts recording the buffers [`Self::new_buffer`] allocates, until
    /// [`BufferRecording::finish`] returns the ones still alive.
    pub fn record_buffers(&self) -> Result<BufferRecording<'_>> {
        let mut recorded = self.recorded.borrow_mut();
        ensure!(recorded.is_none(), "a buffer recording is already open");
        *recorded = Some(Vec::new());
        Ok(BufferRecording { ctx: self })
    }

    /// A buffer holding `len` bytes of `path` at `offset`. Metal maps the
    /// file's pages; Vulkan has no buffer over host pages here, so this
    /// reads them: the buffer is anonymous memory the OS cannot drop. As on
    /// Metal the buffer starts at the page boundary at or below `offset`,
    /// and the second value is the requested range's offset inside it.
    pub fn new_buffer_mapped(
        &self,
        path: &std::path::Path,
        offset: u64,
        len: usize,
    ) -> Result<(Buffer, usize)> {
        use std::os::unix::fs::FileExt;
        ensure!(len > 0, "mapped buffer of zero bytes");
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let base = offset - offset % page;
        let inner = usize::try_from(offset - base)?;
        let buf = self.dev.new_buffer(inner + len, false)?;
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening {} for mapping", path.display()))?;
        // SAFETY: the buffer was just allocated with `inner + len` bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(buf.contents().as_ptr().cast::<u8>(), inner + len)
        };
        file.read_exact_at(bytes, base)
            .with_context(|| format!("reading {} bytes of {}", inner + len, path.display()))?;
        Ok((buf, inner))
    }

    /// A buffer over host memory the caller keeps. Vulkan cannot wrap host
    /// pages here; nothing on a machine that holds the whole model needs it.
    ///
    /// # Safety
    ///
    /// As on Metal.
    pub unsafe fn new_buffer_no_copy(
        &self,
        _ptr: NonNull<c_void>,
        _len: usize,
    ) -> Result<Buffer> {
        bail!("the Vulkan backend has no buffers over host memory (the expert cache needs them)")
    }

    pub fn new_buffer_with_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let buf = self.new_buffer(bytes.len())?;
        // SAFETY: the buffer was just allocated with exactly `bytes.len()` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(buf)
    }

    /// Live buffers of this context's device.
    pub fn resident_allocations(&self) -> usize {
        self.dev.allocations.load(Ordering::Relaxed)
    }

    /// Starts a serial compute pass: dispatches execute in encode order (a
    /// barrier follows each one).
    pub fn begin(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(false)
    }

    /// A compute pass whose dispatches may overlap; the caller owns every
    /// hazard via [`ComputePass::memory_barrier`] at dependency boundaries.
    pub fn begin_concurrent(&self) -> Result<ComputePass<'_>> {
        self.begin_pass(true)
    }

    fn begin_pass(&self, concurrent: bool) -> Result<ComputePass<'_>> {
        let slot = self.pool.take()?;
        Ok(ComputePass {
            ctx: self,
            program: RefCell::new(Vec::new()),
            open: RefCell::new(None),
            concurrent,
            done: RefCell::new(None),
            milestone: RefCell::new(None),
            label: Cell::new(profile::UNLABELED),
            slot: RefCell::new(slot),
        })
    }

    /// A host/GPU synchronization point for [`ComputePass::wait_event`].
    pub fn new_shared_event(&self) -> Result<SharedEvent> {
        Ok(SharedEvent::new())
    }

    /// As [`Self::new_shared_event`]: no wait can be released early here
    /// (the submitter holds the work behind a wait until the host's signal),
    /// so there is nothing to watch.
    pub fn new_watched_event(&self) -> Result<SharedEvent> {
        Ok(SharedEvent::new())
    }

    /// 10, the family of the GPUs whose matrix units the kernels are written
    /// for: the device was checked for them when the context was made.
    pub fn apple_gpu_family(&self) -> i64 {
        10
    }

    /// How much this process may keep on the GPU: the driver's heap (the
    /// kernel's per-client limit, or `heap_memory_percent` of RAM).
    pub fn recommended_working_set(&self) -> usize {
        self.dev.heap_bytes as usize
    }

    /// Bytes currently allocated on the device by this process.
    pub fn current_allocated(&self) -> usize {
        self.dev.allocated.load(Ordering::Relaxed)
    }

    /// The first GPU error on this context, if any; permanent, as on Metal.
    pub fn fault(&self) -> Option<String> {
        self.fault.message()
    }

    /// Records a fault as if the GPU had reported `message`.
    pub fn inject_fault(&self, message: &str) {
        self.fault.record(message.to_owned());
    }

    /// Metal's return from idle; nothing to start here.
    pub fn wake(&self) -> Result<()> {
        self.fault.check()
    }

    /// Copies byte ranges between buffers on the GPU and waits.
    pub fn blit_copy(&self, copies: &[BlitCopy<'_>]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let pass = self.begin()?;
        for copy in copies {
            ensure!(
                copy.src_offset + copy.len <= copy.src.byte_len()
                    && copy.dst_offset + copy.len <= copy.dst.byte_len(),
                "blit copy out of range"
            );
            if copy.len == 0 {
                continue;
            }
            let (src, src_base) = copy.src.binding();
            let (dst, dst_base) = copy.dst.binding();
            pass.copy_buffer(
                src,
                src_base + copy.src_offset,
                dst,
                dst_base + copy.dst_offset,
                copy.len,
            )?;
        }
        pass.commit_wait()
    }

    /// Hands a pass's program to the submitter and returns the fence value
    /// its end raises.
    fn submit(
        &self,
        items: Vec<Item>,
        feedback: &Arc<Feedback>,
        label: &'static str,
    ) -> Result<u64> {
        self.fault.check()?;
        let fence_value = self.fence_value.fetch_add(1, Ordering::AcqRel) + 1;
        self.work
            .as_ref()
            .ok_or_else(|| anyhow!("the context is shutting down"))?
            .send(Submission {
                items,
                fence_value,
                feedback: feedback.clone(),
                profile: self.profile.then_some(label),
            })
            .map_err(|_| anyhow!("the submitter thread is gone"))?;
        Ok(fence_value)
    }

    /// Blocks until the fence reaches `value`, bailing early on a fault.
    fn wait_fence(&self, value: u64) -> Result<()> {
        while !self.fence.wait_until(value, 1000) {
            self.fault.check()?;
        }
        self.fault.check()
    }
}

/// An open [`MetalContext::record_buffers`]; dropping it without
/// [`Self::finish`] (an error path) ends the recording and forgets it.
pub struct BufferRecording<'a> {
    ctx: &'a MetalContext,
}

impl BufferRecording<'_> {
    /// Ends the recording and returns the recorded buffers still alive.
    pub fn finish(self) -> Vec<Buffer> {
        let recorded = self.ctx.recorded.borrow_mut().take().unwrap_or_default();
        recorded.iter().filter_map(Weak::upgrade).collect()
    }
}

impl Drop for BufferRecording<'_> {
    fn drop(&mut self) {
        self.ctx.recorded.borrow_mut().take();
    }
}

/// Command pools with their arenas, recycled across passes.
struct Pool {
    dev: Arc<Dev>,
    free: RefCell<Vec<SlotInner>>,
}

impl Pool {
    fn take(self: &Rc<Self>) -> Result<Slot> {
        let recycled = self.free.borrow_mut().pop();
        let inner = match recycled {
            Some(inner) => inner,
            None => {
                // SAFETY: a valid device.
                let pool = unsafe {
                    self.dev.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(self.dev.queue_family),
                        None,
                    )
                }?;
                SlotInner {
                    dev: self.dev.clone(),
                    pool,
                    cmds: Vec::new(),
                    used: 0,
                    arena: Arena::default(),
                }
            }
        };
        Ok(Slot { inner: Some(inner), pool: self.clone() })
    }

    fn recycle(&self, mut inner: SlotInner) {
        // SAFETY: the pass that used the pool's command buffers completed
        // (or never submitted them).
        if unsafe {
            inner
                .dev
                .device
                .reset_command_pool(inner.pool, vk::CommandPoolResetFlags::empty())
        }
        .is_ok()
        {
            inner.used = 0;
            inner.arena.reset();
            self.free.borrow_mut().push(inner);
        }
    }
}

struct SlotInner {
    dev: Arc<Dev>,
    pool: vk::CommandPool,
    /// Every command buffer of the pool; the first `used` are this pass's.
    cmds: Vec<vk::CommandBuffer>,
    used: usize,
    arena: Arena,
}

impl SlotInner {
    /// A command buffer in the recording state.
    fn begin(&mut self) -> Result<vk::CommandBuffer> {
        if self.used == self.cmds.len() {
            // SAFETY: a valid pool.
            let more = unsafe {
                self.dev.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(8),
                )
            }?;
            self.cmds.extend(more);
        }
        let cmd = self.cmds[self.used];
        self.used += 1;
        // SAFETY: the command buffer is in the initial state.
        unsafe {
            self.dev.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }?;
        Ok(cmd)
    }
}

impl Drop for SlotInner {
    fn drop(&mut self) {
        // SAFETY: dropped with its pool (or after a fault, when nothing is
        // submitted any more).
        unsafe { self.dev.device.destroy_command_pool(self.pool, None) };
    }
}

struct Slot {
    inner: Option<SlotInner>,
    pool: Rc<Pool>,
}

impl Slot {
    fn get(&mut self) -> &mut SlotInner {
        self.inner.as_mut().expect("slot in use")
    }

    /// Gives the slot up without recycling it: after a fault its command
    /// buffers may still be on the queue.
    fn abandon(mut self) {
        if let Some(inner) = self.inner.take() {
            std::mem::forget(inner);
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.pool.recycle(inner);
        }
    }
}

#[derive(Default)]
struct Arena {
    chunks: Vec<Buffer>,
    chunk: usize,
    cursor: usize,
}

impl Arena {
    fn reset(&mut self) {
        self.chunk = 0;
        self.cursor = 0;
    }

    /// Copies `bytes` into the arena and returns their GPU address.
    fn push(&mut self, dev: &Arc<Dev>, bytes: &[u8]) -> Result<u64> {
        let len = bytes.len().max(1);
        loop {
            if let Some(chunk) = self.chunks.get(self.chunk) {
                let start = self.cursor.next_multiple_of(PARAM_ALIGN);
                if start + len <= chunk.length() {
                    // SAFETY: the range lies inside the chunk.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            chunk.contents().as_ptr().cast::<u8>().add(start),
                            bytes.len(),
                        );
                    }
                    self.cursor = start + len;
                    return Ok(chunk.address() + start as u64);
                }
                self.chunk += 1;
                self.cursor = 0;
                continue;
            }
            let size =
                ARENA_CHUNK.max(len.next_power_of_two()) << self.chunks.len().min(8);
            let chunk = dev.new_buffer(size, false)?;
            self.chunks.push(chunk);
        }
    }
}

/// The command buffer being encoded.
struct OpenSegment {
    cmd: vk::CommandBuffer,
    dispatches: usize,
    bound: vk::Pipeline,
    kernel: &'static str,
}

/// A batch of GPU work.
pub struct ComputePass<'a> {
    ctx: &'a MetalContext,
    /// Closed command buffers and the queue operations between them.
    program: RefCell<Vec<Item>>,
    open: RefCell<Option<OpenSegment>>,
    concurrent: bool,
    /// The event and value [`Self::signal_done`] encoded, if any.
    done: RefCell<Option<(SharedEvent, u64)>>,
    /// The last host-signalled wait of the pass ([`Self::set_milestone`]).
    milestone: RefCell<Option<(SharedEvent, u64)>>,
    label: Cell<&'static str>,
    slot: RefCell<Slot>,
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        // An error path leaves a segment open: end it so the pool can be
        // reset. Nothing is committed.
        if let Some(seg) = self.open.borrow_mut().take() {
            // SAFETY: the command buffer is recording.
            let _ = unsafe { self.ctx.dev.device.end_command_buffer(seg.cmd) };
        }
    }
}

/// One inline kernel argument (a `constant T&` in MSL) of a dispatch.
#[derive(Clone, Copy)]
pub enum Param<'p> {
    /// A host constant, copied into the pass's params arena.
    Bytes(&'p [u8]),
    /// A host `uint` constant.
    U32(u32),
    /// A host `float` constant.
    F32(f32),
    /// The word at this byte offset of a buffer, read by the GPU when the
    /// dispatch runs (so a kernel earlier in the pass may set it).
    Gpu(&'p GpuBuffer, usize),
}

/// How to map work items onto the GPU for one dispatch.
#[derive(Clone, Copy)]
pub enum Grid {
    /// Exactly this many threads: Vulkan dispatches whole threadgroups, so
    /// the shader leaves the threads outside the grid idle (`common.glsl`,
    /// `LILY_THREAD_IN_GRID`).
    Threads { grid: (usize, usize, usize), threadgroup: (usize, usize, usize) },
    /// An exact threadgroup count.
    Threadgroups { groups: (usize, usize, usize), threadgroup: (usize, usize, usize) },
}

fn barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    let barriers = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE,
        )];
    // SAFETY: the command buffer is recording.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            stages,
            stages,
            vk::DependencyFlags::empty(),
            &barriers,
            &[],
            &[],
        )
    };
}

impl<'a> ComputePass<'a> {
    /// Runs `f` on the open command buffer, opening one first if none is.
    fn with_cmd<R>(
        &self,
        f: impl FnOnce(&ash::Device, &mut OpenSegment, &mut SlotInner) -> Result<R>,
    ) -> Result<R> {
        let mut open = self.open.borrow_mut();
        let mut slot = self.slot.borrow_mut();
        let inner = slot.get();
        if open.is_none() {
            *open = Some(OpenSegment {
                cmd: inner.begin()?,
                dispatches: 0,
                bound: vk::Pipeline::null(),
                kernel: "(no dispatch)",
            });
        }
        f(&self.ctx.dev.device, open.as_mut().expect("segment open"), inner)
    }

    /// Closes the open command buffer (if any) into the program.
    fn close_segment(&self) -> Result<()> {
        if let Some(seg) = self.open.borrow_mut().take() {
            // SAFETY: the command buffer is recording.
            unsafe { self.ctx.dev.device.end_command_buffer(seg.cmd) }?;
            self.program.borrow_mut().push(Item::Cmd(seg.cmd, seg.kernel));
        }
        Ok(())
    }

    /// Closes the command buffer once it holds a batch (in profile mode,
    /// one dispatch).
    fn close_full_segment(&self) -> Result<()> {
        let limit = if self.ctx.profile {
            1
        } else if self.concurrent {
            self.ctx.batch.1
        } else {
            self.ctx.batch.0
        };
        let full = self.open.borrow().as_ref().is_some_and(|seg| seg.dispatches >= limit);
        if full { self.close_segment() } else { Ok(()) }
    }

    /// Names the pass in the per-kernel profile ([`profile`]).
    pub fn set_label(&self, label: &'static str) {
        self.label.set(label);
    }

    /// Encodes one kernel dispatch with a byte offset per buffer binding.
    /// Every param is a host constant; see [`Self::dispatch_with`].
    pub fn dispatch_at(
        &self,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, usize)],
        params: &[&[u8]],
        grid: Grid,
    ) -> Result<()> {
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        let mut args = [Param::U32(0); MAX_KERNEL_PARAMS];
        for (arg, bytes) in args.iter_mut().zip(params) {
            *arg = Param::Bytes(bytes);
        }
        self.dispatch_with(kernel, buffers, &args[..params.len()], grid)
    }

    /// Like [`Self::dispatch_at`], with each param either a host constant or
    /// the address of a word in a buffer, read when the dispatch runs.
    /// Params bind at indices `buffers.len()..`.
    pub fn dispatch_with(
        &self,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, usize)],
        params: &[Param<'_>],
        grid: Grid,
    ) -> Result<()> {
        ensure!(
            params.len() <= MAX_KERNEL_PARAMS,
            "{} kernel params exceeds MAX_KERNEL_PARAMS ({MAX_KERNEL_PARAMS})",
            params.len()
        );
        ensure!(
            buffers.len() + params.len() <= MAX_BUFFER_BINDINGS,
            "{} buffer bindings exceeds the limit of {MAX_BUFFER_BINDINGS}",
            buffers.len() + params.len()
        );
        let (threads, groups, local) = match grid {
            Grid::Threads { grid, threadgroup } => (
                grid,
                (
                    grid.0.div_ceil(threadgroup.0),
                    grid.1.div_ceil(threadgroup.1),
                    grid.2.div_ceil(threadgroup.2),
                ),
                threadgroup,
            ),
            Grid::Threadgroups { groups, threadgroup } => (
                (
                    groups.0 * threadgroup.0,
                    groups.1 * threadgroup.1,
                    groups.2 * threadgroup.2,
                ),
                groups,
                threadgroup,
            ),
        };
        let local = [local.0 as u32, local.1 as u32, local.2 as u32];
        let started = Instant::now();
        let known = self.ctx.dev.pipelines.lock().is_ok_and(|p| p.contains_key(&(kernel.name, local)));
        let pipeline = self.ctx.dev.pipeline(kernel.name, local)?;
        if !known {
            self.ctx.built.fetch_add(1, Ordering::Relaxed);
            self.ctx
                .build_nanos
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let serial = !self.concurrent;
        let dev = &self.ctx.dev;
        self.with_cmd(|device, seg, inner| {
            let mut table = [0u64; MAX_BUFFER_BINDINGS];
            for (slot, (buf, offset)) in table.iter_mut().zip(buffers) {
                *slot = buf.address() + *offset as u64;
            }
            for (slot, param) in table[buffers.len()..].iter_mut().zip(params) {
                *slot = match param {
                    Param::Bytes(bytes) => {
                        ensure!(!bytes.is_empty(), "empty kernel param");
                        inner.arena.push(dev, bytes)?
                    }
                    Param::U32(v) => inner.arena.push(dev, &v.to_ne_bytes())?,
                    Param::F32(v) => inner.arena.push(dev, &v.to_ne_bytes())?,
                    Param::Gpu(buf, offset) => {
                        ensure!(
                            offset.is_multiple_of(4) && *offset + 4 <= buf.length(),
                            "GPU param at byte {offset} is not a word of its buffer"
                        );
                        buf.address() + *offset as u64
                    }
                };
            }
            let bound = buffers.len() + params.len();
            let address =
                inner.arena.push(dev, bytemuck::cast_slice(&table[..bound.max(1)]))?;
            let mut push = [0u8; PUSH_BYTES as usize];
            push[..8].copy_from_slice(&address.to_ne_bytes());
            push[8..12].copy_from_slice(&(threads.0 as u32).to_ne_bytes());
            push[12..16].copy_from_slice(&(threads.1 as u32).to_ne_bytes());
            push[16..20].copy_from_slice(&(threads.2 as u32).to_ne_bytes());
            // SAFETY: the command buffer is recording; the pipeline and its
            // layout are the device's.
            unsafe {
                if seg.bound != pipeline {
                    device.cmd_bind_pipeline(
                        seg.cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline,
                    );
                    seg.bound = pipeline;
                }
                device.cmd_push_constants(
                    seg.cmd,
                    dev.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                device.cmd_dispatch(
                    seg.cmd,
                    groups.0 as u32,
                    groups.1 as u32,
                    groups.2 as u32,
                );
            }
            if serial {
                barrier(device, seg.cmd);
            }
            seg.dispatches += 1;
            seg.kernel = kernel.name;
            Ok(())
        })?;
        self.close_full_segment()
    }

    /// Copies `len` bytes between buffers (any alignment), ordered like a
    /// dispatch of this pass.
    pub fn copy_buffer(
        &self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &GpuBuffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let serial = !self.concurrent;
        self.with_cmd(|device, seg, _inner| {
            let regions = [vk::BufferCopy::default()
                .src_offset(src_offset as u64)
                .dst_offset(dst_offset as u64)
                .size(len as u64)];
            // SAFETY: ranges validated by the caller against the buffer sizes.
            unsafe { device.cmd_copy_buffer(seg.cmd, src.buffer, dst.buffer, &regions) };
            if serial {
                barrier(device, seg.cmd);
            }
            seg.dispatches += 1;
            seg.kernel = "copy_buffer";
            Ok(())
        })?;
        self.close_full_segment()
    }

    /// Orders one dependency level before the next on a concurrent pass.
    pub fn level_barrier(&self, _written: &[&Tensor]) -> Result<()> {
        profile::count_level();
        self.memory_barrier()
    }

    /// Holds everything encoded after this point until the host has
    /// signalled `event` to at least `value`; everything encoded before
    /// completes first. The pass can then be committed before the inputs
    /// read after this point exist. The event must be signalled by another
    /// thread, or after the commit.
    pub fn wait_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.close_segment()?;
        self.program.borrow_mut().push(Item::Wait(event.clone(), value));
        Ok(())
    }

    /// Raises `event` to `value` once everything encoded before this point
    /// has completed.
    pub fn signal_event(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.close_segment()?;
        self.program.borrow_mut().push(Item::Signal(event.clone(), value));
        Ok(())
    }

    /// Orders all prior dispatches' buffer writes before every subsequent
    /// dispatch: the dependency-level boundary of a concurrent pass. Serial
    /// passes already order every dispatch.
    pub fn memory_barrier(&self) -> Result<()> {
        if self.concurrent && !self.ctx.profile {
            let open = self.open.borrow();
            if let Some(seg) = open.as_ref() {
                barrier(&self.ctx.dev.device, seg.cmd);
            }
        }
        Ok(())
    }

    /// Ends encoding and submits without blocking; pair with
    /// [`PendingPass::wait`].
    pub fn commit(self) -> Result<PendingPass<'a>> {
        self.end()?.commit()
    }

    /// Ends encoding without submitting. The host may still write shared
    /// buffers the encoded kernels read before [`EncodedPass::commit`].
    pub fn end(self) -> Result<EncodedPass<'a>> {
        self.close_segment()?;
        let program = std::mem::take(&mut *self.program.borrow_mut());
        let done = self.done.borrow().clone();
        let milestone = self.milestone.borrow_mut().take();
        let slot = self.slot.replace(Slot { inner: None, pool: self.ctx.pool.clone() });
        Ok(EncodedPass {
            ctx: self.ctx,
            program,
            done,
            milestone,
            slot,
            label: self.label.get(),
        })
    }

    /// Encodes, as the pass's last command, a signal of `event` to `value`,
    /// and remembers it: [`PendingPass::wait_polled`] then polls the event.
    pub fn signal_done(&self, event: &SharedEvent, value: u64) -> Result<()> {
        self.signal_event(event, value)?;
        *self.done.borrow_mut() = Some((event.clone(), value));
        Ok(())
    }

    /// On Metal, submits the pass in parts at its waits; every pass is
    /// submitted that way here.
    pub fn gate_host_waits(&self) {}

    /// Marks `event` reaching `value` as the pass's milestone: a wait the
    /// host satisfies, after which little of the pass is left
    /// ([`PendingPass::wait_polled`]).
    pub fn set_milestone(&self, event: &SharedEvent, value: u64) {
        *self.milestone.borrow_mut() = Some((event.clone(), value));
    }

    /// Ends encoding, submits, and blocks until the GPU finishes.
    pub fn commit_wait(self) -> Result<()> {
        self.commit()?.wait()
    }
}

/// A fully encoded, not yet submitted pass.
pub struct EncodedPass<'a> {
    ctx: &'a MetalContext,
    program: Vec<Item>,
    done: Option<(SharedEvent, u64)>,
    milestone: Option<(SharedEvent, u64)>,
    slot: Slot,
    label: &'static str,
}

impl<'a> EncodedPass<'a> {
    /// Submits without blocking.
    pub fn commit(self) -> Result<PendingPass<'a>> {
        let EncodedPass { ctx, program, done, milestone, slot, label } = self;
        let feedback = Arc::new(Feedback::default());
        let fence_value = ctx.submit(program, &feedback, label)?;
        Ok(PendingPass { ctx, fence_value, done, milestone, feedback, slot: Some(slot) })
    }
}

/// One buffer-to-buffer copy for [`MetalContext::blit_copy`], in bytes.
pub struct BlitCopy<'t> {
    pub src: &'t Tensor,
    pub src_offset: usize,
    pub dst: &'t Tensor,
    pub dst_offset: usize,
    pub len: usize,
}

/// A committed-but-unawaited pass. Holding one while encoding the next pass
/// is the decode pipelining primitive; hosts must not read buffers the
/// pending pass writes until [`Self::wait`] returns. Dropping it without
/// waiting blocks until the GPU is done (its command memory is recycled).
pub struct PendingPass<'a> {
    ctx: &'a MetalContext,
    fence_value: u64,
    done: Option<(SharedEvent, u64)>,
    milestone: Option<(SharedEvent, u64)>,
    feedback: Arc<Feedback>,
    slot: Option<Slot>,
}

impl PendingPass<'_> {
    /// Returns the command memory to the pool, or leaves it alone when the
    /// pass may still be on the queue (a fault ended the wait).
    fn release(&mut self, completed: bool) {
        if let Some(slot) = self.slot.take() {
            if completed { drop(slot) } else { slot.abandon() }
        }
    }
}

impl Drop for PendingPass<'_> {
    fn drop(&mut self) {
        if self.slot.is_some() {
            let completed = self.ctx.wait_fence(self.fence_value).is_ok();
            self.release(completed);
        }
    }
}

/// A pass's GPU execution span, in seconds on the host clock of
/// [`host_secs`], so host and GPU marks can be subtracted.
#[derive(Clone, Copy, Debug)]
pub struct PassTiming {
    pub gpu_start_secs: f64,
    pub gpu_end_secs: f64,
}

/// Host time in seconds on the clock [`PassTiming`] uses (`CLOCK_MONOTONIC`).
pub fn host_secs() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: a valid out-pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

pub struct CompletedPass {
    feedback: Arc<Feedback>,
    fault: Arc<Fault>,
}

impl CompletedPass {
    /// The pass's GPU time span, as the watcher thread saw it.
    pub fn timing(&self) -> Result<PassTiming> {
        let (gpu_start_secs, gpu_end_secs) = self.feedback.complete(&self.fault)?;
        let timing = PassTiming { gpu_start_secs, gpu_end_secs };
        ensure!(
            timing.gpu_start_secs > 0.0 && timing.gpu_end_secs >= timing.gpu_start_secs,
            "invalid GPU timestamps: {timing:?}"
        );
        Ok(timing)
    }
}

impl PendingPass<'_> {
    /// Blocks until the GPU finishes this pass.
    pub fn wait(self) -> Result<()> {
        self.wait_retain().map(|_| ())
    }

    /// Blocks until completion while retaining the pass's feedback.
    pub fn wait_retain(mut self) -> Result<CompletedPass> {
        let waited = self.ctx.wait_fence(self.fence_value);
        self.release(waited.is_ok());
        waited?;
        self.finished()
    }

    /// Like [`Self::wait`], polled: checks the pass's done signal (see
    /// [`ComputePass::signal_done`], falling back to the fence) every
    /// [`POLL_INTERVAL`].
    pub fn wait_polled(self) -> Result<()> {
        self.wait_retain_polled().map(|_| ())
    }

    /// [`Self::wait_polled`], keeping the pass's feedback like
    /// [`Self::wait_retain`].
    pub fn wait_retain_polled(mut self) -> Result<CompletedPass> {
        let (event, value) = match &self.done {
            Some((event, value)) => (event.clone(), *value),
            None => (self.ctx.fence.clone(), self.fence_value),
        };
        let polled = (|| {
            if let Some((milestone, reached)) = &self.milestone {
                while !milestone.wait_until(*reached, MILESTONE_POLL_MS) {
                    if event.signaled_value() >= value {
                        break;
                    }
                    self.ctx.fault.check()?;
                }
            }
            while event.signaled_value() < value {
                self.ctx.fault.check()?;
                std::thread::sleep(POLL_INTERVAL);
            }
            // The done event may lead the fence; the command memory is only
            // safe to reuse after the fence.
            self.ctx.wait_fence(self.fence_value)
        })();
        self.release(polled.is_ok());
        polled?;
        self.finished()
    }

    fn finished(&mut self) -> Result<CompletedPass> {
        self.ctx.fault.check()?;
        Ok(CompletedPass {
            feedback: self.feedback.clone(),
            fault: self.ctx.fault.clone(),
        })
    }
}

/// How often a polled wait blocked on a milestone checks whether the pass
/// ended without it.
const MILESTONE_POLL_MS: u64 = 50;

/// How long a polled wait ([`PendingPass::wait_polled`]) sleeps between
/// checks of the pass's done signal.
pub const POLL_INTERVAL: Duration = Duration::from_micros(20);
