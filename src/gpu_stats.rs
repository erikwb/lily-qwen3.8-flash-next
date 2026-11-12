// The per-kernel profile and the GPU busy time, which every backend records
// the same way: included into `metal.rs` and `vulkan.rs` (`crate::metal`).

/// Per-kernel GPU timing. In profile mode ([`MetalContext::new_with_profile`],
/// or `LILY_KERNEL_PROFILE=1` at [`MetalContext::new`]) every dispatch of a
/// pass is closed into its own command buffer, so the commit feedback's GPU
/// start/end times measure that one kernel. Each completed pass's breakdown
/// is recorded process-wide until [`take`] drains it; nothing is recorded
/// otherwise. The mode changes the work's shape (one command buffer per
/// dispatch instead of one per pass, no encoder barriers), so wall time and
/// pass spans under it are not production numbers; the per-kernel times are.
pub mod profile {
    use std::sync::Mutex;

    /// Label of passes nobody named through [`super::ComputePass::set_label`].
    pub const UNLABELED: &str = "(unlabeled)";

    /// One dispatch's GPU time.
    #[derive(Clone, Debug)]
    pub struct KernelSample {
        pub name: &'static str,
        pub gpu_secs: f64,
    }

    /// One completed pass: its label, its dispatches in encode order (copies
    /// appear as `copy_buffer`), and its GPU span from the first command
    /// buffer's start to the last one's end. The span minus the summed
    /// samples is time between command buffers: the profile mode's own cost
    /// plus whatever gaps exist without it (a parked pass's wait included).
    #[derive(Clone, Debug)]
    pub struct PassProfile {
        pub label: &'static str,
        pub span_secs: f64,
        pub kernels: Vec<KernelSample>,
    }

    static PASSES: Mutex<Vec<PassProfile>> = Mutex::new(Vec::new());

    pub(super) fn record(pass: PassProfile) {
        if let Ok(mut passes) = PASSES.lock() {
            passes.push(pass);
        }
    }

    /// Drains every pass profile recorded so far (in completion order).
    pub fn take() -> Vec<PassProfile> {
        PASSES.lock().map(|mut passes| std::mem::take(&mut *passes)).unwrap_or_default()
    }

    static LEVELS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    pub(super) fn count_level() {
        LEVELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Dependency levels (barriers) encoded so far, process-wide: the
    /// difference across one pass's encoding is that pass's level count.
    pub fn levels() -> usize {
        LEVELS.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// lily's own GPU busy time, process-wide: what the dashboard's estimate of
/// lily's GPU power divides the machine's GPU power by (`serve::machine`,
/// docs/dashboard.md § 6). Every command buffer lily commits goes through
/// [`super::submit_items`] with a commit feedback handler, so every pass is
/// counted, whichever context it ran on (the model, the drafts, vision, the
/// expert cache).
///
/// **What is counted.** Each command buffer's GPU span (`GPUStartTime` to
/// `GPUEndTime`, wall time on the host clock, stalls inside the buffer
/// included). Command buffers of one pass and consecutive passes can overlap,
/// and their feedback arrives out of order, so nothing can be added as it
/// arrives: the handler that completes a pass merges the pass's own spans
/// into their union ([`pass_union`]) and adds that union to a small set of
/// recent intervals ([`Recent`]), counting only the time the set did not
/// cover yet. The set forgets intervals that ended more than
/// [`HORIZON_SECS`] before the newest end it saw, which keeps it small; a pass
/// whose union reaches back past that horizon (its feedback completed more
/// than that after an overlapping pass's) could be counted twice where they
/// overlap. Passes on one queue run one after the other with an overlap of a
/// command buffer at most, so that is not expected to happen. The server
/// has one context and so one queue; several contexts submitting at once in
/// one process (tests, tools) would overlap more and are not what the count
/// is for.
///
/// **When it is counted.** A pass's time is booked when its last feedback
/// arrives, not as it runs: a long prefill pass lands whole in the window its
/// feedback completes in. A pass that never completes (a fault, a pass
/// dropped before all its command buffers were committed) is not counted.
/// Spans with no usable timestamps (not finite, not after 0, ending before
/// they start) are left out, and so are spans no command buffer can have:
/// longer than [`MAX_SPAN_SECS`], or ending more than [`FUTURE_SLACK_SECS`]
/// after the handler runs. One such span would otherwise move the set's
/// newest end far ahead, and every later interval would be forgotten as soon
/// as it is added, counting overlaps twice for the rest of the process.
///
/// The cost on the feedback path (a vector per pass, a sort and a lock held
/// for a merge into a set of about a second of intervals, on Metal's
/// completion thread) is not measured.
pub mod busy {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// How far back from the newest end seen the set keeps intervals, in
    /// seconds.
    pub const HORIZON_SECS: f64 = 1.0;
    /// The longest span one command buffer can plausibly have, in seconds
    /// (lily's longest, a prefill chunk's, takes well under one).
    pub const MAX_SPAN_SECS: f64 = 60.0;
    /// How far past the handler's own clock a span may end, in seconds: the
    /// feedback reports a buffer that has finished.
    pub const FUTURE_SLACK_SECS: f64 = 1.0;

    /// The busy time counted so far, in nanoseconds.
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    static RECENT: Mutex<Recent> = Mutex::new(Recent::new());

    /// The union of a pass's command-buffer spans `(start, end)` in seconds,
    /// sorted and disjoint (touching spans merged); spans without usable
    /// timestamps, or implausible at host time `now` (seconds, the clock of
    /// [`super::host_secs`]), are dropped.
    pub fn pass_union(spans: &[(f64, f64)], now: f64) -> Vec<(f64, f64)> {
        let mut valid: Vec<(f64, f64)> = spans
            .iter()
            .copied()
            .filter(|&(start, end)| {
                start.is_finite()
                    && end.is_finite()
                    && start > 0.0
                    && end > start
                    && end - start <= MAX_SPAN_SECS
                    && end <= now + FUTURE_SLACK_SECS
            })
            .collect();
        valid.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut union: Vec<(f64, f64)> = Vec::with_capacity(valid.len());
        for (start, end) in valid {
            match union.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => union.push((start, end)),
            }
        }
        union
    }

    /// The intervals counted within the last [`HORIZON_SECS`], sorted and
    /// disjoint, and the newest end seen.
    #[derive(Debug, Default)]
    pub struct Recent {
        intervals: Vec<(f64, f64)>,
        newest_end: f64,
    }

    impl Recent {
        pub const fn new() -> Self {
            Self { intervals: Vec::new(), newest_end: 0.0 }
        }

        /// Adds a pass's union (from [`pass_union`]: sorted, disjoint,
        /// valid) and returns the seconds of it the set did not cover yet;
        /// then forgets what ended more than [`HORIZON_SECS`] before the
        /// newest end.
        pub fn add(&mut self, union: &[(f64, f64)]) -> f64 {
            let mut added = 0.0;
            for &(start, end) in union {
                let mut covered = 0.0;
                let (mut lo, mut hi) = (start, end);
                self.intervals.retain(|&(a, b)| {
                    if b < start || a > end {
                        return true;
                    }
                    covered += b.min(end) - a.max(start);
                    lo = lo.min(a);
                    hi = hi.max(b);
                    false
                });
                added += (end - start - covered).max(0.0);
                let at = self.intervals.partition_point(|&(a, _)| a < lo);
                self.intervals.insert(at, (lo, hi));
                self.newest_end = self.newest_end.max(end);
            }
            let horizon = self.newest_end - HORIZON_SECS;
            self.intervals.retain(|&(_, b)| b >= horizon);
            added
        }

        /// The intervals the set holds (tests).
        pub fn intervals(&self) -> &[(f64, f64)] {
            &self.intervals
        }
    }

    /// Counts a completed pass's command-buffer spans.
    pub(super) fn record(spans: &[(f64, f64)]) {
        let union = pass_union(spans, super::host_secs());
        if union.is_empty() {
            return;
        }
        let added = RECENT.lock().unwrap_or_else(|e| e.into_inner()).add(&union);
        let ns = (added * 1e9).round();
        if ns >= 1.0 {
            // Saturating: a counter that wrapped would read as one that went
            // backwards, and the sampler would drop that window's figure.
            let _ = TOTAL_NS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                Some(t.saturating_add(ns as u64))
            });
        }
    }

    /// lily's GPU busy time since the process started, in nanoseconds.
    pub fn total_ns() -> u64 {
        TOTAL_NS.load(Ordering::Relaxed)
    }
}

/// lily's GPU busy time since the process started, in nanoseconds; see
/// [`busy`].
pub fn gpu_busy_ns() -> u64 {
    busy::total_ns()
}
