use super::*;

/// Each class sets the calling thread's own nice value and I/O priority
/// and leaves the other threads of the process as they were.
#[test]
fn lowering_changes_only_the_calling_thread() {
    let main_nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    for (class, nice, ioprio) in [
        (Class::Utility, 10, (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | 7),
        (Class::Background, 19, IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT),
    ] {
        let (took, got_nice, got_ioprio) = std::thread::spawn(move || {
            let took = lower_current_thread(class);
            // SAFETY: plain reads of the calling thread's settings.
            unsafe {
                (
                    took,
                    libc::getpriority(libc::PRIO_PROCESS, 0),
                    libc::syscall(libc::SYS_ioprio_get, IOPRIO_WHO_PROCESS, 0) as c_int,
                )
            }
        })
        .join()
        .unwrap();
        assert!(took, "{class:?}");
        // A test run under `nice` keeps the higher of the two.
        assert_eq!((got_nice, got_ioprio), (nice.max(main_nice), ioprio), "{class:?}");
    }
    assert_eq!(unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) }, main_nice);
}
