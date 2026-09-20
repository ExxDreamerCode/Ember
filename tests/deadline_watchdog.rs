use ember_chess::deadline::DeadlineWatchdog;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn wait_until_set(stopped: &AtomicBool, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if stopped.load(Ordering::Relaxed) {
            return true;
        }
        thread::yield_now();
    }
    stopped.load(Ordering::Relaxed)
}

#[test]
fn already_expired_deadline_is_stopped_before_arm_returns() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let stopped = Arc::new(AtomicBool::new(false));

    let registration = watchdog
        .arm(
            Instant::now() - Duration::from_millis(1),
            Arc::clone(&stopped),
        )
        .unwrap();

    assert!(stopped.load(Ordering::Relaxed));
    drop(registration);
}

#[test]
fn future_deadline_sets_the_supplied_token() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let stopped = Arc::new(AtomicBool::new(false));

    let registration = watchdog
        .arm(
            Instant::now() + Duration::from_millis(20),
            Arc::clone(&stopped),
        )
        .unwrap();

    assert!(!stopped.load(Ordering::Relaxed));
    assert!(wait_until_set(&stopped, Duration::from_secs(2)));
    drop(registration);
    assert_eq!(watchdog.stats().expirations, 1);
}

#[test]
fn disarmed_deadline_does_not_stop_its_token() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let stopped = Arc::new(AtomicBool::new(false));

    watchdog
        .arm(
            Instant::now() + Duration::from_millis(200),
            Arc::clone(&stopped),
        )
        .unwrap()
        .disarm();
    thread::sleep(Duration::from_millis(300));

    assert!(!stopped.load(Ordering::Relaxed));
}

#[test]
fn stale_registration_cannot_stop_the_next_search() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let first = Arc::new(AtomicBool::new(false));
    let second = Arc::new(AtomicBool::new(false));

    let first_registration = watchdog
        .arm(
            Instant::now() + Duration::from_millis(40),
            Arc::clone(&first),
        )
        .unwrap();
    drop(first_registration);
    let second_registration = watchdog
        .arm(Instant::now() + Duration::from_secs(1), Arc::clone(&second))
        .unwrap();
    thread::sleep(Duration::from_millis(100));

    assert!(!second.load(Ordering::Relaxed));
    drop(second_registration);
}

#[test]
fn repeated_arm_and_disarm_cycles_remain_usable() {
    let watchdog = DeadlineWatchdog::new().unwrap();

    for _ in 0..100 {
        let stopped = Arc::new(AtomicBool::new(false));
        watchdog
            .arm(
                Instant::now() + Duration::from_secs(1),
                Arc::clone(&stopped),
            )
            .unwrap()
            .disarm();
        assert!(!stopped.load(Ordering::Relaxed));
    }

    let final_token = Arc::new(AtomicBool::new(false));
    let registration = watchdog
        .arm(Instant::now(), Arc::clone(&final_token))
        .unwrap();
    assert!(final_token.load(Ordering::Relaxed));
    drop(registration);
    assert_eq!(watchdog.stats().arms, 101);
}

#[test]
fn arming_never_clears_an_external_stop() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let stopped = Arc::new(AtomicBool::new(true));

    let registration = watchdog
        .arm(
            Instant::now() + Duration::from_secs(1),
            Arc::clone(&stopped),
        )
        .unwrap();

    assert!(stopped.load(Ordering::Relaxed));
    drop(registration);
}

#[test]
fn dropping_the_final_owner_wakes_and_joins_the_worker() {
    let watchdog = DeadlineWatchdog::new().unwrap();
    let stopped = Arc::new(AtomicBool::new(false));
    let registration = watchdog
        .arm(
            Instant::now() + Duration::from_secs(60),
            Arc::clone(&stopped),
        )
        .unwrap();
    let started = Instant::now();

    drop(registration);
    drop(watchdog);

    assert!(
        started.elapsed() < Duration::from_secs(2),
        "watchdog shutdown did not wake its sleeping worker"
    );
    assert!(!stopped.load(Ordering::Relaxed));
}
