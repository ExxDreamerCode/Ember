use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlineTimerBackend {
    Portable,
    WindowsHighResolution,
    WindowsStandard,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeadlineStats {
    pub arms: u64,
    pub disarms: u64,
    pub expirations: u64,
    pub arm_failures: u64,
    pub last_lateness: Duration,
    pub maximum_lateness: Duration,
}

#[derive(Default)]
struct AtomicDeadlineStats {
    arms: AtomicU64,
    disarms: AtomicU64,
    expirations: AtomicU64,
    arm_failures: AtomicU64,
    last_lateness_ns: AtomicU64,
    maximum_lateness_ns: AtomicU64,
    diagnostics: bool,
}

impl AtomicDeadlineStats {
    fn snapshot(&self) -> DeadlineStats {
        DeadlineStats {
            arms: self.arms.load(Ordering::Relaxed),
            disarms: self.disarms.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            arm_failures: self.arm_failures.load(Ordering::Relaxed),
            last_lateness: Duration::from_nanos(self.last_lateness_ns.load(Ordering::Relaxed)),
            maximum_lateness: Duration::from_nanos(
                self.maximum_lateness_ns.load(Ordering::Relaxed),
            ),
        }
    }

    fn record_expiration(&self, id: u64, deadline: Instant, now: Instant) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
        let lateness = now
            .saturating_duration_since(deadline)
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        self.last_lateness_ns.store(lateness, Ordering::Relaxed);
        self.maximum_lateness_ns
            .fetch_max(lateness, Ordering::Relaxed);
        if self.diagnostics {
            eprintln!(
                "info string deadline watchdog expired id={id} deadline={deadline:?} actual={now:?} lateness_ns={lateness}"
            );
        }
    }
}

struct ActiveDeadline {
    id: u64,
    deadline: Instant,
    stopped: Arc<AtomicBool>,
}

#[derive(Default)]
struct DeadlineState {
    active: Option<ActiveDeadline>,
}

impl DeadlineState {
    /// Replaces the current registration. Returns true when the platform wait
    /// primitive must be armed, or false when the deadline expired immediately.
    fn replace(&mut self, next: ActiveDeadline, now: Instant, stats: &AtomicDeadlineStats) -> bool {
        stats.arms.fetch_add(1, Ordering::Relaxed);
        self.active = None;
        if next.deadline <= now {
            expire_at(next, stats, now);
            false
        } else {
            self.active = Some(next);
            true
        }
    }

    /// Removes only the matching registration so a stale disarm cannot cancel
    /// a replacement timer.
    fn disarm(&mut self, id: u64, stats: &AtomicDeadlineStats) -> bool {
        stats.disarms.fetch_add(1, Ordering::Relaxed);
        if self.active.as_ref().is_some_and(|current| current.id == id) {
            self.active = None;
            true
        } else {
            false
        }
    }

    /// Handles both real timer signals and spurious/early wakes. Returns true
    /// only after the active registration actually expires.
    fn expire_if_due(&mut self, now: Instant, stats: &AtomicDeadlineStats) -> bool {
        let Some(current) = self.active.as_ref() else {
            return false;
        };
        if current.deadline > now {
            return false;
        }
        expire_at(self.active.take().unwrap(), stats, now);
        true
    }

    fn deadline(&self) -> Option<Instant> {
        self.active.as_ref().map(|current| current.deadline)
    }

    #[cfg(windows)]
    fn take(&mut self) -> Option<ActiveDeadline> {
        self.active.take()
    }
}

enum Command {
    Arm {
        active: ActiveDeadline,
        acknowledgement: mpsc::SyncSender<io::Result<()>>,
    },
    Disarm {
        id: u64,
    },
    Shutdown,
}

struct WatchdogInner {
    command_tx: mpsc::Sender<Command>,
    notifier: platform::Notifier,
    worker: Mutex<Option<JoinHandle<()>>>,
    platform: platform::PlatformOwner,
    next_id: AtomicU64,
    stats: Arc<AtomicDeadlineStats>,
    backend: DeadlineTimerBackend,
}

impl WatchdogInner {
    fn send(&self, command: Command) -> io::Result<()> {
        self.command_tx
            .send(command)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "deadline watchdog stopped"))?;
        self.notifier.wake();
        Ok(())
    }
}

impl Drop for WatchdogInner {
    fn drop(&mut self) {
        let _ = self.command_tx.send(Command::Shutdown);
        self.notifier.wake();
        if let Ok(worker) = self.worker.get_mut() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
        // `platform` owns the native handles and drops them after the worker
        // has stopped using them.
        let _ = &self.platform;
    }
}

#[derive(Clone)]
pub struct DeadlineWatchdog {
    inner: Arc<WatchdogInner>,
}

impl fmt::Debug for DeadlineWatchdog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeadlineWatchdog")
            .field("backend", &self.inner.backend)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl DeadlineWatchdog {
    pub fn new() -> io::Result<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let platform = platform::PlatformOwner::new()?;
        let notifier = platform.notifier();
        let worker_platform = platform.worker_platform();
        let backend = platform.backend();
        let diagnostics = std::env::var_os("EMBER_DEADLINE_DIAGNOSTICS").is_some();
        let stats = Arc::new(AtomicDeadlineStats {
            diagnostics,
            ..AtomicDeadlineStats::default()
        });
        let worker_stats = Arc::clone(&stats);
        let worker = thread::Builder::new()
            .name("deadline-watchdog".into())
            .spawn(move || platform::run(command_rx, worker_platform, worker_stats))?;

        Ok(Self {
            inner: Arc::new(WatchdogInner {
                command_tx,
                notifier,
                worker: Mutex::new(Some(worker)),
                platform,
                next_id: AtomicU64::new(1),
                stats,
                backend,
            }),
        })
    }

    pub fn backend(&self) -> DeadlineTimerBackend {
        self.inner.backend
    }

    pub fn stats(&self) -> DeadlineStats {
        self.inner.stats.snapshot()
    }

    pub fn arm(
        &self,
        deadline: Instant,
        stopped: Arc<AtomicBool>,
    ) -> io::Result<DeadlineRegistration> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        if self.inner.stats.diagnostics {
            let remaining = deadline.saturating_duration_since(Instant::now());
            eprintln!(
                "info string deadline watchdog arm id={id} backend={:?} remaining_ns={}",
                self.inner.backend,
                remaining.as_nanos()
            );
        }
        let (acknowledgement, acknowledged) = mpsc::sync_channel(0);
        let command = Command::Arm {
            active: ActiveDeadline {
                id,
                deadline,
                stopped: Arc::clone(&stopped),
            },
            acknowledgement,
        };
        if let Err(error) = self.inner.send(command) {
            stopped.store(true, Ordering::Relaxed);
            self.inner
                .stats
                .arm_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }

        match acknowledged.recv() {
            Ok(Ok(())) => Ok(DeadlineRegistration {
                id,
                inner: Arc::clone(&self.inner),
                armed: true,
            }),
            Ok(Err(error)) => {
                stopped.store(true, Ordering::Relaxed);
                Err(error)
            }
            Err(_) => {
                stopped.store(true, Ordering::Relaxed);
                self.inner
                    .stats
                    .arm_failures
                    .fetch_add(1, Ordering::Relaxed);
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "deadline watchdog stopped before acknowledging the timer",
                ))
            }
        }
    }
}

pub struct DeadlineRegistration {
    id: u64,
    inner: Arc<WatchdogInner>,
    armed: bool,
}

impl fmt::Debug for DeadlineRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeadlineRegistration")
            .field("id", &self.id)
            .field("armed", &self.armed)
            .finish()
    }
}

impl DeadlineRegistration {
    pub fn disarm(mut self) {
        self.disarm_inner();
    }

    fn disarm_inner(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let _ = self.inner.send(Command::Disarm { id: self.id });
    }
}

impl Drop for DeadlineRegistration {
    fn drop(&mut self) {
        self.disarm_inner();
    }
}

fn expire_at(active: ActiveDeadline, stats: &AtomicDeadlineStats, now: Instant) {
    stats.record_expiration(active.id, active.deadline, now);
    active.stopped.store(true, Ordering::Relaxed);
}

#[cfg(any(windows, test))]
fn fail_registration(active: ActiveDeadline, stats: &AtomicDeadlineStats) {
    active.stopped.store(true, Ordering::Relaxed);
    stats.arm_failures.fetch_add(1, Ordering::Relaxed);
}

#[cfg(windows)]
fn acknowledge_failure(
    active: ActiveDeadline,
    acknowledgement: mpsc::SyncSender<io::Result<()>>,
    stats: &AtomicDeadlineStats,
    error: io::Error,
) {
    fail_registration(active, stats);
    let _ = acknowledgement.send(Err(error));
}

#[cfg(not(windows))]
mod platform {
    use super::*;

    #[derive(Clone, Copy)]
    pub(super) struct Notifier;

    impl Notifier {
        pub(super) fn wake(self) {}
    }

    pub(super) struct PlatformOwner;
    pub(super) struct WorkerPlatform;

    impl PlatformOwner {
        pub(super) fn new() -> io::Result<Self> {
            Ok(Self)
        }

        pub(super) fn notifier(&self) -> Notifier {
            Notifier
        }

        pub(super) fn worker_platform(&self) -> WorkerPlatform {
            WorkerPlatform
        }

        pub(super) fn backend(&self) -> DeadlineTimerBackend {
            DeadlineTimerBackend::Portable
        }
    }

    pub(super) fn run(
        command_rx: mpsc::Receiver<Command>,
        _platform: WorkerPlatform,
        stats: Arc<AtomicDeadlineStats>,
    ) {
        let mut state = DeadlineState::default();
        loop {
            let command = if let Some(deadline) = state.deadline() {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    state.expire_if_due(Instant::now(), &stats);
                    continue;
                }
                match command_rx.recv_timeout(remaining) {
                    Ok(command) => command,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        state.expire_if_due(Instant::now(), &stats);
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match command_rx.recv() {
                    Ok(command) => command,
                    Err(_) => break,
                }
            };

            match command {
                Command::Arm {
                    active: next,
                    acknowledgement,
                } => {
                    state.replace(next, Instant::now(), &stats);
                    let _ = acknowledgement.send(Ok(()));
                }
                Command::Disarm { id } => {
                    state.disarm(id, &stats);
                }
                Command::Shutdown => break,
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ptr;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_FAILED, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        CancelWaitableTimer, CreateEventW, CreateWaitableTimerExW, SetEvent, SetWaitableTimerEx,
        WaitForMultipleObjects, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS,
    };

    #[derive(Clone, Copy)]
    pub(super) struct Notifier {
        command_event: usize,
    }

    impl Notifier {
        pub(super) fn wake(self) {
            unsafe {
                SetEvent(self.command_event as HANDLE);
            }
        }
    }

    pub(super) struct PlatformOwner {
        command_event: usize,
        timer: usize,
        backend: DeadlineTimerBackend,
    }

    pub(super) struct WorkerPlatform {
        command_event: usize,
        timer: usize,
    }

    impl PlatformOwner {
        pub(super) fn new() -> io::Result<Self> {
            let command_event = unsafe { CreateEventW(ptr::null(), 0, 0, ptr::null()) };
            if command_event.is_null() {
                return Err(io::Error::last_os_error());
            }

            let mut backend = DeadlineTimerBackend::WindowsHighResolution;
            let mut timer = unsafe {
                CreateWaitableTimerExW(
                    ptr::null(),
                    ptr::null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS,
                )
            };
            if timer.is_null() {
                backend = DeadlineTimerBackend::WindowsStandard;
                timer = unsafe {
                    CreateWaitableTimerExW(ptr::null(), ptr::null(), 0, TIMER_ALL_ACCESS)
                };
            }
            if timer.is_null() {
                let error = io::Error::last_os_error();
                unsafe {
                    CloseHandle(command_event);
                }
                return Err(error);
            }

            Ok(Self {
                command_event: command_event as usize,
                timer: timer as usize,
                backend,
            })
        }

        pub(super) fn notifier(&self) -> Notifier {
            Notifier {
                command_event: self.command_event,
            }
        }

        pub(super) fn worker_platform(&self) -> WorkerPlatform {
            WorkerPlatform {
                command_event: self.command_event,
                timer: self.timer,
            }
        }

        pub(super) fn backend(&self) -> DeadlineTimerBackend {
            self.backend
        }
    }

    impl Drop for PlatformOwner {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.timer as HANDLE);
                CloseHandle(self.command_event as HANDLE);
            }
        }
    }

    fn arm_timer(timer: HANDLE, deadline: Instant) -> io::Result<()> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        let ticks = remaining.as_nanos().div_ceil(100).min(i64::MAX as u128) as i64;
        let due_time = -ticks.max(1);
        let armed =
            unsafe { SetWaitableTimerEx(timer, &due_time, 0, None, ptr::null(), ptr::null(), 0) };
        if armed == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn fail_closed_loop(
        command_rx: &mpsc::Receiver<Command>,
        stats: &AtomicDeadlineStats,
        message: String,
    ) {
        while let Ok(command) = command_rx.recv() {
            match command {
                Command::Arm {
                    active,
                    acknowledgement,
                } => acknowledge_failure(
                    active,
                    acknowledgement,
                    stats,
                    io::Error::other(message.clone()),
                ),
                Command::Disarm { .. } => {
                    stats.disarms.fetch_add(1, Ordering::Relaxed);
                }
                Command::Shutdown => break,
            }
        }
    }

    fn process_commands(
        command_rx: &mpsc::Receiver<Command>,
        timer: HANDLE,
        state: &mut DeadlineState,
        stats: &AtomicDeadlineStats,
    ) -> bool {
        loop {
            let command = match command_rx.try_recv() {
                Ok(command) => command,
                Err(mpsc::TryRecvError::Empty) => return true,
                Err(mpsc::TryRecvError::Disconnected) => return false,
            };
            match command {
                Command::Arm {
                    active: next,
                    acknowledgement,
                } => {
                    unsafe {
                        CancelWaitableTimer(timer);
                    }
                    if !state.replace(next, Instant::now(), stats) {
                        let _ = acknowledgement.send(Ok(()));
                    } else {
                        let deadline = state.deadline().unwrap();
                        match arm_timer(timer, deadline) {
                            Ok(()) => {
                                let _ = acknowledgement.send(Ok(()));
                            }
                            Err(error) => {
                                let next = state.take().unwrap();
                                acknowledge_failure(next, acknowledgement, stats, error);
                            }
                        }
                    }
                }
                Command::Disarm { id } => {
                    if state.disarm(id, stats) {
                        unsafe {
                            CancelWaitableTimer(timer);
                        }
                    }
                }
                Command::Shutdown => return false,
            }
        }
    }

    pub(super) fn run(
        command_rx: mpsc::Receiver<Command>,
        platform: WorkerPlatform,
        stats: Arc<AtomicDeadlineStats>,
    ) {
        let command_event = platform.command_event as HANDLE;
        let timer = platform.timer as HANDLE;
        let handles = [command_event, timer];
        let mut state = DeadlineState::default();

        loop {
            let result = unsafe {
                WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 0, INFINITE)
            };
            if result == WAIT_OBJECT_0 {
                if !process_commands(&command_rx, timer, &mut state, &stats) {
                    break;
                }
            } else if result == WAIT_OBJECT_0 + 1 {
                let now = Instant::now();
                if !state.expire_if_due(now, &stats) {
                    if let Some(deadline) = state.deadline() {
                        if let Err(error) = arm_timer(timer, deadline) {
                            let current = state.take().unwrap();
                            fail_registration(current, &stats);
                            fail_closed_loop(&command_rx, &stats, error.to_string());
                            break;
                        }
                    }
                }
            } else if result == WAIT_FAILED {
                let had_active_registration = if let Some(current) = state.take() {
                    fail_registration(current, &stats);
                    true
                } else {
                    false
                };
                let error = io::Error::last_os_error();
                if !had_active_registration {
                    // Count a failed wait even when no registration was active.
                    stats.arm_failures.fetch_add(1, Ordering::Relaxed);
                }
                fail_closed_loop(&command_rx, &stats, error.to_string());
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(id: u64, deadline: Instant, stopped: &Arc<AtomicBool>) -> ActiveDeadline {
        ActiveDeadline {
            id,
            deadline,
            stopped: Arc::clone(stopped),
        }
    }

    #[test]
    fn fake_clock_future_and_past_deadlines_expire_at_the_boundary() {
        let epoch = Instant::now();
        let stats = AtomicDeadlineStats::default();
        let future = Arc::new(AtomicBool::new(false));
        let past = Arc::new(AtomicBool::new(false));
        let mut state = DeadlineState::default();

        assert!(state.replace(
            registration(1, epoch + Duration::from_secs(1), &future),
            epoch,
            &stats,
        ));
        assert!(!state.expire_if_due(epoch + Duration::from_millis(999), &stats));
        assert!(!future.load(Ordering::Relaxed));
        assert!(state.expire_if_due(epoch + Duration::from_secs(1), &stats));
        assert!(future.load(Ordering::Relaxed));

        assert!(!state.replace(
            registration(2, epoch - Duration::from_nanos(1), &past),
            epoch,
            &stats,
        ));
        assert!(past.load(Ordering::Relaxed));
        assert_eq!(stats.snapshot().expirations, 2);
    }

    #[test]
    fn fake_clock_disarm_and_stale_disarm_cannot_stop_a_replacement() {
        let epoch = Instant::now();
        let stats = AtomicDeadlineStats::default();
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut state = DeadlineState::default();

        state.replace(
            registration(1, epoch + Duration::from_secs(1), &first),
            epoch,
            &stats,
        );
        state.replace(
            registration(2, epoch + Duration::from_secs(2), &second),
            epoch,
            &stats,
        );
        assert!(!state.disarm(1, &stats));
        assert!(!state.expire_if_due(epoch + Duration::from_secs(1), &stats));
        assert!(!first.load(Ordering::Relaxed));
        assert!(!second.load(Ordering::Relaxed));
        assert!(state.disarm(2, &stats));
        assert!(!state.expire_if_due(epoch + Duration::from_secs(3), &stats));
        assert!(!second.load(Ordering::Relaxed));
    }

    #[test]
    fn fake_clock_expiration_race_can_only_touch_the_registered_token() {
        let epoch = Instant::now();
        let stats = AtomicDeadlineStats::default();
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut state = DeadlineState::default();

        state.replace(registration(1, epoch, &first), epoch, &stats);
        assert!(first.load(Ordering::Relaxed));
        state.replace(
            registration(2, epoch + Duration::from_secs(1), &second),
            epoch,
            &stats,
        );
        assert!(!state.disarm(1, &stats));
        assert!(!second.load(Ordering::Relaxed));
    }

    #[test]
    fn fake_clock_early_wake_keeps_the_registration_armed() {
        let epoch = Instant::now();
        let stats = AtomicDeadlineStats::default();
        let stopped = Arc::new(AtomicBool::new(false));
        let mut state = DeadlineState::default();
        let deadline = epoch + Duration::from_secs(10);

        state.replace(registration(1, deadline, &stopped), epoch, &stats);
        assert!(!state.expire_if_due(epoch + Duration::from_secs(1), &stats));
        assert_eq!(state.deadline(), Some(deadline));
        assert!(!stopped.load(Ordering::Relaxed));
    }

    #[test]
    fn fake_arm_failure_stops_safely_and_records_the_failure() {
        let stats = AtomicDeadlineStats::default();
        let stopped = Arc::new(AtomicBool::new(false));
        let active = registration(1, Instant::now(), &stopped);

        fail_registration(active, &stats);

        assert!(stopped.load(Ordering::Relaxed));
        assert_eq!(stats.snapshot().arm_failures, 1);
    }
}
