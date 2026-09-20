use std::sync::atomic::AtomicU64;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn rdtsc() -> u64 {
    #[cfg(all(feature = "search-perf", target_arch = "x86_64"))]
    {
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(all(feature = "search-perf", target_arch = "aarch64"))]
    {
        let counter: u64;
        // SAFETY: Linux and Android expose the architectural virtual counter
        // to EL0. It is a fixed-frequency tick source rather than a core-cycle
        // counter. ISB on both sides orders the read against the measured code;
        // omitting `nomem` also prevents compiler motion of memory accesses.
        unsafe {
            std::arch::asm!(
                "isb",
                "mrs {counter}, cntvct_el0",
                "isb",
                counter = out(reg) counter,
                options(nostack, preserves_flags)
            );
        }
        counter
    }
    #[cfg(any(
        not(feature = "search-perf"),
        not(any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    {
        0
    }
}

#[derive(Default)]
#[cfg_attr(not(feature = "search-perf"), allow(dead_code))]
pub(crate) struct PerfCounters {
    pub eval_cycles: AtomicU64,
    pub eval_calls: AtomicU64,
    pub accupd_cycles: AtomicU64,
    pub accupd_calls: AtomicU64,
    pub movegen_cycles: AtomicU64,
    pub movegen_calls: AtomicU64,
    pub ttget_cycles: AtomicU64,
    pub ttget_calls: AtomicU64,
    pub ttput_cycles: AtomicU64,
    pub ttput_calls: AtomicU64,
    pub see_cycles: AtomicU64,
    pub see_calls: AtomicU64,
    // SEE inside move scoring is already included in score_cycles. Keep it
    // separate from quiescence SEE so top-level attribution remains disjoint.
    #[cfg(feature = "search-perf")]
    pub scoring_see_cycles: AtomicU64,
    #[cfg(feature = "search-perf")]
    pub scoring_see_calls: AtomicU64,
    pub apply_cycles: AtomicU64,
    pub apply_calls: AtomicU64,
    pub draw_cycles: AtomicU64,
    pub draw_calls: AtomicU64,
    pub time_cycles: AtomicU64,
    pub time_calls: AtomicU64,
    pub score_cycles: AtomicU64,
    pub score_calls: AtomicU64,
    pub incheck_cycles: AtomicU64,
    pub incheck_calls: AtomicU64,
    pub nullcopy_cycles: AtomicU64,
    pub nullcopy_calls: AtomicU64,
}

#[cfg(feature = "search-perf")]
pub(crate) static THREAT_SCAN_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static THREAT_SCAN_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_SLOT_UPDATE_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_SLOT_UPDATE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_SCAN_UPDATE_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_SCAN_UPDATE_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REFRESH_UPDATE_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REFRESH_UPDATE_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "search-perf")]
pub(crate) static ACC_COPY_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_DIFF_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_PIECE_ROWS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REBUILD_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REBUILD_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_THREATDIFF_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_THREAT_ROWS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_THREATSORT_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REFRESH_CYCLES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "search-perf")]
pub(crate) static ACC_REFRESH_CALLS: AtomicU64 = AtomicU64::new(0);

macro_rules! perf_time {
    ($cycles:ident, $calls:ident, $self:expr, $call:expr) => {{
        #[cfg(feature = "search-perf")]
        let __perf_t0 = $crate::search::perf::rdtsc();
        let __perf_value = $call;
        #[cfg(feature = "search-perf")]
        {
            let __perf_dt = $crate::search::perf::rdtsc().wrapping_sub(__perf_t0);
            $self
                .perf
                .$cycles
                .fetch_add(__perf_dt, std::sync::atomic::Ordering::Relaxed);
            $self
                .perf
                .$calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        __perf_value
    }};
}

pub(crate) use perf_time;

macro_rules! perf_region_start {
    ($label:ident) => {
        #[cfg(feature = "search-perf")]
        let $label = $crate::search::perf::rdtsc();
    };
}

macro_rules! perf_region_end {
    ($cycles:ident, $calls:ident, $self:expr, $label:ident) => {
        #[cfg(feature = "search-perf")]
        {
            let __perf_dt = $crate::search::perf::rdtsc().wrapping_sub($label);
            $self
                .perf
                .$cycles
                .fetch_add(__perf_dt, std::sync::atomic::Ordering::Relaxed);
            $self
                .perf
                .$calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

pub(crate) use perf_region_end;
pub(crate) use perf_region_start;
