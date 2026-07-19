const DEFAULT_MOVE_OVERHEAD_MS: f64 = 7.0;
const MAX_MOVE_OVERHEAD_MS: f64 = 5_000.0;
const MAX_SEARCH_TIME_MS: f64 = 60_000.0;
const SINGLE_THREAD_BUDGET_MS: f64 = 25.0;
const REDUCED_SMP_BUDGET_MS: f64 = 100.0;
const REDUCED_SMP_THREADS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct TimeBudget {
    pub soft_seconds: f64,
    pub hard_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    move_overhead_ms: f64,
    original_time_adjust: Option<f64>,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
            original_time_adjust: None,
        }
    }
}

impl TimeManager {
    pub fn move_overhead_ms(&self) -> f64 {
        self.move_overhead_ms
    }

    pub fn set_move_overhead_ms(&mut self, value: f64) -> bool {
        if !value.is_finite() || !(0.0..=MAX_MOVE_OVERHEAD_MS).contains(&value) {
            return false;
        }
        self.move_overhead_ms = value;
        true
    }

    pub fn reset_for_new_game(&mut self) {
        self.original_time_adjust = None;
    }

    pub fn clock_budget(
        &mut self,
        remaining_ms: f64,
        increment_ms: f64,
        moves_to_go: i32,
        ply: usize,
    ) -> TimeBudget {
        let time_ms = remaining_ms.max(0.0);
        let increment_ms = increment_ms.max(0.0);
        let overhead_ms = self.move_overhead_ms;
        let mut move_horizon = if moves_to_go > 0 {
            moves_to_go.min(50) as f64
        } else {
            50.0
        };
        if time_ms < 1_000.0 {
            move_horizon = (time_ms * 0.05).floor();
        }

        let time_left_ms = (time_ms + increment_ms * (move_horizon - 1.0)
            - overhead_ms * (move_horizon + 2.0))
            .max(1.0);
        let ply = ply as f64;

        let (opt_scale, max_scale) = if moves_to_go > 0 {
            let horizon_scale = if move_horizon > 0.0 {
                (0.88 + ply / 116.4) / move_horizon
            } else {
                f64::INFINITY
            };
            (
                horizon_scale.min(0.88 * time_ms / time_left_ms),
                1.3 + 0.11 * move_horizon,
            )
        } else {
            let original_time_adjust = *self
                .original_time_adjust
                .get_or_insert_with(|| 0.3272 * time_left_ms.log10() - 0.4141);
            let log_time_seconds = (time_ms.max(1.0) / 1_000.0).log10();
            let opt_constant = (0.0029869 + 0.00033554 * log_time_seconds).min(0.004905);
            let max_constant = (3.3744 + 3.0608 * log_time_seconds).max(3.1441);
            let opt = (0.012112 + (ply + 3.22713).powf(0.46866) * opt_constant)
                .min(0.19404 * time_ms / time_left_ms)
                * original_time_adjust;
            (opt, (max_constant + ply / 12.352).min(6.873))
        };

        let optimum_ms = (opt_scale * time_left_ms).max(1.0);
        let maximum_ms =
            optimum_ms.max((0.8097 * time_ms - overhead_ms).min(max_scale * optimum_ms));

        // The GUI cannot grant the next increment before this move returns.
        // Keep both limits inside the current clock after communication slack.
        let spendable_ms = (time_ms - overhead_ms).max(1.0);
        let mut soft_ms = optimum_ms.min(spendable_ms).min(MAX_SEARCH_TIME_MS);
        if time_ms < 1_000.0 && increment_ms > 0.0 {
            soft_ms = soft_ms.min(increment_ms);
        }
        let short_increment_hard_cap_ms = if time_ms <= 1_000.0 && increment_ms <= 10.0 {
            35.0
        } else {
            MAX_SEARCH_TIME_MS
        };
        let hard_ms = maximum_ms
            .min(spendable_ms)
            .min(MAX_SEARCH_TIME_MS)
            .min(short_increment_hard_cap_ms)
            .max(soft_ms);

        TimeBudget {
            soft_seconds: soft_ms / 1_000.0,
            hard_seconds: hard_ms / 1_000.0,
        }
    }
}

pub fn threads_for_time_budget(configured_threads: usize, soft_seconds: f64) -> usize {
    let configured_threads = configured_threads.max(1);
    let soft_ms = soft_seconds * 1_000.0;
    if soft_ms < SINGLE_THREAD_BUDGET_MS {
        1
    } else if soft_ms < REDUCED_SMP_BUDGET_MS {
        configured_threads.min(REDUCED_SMP_THREADS)
    } else {
        configured_threads
    }
}
