const DEFAULT_MOVE_OVERHEAD_MS: f64 = 7.0;
const MAX_MOVE_OVERHEAD_MS: f64 = 5_000.0;
const MAX_SEARCH_TIME_MS: f64 = 60_000.0;
const SINGLE_THREAD_BUDGET_MS: f64 = 25.0;
const REDUCED_SMP_BUDGET_MS: f64 = 100.0;
const EARLY_PREDICTION_BUDGET_MS: f64 = 500.0;
const REDUCED_SMP_THREADS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct TimeBudget {
    pub soft_seconds: f64,
    pub hard_seconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct IterationTiming {
    pub elapsed_seconds: f64,
    pub iteration_seconds: f64,
    pub previous_iteration_seconds: f64,
    pub score_change_cp: i32,
    pub stable_iterations: u32,
    pub best_move_effort: f64,
    pub worker_disagreement: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct IterationDecision {
    pub target_seconds: f64,
    pub predicted_next_seconds: f64,
    pub stop: bool,
}

pub fn iteration_time_decision(
    soft_seconds: f64,
    hard_seconds: f64,
    legal_moves: usize,
    timing: IterationTiming,
) -> IterationDecision {
    let soft_seconds = soft_seconds.max(0.0).min(hard_seconds);
    let adaptive = hard_seconds > soft_seconds * 1.05 + 0.001;
    let growth = if timing.previous_iteration_seconds > 0.0 {
        (timing.iteration_seconds / timing.previous_iteration_seconds).clamp(1.5, 4.0)
    } else {
        2.0
    };
    let predicted_next_seconds = (timing.iteration_seconds * growth).max(0.0);

    if !adaptive {
        return IterationDecision {
            target_seconds: soft_seconds,
            predicted_next_seconds,
            stop: timing.elapsed_seconds >= soft_seconds,
        };
    }

    let stability = match timing.stable_iterations {
        0 => 1.30,
        1 => 1.15,
        2 => 1.00,
        3 => 0.88,
        _ => 0.75,
    };
    let score_volatility = 1.0 + (f64::from(timing.score_change_cp.abs()) / 240.0).clamp(0.0, 0.35);
    let effort = timing.best_move_effort.clamp(0.0, 1.0);
    let effort_factor = if effort >= 0.90 {
        0.80
    } else if effort >= 0.75 {
        0.90
    } else if effort < 0.45 {
        1.10
    } else {
        1.0
    };
    let disagreement = timing.worker_disagreement.clamp(0.0, 1.0);
    let unsettled = timing.stable_iterations < 2 || timing.score_change_cp.abs() > 80;
    let disagreement_factor = if unsettled {
        1.0 + 0.15 * disagreement
    } else {
        1.0
    };
    let soft_ms = soft_seconds * 1_000.0;
    let can_finish_early = soft_ms >= EARLY_PREDICTION_BUDGET_MS
        && timing.stable_iterations >= 3
        && timing.score_change_cp.abs() <= 35
        && effort >= 0.70
        && disagreement <= 0.25;
    let minimum_scale = if can_finish_early { 0.70 } else { 1.0 };
    let maximum_scale = if soft_ms <= SINGLE_THREAD_BUDGET_MS {
        1.65
    } else {
        1.15
    };
    let scale = (stability * score_volatility * effort_factor * disagreement_factor)
        .clamp(minimum_scale, maximum_scale);
    let mut target_seconds = (soft_seconds * scale).min(hard_seconds);
    if legal_moves == 1 {
        target_seconds = target_seconds.min(0.5).min(hard_seconds);
    }

    let stable_enough =
        timing.stable_iterations >= 2 && timing.score_change_cp.abs() <= 80 && disagreement <= 0.5;
    let prediction_floor = if stable_enough { 0.90 } else { 0.95 };
    let predicted_target_overrun =
        predicted_next_seconds >= (target_seconds - timing.elapsed_seconds).max(0.0);
    let prediction_boundary = if soft_ms < EARLY_PREDICTION_BUDGET_MS {
        stable_enough && timing.elapsed_seconds >= soft_seconds && predicted_target_overrun
    } else if can_finish_early {
        timing.elapsed_seconds >= target_seconds * 0.95 && predicted_target_overrun
    } else {
        timing.elapsed_seconds >= soft_seconds * prediction_floor && predicted_target_overrun
    };
    let hard_boundary = soft_ms >= EARLY_PREDICTION_BUDGET_MS
        && timing.previous_iteration_seconds > 0.0
        && timing.elapsed_seconds + predicted_next_seconds >= hard_seconds;
    let stop = timing.elapsed_seconds >= target_seconds || hard_boundary || prediction_boundary;

    IterationDecision {
        target_seconds,
        predicted_next_seconds,
        stop,
    }
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
        // Start protecting several future increments while there is still
        // enough clock to recover; waiting until the last few moves permits
        // one unstable iteration to consume most of an otherwise safe clock.
        let increment_reserve_cap_ms =
            if increment_ms > 0.0 && time_ms <= increment_ms * 60.0 && time_ms > 1_000.0 {
                ((time_ms - increment_ms * 3.0).max(1.0)).min(time_ms * 0.35)
            } else {
                MAX_SEARCH_TIME_MS
            };
        let mut soft_ms = optimum_ms
            .min(spendable_ms)
            .min(MAX_SEARCH_TIME_MS)
            .min(increment_reserve_cap_ms);
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
            .min(increment_reserve_cap_ms)
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
