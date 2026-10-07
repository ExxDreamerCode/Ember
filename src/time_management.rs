const DEFAULT_MOVE_OVERHEAD_MS: f64 = 7.0;
const MAX_MOVE_OVERHEAD_MS: f64 = 5_000.0;
const MAX_SEARCH_TIME_MS: f64 = 60_000.0;
const SINGLE_THREAD_BUDGET_MS: f64 = 25.0;
const REDUCED_SMP_BUDGET_MS: f64 = 100.0;
const EARLY_PREDICTION_BUDGET_MS: f64 = 500.0;
// Nothing to choose with one legal move: at most a brief confirmation look.
const FORCED_MOVE_CEILING_SECONDS: f64 = 0.4;
const REDUCED_SMP_THREADS: usize = 4;
// A hard deadline can stop search work, but it cannot make a descheduled
// process return bestmove. Keep enough clock to absorb a short scheduler stall
// once an extreme-increment game reaches its low-clock tail.
const SHORT_INCREMENT_CLOCK_RESERVE_MS: f64 = 200.0;

const ASSUMED_MOVE_HORIZON: f64 = 50.0;
const MOVE_HORIZON_CAP: f64 = 50.0;
// Below one second, plan one move per 25ms instead of dust spread over 50.
const MIN_MOVE_SLICE_MS: f64 = 25.0;
// Command, reply, and one scheduler tick: slack the pool never lends out.
const PROTOCOL_TAIL_MOVES: f64 = 3.0;

const OPENING_POOL_SHARE: f64 = 0.0173;
const CLOCK_DECADE_SHARE: f64 = 0.00058;
const MATURITY_GAIN: f64 = 3.2;
const MATURITY_HALF_PLY: f64 = 135.0;
const TEMPO_DECADE: f64 = 0.32;
const TEMPO_REFERENCE_MS: f64 = 1_000.0;
const TEMPO_AT_REFERENCE: f64 = 0.575;
const TEMPO_CEILING: f64 = 1.75;
const CLOCK_CLAIM_CAP: f64 = 0.19;

// Hard headroom over the plan: HARD_BASE + HARD_REACH * t / (t + HARD_HALF)
// + HARD_PER_PLY * ply, capped at HARD_CEILING.
const HARD_BASE: f64 = 2.30;
const HARD_REACH: f64 = 4.60;
const HARD_HALF_SECONDS: f64 = 2.2;
const HARD_PER_PLY: f64 = 0.08;
const HARD_CEILING: f64 = 6.90;
// Absolute guard: one move never reaches this share of the current clock.
const HARD_CLOCK_SHARE: f64 = 0.80;

const FIXED_OPENING_SHARE: f64 = 0.87;
const FIXED_PLY_SPREAD: f64 = 120.0;
const FIXED_CLOCK_SHARE: f64 = 0.87;
const FIXED_HARD_BASE: f64 = 1.25;
const FIXED_HARD_PER_MOVE: f64 = 0.115;

fn pool_tempo(pool_ms: f64) -> f64 {
    (TEMPO_DECADE * (pool_ms / TEMPO_REFERENCE_MS).log10() + TEMPO_AT_REFERENCE)
        .clamp(0.0, TEMPO_CEILING)
}

fn opening_share(time_ms: f64) -> f64 {
    OPENING_POOL_SHARE + CLOCK_DECADE_SHARE * (time_ms.max(1.0) / 1_000.0).log10()
}

fn maturity(ply: f64) -> f64 {
    1.0 + MATURITY_GAIN * ply / (ply + MATURITY_HALF_PLY)
}

fn hard_multiple(time_ms: f64, ply: f64) -> f64 {
    let clock_seconds = time_ms.max(1.0) / 1_000.0;
    (HARD_BASE
        + HARD_REACH * clock_seconds / (clock_seconds + HARD_HALF_SECONDS)
        + HARD_PER_PLY * ply)
        .min(HARD_CEILING)
}

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
        target_seconds = target_seconds.min(FORCED_MOVE_CEILING_SECONDS);
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
    // Tempo anchor: the pool from the first open-ended budget of the game.
    opening_pool_ms: Option<f64>,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
            opening_pool_ms: None,
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
        self.opening_pool_ms = None;
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

        let announced_horizon = moves_to_go > 0;
        let mut move_horizon = if announced_horizon {
            (moves_to_go as f64).min(MOVE_HORIZON_CAP)
        } else {
            ASSUMED_MOVE_HORIZON
        };
        if time_ms < 1_000.0 {
            move_horizon = (time_ms / MIN_MOVE_SLICE_MS).floor();
        }

        // The returning move's own increment is already in the pool.
        let expected_increments = increment_ms * move_horizon;
        let protocol_slack = overhead_ms * (move_horizon + PROTOCOL_TAIL_MOVES);
        let pool_ms = (time_ms + expected_increments - protocol_slack).max(1.0);
        let ply = ply as f64;

        let (planned_ms, headroom) = if announced_horizon {
            let horizon_moves = move_horizon.max(1.0);
            let even_slice = FIXED_OPENING_SHARE / horizon_moves;
            let late_bonus = (ply / FIXED_PLY_SPREAD) / horizon_moves;
            let planned = ((even_slice + late_bonus) * pool_ms).min(FIXED_CLOCK_SHARE * time_ms);
            (
                planned,
                FIXED_HARD_BASE + FIXED_HARD_PER_MOVE * move_horizon,
            )
        } else {
            let tempo = pool_tempo(*self.opening_pool_ms.get_or_insert(pool_ms));
            let claim = opening_share(time_ms) * maturity(ply);
            let planned = (claim * pool_ms).min(CLOCK_CLAIM_CAP * time_ms) * tempo;
            (planned, hard_multiple(time_ms, ply))
        };

        let planned_ms = planned_ms.max(1.0);
        let maximum_ms =
            planned_ms.max((HARD_CLOCK_SHARE * time_ms - overhead_ms).min(headroom * planned_ms));

        // The GUI cannot grant the next increment before this move returns.
        // Keep both limits inside the current clock after communication slack.
        let clock_reserve_ms = if time_ms <= 1_000.0 && increment_ms <= 10.0 {
            overhead_ms.max(SHORT_INCREMENT_CLOCK_RESERVE_MS)
        } else {
            overhead_ms
        };
        let spendable_ms = (time_ms - clock_reserve_ms).max(1.0);
        // Start protecting several future increments while there is still
        // enough clock to recover; waiting until the last few moves permits
        // one unstable iteration to consume most of an otherwise safe clock.
        let increment_reserve_cap_ms =
            if increment_ms > 0.0 && time_ms <= increment_ms * 60.0 && time_ms > 1_000.0 {
                ((time_ms - increment_ms * 3.0).max(1.0)).min(time_ms * 0.35)
            } else {
                MAX_SEARCH_TIME_MS
            };
        let mut soft_ms = planned_ms
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
