use chess_rs_lib::time_management::{iteration_time_decision, IterationTiming};

fn timing(
    elapsed_seconds: f64,
    iteration_seconds: f64,
    previous_iteration_seconds: f64,
    score_change_cp: i32,
    stable_iterations: u32,
    best_move_effort: f64,
    worker_disagreement: f64,
) -> IterationTiming {
    IterationTiming {
        elapsed_seconds,
        iteration_seconds,
        previous_iteration_seconds,
        score_change_cp,
        stable_iterations,
        best_move_effort,
        worker_disagreement,
    }
}

#[test]
fn settled_concentrated_search_can_finish_below_its_nominal_budget() {
    let early = iteration_time_decision(10.0, 40.0, 30, timing(6.0, 2.0, 0.6, 20, 4, 0.90, 0.0));
    let near_target =
        iteration_time_decision(10.0, 40.0, 30, timing(6.8, 2.0, 0.6, 20, 4, 0.90, 0.0));

    assert!(early.target_seconds < 10.0);
    assert!(!early.stop);
    assert!(near_target.stop);
}

#[test]
fn predicted_overrun_can_stop_near_the_nominal_budget() {
    let decision = iteration_time_decision(10.0, 40.0, 30, timing(9.1, 2.0, 0.6, 20, 3, 0.80, 0.0));

    assert!(decision.predicted_next_seconds > decision.target_seconds - 9.1);
    assert!(decision.stop);
}

#[test]
fn fast_search_does_not_stop_before_its_nominal_budget() {
    let decision = iteration_time_decision(
        0.100,
        0.400,
        30,
        timing(0.091, 0.020, 0.006, 20, 3, 0.80, 0.0),
    );

    assert!(decision.predicted_next_seconds > decision.target_seconds - 0.091);
    assert!(!decision.stop);
}

#[test]
fn fast_search_can_use_a_partial_iteration_before_the_hard_limit() {
    let decision = iteration_time_decision(
        0.100,
        0.120,
        30,
        timing(0.090, 0.020, 0.006, 160, 0, 0.50, 0.0),
    );

    assert!(0.090 + decision.predicted_next_seconds > 0.120);
    assert!(!decision.stop);
}

#[test]
fn iteration_that_cannot_finish_before_the_hard_limit_is_not_started() {
    let decision =
        iteration_time_decision(10.0, 40.0, 30, timing(9.0, 10.0, 2.0, 240, 0, 0.40, 1.0));

    assert_eq!(decision.predicted_next_seconds, 40.0);
    assert!(decision.stop);
}

#[test]
fn volatile_search_can_use_more_than_the_nominal_budget() {
    let decision =
        iteration_time_decision(10.0, 40.0, 30, timing(6.0, 2.0, 0.6, 240, 0, 0.40, 1.0));

    assert_eq!(decision.target_seconds, 11.5);
    assert!(!decision.stop);
}

#[test]
fn bullet_search_can_finish_a_valuable_iteration() {
    let decision = iteration_time_decision(
        0.010,
        0.035,
        30,
        timing(0.0096, 0.005, 0.002, 240, 0, 0.40, 1.0),
    );

    assert_eq!(decision.target_seconds, 0.0165);
    assert!(!decision.stop);
}

#[test]
fn worker_disagreement_extends_the_search_target() {
    let agreed = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 1, 0.90, 0.0));
    let split = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 1, 0.90, 1.0));

    assert!(split.target_seconds > agreed.target_seconds);
}

#[test]
fn settled_worker_disagreement_prevents_an_early_finish() {
    let agreed = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 4, 0.80, 0.0));
    let split = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 4, 0.80, 1.0));

    assert!(agreed.target_seconds < 10.0);
    assert_eq!(split.target_seconds, 10.0);
}

#[test]
fn fixed_movetime_is_not_shortened_by_iteration_prediction() {
    let decision = iteration_time_decision(0.5, 0.5, 30, timing(0.3, 0.25, 0.05, 0, 8, 0.95, 0.0));

    assert_eq!(decision.target_seconds, 0.5);
    assert!(!decision.stop);
}

#[test]
fn forced_move_search_has_a_half_second_ceiling() {
    let before_ceiling =
        iteration_time_decision(10.0, 40.0, 1, timing(0.49, 0.1, 0.05, 240, 0, 0.10, 1.0));
    let after_ceiling =
        iteration_time_decision(10.0, 40.0, 1, timing(0.51, 0.1, 0.05, 240, 0, 0.10, 1.0));

    assert_eq!(before_ceiling.target_seconds, 0.5);
    assert!(!before_ceiling.stop);
    assert!(after_ceiling.stop);
}
