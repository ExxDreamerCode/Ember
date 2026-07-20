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
fn stable_search_keeps_its_nominal_budget_before_predicting() {
    let before_soft_limit =
        iteration_time_decision(10.0, 40.0, 30, timing(6.0, 2.0, 0.6, 20, 4, 0.90, 0.0));
    let after_soft_limit =
        iteration_time_decision(10.0, 40.0, 30, timing(10.1, 2.0, 0.6, 20, 4, 0.90, 0.0));

    assert!(before_soft_limit.predicted_next_seconds > before_soft_limit.target_seconds - 6.0);
    assert!(!before_soft_limit.stop);
    assert!(after_soft_limit.stop);
}

#[test]
fn volatile_search_can_use_more_than_the_nominal_budget() {
    let decision =
        iteration_time_decision(10.0, 40.0, 30, timing(6.0, 2.0, 0.6, 240, 0, 0.40, 1.0));

    assert_eq!(decision.target_seconds, 11.5);
    assert!(!decision.stop);
}

#[test]
fn worker_disagreement_extends_the_search_target() {
    let agreed = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 1, 0.90, 0.0));
    let split = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 1, 0.90, 1.0));

    assert!(split.target_seconds > agreed.target_seconds);
}

#[test]
fn settled_worker_disagreement_does_not_inflate_the_budget() {
    let agreed = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 4, 0.80, 0.0));
    let split = iteration_time_decision(10.0, 40.0, 30, timing(4.0, 1.0, 0.5, 20, 4, 0.80, 1.0));

    assert_eq!(split.target_seconds, agreed.target_seconds);
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
