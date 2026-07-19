use chess_rs_lib::time_management::{threads_for_time_budget, TimeManager};

#[test]
fn one_second_bullet_has_no_fifty_millisecond_floor() {
    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(1_000.0, 10.0, 0, 0);

    assert!(
        (0.005..=0.015).contains(&budget.soft_seconds),
        "1+0.01 soft budget is not sustainable: {budget:?}"
    );
    assert!(
        budget.hard_seconds <= 0.035,
        "1+0.01 hard budget leaves too little clock reserve: {budget:?}"
    );
    assert_eq!(threads_for_time_budget(12, budget.soft_seconds), 1);
}

#[test]
fn hard_limit_never_borrows_the_next_increment() {
    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(20.0, 10.0, 0, 80);
    let spendable_seconds = (20.0 - manager.move_overhead_ms()) / 1_000.0;

    assert!(budget.soft_seconds <= budget.hard_seconds);
    assert!(
        budget.hard_seconds <= spendable_seconds,
        "hard limit exceeds the clock available before increment: {budget:?}"
    );
}

#[test]
fn testcorr_late_game_budget_falls_below_the_increment() {
    let mut manager = TimeManager::default();
    let _initial = manager.clock_budget(8_000.0, 80.0, 0, 0);
    let late = manager.clock_budget(500.0, 80.0, 0, 120);

    assert!(
        late.soft_seconds <= 0.080,
        "late 8+0.08 allocation still drains the clock: {late:?}"
    );
    assert!(threads_for_time_budget(12, late.soft_seconds) <= 4);
}

#[test]
fn stockfish_scaling_uses_the_actual_game_ply() {
    let mut opening_manager = TimeManager::default();
    let opening = opening_manager.clock_budget(8_000.0, 80.0, 0, 0);
    let mut middlegame_manager = TimeManager::default();
    let middlegame = middlegame_manager.clock_budget(8_000.0, 80.0, 0, 80);

    assert!(
        middlegame.soft_seconds > opening.soft_seconds,
        "game-ply scaling was lost: opening={opening:?}, middlegame={middlegame:?}"
    );

    let mut opening_manager = TimeManager::default();
    let opening = opening_manager.clock_budget(60_000.0, 0.0, 20, 0);
    let mut middlegame_manager = TimeManager::default();
    let middlegame = middlegame_manager.clock_budget(60_000.0, 0.0, 20, 80);
    assert!(middlegame.soft_seconds > opening.soft_seconds);
}

#[test]
fn move_overhead_rejects_invalid_values() {
    let mut manager = TimeManager::default();
    assert!(!manager.set_move_overhead_ms(-1.0));
    assert!(!manager.set_move_overhead_ms(f64::NAN));
    assert!(!manager.set_move_overhead_ms(5_001.0));
    assert_eq!(manager.move_overhead_ms(), 7.0);
    assert!(manager.set_move_overhead_ms(25.0));
    assert_eq!(manager.move_overhead_ms(), 25.0);
}
