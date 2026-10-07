use ember_chess::time_management::{threads_for_time_budget, TimeManager};

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
fn one_second_bullet_keeps_a_scheduler_tail_reserve() {
    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(205.0, 10.0, 0, 200);

    assert!(budget.soft_seconds <= budget.hard_seconds);
    assert!(
        budget.hard_seconds <= 0.005,
        "low-clock hard budget consumed the scheduler reserve: {budget:?}"
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
fn budget_scaling_uses_the_actual_game_ply() {
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
fn f1w14oir_low_clock_keeps_an_increment_reserve() {
    let mut manager = TimeManager::default();
    let _opening = manager.clock_budget(180_000.0, 2_000.0, 0, 0);
    let before_move_29 = manager.clock_budget(24_350.0, 2_000.0, 0, 58);
    let before_move_42 = manager.clock_budget(10_110.0, 2_000.0, 0, 84);

    assert!(before_move_29.soft_seconds <= before_move_29.hard_seconds);
    assert!(before_move_29.hard_seconds <= 24.350 * 0.35 + 1e-9);
    assert!(before_move_42.soft_seconds <= before_move_42.hard_seconds);
    assert!(
        before_move_42.hard_seconds <= 4.110,
        "three future increments should remain on the clock: {before_move_42:?}"
    );
}

#[test]
fn p1uv2lqo_medium_clock_cannot_spend_most_of_the_remaining_time() {
    let mut manager = TimeManager::default();
    let _opening = manager.clock_budget(178_000.0, 2_000.0, 0, 2);
    let before_move_17 = manager.clock_budget(105_360.0, 2_000.0, 0, 32);
    let before_move_20 = manager.clock_budget(52_120.0, 2_000.0, 0, 38);

    assert!(before_move_17.soft_seconds <= before_move_17.hard_seconds);
    assert!(before_move_17.hard_seconds <= 105.360 * 0.35 + 1e-9);
    assert!(before_move_20.soft_seconds <= before_move_20.hard_seconds);
    assert!(before_move_20.hard_seconds <= 52.120 * 0.35 + 1e-9);
}

/// Numeric anchors of the calibrated clock-planning model: a deliberate
/// constant change re-pins them together with a clocked match run.
#[test]
fn calibrated_budgets_anchor_the_clock_model() {
    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(1_000.0, 10.0, 0, 0);
    assert!(
        (budget.soft_seconds - 0.011_560_072_225).abs() < 1e-6,
        "1s+10ms opening drifted: {budget:?}"
    );
    assert!((budget.hard_seconds - 0.035).abs() < 1e-9, "{budget:?}");

    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(8_000.0, 80.0, 0, 0);
    assert!(
        (budget.soft_seconds - 0.189_856_476_944).abs() < 1e-6,
        "8s+80ms opening drifted: {budget:?}"
    );
    assert!(
        (budget.hard_seconds - 1.121_642_284_376).abs() < 1e-6,
        "8s+80ms opening drifted: {budget:?}"
    );

    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(8_000.0, 80.0, 0, 80);
    assert!(
        (budget.soft_seconds - 0.415_918_142_514).abs() < 1e-6,
        "8s+80ms middlegame drifted: {budget:?}"
    );
    assert!(
        (budget.hard_seconds - 2.869_835_183_335).abs() < 1e-6,
        "8s+80ms middlegame drifted: {budget:?}"
    );

    let mut manager = TimeManager::default();
    let _opening = manager.clock_budget(180_000.0, 2_000.0, 0, 0);
    let budget = manager.clock_budget(24_350.0, 2_000.0, 0, 58);
    assert!(
        (budget.soft_seconds - 5.978_880_039_648).abs() < 1e-6,
        "late-game anchored budget drifted: {budget:?}"
    );
    assert!((budget.hard_seconds - 8.522_5).abs() < 1e-6, "{budget:?}");

    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(60_000.0, 0.0, 20, 0);
    assert!(
        (budget.soft_seconds - 2.602_996_5).abs() < 1e-6,
        "announced-segment opening drifted: {budget:?}"
    );
    assert!(
        (budget.hard_seconds - 9.240_637_575).abs() < 1e-6,
        "announced-segment opening drifted: {budget:?}"
    );

    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(60_000.0, 0.0, 20, 80);
    assert!(
        (budget.soft_seconds - 4.597_629_833_333).abs() < 1e-6,
        "announced-segment middlegame drifted: {budget:?}"
    );
    assert!(
        (budget.hard_seconds - 16.321_585_908_333).abs() < 1e-6,
        "announced-segment middlegame drifted: {budget:?}"
    );

    let mut manager = TimeManager::default();
    let budget = manager.clock_budget(205.0, 10.0, 0, 200);
    assert!(
        (budget.soft_seconds - 0.003_650_326_199).abs() < 1e-6,
        "sub-second bullet budget drifted: {budget:?}"
    );
    assert!((budget.hard_seconds - 0.005).abs() < 1e-9, "{budget:?}");
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
