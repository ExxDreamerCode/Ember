use super::*;
use crate::board::encode_move;
use crate::engine::Engine;

fn state_from_fen(fen: &str) -> BoardState {
    let mut engine = Engine::new();
    engine.set_fen(fen);
    engine.st
}

fn legal_move(st: &BoardState, uci: &str) -> Move {
    generate_moves(st, st.w, &st.cr, st.ep)
        .into_iter()
        .find(|mv| crate::board::move_to_uci(st, *mv) == uci)
        .unwrap_or_else(|| panic!("expected legal move {uci}"))
}

fn qualifying_singular_evidence(mv: Move) -> SingularEvidence {
    SingularEvidence {
        enabled: true,
        ply: 1,
        excluded_move: None,
        in_check: false,
        node_pv: true,
        node_beta: 100,
        actual_depth: SINGULAR_MIN_DEPTH,
        halfmove_clock: 0,
        repetitions: 1,
        repeated_after_root: false,
        shuffling: false,
        path_extensions: 0,
        allow_lower_bound: false,
        tt_move: Some(mv),
        tt_score: Some(300),
        tt_depth: SINGULAR_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN,
        tt_flag: Some(TT_EXACT),
        tt_pv: true,
        tt_age: 0,
        tt_move_is_legal: true,
    }
}

fn qualifying_probcut_candidate() -> ProbCutEligibility {
    probcut_candidate(
        true,
        false,
        1,
        false,
        false,
        None,
        PROBCUT_MIN_DEPTH,
        0,
        0,
        None,
        -1,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn negamax_excluding_move(
    searcher: &mut Searcher,
    st: &mut BoardState,
    excluded_move: Move,
    depth: i32,
    ply: usize,
    alpha: i32,
    beta: i32,
    start: Instant,
    tl: f64,
    nodes: &mut u64,
) -> i32 {
    let previous = searcher.excluded_moves[ply].replace(excluded_move);
    let previous_restricted = searcher.set_restricted_verification(true);
    let score = searcher.negamax(st, depth, ply, alpha, beta, false, start, tl, nodes);
    searcher.set_restricted_verification(previous_restricted);
    searcher.excluded_moves[ply] = previous;
    score
}

#[test]
fn special_move_gives_check_rejects_empty_from_square() {
    let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mv = encode_move(7, 1, 7, 2, 0);

    assert!(!special_move_gives_check(&st, mv));
}

#[test]
fn special_move_gives_check_ignores_normal_rook_check() {
    let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mv = legal_move(&st, "a1a8");

    assert!(!special_move_gives_check(&st, mv));
}

#[test]
fn special_move_gives_check_rejects_quiet_non_check() {
    let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mv = legal_move(&st, "a1a2");

    assert!(!special_move_gives_check(&st, mv));
}

#[test]
fn special_move_gives_check_detects_en_passant_discovery() {
    let st = state_from_fen("8/6pp/8/R2pP1k1/6B1/8/6PP/6K1 w - d6 0 1");
    let mv = legal_move(&st, "e5d6");

    assert!(special_move_gives_check(&st, mv));
}

#[test]
fn special_move_gives_check_rejects_non_check_en_passant() {
    let st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let mv = legal_move(&st, "e5d6");

    assert!(!special_move_gives_check(&st, mv));
}

#[test]
fn special_move_gives_check_detects_castling_rook_discovery() {
    let st = state_from_fen("5k2/8/8/8/8/8/8/4K2R w K - 0 1");
    let mv = legal_move(&st, "e1g1");

    assert!(special_move_gives_check(&st, mv));
}

#[test]
fn qsearch_searches_en_passant_captures() {
    let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let stand_pat = searcher.corrected_eval(&st);
    let mut nodes = 0u64;

    let score = searcher.qsearch(
        &mut st,
        -INF,
        INF,
        QS_DEPTH,
        Instant::now(),
        10.0,
        &mut nodes,
        0,
    );

    assert!(
            score > stand_pat + 50,
            "qsearch should improve on stand-pat by searching e5xd6 en passant: stand_pat={stand_pat}, score={score}"
        );
}

#[test]
fn qsearch_checkmate_score_uses_the_actual_ply() {
    let mut st = state_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let mut nodes = 0u64;
    let ply = 17;

    let score = searcher.qsearch(
        &mut st,
        -INF,
        INF,
        -3,
        Instant::now(),
        10.0,
        &mut nodes,
        ply,
    );

    assert_eq!(score, -MATE + ply as i32);
}

#[test]
fn qsearch_pruning_thresholds_honor_tuning_overrides() {
    tune::reset();
    assert!(!qsearch_delta_prunable(974, 0));
    assert!(qsearch_delta_prunable(976, 0));
    assert!(!qsearch_check_cap_reached(-3));
    assert!(qsearch_check_cap_reached(-4));
    assert_eq!(qsearch_see_threshold_cp(), 0);
    assert!(!qsearch_see_prunable(0, qsearch_see_threshold_cp()));
    assert!(qsearch_see_prunable(-1, qsearch_see_threshold_cp()));

    tune::set(TuneParam::QsearchDeltaMarginCp, 700);
    tune::set(TuneParam::QsearchCheckCapDepth, 2);
    tune::set(TuneParam::QsearchSeeThresholdCp, -50);
    assert!(qsearch_delta_prunable(701, 0));
    assert!(qsearch_check_cap_reached(-2));
    assert_eq!(qsearch_see_threshold_cp(), -50);
    assert!(!qsearch_see_prunable(-50, qsearch_see_threshold_cp()));
    assert!(qsearch_see_prunable(-51, qsearch_see_threshold_cp()));
    tune::reset();
}

#[test]
fn lmp_aggressiveness_controls_preserve_the_default_policy() {
    tune::reset();
    let expected = [4, 7, 11, 17, 24, 33, 44, 57];
    for (depth, move_count) in (1..=8).zip(expected) {
        assert_eq!(lmp_move_count(depth), Some(move_count));
    }
    assert_eq!(lmp_move_count(0), None);
    assert_eq!(lmp_move_count(9), None);
    assert!(lmp_king_pressure_safe(2));
    assert!(!lmp_king_pressure_safe(3));

    tune::set(TuneParam::LmpMoveCountScalePermille, 1200);
    tune::set(TuneParam::LmpKingPressureLimit, 5);
    assert_eq!(lmp_move_count(1), Some(5));
    assert_eq!(lmp_move_count(8), Some(68));
    assert!(lmp_king_pressure_safe(4));
    assert!(!lmp_king_pressure_safe(5));
    tune::reset();
}

#[test]
fn lmr_controls_preserve_default_boundaries_and_reductions() {
    tune::reset();
    assert!(!lmr_policy_eligible(1, 3, true, false));
    assert!(lmr_policy_eligible(2, 3, true, false));
    assert!(!lmr_policy_eligible(2, 2, true, false));
    assert!(!lmr_policy_eligible(2, 3, false, false));
    assert!(!lmr_policy_eligible(2, 3, true, true));
    assert_eq!(lmr_reduction(10, 4, true), 2);
    assert_eq!(lmr_reduction(10, 4, false), 3);

    tune::set(TuneParam::LmrDivisorMillis, 1200);
    assert_eq!(lmr_reduction(10, 4, true), 3);
    tune::reset();

    tune::set(TuneParam::LmrMinMoveIndex, 4);
    tune::set(TuneParam::LmrMinDepth, 5);
    tune::set(TuneParam::LmrBaseMillis, 0);
    tune::set(TuneParam::LmrNonPvExtra, 0);
    assert!(!lmr_policy_eligible(3, 5, true, false));
    assert!(!lmr_policy_eligible(4, 4, true, false));
    assert!(lmr_policy_eligible(4, 5, true, false));
    assert_eq!(lmr_reduction(10, 4, true), 1);
    assert_eq!(lmr_reduction(10, 4, false), 1);
    tune::reset();
}

#[test]
fn aspiration_window_controls_preserve_the_default_boundary() {
    tune::reset();
    assert_eq!(aspiration_window_delta(4), INF);
    assert_eq!(aspiration_window_delta(5), 25);

    tune::set(TuneParam::AspirationMinDepth, 3);
    tune::set(TuneParam::AspirationDeltaCp, 40);
    assert_eq!(aspiration_window_delta(2), INF);
    assert_eq!(aspiration_window_delta(3), 40);
    tune::reset();
}

#[test]
fn tactical_check_extension_depth_honors_tuning_overrides() {
    tune::reset();
    assert!(tactical_check_extension_candidate(2, false, 0, false));
    assert!(!tactical_check_extension_candidate(3, false, 0, false));
    assert!(!tactical_check_extension_candidate(2, true, 0, false));
    assert!(!tactical_check_extension_candidate(2, false, 1, false));
    assert!(!tactical_check_extension_candidate(2, false, 0, true));

    tune::set(TuneParam::TacticalCheckExtensionMaxDepth, 4);
    assert!(tactical_check_extension_candidate(4, false, 0, false));
    assert!(!tactical_check_extension_candidate(5, false, 0, false));
    tune::reset();
}

#[test]
fn restricted_search_ignores_unrestricted_tt_cutoffs() {
    let st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
    let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    assert_eq!(
        legal_moves.len(),
        1,
        "test position must have one legal move"
    );
    let excluded_move = legal_moves[0];
    let ply = 1;

    for flag in [TT_EXACT, TT_BETA] {
        let mut position = st;
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
        searcher.nnue_net = None;
        shared_tt.store(
            position.hash,
            12,
            score_to_tt(900, ply),
            flag,
            Some(excluded_move),
        );
        let mut nodes = 0;

        let score = negamax_excluding_move(
            &mut searcher,
            &mut position,
            excluded_move,
            4,
            ply,
            -200,
            -199,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(
            score, -200,
            "restricted search used an unrestricted TT flag {flag}"
        );
    }
}

#[test]
fn restricted_search_with_no_alternative_fails_low_without_storing_tt() {
    let mut st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
    let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    assert_eq!(
        legal_moves.len(),
        1,
        "test position must have one legal move"
    );
    let excluded_move = legal_moves[0];
    let key = st.hash;
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
    searcher.nnue_net = None;
    let mut nodes = 0;

    let score = negamax_excluding_move(
        &mut searcher,
        &mut st,
        excluded_move,
        4,
        1,
        -300,
        -299,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert_eq!(score, -300);
    assert!(
        shared_tt.get_depth(key).is_none(),
        "restricted result contaminated the unrestricted TT"
    );
}

#[test]
fn stopped_restricted_search_restores_the_excluded_move() {
    let mut st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
    let excluded_move = generate_moves(&st, st.w, &st.cr, st.ep)[0];
    let stopped = Arc::new(AtomicBool::new(true));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    searcher.nnue_net = None;
    let mut nodes = 0;

    let score = negamax_excluding_move(
        &mut searcher,
        &mut st,
        excluded_move,
        4,
        1,
        -300,
        -299,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert_eq!(score, 0);
    assert_eq!(searcher.excluded_moves[1], None);
}

#[test]
fn restricted_search_uses_descendant_tt_without_learning_from_its_root() {
    let mut st = state_from_fen("7k/8/4Q3/5K2/8/8/8/8 b - - 0 1");
    let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    assert_eq!(
        legal_moves.len(),
        2,
        "test position must have two legal moves"
    );
    let excluded_move = legal_moves[0];
    let allowed_move = legal_moves[1];
    let mut child = st;
    apply_move(
        &mut child,
        move_sr(allowed_move),
        move_sc(allowed_move),
        move_er(allowed_move),
        move_ec(allowed_move),
        move_promotion(allowed_move),
    );

    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
    searcher.nnue_net = None;
    shared_tt.store(child.hash, 12, score_to_tt(-5000, 2), TT_EXACT, None);
    let (from, to) = from_to_key(
        move_sr(allowed_move),
        move_sc(allowed_move),
        move_er(allowed_move),
        move_ec(allowed_move),
    );
    let piece_index = piece_to_idx(piece_type(st.mailbox[move_from(allowed_move)]));
    let mut nodes = 0;

    let score = negamax_excluding_move(
        &mut searcher,
        &mut st,
        excluded_move,
        4,
        1,
        9,
        10,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert_eq!(score, 5000, "restricted descendants did not use their TT");
    assert_eq!(searcher.history[from][to], 0);
    assert_eq!(searcher.killers[1], [None; 2]);
    assert_eq!(
        searcher.counter_move[piece_index][move_to(allowed_move)],
        None
    );
    assert!(
        shared_tt.get_depth(st.hash).is_none(),
        "restricted root was stored after a descendant TT cutoff"
    );
}

#[test]
fn restricted_verification_does_not_write_descendant_tt_or_learning() {
    let initial = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let legal_moves = generate_moves(&initial, initial.w, &initial.cr, initial.ep);
    let excluded_move = legal_moves[0];
    let child_hashes: Vec<_> = legal_moves[1..]
        .iter()
        .map(|&mv| {
            let mut child = initial;
            apply_move(
                &mut child,
                move_sr(mv),
                move_sc(mv),
                move_er(mv),
                move_ec(mv),
                move_promotion(mv),
            );
            child.hash
        })
        .collect();

    let control_tt = Arc::new(SharedTT::new(1));
    let mut control = Searcher::new(Arc::clone(&control_tt), Arc::new(AtomicBool::new(false)));
    control.nnue_net = None;
    let mut control_position = initial;
    let mut control_nodes = 0;
    control.excluded_moves[1] = Some(excluded_move);
    control.negamax(
        &mut control_position,
        4,
        1,
        -INF,
        INF,
        false,
        Instant::now(),
        10.0,
        &mut control_nodes,
    );
    control.excluded_moves[1] = None;
    assert!(
        child_hashes
            .iter()
            .any(|&hash| control_tt.get_entry(hash).is_some()),
        "control search did not exercise a descendant TT store"
    );

    let isolated_tt = Arc::new(SharedTT::new(1));
    let mut isolated = Searcher::new(Arc::clone(&isolated_tt), Arc::new(AtomicBool::new(false)));
    isolated.nnue_net = None;
    let mut isolated_position = initial;
    let mut isolated_nodes = 0;
    negamax_excluding_move(
        &mut isolated,
        &mut isolated_position,
        excluded_move,
        4,
        1,
        -INF,
        INF,
        Instant::now(),
        10.0,
        &mut isolated_nodes,
    );

    assert!(
        child_hashes
            .iter()
            .all(|&hash| isolated_tt.get_entry(hash).is_none()),
        "restricted verification polluted a descendant TT entry"
    );
    assert!(isolated.history.iter().flatten().all(|&value| value == 0));
    assert!(isolated.killers.iter().flatten().all(Option::is_none));
    assert!(isolated.counter_move.iter().flatten().all(Option::is_none));
}

#[test]
fn excluded_move_state_is_worker_local_and_cleared_before_search() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut first = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let second = Searcher::new(shared_tt, stopped);
    let excluded_move = encode_move(0, 0, 0, 1, 0);

    first.excluded_moves[3] = Some(excluded_move);

    assert_eq!(second.excluded_moves[3], None);
    first.prepare_for_search();
    assert_eq!(first.excluded_moves[3], None);
}

#[test]
fn reversible_shuffle_requires_both_sides_to_retrace_their_moves() {
    let mut path = [None; MAX_PLY];
    path[0] = Some(encode_move(7, 6, 5, 5, 0));
    path[1] = Some(encode_move(0, 6, 2, 5, 0));
    path[2] = Some(encode_move(5, 5, 7, 6, 0));
    path[3] = Some(encode_move(2, 5, 0, 6, 0));

    assert!(reversible_shuffle(&path, 4, 4));
    assert!(!reversible_shuffle(&path, 4, 3));

    path[3] = Some(encode_move(2, 5, 4, 4, 0));
    assert!(!reversible_shuffle(&path, 4, 4));
}

#[test]
fn singular_path_budget_counts_only_positive_extensions() {
    assert_eq!(next_singular_extension_count(2, 1), 3);
    assert_eq!(next_singular_extension_count(2, 0), 2);
    assert_eq!(next_singular_extension_count(2, -2), 2);
    assert_eq!(next_singular_extension_count(u8::MAX, 1), u8::MAX);
}

#[test]
fn singular_outcome_keeps_adjustments_and_cutoffs_distinct() {
    assert_eq!(
        singular_search_outcome(19, 20, true, None, -1),
        SingularSearchOutcome::Continue(1)
    );
    assert_eq!(
        singular_search_outcome(19, 20, false, None, 0),
        SingularSearchOutcome::Continue(0)
    );
    assert_eq!(
        singular_search_outcome(30, 20, false, Some(25), -1),
        SingularSearchOutcome::Cutoff(25)
    );
    assert_eq!(
        singular_search_outcome(24, 20, false, None, -1),
        SingularSearchOutcome::Continue(-1)
    );
    assert_eq!(combine_move_extensions(0, -2), -2);
    assert_eq!(combine_move_extensions(1, -2), 1);
}

#[test]
fn singular_candidate_requires_deep_reliable_safe_tt_evidence() {
    let mv = encode_move(0, 0, 0, 1, 0);
    let evidence = qualifying_singular_evidence(mv);
    let SingularEligibility::Eligible(candidate) = singular_candidate(evidence) else {
        panic!("qualifying TT evidence was rejected");
    };
    assert_eq!(candidate.mv, mv);
    assert_eq!(candidate.beta, 300 - singular_margin(evidence));
    assert_eq!(candidate.depth, (SINGULAR_MIN_DEPTH - 1) / 2);
    assert!(candidate.positive_extension);
    assert_eq!(
        candidate.max_extension,
        i32::from(singular_path_budget(evidence.actual_depth))
    );

    let mut lower_bound = evidence;
    lower_bound.actual_depth = SINGULAR_POLICY_MIN_DEPTH;
    lower_bound.tt_depth = SINGULAR_POLICY_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN;
    lower_bound.node_pv = false;
    lower_bound.node_beta = 200;
    lower_bound.tt_flag = Some(TT_BETA);
    lower_bound.tt_pv = false;
    let mut allowed_lower_bound = lower_bound;
    allowed_lower_bound.allow_lower_bound = true;
    let SingularEligibility::Eligible(lower_candidate) = singular_candidate(allowed_lower_bound)
    else {
        panic!("enabled lower-bound evidence was rejected");
    };
    assert!(!lower_candidate.positive_extension);
    assert_eq!(lower_candidate.beta, allowed_lower_bound.node_beta);
    assert_eq!(lower_candidate.max_extension, 0);

    let no_candidate_cases = [
        SingularEvidence {
            actual_depth: SINGULAR_MIN_DEPTH - 1,
            ..evidence
        },
        SingularEvidence {
            tt_depth: evidence.tt_depth - 1,
            ..evidence
        },
        SingularEvidence {
            tt_flag: Some(TT_ALPHA),
            ..evidence
        },
        SingularEvidence {
            tt_pv: false,
            ..evidence
        },
        SingularEvidence {
            tt_age: SINGULAR_MAX_TT_AGE + 1,
            ..evidence
        },
        SingularEvidence {
            tt_move: None,
            ..evidence
        },
        SingularEvidence {
            allow_lower_bound: false,
            ..lower_bound
        },
        SingularEvidence {
            actual_depth: SINGULAR_POLICY_MIN_DEPTH - 1,
            tt_depth: SINGULAR_POLICY_MIN_DEPTH - 1,
            ..allowed_lower_bound
        },
        SingularEvidence {
            tt_depth: SINGULAR_POLICY_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN - 1,
            ..allowed_lower_bound
        },
        SingularEvidence {
            node_pv: true,
            ..allowed_lower_bound
        },
        SingularEvidence {
            tt_score: Some(allowed_lower_bound.node_beta - 1),
            ..allowed_lower_bound
        },
        SingularEvidence {
            node_beta: MATE / 2,
            tt_score: Some(MATE / 2),
            ..allowed_lower_bound
        },
    ];
    assert!(no_candidate_cases
        .into_iter()
        .all(|case| singular_candidate(case) == SingularEligibility::NoCandidate));

    let safety_cases = [
        SingularEvidence { ply: 0, ..evidence },
        SingularEvidence {
            excluded_move: Some(mv),
            ..evidence
        },
        SingularEvidence {
            tt_move_is_legal: false,
            ..evidence
        },
        SingularEvidence {
            in_check: true,
            ..evidence
        },
        SingularEvidence {
            repetitions: 2,
            repeated_after_root: true,
            ..evidence
        },
        SingularEvidence {
            halfmove_clock: SINGULAR_MAX_HALF_MOVE_CLOCK,
            ..evidence
        },
        SingularEvidence {
            tt_score: Some(MATE / 2),
            ..evidence
        },
        SingularEvidence {
            shuffling: true,
            ..evidence
        },
        SingularEvidence {
            path_extensions: singular_path_budget(evidence.actual_depth),
            ..evidence
        },
    ];
    assert!(safety_cases
        .into_iter()
        .all(|case| singular_candidate(case) == SingularEligibility::SafetyRejected));
}

#[test]
fn singular_margin_rejects_a_competitive_alternative() {
    let mut st = state_from_fen("7k/8/4Q3/5K2/8/8/8/8 b - - 0 1");
    let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    assert_eq!(legal_moves.len(), 2);
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    searcher.nnue_net = None;
    let singular_beta = -6_000 - singular_margin(qualifying_singular_evidence(legal_moves[0]));
    let mut nodes = 0;

    let alternative_score = negamax_excluding_move(
        &mut searcher,
        &mut st,
        legal_moves[0],
        3,
        1,
        singular_beta - 1,
        singular_beta,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert!(alternative_score >= singular_beta);
}

#[test]
fn singular_multi_ply_extensions_require_progressively_larger_gaps() {
    let mut evidence = qualifying_singular_evidence(encode_move(0, 0, 0, 1, 0));
    evidence.actual_depth = SINGULAR_TRIPLE_MIN_DEPTH;
    evidence.tt_depth = SINGULAR_TRIPLE_MIN_DEPTH;
    let SingularEligibility::Eligible(candidate) = singular_candidate(evidence) else {
        panic!("qualifying TT evidence was rejected");
    };
    assert_eq!(candidate.max_extension, 3);

    assert_eq!(
        singular_extension_from_scores(candidate, candidate.beta - 1, None, None),
        1
    );
    assert_eq!(
        singular_extension_from_scores(
            candidate,
            candidate.beta - 1,
            Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP),
            None,
        ),
        1
    );
    assert_eq!(
        singular_extension_from_scores(
            candidate,
            candidate.beta - 1,
            Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP - 1),
            None,
        ),
        2
    );
    assert_eq!(
        singular_extension_from_scores(
            candidate,
            candidate.beta - 1,
            Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP - 1),
            Some(candidate.score - SINGULAR_TRIPLE_MARGIN_CP - 1),
        ),
        3
    );
    assert_eq!(
        singular_extension_from_scores(candidate, candidate.beta, None, None),
        0
    );
}

#[cfg(feature = "search-debug")]
#[test]
fn singular_extensions_require_explicit_experimental_opt_in() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);

    searcher.debug.enable_singular_extensions = false;
    assert!(!searcher.singular_extensions_enabled());

    searcher.debug.enable_singular_extensions = true;
    assert!(searcher.singular_extensions_enabled());

    searcher.debug.enable_singular_multi_extensions = false;
    assert!(!searcher.singular_multi_extensions_enabled());
    searcher.debug.enable_singular_multi_extensions = true;
    assert!(searcher.singular_multi_extensions_enabled());

    searcher.debug.enable_singular_multicut = false;
    searcher.debug.enable_singular_negative_extensions = false;
    assert!(!searcher.singular_multicut_enabled());
    assert!(!searcher.singular_negative_extensions_enabled());

    searcher.debug.enable_singular_multicut = true;
    assert!(searcher.singular_multicut_enabled());
    assert!(!searcher.singular_negative_extensions_enabled());

    searcher.debug.enable_singular_multicut = false;
    searcher.debug.enable_singular_negative_extensions = true;
    assert!(!searcher.singular_multicut_enabled());
    assert!(searcher.singular_negative_extensions_enabled());
}

#[cfg(feature = "search-debug")]
#[test]
fn singular_search_extends_a_synthetic_only_move_tt_result() {
    let mut st = state_from_fen("7k/8/5K2/5Q2/8/8/8/8 b - - 0 1");
    let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    assert_eq!(legal_moves.len(), 1, "position must have one legal move");
    let tt_move = legal_moves[0];
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
    searcher.nnue_net = None;
    searcher.debug.enable_singular_extensions = true;
    shared_tt.store_with_pv(
        st.hash,
        SINGULAR_MIN_DEPTH,
        score_to_tt(0, 1),
        TT_EXACT,
        Some(tt_move),
        true,
    );
    let mut nodes = 0;

    searcher.negamax(
        &mut st,
        SINGULAR_MIN_DEPTH,
        1,
        -INF,
        INF,
        true,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    let stats = searcher.debug_stats();
    assert_eq!(stats.singular_candidates, 1);
    assert_eq!(stats.singular_verifications, 1);
    assert_eq!(stats.singular_extensions, 1);
    assert_eq!(stats.singular_alternative_rejections, 0);
    assert_eq!(searcher.excluded_moves[1], None);
}

#[test]
fn probcut_candidate_requires_a_safe_non_pv_node() {
    assert_eq!(
        qualifying_probcut_candidate(),
        ProbCutEligibility::Eligible(ProbCutCandidate {
            beta: PROBCUT_MARGIN_CP,
            child_depth: PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION,
            store_depth: PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
        })
    );
    assert_eq!(
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH - 1,
            0,
            0,
            None,
            -1,
            None,
        ),
        ProbCutEligibility::NoCandidate
    );

    let safety_cases = [
        probcut_candidate(
            true,
            false,
            0,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            1,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            false,
            1,
            true,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            false,
            1,
            false,
            true,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            Some(encode_move(0, 0, 0, 1, 0)),
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            true,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        ),
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            MATE / 2,
            MATE / 2,
            None,
            -1,
            None,
        ),
    ];
    assert!(safety_cases
        .into_iter()
        .all(|case| case == ProbCutEligibility::SafetyRejected));
}

#[test]
fn probcut_reduction_override_controls_verification_depth() {
    tune::reset();
    tune::set(TuneParam::ProbCutReduction, 1);
    let candidate = qualifying_probcut_candidate();
    tune::reset();

    assert_eq!(
        candidate,
        ProbCutEligibility::Eligible(ProbCutCandidate {
            beta: PROBCUT_MARGIN_CP,
            child_depth: PROBCUT_MIN_DEPTH - 1,
            store_depth: PROBCUT_MIN_DEPTH,
        })
    );
}

#[test]
fn probcut_respects_tt_evidence_but_not_a_lower_bound() {
    for flag in [TT_EXACT, TT_ALPHA] {
        assert_eq!(
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                Some(0),
                PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
                Some(flag),
            ),
            ProbCutEligibility::TtRejected
        );
    }
    assert!(matches!(
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            Some(0),
            PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
            Some(TT_BETA),
        ),
        ProbCutEligibility::Eligible(_)
    ));
}

#[test]
fn probcut_requires_both_verification_stages_to_pass() {
    let beta = PROBCUT_MARGIN_CP;
    assert_eq!(
        probcut_verdict(beta, beta - 1, Some(beta + 100)),
        ProbCutVerdict::QuiescenceRejected
    );
    assert_eq!(
        probcut_verdict(beta, beta, None),
        ProbCutVerdict::FullSearchRejected
    );
    assert_eq!(
        probcut_verdict(beta, beta, Some(beta - 1)),
        ProbCutVerdict::FullSearchRejected
    );
    assert_eq!(
        probcut_verdict(beta, beta, Some(beta)),
        ProbCutVerdict::Cutoff
    );
}

#[cfg(feature = "search-debug")]
#[test]
fn probcut_stores_only_the_reduced_verified_depth() {
    let mut st = state_from_fen("q6k/8/8/8/8/8/8/Q5K1 w - - 0 1");
    let tactical_move = legal_move(&st, "a1a8");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
    searcher.nnue_net = None;
    let key = st.hash;
    let mut nodes = 0;

    let score = searcher.negamax(
        &mut st,
        PROBCUT_MIN_DEPTH,
        1,
        -1,
        0,
        true,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert_eq!(score, 0);
    let stats = searcher.debug_stats();
    assert_eq!(stats.probcut_eligible_nodes, 1);
    assert_eq!(stats.probcut_qsearch_passes, 1);
    assert_eq!(stats.probcut_verifications, 1);
    assert_eq!(stats.probcut_cutoffs, 1);
    let (depth, tt_score, flag, best_move) = shared_tt
        .get_depth(key)
        .expect("ProbCut did not store a bound");
    assert_eq!(
        depth,
        PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
        "ProbCut stored a depth other than its reduced proof"
    );
    assert_eq!(score_from_tt(tt_score, 1), PROBCUT_MARGIN_CP);
    assert_eq!(flag, TT_BETA);
    assert_eq!(best_move, Some(tactical_move));
    assert!(!searcher.probcut_verification);
}

#[cfg(feature = "search-debug")]
#[test]
fn search_debug_stats_are_reset_between_root_moves() {
    let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let mut nodes = 0u64;

    searcher.qsearch(
        &mut st,
        -INF,
        INF,
        QS_DEPTH,
        Instant::now(),
        10.0,
        &mut nodes,
        0,
    );

    let stats = searcher.debug_stats();
    assert!(stats.qnodes > 1);
    assert!(stats.max_ply > 0);

    searcher.reset_debug_stats();

    assert_eq!(searcher.debug_stats(), SearchDebugStats::default());
}

#[test]
fn settled_lazy_smp_helper_can_coordinate_the_shared_soft_stop() {
    let timing = IterationTiming {
        elapsed_seconds: 1.1,
        iteration_seconds: 0.2,
        previous_iteration_seconds: 0.15,
        score_change_cp: 12,
        stable_iterations: 3,
        best_move_effort: 0.8,
        worker_disagreement: 0.0,
    };
    let agreement = LazySmpAgreement {
        disagreement: 0.0,
        comparable_workers: 2,
        principal_agrees: true,
    };

    assert!(lazy_smp_worker_can_coordinate_stop(
        2, None, 1.0, timing, agreement, true,
    ));
    assert!(!lazy_smp_worker_can_coordinate_stop(
        2,
        None,
        1.0,
        timing,
        LazySmpAgreement {
            disagreement: 0.0,
            comparable_workers: 0,
            principal_agrees: true,
        },
        true,
    ));
    assert!(!lazy_smp_worker_can_coordinate_stop(
        1,
        Some(1),
        1.0,
        timing,
        agreement,
        true,
    ));
    assert!(!lazy_smp_worker_can_coordinate_stop(
        2,
        None,
        1.0,
        IterationTiming {
            elapsed_seconds: 0.9,
            ..timing
        },
        agreement,
        true,
    ));
    assert!(!lazy_smp_worker_can_coordinate_stop(
        2,
        None,
        1.0,
        timing,
        LazySmpAgreement {
            disagreement: 0.5,
            comparable_workers: 2,
            principal_agrees: true,
        },
        true,
    ));
    assert!(!lazy_smp_worker_can_coordinate_stop(
        2,
        None,
        1.0,
        timing,
        LazySmpAgreement {
            disagreement: 0.0,
            comparable_workers: 2,
            principal_agrees: false,
        },
        true,
    ));
}

#[test]
fn lazy_smp_pool_reuses_workers_with_a_fresh_stop_token() {
    let pool = LazySmpPool::new();
    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    let first_stopped = Arc::new(AtomicBool::new(false));
    let first_tt = Arc::new(SharedTT::new(128));
    let mut first_root = Searcher::new(Arc::clone(&first_tt), Arc::clone(&first_stopped));
    let (first_move, _, first_depth, _) = lazy_smp_search(
        &pool,
        first_tt,
        &st,
        &root_moves,
        |_, _| 0,
        LazySmpSearchLimits {
            soft_time: 0.0,
            hard_time: 10.0,
            depth: 4,
            node_limit: None,
            start: Instant::now(),
        },
        2,
        &mut first_root,
    );

    assert!(root_moves.contains(&first_move));
    assert!(first_depth >= 1);
    assert!(first_stopped.load(Ordering::Relaxed));
    let first_worker_ids = pool.worker_ids();
    assert_eq!(first_worker_ids.len(), 2);

    let second_stopped = Arc::new(AtomicBool::new(false));
    let second_tt = Arc::new(SharedTT::new(128));
    let mut second_root = Searcher::new(Arc::clone(&second_tt), Arc::clone(&second_stopped));
    let (second_move, _, second_depth, second_nodes) = lazy_smp_search(
        &pool,
        second_tt,
        &st,
        &root_moves,
        |_, _| 0,
        LazySmpSearchLimits {
            soft_time: 10.0,
            hard_time: 10.0,
            depth: 1,
            node_limit: None,
            start: Instant::now(),
        },
        2,
        &mut second_root,
    );

    assert!(root_moves.contains(&second_move));
    assert_eq!(second_depth, 1);
    assert!(second_nodes > 0);
    assert!(!second_stopped.load(Ordering::Relaxed));
    assert_eq!(pool.worker_ids(), first_worker_ids);
}

#[test]
fn lazy_smp_helpers_prioritize_distinct_root_lanes() {
    let original = vec![
        encode_move(0, 0, 0, 0, 0),
        encode_move(0, 1, 0, 1, 0),
        encode_move(0, 2, 0, 2, 0),
        encode_move(0, 3, 0, 3, 0),
    ];
    let thread_zero = lazy_smp_root_moves(&original, 0, 4);
    let thread_one = lazy_smp_root_moves(&original, 1, 4);
    let thread_two = lazy_smp_root_moves(&original, 2, 4);
    let thread_three = lazy_smp_root_moves(&original, 3, 4);

    assert_eq!(thread_zero, original);
    assert_eq!(thread_one[0], original[1]);
    assert_eq!(thread_two[0], original[2]);
    assert_eq!(thread_three[0], original[3]);

    let mut sorted_original = original.clone();
    sorted_original.sort_unstable();
    for mut helper_moves in [thread_one, thread_two, thread_three] {
        helper_moves.sort_unstable();
        assert_eq!(helper_moves, sorted_original);
    }
}

#[test]
fn lazy_smp_many_helpers_keep_rotated_root_order() {
    let original = (0..16)
        .map(|square| {
            let row = square / 8;
            let col = square % 8;
            encode_move(row, col, row, col, 0)
        })
        .collect::<Vec<_>>();
    let mut expected = original.clone();
    expected.rotate_left(1);

    assert_eq!(lazy_smp_root_moves(&original, 1, 12), expected);
}

// These positions test helper-lane assignment and disagreement accounting directly;
// a final TSV move cannot reveal which worker searched a move or how its vote counted.
#[test]
fn lazy_smp_assigns_a_helper_to_verify_game_ffzk_y782_recapture() {
    let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
    let bxe4 = legal_move(&st, "g2e4");
    let qb2 = legal_move(&st, "c1b2");
    let re1 = legal_move(&st, "d1e1");
    let root_moves = [bxe4, qb2, re1];

    assert_eq!(
        lazy_smp_worker_root_moves(&st, &root_moves, 1, 12),
        vec![bxe4]
    );
    assert_eq!(
        lazy_smp_worker_root_moves(&st, &root_moves, 0, 12),
        root_moves
    );
}

#[test]
fn lazy_smp_tactical_verifier_does_not_inflate_worker_disagreement() {
    let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
    let bxe4 = legal_move(&st, "g2e4");
    let qe3 = legal_move(&st, "c1e3");
    let root_moves = Arc::new(vec![bxe4, qe3]);
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let verification_tt = Arc::new(SharedTT::new(1));
    let root_searcher = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let job = LazySmpSearchJob {
        shared_tt,
        verification_move: Some(bxe4),
        verification_tt: Some(verification_tt),
        stopped,
        st,
        root_moves,
        num_threads: 3,
        root_depth_extension: |_, _| 0,
        limits: LazySmpSearchLimits {
            soft_time: 1.0,
            hard_time: 2.0,
            depth: 20,
            node_limit: None,
            start: Instant::now(),
        },
        root_context: Arc::new(LazySmpRootContext::from_searcher(&root_searcher)),
        start: Instant::now(),
        global_best_depth: Arc::new(AtomicI32::new(0)),
        printed_depth: Arc::new(AtomicI32::new(0)),
        global_nodes: Arc::new(AtomicU64::new(0)),
        node_limit_counter: None,
        worker_best_moves: (0..3).map(|_| AtomicU64::new(0)).collect(),
        worker_depths: (0..3).map(|_| AtomicI32::new(0)).collect(),
    };
    job.worker_best_moves[1].store(u64::from(bxe4), Ordering::Relaxed);
    job.worker_depths[1].store(22, Ordering::Release);
    job.worker_best_moves[2].store(u64::from(qe3), Ordering::Relaxed);
    job.worker_depths[2].store(17, Ordering::Release);

    assert_eq!(lazy_smp_worker_disagreement(&job, 0, qe3, 17), 0.0);
    assert!(!Arc::ptr_eq(
        &job.shared_tt,
        job.verification_tt.as_ref().unwrap()
    ));
}

fn completed_thread(thread_id: usize, best_move: Move, score: i32, depth: i32) -> ThreadResult {
    ThreadResult {
        thread_id,
        best_move,
        score,
        depth,
        nodes: 1,
        learning: None,
    }
}

#[test]
fn final_smp_info_does_not_regress_a_depth_already_published_by_a_helper() {
    // A helper can complete depth N and publish `info depth N` before the
    // principal worker settles on a shallower result. The final aggregate
    // report must not then print `info depth N-1`, because a UCI client
    // would see monotonically decreasing depths although the search itself
    // never went backwards.
    assert!(should_print_final_info(23, 22));
    assert!(
        !should_print_final_info(23, 23),
        "equal depth was already published by a helper and must not be reprinted"
    );
    assert!(!should_print_final_info(22, 23));
    assert!(!should_print_final_info(0, 0));
    assert!(!should_print_final_info(0, 23));
}

// The following recorded positions drive synthetic worker ballots. They verify SMP
// result selection independently of search nondeterminism, which TSV move cases cannot
// express because they only observe a completed single-thread root search.
#[test]
fn lazy_smp_does_not_let_deepest_outlier_repeat_draw_game_kh7() {
    // https://lichess.org/xMs5Nkx3 before 49...Kh7:
    // 2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49
    // 49...Bg7 held the evaluation near equality; the played 49...Kh7
    // conceded a substantial white advantage. A one-ply-deeper dissenting
    // Lazy SMP worker must not overrule two current-depth votes for Bg7.
    let st = state_from_fen("2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49");
    let bg7 = legal_move(&st, "h6g7");
    let kh7 = legal_move(&st, "g8h7");
    let results = [
        completed_thread(0, bg7, -72, 14),
        completed_thread(1, bg7, -68, 14),
        completed_thread(2, kh7, -61, 15),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[bg7, kh7])
            .unwrap()
            .best_move,
        bg7
    );
}

#[test]
fn lazy_smp_does_not_let_deepest_outlier_repeat_loss_game_kf7() {
    // https://lichess.org/VIPYcetR before 22...Kf7:
    // 1r1qk3/Q1p5/5n2/3n1pp1/2BP3r/2P1P3/P2B3P/R3K2R b KQ - 0 22
    // 22...Ne7 was the resilient move; 22...Kf7 was the first major error.
    let st = state_from_fen("1r1qk3/Q1p5/5n2/3n1pp1/2BP3r/2P1P3/P2B3P/R3K2R b KQ - 0 22");
    let ne7 = legal_move(&st, "d5e7");
    let kf7 = legal_move(&st, "e8f7");
    let results = [
        completed_thread(0, ne7, -31, 12),
        completed_thread(1, ne7, -28, 12),
        completed_thread(2, kf7, -20, 13),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[ne7, kf7])
            .unwrap()
            .best_move,
        ne7
    );
}

#[test]
fn lazy_smp_does_not_let_deepest_outlier_repeat_loss_game_g4() {
    // https://lichess.org/VIPYcetR before 28...g4:
    // 1r1q4/Q1p5/1n2B1k1/5ppr/3Pn3/2P1P1R1/P2B3P/2K2R2 b - - 12 28
    // 28...Kh6 resisted; 28...g4 allowed the forcing Bxf5+/Rxg4+
    // sequence. Prefer the supported near-deep result to a deepest outlier.
    let st = state_from_fen("1r1q4/Q1p5/1n2B1k1/5ppr/3Pn3/2P1P1R1/P2B3P/2K2R2 b - - 12 28");
    let kh6 = legal_move(&st, "g6h6");
    let g4 = legal_move(&st, "g5g4");
    let results = [
        completed_thread(0, kh6, -205, 13),
        completed_thread(1, kh6, -198, 13),
        completed_thread(2, g4, -187, 14),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[kh6, g4])
            .unwrap()
            .best_move,
        kh6
    );
}

#[test]
fn lazy_smp_keeps_principal_recapture_from_game_ffzk_y782() {
    // https://lichess.org/ffzkY782 before 31.Re1:
    // 4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31
    // The principal worker found 31.Bxe4, which Stockfish evaluates as
    // equal, but helper consensus replaced it with a losing quiet move.
    let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
    let bxe4 = legal_move(&st, "g2e4");
    let qe3 = legal_move(&st, "c1e3");
    let qb2 = legal_move(&st, "c1b2");
    let results = [
        completed_thread(8, qe3, 0, 19),
        completed_thread(7, bxe4, -91, 18),
        completed_thread(3, qb2, -56, 20),
        completed_thread(11, qe3, 0, 18),
        completed_thread(2, qb2, -56, 20),
        completed_thread(5, qe3, 0, 19),
        completed_thread(4, bxe4, -72, 17),
        completed_thread(6, qe3, 0, 18),
        completed_thread(1, qe3, 0, 18),
        completed_thread(9, qe3, 0, 18),
        completed_thread(10, qe3, 0, 19),
        completed_thread(0, bxe4, -126, 19),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[bxe4, qe3, qb2])
            .unwrap()
            .best_move,
        bxe4
    );
}

#[test]
fn lazy_smp_keeps_deeper_root_recapture_from_game_ffzk_y782() {
    // A dedicated helper can search Bxe4 more deeply than workers that
    // compare every root move. Keep its better score over the principal
    // worker's losing quiet alternative.
    let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
    let bxe4 = legal_move(&st, "g2e4");
    let qb2 = legal_move(&st, "c1b2");
    let re1 = legal_move(&st, "d1e1");
    let results = [
        completed_thread(0, qb2, -128, 17),
        completed_thread(1, bxe4, -65, 18),
        completed_thread(2, re1, -124, 17),
    ];

    assert!(see(&st.bb, move_from(bxe4), move_to(bxe4)) >= -25);
    assert_eq!(
        select_lazy_smp_result(&results, &st, &[bxe4, qb2, re1])
            .unwrap()
            .best_move,
        bxe4
    );
}

#[test]
fn lazy_smp_rejects_a_worse_tactical_verification() {
    let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
    let bxe4 = legal_move(&st, "g2e4");
    let qe3 = legal_move(&st, "c1e3");
    let results = [
        completed_thread(0, qe3, -20, 17),
        completed_thread(1, bxe4, -125, 18),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[bxe4, qe3])
            .unwrap()
            .best_move,
        qe3
    );
}

#[test]
fn lazy_smp_uses_consensus_when_principal_has_no_result() {
    let st = state_from_fen("2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49");
    let bg7 = legal_move(&st, "h6g7");
    let kh7 = legal_move(&st, "g8h7");
    let results = [
        completed_thread(1, bg7, -72, 14),
        completed_thread(2, bg7, -68, 14),
        completed_thread(3, kh7, -61, 15),
    ];

    assert_eq!(
        select_lazy_smp_result(&results, &st, &[bg7, kh7])
            .unwrap()
            .best_move,
        bg7
    );
}

#[test]
fn tt_mate_scores_are_stored_ply_independent() {
    let winning_score = MATE - 9;
    let losing_score = -MATE + 11;

    assert_eq!(score_to_tt(winning_score, 9), MATE);
    assert_eq!(score_from_tt(MATE, 3), MATE - 3);

    assert_eq!(score_to_tt(losing_score, 11), -MATE);
    assert_eq!(score_from_tt(-MATE, 4), -MATE + 4);
}

#[test]
fn tt_non_mate_scores_are_not_adjusted() {
    assert_eq!(score_to_tt(42, 8), 42);
    assert_eq!(score_from_tt(-313, 5), -313);
}

#[test]
fn threefold_repetition_detected_after_long_history() {
    let mut engine = Engine::new();
    engine.book = None;

    engine.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 50");

    for _ in 0..12 {
        assert!(engine.make_move_uci(7, 4, 7, 3, 0));
        assert!(engine.make_move_uci(0, 4, 0, 3, 0));
        assert!(engine.make_move_uci(7, 3, 7, 4, 0));
        assert!(engine.make_move_uci(0, 3, 0, 4, 0));
    }

    assert!(
        engine.searcher.is_repetition(),
        "Threefold repetition should be detected even after 20+ moves of history"
    );
}

#[test]
fn draw_status_distinguishes_claimable_and_automatic_thresholds() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let mut st = state_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1");
    searcher.rep_stack = vec![st.hash];
    searcher.rep_stack_len = 1;

    assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::None);
    st.halfmove_clock = 100;
    assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Claimable);
    st.halfmove_clock = 150;
    assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Automatic);

    st.halfmove_clock = 8;
    searcher.rep_stack = vec![7, 1, 7, 2, 7];
    searcher.rep_stack_len = searcher.rep_stack.len();
    assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Claimable);

    searcher.rep_stack = vec![7, 1, 7, 2, 7, 3, 7, 4, 7];
    searcher.rep_stack_len = searcher.rep_stack.len();
    assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Automatic);
}

#[test]
fn draw_status_terminates_cycles_only_after_the_search_root() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let st = state_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 8 1");

    searcher.rep_stack = vec![9, 8, 7, 6, 7];
    searcher.rep_stack_len = searcher.rep_stack.len();
    searcher.rep_root_len = 3;

    assert_eq!(
        searcher.draw_status(&st, 2, 1),
        DrawStatus::None,
        "a second occurrence of the root is not a legal threefold"
    );
    searcher.rep_root_len = 1;
    assert_eq!(
        searcher.draw_status(&st, 4, 1),
        DrawStatus::SearchCycle,
        "a second occurrence entirely inside the tree terminates the cycle"
    );
}
