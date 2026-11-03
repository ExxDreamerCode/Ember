//! Private evaluator-count, freshness, exit-order and state contracts. TSV root
//! move assertions cannot observe the SearchEval calls or these internal arrays.
use super::*;
use std::cell::RefCell;

#[derive(Clone, Copy)]
struct Counted<'a> {
    calls: &'a RefCell<Vec<(usize, u64, i32)>>,
    stop_after_first: bool,
}
impl SearchEval for Counted<'_> {
    fn static_eval<const CHESS960: bool>(self, s: &Searcher, st: &BoardState, ply: usize) -> i32 {
        let value = ClassicEval.static_eval::<CHESS960>(s, st, ply);
        self.calls.borrow_mut().push((ply, st.hash, value));
        if self.stop_after_first && self.calls.borrow().len() == 1 {
            s.set_stopped();
        }
        value
    }
    fn corrected_eval<const CHESS960: bool>(self, s: &Searcher, st: &BoardState) -> i32 {
        ClassicEval.corrected_eval::<CHESS960>(s, st)
    }
    #[allow(clippy::too_many_arguments)]
    fn push_acc(
        self,
        s: &mut Searcher,
        before: &BoardState,
        after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        ClassicEval.push_acc(s, before, after, sr, sc, er, ec, promotion, ply);
    }
    fn ensure_child_stack(self, s: &mut Searcher, ply: usize) {
        ClassicEval.ensure_child_stack(s, ply);
    }
    fn copy_null_acc(self, s: &mut Searcher, ply: usize) {
        ClassicEval.copy_null_acc(s, ply);
    }
}
fn position(fen: &str) -> BoardState {
    let mut e = crate::engine::Engine::new();
    e.set_fen(fen);
    e.st
}
fn searcher() -> Box<Searcher> {
    tune::reset();
    let mut s = Box::new(Searcher::new(
        Arc::new(SharedTT::new(1)),
        Arc::new(AtomicBool::new(false)),
    ));
    s.nnue_net = None;
    s.ember_v2_net = None;
    s.classic_net = None;
    s.eval_history.fill(-12345);
    s
}
const QUIET: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test]
fn d2e_parent_handoff_uses_one_eval_and_keeps_history() {
    // Actual negamax parent route, not a manually seeded qsearch hint.
    let mut st = position(QUIET);
    let mut s = searcher();
    let trace = RefCell::new(Vec::new());
    let eval = Counted {
        calls: &trace,
        stop_after_first: false,
    };
    let expected = ClassicEval.static_eval::<false>(&s, &st, 2);
    let mut nodes = 0;
    let value = s.negamax_mode_scalar::<false, false, false, _>(
        &mut st, 0, 2, -INF, INF, true, &mut nodes, eval,
    );
    assert_eq!(value, expected);
    assert_eq!(
        s.eval_history[2], expected,
        "original leaf history write survives"
    );
    assert_eq!(
        s.eval_history[0], -12345,
        "prior improving context survives"
    );
    assert_eq!(nodes, 2, "both main and qsearch accounting survive");
    assert_eq!(
        trace.borrow().len(),
        1,
        "immediate stand-pat must reuse the parent evaluation"
    );
}

// D2A_CANDIDATE_ONLY_CONTRACTS: the adapted parent regression above is H-specific.
#[test]
fn d2e_ldse_legal_rejected_probe_invalidates_parent_reuse() {
    let mut st = position(QUIET);
    let before = st;
    let mut s = searcher();
    let mv = generate_moves(&st, st.w, &st.cr, st.ep)[0];
    let expected = ClassicEval.static_eval::<false>(&s, &st, 2);
    s.shared_tt
        .store(st.hash, 0, expected - 100, TT_BETA, Some(mv));
    let _restore_tuning = enable_ldse(0, 1);
    s.singular_extensions_used[2] = 1;
    s.singular_negative_on_line[2] = 1;
    s.singular_extensions_used[3] = 7;
    s.singular_negative_on_line[3] = 5;
    s.singular_ldse_on_line[3] = 4;
    s.path_moves[2] = Some(mv);
    s.rep_stack.push(0x123456789);
    s.rep_stack_len = 1;
    let paths = (
        s.path_moves,
        s.singular_extensions_used,
        s.singular_negative_on_line,
        s.singular_ldse_on_line,
    );
    let repetition = s.rep_stack.clone();
    let contexts = RefCell::new(Vec::new());
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let alpha = expected + 2000;
    let value = s.negamax_mode_scalar::<false, false, true, _>(
        &mut st,
        0,
        2,
        alpha,
        alpha + 1,
        true,
        &mut nodes,
        Observed {
            inner: Counted {
                calls: &trace,
                stop_after_first: false,
            },
            contexts: &contexts,
        },
    );
    tune::reset();
    assert_eq!(
        value, alpha,
        "probe must fall through rather than be accepted"
    );
    assert!(
        nodes > 2 && trace.borrow().iter().any(|(ply, _, _)| *ply == 3),
        "an actual legal descendant was searched"
    );
    assert_eq!(
        trace
            .borrow()
            .iter()
            .filter(|(ply, hash, _)| *ply == 2 && *hash == before.hash)
            .count(),
        2,
        "restoring the board must not resurrect the parent's reuse eligibility"
    );
    assert_eq!(s.eval_history[2], expected);
    assert_eq!(st.hash, before.hash);
    assert_eq!(st.bb, before.bb);
    assert_eq!(s.rep_stack_len, 1);
    assert_eq!(s.rep_stack, repetition);
    assert!(!s.restricted_verification_active());
    assert_eq!(
        (
            s.path_moves,
            s.singular_extensions_used,
            s.singular_negative_on_line,
            s.singular_ldse_on_line
        ),
        paths
    );
    assert!(contexts.borrow().iter().any(|c| c.ply == 3
        && c.restricted
        && c.ldse == 1
        && c.extensions == 2
        && c.negative == 1
        && c.repetition_len == 2));
    let mut child = before;
    apply_move(
        &mut child,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
    assert!(
        s.shared_tt.get_entry(child.hash).is_none(),
        "rejected speculative descendant must not write TT"
    );
    assert!(s.corr_hist.iter().all(|v| *v == 0));
    #[cfg(feature = "node-correction-training")]
    assert!(s.corr_staged_touched.is_empty());
}

#[test]
fn d2e_direct_and_recursive_qsearch_use_current_correction() {
    let mut st = position("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mut s = searcher();
    let old = ClassicEval.static_eval::<false>(&s, &st, 0);
    let slot = corr_idx(compute_pawn_hash(&st), st.w);
    s.corr_hist[slot] = 120;
    let fresh = ClassicEval.static_eval::<false>(&s, &st, 0);
    assert_eq!(fresh, old + 120);
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let score = s.qsearch_mode_scalar::<false, false, _>(
        &mut st,
        -INF,
        INF,
        QS_DEPTH,
        &mut nodes,
        0,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
        None,
    );
    assert_eq!(score, fresh);
    assert_eq!(trace.borrow()[0].2, fresh);

    let mut st = position("7k/8/8/8/8/8/q7/R3K3 w - - 0 1");
    let before = st;
    let mut s = searcher();
    s.corr_hist[corr_idx(compute_pawn_hash(&st), !st.w)] = -90;
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let stand = ClassicEval.static_eval::<false>(&s, &st, 0);
    s.qsearch_mode_scalar::<false, false, _>(
        &mut st,
        -INF,
        INF,
        QS_DEPTH,
        &mut nodes,
        0,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
        Some(stand),
    );
    assert!(
        trace
            .borrow()
            .iter()
            .any(|(ply, hash, value)| *ply > 0 && *hash != before.hash && *value != stand),
        "recursive captures evaluate their own board and corrected side"
    );
    assert_eq!(st.hash, before.hash);
}

#[cfg(feature = "node-correction-training")]
#[test]
fn d2e_descendant_training_boundary_requires_a_fresh_entry() {
    let mut st = position("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let mut s = searcher();
    #[cfg(feature = "search-debug")]
    {
        s.debug.disable_corr_node_train = false;
    }
    let old = ClassicEval.static_eval::<false>(&s, &st, 0);
    let initial_calls = RefCell::new(Vec::new());
    let mut initial_nodes = 0;
    assert_eq!(
        s.negamax_mode_scalar::<false, false, false, _>(
            &mut st,
            0,
            0,
            -INF,
            INF,
            true,
            &mut initial_nodes,
            Counted {
                calls: &initial_calls,
                stop_after_first: false
            }
        ),
        old
    );
    assert_eq!(initial_calls.borrow().len(), 1);

    let mut descendant = st;
    let mv = generate_moves(&st, st.w, &st.cr, st.ep)[0];
    apply_move(
        &mut descendant,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
    // The same pawn/side slot is shared with an independently completed descendant context.
    let reply = generate_moves(&descendant, descendant.w, &descendant.cr, descendant.ep)[0];
    apply_move(
        &mut descendant,
        move_sr(reply),
        move_sc(reply),
        move_er(reply),
        move_ec(reply),
        move_promotion(reply),
    );
    assert_eq!(descendant.w, st.w);
    let ev = ClassicEval.static_eval::<false>(&s, &descendant, 0);
    assert!(s.train_correction_from_node(
        &descendant,
        ev + 200,
        5,
        ev,
        TT_EXACT,
        false,
        false,
        true
    ));
    s.flush_correction_staging();
    let fresh = ClassicEval.static_eval::<false>(&s, &st, 0);
    assert!(fresh > old);
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let score = s.qsearch_mode_scalar::<false, false, _>(
        &mut st,
        -INF,
        INF,
        QS_DEPTH,
        &mut nodes,
        0,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
        None,
    );
    assert_eq!(score, fresh);
    assert_eq!(trace.borrow()[0].2, fresh);
    let parent_calls = RefCell::new(Vec::new());
    let mut parent_nodes = 0;
    assert_eq!(
        s.negamax_mode_scalar::<false, false, false, _>(
            &mut st,
            0,
            0,
            -INF,
            INF,
            true,
            &mut parent_nodes,
            Counted {
                calls: &parent_calls,
                stop_after_first: false
            }
        ),
        fresh
    );
    assert_eq!(parent_calls.borrow().len(), 1);
    assert_eq!(s.eval_history[0], fresh);
}

#[test]
fn d2e_qsearch_exit_order_and_checked_cap_ignore_unused_hints() {
    for mode in 0..3 {
        let mut st = position(QUIET);
        let mut s = searcher();
        let trace = RefCell::new(Vec::new());
        let mut nodes = 0;
        s.set_node_limit(Some(u64::MAX));
        if mode == 0 {
            s.set_stopped();
        }
        if mode == 1 {
            s.set_node_limit(Some(1));
        }
        if mode == 2 {
            st.halfmove_clock = 100;
        }
        let value = s.qsearch_mode_scalar::<false, true, _>(
            &mut st,
            -INF,
            INF,
            QS_DEPTH,
            &mut nodes,
            2,
            Counted {
                calls: &trace,
                stop_after_first: false,
            },
            Some(1234),
        );
        assert_eq!(value, 0);
        assert!(trace.borrow().is_empty());
        assert_eq!(nodes, if mode == 0 { 0 } else { 1 });
    }
    let mut st = position("7k/8/8/8/8/8/7R/4K3 b - - 0 1");
    let mut s = searcher();
    let expected = ClassicEval.static_eval::<false>(&s, &st, 0);
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let value = s.qsearch_mode_scalar::<false, false, _>(
        &mut st,
        -INF,
        INF,
        -4,
        &mut nodes,
        0,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
        Some(-12345),
    );
    assert_eq!(value, expected);
    assert_eq!(
        trace.borrow().len(),
        1,
        "checked depth-cap evaluation remains fresh"
    );
}

#[test]
fn d2e_parent_stop_and_manual_fresh_reference_preserve_state() {
    let mut st = position(QUIET);
    let mut s = searcher();
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let value = s.negamax_mode_scalar::<false, false, false, _>(
        &mut st,
        0,
        2,
        -INF,
        INF,
        true,
        &mut nodes,
        Counted {
            calls: &trace,
            stop_after_first: true,
        },
    );
    assert_eq!(value, 0);
    assert_eq!(
        nodes, 1,
        "qsearch stops before counting a node or consuming a hint"
    );
    assert_eq!(trace.borrow().len(), 1);
    assert_ne!(s.eval_history[2], -12345);

    let mut a = searcher();
    let mut b = searcher();
    let mut st_a = position(QUIET);
    let mut st_b = st_a;
    let trace = RefCell::new(Vec::new());
    let mut na = 0;
    let va = a.negamax_mode_scalar::<false, false, false, _>(
        &mut st_a,
        0,
        2,
        -INF,
        INF,
        true,
        &mut na,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
    );
    b.eval_history[2] = ClassicEval.static_eval::<false>(&b, &st_b, 2);
    let mut nb = 1;
    let vb = b.qsearch_mode_scalar::<false, false, _>(
        &mut st_b,
        -INF,
        INF,
        QS_DEPTH,
        &mut nb,
        2,
        ClassicEval,
        None,
    );
    assert_eq!((va, na), (vb, nb));
    assert_eq!(
        crate::board::board_to_fen(&st_a),
        crate::board::board_to_fen(&st_b)
    );
    assert_eq!(st_a.mailbox, st_b.mailbox);
    assert_eq!(a.eval_history, b.eval_history);
    assert_eq!(a.history, b.history);
    assert_eq!(a.corr_hist, b.corr_hist);
    assert_eq!(a.killers, b.killers);
    assert_eq!(a.path_moves, b.path_moves);
    assert_eq!(a.excluded_moves, b.excluded_moves);
    assert_eq!(a.singular_extensions_used, b.singular_extensions_used);
    assert_eq!(a.singular_negative_on_line, b.singular_negative_on_line);
    assert_eq!(a.singular_ldse_on_line, b.singular_ldse_on_line);
    assert_eq!(a.counter_move, b.counter_move);
    assert_eq!(a.capture_history, b.capture_history);
    assert_eq!(
        a.restricted_verification_active(),
        b.restricted_verification_active()
    );
    assert_eq!(a.rep_stack, b.rep_stack);
    assert_eq!(
        a.shared_tt.get_entry(st_a.hash),
        b.shared_tt.get_entry(st_b.hash)
    );
    #[cfg(feature = "node-correction-training")]
    {
        assert_eq!(a.corr_staged_depth, b.corr_staged_depth);
        assert_eq!(a.corr_staged_diff, b.corr_staged_diff);
    }
}

// These extra H contracts observe private const-LDSE routing and inherited state,
// rather than an exact root move; neither UCI nor the TSV schema exposes them.
struct RestoreTuning;
impl Drop for RestoreTuning {
    fn drop(&mut self) {
        tune::reset();
    }
}
fn enable_ldse(depth: i64, per_line: i64) -> RestoreTuning {
    tune::set(TuneParam::LdseEnabled, 1);
    tune::set(TuneParam::LdseMaxDepth, depth);
    tune::set(TuneParam::LdseMaxPerLine, per_line);
    RestoreTuning
}
#[derive(Clone, Copy)]
struct Context {
    ply: usize,
    restricted: bool,
    ldse: u8,
    extensions: u8,
    negative: u8,
    repetition_len: usize,
}
#[derive(Clone, Copy)]
struct Observed<'a> {
    inner: Counted<'a>,
    contexts: &'a RefCell<Vec<Context>>,
}
impl SearchEval for Observed<'_> {
    fn static_eval<const CHESS960: bool>(self, s: &Searcher, st: &BoardState, ply: usize) -> i32 {
        self.contexts.borrow_mut().push(Context {
            ply,
            restricted: s.restricted_verification_active(),
            ldse: s.singular_ldse_on_line[ply],
            extensions: s.singular_extensions_used[ply],
            negative: s.singular_negative_on_line[ply],
            repetition_len: s.rep_stack_len,
        });
        self.inner.static_eval::<CHESS960>(s, st, ply)
    }
    fn corrected_eval<const CHESS960: bool>(self, s: &Searcher, st: &BoardState) -> i32 {
        self.inner.corrected_eval::<CHESS960>(s, st)
    }
    #[allow(clippy::too_many_arguments)]
    fn push_acc(
        self,
        s: &mut Searcher,
        before: &BoardState,
        after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        self.inner
            .push_acc(s, before, after, sr, sc, er, ec, promotion, ply);
    }
    fn ensure_child_stack(self, s: &mut Searcher, ply: usize) {
        self.inner.ensure_child_stack(s, ply);
    }
    fn copy_null_acc(self, s: &mut Searcher, ply: usize) {
        self.inner.copy_null_acc(s, ply);
    }
}

#[test]
fn d2e_ldse_false_ignores_enabled_probe_configuration() {
    let mut s = searcher();
    let mut st = position(QUIET);
    let mv = generate_moves(&st, st.w, &st.cr, st.ep)[0];
    let expected = ClassicEval.static_eval::<false>(&s, &st, 2);
    s.shared_tt
        .store(st.hash, 0, expected - 100, TT_BETA, Some(mv));
    let _restore_tuning = enable_ldse(0, 1);
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let alpha = expected + 2000;
    let score = s.negamax_mode_scalar::<false, false, false, _>(
        &mut st,
        0,
        2,
        alpha,
        alpha + 1,
        true,
        &mut nodes,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
    );
    assert_eq!(score, alpha);
    assert_eq!(nodes, 2);
    assert_eq!(trace.borrow().len(), 1);
    assert!(trace.borrow().iter().all(|(ply, _, _)| *ply == 2));
}

#[test]
fn d2e_positive_depth_ldse_preserves_routing_and_learning_state() {
    let original = position("7k/8/5K2/5Q2/8/8/8/8 b - - 0 1");
    let mv = generate_moves(&original, original.w, &original.cr, original.ep)[0];
    let mut enabled = searcher();
    let mut reference = searcher();
    for s in [&mut enabled, &mut reference] {
        s.shared_tt
            .store(original.hash, 0, score_to_tt(0, 1), TT_BETA, Some(mv));
        #[cfg(feature = "node-correction-training")]
        #[cfg(feature = "search-debug")]
        {
            s.debug.disable_corr_node_train = false;
        }
    }
    let _restore_tuning = enable_ldse(4, 1);
    let contexts = RefCell::new(Vec::new());
    let trace = RefCell::new(Vec::new());
    let mut a = original;
    let mut b = original;
    let mut na = 0;
    let mut nb = 0;
    let va = enabled.negamax_mode_scalar::<false, false, true, _>(
        &mut a,
        3,
        1,
        5000,
        5001,
        false,
        &mut na,
        Observed {
            inner: Counted {
                calls: &trace,
                stop_after_first: false,
            },
            contexts: &contexts,
        },
    );
    let vb = reference.negamax_mode_scalar::<false, false, false, _>(
        &mut b,
        3,
        1,
        5000,
        5001,
        false,
        &mut nb,
        ClassicEval,
    );
    assert_eq!(va, vb);
    assert!(
        na > nb,
        "positive-depth LDSE must retain its descendant work"
    );
    assert!(contexts
        .borrow()
        .iter()
        .any(|c| c.ply > 1 && c.restricted && c.ldse == 1));
    assert_eq!(
        crate::board::board_to_fen(&a),
        crate::board::board_to_fen(&original)
    );
    assert_eq!(a.mailbox, b.mailbox);
    assert!(!enabled.restricted_verification_active());
    assert_eq!(enabled.rep_stack, reference.rep_stack);
    assert_eq!(enabled.rep_stack_len, reference.rep_stack_len);
    assert_eq!(enabled.path_moves, reference.path_moves);
    assert_eq!(
        enabled.singular_extensions_used,
        reference.singular_extensions_used
    );
    assert_eq!(
        enabled.singular_negative_on_line,
        reference.singular_negative_on_line
    );
    assert_eq!(
        enabled.singular_ldse_on_line,
        reference.singular_ldse_on_line
    );
    assert_eq!(enabled.history, reference.history);
    assert_eq!(enabled.killers, reference.killers);
    assert_eq!(enabled.counter_move, reference.counter_move);
    assert_eq!(enabled.corr_hist, reference.corr_hist);
    assert_eq!(
        enabled.shared_tt.get_entry(original.hash),
        reference.shared_tt.get_entry(original.hash)
    );
    #[cfg(feature = "node-correction-training")]
    {
        assert_eq!(enabled.corr_staged_depth, reference.corr_staged_depth);
        assert_eq!(enabled.corr_staged_diff, reference.corr_staged_diff);
        assert_eq!(enabled.corr_staged_touched, reference.corr_staged_touched);
    }
    #[cfg(feature = "search-debug")]
    {
        assert_eq!(enabled.debug_stats().ldse_probes, 1);
        assert_eq!(enabled.debug_stats().ldse_cutoffs, 0);
        assert_eq!(na - enabled.debug_stats().ldse_probe_nodes, nb);
    }
}

#[test]
fn d2e_illegal_ldse_attempt_keeps_hint_invalidated_without_probe_flag() {
    let mut s = searcher();
    let mut st = position(QUIET);
    let original = st;
    let illegal = crate::board::encode_move(4, 4, 3, 4, 0);
    let mut trial = st;
    assert!(!try_apply_move_mode::<false>(&mut trial, illegal));
    let expected = ClassicEval.static_eval::<false>(&s, &st, 2);
    s.shared_tt
        .store(st.hash, 0, expected - 100, TT_BETA, Some(illegal));
    let _restore_tuning = enable_ldse(0, 1);
    let trace = RefCell::new(Vec::new());
    let mut nodes = 0;
    let alpha = expected + 2000;
    assert!(ldse_candidate(
        true,
        Some(illegal),
        Some(expected - 100),
        Some(TT_BETA),
        false,
        false,
        0,
        alpha,
        expected,
        0
    )
    .is_some());
    let score = s.negamax_mode_scalar::<false, false, true, _>(
        &mut st,
        0,
        2,
        alpha,
        alpha + 1,
        true,
        &mut nodes,
        Counted {
            calls: &trace,
            stop_after_first: false,
        },
    );
    assert_eq!(score, alpha);
    assert_eq!(nodes, 2, "an illegal attempt never enters a descendant");
    assert_eq!(
        trace.borrow().len(),
        2,
        "hint invalidation must precede legality and ldse_probed"
    );
    assert!(trace
        .borrow()
        .iter()
        .all(|(ply, hash, _)| *ply == 2 && *hash == original.hash));
    assert_eq!(
        crate::board::board_to_fen(&st),
        crate::board::board_to_fen(&original)
    );
    assert!(!s.restricted_verification_active());
    assert!(s.rep_stack.is_empty());
    #[cfg(feature = "search-debug")]
    assert_eq!(s.debug_stats().ldse_probes, 0);
}
