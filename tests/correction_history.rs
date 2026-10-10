use ember_chess::Engine;

// This integration executable installs only the embedded network or classic
// evaluation. Restore its prior shared selection even when an assertion fails.
struct RestoreNetwork(bool);

impl RestoreNetwork {
    fn capture() -> Self {
        assert!(ember_chess::evaluate::current_nnue_net().is_none());
        assert!(ember_chess::evaluate::current_classic_net().is_none());
        Self(ember_chess::evaluate::current_ember_v2().is_some())
    }
}

impl Drop for RestoreNetwork {
    fn drop(&mut self) {
        if self.0 {
            ember_chess::evaluate::init_embedded_nnue().unwrap();
        } else {
            ember_chess::evaluate::reset_nnue().unwrap();
        }
    }
}

#[test]
fn correction_history_engine_callers_match_fresh_root_training() {
    use ember_chess::{evaluate, search::Searcher, tt::SharedTT};
    use std::sync::{atomic::AtomicBool, Arc};

    // Public learning-state contract, not a root-move expectation: the engine
    // must train exactly as a fresh evaluation of the pre-search root would.
    // Reuse each engine across changed roots to catch caches outliving a search.
    // Child-node training is disabled here because it writes shared pawn slots
    // and is covered separately by the node-training test below.
    let _restore = RestoreNetwork::capture();
    for (v2, chess960) in [(false, false), (true, true), (true, false)] {
        if v2 {
            evaluate::init_embedded_nnue().unwrap();
        } else {
            evaluate::reset_nnue().unwrap();
        }
        for multi_pv in [1, 3] {
            let mut engine = Engine::new();
            engine.book = None;
            engine.own_book = false;
            engine.multi_pv = multi_pv;
            // MultiPV still routes through the non-SMP branch with Threads > 1.
            engine.num_threads = if multi_pv == 1 { 1 } else { 4 };
            engine.searcher.corr_hist.fill(37);
            let mut trained = false;
            let mut residuals = Vec::new();
            for fen in [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 2",
                "2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14",
                "8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ] {
                engine.st.chess960 = chess960;
                engine.set_fen(fen);
                engine.searcher.refresh_nnue_net();
                engine.searcher.refresh_search_backend();
                #[cfg(feature = "search-debug")]
                {
                    engine.searcher.debug.disable_corr_hist = false;
                    engine.searcher.debug.disable_corr_node_train = true;
                    engine.searcher.debug.enable_endgame_mopup = false;
                }
                let mut oracle =
                    Searcher::new(Arc::new(SharedTT::new(1)), Arc::new(AtomicBool::new(false)));
                engine.searcher.copy_root_context_to(&mut oracle);
                #[cfg(feature = "search-debug")]
                {
                    oracle.debug.disable_corr_hist = engine.searcher.debug.disable_corr_hist;
                    oracle.debug.enable_endgame_mopup = engine.searcher.debug.enable_endgame_mopup;
                }
                let before = oracle.corr_hist;
                let root = engine.st;
                let (_, score, nodes, _) = engine.find_best_move_prepared_untimed(3, None);
                assert!(nodes > 0);
                assert!(!engine.stopped.load(std::sync::atomic::Ordering::Relaxed));
                assert_eq!(engine.st.bb, root.bb);
                assert_eq!(engine.st.hash, root.hash);
                residuals.push(score - oracle.corrected_eval(&root));
                oracle.update_correction_history(&root, score, 3);
                trained |= oracle.corr_hist != before;
                assert_eq!(
                    engine.searcher.corr_hist, oracle.corr_hist,
                    "v2={v2} Chess960={chess960} MultiPV={multi_pv} fen={fen}"
                );
            }
            assert!(
                trained,
                "fixture must exercise training: v2={v2} Chess960={chess960} MultiPV={multi_pv} residuals={residuals:?}"
            );
        }
    }
}

#[test]
#[cfg(feature = "search-debug")]
fn correction_history_trains_at_child_nodes() {
    use ember_chess::{evaluate, search::Searcher, tt::SharedTT};
    use std::sync::{atomic::AtomicBool, Arc};

    // Public learning-state contract: a fixed-depth search must train the
    // correction slots of its child nodes, not only the post-search root.
    // The post-search root call alone provably cannot move any slot other
    // than the root's pawn slot with the root-only arithmetic (the fidelity
    // test above pins that equality), so any divergence from a fresh
    // root-only oracle proves that child nodes trained during the search.
    let _restore = RestoreNetwork::capture();
    evaluate::reset_nnue().unwrap();
    let mut engine = Engine::new();
    engine.book = None;
    engine.own_book = false;
    engine.num_threads = 1;
    engine.set_fen("2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14");
    engine.searcher.refresh_nnue_net();
    engine.searcher.refresh_search_backend();
    engine.searcher.debug.disable_corr_hist = false;
    engine.searcher.debug.disable_corr_node_train = false;
    engine.searcher.debug.enable_endgame_mopup = false;

    let mut oracle = Searcher::new(Arc::new(SharedTT::new(1)), Arc::new(AtomicBool::new(false)));
    engine.searcher.copy_root_context_to(&mut oracle);
    oracle.debug.disable_corr_node_train = true;
    let root = engine.st;

    // Depth 4 gives the root's children the depth-3 node gate; the position
    // includes pawn moves whose subtrees miss the warm transposition table
    // and search fully.
    let (_, score, nodes, _) = engine.find_best_move_prepared_untimed(4, None);
    assert!(nodes > 0);
    let stats = engine.searcher.debug_stats();
    assert!(
        stats.corr_node_exits > 0,
        "completed nodes must reach the training gate: {stats:?}"
    );
    oracle.update_correction_history(&root, score, 4);
    assert_ne!(
        engine.searcher.corr_hist, oracle.corr_hist,
        "child nodes must train during the search, not only the post-search root: {stats:?}"
    );
}

#[test]
fn correction_history_public_updater_observes_successive_training() {
    let _restore = RestoreNetwork::capture();
    // Public state contract: a fresh call must see the correction learned by
    // its predecessor. This is required by SMP's unchanged public callers.
    ember_chess::evaluate::reset_nnue().unwrap();
    let mut engine = Engine::new();
    engine.searcher.refresh_nnue_net();
    #[cfg(feature = "search-debug")]
    {
        engine.searcher.debug.disable_corr_hist = false;
        engine.searcher.debug.enable_endgame_mopup = false;
    }
    engine.searcher.corr_hist.fill(100);
    let score = engine.searcher.corrected_eval(&engine.st) + 64;
    engine
        .searcher
        .update_correction_history(&engine.st, score, 3);
    let first = engine.searcher.corr_hist;
    engine
        .searcher
        .update_correction_history(&engine.st, score, 3);
    let changes: Vec<_> = first
        .iter()
        .zip(engine.searcher.corr_hist)
        .filter(|(a, b)| **a != *b)
        .collect();
    assert_eq!(changes.len(), 1);
    // Depth-3 arithmetic with default constants: residual 64 gives bonus
    // 64*4*10/55 = 46 and gravity 100*46/1024 = 4, so the first update lands
    // at 142; the second sees residual 22, bonus 16, gravity 2 -> 156.
    assert_eq!(*changes[0].0, 142);
    assert_eq!(changes[0].1, 156);
}
