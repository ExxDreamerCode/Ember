use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chess_rs_lib::board::{
    encode_move, move_ec, move_er, move_promotion, move_sc, move_sr, move_to_uci, BoardState, Move,
    INF,
};
use chess_rs_lib::movegen::{apply_move, generate_moves};
use chess_rs_lib::search::{
    extract_pv_line, format_pv_line_uci, lazy_smp_search, LazySmpPool, LazySmpSearchLimits,
    Searcher,
};
use chess_rs_lib::syzygy::SyzygyTables;
use chess_rs_lib::tt::{SharedTT, TT_EXACT};
use chess_rs_lib::zobrist::compute_hash;
use chess_rs_lib::Engine;

fn state_from_fen(fen: &str) -> BoardState {
    let mut engine = Engine::new();
    engine.set_fen(fen);
    engine.st
}

fn legal_move(st: &BoardState, uci: &str) -> Move {
    generate_moves(st, st.w, &st.cr, st.ep)
        .into_iter()
        .find(|mv| move_to_uci(st, *mv) == uci)
        .unwrap_or_else(|| panic!("expected legal move {uci}"))
}

fn apply_encoded_move(st: &mut BoardState, mv: Move) {
    apply_move(
        st,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
}

#[test]
fn negamax_handles_stalemate_with_only_pseudo_king_moves() {
    let mut st = state_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(generate_moves(&st, st.w, &st.cr, st.ep).is_empty());
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut searcher = Searcher::new(shared_tt.clone(), stopped);
    searcher.init_nnue_stack(&st);
    let root_key = compute_hash(&st);
    let mut nodes = 0u64;

    let score = searcher.negamax(
        &mut st,
        2,
        0,
        -INF,
        INF,
        true,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    assert_eq!(score, 0);
    assert!(
        shared_tt
            .get_depth(root_key)
            .and_then(|(_, _, _, best)| best)
            .is_none(),
        "stalemate must not store a pseudo-legal best move"
    );
}

// This deliberately exercises negamax directly and verifies the move stored in its
// root TT entry. The public root driver applies separate ordering and extension policy,
// so a TSV case at the same nominal depth does not preserve this contract.
#[test]
fn negamax_prefers_en_passant_discovered_check() {
    let mut st = state_from_fen("8/6pp/8/R2pP1k1/6B1/8/6PP/6K1 w - d6 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut searcher = Searcher::new(shared_tt.clone(), stopped);
    searcher.init_nnue_stack(&st);
    let root_key = compute_hash(&st);
    let mut nodes = 0u64;

    let score = searcher.negamax(
        &mut st,
        2,
        0,
        -INF,
        INF,
        true,
        Instant::now(),
        10.0,
        &mut nodes,
    );

    let best_move = shared_tt
        .get_depth(root_key)
        .and_then(|(_, _, _, best_move)| best_move)
        .expect("negamax should store the root best move");
    let best_uci = move_to_uci(&st, best_move);
    assert_eq!(
        best_uci, "e5d6",
        "search chose {best_uci} instead of the checking en-passant discovery e5d6; score={score}, nodes={nodes}"
    );
}

#[test]
fn negamax_timeout_sets_stopped_without_storing_tt() {
    let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut searcher = Searcher::new(shared_tt.clone(), stopped);
    let key = compute_hash(&st);
    let mut nodes = 0u64;

    let score = searcher.negamax(
        &mut st,
        4,
        0,
        -INF,
        INF,
        true,
        Instant::now() - Duration::from_secs(1),
        0.0,
        &mut nodes,
    );

    assert_eq!(score, 0);
    assert!(searcher.stopped.load(Ordering::Relaxed));
    assert!(searcher.shared_tt.get_depth(key).is_none());
}

#[test]
fn lazy_smp_worker_context_copies_root_search_state() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let mut worker = Searcher::new(shared_tt, stopped);

    root.rep_stack.extend([11, 22, 33, 44]);
    root.rep_stack_len = 4;
    root.corr_hist[123] = 17;
    root.corr_hist[456] = -23;
    root.history[12][28] = 1_234;
    root.counter_move[7][31] = Some(encode_move(6, 0, 5, 0, 0));
    root.syzygy = SyzygyTables::new();

    root.copy_root_context_to(&mut worker);

    assert_eq!(worker.rep_stack, root.rep_stack);
    assert_eq!(worker.rep_stack_len, root.rep_stack_len);
    assert_eq!(worker.corr_hist[123], 17);
    assert_eq!(worker.corr_hist[456], -23);
    assert_eq!(worker.history[12][28], 1_234);
    assert_eq!(worker.counter_move[7][31], root.counter_move[7][31]);
    assert_eq!(worker.syzygy.tables.is_some(), root.syzygy.tables.is_some());
}

#[test]
fn search_learning_is_aged_between_moves_and_cleared_between_games() {
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(1));
    let mut searcher = Searcher::new(shared_tt, stopped);
    let reply = encode_move(6, 0, 5, 0, 0);
    searcher.killers[2][0] = Some(reply);
    searcher.history[12][28] = 1_600;
    searcher.counter_move[7][31] = Some(reply);
    searcher.corr_hist[123] = 19;

    searcher.prepare_for_search();

    assert_eq!(searcher.killers[2][0], None);
    assert_eq!(searcher.history[12][28], 1_300);
    assert_eq!(searcher.counter_move[7][31], Some(reply));
    assert_eq!(searcher.corr_hist[123], 19);

    searcher.clear_learning();

    assert_eq!(searcher.history[12][28], 0);
    assert_eq!(searcher.counter_move[7][31], None);
    assert_eq!(searcher.corr_hist[123], 0);
}

#[test]
fn lazy_smp_honors_the_root_searcher_stop_token() {
    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let stopped = Arc::new(AtomicBool::new(true));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    let (_, _, depth, nodes) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        |_, _| 0,
        LazySmpSearchLimits {
            soft_time: 10.0,
            hard_time: 10.0,
            depth: 4,
            start: Instant::now(),
        },
        2,
        &mut root,
    );

    assert_eq!(depth, 0, "workers searched despite an external stop");
    assert_eq!(nodes, 0, "workers counted nodes despite an external stop");
}

#[test]
fn lazy_smp_uses_the_caller_start_time() {
    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    let (_, _, depth, nodes) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        |_, _| 0,
        LazySmpSearchLimits {
            soft_time: 0.010,
            hard_time: 0.010,
            depth: 4,
            start: Instant::now() - Duration::from_secs(1),
        },
        2,
        &mut root,
    );

    assert_eq!(depth, 0, "workers ignored the expired caller clock");
    assert_eq!(nodes, 0, "workers searched after the caller clock expired");
    assert!(stopped.load(Ordering::Relaxed));
}

#[test]
fn lazy_smp_counts_work_from_an_interrupted_iteration() {
    static DEEP_ROOT_SEARCH_STARTED: AtomicBool = AtomicBool::new(false);

    fn start_a_deep_root_search(_: &BoardState, _: Move) -> i32 {
        DEEP_ROOT_SEARCH_STARTED.store(true, Ordering::SeqCst);
        12
    }

    let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);
    DEEP_ROOT_SEARCH_STARTED.store(false, Ordering::SeqCst);

    let stop_token = Arc::clone(&stopped);
    let stopper = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(1);
        while !DEEP_ROOT_SEARCH_STARTED.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let search_started = DEEP_ROOT_SEARCH_STARTED.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(5));
        stop_token.store(true, Ordering::SeqCst);
        search_started
    });

    let (_, _, depth, nodes) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        start_a_deep_root_search,
        LazySmpSearchLimits {
            soft_time: 10.0,
            hard_time: 10.0,
            depth: 1,
            start: Instant::now(),
        },
        1,
        &mut root,
    );

    assert!(
        stopper.join().expect("stopper thread completed"),
        "the root search did not start"
    );
    assert_eq!(depth, 0, "the interrupted iteration was not completed");
    assert!(
        nodes > 0,
        "interrupted search work disappeared from the total"
    );
}

#[test]
fn lazy_smp_soft_completion_signals_the_root_searcher() {
    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    let (_, _, depth, _) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        |_, _| 0,
        LazySmpSearchLimits {
            soft_time: 0.0,
            hard_time: 10.0,
            depth: 4,
            start: Instant::now(),
        },
        2,
        &mut root,
    );

    assert!(depth >= 1, "no worker completed the crossing iteration");
    assert!(
        stopped.load(Ordering::Relaxed),
        "the first crossing iteration did not stop sibling workers"
    );
}

#[test]
fn immature_lazy_smp_helper_cannot_end_the_leader_iteration_at_soft_time() {
    static EXPECTED_ROOT_MOVES: AtomicUsize = AtomicUsize::new(0);
    static HELPER_ROOT_VISITS: AtomicUsize = AtomicUsize::new(0);
    static LEADER_ROOT_VISITS: AtomicUsize = AtomicUsize::new(0);

    fn delay_leader_until_the_helper_finishes(_: &BoardState, _: Move) -> i32 {
        if std::thread::current().name() == Some("rts-0") {
            let deadline = Instant::now() + Duration::from_secs(1);
            let expected = EXPECTED_ROOT_MOVES.load(Ordering::SeqCst);
            while HELPER_ROOT_VISITS.load(Ordering::SeqCst) < expected && Instant::now() < deadline
            {
                std::thread::yield_now();
            }
            LEADER_ROOT_VISITS.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(5));
        } else {
            HELPER_ROOT_VISITS.fetch_add(1, Ordering::SeqCst);
        }
        0
    }

    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    EXPECTED_ROOT_MOVES.store(root_moves.len(), Ordering::SeqCst);
    HELPER_ROOT_VISITS.store(0, Ordering::SeqCst);
    LEADER_ROOT_VISITS.store(0, Ordering::SeqCst);
    let (_, _, depth, _) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        delay_leader_until_the_helper_finishes,
        LazySmpSearchLimits {
            soft_time: 0.0,
            hard_time: 10.0,
            depth: 1,
            start: Instant::now(),
        },
        2,
        &mut root,
    );

    assert_eq!(depth, 1);
    assert!(stopped.load(Ordering::Relaxed));
    assert_eq!(HELPER_ROOT_VISITS.load(Ordering::SeqCst), root_moves.len());
    assert_eq!(
        LEADER_ROOT_VISITS.load(Ordering::SeqCst),
        root_moves.len(),
        "a helper stopped the leader before its crossing iteration completed"
    );
}

#[test]
fn lazy_smp_applies_root_depth_extension_policy() {
    static EXTENSION_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn count_extension_calls(_: &BoardState, _: Move) -> i32 {
        EXTENSION_CALLS.fetch_add(1, Ordering::SeqCst);
        0
    }

    let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    let stopped = Arc::new(AtomicBool::new(false));
    let shared_tt = Arc::new(SharedTT::new(128));
    let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
    let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

    EXTENSION_CALLS.store(0, Ordering::SeqCst);
    let (_, _, depth, _) = lazy_smp_search(
        &LazySmpPool::new(),
        shared_tt,
        &st,
        &root_moves,
        count_extension_calls,
        LazySmpSearchLimits {
            soft_time: 10.0,
            hard_time: 10.0,
            depth: 1,
            start: Instant::now(),
        },
        1,
        &mut root,
    );

    assert_eq!(depth, 1);
    assert_eq!(
        EXTENSION_CALLS.load(Ordering::SeqCst),
        root_moves.len(),
        "Lazy SMP did not consult the root extension policy for every root move"
    );
}

#[test]
fn root_search_resets_previous_timeout_state() {
    let mut engine = Engine::new();
    engine.book = None;
    engine.searcher.set_stopped();

    let (best_move, _, nodes, _) = engine.find_best_move(1.0, 1);

    assert_ne!(best_move, "0000");
    assert!(nodes > 0);
    assert!(!engine.searcher.stopped.load(Ordering::Relaxed));
}

#[test]
fn extract_pv_rejects_illegal_first_move_without_promotion() {
    let st = state_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let shared_tt = Arc::new(SharedTT::new(128));

    let bogus = encode_move(1, 2, 0, 0, 0);
    let pv = extract_pv_line(&shared_tt, &st, bogus);
    assert_eq!(pv.len(), 1);
}

#[test]
fn extract_pv_rejects_illegal_tt_move_during_extraction() {
    let st = state_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let shared_tt = Arc::new(SharedTT::new(128));

    let first_move = encode_move(7, 4, 6, 4, 0);
    let bogus_tt_move = encode_move(0, 0, 0, 7, 0);
    let after_st = {
        let mut s = st;
        apply_move(&mut s, 7, 4, 6, 4, 0);
        s
    };
    let after_hash = compute_hash(&after_st);
    shared_tt.store(after_hash, 5, 100, TT_EXACT, Some(bogus_tt_move));

    let pv = extract_pv_line(&shared_tt, &st, first_move);
    assert_eq!(pv.len(), 1, "extract_pv must reject illegal TT moves");
}

#[test]
fn extract_pv_validates_takes_back_king_in_check() {
    let st = state_from_fen("4k3/4r3/8/8/8/8/8/4K3 w - - 0 1");
    let shared_tt = Arc::new(SharedTT::new(128));

    let first_move = encode_move(7, 4, 6, 4, 0);
    let after_st = {
        let mut s = st;
        apply_move(&mut s, 7, 4, 6, 4, 0);
        s
    };
    let after_hash = compute_hash(&after_st);
    let check_move = encode_move(6, 4, 6, 5, 0);
    shared_tt.store(after_hash, 5, 100, TT_EXACT, Some(check_move));

    let pv = extract_pv_line(&shared_tt, &st, first_move);
    assert_eq!(
        pv.len(),
        1,
        "extract_pv must pop moves that leave king in check"
    );
}

#[test]
fn format_pv_line_uci_walks_successive_positions() {
    let st = state_from_fen("2n4k/8/1P6/8/8/8/8/7K w - - 0 1");
    let first = legal_move(&st, "b6b7");
    let mut after_first = st;
    apply_encoded_move(&mut after_first, first);
    let reply = legal_move(&after_first, "h8g8");
    let promotion = encode_move(1, 1, 0, 2, b'Q');

    let pv = [first, reply, promotion];

    assert_eq!(format_pv_line_uci(&st, &pv), "b6b7 h8g8 b7c8q");
}

#[test]
fn format_pv_line_uci_stops_before_suffixless_promotion() {
    let st = state_from_fen("2n4k/8/1P6/8/8/8/8/7K w - - 0 1");
    let first = legal_move(&st, "b6b7");
    let mut after_first = st;
    apply_encoded_move(&mut after_first, first);
    let reply = legal_move(&after_first, "h8g8");
    let suffixless_promotion = encode_move(1, 1, 0, 2, 0);

    let pv = [first, reply, suffixless_promotion];

    assert_eq!(format_pv_line_uci(&st, &pv), "b6b7 h8g8");
}
