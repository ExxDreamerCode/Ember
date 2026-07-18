use std::collections::BTreeSet;
use std::time::Instant;

use chess_rs_lib::board::{
    bit, board_to_fen, is_dead_position, move_ec, move_er, move_promotion, move_sc, move_sr,
    move_to_uci, piece_on, sq, EMPTY_SQ, INF, MATE, WK, WR,
};
use chess_rs_lib::movegen::{apply_move, generate_moves};
use chess_rs_lib::syzygy::SyzygyTables;
use chess_rs_lib::tt::TT_EXACT;
use chess_rs_lib::zobrist::compute_hash;
use chess_rs_lib::Engine;
use shakmaty::{fen::Fen, perft as shakmaty_perft, CastlingMode, Chess, Position};

fn engine_from_fen(fen: &str, chess960: bool) -> Engine {
    let mut engine = Engine::new();
    engine.book = None;
    engine.st.chess960 = chess960;
    engine.set_fen(fen);
    engine
}

fn search_score(engine: &mut Engine, depth: i32, ply: usize, alpha: i32, beta: i32) -> i32 {
    let mut st = engine.st;
    let mut nodes = 0;
    engine.searcher.negamax(
        &mut st,
        depth,
        ply,
        alpha,
        beta,
        true,
        Instant::now(),
        30.0,
        &mut nodes,
    )
}

fn ember_legal_moves(fen: &str, chess960: bool) -> BTreeSet<String> {
    let engine = engine_from_fen(fen, chess960);
    generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
        .iter()
        .map(|&mv| move_to_uci(&engine.st, mv))
        .collect()
}

fn reference_position(fen: &str, chess960: bool) -> Chess {
    let mode = if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    };
    fen.parse::<Fen>()
        .expect("valid FEN")
        .into_position(mode)
        .expect("legal reference position")
}

fn reference_legal_moves(fen: &str, chess960: bool) -> BTreeSet<String> {
    let mode = if chess960 {
        CastlingMode::Chess960
    } else {
        CastlingMode::Standard
    };
    reference_position(fen, chess960)
        .legal_moves()
        .into_iter()
        .map(|mv| mv.to_uci(mode).to_string())
        .collect()
}

fn ember_perft_state(st: &chess_rs_lib::board::BoardState, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_moves(st, st.w, &st.cr, st.ep);
    if depth == 1 {
        return moves.len() as u64;
    }

    moves
        .into_iter()
        .map(|mv| {
            let mut next = *st;
            apply_move(
                &mut next,
                move_sr(mv),
                move_sc(mv),
                move_er(mv),
                move_ec(mv),
                move_promotion(mv),
            );
            ember_perft_state(&next, depth - 1)
        })
        .sum()
}

fn assert_move_sets_match_reference(fen: &str, chess960: bool) {
    assert_eq!(
        ember_legal_moves(fen, chess960),
        reference_legal_moves(fen, chess960),
        "legal move mismatch for {fen}"
    );
}

#[test]
fn standard_legal_moves_match_reference_for_rule_cases() {
    let cases = [
        // Opening pawn/knight movement.
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // Castling, checks, pins, and ordinary captures from the classic kiwipete suite.
        "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
        // Castling through an attacked square is illegal.
        "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1",
        // Legal en passant.
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        // En passant that would expose a horizontal rook check is illegal.
        "8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1",
        // Promotions and underpromotions.
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        // Checkmate.
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
        // Stalemate.
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
    ];

    for fen in cases {
        assert_move_sets_match_reference(fen, false);
    }
}

#[test]
fn chess960_legal_moves_match_reference_for_castling_cases() {
    let cases = [
        "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        "qnr1bkrb/pppp2pp/3np3/5p2/8/P2P2P1/NPP1PP1P/QN1RBKRB w GDg - 3 9",
        // The king is already on the kingside destination square; only the rook moves.
        "6kr/8/8/8/8/8/8/6KR w Hh - 0 1",
    ];

    for fen in cases {
        assert_move_sets_match_reference(fen, true);
    }
}

#[test]
fn perft_matches_reference_for_rule_positions() {
    let cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false, 2),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false, 2),
        ("8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1", false, 2),
        ("4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1", false, 2),
        (
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
            true,
            2,
        ),
        ("6kr/8/8/8/8/8/8/6KR w Hh - 0 1", true, 2),
    ];

    for (fen, chess960, depth) in cases {
        let engine = engine_from_fen(fen, chess960);
        let reference = reference_position(fen, chess960);
        assert_eq!(
            ember_perft_state(&engine.st, depth),
            shakmaty_perft(&reference, depth),
            "perft mismatch for {fen}"
        );
    }
}

#[test]
fn chess960_castling_right_is_revoked_when_non_corner_rook_moves() {
    let mut engine = engine_from_fen("4k3/8/8/8/8/8/8/1R2K1R1 w GB - 0 1", true);

    let before = ember_legal_moves("4k3/8/8/8/8/8/8/1R2K1R1 w GB - 0 1", true);
    assert!(before.contains("e1g1"));
    assert!(before.contains("e1b1"));

    assert!(
        engine.make_move_uci(7, 6, 7, 5, 0),
        "g1f1 rook move should be legal"
    );
    assert!(
        !engine.st.cr[0],
        "moving the g-file castling rook must revoke O-O"
    );
    assert!(
        engine.make_move_uci(0, 4, 1, 4, 0),
        "e8e7 waiting move should be legal"
    );

    let moves: BTreeSet<String> =
        generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
            .iter()
            .map(|&mv| move_to_uci(&engine.st, mv))
            .collect();
    assert!(
        !moves.contains("e1f1"),
        "moved rook on f1 must not become a new castling rook"
    );
    assert!(
        !moves.contains("e1g1"),
        "kingside castling right must stay revoked"
    );
}

#[test]
fn standard_castling_right_is_revoked_when_corner_rook_moves() {
    let mut engine = engine_from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", false);

    assert!(
        engine.make_move_uci(7, 7, 6, 7, 0),
        "h1h2 rook move should be legal"
    );
    assert!(!engine.st.cr[0], "moving h1 rook must revoke O-O");
    assert!(
        engine.make_move_uci(0, 4, 1, 4, 0),
        "e8e7 waiting move should be legal"
    );

    let moves: BTreeSet<String> =
        generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
            .iter()
            .map(|&mv| move_to_uci(&engine.st, mv))
            .collect();
    assert!(!moves.contains("e1g1"));
    assert!(
        moves.contains("e1c1"),
        "queenside castling right should remain"
    );
}

#[test]
fn chess960_castling_works_when_king_already_on_destination() {
    let mut engine = engine_from_fen("6kr/8/8/8/8/8/8/6KR w Hh - 0 1", true);
    let moves = ember_legal_moves("6kr/8/8/8/8/8/8/6KR w Hh - 0 1", true);
    assert!(moves.contains("g1h1"));

    assert!(engine.make_move_uci(7, 6, 7, 7, 0));
    assert_eq!(engine.st.bb[WK], bit(sq(7, 6)), "king remains on g1");
    assert!(engine.st.bb[WR] & bit(sq(7, 5)) != 0, "rook moves to f1");
    assert_eq!(piece_on(&engine.st.bb, sq(7, 7)), EMPTY_SQ, "h1 is vacated");
}

#[test]
fn chess960_castling_keeps_incremental_hash_aligned() {
    let engine = engine_from_fen("6kr/8/8/8/8/8/8/6KR w Hh - 0 1", true);
    let mv = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
        .into_iter()
        .find(|&mv| move_to_uci(&engine.st, mv) == "g1h1")
        .expect("expected legal Chess960 castling move g1h1");

    let mut next = engine.st;
    apply_move(
        &mut next,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );

    assert_eq!(
        next.hash,
        compute_hash(&next),
        "incremental hash must match recomputed hash after Chess960 castling; reached {}",
        board_to_fen(&next)
    );
}

#[test]
fn illegal_uci_move_is_rejected_without_mutating_state() {
    let mut engine = Engine::new();
    let before_state = engine.st;
    let before_hash = compute_hash(&engine.st);
    let before_rep_len = engine.searcher.rep_stack_len;

    assert!(
        !engine.make_move_uci(6, 4, 3, 4, 0),
        "e2e5 is illegal from startpos"
    );
    assert_eq!(compute_hash(&engine.st), before_hash);
    assert_eq!(engine.searcher.rep_stack_len, before_rep_len);
    assert_eq!(engine.st.bb, before_state.bb);
    assert_eq!(engine.st.w, before_state.w);
}

#[test]
fn default_position_move_keeps_repetition_stack_aligned() {
    let mut engine = Engine::new();
    assert_eq!(
        engine.searcher.rep_stack_len, 1,
        "new engine should expose one active repetition hash"
    );
    assert_eq!(
        engine.searcher.rep_stack.len(),
        engine.searcher.rep_stack_len,
        "new engine should not keep inactive hashes ahead of rep_stack_len"
    );
    assert_eq!(engine.searcher.rep_stack[0], compute_hash(&engine.st));

    assert!(engine.make_move_uci(6, 4, 4, 4, 0), "e2e4 is legal");
    let current_hash = compute_hash(&engine.st);
    assert_eq!(
        engine.searcher.rep_stack[engine.searcher.rep_stack_len - 1],
        current_hash,
        "latest repetition hash must describe the current board"
    );
}

#[test]
fn new_engine_reports_default_hash_size() {
    let engine = Engine::new();
    assert_eq!(
        engine.searcher.tt_mb, 256,
        "recorded hash size should match the engine default"
    );
}

#[test]
fn invalid_fen_does_not_replace_current_position() {
    let mut engine = Engine::new();
    let before_fen = board_to_fen(&engine.st);
    let before_hash = compute_hash(&engine.st);
    let before_rep_len = engine.searcher.rep_stack_len;

    engine.set_fen("8/8/8/8/8/8/8/8 w - - 0 1");

    assert_eq!(
        board_to_fen(&engine.st),
        before_fen,
        "invalid FEN must not replace the current board"
    );
    assert_eq!(compute_hash(&engine.st), before_hash);
    assert_eq!(engine.searcher.rep_stack_len, before_rep_len);
}

#[test]
fn halfmove_clock_is_preserved_updated_and_adjudicated() {
    let mut quiet = engine_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 37 12", false);
    assert_eq!(quiet.st.halfmove_clock, 37);
    assert_eq!(board_to_fen(&quiet.st), "4k3/8/8/8/8/8/8/R3K3 w - - 37 12");
    assert!(quiet.make_move_uci(7, 0, 6, 0, 0), "Ra1-a2 is legal");
    assert_eq!(
        quiet.st.halfmove_clock, 38,
        "quiet moves increment the clock"
    );

    let mut pawn = engine_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 88 1", false);
    assert!(pawn.make_move_uci(6, 4, 4, 4, 0), "e2-e4 is legal");
    assert_eq!(pawn.st.halfmove_clock, 0, "pawn moves reset the clock");

    let mut capture = engine_from_fen("n3k3/8/8/8/8/8/8/R3K3 w - - 99 1", false);
    assert!(capture.make_move_uci(7, 0, 0, 0, 0), "Ra1xa8 is legal");
    assert_eq!(capture.st.halfmove_clock, 0, "captures reset the clock");

    let mut adjudication = engine_from_fen("6k1/8/8/8/8/8/R7/K7 w - - 99 1", false);
    let (_, score, _, _) = adjudication.find_best_move(1_000_000.0, 1);
    assert_eq!(score, 0, "a quiet 100th halfmove is scored as a draw");
}

#[test]
fn checkmate_on_the_hundredth_halfmove_outranks_the_draw_threshold() {
    let mut engine = engine_from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 99 1", false);
    assert!(engine.make_move_uci(2, 6, 1, 6, 0), "Qg7# is legal");
    assert_eq!(engine.st.halfmove_clock, 100);
    assert!(engine.is_check());
    assert!(
        generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep).is_empty(),
        "Qg7 is checkmate"
    );

    assert_eq!(
        search_score(&mut engine, 2, 1, -INF, INF),
        -MATE + 1,
        "checkmate must end the game before a draw threshold is considered"
    );
}

#[test]
fn automatic_draw_outranks_a_transposition_table_cutoff() {
    let mut engine = engine_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 150 1", false);
    engine
        .shared_tt
        .store(engine.st.hash, 8, 1234, TT_EXACT, None);

    assert_eq!(
        search_score(&mut engine, 2, 1, -1, 0),
        0,
        "a cached board score must not override an automatic draw"
    );
}

#[test]
fn twofold_repetition_is_not_adjudicated_as_threefold() {
    let mut engine = engine_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 0 1", false);
    let cycle = [(7, 1, 7, 2), (0, 7, 0, 6), (7, 2, 7, 1), (0, 6, 0, 7)];
    for (sr, sc, er, ec) in cycle {
        assert!(engine.make_move_uci(sr, sc, er, ec, 0));
    }
    assert_eq!(
        engine
            .searcher
            .rep_stack
            .iter()
            .filter(|&&hash| hash == engine.st.hash)
            .count(),
        2
    );

    let twofold_score = search_score(&mut engine, 2, 1, -INF, INF);
    let mut control = engine_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 4 3", false);
    let control_score = search_score(&mut control, 2, 1, -INF, INF);
    assert!(
        control_score > 0,
        "the material-winning control scores above draw"
    );
    assert_eq!(
        twofold_score, control_score,
        "a second occurrence is not yet a claimable draw"
    );

    for (sr, sc, er, ec) in cycle {
        assert!(engine.make_move_uci(sr, sc, er, ec, 0));
    }
    assert_eq!(
        search_score(&mut engine, 2, 1, -INF, INF),
        0,
        "the third occurrence is claimable"
    );
}

#[test]
fn dead_material_is_adjudicated_without_false_minor_piece_draws() {
    for fen in [
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/KB6 w - - 0 1",
        "7k/8/8/8/8/8/8/KN6 w - - 0 1",
        "7k/8/8/8/8/3b4/8/KB6 w - - 0 1",
    ] {
        let engine = engine_from_fen(fen, false);
        assert!(
            is_dead_position(&engine.st),
            "expected dead position: {fen}"
        );
    }

    for fen in [
        "7k/8/8/8/8/8/8/KNN5 w - - 0 1",
        "7k/8/8/8/8/8/8/KBN5 w - - 0 1",
        "7k/8/8/8/8/2b5/8/KB6 w - - 0 1",
        "7k/8/8/8/8/8/P7/K7 w - - 0 1",
    ] {
        let engine = engine_from_fen(fen, false);
        assert!(
            !is_dead_position(&engine.st),
            "mating material remains: {fen}"
        );
    }

    let mut engine = engine_from_fen("7k/8/8/8/8/8/8/KB6 w - - 0 1", false);
    assert_eq!(
        search_score(&mut engine, 2, 1, -INF, INF),
        0,
        "K+B versus K is an immediate dead-position draw"
    );
}

#[test]
fn repetition_hash_only_includes_legal_en_passant_rights() {
    let legal_ep = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false);
    let same_without_ep = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1", false);
    assert_ne!(
        compute_hash(&legal_ep.st),
        compute_hash(&same_without_ep.st),
        "a legal en-passant capture changes the legal move set and must affect repetition hash"
    );

    let no_capture_ep = engine_from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1", false);
    let no_capture_without_ep = engine_from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1", false);
    assert_eq!(
        compute_hash(&no_capture_ep.st),
        compute_hash(&no_capture_without_ep.st),
        "a non-capturable en-passant target must not affect repetition hash"
    );

    let pinned_ep = engine_from_fen("8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1", false);
    let pinned_without_ep = engine_from_fen("8/8/8/r2pP2K/8/8/8/4k3 w - - 0 1", false);
    assert_eq!(
        compute_hash(&pinned_ep.st),
        compute_hash(&pinned_without_ep.st),
        "an en-passant target that is illegal because of self-check must not affect repetition hash"
    );
}

#[test]
fn root_search_and_lazy_smp_return_only_legal_moves() {
    for (fen, chess960) in [
        (
            "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
            false,
        ),
        ("6kr/8/8/8/8/8/8/6KR w Hh - 0 1", true),
    ] {
        for threads in [1usize, 2] {
            let mut engine = engine_from_fen(fen, chess960);
            engine.num_threads = threads;
            let legal = ember_legal_moves(fen, chess960);
            let (best_move, _, _, _) = engine.find_best_move(1.0, 1);
            assert!(
                legal.contains(&best_move),
                "threads={threads} produced illegal bestmove {best_move} for {fen}"
            );
        }
    }
}

#[test]
fn syzygy_without_loaded_tables_is_safe_and_search_stays_legal() {
    let fen = "8/8/8/8/8/4k3/8/R3K3 w - - 0 1";
    let mut engine = engine_from_fen(fen, false);
    engine.searcher.syzygy = SyzygyTables::new();

    assert!(SyzygyTables::pieces_ok(&engine.st));
    assert!(engine.searcher.syzygy.probe_wdl(&engine.st).is_none());

    let legal = ember_legal_moves(fen, false);
    let (best_move, _, _, _) = engine.find_best_move(1.0, 1);
    assert!(legal.contains(&best_move));
}
