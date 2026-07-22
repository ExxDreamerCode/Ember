use chess_rs_lib::board::{
    bit, encode_move, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    move_to_uci, piece_on, piece_type, sq, sq_c, sq_r, BoardState, Move, BR, EMPTY_SQ, WR,
};
use chess_rs_lib::movegen::*;
use chess_rs_lib::zobrist::compute_hash;
use chess_rs_lib::Engine;
use std::collections::BTreeSet;

fn state_from_fen(fen: &str) -> BoardState {
    let mut engine = Engine::new();
    engine.set_fen(fen);
    engine.st
}

fn state_from_fen_chess960(fen: &str) -> BoardState {
    let mut engine = Engine::new();
    engine.set_fen(fen);
    engine.st.chess960 = true;
    engine.st.hash = compute_hash(&engine.st);
    engine.st
}

fn assert_same_state(left: &BoardState, right: &BoardState) {
    assert_eq!(left.bb, right.bb);
    assert_eq!(left.mailbox, right.mailbox);
    assert_eq!(left.w, right.w);
    assert_eq!(left.cr, right.cr);
    assert_eq!(left.castling_rooks, right.castling_rooks);
    assert_eq!(left.ep, right.ep);
    assert_eq!(left.mc, right.mc);
    assert_eq!(left.chess960, right.chess960);
}

fn move_name_set(st: &BoardState, moves: impl IntoIterator<Item = Move>) -> BTreeSet<String> {
    moves.into_iter().map(|mv| move_to_uci(st, mv)).collect()
}

fn filtered_pseudo_names(st: &BoardState) -> BTreeSet<String> {
    generate_pseudo_moves(st, st.w, &st.cr, st.ep)
        .into_iter()
        .filter(|mv| {
            let mut next = *st;
            try_apply_move(&mut next, *mv)
        })
        .map(|mv| move_to_uci(st, mv))
        .collect()
}

fn legal_names(st: &BoardState) -> BTreeSet<String> {
    move_name_set(st, generate_moves(st, st.w, &st.cr, st.ep))
}

fn is_legal_tactical(st: &BoardState, mv: Move) -> bool {
    let from = move_from(mv);
    let to = move_to(mv);
    let fpi = st.mailbox[from];
    let tpi = st.mailbox[to];
    if fpi == EMPTY_SQ {
        return false;
    }
    let promotion = piece_type(fpi) == 0 && (sq_r(to) == 0 || sq_r(to) == 7);
    let en_passant =
        piece_type(fpi) == 0 && Some(to) == st.ep && sq_c(from) != sq_c(to) && tpi == EMPTY_SQ;
    let capture = !is_chess960_castling_move(st, mv) && (tpi != EMPTY_SQ || en_passant);
    capture || promotion || move_promotion(mv) != 0
}

fn filtered_pseudo_tactical_names(st: &BoardState) -> BTreeSet<String> {
    let mut pseudo = Vec::new();
    generate_pseudo_captures_promotions_into(st, st.w, &st.cr, st.ep, &mut pseudo);
    pseudo
        .into_iter()
        .filter(|mv| {
            let mut next = *st;
            try_apply_move(&mut next, *mv)
        })
        .map(|mv| move_to_uci(st, mv))
        .collect()
}

#[test]
fn filtered_pseudo_moves_match_legal_moves_for_rule_positions() {
    let positions = [
        state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        state_from_fen("4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1"),
        state_from_fen("k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"),
        state_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        state_from_fen_chess960("6kr/8/8/8/8/8/8/6KR w Hh - 0 1"),
    ];

    for st in positions {
        assert_eq!(filtered_pseudo_names(&st), legal_names(&st));
    }
}

#[test]
fn filtered_pseudo_tactical_moves_match_legal_tactical_subset() {
    let positions = [
        state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        state_from_fen("4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1"),
        state_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
        state_from_fen("7k/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ];

    for st in positions {
        let legal_tactical = generate_moves(&st, st.w, &st.cr, st.ep)
            .into_iter()
            .filter(|&mv| is_legal_tactical(&st, mv));
        assert_eq!(
            filtered_pseudo_tactical_names(&st),
            move_name_set(&st, legal_tactical)
        );
    }
}

#[test]
fn try_apply_rejects_pinned_move_without_mutating_state() {
    let st = state_from_fen("k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1");
    let mv = encode_move(6, 4, 6, 3, 0);
    assert!(generate_pseudo_moves(&st, st.w, &st.cr, st.ep).contains(&mv));

    let mut next = st;
    assert!(!try_apply_move(&mut next, mv));
    assert_same_state(&next, &st);
}

#[test]
fn try_apply_rejects_en_passant_self_check_without_mutating_state() {
    let st = state_from_fen("4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1");
    let mv = encode_move(3, 4, 2, 3, 0);
    assert!(generate_pseudo_moves(&st, st.w, &st.cr, st.ep).contains(&mv));

    let mut next = st;
    assert!(!try_apply_move(&mut next, mv));
    assert_same_state(&next, &st);
}

#[test]
fn try_apply_rejects_standard_castling_through_check() {
    let st = state_from_fen("4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1");
    let mv = encode_move(7, 4, 7, 6, 0);
    assert!(generate_pseudo_moves(&st, st.w, &st.cr, st.ep).contains(&mv));

    let mut next = st;
    assert!(!try_apply_move(&mut next, mv));
    assert_same_state(&next, &st);
}

#[test]
fn try_apply_rejects_chess960_castling_through_check() {
    let mut st = state_from_fen_chess960("1k6/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    st.bb[BR] |= bit(sq(0, 5));
    st.refresh_mailbox();
    let mv = encode_move(7, 4, 7, 7, 0);
    assert!(generate_pseudo_moves(&st, st.w, &st.cr, st.ep).contains(&mv));

    let before = st;
    assert!(!try_apply_move(&mut st, mv));
    assert_same_state(&st, &before);
}

fn perft(st: &BoardState, depth: u32) -> u64 {
    debug_assert_eq!(
        st.hash,
        compute_hash(st),
        "incremental hash diverged from compute_hash at depth {depth}"
    );
    if depth == 0 {
        return 1;
    }
    let moves = generate_moves(st, st.w, &st.cr, st.ep);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for mv in moves {
        let mut next = *st;
        apply_move(
            &mut next,
            move_sr(mv),
            move_sc(mv),
            move_er(mv),
            move_ec(mv),
            move_promotion(mv),
        );
        debug_assert_eq!(
            next.hash,
            compute_hash(&next),
            "incremental hash diverged from compute_hash after {}{}{}{}",
            move_sr(mv),
            move_sc(mv),
            move_er(mv),
            move_ec(mv)
        );
        nodes += perft(&next, depth - 1);
    }
    nodes
}

#[test]
fn incremental_hash_matches_recompute_on_special_move_positions() {
    let positions = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
    ];

    for fen in positions {
        let st = state_from_fen(fen);
        assert_eq!(st.hash, compute_hash(&st), "bad initial hash for {fen}");
        perft(&st, 4);
    }

    let st960 = state_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w AHah - 0 1");
    assert!(
        st960.chess960,
        "Shredder-FEN castling should auto-enable chess960"
    );
    assert_eq!(st960.hash, compute_hash(&st960));
    perft(&st960, 4);
}

#[test]
fn start_position_perft_smoke() {
    let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert_eq!(perft(&st, 1), 20);
    assert_eq!(perft(&st, 2), 400);
    assert_eq!(perft(&st, 3), 8902);
}

#[test]
fn temp_perft_startpos_deep() {
    let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert_eq!(perft(&st, 4), 197281);
    assert_eq!(perft(&st, 5), 4865609);
}

#[test]
fn temp_perft_kiwipete() {
    let st = state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&st, 1), 48);
    assert_eq!(perft(&st, 2), 2039);
    assert_eq!(perft(&st, 3), 97862);
    assert_eq!(perft(&st, 4), 4085603);
}

#[test]
fn temp_perft_position3() {
    let st = state_from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    assert_eq!(perft(&st, 1), 14);
    assert_eq!(perft(&st, 2), 191);
    assert_eq!(perft(&st, 3), 2812);
    assert_eq!(perft(&st, 4), 43238);
    assert_eq!(perft(&st, 5), 674624);
}

#[test]
fn temp_perft_position4() {
    let st = state_from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    assert_eq!(perft(&st, 1), 6);
    assert_eq!(perft(&st, 2), 264);
    assert_eq!(perft(&st, 3), 9467);
    assert_eq!(perft(&st, 4), 422333);
}

#[test]
fn temp_perft_position5() {
    let st = state_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    assert_eq!(perft(&st, 1), 44);
    assert_eq!(perft(&st, 2), 1486);
    assert_eq!(perft(&st, 3), 62379);
    assert_eq!(perft(&st, 4), 2103487);
}

#[test]
fn temp_perft_position6() {
    let st =
        state_from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    assert_eq!(perft(&st, 1), 46);
    assert_eq!(perft(&st, 2), 2079);
    assert_eq!(perft(&st, 3), 89890);
}

#[test]
fn rook_castling_perft_covers_castling_rights() {
    let st = state_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    assert_eq!(perft(&st, 1), 26);
    assert_eq!(perft(&st, 2), 568);
}

#[test]
fn en_passant_move_removes_the_captured_pawn() {
    let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let moves = generate_moves(&st, st.w, &st.cr, st.ep);
    let ep = moves
        .into_iter()
        .find(|mv| *mv == encode_move(3, 4, 2, 3, 0))
        .expect("expected e5d6 en passant to be legal");

    apply_move(
        &mut st,
        move_sr(ep),
        move_sc(ep),
        move_er(ep),
        move_ec(ep),
        move_promotion(ep),
    );

    assert_ne!(piece_on(&st.bb, sq(2, 3)), EMPTY_SQ);
    assert_eq!(piece_on(&st.bb, sq(3, 3)), EMPTY_SQ);
    assert!(!st.w);
}

#[test]
fn chess960_castling_places_pieces_on_standard_squares() {
    let mut engine = Engine::new();
    engine.set_fen("1k6/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    engine.st.chess960 = true;
    let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let uci_moves: Vec<String> = moves
        .iter()
        .map(|mv| move_to_uci(&engine.st, *mv))
        .collect();
    assert!(uci_moves.contains(&"e1a1".to_string()));
    assert!(uci_moves.contains(&"e1h1".to_string()));
    let oo = moves
        .iter()
        .find(|mv| move_to_uci(&engine.st, **mv) == "e1h1")
        .unwrap();
    apply_move(
        &mut engine.st,
        move_sr(*oo),
        move_sc(*oo),
        move_er(*oo),
        move_ec(*oo),
        move_promotion(*oo),
    );
    assert_eq!(engine.st.king_sq(true), 7 * 8 + 6);
    assert!(engine.st.bb[WR] & bit(7 * 8 + 5) != 0);
    assert!(engine.st.bb[WR] & bit(7 * 8 + 7) == 0);
}

#[test]
fn chess960_castling_queenside_places_pieces_on_standard_squares() {
    let mut engine = Engine::new();
    engine.set_fen("r3k2r/8/8/8/8/8/8/1K6 b KQkq - 0 1");
    engine.st.chess960 = true;
    let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let uci_moves: Vec<String> = moves
        .iter()
        .map(|mv| move_to_uci(&engine.st, *mv))
        .collect();
    assert!(uci_moves.contains(&"e8a8".to_string()));
    assert!(uci_moves.contains(&"e8h8".to_string()));
    let ooo = moves
        .iter()
        .find(|mv| move_to_uci(&engine.st, **mv) == "e8a8")
        .unwrap();
    apply_move(
        &mut engine.st,
        move_sr(*ooo),
        move_sc(*ooo),
        move_er(*ooo),
        move_ec(*ooo),
        move_promotion(*ooo),
    );
    assert_eq!(engine.st.king_sq(false), sq(0, 2));
    assert!(engine.st.bb[BR] & bit(sq(0, 3)) != 0);
    assert!(engine.st.bb[BR] & bit(sq(0, 0)) == 0);
}

#[test]
fn chess960_castling_blocked_by_pieces() {
    let mut engine = Engine::new();
    engine.set_fen("1k6/8/8/8/8/8/8/RBNKBNQR w KQkq - 0 1");
    engine.st.chess960 = true;
    let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let uci_moves: Vec<String> = moves
        .iter()
        .map(|mv| move_to_uci(&engine.st, *mv))
        .collect();
    assert!(!uci_moves.contains(&"e1a1".to_string()));
    assert!(!uci_moves.contains(&"e1h1".to_string()));
}

#[test]
fn chess960_castling_king_side_through_check() {
    let mut engine = Engine::new();
    engine.set_fen("1k6/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    engine.st.chess960 = true;
    engine.st.bb[BR] |= bit(sq(0, 5));
    engine.st.refresh_mailbox();
    let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let uci_moves: Vec<String> = moves
        .iter()
        .map(|mv| move_to_uci(&engine.st, *mv))
        .collect();
    assert!(!uci_moves.contains(&"e1h1".to_string()));
    assert!(uci_moves.contains(&"e1a1".to_string()));
}

#[test]
fn chess960_castling_queenside_through_check() {
    let mut engine = Engine::new();
    engine.set_fen("1k6/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    engine.st.chess960 = true;
    engine.st.bb[BR] |= bit(sq(0, 3));
    engine.st.refresh_mailbox();
    let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let uci_moves: Vec<String> = moves
        .iter()
        .map(|mv| move_to_uci(&engine.st, *mv))
        .collect();
    assert!(!uci_moves.contains(&"e1a1".to_string()));
    assert!(uci_moves.contains(&"e1h1".to_string()));
}
