use super::*;

fn engine_from_fen(fen: &str) -> Engine {
    let mut engine = Engine::new();
    engine.book = None;
    engine.set_fen(fen);
    engine
}

fn root_moves(engine: &Engine) -> Vec<Move> {
    generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
}

fn root_move(engine: &Engine, uci: &str) -> Move {
    root_moves(engine)
        .into_iter()
        .find(|mv| move_to_uci(&engine.st, *mv) == uci)
        .unwrap_or_else(|| panic!("expected legal root move {uci}"))
}

fn play_uci(engine: &mut Engine, uci: &str) {
    let bytes = uci.as_bytes();
    assert!(bytes.len() >= 4, "invalid UCI move: {uci}");
    let promotion = bytes
        .get(4)
        .map_or(0, |piece| match piece.to_ascii_lowercase() {
            b'q' => b'Q',
            b'r' => b'R',
            b'b' => b'B',
            b'n' => b'N',
            _ => 0,
        });
    assert!(
        engine.make_move_uci(
            8 - usize::from(bytes[1] - b'0'),
            usize::from(bytes[0] - b'a'),
            8 - usize::from(bytes[3] - b'0'),
            usize::from(bytes[2] - b'a'),
            promotion,
        ),
        "expected legal move {uci}"
    );
}

#[test]
fn engine_defers_hash_materialization_until_ready() {
    let mut engine = Engine::new();
    assert_eq!(engine.searcher.tt_mb, DEFAULT_HASH_MB);
    assert_eq!(engine.shared_tt.allocated_entries(), 1);

    engine.searcher.tt_mb = 1;
    engine.ensure_hash_ready();
    let ready_entries = engine.shared_tt.allocated_entries();
    assert!(ready_entries > 1);

    engine.ensure_hash_ready();
    assert_eq!(engine.shared_tt.allocated_entries(), ready_entries);
}

// These position-backed tests observe the internal root-order predicates, scores,
// extensions, and counterexamples. A TSV best-move assertion cannot distinguish
// those contracts from an unrelated search path that happens to choose the same move.
#[test]
fn root_ordering_prioritizes_the_missed_rook_clearance() {
    let engine = engine_from_fen("8/5k2/2pp2p1/5pP1/P2P4/3n4/2r5/1KB4R b - - 4 46");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);

    assert_eq!(move_to_uci(&engine.st, ordered[0]), "c2c1");
}

#[test]
fn reduced_rook_check_capture_gets_the_tactical_root_extension() {
    let engine = engine_from_fen("8/5k2/2pp2p1/5pP1/P2P4/3n4/2r5/1KB4R b - - 4 46");
    let clearance = root_move(&engine, "c2c1");
    let non_capture = root_move(&engine, "c2c4");

    assert!(root_reduced_rook_check_capture(&engine.st, clearance));
    assert_eq!(root_depth_extension(&engine.st, clearance), 3);
    assert!(!root_reduced_rook_check_capture(&engine.st, non_capture));
    assert_eq!(root_depth_extension(&engine.st, non_capture), 0);
}

#[test]
fn root_rook_invasion_extension_rejects_captures() {
    let mut rook_capture = engine_from_fen("2r3k1/p7/7p/1p4p1/6R1/3B1q1P/2P4P/2B3RK w - - 6 35");
    play_uci(&mut rook_capture, "g1g2");
    let rook_takes_pawn = root_move(&rook_capture, "c8c2");

    assert_eq!(
        root_rook_invasion_score(&rook_capture.st, rook_takes_pawn),
        None
    );
    assert_eq!(root_depth_extension(&rook_capture.st, rook_takes_pawn), 0);
}

#[test]
fn root_ordering_prioritizes_a_missed_mating_check() {
    let mut engine = engine_from_fen("1rb2rk1/q5P1/4p2p/3p3p/3P1P2/2P5/2QK3P/3R2R1 b - - 0 29");
    play_uci(&mut engine, "f8f7");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let mating_check = root_move(&engine, "c2h7");
    let quiet_move = root_move(&engine, "c2g6");

    assert!(root_move_gives_check(&engine.st, mating_check));
    assert!(!root_move_gives_check(&engine.st, quiet_move));
    assert_eq!(
        root_forced_mate_reply_count(&engine.st, mating_check),
        Some(1)
    );
    assert_eq!(
        root_mating_check_order_score(&engine.st, mating_check),
        Some(7_900_000)
    );
    assert_eq!(root_depth_extension(&engine.st, mating_check), 0);
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "c2h7");
}

#[test]
fn root_ordering_prioritizes_checking_non_pawn_capture() {
    let mut engine = engine_from_fen("r4k1r/1pp2p2/p2p3p/3N4/3P2q1/8/PPP5/1K2Q1NR b - - 1 23");
    play_uci(&mut engine, "a8e8");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let checking_capture = root_move(&engine, "e1e8");

    assert!(root_move_gives_check(&engine.st, checking_capture));
    assert!(root_move_is_capture(&engine.st, checking_capture));
    assert_eq!(
        root_checking_non_pawn_capture_order_score(&engine.st, checking_capture),
        Some(6_004_050)
    );
    assert_eq!(root_depth_extension(&engine.st, checking_capture), 0);
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "e1e8");
}

#[test]
fn root_ordering_ignores_noisy_checking_captures() {
    let mut rook_trade = engine_from_fen("4R1k1/p4r1p/1pp2rp1/8/5B1q/4QP1P/P1P2PK1/8 b - - 2 28");
    play_uci(&mut rook_trade, "f7f8");
    let equal_rook_check = root_move(&rook_trade, "e8f8");

    assert!(root_move_gives_check(&rook_trade.st, equal_rook_check));
    assert!(root_move_is_capture(&rook_trade.st, equal_rook_check));
    assert_eq!(
        root_checking_non_pawn_capture_order_score(&rook_trade.st, equal_rook_check),
        None
    );

    let mut queen_harvest =
        engine_from_fen("r2q1rk1/pbpn1p2/1p1bpn1Q/8/8/1B1P1NN1/PPP2PPP/R3K2R b KQ - 0 12");
    play_uci(&mut queen_harvest, "f6h7");
    let queen_takes_knight = root_move(&queen_harvest, "h6h7");
    let queen_takes_rook = root_move(&queen_harvest, "h6f8");

    assert!(root_move_gives_check(&queen_harvest.st, queen_takes_knight));
    assert!(root_move_gives_check(&queen_harvest.st, queen_takes_rook));
    assert_eq!(
        root_checking_non_pawn_capture_order_score(&queen_harvest.st, queen_takes_knight),
        None
    );
    assert_eq!(
        root_checking_non_pawn_capture_order_score(&queen_harvest.st, queen_takes_rook),
        None
    );
}

#[test]
fn root_ordering_prioritizes_quiet_bishop_knight_capture() {
    let mut engine =
        engine_from_fen("r2qkb1r/pp1nppp1/2p2n1p/3p1b2/3P4/BP2PN2/P1P2PPP/RN1QKB1R w KQkq - 2 7");
    play_uci(&mut engine, "c2c4");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let bishop_takes_knight = root_move(&engine, "f5b1");

    assert!(root_move_is_capture(&engine.st, bishop_takes_knight));
    assert!(!root_move_gives_check(&engine.st, bishop_takes_knight));
    assert_eq!(
        root_quiet_bishop_knight_capture_order_score(&engine.st, bishop_takes_knight),
        Some(5_100_000)
    );
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "f5b1");
}

#[test]
fn root_ordering_prioritizes_checking_slider_pawn_capture() {
    let mut engine =
        engine_from_fen("rn1qk2r/pp3ppp/3bp1b1/3p4/3Pn2N/3BB3/PPP2PPP/RN1Q1RK1 w kq - 4 10");
    play_uci(&mut engine, "h4g6");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let bishop_takes_pawn = root_move(&engine, "d6h2");

    assert!(root_move_is_capture(&engine.st, bishop_takes_pawn));
    assert!(root_move_gives_check(&engine.st, bishop_takes_pawn));
    assert_eq!(
        root_checking_slider_pawn_capture_order_score(&engine.st, bishop_takes_pawn),
        Some(5_500_660)
    );
    assert_eq!(root_depth_extension(&engine.st, bishop_takes_pawn), 0);
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "d6h2");

    let mut rook_engine =
        engine_from_fen("r5k1/2p1pp2/pp4p1/1q1r4/5P2/2QP2R1/PP6/1K4R1 b - - 0 32");
    play_uci(&mut rook_engine, "d5h5");
    let moves = root_moves(&rook_engine);
    let ordered = sort_root_moves(&rook_engine.st, &moves, NO_MOVE);
    let rook_takes_pawn = root_move(&rook_engine, "g3g6");

    assert!(root_move_is_capture(&rook_engine.st, rook_takes_pawn));
    assert!(root_move_gives_check(&rook_engine.st, rook_takes_pawn));
    assert_eq!(
        root_checking_slider_pawn_capture_order_score(&rook_engine.st, rook_takes_pawn),
        Some(5_500_500)
    );
    assert_eq!(root_depth_extension(&rook_engine.st, rook_takes_pawn), 1);
    assert_eq!(move_to_uci(&rook_engine.st, ordered[0]), "g3g6");
}

#[test]
fn root_ordering_prioritizes_constrained_quiet_queen_check() {
    let mut engine = engine_from_fen("8/3k4/p6p/1p2Q1pP/3P2b1/1PP2qP1/P3p3/1K2R3 w - - 3 44");
    play_uci(&mut engine, "d4d5");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let queen_check = root_move(&engine, "f3d3");
    let wider_queen_check = root_move(&engine, "f3e4");

    assert_eq!(
        root_quiet_queen_check_reply_count(&engine.st, queen_check),
        Some(3)
    );
    assert_eq!(
        root_quiet_queen_check_reply_count(&engine.st, wider_queen_check),
        Some(4)
    );
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "f3d3");
}

#[test]
fn root_ordering_prioritizes_constrained_queen_pawn_check_capture() {
    let mut engine = engine_from_fen("5rk1/R5p1/5q1p/8/3p4/1P4Q1/P3rPPP/5RK1 w - - 2 38");
    play_uci(&mut engine, "g3g4");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let queen_takes_pawn = root_move(&engine, "f6f2");

    assert!(root_move_is_capture(&engine.st, queen_takes_pawn));
    assert!(root_move_gives_check(&engine.st, queen_takes_pawn));
    assert_eq!(
        root_queen_pawn_check_capture_order_score(&engine.st, queen_takes_pawn),
        Some(5_600_050)
    );
    assert_eq!(move_to_uci(&engine.st, ordered[0]), "f6f2");
}

#[test]
fn root_ordering_prioritizes_the_forced_queen_recapture() {
    let engine = engine_from_fen("1r4k1/2p2p2/2np1bp1/pp6/2Q3P1/2P2N2/PPP2P2/1KBR4 b - - 0 22");
    let moves = root_moves(&engine);
    let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);

    assert_eq!(move_to_uci(&engine.st, ordered[0]), "b5c4");
}

#[test]
fn legal_root_tt_move_is_promoted_between_searches() {
    let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let moves = root_moves(&engine);
    let preferred = root_move(&engine, "g1f3");
    engine
        .shared_tt
        .store(engine.st.hash, 8, 12, crate::tt::TT_EXACT, Some(preferred));

    let tt_move = tt_root_move(&engine.searcher, &engine.st, &moves);
    let ordered = sort_root_moves(&engine.st, &moves, tt_move);

    assert_eq!(tt_move, preferred);
    assert_eq!(ordered[0], preferred);
}

#[test]
fn quiet_root_tt_move_stays_ahead_of_an_unrelated_capture() {
    let engine = engine_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    let moves = root_moves(&engine);
    let preferred = root_move(&engine, "g1f3");
    let pawn_capture = root_move(&engine, "e4d5");
    let ordered = sort_root_moves(&engine.st, &moves, preferred);

    assert_eq!(ordered[0], preferred);
    assert_ne!(ordered[0], pawn_capture);
}

#[test]
fn root_ordering_prioritizes_reported_mating_check() {
    let engine = engine_from_fen("8/5k2/3Q4/7p/8/1p6/3p1P1P/3B2K1 w - - 52 78");
    let moves = root_moves(&engine);
    let sorted = sort_root_moves(&engine.st, &moves, NO_MOVE);
    let mating_check = *moves
        .iter()
        .find(|mv| move_to_uci(&engine.st, **mv) == "d1h5")
        .expect("reported mating check is legal");
    let quiet_start = sorted
        .iter()
        .position(|mv| root_forcing_score(&engine.st, *mv).is_none())
        .unwrap_or(sorted.len());
    let mating_check_pos = sorted
        .iter()
        .position(|mv| *mv == mating_check)
        .expect("reported mating check remains in root moves");

    assert!(root_forcing_score(&engine.st, mating_check).unwrap() >= 4_000_000);
    assert!(mating_check_pos < quiet_start);
}

#[test]
fn root_ordering_preserves_quiet_opening_order() {
    let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let moves = root_moves(&engine);

    assert_eq!(sort_root_moves(&engine.st, &moves, NO_MOVE), moves);
}

#[test]
fn root_ordering_handles_promotion_race_without_major_piece() {
    let engine = engine_from_fen("8/P4k2/8/8/8/8/8/6K1 w - - 0 1");
    let moves = root_moves(&engine);
    let sorted = sort_root_moves(&engine.st, &moves, NO_MOVE);

    assert!(sorted
        .first()
        .is_some_and(|mv| move_to_uci(&engine.st, *mv).starts_with("a7a8")));
}

#[test]
fn fifty_move_verifier_does_not_trust_blessed_loss_rook_moves() {
    let engine = engine_from_fen("R7/8/8/7k/4K3/2r2P2/8/3r4 b - - 86 166");
    let rb3 = root_move(&engine, "c3b3");
    let kh6 = root_move(&engine, "h5g6");
    let rd8 = root_move(&engine, "d1d8");
    let kh4 = root_move(&engine, "h5h4");
    let rf1 = root_move(&engine, "d1f1");

    assert_ne!(
        root_move_preserves_fifty_move_conversion(&engine.st, rb3),
        Some(true),
        "https://lichess.org/v8jiQh6Z: 166...Rb3 must not be trusted as conversion-safe"
    );
    assert_ne!(
        root_move_preserves_fifty_move_conversion(&engine.st, kh6),
        Some(true),
        "https://lichess.org/v8jiQh6Z: 166...Kh6 must not be trusted as conversion-safe"
    );
    assert_ne!(
        root_move_preserves_fifty_move_conversion(&engine.st, rd8),
        Some(true),
        "https://lichess.org/v8jiQh6Z: exact tablebase reports 166...Rd8 as drawn"
    );
    assert_eq!(
        root_move_preserves_fifty_move_conversion(&engine.st, kh4),
        Some(true)
    );
    assert_eq!(
        root_move_preserves_fifty_move_conversion(&engine.st, rf1),
        Some(true)
    );
}

#[test]
fn fifty_move_root_choice_replaces_blessed_loss_bestmove() {
    let engine = engine_from_fen("R7/8/8/7k/4K3/2r2P2/8/3r4 b - - 86 166");
    let moves = sort_root_moves(&engine.st, &root_moves(&engine), NO_MOVE);
    let rb3 = root_move(&engine, "c3b3");
    let chosen = engine.root_fifty_move_conversion_choice(&moves, rb3, 728);
    let chosen_uci = move_to_uci(&engine.st, chosen);

    assert!(
        ["h5h4", "d1f1", "h5g5", "d1d3", "d1e1", "d1d7"].contains(&chosen_uci.as_str()),
        "https://lichess.org/v8jiQh6Z: expected a 50-move preserving root, got {chosen_uci}"
    );
}

#[test]
fn sparse_endgame_root_ordering_is_used_by_search() {
    // This is an integration check for the forcing-move class and both the serial and
    // Lazy SMP paths, rather than an exact single-thread best-move regression.
    for threads in [1usize, 2] {
        let mut engine = engine_from_fen("8/5k2/3Q4/7p/8/1p6/3p1P1P/3B2K1 w - - 52 78");
        engine.num_threads = threads;
        let (best_move, _, _, _) = engine.find_best_move(2.0, 1);
        let best = root_moves(&engine)
            .into_iter()
            .find(|mv| move_to_uci(&engine.st, *mv) == best_move)
            .expect("search best move remains legal");

        assert!(
            root_forcing_score(&engine.st, best).is_some(),
            "threads={threads} should pick a forcing sparse-endgame root move, got {best_move}"
        );
    }
}
