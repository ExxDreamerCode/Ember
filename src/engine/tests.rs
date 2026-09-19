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

#[test]
fn legal_root_tt_move_is_promoted_between_searches() {
    let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let moves = root_moves(&engine);
    let preferred = root_move(&engine, "g1f3");
    engine
        .shared_tt
        .store(engine.st.hash, 8, 12, crate::tt::TT_EXACT, Some(preferred));

    let tt_move = tt_root_move(&engine.searcher, &engine.st, &moves);
    let ordered = sort_root_moves(&moves, tt_move);

    assert_eq!(tt_move, preferred);
    assert_eq!(ordered[0], preferred);
}

#[test]
fn quiet_root_tt_move_stays_ahead_of_an_unrelated_capture() {
    let engine = engine_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    let moves = root_moves(&engine);
    let preferred = root_move(&engine, "g1f3");
    let pawn_capture = root_move(&engine, "e4d5");
    let ordered = sort_root_moves(&moves, preferred);

    assert_eq!(ordered[0], preferred);
    assert_ne!(ordered[0], pawn_capture);
}

#[test]
fn root_ordering_preserves_quiet_opening_order() {
    let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let moves = root_moves(&engine);

    assert_eq!(sort_root_moves(&moves, NO_MOVE), moves);
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
    let moves = sort_root_moves(&root_moves(&engine), NO_MOVE);
    let rb3 = root_move(&engine, "c3b3");
    let chosen = engine.root_fifty_move_conversion_choice(&moves, rb3, 728);
    let chosen_uci = move_to_uci(&engine.st, chosen);

    assert!(
        ["h5h4", "d1f1", "h5g5", "d1d3", "d1e1", "d1d7"].contains(&chosen_uci.as_str()),
        "https://lichess.org/v8jiQh6Z: expected a 50-move preserving root, got {chosen_uci}"
    );
}
