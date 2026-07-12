use chess_rs_lib::{evaluate, Engine};

fn parse_uci_move(mv: &str) -> (usize, usize, usize, usize, u8) {
    let bytes = mv.as_bytes();
    assert!(bytes.len() >= 4, "invalid UCI move: {mv}");
    let sc = (bytes[0] - b'a') as usize;
    let sr = 8 - (bytes[1] - b'0') as usize;
    let ec = (bytes[2] - b'a') as usize;
    let er = 8 - (bytes[3] - b'0') as usize;
    assert!(
        sr < 8 && sc < 8 && er < 8 && ec < 8,
        "invalid UCI move: {mv}"
    );
    let promotion = match bytes.get(4).copied() {
        Some(b'q' | b'Q') => b'Q',
        Some(b'r' | b'R') => b'R',
        Some(b'b' | b'B') => b'B',
        Some(b'n' | b'N') => b'N',
        Some(other) => panic!("invalid promotion piece in {mv}: {}", other as char),
        None => 0,
    };
    (sr, sc, er, ec, promotion)
}

fn replay_history(engine: &mut Engine, history: &str) {
    for mv in history.split_whitespace() {
        let (sr, sc, er, ec, promotion) = parse_uci_move(mv);
        assert!(
            engine.make_move_uci(sr, sc, er, ec, promotion),
            "illegal move in history: {mv}"
        );
    }
}

#[test]
fn ccrl_game_152_prefers_bishop_invasion_over_retreat() {
    evaluate::init_embedded_nnue().expect("embedded NNUE should load");

    let mut engine = Engine::new();
    engine.book = None;
    engine.num_threads = 1;
    replay_history(
        &mut engine,
        "d2d4 g8f6 c2c4 e7e6 b1c3 d7d5 c4d5 e6d5 c1g5 c7c6 \
         e2e3 c8f5 d1f3 f5g6 g5f6 d8f6 f3f6 g7f6 g1f3 b8d7 \
         f3h4 a7a5 f1e2 b7b5 f2f4 h7h5 f4f5 g6h7 e1g1 f8h6 \
         a1e1",
    );

    let depth_limit = if cfg!(debug_assertions) { 8 } else { 16 };
    let (best_move, score, nodes, _) = engine.find_best_move(1_000_000.0, depth_limit);

    if cfg!(debug_assertions) {
        assert_ne!(
            best_move, "h7g8",
            "CCRL game 152 still chooses the losing 16...Bg8 retreat at \
             depth {depth_limit}; score={score}, nodes={nodes}"
        );
    } else {
        assert_eq!(
            best_move, "h6e3",
            "CCRL game 152 should choose 16...Bxe3 instead of the losing \
             16...Bg8 retreat; got {best_move}, score={score}, nodes={nodes}"
        );
    }
}
