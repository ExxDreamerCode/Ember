use chess_rs_lib::Engine;

struct CcrlCase {
    label: &'static str,
    history: &'static str,
    bad_move: &'static str,
}

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

fn ccrl_repetition_cases() -> [CcrlCase; 6] {
    [
        CcrlCase {
            label: "CCRL game 15, Seawall-Ember, move 42...Kf8",
            bad_move: "f7f8",
            history: "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 g1f3 d7d5 e1g1 f8d6 c2c4 c7c6 b2b3 d8e7 f3e5 e8g8 b1d2 b7b6 c1b2 c8b7 e2e3 b8d7 d1e2 a7a5 f1c1 f6e4 d2f3 a5a4 e5d7 e7d7 c4d5 e6d5 b3a4 a8a4 f3e5 d6e5 d4e5 f8a8 e2c2 h7h6 f2f3 a4c4 c2d1 c4c1 d1c1 e4c5 b2d4 c5a4 a1b1 c6c5 d4a1 b7c8 c1c2 d7e8 f3f4 c8e6 b1d1 e8f7 c2b3 c5c4 b3b5 f7e8 b5e8 a8e8 g3g4 g7g6 g4f5 g6f5 a1d4 e8c8 d1b1 c8a8 g2f3 a8a6 b1b5 g8f7 g1f1 a4c5 f3h5 f7g8 h5f3 g8f7 f3h5",
        },
        CcrlCase {
            label: "CCRL game 24, PawnStar-Ember, move 33...a5",
            bad_move: "a6a5",
            history: "d2d4 d7d5 c2c4 d5c4 d1a4 b8d7 g1f3 a7a6 b1c3 a8b8 a4c4 b7b5 c4d3 c8b7 e2e4 e7e6 f1e2 g8f6 a2a3 c7c5 c1f4 b8c8 e1g1 c5d4 d3d4 f8e7 f1d1 e7c5 d4d3 d8b6 f4g3 c5e7 b2b4 e8g8 f3d4 h7h5 h2h3 f8d8 e4e5 d7f8 d3e3 c8c3 e3c3 f6e4 c3b2 e4g3 f2g3 f8d7 e2h5 g7g6 h5f3 b7f3 g2f3 d7e5 g1g2 e7f6 b2f2 e5g4 h3g4 f6d4 f2d2 d4f6 d2f2 f6d4 f2d2",
        },
        CcrlCase {
            label: "CCRL game 34, Revolver-Ember, move 111...Qh4+",
            bad_move: "e4h4",
            history: "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 c2c4 f8e7 g1f3 e8g8 e1g1 d7d6 b2b3 f6e4 c1b2 e7f6 b1d2 b7b6 e2e3 c8b7 f3e1 d8e7 d1c2 d6d5 e1d3 b8a6 c4d5 e6d5 d2f3 a8c8 a1c1 c7c5 d4c5 a6c5 b2f6 e7f6 d3c5 b6c5 f1d1 h7h6 f3d2 d5d4 d1e1 c8d8 e3d4 c5d4 c2d3 f8f7 c1c2 e4d2 g2b7 f7b7 c2d2 b7c7 d3e2 c7c3 e1d1 c3c7 d2d3 g8h8 h2h4 c7c5 e2d2 c5d5 d2f4 h8g8 g1g2 a7a5 d1e1 d5c5 e1d1 c5d5 h4h5 f6c6 f4f3 c6d7 d1e1 a5a4 b3a4 g8h8 a2a3 d7a4 e1e7 a4a5 e7b7 a5a8 b7f7 a8c8 a3a4 c8a8 f7c7 a8a6 f3d1 a6b6 c7c4 b6a7 c4b4 h8h7 d1f3 a7a5 b4c4 a5a6 c4b4 h7h8 f3e2 a6a5 e2b2 a5a8 g2h2 a8a7 b2d2 a7c5 b4b5 c5c4 b5d5 d8d5 d2d1 c4a2 d3d2 a2c4 d2c2 c4a6 d1e1 a6a8 e1e6 d5d8 c2c7 d8g8 c7f7 a8a4 e6d5 a4e8 f7f5 e8c8 f5f3 g8d8 d5e4 c8c5 f3f5 c5d6 e4d3 d6c7 d3f3 c7b8 f5d5 b8a8 d5d8 a8d8 f3d3 h8g8 d3b3 g8f8 b3f3 f8e7 f3e2 e7d6 e2g4 d8f6 h2g1 d6c5 g4e4 f6f7 e4e1 f7b7 e1c1 c5d5 c1c2 b7b4 c2a2 d5c5 a2a6 b4d2 a6a7 c5b4 a7b6 b4a3 b6a7 a3b2 a7g7 d2c3 g7h6 d4d3 h6b6 b2c2 b6g6 c2c1 h5h6 d3d2 g6g5 c1b2 g5b5 b2a3 b5a6 a3b3 a6b5 b3c2 b5f5 c3d3 f5c5 c2b2 c5b4 b2c1 b4f4 c1c2 f4a4 d3b3 a4e4 c2b2 e4e5 b2b1 e5f5 b3c2 f5b5 b1c1 b5g5 c2e4 g1h2 e4d4 h2g2 d4e4 g2h2",
        },
        CcrlCase {
            label: "CCRL game 38, PawnStar-Ember, move 37...a5",
            bad_move: "a6a5",
            history: "d2d4 d7d5 c2c4 d5c4 d1a4 b8d7 g1f3 a7a6 b1c3 a8b8 a4c4 b7b5 c4d3 c8b7 e2e4 e7e6 f1e2 g8f6 a2a3 c7c5 c1f4 c5d4 d3d4 b8c8 e1g1 d7c5 c3b5 d8d4 b5d4 c5e4 d4b3 c8a8 a1c1 b7d5 b3a5 f8d6 f4d6 e4d6 f3e5 e8g8 e5c6 g7g6 f1d1 g8g7 g1f1 h7h5 c6e7 d6f5 e7d5 f6d5 b2b4 h5h4 e2f3 a8c8 a5c6 h4h3 g2g3 f8e8 f3e2 c8a8 e2f3 a8c8 c1c5 f5d6 d1c1 d6b7 c5c2 b7d6 f3g4 e8h8 g4f3 h8e8 f3g4",
        },
        CcrlCase {
            label: "CCRL game 46, Ember-Puffin, move 55.Rb6+",
            bad_move: "a6b6",
            history: "d2d4 d7d5 c2c4 e7e6 g1f3 d5c4 e2e3 g8f6 f1c4 c7c5 e1g1 a7a6 d4c5 f8c5 d1e2 b8c6 a2a3 c5d6 f1d1 b7b5 c4a2 c8b7 b2b4 c6e7 b1d2 e8g8 c1b2 a8c8 d2b3 f6e4 a1c1 b7d5 g2g3 c8c1 b3c1 d8c7 a2d5 e7d5 e2d3 c7c4 d3c4 b5c4 c1a2 d6e7 b2d4 f8c8 f3d2 e4d2 d1d2 f7f5 a2c3 d5c7 d4e5 c7b5 a3a4 b5c3 e5c3 c8b8 d2d4 e7b4 d4c4 b4c3 c4c3 b8b4 a4a5 b4b5 c3c6 b5a5 c6e6 a5a2 e6e5 g7g6 e5e7 a6a5 g1g2 a5a4 e7b7 a4a3 b7a7 h7h6 h2h3 g8f8 a7a6 f8g7 g3g4 f5g4 h3g4 a2a1 a6a7 g7f6 e3e4 f6e5 g2f3 e5d4 a7a4 d4c5 e4e5 h6h5 a4a6 h5h4 f3g2 c5b5 a6a7 b5b6 a7a8 b6c6 a8a6 c6b5",
        },
        CcrlCase {
            label: "CCRL game 60, Ember-KnightX, move 71.Rg8+",
            bad_move: "g7g8",
            history: "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4d5 e6d5 f1d3 c7c6 g1f3 g8e7 e1g1 e8g8 f1e1 e7g6 h2h3 b8d7 c1g5 d7f6 d1d2 b4d6 f3e5 d8b6 g5f6 g7f6 e5f3 d6f4 d2d1 b6b2 c3e2 f4d6 a2a4 b2b6 e2g3 d6g3 f2g3 b6c7 g1h2 h7h5 f3d2 h5h4 d2f1 g8g7 d1f3 c8d7 h2g1 a8e8 g3h4 c7f4 f3f4 g6f4 f1g3 f4d3 c2d3 e8e1 a1e1 f8h8 e1e7 h8d8 g3h5 g7f8 e7e1 f6f5 e1a1 f8e7 h5f4 a7a5 g1f2 d8h8 g2g3 e7f8 a1a3 f8g7 a3c3 g7h6 c3c5 h8a8 f2f3 b7b6 c5c1 a8g8 c1e1 g8e8 e1a1 h6h7 g3g4 f5g4 h3g4 f7f6 f4g2 e8a8 g2e3 b6b5 a4b5 c6b5 f3f4 d7e6 a1c1 a5a4 c1c6 e6g8 c6f6 a4a3 e3c2 a3a2 c2a1 h7g7 g4g5 g8h7 h4h5 h7d3 h5h6 g7h7 f4g4 a8a7 f6e6 b5b4 g4h5 a7a6 e6e7 h7g8 e7g7 g8h8 g5g6 a6b6 h5g5 b6c6 a1b3 c6c1 b3c1 a2a1q c1d3 a1g1 g5f5 g1b1 g7h7 h8g8 h7g7 g8h8",
        },
    ]
}

#[test]
fn ccrl_repetition_histories_do_not_choose_observed_losing_moves() {
    for case in ccrl_repetition_cases() {
        let mut engine = Engine::new();
        engine.book = None;
        engine.num_threads = 1;
        replay_history(&mut engine, case.history);

        let (best_move, score, nodes, _) = engine.find_best_move(1_000_000.0, 6);

        assert_ne!(
            best_move, case.bad_move,
            "{} still chooses the CCRL losing move at depth 6; score={score}, nodes={nodes}",
            case.label
        );
    }
}
