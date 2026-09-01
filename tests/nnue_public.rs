use ember_chess::backend::{
    aarch64_simd_available, available_nnue_backends, nnue_backend_available, x86_v3_available,
    NnueBackendKind,
};
use ember_chess::nnue::{threat_feature_count, NNUEAccumulator, NNUENet, NNUEThreatAccumulator};
use ember_chess::types::{BLACK, WHITE};
use ember_chess::Engine;

const DENSE_NET: &[u8] = include_bytes!("../src/net.nnue");
const COMPACT_NET: &[u8] = include_bytes!("../src/net.compact.nnue");

fn parse_uci_move(mv: &str) -> (usize, usize, usize, usize, u8) {
    let bytes = mv.as_bytes();
    assert!(matches!(bytes.len(), 4 | 5), "invalid UCI move: {mv}");
    let sc = (bytes[0] - b'a') as usize;
    let sr = 8 - (bytes[1] - b'0') as usize;
    let ec = (bytes[2] - b'a') as usize;
    let er = 8 - (bytes[3] - b'0') as usize;
    let promotion = bytes.get(4).copied().unwrap_or(0).to_ascii_uppercase();
    (sr, sc, er, ec, promotion)
}

fn assert_incremental_line_matches_refresh(net: &NNUENet, fen: &str, moves: &[&str]) {
    for backend in available_nnue_backends() {
        let mut engine = Engine::new();
        engine.try_set_fen(fen).expect("critical FEN should parse");

        let mut incremental = NNUEAccumulator::new(net.hidden_size);
        incremental.refresh_with_kind(backend, net, &engine.st);

        for &uci in moves {
            let (sr, sc, er, ec, promotion) = parse_uci_move(uci);
            let before = engine.st;
            let updated =
                incremental.update_move_with_kind(backend, net, &before, sr, sc, er, ec, promotion);
            assert!(
                engine.make_move_uci(sr, sc, er, ec, promotion),
                "{uci} should be legal for {backend:?}"
            );
            if !updated {
                incremental.refresh_with_kind(backend, net, &engine.st);
            }

            let mut refreshed = NNUEAccumulator::new(net.hidden_size);
            refreshed.refresh_with_kind(backend, net, &engine.st);
            assert_eq!(
                incremental.white(),
                refreshed.white(),
                "white accumulator drift after {uci} with {backend:?}"
            );
            assert_eq!(
                incremental.black(),
                refreshed.black(),
                "black accumulator drift after {uci} with {backend:?}"
            );
            assert_eq!(
                (incremental.wk, incremental.bk),
                (refreshed.wk, refreshed.bk),
                "king-square drift after {uci} with {backend:?}"
            );

            let stm = if engine.st.w { WHITE } else { BLACK };
            let piece_count: u32 = engine.st.bb.iter().map(|bb| bb.count_ones()).sum();
            let incremental_score = net.forward_with_kind(backend, &incremental, stm, piece_count);
            let refreshed_score = net.forward_with_kind(backend, &refreshed, stm, piece_count);
            let scalar_score =
                net.forward_with_kind(NnueBackendKind::Scalar, &refreshed, stm, piece_count);
            assert_eq!(
                incremental_score, refreshed_score,
                "NNUE score drift after {uci} with {backend:?}"
            );
            assert_eq!(
                refreshed_score, scalar_score,
                "NNUE backend score differs from scalar after {uci} with {backend:?}"
            );
        }
    }
}

fn nnue_score(net: &NNUENet, fen: &str) -> i32 {
    let mut engine = Engine::new();
    engine.set_fen(fen);

    let mut acc = NNUEAccumulator::new(net.hidden_size);
    acc.refresh(net, &engine.st);
    let stm = if engine.st.w { WHITE } else { BLACK };
    let piece_count = (0..12).map(|i| engine.st.bb[i].count_ones()).sum();
    let score = net.forward(&acc, stm, piece_count);
    if stm == WHITE {
        score
    } else {
        -score
    }
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i16(bytes: &mut Vec<u8>, value: i16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn synthetic_threat_net(threat_weight: impl Fn(usize) -> i8) -> NNUENet {
    let hidden = 16usize;
    let threat_features = threat_feature_count();
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 0x4e4e5545);
    push_u32(&mut bytes, 10);
    bytes.push(0xc6);
    push_u16(&mut bytes, hidden as u16);
    push_u16(&mut bytes, 1);
    push_u16(&mut bytes, 0);
    push_u32(&mut bytes, threat_features as u32);
    bytes.push(16);
    bytes.push(0);
    bytes.push(0);

    for _ in 0..16 * 768 * hidden {
        push_i16(&mut bytes, 0);
    }
    for _ in 0..hidden {
        push_i16(&mut bytes, 0);
    }
    for index in 0..threat_features * hidden {
        bytes.push(threat_weight(index) as u8);
    }
    for _ in 0..hidden {
        push_i16(&mut bytes, 1);
    }
    push_i16(&mut bytes, 0);
    for _ in 0..8 {
        push_i16(&mut bytes, 1);
    }
    for _ in 0..8 {
        push_i32(&mut bytes, 0);
    }

    NNUENet::load_from_bytes(&bytes, "<synthetic threat net>")
        .expect("synthetic threat net should load")
}

#[test]
fn dense_loader_parses_v10_threat_header_before_rejecting_it() {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, 0x4e4e5545);
    push_u32(&mut bytes, 10);
    bytes.push(0xe7);
    push_u16(&mut bytes, 1024);
    push_u16(&mut bytes, 32);
    push_u16(&mut bytes, 32);
    push_u32(&mut bytes, 66_864);
    bytes.push(48);
    bytes.push(1);
    bytes.push(1);

    let error = match NNUENet::load_from_bytes(&bytes, "<v10 threat header>") {
        Ok(_) => panic!("v10 threat net should be rejected"),
        Err(error) => error,
    };
    assert!(
        error.contains("unsupported NNUE threat features"),
        "expected threat-feature rejection, got {error}"
    );
}

#[test]
fn threat_accumulator_incremental_updates_match_refresh() {
    let net = synthetic_threat_net(|index| (index % 5) as i8 - 2);
    let mut engine = Engine::new();

    let mut incremental = NNUEThreatAccumulator::new(net.hidden_size);
    incremental.refresh(&net, &engine.st);

    for uci in [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "d2d4", "e5d4",
    ] {
        let (sr, sc, er, ec, promotion) = parse_uci_move(uci);
        let before = engine.st;
        assert!(
            engine.make_move_uci(sr, sc, er, ec, promotion),
            "{uci} should be legal"
        );
        let parent = incremental.clone();
        if !incremental.update_from_parent(&parent, &net, &before, &engine.st) {
            incremental.refresh(&net, &engine.st);
        }

        let mut refreshed = NNUEThreatAccumulator::new(net.hidden_size);
        refreshed.refresh(&net, &engine.st);
        assert_eq!(
            incremental.white(),
            refreshed.white(),
            "white threat accumulator drift after {uci}"
        );
        assert_eq!(
            incremental.black(),
            refreshed.black(),
            "black threat accumulator drift after {uci}"
        );
    }
}

#[test]
fn threat_accumulator_uses_board_pawn_attack_direction() {
    let net = synthetic_threat_net(|_| 1);

    for fen in [
        "4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1",
        "4k3/8/8/4p3/3N4/8/8/4K3 b - - 0 1",
    ] {
        let mut engine = Engine::new();
        engine
            .try_set_fen(fen)
            .expect("pawn-threat FEN should parse");

        let mut threats = NNUEThreatAccumulator::new(net.hidden_size);
        threats.refresh(&net, &engine.st);
        assert!(
            threats.white().iter().any(|&value| value != 0),
            "white-perspective threats should include pawn attack in {fen}"
        );
        assert!(
            threats.black().iter().any(|&value| value != 0),
            "black-perspective threats should include pawn attack in {fen}"
        );
    }
}

#[test]
fn compact_embedded_nnue_matches_dense_scores() {
    let dense =
        NNUENet::load_from_bytes(DENSE_NET, "<dense test>").expect("dense NNUE should load");
    let compact = NNUENet::load_from_bytes(COMPACT_NET, "<compact test>")
        .expect("general NNUE loader should detect the compact format");

    assert!(
        COMPACT_NET.len() + 3_000_000 < DENSE_NET.len(),
        "compact embedded NNUE should remove the zero feature rows"
    );
    assert_eq!(dense.input_row_map, compact.input_row_map);
    assert_eq!(dense.input_weights, compact.input_weights);
    assert_eq!(
        compact
            .input_row_map
            .iter()
            .filter(|&&row| row == u16::MAX)
            .count(),
        1712
    );
    assert_eq!(compact.input_weights.len() / compact.hidden_size, 10576);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
        "8/8/8/R2pP1k/8/8/6Q1/4K3 w - d6 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 b - - 0 1",
        "8/P4k2/8/8/8/8/8/6K1 w - - 0 1",
    ] {
        assert_eq!(
            nnue_score(&dense, fen),
            nnue_score(&compact, fen),
            "compact NNUE score mismatch for {fen}"
        );
    }
}

#[test]
fn available_nnue_backends_agree_with_nnue_backend_available() {
    let backends = available_nnue_backends();
    assert!(backends.contains(&NnueBackendKind::Scalar));
    for &backend in &backends {
        assert!(
            nnue_backend_available(backend),
            "available_nnue_backends() listed {backend:?} but nnue_backend_available() rejects it"
        );
    }
    for backend in [
        NnueBackendKind::Scalar,
        NnueBackendKind::Simd128,
        NnueBackendKind::Simd256,
        NnueBackendKind::Simd512,
        NnueBackendKind::X86Avx512,
    ] {
        assert_eq!(
            backends.contains(&backend),
            nnue_backend_available(backend),
            "available_nnue_backends() and nnue_backend_available() disagree on {backend:?}"
        );
    }
}

#[test]
fn portable_simd_nnue_backends_available_when_vector_instructions_exist() {
    if x86_v3_available() || aarch64_simd_available() {
        let backends = available_nnue_backends();
        assert!(backends.contains(&NnueBackendKind::Simd128));
        assert!(backends.contains(&NnueBackendKind::Simd256));
        assert!(backends.contains(&NnueBackendKind::Simd512));
    }
}

#[test]
fn critical_game_lines_keep_nnue_incremental_state_exact() {
    let net = NNUENet::load_compact_from_bytes(COMPACT_NET, "<critical PV>")
        .expect("compact NNUE should load");

    assert_incremental_line_matches_refresh(
        &net,
        "rn2nrk1/1pp3b1/3pP2p/p7/2P4q/2N1P1pP/PP1BB3/R1QK3R w - - 2 20",
        &[
            "h1g1", "h4h3", "c1b1", "h3h2", "d1c2", "f8f2", "g1h1", "h2g2", "h1g1", "g2h3", "d2e1",
            "b8c6", "b1d1", "c6b4", "c2b3", "f2g2", "g1g2", "h3g2", "a2a3",
        ],
    );
    assert_incremental_line_matches_refresh(
        &net,
        "r3n1k1/1pp1Prb1/3p3p/p1n5/2P5/2N1P2P/PPKBBqp1/R2Q2R1 w - - 3 25",
        &["e2h5", "g7c3", "b2c3", "f2f5", "c2b2", "f5f2"],
    );
    assert_incremental_line_matches_refresh(
        &net,
        "5k2/1b1r2b1/p4p1N/q3p2Q/2p5/7P/2B2P2/3R2K1 w - - 4 39",
        &["h6f5", "d7d1", "h5d1", "a5d5", "d1d5"],
    );
}
