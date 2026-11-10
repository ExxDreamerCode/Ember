use ember_chess::chainpack::{
    ember_square, pack_stem, scan_chunk, unpack_chain, unpack_stem, ChainPos, MoveScoreListReader,
    MoveScoreListWriter, PackedMove, PackedMoveKind, StemEntry, STEM_SIZE,
};
use ember_chess::movegen::apply_move;
use ember_chess::Engine;

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn state_from_fen(fen: &str) -> ember_chess::board::BoardState {
    let mut engine = Engine::try_new().expect("failed to create engine");
    engine.set_fen(fen);
    engine.st
}

fn replay_game(fen: &str, moves: &[&str]) -> Vec<(ChainPos, PackedMove, i16)> {
    let mut st = state_from_fen(fen);
    let mut out = Vec::new();
    for uci in moves {
        let pos = ChainPos::from_ember(&st);
        let mv = pos.packed_move_from_uci(uci).expect("legal uci move");
        out.push((pos, mv, 0));

        let b = uci.as_bytes();
        let from = (b[1] - b'1') as usize * 8 + (b[0] - b'a') as usize;
        let to = (b[3] - b'1') as usize * 8 + (b[2] - b'a') as usize;
        let promo_char = match b.get(4) {
            Some(b'n') => b'N',
            Some(b'b') => b'B',
            Some(b'r') => b'R',
            Some(b'q') => b'Q',
            _ => 0,
        };
        apply_move(
            &mut st,
            ember_square(from) / 8,
            ember_square(from) % 8,
            ember_square(to) / 8,
            ember_square(to) % 8,
            promo_char,
        );
    }
    out
}

fn pack_and_unpack(
    game: &[(ChainPos, PackedMove, i16)],
    head_score: i16,
    ply: u16,
    result: i16,
) -> Vec<StemEntry> {
    assert!(!game.is_empty());
    let (head_pos, head_mv, _) = &game[0];
    let stem = pack_stem(head_pos, head_mv, head_score, ply, result);
    let mut writer = MoveScoreListWriter::new(head_score);
    for (pos, mv, score) in &game[1..] {
        writer.add_move(pos, mv, *score);
    }

    let head = unpack_stem(&stem).expect("stem decodes");
    let mut reader = MoveScoreListReader::new(&head, writer.movetext(), writer.num_plies());
    let mut entries = vec![head];
    let mut i = 1usize;
    while reader.has_next() {
        let entry = reader
            .next_entry()
            .unwrap_or_else(|| panic!("entry {i} failed to decode"));
        assert_eq!(&entry.pos, &game[i].0, "replayed position {i} diverged");
        assert_eq!(&entry.mv, &game[i].1, "decoded move {i} diverged");
        entries.push(entry);
        i += 1;
    }
    assert_eq!(entries.len(), game.len());
    entries
}

#[test]
fn zigzag_round_trip() {
    use ember_chess::chainpack::{signed_to_unsigned, unsigned_to_signed};
    for v in [i16::MIN, -30000, -1, 0, 1, 12345, i16::MAX] {
        assert_eq!(unsigned_to_signed(signed_to_unsigned(v)), v);
    }
}

#[test]
fn startpos_stem_matches_reference_bytes() {
    let expected: [u8; STEM_SIZE] = [
        0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x2d, 0x84, 0x4a, 0xd2, 0x00, 0x00, 0x00,
        0x00, 0x11, 0x11, 0x11, 0x11, 0x3e, 0x95, 0x5b, 0xe3, 0x0c, 0x70, 0x00, 0x00, 0x80, 0x00,
        0x00, 0x00,
    ];
    let st = state_from_fen(START_FEN);
    let pos = ChainPos::from_ember(&st);
    let mv = pos.packed_move_from_uci("e2e4").unwrap();
    let stem = pack_stem(&pos, &mv, 0, 0, 1);
    assert_eq!(
        stem, expected,
        "startpos e2e4 stem must match the reference byte-for-byte"
    );
}

#[test]
fn italian_game_with_castling_round_trips() {
    let moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f8e7", "d2d3", "d7d6", "c2c3",
        "e8g8",
    ];
    let game = replay_game(START_FEN, &moves);
    let scores: Vec<i16> = (0..game.len() as i16).map(|i| i * 37 - 100).collect();
    let game: Vec<_> = game
        .into_iter()
        .zip(&scores)
        .map(|((pos, mv, _), s)| (pos, mv, *s))
        .collect();

    let entries = pack_and_unpack(&game, scores[0], 10, 1);
    for (i, (entry, (pos, mv, score))) in entries.iter().zip(&game).enumerate() {
        assert_eq!(&entry.pos, pos, "position {i} diverged");
        assert_eq!(&entry.mv, mv, "move {i} diverged");
        assert_eq!(&entry.score, score, "score {i} diverged");
        assert_eq!(entry.ply, 10 + i as u16, "ply {i} diverged");
        assert_eq!(entry.result, if i % 2 == 0 { 1 } else { -1 });
    }
}

#[test]
fn en_passant_round_trip() {
    let moves = [
        "e2e4", "a7a6", "e4e5", "d7d5", "e5d6", "b8c6", "g1f3", "c6d4", "f3d4",
    ];
    let game = replay_game(START_FEN, &moves);
    assert_eq!(game[4].1.kind, PackedMoveKind::EnPassant, "e5d6 must be ep");
    assert_eq!(
        game[7].1.kind,
        PackedMoveKind::Normal,
        "Nc6-d4 is a normal move"
    );

    let scores: Vec<i16> = (0..game.len() as i16).map(|i| 500 - i * 53).collect();
    let game: Vec<_> = game
        .into_iter()
        .zip(&scores)
        .map(|((pos, mv, _), s)| (pos, mv, *s))
        .collect();
    let entries = pack_and_unpack(&game, scores[0], 0, 0);
    for (i, (entry, (pos, mv, score))) in entries.iter().zip(&game).enumerate() {
        assert_eq!(&entry.pos, pos, "position {i} diverged");
        assert_eq!(&entry.mv, mv, "move {i} diverged");
        assert_eq!(&entry.score, score, "score {i} diverged");
    }
}

#[test]
fn promotion_chain_round_trip() {
    let moves = [
        "h2h4", "g7g5", "h4g5", "h7h6", "g5h6", "g8f6", "h6h7", "f6g8", "h7g8q",
    ];
    let game = replay_game(START_FEN, &moves);
    assert_eq!(game[8].1.promotion, 4, "g8q capture promotion");
    assert_eq!(game[8].1.to, 62, "promotes on g8");

    let scores: Vec<i16> = (0..game.len() as i16).map(|i| -i * 211).collect();
    let game: Vec<_> = game
        .into_iter()
        .zip(&scores)
        .map(|((pos, mv, _), s)| (pos, mv, *s))
        .collect();
    let entries = pack_and_unpack(&game, scores[0], 0, -1);
    for (i, (entry, (pos, mv, score))) in entries.iter().zip(&game).enumerate() {
        assert_eq!(&entry.pos, pos, "position {i} diverged");
        assert_eq!(&entry.mv, mv, "move {i} diverged");
        assert_eq!(&entry.score, score, "score {i} diverged");
    }
}

#[test]
fn score_deltas_use_alternating_sign_convention() {
    let moves = ["e2e4", "e7e5", "g1f3", "b8c6"];
    let game = replay_game(START_FEN, &moves);
    let scores = [30000i16, -30000, 30000, -30000];
    let game: Vec<_> = game
        .into_iter()
        .zip(scores)
        .map(|((pos, mv, _), s)| (pos, mv, s))
        .collect();
    let entries = pack_and_unpack(&game, 30000, 0, 1);
    for (entry, (_, _, score)) in entries.iter().zip(&game) {
        assert_eq!(entry.score, *score);
    }
}

#[test]
fn stem_position_decodes_castling_rights_and_ep() {
    let st = state_from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 3");
    let pos = ChainPos::from_ember(&st);
    assert_eq!(pos.castling, [true, true, true, true]);
    let mv = pos.packed_move_from_uci("e5e6").unwrap();
    let stem = pack_stem(&pos, &mv, -40, 4, 0);

    let entry = unpack_stem(&stem).expect("stem decodes");
    assert_eq!(entry.pos.castling, [true, true, true, true]);
    assert_eq!(entry.pos.ep, Some(44), "ep target e6 = std 44");
    assert_eq!(entry.score, -40);
    assert_eq!(entry.ply, 4);
    assert_eq!(entry.result, 0);
    assert_eq!(entry.rule50, 0);
}

#[test]
fn black_to_move_stem_round_trips() {
    let st = state_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let pos = ChainPos::from_ember(&st);
    let mv = pos.packed_move_from_uci("e7e5").unwrap();
    let stem = pack_stem(&pos, &mv, 12, 1, -1);
    let entry = unpack_stem(&stem).expect("stem decodes");
    assert!(!entry.pos.white_stm);
    assert_eq!(entry.pos.ep, Some(20), "ep target e3 = std 20");
    assert_eq!(entry.ply, 1);
    assert_eq!(entry.result, -1);
}

#[test]
fn scan_chunk_counts_chained_entries() {
    let moves = ["e2e4", "e7e5", "g1f3"];
    let game = replay_game(START_FEN, &moves);
    let (ref pos0, mv0, _) = game[0];
    let stem = pack_stem(pos0, &mv0, 10, 0, 1);
    let mut writer = MoveScoreListWriter::new(10);
    for (pos, mv, _) in &game[1..] {
        writer.add_move(pos, mv, 0);
    }
    let mut payload = stem.to_vec();
    payload.extend_from_slice(&writer.num_plies().to_be_bytes());
    payload.extend_from_slice(writer.movetext());

    let (entries, consumed) = scan_chunk(&payload).expect("chunk scans");
    assert_eq!(entries, 3, "one stem plus two chained entries");
    assert_eq!(consumed, payload.len());

    assert!(scan_chunk(&payload[..payload.len() - 1]).is_none());
}

#[test]
fn chained_reader_replays_positions_incrementally() {
    let moves = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"];
    let game = replay_game(START_FEN, &moves);
    let (ref pos0, mv0, _) = game[0];
    let head = StemEntry {
        pos: pos0.clone(),
        mv: mv0,
        score: -25,
        ply: 20,
        result: 1,
        rule50: 0,
    };
    let mut writer = MoveScoreListWriter::new(head.score);
    let mut expected_scores = vec![head.score];
    for (pos, mv, _) in &game[1..] {
        let s = writer.num_plies() as i16 * 13 - 50;
        writer.add_move(pos, mv, s);
        expected_scores.push(s);
    }

    let mut reader = MoveScoreListReader::new(&head, writer.movetext(), writer.num_plies());
    let mut got_scores = vec![head.score];
    let mut prev_pos = head.pos.clone();
    while reader.has_next() {
        let entry = reader.next_entry().expect("entry decodes");

        assert_eq!(entry.pos, game[got_scores.len()].0);
        assert_eq!(entry.ply, 20 + got_scores.len() as u16);
        assert_eq!(entry.result, if got_scores.len() % 2 == 1 { -1 } else { 1 });
        got_scores.push(entry.score);
        prev_pos = entry.pos.clone();
    }
    assert_eq!(got_scores, expected_scores);
    assert_eq!(prev_pos, game.last().unwrap().0);
}

#[test]
fn fen_reconstruction_matches_ember_after_replay() {
    let moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1",
    ];
    let game = replay_game(START_FEN, &moves);
    let (ref pos0, mv0, _) = game[0];
    let stem = pack_stem(pos0, &mv0, 0, 0, 0);
    let mut writer = MoveScoreListWriter::new(0);
    for (pos, mv, _) in &game[1..] {
        writer.add_move(pos, mv, 0);
    }
    let entries = unpack_chain(&stem, writer.num_plies(), writer.movetext()).unwrap();

    for (i, (entry, (pos, mv, _))) in entries.iter().zip(&game).enumerate() {
        assert_eq!(&entry.pos, pos, "position {i} diverged");
        assert_eq!(&entry.mv, mv, "move {i} diverged");
    }

    let mut st = state_from_fen(START_FEN);
    for uci in moves {
        let from = ember_square(if uci.len() >= 4 {
            (uci.as_bytes()[1] - b'1') as usize * 8 + (uci.as_bytes()[0] - b'a') as usize
        } else {
            0
        });
        let to = ember_square(
            (uci.as_bytes()[3] - b'1') as usize * 8 + (uci.as_bytes()[2] - b'a') as usize,
        );
        let promo = match uci.as_bytes().get(4) {
            Some(b'n') => b'N',
            Some(b'b') => b'B',
            Some(b'r') => b'R',
            Some(b'q') => b'Q',
            _ => 0,
        };
        apply_move(&mut st, from / 8, from % 8, to / 8, to % 8, promo);
    }

    let expected = ChainPos::from_ember(&st);
    let last = entries.last().unwrap();
    let mut after_last = last.pos.clone();
    after_last.do_move(&last.mv);
    assert_eq!(after_last, expected);
}
