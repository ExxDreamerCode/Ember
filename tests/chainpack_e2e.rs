use std::env;
use std::fs;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use ember_chess::board::board_to_fen;
use ember_chess::chainpack::{scan_chunk, scan_single_chain, STEM_SIZE};
use ember_chess::Engine;

const MAGIC: &[u8; 4] = b"BINP";

fn temp_path(suffix: &str) -> PathBuf {
    env::temp_dir().join(format!("chainpack_e2e_{}.{suffix}", std::process::id()))
}

fn stem_nibble(stem: &[u8; STEM_SIZE], occupied: u64, std_sq: usize) -> Option<u8> {
    let before = occupied & ((1u64 << std_sq) - 1);
    let idx = before.count_ones() as usize;
    if (occupied >> std_sq) & 1 == 0 {
        return None;
    }
    let byte = stem[8 + idx / 2];
    Some(if idx.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    })
}

#[test]
fn chainpack_writes_one_chain_per_game_with_chained_moves() {
    let pgn = r#"[Event "e1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5

[Event "e2"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 2. c4
"#;
    let pgn_path = temp_path("pgn");
    let out_path = temp_path("binpack");
    fs::write(&pgn_path, pgn).unwrap();

    let bin = env::var("CARGO_BIN_EXE_chainpack")
        .expect("CARGO_BIN_EXE_chainpack is set for integration tests");
    let status = std::process::Command::new(&bin)
        .args([
            "--eval-only",
            pgn_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .status()
        .expect("chainpack binary should run");
    assert!(status.success(), "chainpack should exit successfully");

    let file = fs::File::open(&out_path).unwrap();
    let mut reader = BufReader::new(file);

    let mut chains: Vec<([u8; STEM_SIZE], u16)> = Vec::new();
    let mut total_entries = 0u64;

    loop {
        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() {
            break;
        }
        assert_eq!(&magic, MAGIC, "every chunk must start with the BINP magic");
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).unwrap();
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        reader.read_exact(&mut payload).unwrap();

        let (entries, consumed) =
            scan_chunk(&payload).expect("payload must consist of decodable chains");
        assert_eq!(consumed, payload.len());
        total_entries += entries;

        let mut pos = 0usize;
        while pos < payload.len() {
            let mut stem = [0u8; STEM_SIZE];
            stem.copy_from_slice(&payload[pos..pos + STEM_SIZE]);
            let num_plies = u16::from_be_bytes([payload[pos + 32], payload[pos + 33]]);
            chains.push((stem, num_plies));
            let (_, consumed_single) =
                scan_single_chain(&payload[pos..]).expect("single chain decodes");
            pos += consumed_single;
        }
        assert_eq!(pos, payload.len(), "payload must consist of whole chains");
    }

    assert_eq!(chains.len(), 2, "one chain stem per game");
    let mut plies: Vec<u16> = chains.iter().map(|(_, n)| *n).collect();
    plies.sort_unstable();
    assert_eq!(plies, vec![2, 4], "chained plies follow the game lengths");
    assert_eq!(total_entries, 8, "stems plus chained entries");

    let first = &chains[0].0;
    let occupied = u64::from_be_bytes(first[0..8].try_into().unwrap());
    let mut engine = Engine::try_new().expect("engine");
    engine.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let start = engine.st;
    let mut expected_occupied = 0u64;
    for bb in start.bb {
        expected_occupied |= bb;
    }
    assert_eq!(
        occupied, expected_occupied,
        "first entry is the initial position"
    );

    assert_eq!(
        stem_nibble(first, occupied, 60),
        Some(11),
        "black king nibble without black-to-move marker"
    );
    assert_eq!(
        stem_nibble(first, occupied, 4),
        Some(10),
        "white king nibble"
    );
    assert_eq!(
        u16::from_be_bytes([first[30], first[31]]),
        0,
        "rule fifty starts at zero"
    );

    for (stem, _) in &chains {
        let occ = u64::from_be_bytes(stem[0..8].try_into().unwrap());
        let black_king_markers = (0..64)
            .filter(|&s| {
                let idx = (occ & ((1u64 << s) - 1)).count_ones() as usize;
                (occ >> s) & 1 == 1 && {
                    let byte = stem[8 + idx / 2];
                    let nibble = if idx.is_multiple_of(2) {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    };
                    nibble == 15
                }
            })
            .count();
        assert_eq!(
            black_king_markers, 0,
            "chain stems start from white-to-move positions"
        );
    }

    let _ = fs::remove_file(&pgn_path);
    let _ = fs::remove_file(&out_path);
}

#[test]
fn fen_round_trip_preserves_static_eval() {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    ];
    for fen in fens {
        let mut engine = Engine::try_new().expect("engine");
        engine.set_fen(fen);
        let st = engine.st;
        let fen_again = board_to_fen(&st);
        let mut engine2 = Engine::try_new().expect("engine");
        engine2.set_fen(&fen_again);
        let score1 = engine.searcher.corrected_eval(&st);
        let score2 = engine2.searcher.corrected_eval(&engine2.st);
        assert_eq!(
            score1, score2,
            "fen round-trip must preserve evaluation: {fen}"
        );
    }
}
