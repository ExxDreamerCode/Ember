use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};
use std::thread;

use chess_rs_lib::{evaluate, Engine};

// Fixed subset of the public-domain Lichess puzzle corpus. Active rows are
// puzzles Ember currently solves; commented rows are mined misses kept for
// follow-up engine work.
const CORPUS: &str = include_str!("fixtures/lichess_puzzle_corpus.tsv");
const EXPECTED_CASES: usize = 870;
const DEFAULT_CORPUS_WORKERS: usize = 4;
const EXPECTED_HEADER: &str =
    "id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays";

const SEED_IDS: &[&str] = &[
    "00008", "0008Q", "0009B", "000Pw", "000Sa", "000Zo", "000hf", "000lC", "0017R", "001XA",
    "001h8", "001m3", "001wR", "001wr", "001xl", "002KJ", "002O7", "002Ua", "002bK", "0039T",
    "003Jb", "003S3", "003Tx", "003o0",
];

#[derive(Clone, Copy)]
struct PuzzleCase {
    id: &'static str,
    depth: i32,
    fen_before_blunder: &'static str,
    setup_move: &'static str,
    expected_move: &'static str,
    themes: &'static str,
    rating: u16,
    popularity: u8,
    plays: u32,
}

fn parse_u16(field: &str, value: &str, line_number: usize) -> u16 {
    value
        .parse()
        .unwrap_or_else(|_| panic!("invalid {field} `{value}` on fixture line {line_number}"))
}

fn parse_u8(field: &str, value: &str, line_number: usize) -> u8 {
    value
        .parse()
        .unwrap_or_else(|_| panic!("invalid {field} `{value}` on fixture line {line_number}"))
}

fn parse_u32(field: &str, value: &str, line_number: usize) -> u32 {
    value
        .parse()
        .unwrap_or_else(|_| panic!("invalid {field} `{value}` on fixture line {line_number}"))
}

fn parse_depth(value: &str, line_number: usize) -> i32 {
    let depth = value
        .parse()
        .unwrap_or_else(|_| panic!("invalid depth `{value}` on fixture line {line_number}"));
    assert!(
        (2..=4).contains(&depth),
        "fixture line {line_number} uses depth {depth}, expected 2..=4"
    );
    depth
}

fn corpus_cases() -> Vec<PuzzleCase> {
    let mut lines = CORPUS.lines();
    let header = lines.next().expect("corpus fixture should have a header");
    assert_eq!(header, EXPECTED_HEADER, "corpus fixture header changed");

    let mut ids = HashSet::new();
    let mut cases = Vec::with_capacity(EXPECTED_CASES);
    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let columns: Vec<_> = line.split('\t').collect();
        assert_eq!(
            columns.len(),
            9,
            "fixture line {line_number} should have 9 tab-separated columns"
        );

        let id = columns[0];
        assert!(
            ids.insert(id),
            "duplicate Lichess puzzle id `{id}` on fixture line {line_number}"
        );

        cases.push(PuzzleCase {
            id,
            depth: parse_depth(columns[1], line_number),
            fen_before_blunder: columns[2],
            setup_move: columns[3],
            expected_move: columns[4],
            themes: columns[5],
            rating: parse_u16("rating", columns[6], line_number),
            popularity: parse_u8("popularity", columns[7], line_number),
            plays: parse_u32("plays", columns[8], line_number),
        });
    }

    assert_eq!(
        cases.len(),
        EXPECTED_CASES,
        "unexpected Lichess puzzle fixture size"
    );
    for seed_id in SEED_IDS {
        assert!(
            ids.contains(seed_id),
            "original seed puzzle `{seed_id}` was removed from the fixture"
        );
    }

    cases
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

fn init_nnue() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        evaluate::init_embedded_nnue().expect("embedded NNUE should load");
    });
}

fn corpus_worker_count(case_count: usize) -> usize {
    std::env::var("EMBER_LICHESS_CORPUS_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(DEFAULT_CORPUS_WORKERS)
        .min(case_count.max(1))
}

fn solve_case(case: PuzzleCase) -> Result<(), String> {
    let mut engine = Engine::new();
    engine.book = None;
    engine.num_threads = 1;
    engine
        .try_set_fen(case.fen_before_blunder)
        .expect("fixture FEN should be valid");

    let (sr, sc, er, ec, promotion) = parse_uci_move(case.setup_move);
    if !engine.make_move_uci(sr, sc, er, ec, promotion) {
        return Err(format!(
            "setup move {} should be legal for Lichess puzzle {} ({})",
            case.setup_move, case.id, case.themes
        ));
    }

    let (best_move, score, nodes, elapsed) = engine.find_best_move(1_000_000.0, case.depth);
    if best_move != case.expected_move {
        return Err(format!(
            "failed Lichess puzzle {} ({}) at depth {}; expected={}, got={best_move}; rating={}, popularity={}, plays={}; score={score}, nodes={nodes}, elapsed={elapsed:.3}s",
            case.id,
            case.themes,
            case.depth,
            case.expected_move,
            case.rating,
            case.popularity,
            case.plays
        ));
    }

    Ok(())
}

#[test]
#[ignore = "runs the full 870-position Lichess corpus in a dedicated CI job"]
fn ember_solves_public_lichess_tactic_corpus() {
    init_nnue();

    let cases = corpus_cases();
    let workers = corpus_worker_count(cases.len());
    let next_case = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let case_index = next_case.fetch_add(1, Ordering::Relaxed);
                let Some(case) = cases.get(case_index).copied() else {
                    break;
                };

                if let Err(message) = solve_case(case) {
                    failures.lock().unwrap().push(message);
                }
            });
        }
    });

    let failures = failures.into_inner().unwrap();
    assert!(
        failures.is_empty(),
        "{} Lichess corpus failures across {} cases:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
