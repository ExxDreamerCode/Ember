use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};
use std::thread;

use chess_rs_lib::{evaluate, Engine};

const DEFAULT_CORPUS_WORKERS: usize = 4;
const EXPECTED_HEADER: &str =
    "id\tdepth\tfen_before_blunder\tsetup_move\texpected_move\tthemes\trating\tpopularity\tplays";

struct RegressionCase {
    fixture: String,
    line_number: usize,
    id: String,
    depth: i32,
    fen_before_blunder: String,
    setup_moves: String,
    expected_move: String,
    themes: String,
    rating: u16,
    popularity: u8,
    plays: u32,
}

fn fixture_location(fixture: &str, line_number: usize) -> String {
    format!("{fixture}:{line_number}")
}

fn parse_u16(field: &str, value: &str, fixture: &str, line_number: usize) -> u16 {
    value.parse().unwrap_or_else(|_| {
        panic!(
            "invalid {field} `{value}` at {}",
            fixture_location(fixture, line_number)
        )
    })
}

fn parse_u8(field: &str, value: &str, fixture: &str, line_number: usize) -> u8 {
    value.parse().unwrap_or_else(|_| {
        panic!(
            "invalid {field} `{value}` at {}",
            fixture_location(fixture, line_number)
        )
    })
}

fn parse_u32(field: &str, value: &str, fixture: &str, line_number: usize) -> u32 {
    value.parse().unwrap_or_else(|_| {
        panic!(
            "invalid {field} `{value}` at {}",
            fixture_location(fixture, line_number)
        )
    })
}

fn parse_depth(value: &str, fixture: &str, line_number: usize) -> i32 {
    let depth = value.parse().unwrap_or_else(|_| {
        panic!(
            "invalid depth `{value}` at {}",
            fixture_location(fixture, line_number)
        )
    });
    assert!(
        (1..=64).contains(&depth),
        "{} uses depth {depth}, expected 1..=64",
        fixture_location(fixture, line_number)
    );
    depth
}

fn fixture_paths() -> Vec<PathBuf> {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths = fs::read_dir(&fixture_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture_dir.display()))
        .map(|entry| entry.expect("fixture directory entry should be readable"))
        .filter(|entry| {
            entry
                .file_type()
                .expect("fixture entry type should be readable")
                .is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "tsv")
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    paths.sort();
    assert!(!paths.is_empty(), "no regression fixtures found");
    paths
}

fn parse_fixture(path: &Path, ids: &mut HashSet<String>) -> Vec<RegressionCase> {
    let fixture = path
        .file_name()
        .expect("fixture should have a file name")
        .to_string_lossy()
        .into_owned();
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut lines = contents.lines().enumerate();
    let (header_index, header) = lines
        .by_ref()
        .find(|(_, line)| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with('#')
        })
        .unwrap_or_else(|| panic!("{fixture} should have a header"));
    assert_eq!(
        header.trim(),
        EXPECTED_HEADER,
        "fixture header changed at {}",
        fixture_location(&fixture, header_index + 1)
    );

    let mut cases = Vec::new();
    for (line_index, line) in lines {
        let line_number = line_index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let columns = line.split('\t').collect::<Vec<_>>();
        assert_eq!(
            columns.len(),
            9,
            "{} should have 9 tab-separated columns",
            fixture_location(&fixture, line_number)
        );

        let id = columns[0].to_owned();
        assert!(
            ids.insert(id.clone()),
            "duplicate regression id `{id}` at {}",
            fixture_location(&fixture, line_number)
        );

        cases.push(RegressionCase {
            fixture: fixture.clone(),
            line_number,
            id,
            depth: parse_depth(columns[1], &fixture, line_number),
            fen_before_blunder: columns[2].to_owned(),
            setup_moves: columns[3].to_owned(),
            expected_move: columns[4].to_owned(),
            themes: columns[5].to_owned(),
            rating: parse_u16("rating", columns[6], &fixture, line_number),
            popularity: parse_u8("popularity", columns[7], &fixture, line_number),
            plays: parse_u32("plays", columns[8], &fixture, line_number),
        });
    }

    cases
}

fn regression_cases() -> Vec<RegressionCase> {
    let mut ids = HashSet::new();
    fixture_paths()
        .iter()
        .flat_map(|path| parse_fixture(path, &mut ids))
        .collect()
}

fn parse_uci_move(mv: &str) -> (usize, usize, usize, usize, u8) {
    let bytes = mv.as_bytes();
    assert!(matches!(bytes.len(), 4 | 5), "invalid UCI move: {mv}");
    assert!(
        (b'a'..=b'h').contains(&bytes[0])
            && (b'1'..=b'8').contains(&bytes[1])
            && (b'a'..=b'h').contains(&bytes[2])
            && (b'1'..=b'8').contains(&bytes[3]),
        "invalid UCI move: {mv}"
    );
    let sc = (bytes[0] - b'a') as usize;
    let sr = 8 - (bytes[1] - b'0') as usize;
    let ec = (bytes[2] - b'a') as usize;
    let er = 8 - (bytes[3] - b'0') as usize;
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

fn move_matches_expectation(actual: &str, expectation: &str) -> bool {
    if let Some(forbidden) = expectation.strip_prefix('!') {
        !forbidden.split('|').any(|mv| mv == actual)
    } else {
        expectation.split('|').any(|mv| mv == actual)
    }
}

fn solve_case(case: &RegressionCase) -> Result<(), String> {
    let mut engine = Engine::new();
    engine.book = None;
    engine.num_threads = 1;
    engine
        .try_set_fen(&case.fen_before_blunder)
        .map_err(|error| format!("invalid FEN for {}: {error}", case.id))?;

    if case.setup_moves != "-" {
        for setup_move in case.setup_moves.split_ascii_whitespace() {
            let (sr, sc, er, ec, promotion) = parse_uci_move(setup_move);
            if !engine.make_move_uci(sr, sc, er, ec, promotion) {
                return Err(format!(
                    "setup move {setup_move} should be legal for regression {} ({}) at {}",
                    case.id,
                    case.themes,
                    fixture_location(&case.fixture, case.line_number)
                ));
            }
        }
    }

    let (best_move, score, nodes, elapsed) = engine.find_best_move(1_000_000.0, case.depth);
    if !move_matches_expectation(&best_move, &case.expected_move) {
        return Err(format!(
            "failed regression {} ({}) at {} and depth {}; expected={}, got={best_move}; rating={}, popularity={}, plays={}; score={score}, nodes={nodes}, elapsed={elapsed:.3}s",
            case.id,
            case.themes,
            fixture_location(&case.fixture, case.line_number),
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
fn regression_fixture_files_are_well_formed() {
    assert!(!regression_cases().is_empty(), "no regression cases found");
}

#[test]
#[ignore = "runs every move regression fixture in a dedicated CI job"]
fn ember_solves_move_regression_fixtures() {
    init_nnue();

    let cases = regression_cases();
    let workers = corpus_worker_count(cases.len());
    let next_case = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let case_index = next_case.fetch_add(1, Ordering::Relaxed);
                let Some(case) = cases.get(case_index) else {
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
        "{} move-regression failures across {} cases:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
