use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ember_chess::board::move_to_uci;
use ember_chess::movegen::generate_moves;
use ember_chess::syzygy::SyzygyTables;
use ember_chess::Engine;
use shakmaty_syzygy::{Dtz, Wdl};

fn engine_from_fen(fen: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_fen(fen);
    engine
}

fn temp_syzygy_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ember-syzygy-test-{unique}"));
    fs::create_dir(&dir).unwrap();
    dir
}

fn fake_table(dir: &Path, name: &str) {
    let file = File::create(dir.join(name)).unwrap();
    file.set_len(16).unwrap();
}

#[test]
fn dependency_dtz_recurrence_counts_the_root_ply() {
    assert_eq!((-Dtz(-2)).add_plies(1), Dtz(3));
    assert_eq!((-Dtz(2)).add_plies(1), Dtz(-3));
    assert_eq!(Dtz::before_zeroing(Wdl::Win), Dtz(1));
    assert_eq!(Dtz::before_zeroing(Wdl::CursedWin), Dtz(101));
    assert_eq!(Dtz::before_zeroing(Wdl::BlessedLoss), Dtz(-101));
}

#[test]
fn loaded_capabilities_filter_by_piece_count_and_material() {
    let dir = temp_syzygy_dir();
    fake_table(&dir, "KQvK.rtbw");
    fake_table(&dir, "KQvK.rtbz");

    let mut syzygy = SyzygyTables::new();
    syzygy.load(dir.to_str().unwrap()).unwrap();

    let kqvk = engine_from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1");
    let kvkq = engine_from_fen("q6k/8/8/8/8/8/8/4K3 w - - 0 1");
    let krv_k = engine_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    let kqvkr = engine_from_fen("r6k/8/8/8/8/8/8/Q3K3 w - - 0 1");

    assert_eq!(syzygy.max_pieces(), 3);
    assert!(syzygy.can_probe_wdl(&kqvk.st));
    assert!(syzygy.can_probe_dtz(&kqvk.st));
    assert!(syzygy.can_probe_wdl(&kvkq.st));
    assert!(!syzygy.can_probe_wdl(&krv_k.st));
    assert!(!syzygy.can_probe_wdl(&kqvkr.st));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn fast_check_accepts_en_passant_but_rejects_castling() {
    let dir = temp_syzygy_dir();
    fake_table(&dir, "KRvKR.rtbw");

    let mut syzygy = SyzygyTables::new();
    syzygy.load(dir.to_str().unwrap()).unwrap();

    let no_rights = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1");
    let castling = engine_from_fen("4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1");
    let ep = engine_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");

    assert!(syzygy.can_probe_wdl(&no_rights.st));
    assert!(!syzygy.can_probe_wdl(&castling.st));
    assert!(SyzygyTables::pieces_ok(&ep.st));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn known_six_piece_root_moves_when_tables_are_available() {
    let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
        eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
        return;
    };
    let mut syzygy = SyzygyTables::new();
    syzygy.load(&path).expect("load regression Syzygy tables");
    if syzygy.max_pieces() < 6 {
        eprintln!("skipping six-piece regressions: tablebase has fewer than six pieces");
        return;
    }

    let cases: &[(&str, &[&str])] = &[
        ("8/1r6/7R/3k2p1/5pK1/8/8/8 w - - 0 43", &["h6a6"]),
        ("8/2b4k/p7/4p3/4K3/1N6/8/8 w - - 4 50", &["e4f5", "b3d2"]),
        ("8/6k1/8/r5PR/2K4P/8/8/8 b - - 10 65", &["a5a6"]),
        ("5R2/3k2r1/1K6/1P6/8/8/5p2/8 b - - 1 51", &["g7g2"]),
        ("4q3/6KP/2N2p2/4k3/8/8/8/8 b - - 14 61", &["e5d6", "e5d5"]),
        ("1R6/8/7k/8/6p1/1P6/6r1/1K6 b - - 2 62", &["g2f2"]),
        ("6R1/8/8/1P6/7k/6p1/4r3/2K5 b - - 0 66", &["g3g2"]),
        ("5k2/8/8/p6P/n2K4/8/5P2/8 w - - 2 47", &["f2f4"]),
        ("1R6/4P2k/3K4/8/8/6p1/8/4r3 w - - 0 95", &["e7e8q", "e7e8r"]),
    ];

    for &(fen, expected) in cases {
        let engine = engine_from_fen(fen);
        let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let best = syzygy
            .probe_root_move(&engine.st, &legal)
            .expect("probe canonical root move");
        let actual = move_to_uci(&engine.st, best);
        assert!(
            expected.contains(&actual.as_str()),
            "expected one of {expected:?}, got {actual} for {fen}"
        );
    }
}

#[test]
fn zeroing_capture_regression_only_needs_four_piece_tables() {
    let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
        eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
        return;
    };
    let mut syzygy = SyzygyTables::new();
    syzygy.load(&path).expect("load regression Syzygy tables");
    let engine = engine_from_fen("5k2/R7/8/8/5K2/p7/8/8 w - - 0 62");
    if !syzygy.can_probe_dtz(&engine.st) {
        eprintln!("skipping zeroing regression: KRvKP tables are unavailable");
        return;
    }
    let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let best = syzygy
        .probe_root_move(&engine.st, &legal)
        .expect("probe zeroing capture regression");

    assert_eq!(move_to_uci(&engine.st, best), "a7a3");
}
