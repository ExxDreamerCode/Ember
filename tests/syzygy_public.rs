use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ember_chess::board::move_to_uci;
use ember_chess::movegen::generate_moves;
use ember_chess::syzygy::SyzygyTables;
use ember_chess::Engine;
use pyrrhic_rs::WdlProbeResult;

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
    file.set_len(80).unwrap();
}

#[test]
#[ignore = "run with SYZYGY_CI_PATH pointing to the compact Nix tablebase set"]
fn ci_compact_tables_cover_root_interior_and_transitions() {
    let path = std::env::var("SYZYGY_CI_PATH").expect("SYZYGY_CI_PATH is required");
    let names = fs::read_dir(&path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".rtbw") || name.ends_with(".rtbz"))
        .collect::<std::collections::HashSet<_>>();
    let expected = ["KQvK", "KPvK", "KRvK", "KRvKP", "KNNvKR"]
        .into_iter()
        .flat_map(|stem| [format!("{stem}.rtbw"), format!("{stem}.rtbz")])
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(names, expected);

    let mut syzygy = SyzygyTables::new();
    syzygy.load(&path).unwrap();
    assert_eq!(syzygy.max_pieces(), 5);

    let queen = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 84 1");
    assert_eq!(syzygy.probe_wdl(&queen.st), Some(WdlProbeResult::Win));
    assert_eq!(syzygy.probe_dtz(&queen.st), Some(13));
    assert!(syzygy.probe_search_score(&queen.st, 1).unwrap() > 0);
    let queen_moves = generate_moves(&queen.st, queen.st.w, &queen.st.cr, queen.st.ep);
    assert!(syzygy.probe_root_move(&queen.st, &queen_moves).is_some());
    let boundary = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 87 1");
    let draw = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 89 1");
    assert_eq!(syzygy.probe_search_score(&boundary.st, 1), None);
    assert_eq!(syzygy.probe_search_score(&draw.st, 1), Some(0));

    let pawn = engine_from_fen("6k1/8/8/3P4/4K3/8/8/8 w - - 0 1");
    assert_eq!(syzygy.probe_wdl(&pawn.st), Some(WdlProbeResult::Win));
    assert_eq!(syzygy.probe_dtz(&pawn.st), Some(1));
    let pawn_moves = generate_moves(&pawn.st, pawn.st.w, &pawn.st.cr, pawn.st.ep);
    assert!(syzygy.probe_root_move(&pawn.st, &pawn_moves).is_some());

    let capture = engine_from_fen("5k2/R7/8/8/5K2/p7/8/8 w - - 0 62");
    let capture_moves = generate_moves(&capture.st, capture.st.w, &capture.st.cr, capture.st.ep);
    let best = syzygy.probe_root_move(&capture.st, &capture_moves).unwrap();
    assert_eq!(move_to_uci(&capture.st, best), "a7a3");

    let split = engine_from_fen("6rk/8/8/8/8/8/8/KNN5 w - - 0 1");
    assert_eq!(syzygy.probe_wdl(&split.st), Some(WdlProbeResult::Draw));
    assert_eq!(syzygy.probe_dtz(&split.st), Some(0));
    assert_eq!(syzygy.probe_search_score(&split.st, 1), Some(0));
    let split_moves = generate_moves(&split.st, split.st.w, &split.st.cr, split.st.ep);
    assert!(syzygy.probe_root_move(&split.st, &split_moves).is_some());
}

// Direct probe rejection and subsequent successful probes are API/cache
// contracts; a TSV chosen-move fixture cannot express them.
#[test]
#[ignore = "run with SYZYGY_CI_PATH pointing to the compact Nix tablebase set"]
fn ci_attacked_opposing_king_fails_without_disabling_valid_probes() {
    let path = std::env::var("SYZYGY_CI_PATH").expect("SYZYGY_CI_PATH is required");
    let invalid = engine_from_fen("7k/6Q1/8/8/8/8/8/K7 w - - 0 1");
    let witness = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 0 1");
    let witness_moves = generate_moves(&witness.st, witness.st.w, &witness.st.cr, witness.st.ep);
    for warm in [false, true] {
        let mut syzygy = SyzygyTables::new();
        syzygy.load(&path).unwrap();
        if warm {
            assert_eq!(syzygy.probe_wdl(&witness.st), Some(WdlProbeResult::Win));
            assert_eq!(syzygy.probe_dtz(&witness.st), Some(13));
        }
        assert_eq!(syzygy.probe_wdl(&invalid.st), None);
        assert_eq!(syzygy.probe_dtz(&invalid.st), None);
        assert_eq!(syzygy.probe_root_move(&invalid.st, &witness_moves), None);
        assert_eq!(syzygy.probe_search_score(&invalid.st, 1), None);
        assert_eq!(syzygy.probe_wdl(&witness.st), Some(WdlProbeResult::Win));
        assert_eq!(syzygy.probe_dtz(&witness.st), Some(13));
        assert!(syzygy
            .probe_root_move(&witness.st, &witness_moves)
            .is_some());
    }
}

#[test]
#[ignore = "run with SYZYGY_CI_PATH pointing to the compact Nix tablebase set"]
fn ci_corrupt_dtz_does_not_disable_valid_wdl_tables() {
    let path = std::env::var("SYZYGY_CI_PATH").expect("SYZYGY_CI_PATH is required");
    let dir = temp_syzygy_dir();
    for name in ["KQvK.rtbw", "KRvK.rtbw"] {
        fs::copy(Path::new(&path).join(name), dir.join(name)).unwrap();
    }
    fake_table(&dir, "KQvK.rtbz");
    let mut syzygy = SyzygyTables::new();
    syzygy.load(dir.to_str().unwrap()).unwrap();
    let queen = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 0 1");
    let rook = engine_from_fen("7k/8/8/8/8/8/8/1R2K3 w - - 0 1");
    assert_eq!(syzygy.probe_wdl(&queen.st), Some(WdlProbeResult::Win));
    assert_eq!(syzygy.probe_dtz(&queen.st), None);
    assert_eq!(syzygy.probe_wdl(&rook.st), Some(WdlProbeResult::Win));
    drop(syzygy);
    fs::remove_dir_all(dir).unwrap();
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
fn capabilities_use_the_tables_accepted_by_pyrrhic_discovery() {
    let dir = temp_syzygy_dir();
    File::create(dir.join("KQvK.rtbw"))
        .unwrap()
        .set_len(15)
        .unwrap();
    fake_table(&dir, "KRvK.rtbw");
    File::create(dir.join("KRvK.rtbz"))
        .unwrap()
        .set_len(15)
        .unwrap();
    let mut syzygy = SyzygyTables::new();
    syzygy.load(dir.to_str().unwrap()).unwrap();
    let queen = engine_from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1");
    let rook = engine_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    assert!(!syzygy.can_probe_wdl(&queen.st));
    assert!(syzygy.can_probe_wdl(&rook.st));
    assert!(!syzygy.can_probe_dtz(&rook.st));
    drop(syzygy);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lazy_table_failure_falls_back_to_legal_search() {
    let dir = temp_syzygy_dir();
    fake_table(&dir, "KQvK.rtbw");
    fake_table(&dir, "KQvK.rtbz");
    let mut engine = engine_from_fen("7k/8/8/8/8/8/8/1Q2K3 w - - 0 1");
    engine.searcher.syzygy.load(dir.to_str().unwrap()).unwrap();
    assert!(engine.searcher.syzygy.can_probe_wdl(&engine.st));
    assert!(engine.searcher.syzygy.probe_wdl(&engine.st).is_none());
    let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
    let (best, _, _, _) = engine.find_best_move(1.0, 1);
    assert!(legal.iter().any(|mv| move_to_uci(&engine.st, *mv) == best));
    drop(engine);
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
fn disabled_tables_can_be_reenabled_while_a_worker_handle_exists() {
    let dir = temp_syzygy_dir();
    fake_table(&dir, "KQvK.rtbw");
    fake_table(&dir, "KQvK.rtbz");
    let mut syzygy = SyzygyTables::new();
    syzygy.load(dir.to_str().unwrap()).unwrap();
    let worker = syzygy.clone();
    syzygy.load("<empty>").unwrap();
    assert!(!syzygy.is_loaded());
    syzygy.load(dir.to_str().unwrap()).unwrap();
    assert!(syzygy.is_loaded());
    assert!(worker.is_loaded());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn published_generation_changes_without_changing_an_active_snapshot() {
    let old_dir = temp_syzygy_dir();
    let new_dir = temp_syzygy_dir();
    fake_table(&old_dir, "KQvK.rtbw");
    fake_table(&old_dir, "KQvK.rtbz");
    fake_table(&new_dir, "KRPvKR.rtbw");

    let queen = engine_from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1");
    let split = engine_from_fen("6rk/8/8/8/8/8/8/KR1P4 w - - 0 1");
    let mut published = SyzygyTables::new();
    published.load(old_dir.to_str().unwrap()).unwrap();
    let old_snapshot = published.clone();
    let ready = std::sync::Arc::new(std::sync::Barrier::new(2));
    let changed = std::sync::Arc::new(std::sync::Barrier::new(2));
    let worker = std::thread::spawn({
        let ready = ready.clone();
        let changed = changed.clone();
        move || {
            let queen = engine_from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1");
            let split = engine_from_fen("6rk/8/8/8/8/8/8/KR1P4 w - - 0 1");
            ready.wait();
            changed.wait();
            assert_eq!(old_snapshot.max_pieces(), 3);
            assert!(old_snapshot.can_probe_wdl(&queen.st));
            assert!(!old_snapshot.can_probe_wdl(&split.st));
            old_snapshot
        }
    });
    ready.wait();
    published.load(new_dir.to_str().unwrap()).unwrap();
    assert_eq!(published.max_pieces(), 5);
    assert!(!published.can_probe_wdl(&queen.st));
    assert!(published.can_probe_wdl(&split.st));
    changed.wait();
    let old_snapshot = worker.join().unwrap();

    let empty = temp_syzygy_dir();
    assert!(published.load(empty.to_str().unwrap()).is_err());
    assert_eq!(published.max_pieces(), 5);
    let new_snapshot = published.clone();
    published.load("<empty>").unwrap();
    assert!(!published.is_loaded());
    assert_eq!(new_snapshot.max_pieces(), 5);
    assert_eq!(old_snapshot.max_pieces(), 3);
    published.load(new_dir.to_str().unwrap()).unwrap();
    assert_eq!(published.max_pieces(), 5);

    drop(old_snapshot);
    drop(new_snapshot);
    drop(published);
    fs::remove_dir_all(old_dir).unwrap();
    fs::remove_dir_all(new_dir).unwrap();
    fs::remove_dir_all(empty).unwrap();
}

#[test]
fn fifty_move_boundary_and_shared_root_probe_with_real_tables() {
    let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
        eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
        return;
    };
    let mut syzygy = SyzygyTables::new();
    syzygy.load(&path).expect("load regression Syzygy tables");
    let worker = syzygy.clone();
    let fen = |clock| format!("7k/8/8/8/8/8/8/1Q2K3 w - - {clock} 1");
    let win = engine_from_fen(&fen(84));
    let boundary = engine_from_fen(&fen(87));
    let draw = engine_from_fen(&fen(89));
    assert_eq!(syzygy.probe_dtz(&win.st), Some(13));
    assert!(syzygy.probe_search_score(&win.st, 1).unwrap() > 0);
    assert_eq!(syzygy.probe_search_score(&boundary.st, 1), None);
    assert_eq!(syzygy.probe_search_score(&draw.st, 1), Some(0));

    let legal = generate_moves(&win.st, win.st.w, &win.st.cr, win.st.ep);
    let best = syzygy.probe_root_move(&win.st, &legal);
    assert!(
        best.is_some(),
        "shared tablebase handle must allow a root probe"
    );
    assert!(worker.is_loaded());
}

#[test]
fn five_piece_three_versus_two_tables_probe_when_available() {
    let Ok(path) = std::env::var("EMBER_TEST_SYZYGY_PATH") else {
        eprintln!("skipping real Syzygy regressions: EMBER_TEST_SYZYGY_PATH is unset");
        return;
    };
    let mut syzygy = SyzygyTables::new();
    syzygy.load(&path).expect("load regression Syzygy tables");
    if syzygy.max_pieces() < 5 {
        eprintln!("skipping five-piece regression: tablebase has fewer than five pieces");
        return;
    }

    // Direct probe coverage cannot be represented by a TSV root-move fixture.
    for fen in [
        "8/1r6/7R/3k2K1/5p2/8/8/8 b - - 0 43",
        "8/6KP/2q2p2/4k3/8/8/8/8 w - - 0 62",
    ] {
        let engine = engine_from_fen(fen);
        assert!(
            syzygy.probe_search_score(&engine.st, 1).is_some(),
            "missing interior score for {fen}"
        );
        let legal = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        assert!(
            syzygy.probe_root_move(&engine.st, &legal).is_some(),
            "missing root move for {fen}"
        );
    }
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
