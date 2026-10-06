use ember_chess::tt::{SharedTT, TT_ALPHA, TT_BETA, TT_EXACT};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

#[test]
fn tt_store_get_roundtrip() {
    let tt = SharedTT::new(1);
    let key = 0x123456789ABCDEF0u64;
    tt.store(key, 7, 250, TT_EXACT, Some(0xABCD));
    let result = tt.get_depth(key);
    assert!(result.is_some(), "should find stored entry");
    let (d, s, f, m) = result.unwrap();
    assert_eq!(d, 7, "depth mismatch");
    assert_eq!(s, 250, "score mismatch");
    assert_eq!(f, TT_EXACT, "flag mismatch");
    assert_eq!(m, Some(0xABCD), "move mismatch");
}

#[test]
fn tt_mate_score_survives_roundtrip() {
    let tt = SharedTT::new(1);
    let key = 0xDEAD_BEEF_0000_0001u64;
    tt.store(key, 12, 99_991, TT_EXACT, Some(0x4242));
    let (_, s, _, _) = tt.get_depth(key).unwrap();
    assert_eq!(s, 99_991, "mate score must not be truncated by i16 packing");

    let key2 = key ^ 1;
    tt.store(key2, 8, -100_000, TT_EXACT, None);
    let (_, s2, _, _) = tt.get_depth(key2).unwrap();
    assert_eq!(s2, -100_000, "negative tablebase score must survive");
}

#[test]
fn tt_store_replace_deeper() {
    let tt = SharedTT::new(1);
    let key = 42;
    tt.store(key, 1, 100, TT_ALPHA, None);
    tt.store(key, 2, 200, TT_ALPHA, None);
    let (d, _, _, _) = tt.get_depth(key).unwrap();
    assert!(d >= 2, "deeper entry should replace: got depth {d}");
}

#[test]
fn tt_exact_always_replaces() {
    let tt = SharedTT::new(1);
    let key = 99;
    tt.store(key, 5, 100, TT_BETA, None);
    tt.store(key, 3, 300, TT_EXACT, Some(0x42));
    let (d, s, f, m) = tt.get_depth(key).unwrap();
    assert_eq!(d, 3, "TT_EXACT should replace even if shallower");
    assert_eq!(s, 300, "TT_EXACT score");
    assert_eq!(f, TT_EXACT);
    assert_eq!(m, Some(0x42));
}

#[test]
fn tt_lookup_preserves_score_sign() {
    let tt = SharedTT::new(1);
    tt.store(0x1001, 1, -42, TT_EXACT, None);
    let (_, s, _, _) = tt.get_depth(0x1001).unwrap();
    assert_eq!(s, -42, "negative score must survive round-trip");
}

#[test]
fn tt_miss_on_wrong_key() {
    let tt = SharedTT::new(1);
    tt.store(0xAAAA, 1, 100, TT_EXACT, None);
    assert!(tt.get_depth(0xBBBB).is_none(), "wrong key should miss");
}

#[test]
fn tt_probe_observes_entry_published_before_flag() {
    const ROUNDS: u32 = 40_000;
    const KEY: u64 = 0x5A5A_0000_1234_0001;

    let tt = Arc::new(SharedTT::new(1));
    let published = Arc::new(AtomicBool::new(false));
    let writer = {
        let tt = Arc::clone(&tt);
        let published = Arc::clone(&published);
        thread::spawn(move || {
            for round in 0..ROUNDS {
                while published.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let best_move = 0x1001 + round as u16;
                tt.store(KEY, 4, round as i32, TT_EXACT, Some(best_move));
                published.store(true, Ordering::Release);
            }
        })
    };

    for round in 0..ROUNDS {
        while !published.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        let entry = tt
            .get_entry(KEY)
            .expect("entry stored before the flag must be visible to the probe");
        assert_eq!(
            entry.score, round as i32,
            "probe must observe exactly the entry published before the flag"
        );
        assert_eq!(entry.best_move, Some(0x1001 + round as u16));
        assert_eq!(entry.depth, 4);
        published.store(false, Ordering::Release);
    }
    writer.join().expect("writer thread panicked");
}

#[test]
fn tt_concurrent_probes_never_observe_torn_entries() {
    const ROUNDS: u32 = 200_000;
    const KEY: u64 = 0xC0DE_0000_0BAD_F00D;

    fn assert_consistent(entry: &ember_chess::tt::TTEntry) {
        let expected_depth = (entry.score % 8) + 1;
        let expected_move = 0x2000 + (entry.score as u32 & 0xFFF);
        assert_eq!(
            entry.depth, expected_depth,
            "torn entry: depth does not match score"
        );
        assert_eq!(
            entry.best_move,
            Some(expected_move as u16),
            "torn entry: best_move does not match score"
        );
    }

    let tt = Arc::new(SharedTT::new(1));
    let writer_done = Arc::new(AtomicBool::new(false));
    let writer = {
        let tt = Arc::clone(&tt);
        let writer_done = Arc::clone(&writer_done);
        thread::spawn(move || {
            for i in 0..ROUNDS {
                let score = i as i32;
                let depth = ((i % 8) + 1) as i32;
                let best_move = (0x2000 + (i & 0xFFF)) as u16;
                tt.store(KEY, depth, score, TT_EXACT, Some(best_move));
            }
            writer_done.store(true, Ordering::Release);
        })
    };

    let mut hits = 0u32;
    while !writer_done.load(Ordering::Acquire) {
        if let Some(entry) = tt.get_entry(KEY) {
            assert_consistent(&entry);
            hits += 1;
        }
    }
    writer.join().expect("writer thread panicked");
    let final_entry = tt
        .get_entry(KEY)
        .expect("final entry must be visible after the writer finished");
    assert_consistent(&final_entry);
    assert_eq!(final_entry.score, (ROUNDS - 1) as i32);
    hits += 1;
    assert!(hits > 0, "concurrent probes should have observed hits");
}
