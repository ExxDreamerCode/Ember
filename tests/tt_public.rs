use ember_chess::tt::{SharedTT, TT_ALPHA, TT_BETA, TT_EXACT};

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
