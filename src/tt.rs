use crate::board::Move;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub const TT_EXACT: u8 = 0;
pub const TT_ALPHA: u8 = 1;
pub const TT_BETA: u8 = 2;

#[derive(Default)]
pub struct PackedEntry {
    key: AtomicU64,
    data: AtomicU64,
}

#[inline]
fn pack_data(depth: i32, score: i32, flag: u8, best_move: Option<Move>) -> u64 {
    let s = (score as u32 as u64) & 0xFFFF_FFFF;
    let d = (depth as i8 as u8 as u64) & 0xFF;
    let f = (flag as u64) & 0xFF;
    let m = (best_move.unwrap_or(0) as u64) & 0xFFFF;
    s | (d << 32) | (f << 40) | (m << 48)
}

#[inline]
fn unpack_data(data: u64) -> (i32, i32, u8, Option<Move>) {
    let score = data as u32 as i32;
    let depth = ((data >> 32) as u8) as i8 as i32;
    let flag = ((data >> 40) & 0xFF) as u8;
    let mv = ((data >> 48) & 0xFFFF) as Move;
    let best_move = if mv == 0 { None } else { Some(mv) };
    (depth, score, flag, best_move)
}

struct Inner {
    entries: Box<[PackedEntry]>,
    mask: usize,
}

pub struct SharedTT {
    inner: UnsafeCell<Inner>,
    resize_lock: Mutex<()>,
}

fn table_size_for_mb(mb: usize) -> usize {
    let entry_size = std::mem::size_of::<PackedEntry>();
    ((mb * 1024 * 1024 / entry_size).max(1)).next_power_of_two()
}

impl SharedTT {
    pub fn new(mb: usize) -> Self {
        let size = table_size_for_mb(mb);
        Self {
            inner: UnsafeCell::new(Inner {
                entries: (0..size).map(|_| PackedEntry::default()).collect(),
                mask: size - 1,
            }),
            resize_lock: Mutex::new(()),
        }
    }

    pub fn get_depth(&self, key: u64) -> Option<(i32, i32, u8, Option<Move>)> {
        let inner = unsafe { &*self.inner.get() };
        let idx = (key as usize) & inner.mask;
        let entry = &inner.entries[idx];

        let stored_key_xor = entry.key.load(Ordering::Relaxed);
        let data = entry.data.load(Ordering::Relaxed);

        if stored_key_xor ^ data != key {
            return None;
        }
        Some(unpack_data(data))
    }

    pub fn store(&self, key: u64, depth: i32, score: i32, flag: u8, best_move: Option<Move>) {
        let inner = unsafe { &*self.inner.get() };
        let idx = (key as usize) & inner.mask;
        let entry = &inner.entries[idx];

        let old_key_xor = entry.key.load(Ordering::Relaxed);
        let old_data = entry.data.load(Ordering::Relaxed);
        let old_key = old_key_xor ^ old_data;

        let replace = if old_key == key {
            let old_depth = ((old_data >> 32) as u8) as i8 as i32;
            old_depth <= depth || flag == TT_EXACT
        } else {
            true
        };

        if replace {
            let packed = pack_data(depth, score, flag, best_move);
            entry.data.store(packed, Ordering::Relaxed);
            entry.key.store(key ^ packed, Ordering::Relaxed);
        }
    }

    pub fn resize(&self, mb: usize) {
        let _lock = self.resize_lock.lock().unwrap();
        let inner = unsafe { &mut *self.inner.get() };
        let size = table_size_for_mb(mb);
        inner.entries = (0..size).map(|_| PackedEntry::default()).collect();
        inner.mask = size - 1;
    }
}

unsafe impl Send for SharedTT {}
unsafe impl Sync for SharedTT {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let cases = [
            (5i32, 150i32, TT_EXACT, Some(0xABCD)),
            (0i32, 0i32, TT_ALPHA, None),
            (-1i32, -30000i32, TT_BETA, Some(0x42)),
            (127i32, 32767i32, TT_EXACT, Some(0xFFFF)),
            (-128i32, -32768i32, TT_ALPHA, None),
            (10i32, 100_000i32, TT_EXACT, Some(0x1234)),
            (3i32, -100_000i32, TT_EXACT, Some(0x1234)),
            (5i32, 99_999i32, TT_EXACT, None),
            (5i32, -99_999i32, TT_EXACT, None),
        ];
        for &(depth, score, flag, best_move) in &cases {
            let packed = pack_data(depth, score, flag, best_move);
            let (d2, s2, f2, b2) = unpack_data(packed);
            assert_eq!(depth, d2, "depth mismatch");
            assert_eq!(score, s2, "score mismatch");
            assert_eq!(flag, f2, "flag mismatch");
            assert_eq!(best_move, b2, "best_move mismatch");
        }
    }

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
}
