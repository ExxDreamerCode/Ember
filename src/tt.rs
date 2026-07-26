use crate::board::Move;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

pub const TT_EXACT: u8 = 0;
pub const TT_ALPHA: u8 = 1;
pub const TT_BETA: u8 = 2;
const TT_BOUND_MASK: u8 = 0b0000_0011;
const TT_PV_MASK: u8 = 0b0000_0100;
const TT_GENERATION_SHIFT: u8 = 3;
const TT_GENERATION_MASK: u8 = 0b0001_1111;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TTEntry {
    pub depth: i32,
    pub score: i32,
    pub flag: u8,
    pub best_move: Option<Move>,
    pub pv: bool,
    pub generation: u8,
}

#[derive(Default)]
pub struct PackedEntry {
    key: AtomicU64,
    data: AtomicU64,
}

#[inline]
fn pack_data(
    depth: i32,
    score: i32,
    flag: u8,
    best_move: Option<Move>,
    pv: bool,
    generation: u8,
) -> u64 {
    let s = (score as u32 as u64) & 0xFFFF_FFFF;
    let d = (depth as i8 as u8 as u64) & 0xFF;
    let metadata = (flag & TT_BOUND_MASK)
        | if pv { TT_PV_MASK } else { 0 }
        | ((generation & TT_GENERATION_MASK) << TT_GENERATION_SHIFT);
    let f = u64::from(metadata);
    let m = (best_move.unwrap_or(0) as u64) & 0xFFFF;
    s | (d << 32) | (f << 40) | (m << 48)
}

#[inline]
fn unpack_data(data: u64) -> TTEntry {
    let score = data as u32 as i32;
    let depth = ((data >> 32) as u8) as i8 as i32;
    let metadata = ((data >> 40) & 0xFF) as u8;
    let mv = ((data >> 48) & 0xFFFF) as Move;
    let best_move = if mv == 0 { None } else { Some(mv) };
    TTEntry {
        depth,
        score,
        flag: metadata & TT_BOUND_MASK,
        best_move,
        pv: metadata & TT_PV_MASK != 0,
        generation: (metadata >> TT_GENERATION_SHIFT) & TT_GENERATION_MASK,
    }
}

struct Inner {
    entries: Box<[PackedEntry]>,
    mask: usize,
}

pub struct SharedTT {
    inner: UnsafeCell<Inner>,
    resize_lock: Mutex<()>,
    generation: AtomicU8,
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
            generation: AtomicU8::new(0),
        }
    }

    pub fn get_entry(&self, key: u64) -> Option<TTEntry> {
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

    pub fn get_depth(&self, key: u64) -> Option<(i32, i32, u8, Option<Move>)> {
        self.get_entry(key)
            .map(|entry| (entry.depth, entry.score, entry.flag, entry.best_move))
    }

    pub fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed) & TT_GENERATION_MASK
    }

    pub fn advance_generation(&self) -> u8 {
        let previous = self
            .generation
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                Some(generation.wrapping_add(1) & TT_GENERATION_MASK)
            })
            .unwrap_or_else(|generation| generation);
        previous.wrapping_add(1) & TT_GENERATION_MASK
    }

    pub fn age(&self, entry: TTEntry) -> u8 {
        self.generation().wrapping_sub(entry.generation) & TT_GENERATION_MASK
    }

    pub fn store(&self, key: u64, depth: i32, score: i32, flag: u8, best_move: Option<Move>) {
        self.store_with_pv(key, depth, score, flag, best_move, false);
    }

    pub fn store_with_pv(
        &self,
        key: u64,
        depth: i32,
        score: i32,
        flag: u8,
        best_move: Option<Move>,
        pv: bool,
    ) {
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
            let packed = pack_data(depth, score, flag, best_move, pv, self.generation());
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
            for &(pv, generation) in &[(false, 0), (true, 17), (false, 31)] {
                let packed = pack_data(depth, score, flag, best_move, pv, generation);
                let entry = unpack_data(packed);
                assert_eq!(depth, entry.depth, "depth mismatch");
                assert_eq!(score, entry.score, "score mismatch");
                assert_eq!(flag, entry.flag, "flag mismatch");
                assert_eq!(best_move, entry.best_move, "best_move mismatch");
                assert_eq!(pv, entry.pv, "PV provenance mismatch");
                assert_eq!(generation, entry.generation, "generation mismatch");
            }
        }
    }

    #[test]
    fn table_records_pv_provenance_and_wrapping_age() {
        let table = SharedTT::new(1);
        assert_eq!(table.advance_generation(), 1);
        table.store_with_pv(0x1234, 12, 55, TT_BETA, Some(0x42), true);

        let entry = table.get_entry(0x1234).expect("stored entry");
        assert!(entry.pv);
        assert_eq!(entry.generation, 1);
        assert_eq!(table.age(entry), 0);

        assert_eq!(table.advance_generation(), 2);
        assert_eq!(table.age(entry), 1);
        for _ in 0..31 {
            table.advance_generation();
        }
        assert_eq!(table.generation(), 1);
        assert_eq!(table.age(entry), 0);
    }
}
