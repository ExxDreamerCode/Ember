use crate::board::Move;
use std::sync::RwLock;

#[derive(Clone)]
pub struct TTEntry {
    pub key: u64,
    pub depth: i32,
    pub score: i32,
    pub flag: u8,
    pub best_move: Option<Move>,
}
pub const TT_EXACT: u8 = 0;
pub const TT_ALPHA: u8 = 1;
pub const TT_BETA: u8 = 2;

pub struct TT {
    pub entries: Vec<Option<TTEntry>>,
    pub mask: usize,
}

impl TT {
    pub fn new(mb: usize) -> Self {
        let size = (mb * 1024 * 1024 / 40).next_power_of_two();
        TT {
            entries: vec![None; size],
            mask: size - 1,
        }
    }
    pub fn idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }
    pub fn store(&mut self, key: u64, depth: i32, score: i32, flag: u8, best_move: Option<Move>) {
        let idx = self.idx(key);
        let replace = match &self.entries[idx] {
            None => true,
            Some(e) => e.key != key || e.depth <= depth || flag == TT_EXACT,
        };
        if replace {
            self.entries[idx] = Some(TTEntry {
                key,
                depth,
                score,
                flag,
                best_move,
            });
        }
    }
    pub fn get(&self, key: u64) -> Option<&TTEntry> {
        let idx = self.idx(key);
        self.entries[idx]
            .as_ref()
            .and_then(|e| if e.key == key { Some(e) } else { None })
    }
    pub fn resize(&mut self, mb: usize) {
        let size = (mb * 1024 * 1024 / 40).next_power_of_two();
        self.entries = vec![None; size];
        self.mask = size - 1;
    }
}

pub struct SharedTT {
    inner: RwLock<TT>,
}

impl SharedTT {
    pub fn new(mb: usize) -> Self {
        SharedTT {
            inner: RwLock::new(TT::new(mb)),
        }
    }

    pub fn get_depth(&self, key: u64) -> Option<(i32, i32, u8, Option<Move>)> {
        let tt = self.inner.read().unwrap();
        tt.get(key).map(|e| (e.depth, e.score, e.flag, e.best_move))
    }

    pub fn store(&self, key: u64, depth: i32, score: i32, flag: u8, best_move: Option<Move>) {
        let mut tt = self.inner.write().unwrap();
        tt.store(key, depth, score, flag, best_move);
    }

    pub fn resize(&self, mb: usize) {
        let mut tt = self.inner.write().unwrap();
        tt.resize(mb);
    }
}