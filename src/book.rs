use std::collections::HashMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::polyglot_randoms::POLYGLOT_RANDOMS;
use crate::board::{BoardState, EMPTY, ptype, is_white};

struct BookMove {
    raw_move: u16,
    weight: u16,
}

pub struct OpeningBook {
    entries: HashMap<u64, Vec<BookMove>>,
}

impl OpeningBook {
    pub fn load(path: &str) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        if data.len() % 16 != 0 {
            return Err(format!("book size {} not multiple of 16", data.len()));
        }

        let mut entries: HashMap<u64, Vec<BookMove>> = HashMap::new();
        let num = data.len() / 16;

        for i in 0..num {
            let off = i * 16;
            let key = u64::from_be_bytes(data[off..off+8].try_into().unwrap());
            let mv = u16::from_be_bytes(data[off+8..off+10].try_into().unwrap());
            let weight = u16::from_be_bytes(data[off+10..off+12].try_into().unwrap());

            if weight == 0 { continue; }

            entries.entry(key).or_default().push(BookMove { raw_move: mv, weight });
        }

        for moves in entries.values_mut() {
            moves.sort_by(|a, b| b.weight.cmp(&a.weight));
        }

        let count = entries.len();
        eprintln!("info string Book loaded: {} positions from {}", count, path);

        Ok(OpeningBook { entries })
    }

    pub fn pick_move(&self, st: &BoardState, moves: &[[usize; 4]]) -> Option<[usize; 4]> {
        let hash = polyglot_hash(st);
        let bmoves = self.entries.get(&hash)?;
        if bmoves.is_empty() { return None; }

        let mut candidates: Vec<([usize; 4], u32)> = Vec::new();
        let mut total_weight = 0u32;

        for bm in bmoves {
            if let Some(m) = match_polyglot_move(bm.raw_move, moves, st) {
                let w = bm.weight as u32;
                candidates.push((m, w));
                total_weight += w;
            }
        }

        if candidates.is_empty() { return None; }

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let r = nanos % total_weight;
        let mut cumulative = 0;
        for (m, w) in &candidates {
            cumulative += w;
            if r < cumulative {
                return Some(*m);
            }
        }

        Some(candidates[0].0)
    }
}

fn match_polyglot_move(pm: u16, legal: &[[usize; 4]], st: &BoardState) -> Option<[usize; 4]> {
    let to_file = (pm & 7) as usize;
    let to_rank = ((pm >> 3) & 7) as usize;
    let from_file = ((pm >> 6) & 7) as usize;
    let from_rank = ((pm >> 9) & 7) as usize;
    let promo = ((pm >> 12) & 7) as u8;

    let from_r = 7 - from_rank;
    let from_c = from_file;
    let mut to_r = 7 - to_rank;
    let mut to_c = to_file;

    if from_file == 4 {
        if from_rank == 0 {
            if to_file == 7 { to_r = 7; to_c = 6; }
            else if to_file == 0 { to_r = 7; to_c = 2; }
        } else if from_rank == 7 {
            if to_file == 7 { to_r = 0; to_c = 6; }
            else if to_file == 0 { to_r = 0; to_c = 2; }
        }
    }

    for &mv in legal {
        if mv[0] == from_r && mv[1] == from_c && mv[2] == to_r && mv[3] == to_c {
            let p = st.b[mv[0]][mv[1]];
            if ptype(p) == b'p' && (mv[2] == 0 || mv[2] == 7) {
                let pp = promo;
                if pp == 0 || pp == 4 {
                    return Some(mv);
                }
            } else {
                return Some(mv);
            }
        }
    }

    None
}

const CASTLE_OFFSET: usize = 768;
const EP_OFFSET: usize = 772;
const TURN_OFFSET: usize = 780;

fn polyglot_hash(st: &BoardState) -> u64 {
    let mut hash = 0u64;

    for r in 0..8 {
        for c in 0..8 {
            let p = st.b[r][c];
            if p == EMPTY { continue; }
            let sq = (7 - r) * 8 + c;
            let idx = polyglot_piece_index(is_white(p), ptype(p));
            hash ^= POLYGLOT_RANDOMS[idx * 64 + sq];
        }
    }

    if st.cr[0] { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET]; }
    if st.cr[1] { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 1]; }
    if st.cr[2] { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 2]; }
    if st.cr[3] { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 3]; }

    if let Some((ep_r, ep_c)) = st.ep {
        if polyglot_has_ep_capture(st, ep_r, ep_c) {
            hash ^= POLYGLOT_RANDOMS[EP_OFFSET + ep_c];
        }
    }

    if st.w {
        hash ^= POLYGLOT_RANDOMS[TURN_OFFSET];
    }

    hash
}

fn polyglot_piece_index(w: bool, pt: u8) -> usize {
    let base = match pt {
        b'p' => 0,
        b'n' => 1,
        b'b' => 2,
        b'r' => 3,
        b'q' => 4,
        b'k' => 5,
        _ => 0,
    } * 2;
    if w { base + 1 } else { base }
}

fn polyglot_has_ep_capture(st: &BoardState, ep_r: usize, ep_c: usize) -> bool {
    let opp_r = if st.w { ep_r + 1 } else { ep_r.wrapping_sub(1) };
    if opp_r >= 8 { return false; }

    if ep_c > 0 {
        let p = st.b[opp_r][ep_c - 1];
        if p != EMPTY && is_white(p) == st.w && ptype(p) == b'p' {
            return true;
        }
    }
    if ep_c < 7 {
        let p = st.b[opp_r][ep_c + 1];
        if p != EMPTY && is_white(p) == st.w && ptype(p) == b'p' {
            return true;
        }
    }
    false
}