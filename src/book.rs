use std::collections::HashMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::board::{
    is_white_piece, move_ec, move_promotion, piece_on, piece_type, sq, sq_c, sq_r, BoardState,
    Move, BP, EMPTY_SQ, WP,
};
use crate::polyglot_randoms::POLYGLOT_RANDOMS;

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
        Self::load_from_bytes(&data, path)
    }

    pub fn load_from_bytes(data: &[u8], name: &str) -> Result<Self, String> {
        if !data.len().is_multiple_of(16) {
            return Err(format!("book size {} not multiple of 16", data.len()));
        }
        let mut entries: HashMap<u64, Vec<BookMove>> = HashMap::new();
        let num = data.len() / 16;
        for i in 0..num {
            let off = i * 16;
            let key = u64::from_be_bytes(data[off..off + 8].try_into().unwrap());
            let mv = u16::from_be_bytes(data[off + 8..off + 10].try_into().unwrap());
            let weight = u16::from_be_bytes(data[off + 10..off + 12].try_into().unwrap());
            if weight == 0 {
                continue;
            }
            entries.entry(key).or_default().push(BookMove {
                raw_move: mv,
                weight,
            });
        }
        for moves in entries.values_mut() {
            moves.sort_by_key(|b| std::cmp::Reverse(b.weight));
        }
        let count = entries.len();
        eprintln!("info string Book loaded: {} positions from {}", count, name);
        Ok(OpeningBook { entries })
    }

    pub fn pick_move(&self, st: &BoardState, moves: &[Move]) -> Option<Move> {
        let hash = polyglot_hash(st);
        let bmoves = self.entries.get(&hash)?;
        if bmoves.is_empty() {
            return None;
        }

        let mut candidates: Vec<(Move, u32)> = Vec::new();
        let mut total_weight = 0u32;

        for bm in bmoves {
            if let Some(m) = match_polyglot_move(bm.raw_move, moves, st) {
                let w = bm.weight as u32;
                candidates.push((m, w));
                total_weight += w;
            }
        }
        if candidates.is_empty() {
            return None;
        }

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

fn match_polyglot_move(pm: u16, legal: &[Move], st: &BoardState) -> Option<Move> {
    let to_file = (pm & 7) as usize;
    let to_rank = ((pm >> 3) & 7) as usize;
    let from_file = ((pm >> 6) & 7) as usize;
    let from_rank = ((pm >> 9) & 7) as usize;
    let promo = polyglot_promotion((pm >> 12) & 7);

    let from_r = 7 - from_rank;
    let from_c = from_file;
    let mut to_r = 7 - to_rank;
    let mut to_c = to_file;

    if from_file == 4 {
        if from_rank == 0 {
            if to_file == 7 {
                to_r = 7;
                to_c = 6;
            } else if to_file == 0 {
                to_r = 7;
                to_c = 2;
            }
        } else if from_rank == 7 {
            if to_file == 7 {
                to_r = 0;
                to_c = 6;
            } else if to_file == 0 {
                to_r = 0;
                to_c = 2;
            }
        }
    }

    for &mv in legal {
        if mv[0] == from_r && mv[1] == from_c && mv[2] == to_r && move_ec(&mv) == to_c {
            let from_s = sq(mv[0], mv[1]);
            let pi = piece_on(&st.bb, from_s);
            if pi != EMPTY_SQ && piece_type(pi) == 0 && (mv[2] == 0 || mv[2] == 7) {
                let move_promo = move_promotion(&mv).to_ascii_uppercase();
                if (promo == 0 && move_promo == b'Q') || promo == move_promo {
                    return Some(mv);
                }
            } else {
                return Some(mv);
            }
        }
    }
    None
}

fn polyglot_promotion(promo: u16) -> u8 {
    match promo {
        1 => b'N',
        2 => b'B',
        3 => b'R',
        4 => b'Q',
        _ => 0,
    }
}

const CASTLE_OFFSET: usize = 768;
const EP_OFFSET: usize = 772;
const TURN_OFFSET: usize = 780;

fn polyglot_hash(st: &BoardState) -> u64 {
    let mut hash = 0u64;

    for pi in 0..12usize {
        let white = pi < 6;
        let pt = pi % 6;
        let idx = polyglot_piece_index(white, pt);
        let mut bb = st.bb[pi];
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            let sq_pg = (7 - sq_r(s)) * 8 + sq_c(s);
            hash ^= POLYGLOT_RANDOMS[idx * 64 + sq_pg];
            bb &= bb - 1;
        }
    }

    if st.cr[0] {
        hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET];
    }
    if st.cr[1] {
        hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 1];
    }
    if st.cr[2] {
        hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 2];
    }
    if st.cr[3] {
        hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 3];
    }

    if let Some(ep_s) = st.ep {
        if polyglot_has_ep_capture(st, ep_s) {
            hash ^= POLYGLOT_RANDOMS[EP_OFFSET + sq_c(ep_s)];
        }
    }

    if st.w {
        hash ^= POLYGLOT_RANDOMS[TURN_OFFSET];
    }

    hash
}

fn polyglot_piece_index(white: bool, pt: usize) -> usize {
    let base = pt * 2;
    if white {
        base + 1
    } else {
        base
    }
}

fn polyglot_has_ep_capture(st: &BoardState, ep_s: usize) -> bool {
    let ep_r = sq_r(ep_s);
    let ep_c = sq_c(ep_s);
    let opp_r = if st.w { ep_r + 1 } else { ep_r.wrapping_sub(1) };
    if opp_r >= 8 {
        return false;
    }
    let _pawn_pi = if st.w { WP } else { BP };
    if ep_c > 0 {
        let s = sq(opp_r, ep_c - 1);
        let pi = piece_on(&st.bb, s);
        if pi != EMPTY_SQ && is_white_piece(pi) == st.w && piece_type(pi) == 0 {
            return true;
        }
    }
    if ep_c < 7 {
        let s = sq(opp_r, ep_c + 1);
        let pi = piece_on(&st.bb, s);
        if pi != EMPTY_SQ && is_white_piece(pi) == st.w && piece_type(pi) == 0 {
            return true;
        }
    }
    false
}
