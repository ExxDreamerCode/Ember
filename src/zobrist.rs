use rand::Rng;
use rand::SeedableRng;
use std::sync::OnceLock;

use crate::board::{bit, is_attacked, piece_on, sq_c, BoardState, BK, BP, EMPTY_SQ, WK, WP};

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();
fn zobrist() -> &'static ZobristKeys {
    ZOBRIST.get_or_init(ZobristKeys::new)
}

struct ZobristKeys {
    pieces: [[u64; 64]; 12],
    side: u64,
    ep: [u64; 64],
    castling: [u64; 4],
    castling_rook: [[u64; 64]; 4],
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345678);
        let mut pieces = [[0u64; 64]; 12];
        for pi in 0..12 {
            for sq in 0..64 {
                pieces[pi][sq] = rng.gen();
            }
        }
        let mut ep = [0u64; 64];
        for i in 0..64 {
            ep[i] = rng.gen();
        }
        let mut castling = [0u64; 4];
        for i in 0..4 {
            castling[i] = rng.gen();
        }
        let mut castling_rook = [[0u64; 64]; 4];
        for i in 0..4 {
            for sq in 0..64 {
                castling_rook[i][sq] = rng.gen();
            }
        }
        ZobristKeys {
            pieces,
            side: rng.gen(),
            ep,
            castling,
            castling_rook,
        }
    }
}

pub fn compute_hash(st: &BoardState) -> u64 {
    let z = zobrist();
    let mut key = 0u64;
    for pi in 0..12usize {
        let mut bb = st.bb[pi];
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            key ^= z.pieces[pi][s];
            bb &= bb - 1;
        }
    }
    if !st.w {
        key ^= z.side;
    }
    for i in 0..4 {
        if st.cr[i] {
            key ^= z.castling[i];
            if st.chess960 {
                if let Some(rook_sq) = st.castling_rooks[i] {
                    key ^= z.castling_rook[i][rook_sq];
                }
            }
        }
    }
    if let Some(ep_sq) = st.ep.filter(|&ep_sq| ep_is_legal(st, ep_sq)) {
        key ^= z.ep[ep_sq];
    }
    key
}

fn ep_is_legal(st: &BoardState, ep_sq: usize) -> bool {
    if ep_sq >= 64 {
        return false;
    }

    let ep_row = ep_sq >> 3;
    if (st.w && ep_row != 2) || (!st.w && ep_row != 5) {
        return false;
    }

    let ep_col = sq_c(ep_sq);
    let (candidates, captured_sq, pawn_pi, captured_pi, king_pi) = if st.w {
        let mut candidates = [None; 2];
        if ep_col > 0 && ep_sq + 7 < 64 {
            candidates[0] = Some(ep_sq + 7);
        }
        if ep_col < 7 && ep_sq + 9 < 64 {
            candidates[1] = Some(ep_sq + 9);
        }
        (candidates, ep_sq + 8, WP, BP, WK)
    } else {
        let mut candidates = [None; 2];
        if ep_col > 0 && ep_sq >= 9 {
            candidates[0] = Some(ep_sq - 9);
        }
        if ep_col < 7 && ep_sq >= 7 {
            candidates[1] = Some(ep_sq - 7);
        }
        if ep_sq < 8 {
            return false;
        }
        (candidates, ep_sq - 8, BP, WP, BK)
    };

    if captured_sq >= 64 || piece_on(&st.bb, captured_sq) != captured_pi as u8 {
        return false;
    }
    if piece_on(&st.bb, ep_sq) != EMPTY_SQ {
        return false;
    }

    for from in candidates.into_iter().flatten() {
        if piece_on(&st.bb, from) != pawn_pi as u8 {
            continue;
        }
        let from_col = sq_c(from);
        if from_col.abs_diff(ep_col) != 1 {
            continue;
        }

        let mut bb = st.bb;
        bb[pawn_pi] &= !bit(from);
        bb[captured_pi] &= !bit(captured_sq);
        bb[pawn_pi] |= bit(ep_sq);

        let king_sq = if bb[king_pi] != 0 {
            bb[king_pi].trailing_zeros() as usize
        } else {
            return false;
        };
        if !is_attacked(&bb, king_sq, !st.w) {
            return true;
        }
    }

    false
}

pub fn compute_pawn_hash(st: &BoardState) -> u64 {
    let z = zobrist();
    let mut key = 0u64;
    for &pi in &[0usize, 6usize] {
        let mut bb = st.bb[pi];
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            key ^= z.pieces[pi][s];
            bb &= bb - 1;
        }
    }
    key
}
