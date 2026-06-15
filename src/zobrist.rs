use std::sync::OnceLock;
use rand::Rng;
use rand::SeedableRng;

use crate::board::BoardState;

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();
fn zobrist() -> &'static ZobristKeys { ZOBRIST.get_or_init(ZobristKeys::new) }

struct ZobristKeys {
    pieces:   [[u64; 64]; 12],
    side:     u64,
    ep:       [u64; 64],
    castling: [u64; 4],
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345678);
        let mut pieces = [[0u64; 64]; 12];
        for pi in 0..12 { for sq in 0..64 { pieces[pi][sq] = rng.gen(); } }
        let mut ep = [0u64; 64];
        for i in 0..64 { ep[i] = rng.gen(); }
        let mut castling = [0u64; 4];
        for i in 0..4 { castling[i] = rng.gen(); }
        ZobristKeys { pieces, side: rng.gen(), ep, castling }
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
    if !st.w { key ^= z.side; }
    for i in 0..4 { if st.cr[i] { key ^= z.castling[i]; } }
    if let Some(ep_sq) = st.ep { key ^= z.ep[ep_sq]; }
    key
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