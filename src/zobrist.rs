use rand::Rng;
use rand::SeedableRng;
use std::sync::OnceLock;

use crate::board::BoardState;

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();
pub fn zobrist() -> &'static ZobristKeys {
    ZOBRIST.get_or_init(ZobristKeys::new)
}

pub struct ZobristKeys {
    pub pieces: [[u64; 64]; 12],
    pub side: u64,
    pub ep: [u64; 64],
    pub castling: [u64; 4],
    pub castling_rook: [[u64; 64]; 4],
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345678);
        let mut pieces = [[0u64; 64]; 12];
        for piece in &mut pieces {
            for square in piece {
                *square = rng.gen();
            }
        }
        let mut ep = [0u64; 64];
        for square in &mut ep {
            *square = rng.gen();
        }
        let mut castling = [0u64; 4];
        for right in &mut castling {
            *right = rng.gen();
        }
        let mut castling_rook = [[0u64; 64]; 4];
        for right in &mut castling_rook {
            for square in right {
                *square = rng.gen();
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
    if let Some(ep_sq) = st.ep {
        key ^= z.ep[ep_sq];
    }
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
