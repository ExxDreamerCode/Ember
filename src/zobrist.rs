use std::sync::OnceLock;
use rand::Rng;
use rand::SeedableRng;

use crate::board::{Board8, EMPTY};
use crate::board::ptype;

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();
fn zobrist() -> &'static ZobristKeys { ZOBRIST.get_or_init(ZobristKeys::new) }

struct ZobristKeys { pieces: [[[u64; 8]; 8]; 12], side: u64, ep: [[u64; 8]; 8], castling: [u64; 4] }
impl ZobristKeys {
    fn new() -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345678);
        let mut pieces = [[[0u64; 8]; 8]; 12];
        for idx in 0..12 { for r in 0..8 { for c in 0..8 { pieces[idx][r][c] = rng.gen(); } } }
        let mut ep = [[0u64; 8]; 8];
        for r in 0..8 { for c in 0..8 { ep[r][c] = rng.gen(); } }
        let mut castling = [0u64; 4];
        for i in 0..4 { castling[i] = rng.gen(); }
        ZobristKeys { pieces, side: rng.gen(), ep, castling }
    }
}

fn piece_idx(p: u8) -> usize {
    match p { b'P'=>0,b'N'=>1,b'B'=>2,b'R'=>3,b'Q'=>4,b'K'=>5,
              b'p'=>6,b'n'=>7,b'b'=>8,b'r'=>9,b'q'=>10,b'k'=>11,_=>0 }
}

pub fn compute_hash(b: &Board8, wturn: bool, cr: &[bool; 4], ep: Option<(usize, usize)>) -> u64 {
    let z = zobrist();
    let mut key = 0u64;
    for r in 0..8 { for c in 0..8 { let p = b[r][c]; if p != EMPTY { key ^= z.pieces[piece_idx(p)][r][c]; } } }
    if !wturn { key ^= z.side; }
    for i in 0..4 { if cr[i] { key ^= z.castling[i]; } }
    if let Some((er, ec)) = ep { key ^= z.ep[er][ec]; }
    key
}

pub fn compute_pawn_hash(b: &Board8) -> u64 {
    let z = zobrist();
    let mut key = 0u64;
    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p != EMPTY && ptype(p) == b'p' {
            key ^= z.pieces[piece_idx(p)][r][c];
        }
    }}
    key
}