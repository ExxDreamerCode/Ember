pub type Bitboard = u64;

pub const NOT_FILE_A: u64 = 0xfefefefefefefefe;
pub const NOT_FILE_H: u64 = 0x7f7f7f7f7f7f7f7f;

#[inline(always)]
pub fn pop_lsb(bb: &mut u64) -> u32 {
    let sq = bb.trailing_zeros();
    *bb &= *bb - 1;
    sq
}

#[inline(always)]
pub fn popcount(bb: u64) -> u32 {
    bb.count_ones()
}

pub static PAWN_ATTACKS_WHITE: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let b = 1u64 << sq;
        t[sq] = ((b & NOT_FILE_A) << 7) | ((b & NOT_FILE_H) << 9);
        sq += 1;
    }
    t
};

pub static PAWN_ATTACKS_BLACK: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let b = 1u64 << sq;
        t[sq] = ((b & NOT_FILE_H) >> 7) | ((b & NOT_FILE_A) >> 9);
        sq += 1;
    }
    t
};

pub use crate::board::KING_ATTACKS;
pub use crate::board::KNIGHT_ATTACKS;
pub use crate::magic::{bishop_attacks, rook_attacks};
