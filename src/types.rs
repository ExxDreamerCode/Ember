use crate::board::BoardState;

pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;
pub type Color = u8;

#[inline(always)]
pub fn flip_color(c: Color) -> Color {
    c ^ 1
}

pub const PAWN: u8 = 0;
pub const KNIGHT: u8 = 1;
pub const BISHOP: u8 = 2;
pub const ROOK: u8 = 3;
pub const QUEEN: u8 = 4;
pub const KING: u8 = 5;
pub const NO_PIECE_TYPE: u8 = 6;

pub const NO_PIECE: u8 = 12;

#[inline(always)]
pub fn make_piece(color: Color, pt: u8) -> u8 {
    color * 6 + pt
}

#[inline(always)]
pub fn piece_color(p: u8) -> Color {
    p / 6
}

#[inline(always)]
pub fn piece_type(p: u8) -> u8 {
    p % 6
}

pub type Square = u8;
pub const NO_SQUARE: Square = 64;

#[inline(always)]
pub fn square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline(always)]
pub fn file_of(sq: Square) -> u8 {
    sq & 7
}

#[inline(always)]
pub fn rank_of(sq: Square) -> u8 {
    sq >> 3
}

#[inline(always)]
pub fn square_bb(sq: Square) -> u64 {
    1u64 << sq
}

pub fn square_name(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{}{}", f, r)
}

#[inline(always)]
pub fn piece_type_at(st: &BoardState, sq: Square) -> u8 {
    let s = sq as usize;
    let b = 1u64 << s;
    for i in 0..6 {
        if st.bb[i] & b != 0 {
            return i as u8;
        }
    }
    for i in 6..12 {
        if st.bb[i] & b != 0 {
            return (i - 6) as u8;
        }
    }
    NO_PIECE_TYPE
}

#[inline(always)]
pub fn piece_at(st: &BoardState, sq: Square) -> u8 {
    let s = sq as usize;
    let b = 1u64 << s;
    for i in 0..12 {
        if st.bb[i] & b != 0 {
            return i as u8;
        }
    }
    NO_PIECE
}

pub fn make_mailbox(st: &BoardState) -> [u8; 64] {
    let mut mb = [NO_PIECE; 64];
    for (sq, cell) in mb.iter_mut().enumerate() {
        *cell = piece_at(st, sq as u8);
    }
    mb
}

#[inline(always)]
pub fn color_bb(bbs: &[u64; 12], color: Color) -> u64 {
    if color == WHITE {
        bbs[0] | bbs[1] | bbs[2] | bbs[3] | bbs[4] | bbs[5]
    } else {
        bbs[6] | bbs[7] | bbs[8] | bbs[9] | bbs[10] | bbs[11]
    }
}

#[inline(always)]
pub fn piece_count(st: &BoardState) -> u32 {
    (0..12).map(|i| st.bb[i].count_ones()).sum()
}

#[inline(always)]
pub fn occupancy(bbs: &[u64; 12]) -> u64 {
    (0..12).map(|i| bbs[i]).fold(0, |a, b| a | b)
}

pub type CodaMove = u16;
pub const NO_MOVE: CodaMove = 0;

#[inline(always)]
pub fn ember_move_sq(mv: &[usize; 4]) -> usize {
    mv[2] * 8 + (mv[3] & 7)
}

#[inline(always)]
pub fn ember_move_from(mv: &[usize; 4]) -> usize {
    mv[0] * 8 + mv[1]
}

#[inline(always)]
pub fn ember_move_pt(mv: &[usize; 4], st: &BoardState) -> u8 {
    let s = ember_move_from(mv);
    if s < 64 {
        let b = 1u64 << s;
        for i in 0..12 {
            if st.bb[i] & b != 0 {
                return (i % 6) as u8;
            }
        }
    }
    NO_PIECE_TYPE
}
