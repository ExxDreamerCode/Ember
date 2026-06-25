pub const EMPTY_SQ: u8 = 255;
pub const MATE: i32 = 100_000;
pub const INF: i32 = 1_000_000;
pub const MAX_PLY: usize = 128;
pub const QS_DEPTH: i32 = 0;

pub const WP: usize = 0;
pub const WN: usize = 1;
pub const WB: usize = 2;
pub const WR: usize = 3;
pub const WQ: usize = 4;
pub const WK: usize = 5;
pub const BP: usize = 6;
pub const BN: usize = 7;
pub const BB: usize = 8;
pub const BR: usize = 9;
pub const BQ: usize = 10;
pub const BK: usize = 11;

pub type Move = [usize; 4];

#[inline(always)]
pub fn sq(r: usize, c: usize) -> usize {
    r * 8 + c
}
#[inline(always)]
pub fn sq_r(s: usize) -> usize {
    s >> 3
}
#[inline(always)]
pub fn sq_c(s: usize) -> usize {
    s & 7
}
#[inline(always)]
pub fn bit(s: usize) -> u64 {
    1u64 << s
}

#[inline(always)]
pub fn encode_move(sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) -> Move {
    [sr, sc, er, ec | ((promotion as usize) << 3)]
}

#[inline(always)]
pub fn move_ec(mv: &Move) -> usize {
    mv[3] & 7
}
#[inline(always)]
pub fn move_promotion(mv: &Move) -> u8 {
    (mv[3] >> 3) as u8
}

#[inline(always)]
pub fn promotion_piece_index(white: bool, promotion: u8) -> Option<usize> {
    let pt = match promotion.to_ascii_uppercase() {
        b'N' => 1,
        b'B' => 2,
        b'R' => 3,
        b'Q' => 4,
        _ => return None,
    };
    Some(if white { pt } else { pt + 6 })
}

#[inline(always)]
pub fn piece_on(bbs: &[u64; 12], s: usize) -> u8 {
    let b = bit(s);
    for (i, bb) in bbs.iter().enumerate().take(12) {
        if *bb & b != 0 {
            return i as u8;
        }
    }
    EMPTY_SQ
}

#[inline(always)]
pub fn is_white_piece(pi: u8) -> bool {
    pi < 6
}
#[inline(always)]
pub fn piece_type(pi: u8) -> u8 {
    if pi >= 6 {
        pi - 6
    } else {
        pi
    }
}
pub fn piece_char(pi: u8) -> u8 {
    match pi {
        0 => b'P',
        1 => b'N',
        2 => b'B',
        3 => b'R',
        4 => b'Q',
        5 => b'K',
        6 => b'p',
        7 => b'n',
        8 => b'b',
        9 => b'r',
        10 => b'q',
        11 => b'k',
        _ => b'.',
    }
}
pub fn piece_from_char(ch: u8) -> u8 {
    match ch {
        b'P' => 0,
        b'N' => 1,
        b'B' => 2,
        b'R' => 3,
        b'Q' => 4,
        b'K' => 5,
        b'p' => 6,
        b'n' => 7,
        b'b' => 8,
        b'r' => 9,
        b'q' => 10,
        b'k' => 11,
        _ => EMPTY_SQ,
    }
}

#[inline(always)]
pub fn white_occ(bbs: &[u64; 12]) -> u64 {
    bbs[0] | bbs[1] | bbs[2] | bbs[3] | bbs[4] | bbs[5]
}
#[inline(always)]
pub fn black_occ(bbs: &[u64; 12]) -> u64 {
    bbs[6] | bbs[7] | bbs[8] | bbs[9] | bbs[10] | bbs[11]
}
#[inline(always)]
pub fn all_occ(bbs: &[u64; 12]) -> u64 {
    white_occ(bbs) | black_occ(bbs)
}

#[derive(Clone, Copy)]
pub struct BoardState {
    pub bb: [u64; 12],
    pub w: bool,
    pub cr: [bool; 4],
    pub castling_rooks: [Option<usize>; 4],
    pub ep: Option<usize>,
    pub mc: usize,
    pub chess960: bool,
}

impl BoardState {
    pub fn empty() -> Self {
        BoardState {
            bb: [0u64; 12],
            w: true,
            cr: [false; 4],
            castling_rooks: [None; 4],
            ep: None,
            mc: 0,
            chess960: false,
        }
    }

    #[inline(always)]
    pub fn piece_at(&self, s: usize) -> u8 {
        piece_on(&self.bb, s)
    }

    #[inline(always)]
    pub fn king_sq(&self, white: bool) -> usize {
        let k = if white { self.bb[WK] } else { self.bb[BK] };
        if k == 0 {
            return 0;
        }
        k.trailing_zeros() as usize
    }
}

pub fn coord_to_square(r: usize, c: usize) -> String {
    format!("{}{}", (b'a' + c as u8) as char, 8 - r as u8)
}
pub fn sq_to_str(s: usize) -> String {
    coord_to_square(sq_r(s), sq_c(s))
}

pub fn move_to_uci(st: &BoardState, mv: &Move) -> String {
    let from = sq(mv[0], mv[1]);
    let to = sq(mv[2], move_ec(mv));
    let promo = move_promotion(mv);
    let pi = piece_on(&st.bb, from);

    if st.chess960
        && pi != EMPTY_SQ && piece_type(pi) == 5 {
            let target_pi = piece_on(&st.bb, to);
            if target_pi != EMPTY_SQ
                && piece_type(target_pi) == 3
                && is_white_piece(target_pi) == is_white_piece(pi)
            {
                let king_dst_col = if move_ec(mv) > mv[1] { 6usize } else { 2usize };
                if king_dst_col != mv[1] {
                    return format!("{}{}", sq_to_str(from), sq_to_str(to));
                }
            }
    }

    if promo != 0 {
        let mut out = format!("{}{}", sq_to_str(from), sq_to_str(to));
        out.push(promo.to_ascii_lowercase() as char);
        return out;
    }

    format!("{}{}", sq_to_str(from), sq_to_str(to))
}

pub fn board_to_fen(st: &BoardState) -> String {
    let mut board = String::new();
    for r in 0..8 {
        if r > 0 {
            board.push('/');
        }
        let mut empty = 0usize;
        for c in 0..8 {
            let pi = piece_on(&st.bb, sq(r, c));
            if pi == EMPTY_SQ {
                empty += 1;
            } else {
                if empty > 0 {
                    board.push(char::from_digit(empty as u32, 10).unwrap());
                    empty = 0;
                }
                board.push(piece_char(pi) as char);
            }
        }
        if empty > 0 {
            board.push(char::from_digit(empty as u32, 10).unwrap());
        }
    }

    let side = if st.w { "w" } else { "b" };
    let mut castling = String::new();
    if st.chess960 {
        for cr_idx in 0..4 {
            if st.cr[cr_idx] {
                if let Some(rook_sq) = st.castling_rooks[cr_idx] {
                    let rook_col = sq_c(rook_sq);
                    let ch = if cr_idx < 2 {
                        (b'A' + rook_col as u8) as char
                    } else {
                        (b'a' + rook_col as u8) as char
                    };
                    castling.push(ch);
                } else {
                    castling.push(match cr_idx {
                        0 => 'K',
                        1 => 'Q',
                        2 => 'k',
                        _ => 'q',
                    });
                }
            }
        }
    } else {
        if st.cr[0] {
            castling.push('K');
        }
        if st.cr[1] {
            castling.push('Q');
        }
        if st.cr[2] {
            castling.push('k');
        }
        if st.cr[3] {
            castling.push('q');
        }
    }
    if castling.is_empty() {
        castling.push('-');
    }
    let ep = st.ep.map(sq_to_str).unwrap_or_else(|| "-".to_string());
    let fullmove = st.mc / 2 + 1;
    format!("{} {} {} {} 0 {}", board, side, castling, ep, fullmove)
}

pub fn see_val(pt: u8) -> i32 {
    match pt {
        0 => 100,
        1 => 325,
        2 => 340,
        3 => 500,
        4 => 950,
        5 => 20000,
        _ => 0,
    }
}

pub fn has_non_pawn(bb: &[u64; 12], white: bool) -> bool {
    if white {
        bb[WN] | bb[WB] | bb[WR] | bb[WQ] != 0
    } else {
        bb[BN] | bb[BB] | bb[BR] | bb[BQ] != 0
    }
}

use crate::magic::{bishop_attacks, rook_attacks};

pub fn attacked_by(bb: &[u64; 12], occ: u64, white: bool) -> u64 {
    let (p, n, b, r, q, k) = if white {
        (bb[WP], bb[WN], bb[WB], bb[WR], bb[WQ], bb[WK])
    } else {
        (bb[BP], bb[BN], bb[BB], bb[BR], bb[BQ], bb[BK])
    };
    let mut att = 0u64;
    if white {
        att |= (p & !0x8080808080808080) >> 7 | (p & !0x0101010101010101) >> 9;
    } else {
        att |= (p & !0x0101010101010101) << 7 | (p & !0x8080808080808080) << 9;
    }
    let mut tmp = n;
    while tmp != 0 {
        let s = tmp.trailing_zeros() as usize;
        att |= KNIGHT_ATTACKS[s];
        tmp &= tmp - 1;
    }
    let mut tmp = b | q;
    while tmp != 0 {
        let s = tmp.trailing_zeros() as usize;
        att |= bishop_attacks(s, occ);
        tmp &= tmp - 1;
    }
    let mut tmp = r | q;
    while tmp != 0 {
        let s = tmp.trailing_zeros() as usize;
        att |= rook_attacks(s, occ);
        tmp &= tmp - 1;
    }
    let mut tmp = k;
    while tmp != 0 {
        let s = tmp.trailing_zeros() as usize;
        att |= KING_ATTACKS[s];
        tmp &= tmp - 1;
    }
    att
}

#[inline(always)]
pub fn is_attacked(bb: &[u64; 12], s: usize, by_white: bool) -> bool {
    if s >= 64 {
        return false;
    }
    let occ = all_occ(bb);
    let bit_s = bit(s);
    let (p, n, b, r, q, k) = if by_white {
        (bb[WP], bb[WN], bb[WB], bb[WR], bb[WQ], bb[WK])
    } else {
        (bb[BP], bb[BN], bb[BB], bb[BR], bb[BQ], bb[BK])
    };
    if by_white {
        if p & ((bit_s & !0x0101010101010101) << 7 | (bit_s & !0x8080808080808080) << 9) != 0 {
            return true;
        }
    } else {
        if p & ((bit_s & !0x0101010101010101) >> 9 | (bit_s & !0x8080808080808080) >> 7) != 0 {
            return true;
        }
    }
    if n & KNIGHT_ATTACKS[s] != 0 {
        return true;
    }
    if k & KING_ATTACKS[s] != 0 {
        return true;
    }
    if (b | q) & bishop_attacks(s, occ) != 0 {
        return true;
    }
    if (r | q) & rook_attacks(s, occ) != 0 {
        return true;
    }
    false
}

pub fn see(bb: &[u64; 12], from: usize, to: usize) -> i32 {
    let target_pi = piece_on(bb, to);
    if target_pi == EMPTY_SQ {
        return 0;
    }
    let attacker_pi = piece_on(bb, from);
    if attacker_pi == EMPTY_SQ {
        return 0;
    }

    let mut bbs = *bb;
    let mut occ = all_occ(&bbs);
    let mut side = is_white_piece(attacker_pi);

    let mut gain = [0i32; 32];
    let mut depth = 0usize;
    gain[depth] = see_val(piece_type(target_pi));
    depth += 1;

    bbs[attacker_pi as usize] &= !bit(from);
    occ &= !bit(from);
    let mut current_pt = piece_type(attacker_pi);

    side = !side;

    loop {
        let (lva_sq, lva_pi) = least_valuable_attacker(&bbs, to, occ, side);
        if lva_sq == 64 {
            break;
        }

        gain[depth] = see_val(current_pt) - gain[depth - 1].max(0);
        depth += 1;
        current_pt = piece_type(lva_pi);

        bbs[lva_pi as usize] &= !bit(lva_sq);
        occ &= !bit(lva_sq);
        side = !side;

        if depth >= 32 {
            break;
        }
    }

    let mut i = depth as i32 - 1;
    while i > 0 {
        gain[i as usize - 1] = (-gain[i as usize]).max(gain[i as usize - 1]);
        i -= 1;
    }
    gain[0]
}

fn least_valuable_attacker(bb: &[u64; 12], to: usize, occ: u64, white: bool) -> (usize, u8) {
    let (p, n, b, r, q, k, base) = if white {
        (bb[WP], bb[WN], bb[WB], bb[WR], bb[WQ], bb[WK], 0usize)
    } else {
        (bb[BP], bb[BN], bb[BB], bb[BR], bb[BQ], bb[BK], 6usize)
    };
    let to_bit = bit(to);
    let patt = if white {
        (to_bit & !0x0101010101010101) << 7 | (to_bit & !0x8080808080808080) << 9
    } else {
        (to_bit & !0x0101010101010101) >> 9 | (to_bit & !0x8080808080808080) >> 7
    };
    if p & patt != 0 {
        let s = (p & patt).trailing_zeros() as usize;
        return (s, (base) as u8);
    }
    if n & KNIGHT_ATTACKS[to] != 0 {
        let s = (n & KNIGHT_ATTACKS[to]).trailing_zeros() as usize;
        return (s, (base + 1) as u8);
    }
    let ba = bishop_attacks(to, occ);
    if b & ba != 0 {
        let s = (b & ba).trailing_zeros() as usize;
        return (s, (base + 2) as u8);
    }
    let ra = rook_attacks(to, occ);
    if r & ra != 0 {
        let s = (r & ra).trailing_zeros() as usize;
        return (s, (base + 3) as u8);
    }
    if q & ba != 0 {
        let s = (q & ba).trailing_zeros() as usize;
        return (s, (base + 4) as u8);
    }
    if q & ra != 0 {
        let s = (q & ra).trailing_zeros() as usize;
        return (s, (base + 4) as u8);
    }
    if k & KING_ATTACKS[to] != 0 {
        let s = (k & KING_ATTACKS[to]).trailing_zeros() as usize;
        return (s, (base + 5) as u8);
    }
    (64, EMPTY_SQ)
}

pub static KNIGHT_ATTACKS: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let r = s / 8;
        let c = s % 8;
        let mut v = 0u64;
        if r >= 2 && c >= 1 {
            v |= 1 << ((r - 2) * 8 + (c - 1));
        }
        if r >= 2 && c <= 6 {
            v |= 1 << ((r - 2) * 8 + (c + 1));
        }
        if r >= 1 && c >= 2 {
            v |= 1 << ((r - 1) * 8 + (c - 2));
        }
        if r >= 1 && c <= 5 {
            v |= 1 << ((r - 1) * 8 + (c + 2));
        }
        if r <= 6 && c >= 2 {
            v |= 1 << ((r + 1) * 8 + (c - 2));
        }
        if r <= 6 && c <= 5 {
            v |= 1 << ((r + 1) * 8 + (c + 2));
        }
        if r <= 5 && c >= 1 {
            v |= 1 << ((r + 2) * 8 + (c - 1));
        }
        if r <= 5 && c <= 6 {
            v |= 1 << ((r + 2) * 8 + (c + 1));
        }
        t[s] = v;
        s += 1;
    }
    t
};

pub static KING_ATTACKS: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let r = s / 8;
        let c = s % 8;
        let mut v = 0u64;
        if r > 0 {
            if c > 0 {
                v |= 1 << ((r - 1) * 8 + (c - 1));
            }
            v |= 1 << ((r - 1) * 8 + c);
            if c < 7 {
                v |= 1 << ((r - 1) * 8 + (c + 1));
            }
        }
        if c > 0 {
            v |= 1 << (r * 8 + (c - 1));
        }
        if c < 7 {
            v |= 1 << (r * 8 + (c + 1));
        }
        if r < 7 {
            if c > 0 {
                v |= 1 << ((r + 1) * 8 + (c - 1));
            }
            v |= 1 << ((r + 1) * 8 + c);
            if c < 7 {
                v |= 1 << ((r + 1) * 8 + (c + 1));
            }
        }
        t[s] = v;
        s += 1;
    }
    t
};
