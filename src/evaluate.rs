use crate::board::{
    all_occ, bit, black_occ, sq, sq_c, sq_r, white_occ, BoardState, BB, BK, BN, BP, BQ, BR,
    KING_ATTACKS, KNIGHT_ATTACKS, WB, WK, WN, WP, WQ, WR,
};
use crate::magic::{bishop_attacks, rook_attacks};
use crate::nnue::{NNUEAccumulator, NNUENet};
use crate::types::*;
use std::sync::OnceLock;

const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;

#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
    0,   0,   0,   0,   0,   0,  0,   0,
   98, 134,  61,  95,  68, 126, 34, -11,
   -6,   7,  26,  31,  65,  56, 25, -20,
  -14,  13,   6,  21,  23,  12, 17, -23,
  -27,  -2,  -5,  12,  17,   6, 10, -25,
  -26,  -4,  -4, -10,   3,   3, 33, -12,
  -35,  -1, -20, -23, -15,  24, 38, -22,
    0,   0,   0,   0,   0,   0,  0,   0,
];
#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
  -167, -89, -34, -49,  61, -97, -15, -107,
   -73, -41,  72,  36,  23,  62,   7,  -17,
   -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
   -13,   4,  16,  13,  28,  19,  21,   -8,
   -23,  -9,  12,  10,  19,  17,  25,  -16,
   -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -105, -21, -58, -33, -17, -28, -19,  -23,
];
#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
   -29,   4, -82, -37, -25, -42,   7,  -8,
   -26,  16, -18, -13,  30,  59,  18, -47,
   -16,  37,  43,  40,  35,  50,  37,  -2,
    -4,   5,  19,  50,  37,  37,   7,  -2,
    -6,  13,  13,  26,  34,  12,  10,   4,
     0,  15,  15,  15,  14,  27,  18,  10,
     4,  15,  16,   0,   7,  21,  33,   1,
   -33,  -3, -14, -21, -13, -12, -39, -21,
];
#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
    32,  42,  32,  51, 63,  9,  31,  43,
    27,  32,  58,  62, 80, 67,  26,  44,
    -5,  19,  26,  36, 17, 45,  61,  16,
   -24, -11,   7,  26, 24, 35,  -8, -20,
   -36, -26, -12,  -1,  9, -7,   6, -23,
   -45, -25, -16, -17,  3,  0,  -5, -33,
   -44, -16, -20,  -9, -1, 11,  -6, -71,
   -19, -13,   1,  17, 16,  7, -37, -26,
];
#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
   -28,   0,  29,  12,  59,  44,  43,  45,
   -24, -39,  -5,   1, -16,  57,  28,  54,
   -13, -17,   7,   8,  29,  56,  47,  57,
   -27, -27, -16, -16,  -1,  17,  -2,   1,
    -9, -26,  -9, -10,  -2,  -4,   3,  -3,
   -14,   2, -11,  -2,  -5,   2,  14,   5,
   -35,  -8,  11,   2,   8,  15,  -3,   1,
    -1, -18,  -9,  10, -15, -25, -31, -50,
];
#[rustfmt::skip]
const MG_KING: [i32; 64] = [
   -65,  23,  16, -15, -56, -34,   2,  13,
    29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -9,  24,   2, -16, -20,   6,  22, -22,
   -17, -20, -12, -27, -30, -25, -14, -36,
   -49,  -1, -27, -39, -46, -44, -33, -51,
   -14, -14, -22, -46, -44, -30, -15, -27,
     1,   7,  -8, -64, -43, -16,   9,   8,
   -15,  36,  12, -54,   8, -28,  24,  14,
];
#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
    0,   0,   0,   0,   0,   0,   0,   0,
  178, 173, 158, 134, 147, 132, 165, 187,
   94, 100,  85,  67,  56,  53,  82,  84,
   32,  24,  13,   5,  -2,   4,  17,  17,
   13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    4,   7,  -6,   1,   0,  -5,  -1,  -8,
   13,   8,   8, -10,  -6,  -4,  -1,  -2,
    0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
   -58, -38, -13, -28, -31, -27, -63, -99,
   -25,  -8, -25,  -2,  -9, -25, -24, -52,
   -24, -20,  10,   9,  -1,  -9, -19, -41,
   -17,   3,  22,  22,  22,  11,   8, -18,
   -18,  -6,  16,  25,  16,  17,   4, -18,
   -23,  -3,  -1,  15,  10,  -3, -20, -22,
   -42, -20, -10,  -5,  -2, -20, -23, -44,
   -29, -51, -23, -15, -22, -18, -50, -64,
];
#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
   -14, -21, -11,  -8, -7,  -9, -17, -24,
    -8,  -4,   7, -12, -3, -13,  -4, -14,
     2,  -8,   0,  -1, -2,   6,   0,   4,
    -3,   9,  12,   9, 14,  10,   3,   2,
    -6,   3,  13,  19,  7,  10,  -3,  -9,
   -12,  -3,   8,  10, 13,   3,  -7, -15,
   -14, -18,  -7,  -1,  4,  -9, -15, -27,
   -23,  -9, -23,  -5, -9, -16,  -5, -17,
];
#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
];
#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
   -9,  22,  22,  27,  27,  19,  10,  20,
  -17,  20,  32,  41,  58,  25,  30,   0,
  -20,   6,   9,  49,  47,  35,  19,   9,
    3,  22,  24,  45,  57,  40,  57,  36,
  -18,  28,  19,  47,  31,  34,  39,  23,
  -16, -27,  15,   6,   9,  17,  10,   5,
  -22, -23, -30, -16, -16, -23, -36, -32,
  -33, -28, -22, -43,  -5, -32, -20, -41,
];
#[rustfmt::skip]
const EG_KING: [i32; 64] = [
  -74, -35, -18, -18, -11,  15,   4, -17,
  -12,  17,  14,  17,  17,  38,  23,  11,
   10,  17,  23,  15,  20,  45,  44,  13,
   -8,  22,  24,  27,  26,  33,  26,   3,
  -18,  -4,  21,  24,  27,  23,   9, -11,
  -19,  -3,  11,  21,  23,  16,   7,  -9,
  -27, -11,   4,  13,  14,   4,  -5, -17,
  -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_TABLES: [&[i32; 64]; 6] = [
    &MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING,
];
const EG_TABLES: [&[i32; 64]; 6] = [
    &EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING,
];

const MOB_KNIGHT: [i32; 9] = [-20, -15, -5, 0, 5, 10, 15, 18, 20];
const MOB_BISHOP: [i32; 14] = [-20, -10, -5, 0, 5, 8, 11, 13, 15, 16, 17, 18, 18, 18];
const MOB_ROOK: [i32; 15] = [-15, -8, -4, 0, 3, 5, 7, 9, 11, 12, 13, 14, 14, 14, 14];
const MOB_QUEEN: [i32; 28] = [
    -20, -14, -8, -2, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14,
];

#[inline]
fn flip_sq(sq: usize) -> usize {
    sq ^ 56
}

fn count_mobility_bb(bb: &[u64; 12], s: usize, occ: u64, white: bool) -> usize {
    let own = if white { white_occ(bb) } else { black_occ(bb) };
    let pi_type = {
        let b = bit(s);
        let base = if white { 0usize } else { 6 };
        let mut t = 6usize;
        for i in 0..6 {
            if bb[base + i] & b != 0 {
                t = i;
                break;
            }
        }
        t
    };
    match pi_type {
        1 => (KNIGHT_ATTACKS[s] & !own).count_ones() as usize,
        2 => (bishop_attacks(s, occ) & !own).count_ones() as usize,
        3 => (rook_attacks(s, occ) & !own).count_ones() as usize,
        4 => ((bishop_attacks(s, occ) | rook_attacks(s, occ)) & !own).count_ones() as usize,
        _ => 0,
    }
}

fn eval_pawns(bb: &[u64; 12]) -> i32 {
    let mut score = 0i32;
    let mut w_file = [0i32; 8];
    let mut b_file = [0i32; 8];
    let mut w_pawns = [[false; 8]; 8];
    let mut b_pawns = [[false; 8]; 8];

    let mut wp = bb[WP];
    while wp != 0 {
        let s = wp.trailing_zeros() as usize;
        let r = sq_r(s);
        let c = sq_c(s);
        w_file[c] += 1;
        w_pawns[c][r] = true;
        wp &= wp - 1;
    }
    let mut bp = bb[BP];
    while bp != 0 {
        let s = bp.trailing_zeros() as usize;
        let r = sq_r(s);
        let c = sq_c(s);
        b_file[c] += 1;
        b_pawns[c][r] = true;
        bp &= bp - 1;
    }

    for c in 0..8 {
        if w_file[c] > 1 {
            score -= 15 * (w_file[c] - 1);
        }
        if b_file[c] > 1 {
            score += 15 * (b_file[c] - 1);
        }

        let wn = (if c > 0 { w_file[c - 1] } else { 0 }) + (if c < 7 { w_file[c + 1] } else { 0 });
        let bn = (if c > 0 { b_file[c - 1] } else { 0 }) + (if c < 7 { b_file[c + 1] } else { 0 });
        if w_file[c] > 0 && wn == 0 {
            score -= 20;
        }
        if b_file[c] > 0 && bn == 0 {
            score += 20;
        }

        for r in 0..8 {
            if w_pawns[c][r] {
                let blocked = (0..r).any(|r2| {
                    b_pawns[c][r2] || (c > 0 && b_pawns[c - 1][r2]) || (c < 7 && b_pawns[c + 1][r2])
                });
                if !blocked {
                    let rank = (7 - r) as i32;
                    score += 10 + rank * rank * 3;
                }
            }
            if b_pawns[c][r] {
                let blocked = (r + 1..8).any(|r2| {
                    w_pawns[c][r2] || (c > 0 && w_pawns[c - 1][r2]) || (c < 7 && w_pawns[c + 1][r2])
                });
                if !blocked {
                    let rank = r as i32;
                    score -= 10 + rank * rank * 3;
                }
            }
        }
    }
    score
}

fn king_safety(bb: &[u64; 12], white: bool, phase: i32) -> i32 {
    if phase <= 6 {
        return 0;
    }
    let kbb = if white { bb[WK] } else { bb[BK] };
    if kbb == 0 {
        return 0;
    }
    let ks = kbb.trailing_zeros() as usize;
    let kr = sq_r(ks);
    let kc = sq_c(ks);
    let opp = !white;

    let mut danger = 0i32;
    let zone = KING_ATTACKS[ks] | bit(ks);
    let mut z = zone;
    while z != 0 {
        let t = z.trailing_zeros() as usize;
        let occ = all_occ(bb);
        let (p, n, b, r, q) = if opp {
            (bb[WP], bb[WN], bb[WB], bb[WR], bb[WQ])
        } else {
            (bb[BP], bb[BN], bb[BB], bb[BR], bb[BQ])
        };
        let tb = bit(t);
        let patt = if opp {
            (tb & !0x0101010101010101u64) << 7 | (tb & !0x8080808080808080u64) << 9
        } else {
            (tb & !0x0101010101010101u64) >> 9 | (tb & !0x8080808080808080u64) >> 7
        };
        if p & patt != 0 {
            danger += 8;
        }
        if n & KNIGHT_ATTACKS[t] != 0 {
            danger += 15;
        }
        let ba = bishop_attacks(t, occ);
        let ra = rook_attacks(t, occ);
        if b & ba != 0 {
            danger += 15;
        }
        if r & ra != 0 {
            danger += 20;
        }
        if q & (ba | ra) != 0 {
            danger += 40;
        }
        z &= z - 1;
    }
    let front_r = if white { kr.wrapping_sub(1) } else { kr + 1 };
    if front_r < 8 {
        for dc in 0usize..3 {
            let fc = match dc {
                0 => kc,
                1 => kc.wrapping_sub(1),
                _ => kc + 1,
            };
            if fc >= 8 {
                continue;
            }
            let shelter_bit = bit(sq(front_r, fc));
            let has = if white {
                bb[WP] & shelter_bit != 0
            } else {
                bb[BP] & shelter_bit != 0
            };
            if !has {
                danger += 10;
            }
        }
    }
    -(danger * phase / 24).max(0)
}

static NNUE_NET: OnceLock<NNUENet> = OnceLock::new();

pub fn init_nnue(path: &str) -> Result<(), String> {
    let net = NNUENet::load(path)?;
    NNUE_NET
        .set(net)
        .map_err(|_| "NNUE already initialised".to_string())
}

pub fn init_nnue_from_bytes(data: &[u8]) -> Result<(), String> {
    let path = format!("{}/.nnue_temp", std::env::temp_dir().display());
    std::fs::write(&path, data).map_err(|e| format!("write temp: {}", e))?;
    let net = NNUENet::load(&path)?;
    let _ = std::fs::remove_file(&path);
    NNUE_NET
        .set(net)
        .map_err(|_| "NNUE already initialised".to_string())
}

pub fn nnue_loaded() -> bool {
    NNUE_NET.get().is_some()
}

pub fn get_nnue_net() -> Option<&'static NNUENet> {
    NNUE_NET.get()
}

pub fn evaluate_nnue_acc(net: &NNUENet, acc: &NNUEAccumulator, st: &BoardState) -> i32 {
    let stm = if st.w { WHITE } else { BLACK };
    let pc: u32 = (0..12).map(|i| st.bb[i].count_ones()).sum();
    let score = net.forward(acc, stm, pc);
    if stm == WHITE {
        score
    } else {
        -score
    }
}

pub fn evaluate_nnue(st: &BoardState) -> i32 {
    let net = match NNUE_NET.get() {
        Some(n) => n,
        None => return evaluate(st),
    };
    let mut acc = NNUEAccumulator::new(net.hidden_size);
    acc.refresh(net, st);
    evaluate_nnue_acc(net, &acc, st)
}

pub fn evaluate(st: &BoardState) -> i32 {
    let occ = all_occ(&st.bb);
    let mut phase = 0i32;
    let mut mg_score = 0i32;
    let mut eg_score = 0i32;

    for pi in 0..12usize {
        let white = pi < 6;
        let pt = pi % 6;
        let sign = if white { 1 } else { -1 };
        let mut bb = st.bb[pi];
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            let table_sq = if white { s } else { flip_sq(s) };
            mg_score += sign * (MG_VALUE[pt] + MG_TABLES[pt][table_sq]);
            eg_score += sign * (EG_VALUE[pt] + EG_TABLES[pt][table_sq]);
            phase += PHASE_INC[pt];

            let mob = count_mobility_bb(&st.bb, s, occ, white).min(27);
            let mob_bonus = match pt {
                1 => MOB_KNIGHT[mob.min(8)],
                2 => MOB_BISHOP[mob.min(13)],
                3 => MOB_ROOK[mob.min(14)],
                4 => MOB_QUEEN[mob.min(27)],
                _ => 0,
            };
            mg_score += sign * mob_bonus;
            eg_score += sign * mob_bonus;

            bb &= bb - 1;
        }
    }

    phase = phase.min(TOTAL_PHASE);

    if st.bb[WB].count_ones() >= 2 {
        mg_score += 40;
        eg_score += 40;
    }
    if st.bb[BB].count_ones() >= 2 {
        mg_score -= 40;
        eg_score -= 40;
    }

    let ps = eval_pawns(&st.bb);
    mg_score += ps;
    eg_score += ps;

    mg_score += king_safety(&st.bb, true, phase);
    mg_score -= king_safety(&st.bb, false, phase);

    for pi in [WR, BR] {
        let white = pi == WR;
        let sign = if white { 1 } else { -1 };
        let mut rooks = st.bb[pi];
        while rooks != 0 {
            let s = rooks.trailing_zeros() as usize;
            let c = sq_c(s);
            let file_mask: u64 = 0x0101010101010101u64 << c;
            let wp = (st.bb[WP] & file_mask).count_ones() as i32;
            let bp = (st.bb[BP] & file_mask).count_ones() as i32;
            let own_p = if white { wp } else { bp };
            let opp_p = if white { bp } else { wp };
            let bonus = if own_p == 0 && opp_p == 0 {
                20
            } else if own_p == 0 {
                10
            } else {
                0
            };
            mg_score += sign * bonus;
            eg_score += sign * bonus;
            rooks &= rooks - 1;
        }
    }

    (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}
