use crate::board::{Board8, EMPTY, is_white, ptype, find_king, can_attack};

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
    &MG_PAWN, &MG_KNIGHT, &MG_BISHOP,
    &MG_ROOK, &MG_QUEEN, &MG_KING,
];
const EG_TABLES: [&[i32; 64]; 6] = [
    &EG_PAWN, &EG_KNIGHT, &EG_BISHOP,
    &EG_ROOK, &EG_QUEEN, &EG_KING,
];

fn pt_index(pt: u8) -> usize {
    match pt { b'p' => 0, b'n' => 1, b'b' => 2, b'r' => 3, b'q' => 4, b'k' => 5, _ => 0 }
}

fn flip_sq(sq: usize) -> usize {
    sq ^ 56
}

const MOB_KNIGHT: [i32; 9]  = [-20,-15,-5, 0, 5, 10, 15, 18, 20];
const MOB_BISHOP: [i32; 14] = [-20,-10,-5, 0, 5,  8, 11, 13, 15, 16, 17, 18, 18, 18];
const MOB_ROOK:   [i32; 15] = [-15,-8, -4, 0, 3,  5,  7,  9, 11, 12, 13, 14, 14, 14, 14];
const MOB_QUEEN:  [i32; 28] = [-20,-14,-8,-2, 0,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14];

fn count_mobility(b: &Board8, r: usize, c: usize) -> usize {
    let p = b[r][c];
    if p == EMPTY { return 0; }
    let pt = ptype(p);
    let wturn = is_white(p);
    let mut mob = 0usize;
    match pt {
        b'n' => {
            for &[dr, dc] in &[[-2i32,-1i32],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                    let t = b[nr as usize][nc as usize];
                    if t == EMPTY || is_white(t) != wturn { mob += 1; }
                }
            }
        }
        b'b' => {
            for &[dr, dc] in &[[-1i32,-1i32],[-1,1],[1,-1],[1,1]] {
                for s in 1..8i32 {
                    let nr = r as i32 + dr * s;
                    let nc = c as i32 + dc * s;
                    if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                    let t = b[nr as usize][nc as usize];
                    if t == EMPTY { mob += 1; }
                    else { if is_white(t) != wturn { mob += 1; } break; }
                }
            }
        }
        b'r' => {
            for &[dr, dc] in &[[-1i32,0i32],[1,0],[0,-1],[0,1]] {
                for s in 1..8i32 {
                    let nr = r as i32 + dr * s;
                    let nc = c as i32 + dc * s;
                    if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                    let t = b[nr as usize][nc as usize];
                    if t == EMPTY { mob += 1; }
                    else { if is_white(t) != wturn { mob += 1; } break; }
                }
            }
        }
        b'q' => {
            for &[dr, dc] in &[[-1i32,-1i32],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]] {
                for s in 1..8i32 {
                    let nr = r as i32 + dr * s;
                    let nc = c as i32 + dc * s;
                    if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                    let t = b[nr as usize][nc as usize];
                    if t == EMPTY { mob += 1; }
                    else { if is_white(t) != wturn { mob += 1; } break; }
                }
            }
        }
        _ => {}
    }
    mob
}

fn eval_pawns(b: &Board8) -> i32 {
    let mut score = 0i32;

    let mut w_pawns_on_file = [0i32; 8];
    let mut b_pawns_on_file = [0i32; 8];
    let mut w_pawn_ranks = [[false; 8]; 8];
    let mut b_pawn_ranks = [[false; 8]; 8];

    for r in 0..8 {
        for c in 0..8 {
            let p = b[r][c];
            if p == b'P' {
                w_pawns_on_file[c] += 1;
                w_pawn_ranks[c][7 - r] = true;
            } else if p == b'p' {
                b_pawns_on_file[c] += 1;
                b_pawn_ranks[c][r] = true;
            }
        }
    }

    for c in 0..8 {
        if w_pawns_on_file[c] > 1 { score -= 15 * (w_pawns_on_file[c] - 1); }
        if b_pawns_on_file[c] > 1 { score += 15 * (b_pawns_on_file[c] - 1); }

        let w_neighbors = (if c > 0 { w_pawns_on_file[c-1] } else { 0 }) +
                          (if c < 7 { w_pawns_on_file[c+1] } else { 0 });
        let b_neighbors = (if c > 0 { b_pawns_on_file[c-1] } else { 0 }) +
                          (if c < 7 { b_pawns_on_file[c+1] } else { 0 });
        if w_pawns_on_file[c] > 0 && w_neighbors == 0 { score -= 20; }
        if b_pawns_on_file[c] > 0 && b_neighbors == 0 { score += 20; }

        if w_pawns_on_file[c] > 0 {
            for rank in 0..8 {
                if !w_pawn_ranks[c][rank] { continue; }
                let blocked = (0..rank).any(|r2| {
                    b_pawn_ranks[c][r2] ||
                    (c > 0 && b_pawn_ranks[c-1][r2]) ||
                    (c < 7 && b_pawn_ranks[c+1][r2])
                });
                if !blocked {
                    score += 10 + rank as i32 * rank as i32 * 3;
                }
            }
        }
        if b_pawns_on_file[c] > 0 {
            for rank in 0..8 {
                if !b_pawn_ranks[c][rank] { continue; }
                let blocked = (0..rank).any(|r2| {
                    w_pawn_ranks[c][r2] ||
                    (c > 0 && w_pawn_ranks[c-1][r2]) ||
                    (c < 7 && w_pawn_ranks[c+1][r2])
                });
                if !blocked {
                    score -= 10 + rank as i32 * rank as i32 * 3;
                }
            }
        }
    }
    score
}

fn king_safety(b: &Board8, wturn: bool, phase: i32) -> i32 {
    if phase <= 6 { return 0; }
    let (kr, kc) = find_king(b, wturn);
    let opp = !wturn;
    let mut danger = 0i32;
    let front_r = if wturn { kr.wrapping_sub(1) } else { kr + 1 };
    for &(r, c) in &[(kr.wrapping_sub(1), kc.wrapping_sub(1)), (kr.wrapping_sub(1), kc),
                     (kr.wrapping_sub(1), kc+1), (kr, kc.wrapping_sub(1)),
                     (kr, kc+1), (kr+1, kc.wrapping_sub(1)), (kr+1, kc), (kr+1, kc+1)] {
        if r >= 8 || c >= 8 { continue; }
        for r2 in 0..8 { for c2 in 0..8 {
            let p = b[r2][c2];
            if p == EMPTY || is_white(p) != opp { continue; }
            let pt = ptype(p);
            if pt == b'k' { continue; }
            if can_attack(b, r2, c2, r, c) {
                danger += match pt {
                    b'q' => 40, b'r' => 20, b'b' | b'n' => 15, b'p' => 8, _ => 0,
                };
            }
        }}
    }
    if front_r < 8 {
        for dc in [0usize, 1, 2] {
            let fc = if dc == 0 { kc } else if dc == 1 { kc.wrapping_sub(1) } else { kc + 1 };
            if fc >= 8 { continue; }
            let shelter_p = if wturn { b'P' } else { b'p' };
            if b[front_r][fc] != shelter_p { danger += 10; }
        }
    }
    -(danger * phase / 24).max(0)
}

pub fn evaluate(b: &Board8) -> i32 {
    let mut phase = 0i32;
    let mut mg_score = 0i32;
    let mut eg_score = 0i32;

    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p == EMPTY { continue; }
        let w = is_white(p);
        let pt = ptype(p);
        let pi = pt_index(pt);
        let sq = r * 8 + c;
        let table_sq = if w { sq } else { flip_sq(sq) };

        mg_score += (MG_VALUE[pi] + MG_TABLES[pi][table_sq]) * if w { 1 } else { -1 };
        eg_score += (EG_VALUE[pi] + EG_TABLES[pi][table_sq]) * if w { 1 } else { -1 };
        phase += PHASE_INC[pi];

        let mob = count_mobility(b, r, c).min(27);
        let mob_bonus = match pt {
            b'n' => MOB_KNIGHT[mob.min(8)],
            b'b' => MOB_BISHOP[mob.min(13)],
            b'r' => MOB_ROOK[mob.min(14)],
            b'q' => MOB_QUEEN[mob.min(27)],
            _ => 0,
        };
        mg_score += mob_bonus * if w { 1 } else { -1 };
        eg_score += mob_bonus * if w { 1 } else { -1 };
    }}

    phase = phase.min(TOTAL_PHASE);

    let mut wb = 0; let mut bb = 0;
    for r in 0..8 { for c in 0..8 {
        if b[r][c] == b'B' { wb += 1; }
        if b[r][c] == b'b' { bb += 1; }
    }}
    if wb >= 2 { mg_score += 40; eg_score += 40; }
    if bb >= 2 { mg_score -= 40; eg_score -= 40; }

    let pawn_score = eval_pawns(b);
    mg_score += pawn_score;
    eg_score += pawn_score;

    mg_score += king_safety(b, true, phase);
    mg_score -= king_safety(b, false, phase);

    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p == EMPTY { continue; }
        if ptype(p) == b'r' {
            let w = is_white(p);
            let (wp, bp) = (0..8).fold((0, 0), |(w, bk), row| {
                if b[row][c] == b'P' { (w+1, bk) }
                else if b[row][c] == b'p' { (w, bk+1) }
                else { (w, bk) }
            });
            let own_pawns = if w { wp } else { bp };
            let opp_pawns = if w { bp } else { wp };
            let bonus = if own_pawns == 0 && opp_pawns == 0 { 20 }
                        else if own_pawns == 0 { 10 }
                        else { 0 };
            mg_score += bonus * if w { 1 } else { -1 };
            eg_score += bonus * if w { 1 } else { -1 };
        }
    }}

    (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}