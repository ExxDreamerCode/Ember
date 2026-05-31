use std::time::Instant;
use std::sync::OnceLock;
use rand::Rng;
use rand::SeedableRng;

const EMPTY: u8 = b'.';
const MATE: i32 = 100_000;
const INF: i32 = 1_000_000;
const MAX_PLY: usize = 128;
const QS_DEPTH: i32 = 8;

fn see_val(pt: u8) -> i32 {
    match pt {
        b'p' => 100, b'n' => 325, b'b' => 340, b'r' => 500, b'q' => 950, b'k' => 20000,
        _ => 0,
    }
}

pub fn see(b: &Board8, fr: usize, fc: usize, er: usize, ec: usize) -> i32 {
    let target = b[er][ec];
    if target == EMPTY { return 0; }
    let mut board = *b;
    let target_pt = ptype(target);
    let side = is_white(b[fr][fc]);

    let mut gain = [0i32; 32];
    let mut depth = 0;
    let mut current_val = see_val(target_pt);
    let mut current_side = side;

    board[er][ec] = board[fr][fc];
    board[fr][fc] = EMPTY;
    gain[depth] = current_val;
    depth += 1;
    current_side = !current_side;

    loop {
        let mut best_val = i32::MAX;
        let mut best_r = 8;
        let mut best_c = 8;
        for r in 0..8 {
            for c in 0..8 {
                let p = board[r][c];
                if p == EMPTY { continue; }
                if is_white(p) != current_side { continue; }
                if can_attack(&board, r, c, er, ec) {
                    let v = see_val(ptype(p));
                    if v < best_val {
                        best_val = v;
                        best_r = r;
                        best_c = c;
                    }
                }
            }
        }
        if best_r == 8 { break; }

        current_val = best_val;
        board[er][ec] = board[best_r][best_c];
        board[best_r][best_c] = EMPTY;
        gain[depth] = current_val - gain[depth - 1].max(0);
        depth += 1;
        current_side = !current_side;
    }

    let mut i = depth as i32 - 1;
    while i > 0 {
        gain[i as usize - 1] = (-gain[i as usize]).max(gain[i as usize - 1]);
        i -= 1;
    }
    gain[0]
}

pub type Board8 = [[u8; 8]; 8];

#[inline(always)] pub fn is_white(p: u8) -> bool { p.is_ascii_uppercase() }
#[inline(always)] pub fn ptype(p: u8) -> u8 { if p.is_ascii_uppercase() { p + 32 } else { p } }

pub fn find_king(b: &Board8, wturn: bool) -> (usize, usize) {
    let kc = if wturn { b'K' } else { b'k' };
    for r in 0..8 { for c in 0..8 { if b[r][c] == kc { return (r, c); } } }
    (0, 0)
}

pub fn coord_to_square(r: usize, c: usize) -> String {
    format!("{}{}", (b'a' + c as u8) as char, 8 - r as u8)
}

pub fn can_attack(b: &Board8, fr: usize, fc: usize, tr: usize, tc: usize) -> bool {
    let p = b[fr][fc];
    if p == EMPTY { return false; }
    let pt = ptype(p);
    let dr = tr as i32 - fr as i32;
    let dc = tc as i32 - fc as i32;
    match pt {
        b'p' => {
            let d = if is_white(p) { -1i32 } else { 1 };
            dr == d && dc.abs() == 1
        }
        b'n' => (dr.abs() == 2 && dc.abs() == 1) || (dr.abs() == 1 && dc.abs() == 2),
        b'k' => dr.abs() <= 1 && dc.abs() <= 1 && (dr != 0 || dc != 0),
        b'b' => {
            if dr.abs() != dc.abs() || dr == 0 { return false; }
            let sr = dr.signum(); let sc = dc.signum();
            let (mut r, mut c) = (fr as i32 + sr, fc as i32 + sc);
            while (r, c) != (tr as i32, tc as i32) {
                if b[r as usize][c as usize] != EMPTY { return false; }
                r += sr; c += sc;
            }
            true
        }
        b'r' => {
            if dr != 0 && dc != 0 { return false; }
            if dr == 0 && dc == 0 { return false; }
            let sr = dr.signum(); let sc = dc.signum();
            let (mut r, mut c) = (fr as i32 + sr, fc as i32 + sc);
            while (r, c) != (tr as i32, tc as i32) {
                if b[r as usize][c as usize] != EMPTY { return false; }
                r += sr; c += sc;
            }
            true
        }
        b'q' => {
            if dr == 0 && dc == 0 { return false; }
            if dr != 0 && dc != 0 && dr.abs() != dc.abs() { return false; }
            let sr = dr.signum(); let sc = dc.signum();
            let (mut r, mut c) = (fr as i32 + sr, fc as i32 + sc);
            while (r, c) != (tr as i32, tc as i32) {
                if b[r as usize][c as usize] != EMPTY { return false; }
                r += sr; c += sc;
            }
            true
        }
        _ => false
    }
}

pub fn is_attacked(b: &Board8, row: usize, col: usize, attacker_w: bool) -> bool {
    for r in 0..8 {
        for c in 0..8 {
            let p = b[r][c];
            if p != EMPTY && is_white(p) == attacker_w && can_attack(b, r, c, row, col) {
                return true;
            }
        }
    }
    false
}

pub fn has_non_pawn(b: &Board8, wturn: bool) -> bool {
    for r in 0..8 {
        for c in 0..8 {
            let p = b[r][c];
            if p != EMPTY && is_white(p) == wturn && ptype(p) != b'p' && ptype(p) != b'k' {
                return true;
            }
        }
    }
    false
}

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

fn compute_hash(b: &Board8, wturn: bool, cr: &[bool; 4], ep: Option<(usize, usize)>) -> u64 {
    let z = zobrist();
    let mut key = 0u64;
    for r in 0..8 { for c in 0..8 { let p = b[r][c]; if p != EMPTY { key ^= z.pieces[piece_idx(p)][r][c]; } } }
    if !wturn { key ^= z.side; }
    for i in 0..4 { if cr[i] { key ^= z.castling[i]; } }
    if let Some((er, ec)) = ep { key ^= z.ep[er][ec]; }
    key
}

fn game_phase(b: &Board8) -> i32 {
    let mut phase = 0i32;
    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p == EMPTY { continue; }
        match ptype(p) {
            b'n' | b'b' => phase += 1,
            b'r' => phase += 2,
            b'q' => phase += 4,
            _ => {}
        }
    }}
    phase
}

const PST_PAWN: [[i32;8];8] = [
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [ 50, 50, 50, 50, 50, 50, 50, 50],
    [ 10, 10, 20, 30, 30, 20, 10, 10],
    [  5,  5, 10, 27, 27, 10,  5,  5],
    [  0,  0,  0, 25, 25,  0,  0,  0],
    [  5, -5,-10,  0,  0,-10, -5,  5],
    [  5, 10, 10,-20,-20, 10, 10,  5],
    [  0,  0,  0,  0,  0,  0,  0,  0],
];
const PST_KNIGHT: [[i32;8];8] = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
];
const PST_BISHOP: [[i32;8];8] = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
];
const PST_ROOK: [[i32;8];8] = [
    [  0,  0,  0,  5,  5,  0,  0,  0],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [  5, 10, 10, 10, 10, 10, 10,  5],
    [  0,  0,  0,  0,  0,  0,  0,  0],
];
const PST_QUEEN: [[i32;8];8] = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
];
const PST_KING_MG: [[i32;8];8] = [
    [ 20, 30, 10,  0,  0, 10, 30, 20],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
];
const PST_KING_EG: [[i32;8];8] = [
    [-50,-30,-30,-30,-30,-30,-30,-50],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-50,-40,-30,-20,-20,-30,-40,-50],
];

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
    let phase = game_phase(b);
    let endgame = phase <= 6;

    let mut score = 0i32;
    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p == EMPTY { continue; }
        let w = is_white(p);
        let pt = ptype(p);
        let rp = if w { 7 - r } else { r };
        let (mat, pst) = match pt {
            b'p' => (100, PST_PAWN[rp][c]),
            b'n' => (325, PST_KNIGHT[rp][c]),
            b'b' => (340, PST_BISHOP[rp][c]),
            b'r' => (500, PST_ROOK[rp][c]),
            b'q' => (950, PST_QUEEN[rp][c]),
            _    => (0, if endgame { PST_KING_EG[rp][c] } else { PST_KING_MG[rp][c] }),
        };
        score += (mat + pst) * if w { 1 } else { -1 };

        let mob = count_mobility(b, r, c).min(27);
        let mob_bonus = match pt {
            b'n' => MOB_KNIGHT[mob.min(8)],
            b'b' => MOB_BISHOP[mob.min(13)],
            b'r' => MOB_ROOK[mob.min(14)],
            b'q' => MOB_QUEEN[mob.min(27)],
            _ => 0,
        };
        score += mob_bonus * if w { 1 } else { -1 };
    }}

    let mut wb = 0; let mut bb = 0;
    for r in 0..8 { for c in 0..8 {
        if b[r][c] == b'B' { wb += 1; }
        if b[r][c] == b'b' { bb += 1; }
    }}
    if wb >= 2 { score += 40; }
    if bb >= 2 { score -= 40; }

    score += eval_pawns(b);

    score += king_safety(b, true, phase);
    score -= king_safety(b, false, phase);

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
            if own_pawns == 0 && opp_pawns == 0 { score += 20 * if w { 1 } else { -1 }; }
            else if own_pawns == 0 { score += 10 * if w { 1 } else { -1 }; }
        }
    }}

    score
}

#[derive(Clone, Copy)]
pub struct BoardState {
    pub b: Board8,
    pub w: bool,
    pub cr: [bool; 4],
    pub ep: Option<(usize, usize)>,
    pub mc: usize,
}

pub fn apply_move(st: &mut BoardState, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
    let p = st.b[sr][sc];
    let pt = ptype(p);

    if pt == b'p' && ec != sc && st.b[er][ec] == EMPTY {
        let cap_row = if st.w { er + 1 } else { er.wrapping_sub(1) };
        if cap_row < 8 { st.b[cap_row][ec] = EMPTY; }
    }

    if pt == b'k' && (ec as i32 - sc as i32).abs() == 2 {
        st.b[er][ec] = p;
        st.b[sr][sc] = EMPTY;
        if ec > sc {
            st.b[sr][5] = st.b[sr][7];
            st.b[sr][7] = EMPTY;
        } else {
            st.b[sr][3] = st.b[sr][0];
            st.b[sr][0] = EMPTY;
        }
    } else {
        let prom = if pt == b'p' && (er == 0 || er == 7) {
            if promotion != 0 {
                if st.w { promotion } else { promotion + 32 }
            } else {
                if st.w { b'Q' } else { b'q' }
            }
        } else { p };
        st.b[er][ec] = prom;
        st.b[sr][sc] = EMPTY;
    }

    if p == b'K' { st.cr[0] = false; st.cr[1] = false; }
    if p == b'k' { st.cr[2] = false; st.cr[3] = false; }
    if p == b'R' && sr == 7 && sc == 7 { st.cr[0] = false; }
    if p == b'R' && sr == 7 && sc == 0 { st.cr[1] = false; }
    if p == b'r' && sr == 0 && sc == 7 { st.cr[2] = false; }
    if p == b'r' && sr == 0 && sc == 0 { st.cr[3] = false; }
    if er == 7 && ec == 7 { st.cr[0] = false; }
    if er == 7 && ec == 0 { st.cr[1] = false; }
    if er == 0 && ec == 7 { st.cr[2] = false; }
    if er == 0 && ec == 0 { st.cr[3] = false; }

    st.ep = if pt == b'p' && (er as i32 - sr as i32).abs() == 2 {
        Some(((sr + er) / 2, sc))
    } else { None };

    st.w = !st.w;
    st.mc += 1;
}

pub fn generate_moves(b: &Board8, wturn: bool, cr: &[bool; 4], ep: Option<(usize, usize)>) -> Vec<[usize; 4]> {
    let opp = !wturn;
    let back = if wturn { 7usize } else { 0usize };
    let (kr, kc) = find_king(b, wturn);
    let in_check = is_attacked(b, kr, kc, opp);
    let mut result = Vec::with_capacity(48);

    for r in 0..8 { for c in 0..8 {
        let p = b[r][c];
        if p == EMPTY || is_white(p) != wturn { continue; }
        let pt = ptype(p);

        let pseudo: Vec<[usize; 2]> = match pt {
            b'p' => {
                let d: i32 = if wturn { -1 } else { 1 };
                let start_row = if wturn { 6usize } else { 1usize };
                let mut ms = Vec::with_capacity(4);
                let nr = r as i32 + d;
                if nr >= 0 && nr < 8 {
                    let nr = nr as usize;
                    if b[nr][c] == EMPTY {
                        ms.push([nr, c]);
                        let nr2 = (r as i32 + 2 * d) as usize;
                        if r == start_row && nr2 < 8 && b[nr2][c] == EMPTY {
                            ms.push([nr2, c]);
                        }
                    }
                    for dc in [-1i32, 1] {
                        let nc = c as i32 + dc;
                        if nc >= 0 && nc < 8 {
                            let nc = nc as usize;
                            let t = b[nr][nc];
                            if t != EMPTY && is_white(t) != wturn {
                                ms.push([nr, nc]);
                            }
                            if let Some((epr, epc)) = ep {
                                if nr == epr && nc == epc {
                                    ms.push([nr, nc]);
                                }
                            }
                        }
                    }
                }
                ms
            }
            b'n' => {
                [[-2i32,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]
                    .iter().filter_map(|&[dr, dc]| {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                            let t = b[nr as usize][nc as usize];
                            if t == EMPTY || is_white(t) != wturn { Some([nr as usize, nc as usize]) } else { None }
                        } else { None }
                    }).collect()
            }
            b'b' => {
                let mut ms = Vec::new();
                for &[dr, dc] in &[[-1i32,-1i32],[-1,1],[1,-1],[1,1]] {
                    for s in 1..8i32 {
                        let nr = r as i32 + dr * s;
                        let nc = c as i32 + dc * s;
                        if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                        let t = b[nr as usize][nc as usize];
                        if t == EMPTY { ms.push([nr as usize, nc as usize]); }
                        else { if is_white(t) != wturn { ms.push([nr as usize, nc as usize]); } break; }
                    }
                }
                ms
            }
            b'r' => {
                let mut ms = Vec::new();
                for &[dr, dc] in &[[-1i32,0i32],[1,0],[0,-1],[0,1]] {
                    for s in 1..8i32 {
                        let nr = r as i32 + dr * s;
                        let nc = c as i32 + dc * s;
                        if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                        let t = b[nr as usize][nc as usize];
                        if t == EMPTY { ms.push([nr as usize, nc as usize]); }
                        else { if is_white(t) != wturn { ms.push([nr as usize, nc as usize]); } break; }
                    }
                }
                ms
            }
            b'q' => {
                let mut ms = Vec::new();
                for &[dr, dc] in &[[-1i32,-1i32],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]] {
                    for s in 1..8i32 {
                        let nr = r as i32 + dr * s;
                        let nc = c as i32 + dc * s;
                        if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                        let t = b[nr as usize][nc as usize];
                        if t == EMPTY { ms.push([nr as usize, nc as usize]); }
                        else { if is_white(t) != wturn { ms.push([nr as usize, nc as usize]); } break; }
                    }
                }
                ms
            }
            b'k' => {
                let mut ms: Vec<[usize; 2]> = [[-1i32,-1i32],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
                    .iter().filter_map(|&[dr, dc]| {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                            let t = b[nr as usize][nc as usize];
                            if t == EMPTY || is_white(t) != wturn { Some([nr as usize, nc as usize]) } else { None }
                        } else { None }
                    }).collect();

                if !in_check {
                    let rk = if wturn { b'R' } else { b'r' };
                    let cr_ks = if wturn { 0 } else { 2 };
                    if cr[cr_ks]
                        && b[back][7] == rk
                        && b[back][5] == EMPTY
                        && b[back][6] == EMPTY
                        && !is_attacked(b, back, 4, opp)
                        && !is_attacked(b, back, 5, opp)
                        && !is_attacked(b, back, 6, opp)
                    {
                        ms.push([back, 6]);
                    }
                    let cr_ql = if wturn { 1 } else { 3 };
                    if cr[cr_ql]
                        && b[back][0] == rk
                        && b[back][1] == EMPTY
                        && b[back][2] == EMPTY
                        && b[back][3] == EMPTY
                        && !is_attacked(b, back, 4, opp)
                        && !is_attacked(b, back, 3, opp)
                        && !is_attacked(b, back, 2, opp)
                    {
                        ms.push([back, 2]);
                    }
                }
                ms
            }
            _ => vec![]
        };

        for &[nr, nc] in &pseudo {
            let mut b2 = *b;
            let castle = pt == b'k' && (nc as i32 - c as i32).abs() == 2;
            if castle {
                b2[nr][nc] = p; b2[r][c] = EMPTY;
                if nc > c { b2[r][5] = b2[r][7]; b2[r][7] = EMPTY; }
                else       { b2[r][3] = b2[r][0]; b2[r][0] = EMPTY; }
            } else {
                if pt == b'p' && nc != c && b[nr][nc] == EMPTY {
                    let cap_row = if wturn { nr + 1 } else { nr.wrapping_sub(1) };
                    if cap_row < 8 { b2[cap_row][nc] = EMPTY; }
                }
                b2[nr][nc] = p; b2[r][c] = EMPTY;
                if pt == b'p' && (nr == 0 || nr == 7) {
                    b2[nr][nc] = if wturn { b'Q' } else { b'q' };
                }
            }
            let (kr2, kc2) = if pt == b'k' { (nr, nc) } else { (kr, kc) };
            if !is_attacked(&b2, kr2, kc2, opp) {
                result.push([r, c, nr, nc]);
            }
        }
    }}
    result
}

#[derive(Clone)]
struct TTEntry { key: u64, depth: i32, score: i32, flag: u8, best_move: Option<[usize; 4]> }
const TT_EXACT: u8 = 0;
const TT_ALPHA: u8 = 1;
const TT_BETA:  u8 = 2;

struct TT { entries: Vec<Option<TTEntry>>, mask: usize }
impl TT {
    fn new(mb: usize) -> Self {
        let size = (mb * 1024 * 1024 / 40).next_power_of_two();
        TT { entries: vec![None; size], mask: size - 1 }
    }
    fn idx(&self, key: u64) -> usize { (key as usize) & self.mask }
    fn store(&mut self, key: u64, depth: i32, score: i32, flag: u8, best_move: Option<[usize; 4]>) {
        let idx = self.idx(key);
        let replace = match &self.entries[idx] {
            None => true,
            Some(e) => e.key != key || e.depth <= depth || flag == TT_EXACT,
        };
        if replace {
            self.entries[idx] = Some(TTEntry { key, depth, score, flag, best_move });
        }
    }
    fn get(&self, key: u64) -> Option<&TTEntry> {
        let idx = self.idx(key);
        self.entries[idx].as_ref().and_then(|e| if e.key == key { Some(e) } else { None })
    }
    fn resize(&mut self, mb: usize) {
        let size = (mb * 1024 * 1024 / 40).next_power_of_two();
        self.entries = vec![None; size];
        self.mask = size - 1;
    }
}

fn piece_to_idx(pt: u8) -> usize {
    match pt { b'p' => 1, b'n' => 2, b'b' => 3, b'r' => 4, b'q' => 5, b'k' => 6, _ => 0 }
}
fn from_to_key(sr: usize, sc: usize, er: usize, ec: usize) -> (usize, usize) {
    (sr * 8 + sc, er * 8 + ec)
}

const CORR_HIST_SIZE: usize = 16384;
fn corr_idx(h: u64, side: bool) -> usize {
    let k = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(if side { 1 } else { 0 });
    k as usize & (CORR_HIST_SIZE - 1)
}

pub struct Searcher {
    tt: TT,
    killers: [[Option<[usize; 4]>; 2]; MAX_PLY],
    history: [[i32; 64]; 64],
    counter_move: [[Option<[usize; 4]>; 64]; 13],
    corr_hist: [i32; CORR_HIST_SIZE * 2],
    rep_stack: Vec<u64>,
    rep_stack_len: usize,
    pub tt_mb: usize,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            tt: TT::new(128),
            killers: [[None; 2]; MAX_PLY],
            history: [[0i32; 64]; 64],
            counter_move: [[None; 64]; 13],
            corr_hist: [0i32; CORR_HIST_SIZE * 2],
            rep_stack: Vec::with_capacity(512),
            rep_stack_len: 0,
            tt_mb: 128,
        }
    }

    pub fn resize_tt(&mut self, mb: usize) {
        self.tt.resize(mb);
        self.tt_mb = mb;
    }

    fn piece_val(pt: u8) -> i32 {
        match pt { b'p'=>100, b'n'=>325, b'b'=>340, b'r'=>500, b'q'=>950, _=>0 }
    }

    fn corrected_eval(&self, st: &BoardState) -> i32 {
        let base = evaluate(&st.b) * if st.w { 1 } else { -1 };
        let ph = compute_pawn_hash(&st.b);
        let idx = corr_idx(ph, st.w);
        let corr = self.corr_hist[idx];
        base + corr.clamp(-200, 200)
    }

    fn is_repetition(&self) -> bool {
        if self.rep_stack_len < 4 { return false; }
        let last = self.rep_stack[self.rep_stack_len - 1];
        let mut count = 0;
        let start = if self.rep_stack_len > 20 { self.rep_stack_len - 20 } else { 0 };
        for i in (start..self.rep_stack_len - 1).rev() {
            if self.rep_stack[i] == last {
                count += 1;
                if count >= 1 { return true; }
            }
        }
        false
    }

    fn qsearch(&mut self, st: &mut BoardState, mut alpha: i32, beta: i32, depth: i32,
               start: Instant, tl: f64, cnt: &mut u64) -> i32 {
        *cnt += 1;
        if start.elapsed().as_secs_f64() > tl { return 0; }
        let (kr, kc) = find_king(&st.b, st.w);
        let in_check = is_attacked(&st.b, kr, kc, !st.w);

        if !in_check {
            let stand = self.corrected_eval(st);
            if stand >= beta { return stand; }
            if stand > alpha { alpha = stand; }
            if depth <= 0 { return alpha; }
            let big_delta = 975;
            if alpha - big_delta > stand { return alpha; }
        } else if depth <= -8 {
            return self.corrected_eval(st);
        }

        let moves = generate_moves(&st.b, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + 1000 } else { alpha };
        }

        let mut caps: Vec<[usize; 4]> = if in_check {
            moves
        } else {
            moves.into_iter().filter(|mv| {
                st.b[mv[2]][mv[3]] != EMPTY ||
                (ptype(st.b[mv[0]][mv[1]]) == b'p' && (mv[2] == 0 || mv[2] == 7))
            }).collect()
        };
        if caps.is_empty() { return alpha; }

        caps.sort_by_key(|mv| {
            let victim = Self::piece_val(ptype(st.b[mv[2]][mv[3]]));
            let attacker = Self::piece_val(ptype(st.b[mv[0]][mv[1]]));
            -(victim * 10 - attacker)
        });

        for mv in caps {
            if start.elapsed().as_secs_f64() > tl { return 0; }
            if !in_check && see(&st.b, mv[0], mv[1], mv[2], mv[3]) < 0 { continue; }
            let old = *st;
            apply_move(st, mv[0], mv[1], mv[2], mv[3], 0);
            let score = -self.qsearch(st, -beta, -alpha, depth - 1, start, tl, cnt);
            *st = old;
            if score >= beta { return score; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    fn negamax(&mut self, st: &mut BoardState, depth: i32, ply: usize,
               mut alpha: i32, beta: i32, can_null: bool,
               start: Instant, tl: f64, cnt: &mut u64) -> i32 {
        *cnt += 1;
        if start.elapsed().as_secs_f64() > tl { return 0; }
        if ply >= MAX_PLY { return self.corrected_eval(st); }

        let h = compute_hash(&st.b, st.w, &st.cr, st.ep);
        let (kr, kc) = find_king(&st.b, st.w);
        let in_check = is_attacked(&st.b, kr, kc, !st.w);
        let is_pv = beta - alpha > 1;
        let is_root = ply == 0;

        if !is_root && (ply >= 2 && self.is_repetition()) { return 0; }

        let ext = if in_check { 1 } else { 0 };
        let actual_depth = depth + ext;

        let tt_entry = self.tt.get(h);
        let tt_move = tt_entry.and_then(|e| e.best_move);
        let tt_score = tt_entry.map(|e| e.score);
        let tt_depth = tt_entry.map(|e| e.depth).unwrap_or(-1);
        let tt_flag = tt_entry.map(|e| e.flag);

        if !is_pv && tt_depth >= actual_depth && tt_flag.is_some() {
            let s = tt_score.unwrap();
            match tt_flag.unwrap() {
                TT_EXACT => return s,
                TT_ALPHA => if s <= alpha { return alpha; },
                TT_BETA  => if s >= beta  { return beta; },
                _ => {}
            }
        }

        if actual_depth <= 0 {
            return self.qsearch(st, alpha, beta, QS_DEPTH, start, tl, cnt);
        }

        let eval_score = self.corrected_eval(st);

        if !in_check && !is_pv && actual_depth <= 8 && ply > 0 {
            let margin = 80 + 65 * actual_depth;
            if eval_score - margin >= beta { return eval_score - margin; }
        }

        if !in_check && !is_pv && actual_depth <= 3 && ply > 0 {
            let margin = 150 * actual_depth;
            if eval_score + margin <= alpha {
                let q = self.qsearch(st, alpha - margin, beta - margin, QS_DEPTH, start, tl, cnt);
                if q + margin <= alpha { return alpha; }
            }
        }

        if !in_check && can_null && !is_pv && ply > 0 && actual_depth >= 3 && has_non_pawn(&st.b, st.w) {
            if eval_score >= beta {
                let r = 3 + actual_depth / 4 + ((eval_score - beta) / 200).min(3);
                let ow = st.w; let oe = st.ep;
                st.w = !st.w; st.ep = None;
                let null_h = compute_hash(&st.b, st.w, &st.cr, st.ep);
                self.rep_stack.push(null_h);
                self.rep_stack_len += 1;
                let s = -self.negamax(st, actual_depth - r - 1, ply + 1, -beta, -beta + 1, false, start, tl, cnt);
                self.rep_stack.pop();
                self.rep_stack_len -= 1;
                st.w = ow; st.ep = oe;
                if start.elapsed().as_secs_f64() > tl { return 0; }
                if s >= beta { return beta; }
            }
        }

        let moves = generate_moves(&st.b, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let actual_depth = if tt_move.is_none() && actual_depth >= 4 && is_pv {
            actual_depth - 1
        } else {
            actual_depth
        };

        let mut scored: Vec<(i32, [usize; 4])> = moves.into_iter().map(|mv| {
            let mut s = 0i32;
            if Some(mv) == tt_move { s = 10_000_000; }
            else {
                let t = st.b[mv[2]][mv[3]];
                let is_promo = ptype(st.b[mv[0]][mv[1]]) == b'p' && (mv[2] == 0 || mv[2] == 7);
                if t != EMPTY || is_promo {
                    let v = Self::piece_val(ptype(t));
                    let a = Self::piece_val(ptype(st.b[mv[0]][mv[1]]));
                    let see_sc = see(&st.b, mv[0], mv[1], mv[2], mv[3]);
                    if see_sc >= 0 {
                        s += 2_000_000 + v * 10 - a + see_sc;
                    } else {
                        s += 500_000 + v * 10 - a;
                    }
                    if is_promo { s += 1_500_000; }
                } else {
                    if self.killers[ply][0] == Some(mv) { s += 900_000; }
                    else if self.killers[ply][1] == Some(mv) { s += 800_000; }
                    let prev_p = st.b[mv[0]][mv[1]];
                    let p_idx = piece_to_idx(ptype(prev_p));
                    if self.counter_move[p_idx][mv[2]*8+mv[3]] == Some(mv) { s += 700_000; }
                    let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], mv[3]);
                    s += self.history[fk][tk].clamp(-32768, 32768);
                }
            }
            (s, mv)
        }).collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let lmp_count = if !is_pv && !in_check && actual_depth <= 8 {
            match actual_depth { 1 => 4, 2 => 7, 3 => 11, 4 => 17, 5 => 24, 6 => 33, 7 => 44, 8 => 57, _ => usize::MAX }
        } else { usize::MAX };

        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = scored.first().map(|&(_, mv)| mv);
        let mut quiets_tried: Vec<[usize; 4]> = Vec::new();

        for (i, &(_, mv)) in scored.iter().enumerate() {
            if start.elapsed().as_secs_f64() > tl { return 0; }

            let capture = st.b[mv[2]][mv[3]] != EMPTY;
            let is_promo = ptype(st.b[mv[0]][mv[1]]) == b'p' && (mv[2] == 0 || mv[2] == 7);
            let is_quiet = !capture && !is_promo;
            let gives_check = {
                let mut b2 = st.b;
                let p2 = b2[mv[0]][mv[1]];
                b2[mv[2]][mv[3]] = p2; b2[mv[0]][mv[1]] = EMPTY;
                if ptype(p2) == b'p' && mv[1] != mv[3] && st.b[mv[2]][mv[3]] == EMPTY {
                    let cap_row = if st.w { mv[2] + 1 } else { mv[2].wrapping_sub(1) };
                    if cap_row < 8 { b2[cap_row][mv[3]] = EMPTY; }
                }
                let (opp_kr, opp_kc) = find_king(&b2, !st.w);
                is_attacked(&b2, opp_kr, opp_kc, st.w)
            };

            if !is_pv && !in_check && is_quiet && i >= lmp_count { break; }

            if !is_pv && !in_check && i > 0 && best_score > -MATE / 2 {
                if capture {
                    let see_sc = see(&st.b, mv[0], mv[1], mv[2], mv[3]);
                    if see_sc < -80 * actual_depth { continue; }
                } else if is_quiet {
                    let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], mv[3]);
                    let hist = self.history[fk][tk];
                    if actual_depth <= 5 && hist < -1024 * actual_depth { continue; }
                }
            }

            let move_ext = if gives_check && !in_check && i == 0 { 1 } else { 0 };

            let old = *st;
            apply_move(st, mv[0], mv[1], mv[2], mv[3], 0);
            let h_after = compute_hash(&st.b, st.w, &st.cr, st.ep);
            self.rep_stack.push(h_after);
            self.rep_stack_len += 1;

            let new_depth = actual_depth - 1 + move_ext;

            let lmr_eligible = i >= 2 && actual_depth >= 3 && is_quiet && !in_check && !gives_check;
            let s = if i == 0 {
                -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
            } else if lmr_eligible {
                let r = {
                    let base = (0.5 + (i as f64).ln() * (actual_depth as f64).ln() / 1.8) as i32;
                    let r = base.min(actual_depth - 1).max(1);
                    if !is_pv { (r + 1).min(actual_depth - 1) } else { r }
                };
                let s2 = -self.negamax(st, new_depth - r, ply + 1, -alpha - 1, -alpha, true, start, tl, cnt);
                if s2 > alpha {
                    let s3 = -self.negamax(st, new_depth, ply + 1, -alpha - 1, -alpha, true, start, tl, cnt);
                    if s3 > alpha && is_pv {
                        -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
                    } else { s3 }
                } else { s2 }
            } else if is_pv {
                let s2 = -self.negamax(st, new_depth, ply + 1, -alpha - 1, -alpha, true, start, tl, cnt);
                if s2 > alpha && s2 < beta {
                    -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
                } else { s2 }
            } else {
                -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
            };

            self.rep_stack.pop();
            self.rep_stack_len -= 1;
            *st = old;

            if is_quiet { quiets_tried.push(mv); }

            if s > best_score {
                best_score = s;
                if s > alpha {
                    alpha = s;
                    best_move = Some(mv);
                    if alpha >= beta {
                        if is_quiet {
                            if self.killers[ply][0] != Some(mv) {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = Some(mv);
                            }
                            let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], mv[3]);
                            let bonus = (actual_depth * actual_depth).min(512);
                            self.history[fk][tk] += bonus;
                            if self.history[fk][tk] > 16384 {
                                for a in 0..64 { for b in 0..64 { self.history[a][b] /= 2; } }
                            }
                            for &qmv in &quiets_tried {
                                if qmv == mv { continue; }
                                let (qfk, qtk) = from_to_key(qmv[0], qmv[1], qmv[2], qmv[3]);
                                self.history[qfk][qtk] -= bonus;
                                if self.history[qfk][qtk] < -16384 {
                                    for a in 0..64 { for b in 0..64 { self.history[a][b] /= 2; } }
                                }
                            }
                            let prev_p = old.b[mv[0]][mv[1]];
                            let p_idx = piece_to_idx(ptype(prev_p));
                            self.counter_move[p_idx][tk] = Some(mv);
                        }
                        break;
                    }
                }
            }
        }

        if !in_check && actual_depth >= 3 && best_score != -INF {
            let ev = self.corrected_eval(st);
            let diff = best_score - ev;
            if diff.abs() < 500 {
                let ph = compute_pawn_hash(&st.b);
                let idx = corr_idx(ph, st.w);
                let corr = &mut self.corr_hist[idx];
                *corr = (*corr + diff.clamp(-64, 64) / 8).clamp(-1024, 1024);
            }
        }

        let flag = if best_score <= orig_alpha { TT_ALPHA }
                   else if best_score >= beta  { TT_BETA }
                   else                        { TT_EXACT };
        self.tt.store(h, actual_depth, best_score, flag, best_move);
        best_score
    }
}

fn compute_pawn_hash(b: &Board8) -> u64 {
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

pub struct Engine {
    pub st: BoardState,
    pub searcher: Searcher,
}

impl Engine {
    pub fn new() -> Self {
        let mut e = Engine {
            st: BoardState { b: [[EMPTY; 8]; 8], w: true, cr: [false; 4], ep: None, mc: 0 },
            searcher: Searcher::new(),
        };
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let h = compute_hash(&e.st.b, e.st.w, &e.st.cr, e.st.ep);
        e.searcher.rep_stack.push(h);
        e.searcher.rep_stack_len = 1;
        e
    }

    pub fn set_fen(&mut self, fen: &str) {
        let parts: Vec<&str> = fen.split(' ').collect();
        self.st.b = [[EMPTY; 8]; 8];
        for (ri, rs) in parts[0].split('/').enumerate() {
            let mut ci = 0;
            for ch in rs.chars() {
                if ch.is_ascii_digit() {
                    ci += ch.to_digit(10).unwrap() as usize;
                } else {
                    self.st.b[ri][ci] = ch as u8;
                    ci += 1;
                }
            }
        }
        self.st.w = parts.len() > 1 && parts[1] == "w";
        self.st.cr = [false; 4];
        if parts.len() > 2 {
            let r = parts[2];
            self.st.cr[0] = r.contains('K');
            self.st.cr[1] = r.contains('Q');
            self.st.cr[2] = r.contains('k');
            self.st.cr[3] = r.contains('q');
        }
        self.st.ep = if parts.len() > 3 && parts[3] != "-" {
            let b = parts[3].as_bytes();
            if b.len() >= 2 {
                let col = (b[0] - b'a') as usize;
                let row = 8usize.wrapping_sub((b[1] - b'0') as usize);
                if row < 8 { Some((row, col)) } else { None }
            } else { None }
        } else { None };
        self.st.mc = if parts.len() > 4 { parts[4].parse().unwrap_or(0) } else { 0 };
        self.searcher.rep_stack.clear();
        self.searcher.rep_stack_len = 0;
        let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len = 1;
    }

    pub fn make_move_uci(&mut self, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
        apply_move(&mut self.st, sr, sc, er, ec, promotion);
        let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len += 1;
    }

    pub fn is_check(&self) -> bool {
        let (kr, kc) = find_king(&self.st.b, self.st.w);
        is_attacked(&self.st.b, kr, kc, !self.st.w)
    }

    pub fn find_best_move(&mut self, time_limit: f64, depth_limit: i32) -> (String, i32, u64, f64) {
        let moves = generate_moves(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        if moves.is_empty() {
            let (kr, kc) = find_king(&self.st.b, self.st.w);
            let in_check = is_attacked(&self.st.b, kr, kc, !self.st.w);
            if in_check {
                eprintln!("info depth 0 score mate 0");
                println!("info depth 0 score mate 0");
                return ("0000".into(), -MATE, 0, 0.0);
            } else {
                eprintln!("info depth 0 score cp 0");
                println!("info depth 0 score cp 0");
                return ("0000".into(), 0, 0, 0.0);
            }
        }

        self.searcher.killers = [[None; 2]; MAX_PLY];
        self.searcher.history = [[0i32; 64]; 64];

        let start = Instant::now();
        let mut best_move = moves[0];
        let mut best_score = 0i32;
        let mut total_nodes = 0u64;

        let init_eval = self.searcher.corrected_eval(&self.st);
        let mut prev_score = init_eval;

        for depth in 1..=depth_limit {
            if start.elapsed().as_secs_f64() > time_limit { break; }

            let mut nd = 0u64;

            let init_delta = if depth >= 5 { 25 } else { INF };
            let mut asp_delta = init_delta;
            let (mut alpha, mut beta) = if asp_delta < INF {
                (prev_score - asp_delta, prev_score + asp_delta)
            } else {
                (-INF, INF)
            };

            let mut asp_best = best_move;
            let mut asp_score = -INF;

            'asp: loop {
                let mut sorted_moves = moves.clone();
                if asp_best != moves[0] {
                    if let Some(pos) = sorted_moves.iter().position(|&m| m == asp_best) {
                        sorted_moves.swap(0, pos);
                    }
                }

                let mut current_best = sorted_moves[0];
                let mut current_score = -INF;
                let mut loop_alpha = alpha;

                for &mv in &sorted_moves {
                    if start.elapsed().as_secs_f64() > time_limit { break; }
                    let old = self.st;
                    apply_move(&mut self.st, mv[0], mv[1], mv[2], mv[3], 0);
                    let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
                    self.searcher.rep_stack.push(h);
                    self.searcher.rep_stack_len += 1;

                    let score = if current_score == -INF {
                        -self.searcher.negamax(&mut self.st, depth - 1, 1, -beta, -loop_alpha, true, start, time_limit, &mut nd)
                    } else {
                        let s = -self.searcher.negamax(&mut self.st, depth - 1, 1, -loop_alpha - 1, -loop_alpha, true, start, time_limit, &mut nd);
                        if s > loop_alpha && s < beta {
                            -self.searcher.negamax(&mut self.st, depth - 1, 1, -beta, -loop_alpha, true, start, time_limit, &mut nd)
                        } else { s }
                    };

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;

                    if score > current_score {
                        current_score = score;
                        current_best = mv;
                    }
                    if score > loop_alpha { loop_alpha = score; }
                    if loop_alpha >= beta { break; }
                }

                if start.elapsed().as_secs_f64() > time_limit { break 'asp; }

                if current_score <= alpha {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    alpha = (prev_score - asp_delta).max(-INF);
                    beta = prev_score + init_delta;
                    continue 'asp;
                }
                if current_score >= beta {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    beta = (prev_score + asp_delta).min(INF);
                    asp_best = current_best;
                    continue 'asp;
                }
                asp_best = current_best;
                asp_score = current_score;
                break;
            }

            total_nodes += nd;
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed <= time_limit {
                best_move = asp_best;
                best_score = asp_score;
                prev_score = best_score;
                let nps = if elapsed > 0.0 { (total_nodes as f64 / elapsed) as i64 } else { 0 };
                let score_str = if best_score.abs() > 90_000 {
                    let mate_in = (MATE - best_score.abs()) / 2 + 1;
                    if best_score > 0 { format!("mate {}", mate_in) } else { format!("mate -{}", mate_in) }
                } else {
                    format!("cp {}", best_score)
                };
                let pv = format!("{}{}", coord_to_square(best_move[0], best_move[1]),
                                         coord_to_square(best_move[2], best_move[3]));
                eprintln!("info depth {} score {} nodes {} nps {} pv {}", depth, score_str, total_nodes, nps, pv);
                println!("info depth {} score {} nodes {} nps {} pv {}", depth, score_str, total_nodes, nps, pv);
            } else {
                break;
            }
        }

        let mv_str = format!("{}{}", coord_to_square(best_move[0], best_move[1]),
                                     coord_to_square(best_move[2], best_move[3]));
        (mv_str, best_score, total_nodes, start.elapsed().as_secs_f64())
    }
}