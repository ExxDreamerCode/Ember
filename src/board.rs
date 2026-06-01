pub const EMPTY: u8 = b'.';
pub const MATE: i32 = 100_000;
pub const INF: i32 = 1_000_000;
pub const MAX_PLY: usize = 128;
pub const QS_DEPTH: i32 = 8;

pub type Board8 = [[u8; 8]; 8];

#[derive(Clone, Copy)]
pub struct BoardState {
    pub b: Board8,
    pub w: bool,
    pub cr: [bool; 4],
    pub ep: Option<(usize, usize)>,
    pub mc: usize,
}

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