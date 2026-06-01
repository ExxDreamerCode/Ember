use crate::board::{Board8, BoardState, EMPTY, is_white, ptype, find_king, is_attacked};

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