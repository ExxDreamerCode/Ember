use std::time::Instant;
use crate::board::{BoardState, EMPTY, MATE, INF, MAX_PLY, QS_DEPTH, ptype, find_king, is_attacked, has_non_pawn, see};
use crate::evaluate::evaluate;
use crate::zobrist::{compute_hash, compute_pawn_hash};
use crate::movegen::{apply_move, generate_moves};
use crate::tt::{TT, TT_EXACT, TT_ALPHA, TT_BETA};

fn piece_val(pt: u8) -> i32 {
    match pt { b'p'=>100, b'n'=>325, b'b'=>340, b'r'=>500, b'q'=>950, _=>0 }
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
    pub tt: TT,
    pub killers: [[Option<[usize; 4]>; 2]; MAX_PLY],
    pub history: [[i32; 64]; 64],
    pub counter_move: [[Option<[usize; 4]>; 64]; 13],
    pub corr_hist: [i32; CORR_HIST_SIZE * 2],
    pub rep_stack: Vec<u64>,
    pub rep_stack_len: usize,
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

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
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
            let victim = piece_val(ptype(st.b[mv[2]][mv[3]]));
            let attacker = piece_val(ptype(st.b[mv[0]][mv[1]]));
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

    pub fn negamax(&mut self, st: &mut BoardState, depth: i32, ply: usize,
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
                    let v = piece_val(ptype(t));
                    let a = piece_val(ptype(st.b[mv[0]][mv[1]]));
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