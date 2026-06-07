use std::time::Instant;
use crate::board::{
    BoardState, MATE, INF, MAX_PLY, QS_DEPTH,
    piece_on, piece_type, is_white_piece, EMPTY_SQ,
    has_non_pawn, see, all_occ, bit, attacked_by, KING_ATTACKS,
};
use crate::evaluate::evaluate;
use crate::zobrist::{compute_hash, compute_pawn_hash};
use crate::movegen::{apply_move, generate_moves};
use crate::tt::{TT, TT_EXACT, TT_ALPHA, TT_BETA};

fn piece_val(pt: u8) -> i32 {
    match pt { 0=>100, 1=>325, 2=>340, 3=>500, 4=>950, _=>0 }
}

fn piece_to_idx(pt: u8) -> usize {
    match pt { 0=>1, 1=>2, 2=>3, 3=>4, 4=>5, 5=>6, _=>0 }
}

fn from_to_key(sr: usize, sc: usize, er: usize, ec: usize) -> (usize, usize) {
    (sr * 8 + sc, er * 8 + ec)
}

const CORR_HIST_SIZE: usize = 16384;
fn corr_idx(h: u64, side: bool) -> usize {
    let k = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(if side { 1 } else { 0 });
    k as usize & (CORR_HIST_SIZE - 1)
}

fn king_zone_pressure(st: &BoardState, white: bool) -> u32 {
    let ks = st.king_sq(white);
    let zone = KING_ATTACKS[ks] | bit(ks);
    let occ = all_occ(&st.bb);
    (attacked_by(&st.bb, occ, !white) & zone).count_ones()
}

fn tactical_king_pressure(st: &BoardState) -> u32 {
    king_zone_pressure(st, true).max(king_zone_pressure(st, false))
}

pub struct Searcher {
    pub tt:           TT,
    pub killers:      [[Option<[usize; 4]>; 2]; MAX_PLY],
    pub history:      [[i32; 64]; 64],
    pub counter_move: [[Option<[usize; 4]>; 64]; 13],
    pub corr_hist:    [i32; CORR_HIST_SIZE * 2],
    pub rep_stack:    Vec<u64>,
    pub rep_stack_len: usize,
    pub tt_mb:        usize,
    #[cfg(feature = "search-debug")]
    pub debug:        SearchDebug,
}

#[cfg(feature = "search-debug")]
pub struct SearchDebug {
    pub disable_corr_hist: bool,
    pub disable_futility: bool,
    pub disable_history_pruning: bool,
    pub disable_iid_reduction: bool,
    pub disable_lmp: bool,
    pub disable_lmr: bool,
    pub disable_null_move: bool,
    pub disable_reverse_futility: bool,
    pub disable_see_pruning: bool,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            tt:           TT::new(128),
            killers:      [[None; 2]; MAX_PLY],
            history:      [[0i32; 64]; 64],
            counter_move: [[None; 64]; 13],
            corr_hist:    [0i32; CORR_HIST_SIZE * 2],
            rep_stack:    Vec::with_capacity(512),
            rep_stack_len: 0,
            tt_mb:        128,
            #[cfg(feature = "search-debug")]
            debug:        SearchDebug::from_env(),
        }
    }

    pub fn resize_tt(&mut self, mb: usize) {
        self.tt.resize(mb);
        self.tt_mb = mb;
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn corr_hist_enabled(&self) -> bool { !self.debug.disable_corr_hist }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn corr_hist_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn futility_enabled(&self) -> bool { !self.debug.disable_futility }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn futility_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn history_pruning_enabled(&self) -> bool { !self.debug.disable_history_pruning }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn history_pruning_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn iid_reduction_enabled(&self) -> bool { !self.debug.disable_iid_reduction }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn iid_reduction_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn lmp_enabled(&self) -> bool { !self.debug.disable_lmp }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmp_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn lmr_enabled(&self) -> bool { !self.debug.disable_lmr }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmr_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn null_move_enabled(&self) -> bool { !self.debug.disable_null_move }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn null_move_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn reverse_futility_enabled(&self) -> bool { !self.debug.disable_reverse_futility }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn reverse_futility_enabled(&self) -> bool { true }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn see_pruning_enabled(&self) -> bool { !self.debug.disable_see_pruning }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn see_pruning_enabled(&self) -> bool { true }

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
        let base = evaluate(st) * if st.w { 1 } else { -1 };
        if self.corr_hist_enabled() {
            let ph   = compute_pawn_hash(st);
            let idx  = corr_idx(ph, st.w);
            let corr = self.corr_hist[idx];
            return base + corr.clamp(-200, 200);
        }
        base
    }

    pub fn update_correction_history(&mut self, st: &BoardState, score: i32, depth: i32) {
        if !self.corr_hist_enabled() || depth < 3 || score.abs() > MATE / 2 {
            return;
        }
        let ev = self.corrected_eval(st);
        let diff = score - ev;
        if diff.abs() < 500 {
            let ph = compute_pawn_hash(st);
            let idx = corr_idx(ph, st.w);
            let corr = &mut self.corr_hist[idx];
            *corr = (*corr + diff.clamp(-64, 64) / 8).clamp(-1024, 1024);
        }
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
        let ks = st.king_sq(st.w);
        let in_check = crate::board::is_attacked(&st.bb, ks, !st.w);

        if !in_check {
            let stand = self.corrected_eval(st);
            if stand >= beta  { return stand; }
            if stand > alpha  { alpha = stand; }
            if depth <= 0     { return alpha; }
            if alpha - 975 > stand { return alpha; }
        } else if depth <= -8 {
            return self.corrected_eval(st);
        }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + 1000 } else { alpha };
        }

        let mut caps: Vec<[usize; 4]> = if in_check {
            moves
        } else {
            moves.into_iter().filter(|mv| {
                let to = mv[2]*8 + mv[3];
                piece_on(&st.bb, to) != EMPTY_SQ ||
                (piece_type(piece_on(&st.bb, mv[0]*8+mv[1])) == 0 && (mv[2]==0||mv[2]==7))
            }).collect()
        };
        if caps.is_empty() { return alpha; }

        caps.sort_by_key(|mv| {
            let to = mv[2]*8+mv[3]; let from = mv[0]*8+mv[1];
            let vpi = piece_on(&st.bb, to);
            let api = piece_on(&st.bb, from);
            let victim   = if vpi != EMPTY_SQ { piece_val(piece_type(vpi)) } else { 0 };
            let attacker = if api != EMPTY_SQ { piece_val(piece_type(api)) } else { 0 };
            -(victim * 10 - attacker)
        });

        for mv in caps {
            if start.elapsed().as_secs_f64() > tl { return 0; }
            let from = mv[0]*8+mv[1]; let to = mv[2]*8+mv[3];
            if !in_check && see(&st.bb, from, to) < 0 { continue; }
            let old = *st;
            apply_move(st, mv[0], mv[1], mv[2], mv[3], 0);
            let score = -self.qsearch(st, -beta, -alpha, depth - 1, start, tl, cnt);
            *st = old;
            if score >= beta  { return score; }
            if score > alpha  { alpha = score; }
        }
        alpha
    }

    pub fn negamax(&mut self, st: &mut BoardState, depth: i32, ply: usize,
                   mut alpha: i32, beta: i32, can_null: bool,
                   start: Instant, tl: f64, cnt: &mut u64) -> i32 {
        *cnt += 1;
        if start.elapsed().as_secs_f64() > tl { return 0; }
        if ply >= MAX_PLY { return self.corrected_eval(st); }

        let h = compute_hash(st);
        let ks = st.king_sq(st.w);
        let in_check = crate::board::is_attacked(&st.bb, ks, !st.w);
        let is_pv  = beta - alpha > 1;
        let is_root = ply == 0;
        let king_pressure = if in_check { 8 } else { tactical_king_pressure(st) };

        if !is_root && ply >= 2 && self.is_repetition() { return 0; }

        let ext = if in_check { 1 } else { 0 };
        let actual_depth = depth + ext;

        let tt_entry = self.tt.get(h);
        let tt_move  = tt_entry.and_then(|e| e.best_move);
        let tt_score = tt_entry.map(|e| e.score);
        let tt_depth = tt_entry.map(|e| e.depth).unwrap_or(-1);
        let tt_flag  = tt_entry.map(|e| e.flag);

        if !is_pv && tt_depth >= actual_depth && tt_flag.is_some() {
            let s = tt_score.unwrap();
            match tt_flag.unwrap() {
                TT_EXACT => return s,
                TT_ALPHA => if s <= alpha { return alpha; },
                TT_BETA  => if s >= beta  { return beta;  },
                _ => {}
            }
        }

        if actual_depth <= 0 {
            return self.qsearch(st, alpha, beta, QS_DEPTH, start, tl, cnt);
        }

        let eval_score = self.corrected_eval(st);

        if self.reverse_futility_enabled() && !in_check && !is_pv && actual_depth <= 8 && ply > 0 {
            let margin = 80 + 65 * actual_depth;
            if eval_score - margin >= beta { return eval_score - margin; }
        }
        if self.futility_enabled() && !in_check && !is_pv && actual_depth <= 3 && ply > 0 {
            let margin = 150 * actual_depth;
            if eval_score + margin <= alpha {
                let q = self.qsearch(st, alpha - margin, beta - margin, QS_DEPTH, start, tl, cnt);
                if q + margin <= alpha { return alpha; }
            }
        }
        if self.null_move_enabled() && king_pressure < 3 && !in_check && can_null && !is_pv && ply > 0 && actual_depth >= 3
            && has_non_pawn(&st.bb, st.w) && eval_score >= beta
        {
            let r = 3 + actual_depth / 4 + ((eval_score - beta) / 200).min(3);
            let ow = st.w; let oe = st.ep;
            st.w = !st.w; st.ep = None;
            let null_h = compute_hash(st);
            self.rep_stack.push(null_h);
            self.rep_stack_len += 1;
            let s = -self.negamax(st, actual_depth - r - 1, ply + 1, -beta, -beta + 1,
                                  false, start, tl, cnt);
            self.rep_stack.pop();
            self.rep_stack_len -= 1;
            st.w = ow; st.ep = oe;
            if start.elapsed().as_secs_f64() > tl { return 0; }
            if s >= beta { return beta; }
        }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let actual_depth = if self.iid_reduction_enabled() && tt_move.is_none() && actual_depth >= 4 && is_pv {
            actual_depth - 1
        } else { actual_depth };

        let mut scored: Vec<(i32, [usize; 4])> = moves.into_iter().map(|mv| {
            let mut s = 0i32;
            if Some(mv) == tt_move { s = 10_000_000; }
            else {
                let from = mv[0]*8+mv[1]; let to = mv[2]*8+mv[3];
                let tpi  = piece_on(&st.bb, to);
                let fpi  = piece_on(&st.bb, from);
                let is_promo = fpi != EMPTY_SQ && piece_type(fpi) == 0 && (mv[2]==0||mv[2]==7);
                if tpi != EMPTY_SQ || is_promo {
                    let v   = if tpi != EMPTY_SQ { piece_val(piece_type(tpi)) } else { 0 };
                    let a   = if fpi != EMPTY_SQ { piece_val(piece_type(fpi)) } else { 0 };
                    let see_sc = see(&st.bb, from, to);
                    if see_sc >= 0 { s += 2_000_000 + v * 10 - a + see_sc; }
                    else           { s += 500_000   + v * 10 - a; }
                    if is_promo    { s += 1_500_000; }
                } else {
                    if self.killers[ply][0] == Some(mv) { s += 900_000; }
                    else if self.killers[ply][1] == Some(mv) { s += 800_000; }
                    let p_idx = if fpi != EMPTY_SQ { piece_to_idx(piece_type(fpi)) } else { 0 };
                    if self.counter_move[p_idx][to] == Some(mv) { s += 700_000; }
                    let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], mv[3]);
                    s += self.history[fk][tk].clamp(-32768, 32768);
                }
            }
            (s, mv)
        }).collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let lmp_count = if self.lmp_enabled() && king_pressure < 3 && !is_pv && !in_check && actual_depth <= 8 {
            match actual_depth { 1=>4, 2=>7, 3=>11, 4=>17, 5=>24, 6=>33, 7=>44, 8=>57, _=>usize::MAX }
        } else { usize::MAX };

        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move  = scored.first().map(|&(_, mv)| mv);
        let mut quiets_tried: Vec<[usize; 4]> = Vec::new();

        for (i, &(_, mv)) in scored.iter().enumerate() {
            if start.elapsed().as_secs_f64() > tl { return 0; }

            let from = mv[0]*8+mv[1]; let to = mv[2]*8+mv[3];
            let fpi  = piece_on(&st.bb, from);
            let tpi  = piece_on(&st.bb, to);
            let capture  = tpi != EMPTY_SQ;
            let is_promo = fpi != EMPTY_SQ && piece_type(fpi) == 0 && (mv[2]==0||mv[2]==7);
            let is_quiet = !capture && !is_promo;

            let gives_check = {
                let mut bb2 = st.bb;
                let pi = piece_on(&bb2, from);
                if pi != EMPTY_SQ {
                    let cap = piece_on(&bb2, to);
                    if cap != EMPTY_SQ { bb2[cap as usize] &= !bit(to); }
                    if piece_type(pi) == 0 && mv[1] != mv[3] && cap == EMPTY_SQ {
                        let cap_sq = if st.w { to + 8 } else { to - 8 };
                        let cpi = piece_on(&bb2, cap_sq);
                        if cpi != EMPTY_SQ { bb2[cpi as usize] &= !bit(cap_sq); }
                    }
                    bb2[pi as usize] &= !bit(from);
                    bb2[pi as usize] |=  bit(to);
                    let _opp_ks = st.king_sq(!st.w);
                    let opp_k_pi = if st.w { crate::board::BK } else { crate::board::WK };
                    let opp_ks2 = if bb2[opp_k_pi] != 0 { bb2[opp_k_pi].trailing_zeros() as usize } else { 64 };
                    crate::board::is_attacked(&bb2, opp_ks2, st.w)
                } else { false }
            };

            if !is_pv && !in_check && is_quiet && i >= lmp_count { break; }
            if !is_pv && !in_check && i > 0 && best_score > -MATE / 2 {
                if capture {
                    if self.see_pruning_enabled() && see(&st.bb, from, to) < -80 * actual_depth { continue; }
                } else if is_quiet && self.history_pruning_enabled() {
                    let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], mv[3]);
                    if actual_depth <= 5 && self.history[fk][tk] < -1024 * actual_depth { continue; }
                }
            }

            let move_ext = if gives_check && !in_check && i == 0 { 1 } else { 0 };

            let old = *st;
            apply_move(st, mv[0], mv[1], mv[2], mv[3], 0);
            let h_after = compute_hash(st);
            self.rep_stack.push(h_after);
            self.rep_stack_len += 1;

            let new_depth = actual_depth - 1 + move_ext;

            let lmr_eligible = self.lmr_enabled() && i >= 2 && actual_depth >= 3 && is_quiet && !in_check && !gives_check;
            let s = if i == 0 {
                -self.negamax(st, new_depth, ply+1, -beta, -alpha, true, start, tl, cnt)
            } else if lmr_eligible {
                let r = {
                    let base = (0.5 + (i as f64).ln() * (actual_depth as f64).ln() / 1.8) as i32;
                    let r = base.min(actual_depth - 1).max(1);
                    if !is_pv { (r + 1).min(actual_depth - 1) } else { r }
                };
                let s2 = -self.negamax(st, new_depth - r, ply+1, -alpha-1, -alpha, true, start, tl, cnt);
                if s2 > alpha {
                    let s3 = -self.negamax(st, new_depth, ply+1, -alpha-1, -alpha, true, start, tl, cnt);
                    if s3 > alpha && is_pv {
                        -self.negamax(st, new_depth, ply+1, -beta, -alpha, true, start, tl, cnt)
                    } else { s3 }
                } else { s2 }
            } else if is_pv {
                let s2 = -self.negamax(st, new_depth, ply+1, -alpha-1, -alpha, true, start, tl, cnt);
                if s2 > alpha && s2 < beta {
                    -self.negamax(st, new_depth, ply+1, -beta, -alpha, true, start, tl, cnt)
                } else { s2 }
            } else {
                -self.negamax(st, new_depth, ply+1, -beta, -alpha, true, start, tl, cnt)
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
                            let p_idx = if fpi != EMPTY_SQ { piece_to_idx(piece_type(fpi)) } else { 0 };
                            self.counter_move[p_idx][to] = Some(mv);
                        }
                        break;
                    }
                }
            }
        }

        let flag = if best_score <= orig_alpha { TT_ALPHA }
                   else if best_score >= beta  { TT_BETA  }
                   else                        { TT_EXACT };
        self.tt.store(h, actual_depth, best_score, flag, best_move);
        best_score
    }
}

#[cfg(feature = "search-debug")]
impl SearchDebug {
    fn from_env() -> Self {
        Self {
            disable_corr_hist: env_flag("EMBER_DISABLE_CORR_HIST"),
            disable_futility: env_flag("EMBER_DISABLE_FUTILITY"),
            disable_history_pruning: env_flag("EMBER_DISABLE_HISTORY_PRUNING"),
            disable_iid_reduction: env_flag("EMBER_DISABLE_IID_REDUCTION"),
            disable_lmp: env_flag("EMBER_DISABLE_LMP"),
            disable_lmr: env_flag("EMBER_DISABLE_LMR"),
            disable_null_move: env_flag("EMBER_DISABLE_NULL_MOVE"),
            disable_reverse_futility: env_flag("EMBER_DISABLE_REVERSE_FUTILITY"),
            disable_see_pruning: env_flag("EMBER_DISABLE_SEE_PRUNING"),
        }
    }
}

#[cfg(feature = "search-debug")]
fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes" || value == "on"
        })
        .unwrap_or(false)
}
