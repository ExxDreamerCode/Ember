use crate::board::{
    all_occ, attacked_by, bit, has_non_pawn, move_ec, move_promotion, piece_on, piece_type, see,
    BoardState, Move, EMPTY_SQ, INF, KING_ATTACKS, MATE, MAX_PLY, QS_DEPTH,
};
use crate::evaluate::{evaluate, evaluate_nnue_acc, with_nnue_net};
use crate::movegen::{apply_move, generate_moves, is_chess960_castling_move};
use crate::nnue::NNUEAccumulator;
use crate::syzygy::SyzygyTables;
use crate::tt::{SharedTT, TT_ALPHA, TT_BETA, TT_EXACT};
use crate::zobrist::{compute_hash, compute_pawn_hash};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn piece_val(pt: u8) -> i32 {
    match pt {
        0 => 100,
        1 => 325,
        2 => 340,
        3 => 500,
        4 => 950,
        _ => 0,
    }
}

fn piece_to_idx(pt: u8) -> usize {
    match pt {
        0 => 1,
        1 => 2,
        2 => 3,
        3 => 4,
        4 => 5,
        5 => 6,
        _ => 0,
    }
}

fn from_to_key(sr: usize, sc: usize, er: usize, ec: usize) -> (usize, usize) {
    (sr * 8 + sc, er * 8 + ec)
}

fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE / 2 {
        score + ply as i32
    } else if score < -MATE / 2 {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE / 2 {
        score - ply as i32
    } else if score < -MATE / 2 {
        score + ply as i32
    } else {
        score
    }
}

#[inline]
fn is_promotion_move(fpi: u8, mv: &Move) -> bool {
    move_promotion(mv) != 0
        || (fpi != EMPTY_SQ && piece_type(fpi) == 0 && (mv[2] == 0 || mv[2] == 7))
}

fn promotion_value(mv: &Move) -> i32 {
    match move_promotion(mv).to_ascii_uppercase() {
        b'N' => piece_val(1),
        b'B' => piece_val(2),
        b'R' => piece_val(3),
        b'Q' => piece_val(4),
        _ => 0,
    }
}

#[inline]
fn is_en_passant_capture(st: &BoardState, fpi: u8, mv: &Move, to: usize, tpi: u8) -> bool {
    fpi != EMPTY_SQ
        && tpi == EMPTY_SQ
        && piece_type(fpi) == 0
        && Some(to) == st.ep
        && mv[1] != move_ec(mv)
}

#[inline]
fn capture_victim_value(st: &BoardState, fpi: u8, mv: &Move, to: usize, tpi: u8) -> i32 {
    if is_chess960_castling_move(st, mv) {
        0
    } else if tpi != EMPTY_SQ {
        piece_val(piece_type(tpi))
    } else if is_en_passant_capture(st, fpi, mv, to, tpi) {
        piece_val(0)
    } else {
        0
    }
}

#[inline]
fn move_is_capture(st: &BoardState, fpi: u8, mv: &Move, to: usize, tpi: u8) -> bool {
    !is_chess960_castling_move(st, mv)
        && (tpi != EMPTY_SQ || is_en_passant_capture(st, fpi, mv, to, tpi))
}

#[inline]
fn move_see(st: &BoardState, mv: &Move, from: usize, to: usize, fpi: u8, tpi: u8) -> i32 {
    if is_chess960_castling_move(st, mv) || is_en_passant_capture(st, fpi, mv, to, tpi) {
        0
    } else {
        see(&st.bb, from, to)
    }
}

const CORR_HIST_SIZE: usize = 16384;
fn corr_idx(h: u64, side: bool) -> usize {
    let k = h
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(if side { 1 } else { 0 });
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
    pub shared_tt: Arc<SharedTT>,
    pub killers: [[Option<Move>; 2]; MAX_PLY],
    pub history: [[i32; 64]; 64],
    pub counter_move: [[Option<Move>; 64]; 13],
    pub corr_hist: [i32; CORR_HIST_SIZE * 2],
    pub rep_stack: Vec<u64>,
    pub rep_stack_len: usize,
    pub tt_mb: usize,
    pub stopped: Arc<AtomicBool>,
    pub nnue_stack: Vec<NNUEAccumulator>,
    pub syzygy: SyzygyTables,
    #[cfg(feature = "search-debug")]
    pub debug: SearchDebug,
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
    pub fn new(shared_tt: Arc<SharedTT>, stopped: Arc<AtomicBool>) -> Self {
        Searcher {
            shared_tt,
            killers: [[None; 2]; MAX_PLY],
            history: [[0i32; 64]; 64],
            counter_move: [[None; 64]; 13],
            corr_hist: [0i32; CORR_HIST_SIZE * 2],
            rep_stack: Vec::with_capacity(512),
            rep_stack_len: 0,
            tt_mb: 128,
            stopped,
            nnue_stack: Vec::new(),
            syzygy: SyzygyTables::new(),
            #[cfg(feature = "search-debug")]
            debug: SearchDebug::from_env(),
        }
    }

    pub fn resize_tt(&mut self, mb: usize) {
        self.shared_tt.resize(mb);
        self.tt_mb = mb;
    }

    pub fn set_syzygy(&mut self, syzygy: SyzygyTables) {
        self.syzygy = syzygy;
    }

    pub fn init_nnue_stack(&mut self, st: &BoardState) {
        with_nnue_net(|net| {
            if self.nnue_stack.len() < MAX_PLY + 1 {
                self.nnue_stack
                    .resize(MAX_PLY + 1, NNUEAccumulator::new(net.hidden_size));
            }
            self.nnue_stack[0].refresh(net, st);
        });
    }

    #[inline]
    fn time_up(&self, start: Instant, tl: f64) -> bool {
        if self.stopped.load(Ordering::Relaxed) {
            return true;
        }
        if start.elapsed().as_secs_f64() > tl {
            self.set_stopped();
            true
        } else {
            false
        }
    }

    pub fn set_stopped(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    #[cfg(feature = "search-debug")]
    fn corr_hist_enabled(&self) -> bool {
        !self.debug.disable_corr_hist
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn corr_hist_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn futility_enabled(&self) -> bool {
        !self.debug.disable_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn history_pruning_enabled(&self) -> bool {
        !self.debug.disable_history_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn history_pruning_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn iid_reduction_enabled(&self) -> bool {
        !self.debug.disable_iid_reduction
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn iid_reduction_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn lmp_enabled(&self) -> bool {
        !self.debug.disable_lmp
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmp_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn lmr_enabled(&self) -> bool {
        !self.debug.disable_lmr
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmr_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn null_move_enabled(&self) -> bool {
        !self.debug.disable_null_move
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn null_move_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn reverse_futility_enabled(&self) -> bool {
        !self.debug.disable_reverse_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn reverse_futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn see_pruning_enabled(&self) -> bool {
        !self.debug.disable_see_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn see_pruning_enabled(&self) -> bool {
        true
    }

    fn static_eval(&self, st: &BoardState, ply: usize) -> i32 {
        if st.chess960 && st.mc <= 3 {
            return evaluate(st) * if st.w { 1 } else { -1 };
        }
        with_nnue_net(|net| {
            let score = if ply < self.nnue_stack.len() {
                evaluate_nnue_acc(net, &self.nnue_stack[ply], st)
            } else {
                let mut acc = NNUEAccumulator::new(net.hidden_size);
                acc.refresh(net, st);
                evaluate_nnue_acc(net, &acc, st)
            };
            if st.w {
                score
            } else {
                -score
            }
        })
        .unwrap_or_else(|| evaluate(st) * if st.w { 1 } else { -1 })
    }

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
        if st.chess960 && st.mc <= 3 {
            let base = evaluate(st) * if st.w { 1 } else { -1 };
            if self.corr_hist_enabled() {
                let ph = compute_pawn_hash(st);
                let idx = corr_idx(ph, st.w);
                return base + self.corr_hist[idx].clamp(-200, 200);
            }
            return base;
        }
        if let Some(nnue_score) = with_nnue_net(|net| {
            let mut acc = NNUEAccumulator::new(net.hidden_size);
            acc.refresh(net, st);
            let score = evaluate_nnue_acc(net, &acc, st);
            if st.w {
                score
            } else {
                -score
            }
        }) {
            return nnue_score;
        }
        let base = evaluate(st) * if st.w { 1 } else { -1 };
        if self.corr_hist_enabled() {
            let ph = compute_pawn_hash(st);
            let idx = corr_idx(ph, st.w);
            return base + self.corr_hist[idx].clamp(-200, 200);
        }
        base
    }

    pub fn probe_syzygy(&self, st: &BoardState) -> Option<i32> {
        self.syzygy
            .probe_wdl(st)
            .and_then(SyzygyTables::wdl_to_score)
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
        if self.rep_stack_len < 4 {
            return false;
        }
        let last = self.rep_stack[self.rep_stack_len - 1];
        let mut count = 0;
        for i in (0..self.rep_stack_len - 1).rev() {
            if self.rep_stack[i] == last {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    fn push_nnue_acc(
        &mut self,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) -> bool {
        with_nnue_net(|net| {
            if ply + 1 >= self.nnue_stack.len() {
                return false;
            }

            let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);

            let ok =
                self.nnue_stack[ply + 1].update_move(net, st_before, sr, sc, er, ec, promotion);

            if !ok {
                self.nnue_stack[ply + 1].refresh(net, st_after);
            }
            true
        })
        .unwrap_or(false)
    }

    fn qsearch(
        &mut self,
        st: &mut BoardState,
        mut alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
    ) -> i32 {
        *cnt += 1;
        if self.time_up(start, tl) {
            return 0;
        }
        let ks = st.king_sq(st.w);
        let in_check = crate::board::is_attacked(&st.bb, ks, !st.w);

        if !in_check && self.syzygy.tables.is_some() && SyzygyTables::pieces_ok(st) {
            if let Some(cutoff) = self.syzygy.probe_cutoff(st, beta, alpha) {
                return cutoff;
            }
        }

        if ply >= 2 && self.is_repetition() {
            return 0;
        }

            if !in_check {
                let stand = self.static_eval(st, ply);
                if stand >= beta {
                    return stand;
                }
                if stand > alpha {
                    alpha = stand;
                }
                if depth <= 0 && alpha - 975 > stand {
                    return alpha;
                }
            } else if depth <= -4 {
                return self.static_eval(st, ply);
            }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + 1000 } else { alpha };
        }
        if with_nnue_net(|_| true).unwrap_or(false) && ply + 1 >= self.nnue_stack.len() {
            if ply + 1 < MAX_PLY + 1 {
                self.nnue_stack
                    .resize(ply + 2, NNUEAccumulator::new(self.nnue_stack[0].hs));
            }
        }

        let mut caps: Vec<Move> = if in_check {
            moves
        } else {
            moves
                .into_iter()
                .filter(|mv| {
                    let to = mv[2] * 8 + move_ec(mv);
                    let from = mv[0] * 8 + mv[1];
                    let fpi = piece_on(&st.bb, from);
                    let tpi = piece_on(&st.bb, to);
                    move_is_capture(st, fpi, mv, to, tpi) || is_promotion_move(fpi, mv)
                })
                .collect()
        };
        if caps.is_empty() {
            return alpha;
        }

        caps.sort_by_key(|mv| {
            let to = mv[2] * 8 + move_ec(mv);
            let from = mv[0] * 8 + mv[1];
            let vpi = piece_on(&st.bb, to);
            let api = piece_on(&st.bb, from);
            let victim = capture_victim_value(st, api, mv, to, vpi);
            let attacker = if api != EMPTY_SQ {
                piece_val(piece_type(api))
            } else {
                0
            };
            -(victim * 10 - attacker + promotion_value(mv))
        });

        for mv in caps {
            if self.time_up(start, tl) {
                return 0;
            }
            let from = mv[0] * 8 + mv[1];
            let to = mv[2] * 8 + move_ec(&mv);
            let fpi = piece_on(&st.bb, from);
            let tpi = piece_on(&st.bb, to);
            if !in_check && move_see(st, &mv, from, to, fpi, tpi) < 0 {
                continue;
            }
            let st_before = *st;
            apply_move(st, mv[0], mv[1], mv[2], move_ec(&mv), move_promotion(&mv));
            self.push_nnue_acc(
                &st_before,
                st,
                mv[0],
                mv[1],
                mv[2],
                move_ec(&mv),
                move_promotion(&mv),
                ply,
            );
            let score = -self.qsearch(st, -beta, -alpha, depth - 1, start, tl, cnt, ply + 1);
            *st = st_before;
            if self.stopped.load(Ordering::Relaxed) {
                return 0;
            }
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    pub fn negamax(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        *cnt += 1;
        if self.time_up(start, tl) {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.static_eval(st, ply);
        }

        let h = compute_hash(st);
        let ks = st.king_sq(st.w);
        let in_check = crate::board::is_attacked(&st.bb, ks, !st.w);
        let is_pv = beta - alpha > 1;
        let is_root = ply == 0;
        let king_pressure = if in_check {
            8
        } else {
            tactical_king_pressure(st)
        };

        let tb_available = !in_check && self.syzygy.tables.is_some() && SyzygyTables::pieces_ok(st);

        let eval_score = if tb_available {
            self.probe_syzygy(st)
                .unwrap_or_else(|| self.static_eval(st, ply))
        } else {
            self.static_eval(st, ply)
        };

        if tb_available && !is_pv && !is_root {
            if let Some(cutoff) = self.syzygy.probe_cutoff(st, beta, alpha) {
                return cutoff;
            }
        }

        if ply > 0 && self.is_repetition() {
            return 0;
        }

        let ext = if in_check && depth < 16 { 1 } else { 0 };
        let actual_depth = depth + ext;

        let tt_data = self.shared_tt.get_depth(h);
        let tt_move = tt_data.and_then(|(_, _, _, best)| best);
        let tt_score = tt_data.map(|(_, s, _, _)| score_from_tt(s, ply));
        let tt_depth = tt_data.map(|(d, _, _, _)| d).unwrap_or(-1);
        let tt_flag = tt_data.map(|(_, _, f, _)| f);

        if !is_pv && tt_depth >= actual_depth && tt_flag.is_some() {
            let s = tt_score.unwrap();
            match tt_flag.unwrap() {
                TT_EXACT => return s,
                TT_ALPHA => {
                    if s <= alpha {
                        return alpha;
                    }
                }
                TT_BETA => {
                    if s >= beta {
                        return beta;
                    }
                }
                _ => {}
            }
        }

        if actual_depth <= 0 {
            return self.qsearch(st, alpha, beta, QS_DEPTH, start, tl, cnt, ply);
        }

        if self.reverse_futility_enabled() && !in_check && !is_pv && actual_depth <= 8 && ply > 0 {
            let margin = 80 + 65 * actual_depth;
            if eval_score - margin >= beta {
                return eval_score - margin;
            }
        }
        if self.futility_enabled()
            && !in_check
            && !is_pv
            && actual_depth <= 3
            && ply > 0
        {
            let margin = 150 * actual_depth;
            if eval_score + margin <= alpha {
                let q = self.qsearch(
                    st,
                    alpha - margin,
                    beta - margin,
                    QS_DEPTH,
                    start,
                    tl,
                    cnt,
                    ply,
                );
                if q + margin <= alpha {
                    return alpha;
                }
            }
        }
        if self.null_move_enabled()
            && king_pressure < 3
            && !in_check
            && can_null
            && !is_pv
            && ply > 0
            && actual_depth >= 3
            && has_non_pawn(&st.bb, st.w)
            && eval_score >= beta
        {
            let r = 3 + actual_depth / 4 + ((eval_score - beta) / 200).min(3);
            let ow = st.w;
            let oe = st.ep;
            st.w = !st.w;
            st.ep = None;
            let null_h = compute_hash(st);
            self.rep_stack.push(null_h);
            self.rep_stack_len += 1;
            let s = -self.negamax(
                st,
                actual_depth - r - 1,
                ply + 1,
                -beta,
                -beta + 1,
                false,
                start,
                tl,
                cnt,
            );
            self.rep_stack.pop();
            self.rep_stack_len -= 1;
            st.w = ow;
            st.ep = oe;
            if self.time_up(start, tl) {
                return 0;
            }
            if s >= beta {
                return beta;
            }
        }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let actual_depth =
            if self.iid_reduction_enabled() && tt_move.is_none() && actual_depth >= 4 && is_pv {
                actual_depth - 1
            } else {
                actual_depth
            };

        let mut scored: Vec<(i32, Move)> = moves
            .into_iter()
            .map(|mv| {
                let mut s = 0i32;
                if Some(mv) == tt_move {
                    s = 10_000_000;
                } else {
                    let from = mv[0] * 8 + mv[1];
                    let to = mv[2] * 8 + move_ec(&mv);
                    let tpi = piece_on(&st.bb, to);
                    let fpi = piece_on(&st.bb, from);
                    let is_promo = is_promotion_move(fpi, &mv);
                    if move_is_capture(st, fpi, &mv, to, tpi) || is_promo {
                        let v = capture_victim_value(st, fpi, &mv, to, tpi);
                        let a = if fpi != EMPTY_SQ {
                            piece_val(piece_type(fpi))
                        } else {
                            0
                        };
                        let see_sc = move_see(st, &mv, from, to, fpi, tpi);
                        if see_sc >= 0 {
                            s += 2_000_000 + v * 10 - a + see_sc;
                        } else {
                            s += 500_000 + v * 10 - a;
                        }
                        if is_promo {
                            s += 1_500_000 + promotion_value(&mv);
                        }
                    } else {
                        if self.killers[ply][0] == Some(mv) {
                            s += 900_000;
                        } else if self.killers[ply][1] == Some(mv) {
                            s += 800_000;
                        }
                        let p_idx = if fpi != EMPTY_SQ {
                            piece_to_idx(piece_type(fpi))
                        } else {
                            0
                        };
                        if self.counter_move[p_idx][to] == Some(mv) {
                            s += 700_000;
                        }
                        let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], move_ec(&mv));
                        s += self.history[fk][tk].clamp(-32768, 32768);
                    }
                }
                (s, mv)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let lmp_count = if self.lmp_enabled()
            && king_pressure < 3
            && !is_pv
            && !in_check
            && actual_depth <= 8
        {
            let base = match actual_depth {
                1 => 4,
                2 => 7,
                3 => 11,
                4 => 17,
                5 => 24,
                6 => 33,
                7 => 44,
                8 => 57,
                _ => usize::MAX,
            };
            base
        } else {
            usize::MAX
        };

        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = scored.first().map(|&(_, mv)| mv);
        let mut quiets_tried: Vec<Move> = Vec::new();

        for (i, &(_, mv)) in scored.iter().enumerate() {
            if self.time_up(start, tl) {
                return 0;
            }

            let from = mv[0] * 8 + mv[1];
            let to = mv[2] * 8 + move_ec(&mv);
            let fpi = piece_on(&st.bb, from);
            let tpi = piece_on(&st.bb, to);
            let capture = move_is_capture(st, fpi, &mv, to, tpi);
            let is_promo = is_promotion_move(fpi, &mv);
            let is_quiet = !capture && !is_promo;

            let gives_check = {
                let mut after = *st;
                apply_move(
                    &mut after,
                    mv[0],
                    mv[1],
                    mv[2],
                    move_ec(&mv),
                    move_promotion(&mv),
                );
                let opp_ks = after.king_sq(after.w);
                crate::board::is_attacked(&after.bb, opp_ks, !after.w)
            };

            if !is_pv && !in_check && is_quiet && i >= lmp_count {
                break;
            }
            if !is_pv && !in_check && i > 0 && best_score > -MATE / 2 {
                if capture {
                    if self.see_pruning_enabled()
                        && move_see(st, &mv, from, to, fpi, tpi) < -80 * actual_depth
                    {
                        continue;
                    }
                } else if is_quiet && self.history_pruning_enabled() {
                    let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], move_ec(&mv));
                    if actual_depth <= 5 && self.history[fk][tk] < -1024 * actual_depth {
                        continue;
                    }
                }
            }

            let move_ext = if gives_check && !in_check && i == 0 && actual_depth < 16 {
                1
            } else {
                0
            };

            let st_before = *st;
            apply_move(st, mv[0], mv[1], mv[2], move_ec(&mv), move_promotion(&mv));

            self.push_nnue_acc(
                &st_before,
                st,
                mv[0],
                mv[1],
                mv[2],
                move_ec(&mv),
                move_promotion(&mv),
                ply,
            );

            let h_after = compute_hash(st);
            self.rep_stack.push(h_after);
            self.rep_stack_len += 1;

            let new_depth = actual_depth - 1 + move_ext;

            let lmr_eligible = self.lmr_enabled()
                && i >= 2
                && actual_depth >= 3
                && is_quiet
                && !in_check
                && !gives_check;
            let s = if i == 0 {
                -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
            } else if lmr_eligible {
                let r = {
                    let base = (0.5 + (i as f64).ln() * (actual_depth as f64).ln() / 1.8) as i32;
                    let r = base.min(actual_depth - 1).max(1);
                    if !is_pv {
                        (r + 1).min(actual_depth - 1)
                    } else {
                        r
                    }
                };
                let s2 = -self.negamax(
                    st,
                    new_depth - r,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    true,
                    start,
                    tl,
                    cnt,
                );
                if s2 > alpha {
                    let s3 = -self.negamax(
                        st,
                        new_depth,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        true,
                        start,
                        tl,
                        cnt,
                    );
                    if s3 > alpha && is_pv {
                        -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
                    } else {
                        s3
                    }
                } else {
                    s2
                }
            } else if is_pv {
                let s2 = -self.negamax(
                    st,
                    new_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    true,
                    start,
                    tl,
                    cnt,
                );
                if s2 > alpha && s2 < beta {
                    -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
                } else {
                    s2
                }
            } else {
                -self.negamax(st, new_depth, ply + 1, -beta, -alpha, true, start, tl, cnt)
            };

            self.rep_stack.pop();
            self.rep_stack_len -= 1;
            *st = st_before;

            if self.stopped.load(Ordering::Relaxed) {
                return 0;
            }

            if is_quiet {
                quiets_tried.push(mv);
            }

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
                            let (fk, tk) = from_to_key(mv[0], mv[1], mv[2], move_ec(&mv));
                            let bonus = (actual_depth * actual_depth).min(512);
                            self.history[fk][tk] += bonus;
                            if self.history[fk][tk] > 16384 {
                                for a in 0..64 {
                                    for b in 0..64 {
                                        self.history[a][b] /= 2;
                                    }
                                }
                            }
                            for &qmv in &quiets_tried {
                                if qmv == mv {
                                    continue;
                                }
                                let (qfk, qtk) = from_to_key(qmv[0], qmv[1], qmv[2], move_ec(&qmv));
                                self.history[qfk][qtk] -= bonus;
                                if self.history[qfk][qtk] < -16384 {
                                    for a in 0..64 {
                                        for b in 0..64 {
                                            self.history[a][b] /= 2;
                                        }
                                    }
                                }
                            }
                            let p_idx = if fpi != EMPTY_SQ {
                                piece_to_idx(piece_type(fpi))
                            } else {
                                0
                            };
                            self.counter_move[p_idx][to] = Some(mv);
                        }
                        break;
                    }
                }
            }
        }

        if self.stopped.load(Ordering::Relaxed) {
            return 0;
        }

        let flag = if best_score <= orig_alpha {
            TT_ALPHA
        } else if best_score >= beta {
            TT_BETA
        } else {
            TT_EXACT
        };
        self.shared_tt.store(
            h,
            actual_depth,
            score_to_tt(best_score, ply),
            flag,
            best_move,
        );
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

struct ThreadResult {
    best_move: Move,
    score: i32,
    depth: i32,
    nodes: u64,
}

pub fn lazy_smp_search(
    shared_tt: Arc<SharedTT>,
    st: &BoardState,
    time_limit: f64,
    depth_limit: i32,
    num_threads: usize,
) -> (Move, i32, i32, u64) {
    let stopped = Arc::new(AtomicBool::new(false));
    let all_moves = generate_moves(st, st.w, &st.cr, st.ep);

    let actual_threads = if all_moves.len() < num_threads {
        1
    } else {
        num_threads
    };

    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let global_best_depth: Arc<AtomicI32> = Arc::new(AtomicI32::new(0));
    let global_nodes: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let root_hash = compute_hash(st);

    let mut handles = Vec::with_capacity(actual_threads);

    let moves_per_thread = (all_moves.len() + actual_threads - 1) / actual_threads;

    for thread_id in 0..actual_threads {
        let start_idx = thread_id * moves_per_thread;
        let end_idx = (start_idx + moves_per_thread).min(all_moves.len());
        if start_idx >= end_idx {
            break;
        }
        let mut my_moves: Vec<Move> = all_moves[start_idx..end_idx].to_vec();

        let shared_tt = Arc::clone(&shared_tt);
        let stopped = Arc::clone(&stopped);
        let results = Arc::clone(&results);
        let global_best_depth = Arc::clone(&global_best_depth);
        let global_nodes = Arc::clone(&global_nodes);
        let st = *st;

        let handle = std::thread::Builder::new()
            .name(format!("rts-{}", thread_id))
            .spawn(move || {
                let mut searcher = Searcher::new(shared_tt, Arc::clone(&stopped));
                searcher.syzygy = SyzygyTables::new();
                searcher.init_nnue_stack(&st);

                let mut best_move = my_moves[0];
                let mut best_score = 0i32;
                let mut best_depth = 0;
                let mut total_nodes = 0u64;

                let init_eval = searcher.corrected_eval(&st);
                let mut prev_score = init_eval;

                for depth in 1..=depth_limit {
                    if searcher.time_up(start, time_limit) {
                        break;
                    }

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

                    if let Some((tt_d, _, _, Some(tt_mv))) = searcher.shared_tt.get_depth(root_hash)
                    {
                        if tt_d >= 1 && !my_moves.contains(&tt_mv) {
                            my_moves.push(tt_mv);
                        }
                    }

                    'asp: loop {
                        let mut sorted = my_moves.clone();
                        if asp_best != my_moves[0] {
                            if let Some(pos) = sorted.iter().position(|&m| m == asp_best) {
                                sorted.swap(0, pos);
                            }
                        }

                        let mut cur_best = sorted[0];
                        let mut cur_score = -INF;
                        let mut loop_alpha = alpha;

                        for &mv in &sorted {
                            if searcher.time_up(start, time_limit) {
                                break;
                            }
                            let mut s = st;
                            apply_move(
                                &mut s,
                                mv[0],
                                mv[1],
                                mv[2],
                                move_ec(&mv),
                                move_promotion(&mv),
                            );
                            crate::evaluate::with_nnue_net(|net| {
                                if !searcher.nnue_stack.is_empty() {
                                    searcher.nnue_stack[1].refresh(net, &s);
                                }
                            });
                            let h = compute_hash(&s);
                            searcher.rep_stack.push(h);
                            searcher.rep_stack_len += 1;

                            let score = if cur_score == -INF {
                                -searcher.negamax(
                                    &mut s,
                                    depth - 1,
                                    1,
                                    -beta,
                                    -loop_alpha,
                                    true,
                                    start,
                                    time_limit,
                                    &mut nd,
                                )
                            } else {
                                let sc = -searcher.negamax(
                                    &mut s,
                                    depth - 1,
                                    1,
                                    -loop_alpha - 1,
                                    -loop_alpha,
                                    true,
                                    start,
                                    time_limit,
                                    &mut nd,
                                );
                                if sc > loop_alpha && sc < beta {
                                    -searcher.negamax(
                                        &mut s,
                                        depth - 1,
                                        1,
                                        -beta,
                                        -loop_alpha,
                                        true,
                                        start,
                                        time_limit,
                                        &mut nd,
                                    )
                                } else {
                                    sc
                                }
                            };

                            searcher.rep_stack.pop();
                            searcher.rep_stack_len -= 1;

                            if stopped.load(Ordering::Relaxed) {
                                break;
                            }
                            if score > cur_score {
                                cur_score = score;
                                cur_best = mv;
                            }
                            if score > loop_alpha {
                                loop_alpha = score;
                            }
                            if loop_alpha >= beta {
                                break;
                            }
                        }

                        if stopped.load(Ordering::Relaxed)
                            || start.elapsed().as_secs_f64() > time_limit
                        {
                            break 'asp;
                        }

                        if cur_score <= alpha {
                            asp_delta = asp_delta.saturating_mul(2).min(INF);
                            alpha = (prev_score - asp_delta).max(-INF);
                            beta = prev_score + init_delta;
                            continue 'asp;
                        }
                        if cur_score >= beta {
                            asp_delta = asp_delta.saturating_mul(2).min(INF);
                            beta = (prev_score + asp_delta).min(INF);
                            asp_best = cur_best;
                            continue 'asp;
                        }
                        asp_best = cur_best;
                        asp_score = cur_score;
                        break;
                    }

                    if stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    total_nodes += nd;
                    global_nodes.fetch_add(nd, Ordering::Relaxed);
                    let elapsed = start.elapsed().as_secs_f64();

                    if elapsed <= time_limit {
                        let prev = global_best_depth.fetch_max(depth, Ordering::SeqCst);
                        if prev < depth {
                            let score_str = if asp_score.abs() > 90_000 {
                                let mate_in = (MATE - asp_score.abs()) / 2 + 1;
                                if asp_score > 0 {
                                    format!("mate {}", mate_in)
                                } else {
                                    format!("mate -{}", mate_in)
                                }
                            } else {
                                format!("cp {}", asp_score)
                            };
                            let mv_str = crate::board::move_to_uci(&st, &asp_best);
                            let g_nodes = global_nodes.load(Ordering::Relaxed);
                            let nps = if elapsed > 0.0 {
                                (g_nodes as f64 / elapsed) as i64
                            } else {
                                0
                            };
                            println!(
                                "info depth {} score {} nodes {} nps {} time {} pv {}",
                                depth,
                                score_str,
                                g_nodes,
                                nps,
                                (elapsed * 1000.0) as u64,
                                mv_str
                            );
                        }
                        best_move = asp_best;
                        best_score = asp_score;
                        best_depth = depth;
                        prev_score = best_score;
                        searcher.update_correction_history(&st, best_score, best_depth);
                    } else {
                        break;
                    }
                }

                let mut lock = results.lock().unwrap();
                lock.push(ThreadResult {
                    best_move,
                    score: best_score,
                    depth: best_depth,
                    nodes: total_nodes,
                });
            });

        if let Ok(h) = handle {
            handles.push(h);
        }
    }

    for h in handles {
        let _ = h.join();
    }

    let lock = results.lock().unwrap();
    let best = lock
        .iter()
        .max_by(|a, b| a.depth.cmp(&b.depth).then_with(|| a.score.cmp(&b.score)))
        .unwrap_or(&lock[0]);

    let best_depth = best.depth;
    let total_nodes: u64 = lock.iter().map(|r| r.nodes).sum();

    (best.best_move, best.score, best_depth, total_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use std::time::Duration;

    fn state_from_fen(fen: &str) -> BoardState {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        engine.st
    }

    #[test]
    fn qsearch_searches_en_passant_captures() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut searcher = Searcher::new(shared_tt, stopped);
        let stand_pat = searcher.corrected_eval(&st);
        let mut nodes = 0u64;

        let score = searcher.qsearch(
            &mut st,
            -INF,
            INF,
            QS_DEPTH,
            Instant::now(),
            10.0,
            &mut nodes,
            0,
        );

        assert!(
            score > stand_pat + 50,
            "qsearch should improve on stand-pat by searching e5xd6 en passant: stand_pat={stand_pat}, score={score}"
        );
    }

    #[test]
    fn negamax_timeout_sets_stopped_without_storing_tt() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut searcher = Searcher::new(shared_tt.clone(), stopped);
        let key = compute_hash(&st);
        let mut nodes = 0u64;

        let score = searcher.negamax(
            &mut st,
            4,
            0,
            -INF,
            INF,
            true,
            Instant::now() - Duration::from_secs(1),
            0.0,
            &mut nodes,
        );

        assert_eq!(score, 0);
        assert!(searcher.stopped.load(Ordering::Relaxed));
        assert!(searcher.shared_tt.get_depth(key).is_none());
    }

    #[test]
    fn root_search_resets_previous_timeout_state() {
        let mut engine = Engine::new();
        engine.book = None;
        engine.searcher.set_stopped();

        let (best_move, _, nodes, _) = engine.find_best_move(1.0, 1);

        assert_ne!(best_move, "0000");
        assert!(nodes > 0);
        assert!(!engine.searcher.stopped.load(Ordering::Relaxed));
    }

    #[test]
    fn tt_mate_scores_are_stored_ply_independent() {
        let winning_score = MATE - 9;
        let losing_score = -MATE + 11;

        assert_eq!(score_to_tt(winning_score, 9), MATE);
        assert_eq!(score_from_tt(MATE, 3), MATE - 3);

        assert_eq!(score_to_tt(losing_score, 11), -MATE);
        assert_eq!(score_from_tt(-MATE, 4), -MATE + 4);
    }

    #[test]
    fn tt_non_mate_scores_are_not_adjusted() {
        assert_eq!(score_to_tt(42, 8), 42);
        assert_eq!(score_from_tt(-313, 5), -313);
    }

    #[test]
    fn threefold_repetition_detected_after_long_history() {
        let mut engine = Engine::new();
        engine.book = None;

        engine.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 50");

        for _ in 0..12 {
            assert!(engine.make_move_uci(7, 4, 7, 3, 0));
            assert!(engine.make_move_uci(0, 4, 0, 3, 0));
            assert!(engine.make_move_uci(7, 3, 7, 4, 0));
            assert!(engine.make_move_uci(0, 3, 0, 4, 0));
        }

        assert!(
            engine.searcher.is_repetition(),
            "Threefold repetition should be detected even after 20+ moves of history"
        );
    }
}
