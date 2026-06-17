use crate::board::{
    all_occ, attacked_by, bit, has_non_pawn, move_ec, move_promotion, piece_on, piece_type,
    promotion_piece_index, see, BoardState, Move, BK, BP, EMPTY_SQ, INF, KING_ATTACKS, MATE,
    MAX_PLY, QS_DEPTH, WK, WP,
};
use crate::evaluate::{evaluate, evaluate_nnue_acc, with_nnue_net};
use crate::movegen::{apply_move, generate_moves};
use crate::nnue::{NNUEAccumulator, NNUENet};
use crate::tt::{TT, TT_ALPHA, TT_BETA, TT_EXACT};
use crate::zobrist::{compute_hash, compute_pawn_hash};
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
    if tpi != EMPTY_SQ {
        piece_val(piece_type(tpi))
    } else if is_en_passant_capture(st, fpi, mv, to, tpi) {
        piece_val(0)
    } else {
        0
    }
}

#[inline]
fn move_is_capture(st: &BoardState, fpi: u8, mv: &Move, to: usize, tpi: u8) -> bool {
    tpi != EMPTY_SQ || is_en_passant_capture(st, fpi, mv, to, tpi)
}

#[inline]
fn move_see(st: &BoardState, mv: &Move, from: usize, to: usize, fpi: u8, tpi: u8) -> i32 {
    if is_en_passant_capture(st, fpi, mv, to, tpi) {
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

fn promotion_race(st: &BoardState) -> bool {
    const WHITE_ADVANCED: u64 = 0x0000_0000_00FF_FF00;
    const BLACK_ADVANCED: u64 = 0x00FF_FF00_0000_0000;
    (st.bb[WP] & WHITE_ADVANCED) != 0 || (st.bb[BP] & BLACK_ADVANCED) != 0
}

fn sparse_endgame(st: &BoardState) -> bool {
    let mut pieces = 0;
    for idx in 0..12 {
        if idx != WK && idx != BK {
            pieces += st.bb[idx].count_ones();
        }
    }
    pieces <= 8
}

fn selective_pruning_unsafe(st: &BoardState) -> bool {
    promotion_race(st) || sparse_endgame(st)
}

pub struct Searcher {
    pub tt: TT,
    pub killers: [[Option<Move>; 2]; MAX_PLY],
    pub history: [[i32; 64]; 64],
    pub counter_move: [[Option<Move>; 64]; 13],
    pub corr_hist: [i32; CORR_HIST_SIZE * 2],
    pub rep_stack: Vec<u64>,
    pub rep_stack_len: usize,
    pub tt_mb: usize,
    pub stopped: bool,
    pub nnue_stack: Vec<NNUEAccumulator>,
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
            stopped: false,
            nnue_stack: Vec::new(),
            #[cfg(feature = "search-debug")]
            debug: SearchDebug::from_env(),
        }
    }

    pub fn resize_tt(&mut self, mb: usize) {
        self.tt.resize(mb);
        self.tt_mb = mb;
    }

    pub fn init_nnue_stack(&mut self, st: &BoardState) {
        with_nnue_net(|net| {
            self.ensure_nnue_stack_len(net, MAX_PLY + 1);
            self.nnue_stack[0].refresh(net, st);
        });
    }

    fn ensure_nnue_stack_len(&mut self, net: &NNUENet, len: usize) {
        if self.nnue_stack.len() < len {
            self.nnue_stack
                .resize_with(len, || NNUEAccumulator::new(net.hidden_size));
        }
    }

    #[inline]
    fn time_up(&mut self, start: Instant, tl: f64) -> bool {
        if self.stopped || start.elapsed().as_secs_f64() > tl {
            self.stopped = true;
            true
        } else {
            false
        }
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn corr_hist_enabled(&self) -> bool {
        !self.debug.disable_corr_hist
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn corr_hist_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn futility_enabled(&self) -> bool {
        !self.debug.disable_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn history_pruning_enabled(&self) -> bool {
        !self.debug.disable_history_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn history_pruning_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn iid_reduction_enabled(&self) -> bool {
        !self.debug.disable_iid_reduction
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn iid_reduction_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn lmp_enabled(&self) -> bool {
        !self.debug.disable_lmp
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmp_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn lmr_enabled(&self) -> bool {
        !self.debug.disable_lmr
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn lmr_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn null_move_enabled(&self) -> bool {
        !self.debug.disable_null_move
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn null_move_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn reverse_futility_enabled(&self) -> bool {
        !self.debug.disable_reverse_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn reverse_futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    #[inline(always)]
    fn see_pruning_enabled(&self) -> bool {
        !self.debug.disable_see_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn see_pruning_enabled(&self) -> bool {
        true
    }

    fn static_eval(&self, st: &BoardState, ply: usize) -> i32 {
        with_nnue_net(|net| {
            let score = if ply < self.nnue_stack.len() {
                evaluate_nnue_acc(net, &self.nnue_stack[ply], st)
            } else {
                let mut acc = NNUEAccumulator::new(net.hidden_size);
                acc.refresh(net, st);
                evaluate_nnue_acc(net, &acc, st)
            };
            if st.w { score } else { -score }
        })
        .unwrap_or_else(|| evaluate(st) * if st.w { 1 } else { -1 })
    }

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
        if let Some(nnue_score) = with_nnue_net(|net| {
            let mut acc = NNUEAccumulator::new(net.hidden_size);
            acc.refresh(net, st);
            let score = evaluate_nnue_acc(net, &acc, st);
            if st.w { score } else { -score }
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
        let start = if self.rep_stack_len > 20 {
            self.rep_stack_len - 20
        } else {
            0
        };
        for i in (start..self.rep_stack_len - 1).rev() {
            if self.rep_stack[i] == last {
                count += 1;
                if count >= 1 {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn push_nnue_acc(
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
            if ply >= MAX_PLY {
                return false;
            }
            if self.nnue_stack.len() <= ply {
                self.ensure_nnue_stack_len(net, ply + 1);
                self.nnue_stack[ply].refresh(net, st_before);
            }
            self.ensure_nnue_stack_len(net, ply + 2);

            let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);

            let ok = self.nnue_stack[ply + 1].update_move(net, st_before, sr, sc, er, ec, promotion);

            if !ok {
                self.nnue_stack[ply + 1].refresh(net, st_after);
            }
            true
        })
        .unwrap_or(false)
    }

    fn push_null_nnue_acc(&mut self, st: &BoardState, ply: usize) -> bool {
        with_nnue_net(|net| {
            if ply >= MAX_PLY {
                return false;
            }
            if self.nnue_stack.len() <= ply {
                self.ensure_nnue_stack_len(net, ply + 1);
                self.nnue_stack[ply].refresh(net, st);
            }
            self.ensure_nnue_stack_len(net, ply + 2);
            let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
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

        if !in_check {
            let stand = self.static_eval(st, ply);
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
            if depth <= 0 {
                return alpha;
            }
            if alpha - 975 > stand {
                return alpha;
            }
        } else if depth <= -8 {
            return self.static_eval(st, ply);
        }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if moves.is_empty() {
            return if in_check { -MATE + 1000 } else { alpha };
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
            if self.stopped {
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

        if !is_root && ply >= 2 && self.is_repetition() {
            return 0;
        }

        let ext = if in_check { 1 } else { 0 };
        let actual_depth = depth + ext;

        let tt_entry = self.tt.get(h);
        let tt_move = tt_entry.and_then(|e| e.best_move);
        let tt_score = tt_entry.map(|e| score_from_tt(e.score, ply));
        let tt_depth = tt_entry.map(|e| e.depth).unwrap_or(-1);
        let tt_flag = tt_entry.map(|e| e.flag);

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

        let eval_score = self.static_eval(st, ply);

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
            && !(actual_depth >= 2 && selective_pruning_unsafe(st))
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
            self.push_null_nnue_acc(st, ply);
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
            && !selective_pruning_unsafe(st)
        {
            match actual_depth {
                1 => 4,
                2 => 7,
                3 => 11,
                4 => 17,
                5 => 24,
                6 => 33,
                7 => 44,
                8 => 57,
                _ => usize::MAX,
            }
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
                let mut bb2 = st.bb;
                let pi = piece_on(&bb2, from);
                if pi != EMPTY_SQ {
                    let cap = piece_on(&bb2, to);
                    if cap != EMPTY_SQ {
                        bb2[cap as usize] &= !bit(to);
                    }
                    if piece_type(pi) == 0 && mv[1] != move_ec(&mv) && cap == EMPTY_SQ {
                        let cap_sq = if st.w { to + 8 } else { to - 8 };
                        let cpi = piece_on(&bb2, cap_sq);
                        if cpi != EMPTY_SQ {
                            bb2[cpi as usize] &= !bit(cap_sq);
                        }
                    }
                    bb2[pi as usize] &= !bit(from);
                    if let Some(promo_pi) = promotion_piece_index(st.w, move_promotion(&mv)) {
                        bb2[promo_pi] |= bit(to);
                    } else {
                        bb2[pi as usize] |= bit(to);
                    }
                    let opp_k_pi = if st.w {
                        crate::board::BK
                    } else {
                        crate::board::WK
                    };
                    let opp_ks2 = if bb2[opp_k_pi] != 0 {
                        bb2[opp_k_pi].trailing_zeros() as usize
                    } else {
                        64
                    };
                    crate::board::is_attacked(&bb2, opp_ks2, st.w)
                } else {
                    false
                }
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

            let move_ext = if gives_check && !in_check && i == 0 {
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
                && !gives_check
                && !selective_pruning_unsafe(st);
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

            if self.stopped {
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

        if self.stopped {
            return 0;
        }

        let flag = if best_score <= orig_alpha {
            TT_ALPHA
        } else if best_score >= beta {
            TT_BETA
        } else {
            TT_EXACT
        };
        self.tt.store(
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

    fn ensure_nnue_loaded() {
        crate::evaluate::init_embedded_nnue().expect("embedded test NNUE should load");
    }

    #[test]
    fn qsearch_searches_en_passant_captures() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let mut searcher = Searcher::new();
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
            score > stand_pat,
            "qsearch should improve on stand-pat by searching e5xd6 en passant: stand_pat={stand_pat}, score={score}"
        );
    }

    #[test]
    fn negamax_timeout_sets_stopped_without_storing_tt() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let mut searcher = Searcher::new();
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
        assert!(searcher.stopped);
        assert!(searcher.tt.get(key).is_none());
    }

    #[test]
    fn root_search_resets_previous_timeout_state() {
        let mut engine = Engine::new();
        engine.book = None;
        engine.searcher.stopped = true;

        let (best_move, _, nodes, _) = engine.find_best_move(1.0, 1);

        assert_ne!(best_move, "0000");
        assert!(nodes > 0);
        assert!(!engine.searcher.stopped);
    }

    #[test]
    fn null_move_copies_current_nnue_accumulator() {
        ensure_nnue_loaded();
        let mut st = state_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let mut searcher = Searcher::new();
        searcher.init_nnue_stack(&st);

        assert!(searcher.push_null_nnue_acc(&st, 0));
        st.w = !st.w;
        st.ep = None;

        let copied_eval = searcher.static_eval(&st, 1);
        let expected = crate::evaluate::with_nnue_net(|net| {
            let mut refreshed = NNUEAccumulator::new(net.hidden_size);
            refreshed.refresh(net, &st);
            let refreshed_eval = evaluate_nnue_acc(net, &refreshed, &st);
            if st.w {
                refreshed_eval
            } else {
                -refreshed_eval
            }
        })
        .expect("embedded test NNUE should load");

        assert_eq!(copied_eval, expected);
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
}
