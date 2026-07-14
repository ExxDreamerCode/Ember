use crate::backend::{
    default_search_backend, parse_search_backend_name, search_backend_available,
    x86_avx512_available, x86_v3_available,
};
use crate::board::{
    all_occ, attacked_by, bit, has_non_pawn, is_attacked, is_white_piece, move_ec, move_er,
    move_from, move_promotion, move_sc, move_sr, move_to, piece_on, piece_type,
    promotion_piece_index, see, BoardState, Move, BK, BP, BR, EMPTY_SQ, INF, KING_ATTACKS, MATE,
    MAX_PLY, QS_DEPTH, WK, WP, WR,
};
use crate::evaluate::{current_nnue_net, evaluate, evaluate_nnue_acc_with_backend};
use crate::movegen::{
    apply_move, apply_move_mode, generate_moves, generate_moves_into_mode,
    generate_pseudo_captures_promotions_into_mode, generate_pseudo_moves_into_mode,
    is_chess960_castling_move_mode, try_apply_move_mode,
};
#[cfg(target_arch = "x86_64")]
use crate::nnue::Avx512NnueBackend;
use crate::nnue::{NNUEAccumulator, NNUENet, NnueBackend, ScalarNnueBackend, SimdNnueBackend};
use crate::syzygy::SyzygyTables;
use crate::tt::{SharedTT, TT_ALPHA, TT_BETA, TT_EXACT};
use crate::zobrist::{compute_pawn_hash, ep_hash_square, zobrist};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub use crate::backend::SearchBackendKind;

const SEARCH_BACKEND_ENV: &str = "EMBER_SEARCH_BACKEND";

static SEARCH_BACKEND: OnceLock<SearchBackendKind> = OnceLock::new();
static SEARCH_BACKEND_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[inline]
pub fn active_search_backend() -> SearchBackendKind {
    if let Some(backend) = search_backend_from_id(SEARCH_BACKEND_OVERRIDE.load(Ordering::Relaxed)) {
        return backend;
    }
    *SEARCH_BACKEND.get_or_init(detect_search_backend)
}

pub fn set_search_backend_override(backend: Option<SearchBackendKind>) {
    let id = backend.map(search_backend_id).unwrap_or(0);
    SEARCH_BACKEND_OVERRIDE.store(id, Ordering::SeqCst);
}

fn detect_search_backend() -> SearchBackendKind {
    if let Ok(value) = std::env::var(SEARCH_BACKEND_ENV) {
        if let Some(backend) = parse_search_backend_name(&value) {
            if search_backend_available(backend) {
                return backend;
            }
        }
    }

    default_search_backend()
}

fn search_backend_id(backend: SearchBackendKind) -> u8 {
    match backend {
        SearchBackendKind::Scalar => 1,
        SearchBackendKind::X86V3 => 2,
        SearchBackendKind::Aarch64Simd => 3,
        SearchBackendKind::X86Avx512 => 4,
    }
}

fn search_backend_from_id(id: u8) -> Option<SearchBackendKind> {
    match id {
        1 => Some(SearchBackendKind::Scalar),
        2 => Some(SearchBackendKind::X86V3),
        3 => Some(SearchBackendKind::Aarch64Simd),
        4 => Some(SearchBackendKind::X86Avx512),
        _ => None,
    }
}

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
fn is_promotion_move(fpi: u8, mv: Move) -> bool {
    move_promotion(mv) != 0
        || (fpi != EMPTY_SQ && piece_type(fpi) == 0 && (move_er(mv) == 0 || move_er(mv) == 7))
}

fn promotion_value(mv: Move) -> i32 {
    match move_promotion(mv).to_ascii_uppercase() {
        b'N' => piece_val(1),
        b'B' => piece_val(2),
        b'R' => piece_val(3),
        b'Q' => piece_val(4),
        _ => 0,
    }
}

#[inline]
fn is_en_passant_capture(st: &BoardState, fpi: u8, mv: Move, to: usize, tpi: u8) -> bool {
    fpi != EMPTY_SQ
        && tpi == EMPTY_SQ
        && piece_type(fpi) == 0
        && Some(to) == st.ep
        && move_sc(mv) != move_ec(mv)
}

#[inline]
fn capture_victim_value<const CHESS960: bool>(
    st: &BoardState,
    fpi: u8,
    mv: Move,
    to: usize,
    tpi: u8,
) -> i32 {
    if is_chess960_castling_move_mode::<CHESS960>(st, mv) {
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
fn move_is_capture<const CHESS960: bool>(
    st: &BoardState,
    fpi: u8,
    mv: Move,
    to: usize,
    tpi: u8,
) -> bool {
    !is_chess960_castling_move_mode::<CHESS960>(st, mv)
        && (tpi != EMPTY_SQ || is_en_passant_capture(st, fpi, mv, to, tpi))
}

#[inline]
fn move_see<const CHESS960: bool>(
    st: &BoardState,
    mv: Move,
    from: usize,
    to: usize,
    fpi: u8,
    tpi: u8,
) -> i32 {
    if is_chess960_castling_move_mode::<CHESS960>(st, mv)
        || is_en_passant_capture(st, fpi, mv, to, tpi)
    {
        0
    } else {
        see(&st.bb, from, to)
    }
}

#[inline(always)]
fn special_move_gives_check_mode<const CHESS960: bool>(st: &BoardState, mv: Move) -> bool {
    let from = move_from(mv);
    let to = move_to(mv);
    let fpi = st.mailbox[from];
    if fpi == EMPTY_SQ {
        return false;
    }

    let mut bb = st.bb;
    let mover_is_white = is_white_piece(fpi);
    let mover_type = piece_type(fpi);
    let is_chess960_castle = is_chess960_castling_move_mode::<CHESS960>(st, mv);
    let is_en_passant = mover_type == 0 && Some(to) == st.ep && move_sc(mv) != move_ec(mv);
    let is_standard_castle =
        mover_type == 5 && !CHESS960 && move_sc(mv) == 4 && (move_ec(mv) == 6 || move_ec(mv) == 2);

    if !is_en_passant && !is_chess960_castle && !is_standard_castle {
        return false;
    }

    if !is_chess960_castle {
        let tpi = piece_on(&bb, to);
        if tpi != EMPTY_SQ {
            bb[tpi as usize] &= !bit(to);
        }
    }

    if is_en_passant {
        let cap_sq = if mover_is_white { to + 8 } else { to - 8 };
        let ep_pi = piece_on(&bb, cap_sq);
        if ep_pi != EMPTY_SQ {
            bb[ep_pi as usize] &= !bit(cap_sq);
        }
    }

    if mover_type == 5 && is_chess960_castle {
        let rook_pi = if mover_is_white { WR } else { BR };
        let rook_col = move_ec(mv);
        let (king_dst_col, rook_dst_col) = if rook_col > move_sc(mv) {
            (6usize, 5usize)
        } else {
            (2usize, 3usize)
        };
        bb[rook_pi] &= !bit(move_er(mv) * 8 + rook_col);
        bb[rook_pi] |= bit(move_sr(mv) * 8 + rook_dst_col);
        bb[fpi as usize] &= !bit(from);
        bb[fpi as usize] |= bit(move_sr(mv) * 8 + king_dst_col);
    } else {
        bb[fpi as usize] &= !bit(from);

        if mover_type == 5
            && !CHESS960
            && move_sc(mv) == 4
            && (move_ec(mv) == 6 || move_ec(mv) == 2)
        {
            let rook_pi = if mover_is_white { WR } else { BR };
            let (rook_from, rook_to) = if move_ec(mv) == 6 {
                (move_sr(mv) * 8 + 7, move_sr(mv) * 8 + 5)
            } else {
                (move_sr(mv) * 8, move_sr(mv) * 8 + 3)
            };
            bb[rook_pi] &= !bit(rook_from);
            bb[rook_pi] |= bit(rook_to);
        }

        if mover_type == 0 && (move_er(mv) == 0 || move_er(mv) == 7) {
            if let Some(ppi) = promotion_piece_index(mover_is_white, move_promotion(mv)) {
                bb[ppi] |= bit(to);
            } else {
                bb[fpi as usize] |= bit(to);
            }
        } else {
            bb[fpi as usize] |= bit(to);
        }
    }

    let opponent_king = if st.w { bb[BK] } else { bb[WK] };
    opponent_king != 0 && is_attacked(&bb, opponent_king.trailing_zeros() as usize, st.w)
}

#[cfg(test)]
fn special_move_gives_check(st: &BoardState, mv: Move) -> bool {
    if st.chess960 {
        special_move_gives_check_mode::<true>(st, mv)
    } else {
        special_move_gives_check_mode::<false>(st, mv)
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
    pub nnue_net: Option<Arc<NNUENet>>,
    pub syzygy: SyzygyTables,
    move_bufs: Vec<Vec<Move>>,
    scored_bufs: Vec<Vec<(i32, Move)>>,
    quiets_bufs: Vec<Vec<Move>>,
    caps_bufs: Vec<Vec<Move>>,
    #[cfg(feature = "search-debug")]
    pub debug: SearchDebug,
}

#[derive(Clone, Copy)]
struct ClassicEval;

#[derive(Clone, Copy)]
struct NnueEval<'a, B: NnueBackend> {
    net: &'a NNUENet,
    _backend: B,
}

trait SearchEval: Copy {
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32;

    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn push_acc(
        self,
        searcher: &mut Searcher,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    );

    fn ensure_child_stack(self, searcher: &mut Searcher, ply: usize);

    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize);
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

macro_rules! qsearch_mode_body {
    (
        $this:tt,
        $qsearch_mode:ident,
        $st:ident,
        $alpha:ident,
        $beta:ident,
        $depth:ident,
        $start:ident,
        $tl:ident,
        $cnt:ident,
        $ply:ident,
        $eval:ident
    ) => {{
        *$cnt += 1;
        if $this.time_up($start, $tl) {
            return 0;
        }
        let ks = $st.king_sq($st.w);
        let in_check = crate::board::is_attacked(&$st.bb, ks, !$st.w);

        if !in_check && $this.syzygy.tables.is_some() && SyzygyTables::pieces_ok($st) {
            if let Some(cutoff) = $this.syzygy.probe_cutoff($st, $beta, $alpha) {
                return cutoff;
            }
        }

        if $ply >= 2 && $this.is_repetition() {
            return 0;
        }

        if !in_check {
            let stand = $eval.static_eval::<CHESS960>($this, $st, $ply);
            if stand >= $beta {
                return stand;
            }
            if stand > $alpha {
                $alpha = stand;
            }
            if $depth <= 0 && $alpha - 975 > stand {
                return $alpha;
            }
        } else if $depth <= -4 {
            return $eval.static_eval::<CHESS960>($this, $st, $ply);
        }

        $this.ensure_buf_pools($ply);
        let mut caps = Self::take_buf(&mut $this.move_bufs, $ply);
        if in_check {
            generate_moves_into_mode::<CHESS960>($st, $st.w, &$st.cr, $st.ep, &mut caps);
        } else {
            generate_pseudo_captures_promotions_into_mode::<CHESS960>(
                $st, $st.w, &$st.cr, $st.ep, &mut caps,
            );
        }
        if caps.is_empty() {
            Self::return_buf(&mut $this.move_bufs, $ply, caps);
            return if in_check { -MATE + 1000 } else { $alpha };
        }
        $eval.ensure_child_stack($this, $ply);

        caps.sort_by_key(|mv| {
            let to = move_to(*mv);
            let from = move_from(*mv);
            let vpi = $st.mailbox[to];
            let api = $st.mailbox[from];
            let victim = capture_victim_value::<CHESS960>($st, api, *mv, to, vpi);
            let attacker = if api != EMPTY_SQ {
                piece_val(piece_type(api))
            } else {
                0
            };
            -(victim * 10 - attacker + promotion_value(*mv))
        });

        let mut cap_idx = 0usize;
        while cap_idx < caps.len() {
            let mv = caps[cap_idx];
            cap_idx += 1;
            if $this.time_up($start, $tl) {
                return 0;
            }
            let from = move_from(mv);
            let to = move_to(mv);
            let fpi = $st.mailbox[from];
            let tpi = $st.mailbox[to];
            if !in_check && move_see::<CHESS960>($st, mv, from, to, fpi, tpi) < 0 {
                continue;
            }
            let st_before = *$st;
            let legal = if in_check {
                apply_move_mode::<CHESS960>(
                    $st,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                );
                true
            } else {
                try_apply_move_mode::<CHESS960>($st, mv)
            };
            if !legal {
                continue;
            }
            $eval.push_acc(
                $this,
                &st_before,
                $st,
                move_sr(mv),
                move_sc(mv),
                move_er(mv),
                move_ec(mv),
                move_promotion(mv),
                $ply,
            );
            let score = -$this.$qsearch_mode::<CHESS960, E>(
                $st,
                -$beta,
                -$alpha,
                $depth - 1,
                $start,
                $tl,
                $cnt,
                $ply + 1,
                $eval,
            );
            *$st = st_before;
            if $this.stopped.load(Ordering::Relaxed) {
                return 0;
            }
            if score >= $beta {
                Self::return_buf(&mut $this.move_bufs, $ply, caps);
                return score;
            }
            if score > $alpha {
                $alpha = score;
            }
        }
        Self::return_buf(&mut $this.move_bufs, $ply, caps);
        $alpha
    }};
}

macro_rules! negamax_mode_body {
    (
        $this:tt,
        $negamax_mode:ident,
        $qsearch_mode:ident,
        $st:ident,
        $depth:ident,
        $ply:ident,
        $alpha:ident,
        $beta:ident,
        $can_null:ident,
        $start:ident,
        $tl:ident,
        $cnt:ident,
        $eval:ident
    ) => {{
        *$cnt += 1;
        if $this.time_up($start, $tl) {
            return 0;
        }
        if $ply >= MAX_PLY {
            return $eval.static_eval::<CHESS960>($this, $st, $ply);
        }

        let mut beta = $beta;
        if $ply > 0 {
            let mate_alpha = -MATE + $ply as i32;
            let mate_beta = MATE - $ply as i32;
            if $alpha < mate_alpha {
                $alpha = mate_alpha;
            }
            if beta > mate_beta {
                beta = mate_beta;
            }
            if $alpha >= beta {
                return $alpha;
            }
        }

        let h = $st.hash;

        let tt_data = $this.shared_tt.get_depth(h);
        let tt_move = tt_data.and_then(|(_, _, _, best)| best);
        let tt_score = tt_data.map(|(_, s, _, _)| score_from_tt(s, $ply));
        let tt_depth = tt_data.map(|(d, _, _, _)| d).unwrap_or(-1);
        let tt_flag = tt_data.map(|(_, _, f, _)| f);

        let ks = $st.king_sq($st.w);
        let in_check = crate::board::is_attacked(&$st.bb, ks, !$st.w);
        let is_pv = beta - $alpha > 1;
        let is_root = $ply == 0;

        let ext = if in_check && $depth < 16 { 1 } else { 0 };
        let actual_depth = $depth + ext;

        if !is_pv && tt_depth >= actual_depth {
            if let (Some(flag), Some(s)) = (tt_flag, tt_score) {
                match flag {
                    TT_EXACT => return s,
                    TT_ALPHA if s <= $alpha => return $alpha,
                    TT_BETA if s >= beta => return beta,
                    _ => {}
                }
            }
        }

        let king_pressure = if in_check {
            8
        } else {
            tactical_king_pressure($st)
        };

        let tb_available =
            !in_check && $this.syzygy.tables.is_some() && SyzygyTables::pieces_ok($st);

        let eval_score = if tb_available {
            $this
                .probe_syzygy($st)
                .unwrap_or_else(|| $eval.static_eval::<CHESS960>($this, $st, $ply))
        } else {
            $eval.static_eval::<CHESS960>($this, $st, $ply)
        };

        if tb_available && !is_pv && !is_root {
            if let Some(cutoff) = $this.syzygy.probe_cutoff($st, beta, $alpha) {
                return cutoff;
            }
        }

        if $ply > 0 && $this.is_repetition() {
            return 0;
        }

        if actual_depth <= 0 {
            return $this.$qsearch_mode::<CHESS960, E>(
                $st, $alpha, beta, QS_DEPTH, $start, $tl, $cnt, $ply, $eval,
            );
        }

        if $this.reverse_futility_enabled() && !in_check && !is_pv && actual_depth <= 8 && $ply > 0
        {
            let margin = 80 + 65 * actual_depth;
            if eval_score - margin >= beta {
                return eval_score - margin;
            }
        }
        if $this.futility_enabled() && !in_check && !is_pv && actual_depth <= 3 && $ply > 0 {
            let margin = 150 * actual_depth;
            if eval_score + margin <= $alpha {
                let q = $this.$qsearch_mode::<CHESS960, E>(
                    $st,
                    $alpha - margin,
                    beta - margin,
                    QS_DEPTH,
                    $start,
                    $tl,
                    $cnt,
                    $ply,
                    $eval,
                );
                if q + margin <= $alpha {
                    return $alpha;
                }
            }
        }
        if $this.null_move_enabled()
            && king_pressure < 3
            && !in_check
            && $can_null
            && !is_pv
            && $ply > 0
            && actual_depth >= 3
            && has_non_pawn(&$st.bb, $st.w)
            && eval_score >= beta
        {
            let total_non_pawn = (all_occ(&$st.bb) & !($st.bb[WP] | $st.bb[BP])).count_ones();
            if total_non_pawn > 4 {
                let r = 3 + actual_depth / 4 + ((eval_score - beta) / 200).min(3);
                let ow = $st.w;
                let oe = $st.ep;
                let old_ep_hash = ep_hash_square($st);
                let z = zobrist();
                if let Some(ep_sq) = old_ep_hash {
                    $st.hash ^= z.ep[ep_sq];
                }
                $st.hash ^= z.side;
                $st.ep = None;
                $st.w = !$st.w;
                $eval.copy_null_acc($this, $ply);
                let null_h = $st.hash;
                $this.rep_stack.push(null_h);
                $this.rep_stack_len += 1;
                let s = -$this.$negamax_mode::<CHESS960, E>(
                    $st,
                    actual_depth - r - 1,
                    $ply + 1,
                    -beta,
                    -beta + 1,
                    false,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                );
                $this.rep_stack.pop();
                $this.rep_stack_len -= 1;
                $st.hash ^= z.side;
                if let Some(ep_sq) = old_ep_hash {
                    $st.hash ^= z.ep[ep_sq];
                }
                $st.w = ow;
                $st.ep = oe;
                if $this.time_up($start, $tl) {
                    return 0;
                }
                if s >= beta {
                    return beta;
                }
            }
        }

        $this.ensure_buf_pools($ply);
        let mut moves_buf = Self::take_buf(&mut $this.move_bufs, $ply);
        let pseudo_moves = !in_check;
        if pseudo_moves {
            generate_pseudo_moves_into_mode::<CHESS960>(
                $st,
                $st.w,
                &$st.cr,
                $st.ep,
                &mut moves_buf,
            );
        } else {
            generate_moves_into_mode::<CHESS960>($st, $st.w, &$st.cr, $st.ep, &mut moves_buf);
        }
        if moves_buf.is_empty() {
            Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
            return if in_check { -MATE + $ply as i32 } else { 0 };
        }

        let actual_depth =
            if $this.iid_reduction_enabled() && tt_move.is_none() && actual_depth >= 4 && is_pv {
                actual_depth - 1
            } else {
                actual_depth
            };

        let mut scored = Self::take_buf(&mut $this.scored_bufs, $ply);
        scored.clear();
        scored.reserve(moves_buf.len());
        for &mv in moves_buf.iter() {
            let mut s = 0i32;
            if Some(mv) == tt_move {
                s = 10_000_000;
            } else {
                let from = move_from(mv);
                let to = move_to(mv);
                let tpi = $st.mailbox[to];
                let fpi = $st.mailbox[from];
                let is_promo = is_promotion_move(fpi, mv);
                if move_is_capture::<CHESS960>($st, fpi, mv, to, tpi) || is_promo {
                    let v = capture_victim_value::<CHESS960>($st, fpi, mv, to, tpi);
                    let a = if fpi != EMPTY_SQ {
                        piece_val(piece_type(fpi))
                    } else {
                        0
                    };
                    let see_sc = move_see::<CHESS960>($st, mv, from, to, fpi, tpi);
                    if see_sc >= 0 {
                        s += 2_000_000 + v * 10 - a + see_sc;
                    } else {
                        s += 500_000 + v * 10 - a;
                    }
                    if is_promo {
                        s += 1_500_000 + promotion_value(mv);
                    }
                } else {
                    if $this.killers[$ply][0] == Some(mv) {
                        s += 900_000;
                    } else if $this.killers[$ply][1] == Some(mv) {
                        s += 800_000;
                    }
                    let p_idx = if fpi != EMPTY_SQ {
                        piece_to_idx(piece_type(fpi))
                    } else {
                        0
                    };
                    if $this.counter_move[p_idx][to] == Some(mv) {
                        s += 700_000;
                    }
                    let (fk, tk) = from_to_key(move_sr(mv), move_sc(mv), move_er(mv), move_ec(mv));
                    s += $this.history[fk][tk].clamp(-32768, 32768);
                }
            }
            scored.push((s, mv));
        }
        Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
        scored.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));

        let lmp_count =
            if $this.lmp_enabled() && king_pressure < 3 && !is_pv && !in_check && actual_depth <= 8
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

        let orig_alpha = $alpha;
        let mut best_score = -INF;
        let mut best_move = None;
        let mut legal_moves_seen = 0usize;
        let mut quiets_tried = Self::take_buf(&mut $this.quiets_bufs, $ply);
        quiets_tried.clear();

        for &(_, mv) in scored.iter() {
            if $this.time_up($start, $tl) {
                return 0;
            }

            let from = move_from(mv);
            let to = move_to(mv);
            let fpi = $st.mailbox[from];
            let tpi = $st.mailbox[to];
            let capture = move_is_capture::<CHESS960>($st, fpi, mv, to, tpi);
            let is_promo = is_promotion_move(fpi, mv);
            let is_quiet = !capture && !is_promo;

            if !is_pv && !in_check && is_quiet && legal_moves_seen >= lmp_count {
                break;
            }
            if !is_pv && !in_check && legal_moves_seen > 0 && best_score > -MATE / 2 {
                if capture {
                    if $this.see_pruning_enabled()
                        && move_see::<CHESS960>($st, mv, from, to, fpi, tpi) < -80 * actual_depth
                    {
                        continue;
                    }
                } else if is_quiet && $this.history_pruning_enabled() {
                    let (fk, tk) = from_to_key(move_sr(mv), move_sc(mv), move_er(mv), move_ec(mv));
                    if actual_depth <= 5 && $this.history[fk][tk] < -1024 * actual_depth {
                        continue;
                    }
                }
            }

            let move_ext = if !in_check
                && legal_moves_seen == 0
                && !is_quiet
                && actual_depth <= 2
                && special_move_gives_check_mode::<CHESS960>($st, mv)
            {
                1
            } else {
                0
            };

            let st_before = *$st;
            let legal = if pseudo_moves {
                try_apply_move_mode::<CHESS960>($st, mv)
            } else {
                apply_move_mode::<CHESS960>(
                    $st,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                );
                true
            };
            if !legal {
                continue;
            }
            let move_index = legal_moves_seen;
            legal_moves_seen += 1;

            $eval.push_acc(
                $this,
                &st_before,
                $st,
                move_sr(mv),
                move_sc(mv),
                move_er(mv),
                move_ec(mv),
                move_promotion(mv),
                $ply,
            );

            let h_after = $st.hash;
            $this.rep_stack.push(h_after);
            $this.rep_stack_len += 1;

            let new_depth = actual_depth - 1 + move_ext;

            let lmr_eligible = $this.lmr_enabled()
                && move_index >= 2
                && actual_depth >= 3
                && is_quiet
                && !in_check;
            let s = if move_index == 0 {
                -$this.$negamax_mode::<CHESS960, E>(
                    $st,
                    new_depth,
                    $ply + 1,
                    -beta,
                    -$alpha,
                    true,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                )
            } else if lmr_eligible {
                let r = {
                    let base =
                        (0.5 + (move_index as f64).ln() * (actual_depth as f64).ln() / 1.8) as i32;
                    let r = base.min(actual_depth - 1).max(1);
                    if !is_pv {
                        (r + 1).min(actual_depth - 1)
                    } else {
                        r
                    }
                };
                let s2 = -$this.$negamax_mode::<CHESS960, E>(
                    $st,
                    new_depth - r,
                    $ply + 1,
                    -$alpha - 1,
                    -$alpha,
                    true,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                );
                if s2 > $alpha {
                    let s3 = -$this.$negamax_mode::<CHESS960, E>(
                        $st,
                        new_depth,
                        $ply + 1,
                        -$alpha - 1,
                        -$alpha,
                        true,
                        $start,
                        $tl,
                        $cnt,
                        $eval,
                    );
                    if s3 > $alpha && is_pv {
                        -$this.$negamax_mode::<CHESS960, E>(
                            $st,
                            new_depth,
                            $ply + 1,
                            -beta,
                            -$alpha,
                            true,
                            $start,
                            $tl,
                            $cnt,
                            $eval,
                        )
                    } else {
                        s3
                    }
                } else {
                    s2
                }
            } else if is_pv {
                let s2 = -$this.$negamax_mode::<CHESS960, E>(
                    $st,
                    new_depth,
                    $ply + 1,
                    -$alpha - 1,
                    -$alpha,
                    true,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                );
                if s2 > $alpha && s2 < beta {
                    -$this.$negamax_mode::<CHESS960, E>(
                        $st,
                        new_depth,
                        $ply + 1,
                        -beta,
                        -$alpha,
                        true,
                        $start,
                        $tl,
                        $cnt,
                        $eval,
                    )
                } else {
                    s2
                }
            } else {
                -$this.$negamax_mode::<CHESS960, E>(
                    $st,
                    new_depth,
                    $ply + 1,
                    -beta,
                    -$alpha,
                    true,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                )
            };

            $this.rep_stack.pop();
            $this.rep_stack_len -= 1;
            *$st = st_before;

            if $this.stopped.load(Ordering::Relaxed) {
                return 0;
            }

            if is_quiet {
                quiets_tried.push(mv);
            }

            if s > best_score {
                best_score = s;
                best_move = Some(mv);
                if s > $alpha {
                    $alpha = s;
                    if $alpha >= beta {
                        if is_quiet {
                            if $this.killers[$ply][0] != Some(mv) {
                                $this.killers[$ply][1] = $this.killers[$ply][0];
                                $this.killers[$ply][0] = Some(mv);
                            }
                            let (fk, tk) =
                                from_to_key(move_sr(mv), move_sc(mv), move_er(mv), move_ec(mv));
                            let bonus = (actual_depth * actual_depth).min(512);
                            $this.history[fk][tk] += bonus;
                            if $this.history[fk][tk] > 16384 {
                                for a in 0..64 {
                                    for b in 0..64 {
                                        $this.history[a][b] /= 2;
                                    }
                                }
                            }
                            for &qmv in &quiets_tried {
                                if qmv == mv {
                                    continue;
                                }
                                let (qfk, qtk) = from_to_key(
                                    move_sr(qmv),
                                    move_sc(qmv),
                                    move_er(qmv),
                                    move_ec(qmv),
                                );
                                $this.history[qfk][qtk] -= bonus;
                                if $this.history[qfk][qtk] < -16384 {
                                    for a in 0..64 {
                                        for b in 0..64 {
                                            $this.history[a][b] /= 2;
                                        }
                                    }
                                }
                            }
                            let p_idx = if fpi != EMPTY_SQ {
                                piece_to_idx(piece_type(fpi))
                            } else {
                                0
                            };
                            $this.counter_move[p_idx][to] = Some(mv);
                        }
                        break;
                    }
                }
            }
        }

        Self::return_buf(&mut $this.scored_bufs, $ply, scored);
        Self::return_buf(&mut $this.quiets_bufs, $ply, quiets_tried);

        if $this.stopped.load(Ordering::Relaxed) {
            return 0;
        }
        if legal_moves_seen == 0 {
            return if in_check { -MATE + $ply as i32 } else { 0 };
        }

        let flag = if best_score <= orig_alpha {
            TT_ALPHA
        } else if best_score >= beta {
            TT_BETA
        } else {
            TT_EXACT
        };
        $this.shared_tt.store(
            h,
            actual_depth,
            score_to_tt(best_score, $ply),
            flag,
            best_move,
        );
        best_score
    }};
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
            nnue_net: current_nnue_net(),
            syzygy: SyzygyTables::new(),
            move_bufs: Vec::new(),
            scored_bufs: Vec::new(),
            quiets_bufs: Vec::new(),
            caps_bufs: Vec::new(),
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

    pub fn refresh_nnue_net(&mut self) {
        self.nnue_net = current_nnue_net();
    }

    pub fn init_nnue_stack(&mut self, st: &BoardState) {
        if let Some(net) = self.nnue_net.as_deref() {
            if self.nnue_stack.len() < MAX_PLY + 1 {
                self.nnue_stack
                    .resize(MAX_PLY + 1, NNUEAccumulator::new(net.hidden_size));
            }
            self.nnue_stack[0].refresh(net, st);
        }
    }

    pub fn refresh_nnue_stack_at(&mut self, ply: usize, st: &BoardState) {
        let Some(net) = self.nnue_net.as_deref() else {
            return;
        };
        if self.nnue_stack.len() <= ply {
            self.nnue_stack
                .resize(ply + 1, NNUEAccumulator::new(net.hidden_size));
        }
        self.nnue_stack[ply].refresh(net, st);
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

    const BUF_POOL_CAP: usize = MAX_PLY + 64;

    fn ensure_buf_pools(&mut self, ply: usize) {
        let need = (ply + 1).min(Self::BUF_POOL_CAP);
        if self.move_bufs.len() < need {
            self.move_bufs.resize_with(need, Vec::new);
            self.scored_bufs.resize_with(need, Vec::new);
            self.quiets_bufs.resize_with(need, Vec::new);
            self.caps_bufs.resize_with(need, Vec::new);
        }
    }

    #[inline]
    fn take_buf<T>(pool: &mut [Vec<T>], ply: usize) -> Vec<T> {
        if ply < pool.len() {
            std::mem::take(&mut pool[ply])
        } else {
            Vec::new()
        }
    }

    #[inline]
    fn return_buf<T>(pool: &mut [Vec<T>], ply: usize, buf: Vec<T>) {
        if ply < pool.len() {
            pool[ply] = buf;
        }
    }

    pub fn copy_root_context_to(&self, dst: &mut Searcher) {
        dst.rep_stack = self.rep_stack.clone();
        dst.rep_stack_len = self.rep_stack_len;
        dst.corr_hist = self.corr_hist;
        dst.nnue_net = self.nnue_net.clone();
        dst.syzygy = self.syzygy.clone();
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

    #[inline(always)]
    fn static_eval_classic<const CHESS960: bool>(&self, st: &BoardState) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return evaluate(st) * if st.w { 1 } else { -1 };
        }
        evaluate(st) * if st.w { 1 } else { -1 }
    }

    #[inline(always)]
    fn static_eval_nnue<const CHESS960: bool, B: NnueBackend>(
        &self,
        st: &BoardState,
        ply: usize,
        net: &NNUENet,
    ) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return evaluate(st) * if st.w { 1 } else { -1 };
        }
        let score = if ply < self.nnue_stack.len() {
            evaluate_nnue_acc_with_backend::<B>(net, &self.nnue_stack[ply], st)
        } else {
            let mut acc = NNUEAccumulator::new(net.hidden_size);
            B::refresh(&mut acc, net, st);
            evaluate_nnue_acc_with_backend::<B>(net, &acc, st)
        };
        if st.w {
            score
        } else {
            -score
        }
    }

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
        match (st.chess960, self.nnue_net.as_deref()) {
            (true, Some(net)) => NnueEval {
                net,
                _backend: ScalarNnueBackend,
            }
            .corrected_eval::<true>(self, st),
            (true, None) => ClassicEval.corrected_eval::<true>(self, st),
            (false, Some(net)) => NnueEval {
                net,
                _backend: ScalarNnueBackend,
            }
            .corrected_eval::<false>(self, st),
            (false, None) => ClassicEval.corrected_eval::<false>(self, st),
        }
    }

    fn corrected_eval_classic<const CHESS960: bool>(&self, st: &BoardState) -> i32 {
        if CHESS960 && st.mc <= 3 {
            let base = evaluate(st) * if st.w { 1 } else { -1 };
            if self.corr_hist_enabled() {
                let ph = compute_pawn_hash(st);
                let idx = corr_idx(ph, st.w);
                return base + self.corr_hist[idx].clamp(-200, 200);
            }
            return base;
        }
        let base = evaluate(st) * if st.w { 1 } else { -1 };
        if self.corr_hist_enabled() {
            let ph = compute_pawn_hash(st);
            let idx = corr_idx(ph, st.w);
            return base + self.corr_hist[idx].clamp(-200, 200);
        }
        base
    }

    #[inline(always)]
    fn corrected_eval_nnue<const CHESS960: bool, B: NnueBackend>(
        &self,
        st: &BoardState,
        net: &NNUENet,
    ) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return self.corrected_eval_classic::<CHESS960>(st);
        }
        let mut acc = NNUEAccumulator::new(net.hidden_size);
        B::refresh(&mut acc, net, st);
        let score = evaluate_nnue_acc_with_backend::<B>(net, &acc, st);
        if st.w {
            score
        } else {
            -score
        }
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
        for i in (0..self.rep_stack_len - 1).rev() {
            if self.rep_stack[i] == last {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_nnue_acc<B: NnueBackend>(
        &mut self,
        net: &NNUENet,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        if ply + 1 >= self.nnue_stack.len() {
            return;
        }
        let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
        right[0].clone_from(&left[ply]);

        let ok = B::update_move(
            &mut self.nnue_stack[ply + 1],
            net,
            st_before,
            sr,
            sc,
            er,
            ec,
            promotion,
        );

        if !ok {
            B::refresh(&mut self.nnue_stack[ply + 1], net, st_after);
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn qsearch(
        &mut self,
        st: &mut BoardState,
        alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
    ) -> i32 {
        self.qsearch_scalar(st, alpha, beta, depth, start, tl, cnt, ply)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn qsearch_scalar(
        &mut self,
        st: &mut BoardState,
        alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
    ) -> i32 {
        let nnue_net = self.nnue_net.clone();
        match (st.chess960, nnue_net.as_deref()) {
            (true, Some(net)) => self.qsearch_mode_scalar::<true, _>(
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                NnueEval {
                    net,
                    _backend: ScalarNnueBackend,
                },
            ),
            (true, None) => self.qsearch_mode_scalar::<true, _>(
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                ClassicEval,
            ),
            (false, Some(net)) => self.qsearch_mode_scalar::<false, _>(
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                NnueEval {
                    net,
                    _backend: ScalarNnueBackend,
                },
            ),
            (false, None) => self.qsearch_mode_scalar::<false, _>(
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                ClassicEval,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qsearch_mode_scalar<const CHESS960: bool, E: SearchEval>(
        &mut self,
        st: &mut BoardState,
        mut alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
        eval: E,
    ) -> i32 {
        qsearch_mode_body!(
            self,
            qsearch_mode_scalar,
            st,
            alpha,
            beta,
            depth,
            start,
            tl,
            cnt,
            ply,
            eval
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qsearch_mode_simd256<const CHESS960: bool, E: SearchEval>(
        &mut self,
        st: &mut BoardState,
        mut alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
        eval: E,
    ) -> i32 {
        qsearch_mode_body!(
            self,
            qsearch_mode_simd256,
            st,
            alpha,
            beta,
            depth,
            start,
            tl,
            cnt,
            ply,
            eval
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn qsearch_mode_x86_v3<const CHESS960: bool, E: SearchEval>(
        &mut self,
        st: &mut BoardState,
        mut alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
        eval: E,
    ) -> i32 {
        unsafe {
            qsearch_mode_body!(
                self,
                qsearch_mode_x86_v3,
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                eval
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt"
    )]
    #[allow(clippy::too_many_arguments)]
    unsafe fn qsearch_mode_x86_avx512<const CHESS960: bool, E: SearchEval>(
        &mut self,
        st: &mut BoardState,
        mut alpha: i32,
        beta: i32,
        depth: i32,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
        ply: usize,
        eval: E,
    ) -> i32 {
        unsafe {
            qsearch_mode_body!(
                self,
                qsearch_mode_x86_avx512,
                st,
                alpha,
                beta,
                depth,
                start,
                tl,
                cnt,
                ply,
                eval
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn negamax(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        match active_search_backend() {
            SearchBackendKind::X86Avx512 if x86_avx512_available() => {
                #[cfg(target_arch = "x86_64")]
                {
                    return unsafe {
                        self.negamax_x86_avx512(
                            st, depth, ply, alpha, beta, can_null, start, tl, cnt,
                        )
                    };
                }
                #[allow(unreachable_code)]
                self.negamax_scalar(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
            }
            SearchBackendKind::X86V3 if x86_v3_available() => {
                #[cfg(target_arch = "x86_64")]
                {
                    return unsafe {
                        self.negamax_x86_v3(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
                    };
                }
                #[allow(unreachable_code)]
                self.negamax_scalar(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
            }
            SearchBackendKind::Aarch64Simd
                if search_backend_available(SearchBackendKind::Aarch64Simd) =>
            {
                self.negamax_simd256(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
            }
            _ => self.negamax_scalar(st, depth, ply, alpha, beta, can_null, start, tl, cnt),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_scalar(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        let nnue_net = self.nnue_net.clone();
        match (st.chess960, nnue_net.as_deref()) {
            (true, Some(net)) => self.negamax_mode_scalar::<true, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                NnueEval {
                    net,
                    _backend: ScalarNnueBackend,
                },
            ),
            (true, None) => self.negamax_mode_scalar::<true, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                ClassicEval,
            ),
            (false, Some(net)) => self.negamax_mode_scalar::<false, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                NnueEval {
                    net,
                    _backend: ScalarNnueBackend,
                },
            ),
            (false, None) => self.negamax_mode_scalar::<false, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                ClassicEval,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_simd256(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        let nnue_net = self.nnue_net.clone();
        match (st.chess960, nnue_net.as_deref()) {
            (true, Some(net)) => self.negamax_mode_simd256::<true, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                NnueEval {
                    net,
                    _backend: SimdNnueBackend,
                },
            ),
            (true, None) => self.negamax_mode_simd256::<true, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                ClassicEval,
            ),
            (false, Some(net)) => self.negamax_mode_simd256::<false, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                NnueEval {
                    net,
                    _backend: SimdNnueBackend,
                },
            ),
            (false, None) => self.negamax_mode_simd256::<false, _>(
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                ClassicEval,
            ),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn negamax_x86_v3(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        let nnue_net = self.nnue_net.clone();
        unsafe {
            match (st.chess960, nnue_net.as_deref()) {
                (true, Some(net)) => self.negamax_mode_x86_v3::<true, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    NnueEval {
                        net,
                        _backend: SimdNnueBackend,
                    },
                ),
                (true, None) => self.negamax_mode_x86_v3::<true, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    ClassicEval,
                ),
                (false, Some(net)) => self.negamax_mode_x86_v3::<false, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    NnueEval {
                        net,
                        _backend: SimdNnueBackend,
                    },
                ),
                (false, None) => self.negamax_mode_x86_v3::<false, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    ClassicEval,
                ),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt"
    )]
    #[allow(clippy::too_many_arguments)]
    unsafe fn negamax_x86_avx512(
        &mut self,
        st: &mut BoardState,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        can_null: bool,
        start: Instant,
        tl: f64,
        cnt: &mut u64,
    ) -> i32 {
        let nnue_net = self.nnue_net.clone();
        unsafe {
            match (st.chess960, nnue_net.as_deref()) {
                (true, Some(net)) => self.negamax_mode_x86_avx512::<true, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    NnueEval {
                        net,
                        _backend: Avx512NnueBackend,
                    },
                ),
                (true, None) => self.negamax_mode_x86_avx512::<true, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    ClassicEval,
                ),
                (false, Some(net)) => self.negamax_mode_x86_avx512::<false, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    NnueEval {
                        net,
                        _backend: Avx512NnueBackend,
                    },
                ),
                (false, None) => self.negamax_mode_x86_avx512::<false, _>(
                    st,
                    depth,
                    ply,
                    alpha,
                    beta,
                    can_null,
                    start,
                    tl,
                    cnt,
                    ClassicEval,
                ),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_mode_scalar<const CHESS960: bool, E: SearchEval>(
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
        eval: E,
    ) -> i32 {
        negamax_mode_body!(
            self,
            negamax_mode_scalar,
            qsearch_mode_scalar,
            st,
            depth,
            ply,
            alpha,
            beta,
            can_null,
            start,
            tl,
            cnt,
            eval
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_mode_simd256<const CHESS960: bool, E: SearchEval>(
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
        eval: E,
    ) -> i32 {
        negamax_mode_body!(
            self,
            negamax_mode_simd256,
            qsearch_mode_simd256,
            st,
            depth,
            ply,
            alpha,
            beta,
            can_null,
            start,
            tl,
            cnt,
            eval
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn negamax_mode_x86_v3<const CHESS960: bool, E: SearchEval>(
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
        eval: E,
    ) -> i32 {
        unsafe {
            negamax_mode_body!(
                self,
                negamax_mode_x86_v3,
                qsearch_mode_x86_v3,
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                eval
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt"
    )]
    #[allow(clippy::too_many_arguments)]
    unsafe fn negamax_mode_x86_avx512<const CHESS960: bool, E: SearchEval>(
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
        eval: E,
    ) -> i32 {
        unsafe {
            negamax_mode_body!(
                self,
                negamax_mode_x86_avx512,
                qsearch_mode_x86_avx512,
                st,
                depth,
                ply,
                alpha,
                beta,
                can_null,
                start,
                tl,
                cnt,
                eval
            )
        }
    }
}

impl SearchEval for ClassicEval {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        _ply: usize,
    ) -> i32 {
        searcher.static_eval_classic::<CHESS960>(st)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_classic::<CHESS960>(st)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_acc(
        self,
        _searcher: &mut Searcher,
        _st_before: &BoardState,
        _st_after: &BoardState,
        _sr: usize,
        _sc: usize,
        _er: usize,
        _ec: usize,
        _promotion: u8,
        _ply: usize,
    ) {
    }

    #[inline(always)]
    fn ensure_child_stack(self, _searcher: &mut Searcher, _ply: usize) {}

    #[inline(always)]
    fn copy_null_acc(self, _searcher: &mut Searcher, _ply: usize) {}
}

impl<'a, B: NnueBackend> SearchEval for NnueEval<'a, B> {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32 {
        searcher.static_eval_nnue::<CHESS960, B>(st, ply, self.net)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_nnue::<CHESS960, B>(st, self.net)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn push_acc(
        self,
        searcher: &mut Searcher,
        st_before: &BoardState,
        st_after: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
        ply: usize,
    ) {
        searcher.push_nnue_acc::<B>(
            self.net, st_before, st_after, sr, sc, er, ec, promotion, ply,
        );
    }

    #[inline(always)]
    fn ensure_child_stack(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 >= searcher.nnue_stack.len() && ply + 1 < MAX_PLY + 1 {
            searcher
                .nnue_stack
                .resize(ply + 2, NNUEAccumulator::new(self.net.hidden_size));
        }
    }

    #[inline(always)]
    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 < searcher.nnue_stack.len() {
            let (left, right) = searcher.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
        }
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

fn diversify_lazy_smp_root_moves(moves: &mut [Move], thread_id: usize) {
    if moves.len() <= 1 || thread_id == 0 {
        return;
    }
    let offset = thread_id % moves.len();
    moves.rotate_left(offset);
}

pub fn extract_pv_line(shared_tt: &SharedTT, st: &BoardState, first_move: Move) -> Vec<Move> {
    let first_promo = move_promotion(first_move);
    let first_fpi = st.mailbox[move_from(first_move)];
    if first_fpi != EMPTY_SQ
        && piece_type(first_fpi) == 0
        && (move_er(first_move) == 0 || move_er(first_move) == 7)
        && (first_promo == 0
            || (first_promo != b'Q'
                && first_promo != b'R'
                && first_promo != b'B'
                && first_promo != b'N'))
    {
        return vec![];
    }

    let mut pv = vec![first_move];
    let mut prev_st = *st;
    apply_move(
        &mut prev_st,
        move_sr(first_move),
        move_sc(first_move),
        move_er(first_move),
        move_ec(first_move),
        move_promotion(first_move),
    );

    let moved_king_sq = prev_st.king_sq(!prev_st.w);
    if moved_king_sq == 0 || crate::board::is_attacked(&prev_st.bb, moved_king_sq, prev_st.w) {
        return pv;
    }

    let mut seen_hashes = std::collections::HashSet::new();
    seen_hashes.insert(st.hash);
    seen_hashes.insert(prev_st.hash);

    for _ in 0..MAX_PLY.saturating_sub(1) {
        let h = prev_st.hash;
        if let Some((_, _, _, Some(best))) = shared_tt.get_depth(h) {
            let moves = generate_moves(&prev_st, prev_st.w, &prev_st.cr, prev_st.ep);
            if !moves.contains(&best) {
                break;
            }
            let promo = move_promotion(best);
            let fpi = prev_st.mailbox[move_from(best)];
            if fpi != EMPTY_SQ
                && piece_type(fpi) == 0
                && (move_er(best) == 0 || move_er(best) == 7)
                && (promo == 0
                    || (promo != b'Q' && promo != b'R' && promo != b'B' && promo != b'N'))
            {
                break;
            }
            pv.push(best);
            apply_move(
                &mut prev_st,
                move_sr(best),
                move_sc(best),
                move_er(best),
                move_ec(best),
                promo,
            );
            let moved_king_sq = prev_st.king_sq(!prev_st.w);
            if moved_king_sq == 0
                || crate::board::is_attacked(&prev_st.bb, moved_king_sq, prev_st.w)
            {
                pv.pop();
                break;
            }
            let h_after = prev_st.hash;
            if !seen_hashes.insert(h_after) {
                pv.pop();
                break;
            }
        } else {
            break;
        }
    }
    pv
}

pub fn lazy_smp_search(
    shared_tt: Arc<SharedTT>,
    st: &BoardState,
    root_moves: &[Move],
    time_limit: f64,
    depth_limit: i32,
    num_threads: usize,
    root_searcher: &Searcher,
) -> (Move, i32, i32, u64) {
    let stopped = Arc::new(AtomicBool::new(false));
    let all_moves = root_moves.to_vec();

    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let global_best_depth: Arc<AtomicI32> = Arc::new(AtomicI32::new(0));
    let global_nodes: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let root_hash = st.hash;

    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let mut my_moves = all_moves.clone();
        diversify_lazy_smp_root_moves(&mut my_moves, thread_id);

        let shared_tt = Arc::clone(&shared_tt);
        let stopped = Arc::clone(&stopped);
        let results = Arc::clone(&results);
        let global_best_depth = Arc::clone(&global_best_depth);
        let global_nodes = Arc::clone(&global_nodes);
        let st = *st;
        let mut root_context = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
        root_searcher.copy_root_context_to(&mut root_context);
        let handle = std::thread::Builder::new()
            .name(format!("rts-{}", thread_id))
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let mut searcher = Searcher::new(shared_tt, Arc::clone(&stopped));
                root_context.copy_root_context_to(&mut searcher);
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
                            let legal_root = generate_moves(&st, st.w, &st.cr, st.ep);
                            if legal_root.contains(&tt_mv) {
                                my_moves.push(tt_mv);
                            }
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
                                move_sr(mv),
                                move_sc(mv),
                                move_er(mv),
                                move_ec(mv),
                                move_promotion(mv),
                            );
                            searcher.refresh_nnue_stack_at(1, &s);
                            let h = s.hash;
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
                            let pv_line = extract_pv_line(&searcher.shared_tt, &st, asp_best);
                            let pv_str = pv_line
                                .iter()
                                .map(|m| crate::board::move_to_uci(&st, *m))
                                .collect::<Vec<_>>()
                                .join(" ");
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
                                pv_str
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
    use crate::board::encode_move;
    use crate::engine::Engine;
    use crate::zobrist::compute_hash;
    use std::time::Duration;

    fn state_from_fen(fen: &str) -> BoardState {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        engine.st
    }

    fn legal_move(st: &BoardState, uci: &str) -> Move {
        generate_moves(st, st.w, &st.cr, st.ep)
            .into_iter()
            .find(|mv| crate::board::move_to_uci(st, *mv) == uci)
            .unwrap_or_else(|| panic!("expected legal move {uci}"))
    }

    #[test]
    fn negamax_handles_stalemate_with_only_pseudo_king_moves() {
        let mut st = state_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        assert!(generate_moves(&st, st.w, &st.cr, st.ep).is_empty());
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut searcher = Searcher::new(shared_tt.clone(), stopped);
        searcher.init_nnue_stack(&st);
        let root_key = compute_hash(&st);
        let mut nodes = 0u64;

        let score = searcher.negamax(
            &mut st,
            2,
            0,
            -INF,
            INF,
            true,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(score, 0);
        assert!(
            shared_tt
                .get_depth(root_key)
                .and_then(|(_, _, _, best)| best)
                .is_none(),
            "stalemate must not store a pseudo-legal best move"
        );
    }

    #[test]
    fn special_move_gives_check_rejects_empty_from_square() {
        let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
        let mv = encode_move(7, 1, 7, 2, 0);

        assert!(!special_move_gives_check(&st, mv));
    }

    #[test]
    fn special_move_gives_check_ignores_normal_rook_check() {
        let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
        let mv = legal_move(&st, "a1a8");

        assert!(!special_move_gives_check(&st, mv));
    }

    #[test]
    fn special_move_gives_check_rejects_quiet_non_check() {
        let st = state_from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
        let mv = legal_move(&st, "a1a2");

        assert!(!special_move_gives_check(&st, mv));
    }

    #[test]
    fn special_move_gives_check_detects_en_passant_discovery() {
        let st = state_from_fen("8/6pp/8/R2pP1k1/6B1/8/6PP/6K1 w - d6 0 1");
        let mv = legal_move(&st, "e5d6");

        assert!(special_move_gives_check(&st, mv));
    }

    #[test]
    fn special_move_gives_check_rejects_non_check_en_passant() {
        let st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let mv = legal_move(&st, "e5d6");

        assert!(!special_move_gives_check(&st, mv));
    }

    #[test]
    fn special_move_gives_check_detects_castling_rook_discovery() {
        let st = state_from_fen("5k2/8/8/8/8/8/8/4K2R w K - 0 1");
        let mv = legal_move(&st, "e1g1");

        assert!(special_move_gives_check(&st, mv));
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
    fn negamax_prefers_en_passant_discovered_check() {
        let mut st = state_from_fen("8/6pp/8/R2pP1k1/6B1/8/6PP/6K1 w - d6 0 1");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut searcher = Searcher::new(shared_tt.clone(), stopped);
        searcher.init_nnue_stack(&st);
        let root_key = compute_hash(&st);
        let mut nodes = 0u64;

        let score = searcher.negamax(
            &mut st,
            2,
            0,
            -INF,
            INF,
            true,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        let best_move = shared_tt
            .get_depth(root_key)
            .and_then(|(_, _, _, best_move)| best_move)
            .expect("negamax should store the root best move");
        let best_uci = crate::board::move_to_uci(&st, best_move);
        assert_eq!(
            best_uci, "e5d6",
            "search chose {best_uci} instead of the checking en-passant discovery e5d6; score={score}, nodes={nodes}"
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
    fn lazy_smp_worker_context_copies_root_search_state() {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut root = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
        let mut worker = Searcher::new(shared_tt, stopped);

        root.rep_stack.extend([11, 22, 33, 44]);
        root.rep_stack_len = 4;
        root.corr_hist[123] = 17;
        root.corr_hist[456] = -23;
        root.syzygy = SyzygyTables::new();

        root.copy_root_context_to(&mut worker);

        assert_eq!(worker.rep_stack, root.rep_stack);
        assert_eq!(worker.rep_stack_len, root.rep_stack_len);
        assert_eq!(worker.corr_hist[123], 17);
        assert_eq!(worker.corr_hist[456], -23);
        assert_eq!(worker.syzygy.tables.is_some(), root.syzygy.tables.is_some());
    }

    #[test]
    fn lazy_smp_root_diversification_changes_nonzero_worker_order() {
        let original = vec![
            encode_move(0, 0, 0, 0, 0),
            encode_move(0, 1, 0, 1, 0),
            encode_move(0, 2, 0, 2, 0),
            encode_move(0, 3, 0, 3, 0),
        ];
        let mut thread_zero = original.clone();
        let mut thread_one = original.clone();

        diversify_lazy_smp_root_moves(&mut thread_zero, 0);
        diversify_lazy_smp_root_moves(&mut thread_one, 1);

        assert_eq!(thread_zero, original);
        assert_ne!(thread_one, original);

        let mut sorted_original = original.clone();
        let mut sorted_thread_one = thread_one;
        sorted_original.sort_unstable();
        sorted_thread_one.sort_unstable();
        assert_eq!(sorted_thread_one, sorted_original);
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

    #[test]
    fn extract_pv_rejects_illegal_first_move_without_promotion() {
        let st = state_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let _stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));

        let bogus = encode_move(1, 2, 0, 0, 0);
        let pv = extract_pv_line(&shared_tt, &st, bogus);
        assert_eq!(pv.len(), 1);
    }

    #[test]
    fn extract_pv_rejects_illegal_tt_move_during_extraction() {
        let st = state_from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let _stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));

        let first_move = encode_move(7, 4, 6, 4, 0);
        let bogus_tt_move = encode_move(0, 0, 0, 7, 0);
        let after_st = {
            let mut s = st;
            apply_move(&mut s, 7, 4, 6, 4, 0);
            s
        };
        let after_hash = compute_hash(&after_st);
        shared_tt.store(after_hash, 5, 100, TT_EXACT, Some(bogus_tt_move));

        let pv = extract_pv_line(&shared_tt, &st, first_move);
        assert_eq!(pv.len(), 1, "extract_pv must reject illegal TT moves");
    }

    #[test]
    fn extract_pv_validates_takes_back_king_in_check() {
        let st = state_from_fen("4k3/4r3/8/8/8/8/8/4K3 w - - 0 1");
        let _stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));

        let first_move = encode_move(7, 4, 6, 4, 0);
        let after_st = {
            let mut s = st;
            apply_move(&mut s, 7, 4, 6, 4, 0);
            s
        };
        let after_hash = compute_hash(&after_st);
        let check_move = encode_move(6, 4, 6, 5, 0);
        shared_tt.store(after_hash, 5, 100, TT_EXACT, Some(check_move));

        let pv = extract_pv_line(&shared_tt, &st, first_move);
        assert_eq!(
            pv.len(),
            1,
            "extract_pv must pop moves that leave king in check"
        );
    }
}
