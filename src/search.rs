use crate::backend::{default_search_backend, parse_search_backend_name, search_backend_available};
use crate::board::{
    all_occ, attacked_by, bit, has_non_pawn, is_attacked, is_dead_position, is_white_piece,
    move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to, piece_on, piece_type,
    promotion_piece_index, see, BoardState, Move, BK, BP, BR, EMPTY_SQ, INF, KING_ATTACKS, MATE,
    MAX_HALF_MOVE_CLOCK, MAX_PLY, NO_MOVE, QS_DEPTH, WK, WP, WR,
};
use crate::evaluate::{current_nnue_net, evaluate, evaluate_nnue_acc_with_backend};
use crate::movegen::{
    apply_move, apply_move_mode, generate_moves, generate_moves_into_mode,
    generate_pseudo_captures_promotions_into_mode, generate_pseudo_moves_into_mode,
    is_chess960_castling_move_mode, try_apply_move_mode,
};
#[cfg(target_arch = "x86_64")]
use crate::nnue::Avx512NnueBackend;
use crate::nnue::{
    NNUEAccumulator, NNUENet, NNUEThreatAccumulator, NnueBackend, ScalarNnueBackend,
    Simd128NnueBackend, Simd512NnueBackend, SimdNnueBackend,
};
use crate::syzygy::SyzygyTables;
use crate::time_management::{iteration_time_decision, IterationTiming};
use crate::tt::{SharedTT, TT_ALPHA, TT_BETA, TT_EXACT};
use crate::types::{BLACK, WHITE};
use crate::zobrist::{compute_pawn_hash, ep_hash_square, zobrist};
#[cfg(feature = "search-debug")]
use std::collections::BTreeMap;
#[cfg(feature = "search-debug")]
use std::fs::OpenOptions;
#[cfg(feature = "search-debug")]
use std::io::Write;
#[cfg(feature = "search-debug")]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

pub use crate::backend::SearchBackendKind;

const SEARCH_BACKEND_ENV: &str = "EMBER_SEARCH_BACKEND";
const LAZY_SMP_VERIFICATION_MARGIN_CP: i32 = 25;
const LAZY_SMP_VERIFICATION_TT_MB: usize = 4;
// Singular extensions remain available for controlled search experiments, but
// are not part of the production search until they pass the strength gates.
const SINGULAR_MIN_DEPTH: i32 = 12;
const SINGULAR_TT_DEPTH_MARGIN: i32 = 1;
const SINGULAR_BASE_MARGIN_CP: i32 = 44;
const SINGULAR_MARGIN_PER_DEPTH_CP: i32 = 3;
const SINGULAR_MAX_TT_AGE: u8 = 0;
const SINGULAR_MAX_HALF_MOVE_CLOCK: u8 = 80;
const SINGULAR_POLICY_MIN_DEPTH: i32 = 15;
const SINGULAR_DOUBLE_MIN_DEPTH: i32 = 16;
const SINGULAR_TRIPLE_MIN_DEPTH: i32 = 24;
const SINGULAR_DOUBLE_MARGIN_CP: i32 = 160;
const SINGULAR_TRIPLE_MARGIN_CP: i32 = 240;
const PROBCUT_MIN_DEPTH: i32 = 8;
const PROBCUT_REDUCTION: i32 = 2;
const PROBCUT_MARGIN_CP: i32 = 350;
const ROOT_REPETITION_TIE_MIN_SCORE: i32 = 300;
const ROOT_REPETITION_TIE_MIN_HALFMOVE_CLOCK: u8 = 40;
const ROOT_REPETITION_TIE_MAX_PIECES: u32 = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DrawStatus {
    None,
    SearchCycle,
    Claimable,
    Automatic,
}

pub(crate) fn root_repetition_tie_scope(st: &BoardState) -> bool {
    // A broad "avoid any root twofold" policy lost Elo. Keep this as a narrow
    // high-halfmove conversion guard where another reversible shuffle has real
    // fifty-move-rule cost and the normal search has no score preference.
    st.halfmove_clock >= ROOT_REPETITION_TIE_MIN_HALFMOVE_CLOCK
        && (0..12).map(|piece| st.bb[piece].count_ones()).sum::<u32>()
            <= ROOT_REPETITION_TIE_MAX_PIECES
}

pub(crate) fn prefer_non_repeating_root_on_tie(
    score: i32,
    current_best_repeats: bool,
    candidate_repeats: bool,
) -> bool {
    score >= ROOT_REPETITION_TIE_MIN_SCORE && current_best_repeats && !candidate_repeats
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SingularCandidate {
    mv: Move,
    score: i32,
    beta: i32,
    depth: i32,
    positive_extension: bool,
    max_extension: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SingularMoveAdjustment {
    mv: Move,
    extension: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SingularSearchOutcome {
    Continue(i32),
    Cutoff(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SingularEligibility {
    NoCandidate,
    SafetyRejected,
    Eligible(SingularCandidate),
}

#[derive(Clone, Copy, Debug)]
struct SingularEvidence {
    enabled: bool,
    ply: usize,
    excluded_move: Option<Move>,
    in_check: bool,
    node_pv: bool,
    node_beta: i32,
    actual_depth: i32,
    halfmove_clock: u8,
    repetitions: u8,
    repeated_after_root: bool,
    shuffling: bool,
    path_extensions: u8,
    allow_lower_bound: bool,
    tt_move: Option<Move>,
    tt_score: Option<i32>,
    tt_depth: i32,
    tt_flag: Option<u8>,
    tt_pv: bool,
    tt_age: u8,
    tt_move_is_legal: bool,
}

#[cfg(feature = "search-debug")]
#[derive(Clone, Copy)]
struct ChildPathState {
    ply: usize,
    previous_move: Option<Move>,
    child_ply: Option<usize>,
    previous_extensions: u8,
}

#[cfg(not(feature = "search-debug"))]
#[derive(Clone, Copy)]
struct ChildPathState;

#[cfg(any(feature = "search-debug", test))]
fn next_singular_extension_count(current: u8, extension: i32) -> u8 {
    current.saturating_add(extension.max(0) as u8)
}

fn singular_search_outcome(
    alternative_score: i32,
    singular_beta: i32,
    positive_extension: bool,
    multi_cut_beta: Option<i32>,
    negative_extension: i32,
) -> SingularSearchOutcome {
    if let Some(beta) = multi_cut_beta {
        if alternative_score >= beta {
            return SingularSearchOutcome::Cutoff(beta);
        }
    }
    if alternative_score < singular_beta {
        SingularSearchOutcome::Continue(i32::from(positive_extension))
    } else {
        SingularSearchOutcome::Continue(negative_extension.min(0))
    }
}

fn combine_move_extensions(tactical_extension: i32, singular_extension: i32) -> i32 {
    if tactical_extension > 0 {
        tactical_extension
    } else {
        singular_extension
    }
}

fn singular_extension_from_scores(
    candidate: SingularCandidate,
    base_alternative_score: i32,
    double_alternative_score: Option<i32>,
    triple_alternative_score: Option<i32>,
) -> i32 {
    if !candidate.positive_extension || base_alternative_score >= candidate.beta {
        return 0;
    }
    let mut extension = 1;
    if candidate.max_extension >= 2
        && double_alternative_score
            .is_some_and(|score| score < candidate.score - SINGULAR_DOUBLE_MARGIN_CP)
    {
        extension = 2;
    }
    if extension == 2
        && candidate.max_extension >= 3
        && triple_alternative_score
            .is_some_and(|score| score < candidate.score - SINGULAR_TRIPLE_MARGIN_CP)
    {
        extension = 3;
    }
    extension
}

#[cfg(any(feature = "search-debug", test))]
fn reversible_shuffle(path_moves: &[Option<Move>], ply: usize, halfmove_clock: u8) -> bool {
    if ply < 4 || halfmove_clock < 4 {
        return false;
    }
    let (Some(latest), Some(previous), Some(latest_same_side), Some(previous_same_side)) = (
        path_moves.get(ply - 1).copied().flatten(),
        path_moves.get(ply - 2).copied().flatten(),
        path_moves.get(ply - 3).copied().flatten(),
        path_moves.get(ply - 4).copied().flatten(),
    ) else {
        return false;
    };
    move_from(latest) == move_to(latest_same_side)
        && move_to(latest) == move_from(latest_same_side)
        && move_from(previous) == move_to(previous_same_side)
        && move_to(previous) == move_from(previous_same_side)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProbCutCandidate {
    beta: i32,
    child_depth: i32,
    store_depth: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbCutEligibility {
    NoCandidate,
    SafetyRejected,
    TtRejected,
    Eligible(ProbCutCandidate),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbCutVerdict {
    QuiescenceRejected,
    FullSearchRejected,
    Cutoff,
}

#[allow(clippy::too_many_arguments)]
fn probcut_candidate(
    enabled: bool,
    verification_active: bool,
    ply: usize,
    is_pv: bool,
    in_check: bool,
    excluded_move: Option<Move>,
    actual_depth: i32,
    beta: i32,
    static_eval: i32,
    tt_score: Option<i32>,
    tt_depth: i32,
    tt_flag: Option<u8>,
) -> ProbCutEligibility {
    if !enabled || actual_depth < PROBCUT_MIN_DEPTH {
        return ProbCutEligibility::NoCandidate;
    }
    let probcut_beta = beta + PROBCUT_MARGIN_CP;
    if verification_active
        || ply == 0
        || is_pv
        || in_check
        || excluded_move.is_some()
        || static_eval < beta
        || beta.abs() >= MATE / 2
        || probcut_beta.abs() >= MATE / 2
    {
        return ProbCutEligibility::SafetyRejected;
    }
    let child_depth = actual_depth - PROBCUT_REDUCTION;
    let store_depth = child_depth + 1;
    if tt_depth >= store_depth
        && matches!(tt_flag, Some(TT_EXACT) | Some(TT_ALPHA))
        && tt_score.is_some_and(|score| score < probcut_beta)
    {
        return ProbCutEligibility::TtRejected;
    }
    ProbCutEligibility::Eligible(ProbCutCandidate {
        beta: probcut_beta,
        child_depth,
        store_depth,
    })
}

fn probcut_verdict(
    probcut_beta: i32,
    qsearch_score: i32,
    full_search_score: Option<i32>,
) -> ProbCutVerdict {
    if qsearch_score < probcut_beta {
        ProbCutVerdict::QuiescenceRejected
    } else if !full_search_score.is_some_and(|score| score >= probcut_beta) {
        ProbCutVerdict::FullSearchRejected
    } else {
        ProbCutVerdict::Cutoff
    }
}

fn singular_margin(evidence: SingularEvidence) -> i32 {
    SINGULAR_BASE_MARGIN_CP
        + SINGULAR_MARGIN_PER_DEPTH_CP * evidence.actual_depth
        + i32::from(!evidence.tt_pv) * 16
        + i32::from(!evidence.node_pv) * 8
        + i32::from(evidence.tt_age) * 8
}

fn singular_path_budget(depth: i32) -> u8 {
    (1 + depth.max(0) / 12).min(3) as u8
}

fn singular_candidate(evidence: SingularEvidence) -> SingularEligibility {
    if !evidence.enabled {
        return SingularEligibility::NoCandidate;
    }
    let (Some(mv), Some(score), Some(flag)) =
        (evidence.tt_move, evidence.tt_score, evidence.tt_flag)
    else {
        return SingularEligibility::NoCandidate;
    };
    let positive_extension = flag == TT_EXACT && evidence.tt_pv;
    let lower_bound_policy = evidence.allow_lower_bound && flag == TT_BETA;
    let reliable_bound = positive_extension || lower_bound_policy;
    if !reliable_bound
        || evidence.actual_depth < SINGULAR_MIN_DEPTH
        || evidence.tt_depth < evidence.actual_depth - SINGULAR_TT_DEPTH_MARGIN
        || evidence.tt_age > SINGULAR_MAX_TT_AGE
        || lower_bound_policy
            && (evidence.node_pv
                || evidence.actual_depth < SINGULAR_POLICY_MIN_DEPTH
                || evidence.node_beta.abs() >= MATE / 2
                || score < evidence.node_beta)
    {
        return SingularEligibility::NoCandidate;
    }
    if evidence.ply == 0
        || evidence.excluded_move.is_some()
        || evidence.in_check
        || score.abs() >= MATE / 2
        || evidence.halfmove_clock >= SINGULAR_MAX_HALF_MOVE_CLOCK
        || evidence.repetitions > 1
        || evidence.repeated_after_root
        || evidence.shuffling
        || evidence.path_extensions >= singular_path_budget(evidence.actual_depth)
        || !evidence.tt_move_is_legal
    {
        return SingularEligibility::SafetyRejected;
    }
    SingularEligibility::Eligible(SingularCandidate {
        mv,
        score,
        beta: if lower_bound_policy {
            evidence.node_beta
        } else {
            score - singular_margin(evidence)
        },
        depth: (evidence.actual_depth - 1) / 2,
        positive_extension,
        max_extension: if positive_extension {
            i32::from(
                singular_path_budget(evidence.actual_depth)
                    .saturating_sub(evidence.path_extensions),
            )
        } else {
            0
        },
    })
}

static SEARCH_BACKEND: OnceLock<SearchBackendKind> = OnceLock::new();
static SEARCH_BACKEND_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[inline]
pub fn active_search_backend() -> SearchBackendKind {
    if let Some(backend) = search_backend_from_id(SEARCH_BACKEND_OVERRIDE.load(Ordering::Relaxed)) {
        return backend;
    }
    *SEARCH_BACKEND.get_or_init(detect_search_backend)
}

pub fn set_search_backend_override(backend: Option<SearchBackendKind>) -> bool {
    if backend.is_some_and(|backend| !search_backend_available(backend)) {
        return false;
    }
    let id = backend.map(search_backend_id).unwrap_or(0);
    SEARCH_BACKEND_OVERRIDE.store(id, Ordering::SeqCst);
    true
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
        SearchBackendKind::Aarch64Simd128 => 3,
        SearchBackendKind::Aarch64Simd256 => 5,
        SearchBackendKind::Aarch64Simd512 => 6,
        SearchBackendKind::X86Avx512 => 4,
    }
}

fn search_backend_from_id(id: u8) -> Option<SearchBackendKind> {
    match id {
        1 => Some(SearchBackendKind::Scalar),
        2 => Some(SearchBackendKind::X86V3),
        3 => Some(SearchBackendKind::Aarch64Simd128),
        4 => Some(SearchBackendKind::X86Avx512),
        5 => Some(SearchBackendKind::Aarch64Simd256),
        6 => Some(SearchBackendKind::Aarch64Simd512),
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
pub const SEARCH_THREAD_STACK_SIZE: usize = 16 * 1024 * 1024;
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
    rep_root_len: usize,
    excluded_moves: [Option<Move>; MAX_PLY],
    #[cfg(feature = "search-debug")]
    path_moves: [Option<Move>; MAX_PLY],
    #[cfg(feature = "search-debug")]
    singular_extensions_used: [u8; MAX_PLY],
    #[cfg(any(feature = "search-debug", test))]
    restricted_verification: bool,
    probcut_verification: bool,
    pub tt_mb: usize,
    pub stopped: Arc<AtomicBool>,
    pub pondering: Arc<AtomicBool>,
    node_limit: Option<u64>,
    shared_node_counter: Option<Arc<AtomicU64>>,
    pub nnue_stack: Vec<NNUEAccumulator>,
    pub threat_stack: Vec<NNUEThreatAccumulator>,
    pub nnue_net: Option<Arc<NNUENet>>,
    pub search_backend: SearchBackendKind,
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

#[derive(Clone, Copy)]
struct ThreatNnueEval<'a, B: NnueBackend> {
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
    pub disable_qsearch_check_cap: bool,
    pub disable_qsearch_delta: bool,
    pub disable_qsearch_see: bool,
    pub disable_probcut: bool,
    pub disable_reverse_futility: bool,
    pub disable_see_pruning: bool,
    pub enable_singular_extensions: bool,
    pub enable_singular_multi_extensions: bool,
    pub enable_singular_multicut: bool,
    pub enable_singular_negative_extensions: bool,
    trace_roots: bool,
    trace_singular_candidates: bool,
    dag: SearchDagTrace,
    stats: SearchDebugStats,
}

#[cfg(feature = "search-debug")]
#[derive(Clone, Debug)]
struct SearchDagNode {
    fen: String,
    main_visits: u64,
    qsearch_visits: u64,
    min_ply: usize,
    max_ply: usize,
    min_depth: i32,
    max_depth: i32,
    min_alpha: i32,
    max_alpha: i32,
    min_beta: i32,
    max_beta: i32,
    eval_visits: u64,
    min_eval: i32,
    max_eval: i32,
    tt_visits: u64,
    min_tt_depth: i32,
    max_tt_depth: i32,
    min_tt_score: i32,
    max_tt_score: i32,
    tt_alpha: u64,
    tt_beta: u64,
    tt_exact: u64,
    q_delta_cutoffs: u64,
    min_q_delta_gap: i32,
    max_q_delta_gap: i32,
    search_cycle_returns: u64,
    claimable_draw_returns: u64,
    automatic_draw_returns: u64,
}

#[cfg(feature = "search-debug")]
impl SearchDagNode {
    fn new(st: &BoardState, ply: usize, depth: i32, alpha: i32, beta: i32, qsearch: bool) -> Self {
        Self {
            fen: crate::board::board_to_fen(st),
            main_visits: u64::from(!qsearch),
            qsearch_visits: u64::from(qsearch),
            min_ply: ply,
            max_ply: ply,
            min_depth: depth,
            max_depth: depth,
            min_alpha: alpha,
            max_alpha: alpha,
            min_beta: beta,
            max_beta: beta,
            eval_visits: 0,
            min_eval: i32::MAX,
            max_eval: i32::MIN,
            tt_visits: 0,
            min_tt_depth: i32::MAX,
            max_tt_depth: i32::MIN,
            min_tt_score: i32::MAX,
            max_tt_score: i32::MIN,
            tt_alpha: 0,
            tt_beta: 0,
            tt_exact: 0,
            q_delta_cutoffs: 0,
            min_q_delta_gap: i32::MAX,
            max_q_delta_gap: i32::MIN,
            search_cycle_returns: 0,
            claimable_draw_returns: 0,
            automatic_draw_returns: 0,
        }
    }

    fn record_visit(&mut self, ply: usize, depth: i32, alpha: i32, beta: i32, qsearch: bool) {
        if qsearch {
            self.qsearch_visits += 1;
        } else {
            self.main_visits += 1;
        }
        self.min_ply = self.min_ply.min(ply);
        self.max_ply = self.max_ply.max(ply);
        self.min_depth = self.min_depth.min(depth);
        self.max_depth = self.max_depth.max(depth);
        self.min_alpha = self.min_alpha.min(alpha);
        self.max_alpha = self.max_alpha.max(alpha);
        self.min_beta = self.min_beta.min(beta);
        self.max_beta = self.max_beta.max(beta);
    }

    fn record_eval(&mut self, eval: i32) {
        self.eval_visits += 1;
        self.min_eval = self.min_eval.min(eval);
        self.max_eval = self.max_eval.max(eval);
    }

    fn record_tt(&mut self, depth: i32, score: i32, flag: u8) {
        self.tt_visits += 1;
        self.min_tt_depth = self.min_tt_depth.min(depth);
        self.max_tt_depth = self.max_tt_depth.max(depth);
        self.min_tt_score = self.min_tt_score.min(score);
        self.max_tt_score = self.max_tt_score.max(score);
        match flag {
            TT_ALPHA => self.tt_alpha += 1,
            TT_BETA => self.tt_beta += 1,
            TT_EXACT => self.tt_exact += 1,
            _ => {}
        }
    }

    fn record_q_delta(&mut self, alpha: i32, stand: i32) {
        let gap = alpha - stand;
        self.q_delta_cutoffs += 1;
        self.min_q_delta_gap = self.min_q_delta_gap.min(gap);
        self.max_q_delta_gap = self.max_q_delta_gap.max(gap);
    }

    fn record_draw(&mut self, status: DrawStatus) {
        match status {
            DrawStatus::None => {}
            DrawStatus::SearchCycle => self.search_cycle_returns += 1,
            DrawStatus::Claimable => self.claimable_draw_returns += 1,
            DrawStatus::Automatic => self.automatic_draw_returns += 1,
        }
    }
}

#[cfg(feature = "search-debug")]
struct SearchDagTrace {
    output: Option<PathBuf>,
    root_depth: Option<i32>,
    root_moves: Vec<String>,
    max_ply: usize,
    max_positions: usize,
    active: bool,
    active_depth: i32,
    active_move: String,
    sequence: u64,
    truncated: bool,
    nodes: BTreeMap<u64, SearchDagNode>,
    edges: BTreeMap<(u64, u64), u64>,
}

#[cfg(feature = "search-debug")]
impl SearchDagTrace {
    fn from_env() -> Self {
        let output = std::env::var_os("EMBER_TRACE_SEARCH_DAG").map(PathBuf::from);
        let root_depth = env_usize("EMBER_TRACE_SEARCH_DAG_DEPTH").map(|depth| depth as i32);
        let root_moves = std::env::var("EMBER_TRACE_SEARCH_DAG_ROOTS")
            .ok()
            .map(|moves| {
                moves
                    .split(',')
                    .map(str::trim)
                    .filter(|mv| !mv.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        Self {
            output,
            root_depth,
            root_moves,
            max_ply: env_usize("EMBER_TRACE_SEARCH_DAG_MAX_PLY").unwrap_or(MAX_PLY),
            max_positions: env_usize("EMBER_TRACE_SEARCH_DAG_MAX_POSITIONS").unwrap_or(1_000_000),
            active: false,
            active_depth: 0,
            active_move: String::new(),
            sequence: 0,
            truncated: false,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
        }
    }

    fn begin_root(&mut self, depth: i32, mv: &str) {
        self.active = self.output.is_some()
            && self.root_depth.is_none_or(|wanted| wanted == depth)
            && (self.root_moves.is_empty() || self.root_moves.iter().any(|wanted| wanted == mv));
        self.active_depth = depth;
        self.active_move.clear();
        self.active_move.push_str(mv);
        self.truncated = false;
        self.nodes.clear();
        self.edges.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn record_node(
        &mut self,
        st: &BoardState,
        parent: Option<u64>,
        ply: usize,
        depth: i32,
        alpha: i32,
        beta: i32,
        qsearch: bool,
    ) {
        if !self.active || ply > self.max_ply {
            return;
        }
        let hash = st.hash;
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_visit(ply, depth, alpha, beta, qsearch);
        } else if self.nodes.len() < self.max_positions {
            self.nodes.insert(
                hash,
                SearchDagNode::new(st, ply, depth, alpha, beta, qsearch),
            );
        } else {
            self.truncated = true;
            return;
        }
        if let Some(parent) = parent.filter(|parent| *parent != hash) {
            *self.edges.entry((parent, hash)).or_default() += 1;
        }
    }

    fn record_eval(&mut self, hash: u64, eval: i32) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_eval(eval);
        }
    }

    fn record_tt(&mut self, hash: u64, depth: i32, score: i32, flag: u8) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_tt(depth, score, flag);
        }
    }

    fn record_q_delta(&mut self, hash: u64, alpha: i32, stand: i32) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_q_delta(alpha, stand);
        }
    }

    fn record_draw(&mut self, hash: u64, status: DrawStatus) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_draw(status);
        }
    }

    fn emit(&mut self, score: i32, searched_nodes: u64) {
        if !self.active {
            return;
        }
        let Some(output) = &self.output else {
            return;
        };
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(output) else {
            eprintln!(
                "info string search-debug could not open DAG trace {}",
                output.display()
            );
            self.active = false;
            return;
        };
        self.sequence += 1;
        let sequence = self.sequence;
        let _ = writeln!(
            file,
            "{{\"type\":\"root\",\"sequence\":{sequence},\"depth\":{},\"move\":\"{}\",\
             \"score\":{score},\"searched_nodes\":{searched_nodes},\"positions\":{},\
             \"edges\":{},\"truncated\":{}}}",
            self.active_depth,
            self.active_move,
            self.nodes.len(),
            self.edges.len(),
            self.truncated,
        );
        for (&hash, node) in &self.nodes {
            let (min_eval, max_eval) = if node.eval_visits == 0 {
                (0, 0)
            } else {
                (node.min_eval, node.max_eval)
            };
            let (min_tt_depth, max_tt_depth, min_tt_score, max_tt_score) = if node.tt_visits == 0 {
                (0, 0, 0, 0)
            } else {
                (
                    node.min_tt_depth,
                    node.max_tt_depth,
                    node.min_tt_score,
                    node.max_tt_score,
                )
            };
            let (min_q_delta_gap, max_q_delta_gap) = if node.q_delta_cutoffs == 0 {
                (0, 0)
            } else {
                (node.min_q_delta_gap, node.max_q_delta_gap)
            };
            let _ = writeln!(
                file,
                "{{\"type\":\"node\",\"sequence\":{sequence},\"hash\":\"{hash:016x}\",\
                 \"fen\":\"{}\",\"main_visits\":{},\"qsearch_visits\":{},\
                 \"min_ply\":{},\"max_ply\":{},\"min_depth\":{},\"max_depth\":{},\
                 \"min_alpha\":{},\"max_alpha\":{},\"min_beta\":{},\"max_beta\":{},\
                 \"eval_visits\":{},\"min_eval\":{min_eval},\"max_eval\":{max_eval},\
                 \"tt_visits\":{},\"min_tt_depth\":{min_tt_depth},\
                 \"max_tt_depth\":{max_tt_depth},\"min_tt_score\":{min_tt_score},\
                 \"max_tt_score\":{max_tt_score},\"tt_alpha\":{},\"tt_beta\":{},\
                 \"tt_exact\":{},\"q_delta_cutoffs\":{},\
                 \"min_q_delta_gap\":{min_q_delta_gap},\
                 \"max_q_delta_gap\":{max_q_delta_gap},\
                 \"search_cycle_returns\":{},\"claimable_draw_returns\":{},\
                 \"automatic_draw_returns\":{}}}",
                node.fen,
                node.main_visits,
                node.qsearch_visits,
                node.min_ply,
                node.max_ply,
                node.min_depth,
                node.max_depth,
                node.min_alpha,
                node.max_alpha,
                node.min_beta,
                node.max_beta,
                node.eval_visits,
                node.tt_visits,
                node.tt_alpha,
                node.tt_beta,
                node.tt_exact,
                node.q_delta_cutoffs,
                node.search_cycle_returns,
                node.claimable_draw_returns,
                node.automatic_draw_returns,
            );
        }
        for (&(parent, child), &visits) in &self.edges {
            let _ = writeln!(
                file,
                "{{\"type\":\"edge\",\"sequence\":{sequence},\
                 \"parent\":\"{parent:016x}\",\"child\":\"{child:016x}\",\
                 \"visits\":{visits}}}"
            );
        }
        self.active = false;
    }
}

#[cfg(feature = "search-debug")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchDebugStats {
    pub max_ply: usize,
    pub tt_hits: u64,
    pub tt_max_depth: i32,
    pub tt_cutoffs: u64,
    pub reverse_futility_cutoffs: u64,
    pub futility_cutoffs: u64,
    pub null_attempts: u64,
    pub null_cutoffs: u64,
    pub iid_reductions: u64,
    pub lmp_cutoffs: u64,
    pub history_skips: u64,
    pub see_skips: u64,
    pub lmr_searches: u64,
    pub lmr_researches: u64,
    pub lmr_reduction_sum: u64,
    pub lmr_max_reduction: i32,
    pub qnodes: u64,
    pub q_delta_cutoffs: u64,
    pub q_see_skips: u64,
    pub q_checked_depth_exits: u64,
    pub probcut_eligible_nodes: u64,
    pub probcut_safety_rejections: u64,
    pub probcut_tt_rejections: u64,
    pub probcut_candidates: u64,
    pub probcut_see_rejections: u64,
    pub probcut_qsearch_passes: u64,
    pub probcut_verifications: u64,
    pub probcut_verification_nodes: u64,
    pub probcut_cutoffs: u64,
    pub probcut_stop_rejections: u64,
    pub singular_candidates: u64,
    pub singular_safety_rejections: u64,
    pub singular_verifications: u64,
    pub singular_verification_nodes: u64,
    pub singular_extensions: u64,
    pub singular_extension_plies: u64,
    pub singular_negative_extensions: u64,
    pub singular_multicut_cutoffs: u64,
    pub singular_alternative_rejections: u64,
    pub singular_stop_rejections: u64,
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
        #[cfg(feature = "search-debug")]
        {
            $this.debug.stats.qnodes += 1;
            $this.debug.stats.max_ply = $this.debug.stats.max_ply.max($ply);
            $this.record_debug_dag_node($st, $ply, $depth, $alpha, $beta, true);
        }
        if $this.search_limit_reached::<NODE_LIMITED>($start, $tl, *$cnt) {
            return 0;
        }
        let excluded_move = $this.excluded_moves.get($ply).copied().flatten();
        let ks = $st.king_sq($st.w);
        let in_check = crate::board::is_attacked(&$st.bb, ks, !$st.w);

        if let Some(score) = $this.draw_score($st, $ply, 2, in_check) {
            return score;
        }

        if !in_check && excluded_move.is_none() {
            if let Some(score) = $this.syzygy.probe_search_score($st, $ply) {
                return score;
            }
        }

        if !in_check {
            let stand = $eval.static_eval::<CHESS960>($this, $st, $ply);
            #[cfg(feature = "search-debug")]
            $this.record_debug_dag_eval($st.hash, stand);
            if stand >= $beta {
                return stand;
            }
            if stand > $alpha {
                $alpha = stand;
            }
            if $this.qsearch_delta_enabled()
                && excluded_move.is_none()
                && $depth <= 0
                && $alpha - 975 > stand
            {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.q_delta_cutoffs += 1;
                    $this.debug.dag.record_q_delta($st.hash, $alpha, stand);
                }
                return $alpha;
            }
        } else if $this.qsearch_check_cap_enabled() && excluded_move.is_none() && $depth <= -4 {
            #[cfg(feature = "search-debug")]
            {
                $this.debug.stats.q_checked_depth_exits += 1;
            }
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
            return if excluded_move.is_some() {
                $alpha
            } else if in_check {
                -MATE + $ply as i32
            } else {
                $alpha
            };
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
            if Some(mv) == excluded_move {
                continue;
            }
            let from = move_from(mv);
            let to = move_to(mv);
            let fpi = $st.mailbox[from];
            let tpi = $st.mailbox[to];
            if !in_check
                && excluded_move.is_none()
                && $this.qsearch_see_enabled()
                && move_see::<CHESS960>($st, mv, from, to, fpi, tpi) < 0
            {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.q_see_skips += 1;
                }
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
            $this.rep_stack.push($st.hash);
            $this.rep_stack_len += 1;
            let score = -$this.$qsearch_mode::<CHESS960, NODE_LIMITED, E>(
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
            $this.rep_stack.pop();
            $this.rep_stack_len -= 1;
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
        #[cfg(feature = "search-debug")]
        {
            $this.debug.stats.max_ply = $this.debug.stats.max_ply.max($ply);
            $this.record_debug_dag_node($st, $ply, $depth, $alpha, $beta, false);
        }
        if $this.search_limit_reached::<NODE_LIMITED>($start, $tl, *$cnt) {
            return 0;
        }
        if $ply >= MAX_PLY {
            return $eval.static_eval::<CHESS960>($this, $st, $ply);
        }
        let excluded_move = $this.excluded_moves[$ply];

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

        let ks = $st.king_sq($st.w);
        let in_check = crate::board::is_attacked(&$st.bb, ks, !$st.w);
        if let Some(score) = $this.draw_score($st, $ply, 1, in_check) {
            return score;
        }

        if $ply > 0 && !in_check && $can_null && excluded_move.is_none() {
            if let Some(score) = $this.syzygy.probe_search_score($st, $ply) {
                return score;
            }
        }

        let tt_data = $this.shared_tt.get_entry(h);
        let tt_move = if excluded_move.is_none() {
            tt_data.and_then(|entry| entry.best_move)
        } else {
            None
        };
        let tt_score = tt_data.map(|entry| score_from_tt(entry.score, $ply));
        let tt_depth = tt_data.map(|entry| entry.depth).unwrap_or(-1);
        let tt_flag = tt_data.map(|entry| entry.flag);
        let tt_pv = tt_data.is_some_and(|entry| entry.pv);
        let tt_age = tt_data
            .map(|entry| $this.shared_tt.age(entry))
            .unwrap_or(u8::MAX);

        #[cfg(feature = "search-debug")]
        if let (Some(score), Some(flag)) = (tt_score, tt_flag) {
            $this.debug.stats.tt_hits += 1;
            $this.debug.stats.tt_max_depth = $this.debug.stats.tt_max_depth.max(tt_depth);
            $this.record_debug_dag_tt(h, tt_depth, score, flag);
        }

        let is_pv = beta - $alpha > 1;
        let ext = if in_check && $depth < 16 { 1 } else { 0 };
        let actual_depth = $depth + ext;

        if excluded_move.is_none() && !is_pv && tt_depth >= actual_depth {
            if let (Some(flag), Some(s)) = (tt_flag, tt_score) {
                match flag {
                    TT_EXACT => {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.tt_cutoffs += 1;
                        }
                        return s;
                    }
                    TT_ALPHA if s <= $alpha => {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.tt_cutoffs += 1;
                        }
                        return $alpha;
                    }
                    TT_BETA if s >= beta => {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.tt_cutoffs += 1;
                        }
                        return beta;
                    }
                    _ => {}
                }
            }
        }

        let king_pressure = if in_check {
            8
        } else {
            tactical_king_pressure($st)
        };

        let eval_score = $eval.static_eval::<CHESS960>($this, $st, $ply);
        #[cfg(feature = "search-debug")]
        $this.record_debug_dag_eval(h, eval_score);

        if actual_depth <= 0 {
            return $this.$qsearch_mode::<CHESS960, NODE_LIMITED, E>(
                $st, $alpha, beta, QS_DEPTH, $start, $tl, $cnt, $ply, $eval,
            );
        }

        if $this.reverse_futility_enabled() && !in_check && !is_pv && actual_depth <= 8 && $ply > 0
        {
            let margin = 80 + 65 * actual_depth;
            if eval_score - margin >= beta {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.reverse_futility_cutoffs += 1;
                }
                return eval_score - margin;
            }
        }
        if $this.futility_enabled()
            && excluded_move.is_none()
            && !in_check
            && !is_pv
            && actual_depth <= 3
            && $ply > 0
        {
            let margin = 150 * actual_depth;
            if eval_score + margin <= $alpha {
                let q = $this.$qsearch_mode::<CHESS960, NODE_LIMITED, E>(
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
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.futility_cutoffs += 1;
                    }
                    return $alpha;
                }
            }
        }
        if $this.null_move_enabled()
            && excluded_move.is_none()
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
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.null_attempts += 1;
                }
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
                let path_state = $this.enter_null_path($ply);
                let s = -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                $this.leave_child_path(path_state);
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
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.null_cutoffs += 1;
                    }
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
            return if excluded_move.is_some() {
                $alpha
            } else if in_check {
                -MATE + $ply as i32
            } else {
                0
            };
        }

        let actual_depth = if $this.iid_reduction_enabled()
            && excluded_move.is_none()
            && tt_move.is_none()
            && actual_depth >= 4
            && is_pv
        {
            #[cfg(feature = "search-debug")]
            {
                $this.debug.stats.iid_reductions += 1;
            }
            actual_depth - 1
        } else {
            actual_depth
        };

        match probcut_candidate(
            $this.probcut_enabled(),
            $this.probcut_verification,
            $ply,
            is_pv,
            in_check,
            excluded_move,
            actual_depth,
            beta,
            eval_score,
            tt_score,
            tt_depth,
            tt_flag,
        ) {
            ProbCutEligibility::NoCandidate => {}
            ProbCutEligibility::SafetyRejected => {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.probcut_safety_rejections += 1;
                }
            }
            ProbCutEligibility::TtRejected => {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.probcut_tt_rejections += 1;
                }
            }
            ProbCutEligibility::Eligible(candidate) => {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.probcut_eligible_nodes += 1;
                }
                let mut caps = Self::take_buf(&mut $this.caps_bufs, $ply);
                generate_pseudo_captures_promotions_into_mode::<CHESS960>(
                    $st, $st.w, &$st.cr, $st.ep, &mut caps,
                );
                caps.sort_by_key(|mv| {
                    let from = move_from(*mv);
                    let to = move_to(*mv);
                    let fpi = $st.mailbox[from];
                    let tpi = $st.mailbox[to];
                    let victim = capture_victim_value::<CHESS960>($st, fpi, *mv, to, tpi);
                    let attacker = if fpi != EMPTY_SQ {
                        piece_val(piece_type(fpi))
                    } else {
                        0
                    };
                    -(victim * 10 - attacker + promotion_value(*mv))
                });
                $eval.ensure_child_stack($this, $ply);

                let mut cap_idx = 0usize;
                while cap_idx < caps.len() {
                    let mv = caps[cap_idx];
                    cap_idx += 1;
                    if $this.time_up($start, $tl) {
                        Self::return_buf(&mut $this.caps_bufs, $ply, caps);
                        Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
                        return 0;
                    }
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.probcut_candidates += 1;
                    }
                    let from = move_from(mv);
                    let to = move_to(mv);
                    let fpi = $st.mailbox[from];
                    let tpi = $st.mailbox[to];
                    let is_capture = move_is_capture::<CHESS960>($st, fpi, mv, to, tpi);
                    let is_queen_promotion = move_promotion(mv).eq_ignore_ascii_case(&b'Q');
                    if !is_queen_promotion
                        && (!is_capture || move_see::<CHESS960>($st, mv, from, to, fpi, tpi) < 0)
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.probcut_see_rejections += 1;
                        }
                        continue;
                    }

                    let st_before = *$st;
                    if !try_apply_move_mode::<CHESS960>($st, mv) {
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
                    $this.rep_stack.push($st.hash);
                    $this.rep_stack_len += 1;
                    let path_state = $this.enter_child_path($ply, mv, 0);

                    let qsearch_score = -$this.$qsearch_mode::<CHESS960, NODE_LIMITED, E>(
                        $st,
                        -candidate.beta,
                        -candidate.beta + 1,
                        QS_DEPTH,
                        $start,
                        $tl,
                        $cnt,
                        $ply + 1,
                        $eval,
                    );
                    let mut full_search_score = None;
                    if qsearch_score >= candidate.beta && !$this.stopped.load(Ordering::Relaxed) {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.probcut_qsearch_passes += 1;
                            $this.debug.stats.probcut_verifications += 1;
                        }
                        #[cfg(feature = "search-debug")]
                        let nodes_before = *$cnt;
                        let previous_verification = $this.probcut_verification;
                        $this.probcut_verification = true;
                        full_search_score =
                            Some(-$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
                                $st,
                                candidate.child_depth,
                                $ply + 1,
                                -candidate.beta,
                                -candidate.beta + 1,
                                false,
                                $start,
                                $tl,
                                $cnt,
                                $eval,
                            ));
                        $this.probcut_verification = previous_verification;
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.probcut_verification_nodes +=
                                (*$cnt).saturating_sub(nodes_before);
                        }
                    }

                    $this.leave_child_path(path_state);
                    $this.rep_stack.pop();
                    $this.rep_stack_len -= 1;
                    *$st = st_before;

                    if $this.stopped.load(Ordering::Relaxed) {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.probcut_stop_rejections += 1;
                        }
                        Self::return_buf(&mut $this.caps_bufs, $ply, caps);
                        Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
                        return 0;
                    }
                    if probcut_verdict(candidate.beta, qsearch_score, full_search_score)
                        == ProbCutVerdict::Cutoff
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.probcut_cutoffs += 1;
                        }
                        if !$this.restricted_verification_active() {
                            $this.shared_tt.store(
                                h,
                                candidate.store_depth,
                                score_to_tt(candidate.beta, $ply),
                                TT_BETA,
                                Some(mv),
                            );
                        }
                        Self::return_buf(&mut $this.caps_bufs, $ply, caps);
                        Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
                        return beta;
                    }
                }
                Self::return_buf(&mut $this.caps_bufs, $ply, caps);
            }
        }

        let singular_enabled =
            $this.singular_extensions_enabled() && !$this.restricted_verification_active();
        let singular_policy_enabled =
            $this.singular_multicut_enabled() || $this.singular_negative_extensions_enabled();
        let inspect_singular_safety = singular_enabled
            && actual_depth >= SINGULAR_MIN_DEPTH
            && tt_depth >= actual_depth - SINGULAR_TT_DEPTH_MARGIN
            && (tt_flag == Some(TT_EXACT) && tt_pv
                || singular_policy_enabled && tt_flag == Some(TT_BETA))
            && tt_age <= SINGULAR_MAX_TT_AGE
            && tt_move.is_some()
            && tt_score.is_some();
        let tt_move_is_legal = inspect_singular_safety
            && tt_move.is_some_and(|mv| {
                if !moves_buf.contains(&mv) {
                    return false;
                }
                let mut probe = *$st;
                try_apply_move_mode::<CHESS960>(&mut probe, mv)
            });
        let (repetitions, repeated_after_root) = if inspect_singular_safety {
            $this.repetition_info(usize::from($st.halfmove_clock))
        } else {
            (0, false)
        };
        let singular_evidence = SingularEvidence {
            enabled: singular_enabled,
            ply: $ply,
            excluded_move,
            in_check,
            node_pv: is_pv,
            node_beta: beta,
            actual_depth,
            halfmove_clock: $st.halfmove_clock,
            repetitions,
            repeated_after_root,
            shuffling: $this.singular_shuffling($ply, $st.halfmove_clock),
            path_extensions: $this.singular_path_extensions($ply),
            allow_lower_bound: singular_policy_enabled,
            tt_move,
            tt_score,
            tt_depth,
            tt_flag,
            tt_pv,
            tt_age,
            tt_move_is_legal,
        };
        let singular_adjustment = match singular_candidate(singular_evidence) {
            SingularEligibility::NoCandidate => None,
            SingularEligibility::SafetyRejected => {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.singular_safety_rejections += 1;
                }
                None
            }
            SingularEligibility::Eligible(candidate) => {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.singular_candidates += 1;
                    $this.debug.stats.singular_verifications += 1;
                }
                #[cfg(feature = "search-debug")]
                let nodes_before = *$cnt;
                let previous = $this.excluded_moves[$ply].replace(candidate.mv);
                let previous_restricted = $this.set_restricted_verification(true);
                let alternative_score = $this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
                    $st,
                    candidate.depth,
                    $ply,
                    candidate.beta - 1,
                    candidate.beta,
                    false,
                    $start,
                    $tl,
                    $cnt,
                    $eval,
                );
                let mut double_alternative_score = None;
                let mut triple_alternative_score = None;
                if !$this.stopped.load(Ordering::Relaxed)
                    && $this.singular_multi_extensions_enabled()
                    && candidate.positive_extension
                    && candidate.max_extension >= 2
                    && actual_depth >= SINGULAR_DOUBLE_MIN_DEPTH
                    && alternative_score < candidate.beta
                {
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.singular_verifications += 1;
                    }
                    let threshold = candidate.score - SINGULAR_DOUBLE_MARGIN_CP;
                    double_alternative_score =
                        Some($this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
                            $st,
                            candidate.depth,
                            $ply,
                            threshold - 1,
                            threshold,
                            false,
                            $start,
                            $tl,
                            $cnt,
                            $eval,
                        ));
                    if !$this.stopped.load(Ordering::Relaxed)
                        && candidate.max_extension >= 3
                        && actual_depth >= SINGULAR_TRIPLE_MIN_DEPTH
                        && double_alternative_score.is_some_and(|score| score < threshold)
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.singular_verifications += 1;
                        }
                        let threshold = candidate.score - SINGULAR_TRIPLE_MARGIN_CP;
                        triple_alternative_score =
                            Some($this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
                                $st,
                                candidate.depth,
                                $ply,
                                threshold - 1,
                                threshold,
                                false,
                                $start,
                                $tl,
                                $cnt,
                                $eval,
                            ));
                    }
                }
                $this.excluded_moves[$ply] = previous;
                $this.set_restricted_verification(previous_restricted);
                #[cfg(feature = "search-debug")]
                let verification_nodes = (*$cnt).saturating_sub(nodes_before);
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.singular_verification_nodes += verification_nodes;
                }
                if $this.stopped.load(Ordering::Relaxed) {
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.singular_stop_rejections += 1;
                        $this.emit_debug_singular_candidate(
                            $st,
                            candidate.mv,
                            $ply,
                            is_pv,
                            actual_depth,
                            $alpha,
                            beta,
                            eval_score,
                            tt_score.expect("eligible singular candidate has a TT score"),
                            tt_depth,
                            tt_flag.expect("eligible singular candidate has a TT flag"),
                            tt_pv,
                            tt_age,
                            candidate.beta,
                            candidate.depth,
                            alternative_score,
                            verification_nodes,
                            repetitions,
                            repeated_after_root,
                            0,
                            "stopped",
                        );
                    }
                    Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
                    return 0;
                }
                let lower_bound_policy = !candidate.positive_extension;
                let multi_cut_beta = ($this.singular_multicut_enabled() && lower_bound_policy)
                    .then_some(candidate.beta);
                let negative_extension = if $this.singular_negative_extensions_enabled()
                    && lower_bound_policy
                    && alternative_score >= candidate.beta
                {
                    -1
                } else {
                    0
                };
                match singular_search_outcome(
                    alternative_score,
                    candidate.beta,
                    candidate.positive_extension,
                    multi_cut_beta,
                    negative_extension,
                ) {
                    SingularSearchOutcome::Continue(mut extension) => {
                        if extension > 0 && $this.singular_multi_extensions_enabled() {
                            extension = singular_extension_from_scores(
                                candidate,
                                alternative_score,
                                double_alternative_score,
                                triple_alternative_score,
                            );
                        }
                        #[cfg(feature = "search-debug")]
                        {
                            let outcome = if extension > 0 {
                                $this.debug.stats.singular_extensions += 1;
                                $this.debug.stats.singular_extension_plies += extension as u64;
                                "extended"
                            } else if extension < 0 {
                                $this.debug.stats.singular_negative_extensions += 1;
                                "reduced"
                            } else {
                                $this.debug.stats.singular_alternative_rejections += 1;
                                "rejected"
                            };
                            $this.emit_debug_singular_candidate(
                                $st,
                                candidate.mv,
                                $ply,
                                is_pv,
                                actual_depth,
                                $alpha,
                                beta,
                                eval_score,
                                tt_score.expect("eligible singular candidate has a TT score"),
                                tt_depth,
                                tt_flag.expect("eligible singular candidate has a TT flag"),
                                tt_pv,
                                tt_age,
                                candidate.beta,
                                candidate.depth,
                                alternative_score,
                                verification_nodes,
                                repetitions,
                                repeated_after_root,
                                extension,
                                outcome,
                            );
                        }
                        (extension != 0).then_some(SingularMoveAdjustment {
                            mv: candidate.mv,
                            extension,
                        })
                    }
                    SingularSearchOutcome::Cutoff(score) => {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.singular_multicut_cutoffs += 1;
                            $this.emit_debug_singular_candidate(
                                $st,
                                candidate.mv,
                                $ply,
                                is_pv,
                                actual_depth,
                                $alpha,
                                beta,
                                eval_score,
                                tt_score.expect("eligible singular candidate has a TT score"),
                                tt_depth,
                                tt_flag.expect("eligible singular candidate has a TT flag"),
                                tt_pv,
                                tt_age,
                                candidate.beta,
                                candidate.depth,
                                alternative_score,
                                verification_nodes,
                                repetitions,
                                repeated_after_root,
                                0,
                                "multi_cut",
                            );
                        }
                        Self::return_buf(&mut $this.move_bufs, $ply, moves_buf);
                        return score;
                    }
                }
            }
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

        let lmp_count = if $this.lmp_enabled()
            && excluded_move.is_none()
            && king_pressure < 3
            && !is_pv
            && !in_check
            && actual_depth <= 8
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
            if Some(mv) == excluded_move {
                continue;
            }

            let from = move_from(mv);
            let to = move_to(mv);
            let fpi = $st.mailbox[from];
            let tpi = $st.mailbox[to];
            let capture = move_is_capture::<CHESS960>($st, fpi, mv, to, tpi);
            let is_promo = is_promotion_move(fpi, mv);
            let is_quiet = !capture && !is_promo;

            if excluded_move.is_none()
                && !is_pv
                && !in_check
                && is_quiet
                && legal_moves_seen >= lmp_count
            {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.lmp_cutoffs += 1;
                }
                break;
            }
            if excluded_move.is_none()
                && !is_pv
                && !in_check
                && legal_moves_seen > 0
                && best_score > -MATE / 2
            {
                if capture {
                    if $this.see_pruning_enabled()
                        && move_see::<CHESS960>($st, mv, from, to, fpi, tpi) < -80 * actual_depth
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.see_skips += 1;
                        }
                        continue;
                    }
                } else if is_quiet && $this.history_pruning_enabled() {
                    let (fk, tk) = from_to_key(move_sr(mv), move_sc(mv), move_er(mv), move_ec(mv));
                    if actual_depth <= 5 && $this.history[fk][tk] < -1024 * actual_depth {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.history_skips += 1;
                        }
                        continue;
                    }
                }
            }

            let tactical_move_ext = if !in_check
                && legal_moves_seen == 0
                && !is_quiet
                && actual_depth <= 2
                && special_move_gives_check_mode::<CHESS960>($st, mv)
            {
                1
            } else {
                0
            };
            let singular_extension = singular_adjustment
                .filter(|adjustment| adjustment.mv == mv)
                .map(|adjustment| adjustment.extension)
                .unwrap_or(0);
            let move_ext = combine_move_extensions(tactical_move_ext, singular_extension);

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
            let path_state = $this.enter_child_path($ply, mv, singular_extension);

            let new_depth = actual_depth - 1 + move_ext;

            let lmr_eligible = $this.lmr_enabled()
                && excluded_move.is_none()
                && move_index >= 2
                && actual_depth >= 3
                && is_quiet
                && !in_check;
            let s = if move_index == 0 {
                -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.lmr_searches += 1;
                    $this.debug.stats.lmr_reduction_sum += r as u64;
                    $this.debug.stats.lmr_max_reduction =
                        $this.debug.stats.lmr_max_reduction.max(r);
                }
                let s2 = -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                    #[cfg(feature = "search-debug")]
                    {
                        $this.debug.stats.lmr_researches += 1;
                    }
                    let s3 = -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                        -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                let s2 = -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                    -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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
                -$this.$negamax_mode::<CHESS960, NODE_LIMITED, E>(
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

            $this.leave_child_path(path_state);
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
                        if is_quiet
                            && excluded_move.is_none()
                            && !$this.restricted_verification_active()
                        {
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
            return if excluded_move.is_some() {
                $alpha
            } else if in_check {
                -MATE + $ply as i32
            } else {
                0
            };
        }

        let flag = if best_score <= orig_alpha {
            TT_ALPHA
        } else if best_score >= beta {
            TT_BETA
        } else {
            TT_EXACT
        };
        if excluded_move.is_none() && !$this.restricted_verification_active() {
            $this.shared_tt.store_with_pv(
                h,
                actual_depth,
                score_to_tt(best_score, $ply),
                flag,
                best_move,
                is_pv,
            );
        }
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
            rep_root_len: 0,
            excluded_moves: [None; MAX_PLY],
            #[cfg(feature = "search-debug")]
            path_moves: [None; MAX_PLY],
            #[cfg(feature = "search-debug")]
            singular_extensions_used: [0; MAX_PLY],
            #[cfg(any(feature = "search-debug", test))]
            restricted_verification: false,
            probcut_verification: false,
            tt_mb: 128,
            stopped,
            pondering: Arc::new(AtomicBool::new(false)),
            node_limit: None,
            shared_node_counter: None,
            nnue_stack: Vec::new(),
            threat_stack: Vec::new(),
            nnue_net: current_nnue_net(),
            search_backend: active_search_backend(),
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

    pub fn refresh_search_backend(&mut self) {
        self.search_backend = active_search_backend();
    }

    pub fn init_nnue_stack(&mut self, st: &BoardState) {
        if let Some(net) = self.nnue_net.as_deref() {
            if self.nnue_stack.len() < MAX_PLY + 1 {
                self.nnue_stack
                    .resize(MAX_PLY + 1, NNUEAccumulator::new(net.hidden_size));
            }
            if net.has_threat_features() {
                self.nnue_stack[0].refresh_with_backend::<ScalarNnueBackend>(net, st);
                if self.threat_stack.len() < MAX_PLY + 1 {
                    self.threat_stack
                        .resize(MAX_PLY + 1, NNUEThreatAccumulator::new(net.hidden_size));
                }
                self.threat_stack[0].refresh(net, st);
            } else {
                self.nnue_stack[0].refresh(net, st);
            }
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
        if net.has_threat_features() {
            self.nnue_stack[ply].refresh_with_backend::<ScalarNnueBackend>(net, st);
            if self.threat_stack.len() <= ply {
                self.threat_stack
                    .resize(ply + 1, NNUEThreatAccumulator::new(net.hidden_size));
            }
            self.threat_stack[ply].refresh(net, st);
        } else {
            self.nnue_stack[ply].refresh(net, st);
        }
    }

    #[inline]
    fn time_up(&self, start: Instant, tl: f64) -> bool {
        if self.stopped.load(Ordering::Relaxed) {
            return true;
        }
        if self.pondering.load(Ordering::Relaxed) {
            return false;
        }
        if start.elapsed().as_secs_f64() > tl {
            self.set_stopped();
            true
        } else {
            false
        }
    }

    #[inline]
    fn search_limit_reached<const NODE_LIMITED: bool>(
        &self,
        start: Instant,
        tl: f64,
        local_nodes: u64,
    ) -> bool {
        if self.stopped.load(Ordering::Relaxed) {
            return true;
        }
        if NODE_LIMITED {
            let limit = self
                .node_limit
                .expect("node-limited search was started without a node limit");
            let searched_nodes = if let Some(counter) = &self.shared_node_counter {
                counter.fetch_add(1, Ordering::Relaxed).saturating_add(1)
            } else {
                local_nodes
            };
            if searched_nodes >= limit {
                self.set_stopped();
                return true;
            }
        }
        if self.pondering.load(Ordering::Relaxed) {
            return false;
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

    pub fn set_node_limit(&mut self, node_limit: Option<u64>) {
        self.node_limit = node_limit;
        self.shared_node_counter = None;
    }

    fn set_shared_node_limit(&mut self, node_limit: Option<u64>, counter: Option<Arc<AtomicU64>>) {
        self.node_limit = node_limit;
        self.shared_node_counter = counter;
    }

    pub fn clear_node_limit(&mut self) {
        self.node_limit = None;
        self.shared_node_counter = None;
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
        dst.rep_root_len = dst.rep_stack_len;
        dst.import_learning(&self.export_learning());
        dst.nnue_net = self.nnue_net.clone();
        dst.search_backend = self.search_backend;
        dst.syzygy = self.syzygy.clone();
        dst.pondering = Arc::clone(&self.pondering);
    }

    pub fn export_learning(&self) -> SearchLearning {
        SearchLearning {
            history: self.history,
            counter_move: self.counter_move,
            corr_hist: self.corr_hist,
        }
    }

    pub fn import_learning(&mut self, learning: &SearchLearning) {
        self.history = learning.history;
        self.counter_move = learning.counter_move;
        self.corr_hist = learning.corr_hist;
    }

    pub fn prepare_for_search(&mut self) {
        self.rep_root_len = self.rep_stack_len;
        self.excluded_moves.fill(None);
        #[cfg(feature = "search-debug")]
        {
            self.path_moves.fill(None);
            self.singular_extensions_used.fill(0);
        }
        #[cfg(any(feature = "search-debug", test))]
        {
            self.restricted_verification = false;
        }
        self.probcut_verification = false;
        self.killers = [[None; 2]; MAX_PLY];
        for row in &mut self.history {
            for value in row {
                *value = *value * 13 / 16;
            }
        }
    }

    #[cfg(feature = "search-debug")]
    fn enter_path(
        &mut self,
        ply: usize,
        mv: Option<Move>,
        singular_extension: i32,
    ) -> ChildPathState {
        let previous_move = std::mem::replace(&mut self.path_moves[ply], mv);
        let child_ply = (ply + 1 < MAX_PLY).then_some(ply + 1);
        let previous_extensions = child_ply
            .map(|child| {
                let previous = self.singular_extensions_used[child];
                self.singular_extensions_used[child] = next_singular_extension_count(
                    self.singular_extensions_used[ply],
                    singular_extension,
                );
                previous
            })
            .unwrap_or(0);
        ChildPathState {
            ply,
            previous_move,
            child_ply,
            previous_extensions,
        }
    }

    #[cfg(feature = "search-debug")]
    fn enter_child_path(
        &mut self,
        ply: usize,
        mv: Move,
        singular_extension: i32,
    ) -> ChildPathState {
        self.enter_path(ply, Some(mv), singular_extension)
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn enter_child_path(
        &mut self,
        _ply: usize,
        _mv: Move,
        _singular_extension: i32,
    ) -> ChildPathState {
        ChildPathState
    }

    #[cfg(feature = "search-debug")]
    fn enter_null_path(&mut self, ply: usize) -> ChildPathState {
        self.enter_path(ply, None, 0)
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn enter_null_path(&mut self, _ply: usize) -> ChildPathState {
        ChildPathState
    }

    #[cfg(feature = "search-debug")]
    fn leave_child_path(&mut self, state: ChildPathState) {
        self.path_moves[state.ply] = state.previous_move;
        if let Some(child) = state.child_ply {
            self.singular_extensions_used[child] = state.previous_extensions;
        }
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn leave_child_path(&mut self, _state: ChildPathState) {}

    #[cfg(feature = "search-debug")]
    pub(crate) fn enter_root_path(&mut self, mv: Move) {
        self.path_moves[0] = Some(mv);
        self.singular_extensions_used[0] = 0;
        self.singular_extensions_used[1] = 0;
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(crate) fn enter_root_path(&mut self, _mv: Move) {}

    #[cfg(feature = "search-debug")]
    pub(crate) fn leave_root_path(&mut self) {
        self.path_moves[0] = None;
        self.singular_extensions_used[1] = 0;
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(crate) fn leave_root_path(&mut self) {}

    #[cfg(feature = "search-debug")]
    fn singular_shuffling(&self, ply: usize, halfmove_clock: u8) -> bool {
        reversible_shuffle(&self.path_moves, ply, halfmove_clock)
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_shuffling(&self, _ply: usize, _halfmove_clock: u8) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    fn singular_path_extensions(&self, ply: usize) -> u8 {
        self.singular_extensions_used[ply]
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_path_extensions(&self, _ply: usize) -> u8 {
        0
    }

    pub fn clear_learning(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[0; 64]; 64];
        self.counter_move = [[None; 64]; 13];
        self.corr_hist = [0; CORR_HIST_SIZE * 2];
    }

    #[cfg(feature = "search-debug")]
    pub fn reset_debug_stats(&mut self) {
        self.debug.reset_stats();
    }

    #[cfg(feature = "search-debug")]
    pub(crate) fn begin_debug_search_dag(&mut self, depth: i32, mv: &str) {
        self.debug.dag.begin_root(depth, mv);
    }

    #[cfg(feature = "search-debug")]
    pub(crate) fn emit_debug_search_dag(&mut self, score: i32, searched_nodes: u64) {
        self.debug.dag.emit(score, searched_nodes);
    }

    #[cfg(feature = "search-debug")]
    fn record_debug_dag_node(
        &mut self,
        st: &BoardState,
        ply: usize,
        depth: i32,
        alpha: i32,
        beta: i32,
        qsearch: bool,
    ) {
        let parent = self
            .rep_stack_len
            .checked_sub(2)
            .and_then(|index| self.rep_stack.get(index))
            .copied();
        self.debug
            .dag
            .record_node(st, parent, ply, depth, alpha, beta, qsearch);
    }

    #[cfg(feature = "search-debug")]
    fn record_debug_dag_eval(&mut self, hash: u64, eval: i32) {
        self.debug.dag.record_eval(hash, eval);
    }

    #[cfg(feature = "search-debug")]
    fn record_debug_dag_tt(&mut self, hash: u64, depth: i32, score: i32, flag: u8) {
        self.debug.dag.record_tt(hash, depth, score, flag);
    }

    #[cfg(feature = "search-debug")]
    fn record_debug_dag_draw(&mut self, hash: u64, status: DrawStatus) {
        self.debug.dag.record_draw(hash, status);
    }

    #[cfg(feature = "search-debug")]
    pub fn debug_stats(&self) -> SearchDebugStats {
        self.debug.stats
    }

    #[cfg(feature = "search-debug")]
    #[allow(clippy::too_many_arguments)]
    pub fn emit_debug_root_trace(
        &self,
        depth: i32,
        order: usize,
        mv: &str,
        alpha: i32,
        beta: i32,
        score: i32,
        nodes: u64,
    ) {
        if !self.debug.trace_roots {
            return;
        }
        let s = self.debug.stats;
        eprintln!(
            "info string search-debug root depth={depth} order={order} move={mv} alpha={alpha} beta={beta} score={score} nodes={nodes} seldepth={} tt_hits={} tt_max_depth={} tt_cutoffs={} rfp={} futility={} null={}/{} iid={} lmp={} history={} see={} lmr={}/{} lmr_sum={} lmr_max={} qnodes={} qdelta={} qsee={} qcheck_cap={} probcut_eligible={} probcut_safety={} probcut_tt={} probcut_candidates={} probcut_see={} probcut_qpass={} probcut_verify={} probcut_nodes={} probcut_cutoffs={} probcut_stops={} singular_candidates={} singular_safety={} singular_verify={} singular_nodes={} singular_extensions={} singular_extension_plies={} singular_negative={} singular_multicut={} singular_alternatives={} singular_stops={}",
            s.max_ply,
            s.tt_hits,
            s.tt_max_depth,
            s.tt_cutoffs,
            s.reverse_futility_cutoffs,
            s.futility_cutoffs,
            s.null_cutoffs,
            s.null_attempts,
            s.iid_reductions,
            s.lmp_cutoffs,
            s.history_skips,
            s.see_skips,
            s.lmr_researches,
            s.lmr_searches,
            s.lmr_reduction_sum,
            s.lmr_max_reduction,
            s.qnodes,
            s.q_delta_cutoffs,
            s.q_see_skips,
            s.q_checked_depth_exits,
            s.probcut_eligible_nodes,
            s.probcut_safety_rejections,
            s.probcut_tt_rejections,
            s.probcut_candidates,
            s.probcut_see_rejections,
            s.probcut_qsearch_passes,
            s.probcut_verifications,
            s.probcut_verification_nodes,
            s.probcut_cutoffs,
            s.probcut_stop_rejections,
            s.singular_candidates,
            s.singular_safety_rejections,
            s.singular_verifications,
            s.singular_verification_nodes,
            s.singular_extensions,
            s.singular_extension_plies,
            s.singular_negative_extensions,
            s.singular_multicut_cutoffs,
            s.singular_alternative_rejections,
            s.singular_stop_rejections,
        );
    }

    #[cfg(feature = "search-debug")]
    pub fn emit_debug_aspiration_trace(
        &self,
        depth: i32,
        alpha: i32,
        beta: i32,
        score: i32,
        result: &str,
    ) {
        if self.debug.trace_roots {
            eprintln!(
                "info string search-debug aspiration depth={depth} alpha={alpha} beta={beta} score={score} result={result}"
            );
        }
    }

    #[cfg(feature = "search-debug")]
    #[allow(clippy::too_many_arguments)]
    fn emit_debug_singular_candidate(
        &self,
        st: &BoardState,
        mv: Move,
        ply: usize,
        is_pv: bool,
        depth: i32,
        alpha: i32,
        beta: i32,
        eval: i32,
        tt_score: i32,
        tt_depth: i32,
        tt_flag: u8,
        tt_pv: bool,
        tt_age: u8,
        threshold: i32,
        verification_depth: i32,
        verification_score: i32,
        verification_nodes: u64,
        repetitions: u8,
        repeated_after_root: bool,
        extension: i32,
        outcome: &str,
    ) {
        if !self.debug.trace_singular_candidates {
            return;
        }
        let from = move_from(mv);
        let to = move_to(mv);
        let moved_piece = st.mailbox[from];
        let en_passant_capture = moved_piece != EMPTY_SQ
            && piece_type(moved_piece) == 0
            && st.ep == Some(to)
            && st.mailbox[to] == EMPTY_SQ;
        let capture = st.mailbox[to] != EMPTY_SQ || en_passant_capture;
        let promotion = move_promotion(mv) != 0;
        let shuffling = reversible_shuffle(&self.path_moves, ply, st.halfmove_clock);
        let path_extensions = self
            .singular_extensions_used
            .get(ply)
            .copied()
            .unwrap_or_default();
        eprintln!(
            "info string search-debug singular-event \
             {{\"hash\":\"{:016x}\",\"fen\":\"{}\",\"move\":\"{}\",\
             \"ply\":{},\"pv\":{},\"depth\":{},\"alpha\":{},\"beta\":{},\
             \"eval\":{},\"tt_score\":{},\"tt_depth\":{},\"tt_flag\":{},\
             \"tt_pv\":{},\"tt_age\":{},\
             \"threshold\":{},\"verification_depth\":{},\
             \"verification_score\":{},\"verification_nodes\":{},\
             \"halfmove_clock\":{},\"repetitions\":{},\
             \"repeated_after_root\":{},\"shuffling\":{},\
             \"path_extensions\":{},\"capture\":{},\"promotion\":{},\
             \"extension\":{},\"outcome\":\"{}\"}}",
            st.hash,
            crate::board::board_to_fen(st),
            crate::board::move_to_uci(st, mv),
            ply,
            is_pv,
            depth,
            alpha,
            beta,
            eval,
            tt_score,
            tt_depth,
            tt_flag,
            tt_pv,
            tt_age,
            threshold,
            verification_depth,
            verification_score,
            verification_nodes,
            st.halfmove_clock,
            repetitions,
            repeated_after_root,
            shuffling,
            path_extensions,
            capture,
            promotion,
            extension,
            outcome,
        );
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

    #[cfg(feature = "search-debug")]
    fn singular_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(any(feature = "search-debug", test))]
    fn set_restricted_verification(&mut self, active: bool) -> bool {
        std::mem::replace(&mut self.restricted_verification, active)
    }

    #[cfg(not(any(feature = "search-debug", test)))]
    #[inline(always)]
    fn set_restricted_verification(&mut self, _active: bool) -> bool {
        false
    }

    #[cfg(any(feature = "search-debug", test))]
    fn restricted_verification_active(&self) -> bool {
        self.restricted_verification
    }

    #[cfg(not(any(feature = "search-debug", test)))]
    #[inline(always)]
    fn restricted_verification_active(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    fn singular_multi_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_multi_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_multi_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    fn singular_multicut_enabled(&self) -> bool {
        self.debug.enable_singular_multicut
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_multicut_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    fn singular_negative_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_negative_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn singular_negative_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    fn qsearch_check_cap_enabled(&self) -> bool {
        !self.debug.disable_qsearch_check_cap
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn qsearch_check_cap_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn qsearch_delta_enabled(&self) -> bool {
        !self.debug.disable_qsearch_delta
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn qsearch_delta_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn qsearch_see_enabled(&self) -> bool {
        !self.debug.disable_qsearch_see
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn qsearch_see_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    fn probcut_enabled(&self) -> bool {
        !self.debug.disable_probcut
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    fn probcut_enabled(&self) -> bool {
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

    #[inline(always)]
    fn static_eval_threat_nnue<const CHESS960: bool, B: NnueBackend>(
        &self,
        st: &BoardState,
        ply: usize,
        net: &NNUENet,
    ) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return evaluate(st) * if st.w { 1 } else { -1 };
        }
        let stm = if st.w { WHITE } else { BLACK };
        let pc: u32 = (0..12).map(|i| st.bb[i].count_ones()).sum();
        if ply < self.nnue_stack.len() && ply < self.threat_stack.len() {
            net.forward_with_threats::<B>(&self.nnue_stack[ply], &self.threat_stack[ply], stm, pc)
        } else {
            let mut acc = NNUEAccumulator::new(net.hidden_size);
            acc.refresh_with_backend::<B>(net, st);
            let mut threats = NNUEThreatAccumulator::new(net.hidden_size);
            threats.refresh(net, st);
            net.forward_with_threats::<B>(&acc, &threats, stm, pc)
        }
    }

    pub fn corrected_eval(&self, st: &BoardState) -> i32 {
        match (st.chess960, self.nnue_net.as_deref()) {
            (true, Some(net)) => {
                if net.has_threat_features() {
                    ThreatNnueEval {
                        net,
                        _backend: ScalarNnueBackend,
                    }
                    .corrected_eval::<true>(self, st)
                } else {
                    NnueEval {
                        net,
                        _backend: ScalarNnueBackend,
                    }
                    .corrected_eval::<true>(self, st)
                }
            }
            (true, None) => ClassicEval.corrected_eval::<true>(self, st),
            (false, Some(net)) => {
                if net.has_threat_features() {
                    ThreatNnueEval {
                        net,
                        _backend: ScalarNnueBackend,
                    }
                    .corrected_eval::<false>(self, st)
                } else {
                    NnueEval {
                        net,
                        _backend: ScalarNnueBackend,
                    }
                    .corrected_eval::<false>(self, st)
                }
            }
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

    #[inline(always)]
    fn corrected_eval_threat_nnue<const CHESS960: bool, B: NnueBackend>(
        &self,
        st: &BoardState,
        net: &NNUENet,
    ) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return self.corrected_eval_classic::<CHESS960>(st);
        }
        let mut acc = NNUEAccumulator::new(net.hidden_size);
        acc.refresh_with_backend::<B>(net, st);
        let mut threats = NNUEThreatAccumulator::new(net.hidden_size);
        threats.refresh(net, st);
        let stm = if st.w { WHITE } else { BLACK };
        let pc: u32 = (0..12).map(|i| st.bb[i].count_ones()).sum();
        net.forward_with_threats::<B>(&acc, &threats, stm, pc)
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

    fn repetition_info(&self, reversible_plies: usize) -> (u8, bool) {
        let len = self.rep_stack_len.min(self.rep_stack.len());
        if len == 0 {
            return (0, false);
        }

        let current_idx = len - 1;
        let root_idx = self.rep_root_len.saturating_sub(1);
        let reversible_plies = reversible_plies.min(current_idx);
        let earliest_idx = current_idx - reversible_plies;
        let current = self.rep_stack[current_idx];
        let mut occurrences = 0u8;
        let mut repeated_after_root = false;
        let mut idx = current_idx;

        loop {
            if self.rep_stack[idx] == current {
                occurrences += 1;
                // One prior occurrence is enough to stop a cycle created entirely by
                // this search. At or before the root it is only game history, where a
                // claim still requires the normal three occurrences.
                if occurrences > 1 && idx > root_idx {
                    repeated_after_root = true;
                }
                if occurrences == 5 {
                    break;
                }
            }
            if idx < 2 || idx - 2 < earliest_idx {
                break;
            }
            idx -= 2;
        }

        (occurrences, repeated_after_root)
    }

    pub(crate) fn current_position_repeats(&self, reversible_plies: usize) -> bool {
        self.repetition_info(reversible_plies).0 >= 2
    }

    #[cfg(test)]
    fn is_repetition(&self) -> bool {
        self.repetition_info(usize::MAX).0 >= 3
    }

    fn draw_status(&self, st: &BoardState, ply: usize, minimum_ply: usize) -> DrawStatus {
        if is_dead_position(st) {
            return DrawStatus::Automatic;
        }
        if ply < minimum_ply {
            return DrawStatus::None;
        }

        let (repetitions, repeated_after_root) =
            self.repetition_info(usize::from(st.halfmove_clock));
        if st.halfmove_clock >= MAX_HALF_MOVE_CLOCK || repetitions >= 5 {
            DrawStatus::Automatic
        } else if st.halfmove_clock >= 100 || repetitions >= 3 {
            DrawStatus::Claimable
        } else if repeated_after_root {
            DrawStatus::SearchCycle
        } else {
            DrawStatus::None
        }
    }

    fn draw_score(
        &mut self,
        st: &BoardState,
        ply: usize,
        minimum_ply: usize,
        in_check: bool,
    ) -> Option<i32> {
        let status = self.draw_status(st, ply, minimum_ply);
        #[cfg(feature = "search-debug")]
        self.record_debug_dag_draw(st.hash, status);
        if status == DrawStatus::None {
            return None;
        }
        if in_check && generate_moves(st, st.w, &st.cr, st.ep).is_empty() {
            return Some(-MATE + ply as i32);
        }
        Some(0)
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
        let ok = {
            let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
            right[0].update_from_parent_with_backend::<B>(
                &left[ply], net, st_before, sr, sc, er, ec, promotion,
            )
        };

        if !ok {
            B::refresh(&mut self.nnue_stack[ply + 1], net, st_after);
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn push_threat_nnue_acc<B: NnueBackend>(
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
        if ply + 1 >= self.nnue_stack.len() || ply + 1 > MAX_PLY {
            return;
        }
        let ok = {
            let (left, right) = self.nnue_stack.split_at_mut(ply + 1);
            right[0].update_from_parent_with_backend::<B>(
                &left[ply], net, st_before, sr, sc, er, ec, promotion,
            )
        };

        if !ok {
            self.nnue_stack[ply + 1].refresh_with_backend::<B>(net, st_after);
        }

        if ply + 1 >= self.threat_stack.len() {
            self.threat_stack
                .resize(ply + 2, NNUEThreatAccumulator::new(net.hidden_size));
        }
        let updated = {
            let (left, right) = self.threat_stack.split_at_mut(ply + 1);
            right[0].update_from_parent(&left[ply], net, st_before, st_after)
        };
        if !updated {
            self.threat_stack[ply + 1].refresh(net, st_after);
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
            (true, Some(net)) => {
                if net.has_threat_features() {
                    self.qsearch_mode_scalar::<true, false, _>(
                        st,
                        alpha,
                        beta,
                        depth,
                        start,
                        tl,
                        cnt,
                        ply,
                        ThreatNnueEval {
                            net,
                            _backend: ScalarNnueBackend,
                        },
                    )
                } else {
                    self.qsearch_mode_scalar::<true, false, _>(
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
                    )
                }
            }
            (true, None) => self.qsearch_mode_scalar::<true, false, _>(
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
            (false, Some(net)) => {
                if net.has_threat_features() {
                    self.qsearch_mode_scalar::<false, false, _>(
                        st,
                        alpha,
                        beta,
                        depth,
                        start,
                        tl,
                        cnt,
                        ply,
                        ThreatNnueEval {
                            net,
                            _backend: ScalarNnueBackend,
                        },
                    )
                } else {
                    self.qsearch_mode_scalar::<false, false, _>(
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
                    )
                }
            }
            (false, None) => self.qsearch_mode_scalar::<false, false, _>(
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
    fn qsearch_mode_scalar<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
    fn qsearch_mode_simd128<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
            qsearch_mode_simd128,
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
    fn qsearch_mode_simd256<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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

    #[allow(clippy::too_many_arguments)]
    fn qsearch_mode_simd512<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
            qsearch_mode_simd512,
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
    unsafe fn qsearch_mode_x86_v3<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
    unsafe fn qsearch_mode_x86_avx512<
        const CHESS960: bool,
        const NODE_LIMITED: bool,
        E: SearchEval,
    >(
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
        if self.node_limit.is_some() {
            self.negamax_with_limits::<true>(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
        } else {
            self.negamax_with_limits::<false>(st, depth, ply, alpha, beta, can_null, start, tl, cnt)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_with_limits<const NODE_LIMITED: bool>(
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
        match self.search_backend {
            SearchBackendKind::X86Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    return unsafe {
                        self.negamax_x86_avx512::<NODE_LIMITED>(
                            st, depth, ply, alpha, beta, can_null, start, tl, cnt,
                        )
                    };
                }
                #[allow(unreachable_code)]
                self.negamax_scalar::<NODE_LIMITED>(
                    st, depth, ply, alpha, beta, can_null, start, tl, cnt,
                )
            }
            SearchBackendKind::X86V3 => {
                #[cfg(target_arch = "x86_64")]
                {
                    return unsafe {
                        self.negamax_x86_v3::<NODE_LIMITED>(
                            st, depth, ply, alpha, beta, can_null, start, tl, cnt,
                        )
                    };
                }
                #[allow(unreachable_code)]
                self.negamax_scalar::<NODE_LIMITED>(
                    st, depth, ply, alpha, beta, can_null, start, tl, cnt,
                )
            }
            SearchBackendKind::Aarch64Simd128 => self.negamax_simd128::<NODE_LIMITED>(
                st, depth, ply, alpha, beta, can_null, start, tl, cnt,
            ),
            SearchBackendKind::Aarch64Simd256 => self.negamax_simd256::<NODE_LIMITED>(
                st, depth, ply, alpha, beta, can_null, start, tl, cnt,
            ),
            SearchBackendKind::Aarch64Simd512 => self.negamax_simd512::<NODE_LIMITED>(
                st, depth, ply, alpha, beta, can_null, start, tl, cnt,
            ),
            _ => self.negamax_scalar::<NODE_LIMITED>(
                st, depth, ply, alpha, beta, can_null, start, tl, cnt,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_scalar<const NODE_LIMITED: bool>(
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
            (true, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_scalar::<true, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: ScalarNnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_scalar::<true, NODE_LIMITED, _>(
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
                    )
                }
            }
            (true, None) => self.negamax_mode_scalar::<true, NODE_LIMITED, _>(
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
            (false, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_scalar::<false, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: ScalarNnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_scalar::<false, NODE_LIMITED, _>(
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
                    )
                }
            }
            (false, None) => self.negamax_mode_scalar::<false, NODE_LIMITED, _>(
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
    fn negamax_simd128<const NODE_LIMITED: bool>(
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
            (true, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd128::<true, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: Simd128NnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd128::<true, NODE_LIMITED, _>(
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
                            _backend: Simd128NnueBackend,
                        },
                    )
                }
            }
            (true, None) => self.negamax_mode_simd128::<true, NODE_LIMITED, _>(
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
            (false, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd128::<false, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: Simd128NnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd128::<false, NODE_LIMITED, _>(
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
                            _backend: Simd128NnueBackend,
                        },
                    )
                }
            }
            (false, None) => self.negamax_mode_simd128::<false, NODE_LIMITED, _>(
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
    fn negamax_simd256<const NODE_LIMITED: bool>(
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
            (true, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd256::<true, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: SimdNnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd256::<true, NODE_LIMITED, _>(
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
                    )
                }
            }
            (true, None) => self.negamax_mode_simd256::<true, NODE_LIMITED, _>(
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
            (false, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd256::<false, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: SimdNnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd256::<false, NODE_LIMITED, _>(
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
                    )
                }
            }
            (false, None) => self.negamax_mode_simd256::<false, NODE_LIMITED, _>(
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
    fn negamax_simd512<const NODE_LIMITED: bool>(
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
            (true, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd512::<true, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: Simd512NnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd512::<true, NODE_LIMITED, _>(
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
                            _backend: Simd512NnueBackend,
                        },
                    )
                }
            }
            (true, None) => self.negamax_mode_simd512::<true, NODE_LIMITED, _>(
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
            (false, Some(net)) => {
                if net.has_threat_features() {
                    self.negamax_mode_simd512::<false, NODE_LIMITED, _>(
                        st,
                        depth,
                        ply,
                        alpha,
                        beta,
                        can_null,
                        start,
                        tl,
                        cnt,
                        ThreatNnueEval {
                            net,
                            _backend: Simd512NnueBackend,
                        },
                    )
                } else {
                    self.negamax_mode_simd512::<false, NODE_LIMITED, _>(
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
                            _backend: Simd512NnueBackend,
                        },
                    )
                }
            }
            (false, None) => self.negamax_mode_simd512::<false, NODE_LIMITED, _>(
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
    unsafe fn negamax_x86_v3<const NODE_LIMITED: bool>(
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
                (true, Some(net)) => {
                    if net.has_threat_features() {
                        self.negamax_mode_x86_v3::<true, NODE_LIMITED, _>(
                            st,
                            depth,
                            ply,
                            alpha,
                            beta,
                            can_null,
                            start,
                            tl,
                            cnt,
                            ThreatNnueEval {
                                net,
                                _backend: SimdNnueBackend,
                            },
                        )
                    } else {
                        self.negamax_mode_x86_v3::<true, NODE_LIMITED, _>(
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
                        )
                    }
                }
                (true, None) => self.negamax_mode_x86_v3::<true, NODE_LIMITED, _>(
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
                (false, Some(net)) => {
                    if net.has_threat_features() {
                        self.negamax_mode_x86_v3::<false, NODE_LIMITED, _>(
                            st,
                            depth,
                            ply,
                            alpha,
                            beta,
                            can_null,
                            start,
                            tl,
                            cnt,
                            ThreatNnueEval {
                                net,
                                _backend: SimdNnueBackend,
                            },
                        )
                    } else {
                        self.negamax_mode_x86_v3::<false, NODE_LIMITED, _>(
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
                        )
                    }
                }
                (false, None) => self.negamax_mode_x86_v3::<false, NODE_LIMITED, _>(
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
    unsafe fn negamax_x86_avx512<const NODE_LIMITED: bool>(
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
                (true, Some(net)) => {
                    if net.has_threat_features() {
                        self.negamax_mode_x86_avx512::<true, NODE_LIMITED, _>(
                            st,
                            depth,
                            ply,
                            alpha,
                            beta,
                            can_null,
                            start,
                            tl,
                            cnt,
                            ThreatNnueEval {
                                net,
                                _backend: Avx512NnueBackend,
                            },
                        )
                    } else {
                        self.negamax_mode_x86_avx512::<true, NODE_LIMITED, _>(
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
                        )
                    }
                }
                (true, None) => self.negamax_mode_x86_avx512::<true, NODE_LIMITED, _>(
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
                (false, Some(net)) => {
                    if net.has_threat_features() {
                        self.negamax_mode_x86_avx512::<false, NODE_LIMITED, _>(
                            st,
                            depth,
                            ply,
                            alpha,
                            beta,
                            can_null,
                            start,
                            tl,
                            cnt,
                            ThreatNnueEval {
                                net,
                                _backend: Avx512NnueBackend,
                            },
                        )
                    } else {
                        self.negamax_mode_x86_avx512::<false, NODE_LIMITED, _>(
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
                        )
                    }
                }
                (false, None) => self.negamax_mode_x86_avx512::<false, NODE_LIMITED, _>(
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
    fn negamax_mode_scalar<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
    fn negamax_mode_simd128<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
            negamax_mode_simd128,
            qsearch_mode_simd128,
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
    fn negamax_mode_simd256<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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

    #[allow(clippy::too_many_arguments)]
    fn negamax_mode_simd512<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
            negamax_mode_simd512,
            qsearch_mode_simd512,
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
    unsafe fn negamax_mode_x86_v3<const CHESS960: bool, const NODE_LIMITED: bool, E: SearchEval>(
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
    unsafe fn negamax_mode_x86_avx512<
        const CHESS960: bool,
        const NODE_LIMITED: bool,
        E: SearchEval,
    >(
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

impl<'a, B: NnueBackend> SearchEval for ThreatNnueEval<'a, B> {
    #[inline(always)]
    fn static_eval<const CHESS960: bool>(
        self,
        searcher: &Searcher,
        st: &BoardState,
        ply: usize,
    ) -> i32 {
        searcher.static_eval_threat_nnue::<CHESS960, B>(st, ply, self.net)
    }

    #[inline(always)]
    fn corrected_eval<const CHESS960: bool>(self, searcher: &Searcher, st: &BoardState) -> i32 {
        searcher.corrected_eval_threat_nnue::<CHESS960, B>(st, self.net)
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
        searcher.push_threat_nnue_acc::<B>(
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
        if ply + 1 >= searcher.threat_stack.len() && ply + 1 < MAX_PLY + 1 {
            searcher
                .threat_stack
                .resize(ply + 2, NNUEThreatAccumulator::new(self.net.hidden_size));
        }
    }

    #[inline(always)]
    fn copy_null_acc(self, searcher: &mut Searcher, ply: usize) {
        if ply + 1 < searcher.nnue_stack.len() {
            let (left, right) = searcher.nnue_stack.split_at_mut(ply + 1);
            right[0].clone_from(&left[ply]);
        }
        if ply + 1 < searcher.threat_stack.len() {
            let (left, right) = searcher.threat_stack.split_at_mut(ply + 1);
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
            disable_qsearch_check_cap: env_flag("EMBER_DISABLE_QSEARCH_CHECK_CAP"),
            disable_qsearch_delta: env_flag("EMBER_DISABLE_QSEARCH_DELTA"),
            disable_qsearch_see: env_flag("EMBER_DISABLE_QSEARCH_SEE"),
            disable_probcut: env_flag("EMBER_DISABLE_PROBCUT"),
            disable_reverse_futility: env_flag("EMBER_DISABLE_REVERSE_FUTILITY"),
            disable_see_pruning: env_flag("EMBER_DISABLE_SEE_PRUNING"),
            enable_singular_extensions: env_flag("EMBER_ENABLE_SINGULAR_EXTENSIONS"),
            enable_singular_multi_extensions: env_flag("EMBER_ENABLE_SINGULAR_MULTI_EXTENSIONS"),
            enable_singular_multicut: env_flag("EMBER_ENABLE_SINGULAR_MULTICUT"),
            enable_singular_negative_extensions: env_flag(
                "EMBER_ENABLE_SINGULAR_NEGATIVE_EXTENSIONS",
            ),
            trace_roots: env_flag("EMBER_TRACE_ROOT_SEARCH"),
            trace_singular_candidates: env_flag("EMBER_TRACE_SINGULAR_CANDIDATES"),
            dag: SearchDagTrace::from_env(),
            stats: SearchDebugStats::default(),
        }
    }

    fn reset_stats(&mut self) {
        self.stats = SearchDebugStats::default();
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

#[cfg(feature = "search-debug")]
fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

#[derive(Clone)]
pub struct SearchLearning {
    history: [[i32; 64]; 64],
    counter_move: [[Option<Move>; 64]; 13],
    corr_hist: [i32; CORR_HIST_SIZE * 2],
}

struct ThreadResult {
    thread_id: usize,
    best_move: Move,
    score: i32,
    depth: i32,
    nodes: u64,
    learning: Option<Box<SearchLearning>>,
}

#[derive(Clone, Copy)]
pub struct LazySmpSearchLimits {
    pub soft_time: f64,
    pub hard_time: f64,
    pub depth: i32,
    pub node_limit: Option<u64>,
    pub start: Instant,
}

#[derive(Clone)]
struct LazySmpRootContext {
    rep_stack: Vec<u64>,
    rep_stack_len: usize,
    nnue_net: Option<Arc<NNUENet>>,
    search_backend: SearchBackendKind,
    syzygy: SyzygyTables,
    tt_mb: usize,
    pondering: Arc<AtomicBool>,
    learning: SearchLearning,
}

impl LazySmpRootContext {
    fn from_searcher(searcher: &Searcher) -> Self {
        Self {
            rep_stack: searcher.rep_stack.clone(),
            rep_stack_len: searcher.rep_stack_len,
            nnue_net: searcher.nnue_net.clone(),
            search_backend: searcher.search_backend,
            syzygy: searcher.syzygy.clone(),
            tt_mb: searcher.tt_mb,
            pondering: Arc::clone(&searcher.pondering),
            learning: searcher.export_learning(),
        }
    }

    fn prepare_worker(
        &self,
        searcher: &mut Searcher,
        shared_tt: Arc<SharedTT>,
        stopped: Arc<AtomicBool>,
        st: &BoardState,
        initialize_learning: bool,
    ) {
        searcher.shared_tt = shared_tt;
        searcher.stopped = stopped;
        searcher.pondering = Arc::clone(&self.pondering);
        if initialize_learning {
            searcher.import_learning(&self.learning);
        }
        searcher.prepare_for_search();
        searcher.rep_stack.clone_from(&self.rep_stack);
        searcher.rep_stack_len = self.rep_stack_len;
        searcher.rep_root_len = searcher.rep_stack_len;
        searcher.search_backend = self.search_backend;
        searcher.syzygy = self.syzygy.clone();
        searcher.tt_mb = self.tt_mb;

        let old_hidden_size = searcher.nnue_stack.first().map(|acc| acc.hs);
        let new_hidden_size = self.nnue_net.as_deref().map(|net| net.hidden_size);
        if old_hidden_size != new_hidden_size {
            searcher.nnue_stack.clear();
        }
        searcher.nnue_net = self.nnue_net.clone();
        searcher.init_nnue_stack(st);
    }
}

struct LazySmpSearchJob {
    shared_tt: Arc<SharedTT>,
    verification_move: Option<Move>,
    verification_tt: Option<Arc<SharedTT>>,
    stopped: Arc<AtomicBool>,
    st: BoardState,
    root_moves: Arc<Vec<Move>>,
    num_threads: usize,
    root_depth_extension: fn(&BoardState, Move) -> i32,
    root_lmr_exemptions: Arc<Vec<Move>>,
    limits: LazySmpSearchLimits,
    root_context: Arc<LazySmpRootContext>,
    start: Instant,
    global_best_depth: Arc<AtomicI32>,
    global_nodes: Arc<AtomicU64>,
    node_limit_counter: Option<Arc<AtomicU64>>,
    worker_best_moves: Vec<AtomicU64>,
    worker_depths: Vec<AtomicI32>,
}

enum LazySmpWorkerCommand {
    Search {
        job: Arc<LazySmpSearchJob>,
        result_tx: mpsc::Sender<ThreadResult>,
    },
    ClearLearning {
        done_tx: mpsc::Sender<()>,
    },
    Shutdown,
}

struct LazySmpWorker {
    command_tx: mpsc::Sender<LazySmpWorkerCommand>,
    handle: Option<std::thread::JoinHandle<()>>,
}

fn spawn_lazy_smp_worker(thread_id: usize) -> std::io::Result<LazySmpWorker> {
    let (command_tx, command_rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(format!("rts-{thread_id}"))
        .stack_size(SEARCH_THREAD_STACK_SIZE)
        .spawn(move || {
            let mut searcher = None;
            while let Ok(command) = command_rx.recv() {
                match command {
                    LazySmpWorkerCommand::Search { job, result_tx } => {
                        let initialize_learning = searcher.is_none();
                        let searcher = searcher.get_or_insert_with(|| {
                            Searcher::new(Arc::clone(&job.shared_tt), Arc::clone(&job.stopped))
                        });
                        job.root_context.prepare_worker(
                            searcher,
                            Arc::clone(&job.shared_tt),
                            Arc::clone(&job.stopped),
                            &job.st,
                            initialize_learning,
                        );
                        if thread_id == 1 {
                            if let Some(verification_tt) = &job.verification_tt {
                                searcher.shared_tt = Arc::clone(verification_tt);
                            }
                        }
                        let result = run_lazy_smp_worker(searcher, thread_id, &job);
                        let _ = result_tx.send(result);
                    }
                    LazySmpWorkerCommand::ClearLearning { done_tx } => {
                        if let Some(searcher) = searcher.as_mut() {
                            searcher.clear_learning();
                        }
                        let _ = done_tx.send(());
                    }
                    LazySmpWorkerCommand::Shutdown => break,
                }
            }
        })?;
    Ok(LazySmpWorker {
        command_tx,
        handle: Some(handle),
    })
}

pub struct LazySmpPool {
    workers: Mutex<Vec<LazySmpWorker>>,
    search_lock: Mutex<()>,
}

impl Default for LazySmpPool {
    fn default() -> Self {
        Self::new()
    }
}

impl LazySmpPool {
    pub fn new() -> Self {
        Self {
            workers: Mutex::new(Vec::new()),
            search_lock: Mutex::new(()),
        }
    }

    fn ensure_workers(&self, count: usize) {
        let mut workers = self.workers.lock().unwrap();
        for thread_id in 0..count.min(workers.len()) {
            let finished = workers[thread_id]
                .handle
                .as_ref()
                .is_some_and(|handle| handle.is_finished());
            if finished {
                let mut replacement = spawn_lazy_smp_worker(thread_id)
                    .unwrap_or_else(|error| panic!("failed to replace search worker: {error}"));
                std::mem::swap(&mut workers[thread_id], &mut replacement);
                if let Some(handle) = replacement.handle.take() {
                    let _ = handle.join();
                }
            }
        }
        while workers.len() < count {
            let thread_id = workers.len();
            workers.push(
                spawn_lazy_smp_worker(thread_id)
                    .unwrap_or_else(|error| panic!("failed to spawn search worker: {error}")),
            );
        }
    }

    pub fn clear_learning(&self) {
        let _search_guard = self.search_lock.lock().unwrap();
        let workers = self.workers.lock().unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let mut sent = 0;
        for worker in workers.iter() {
            if worker
                .command_tx
                .send(LazySmpWorkerCommand::ClearLearning {
                    done_tx: done_tx.clone(),
                })
                .is_ok()
            {
                sent += 1;
            }
        }
        drop(done_tx);
        for _ in 0..sent {
            let _ = done_rx.recv();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        shared_tt: Arc<SharedTT>,
        st: &BoardState,
        root_moves: &[Move],
        root_depth_extension: fn(&BoardState, Move) -> i32,
        root_lmr_exemptions: &[Move],
        limits: LazySmpSearchLimits,
        num_threads: usize,
        root_searcher: &mut Searcher,
    ) -> (Move, i32, i32, u64) {
        assert!(!root_moves.is_empty());
        let num_threads = num_threads.max(1);
        let _search_guard = self.search_lock.lock().unwrap();
        self.ensure_workers(num_threads);

        let verification_move = if num_threads > 1 {
            lazy_smp_verification_move(st, root_moves)
        } else {
            None
        };
        // Keep the forced-line result independent of horizon-sensitive writes
        // from workers that are comparing the complete root.
        let verification_tt =
            verification_move.map(|_| Arc::new(SharedTT::new(LAZY_SMP_VERIFICATION_TT_MB)));
        let job = Arc::new(LazySmpSearchJob {
            shared_tt,
            verification_move,
            verification_tt,
            stopped: Arc::clone(&root_searcher.stopped),
            st: *st,
            root_moves: Arc::new(root_moves.to_vec()),
            num_threads,
            root_depth_extension,
            root_lmr_exemptions: Arc::new(root_lmr_exemptions.to_vec()),
            limits,
            root_context: Arc::new(LazySmpRootContext::from_searcher(root_searcher)),
            start: limits.start,
            global_best_depth: Arc::new(AtomicI32::new(0)),
            global_nodes: Arc::new(AtomicU64::new(0)),
            node_limit_counter: limits.node_limit.map(|_| Arc::new(AtomicU64::new(0))),
            worker_best_moves: (0..num_threads).map(|_| AtomicU64::new(0)).collect(),
            worker_depths: (0..num_threads).map(|_| AtomicI32::new(0)).collect(),
        });
        let (result_tx, result_rx) = mpsc::channel();

        {
            let workers = self.workers.lock().unwrap();
            for worker in workers.iter().take(num_threads) {
                let command = LazySmpWorkerCommand::Search {
                    job: Arc::clone(&job),
                    result_tx: result_tx.clone(),
                };
                if worker.command_tx.send(command).is_err() {
                    panic!("persistent search worker stopped unexpectedly");
                }
            }
        }
        drop(result_tx);

        let mut results = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            match result_rx.recv() {
                Ok(result) => results.push(result),
                Err(_) => break,
            }
        }
        let total_nodes = results.iter().map(|result| result.nodes).sum();
        if let Some(learning) = results
            .iter_mut()
            .find(|result| result.thread_id == 0)
            .and_then(|result| result.learning.take())
        {
            root_searcher.import_learning(&learning);
        }
        let Some(best) = select_lazy_smp_result(&results, st, root_moves) else {
            return (root_moves[0], 0, 0, total_nodes);
        };
        if best.depth > 0 {
            print_lazy_smp_info(
                &job,
                best.best_move,
                best.score,
                best.depth,
                total_nodes,
                job.start.elapsed().as_secs_f64(),
            );
        }
        (best.best_move, best.score, best.depth, total_nodes)
    }

    #[cfg(test)]
    fn worker_ids(&self) -> Vec<std::thread::ThreadId> {
        self.workers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|worker| worker.handle.as_ref().map(|handle| handle.thread().id()))
            .collect()
    }
}

impl Drop for LazySmpPool {
    fn drop(&mut self) {
        let workers = self.workers.get_mut().unwrap();
        for worker in workers.iter() {
            let _ = worker.command_tx.send(LazySmpWorkerCommand::Shutdown);
        }
        for worker in workers.iter_mut() {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn select_lazy_smp_result<'a>(
    results: &'a [ThreadResult],
    st: &BoardState,
    root_moves: &[Move],
) -> Option<&'a ThreadResult> {
    let max_depth = results.iter().map(|result| result.depth).max()?;
    let near_deep_floor = max_depth.saturating_sub(1).max(1);
    let principal = results
        .iter()
        .find(|result| result.thread_id == 0 && result.depth > 0);

    // A helper can spend the whole search verifying a tactically ordered root
    // capture. Accept its deeper result only within a small score-noise margin
    // of the principal worker's alternative.
    if let Some(root_move) = lazy_smp_verification_move(st, root_moves) {
        if let Some(tactical) = results
            .iter()
            .filter(|result| {
                result.thread_id == 1
                    && result.depth >= near_deep_floor
                    && result.best_move == root_move
                    && principal.is_none_or(|principal| {
                        result.score.saturating_add(LAZY_SMP_VERIFICATION_MARGIN_CP)
                            >= principal.score
                    })
            })
            .max_by(|a, b| {
                a.depth
                    .cmp(&b.depth)
                    .then_with(|| a.score.cmp(&b.score))
                    .then_with(|| b.thread_id.cmp(&a.thread_id))
            })
        {
            return Some(tactical);
        }
    }

    // Thread zero owns the selected result, persistent learning, and the root
    // TT entry. Settled helpers can coordinate its stop only when its published
    // move agrees; their horizon-dependent votes never replace that result.
    if let Some(principal) = principal {
        return Some(principal);
    }

    results
        .iter()
        .filter(|result| result.depth >= near_deep_floor)
        .max_by(|a, b| {
            let a_support = results
                .iter()
                .filter(|result| result.depth >= near_deep_floor && result.best_move == a.best_move)
                .count();
            let b_support = results
                .iter()
                .filter(|result| result.depth >= near_deep_floor && result.best_move == b.best_move)
                .count();

            a_support
                .cmp(&b_support)
                .then_with(|| a.depth.cmp(&b.depth))
                .then_with(|| a.score.cmp(&b.score))
        })
}

fn lazy_smp_verification_move(st: &BoardState, root_moves: &[Move]) -> Option<Move> {
    let &root_move = root_moves.first()?;
    let attacker = st.mailbox[move_from(root_move)];
    let victim = st.mailbox[move_to(root_move)];
    let verifies_minor_exchange = attacker != EMPTY_SQ
        && victim != EMPTY_SQ
        && piece_type(attacker) == 2
        && piece_type(victim) == 1
        && see(&st.bb, move_from(root_move), move_to(root_move)) >= -25;
    verifies_minor_exchange.then_some(root_move)
}

fn lazy_smp_worker_root_moves(
    st: &BoardState,
    root_moves: &[Move],
    thread_id: usize,
    num_threads: usize,
) -> Vec<Move> {
    if thread_id == 1 && num_threads > 1 {
        if let Some(root_move) = lazy_smp_verification_move(st, root_moves) {
            return vec![root_move];
        }
    }

    lazy_smp_root_moves(root_moves, thread_id, num_threads)
}

fn lazy_smp_root_moves(root_moves: &[Move], thread_id: usize, num_threads: usize) -> Vec<Move> {
    if (4..=8).contains(&num_threads) && root_moves.len() >= num_threads && thread_id > 0 {
        let helper_count = num_threads - 1;
        let helper_lane = thread_id - 1;
        let mut moves = Vec::with_capacity(root_moves.len());

        // Give each helper a disjoint prefix of non-PV root moves, then let it
        // search the complete root so every worker still produces a full vote.
        for (index, &mv) in root_moves.iter().enumerate() {
            if index > 0 && (index - 1) % helper_count == helper_lane {
                moves.push(mv);
            }
        }
        for (index, &mv) in root_moves.iter().enumerate() {
            if index == 0 || (index - 1) % helper_count != helper_lane {
                moves.push(mv);
            }
        }
        debug_assert_eq!(moves.len(), root_moves.len());
        return moves;
    }

    let mut moves = root_moves.to_vec();
    if moves.len() > 1 && thread_id > 0 {
        let offset = thread_id % moves.len();
        moves.rotate_left(offset);
    }
    moves
}

fn root_lmr_exemption_depth_bonus(exemptions: &[Move], mv: Move) -> i32 {
    i32::from(exemptions.contains(&mv))
}

fn print_lazy_smp_info(
    job: &LazySmpSearchJob,
    best_move: Move,
    score: i32,
    depth: i32,
    nodes: u64,
    elapsed: f64,
) {
    let score_str = if score.abs() > 90_000 {
        let mate_in = (MATE - score.abs()) / 2 + 1;
        if score > 0 {
            format!("mate {mate_in}")
        } else {
            format!("mate -{mate_in}")
        }
    } else {
        format!("cp {score}")
    };
    let pv_line = extract_pv_line(&job.shared_tt, &job.st, best_move);
    let pv_str = format_pv_line_uci(&job.st, &pv_line);
    let nps = if elapsed > 0.0 {
        (nodes as f64 / elapsed) as i64
    } else {
        0
    };
    println!(
        "info depth {} score {} nodes {} nps {} time {} pv {}",
        depth,
        score_str,
        nodes,
        nps,
        (elapsed * 1000.0) as u64,
        pv_str
    );
}

fn malformed_promotion_move(st: &BoardState, mv: Move) -> bool {
    let promo = move_promotion(mv).to_ascii_uppercase();
    let from = move_from(mv);
    let to_rank = move_er(mv);
    let fpi = st.mailbox[from];
    let reaches_back_rank = to_rank == 0 || to_rank == 7;
    let valid_promo = matches!(promo, b'Q' | b'R' | b'B' | b'N');

    if promo != 0 {
        return fpi == EMPTY_SQ || piece_type(fpi) != 0 || !reaches_back_rank || !valid_promo;
    }

    fpi != EMPTY_SQ && piece_type(fpi) == 0 && reaches_back_rank
}

pub fn format_pv_line_uci(st: &BoardState, pv_line: &[Move]) -> String {
    let mut current = *st;
    let mut out = Vec::with_capacity(pv_line.len());

    for &mv in pv_line {
        if malformed_promotion_move(&current, mv) {
            break;
        }

        let legal_moves = generate_moves(&current, current.w, &current.cr, current.ep);
        if !legal_moves.contains(&mv) {
            break;
        }

        out.push(crate::board::move_to_uci(&current, mv));
        apply_move(
            &mut current,
            move_sr(mv),
            move_sc(mv),
            move_er(mv),
            move_ec(mv),
            move_promotion(mv),
        );
    }

    out.join(" ")
}

pub fn extract_pv_line(shared_tt: &SharedTT, st: &BoardState, first_move: Move) -> Vec<Move> {
    if malformed_promotion_move(st, first_move) {
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
            if malformed_promotion_move(&prev_st, best) {
                break;
            }
            let promo = move_promotion(best);
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

#[allow(clippy::too_many_arguments)]
pub fn lazy_smp_search(
    pool: &LazySmpPool,
    shared_tt: Arc<SharedTT>,
    st: &BoardState,
    root_moves: &[Move],
    root_depth_extension: fn(&BoardState, Move) -> i32,
    root_lmr_exemptions: &[Move],
    limits: LazySmpSearchLimits,
    num_threads: usize,
    root_searcher: &mut Searcher,
) -> (Move, i32, i32, u64) {
    pool.search(
        shared_tt,
        st,
        root_moves,
        root_depth_extension,
        root_lmr_exemptions,
        limits,
        num_threads,
        root_searcher,
    )
}

#[derive(Clone, Copy, Debug)]
struct LazySmpAgreement {
    disagreement: f64,
    comparable_workers: usize,
    principal_agrees: bool,
}

fn lazy_smp_worker_agreement(
    job: &LazySmpSearchJob,
    thread_id: usize,
    best_move: Move,
    depth: i32,
) -> LazySmpAgreement {
    let verification_thread = job.verification_move.map(|_| 1);
    if verification_thread == Some(thread_id) {
        return LazySmpAgreement {
            disagreement: 0.0,
            comparable_workers: 0,
            principal_agrees: false,
        };
    }

    let mut comparable = 0usize;
    let mut different = 0usize;
    for other in 0..job.num_threads {
        if other == thread_id
            || verification_thread == Some(other)
            || job.worker_depths[other].load(Ordering::Acquire) < depth
        {
            continue;
        }
        let other_move = job.worker_best_moves[other].load(Ordering::Relaxed) as Move;
        if other_move == NO_MOVE {
            continue;
        }
        comparable += 1;
        different += usize::from(other_move != best_move);
    }
    LazySmpAgreement {
        disagreement: if comparable == 0 {
            0.0
        } else {
            different as f64 / comparable as f64
        },
        comparable_workers: comparable,
        principal_agrees: thread_id == 0
            || (job.worker_depths[0].load(Ordering::Acquire) > 0
                && job.worker_best_moves[0].load(Ordering::Relaxed) as Move == best_move),
    }
}

#[cfg(test)]
fn lazy_smp_worker_disagreement(
    job: &LazySmpSearchJob,
    thread_id: usize,
    best_move: Move,
    depth: i32,
) -> f64 {
    lazy_smp_worker_agreement(job, thread_id, best_move, depth).disagreement
}

fn lazy_smp_worker_can_coordinate_stop(
    thread_id: usize,
    verification_thread: Option<usize>,
    soft_time: f64,
    timing: IterationTiming,
    agreement: LazySmpAgreement,
    time_decision_stop: bool,
) -> bool {
    if !time_decision_stop {
        return false;
    }
    if thread_id == 0 {
        return true;
    }

    // A helper may interrupt a long leader iteration only after another
    // complete root search confirms a settled result. The leader still owns
    // the final move and persistent learning; helpers merely coordinate the
    // shared stop token once the nominal allocation has been consumed.
    verification_thread != Some(thread_id)
        && timing.elapsed_seconds >= soft_time
        && timing.stable_iterations >= 2
        && timing.score_change_cp.abs() <= 80
        && agreement.comparable_workers > 0
        && agreement.principal_agrees
        && agreement.disagreement <= 0.25
}

fn run_lazy_smp_worker(
    searcher: &mut Searcher,
    thread_id: usize,
    job: &LazySmpSearchJob,
) -> ThreadResult {
    let st = job.st;
    let limits = job.limits;
    let start = job.start;
    let stopped = &job.stopped;
    let my_moves = lazy_smp_worker_root_moves(&st, &job.root_moves, thread_id, job.num_threads);
    searcher.set_shared_node_limit(limits.node_limit, job.node_limit_counter.clone());

    let mut best_move = my_moves[0];
    let mut best_score = 0i32;
    let mut best_depth = 0;
    let mut total_nodes = 0u64;

    let init_eval = searcher.corrected_eval(&st);
    let mut prev_score = init_eval;
    let mut stable_iterations = 0u32;
    let mut previous_iteration_seconds = 0.0;
    let mut previous_completed_elapsed = 0.0;

    for depth in 1..=limits.depth {
        if searcher.time_up(start, limits.hard_time) {
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
        let mut asp_best_nodes = 0u64;

        'asp: loop {
            let mut sorted = my_moves.clone();
            if asp_best != my_moves[0] {
                if let Some(pos) = sorted.iter().position(|&m| m == asp_best) {
                    sorted.swap(0, pos);
                }
            }
            let repetition_tie_scope = root_repetition_tie_scope(&st);

            let mut cur_best = sorted[0];
            let mut cur_score = -INF;
            let mut cur_best_nodes = 0u64;
            let mut cur_best_repeats = false;
            let mut loop_alpha = alpha;

            for &mv in &sorted {
                if searcher.time_up(start, limits.hard_time) {
                    break;
                }
                let mut s = st;
                searcher.enter_root_path(mv);
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
                let root_ext = (job.root_depth_extension)(&st, mv)
                    + root_lmr_exemption_depth_bonus(&job.root_lmr_exemptions, mv);
                let move_nodes_before = nd;

                let score = if cur_score == -INF {
                    -searcher.negamax(
                        &mut s,
                        depth - 1 + root_ext,
                        1,
                        -beta,
                        -loop_alpha,
                        true,
                        start,
                        limits.hard_time,
                        &mut nd,
                    )
                } else {
                    let sc = -searcher.negamax(
                        &mut s,
                        depth - 1 + root_ext,
                        1,
                        -loop_alpha - 1,
                        -loop_alpha,
                        true,
                        start,
                        limits.hard_time,
                        &mut nd,
                    );
                    if sc > loop_alpha && sc < beta {
                        -searcher.negamax(
                            &mut s,
                            depth - 1 + root_ext,
                            1,
                            -beta,
                            -loop_alpha,
                            true,
                            start,
                            limits.hard_time,
                            &mut nd,
                        )
                    } else {
                        sc
                    }
                };
                let move_nodes = nd.saturating_sub(move_nodes_before);
                let root_repeats = if repetition_tie_scope
                    && (score > cur_score || (score == cur_score && cur_best_repeats))
                {
                    searcher.current_position_repeats(usize::from(s.halfmove_clock))
                } else {
                    false
                };

                searcher.rep_stack.pop();
                searcher.rep_stack_len -= 1;
                searcher.leave_root_path();

                if stopped.load(Ordering::Relaxed) {
                    break;
                }
                if score > cur_score
                    || (score == cur_score
                        && prefer_non_repeating_root_on_tie(score, cur_best_repeats, root_repeats))
                {
                    cur_score = score;
                    cur_best = mv;
                    cur_best_nodes = move_nodes;
                    cur_best_repeats = root_repeats;
                }
                if score > loop_alpha {
                    loop_alpha = score;
                }
                if loop_alpha >= beta {
                    break;
                }
            }

            if stopped.load(Ordering::Relaxed)
                || (!searcher.pondering.load(Ordering::Relaxed)
                    && start.elapsed().as_secs_f64() > limits.hard_time)
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
            asp_best_nodes = cur_best_nodes;
            break;
        }

        total_nodes += nd;
        job.global_nodes.fetch_add(nd, Ordering::Relaxed);
        if stopped.load(Ordering::Relaxed) {
            break;
        }
        let elapsed = start.elapsed().as_secs_f64();

        if elapsed <= limits.hard_time || searcher.pondering.load(Ordering::Relaxed) {
            let score_change_cp = asp_score.saturating_sub(prev_score).abs();
            if best_depth == 0 || asp_best != best_move {
                stable_iterations = 0;
            } else {
                stable_iterations = stable_iterations.saturating_add(1);
            }
            let iteration_seconds = (elapsed - previous_completed_elapsed).max(0.0);
            job.worker_best_moves[thread_id].store(u64::from(asp_best), Ordering::Relaxed);
            job.worker_depths[thread_id].store(depth, Ordering::Release);
            let agreement = lazy_smp_worker_agreement(job, thread_id, asp_best, depth);
            let timing = IterationTiming {
                elapsed_seconds: elapsed,
                iteration_seconds,
                previous_iteration_seconds,
                score_change_cp,
                stable_iterations,
                best_move_effort: asp_best_nodes as f64 / nd.max(1) as f64,
                worker_disagreement: agreement.disagreement,
            };
            let time_decision = iteration_time_decision(
                limits.soft_time,
                limits.hard_time,
                job.root_moves.len(),
                timing,
            );
            let prev = job.global_best_depth.fetch_max(depth, Ordering::SeqCst);
            if prev < depth {
                let global_nodes = job.global_nodes.load(Ordering::Relaxed);
                print_lazy_smp_info(job, asp_best, asp_score, depth, global_nodes, elapsed);
            }
            best_move = asp_best;
            best_score = asp_score;
            best_depth = depth;
            prev_score = best_score;
            previous_iteration_seconds = iteration_seconds;
            previous_completed_elapsed = elapsed;
            if thread_id == 0 {
                searcher.shared_tt.store_with_pv(
                    st.hash,
                    depth,
                    score_to_tt(best_score, 0),
                    TT_EXACT,
                    Some(best_move),
                    true,
                );
            }
            searcher.update_correction_history(&st, best_score, best_depth);
            let verification_thread = job.verification_move.map(|_| 1);
            if !searcher.pondering.load(Ordering::Relaxed)
                && lazy_smp_worker_can_coordinate_stop(
                    thread_id,
                    verification_thread,
                    limits.soft_time,
                    timing,
                    agreement,
                    time_decision.stop,
                )
            {
                stopped.store(true, Ordering::SeqCst);
                break;
            }
        } else {
            break;
        }
    }

    let result = ThreadResult {
        thread_id,
        best_move,
        score: best_score,
        depth: best_depth,
        nodes: total_nodes,
        learning: (thread_id == 0).then(|| Box::new(searcher.export_learning())),
    };
    searcher.clear_node_limit();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::encode_move;
    use crate::engine::Engine;

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

    fn qualifying_singular_evidence(mv: Move) -> SingularEvidence {
        SingularEvidence {
            enabled: true,
            ply: 1,
            excluded_move: None,
            in_check: false,
            node_pv: true,
            node_beta: 100,
            actual_depth: SINGULAR_MIN_DEPTH,
            halfmove_clock: 0,
            repetitions: 1,
            repeated_after_root: false,
            shuffling: false,
            path_extensions: 0,
            allow_lower_bound: false,
            tt_move: Some(mv),
            tt_score: Some(300),
            tt_depth: SINGULAR_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN,
            tt_flag: Some(TT_EXACT),
            tt_pv: true,
            tt_age: 0,
            tt_move_is_legal: true,
        }
    }

    fn qualifying_probcut_candidate() -> ProbCutEligibility {
        probcut_candidate(
            true,
            false,
            1,
            false,
            false,
            None,
            PROBCUT_MIN_DEPTH,
            0,
            0,
            None,
            -1,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_excluding_move(
        searcher: &mut Searcher,
        st: &mut BoardState,
        excluded_move: Move,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        start: Instant,
        tl: f64,
        nodes: &mut u64,
    ) -> i32 {
        let previous = searcher.excluded_moves[ply].replace(excluded_move);
        let previous_restricted = searcher.set_restricted_verification(true);
        let score = searcher.negamax(st, depth, ply, alpha, beta, false, start, tl, nodes);
        searcher.set_restricted_verification(previous_restricted);
        searcher.excluded_moves[ply] = previous;
        score
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
    fn qsearch_checkmate_score_uses_the_actual_ply() {
        let mut st = state_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        let mut nodes = 0u64;
        let ply = 17;

        let score = searcher.qsearch(
            &mut st,
            -INF,
            INF,
            -3,
            Instant::now(),
            10.0,
            &mut nodes,
            ply,
        );

        assert_eq!(score, -MATE + ply as i32);
    }

    #[test]
    fn restricted_search_ignores_unrestricted_tt_cutoffs() {
        let st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
        let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
        assert_eq!(
            legal_moves.len(),
            1,
            "test position must have one legal move"
        );
        let excluded_move = legal_moves[0];
        let ply = 1;

        for flag in [TT_EXACT, TT_BETA] {
            let mut position = st;
            let stopped = Arc::new(AtomicBool::new(false));
            let shared_tt = Arc::new(SharedTT::new(1));
            let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
            searcher.nnue_net = None;
            shared_tt.store(
                position.hash,
                12,
                score_to_tt(900, ply),
                flag,
                Some(excluded_move),
            );
            let mut nodes = 0;

            let score = negamax_excluding_move(
                &mut searcher,
                &mut position,
                excluded_move,
                4,
                ply,
                -200,
                -199,
                Instant::now(),
                10.0,
                &mut nodes,
            );

            assert_eq!(
                score, -200,
                "restricted search used an unrestricted TT flag {flag}"
            );
        }
    }

    #[test]
    fn restricted_search_with_no_alternative_fails_low_without_storing_tt() {
        let mut st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
        let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
        assert_eq!(
            legal_moves.len(),
            1,
            "test position must have one legal move"
        );
        let excluded_move = legal_moves[0];
        let key = st.hash;
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
        searcher.nnue_net = None;
        let mut nodes = 0;

        let score = negamax_excluding_move(
            &mut searcher,
            &mut st,
            excluded_move,
            4,
            1,
            -300,
            -299,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(score, -300);
        assert!(
            shared_tt.get_depth(key).is_none(),
            "restricted result contaminated the unrestricted TT"
        );
    }

    #[test]
    fn stopped_restricted_search_restores_the_excluded_move() {
        let mut st = state_from_fen("7k/4Q3/5K2/8/8/8/8/8 b - - 0 1");
        let excluded_move = generate_moves(&st, st.w, &st.cr, st.ep)[0];
        let stopped = Arc::new(AtomicBool::new(true));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        searcher.nnue_net = None;
        let mut nodes = 0;

        let score = negamax_excluding_move(
            &mut searcher,
            &mut st,
            excluded_move,
            4,
            1,
            -300,
            -299,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(score, 0);
        assert_eq!(searcher.excluded_moves[1], None);
    }

    #[test]
    fn restricted_search_uses_descendant_tt_without_learning_from_its_root() {
        let mut st = state_from_fen("7k/8/4Q3/5K2/8/8/8/8 b - - 0 1");
        let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
        assert_eq!(
            legal_moves.len(),
            2,
            "test position must have two legal moves"
        );
        let excluded_move = legal_moves[0];
        let allowed_move = legal_moves[1];
        let mut child = st;
        apply_move(
            &mut child,
            move_sr(allowed_move),
            move_sc(allowed_move),
            move_er(allowed_move),
            move_ec(allowed_move),
            move_promotion(allowed_move),
        );

        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
        searcher.nnue_net = None;
        shared_tt.store(child.hash, 12, score_to_tt(-5000, 2), TT_EXACT, None);
        let (from, to) = from_to_key(
            move_sr(allowed_move),
            move_sc(allowed_move),
            move_er(allowed_move),
            move_ec(allowed_move),
        );
        let piece_index = piece_to_idx(piece_type(st.mailbox[move_from(allowed_move)]));
        let mut nodes = 0;

        let score = negamax_excluding_move(
            &mut searcher,
            &mut st,
            excluded_move,
            4,
            1,
            9,
            10,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(score, 5000, "restricted descendants did not use their TT");
        assert_eq!(searcher.history[from][to], 0);
        assert_eq!(searcher.killers[1], [None; 2]);
        assert_eq!(
            searcher.counter_move[piece_index][move_to(allowed_move)],
            None
        );
        assert!(
            shared_tt.get_depth(st.hash).is_none(),
            "restricted root was stored after a descendant TT cutoff"
        );
    }

    #[test]
    fn restricted_verification_does_not_write_descendant_tt_or_learning() {
        let initial = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let legal_moves = generate_moves(&initial, initial.w, &initial.cr, initial.ep);
        let excluded_move = legal_moves[0];
        let child_hashes: Vec<_> = legal_moves[1..]
            .iter()
            .map(|&mv| {
                let mut child = initial;
                apply_move(
                    &mut child,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                );
                child.hash
            })
            .collect();

        let control_tt = Arc::new(SharedTT::new(1));
        let mut control = Searcher::new(Arc::clone(&control_tt), Arc::new(AtomicBool::new(false)));
        control.nnue_net = None;
        let mut control_position = initial;
        let mut control_nodes = 0;
        control.excluded_moves[1] = Some(excluded_move);
        control.negamax(
            &mut control_position,
            4,
            1,
            -INF,
            INF,
            false,
            Instant::now(),
            10.0,
            &mut control_nodes,
        );
        control.excluded_moves[1] = None;
        assert!(
            child_hashes
                .iter()
                .any(|&hash| control_tt.get_entry(hash).is_some()),
            "control search did not exercise a descendant TT store"
        );

        let isolated_tt = Arc::new(SharedTT::new(1));
        let mut isolated =
            Searcher::new(Arc::clone(&isolated_tt), Arc::new(AtomicBool::new(false)));
        isolated.nnue_net = None;
        let mut isolated_position = initial;
        let mut isolated_nodes = 0;
        negamax_excluding_move(
            &mut isolated,
            &mut isolated_position,
            excluded_move,
            4,
            1,
            -INF,
            INF,
            Instant::now(),
            10.0,
            &mut isolated_nodes,
        );

        assert!(
            child_hashes
                .iter()
                .all(|&hash| isolated_tt.get_entry(hash).is_none()),
            "restricted verification polluted a descendant TT entry"
        );
        assert!(isolated.history.iter().flatten().all(|&value| value == 0));
        assert!(isolated.killers.iter().flatten().all(Option::is_none));
        assert!(isolated.counter_move.iter().flatten().all(Option::is_none));
    }

    #[test]
    fn excluded_move_state_is_worker_local_and_cleared_before_search() {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut first = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
        let second = Searcher::new(shared_tt, stopped);
        let excluded_move = encode_move(0, 0, 0, 1, 0);

        first.excluded_moves[3] = Some(excluded_move);

        assert_eq!(second.excluded_moves[3], None);
        first.prepare_for_search();
        assert_eq!(first.excluded_moves[3], None);
    }

    #[test]
    fn reversible_shuffle_requires_both_sides_to_retrace_their_moves() {
        let mut path = [None; MAX_PLY];
        path[0] = Some(encode_move(7, 6, 5, 5, 0));
        path[1] = Some(encode_move(0, 6, 2, 5, 0));
        path[2] = Some(encode_move(5, 5, 7, 6, 0));
        path[3] = Some(encode_move(2, 5, 0, 6, 0));

        assert!(reversible_shuffle(&path, 4, 4));
        assert!(!reversible_shuffle(&path, 4, 3));

        path[3] = Some(encode_move(2, 5, 4, 4, 0));
        assert!(!reversible_shuffle(&path, 4, 4));
    }

    #[test]
    fn singular_path_budget_counts_only_positive_extensions() {
        assert_eq!(next_singular_extension_count(2, 1), 3);
        assert_eq!(next_singular_extension_count(2, 0), 2);
        assert_eq!(next_singular_extension_count(2, -2), 2);
        assert_eq!(next_singular_extension_count(u8::MAX, 1), u8::MAX);
    }

    #[test]
    fn singular_outcome_keeps_adjustments_and_cutoffs_distinct() {
        assert_eq!(
            singular_search_outcome(19, 20, true, None, -1),
            SingularSearchOutcome::Continue(1)
        );
        assert_eq!(
            singular_search_outcome(19, 20, false, None, 0),
            SingularSearchOutcome::Continue(0)
        );
        assert_eq!(
            singular_search_outcome(30, 20, false, Some(25), -1),
            SingularSearchOutcome::Cutoff(25)
        );
        assert_eq!(
            singular_search_outcome(24, 20, false, None, -1),
            SingularSearchOutcome::Continue(-1)
        );
        assert_eq!(combine_move_extensions(0, -2), -2);
        assert_eq!(combine_move_extensions(1, -2), 1);
    }

    #[test]
    fn singular_candidate_requires_deep_reliable_safe_tt_evidence() {
        let mv = encode_move(0, 0, 0, 1, 0);
        let evidence = qualifying_singular_evidence(mv);
        let SingularEligibility::Eligible(candidate) = singular_candidate(evidence) else {
            panic!("qualifying TT evidence was rejected");
        };
        assert_eq!(candidate.mv, mv);
        assert_eq!(candidate.beta, 300 - singular_margin(evidence));
        assert_eq!(candidate.depth, (SINGULAR_MIN_DEPTH - 1) / 2);
        assert!(candidate.positive_extension);
        assert_eq!(
            candidate.max_extension,
            i32::from(singular_path_budget(evidence.actual_depth))
        );

        let mut lower_bound = evidence;
        lower_bound.actual_depth = SINGULAR_POLICY_MIN_DEPTH;
        lower_bound.tt_depth = SINGULAR_POLICY_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN;
        lower_bound.node_pv = false;
        lower_bound.node_beta = 200;
        lower_bound.tt_flag = Some(TT_BETA);
        lower_bound.tt_pv = false;
        let mut allowed_lower_bound = lower_bound;
        allowed_lower_bound.allow_lower_bound = true;
        let SingularEligibility::Eligible(lower_candidate) =
            singular_candidate(allowed_lower_bound)
        else {
            panic!("enabled lower-bound evidence was rejected");
        };
        assert!(!lower_candidate.positive_extension);
        assert_eq!(lower_candidate.beta, allowed_lower_bound.node_beta);
        assert_eq!(lower_candidate.max_extension, 0);

        let no_candidate_cases = [
            SingularEvidence {
                actual_depth: SINGULAR_MIN_DEPTH - 1,
                ..evidence
            },
            SingularEvidence {
                tt_depth: evidence.tt_depth - 1,
                ..evidence
            },
            SingularEvidence {
                tt_flag: Some(TT_ALPHA),
                ..evidence
            },
            SingularEvidence {
                tt_pv: false,
                ..evidence
            },
            SingularEvidence {
                tt_age: SINGULAR_MAX_TT_AGE + 1,
                ..evidence
            },
            SingularEvidence {
                tt_move: None,
                ..evidence
            },
            SingularEvidence {
                allow_lower_bound: false,
                ..lower_bound
            },
            SingularEvidence {
                actual_depth: SINGULAR_POLICY_MIN_DEPTH - 1,
                tt_depth: SINGULAR_POLICY_MIN_DEPTH - 1,
                ..allowed_lower_bound
            },
            SingularEvidence {
                tt_depth: SINGULAR_POLICY_MIN_DEPTH - SINGULAR_TT_DEPTH_MARGIN - 1,
                ..allowed_lower_bound
            },
            SingularEvidence {
                node_pv: true,
                ..allowed_lower_bound
            },
            SingularEvidence {
                tt_score: Some(allowed_lower_bound.node_beta - 1),
                ..allowed_lower_bound
            },
            SingularEvidence {
                node_beta: MATE / 2,
                tt_score: Some(MATE / 2),
                ..allowed_lower_bound
            },
        ];
        assert!(no_candidate_cases
            .into_iter()
            .all(|case| singular_candidate(case) == SingularEligibility::NoCandidate));

        let safety_cases = [
            SingularEvidence { ply: 0, ..evidence },
            SingularEvidence {
                excluded_move: Some(mv),
                ..evidence
            },
            SingularEvidence {
                tt_move_is_legal: false,
                ..evidence
            },
            SingularEvidence {
                in_check: true,
                ..evidence
            },
            SingularEvidence {
                repetitions: 2,
                repeated_after_root: true,
                ..evidence
            },
            SingularEvidence {
                halfmove_clock: SINGULAR_MAX_HALF_MOVE_CLOCK,
                ..evidence
            },
            SingularEvidence {
                tt_score: Some(MATE / 2),
                ..evidence
            },
            SingularEvidence {
                shuffling: true,
                ..evidence
            },
            SingularEvidence {
                path_extensions: singular_path_budget(evidence.actual_depth),
                ..evidence
            },
        ];
        assert!(safety_cases
            .into_iter()
            .all(|case| singular_candidate(case) == SingularEligibility::SafetyRejected));
    }

    #[test]
    fn singular_margin_rejects_a_competitive_alternative() {
        let mut st = state_from_fen("7k/8/4Q3/5K2/8/8/8/8 b - - 0 1");
        let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
        assert_eq!(legal_moves.len(), 2);
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        searcher.nnue_net = None;
        let singular_beta = -6_000 - singular_margin(qualifying_singular_evidence(legal_moves[0]));
        let mut nodes = 0;

        let alternative_score = negamax_excluding_move(
            &mut searcher,
            &mut st,
            legal_moves[0],
            3,
            1,
            singular_beta - 1,
            singular_beta,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert!(alternative_score >= singular_beta);
    }

    #[test]
    fn singular_multi_ply_extensions_require_progressively_larger_gaps() {
        let mut evidence = qualifying_singular_evidence(encode_move(0, 0, 0, 1, 0));
        evidence.actual_depth = SINGULAR_TRIPLE_MIN_DEPTH;
        evidence.tt_depth = SINGULAR_TRIPLE_MIN_DEPTH;
        let SingularEligibility::Eligible(candidate) = singular_candidate(evidence) else {
            panic!("qualifying TT evidence was rejected");
        };
        assert_eq!(candidate.max_extension, 3);

        assert_eq!(
            singular_extension_from_scores(candidate, candidate.beta - 1, None, None),
            1
        );
        assert_eq!(
            singular_extension_from_scores(
                candidate,
                candidate.beta - 1,
                Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP),
                None,
            ),
            1
        );
        assert_eq!(
            singular_extension_from_scores(
                candidate,
                candidate.beta - 1,
                Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP - 1),
                None,
            ),
            2
        );
        assert_eq!(
            singular_extension_from_scores(
                candidate,
                candidate.beta - 1,
                Some(candidate.score - SINGULAR_DOUBLE_MARGIN_CP - 1),
                Some(candidate.score - SINGULAR_TRIPLE_MARGIN_CP - 1),
            ),
            3
        );
        assert_eq!(
            singular_extension_from_scores(candidate, candidate.beta, None, None),
            0
        );
    }

    #[cfg(feature = "search-debug")]
    #[test]
    fn singular_extensions_require_explicit_experimental_opt_in() {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);

        searcher.debug.enable_singular_extensions = false;
        assert!(!searcher.singular_extensions_enabled());

        searcher.debug.enable_singular_extensions = true;
        assert!(searcher.singular_extensions_enabled());

        searcher.debug.enable_singular_multi_extensions = false;
        assert!(!searcher.singular_multi_extensions_enabled());
        searcher.debug.enable_singular_multi_extensions = true;
        assert!(searcher.singular_multi_extensions_enabled());

        searcher.debug.enable_singular_multicut = false;
        searcher.debug.enable_singular_negative_extensions = false;
        assert!(!searcher.singular_multicut_enabled());
        assert!(!searcher.singular_negative_extensions_enabled());

        searcher.debug.enable_singular_multicut = true;
        assert!(searcher.singular_multicut_enabled());
        assert!(!searcher.singular_negative_extensions_enabled());

        searcher.debug.enable_singular_multicut = false;
        searcher.debug.enable_singular_negative_extensions = true;
        assert!(!searcher.singular_multicut_enabled());
        assert!(searcher.singular_negative_extensions_enabled());
    }

    #[cfg(feature = "search-debug")]
    #[test]
    fn singular_search_extends_a_synthetic_only_move_tt_result() {
        let mut st = state_from_fen("7k/8/5K2/5Q2/8/8/8/8 b - - 0 1");
        let legal_moves = generate_moves(&st, st.w, &st.cr, st.ep);
        assert_eq!(legal_moves.len(), 1, "position must have one legal move");
        let tt_move = legal_moves[0];
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
        searcher.nnue_net = None;
        searcher.debug.enable_singular_extensions = true;
        shared_tt.store_with_pv(
            st.hash,
            SINGULAR_MIN_DEPTH,
            score_to_tt(0, 1),
            TT_EXACT,
            Some(tt_move),
            true,
        );
        let mut nodes = 0;

        searcher.negamax(
            &mut st,
            SINGULAR_MIN_DEPTH,
            1,
            -INF,
            INF,
            true,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        let stats = searcher.debug_stats();
        assert_eq!(stats.singular_candidates, 1);
        assert_eq!(stats.singular_verifications, 1);
        assert_eq!(stats.singular_extensions, 1);
        assert_eq!(stats.singular_alternative_rejections, 0);
        assert_eq!(searcher.excluded_moves[1], None);
    }

    #[test]
    fn probcut_candidate_requires_a_safe_non_pv_node() {
        assert_eq!(
            qualifying_probcut_candidate(),
            ProbCutEligibility::Eligible(ProbCutCandidate {
                beta: PROBCUT_MARGIN_CP,
                child_depth: PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION,
                store_depth: PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
            })
        );
        assert_eq!(
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH - 1,
                0,
                0,
                None,
                -1,
                None,
            ),
            ProbCutEligibility::NoCandidate
        );

        let safety_cases = [
            probcut_candidate(
                true,
                false,
                0,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                1,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                false,
                1,
                true,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                false,
                1,
                false,
                true,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                Some(encode_move(0, 0, 0, 1, 0)),
                PROBCUT_MIN_DEPTH,
                0,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                true,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                None,
                -1,
                None,
            ),
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                MATE / 2,
                MATE / 2,
                None,
                -1,
                None,
            ),
        ];
        assert!(safety_cases
            .into_iter()
            .all(|case| case == ProbCutEligibility::SafetyRejected));
    }

    #[test]
    fn probcut_respects_tt_evidence_but_not_a_lower_bound() {
        for flag in [TT_EXACT, TT_ALPHA] {
            assert_eq!(
                probcut_candidate(
                    true,
                    false,
                    1,
                    false,
                    false,
                    None,
                    PROBCUT_MIN_DEPTH,
                    0,
                    0,
                    Some(0),
                    PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
                    Some(flag),
                ),
                ProbCutEligibility::TtRejected
            );
        }
        assert!(matches!(
            probcut_candidate(
                true,
                false,
                1,
                false,
                false,
                None,
                PROBCUT_MIN_DEPTH,
                0,
                0,
                Some(0),
                PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
                Some(TT_BETA),
            ),
            ProbCutEligibility::Eligible(_)
        ));
    }

    #[test]
    fn probcut_requires_both_verification_stages_to_pass() {
        let beta = PROBCUT_MARGIN_CP;
        assert_eq!(
            probcut_verdict(beta, beta - 1, Some(beta + 100)),
            ProbCutVerdict::QuiescenceRejected
        );
        assert_eq!(
            probcut_verdict(beta, beta, None),
            ProbCutVerdict::FullSearchRejected
        );
        assert_eq!(
            probcut_verdict(beta, beta, Some(beta - 1)),
            ProbCutVerdict::FullSearchRejected
        );
        assert_eq!(
            probcut_verdict(beta, beta, Some(beta)),
            ProbCutVerdict::Cutoff
        );
    }

    #[cfg(feature = "search-debug")]
    #[test]
    fn probcut_stores_only_the_reduced_verified_depth() {
        let mut st = state_from_fen("q6k/8/8/8/8/8/8/Q5K1 w - - 0 1");
        let tactical_move = legal_move(&st, "a1a8");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(Arc::clone(&shared_tt), stopped);
        searcher.nnue_net = None;
        let key = st.hash;
        let mut nodes = 0;

        let score = searcher.negamax(
            &mut st,
            PROBCUT_MIN_DEPTH,
            1,
            -1,
            0,
            true,
            Instant::now(),
            10.0,
            &mut nodes,
        );

        assert_eq!(score, 0);
        let stats = searcher.debug_stats();
        assert_eq!(stats.probcut_eligible_nodes, 1);
        assert_eq!(stats.probcut_qsearch_passes, 1);
        assert_eq!(stats.probcut_verifications, 1);
        assert_eq!(stats.probcut_cutoffs, 1);
        let (depth, tt_score, flag, best_move) = shared_tt
            .get_depth(key)
            .expect("ProbCut did not store a bound");
        assert_eq!(
            depth,
            PROBCUT_MIN_DEPTH - PROBCUT_REDUCTION + 1,
            "ProbCut stored a depth other than its reduced proof"
        );
        assert_eq!(score_from_tt(tt_score, 1), PROBCUT_MARGIN_CP);
        assert_eq!(flag, TT_BETA);
        assert_eq!(best_move, Some(tactical_move));
        assert!(!searcher.probcut_verification);
    }

    #[cfg(feature = "search-debug")]
    #[test]
    fn search_debug_stats_are_reset_between_root_moves() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        let mut nodes = 0u64;

        searcher.qsearch(
            &mut st,
            -INF,
            INF,
            QS_DEPTH,
            Instant::now(),
            10.0,
            &mut nodes,
            0,
        );

        let stats = searcher.debug_stats();
        assert!(stats.qnodes > 1);
        assert!(stats.max_ply > 0);

        searcher.reset_debug_stats();

        assert_eq!(searcher.debug_stats(), SearchDebugStats::default());
    }

    #[test]
    fn settled_lazy_smp_helper_can_coordinate_the_shared_soft_stop() {
        let timing = IterationTiming {
            elapsed_seconds: 1.1,
            iteration_seconds: 0.2,
            previous_iteration_seconds: 0.15,
            score_change_cp: 12,
            stable_iterations: 3,
            best_move_effort: 0.8,
            worker_disagreement: 0.0,
        };
        let agreement = LazySmpAgreement {
            disagreement: 0.0,
            comparable_workers: 2,
            principal_agrees: true,
        };

        assert!(lazy_smp_worker_can_coordinate_stop(
            2, None, 1.0, timing, agreement, true,
        ));
        assert!(!lazy_smp_worker_can_coordinate_stop(
            2,
            None,
            1.0,
            timing,
            LazySmpAgreement {
                disagreement: 0.0,
                comparable_workers: 0,
                principal_agrees: true,
            },
            true,
        ));
        assert!(!lazy_smp_worker_can_coordinate_stop(
            1,
            Some(1),
            1.0,
            timing,
            agreement,
            true,
        ));
        assert!(!lazy_smp_worker_can_coordinate_stop(
            2,
            None,
            1.0,
            IterationTiming {
                elapsed_seconds: 0.9,
                ..timing
            },
            agreement,
            true,
        ));
        assert!(!lazy_smp_worker_can_coordinate_stop(
            2,
            None,
            1.0,
            timing,
            LazySmpAgreement {
                disagreement: 0.5,
                comparable_workers: 2,
                principal_agrees: true,
            },
            true,
        ));
        assert!(!lazy_smp_worker_can_coordinate_stop(
            2,
            None,
            1.0,
            timing,
            LazySmpAgreement {
                disagreement: 0.0,
                comparable_workers: 2,
                principal_agrees: false,
            },
            true,
        ));
    }

    #[test]
    fn lazy_smp_pool_reuses_workers_with_a_fresh_stop_token() {
        let pool = LazySmpPool::new();
        let st = state_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
        let root_moves = generate_moves(&st, st.w, &st.cr, st.ep);

        let first_stopped = Arc::new(AtomicBool::new(false));
        let first_tt = Arc::new(SharedTT::new(128));
        let mut first_root = Searcher::new(Arc::clone(&first_tt), Arc::clone(&first_stopped));
        let (first_move, _, first_depth, _) = lazy_smp_search(
            &pool,
            first_tt,
            &st,
            &root_moves,
            |_, _| 0,
            &[],
            LazySmpSearchLimits {
                soft_time: 0.0,
                hard_time: 10.0,
                depth: 4,
                node_limit: None,
                start: Instant::now(),
            },
            2,
            &mut first_root,
        );

        assert!(root_moves.contains(&first_move));
        assert!(first_depth >= 1);
        assert!(first_stopped.load(Ordering::Relaxed));
        let first_worker_ids = pool.worker_ids();
        assert_eq!(first_worker_ids.len(), 2);

        let second_stopped = Arc::new(AtomicBool::new(false));
        let second_tt = Arc::new(SharedTT::new(128));
        let mut second_root = Searcher::new(Arc::clone(&second_tt), Arc::clone(&second_stopped));
        let (second_move, _, second_depth, second_nodes) = lazy_smp_search(
            &pool,
            second_tt,
            &st,
            &root_moves,
            |_, _| 0,
            &[],
            LazySmpSearchLimits {
                soft_time: 10.0,
                hard_time: 10.0,
                depth: 1,
                node_limit: None,
                start: Instant::now(),
            },
            2,
            &mut second_root,
        );

        assert!(root_moves.contains(&second_move));
        assert_eq!(second_depth, 1);
        assert!(second_nodes > 0);
        assert!(!second_stopped.load(Ordering::Relaxed));
        assert_eq!(pool.worker_ids(), first_worker_ids);
    }

    #[test]
    fn lazy_smp_helpers_prioritize_distinct_root_lanes() {
        let original = vec![
            encode_move(0, 0, 0, 0, 0),
            encode_move(0, 1, 0, 1, 0),
            encode_move(0, 2, 0, 2, 0),
            encode_move(0, 3, 0, 3, 0),
        ];
        let thread_zero = lazy_smp_root_moves(&original, 0, 4);
        let thread_one = lazy_smp_root_moves(&original, 1, 4);
        let thread_two = lazy_smp_root_moves(&original, 2, 4);
        let thread_three = lazy_smp_root_moves(&original, 3, 4);

        assert_eq!(thread_zero, original);
        assert_eq!(thread_one[0], original[1]);
        assert_eq!(thread_two[0], original[2]);
        assert_eq!(thread_three[0], original[3]);

        let mut sorted_original = original.clone();
        sorted_original.sort_unstable();
        for mut helper_moves in [thread_one, thread_two, thread_three] {
            helper_moves.sort_unstable();
            assert_eq!(helper_moves, sorted_original);
        }
    }

    #[test]
    fn lazy_smp_many_helpers_keep_rotated_root_order() {
        let original = (0..16)
            .map(|square| {
                let row = square / 8;
                let col = square % 8;
                encode_move(row, col, row, col, 0)
            })
            .collect::<Vec<_>>();
        let mut expected = original.clone();
        expected.rotate_left(1);

        assert_eq!(lazy_smp_root_moves(&original, 1, 12), expected);
    }

    // These positions test helper-lane assignment and disagreement accounting directly;
    // a final TSV move cannot reveal which worker searched a move or how its vote counted.
    #[test]
    fn lazy_smp_assigns_a_helper_to_verify_game_ffzk_y782_recapture() {
        let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
        let bxe4 = legal_move(&st, "g2e4");
        let qb2 = legal_move(&st, "c1b2");
        let re1 = legal_move(&st, "d1e1");
        let root_moves = [bxe4, qb2, re1];

        assert_eq!(
            lazy_smp_worker_root_moves(&st, &root_moves, 1, 12),
            vec![bxe4]
        );
        assert_eq!(
            lazy_smp_worker_root_moves(&st, &root_moves, 0, 12),
            root_moves
        );
    }

    #[test]
    fn lazy_smp_tactical_verifier_does_not_inflate_worker_disagreement() {
        let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
        let bxe4 = legal_move(&st, "g2e4");
        let qe3 = legal_move(&st, "c1e3");
        let root_moves = Arc::new(vec![bxe4, qe3]);
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let verification_tt = Arc::new(SharedTT::new(1));
        let root_searcher = Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped));
        let job = LazySmpSearchJob {
            shared_tt,
            verification_move: Some(bxe4),
            verification_tt: Some(verification_tt),
            stopped,
            st,
            root_moves,
            num_threads: 3,
            root_depth_extension: |_, _| 0,
            root_lmr_exemptions: Arc::new(Vec::new()),
            limits: LazySmpSearchLimits {
                soft_time: 1.0,
                hard_time: 2.0,
                depth: 20,
                node_limit: None,
                start: Instant::now(),
            },
            root_context: Arc::new(LazySmpRootContext::from_searcher(&root_searcher)),
            start: Instant::now(),
            global_best_depth: Arc::new(AtomicI32::new(0)),
            global_nodes: Arc::new(AtomicU64::new(0)),
            node_limit_counter: None,
            worker_best_moves: (0..3).map(|_| AtomicU64::new(0)).collect(),
            worker_depths: (0..3).map(|_| AtomicI32::new(0)).collect(),
        };
        job.worker_best_moves[1].store(u64::from(bxe4), Ordering::Relaxed);
        job.worker_depths[1].store(22, Ordering::Release);
        job.worker_best_moves[2].store(u64::from(qe3), Ordering::Relaxed);
        job.worker_depths[2].store(17, Ordering::Release);

        assert_eq!(lazy_smp_worker_disagreement(&job, 0, qe3, 17), 0.0);
        assert!(!Arc::ptr_eq(
            &job.shared_tt,
            job.verification_tt.as_ref().unwrap()
        ));
    }

    fn completed_thread(thread_id: usize, best_move: Move, score: i32, depth: i32) -> ThreadResult {
        ThreadResult {
            thread_id,
            best_move,
            score,
            depth,
            nodes: 1,
            learning: None,
        }
    }

    // The following recorded positions drive synthetic worker ballots. They verify SMP
    // result selection independently of search nondeterminism, which TSV move cases cannot
    // express because they only observe a completed single-thread root search.
    #[test]
    fn lazy_smp_does_not_let_deepest_outlier_repeat_draw_game_kh7() {
        // https://lichess.org/xMs5Nkx3 before 49...Kh7:
        // 2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49
        // 49...Bg7 held the evaluation near equality; the played 49...Kh7
        // conceded a substantial white advantage. A one-ply-deeper dissenting
        // Lazy SMP worker must not overrule two current-depth votes for Bg7.
        let st = state_from_fen("2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49");
        let bg7 = legal_move(&st, "h6g7");
        let kh7 = legal_move(&st, "g8h7");
        let results = [
            completed_thread(0, bg7, -72, 14),
            completed_thread(1, bg7, -68, 14),
            completed_thread(2, kh7, -61, 15),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[bg7, kh7])
                .unwrap()
                .best_move,
            bg7
        );
    }

    #[test]
    fn lazy_smp_does_not_let_deepest_outlier_repeat_loss_game_kf7() {
        // https://lichess.org/VIPYcetR before 22...Kf7:
        // 1r1qk3/Q1p5/5n2/3n1pp1/2BP3r/2P1P3/P2B3P/R3K2R b KQ - 0 22
        // 22...Ne7 was the resilient move; 22...Kf7 was the first major error.
        let st = state_from_fen("1r1qk3/Q1p5/5n2/3n1pp1/2BP3r/2P1P3/P2B3P/R3K2R b KQ - 0 22");
        let ne7 = legal_move(&st, "d5e7");
        let kf7 = legal_move(&st, "e8f7");
        let results = [
            completed_thread(0, ne7, -31, 12),
            completed_thread(1, ne7, -28, 12),
            completed_thread(2, kf7, -20, 13),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[ne7, kf7])
                .unwrap()
                .best_move,
            ne7
        );
    }

    #[test]
    fn lazy_smp_does_not_let_deepest_outlier_repeat_loss_game_g4() {
        // https://lichess.org/VIPYcetR before 28...g4:
        // 1r1q4/Q1p5/1n2B1k1/5ppr/3Pn3/2P1P1R1/P2B3P/2K2R2 b - - 12 28
        // 28...Kh6 resisted; 28...g4 allowed the forcing Bxf5+/Rxg4+
        // sequence. Prefer the supported near-deep result to a deepest outlier.
        let st = state_from_fen("1r1q4/Q1p5/1n2B1k1/5ppr/3Pn3/2P1P1R1/P2B3P/2K2R2 b - - 12 28");
        let kh6 = legal_move(&st, "g6h6");
        let g4 = legal_move(&st, "g5g4");
        let results = [
            completed_thread(0, kh6, -205, 13),
            completed_thread(1, kh6, -198, 13),
            completed_thread(2, g4, -187, 14),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[kh6, g4])
                .unwrap()
                .best_move,
            kh6
        );
    }

    #[test]
    fn lazy_smp_keeps_principal_recapture_from_game_ffzk_y782() {
        // https://lichess.org/ffzkY782 before 31.Re1:
        // 4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31
        // The principal worker found 31.Bxe4, which Stockfish evaluates as
        // equal, but helper consensus replaced it with a losing quiet move.
        let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
        let bxe4 = legal_move(&st, "g2e4");
        let qe3 = legal_move(&st, "c1e3");
        let qb2 = legal_move(&st, "c1b2");
        let results = [
            completed_thread(8, qe3, 0, 19),
            completed_thread(7, bxe4, -91, 18),
            completed_thread(3, qb2, -56, 20),
            completed_thread(11, qe3, 0, 18),
            completed_thread(2, qb2, -56, 20),
            completed_thread(5, qe3, 0, 19),
            completed_thread(4, bxe4, -72, 17),
            completed_thread(6, qe3, 0, 18),
            completed_thread(1, qe3, 0, 18),
            completed_thread(9, qe3, 0, 18),
            completed_thread(10, qe3, 0, 19),
            completed_thread(0, bxe4, -126, 19),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[bxe4, qe3, qb2])
                .unwrap()
                .best_move,
            bxe4
        );
    }

    #[test]
    fn lazy_smp_keeps_deeper_root_recapture_from_game_ffzk_y782() {
        // A dedicated helper can search Bxe4 more deeply than workers that
        // compare every root move. Keep its better score over the principal
        // worker's losing quiet alternative.
        let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
        let bxe4 = legal_move(&st, "g2e4");
        let qb2 = legal_move(&st, "c1b2");
        let re1 = legal_move(&st, "d1e1");
        let results = [
            completed_thread(0, qb2, -128, 17),
            completed_thread(1, bxe4, -65, 18),
            completed_thread(2, re1, -124, 17),
        ];

        assert!(see(&st.bb, move_from(bxe4), move_to(bxe4)) >= -25);
        assert_eq!(
            select_lazy_smp_result(&results, &st, &[bxe4, qb2, re1])
                .unwrap()
                .best_move,
            bxe4
        );
    }

    #[test]
    fn lazy_smp_rejects_a_worse_tactical_verification() {
        let st = state_from_fen("4r1k1/q3nppp/2p1p2P/1p2B3/pP1rn3/3N2P1/P4PB1/2QR2K1 w - - 0 31");
        let bxe4 = legal_move(&st, "g2e4");
        let qe3 = legal_move(&st, "c1e3");
        let results = [
            completed_thread(0, qe3, -20, 17),
            completed_thread(1, bxe4, -125, 18),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[bxe4, qe3])
                .unwrap()
                .best_move,
            qe3
        );
    }

    #[test]
    fn lazy_smp_uses_consensus_when_principal_has_no_result() {
        let st = state_from_fen("2r3k1/5q2/2p3pb/4Qp1p/pB1P3P/P1P5/4RKP1/8 b - - 19 49");
        let bg7 = legal_move(&st, "h6g7");
        let kh7 = legal_move(&st, "g8h7");
        let results = [
            completed_thread(1, bg7, -72, 14),
            completed_thread(2, bg7, -68, 14),
            completed_thread(3, kh7, -61, 15),
        ];

        assert_eq!(
            select_lazy_smp_result(&results, &st, &[bg7, kh7])
                .unwrap()
                .best_move,
            bg7
        );
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
    fn draw_status_distinguishes_claimable_and_automatic_thresholds() {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        let mut st = state_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1");
        searcher.rep_stack = vec![st.hash];
        searcher.rep_stack_len = 1;

        assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::None);
        st.halfmove_clock = 100;
        assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Claimable);
        st.halfmove_clock = 150;
        assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Automatic);

        st.halfmove_clock = 8;
        searcher.rep_stack = vec![7, 1, 7, 2, 7];
        searcher.rep_stack_len = searcher.rep_stack.len();
        assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Claimable);

        searcher.rep_stack = vec![7, 1, 7, 2, 7, 3, 7, 4, 7];
        searcher.rep_stack_len = searcher.rep_stack.len();
        assert_eq!(searcher.draw_status(&st, 1, 1), DrawStatus::Automatic);
    }

    #[test]
    fn draw_status_terminates_cycles_only_after_the_search_root() {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(1));
        let mut searcher = Searcher::new(shared_tt, stopped);
        let st = state_from_fen("7k/8/8/8/8/8/8/KQ6 w - - 8 1");

        searcher.rep_stack = vec![9, 8, 7, 6, 7];
        searcher.rep_stack_len = searcher.rep_stack.len();
        searcher.rep_root_len = 3;

        assert_eq!(
            searcher.draw_status(&st, 2, 1),
            DrawStatus::None,
            "a second occurrence of the root is not a legal threefold"
        );
        searcher.rep_root_len = 1;
        assert_eq!(
            searcher.draw_status(&st, 4, 1),
            DrawStatus::SearchCycle,
            "a second occurrence entirely inside the tree terminates the cycle"
        );
    }
}
