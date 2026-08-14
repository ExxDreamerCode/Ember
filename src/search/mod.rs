use crate::board::{
    all_occ, attacked_by, bit, has_non_pawn, is_dead_position, move_ec, move_er, move_from,
    move_promotion, move_sc, move_sr, move_to, piece_type, see, BoardState, Move, BP, EMPTY_SQ,
    INF, KING_ATTACKS, MATE, MAX_HALF_MOVE_CLOCK, MAX_PLY, NO_MOVE, QS_DEPTH, WP,
};
use crate::evaluate::{current_nnue_net, evaluate, evaluate_nnue_acc_with_backend};
use crate::movegen::{
    apply_move, apply_move_mode, generate_moves, generate_moves_into_mode,
    generate_pseudo_captures_promotions_into_mode, generate_pseudo_moves_into_mode,
    try_apply_move_mode,
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
use crate::tune::{self, TuneParam};
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
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

mod interface;
pub use self::interface::{
    active_search_backend, extract_pv_line, format_pv_line_uci, set_search_backend_override,
};
pub use crate::backend::SearchBackendKind;

#[cfg(feature = "search-debug")]
mod debug;
mod lazy_smp;
#[cfg(feature = "search-debug")]
pub use self::debug::{SearchDebug, SearchDebugStats};
#[cfg(test)]
use self::lazy_smp::{
    lazy_smp_root_moves, lazy_smp_worker_can_coordinate_stop, lazy_smp_worker_disagreement,
    lazy_smp_worker_root_moves, select_lazy_smp_result, LazySmpAgreement, LazySmpRootContext,
    LazySmpSearchJob, ThreadResult,
};
pub use self::lazy_smp::{lazy_smp_search, LazySmpPool, LazySmpSearchLimits, SearchLearning};

const LAZY_SMP_VERIFICATION_MARGIN_CP: i32 = 25;
const LAZY_SMP_VERIFICATION_TT_MB: usize = 4;
mod selectivity;
use self::selectivity::{
    combine_move_extensions, probcut_candidate, probcut_verdict, singular_candidate,
    singular_extension_from_scores, singular_search_outcome, ChildPathState, DrawStatus,
    ProbCutEligibility, ProbCutVerdict, SingularEligibility, SingularEvidence,
    SingularMoveAdjustment, SingularSearchOutcome, SINGULAR_DOUBLE_MARGIN_CP,
    SINGULAR_DOUBLE_MIN_DEPTH, SINGULAR_MAX_TT_AGE, SINGULAR_MIN_DEPTH, SINGULAR_TRIPLE_MARGIN_CP,
    SINGULAR_TRIPLE_MIN_DEPTH, SINGULAR_TT_DEPTH_MARGIN,
};
#[cfg(any(feature = "search-debug", test))]
use self::selectivity::{next_singular_extension_count, reversible_shuffle};
pub(crate) use self::selectivity::{prefer_non_repeating_root_on_tie, root_repetition_tie_scope};
#[cfg(test)]
use self::selectivity::{
    singular_margin, singular_path_budget, ProbCutCandidate, PROBCUT_MARGIN_CP, PROBCUT_MIN_DEPTH,
    PROBCUT_REDUCTION, SINGULAR_MAX_HALF_MOVE_CLOCK, SINGULAR_POLICY_MIN_DEPTH,
};

mod move_helpers;
use self::move_helpers::*;

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

mod eval;
use self::eval::{ClassicEval, NnueEval, SearchEval, ThreatNnueEval};

mod negamax;
#[cfg(test)]
use self::negamax::{lmp_king_pressure_safe, lmp_move_count};
mod qsearch;
#[cfg(test)]
use self::qsearch::{
    qsearch_check_cap_reached, qsearch_delta_prunable, qsearch_see_prunable,
    qsearch_see_threshold_cp,
};
mod state;

#[cfg(test)]
mod tests;
