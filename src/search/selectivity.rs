#[cfg(any(feature = "search-debug", test))]
use crate::board::{move_from, move_to};
use crate::board::{BoardState, Move, MATE};
use crate::tt::{TT_ALPHA, TT_BETA, TT_EXACT};

// Singular extensions remain available for controlled search experiments, but
// are not part of the production search until they pass the strength gates.
pub(super) const SINGULAR_MIN_DEPTH: i32 = 12;
pub(super) const SINGULAR_TT_DEPTH_MARGIN: i32 = 1;
pub(super) const SINGULAR_BASE_MARGIN_CP: i32 = 44;
pub(super) const SINGULAR_MARGIN_PER_DEPTH_CP: i32 = 3;
pub(super) const SINGULAR_MAX_TT_AGE: u8 = 0;
pub(super) const SINGULAR_MAX_HALF_MOVE_CLOCK: u8 = 80;
pub(super) const SINGULAR_POLICY_MIN_DEPTH: i32 = 15;
pub(super) const SINGULAR_DOUBLE_MIN_DEPTH: i32 = 16;
pub(super) const SINGULAR_TRIPLE_MIN_DEPTH: i32 = 24;
pub(super) const SINGULAR_DOUBLE_MARGIN_CP: i32 = 160;
pub(super) const SINGULAR_TRIPLE_MARGIN_CP: i32 = 240;
pub(super) const PROBCUT_MIN_DEPTH: i32 = 8;
pub(super) const PROBCUT_REDUCTION: i32 = 2;
pub(super) const PROBCUT_MARGIN_CP: i32 = 350;
pub(super) const ROOT_REPETITION_TIE_MIN_SCORE: i32 = 300;
pub(super) const ROOT_REPETITION_TIE_MIN_HALFMOVE_CLOCK: u8 = 40;
pub(super) const ROOT_REPETITION_TIE_MAX_PIECES: u32 = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DrawStatus {
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
pub(super) struct SingularCandidate {
    pub(super) mv: Move,
    pub(super) score: i32,
    pub(super) beta: i32,
    pub(super) depth: i32,
    pub(super) positive_extension: bool,
    pub(super) max_extension: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SingularMoveAdjustment {
    pub(super) mv: Move,
    pub(super) extension: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SingularSearchOutcome {
    Continue(i32),
    Cutoff(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SingularEligibility {
    NoCandidate,
    SafetyRejected,
    Eligible(SingularCandidate),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SingularEvidence {
    pub(super) enabled: bool,
    pub(super) ply: usize,
    pub(super) excluded_move: Option<Move>,
    pub(super) in_check: bool,
    pub(super) node_pv: bool,
    pub(super) node_beta: i32,
    pub(super) actual_depth: i32,
    pub(super) halfmove_clock: u8,
    pub(super) repetitions: u8,
    pub(super) repeated_after_root: bool,
    pub(super) shuffling: bool,
    pub(super) path_extensions: u8,
    pub(super) allow_lower_bound: bool,
    pub(super) tt_move: Option<Move>,
    pub(super) tt_score: Option<i32>,
    pub(super) tt_depth: i32,
    pub(super) tt_flag: Option<u8>,
    pub(super) tt_pv: bool,
    pub(super) tt_age: u8,
    pub(super) tt_move_is_legal: bool,
}

#[cfg(feature = "search-debug")]
#[derive(Clone, Copy)]
pub(super) struct ChildPathState {
    pub(super) ply: usize,
    pub(super) previous_move: Option<Move>,
    pub(super) child_ply: Option<usize>,
    pub(super) previous_extensions: u8,
}

#[cfg(not(feature = "search-debug"))]
#[derive(Clone, Copy)]
pub(super) struct ChildPathState;

#[cfg(any(feature = "search-debug", test))]
pub(super) fn next_singular_extension_count(current: u8, extension: i32) -> u8 {
    current.saturating_add(extension.max(0) as u8)
}

pub(super) fn singular_search_outcome(
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

pub(super) fn combine_move_extensions(tactical_extension: i32, singular_extension: i32) -> i32 {
    if tactical_extension > 0 {
        tactical_extension
    } else {
        singular_extension
    }
}

pub(super) fn singular_extension_from_scores(
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
pub(super) fn reversible_shuffle(
    path_moves: &[Option<Move>],
    ply: usize,
    halfmove_clock: u8,
) -> bool {
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
pub(super) struct ProbCutCandidate {
    pub(super) beta: i32,
    pub(super) child_depth: i32,
    pub(super) store_depth: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProbCutEligibility {
    NoCandidate,
    SafetyRejected,
    TtRejected,
    Eligible(ProbCutCandidate),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProbCutVerdict {
    QuiescenceRejected,
    FullSearchRejected,
    Cutoff,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn probcut_candidate(
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

pub(super) fn probcut_verdict(
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

pub(super) fn singular_margin(evidence: SingularEvidence) -> i32 {
    SINGULAR_BASE_MARGIN_CP
        + SINGULAR_MARGIN_PER_DEPTH_CP * evidence.actual_depth
        + i32::from(!evidence.tt_pv) * 16
        + i32::from(!evidence.node_pv) * 8
        + i32::from(evidence.tt_age) * 8
}

pub(super) fn singular_path_budget(depth: i32) -> u8 {
    (1 + depth.max(0) / 12).min(3) as u8
}

pub(super) fn singular_candidate(evidence: SingularEvidence) -> SingularEligibility {
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
