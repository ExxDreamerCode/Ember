use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

static TUNING_ACTIVE: AtomicBool = AtomicBool::new(false);

const PARAM_COUNT: usize = 22;

static OVERRIDES: [AtomicI64; PARAM_COUNT] = [const { AtomicI64::new(0) }; PARAM_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum TuneParam {
    ProbCutMinDepth,
    ProbCutMarginCp,
    RootRepetitionTieMinScore,
    ReverseFutilityBaseCp,
    ReverseFutilityPerDepthCp,
    ReverseFutilityMaxDepth,
    FutilityMarginPerDepthCp,
    FutilityMaxDepth,
    NullMoveMinDepth,
    NullMoveReductionBase,
    NullMoveReductionDivisor,
    NullMoveMarginDivisor,
    NullMoveMarginCap,
    NullMoveKingPressureLimit,
    NullMoveNonPawnLimit,
    SeeMarginPerDepthCp,
    HistoryPruneMarginPerDepth,
    HistoryPruneMaxDepth,
    CheckExtensionMaxDepth,
    LmpMaxDepth,
    IidMinDepth,
    LmrDivisorMillis,
}

impl TuneParam {
    pub const COUNT: usize = PARAM_COUNT;

    pub fn name(self) -> &'static str {
        match self {
            TuneParam::ProbCutMinDepth => "PROBCUT_MIN_DEPTH",
            TuneParam::ProbCutMarginCp => "PROBCUT_MARGIN_CP",
            TuneParam::RootRepetitionTieMinScore => "ROOT_REPETITION_TIE_MIN_SCORE",
            TuneParam::ReverseFutilityBaseCp => "REVERSE_FUTILITY_BASE_CP",
            TuneParam::ReverseFutilityPerDepthCp => "REVERSE_FUTILITY_PER_DEPTH_CP",
            TuneParam::ReverseFutilityMaxDepth => "REVERSE_FUTILITY_MAX_DEPTH",
            TuneParam::FutilityMarginPerDepthCp => "FUTILITY_MARGIN_PER_DEPTH_CP",
            TuneParam::FutilityMaxDepth => "FUTILITY_MAX_DEPTH",
            TuneParam::NullMoveMinDepth => "NULL_MOVE_MIN_DEPTH",
            TuneParam::NullMoveReductionBase => "NULL_MOVE_REDUCTION_BASE",
            TuneParam::NullMoveReductionDivisor => "NULL_MOVE_REDUCTION_DIVISOR",
            TuneParam::NullMoveMarginDivisor => "NULL_MOVE_MARGIN_DIVISOR",
            TuneParam::NullMoveMarginCap => "NULL_MOVE_MARGIN_CAP",
            TuneParam::NullMoveKingPressureLimit => "NULL_MOVE_KING_PRESSURE_LIMIT",
            TuneParam::NullMoveNonPawnLimit => "NULL_MOVE_NON_PAWN_LIMIT",
            TuneParam::SeeMarginPerDepthCp => "SEE_MARGIN_PER_DEPTH_CP",
            TuneParam::HistoryPruneMarginPerDepth => "HISTORY_PRUNE_MARGIN_PER_DEPTH",
            TuneParam::HistoryPruneMaxDepth => "HISTORY_PRUNE_MAX_DEPTH",
            TuneParam::CheckExtensionMaxDepth => "CHECK_EXTENSION_MAX_DEPTH",
            TuneParam::LmpMaxDepth => "LMP_MAX_DEPTH",
            TuneParam::IidMinDepth => "IID_MIN_DEPTH",
            TuneParam::LmrDivisorMillis => "LMR_DIVISOR_MILLIS",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "PROBCUT_MIN_DEPTH" | "PROBCUT-MIN-DEPTH" => Some(TuneParam::ProbCutMinDepth),
            "PROBCUT_MARGIN_CP" | "PROBCUT-MARGIN-CP" => Some(TuneParam::ProbCutMarginCp),
            "ROOT_REPETITION_TIE_MIN_SCORE" | "ROOT-REPETITION-TIE-MIN-SCORE" => {
                Some(TuneParam::RootRepetitionTieMinScore)
            }
            "REVERSE_FUTILITY_BASE_CP" | "REVERSE-FUTILITY-BASE-CP" => {
                Some(TuneParam::ReverseFutilityBaseCp)
            }
            "REVERSE_FUTILITY_PER_DEPTH_CP" | "REVERSE-FUTILITY-PER-DEPTH-CP" => {
                Some(TuneParam::ReverseFutilityPerDepthCp)
            }
            "REVERSE_FUTILITY_MAX_DEPTH" | "REVERSE-FUTILITY-MAX-DEPTH" => {
                Some(TuneParam::ReverseFutilityMaxDepth)
            }
            "FUTILITY_MARGIN_PER_DEPTH_CP" | "FUTILITY-MARGIN-PER-DEPTH-CP" => {
                Some(TuneParam::FutilityMarginPerDepthCp)
            }
            "FUTILITY_MAX_DEPTH" | "FUTILITY-MAX-DEPTH" => Some(TuneParam::FutilityMaxDepth),
            "NULL_MOVE_MIN_DEPTH" | "NULL-MOVE-MIN-DEPTH" => Some(TuneParam::NullMoveMinDepth),
            "NULL_MOVE_REDUCTION_BASE" | "NULL-MOVE-REDUCTION-BASE" => {
                Some(TuneParam::NullMoveReductionBase)
            }
            "NULL_MOVE_REDUCTION_DIVISOR" | "NULL-MOVE-REDUCTION-DIVISOR" => {
                Some(TuneParam::NullMoveReductionDivisor)
            }
            "NULL_MOVE_MARGIN_DIVISOR" | "NULL-MOVE-MARGIN-DIVISOR" => {
                Some(TuneParam::NullMoveMarginDivisor)
            }
            "NULL_MOVE_MARGIN_CAP" | "NULL-MOVE-MARGIN-CAP" => Some(TuneParam::NullMoveMarginCap),
            "NULL_MOVE_KING_PRESSURE_LIMIT" | "NULL-MOVE-KING-PRESSURE-LIMIT" => {
                Some(TuneParam::NullMoveKingPressureLimit)
            }
            "NULL_MOVE_NON_PAWN_LIMIT" | "NULL-MOVE-NON-PAWN-LIMIT" => {
                Some(TuneParam::NullMoveNonPawnLimit)
            }
            "SEE_MARGIN_PER_DEPTH_CP" | "SEE-MARGIN-PER-DEPTH-CP" => {
                Some(TuneParam::SeeMarginPerDepthCp)
            }
            "HISTORY_PRUNE_MARGIN_PER_DEPTH" | "HISTORY-PRUNE-MARGIN-PER-DEPTH" => {
                Some(TuneParam::HistoryPruneMarginPerDepth)
            }
            "HISTORY_PRUNE_MAX_DEPTH" | "HISTORY-PRUNE-MAX-DEPTH" => {
                Some(TuneParam::HistoryPruneMaxDepth)
            }
            "CHECK_EXTENSION_MAX_DEPTH" | "CHECK-EXTENSION-MAX-DEPTH" => {
                Some(TuneParam::CheckExtensionMaxDepth)
            }
            "LMP_MAX_DEPTH" | "LMP-MAX-DEPTH" => Some(TuneParam::LmpMaxDepth),
            "IID_MIN_DEPTH" | "IID-MIN-DEPTH" => Some(TuneParam::IidMinDepth),
            "LMR_DIVISOR_MILLIS" | "LMR-DIVISOR-MILLIS" => Some(TuneParam::LmrDivisorMillis),
            _ => None,
        }
    }

    fn idx(self) -> usize {
        self as usize
    }

    fn from_idx(idx: usize) -> TuneParam {
        match idx {
            0 => TuneParam::ProbCutMinDepth,
            1 => TuneParam::ProbCutMarginCp,
            2 => TuneParam::RootRepetitionTieMinScore,
            3 => TuneParam::ReverseFutilityBaseCp,
            4 => TuneParam::ReverseFutilityPerDepthCp,
            5 => TuneParam::ReverseFutilityMaxDepth,
            6 => TuneParam::FutilityMarginPerDepthCp,
            7 => TuneParam::FutilityMaxDepth,
            8 => TuneParam::NullMoveMinDepth,
            9 => TuneParam::NullMoveReductionBase,
            10 => TuneParam::NullMoveReductionDivisor,
            11 => TuneParam::NullMoveMarginDivisor,
            12 => TuneParam::NullMoveMarginCap,
            13 => TuneParam::NullMoveKingPressureLimit,
            14 => TuneParam::NullMoveNonPawnLimit,
            15 => TuneParam::SeeMarginPerDepthCp,
            16 => TuneParam::HistoryPruneMarginPerDepth,
            17 => TuneParam::HistoryPruneMaxDepth,
            18 => TuneParam::CheckExtensionMaxDepth,
            19 => TuneParam::LmpMaxDepth,
            20 => TuneParam::IidMinDepth,
            _ => TuneParam::LmrDivisorMillis,
        }
    }
}

#[inline(always)]
pub fn get_int(param: TuneParam, default: i64) -> i64 {
    if !TUNING_ACTIVE.load(Ordering::Relaxed) {
        return default;
    }
    let value = OVERRIDES[param.idx()].load(Ordering::Relaxed);
    if value == 0 {
        default
    } else {
        value
    }
}

pub fn set(param: TuneParam, value: i64) {
    TUNING_ACTIVE.store(true, Ordering::Relaxed);
    OVERRIDES[param.idx()].store(value, Ordering::Relaxed);
}

pub fn reset() {
    TUNING_ACTIVE.store(false, Ordering::Relaxed);
    for slot in &OVERRIDES {
        slot.store(0, Ordering::Relaxed);
    }
}

pub fn is_active() -> bool {
    TUNING_ACTIVE.load(Ordering::Relaxed)
}

pub fn active_overrides() -> Vec<(TuneParam, i64)> {
    if !TUNING_ACTIVE.load(Ordering::Relaxed) {
        return Vec::new();
    }
    (0..PARAM_COUNT)
        .filter_map(|idx| {
            let value = OVERRIDES[idx].load(Ordering::Relaxed);
            (value != 0).then(|| (TuneParam::from_idx(idx), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_returns_default_when_tuning_inactive() {
        reset();
        assert!(!is_active());
        assert_eq!(get_int(TuneParam::ProbCutMinDepth, 8), 8);
        assert_eq!(get_int(TuneParam::ReverseFutilityBaseCp, 80), 80);
        assert_eq!(get_int(TuneParam::LmrDivisorMillis, 1800), 1800);
    }

    #[test]
    fn set_activates_and_overrides() {
        set(TuneParam::NullMoveReductionBase, 4);
        assert!(is_active());
        assert_eq!(get_int(TuneParam::NullMoveReductionBase, 3), 4);
        reset();
        assert!(!is_active());
        assert_eq!(get_int(TuneParam::NullMoveReductionBase, 3), 3);
    }

    #[test]
    fn active_overrides_lists_only_set_values() {
        reset();
        set(TuneParam::SeeMarginPerDepthCp, 96);
        let overrides = active_overrides();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0], (TuneParam::SeeMarginPerDepthCp, 96));
        reset();
        assert!(active_overrides().is_empty());
    }
}
