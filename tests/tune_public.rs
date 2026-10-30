use ember_chess::tune::{active_overrides, get_int, is_active, reset, set, TuneParam};

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

#[test]
fn zero_and_negative_values_are_real_overrides() {
    reset();
    set(TuneParam::LmrDivisorMillis, 0);
    set(TuneParam::RootRepetitionTieMinScore, -25);

    assert_eq!(get_int(TuneParam::LmrDivisorMillis, 1800), 0);
    assert_eq!(get_int(TuneParam::RootRepetitionTieMinScore, 300), -25);
    assert_eq!(
        active_overrides(),
        vec![
            (TuneParam::RootRepetitionTieMinScore, -25),
            (TuneParam::LmrDivisorMillis, 0),
        ]
    );

    reset();
    assert_eq!(get_int(TuneParam::LmrDivisorMillis, 1800), 1800);
}

#[test]
fn every_parameter_round_trips_through_names_values_and_listing() {
    use TuneParam::*;
    let params = [
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
        ProbCutReduction,
        QsearchDeltaMarginCp,
        QsearchCheckCapDepth,
        QsearchSeeThresholdCp,
        LmpMoveCountScalePermille,
        LmpKingPressureLimit,
        LmrBaseMillis,
        LmrMinMoveIndex,
        LmrMinDepth,
        LmrNonPvExtra,
        AspirationMinDepth,
        AspirationDeltaCp,
        TacticalCheckExtensionMaxDepth,
        CorrGravityLimit,
        CorrBonusCapPermille,
        CorrDepthCap,
        CorrWeightNum,
        CorrWeightDen,
        LmrCaptureDivisorMillis,
        HistoryBonusMult,
        KingShieldPenalty,
        OutpostBonus,
        PassedPushBonus,
        AttackOrderBonus,
        ExtensionMinDepth,
        ExtensionBaseMarginCp,
        ExtensionMarginPerDepthCp,
        ExtensionMaxTtAge,
        ExtensionPolicyMinDepth,
        ExtensionDoubleMinDepth,
        ExtensionTripleMinDepth,
        ExtensionLadderSteps,
        ExtensionLadderStepCp,
        LdseEnabled,
        LdseMarginCp,
        LdseMaxDepth,
        LdseMaxPerLine,
        NegExtMaxPerLine,
        ExtensionTtDepthMargin,
        ExtensionMaxHalfmoveClock,
    ];
    assert_eq!(TuneParam::COUNT, params.len());
    reset();
    for (idx, param) in params.into_iter().enumerate() {
        assert_eq!(TuneParam::from_name(param.name()), Some(param));
        assert_eq!(
            TuneParam::from_name(&param.name().to_ascii_lowercase().replace('_', "-")),
            Some(param)
        );
        let value = idx as i64 - 20;
        set(param, value);
        assert_eq!(get_int(param, 123), value);
        assert_eq!(active_overrides(), vec![(param, value)]);
        reset();
        assert_eq!(get_int(param, 123), 123);
    }
    let expected: Vec<_> = params
        .into_iter()
        .enumerate()
        .map(|(idx, param)| (param, idx as i64))
        .collect();
    for &(param, value) in &expected {
        set(param, value);
    }
    assert_eq!(active_overrides(), expected);
    reset();
}
