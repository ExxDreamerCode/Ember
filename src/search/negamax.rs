use super::*;

const LMP_MOVE_COUNTS: [usize; 8] = [4, 7, 11, 17, 24, 33, 44, 57];
const LMP_MOVE_COUNT_SCALE_PERMILLE: i64 = 1000;
const LMP_KING_PRESSURE_LIMIT: i64 = 1;
const LMR_BASE_MILLIS: i64 = 500;
const LMR_MIN_MOVE_INDEX: i64 = 2;
const LMR_MIN_DEPTH: i64 = 2;
const LMR_NON_PV_EXTRA: i64 = 1;
const TACTICAL_CHECK_EXTENSION_MAX_DEPTH: i64 = 2;

#[inline(always)]
pub(super) fn lmp_move_count(depth: i32) -> Option<usize> {
    if !(1..=LMP_MOVE_COUNTS.len() as i32).contains(&depth) {
        return None;
    }
    let base = LMP_MOVE_COUNTS[(depth - 1) as usize];
    let scale = tune::get_int(
        TuneParam::LmpMoveCountScalePermille,
        LMP_MOVE_COUNT_SCALE_PERMILLE,
    );
    Some(((base as i64 * scale + 500) / 1000).max(1) as usize)
}

#[inline(always)]
pub(super) fn lmp_king_pressure_safe(king_pressure: u32) -> bool {
    let limit = tune::get_int(TuneParam::LmpKingPressureLimit, LMP_KING_PRESSURE_LIMIT) as u32;
    king_pressure < limit
}

#[inline(always)]
pub(super) fn lmr_policy_eligible(
    move_index: usize,
    actual_depth: i32,
    is_quiet: bool,
    in_check: bool,
) -> bool {
    if !is_quiet || in_check {
        return false;
    }
    let min_move_index = tune::get_int(TuneParam::LmrMinMoveIndex, LMR_MIN_MOVE_INDEX);
    let min_depth = tune::get_int(TuneParam::LmrMinDepth, LMR_MIN_DEPTH);
    move_index as i64 >= min_move_index && i64::from(actual_depth) >= min_depth
}

#[inline(always)]
pub(super) fn lmr_reduction(move_index: usize, actual_depth: i32, is_pv: bool) -> i32 {
    let base_millis = tune::get_int(TuneParam::LmrBaseMillis, LMR_BASE_MILLIS) as f64;
    let divisor = tune::get_int(TuneParam::LmrDivisorMillis, 1800) as f64;
    let non_pv_extra = tune::get_int(TuneParam::LmrNonPvExtra, LMR_NON_PV_EXTRA) as i32;
    let max_reduction = (actual_depth - 1).max(1);
    let reduction = (base_millis / 1000.0
        + (move_index as f64).ln() * (actual_depth as f64).ln() * 1000.0 / divisor)
        as i32;
    let reduction = reduction.clamp(1, max_reduction);
    if is_pv {
        reduction
    } else {
        (reduction + non_pv_extra).clamp(1, max_reduction)
    }
}

#[inline(always)]
pub(super) fn tactical_check_extension_candidate(
    actual_depth: i32,
    in_check: bool,
    legal_moves_seen: usize,
    is_quiet: bool,
) -> bool {
    if in_check || legal_moves_seen != 0 || is_quiet {
        return false;
    }
    let max_depth = tune::get_int(
        TuneParam::TacticalCheckExtensionMaxDepth,
        TACTICAL_CHECK_EXTENSION_MAX_DEPTH,
    );
    i64::from(actual_depth) <= max_depth
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
        let ext_depth = tune::get_int(TuneParam::CheckExtensionMaxDepth, 16) as i32;
        let ext = if in_check && $depth < ext_depth { 1 } else { 0 };
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

        let rfp_max_depth = tune::get_int(TuneParam::ReverseFutilityMaxDepth, 8) as i32;
        let rfp_base = tune::get_int(TuneParam::ReverseFutilityBaseCp, 80) as i32;
        let rfp_per_depth = tune::get_int(TuneParam::ReverseFutilityPerDepthCp, 65) as i32;
        if $this.reverse_futility_enabled()
            && !in_check
            && !is_pv
            && actual_depth <= rfp_max_depth
            && $ply > 0
        {
            let margin = rfp_base + rfp_per_depth * actual_depth;
            if eval_score - margin >= beta {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.reverse_futility_cutoffs += 1;
                }
                return eval_score - margin;
            }
        }
        let futility_max_depth = tune::get_int(TuneParam::FutilityMaxDepth, 3) as i32;
        let futility_margin = tune::get_int(TuneParam::FutilityMarginPerDepthCp, 150) as i32;
        if $this.futility_enabled()
            && excluded_move.is_none()
            && !in_check
            && !is_pv
            && actual_depth <= futility_max_depth
            && $ply > 0
        {
            let margin = futility_margin * actual_depth;
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
        let null_king_pressure = tune::get_int(TuneParam::NullMoveKingPressureLimit, 3) as u32;
        let null_min_depth = tune::get_int(TuneParam::NullMoveMinDepth, 3) as i32;
        let null_non_pawn = tune::get_int(TuneParam::NullMoveNonPawnLimit, 4) as u32;
        let null_base = tune::get_int(TuneParam::NullMoveReductionBase, 3) as i32;
        let null_divisor = tune::get_int(TuneParam::NullMoveReductionDivisor, 4) as i32;
        let null_margin_divisor = tune::get_int(TuneParam::NullMoveMarginDivisor, 200) as i32;
        let null_margin_cap = tune::get_int(TuneParam::NullMoveMarginCap, 3) as i32;
        if $this.null_move_enabled()
            && excluded_move.is_none()
            && king_pressure < null_king_pressure
            && !in_check
            && $can_null
            && !is_pv
            && $ply > 0
            && actual_depth >= null_min_depth
            && has_non_pawn(&$st.bb, $st.w)
            && eval_score >= beta
        {
            let total_non_pawn = (all_occ(&$st.bb) & !($st.bb[WP] | $st.bb[BP])).count_ones();
            if total_non_pawn > null_non_pawn {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.null_attempts += 1;
                }
                let r = null_base
                    + actual_depth / null_divisor
                    + ((eval_score - beta) / null_margin_divisor).min(null_margin_cap);
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

        let iid_min_depth = tune::get_int(TuneParam::IidMinDepth, 4) as i32;
        let actual_depth = if $this.iid_reduction_enabled()
            && excluded_move.is_none()
            && tt_move.is_none()
            && actual_depth >= iid_min_depth
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
                            #[cfg(feature = "search-debug")]
                            $this.record_debug_dag_tt_store(
                                h,
                                candidate.store_depth,
                                candidate.beta,
                                TT_BETA,
                                true,
                            );
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

        let lmp_max_depth = tune::get_int(TuneParam::LmpMaxDepth, 8) as i32;
        let lmp_count = if $this.lmp_enabled()
            && excluded_move.is_none()
            && lmp_king_pressure_safe(king_pressure)
            && !is_pv
            && !in_check
            && actual_depth <= lmp_max_depth
        {
            lmp_move_count(actual_depth).unwrap_or(usize::MAX)
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
                    let see_margin = tune::get_int(TuneParam::SeeMarginPerDepthCp, 80) as i32;
                    if $this.see_pruning_enabled()
                        && move_see::<CHESS960>($st, mv, from, to, fpi, tpi)
                            < -see_margin * actual_depth
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.see_skips += 1;
                        }
                        continue;
                    }
                } else if is_quiet && $this.history_pruning_enabled() {
                    let (fk, tk) = from_to_key(move_sr(mv), move_sc(mv), move_er(mv), move_ec(mv));
                    let history_max_depth =
                        tune::get_int(TuneParam::HistoryPruneMaxDepth, 5) as i32;
                    let history_margin =
                        tune::get_int(TuneParam::HistoryPruneMarginPerDepth, 1024) as i32;
                    if actual_depth <= history_max_depth
                        && $this.history[fk][tk] < -history_margin * actual_depth
                    {
                        #[cfg(feature = "search-debug")]
                        {
                            $this.debug.stats.history_skips += 1;
                        }
                        continue;
                    }
                }
            }

            let tactical_move_ext = if tactical_check_extension_candidate(
                actual_depth,
                in_check,
                legal_moves_seen,
                is_quiet,
            ) && special_move_gives_check_mode::<CHESS960>($st, mv)
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
                && move_index > 0
                && lmr_policy_eligible(move_index, actual_depth, is_quiet, in_check);
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
                let r = lmr_reduction(move_index, actual_depth, is_pv);
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
            #[cfg(feature = "search-debug")]
            $this.record_debug_dag_tt_store(h, actual_depth, best_score, flag, false);
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
    pub(super) fn negamax_with_limits<const NODE_LIMITED: bool>(
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
    pub(super) fn negamax_scalar<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        if let Some(net) = other_net.as_deref() {
            return if st.chess960 {
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
                    OtherNnueEval { net },
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
                    OtherNnueEval { net },
                )
            };
        }
        let classic_net = self.classic_net.clone();
        if let Some(net) = classic_net.as_deref() {
            return if st.chess960 {
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
                    ClassicHalfKpEval { net },
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
                    ClassicHalfKpEval { net },
                )
            };
        }
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
    pub(super) fn negamax_simd128<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        if let Some(net) = other_net.as_deref() {
            return if st.chess960 {
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
                    OtherNnueEval { net },
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
                    OtherNnueEval { net },
                )
            };
        }
        let classic_net = self.classic_net.clone();
        if let Some(net) = classic_net.as_deref() {
            return if st.chess960 {
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
                    ClassicHalfKpEval { net },
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
                    ClassicHalfKpEval { net },
                )
            };
        }
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
    pub(super) fn negamax_simd256<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        if let Some(net) = other_net.as_deref() {
            return if st.chess960 {
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
                    OtherNnueEval { net },
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
                    OtherNnueEval { net },
                )
            };
        }
        let classic_net = self.classic_net.clone();
        if let Some(net) = classic_net.as_deref() {
            return if st.chess960 {
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
                    ClassicHalfKpEval { net },
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
                    ClassicHalfKpEval { net },
                )
            };
        }
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
    pub(super) fn negamax_simd512<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        if let Some(net) = other_net.as_deref() {
            return if st.chess960 {
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
                    OtherNnueEval { net },
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
                    OtherNnueEval { net },
                )
            };
        }
        let classic_net = self.classic_net.clone();
        if let Some(net) = classic_net.as_deref() {
            return if st.chess960 {
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
                    ClassicHalfKpEval { net },
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
                    ClassicHalfKpEval { net },
                )
            };
        }
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
    pub(super) unsafe fn negamax_x86_v3<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        let nnue_net = self.nnue_net.clone();
        let classic_net = self.classic_net.clone();
        unsafe {
            if let Some(net) = other_net.as_deref() {
                return if st.chess960 {
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
                        OtherNnueEval { net },
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
                        OtherNnueEval { net },
                    )
                };
            }
            if let Some(net) = classic_net.as_deref() {
                return if st.chess960 {
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
                        ClassicHalfKpEval { net },
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
                        ClassicHalfKpEval { net },
                    )
                };
            }
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
    pub(super) unsafe fn negamax_x86_avx512<const NODE_LIMITED: bool>(
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
        let other_net = self.other_net.clone();
        let nnue_net = self.nnue_net.clone();
        let classic_net = self.classic_net.clone();
        unsafe {
            if let Some(net) = other_net.as_deref() {
                return if st.chess960 {
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
                        OtherNnueEval { net },
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
                        OtherNnueEval { net },
                    )
                };
            }
            if let Some(net) = classic_net.as_deref() {
                return if st.chess960 {
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
                        ClassicHalfKpEval { net },
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
                        ClassicHalfKpEval { net },
                    )
                };
            }
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
    pub(super) fn negamax_mode_scalar<
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
    pub(super) fn negamax_mode_simd128<
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
    pub(super) fn negamax_mode_simd256<
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
    pub(super) fn negamax_mode_simd512<
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
    pub(super) unsafe fn negamax_mode_x86_v3<
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
    pub(super) unsafe fn negamax_mode_x86_avx512<
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
