use super::*;

const QSEARCH_DELTA_MARGIN_CP: i32 = 1125;
const QSEARCH_CHECK_CAP_DEPTH: i32 = 4;
const QSEARCH_SEE_THRESHOLD_CP: i32 = 0;

#[inline(always)]
pub(super) fn qsearch_delta_prunable(alpha: i32, stand: i32) -> bool {
    let margin = tune::get_int(
        TuneParam::QsearchDeltaMarginCp,
        i64::from(QSEARCH_DELTA_MARGIN_CP),
    ) as i32;
    alpha - margin > stand
}

#[inline(always)]
pub(super) fn qsearch_check_cap_reached(depth: i32) -> bool {
    let max_depth = tune::get_int(
        TuneParam::QsearchCheckCapDepth,
        i64::from(QSEARCH_CHECK_CAP_DEPTH),
    ) as i32;
    depth <= -max_depth
}

#[inline(always)]
pub(super) fn qsearch_see_threshold_cp() -> i32 {
    tune::get_int(
        TuneParam::QsearchSeeThresholdCp,
        i64::from(QSEARCH_SEE_THRESHOLD_CP),
    ) as i32
}

#[inline(always)]
pub(super) fn qsearch_see_prunable(see_score: i32, threshold: i32) -> bool {
    see_score < threshold
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
                && qsearch_delta_prunable($alpha, stand)
            {
                #[cfg(feature = "search-debug")]
                {
                    $this.debug.stats.q_delta_cutoffs += 1;
                    $this.debug.dag.record_q_delta($st.hash, $alpha, stand);
                }
                return $alpha;
            }
        } else if $this.qsearch_check_cap_enabled()
            && excluded_move.is_none()
            && qsearch_check_cap_reached($depth)
        {
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
        let qsearch_see_threshold =
            if !in_check && excluded_move.is_none() && $this.qsearch_see_enabled() {
                Some(qsearch_see_threshold_cp())
            } else {
                None
            };
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
                && qsearch_see_threshold.is_some_and(|threshold| {
                    qsearch_see_prunable(
                        move_see::<CHESS960>($st, mv, from, to, fpi, tpi),
                        threshold,
                    )
                })
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

impl Searcher {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsearch(
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
        let other_net = self.other_net.clone();
        if let Some(net) = other_net.as_deref() {
            return if st.chess960 {
                self.qsearch_mode_scalar::<true, false, _>(
                    st,
                    alpha,
                    beta,
                    depth,
                    start,
                    tl,
                    cnt,
                    ply,
                    OtherNnueEval { net },
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
                    OtherNnueEval { net },
                )
            };
        }
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
    pub(super) fn qsearch_mode_scalar<
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
    pub(super) fn qsearch_mode_simd128<
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
    pub(super) fn qsearch_mode_simd256<
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
    pub(super) fn qsearch_mode_simd512<
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
    pub(super) unsafe fn qsearch_mode_x86_v3<
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
    pub(super) unsafe fn qsearch_mode_x86_avx512<
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
}
