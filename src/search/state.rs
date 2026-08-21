use super::*;

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
            prev_moves: [None; MAX_PLY],
            prev_pieces: [0; MAX_PLY],
            continuation_history: vec![0i32; CONTINUATION_HIST_SIZE].into_boxed_slice(),
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
    pub(super) fn time_up(&self, start: Instant, tl: f64) -> bool {
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
    pub(super) fn search_limit_reached<const NODE_LIMITED: bool>(
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

    pub(super) fn set_shared_node_limit(
        &mut self,
        node_limit: Option<u64>,
        counter: Option<Arc<AtomicU64>>,
    ) {
        self.node_limit = node_limit;
        self.shared_node_counter = counter;
    }

    pub fn clear_node_limit(&mut self) {
        self.node_limit = None;
        self.shared_node_counter = None;
    }

    const BUF_POOL_CAP: usize = MAX_PLY + 64;

    pub(super) fn ensure_buf_pools(&mut self, ply: usize) {
        let need = (ply + 1).min(Self::BUF_POOL_CAP);
        if self.move_bufs.len() < need {
            self.move_bufs.resize_with(need, Vec::new);
            self.scored_bufs.resize_with(need, Vec::new);
            self.quiets_bufs.resize_with(need, Vec::new);
            self.caps_bufs.resize_with(need, Vec::new);
        }
    }

    #[inline]
    pub(super) fn take_buf<T>(pool: &mut [Vec<T>], ply: usize) -> Vec<T> {
        if ply < pool.len() {
            std::mem::take(&mut pool[ply])
        } else {
            Vec::new()
        }
    }

    #[inline]
    pub(super) fn return_buf<T>(pool: &mut [Vec<T>], ply: usize, buf: Vec<T>) {
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
            continuation_history: self.continuation_history.clone(),
        }
    }

    pub fn import_learning(&mut self, learning: &SearchLearning) {
        self.history = learning.history;
        self.counter_move = learning.counter_move;
        self.corr_hist = learning.corr_hist;
        self.continuation_history = learning.continuation_history.clone();
    }

    pub fn prepare_for_search(&mut self) {
        self.rep_root_len = self.rep_stack_len;
        self.excluded_moves.fill(None);
        self.prev_moves.fill(None);
        self.prev_pieces.fill(0);
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
        self.decay_continuation_history();
    }

    pub(super) fn enter_child_path(
        &mut self,
        ply: usize,
        mv: Move,
        mover_piece: u8,
        singular_extension: i32,
    ) -> ChildPathState {
        #[cfg(not(feature = "search-debug"))]
        let _ = singular_extension;
        let previous_move = self.prev_moves[ply].replace(mv);
        let previous_piece = std::mem::replace(&mut self.prev_pieces[ply], mover_piece);
        #[cfg(feature = "search-debug")]
        let previous_debug_move = self.path_moves[ply].replace(mv);
        #[cfg(feature = "search-debug")]
        let child_ply = (ply + 1 < MAX_PLY).then_some(ply + 1);
        #[cfg(feature = "search-debug")]
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
            previous_piece,
            is_null: false,
            #[cfg(feature = "search-debug")]
            previous_debug_move,
            #[cfg(feature = "search-debug")]
            child_ply,
            #[cfg(feature = "search-debug")]
            previous_extensions,
        }
    }

    pub(super) fn enter_null_path(&mut self, ply: usize) -> ChildPathState {
        #[cfg(feature = "search-debug")]
        let previous_debug_move = self.path_moves[ply].take();
        #[cfg(feature = "search-debug")]
        let child_ply = (ply + 1 < MAX_PLY).then_some(ply + 1);
        #[cfg(feature = "search-debug")]
        let previous_extensions = child_ply
            .map(|child| {
                let previous = self.singular_extensions_used[child];
                self.singular_extensions_used[child] =
                    next_singular_extension_count(self.singular_extensions_used[ply], 0);
                previous
            })
            .unwrap_or(0);
        ChildPathState {
            ply,
            previous_move: None,
            previous_piece: 0,
            is_null: true,
            #[cfg(feature = "search-debug")]
            previous_debug_move,
            #[cfg(feature = "search-debug")]
            child_ply,
            #[cfg(feature = "search-debug")]
            previous_extensions,
        }
    }

    pub(super) fn leave_child_path(&mut self, state: ChildPathState) {
        if !state.is_null {
            self.prev_moves[state.ply] = state.previous_move;
            self.prev_pieces[state.ply] = state.previous_piece;
        }
        #[cfg(feature = "search-debug")]
        {
            self.path_moves[state.ply] = state.previous_debug_move;
            if let Some(child) = state.child_ply {
                self.singular_extensions_used[child] = state.previous_extensions;
            }
        }
    }

    pub(crate) fn enter_root_path(&mut self, mv: Move, mover_piece: u8) {
        self.prev_moves[0] = Some(mv);
        self.prev_pieces[0] = mover_piece;
        #[cfg(feature = "search-debug")]
        {
            self.path_moves[0] = Some(mv);
            self.singular_extensions_used[0] = 0;
            self.singular_extensions_used[1] = 0;
        }
    }

    pub(crate) fn leave_root_path(&mut self) {
        self.prev_moves[0] = None;
        self.prev_pieces[0] = 0;
        #[cfg(feature = "search-debug")]
        {
            self.path_moves[0] = None;
            self.singular_extensions_used[1] = 0;
        }
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_shuffling(&self, ply: usize, halfmove_clock: u8) -> bool {
        reversible_shuffle(&self.path_moves, ply, halfmove_clock)
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_shuffling(&self, _ply: usize, _halfmove_clock: u8) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_path_extensions(&self, ply: usize) -> u8 {
        self.singular_extensions_used[ply]
    }

    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_path_extensions(&self, _ply: usize) -> u8 {
        0
    }

    pub fn clear_learning(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[0; 64]; 64];
        self.counter_move = [[None; 64]; 13];
        self.corr_hist = [0; CORR_HIST_SIZE * 2];
        self.continuation_history = vec![0i32; CONTINUATION_HIST_SIZE].into_boxed_slice();
    }

    #[inline(always)]
    pub(super) fn continuation_score(&self, ply: usize, piece_idx: usize, to: usize) -> i32 {
        debug_assert!(piece_idx < 7 && to < 64);
        let mut s = 0i32;
        if ply >= 1 {
            if let (Some(prev), Some(prev_p)) = (
                self.prev_moves.get(ply - 1).copied().flatten(),
                self.prev_pieces.get(ply - 1).copied().filter(|&p| p != 0),
            ) {
                s += self.continuation_history
                    [continuation_idx(0, prev_p as usize, move_to(prev), piece_idx, to)];
            }
        }
        if ply >= 2 {
            if let (Some(prev), Some(prev_p)) = (
                self.prev_moves.get(ply - 2).copied().flatten(),
                self.prev_pieces.get(ply - 2).copied().filter(|&p| p != 0),
            ) {
                s += self.continuation_history
                    [continuation_idx(1, prev_p as usize, move_to(prev), piece_idx, to)];
            }
        }
        s.clamp(-32768, 32768)
    }

    #[inline(always)]
    pub(super) fn update_continuation_history(
        &mut self,
        ply: usize,
        mover_piece: u8,
        mv: Move,
        delta: i32,
    ) {
        let m_to = move_to(mv);
        let mut halved = false;
        for (offset, prev_ply) in [ply.wrapping_sub(1), ply.wrapping_sub(2)]
            .into_iter()
            .enumerate()
        {
            if prev_ply >= MAX_PLY {
                continue;
            }
            let (Some(prev), Some(prev_p)) = (
                self.prev_moves.get(prev_ply).copied().flatten(),
                self.prev_pieces.get(prev_ply).copied().filter(|&p| p != 0),
            ) else {
                continue;
            };
            let cell = &mut self.continuation_history[continuation_idx(
                offset,
                prev_p as usize,
                move_to(prev),
                mover_piece as usize,
                m_to,
            )];
            *cell += delta;
            if !halved && (*cell > 16384 || *cell < -16384) {
                for value in self.continuation_history.iter_mut() {
                    *value /= 2;
                }
                halved = true;
            }
        }
    }
    pub(super) fn decay_continuation_history(&mut self) {
        for value in self.continuation_history.iter_mut() {
            *value = *value * 13 / 16;
        }
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
    pub(super) fn record_debug_dag_node(
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
    pub(super) fn record_debug_dag_eval(&mut self, hash: u64, eval: i32) {
        self.debug.dag.record_eval(hash, eval);
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn record_debug_dag_tt(&mut self, hash: u64, depth: i32, score: i32, flag: u8) {
        self.debug.dag.record_tt(hash, depth, score, flag);
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn record_debug_dag_tt_store(
        &mut self,
        hash: u64,
        depth: i32,
        score: i32,
        flag: u8,
        probcut: bool,
    ) {
        self.debug
            .dag
            .record_tt_store(hash, depth, score, flag, probcut);
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn record_debug_dag_draw(&mut self, hash: u64, status: DrawStatus) {
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
    pub(super) fn emit_debug_singular_candidate(
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
    pub(super) fn corr_hist_enabled(&self) -> bool {
        !self.debug.disable_corr_hist
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn corr_hist_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn futility_enabled(&self) -> bool {
        !self.debug.disable_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn history_pruning_enabled(&self) -> bool {
        !self.debug.disable_history_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn history_pruning_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn iid_reduction_enabled(&self) -> bool {
        !self.debug.disable_iid_reduction
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn iid_reduction_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn lmp_enabled(&self) -> bool {
        !self.debug.disable_lmp
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn lmp_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn lmr_enabled(&self) -> bool {
        !self.debug.disable_lmr
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn lmr_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn null_move_enabled(&self) -> bool {
        !self.debug.disable_null_move
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn null_move_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn reverse_futility_enabled(&self) -> bool {
        !self.debug.disable_reverse_futility
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn reverse_futility_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn see_pruning_enabled(&self) -> bool {
        !self.debug.disable_see_pruning
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn see_pruning_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(any(feature = "search-debug", test))]
    pub(super) fn set_restricted_verification(&mut self, active: bool) -> bool {
        std::mem::replace(&mut self.restricted_verification, active)
    }

    #[cfg(not(any(feature = "search-debug", test)))]
    #[inline(always)]
    pub(super) fn set_restricted_verification(&mut self, _active: bool) -> bool {
        false
    }

    #[cfg(any(feature = "search-debug", test))]
    pub(super) fn restricted_verification_active(&self) -> bool {
        self.restricted_verification
    }

    #[cfg(not(any(feature = "search-debug", test)))]
    #[inline(always)]
    pub(super) fn restricted_verification_active(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_multi_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_multi_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_multi_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_multicut_enabled(&self) -> bool {
        self.debug.enable_singular_multicut
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_multicut_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn singular_negative_extensions_enabled(&self) -> bool {
        self.debug.enable_singular_negative_extensions
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn singular_negative_extensions_enabled(&self) -> bool {
        false
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn qsearch_check_cap_enabled(&self) -> bool {
        !self.debug.disable_qsearch_check_cap
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn qsearch_check_cap_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn qsearch_delta_enabled(&self) -> bool {
        !self.debug.disable_qsearch_delta
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn qsearch_delta_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn qsearch_see_enabled(&self) -> bool {
        !self.debug.disable_qsearch_see
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn qsearch_see_enabled(&self) -> bool {
        true
    }

    #[cfg(feature = "search-debug")]
    pub(super) fn probcut_enabled(&self) -> bool {
        !self.debug.disable_probcut
    }
    #[cfg(not(feature = "search-debug"))]
    #[inline(always)]
    pub(super) fn probcut_enabled(&self) -> bool {
        true
    }

    #[inline(always)]
    pub(super) fn static_eval_classic<const CHESS960: bool>(&self, st: &BoardState) -> i32 {
        if CHESS960 && st.mc <= 3 {
            return evaluate(st) * if st.w { 1 } else { -1 };
        }
        evaluate(st) * if st.w { 1 } else { -1 }
    }

    #[inline(always)]
    pub(super) fn static_eval_nnue<const CHESS960: bool, B: NnueBackend>(
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
        #[cfg(feature = "search-debug")]
        if self.debug.trace_nnue_parity && ply < self.nnue_stack.len() {
            let mut refreshed = NNUEAccumulator::new(net.hidden_size);
            B::refresh(&mut refreshed, net, st);
            let fresh_score = evaluate_nnue_acc_with_backend::<B>(net, &refreshed, st);
            let scalar_score =
                evaluate_nnue_acc_with_backend::<ScalarNnueBackend>(net, &refreshed, st);
            if score != fresh_score || score != scalar_score {
                eprintln!(
                    "info string search-debug nnue-parity \
                     {{\"hash\":\"{:016x}\",\"fen\":\"{}\",\"ply\":{},\
                     \"incremental\":{},\"refreshed\":{},\"scalar\":{}}}",
                    st.hash,
                    crate::board::board_to_fen(st),
                    ply,
                    score,
                    fresh_score,
                    scalar_score,
                );
            }
        }
        if st.w {
            score
        } else {
            -score
        }
    }

    #[inline(always)]
    pub(super) fn static_eval_threat_nnue<const CHESS960: bool, B: NnueBackend>(
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

    pub(super) fn corrected_eval_classic<const CHESS960: bool>(&self, st: &BoardState) -> i32 {
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
    pub(super) fn corrected_eval_nnue<const CHESS960: bool, B: NnueBackend>(
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
    pub(super) fn corrected_eval_threat_nnue<const CHESS960: bool, B: NnueBackend>(
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

    pub(super) fn repetition_info(&self, reversible_plies: usize) -> (u8, bool) {
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
    pub(super) fn is_repetition(&self) -> bool {
        self.repetition_info(usize::MAX).0 >= 3
    }

    pub(super) fn draw_status(
        &self,
        st: &BoardState,
        ply: usize,
        minimum_ply: usize,
    ) -> DrawStatus {
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

    pub(super) fn draw_score(
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
    pub(super) fn push_nnue_acc<B: NnueBackend>(
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
    pub(super) fn push_threat_nnue_acc<B: NnueBackend>(
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
}
