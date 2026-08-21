use super::*;

#[derive(Clone)]
pub struct SearchLearning {
    pub(super) history: [[i32; 64]; 64],
    pub(super) counter_move: [[Option<Move>; 64]; 13],
    pub(super) corr_hist: [i32; CORR_HIST_SIZE * 2],
}

pub(super) struct ThreadResult {
    pub(super) thread_id: usize,
    pub(super) best_move: Move,
    pub(super) score: i32,
    pub(super) depth: i32,
    pub(super) nodes: u64,
    pub(super) learning: Option<Box<SearchLearning>>,
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
pub(super) struct LazySmpRootContext {
    pub(super) rep_stack: Vec<u64>,
    pub(super) rep_stack_len: usize,
    pub(super) nnue_net: Option<Arc<NNUENet>>,
    pub(super) search_backend: SearchBackendKind,
    pub(super) syzygy: SyzygyTables,
    pub(super) tt_mb: usize,
    pub(super) pondering: Arc<AtomicBool>,
    pub(super) learning: SearchLearning,
}

impl LazySmpRootContext {
    pub(super) fn from_searcher(searcher: &Searcher) -> Self {
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

pub(super) struct LazySmpSearchJob {
    pub(super) shared_tt: Arc<SharedTT>,
    pub(super) verification_move: Option<Move>,
    pub(super) verification_tt: Option<Arc<SharedTT>>,
    pub(super) stopped: Arc<AtomicBool>,
    pub(super) st: BoardState,
    pub(super) root_moves: Arc<Vec<Move>>,
    pub(super) num_threads: usize,
    pub(super) root_depth_extension: fn(&BoardState, Move) -> i32,
    pub(super) limits: LazySmpSearchLimits,
    pub(super) root_context: Arc<LazySmpRootContext>,
    pub(super) start: Instant,
    pub(super) global_best_depth: Arc<AtomicI32>,
    pub(super) printed_depth: Arc<AtomicI32>,
    pub(super) global_nodes: Arc<AtomicU64>,
    pub(super) node_limit_counter: Option<Arc<AtomicU64>>,
    pub(super) worker_best_moves: Vec<AtomicU64>,
    pub(super) worker_depths: Vec<AtomicI32>,
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
            limits,
            root_context: Arc::new(LazySmpRootContext::from_searcher(root_searcher)),
            start: limits.start,
            global_best_depth: Arc::new(AtomicI32::new(0)),
            printed_depth: Arc::new(AtomicI32::new(0)),
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
        if should_print_final_info(best.depth, job.printed_depth.load(Ordering::SeqCst)) {
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
    pub(super) fn worker_ids(&self) -> Vec<std::thread::ThreadId> {
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

pub(super) fn select_lazy_smp_result<'a>(
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

pub(super) fn lazy_smp_worker_root_moves(
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

pub(super) fn lazy_smp_root_moves(
    root_moves: &[Move],
    thread_id: usize,
    num_threads: usize,
) -> Vec<Move> {
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

pub(super) fn should_print_final_info(best_depth: i32, printed_depth: i32) -> bool {
    best_depth > 0 && best_depth > printed_depth
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

#[allow(clippy::too_many_arguments)]
pub fn lazy_smp_search(
    pool: &LazySmpPool,
    shared_tt: Arc<SharedTT>,
    st: &BoardState,
    root_moves: &[Move],
    root_depth_extension: fn(&BoardState, Move) -> i32,
    limits: LazySmpSearchLimits,
    num_threads: usize,
    root_searcher: &mut Searcher,
) -> (Move, i32, i32, u64) {
    pool.search(
        shared_tt,
        st,
        root_moves,
        root_depth_extension,
        limits,
        num_threads,
        root_searcher,
    )
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LazySmpAgreement {
    pub(super) disagreement: f64,
    pub(super) comparable_workers: usize,
    pub(super) principal_agrees: bool,
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
pub(super) fn lazy_smp_worker_disagreement(
    job: &LazySmpSearchJob,
    thread_id: usize,
    best_move: Move,
    depth: i32,
) -> f64 {
    lazy_smp_worker_agreement(job, thread_id, best_move, depth).disagreement
}

pub(super) fn lazy_smp_worker_can_coordinate_stop(
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
        let init_delta = aspiration_window_delta(depth);
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
                let root_ext = (job.root_depth_extension)(&st, mv);
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
            job.global_best_depth.fetch_max(depth, Ordering::SeqCst);
            let prev_printed = job.printed_depth.fetch_max(depth, Ordering::SeqCst);
            if prev_printed < depth {
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
