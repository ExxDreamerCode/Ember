use super::*;

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
    pub(super) trace_roots: bool,
    pub(super) trace_nnue_parity: bool,
    pub(super) trace_singular_candidates: bool,
    pub(super) dag: SearchDagTrace,
    pub(super) stats: SearchDebugStats,
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
    tt_store_visits: u64,
    min_tt_store_depth: i32,
    max_tt_store_depth: i32,
    min_tt_store_score: i32,
    max_tt_store_score: i32,
    tt_store_alpha: u64,
    tt_store_beta: u64,
    tt_store_exact: u64,
    tt_probcut_stores: u64,
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
            tt_store_visits: 0,
            min_tt_store_depth: i32::MAX,
            max_tt_store_depth: i32::MIN,
            min_tt_store_score: i32::MAX,
            max_tt_store_score: i32::MIN,
            tt_store_alpha: 0,
            tt_store_beta: 0,
            tt_store_exact: 0,
            tt_probcut_stores: 0,
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

    pub(super) fn record_eval(&mut self, eval: i32) {
        self.eval_visits += 1;
        self.min_eval = self.min_eval.min(eval);
        self.max_eval = self.max_eval.max(eval);
    }

    pub(super) fn record_tt(&mut self, depth: i32, score: i32, flag: u8) {
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

    pub(super) fn record_tt_store(&mut self, depth: i32, score: i32, flag: u8, probcut: bool) {
        self.tt_store_visits += 1;
        self.min_tt_store_depth = self.min_tt_store_depth.min(depth);
        self.max_tt_store_depth = self.max_tt_store_depth.max(depth);
        self.min_tt_store_score = self.min_tt_store_score.min(score);
        self.max_tt_store_score = self.max_tt_store_score.max(score);
        self.tt_probcut_stores += u64::from(probcut);
        match flag {
            TT_ALPHA => self.tt_store_alpha += 1,
            TT_BETA => self.tt_store_beta += 1,
            TT_EXACT => self.tt_store_exact += 1,
            _ => {}
        }
    }

    pub(super) fn record_q_delta(&mut self, alpha: i32, stand: i32) {
        let gap = alpha - stand;
        self.q_delta_cutoffs += 1;
        self.min_q_delta_gap = self.min_q_delta_gap.min(gap);
        self.max_q_delta_gap = self.max_q_delta_gap.max(gap);
    }

    pub(super) fn record_draw(&mut self, status: DrawStatus) {
        match status {
            DrawStatus::None => {}
            DrawStatus::SearchCycle => self.search_cycle_returns += 1,
            DrawStatus::Claimable => self.claimable_draw_returns += 1,
            DrawStatus::Automatic => self.automatic_draw_returns += 1,
        }
    }
}

#[cfg(feature = "search-debug")]
pub(super) struct SearchDagTrace {
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
    pub(super) fn from_env() -> Self {
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

    pub(super) fn begin_root(&mut self, depth: i32, mv: &str) {
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
    pub(super) fn record_node(
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

    pub(super) fn record_eval(&mut self, hash: u64, eval: i32) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_eval(eval);
        }
    }

    pub(super) fn record_tt(&mut self, hash: u64, depth: i32, score: i32, flag: u8) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_tt(depth, score, flag);
        }
    }

    pub(super) fn record_tt_store(
        &mut self,
        hash: u64,
        depth: i32,
        score: i32,
        flag: u8,
        probcut: bool,
    ) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_tt_store(depth, score, flag, probcut);
        }
    }

    pub(super) fn record_q_delta(&mut self, hash: u64, alpha: i32, stand: i32) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_q_delta(alpha, stand);
        }
    }

    pub(super) fn record_draw(&mut self, hash: u64, status: DrawStatus) {
        if let Some(node) = self.nodes.get_mut(&hash) {
            node.record_draw(status);
        }
    }

    pub(super) fn emit(&mut self, score: i32, searched_nodes: u64) {
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
            let (min_tt_store_depth, max_tt_store_depth, min_tt_store_score, max_tt_store_score) =
                if node.tt_store_visits == 0 {
                    (0, 0, 0, 0)
                } else {
                    (
                        node.min_tt_store_depth,
                        node.max_tt_store_depth,
                        node.min_tt_store_score,
                        node.max_tt_store_score,
                    )
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
                 \"tt_exact\":{},\"tt_store_visits\":{},\
                 \"min_tt_store_depth\":{min_tt_store_depth},\
                 \"max_tt_store_depth\":{max_tt_store_depth},\
                 \"min_tt_store_score\":{min_tt_store_score},\
                 \"max_tt_store_score\":{max_tt_store_score},\
                 \"tt_store_alpha\":{},\"tt_store_beta\":{},\
                 \"tt_store_exact\":{},\"tt_probcut_stores\":{},\
                 \"q_delta_cutoffs\":{},\
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
                node.tt_store_visits,
                node.tt_store_alpha,
                node.tt_store_beta,
                node.tt_store_exact,
                node.tt_probcut_stores,
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

#[cfg(feature = "search-debug")]
impl SearchDebug {
    pub(super) fn from_env() -> Self {
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
            trace_nnue_parity: env_flag("EMBER_TRACE_NNUE_PARITY"),
            trace_singular_candidates: env_flag("EMBER_TRACE_SINGULAR_CANDIDATES"),
            dag: SearchDagTrace::from_env(),
            stats: SearchDebugStats::default(),
        }
    }

    pub(super) fn reset_stats(&mut self) {
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
