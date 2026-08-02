#[cfg(feature = "decision-trace")]
use crate::board::board_to_fen;
#[cfg(test)]
use crate::board::NO_MOVE;
use crate::board::{
    bit, is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    move_to_uci, piece_from_char, piece_type, sq, sq_c, BoardState, Move, BK, EMPTY_SQ, INF, MATE,
    MAX_HALF_MOVE_CLOCK, WK,
};
use crate::book::{
    OpeningBook, DEFAULT_BOOK_MIN_MOVE_WEIGHT, DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE,
};
use crate::movegen::{apply_move, generate_moves};
use crate::search::{
    format_pv_line_uci, lazy_smp_search, prefer_non_repeating_root_on_tie,
    root_repetition_tie_scope, LazySmpPool, LazySmpSearchLimits, Searcher,
};
use crate::time_management::{iteration_time_decision, threads_for_time_budget, IterationTiming};
#[cfg(feature = "decision-trace")]
use crate::trace::{DecisionTrace, DepthInfo, TraceLogger};
use crate::tt::SharedTT;
use crate::zobrist::compute_hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_HASH_MB: usize = 256;
const FIFTY_MOVE_ROOT_MIN_CLOCK: u8 = 80;
const FIFTY_MOVE_ROOT_MAX_PIECES: u32 = 7;
const FIFTY_MOVE_ROOT_MIN_SCORE: i32 = 500;
const FIFTY_MOVE_ROOT_STATIC_MARGIN_CP: i32 = 350;
const FIFTY_MOVE_ROOT_MATERIAL_MARGIN_CP: i32 = 150;
const FIFTY_MOVE_ROOT_VERIFY_NODE_LIMIT: u32 = 1_000;

#[derive(Clone, Copy)]
enum SearchTimerStart {
    BeforeSetup(Instant),
    AfterSetup,
}

pub struct Engine {
    pub st: BoardState,
    pub searcher: Searcher,
    pub shared_tt: Arc<SharedTT>,
    pub search_pool: Arc<LazySmpPool>,
    pub num_threads: usize,
    pub stopped: Arc<AtomicBool>,
    pub book: Option<OpeningBook>,
    pub random_book_move: bool,
    pub book_min_move_weight: u16,
    pub book_min_move_weight_permille: u16,
    #[cfg(feature = "decision-trace")]
    pub trace: TraceLogger,
}

pub struct EngineBookConfig {
    pub book: Option<OpeningBook>,
    pub random_book_move: bool,
    pub min_move_weight: u16,
    pub min_move_weight_permille: u16,
}

impl EngineBookConfig {
    pub fn new(
        book: Option<OpeningBook>,
        min_move_weight: u16,
        min_move_weight_permille: u16,
    ) -> Self {
        Self {
            book,
            random_book_move: false,
            min_move_weight,
            min_move_weight_permille,
        }
    }

    pub fn with_random_book_move(mut self, random_book_move: bool) -> Self {
        self.random_book_move = random_book_move;
        self
    }
}

fn set_castling_rook_by_side(st: &mut BoardState, white: bool, kingside: bool) {
    let rank = if white { 7usize } else { 0usize };
    let king_col = sq_c(st.king_sq(white));
    let mut candidate = None;

    for col in 0..8 {
        let rook_sq = sq(rank, col);
        let pi = st.mailbox[rook_sq];
        if pi == EMPTY_SQ || piece_type(pi) != 3 || (pi < 6) != white {
            continue;
        }
        let better_candidate = if kingside {
            col > king_col && candidate.is_none_or(|prev| col < prev)
        } else {
            col < king_col && candidate.is_none_or(|prev| col > prev)
        };
        if better_candidate {
            candidate = Some(col);
        }
    }

    if let Some(col) = candidate {
        let idx = match (white, kingside) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        };
        st.cr[idx] = true;
        st.castling_rooks[idx] = Some(sq(rank, col));
    }
}

mod root;
#[cfg(test)]
use self::root::*;
#[cfg(not(test))]
use self::root::{
    root_child_after, root_depth_extension, root_move_preserves_fifty_move_conversion,
    root_total_piece_count, sort_root_moves, tt_root_move,
};

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::placeholder());
        let search_pool = Arc::new(LazySmpPool::new());
        let mut e = Engine {
            st: BoardState::empty(),
            searcher: Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped)),
            shared_tt,
            search_pool,
            num_threads: 1,
            stopped,
            book: None,
            random_book_move: false,
            book_min_move_weight: DEFAULT_BOOK_MIN_MOVE_WEIGHT,
            book_min_move_weight_permille: DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE,
            #[cfg(feature = "decision-trace")]
            trace: TraceLogger::from_env(),
        };
        e.searcher.tt_mb = DEFAULT_HASH_MB;
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        e
    }

    pub fn ensure_hash_ready(&mut self) {
        self.shared_tt.ensure_size(self.searcher.tt_mb);
    }

    fn root_static_score_after(&self, mv: Move) -> i32 {
        let child = root_child_after(&self.st, mv);
        -self.searcher.corrected_eval(&child)
    }

    fn root_fifty_move_conversion_choice(
        &self,
        legal_root_moves: &[Move],
        best_move: Move,
        best_score: i32,
    ) -> Move {
        if self.st.halfmove_clock < FIFTY_MOVE_ROOT_MIN_CLOCK
            || root_total_piece_count(&self.st) > FIFTY_MOVE_ROOT_MAX_PIECES
            || best_score < FIFTY_MOVE_ROOT_MIN_SCORE
        {
            return best_move;
        }

        if root_move_preserves_fifty_move_conversion(&self.st, best_move) == Some(true) {
            return best_move;
        }

        // This verifier is a bounded fallback, not exact DTZ. Near the claim boundary an
        // unproven best move is not trusted when a close root alternative has a proven
        // capture, pawn move, or mate before the fifty-move draw.
        let best_static = self.root_static_score_after(best_move);
        let mut replacement: Option<(Move, i32)> = None;
        for &candidate in legal_root_moves {
            if candidate == best_move {
                continue;
            }

            let static_score = self.root_static_score_after(candidate);
            if static_score + FIFTY_MOVE_ROOT_STATIC_MARGIN_CP < best_static {
                continue;
            }
            if root_move_preserves_fifty_move_conversion(&self.st, candidate) != Some(true) {
                continue;
            }

            if replacement.is_none_or(|(_, replacement_score)| static_score > replacement_score) {
                replacement = Some((candidate, static_score));
            }
        }

        if let Some((replacement, _)) = replacement {
            println!(
                "info string fifty-move root verifier replaced {} with {}",
                move_to_uci(&self.st, best_move),
                move_to_uci(&self.st, replacement)
            );
            replacement
        } else {
            best_move
        }
    }

    pub fn new_with(
        st: BoardState,
        searcher: Searcher,
        shared_tt: Arc<SharedTT>,
        search_pool: Arc<LazySmpPool>,
        num_threads: usize,
        stopped: Arc<AtomicBool>,
        book_config: EngineBookConfig,
    ) -> Self {
        Engine {
            st,
            searcher,
            shared_tt,
            search_pool,
            num_threads,
            stopped,
            book: book_config.book,
            random_book_move: book_config.random_book_move,
            book_min_move_weight: book_config.min_move_weight,
            book_min_move_weight_permille: book_config.min_move_weight_permille,
            #[cfg(feature = "decision-trace")]
            trace: TraceLogger::default(),
        }
    }

    pub fn set_fen(&mut self, fen: &str) {
        if let Err(e) = self.try_set_fen(fen) {
            eprintln!("info string Ignoring invalid FEN: {}", e);
        }
    }

    pub fn try_set_fen(&mut self, fen: &str) -> Result<(), String> {
        let chess960_mode = self.st.chess960;
        let mut next = BoardState::empty();
        next.chess960 = chess960_mode;
        let parts: Vec<&str> = fen.split(' ').collect();
        if parts.len() < 4 {
            return Err(
                "expected at least board, side, castling and en-passant fields".to_string(),
            );
        }

        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("board must contain exactly 8 ranks".to_string());
        }
        for (ri, rs) in ranks.iter().enumerate() {
            let mut ci = 0usize;
            for ch in rs.chars() {
                if ch.is_ascii_digit() {
                    let empty = ch.to_digit(10).unwrap() as usize;
                    if empty == 0 || ci + empty > 8 {
                        return Err("rank has invalid empty-square count".to_string());
                    }
                    ci += empty;
                } else {
                    let pi = piece_from_char(ch as u8);
                    if pi == EMPTY_SQ || ci >= 8 {
                        return Err("board contains an invalid piece placement".to_string());
                    }
                    next.bb[pi as usize] |= bit(ri * 8 + ci);
                    ci += 1;
                }
            }
            if ci != 8 {
                return Err("rank does not contain exactly 8 squares".to_string());
            }
        }
        if next.bb[WK].count_ones() != 1 || next.bb[BK].count_ones() != 1 {
            return Err("position must contain exactly one king per side".to_string());
        }
        next.refresh_mailbox();

        next.w = match parts[1] {
            "w" => true,
            "b" => false,
            _ => return Err("side-to-move must be 'w' or 'b'".to_string()),
        };

        next.cr = [false; 4];
        next.castling_rooks = [None; 4];
        if parts.len() > 2 {
            let r = parts[2];
            if r == "-" {
            } else {
                let has_file_rights = r.chars().any(|ch| {
                    let b = ch as u8;
                    (b'A'..=b'H').contains(&b) || (b'a'..=b'h').contains(&b)
                });
                if has_file_rights {
                    next.chess960 = true;
                    for ch in r.chars() {
                        let b = ch as u8;
                        if !(b'A'..=b'H').contains(&b) && !(b'a'..=b'h').contains(&b) {
                            return Err("invalid Chess960 castling rights".to_string());
                        }
                        let col = (b.to_ascii_lowercase() - b'a') as usize;
                        let white = ch.is_uppercase();
                        let rank = if white { 7usize } else { 0usize };
                        let rook_sq = sq(rank, col);
                        let pi = next.mailbox[rook_sq];
                        if pi != EMPTY_SQ && piece_type(pi) == 3 && (pi < 6) == white {
                            let king_sq = next.king_sq(white);
                            let idx = if white {
                                if col > sq_c(king_sq) {
                                    0
                                } else {
                                    1
                                }
                            } else if col > sq_c(king_sq) {
                                2
                            } else {
                                3
                            };
                            next.cr[idx] = true;
                            next.castling_rooks[idx] = Some(rook_sq);
                        }
                    }
                } else if r.chars().all(|ch| matches!(ch, 'K' | 'Q' | 'k' | 'q')) {
                    if next.chess960 {
                        if r.contains('K') {
                            set_castling_rook_by_side(&mut next, true, true);
                        }
                        if r.contains('Q') {
                            set_castling_rook_by_side(&mut next, true, false);
                        }
                        if r.contains('k') {
                            set_castling_rook_by_side(&mut next, false, true);
                        }
                        if r.contains('q') {
                            set_castling_rook_by_side(&mut next, false, false);
                        }
                    } else {
                        if r.contains('K') {
                            next.cr[0] = true;
                            next.castling_rooks[0] = Some(sq(7, 7));
                        }
                        if r.contains('Q') {
                            next.cr[1] = true;
                            next.castling_rooks[1] = Some(sq(7, 0));
                        }
                        if r.contains('k') {
                            next.cr[2] = true;
                            next.castling_rooks[2] = Some(sq(0, 7));
                        }
                        if r.contains('q') {
                            next.cr[3] = true;
                            next.castling_rooks[3] = Some(sq(0, 0));
                        }
                    }
                } else {
                    return Err("invalid castling rights".to_string());
                }
            }
        }

        next.ep = if parts.len() > 3 && parts[3] != "-" {
            let b = parts[3].as_bytes();
            if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
                return Err("invalid en-passant square".to_string());
            }
            let col = (b[0] - b'a') as usize;
            let row = 8usize - (b[1] - b'0') as usize;
            Some(row * 8 + col)
        } else {
            None
        };

        next.mc = if parts.len() > 5 {
            let fullmove = parts[5].parse::<usize>().unwrap_or(1).saturating_sub(1);
            fullmove * 2 + usize::from(!next.w)
        } else {
            0
        };

        next.halfmove_clock = if parts.len() > 4 {
            let parsed = parts[4]
                .parse::<u64>()
                .map_err(|_| "halfmove clock must be a nonnegative integer".to_string())?;
            parsed.min(u64::from(MAX_HALF_MOVE_CLOCK)) as u8
        } else {
            0
        };

        self.st = next;
        self.st.hash = compute_hash(&self.st);
        self.searcher.rep_stack.clear();
        self.searcher.rep_stack_len = 0;
        let h = self.st.hash;
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len = 1;
        Ok(())
    }

    pub fn make_move_uci(
        &mut self,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        let Some(mv) = self.legal_move_from_uci(sr, sc, er, ec, promotion) else {
            return false;
        };
        apply_move(
            &mut self.st,
            move_sr(mv),
            move_sc(mv),
            move_er(mv),
            move_ec(mv),
            move_promotion(mv),
        );
        let h = self.st.hash;
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len += 1;
        true
    }

    fn legal_move_from_uci(
        &self,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> Option<Move> {
        let moves = generate_moves(&self.st, self.st.w, &self.st.cr, self.st.ep);
        moves.into_iter().find(|mv| {
            if move_sr(*mv) != sr || move_sc(*mv) != sc {
                return false;
            }

            let move_promo = move_promotion(*mv).to_ascii_uppercase();
            let input_promo = promotion.to_ascii_uppercase();
            let promo_matches =
                move_promo == input_promo || (input_promo == 0 && move_promo == b'Q');
            if !promo_matches {
                return false;
            }

            if move_er(*mv) == er && move_ec(*mv) == ec {
                return true;
            }

            let from = move_from(*mv);
            let to = move_to(*mv);
            let pi = self.st.mailbox[from];
            let target = self.st.mailbox[to];
            if !self.st.chess960
                || pi == EMPTY_SQ
                || piece_type(pi) != 5
                || target == EMPTY_SQ
                || piece_type(target) != 3
                || (target < 6) != (pi < 6)
                || move_er(*mv) != move_sr(*mv)
            {
                return false;
            }

            let king_dst_col = if move_ec(*mv) > move_sc(*mv) {
                6usize
            } else {
                2usize
            };
            ec == king_dst_col
        })
    }

    pub fn is_check(&self) -> bool {
        let ks = self.st.king_sq(self.st.w);
        is_attacked(&self.st.bb, ks, !self.st.w)
    }

    pub fn load_book(&mut self, path: &str) -> Result<(), String> {
        self.book = Some(OpeningBook::load(path)?);
        Ok(())
    }

    #[cfg(feature = "decision-trace")]
    pub fn set_trace_file(&mut self, path: &str) {
        self.trace.set_path(path);
    }

    pub fn find_best_move(&mut self, time_limit: f64, depth_limit: i32) -> (String, i32, u64, f64) {
        self.find_best_move_with_time_limits(time_limit, time_limit, depth_limit)
    }

    pub fn find_best_move_with_time_limits(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
    ) -> (String, i32, u64, f64) {
        self.searcher.stopped.store(false, Ordering::SeqCst);
        self.searcher.pondering.store(false, Ordering::SeqCst);
        self.find_best_move_with_time_limits_prepared(soft_time_limit, time_limit, depth_limit)
    }

    pub fn find_best_move_with_time_limits_started_at(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
        node_limit: Option<u64>,
        start: Instant,
    ) -> (String, i32, u64, f64) {
        self.searcher.stopped.store(false, Ordering::SeqCst);
        self.searcher.pondering.store(false, Ordering::SeqCst);
        self.find_best_move_with_time_limits_prepared_started_at(
            soft_time_limit,
            time_limit,
            depth_limit,
            node_limit,
            start,
        )
    }

    pub fn find_best_move_with_time_limits_prepared(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
    ) -> (String, i32, u64, f64) {
        self.find_best_move_with_time_limits_prepared_with_timer(
            soft_time_limit,
            time_limit,
            depth_limit,
            None,
            SearchTimerStart::AfterSetup,
        )
    }

    pub fn find_best_move_with_time_limits_prepared_with_node_limit(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
        node_limit: Option<u64>,
    ) -> (String, i32, u64, f64) {
        self.find_best_move_with_time_limits_prepared_with_timer(
            soft_time_limit,
            time_limit,
            depth_limit,
            node_limit,
            SearchTimerStart::AfterSetup,
        )
    }

    pub fn find_best_move_with_time_limits_prepared_started_at(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
        node_limit: Option<u64>,
        start: Instant,
    ) -> (String, i32, u64, f64) {
        self.find_best_move_with_time_limits_prepared_with_timer(
            soft_time_limit,
            time_limit,
            depth_limit,
            node_limit,
            SearchTimerStart::BeforeSetup(start),
        )
    }

    fn find_best_move_with_time_limits_prepared_with_timer(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
        node_limit: Option<u64>,
        timer_start: SearchTimerStart,
    ) -> (String, i32, u64, f64) {
        let soft_time_limit = soft_time_limit.min(time_limit);
        self.searcher.refresh_nnue_net();
        self.searcher.refresh_search_backend();
        let legal_root_moves = generate_moves(&self.st, self.st.w, &self.st.cr, self.st.ep);
        #[cfg(feature = "decision-trace")]
        let root_fen = board_to_fen(&self.st);
        #[cfg(feature = "decision-trace")]
        let legal_moves: Vec<String> = legal_root_moves
            .iter()
            .map(|mv| move_to_uci(&self.st, *mv))
            .collect();
        #[cfg(feature = "decision-trace")]
        let side = if self.st.w { "white" } else { "black" };
        if legal_root_moves.is_empty() {
            let ks = self.st.king_sq(self.st.w);
            let in_check = is_attacked(&self.st.bb, ks, !self.st.w);
            if in_check {
                println!("info depth 0 score mate 0");
                #[cfg(feature = "decision-trace")]
                self.trace.emit_decision(DecisionTrace {
                    fen: &root_fen,
                    side,
                    legal_moves: &legal_moves,
                    chosen_move: "0000",
                    source: "terminal",
                    depth_reached: 0,
                    score_cp: -MATE,
                    nodes: 0,
                    elapsed_ms: 0,
                    depth_infos: &[],
                });
                return ("0000".into(), -MATE, 0, 0.0);
            } else {
                println!("info depth 0 score cp 0");
                #[cfg(feature = "decision-trace")]
                self.trace.emit_decision(DecisionTrace {
                    fen: &root_fen,
                    side,
                    legal_moves: &legal_moves,
                    chosen_move: "0000",
                    source: "terminal",
                    depth_reached: 0,
                    score_cp: 0,
                    nodes: 0,
                    elapsed_ms: 0,
                    depth_infos: &[],
                });
                return ("0000".into(), 0, 0, 0.0);
            }
        }

        let tablebase_start = match timer_start {
            SearchTimerStart::BeforeSetup(start) => start,
            SearchTimerStart::AfterSetup => Instant::now(),
        };
        if let Some(best_move) = self
            .searcher
            .syzygy
            .probe_root_move(&self.st, &legal_root_moves)
        {
            let mv_str = move_to_uci(&self.st, best_move);
            let score = self.searcher.syzygy.probe_root_score(&self.st).unwrap_or(0);
            let elapsed = tablebase_start.elapsed().as_secs_f64();
            println!(
                "info depth 1 score cp {} nodes 0 nps 0 time {} pv {}",
                score,
                (elapsed * 1000.0) as u64,
                mv_str
            );
            #[cfg(feature = "decision-trace")]
            self.trace.emit_decision(DecisionTrace {
                fen: &root_fen,
                side,
                legal_moves: &legal_moves,
                chosen_move: &mv_str,
                source: "syzygy",
                depth_reached: 1,
                score_cp: score,
                nodes: 0,
                elapsed_ms: (elapsed * 1000.0) as u128,
                depth_infos: &[],
            });
            return (mv_str, score, 0, elapsed);
        }
        let moves = legal_root_moves;

        if !self.st.chess960 {
            if let Some(ref book) = self.book {
                let choice = if self.random_book_move {
                    book.pick_move_with_quality(
                        &self.st,
                        &moves,
                        self.book_min_move_weight,
                        self.book_min_move_weight_permille,
                        crate::book::DEFAULT_BOOK_MAX_EVAL_LOSS_CP,
                        |mv| {
                            let mut child = self.st;
                            apply_move(
                                &mut child,
                                move_sr(mv),
                                move_sc(mv),
                                move_er(mv),
                                move_ec(mv),
                                move_promotion(mv),
                            );
                            -self.searcher.corrected_eval(&child)
                        },
                    )
                } else {
                    book.best_move_with_confidence(
                        &self.st,
                        &moves,
                        self.book_min_move_weight,
                        self.book_min_move_weight_permille,
                    )
                };
                if let Some(choice) = choice {
                    let mv_str = move_to_uci(&self.st, choice.mv);
                    let eval_score = self.searcher.corrected_eval(&self.st);
                    let elapsed = match timer_start {
                        SearchTimerStart::BeforeSetup(start) => start.elapsed().as_secs_f64(),
                        SearchTimerStart::AfterSetup => 0.0,
                    };
                    println!(
                        "info depth 1 score cp {} nodes 0 nps 0 time {} pv {}",
                        eval_score,
                        (elapsed * 1000.0) as u64,
                        mv_str
                    );
                    #[cfg(feature = "decision-trace")]
                    self.trace.emit_decision(DecisionTrace {
                        fen: &root_fen,
                        side,
                        legal_moves: &legal_moves,
                        chosen_move: &mv_str,
                        source: "book",
                        depth_reached: 1,
                        score_cp: eval_score,
                        nodes: 0,
                        elapsed_ms: (elapsed * 1000.0) as u128,
                        depth_infos: &[],
                    });
                    return (mv_str, eval_score, 0, elapsed);
                }
            }
        }

        let search_threads = threads_for_time_budget(self.num_threads, soft_time_limit);
        self.ensure_hash_ready();
        self.shared_tt.advance_generation();
        let preferred = tt_root_move(&self.searcher, &self.st, &moves);
        let ordered_moves = sort_root_moves(&self.st, &moves, preferred);
        if search_threads > 1 {
            let start = match timer_start {
                SearchTimerStart::BeforeSetup(start) => start,
                SearchTimerStart::AfterSetup => Instant::now(),
            };
            let (best_move, best_score, best_depth, total_nodes) = lazy_smp_search(
                &self.search_pool,
                Arc::clone(&self.shared_tt),
                &self.st,
                &ordered_moves,
                root_depth_extension,
                LazySmpSearchLimits {
                    soft_time: soft_time_limit,
                    hard_time: time_limit,
                    depth: depth_limit,
                    node_limit,
                    start,
                },
                search_threads,
                &mut self.searcher,
            );

            let best_move =
                self.root_fifty_move_conversion_choice(&ordered_moves, best_move, best_score);
            let mv_str = move_to_uci(&self.st, best_move);
            let elapsed = start.elapsed().as_secs_f64();
            self.searcher
                .update_correction_history(&self.st, best_score, best_depth);
            return (mv_str, best_score, total_nodes, elapsed);
        }

        self.searcher.prepare_for_search();
        self.searcher.set_node_limit(node_limit);
        self.searcher.init_nnue_stack(&self.st);

        let start = match timer_start {
            SearchTimerStart::BeforeSetup(start) => start,
            SearchTimerStart::AfterSetup => Instant::now(),
        };
        let mut best_move = ordered_moves[0];
        let mut best_score = 0i32;
        let mut total_nodes = 0u64;

        let init_eval = self.searcher.corrected_eval(&self.st);
        let mut prev_score = init_eval;
        let mut best_depth = 0;
        let mut stable_iterations = 0u32;
        let mut previous_iteration_seconds = 0.0;
        let mut previous_completed_elapsed = 0.0;
        #[cfg(feature = "decision-trace")]
        let mut depth_infos = Vec::new();

        for depth in 1..=depth_limit {
            if !self.searcher.pondering.load(Ordering::Relaxed)
                && start.elapsed().as_secs_f64() > time_limit
            {
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
                let sorted = sort_root_moves(&self.st, &ordered_moves, asp_best);
                let repetition_tie_scope = root_repetition_tie_scope(&self.st);

                let mut cur_best = sorted[0];
                let mut cur_score = -INF;
                let mut cur_best_nodes = 0u64;
                let mut cur_best_repeats = false;
                let mut loop_alpha = alpha;

                for (_root_index, &mv) in sorted.iter().enumerate() {
                    if !self.searcher.pondering.load(Ordering::Relaxed)
                        && start.elapsed().as_secs_f64() > time_limit
                    {
                        break;
                    }
                    let old = self.st;
                    self.searcher.enter_root_path(mv);
                    apply_move(
                        &mut self.st,
                        move_sr(mv),
                        move_sc(mv),
                        move_er(mv),
                        move_ec(mv),
                        move_promotion(mv),
                    );
                    self.searcher.refresh_nnue_stack_at(1, &self.st);
                    let h = self.st.hash;
                    self.searcher.rep_stack.push(h);
                    self.searcher.rep_stack_len += 1;
                    let root_ext = root_depth_extension(&old, mv);
                    let move_nodes_before = nd;
                    #[cfg(feature = "search-debug")]
                    {
                        self.searcher.reset_debug_stats();
                        self.searcher
                            .begin_debug_search_dag(depth, &move_to_uci(&old, mv));
                    }

                    let score = if cur_score == -INF {
                        -self.searcher.negamax(
                            &mut self.st,
                            depth - 1 + root_ext,
                            1,
                            -beta,
                            -loop_alpha,
                            true,
                            start,
                            time_limit,
                            &mut nd,
                        )
                    } else {
                        let s = -self.searcher.negamax(
                            &mut self.st,
                            depth - 1 + root_ext,
                            1,
                            -loop_alpha - 1,
                            -loop_alpha,
                            true,
                            start,
                            time_limit,
                            &mut nd,
                        );
                        if s > loop_alpha && s < beta {
                            -self.searcher.negamax(
                                &mut self.st,
                                depth - 1 + root_ext,
                                1,
                                -beta,
                                -loop_alpha,
                                true,
                                start,
                                time_limit,
                                &mut nd,
                            )
                        } else {
                            s
                        }
                    };
                    let move_nodes = nd.saturating_sub(move_nodes_before);
                    let root_repeats = if repetition_tie_scope
                        && (score > cur_score || (score == cur_score && cur_best_repeats))
                    {
                        self.searcher
                            .current_position_repeats(usize::from(self.st.halfmove_clock))
                    } else {
                        false
                    };

                    #[cfg(feature = "search-debug")]
                    self.searcher.emit_debug_root_trace(
                        depth,
                        _root_index,
                        &move_to_uci(&old, mv),
                        loop_alpha,
                        beta,
                        score,
                        move_nodes,
                    );
                    #[cfg(feature = "search-debug")]
                    self.searcher.emit_debug_search_dag(score, move_nodes);

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;
                    self.searcher.leave_root_path();

                    if self.searcher.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    if score > cur_score
                        || (score == cur_score
                            && prefer_non_repeating_root_on_tie(
                                score,
                                cur_best_repeats,
                                root_repeats,
                            ))
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

                if self.searcher.stopped.load(Ordering::Relaxed)
                    || (!self.searcher.pondering.load(Ordering::Relaxed)
                        && start.elapsed().as_secs_f64() > time_limit)
                {
                    break 'asp;
                }

                if cur_score <= alpha {
                    #[cfg(feature = "search-debug")]
                    self.searcher
                        .emit_debug_aspiration_trace(depth, alpha, beta, cur_score, "fail-low");
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    alpha = (prev_score - asp_delta).max(-INF);
                    beta = prev_score + init_delta;
                    continue 'asp;
                }
                if cur_score >= beta {
                    #[cfg(feature = "search-debug")]
                    self.searcher.emit_debug_aspiration_trace(
                        depth,
                        alpha,
                        beta,
                        cur_score,
                        "fail-high",
                    );
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    beta = (prev_score + asp_delta).min(INF);
                    asp_best = cur_best;
                    continue 'asp;
                }
                asp_best = cur_best;
                asp_score = cur_score;
                asp_best_nodes = cur_best_nodes;
                #[cfg(feature = "search-debug")]
                self.searcher
                    .emit_debug_aspiration_trace(depth, alpha, beta, cur_score, "exact");
                break;
            }

            total_nodes += nd;
            if self.searcher.stopped.load(Ordering::Relaxed) {
                break;
            }
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed <= time_limit || self.searcher.pondering.load(Ordering::Relaxed) {
                let score_change_cp = asp_score.saturating_sub(prev_score).abs();
                if best_depth == 0 || asp_best != best_move {
                    stable_iterations = 0;
                } else {
                    stable_iterations = stable_iterations.saturating_add(1);
                }
                let iteration_seconds = (elapsed - previous_completed_elapsed).max(0.0);
                let timing = IterationTiming {
                    elapsed_seconds: elapsed,
                    iteration_seconds,
                    previous_iteration_seconds,
                    score_change_cp,
                    stable_iterations,
                    best_move_effort: asp_best_nodes as f64 / nd.max(1) as f64,
                    worker_disagreement: 0.0,
                };
                let time_decision =
                    iteration_time_decision(soft_time_limit, time_limit, moves.len(), timing);
                best_move = asp_best;
                best_score = asp_score;
                best_depth = depth;
                prev_score = best_score;
                previous_iteration_seconds = iteration_seconds;
                previous_completed_elapsed = elapsed;
                self.searcher.shared_tt.store_with_pv(
                    self.st.hash,
                    depth,
                    best_score,
                    crate::tt::TT_EXACT,
                    Some(best_move),
                    true,
                );
                let nps = if elapsed > 0.0 {
                    (total_nodes as f64 / elapsed) as i64
                } else {
                    0
                };
                let time_ms = (elapsed * 1000.0) as u64;
                let score_str = if best_score.abs() > 90_000 {
                    let mate_in = (MATE - best_score.abs()) / 2 + 1;
                    if best_score > 0 {
                        format!("mate {}", mate_in)
                    } else {
                        format!("mate -{}", mate_in)
                    }
                } else {
                    format!("cp {}", best_score)
                };
                let pv_line =
                    crate::search::extract_pv_line(&self.searcher.shared_tt, &self.st, best_move);
                let pv_str = format_pv_line_uci(&self.st, &pv_line);
                println!(
                    "info depth {} score {} nodes {} nps {} time {} pv {}",
                    depth, score_str, total_nodes, nps, time_ms, pv_str
                );
                #[cfg(feature = "decision-trace")]
                depth_infos.push(DepthInfo {
                    depth,
                    score_cp: best_score,
                    nodes: total_nodes,
                    elapsed_ms: (elapsed * 1000.0) as u128,
                    pv: pv_str,
                });
                if !self.searcher.pondering.load(Ordering::Relaxed) && time_decision.stop {
                    break;
                }
            } else {
                break;
            }
        }

        let best_move =
            self.root_fifty_move_conversion_choice(&ordered_moves, best_move, best_score);
        let mv_str = move_to_uci(&self.st, best_move);
        let elapsed = start.elapsed().as_secs_f64();
        self.searcher
            .update_correction_history(&self.st, best_score, best_depth);
        self.searcher.clear_node_limit();
        #[cfg(feature = "decision-trace")]
        self.trace.emit_decision(DecisionTrace {
            fen: &root_fen,
            side,
            legal_moves: &legal_moves,
            chosen_move: &mv_str,
            source: "search",
            depth_reached: depth_infos.last().map(|d| d.depth).unwrap_or(0),
            score_cp: best_score,
            nodes: total_nodes,
            elapsed_ms: (elapsed * 1000.0) as u128,
            depth_infos: &depth_infos,
        });
        (mv_str, best_score, total_nodes, elapsed)
    }

    pub fn ponder_move_after(&self, best_move: &str) -> Option<String> {
        let bytes = best_move.as_bytes();
        if bytes.len() < 4
            || !(b'a'..=b'h').contains(&bytes[0])
            || !(b'1'..=b'8').contains(&bytes[1])
            || !(b'a'..=b'h').contains(&bytes[2])
            || !(b'1'..=b'8').contains(&bytes[3])
        {
            return None;
        }
        let promotion = bytes
            .get(4)
            .map_or(0, |piece| match piece.to_ascii_lowercase() {
                b'q' => b'Q',
                b'r' => b'R',
                b'b' => b'B',
                b'n' => b'N',
                _ => 0,
            });
        let root_move = self.legal_move_from_uci(
            8 - usize::from(bytes[1] - b'0'),
            usize::from(bytes[0] - b'a'),
            8 - usize::from(bytes[3] - b'0'),
            usize::from(bytes[2] - b'a'),
            promotion,
        )?;

        let mut child = self.st;
        apply_move(
            &mut child,
            move_sr(root_move),
            move_sc(root_move),
            move_er(root_move),
            move_ec(root_move),
            move_promotion(root_move),
        );
        let replies = generate_moves(&child, child.w, &child.cr, child.ep);
        if let Some(reply) = self
            .shared_tt
            .get_depth(child.hash)
            .and_then(|(_, _, _, best)| best)
        {
            if replies.contains(&reply) {
                return Some(move_to_uci(&child, reply));
            }
        }

        if !child.chess960 {
            if let Some(ref book) = self.book {
                if let Some(choice) = book.best_move_with_confidence(
                    &child,
                    &replies,
                    self.book_min_move_weight,
                    self.book_min_move_weight_permille,
                ) {
                    return Some(move_to_uci(&child, choice.mv));
                }
                if let Some(choice) = book.best_move_with_confidence(&child, &replies, 1, 0) {
                    return Some(move_to_uci(&child, choice.mv));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests;
