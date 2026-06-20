#[cfg(feature = "decision-trace")]
use crate::board::board_to_fen;
use crate::board::{
    bit, is_attacked, move_ec, move_promotion, move_to_uci, piece_from_char, piece_on, piece_type,
    sq, sq_c, BoardState, EMPTY_SQ, INF, MATE, MAX_PLY,
};
use crate::book::OpeningBook;
use crate::movegen::Move;
use crate::movegen::{apply_move, generate_moves};
use crate::search::{lazy_smp_search, Searcher};
use crate::syzygy::SyzygyTables;
#[cfg(feature = "decision-trace")]
use crate::trace::{DecisionTrace, DepthInfo, TraceLogger};
use crate::tt::SharedTT;
use crate::zobrist::compute_hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct Engine {
    pub st: BoardState,
    pub searcher: Searcher,
    pub shared_tt: Arc<SharedTT>,
    pub num_threads: usize,
    pub stopped: Arc<AtomicBool>,
    pub book: Option<OpeningBook>,
    #[cfg(feature = "decision-trace")]
    pub trace: TraceLogger,
}

fn set_castling_rook_by_side(st: &mut BoardState, white: bool, kingside: bool) {
    let rank = if white { 7usize } else { 0usize };
    let king_col = sq_c(st.king_sq(white));
    let mut candidate = None;

    for col in 0..8 {
        let rook_sq = sq(rank, col);
        let pi = piece_on(&st.bb, rook_sq);
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

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(128));
        let mut e = Engine {
            st: BoardState::empty(),
            searcher: Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped)),
            shared_tt,
            num_threads: 1,
            stopped,
            book: None,
            #[cfg(feature = "decision-trace")]
            trace: TraceLogger::from_env(),
        };
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let h = compute_hash(&e.st);
        e.searcher.rep_stack.push(h);
        e.searcher.rep_stack_len = 1;
        e
    }

    pub fn set_fen(&mut self, fen: &str) {
        let chess960_mode = self.st.chess960;
        self.st = BoardState::empty();
        self.st.chess960 = chess960_mode;
        let parts: Vec<&str> = fen.split(' ').collect();

        for (ri, rs) in parts[0].split('/').enumerate() {
            let mut ci = 0usize;
            for ch in rs.chars() {
                if ch.is_ascii_digit() {
                    ci += ch.to_digit(10).unwrap() as usize;
                } else {
                    let pi = piece_from_char(ch as u8);
                    if pi != EMPTY_SQ {
                        self.st.bb[pi as usize] |= bit(ri * 8 + ci);
                    }
                    ci += 1;
                }
            }
        }

        self.st.w = parts.len() > 1 && parts[1] == "w";

        self.st.cr = [false; 4];
        self.st.castling_rooks = [None; 4];
        if parts.len() > 2 {
            let r = parts[2];
            if r == "-" {
            } else {
                let has_file_rights = r.chars().any(|ch| {
                    let b = ch as u8;
                    (b'A'..=b'H').contains(&b) || (b'a'..=b'h').contains(&b)
                });
                if has_file_rights {
                    self.st.chess960 = true;
                    for ch in r.chars() {
                        let b = ch as u8;
                        if !(b'A'..=b'H').contains(&b) && !(b'a'..=b'h').contains(&b) {
                            continue;
                        }
                        let col = (b.to_ascii_lowercase() - b'a') as usize;
                        let white = ch.is_uppercase();
                        let rank = if white { 7usize } else { 0usize };
                        let rook_sq = sq(rank, col);
                        let pi = piece_on(&self.st.bb, rook_sq);
                        if pi != EMPTY_SQ && piece_type(pi) == 3 && (pi < 6) == white {
                            let king_sq = self.st.king_sq(white);
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
                            self.st.cr[idx] = true;
                            self.st.castling_rooks[idx] = Some(rook_sq);
                        }
                    }
                } else if self.st.chess960 {
                    if r.contains('K') {
                        set_castling_rook_by_side(&mut self.st, true, true);
                    }
                    if r.contains('Q') {
                        set_castling_rook_by_side(&mut self.st, true, false);
                    }
                    if r.contains('k') {
                        set_castling_rook_by_side(&mut self.st, false, true);
                    }
                    if r.contains('q') {
                        set_castling_rook_by_side(&mut self.st, false, false);
                    }
                } else {
                    if r.contains('K') {
                        self.st.cr[0] = true;
                        self.st.castling_rooks[0] = Some(sq(7, 7));
                    }
                    if r.contains('Q') {
                        self.st.cr[1] = true;
                        self.st.castling_rooks[1] = Some(sq(7, 0));
                    }
                    if r.contains('k') {
                        self.st.cr[2] = true;
                        self.st.castling_rooks[2] = Some(sq(0, 7));
                    }
                    if r.contains('q') {
                        self.st.cr[3] = true;
                        self.st.castling_rooks[3] = Some(sq(0, 0));
                    }
                }
            }
        }

        self.st.ep = if parts.len() > 3 && parts[3] != "-" {
            let b = parts[3].as_bytes();
            if b.len() >= 2 {
                let col = (b[0] - b'a') as usize;
                let row = 8usize.wrapping_sub((b[1] - b'0') as usize);
                if row < 8 {
                    Some(row * 8 + col)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        self.st.mc = if parts.len() > 5 {
            let fullmove = parts[5].parse::<usize>().unwrap_or(1).saturating_sub(1);
            fullmove * 2 + usize::from(!self.st.w)
        } else {
            0
        };

        self.searcher.rep_stack.clear();
        self.searcher.rep_stack_len = 0;
        let h = compute_hash(&self.st);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len = 1;
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
            mv[0],
            mv[1],
            mv[2],
            move_ec(&mv),
            move_promotion(&mv),
        );
        let h = compute_hash(&self.st);
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
            if mv[0] != sr || mv[1] != sc {
                return false;
            }

            let move_promo = move_promotion(mv).to_ascii_uppercase();
            let input_promo = promotion.to_ascii_uppercase();
            let promo_matches =
                move_promo == input_promo || (input_promo == 0 && move_promo == b'Q');
            if !promo_matches {
                return false;
            }

            if mv[2] == er && move_ec(mv) == ec {
                return true;
            }

            let from = sq(mv[0], mv[1]);
            let to = sq(mv[2], move_ec(mv));
            let pi = piece_on(&self.st.bb, from);
            let target = piece_on(&self.st.bb, to);
            if !self.st.chess960
                || pi == EMPTY_SQ
                || piece_type(pi) != 5
                || target == EMPTY_SQ
                || piece_type(target) != 3
                || (target < 6) != (pi < 6)
                || er != mv[0]
            {
                return false;
            }

            let king_dst_col = if move_ec(mv) > mv[1] { 6usize } else { 2usize };
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
        let moves = generate_moves(&self.st, self.st.w, &self.st.cr, self.st.ep);
        #[cfg(feature = "decision-trace")]
        let root_fen = board_to_fen(&self.st);
        #[cfg(feature = "decision-trace")]
        let legal_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&self.st, mv)).collect();
        #[cfg(feature = "decision-trace")]
        let side = if self.st.w { "white" } else { "black" };
        if moves.is_empty() {
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

        if !self.st.chess960 {
            if let Some(ref book) = self.book {
                if let Some(bm) = book.pick_move(&self.st, &moves) {
                    let mv_str = move_to_uci(&self.st, &bm);
                    let eval_score = self.searcher.corrected_eval(&self.st);
                    println!(
                        "info depth 1 score cp {} nodes 0 nps 0 time 0 pv {}",
                        eval_score, mv_str
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
                        elapsed_ms: 0,
                        depth_infos: &[],
                    });
                    return (mv_str, eval_score, 0, 0.0);
                }
            }
        }

        if self.num_threads > 1 {
            let start = Instant::now();
            self.searcher.stopped.store(false, Ordering::SeqCst);

            let (best_move, best_score, best_depth, total_nodes) = lazy_smp_search(
                Arc::clone(&self.shared_tt),
                &self.st,
                time_limit,
                depth_limit,
                self.num_threads,
            );

            let mv_str = move_to_uci(&self.st, &best_move);
            let elapsed = start.elapsed().as_secs_f64();
            self.searcher
                .update_correction_history(&self.st, best_score, best_depth);
            return (mv_str, best_score, total_nodes, elapsed);
        }

        self.searcher.killers = [[None; 2]; MAX_PLY];
        self.searcher.history = [[0i32; 64]; 64];
        self.searcher.stopped.store(false, Ordering::SeqCst);
        self.searcher.init_nnue_stack(&self.st);

        let start = Instant::now();
        let mut best_move = moves[0];
        let mut best_score = 0i32;
        let mut total_nodes = 0u64;

        let init_eval = self.searcher.corrected_eval(&self.st);
        let mut prev_score = init_eval;
        let mut best_depth = 0;
        #[cfg(feature = "decision-trace")]
        let mut depth_infos = Vec::new();

        for depth in 1..=depth_limit {
            if start.elapsed().as_secs_f64() > time_limit {
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

            'asp: loop {
                let sorted = if SyzygyTables::pieces_ok(&self.st)
                    && self.searcher.syzygy.tables.is_some()
                    && depth >= 2
                {
                    let mut with_dtz: Vec<(i32, Move)> = moves
                        .iter()
                        .map(|&mv| {
                            let old = self.st;
                            apply_move(
                                &mut self.st,
                                mv[0],
                                mv[1],
                                mv[2],
                                move_ec(&mv),
                                move_promotion(&mv),
                            );
                            let bonus = self.searcher.syzygy.dtz_bonus(&self.st).unwrap_or(0);
                            self.st = old;
                            (bonus, mv)
                        })
                        .collect();
                    with_dtz.sort_unstable_by_key(|b| std::cmp::Reverse(b.0));
                    if asp_best != with_dtz[0].1 {
                        if let Some(pos) = with_dtz.iter().position(|&(_, m)| m == asp_best) {
                            with_dtz.swap(0, pos);
                        }
                    }
                    with_dtz.into_iter().map(|(_, mv)| mv).collect()
                } else {
                    let mut s = moves.clone();
                    if asp_best != moves[0] {
                        if let Some(pos) = s.iter().position(|&m| m == asp_best) {
                            s.swap(0, pos);
                        }
                    }
                    s
                };

                let mut cur_best = sorted[0];
                let mut cur_score = -INF;
                let mut loop_alpha = alpha;

                for &mv in &sorted {
                    if start.elapsed().as_secs_f64() > time_limit {
                        break;
                    }
                    let old = self.st;
                    apply_move(
                        &mut self.st,
                        mv[0],
                        mv[1],
                        mv[2],
                        move_ec(&mv),
                        move_promotion(&mv),
                    );
                    crate::evaluate::with_nnue_net(|net| {
                        if !self.searcher.nnue_stack.is_empty() {
                            self.searcher.nnue_stack[1].refresh(net, &self.st);
                        }
                    });
                    let h = compute_hash(&self.st);
                    self.searcher.rep_stack.push(h);
                    self.searcher.rep_stack_len += 1;

                    let score = if cur_score == -INF {
                        -self.searcher.negamax(
                            &mut self.st,
                            depth - 1,
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
                            depth - 1,
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
                                depth - 1,
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

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;

                    if self.searcher.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    if score > cur_score {
                        cur_score = score;
                        cur_best = mv;
                    }
                    if score > loop_alpha {
                        loop_alpha = score;
                    }
                    if loop_alpha >= beta {
                        break;
                    }
                }

                if self.searcher.stopped.load(Ordering::Relaxed)
                    || start.elapsed().as_secs_f64() > time_limit
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
                break;
            }

            if self.searcher.stopped.load(Ordering::Relaxed) {
                break;
            }
            total_nodes += nd;
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed <= time_limit {
                best_move = asp_best;
                best_score = asp_score;
                best_depth = depth;
                prev_score = best_score;
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
                let pv = move_to_uci(&self.st, &best_move);
                println!(
                    "info depth {} score {} nodes {} nps {} time {} pv {}",
                    depth, score_str, total_nodes, nps, time_ms, pv
                );
                #[cfg(feature = "decision-trace")]
                depth_infos.push(DepthInfo {
                    depth,
                    score_cp: best_score,
                    nodes: total_nodes,
                    elapsed_ms: (elapsed * 1000.0) as u128,
                    pv,
                });
            } else {
                break;
            }
        }

        let mv_str = move_to_uci(&self.st, &best_move);
        let elapsed = start.elapsed().as_secs_f64();
        self.searcher
            .update_correction_history(&self.st, best_score, best_depth);
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
}
