use std::time::Instant;
use crate::board::{
    BoardState, EMPTY_SQ, MATE, INF, MAX_PLY,
    piece_from_char, bit, sq_to_str, move_to_uci,
    is_attacked,
};
#[cfg(feature = "decision-trace")]
use crate::board::board_to_fen;
use crate::search::Searcher;
#[cfg(feature = "decision-trace")]
use crate::trace::{DecisionTrace, DepthInfo, TraceLogger};
use crate::zobrist::compute_hash;
use crate::movegen::{apply_move, generate_moves};
use crate::book::OpeningBook;

pub struct Engine {
    pub st:       BoardState,
    pub searcher: Searcher,
    pub book:     Option<OpeningBook>,
    #[cfg(feature = "decision-trace")]
    pub trace:    TraceLogger,
}

impl Engine {
    pub fn new() -> Self {
        let mut e = Engine {
            st:       BoardState::empty(),
            searcher: Searcher::new(),
            book:     None,
            #[cfg(feature = "decision-trace")]
            trace:    TraceLogger::from_env(),
        };
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let h = compute_hash(&e.st);
        e.searcher.rep_stack.push(h);
        e.searcher.rep_stack_len = 1;
        e
    }

    pub fn set_fen(&mut self, fen: &str) {
        self.st = BoardState::empty();
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
        if parts.len() > 2 {
            let r = parts[2];
            self.st.cr[0] = r.contains('K');
            self.st.cr[1] = r.contains('Q');
            self.st.cr[2] = r.contains('k');
            self.st.cr[3] = r.contains('q');
        }

        self.st.ep = if parts.len() > 3 && parts[3] != "-" {
            let b = parts[3].as_bytes();
            if b.len() >= 2 {
                let col = (b[0] - b'a') as usize;
                let row = 8usize.wrapping_sub((b[1] - b'0') as usize);
                if row < 8 { Some(row * 8 + col) } else { None }
            } else { None }
        } else { None };

        self.st.mc = if parts.len() > 4 { parts[4].parse().unwrap_or(0) } else { 0 };

        self.searcher.rep_stack.clear();
        self.searcher.rep_stack_len = 0;
        let h = compute_hash(&self.st);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len = 1;
    }

    pub fn make_move_uci(&mut self, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
        apply_move(&mut self.st, sr, sc, er, ec, promotion);
        let h = compute_hash(&self.st);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len += 1;
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
                eprintln!("info depth 0 score mate 0");
                println! ("info depth 0 score mate 0");
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
                eprintln!("info depth 0 score cp 0");
                println! ("info depth 0 score cp 0");
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

        if let Some(ref book) = self.book {
            if let Some(bm) = book.pick_move(&self.st, &moves) {
                let mv_str = move_to_uci(&self.st, &bm);
                let eval_score = self.searcher.corrected_eval(&self.st);
                println!("info depth 1 score cp {} nodes 0 nps 0 time 0 pv {}", eval_score, mv_str);
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

        self.searcher.killers  = [[None; 2]; MAX_PLY];
        self.searcher.history  = [[0i32; 64]; 64];

        let start      = Instant::now();
        let mut best_move  = moves[0];
        let mut best_score = 0i32;
        let mut total_nodes = 0u64;

        let init_eval = self.searcher.corrected_eval(&self.st);
        let mut prev_score = init_eval;
        #[cfg(feature = "decision-trace")]
        let mut depth_infos = Vec::new();

        for depth in 1..=depth_limit {
            if start.elapsed().as_secs_f64() > time_limit { break; }

            let mut nd = 0u64;
            let init_delta = if depth >= 5 { 25 } else { INF };
            let mut asp_delta = init_delta;
            let (mut alpha, mut beta) = if asp_delta < INF {
                (prev_score - asp_delta, prev_score + asp_delta)
            } else { (-INF, INF) };

            let mut asp_best  = best_move;
            let mut asp_score = -INF;

            'asp: loop {
                let mut sorted = moves.clone();
                if asp_best != moves[0] {
                    if let Some(pos) = sorted.iter().position(|&m| m == asp_best) {
                        sorted.swap(0, pos);
                    }
                }

                let mut cur_best  = sorted[0];
                let mut cur_score = -INF;
                let mut loop_alpha = alpha;

                for &mv in &sorted {
                    if start.elapsed().as_secs_f64() > time_limit { break; }
                    let old = self.st;
                    apply_move(&mut self.st, mv[0], mv[1], mv[2], mv[3], 0);
                    let h = compute_hash(&self.st);
                    self.searcher.rep_stack.push(h);
                    self.searcher.rep_stack_len += 1;

                    let score = if cur_score == -INF {
                        -self.searcher.negamax(&mut self.st, depth-1, 1, -beta, -loop_alpha,
                                               true, start, time_limit, &mut nd)
                    } else {
                        let s = -self.searcher.negamax(&mut self.st, depth-1, 1,
                                                       -loop_alpha-1, -loop_alpha,
                                                       true, start, time_limit, &mut nd);
                        if s > loop_alpha && s < beta {
                            -self.searcher.negamax(&mut self.st, depth-1, 1, -beta, -loop_alpha,
                                                   true, start, time_limit, &mut nd)
                        } else { s }
                    };

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;

                    if score > cur_score { cur_score = score; cur_best = mv; }
                    if score > loop_alpha { loop_alpha = score; }
                    if loop_alpha >= beta { break; }
                }

                if start.elapsed().as_secs_f64() > time_limit { break 'asp; }

                if cur_score <= alpha {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    alpha = (prev_score - asp_delta).max(-INF);
                    beta  = prev_score + init_delta;
                    continue 'asp;
                }
                if cur_score >= beta {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    beta  = (prev_score + asp_delta).min(INF);
                    asp_best = cur_best;
                    continue 'asp;
                }
                asp_best  = cur_best;
                asp_score = cur_score;
                break;
            }

            total_nodes += nd;
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed <= time_limit {
                best_move  = asp_best;
                best_score = asp_score;
                prev_score = best_score;
                let nps = if elapsed > 0.0 { (total_nodes as f64 / elapsed) as i64 } else { 0 };
                let score_str = if best_score.abs() > 90_000 {
                    let mate_in = (MATE - best_score.abs()) / 2 + 1;
                    if best_score > 0 { format!("mate {}", mate_in) }
                    else              { format!("mate -{}", mate_in) }
                } else { format!("cp {}", best_score) };
                let pv = format!("{}{}", sq_to_str(best_move[0]*8+best_move[1]),
                                         sq_to_str(best_move[2]*8+best_move[3]));
                eprintln!("info depth {} score {} nodes {} nps {} pv {}",
                          depth, score_str, total_nodes, nps, pv);
                println! ("info depth {} score {} nodes {} nps {} pv {}",
                          depth, score_str, total_nodes, nps, pv);
                #[cfg(feature = "decision-trace")]
                depth_infos.push(DepthInfo {
                    depth,
                    score_cp: best_score,
                    nodes: total_nodes,
                    elapsed_ms: (elapsed * 1000.0) as u128,
                    pv,
                });
            } else { break; }
        }

        let mv_str = move_to_uci(&self.st, &best_move);
        let elapsed = start.elapsed().as_secs_f64();
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
