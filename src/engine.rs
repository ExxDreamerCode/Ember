use std::time::Instant;
use crate::board::{BoardState, EMPTY, MATE, INF, MAX_PLY, find_king, is_attacked, coord_to_square};
use crate::search::Searcher;
use crate::zobrist::compute_hash;
use crate::movegen::{apply_move, generate_moves};
use crate::book::OpeningBook;

pub struct Engine {
    pub st: BoardState,
    pub searcher: Searcher,
    pub book: Option<OpeningBook>,
}

impl Engine {
    pub fn new() -> Self {
        let mut e = Engine {
            st: BoardState { b: [[EMPTY; 8]; 8], w: true, cr: [false; 4], ep: None, mc: 0 },
            searcher: Searcher::new(),
            book: None,
        };
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let h = compute_hash(&e.st.b, e.st.w, &e.st.cr, e.st.ep);
        e.searcher.rep_stack.push(h);
        e.searcher.rep_stack_len = 1;
        e
    }

    pub fn set_fen(&mut self, fen: &str) {
        let parts: Vec<&str> = fen.split(' ').collect();
        self.st.b = [[EMPTY; 8]; 8];
        for (ri, rs) in parts[0].split('/').enumerate() {
            let mut ci = 0;
            for ch in rs.chars() {
                if ch.is_ascii_digit() {
                    ci += ch.to_digit(10).unwrap() as usize;
                } else {
                    self.st.b[ri][ci] = ch as u8;
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
                if row < 8 { Some((row, col)) } else { None }
            } else { None }
        } else { None };
        self.st.mc = if parts.len() > 4 { parts[4].parse().unwrap_or(0) } else { 0 };
        self.searcher.rep_stack.clear();
        self.searcher.rep_stack_len = 0;
        let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len = 1;
    }

    pub fn make_move_uci(&mut self, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
        apply_move(&mut self.st, sr, sc, er, ec, promotion);
        let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        self.searcher.rep_stack.push(h);
        self.searcher.rep_stack_len += 1;
    }

    pub fn is_check(&self) -> bool {
        let (kr, kc) = find_king(&self.st.b, self.st.w);
        is_attacked(&self.st.b, kr, kc, !self.st.w)
    }

    pub fn load_book(&mut self, path: &str) -> Result<(), String> {
        let book = OpeningBook::load(path)?;
        self.book = Some(book);
        Ok(())
    }

    pub fn find_best_move(&mut self, time_limit: f64, depth_limit: i32) -> (String, i32, u64, f64) {
        let moves = generate_moves(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
        if moves.is_empty() {
            let (kr, kc) = find_king(&self.st.b, self.st.w);
            let in_check = is_attacked(&self.st.b, kr, kc, !self.st.w);
            if in_check {
                eprintln!("info depth 0 score mate 0");
                println!("info depth 0 score mate 0");
                return ("0000".into(), -MATE, 0, 0.0);
            } else {
                eprintln!("info depth 0 score cp 0");
                println!("info depth 0 score cp 0");
                return ("0000".into(), 0, 0, 0.0);
            }
        }

        if let Some(ref book) = self.book {
            if let Some(bm) = book.pick_move(&self.st, &moves) {
                let mv_str = format!("{}{}", coord_to_square(bm[0], bm[1]),
                                             coord_to_square(bm[2], bm[3]));
                let eval_score = self.searcher.corrected_eval(&self.st);
                println!("info depth 1 score cp {} nodes 0 nps 0 time 0 pv {}", eval_score, mv_str);
                return (mv_str, eval_score, 0, 0.0);
            }
        }

        self.searcher.killers = [[None; 2]; MAX_PLY];
        self.searcher.history = [[0i32; 64]; 64];

        let start = Instant::now();
        let mut best_move = moves[0];
        let mut best_score = 0i32;
        let mut total_nodes = 0u64;

        let init_eval = self.searcher.corrected_eval(&self.st);
        let mut prev_score = init_eval;

        for depth in 1..=depth_limit {
            if start.elapsed().as_secs_f64() > time_limit { break; }

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
                let mut sorted_moves = moves.clone();
                if asp_best != moves[0] {
                    if let Some(pos) = sorted_moves.iter().position(|&m| m == asp_best) {
                        sorted_moves.swap(0, pos);
                    }
                }

                let mut current_best = sorted_moves[0];
                let mut current_score = -INF;
                let mut loop_alpha = alpha;

                for &mv in &sorted_moves {
                    if start.elapsed().as_secs_f64() > time_limit { break; }
                    let old = self.st;
                    apply_move(&mut self.st, mv[0], mv[1], mv[2], mv[3], 0);
                    let h = compute_hash(&self.st.b, self.st.w, &self.st.cr, self.st.ep);
                    self.searcher.rep_stack.push(h);
                    self.searcher.rep_stack_len += 1;

                    let score = if current_score == -INF {
                        -self.searcher.negamax(&mut self.st, depth - 1, 1, -beta, -loop_alpha, true, start, time_limit, &mut nd)
                    } else {
                        let s = -self.searcher.negamax(&mut self.st, depth - 1, 1, -loop_alpha - 1, -loop_alpha, true, start, time_limit, &mut nd);
                        if s > loop_alpha && s < beta {
                            -self.searcher.negamax(&mut self.st, depth - 1, 1, -beta, -loop_alpha, true, start, time_limit, &mut nd)
                        } else { s }
                    };

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;

                    if score > current_score {
                        current_score = score;
                        current_best = mv;
                    }
                    if score > loop_alpha { loop_alpha = score; }
                    if loop_alpha >= beta { break; }
                }

                if start.elapsed().as_secs_f64() > time_limit { break 'asp; }

                if current_score <= alpha {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    alpha = (prev_score - asp_delta).max(-INF);
                    beta = prev_score + init_delta;
                    continue 'asp;
                }
                if current_score >= beta {
                    asp_delta = asp_delta.saturating_mul(2).min(INF);
                    beta = (prev_score + asp_delta).min(INF);
                    asp_best = current_best;
                    continue 'asp;
                }
                asp_best = current_best;
                asp_score = current_score;
                break;
            }

            total_nodes += nd;
            let elapsed = start.elapsed().as_secs_f64();

            if elapsed <= time_limit {
                best_move = asp_best;
                best_score = asp_score;
                prev_score = best_score;
                let nps = if elapsed > 0.0 { (total_nodes as f64 / elapsed) as i64 } else { 0 };
                let score_str = if best_score.abs() > 90_000 {
                    let mate_in = (MATE - best_score.abs()) / 2 + 1;
                    if best_score > 0 { format!("mate {}", mate_in) } else { format!("mate -{}", mate_in) }
                } else {
                    format!("cp {}", best_score)
                };
                let pv = format!("{}{}", coord_to_square(best_move[0], best_move[1]),
                                         coord_to_square(best_move[2], best_move[3]));
                eprintln!("info depth {} score {} nodes {} nps {} pv {}", depth, score_str, total_nodes, nps, pv);
                println!("info depth {} score {} nodes {} nps {} pv {}", depth, score_str, total_nodes, nps, pv);
            } else {
                break;
            }
        }

        let mv_str = format!("{}{}", coord_to_square(best_move[0], best_move[1]),
                                     coord_to_square(best_move[2], best_move[3]));
        (mv_str, best_score, total_nodes, start.elapsed().as_secs_f64())
    }
}