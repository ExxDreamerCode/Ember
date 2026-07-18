#[cfg(feature = "decision-trace")]
use crate::board::board_to_fen;
use crate::board::{
    bit, is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    move_to_uci, piece_from_char, piece_type, sq, sq_c, BoardState, Move, BK, BP, BQ, BR, EMPTY_SQ,
    INF, MATE, MAX_HALF_MOVE_CLOCK, MAX_PLY, NO_MOVE, WK, WP, WQ, WR,
};
use crate::book::OpeningBook;
use crate::movegen::{apply_move, generate_moves};
use crate::search::{lazy_smp_search, LazySmpSearchLimits, Searcher};
#[cfg(feature = "decision-trace")]
use crate::trace::{DecisionTrace, DepthInfo, TraceLogger};
use crate::tt::SharedTT;
use crate::zobrist::compute_hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_HASH_MB: usize = 256;

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

fn root_non_king_piece_count(st: &BoardState) -> u32 {
    (0..12)
        .filter(|&pi| piece_type(pi as u8) != 5)
        .map(|pi| st.bb[pi].count_ones())
        .sum()
}

fn root_side_has_major(st: &BoardState, white: bool) -> bool {
    let rook = if white { WR } else { BR };
    let queen = if white { WQ } else { BQ };
    (st.bb[rook] | st.bb[queen]) != 0
}

fn root_promotion_race(st: &BoardState) -> bool {
    let mut white_pawns = st.bb[WP];
    while white_pawns != 0 {
        let sq = white_pawns.trailing_zeros() as usize;
        if sq / 8 <= 2 {
            return true;
        }
        white_pawns &= white_pawns - 1;
    }

    let mut black_pawns = st.bb[BP];
    while black_pawns != 0 {
        let sq = black_pawns.trailing_zeros() as usize;
        if sq / 8 >= 5 {
            return true;
        }
        black_pawns &= black_pawns - 1;
    }

    false
}

fn root_move_gives_check(st: &BoardState, mv: Move) -> bool {
    let mut after = *st;
    apply_move(
        &mut after,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
    let opp_ks = after.king_sq(after.w);
    is_attacked(&after.bb, opp_ks, !after.w)
}

fn root_move_is_capture(st: &BoardState, mv: Move) -> bool {
    let to = move_to(mv);
    let from = move_from(mv);
    let fpi = st.mailbox[from];
    let tpi = st.mailbox[to];
    if tpi != EMPTY_SQ {
        return fpi == EMPTY_SQ || (tpi < 6) != (fpi < 6);
    }

    fpi != EMPTY_SQ && piece_type(fpi) == 0 && Some(to) == st.ep && move_sc(mv) != move_ec(mv)
}

fn root_move_is_promotion(st: &BoardState, mv: Move) -> bool {
    if move_promotion(mv) != 0 {
        return true;
    }
    let from = move_from(mv);
    let fpi = st.mailbox[from];
    fpi != EMPTY_SQ && piece_type(fpi) == 0 && (move_er(mv) == 0 || move_er(mv) == 7)
}

fn root_piece_value(pi: u8) -> i32 {
    if pi == EMPTY_SQ {
        return 0;
    }
    match piece_type(pi) {
        0 => 100,
        1 => 325,
        2 => 340,
        3 => 500,
        4 => 950,
        _ => 0,
    }
}

fn root_forcing_score(st: &BoardState, mv: Move) -> Option<i32> {
    let gives_check = root_move_gives_check(st, mv);
    let is_promo = root_move_is_promotion(st, mv);
    let is_capture = root_move_is_capture(st, mv);
    if !gives_check && !is_promo && !is_capture {
        return None;
    }

    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    let victim = st.mailbox[to];
    let mut score = 0;
    if gives_check {
        score += 4_000_000;
    }
    if is_promo {
        score += 2_000_000;
    }
    if is_capture {
        score += 1_000_000 + root_piece_value(victim) * 10 - root_piece_value(attacker);
    }
    Some(score)
}

fn root_rook_invasion_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    if attacker == EMPTY_SQ || piece_type(attacker) != 3 {
        return None;
    }

    let target_row = if st.w { 1 } else { 6 };
    if to / 8 != target_row {
        return None;
    }

    if root_non_king_piece_count(st) > 8 && !rook_attacks_enemy_non_pawn_on_rank(st, to, attacker) {
        return None;
    }

    Some(600_000)
}

fn rook_attacks_enemy_non_pawn_on_rank(st: &BoardState, rook_sq: usize, rook: u8) -> bool {
    let moving_white = rook < 6;
    let row = rook_sq / 8;
    let col = rook_sq % 8;

    for c in (0..col).rev() {
        let pi = st.mailbox[row * 8 + c];
        if pi == EMPTY_SQ {
            continue;
        }
        return (pi < 6) != moving_white && piece_type(pi) != 0;
    }

    for c in (col + 1)..8 {
        let pi = st.mailbox[row * 8 + c];
        if pi == EMPTY_SQ {
            continue;
        }
        return (pi < 6) != moving_white && piece_type(pi) != 0;
    }

    false
}

fn root_order_score(st: &BoardState, mv: Move, preferred: Move) -> i32 {
    let mut score = root_forcing_score(st, mv).unwrap_or(0);
    score += root_rook_invasion_score(st, mv).unwrap_or(0);
    if mv == preferred {
        score += 500_000;
    }
    score
}

fn root_enemy_pawn_attacks_square(st: &BoardState, target: usize) -> bool {
    let row = target / 8;
    let col = target % 8;
    let pawn = if st.w { BP } else { WP };

    if st.w {
        if row == 0 {
            return false;
        }
        let pawn_row = row - 1;
        if col > 0 && (st.bb[pawn] & bit(pawn_row * 8 + col - 1)) != 0 {
            return true;
        }
        if col < 7 && (st.bb[pawn] & bit(pawn_row * 8 + col + 1)) != 0 {
            return true;
        }
    } else {
        if row == 7 {
            return false;
        }
        let pawn_row = row + 1;
        if col > 0 && (st.bb[pawn] & bit(pawn_row * 8 + col - 1)) != 0 {
            return true;
        }
        if col < 7 && (st.bb[pawn] & bit(pawn_row * 8 + col + 1)) != 0 {
            return true;
        }
    }

    false
}

fn root_minor_king_zone_capture(st: &BoardState, mv: Move) -> bool {
    if move_promotion(mv) != 0 {
        return false;
    }

    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    let victim = st.mailbox[to];
    if attacker == EMPTY_SQ || victim == EMPTY_SQ {
        return false;
    }
    if (attacker < 6) != st.w || (victim < 6) == st.w {
        return false;
    }

    let attacker_type = piece_type(attacker);
    if attacker_type != 1 && attacker_type != 2 {
        return false;
    }
    if (st.bb[WQ] | st.bb[BQ]) != 0 {
        return false;
    }
    let own_can_castle = if st.w {
        st.cr[0] || st.cr[1]
    } else {
        st.cr[2] || st.cr[3]
    };
    if !own_can_castle {
        return false;
    }

    let target_row = to / 8;
    if (st.w && target_row > 3) || (!st.w && target_row < 4) {
        return false;
    }

    let king = st.king_sq(!st.w);
    let row_dist = target_row.abs_diff(king / 8);
    let col_dist = (to % 8).abs_diff(king % 8);
    if row_dist.max(col_dist) > 2 {
        return false;
    }

    !root_enemy_pawn_attacks_square(st, to)
}

fn sort_tactical_root_moves(st: &BoardState, moves: &[Move], preferred: Move) -> Option<Vec<Move>> {
    let mut scored = Vec::with_capacity(moves.len());
    let mut has_tactical = false;
    for (idx, &mv) in moves.iter().enumerate() {
        let tactical = root_minor_king_zone_capture(st, mv);
        has_tactical |= tactical;
        let bonus = if tactical { 1_500_000 } else { 0 };
        scored.push((root_order_score(st, mv, preferred) + bonus, idx, mv));
    }
    if !has_tactical {
        return None;
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    Some(scored.into_iter().map(|(_, _, mv)| mv).collect())
}

fn sort_sparse_root_moves(st: &BoardState, moves: &[Move], preferred: Move) -> Option<Vec<Move>> {
    let sparse_endgame = root_non_king_piece_count(st) <= 8
        && (root_side_has_major(st, st.w) || root_promotion_race(st));
    let has_rook_invasion = moves
        .iter()
        .any(|&mv| root_rook_invasion_score(st, mv).is_some());

    if !sparse_endgame && !has_rook_invasion {
        return None;
    }

    let mut scored: Vec<(i32, usize, Move)> = moves
        .iter()
        .enumerate()
        .map(|(idx, &mv)| (root_order_score(st, mv, preferred), idx, mv))
        .collect();
    let has_priority = scored.iter().any(|(score, _, _)| *score >= 600_000);
    if !has_priority {
        return None;
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    Some(scored.into_iter().map(|(_, _, mv)| mv).collect())
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(DEFAULT_HASH_MB));
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
        e.searcher.tt_mb = DEFAULT_HASH_MB;
        e.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        e
    }

    pub fn new_with(
        st: BoardState,
        searcher: Searcher,
        shared_tt: Arc<SharedTT>,
        num_threads: usize,
        stopped: Arc<AtomicBool>,
        book: Option<OpeningBook>,
    ) -> Self {
        Engine {
            st,
            searcher,
            shared_tt,
            num_threads,
            stopped,
            book,
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
        let soft_time_limit = soft_time_limit.min(time_limit);
        self.searcher.refresh_nnue_net();
        self.searcher.refresh_search_backend();
        let moves = generate_moves(&self.st, self.st.w, &self.st.cr, self.st.ep);
        #[cfg(feature = "decision-trace")]
        let root_fen = board_to_fen(&self.st);
        #[cfg(feature = "decision-trace")]
        let legal_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&self.st, *mv)).collect();
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
                    let mv_str = move_to_uci(&self.st, bm);
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
            let threaded_moves = sort_tactical_root_moves(&self.st, &moves, NO_MOVE)
                .or_else(|| sort_sparse_root_moves(&self.st, &moves, NO_MOVE))
                .unwrap_or_else(|| moves.clone());

            let (best_move, best_score, best_depth, total_nodes) = lazy_smp_search(
                Arc::clone(&self.shared_tt),
                &self.st,
                &threaded_moves,
                LazySmpSearchLimits {
                    soft_time: soft_time_limit,
                    hard_time: time_limit,
                    depth: depth_limit,
                },
                self.num_threads,
                &self.searcher,
            );

            let mv_str = move_to_uci(&self.st, best_move);
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
                let use_syzygy_dtz =
                    depth >= 2 && self.searcher.syzygy.can_probe_dtz_after_one_move(&self.st);
                let sorted = if use_syzygy_dtz {
                    let mut with_dtz: Vec<(i32, Move)> = moves
                        .iter()
                        .map(|&mv| {
                            let old = self.st;
                            apply_move(
                                &mut self.st,
                                move_sr(mv),
                                move_sc(mv),
                                move_er(mv),
                                move_ec(mv),
                                move_promotion(mv),
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
                } else if let Some(s) = sort_tactical_root_moves(&self.st, &moves, asp_best) {
                    s
                } else if let Some(s) = sort_sparse_root_moves(&self.st, &moves, asp_best) {
                    s
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
                    let root_ext = i32::from(root_rook_invasion_score(&old, mv).is_some());

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
                let pv_line =
                    crate::search::extract_pv_line(&self.searcher.shared_tt, &self.st, best_move);
                let pv_str = pv_line
                    .iter()
                    .map(|m| move_to_uci(&self.st, *m))
                    .collect::<Vec<_>>()
                    .join(" ");
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
                if elapsed >= soft_time_limit {
                    break;
                }
            } else {
                break;
            }
        }

        let mv_str = move_to_uci(&self.st, best_move);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_from_fen(fen: &str) -> Engine {
        let mut engine = Engine::new();
        engine.book = None;
        engine.set_fen(fen);
        engine
    }

    fn root_moves(engine: &Engine) -> Vec<Move> {
        generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
    }

    #[test]
    fn sparse_endgame_root_ordering_prioritizes_reported_mating_check() {
        let engine = engine_from_fen("8/5k2/3Q4/7p/8/1p6/3p1P1P/3B2K1 w - - 52 78");
        let moves = root_moves(&engine);
        let sorted = sort_sparse_root_moves(&engine.st, &moves, NO_MOVE).expect("sparse ordering");
        let mating_check = *moves
            .iter()
            .find(|mv| move_to_uci(&engine.st, **mv) == "d1h5")
            .expect("reported mating check is legal");
        let quiet_start = sorted
            .iter()
            .position(|mv| root_forcing_score(&engine.st, *mv).is_none())
            .unwrap_or(sorted.len());
        let mating_check_pos = sorted
            .iter()
            .position(|mv| *mv == mating_check)
            .expect("reported mating check remains in root moves");

        assert!(root_forcing_score(&engine.st, mating_check).unwrap() >= 4_000_000);
        assert!(mating_check_pos < quiet_start);
    }

    #[test]
    fn sparse_root_ordering_does_not_activate_in_opening_positions() {
        let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let moves = root_moves(&engine);

        assert!(sort_sparse_root_moves(&engine.st, &moves, NO_MOVE).is_none());
    }

    #[test]
    fn sparse_root_ordering_handles_promotion_race_without_major_piece() {
        let engine = engine_from_fen("8/P4k2/8/8/8/8/8/6K1 w - - 0 1");
        let moves = root_moves(&engine);
        let sorted = sort_sparse_root_moves(&engine.st, &moves, NO_MOVE).expect("promotion race");

        assert!(sorted
            .first()
            .is_some_and(|mv| move_to_uci(&engine.st, *mv).starts_with("a7a8")));
    }

    #[test]
    fn sparse_endgame_root_ordering_is_used_by_search() {
        for threads in [1usize, 2] {
            let mut engine = engine_from_fen("8/5k2/3Q4/7p/8/1p6/3p1P1P/3B2K1 w - - 52 78");
            engine.num_threads = threads;
            let (best_move, _, _, _) = engine.find_best_move(2.0, 1);
            let best = root_moves(&engine)
                .into_iter()
                .find(|mv| move_to_uci(&engine.st, *mv) == best_move)
                .expect("search best move remains legal");

            assert!(
                root_forcing_score(&engine.st, best).is_some(),
                "threads={threads} should pick a forcing sparse-endgame root move, got {best_move}"
            );
        }
    }
}
