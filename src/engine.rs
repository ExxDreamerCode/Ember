#[cfg(feature = "decision-trace")]
use crate::board::board_to_fen;
use crate::board::{
    bit, is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    move_to_uci, piece_from_char, piece_type, sq, sq_c, BoardState, Move, BK, BP, BQ, BR, EMPTY_SQ,
    INF, MATE, MAX_HALF_MOVE_CLOCK, NO_MOVE, WK, WP, WQ, WR,
};
use crate::book::{
    OpeningBook, DEFAULT_BOOK_MIN_MOVE_WEIGHT, DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE,
};
use crate::movegen::{apply_move, generate_moves};
use crate::search::{lazy_smp_search, LazySmpPool, LazySmpSearchLimits, Searcher};
use crate::time_management::{iteration_time_decision, threads_for_time_budget, IterationTiming};
#[cfg(feature = "decision-trace")]
use crate::trace::{DecisionTrace, DepthInfo, TraceLogger};
use crate::tt::SharedTT;
use crate::zobrist::compute_hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_HASH_MB: usize = 256;

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
    pub book_min_move_weight: u16,
    pub book_min_move_weight_permille: u16,
    #[cfg(feature = "decision-trace")]
    pub trace: TraceLogger,
}

pub struct EngineBookConfig {
    pub book: Option<OpeningBook>,
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
            min_move_weight,
            min_move_weight_permille,
        }
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

fn root_has_queen(st: &BoardState) -> bool {
    (st.bb[WQ] | st.bb[BQ]) != 0
}

fn root_promotion_race(st: &BoardState) -> bool {
    let mut white_pawns = st.bb[WP];
    while white_pawns != 0 {
        let square = white_pawns.trailing_zeros() as usize;
        if square / 8 <= 2 {
            return true;
        }
        white_pawns &= white_pawns - 1;
    }

    let mut black_pawns = st.bb[BP];
    while black_pawns != 0 {
        let square = black_pawns.trailing_zeros() as usize;
        if square / 8 >= 5 {
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

fn root_move_gives_checkmate(st: &BoardState, mv: Move) -> bool {
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
        && generate_moves(&after, after.w, &after.cr, after.ep).is_empty()
}

fn root_forced_mate_reply_count(st: &BoardState, mv: Move) -> Option<usize> {
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
    if !is_attacked(&after.bb, opp_ks, !after.w) {
        return None;
    }

    let replies = generate_moves(&after, after.w, &after.cr, after.ep);
    if replies.len() > 2 {
        return None;
    }
    if replies.is_empty() {
        return Some(0);
    }

    for reply in replies.iter().copied() {
        let mut after_reply = after;
        apply_move(
            &mut after_reply,
            move_sr(reply),
            move_sc(reply),
            move_er(reply),
            move_ec(reply),
            move_promotion(reply),
        );
        if !generate_moves(&after_reply, after_reply.w, &after_reply.cr, after_reply.ep)
            .into_iter()
            .any(|mate| root_move_gives_checkmate(&after_reply, mate))
        {
            return None;
        }
    }

    Some(replies.len())
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

fn root_reduced_rook_check_capture(st: &BoardState, mv: Move) -> bool {
    let attacker = st.mailbox[move_from(mv)];
    root_non_king_piece_count(st) <= 12
        && !root_has_queen(st)
        && attacker != EMPTY_SQ
        && piece_type(attacker) == 3
        && root_move_is_capture(st, mv)
        && root_move_gives_check(st, mv)
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

fn root_depth_extension(st: &BoardState, mv: Move) -> i32 {
    if root_reduced_rook_check_capture(st, mv) {
        3
    } else {
        i32::from(root_rook_invasion_score(st, mv).is_some())
    }
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
    if root_minor_king_zone_capture(st, mv) {
        score += 1_500_000;
    }
    if mv == preferred {
        score += 500_000;
    }
    score
}

fn root_mating_check_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let reply_count = root_forced_mate_reply_count(st, mv)?;
    let mut score = 8_000_000 - reply_count as i32 * 100_000;
    if root_move_is_promotion(st, mv) {
        score += 2_000_000;
    }
    if root_move_is_capture(st, mv) {
        let from = move_from(mv);
        let to = move_to(mv);
        score +=
            1_000_000 + root_piece_value(st.mailbox[to]) * 10 - root_piece_value(st.mailbox[from]);
    }
    Some(score)
}

fn root_checking_non_pawn_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let to = move_to(mv);
    let victim = st.mailbox[to];
    if victim == EMPTY_SQ || piece_type(victim) == 0 {
        return None;
    }
    if !root_move_is_capture(st, mv) {
        return None;
    }

    let attacker = st.mailbox[move_from(mv)];
    if attacker == EMPTY_SQ {
        return None;
    }
    let attacker_type = piece_type(attacker);
    let victim_type = piece_type(victim);
    if victim_type == 4 || (attacker_type == 3 && victim_type == 3) {
        return None;
    }
    if attacker_type == 4 && victim_type == 1 {
        return None;
    }

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
    if !is_attacked(&after.bb, opp_ks, !after.w) {
        return None;
    }
    if generate_moves(&after, after.w, &after.cr, after.ep).len() > 3 {
        return None;
    }

    Some(6_000_000 + root_piece_value(victim) * 10 - root_piece_value(attacker))
}

fn root_quiet_bishop_knight_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    let victim = st.mailbox[to];
    if attacker == EMPTY_SQ || victim == EMPTY_SQ {
        return None;
    }
    if piece_type(attacker) != 2 || piece_type(victim) != 1 {
        return None;
    }
    if !root_move_is_capture(st, mv) || root_move_gives_check(st, mv) {
        return None;
    }

    let pawn_safe_bonus = if root_enemy_pawn_attacks_square(st, to) {
        0
    } else {
        100_000
    };
    Some(5_000_000 + pawn_safe_bonus)
}

fn root_checking_slider_pawn_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    let victim = st.mailbox[to];
    if attacker == EMPTY_SQ || victim == EMPTY_SQ {
        return None;
    }
    if !matches!(piece_type(attacker), 2 | 3) || piece_type(victim) != 0 {
        return None;
    }
    if !root_move_is_capture(st, mv) {
        return None;
    }

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
    if !is_attacked(&after.bb, opp_ks, !after.w) {
        return None;
    }
    if generate_moves(&after, after.w, &after.cr, after.ep).len() > 3 {
        return None;
    }

    Some(5_500_000 + root_piece_value(victim) * 10 - root_piece_value(attacker))
}

fn root_quiet_queen_check_reply_count(st: &BoardState, mv: Move) -> Option<usize> {
    let attacker = st.mailbox[move_from(mv)];
    if attacker == EMPTY_SQ || piece_type(attacker) != 4 || root_move_is_capture(st, mv) {
        return None;
    }

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
    if !is_attacked(&after.bb, opp_ks, !after.w) {
        return None;
    }

    Some(generate_moves(&after, after.w, &after.cr, after.ep).len())
}

fn root_queen_pawn_check_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    let victim = st.mailbox[to];
    if attacker == EMPTY_SQ || victim == EMPTY_SQ {
        return None;
    }
    if piece_type(attacker) != 4 || piece_type(victim) != 0 || !root_move_is_capture(st, mv) {
        return None;
    }

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
    if !is_attacked(&after.bb, opp_ks, !after.w) {
        return None;
    }
    if generate_moves(&after, after.w, &after.cr, after.ep).len() != 2 {
        return None;
    }

    Some(5_600_000 + root_piece_value(victim) * 10 - root_piece_value(attacker))
}

fn sort_root_moves(st: &BoardState, moves: &[Move], preferred: Move) -> Vec<Move> {
    let sparse_endgame = root_non_king_piece_count(st) <= 8
        && (root_side_has_major(st, st.w) || root_promotion_race(st));
    let has_rook_invasion = moves
        .iter()
        .any(|&mv| root_rook_invasion_score(st, mv).is_some());
    let has_reduced_rook_check = root_non_king_piece_count(st) <= 12
        && !root_has_queen(st)
        && moves.iter().any(|&mv| {
            let attacker = st.mailbox[move_from(mv)];
            attacker != EMPTY_SQ && piece_type(attacker) == 3 && root_move_gives_check(st, mv)
        });
    let has_minor_tactic = moves.iter().any(|&mv| root_minor_king_zone_capture(st, mv));
    let has_queen_capture = moves
        .iter()
        .any(|&mv| root_move_is_capture(st, mv) && piece_type(st.mailbox[move_to(mv)]) == 4);
    let use_tactical_order = has_minor_tactic
        || has_queen_capture
        || has_reduced_rook_check
        || ((sparse_endgame || has_rook_invasion)
            && moves
                .iter()
                .any(|&mv| root_order_score(st, mv, NO_MOVE) >= 600_000));
    let mating_check_scores: Vec<i32> = if use_tactical_order {
        Vec::new()
    } else {
        moves
            .iter()
            .map(|&mv| root_mating_check_order_score(st, mv).unwrap_or(0))
            .collect()
    };
    let use_mating_check_order = mating_check_scores.iter().any(|&score| score != 0);
    let checking_non_pawn_capture_scores: Vec<i32> = if use_tactical_order || use_mating_check_order
    {
        Vec::new()
    } else {
        moves
            .iter()
            .map(|&mv| root_checking_non_pawn_capture_order_score(st, mv).unwrap_or(0))
            .collect()
    };
    let use_checking_non_pawn_capture_order = checking_non_pawn_capture_scores
        .iter()
        .any(|&score| score != 0);
    let quiet_bishop_knight_capture_scores: Vec<i32> =
        if use_tactical_order || use_mating_check_order || use_checking_non_pawn_capture_order {
            Vec::new()
        } else {
            moves
                .iter()
                .map(|&mv| root_quiet_bishop_knight_capture_order_score(st, mv).unwrap_or(0))
                .collect()
        };
    let use_quiet_bishop_knight_capture_order = quiet_bishop_knight_capture_scores
        .iter()
        .any(|&score| score != 0);
    let checking_pawn_capture_scores: Vec<i32> = if use_tactical_order
        || use_mating_check_order
        || use_checking_non_pawn_capture_order
        || use_quiet_bishop_knight_capture_order
    {
        Vec::new()
    } else {
        moves
            .iter()
            .map(|&mv| root_checking_slider_pawn_capture_order_score(st, mv).unwrap_or(0))
            .collect()
    };
    let use_checking_pawn_capture_order =
        checking_pawn_capture_scores.iter().any(|&score| score != 0);
    let queen_pawn_check_capture_scores: Vec<i32> = if use_tactical_order
        || use_mating_check_order
        || use_checking_non_pawn_capture_order
        || use_quiet_bishop_knight_capture_order
        || use_checking_pawn_capture_order
    {
        Vec::new()
    } else if moves
        .iter()
        .filter(|&&mv| root_move_gives_check(st, mv))
        .count()
        == 1
    {
        moves
            .iter()
            .map(|&mv| root_queen_pawn_check_capture_order_score(st, mv).unwrap_or(0))
            .collect()
    } else {
        vec![0; moves.len()]
    };
    let use_queen_pawn_check_capture_order = queen_pawn_check_capture_scores
        .iter()
        .any(|&score| score != 0);
    let quiet_queen_check_scores: Vec<i32> = if use_tactical_order
        || use_mating_check_order
        || use_checking_non_pawn_capture_order
        || use_quiet_bishop_knight_capture_order
        || use_checking_pawn_capture_order
        || use_queen_pawn_check_capture_order
    {
        Vec::new()
    } else {
        let has_checking_non_pawn_capture = moves.iter().any(|&mv| {
            let victim = st.mailbox[move_to(mv)];
            victim != EMPTY_SQ
                && piece_type(victim) != 0
                && root_move_is_capture(st, mv)
                && root_move_gives_check(st, mv)
        });
        let reply_counts: Vec<Option<usize>> = moves
            .iter()
            .map(|&mv| root_quiet_queen_check_reply_count(st, mv))
            .collect();
        let quiet_queen_check_count = reply_counts.iter().flatten().count();
        let best_reply_count = reply_counts
            .iter()
            .flatten()
            .copied()
            .filter(|&reply_count| (2..=3).contains(&reply_count))
            .min();
        if let Some(best_reply_count) = best_reply_count {
            let narrow_enough = !has_checking_non_pawn_capture
                && (quiet_queen_check_count > 1 || best_reply_count == 2);
            if reply_counts
                .iter()
                .filter(|&&reply_count| reply_count == Some(best_reply_count))
                .count()
                == 1
                && narrow_enough
            {
                reply_counts
                    .iter()
                    .map(|&reply_count| {
                        if reply_count == Some(best_reply_count) {
                            5_250_000 - best_reply_count as i32 * 100_000
                        } else {
                            0
                        }
                    })
                    .collect()
            } else {
                vec![0; moves.len()]
            }
        } else {
            vec![0; moves.len()]
        }
    };
    let use_quiet_queen_check_order = quiet_queen_check_scores.iter().any(|&score| score != 0);

    if !use_tactical_order
        && !use_mating_check_order
        && !use_checking_non_pawn_capture_order
        && !use_quiet_bishop_knight_capture_order
        && !use_checking_pawn_capture_order
        && !use_queen_pawn_check_capture_order
        && !use_quiet_queen_check_order
    {
        let mut ordered = moves.to_vec();
        if let Some(position) = ordered.iter().position(|&mv| mv == preferred) {
            ordered.swap(0, position);
        }
        return ordered;
    }

    let mut scored: Vec<(i32, usize, Move)> = moves
        .iter()
        .enumerate()
        .map(|(idx, &mv)| {
            let score = if use_tactical_order {
                root_order_score(st, mv, preferred)
            } else if use_mating_check_order {
                mating_check_scores[idx] + i32::from(mv == preferred) * 500_000
            } else {
                let fallback_score = if use_checking_non_pawn_capture_order {
                    checking_non_pawn_capture_scores[idx]
                } else if use_quiet_bishop_knight_capture_order {
                    quiet_bishop_knight_capture_scores[idx]
                } else if use_checking_pawn_capture_order {
                    checking_pawn_capture_scores[idx]
                } else if use_queen_pawn_check_capture_order {
                    queen_pawn_check_capture_scores[idx]
                } else {
                    quiet_queen_check_scores[idx]
                };
                fallback_score + i32::from(mv == preferred) * 500_000
            };
            (score, idx, mv)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, mv)| mv).collect()
}

fn tt_root_move(searcher: &Searcher, st: &BoardState, moves: &[Move]) -> Move {
    searcher
        .shared_tt
        .get_depth(st.hash)
        .and_then(|(_, _, _, best_move)| best_move)
        .filter(|best_move| moves.contains(best_move))
        .unwrap_or(NO_MOVE)
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

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let shared_tt = Arc::new(SharedTT::new(DEFAULT_HASH_MB));
        let search_pool = Arc::new(LazySmpPool::new());
        let mut e = Engine {
            st: BoardState::empty(),
            searcher: Searcher::new(Arc::clone(&shared_tt), Arc::clone(&stopped)),
            shared_tt,
            search_pool,
            num_threads: 1,
            stopped,
            book: None,
            book_min_move_weight: DEFAULT_BOOK_MIN_MOVE_WEIGHT,
            book_min_move_weight_permille: DEFAULT_BOOK_MIN_MOVE_WEIGHT_PERMILLE,
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
        start: Instant,
    ) -> (String, i32, u64, f64) {
        self.searcher.stopped.store(false, Ordering::SeqCst);
        self.searcher.pondering.store(false, Ordering::SeqCst);
        self.find_best_move_with_time_limits_prepared_started_at(
            soft_time_limit,
            time_limit,
            depth_limit,
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
            SearchTimerStart::AfterSetup,
        )
    }

    pub fn find_best_move_with_time_limits_prepared_started_at(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
        start: Instant,
    ) -> (String, i32, u64, f64) {
        self.find_best_move_with_time_limits_prepared_with_timer(
            soft_time_limit,
            time_limit,
            depth_limit,
            SearchTimerStart::BeforeSetup(start),
        )
    }

    fn find_best_move_with_time_limits_prepared_with_timer(
        &mut self,
        soft_time_limit: f64,
        time_limit: f64,
        depth_limit: i32,
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
                if let Some(choice) = book.pick_move_with_confidence(
                    &self.st,
                    &moves,
                    self.book_min_move_weight,
                    self.book_min_move_weight_permille,
                ) {
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
                    start,
                },
                search_threads,
                &mut self.searcher,
            );

            let mv_str = move_to_uci(&self.st, best_move);
            let elapsed = start.elapsed().as_secs_f64();
            self.searcher
                .update_correction_history(&self.st, best_score, best_depth);
            return (mv_str, best_score, total_nodes, elapsed);
        }

        self.searcher.prepare_for_search();
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

                let mut cur_best = sorted[0];
                let mut cur_score = -INF;
                let mut cur_best_nodes = 0u64;
                let mut loop_alpha = alpha;

                for &mv in &sorted {
                    if !self.searcher.pondering.load(Ordering::Relaxed)
                        && start.elapsed().as_secs_f64() > time_limit
                    {
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
                    let root_ext = root_depth_extension(&old, mv);
                    let move_nodes_before = nd;

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

                    self.searcher.rep_stack.pop();
                    self.searcher.rep_stack_len -= 1;
                    self.st = old;

                    if self.searcher.stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    if score > cur_score {
                        cur_score = score;
                        cur_best = mv;
                        cur_best_nodes = move_nodes;
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

            if self.searcher.stopped.load(Ordering::Relaxed) {
                break;
            }
            total_nodes += nd;
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
                self.searcher.shared_tt.store(
                    self.st.hash,
                    depth,
                    best_score,
                    crate::tt::TT_EXACT,
                    Some(best_move),
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
                if !self.searcher.pondering.load(Ordering::Relaxed) && time_decision.stop {
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
        let reply = self
            .shared_tt
            .get_depth(child.hash)
            .and_then(|(_, _, _, best)| best)?;
        generate_moves(&child, child.w, &child.cr, child.ep)
            .contains(&reply)
            .then(|| move_to_uci(&child, reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn engine_from_fen(fen: &str) -> Engine {
        let mut engine = Engine::new();
        engine.book = None;
        engine.set_fen(fen);
        engine
    }

    fn root_moves(engine: &Engine) -> Vec<Move> {
        generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep)
    }

    fn root_move(engine: &Engine, uci: &str) -> Move {
        root_moves(engine)
            .into_iter()
            .find(|mv| move_to_uci(&engine.st, *mv) == uci)
            .unwrap_or_else(|| panic!("expected legal root move {uci}"))
    }

    fn play_uci(engine: &mut Engine, uci: &str) {
        let bytes = uci.as_bytes();
        assert!(bytes.len() >= 4, "invalid UCI move: {uci}");
        let promotion = bytes
            .get(4)
            .map_or(0, |piece| match piece.to_ascii_lowercase() {
                b'q' => b'Q',
                b'r' => b'R',
                b'b' => b'B',
                b'n' => b'N',
                _ => 0,
            });
        assert!(
            engine.make_move_uci(
                8 - usize::from(bytes[1] - b'0'),
                usize::from(bytes[0] - b'a'),
                8 - usize::from(bytes[3] - b'0'),
                usize::from(bytes[2] - b'a'),
                promotion,
            ),
            "expected legal move {uci}"
        );
    }

    #[test]
    fn root_ordering_prioritizes_the_missed_rook_clearance() {
        let engine = engine_from_fen("8/5k2/2pp2p1/5pP1/P2P4/3n4/2r5/1KB4R b - - 4 46");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);

        assert_eq!(move_to_uci(&engine.st, ordered[0]), "c2c1");
    }

    #[test]
    fn reduced_rook_check_capture_gets_the_tactical_root_extension() {
        let engine = engine_from_fen("8/5k2/2pp2p1/5pP1/P2P4/3n4/2r5/1KB4R b - - 4 46");
        let clearance = root_move(&engine, "c2c1");
        let non_capture = root_move(&engine, "c2c4");

        assert!(root_reduced_rook_check_capture(&engine.st, clearance));
        assert_eq!(root_depth_extension(&engine.st, clearance), 3);
        assert!(!root_reduced_rook_check_capture(&engine.st, non_capture));
        assert_eq!(root_depth_extension(&engine.st, non_capture), 0);
    }

    #[test]
    fn root_ordering_prioritizes_a_missed_mating_check() {
        let mut engine = engine_from_fen("1rb2rk1/q5P1/4p2p/3p3p/3P1P2/2P5/2QK3P/3R2R1 b - - 0 29");
        play_uci(&mut engine, "f8f7");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let mating_check = root_move(&engine, "c2h7");
        let quiet_move = root_move(&engine, "c2g6");

        assert!(root_move_gives_check(&engine.st, mating_check));
        assert!(!root_move_gives_check(&engine.st, quiet_move));
        assert_eq!(
            root_forced_mate_reply_count(&engine.st, mating_check),
            Some(1)
        );
        assert_eq!(
            root_mating_check_order_score(&engine.st, mating_check),
            Some(7_900_000)
        );
        assert_eq!(root_depth_extension(&engine.st, mating_check), 0);
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "c2h7");
    }

    #[test]
    fn root_ordering_prioritizes_checking_non_pawn_capture() {
        let mut engine = engine_from_fen("r4k1r/1pp2p2/p2p3p/3N4/3P2q1/8/PPP5/1K2Q1NR b - - 1 23");
        play_uci(&mut engine, "a8e8");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let checking_capture = root_move(&engine, "e1e8");

        assert!(root_move_gives_check(&engine.st, checking_capture));
        assert!(root_move_is_capture(&engine.st, checking_capture));
        assert_eq!(
            root_checking_non_pawn_capture_order_score(&engine.st, checking_capture),
            Some(6_004_050)
        );
        assert_eq!(root_depth_extension(&engine.st, checking_capture), 0);
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "e1e8");
    }

    #[test]
    fn root_ordering_ignores_noisy_checking_captures() {
        let mut rook_trade =
            engine_from_fen("4R1k1/p4r1p/1pp2rp1/8/5B1q/4QP1P/P1P2PK1/8 b - - 2 28");
        play_uci(&mut rook_trade, "f7f8");
        let equal_rook_check = root_move(&rook_trade, "e8f8");

        assert!(root_move_gives_check(&rook_trade.st, equal_rook_check));
        assert!(root_move_is_capture(&rook_trade.st, equal_rook_check));
        assert_eq!(
            root_checking_non_pawn_capture_order_score(&rook_trade.st, equal_rook_check),
            None
        );

        let mut queen_harvest =
            engine_from_fen("r2q1rk1/pbpn1p2/1p1bpn1Q/8/8/1B1P1NN1/PPP2PPP/R3K2R b KQ - 0 12");
        play_uci(&mut queen_harvest, "f6h7");
        let queen_takes_knight = root_move(&queen_harvest, "h6h7");
        let queen_takes_rook = root_move(&queen_harvest, "h6f8");

        assert!(root_move_gives_check(&queen_harvest.st, queen_takes_knight));
        assert!(root_move_gives_check(&queen_harvest.st, queen_takes_rook));
        assert_eq!(
            root_checking_non_pawn_capture_order_score(&queen_harvest.st, queen_takes_knight),
            None
        );
        assert_eq!(
            root_checking_non_pawn_capture_order_score(&queen_harvest.st, queen_takes_rook),
            None
        );
    }

    #[test]
    fn root_ordering_prioritizes_quiet_bishop_knight_capture() {
        let mut engine = engine_from_fen(
            "r2qkb1r/pp1nppp1/2p2n1p/3p1b2/3P4/BP2PN2/P1P2PPP/RN1QKB1R w KQkq - 2 7",
        );
        play_uci(&mut engine, "c2c4");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let bishop_takes_knight = root_move(&engine, "f5b1");

        assert!(root_move_is_capture(&engine.st, bishop_takes_knight));
        assert!(!root_move_gives_check(&engine.st, bishop_takes_knight));
        assert_eq!(
            root_quiet_bishop_knight_capture_order_score(&engine.st, bishop_takes_knight),
            Some(5_100_000)
        );
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "f5b1");
    }

    #[test]
    fn root_ordering_prioritizes_checking_slider_pawn_capture() {
        let mut engine =
            engine_from_fen("rn1qk2r/pp3ppp/3bp1b1/3p4/3Pn2N/3BB3/PPP2PPP/RN1Q1RK1 w kq - 4 10");
        play_uci(&mut engine, "h4g6");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let bishop_takes_pawn = root_move(&engine, "d6h2");

        assert!(root_move_is_capture(&engine.st, bishop_takes_pawn));
        assert!(root_move_gives_check(&engine.st, bishop_takes_pawn));
        assert_eq!(
            root_checking_slider_pawn_capture_order_score(&engine.st, bishop_takes_pawn),
            Some(5_500_660)
        );
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "d6h2");

        let mut rook_engine =
            engine_from_fen("r5k1/2p1pp2/pp4p1/1q1r4/5P2/2QP2R1/PP6/1K4R1 b - - 0 32");
        play_uci(&mut rook_engine, "d5h5");
        let moves = root_moves(&rook_engine);
        let ordered = sort_root_moves(&rook_engine.st, &moves, NO_MOVE);
        let rook_takes_pawn = root_move(&rook_engine, "g3g6");

        assert!(root_move_is_capture(&rook_engine.st, rook_takes_pawn));
        assert!(root_move_gives_check(&rook_engine.st, rook_takes_pawn));
        assert_eq!(
            root_checking_slider_pawn_capture_order_score(&rook_engine.st, rook_takes_pawn),
            Some(5_500_500)
        );
        assert_eq!(move_to_uci(&rook_engine.st, ordered[0]), "g3g6");
    }

    #[test]
    fn root_ordering_prioritizes_constrained_quiet_queen_check() {
        let mut engine = engine_from_fen("8/3k4/p6p/1p2Q1pP/3P2b1/1PP2qP1/P3p3/1K2R3 w - - 3 44");
        play_uci(&mut engine, "d4d5");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let queen_check = root_move(&engine, "f3d3");
        let wider_queen_check = root_move(&engine, "f3e4");

        assert_eq!(
            root_quiet_queen_check_reply_count(&engine.st, queen_check),
            Some(3)
        );
        assert_eq!(
            root_quiet_queen_check_reply_count(&engine.st, wider_queen_check),
            Some(4)
        );
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "f3d3");
    }

    #[test]
    fn root_ordering_prioritizes_constrained_queen_pawn_check_capture() {
        let mut engine = engine_from_fen("5rk1/R5p1/5q1p/8/3p4/1P4Q1/P3rPPP/5RK1 w - - 2 38");
        play_uci(&mut engine, "g3g4");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);
        let queen_takes_pawn = root_move(&engine, "f6f2");

        assert!(root_move_is_capture(&engine.st, queen_takes_pawn));
        assert!(root_move_gives_check(&engine.st, queen_takes_pawn));
        assert_eq!(
            root_queen_pawn_check_capture_order_score(&engine.st, queen_takes_pawn),
            Some(5_600_050)
        );
        assert_eq!(move_to_uci(&engine.st, ordered[0]), "f6f2");
    }

    #[test]
    fn root_ordering_prioritizes_the_forced_queen_recapture() {
        let engine = engine_from_fen("1r4k1/2p2p2/2np1bp1/pp6/2Q3P1/2P2N2/PPP2P2/1KBR4 b - - 0 22");
        let moves = root_moves(&engine);
        let ordered = sort_root_moves(&engine.st, &moves, NO_MOVE);

        assert_eq!(move_to_uci(&engine.st, ordered[0]), "b5c4");
    }

    #[test]
    fn legal_root_tt_move_is_promoted_between_searches() {
        let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let moves = root_moves(&engine);
        let preferred = root_move(&engine, "g1f3");
        engine
            .shared_tt
            .store(engine.st.hash, 8, 12, crate::tt::TT_EXACT, Some(preferred));

        let tt_move = tt_root_move(&engine.searcher, &engine.st, &moves);
        let ordered = sort_root_moves(&engine.st, &moves, tt_move);

        assert_eq!(tt_move, preferred);
        assert_eq!(ordered[0], preferred);
    }

    #[test]
    fn quiet_root_tt_move_stays_ahead_of_an_unrelated_capture() {
        let engine =
            engine_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let moves = root_moves(&engine);
        let preferred = root_move(&engine, "g1f3");
        let pawn_capture = root_move(&engine, "e4d5");
        let ordered = sort_root_moves(&engine.st, &moves, preferred);

        assert_eq!(ordered[0], preferred);
        assert_ne!(ordered[0], pawn_capture);
    }

    #[test]
    fn root_ordering_prioritizes_reported_mating_check() {
        let engine = engine_from_fen("8/5k2/3Q4/7p/8/1p6/3p1P1P/3B2K1 w - - 52 78");
        let moves = root_moves(&engine);
        let sorted = sort_root_moves(&engine.st, &moves, NO_MOVE);
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
    fn root_ordering_preserves_quiet_opening_order() {
        let engine = engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let moves = root_moves(&engine);

        assert_eq!(sort_root_moves(&engine.st, &moves, NO_MOVE), moves);
    }

    #[test]
    fn root_ordering_handles_promotion_race_without_major_piece() {
        let engine = engine_from_fen("8/P4k2/8/8/8/8/8/6K1 w - - 0 1");
        let moves = root_moves(&engine);
        let sorted = sort_root_moves(&engine.st, &moves, NO_MOVE);

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

    #[test]
    fn book_confidence_cutoff_rejects_weight_one_tail_move() {
        let mut engine = Engine::new();
        engine.book = Some(
            OpeningBook::load_from_bytes(crate::opening_book::BOOK_DATA, "<embedded>").unwrap(),
        );
        for mv in [
            "e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "c5d4", "c3d4", "b8c6", "g1f3",
            "g8e7", "f1d3", "e7f5", "d3f5", "e6f5", "b1c3", "f8e7",
        ] {
            play_uci(&mut engine, mv);
        }

        let (_best_move, _score, nodes, _elapsed) =
            engine.find_best_move_with_time_limits(0.01, 0.01, 1);

        assert!(
            nodes > 0,
            "the weight-one 10.h4 book tail should be rejected so search starts"
        );
    }

    #[test]
    fn caller_supplied_start_time_is_used_for_clock_search() {
        let mut engine =
            engine_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let expired_start = Instant::now() - Duration::from_millis(50);

        let (_, _, nodes, elapsed) = engine.find_best_move_with_time_limits_prepared_started_at(
            0.005,
            0.010,
            64,
            expired_start,
        );

        assert_eq!(nodes, 0, "search ignored the already-expired clock");
        assert!(
            elapsed >= 0.050,
            "reported elapsed time must include the caller's start point: {elapsed}"
        );
    }
}
