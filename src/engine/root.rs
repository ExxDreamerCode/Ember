use super::{FIFTY_MOVE_ROOT_MATERIAL_MARGIN_CP, FIFTY_MOVE_ROOT_VERIFY_NODE_LIMIT};
use crate::board::{
    bit, is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    piece_type, BoardState, Move, BK, BP, BQ, BR, EMPTY_SQ, KNIGHT_ATTACKS, NO_MOVE, WK, WP, WQ,
    WR,
};
use crate::movegen::{apply_move, generate_moves};
use crate::search::Searcher;
use std::collections::HashMap;

pub(super) fn root_non_king_piece_count(st: &BoardState) -> u32 {
    (0..12)
        .filter(|&pi| piece_type(pi as u8) != 5)
        .map(|pi| st.bb[pi].count_ones())
        .sum()
}

pub(super) fn root_side_has_major(st: &BoardState, white: bool) -> bool {
    let rook = if white { WR } else { BR };
    let queen = if white { WQ } else { BQ };
    (st.bb[rook] | st.bb[queen]) != 0
}

pub(super) fn root_has_queen(st: &BoardState) -> bool {
    (st.bb[WQ] | st.bb[BQ]) != 0
}

pub(super) fn root_promotion_race(st: &BoardState) -> bool {
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

pub(super) fn root_move_gives_check(st: &BoardState, mv: Move) -> bool {
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

pub(super) fn root_move_gives_checkmate(st: &BoardState, mv: Move) -> bool {
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

pub(super) fn root_forced_mate_reply_count(st: &BoardState, mv: Move) -> Option<usize> {
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

pub(super) fn root_move_is_capture(st: &BoardState, mv: Move) -> bool {
    let to = move_to(mv);
    let from = move_from(mv);
    let fpi = st.mailbox[from];
    let tpi = st.mailbox[to];
    if tpi != EMPTY_SQ {
        return fpi == EMPTY_SQ || (tpi < 6) != (fpi < 6);
    }

    fpi != EMPTY_SQ && piece_type(fpi) == 0 && Some(to) == st.ep && move_sc(mv) != move_ec(mv)
}

pub(super) fn root_reduced_rook_check_capture(st: &BoardState, mv: Move) -> bool {
    let attacker = st.mailbox[move_from(mv)];
    root_non_king_piece_count(st) <= 12
        && !root_has_queen(st)
        && attacker != EMPTY_SQ
        && piece_type(attacker) == 3
        && root_move_is_capture(st, mv)
        && root_move_gives_check(st, mv)
}

pub(super) fn root_move_is_promotion(st: &BoardState, mv: Move) -> bool {
    if move_promotion(mv) != 0 {
        return true;
    }
    let from = move_from(mv);
    let fpi = st.mailbox[from];
    fpi != EMPTY_SQ && piece_type(fpi) == 0 && (move_er(mv) == 0 || move_er(mv) == 7)
}

pub(super) fn root_equal_knight_capture(st: &BoardState, mv: Move) -> bool {
    if !root_move_is_capture(st, mv) {
        return false;
    }
    let attacker = st.mailbox[move_from(mv)];
    let victim = st.mailbox[move_to(mv)];
    attacker != EMPTY_SQ
        && victim != EMPTY_SQ
        && piece_type(attacker) == 1
        && piece_type(victim) == 1
}

pub(super) fn root_capture_can_be_recaptured(st: &BoardState, mv: Move) -> bool {
    let target = move_to(mv);
    let mut after = *st;
    apply_move(
        &mut after,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
    generate_moves(&after, after.w, &after.cr, after.ep)
        .into_iter()
        .any(|reply| move_to(reply) == target && root_move_is_capture(&after, reply))
}

pub(super) fn root_has_more_valuable_capture(st: &BoardState, mv: Move) -> bool {
    let victim_value = root_piece_value(st.mailbox[move_to(mv)]);
    generate_moves(st, st.w, &st.cr, st.ep)
        .into_iter()
        .any(|candidate| {
            root_move_is_capture(st, candidate)
                && root_piece_value(st.mailbox[move_to(candidate)]) > victim_value
        })
}

pub(super) fn root_piece_value(pi: u8) -> i32 {
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

pub(super) fn root_forcing_score(st: &BoardState, mv: Move) -> Option<i32> {
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

pub(super) fn root_rook_invasion_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    if attacker == EMPTY_SQ || piece_type(attacker) != 3 {
        return None;
    }
    if root_move_is_capture(st, mv) {
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

pub(super) fn root_depth_extension(st: &BoardState, mv: Move) -> i32 {
    if root_reduced_rook_check_capture(st, mv)
        || (root_equal_knight_capture(st, mv)
            && root_capture_can_be_recaptured(st, mv)
            && !root_has_more_valuable_capture(st, mv))
    {
        3
    } else {
        let attacker = st.mailbox[move_from(mv)];
        i32::from(
            root_rook_invasion_score(st, mv).is_some()
                || root_quiet_queen_rook_battery_order_score(st, mv).is_some()
                || (attacker != EMPTY_SQ
                    && piece_type(attacker) == 3
                    && root_checking_slider_pawn_capture_order_score(st, mv).is_some()),
        )
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

pub(super) fn root_order_score(st: &BoardState, mv: Move, preferred: Move) -> i32 {
    let mut score = root_forcing_score(st, mv).unwrap_or(0);
    score += root_rook_invasion_score(st, mv).unwrap_or(0);
    score += root_quiet_knight_major_fork_order_score(st, mv).unwrap_or(0);
    score += root_quiet_queen_rook_battery_order_score(st, mv).unwrap_or(0);
    if root_minor_king_zone_capture(st, mv) {
        score += 1_500_000;
    }
    if mv == preferred {
        score += 500_000;
    }
    score
}

pub(super) fn root_total_piece_count(st: &BoardState) -> u32 {
    st.bb.iter().map(|bb| bb.count_ones()).sum()
}

pub(super) fn root_material_score(st: &BoardState, white: bool) -> i32 {
    st.bb
        .iter()
        .enumerate()
        .map(|(pi, bb)| {
            let value = root_piece_value(pi as u8) * bb.count_ones() as i32;
            if (pi < 6) == white {
                value
            } else {
                -value
            }
        })
        .sum()
}

pub(super) fn root_child_after(st: &BoardState, mv: Move) -> BoardState {
    let mut after = *st;
    apply_move(
        &mut after,
        move_sr(mv),
        move_sc(mv),
        move_er(mv),
        move_ec(mv),
        move_promotion(mv),
    );
    after
}

pub(super) fn root_move_resets_halfmove(st: &BoardState, mv: Move) -> bool {
    let from = move_from(mv);
    let mover = st.mailbox[from];
    mover != EMPTY_SQ && (piece_type(mover) == 0 || root_move_is_capture(st, mv))
}

pub(super) fn root_move_preserves_fifty_move_conversion(st: &BoardState, mv: Move) -> Option<bool> {
    let attacker = st.w;
    let after = root_child_after(st, mv);
    let material_floor =
        root_material_score(&after, attacker).saturating_sub(FIFTY_MOVE_ROOT_MATERIAL_MARGIN_CP);
    fifty_move_attacker_can_force_progress(&after, attacker, material_floor)
}

fn fifty_move_attacker_can_force_progress(
    st: &BoardState,
    attacker: bool,
    material_floor: i32,
) -> Option<bool> {
    let defender = st.w;
    let mut memo = HashMap::new();
    let mut nodes = 0u32;
    fifty_move_attacker_can_force_progress_inner(
        st,
        defender,
        attacker,
        material_floor,
        &mut memo,
        &mut nodes,
    )
}

fn fifty_move_attacker_can_force_progress_inner(
    st: &BoardState,
    defender: bool,
    attacker: bool,
    material_floor: i32,
    memo: &mut HashMap<(u64, u8), bool>,
    nodes: &mut u32,
) -> Option<bool> {
    if *nodes >= FIFTY_MOVE_ROOT_VERIFY_NODE_LIMIT {
        return None;
    }
    *nodes += 1;

    let mut legal = generate_moves(st, st.w, &st.cr, st.ep);
    if legal.is_empty() {
        let king = st.king_sq(st.w);
        let checkmate = is_attacked(&st.bb, king, !st.w);
        return Some(checkmate && st.w == defender);
    }
    if st.halfmove_clock >= 100 {
        return Some(false);
    }

    let key = (st.hash, st.halfmove_clock);
    if let Some(&cached) = memo.get(&key) {
        return Some(cached);
    }

    let defender_to_move = st.w == defender;
    legal.sort_by_key(|mv| {
        let resets = root_move_resets_halfmove(st, *mv);
        if defender_to_move {
            resets
        } else {
            !resets
        }
    });
    let mut saw_unknown = false;
    for mv in legal {
        let outcome = if root_move_resets_halfmove(st, mv) {
            let child = root_child_after(st, mv);
            Some(root_material_score(&child, attacker) >= material_floor)
        } else {
            let child = root_child_after(st, mv);
            fifty_move_attacker_can_force_progress_inner(
                &child,
                defender,
                attacker,
                material_floor,
                memo,
                nodes,
            )
        };

        match (defender_to_move, outcome) {
            (true, Some(false)) => {
                memo.insert(key, false);
                return Some(false);
            }
            (true, Some(true)) => {}
            (true, None) => saw_unknown = true,
            (false, Some(true)) => {
                memo.insert(key, true);
                return Some(true);
            }
            (false, Some(false)) => {}
            (false, None) => saw_unknown = true,
        }
    }

    if saw_unknown {
        None
    } else {
        let result = defender_to_move;
        memo.insert(key, result);
        Some(result)
    }
}

pub(super) fn root_quiet_knight_major_fork_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    if attacker == EMPTY_SQ
        || piece_type(attacker) != 1
        || root_move_is_capture(st, mv)
        || root_move_is_promotion(st, mv)
    {
        return None;
    }
    let (enemy_queen, enemy_rook, enemy_king) = if st.w { (BQ, BR, BK) } else { (WQ, WR, WK) };
    let attacks = KNIGHT_ATTACKS[to];
    ((attacks & st.bb[enemy_queen]) != 0
        && (attacks & (st.bb[enemy_rook] | st.bb[enemy_king])) != 0)
        .then_some(6_500_000)
}

pub(super) fn root_quiet_queen_rook_battery_order_score(st: &BoardState, mv: Move) -> Option<i32> {
    let from = move_from(mv);
    let to = move_to(mv);
    let attacker = st.mailbox[from];
    if attacker == EMPTY_SQ
        || piece_type(attacker) != 4
        || root_move_is_capture(st, mv)
        || root_move_is_promotion(st, mv)
    {
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
    let own_rook = if st.w { WR } else { BR };
    for (dr, dc) in [
        (-1isize, -1isize),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] {
        let row = to as isize / 8 + dr;
        let col = to as isize % 8 + dc;
        if !(0..8).contains(&row) || !(0..8).contains(&col) {
            continue;
        }
        let rook_square = row as usize * 8 + col as usize;
        if after.mailbox[rook_square] as usize == own_rook
            && is_attacked(&after.bb, rook_square, !st.w)
        {
            return Some(2_000_000);
        }
    }
    None
}

pub(super) fn root_mating_check_order_score(st: &BoardState, mv: Move) -> Option<i32> {
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

pub(super) fn root_checking_non_pawn_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
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

pub(super) fn root_quiet_bishop_knight_capture_order_score(
    st: &BoardState,
    mv: Move,
) -> Option<i32> {
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

pub(super) fn root_checking_slider_pawn_capture_order_score(
    st: &BoardState,
    mv: Move,
) -> Option<i32> {
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

pub(super) fn root_quiet_queen_check_reply_count(st: &BoardState, mv: Move) -> Option<usize> {
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

pub(super) fn root_queen_pawn_check_capture_order_score(st: &BoardState, mv: Move) -> Option<i32> {
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

pub(super) fn sort_root_moves(st: &BoardState, moves: &[Move], preferred: Move) -> Vec<Move> {
    let sparse_endgame = root_non_king_piece_count(st) <= 8
        && (root_side_has_major(st, st.w) || root_promotion_race(st));
    let has_rook_invasion = moves
        .iter()
        .any(|&mv| root_rook_invasion_score(st, mv).is_some());
    let has_reduced_rook_tactic = moves
        .iter()
        .any(|&mv| root_reduced_rook_check_capture(st, mv))
        || (root_non_king_piece_count(st) <= 11
            && !root_has_queen(st)
            && moves.iter().any(|&mv| {
                let attacker = st.mailbox[move_from(mv)];
                attacker != EMPTY_SQ && piece_type(attacker) == 3 && root_move_gives_check(st, mv)
            }));
    let has_minor_tactic = moves.iter().any(|&mv| root_minor_king_zone_capture(st, mv));
    let has_queen_capture = moves
        .iter()
        .any(|&mv| root_move_is_capture(st, mv) && piece_type(st.mailbox[move_to(mv)]) == 4);
    let use_tactical_order = has_minor_tactic
        || has_queen_capture
        || has_reduced_rook_tactic
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

pub(super) fn tt_root_move(searcher: &Searcher, st: &BoardState, moves: &[Move]) -> Move {
    searcher
        .shared_tt
        .get_depth(st.hash)
        .and_then(|(_, _, _, best_move)| best_move)
        .filter(|best_move| moves.contains(best_move))
        .unwrap_or(NO_MOVE)
}

pub(super) fn root_enemy_pawn_attacks_square(st: &BoardState, target: usize) -> bool {
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

pub(super) fn root_minor_king_zone_capture(st: &BoardState, mv: Move) -> bool {
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
