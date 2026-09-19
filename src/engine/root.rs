use super::{FIFTY_MOVE_ROOT_MATERIAL_MARGIN_CP, FIFTY_MOVE_ROOT_VERIFY_NODE_LIMIT};
use crate::board::{
    is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to,
    piece_type, BoardState, Move, EMPTY_SQ, NO_MOVE,
};
use crate::movegen::{apply_move, generate_moves};
use crate::search::Searcher;
use std::collections::HashMap;

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

pub(super) fn sort_root_moves(moves: &[Move], preferred: Move) -> Vec<Move> {
    let mut ordered = moves.to_vec();
    if let Some(position) = ordered.iter().position(|&mv| mv == preferred) {
        ordered.swap(0, position);
    }
    ordered
}

pub(super) fn tt_root_move(searcher: &Searcher, st: &BoardState, moves: &[Move]) -> Move {
    searcher
        .shared_tt
        .get_depth(st.hash)
        .and_then(|(_, _, _, best_move)| best_move)
        .filter(|best_move| moves.contains(best_move))
        .unwrap_or(NO_MOVE)
}
