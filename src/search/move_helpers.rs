use crate::board::{
    bit, is_attacked, is_white_piece, move_ec, move_er, move_from, move_promotion, move_sc,
    move_sr, move_to, piece_on, piece_type, promotion_piece_index, see, BoardState, Move, BK, BR,
    EMPTY_SQ, MATE, WK, WR,
};
use crate::movegen::is_chess960_castling_move_mode;

pub(super) fn piece_val(pt: u8) -> i32 {
    match pt {
        0 => 100,
        1 => 325,
        2 => 340,
        3 => 500,
        4 => 950,
        _ => 0,
    }
}

pub(super) fn piece_to_idx(pt: u8) -> usize {
    match pt {
        0 => 1,
        1 => 2,
        2 => 3,
        3 => 4,
        4 => 5,
        5 => 6,
        _ => 0,
    }
}

pub(super) fn from_to_key(sr: usize, sc: usize, er: usize, ec: usize) -> (usize, usize) {
    (sr * 8 + sc, er * 8 + ec)
}

pub(super) fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE / 2 {
        score + ply as i32
    } else if score < -MATE / 2 {
        score - ply as i32
    } else {
        score
    }
}

pub(super) fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE / 2 {
        score - ply as i32
    } else if score < -MATE / 2 {
        score + ply as i32
    } else {
        score
    }
}

#[inline]
pub(super) fn is_promotion_move(fpi: u8, mv: Move) -> bool {
    move_promotion(mv) != 0
        || (fpi != EMPTY_SQ && piece_type(fpi) == 0 && (move_er(mv) == 0 || move_er(mv) == 7))
}

pub(super) fn promotion_value(mv: Move) -> i32 {
    match move_promotion(mv).to_ascii_uppercase() {
        b'N' => piece_val(1),
        b'B' => piece_val(2),
        b'R' => piece_val(3),
        b'Q' => piece_val(4),
        _ => 0,
    }
}

#[inline]
pub(super) fn is_en_passant_capture(
    st: &BoardState,
    fpi: u8,
    mv: Move,
    to: usize,
    tpi: u8,
) -> bool {
    fpi != EMPTY_SQ
        && tpi == EMPTY_SQ
        && piece_type(fpi) == 0
        && Some(to) == st.ep
        && move_sc(mv) != move_ec(mv)
}

#[inline]
pub(super) fn capture_victim_value<const CHESS960: bool>(
    st: &BoardState,
    fpi: u8,
    mv: Move,
    to: usize,
    tpi: u8,
) -> i32 {
    if is_chess960_castling_move_mode::<CHESS960>(st, mv) {
        0
    } else if tpi != EMPTY_SQ {
        piece_val(piece_type(tpi))
    } else if is_en_passant_capture(st, fpi, mv, to, tpi) {
        piece_val(0)
    } else {
        0
    }
}

#[inline]
pub(super) fn move_is_capture<const CHESS960: bool>(
    st: &BoardState,
    fpi: u8,
    mv: Move,
    to: usize,
    tpi: u8,
) -> bool {
    !is_chess960_castling_move_mode::<CHESS960>(st, mv)
        && (tpi != EMPTY_SQ || is_en_passant_capture(st, fpi, mv, to, tpi))
}

#[inline]
pub(super) fn move_see<const CHESS960: bool>(
    st: &BoardState,
    mv: Move,
    from: usize,
    to: usize,
    fpi: u8,
    tpi: u8,
) -> i32 {
    if is_chess960_castling_move_mode::<CHESS960>(st, mv)
        || is_en_passant_capture(st, fpi, mv, to, tpi)
    {
        0
    } else {
        see(&st.bb, from, to)
    }
}

#[inline(always)]
pub(super) fn special_move_gives_check_mode<const CHESS960: bool>(
    st: &BoardState,
    mv: Move,
) -> bool {
    let from = move_from(mv);
    let to = move_to(mv);
    let fpi = st.mailbox[from];
    if fpi == EMPTY_SQ {
        return false;
    }

    let mut bb = st.bb;
    let mover_is_white = is_white_piece(fpi);
    let mover_type = piece_type(fpi);
    let is_chess960_castle = is_chess960_castling_move_mode::<CHESS960>(st, mv);
    let is_en_passant = mover_type == 0 && Some(to) == st.ep && move_sc(mv) != move_ec(mv);
    let is_standard_castle =
        mover_type == 5 && !CHESS960 && move_sc(mv) == 4 && (move_ec(mv) == 6 || move_ec(mv) == 2);

    if !is_en_passant && !is_chess960_castle && !is_standard_castle {
        return false;
    }

    if !is_chess960_castle {
        let tpi = piece_on(&bb, to);
        if tpi != EMPTY_SQ {
            bb[tpi as usize] &= !bit(to);
        }
    }

    if is_en_passant {
        let cap_sq = if mover_is_white { to + 8 } else { to - 8 };
        let ep_pi = piece_on(&bb, cap_sq);
        if ep_pi != EMPTY_SQ {
            bb[ep_pi as usize] &= !bit(cap_sq);
        }
    }

    if mover_type == 5 && is_chess960_castle {
        let rook_pi = if mover_is_white { WR } else { BR };
        let rook_col = move_ec(mv);
        let (king_dst_col, rook_dst_col) = if rook_col > move_sc(mv) {
            (6usize, 5usize)
        } else {
            (2usize, 3usize)
        };
        bb[rook_pi] &= !bit(move_er(mv) * 8 + rook_col);
        bb[rook_pi] |= bit(move_sr(mv) * 8 + rook_dst_col);
        bb[fpi as usize] &= !bit(from);
        bb[fpi as usize] |= bit(move_sr(mv) * 8 + king_dst_col);
    } else {
        bb[fpi as usize] &= !bit(from);

        if mover_type == 5
            && !CHESS960
            && move_sc(mv) == 4
            && (move_ec(mv) == 6 || move_ec(mv) == 2)
        {
            let rook_pi = if mover_is_white { WR } else { BR };
            let (rook_from, rook_to) = if move_ec(mv) == 6 {
                (move_sr(mv) * 8 + 7, move_sr(mv) * 8 + 5)
            } else {
                (move_sr(mv) * 8, move_sr(mv) * 8 + 3)
            };
            bb[rook_pi] &= !bit(rook_from);
            bb[rook_pi] |= bit(rook_to);
        }

        if mover_type == 0 && (move_er(mv) == 0 || move_er(mv) == 7) {
            if let Some(ppi) = promotion_piece_index(mover_is_white, move_promotion(mv)) {
                bb[ppi] |= bit(to);
            } else {
                bb[fpi as usize] |= bit(to);
            }
        } else {
            bb[fpi as usize] |= bit(to);
        }
    }

    let opponent_king = if st.w { bb[BK] } else { bb[WK] };
    opponent_king != 0 && is_attacked(&bb, opponent_king.trailing_zeros() as usize, st.w)
}

#[cfg(test)]
pub(super) fn special_move_gives_check(st: &BoardState, mv: Move) -> bool {
    if st.chess960 {
        special_move_gives_check_mode::<true>(st, mv)
    } else {
        special_move_gives_check_mode::<false>(st, mv)
    }
}
