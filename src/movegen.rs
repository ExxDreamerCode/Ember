use crate::board::{
    all_occ, bit, black_occ, encode_move, is_attacked, is_white_piece, move_ec, piece_on,
    piece_type, promotion_piece_index, sq, sq_c, sq_r, white_occ, BoardState, BB, BK, BN, BP, BQ,
    BR, EMPTY_SQ, KING_ATTACKS, KNIGHT_ATTACKS, WB, WK, WN, WP, WQ, WR,
};
use crate::magic::{bishop_attacks, rook_attacks};

pub use crate::board::Move;

const ROOK_DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

#[inline]
fn attackers_to(bb: &[u64; 12], occ: u64, sq: usize, by_white: bool) -> u64 {
    let (p, n, b, r, q, k) = if by_white {
        (bb[WP], bb[WN], bb[WB], bb[WR], bb[WQ], bb[WK])
    } else {
        (bb[BP], bb[BN], bb[BB], bb[BR], bb[BQ], bb[BK])
    };
    let bit_s = bit(sq);
    let mut att = 0u64;
    if by_white {
        att |= p & ((bit_s & !0x0101010101010101u64) << 7 | (bit_s & !0x8080808080808080u64) << 9);
    } else {
        att |= p & ((bit_s & !0x0101010101010101u64) >> 9 | (bit_s & !0x8080808080808080u64) >> 7);
    }
    att |= n & KNIGHT_ATTACKS[sq];
    att |= k & KING_ATTACKS[sq];
    att |= (b | q) & bishop_attacks(sq, occ);
    att |= (r | q) & rook_attacks(sq, occ);
    att
}

#[inline]
fn ray_between(a: usize, b: usize) -> u64 {
    let ar = (a / 8) as i32;
    let ac = (a % 8) as i32;
    let br = (b / 8) as i32;
    let bc = (b % 8) as i32;
    if ar == br && ac == bc {
        return 0;
    }
    let aligned = ar == br || ac == bc || (br - ar).abs() == (bc - ac).abs();
    if !aligned {
        return 0;
    }
    let dr = (br - ar).signum();
    let dc = (bc - ac).signum();
    let mut mask = 0u64;
    let mut r = ar + dr;
    let mut c = ac + dc;
    while r != br || c != bc {
        mask |= bit((r * 8 + c) as usize);
        r += dr;
        c += dc;
    }
    mask
}

fn compute_pins(bb: &[u64; 12], occ: u64, own: u64, king_sq: usize, wturn: bool) -> (u64, [u64; 64]) {
    let (opp_rook_like, opp_bishop_like) = if wturn {
        (bb[BR] | bb[BQ], bb[BB] | bb[BQ])
    } else {
        (bb[WR] | bb[WQ], bb[WB] | bb[WQ])
    };

    let mut pinned = 0u64;
    let mut pin_mask = [0u64; 64];

    let kr = (king_sq / 8) as i32;
    let kc = (king_sq % 8) as i32;

    for (idx, &(dr, dc)) in ROOK_DIRS.iter().chain(BISHOP_DIRS.iter()).enumerate() {
        let slider_bb = if idx < 4 { opp_rook_like } else { opp_bishop_like };

        let mut ray = 0u64;
        let mut r = kr + dr;
        let mut c = kc + dc;
        let mut first_blocker: Option<usize> = None;
        while (0..8).contains(&r) && (0..8).contains(&c) {
            let s = (r * 8 + c) as usize;
            ray |= bit(s);
            if occ & bit(s) != 0 {
                first_blocker = Some(s);
                break;
            }
            r += dr;
            c += dc;
        }

        let Some(b1) = first_blocker else {
            continue;
        };
        if own & bit(b1) == 0 {
            continue;
        }

        let mut ray2 = ray;
        let mut r2 = r + dr;
        let mut c2 = c + dc;
        while (0..8).contains(&r2) && (0..8).contains(&c2) {
            let s2 = (r2 * 8 + c2) as usize;
            ray2 |= bit(s2);
            if occ & bit(s2) != 0 {
                if slider_bb & bit(s2) != 0 {
                    pinned |= bit(b1);
                    pin_mask[b1] = ray2;
                }
                break;
            }
            r2 += dr;
            c2 += dc;
        }
    }

    (pinned, pin_mask)
}

#[inline]
pub fn is_chess960_castling_move(st: &BoardState, mv: &Move) -> bool {
    if !st.chess960 {
        return false;
    }
    let from = sq(mv[0], mv[1]);
    let to = sq(mv[2], move_ec(mv));
    let mover_pi = piece_on(&st.bb, from);
    let target_pi = piece_on(&st.bb, to);
    mover_pi != EMPTY_SQ
        && target_pi != EMPTY_SQ
        && piece_type(mover_pi) == 5
        && piece_type(target_pi) == 3
        && is_white_piece(mover_pi) == is_white_piece(target_pi)
}

fn revoke_castling_rights_for_square(st: &mut BoardState, square: usize) {
    for idx in 0..4 {
        if st.cr[idx] && st.castling_rooks[idx] == Some(square) {
            st.cr[idx] = false;
            st.castling_rooks[idx] = None;
        }
    }
}

pub fn apply_move(st: &mut BoardState, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
    let from = sq(sr, sc);
    let to = sq(er, ec);
    let mover_pi = st.mailbox[from];
    if mover_pi == EMPTY_SQ {
        return;
    }
    let mover_type = piece_type(mover_pi);
    let white = is_white_piece(mover_pi);

    let is_chess960_castle = if mover_type == 5 && st.chess960 {
        let target_pi = st.mailbox[to];
        target_pi != EMPTY_SQ && piece_type(target_pi) == 3 && is_white_piece(target_pi) == white
    } else {
        false
    };

    if !is_chess960_castle {
        let cap_pi = st.mailbox[to];
        if cap_pi != EMPTY_SQ {
            st.bb[cap_pi as usize] &= !bit(to);
            st.mailbox[to] = EMPTY_SQ;
        }
    }

    if mover_type == 0 && Some(to) == st.ep {
        let cap_sq = if white { to + 8 } else { to - 8 };
        let ep_pi = st.mailbox[cap_sq];
        if ep_pi != EMPTY_SQ {
            st.bb[ep_pi as usize] &= !bit(cap_sq);
            st.mailbox[cap_sq] = EMPTY_SQ;
        }
    }

    if mover_type == 5 {
        if st.chess960 {
            let target_pi = st.mailbox[to];
            if target_pi != EMPTY_SQ
                && piece_type(target_pi) == 3
                && is_white_piece(target_pi) == white
            {
                let rook_pi = if white { WR } else { BR };
                let rook_col = ec;
                let (king_dst_col, rook_dst_col) = if rook_col > sc {
                    (6usize, 5usize)
                } else {
                    (2usize, 3usize)
                };
                let rook_from = sq(sr, rook_col);
                let rook_to = sq(sr, rook_dst_col);
                let king_to = sq(sr, king_dst_col);
                st.bb[rook_pi] &= !bit(rook_from);
                st.bb[rook_pi] |= bit(rook_to);
                st.bb[mover_pi as usize] &= !bit(from);
                st.bb[mover_pi as usize] |= bit(king_to);
                st.mailbox[from] = EMPTY_SQ;
                st.mailbox[rook_from] = EMPTY_SQ;
                st.mailbox[king_to] = mover_pi;
                st.mailbox[rook_to] = rook_pi as u8;
            }
        } else if sc == 4 && (ec == 6 || ec == 2) {
            let rook_pi = if white { WR } else { BR };
            let (r_from, r_to) = if ec == 6 {
                (sq(sr, 7), sq(sr, 5))
            } else {
                (sq(sr, 0), sq(sr, 3))
            };
            st.bb[rook_pi] &= !bit(r_from);
            st.bb[rook_pi] |= bit(r_to);
            st.mailbox[r_from] = EMPTY_SQ;
            st.mailbox[r_to] = rook_pi as u8;
        }
    }

    if !is_chess960_castle {
        st.bb[mover_pi as usize] &= !bit(from);
        st.mailbox[from] = EMPTY_SQ;

        if mover_type == 0 && (er == 0 || er == 7) {
            let promo_pi =
                promotion_piece_index(white, promotion).unwrap_or(if white { WQ } else { BQ });
            st.bb[promo_pi] |= bit(to);
            st.mailbox[to] = promo_pi as u8;
        } else {
            st.bb[mover_pi as usize] |= bit(to);
            st.mailbox[to] = mover_pi;
        }
    }

    if mover_pi == WK as u8 {
        st.cr[0] = false;
        st.cr[1] = false;
        st.castling_rooks[0] = None;
        st.castling_rooks[1] = None;
    }
    if mover_pi == BK as u8 {
        st.cr[2] = false;
        st.cr[3] = false;
        st.castling_rooks[2] = None;
        st.castling_rooks[3] = None;
    }
    revoke_castling_rights_for_square(st, from);
    revoke_castling_rights_for_square(st, to);
    if !st.chess960 {
        if from == sq(7, 7) || to == sq(7, 7) {
            st.cr[0] = false;
            st.castling_rooks[0] = None;
        }
        if from == sq(7, 0) || to == sq(7, 0) {
            st.cr[1] = false;
            st.castling_rooks[1] = None;
        }
        if from == sq(0, 7) || to == sq(0, 7) {
            st.cr[2] = false;
            st.castling_rooks[2] = None;
        }
        if from == sq(0, 0) || to == sq(0, 0) {
            st.cr[3] = false;
            st.castling_rooks[3] = None;
        }
    }

    st.ep = if mover_type == 0 && er.abs_diff(sr) == 2 {
        Some(sq((sr + er) / 2, sc))
    } else {
        None
    };

    st.w = !st.w;
    st.mc += 1;
}

fn castling_rook_square(
    st: &BoardState,
    wturn: bool,
    kingside: bool,
    king_col: usize,
) -> Option<usize> {
    let idx = match (wturn, kingside) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    };
    if let Some(rook_sq) = st.castling_rooks[idx] {
        return Some(rook_sq);
    }

    let rook_pi = if wturn { WR } else { BR };
    let rank = if wturn { 7usize } else { 0usize };
    let mut candidate = None;
    for col in 0..8 {
        let rs = sq(rank, col);
        if st.bb[rook_pi] & bit(rs) == 0 {
            continue;
        }
        let better_candidate = if kingside {
            col > king_col && candidate.map_or(true, |prev| col < prev)
        } else {
            col < king_col && candidate.map_or(true, |prev| col > prev)
        };
        if better_candidate {
            candidate = Some(col);
        }
    }
    candidate.map(|col| sq(rank, col))
}

fn path_is_clear_for_castling(
    occ: u64,
    rank: usize,
    from_col: usize,
    to_col: usize,
    king_col: usize,
    rook_col: usize,
) -> bool {
    let lo = from_col.min(to_col);
    let hi = from_col.max(to_col);
    for col in lo..=hi {
        if col == king_col || col == rook_col {
            continue;
        }
        if occ & bit(sq(rank, col)) != 0 {
            return false;
        }
    }
    true
}

fn square_attacked_with_king_removed(
    st: &BoardState,
    king_pi: usize,
    king_from: usize,
    square: usize,
    by_white: bool,
) -> bool {
    let mut bb = st.bb;
    bb[king_pi] &= !bit(king_from);
    is_attacked(&bb, square, by_white)
}

#[allow(clippy::too_many_arguments)]
fn try_chess960_castle(
    st: &BoardState,
    wturn: bool,
    cr: &[bool; 4],
    result: &mut Vec<Move>,
    kingside: bool,
    kf: usize,
    kr: usize,
    king_col: usize,
) {
    let right_idx = match (wturn, kingside) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    };
    if !cr[right_idx] {
        return;
    }

    let Some(rook_sq) = castling_rook_square(st, wturn, kingside, king_col) else {
        return;
    };
    if sq_r(rook_sq) != kr {
        return;
    }

    let rook_pi = if wturn { WR } else { BR };
    let king_pi = if wturn { WK } else { BK };
    if st.bb[rook_pi] & bit(rook_sq) == 0 {
        return;
    }

    let rook_col = sq_c(rook_sq);
    let king_dst_col = if kingside { 6usize } else { 2usize };
    let rook_dst_col = if kingside { 5usize } else { 3usize };
    let occ = all_occ(&st.bb);

    if !path_is_clear_for_castling(occ, kr, king_col, king_dst_col, king_col, rook_col) {
        return;
    }
    if !path_is_clear_for_castling(occ, kr, rook_col, rook_dst_col, king_col, rook_col) {
        return;
    }

    let lo = king_col.min(king_dst_col);
    let hi = king_col.max(king_dst_col);
    for col in lo..=hi {
        if square_attacked_with_king_removed(st, king_pi, kf, sq(kr, col), !wturn) {
            return;
        }
    }

    let mut bb2 = st.bb;
    bb2[king_pi] &= !bit(sq(kr, king_col));
    bb2[rook_pi] &= !bit(sq(kr, rook_col));
    bb2[king_pi] |= bit(sq(kr, king_dst_col));
    bb2[rook_pi] |= bit(sq(kr, rook_dst_col));
    if is_attacked(&bb2, sq(kr, king_dst_col), !wturn) {
        return;
    }

    result.push(encode_move(kr, king_col, kr, rook_col, 0));
}

pub fn generate_moves(
    st: &BoardState,
    wturn: bool,
    cr: &[bool; 4],
    ep: Option<usize>,
) -> Vec<Move> {
    let occ = all_occ(&st.bb);
    let own = if wturn {
        white_occ(&st.bb)
    } else {
        black_occ(&st.bb)
    };
    let opp = occ ^ own;
    let free = !occ;
    let king_sq_own = st.king_sq(wturn);
    let checkers_bb = attackers_to(&st.bb, occ, king_sq_own, !wturn);
    let in_check = checkers_bb != 0;
    let num_checkers = checkers_bb.count_ones();
    let check_mask: u64 = if num_checkers == 0 {
        !0u64
    } else if num_checkers == 1 {
        let checker_sq = checkers_bb.trailing_zeros() as usize;
        let checker_pi = st.mailbox[checker_sq];
        let is_slider = checker_pi != EMPTY_SQ && matches!(piece_type(checker_pi), 2 | 3 | 4);
        if is_slider {
            ray_between(king_sq_own, checker_sq) | bit(checker_sq)
        } else {
            bit(checker_sq)
        }
    } else {
        0u64
    };
    let (pinned_bb, pin_mask) = compute_pins(&st.bb, occ, own, king_sq_own, wturn);
    let _back = if wturn { 7usize } else { 0usize };

    let mut result: Vec<Move> = Vec::with_capacity(48);

    macro_rules! try_push {
        ($from:expr, $to:expr) => {{
            let f = $from;
            let t = $to;
            if check_mask & bit(t) != 0 {
                let allowed = if pinned_bb & bit(f) != 0 {
                    pin_mask[f]
                } else {
                    !0u64
                };
                if allowed & bit(t) != 0 {
                    result.push(encode_move(sq_r(f), sq_c(f), sq_r(t), sq_c(t), 0));
                }
            }
        }};
    }

    macro_rules! try_push_ep {
        ($from:expr, $to:expr) => {{
            let f = $from;
            let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap_sq = if wturn { t + 8 } else { t - 8 };
                let cap = piece_on(&bb2, cap_sq);
                let expected_cap = if wturn { BP as u8 } else { WP as u8 };
                if cap == expected_cap && piece_on(&bb2, t) == EMPTY_SQ {
                    bb2[cap as usize] &= !bit(cap_sq);
                    bb2[pi as usize] &= !bit(f);
                    bb2[pi as usize] |= bit(t);
                    if !is_attacked(&bb2, king_sq_own, !wturn) {
                        result.push(encode_move(sq_r(f), sq_c(f), sq_r(t), sq_c(t), 0));
                    }
                }
            }
        }};
    }

    macro_rules! try_push_promo {
        ($from:expr, $to:expr, $promotion:expr) => {{
            let f = $from;
            let t = $to;
            if check_mask & bit(t) != 0 {
                let allowed = if pinned_bb & bit(f) != 0 {
                    pin_mask[f]
                } else {
                    !0u64
                };
                if allowed & bit(t) != 0 {
                    result.push(encode_move(sq_r(f), sq_c(f), sq_r(t), sq_c(t), $promotion));
                }
            }
        }};
    }

    {
        let pawns = if wturn { st.bb[WP] } else { st.bb[BP] };
        let promo_rank_bb: u64 = if wturn {
            0x000000000000FF00u64
        } else {
            0x00FF000000000000u64
        };
        let start_rank: u64 = if wturn {
            0x00FF000000000000u64
        } else {
            0x000000000000FF00u64
        };
        let pushed = if wturn {
            (pawns & !promo_rank_bb & !start_rank) >> 8 & free
        } else {
            (pawns & !promo_rank_bb & !start_rank) << 8 & free
        };
        let mut tmp = pushed;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 8 } else { t - 8 };
            try_push!(f, t);
            tmp &= tmp - 1;
        }
        let pushed2 = if wturn {
            let p1 = (pawns & start_rank) >> 8 & free;
            p1 >> 8 & free
        } else {
            let p1 = (pawns & start_rank) << 8 & free;
            p1 << 8 & free
        };
        let mut tmp = pushed2;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 16 } else { t - 16 };
            try_push!(f, t);
            tmp &= tmp - 1;
        }
        let promo_pawns = pawns & promo_rank_bb;
        let normal_pawns = pawns & !promo_rank_bb;
        let cap_targets = opp | ep.map_or(0, bit);
        let att_c1 = if wturn {
            (normal_pawns & !0x0101010101010101u64) >> 9
        } else {
            (normal_pawns & !0x0101010101010101u64) << 7
        };
        let att_c2 = if wturn {
            (normal_pawns & !0x8080808080808080u64) >> 7
        } else {
            (normal_pawns & !0x8080808080808080u64) << 9
        };
        let mut tmp = att_c1 & cap_targets;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 9 } else { t - 7 };
            if Some(t) == ep {
                try_push_ep!(f, t);
            } else {
                try_push!(f, t);
            }
            tmp &= tmp - 1;
        }
        let mut tmp = att_c2 & cap_targets;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 7 } else { t - 9 };
            if Some(t) == ep {
                try_push_ep!(f, t);
            } else {
                try_push!(f, t);
            }
            tmp &= tmp - 1;
        }

        let start_pushed = if wturn {
            (pawns & start_rank) >> 8 & free
        } else {
            (pawns & start_rank) << 8 & free
        };
        let mut tmp = start_pushed;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 8 } else { t - 8 };
            try_push!(f, t);
            tmp &= tmp - 1;
        }

        let promo_push = if wturn {
            promo_pawns >> 8 & free
        } else {
            promo_pawns << 8 & free
        };
        let mut tmp = promo_push;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 8 } else { t - 8 };
            for promotion in [b'Q', b'R', b'B', b'N'] {
                try_push_promo!(f, t, promotion);
            }
            tmp &= tmp - 1;
        }
        let pc1 = if wturn {
            (promo_pawns & !0x0101010101010101u64) >> 9
        } else {
            (promo_pawns & !0x0101010101010101u64) << 7
        };
        let mut tmp = pc1 & opp;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 9 } else { t - 7 };
            for promotion in [b'Q', b'R', b'B', b'N'] {
                try_push_promo!(f, t, promotion);
            }
            tmp &= tmp - 1;
        }
        let pc2 = if wturn {
            (promo_pawns & !0x8080808080808080u64) >> 7
        } else {
            (promo_pawns & !0x8080808080808080u64) << 9
        };
        let mut tmp = pc2 & opp;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 7 } else { t - 9 };
            for promotion in [b'Q', b'R', b'B', b'N'] {
                try_push_promo!(f, t, promotion);
            }
            tmp &= tmp - 1;
        }
    }

    {
        let mut knights = if wturn { st.bb[WN] } else { st.bb[BN] };
        while knights != 0 {
            let f = knights.trailing_zeros() as usize;
            let mut att = KNIGHT_ATTACKS[f] & !own;
            while att != 0 {
                let t = att.trailing_zeros() as usize;
                try_push!(f, t);
                att &= att - 1;
            }
            knights &= knights - 1;
        }
    }

    {
        let mut bishops = if wturn { st.bb[WB] } else { st.bb[BB] };
        while bishops != 0 {
            let f = bishops.trailing_zeros() as usize;
            let mut att = bishop_attacks(f, occ) & !own;
            while att != 0 {
                let t = att.trailing_zeros() as usize;
                try_push!(f, t);
                att &= att - 1;
            }
            bishops &= bishops - 1;
        }
    }

    {
        let mut rooks = if wturn { st.bb[WR] } else { st.bb[BR] };
        while rooks != 0 {
            let f = rooks.trailing_zeros() as usize;
            let mut att = rook_attacks(f, occ) & !own;
            while att != 0 {
                let t = att.trailing_zeros() as usize;
                try_push!(f, t);
                att &= att - 1;
            }
            rooks &= rooks - 1;
        }
    }

    {
        let mut queens = if wturn { st.bb[WQ] } else { st.bb[BQ] };
        while queens != 0 {
            let f = queens.trailing_zeros() as usize;
            let att = (bishop_attacks(f, occ) | rook_attacks(f, occ)) & !own;
            let mut att = att;
            while att != 0 {
                let t = att.trailing_zeros() as usize;
                try_push!(f, t);
                att &= att - 1;
            }
            queens &= queens - 1;
        }
    }

    {
        let kf = king_sq_own;
        let mut att = KING_ATTACKS[kf] & !own;
        while att != 0 {
            let t = att.trailing_zeros() as usize;
            let cap = piece_on(&st.bb, t);
            if cap != EMPTY_SQ && piece_type(cap) == 5 {
                att &= att - 1;
                continue;
            }
            let mut bb2 = st.bb;
            bb2[if wturn { WK } else { BK }] &= !bit(kf);
            bb2[if wturn { WK } else { BK }] |= bit(t);
            if cap != EMPTY_SQ {
                bb2[cap as usize] &= !bit(t);
            }
            if !is_attacked(&bb2, t, !wturn) {
                result.push(encode_move(sq_r(kf), sq_c(kf), sq_r(t), sq_c(t), 0));
            }
            att &= att - 1;
        }
        if !in_check && st.chess960 {
            let kr = if wturn { 7usize } else { 0usize };
            let king_col = sq_c(kf);
            try_chess960_castle(st, wturn, cr, &mut result, true, kf, kr, king_col);
            try_chess960_castle(st, wturn, cr, &mut result, false, kf, kr, king_col);
        } else if !in_check {
            let rook_pi = if wturn { WR } else { BR };
            let kr = if wturn { 7usize } else { 0usize };
            let king_col = sq_c(kf);
            if cr[if wturn { 0 } else { 2 }]
                && king_col == 4
                && st.bb[rook_pi] & bit(sq(kr, 7)) != 0
                && all_occ(&st.bb) & (bit(sq(kr, 5)) | bit(sq(kr, 6))) == 0
                && !is_attacked(&st.bb, sq(kr, 4), !wturn)
                && !is_attacked(&st.bb, sq(kr, 5), !wturn)
                && !is_attacked(&st.bb, sq(kr, 6), !wturn)
            {
                result.push(encode_move(kr, 4, kr, 6, 0));
            }
            if cr[if wturn { 1 } else { 3 }]
                && king_col == 4
                && st.bb[rook_pi] & bit(sq(kr, 0)) != 0
                && all_occ(&st.bb) & (bit(sq(kr, 1)) | bit(sq(kr, 2)) | bit(sq(kr, 3))) == 0
                && !is_attacked(&st.bb, sq(kr, 4), !wturn)
                && !is_attacked(&st.bb, sq(kr, 3), !wturn)
                && !is_attacked(&st.bb, sq(kr, 2), !wturn)
            {
                result.push(encode_move(kr, 4, kr, 2, 0));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{move_ec, move_promotion, move_to_uci};
    use crate::engine::Engine;
    use std::collections::BTreeSet;

    fn state_from_fen(fen: &str) -> BoardState {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        engine.st
    }

    fn perft(st: &BoardState, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = generate_moves(st, st.w, &st.cr, st.ep);
        if depth == 1 {
            return moves.len() as u64;
        }

        let mut nodes = 0u64;
        for mv in moves {
            let mut next = *st;
            apply_move(
                &mut next,
                mv[0],
                mv[1],
                mv[2],
                move_ec(&mv),
                move_promotion(&mv),
            );
            nodes += perft(&next, depth - 1);
        }
        nodes
    }

    #[test]
    fn start_position_perft_smoke() {
        let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(perft(&st, 1), 20);
        assert_eq!(perft(&st, 2), 400);
        assert_eq!(perft(&st, 3), 8902);
    }

    #[test]
    fn temp_perft_startpos_deep() {
        let st = state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(perft(&st, 4), 197281);
        assert_eq!(perft(&st, 5), 4865609);
    }

    #[test]
    fn temp_perft_kiwipete() {
        let st = state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&st, 1), 48);
        assert_eq!(perft(&st, 2), 2039);
        assert_eq!(perft(&st, 3), 97862);
        assert_eq!(perft(&st, 4), 4085603);
    }

    #[test]
    fn temp_perft_position3() {
        let st = state_from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(perft(&st, 1), 14);
        assert_eq!(perft(&st, 2), 191);
        assert_eq!(perft(&st, 3), 2812);
        assert_eq!(perft(&st, 4), 43238);
        assert_eq!(perft(&st, 5), 674624);
    }

    #[test]
    fn temp_perft_position4() {
        let st = state_from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        assert_eq!(perft(&st, 1), 6);
        assert_eq!(perft(&st, 2), 264);
        assert_eq!(perft(&st, 3), 9467);
        assert_eq!(perft(&st, 4), 422333);
    }

    #[test]
    fn temp_perft_position5() {
        let st = state_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
        assert_eq!(perft(&st, 1), 44);
        assert_eq!(perft(&st, 2), 1486);
        assert_eq!(perft(&st, 3), 62379);
        assert_eq!(perft(&st, 4), 2103487);
    }


    #[test]
    #[ignore]
    fn temp_perft_kiwipete_depth5_stress() {
        let st = state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&st, 5), 193690690);
    }

    #[test]
    #[ignore]
    fn temp_perft_position3_depth6_stress() {
        let st = state_from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(perft(&st, 6), 11030083);
    }

    #[test]
    fn temp_perft_position6() {
        let st = state_from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        );
        assert_eq!(perft(&st, 1), 46);
        assert_eq!(perft(&st, 2), 2079);
        assert_eq!(perft(&st, 3), 89890);
    }

    #[test]
    fn rook_castling_perft_covers_castling_rights() {
        let st = state_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&st, 1), 26);
        assert_eq!(perft(&st, 2), 568);
    }

    #[test]
    fn en_passant_move_removes_the_captured_pawn() {
        let mut st = state_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let moves = generate_moves(&st, st.w, &st.cr, st.ep);
        let ep = moves
            .into_iter()
            .find(|mv| *mv == [3, 4, 2, 3])
            .expect("expected e5d6 en passant to be legal");

        apply_move(
            &mut st,
            ep[0],
            ep[1],
            ep[2],
            move_ec(&ep),
            move_promotion(&ep),
        );

        assert_ne!(piece_on(&st.bb, sq(2, 3)), EMPTY_SQ);
        assert_eq!(piece_on(&st.bb, sq(3, 3)), EMPTY_SQ);
        assert!(!st.w);
    }

    #[test]
    fn chess960_castling_places_pieces_on_standard_squares() {
        let mut engine = Engine::new();
        engine.set_fen("1k6/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            uci_moves.contains(&"e1a1".to_string()),
            "Queenside castling e1a1 should be legal"
        );
        assert!(
            uci_moves.contains(&"e1h1".to_string()),
            "Kingside castling e1h1 should be legal"
        );

        let oo = moves
            .iter()
            .find(|mv| move_to_uci(&engine.st, mv) == "e1h1")
            .unwrap();
        apply_move(
            &mut engine.st,
            oo[0],
            oo[1],
            oo[2],
            move_ec(oo),
            move_promotion(oo),
        );
        assert_eq!(
            engine.st.king_sq(true),
            7 * 8 + 6,
            "King should be on g1 after O-O"
        );
        assert!(
            engine.st.bb[WR] & bit(7 * 8 + 5) != 0,
            "Rook should be on f1 after O-O"
        );
        assert!(
            engine.st.bb[WR] & bit(7 * 8 + 7) == 0,
            "Rook should no longer be on h1 after O-O"
        );
    }

    #[test]
    fn chess960_castling_queenside_places_pieces_on_standard_squares() {
        let mut engine = Engine::new();
        engine.set_fen("r3k2r/8/8/8/8/8/8/1K6 b KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            uci_moves.contains(&"e8a8".to_string()),
            "Black queenside castling e8a8 should be legal"
        );
        assert!(
            uci_moves.contains(&"e8h8".to_string()),
            "Black kingside castling e8h8 should be legal"
        );

        let ooo = moves
            .iter()
            .find(|mv| move_to_uci(&engine.st, mv) == "e8a8")
            .unwrap();
        apply_move(
            &mut engine.st,
            ooo[0],
            ooo[1],
            ooo[2],
            move_ec(ooo),
            move_promotion(ooo),
        );
        assert_eq!(
            engine.st.king_sq(false),
            sq(0, 2),
            "Black king should be on c8 after O-O-O"
        );
        assert!(
            engine.st.bb[BR] & bit(sq(0, 3)) != 0,
            "Black rook should be on d8 after O-O-O"
        );
        assert!(
            engine.st.bb[BR] & bit(sq(0, 0)) == 0,
            "Black rook should no longer be on a8 after O-O-O"
        );
    }

    #[test]
    fn chess960_castling_blocked_by_pieces() {
        let mut engine = Engine::new();
        engine.set_fen("1k6/8/8/8/8/8/8/RBNKBNQR w KQkq - 0 1");
        engine.st.chess960 = true;

        let king_sq = engine.st.king_sq(true);
        assert_eq!(king_sq, 7 * 8 + 3, "White king should be on d1");

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            !uci_moves.contains(&"d1a1".to_string()),
            "Queenside castling d1a1 blocked by bishop on b1"
        );
        assert!(
            !uci_moves.contains(&"d1h1".to_string()),
            "Kingside castling d1h1 blocked by bishop on f1"
        );
    }

    #[test]
    fn chess960_castling_blocked_by_piece_on_king_destination() {
        let mut engine = Engine::new();
        engine.set_fen("1k6/8/8/8/8/8/8/R3KBNR w KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            !uci_moves.contains(&"e1h1".to_string()),
            "Kingside castling e1h1 should be blocked by bishop on g1"
        );
    }

    #[test]
    fn chess960_castling_queenside_blocked_by_piece_on_king_destination() {
        let mut engine = Engine::new();
        engine.set_fen("1k6/8/8/8/8/8/8/BNR1K2R w KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            !uci_moves.contains(&"e1a1".to_string()),
            "Queenside castling e1a1 should be blocked by bishop on c1"
        );
    }

    #[test]
    fn chess960_castling_king_ends_in_check_from_discovered_attack() {
        let mut engine = Engine::new();
        engine.set_fen("4r1k1/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(
            !uci_moves.contains(&"e1h1".to_string()),
            "Kingside castling e1h1 should be illegal: king on g1 would be attacked by rook on e8"
        );
    }

    #[test]
    fn generate_moves_includes_underpromotions() {
        let mut st = state_from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        let moves = generate_moves(&st, st.w, &st.cr, st.ep);
        let uci: BTreeSet<String> = moves.iter().map(|mv| move_to_uci(&st, mv)).collect();

        for suffix in ["q", "r", "b", "n"] {
            assert!(uci.contains(&format!("a7a8{suffix}")));
            assert!(uci.contains(&format!("a7b8{suffix}")));
        }

        let knight_promotion = moves
            .into_iter()
            .find(|mv| move_to_uci(&st, mv) == "a7a8n")
            .expect("expected a7a8n to be legal");
        apply_move(
            &mut st,
            knight_promotion[0],
            knight_promotion[1],
            knight_promotion[2],
            move_ec(&knight_promotion),
            move_promotion(&knight_promotion),
        );

        assert_eq!(piece_on(&st.bb, sq(0, 0)), WN as u8);
        assert_eq!(piece_on(&st.bb, sq(1, 0)), EMPTY_SQ);
        assert!(!st.w);
    }
}