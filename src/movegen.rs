use crate::board::{
    all_occ, bit, black_occ, encode_move, is_attacked, is_white_piece, piece_on, piece_type,
    promotion_piece_index, sq, sq_c, sq_r, white_occ, BoardState, BB, BK, BN, BP, BQ, BR, EMPTY_SQ,
    KING_ATTACKS, KNIGHT_ATTACKS, WB, WK, WN, WP, WQ, WR,
};
use crate::magic::{bishop_attacks, rook_attacks};

pub use crate::board::Move;

pub fn apply_move(st: &mut BoardState, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
    let from = sq(sr, sc);
    let to = sq(er, ec);
    let mover_pi = piece_on(&st.bb, from);
    if mover_pi == EMPTY_SQ {
        return;
    }
    let mover_type = piece_type(mover_pi);
    let white = is_white_piece(mover_pi);

    let is_chess960_castle = if mover_type == 5 && st.chess960 {
        let target_pi = piece_on(&st.bb, to);
        if target_pi != EMPTY_SQ && piece_type(target_pi) == 3 && is_white_piece(target_pi) == white {
            let king_dst_col = if ec > sc { 6usize } else { 2usize };
            king_dst_col != sc
        } else {
            false
        }
    } else {
        false
    };

    if !is_chess960_castle {
        let cap_pi = piece_on(&st.bb, to);
        if cap_pi != EMPTY_SQ {
            st.bb[cap_pi as usize] &= !bit(to);
        }
    }

    if mover_type == 0 && Some(to) == st.ep {
        let cap_sq = if white { to + 8 } else { to - 8 };
        let ep_pi = piece_on(&st.bb, cap_sq);
        if ep_pi != EMPTY_SQ {
            st.bb[ep_pi as usize] &= !bit(cap_sq);
        }
    }

    if mover_type == 5 {
        if st.chess960 {
            let target_pi = piece_on(&st.bb, to);
            if target_pi != EMPTY_SQ && piece_type(target_pi) == 3 && is_white_piece(target_pi) == white {
                let rook_pi = if white { WR } else { BR };
                let rook_col = ec;
                let (king_dst_col, rook_dst_col) = if rook_col > sc {
                    (6usize, 5usize)
                } else {
                    (2usize, 3usize)
                };
                st.bb[rook_pi] &= !bit(sq(sr, rook_col));
                st.bb[rook_pi] |= bit(sq(sr, rook_dst_col));
                st.bb[mover_pi as usize] &= !bit(sq(sr, sc));
                st.bb[mover_pi as usize] |= bit(sq(sr, king_dst_col));
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
        }
    }

    if !is_chess960_castle {
        st.bb[mover_pi as usize] &= !bit(from);

        if mover_type == 0 && (er == 0 || er == 7) {
            let promo_pi =
                promotion_piece_index(white, promotion).unwrap_or(if white { WQ } else { BQ });
            st.bb[promo_pi] |= bit(to);
        } else {
            st.bb[mover_pi as usize] |= bit(to);
        }
    }

    if mover_pi == WK as u8 {
        st.cr[0] = false;
        st.cr[1] = false;
    }
    if mover_pi == BK as u8 {
        st.cr[2] = false;
        st.cr[3] = false;
    }
    if from == sq(7, 7) || to == sq(7, 7) {
        st.cr[0] = false;
    }
    if from == sq(7, 0) || to == sq(7, 0) {
        st.cr[1] = false;
    }
    if from == sq(0, 7) || to == sq(0, 7) {
        st.cr[2] = false;
    }
    if from == sq(0, 0) || to == sq(0, 0) {
        st.cr[3] = false;
    }

    st.ep = if mover_type == 0 && er.abs_diff(sr) == 2 {
        Some(sq((sr + er) / 2, sc))
    } else {
        None
    };

    st.w = !st.w;
    st.mc += 1;
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
    let in_check = is_attacked(&st.bb, king_sq_own, !wturn);
    let _back = if wturn { 7usize } else { 0usize };

    let mut result: Vec<Move> = Vec::with_capacity(48);

    macro_rules! try_push {
        ($from:expr, $to:expr) => {{
            let f = $from;
            let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap = piece_on(&bb2, t);
                if cap != EMPTY_SQ {
                    bb2[cap as usize] &= !bit(t);
                }
                bb2[pi as usize] &= !bit(f);
                bb2[pi as usize] |= bit(t);
                let ks = if piece_type(pi) == 5 { t } else { king_sq_own };
                if !is_attacked(&bb2, ks, !wturn) {
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
                if cap != EMPTY_SQ {
                    bb2[cap as usize] &= !bit(cap_sq);
                }
                bb2[pi as usize] &= !bit(f);
                bb2[pi as usize] |= bit(t);
                if !is_attacked(&bb2, king_sq_own, !wturn) {
                    result.push(encode_move(sq_r(f), sq_c(f), sq_r(t), sq_c(t), 0));
                }
            }
        }};
    }

    macro_rules! try_push_promo {
        ($from:expr, $to:expr, $promotion:expr) => {{
            let f = $from;
            let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap = piece_on(&bb2, t);
                if cap != EMPTY_SQ {
                    bb2[cap as usize] &= !bit(t);
                }
                bb2[pi as usize] &= !bit(f);
                if let Some(promo_pi) = promotion_piece_index(wturn, $promotion) {
                    bb2[promo_pi] |= bit(t);
                }
                if !is_attacked(&bb2, king_sq_own, !wturn) {
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
            let mut bb2 = st.bb;
            bb2[if wturn { WK } else { BK }] &= !bit(kf);
            bb2[if wturn { WK } else { BK }] |= bit(t);
            let cap = piece_on(&st.bb, t);
            if cap != EMPTY_SQ {
                bb2[cap as usize] &= !bit(t);
            }
            if !is_attacked(&bb2, t, !wturn) {
                result.push(encode_move(sq_r(kf), sq_c(kf), sq_r(t), sq_c(t), 0));
            }
            att &= att - 1;
        }
        if !in_check && st.chess960 {
            let rook_pi = if wturn { WR } else { BR };
            let kr = if wturn { 7usize } else { 0usize };
            let rooks = st.bb[rook_pi] & if wturn { 0xFF00000000000000 } else { 0x00000000000000FF };
            let king_col = sq_c(kf);
            let occ = all_occ(&st.bb);
            let mut rook_list: Vec<usize> = Vec::new();
            let mut tmp = rooks;
            while tmp != 0 {
                let rs = tmp.trailing_zeros() as usize;
                rook_list.push(rs % 8);
                tmp &= tmp - 1;
            }
            rook_list.sort();
            if cr[if wturn { 0 } else { 2 }] {
                if let Some(&rc) = rook_list.iter().find(|&&c| c > king_col) {
                    let king_dst = 6usize;
                    let rook_dst = 5usize;
                    if king_dst != king_col {
                        let mut blocked = false;
                        for c in (king_col + 1)..rc {
                            if occ & bit(sq(kr, c)) != 0 { blocked = true; break; }
                        }
                        if !blocked {
                            let rlo = rook_dst.min(rc);
                            let rhi = rook_dst.max(rc);
                            for c in rlo..=rhi {
                                if occ & bit(sq(kr, c)) != 0 && c != rc { blocked = true; break; }
                            }
                        }
                        if !blocked {
                            let mut attacked = false;
                            for c in king_col..=king_dst {
                                if is_attacked(&st.bb, sq(kr, c), !wturn) { attacked = true; break; }
                            }
                            if !attacked {
                                result.push(encode_move(kr, king_col, kr, rc, 0));
                            }
                        }
                    }
                }
            }
            if cr[if wturn { 1 } else { 3 }] {
                if let Some(&rc) = rook_list.iter().rev().find(|&&c| c < king_col) {
                    let king_dst = 2usize;
                    let rook_dst = 3usize;
                    if king_dst != king_col {
                        let mut blocked = false;
                        for c in (rc + 1)..king_col {
                            if occ & bit(sq(kr, c)) != 0 { blocked = true; break; }
                        }
                        if !blocked {
                            let rlo = rook_dst.min(rc);
                            let rhi = rook_dst.max(rc);
                            for c in rlo..=rhi {
                                if occ & bit(sq(kr, c)) != 0 && c != rc { blocked = true; break; }
                            }
                        }
                        if !blocked {
                            let mut attacked = false;
                            for c in king_dst..=king_col {
                                if is_attacked(&st.bb, sq(kr, c), !wturn) { attacked = true; break; }
                            }
                            if !attacked {
                                result.push(encode_move(kr, king_col, kr, rc, 0));
                            }
                        }
                    }
                }
            }
        } else if !in_check {
            let rook_pi = if wturn { WR } else { BR };
            let kr = if wturn { 7usize } else { 0usize };
            if cr[if wturn { 0 } else { 2 }]
                && st.bb[rook_pi] & bit(sq(kr, 7)) != 0
                && all_occ(&st.bb) & (bit(sq(kr, 5)) | bit(sq(kr, 6))) == 0
                && !is_attacked(&st.bb, sq(kr, 4), !wturn)
                && !is_attacked(&st.bb, sq(kr, 5), !wturn)
                && !is_attacked(&st.bb, sq(kr, 6), !wturn)
            {
                result.push(encode_move(kr, 4, kr, 6, 0));
            }
            if cr[if wturn { 1 } else { 3 }]
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

        assert!(uci_moves.contains(&"e1a1".to_string()), "Queenside castling e1a1 should be legal");
        assert!(uci_moves.contains(&"e1h1".to_string()), "Kingside castling e1h1 should be legal");

        let oo = moves.iter().find(|mv| move_to_uci(&engine.st, mv) == "e1h1").unwrap();
        apply_move(&mut engine.st, oo[0], oo[1], oo[2], move_ec(oo), move_promotion(oo));
        assert_eq!(engine.st.king_sq(true), 7 * 8 + 6, "King should be on g1 after O-O");
        assert!(engine.st.bb[WR] & bit(7 * 8 + 5) != 0, "Rook should be on f1 after O-O");
        assert!(engine.st.bb[WR] & bit(7 * 8 + 7) == 0, "Rook should no longer be on h1 after O-O");
    }

    #[test]
    fn chess960_castling_queenside_places_pieces_on_standard_squares() {
        let mut engine = Engine::new();
        engine.set_fen("r3k2r/8/8/8/8/8/8/1K6 b KQkq - 0 1");
        engine.st.chess960 = true;

        let moves = generate_moves(&engine.st, engine.st.w, &engine.st.cr, engine.st.ep);
        let uci_moves: Vec<String> = moves.iter().map(|mv| move_to_uci(&engine.st, mv)).collect();

        assert!(uci_moves.contains(&"e8a8".to_string()), "Black queenside castling e8a8 should be legal");
        assert!(uci_moves.contains(&"e8h8".to_string()), "Black kingside castling e8h8 should be legal");

        let ooo = moves.iter().find(|mv| move_to_uci(&engine.st, mv) == "e8a8").unwrap();
        apply_move(&mut engine.st, ooo[0], ooo[1], ooo[2], move_ec(ooo), move_promotion(ooo));
        assert_eq!(engine.st.king_sq(false), 0 * 8 + 2, "Black king should be on c8 after O-O-O");
        assert!(engine.st.bb[BR] & bit(0 * 8 + 3) != 0, "Black rook should be on d8 after O-O-O");
        assert!(engine.st.bb[BR] & bit(0 * 8 + 0) == 0, "Black rook should no longer be on a8 after O-O-O");
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

        assert!(!uci_moves.contains(&"d1a1".to_string()), "Queenside castling d1a1 blocked by bishop on b1");
        assert!(!uci_moves.contains(&"d1h1".to_string()), "Kingside castling d1h1 blocked by bishop on f1");
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