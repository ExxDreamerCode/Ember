use crate::board::{
    BoardState, EMPTY_SQ, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    bit, sq, sq_r, sq_c, piece_on, is_white_piece, piece_type, piece_char, piece_from_char,
    white_occ, black_occ, all_occ, is_attacked, KNIGHT_ATTACKS, KING_ATTACKS,
};
use crate::magic::{bishop_attacks, rook_attacks};

pub type Move = [usize; 4];

pub fn apply_move(st: &mut BoardState, sr: usize, sc: usize, er: usize, ec: usize, promotion: u8) {
    let from = sq(sr, sc);
    let to   = sq(er, ec);
    let mover_pi = piece_on(&st.bb, from);
    if mover_pi == EMPTY_SQ { return; }
    let mover_type = piece_type(mover_pi);
    let white = is_white_piece(mover_pi);

    let cap_pi = piece_on(&st.bb, to);
    if cap_pi != EMPTY_SQ { st.bb[cap_pi as usize] &= !bit(to); }

    if mover_type == 0 && Some(to) == st.ep {
        let cap_sq = if white { to + 8 } else { to - 8 };
        let ep_pi = piece_on(&st.bb, cap_sq);
        if ep_pi != EMPTY_SQ { st.bb[ep_pi as usize] &= !bit(cap_sq); }
    }

    if mover_type == 5 && sc == 4 && (ec == 6 || ec == 2) {
        let rook_pi = if white { WR } else { BR };
        if ec == 6 {
            st.bb[rook_pi] &= !bit(sq(sr, 7));
            st.bb[rook_pi] |=  bit(sq(sr, 5));
        } else {
            st.bb[rook_pi] &= !bit(sq(sr, 0));
            st.bb[rook_pi] |=  bit(sq(sr, 3));
        }
    }

    st.bb[mover_pi as usize] &= !bit(from);

    if mover_type == 0 && (er == 0 || er == 7) {
        let promo_type: usize = if promotion != 0 {
            match promotion {
                b'Q' | b'q' => 4,
                b'R' | b'r' => 3,
                b'B' | b'b' => 2,
                b'N' | b'n' => 1,
                _ => 4,
            }
        } else { 4 };
        let promo_pi = if white { promo_type } else { promo_type + 6 };
        st.bb[promo_pi] |= bit(to);
    } else {
        st.bb[mover_pi as usize] |= bit(to);
    }

    if mover_pi == WK as u8 { st.cr[0] = false; st.cr[1] = false; }
    if mover_pi == BK as u8 { st.cr[2] = false; st.cr[3] = false; }
    if from == sq(7,7) || to == sq(7,7) { st.cr[0] = false; }
    if from == sq(7,0) || to == sq(7,0) { st.cr[1] = false; }
    if from == sq(0,7) || to == sq(0,7) { st.cr[2] = false; }
    if from == sq(0,0) || to == sq(0,0) { st.cr[3] = false; }

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
    let occ   = all_occ(&st.bb);
    let own   = if wturn { white_occ(&st.bb) } else { black_occ(&st.bb) };
    let opp   = occ ^ own;
    let free  = !occ;
    let king_sq_own = st.king_sq(wturn);
    let in_check = is_attacked(&st.bb, king_sq_own, !wturn);
    let _back  = if wturn { 7usize } else { 0usize };

    let mut result: Vec<Move> = Vec::with_capacity(48);

    macro_rules! try_push {
        ($from:expr, $to:expr) => {{
            let f = $from; let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap = piece_on(&bb2, t);
                if cap != EMPTY_SQ { bb2[cap as usize] &= !bit(t); }
                bb2[pi as usize] &= !bit(f);
                bb2[pi as usize] |=  bit(t);
                let ks = if piece_type(pi) == 5 { t } else { king_sq_own };
                if !is_attacked(&bb2, ks, !wturn) {
                    result.push([sq_r(f), sq_c(f), sq_r(t), sq_c(t)]);
                }
            }
        }};
    }

    macro_rules! try_push_ep {
        ($from:expr, $to:expr) => {{
            let f = $from; let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap_sq = if wturn { t + 8 } else { t - 8 };
                let cap = piece_on(&bb2, cap_sq);
                if cap != EMPTY_SQ { bb2[cap as usize] &= !bit(cap_sq); }
                bb2[pi as usize] &= !bit(f);
                bb2[pi as usize] |=  bit(t);
                if !is_attacked(&bb2, king_sq_own, !wturn) {
                    result.push([sq_r(f), sq_c(f), sq_r(t), sq_c(t)]);
                }
            }
        }};
    }

    macro_rules! try_push_promo {
        ($from:expr, $to:expr) => {{
            let f = $from; let t = $to;
            let mut bb2 = st.bb;
            let pi = piece_on(&bb2, f);
            if pi != EMPTY_SQ {
                let cap = piece_on(&bb2, t);
                if cap != EMPTY_SQ { bb2[cap as usize] &= !bit(t); }
                bb2[pi as usize] &= !bit(f);
                let qpi = if wturn { WQ } else { BQ };
                bb2[qpi] |= bit(t);
                if !is_attacked(&bb2, king_sq_own, !wturn) {
                    result.push([sq_r(f), sq_c(f), sq_r(t), sq_c(t)]);
                }
            }
        }};
    }

    {
        let pawns = if wturn { st.bb[WP] } else { st.bb[BP] };
        let promo_rank_bb: u64 = if wturn { 0x00FF000000000000u64 } else { 0x000000000000FF00u64 };
        let pushed = if wturn { (pawns & !promo_rank_bb) >> 8 & free }
                     else     { (pawns & !promo_rank_bb) << 8 & free };
        let mut tmp = pushed;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 8 } else { t - 8 };
            try_push!(f, t);
            tmp &= tmp - 1;
        }
        let _start_rank: u64 = if wturn { 0x00FF000000000000u64 >> 8 } else { 0x000000FF00000000u64 >> (8*3) };
        let start_rank: u64 = if wturn { 0x00FF000000000000u64 }
                              else      { 0x000000000000FF00u64 };
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
        let att_c1 = if wturn { (normal_pawns & !0x0101010101010101u64) >> 9 }
                     else     { (normal_pawns & !0x0101010101010101u64) << 7 };
        let att_c2 = if wturn { (normal_pawns & !0x8080808080808080u64) >> 7 }
                     else     { (normal_pawns & !0x8080808080808080u64) << 9 };
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

        let promo_push = if wturn { promo_pawns >> 8 & free }
                         else     { promo_pawns << 8 & free };
        let mut tmp = promo_push;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 8 } else { t - 8 };
            try_push_promo!(f, t);
            tmp &= tmp - 1;
        }
        let pc1 = if wturn { (promo_pawns & !0x0101010101010101u64) >> 9 }
                  else     { (promo_pawns & !0x0101010101010101u64) << 7 };
        let mut tmp = pc1 & opp;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 9 } else { t - 7 };
            try_push_promo!(f, t);
            tmp &= tmp - 1;
        }
        let pc2 = if wturn { (promo_pawns & !0x8080808080808080u64) >> 7 }
                  else     { (promo_pawns & !0x8080808080808080u64) << 9 };
        let mut tmp = pc2 & opp;
        while tmp != 0 {
            let t = tmp.trailing_zeros() as usize;
            let f = if wturn { t + 7 } else { t - 9 };
            try_push_promo!(f, t);
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
            bb2[if wturn {WK} else {BK}] &= !bit(kf);
            bb2[if wturn {WK} else {BK}] |=  bit(t);
            let cap = piece_on(&st.bb, t);
            if cap != EMPTY_SQ { bb2[cap as usize] &= !bit(t); }
            if !is_attacked(&bb2, t, !wturn) {
                result.push([sq_r(kf), sq_c(kf), sq_r(t), sq_c(t)]);
            }
            att &= att - 1;
        }
        if !in_check {
            let rook_pi = if wturn { WR } else { BR };
            let kr = if wturn { 7usize } else { 0usize };
            if cr[if wturn {0} else {2}]
                && st.bb[rook_pi] & bit(sq(kr,7)) != 0
                && all_occ(&st.bb) & (bit(sq(kr,5))|bit(sq(kr,6))) == 0
                && !is_attacked(&st.bb, sq(kr,4), !wturn)
                && !is_attacked(&st.bb, sq(kr,5), !wturn)
                && !is_attacked(&st.bb, sq(kr,6), !wturn)
            {
                result.push([kr, 4, kr, 6]);
            }
            if cr[if wturn {1} else {3}]
                && st.bb[rook_pi] & bit(sq(kr,0)) != 0
                && all_occ(&st.bb) & (bit(sq(kr,1))|bit(sq(kr,2))|bit(sq(kr,3))) == 0
                && !is_attacked(&st.bb, sq(kr,4), !wturn)
                && !is_attacked(&st.bb, sq(kr,3), !wturn)
                && !is_attacked(&st.bb, sq(kr,2), !wturn)
            {
                result.push([kr, 4, kr, 2]);
            }
        }
    }

    result
}