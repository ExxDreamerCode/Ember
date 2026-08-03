use crate::backend::{
    default_search_backend, parse_search_backend_name, search_backend_available, SearchBackendKind,
};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

const SEARCH_BACKEND_ENV: &str = "EMBER_SEARCH_BACKEND";

static SEARCH_BACKEND: OnceLock<SearchBackendKind> = OnceLock::new();
static SEARCH_BACKEND_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[inline]
pub fn active_search_backend() -> SearchBackendKind {
    if let Some(backend) = search_backend_from_id(SEARCH_BACKEND_OVERRIDE.load(Ordering::Relaxed)) {
        return backend;
    }
    *SEARCH_BACKEND.get_or_init(detect_search_backend)
}

pub fn set_search_backend_override(backend: Option<SearchBackendKind>) -> bool {
    if backend.is_some_and(|backend| !search_backend_available(backend)) {
        return false;
    }
    let id = backend.map(search_backend_id).unwrap_or(0);
    SEARCH_BACKEND_OVERRIDE.store(id, Ordering::SeqCst);
    true
}

fn detect_search_backend() -> SearchBackendKind {
    if let Ok(value) = std::env::var(SEARCH_BACKEND_ENV) {
        if let Some(backend) = parse_search_backend_name(&value) {
            if search_backend_available(backend) {
                return backend;
            }
        }
    }

    default_search_backend()
}

fn search_backend_id(backend: SearchBackendKind) -> u8 {
    match backend {
        SearchBackendKind::Scalar => 1,
        SearchBackendKind::X86V3 => 2,
        SearchBackendKind::Aarch64Simd128 => 3,
        SearchBackendKind::Aarch64Simd256 => 5,
        SearchBackendKind::Aarch64Simd512 => 6,
        SearchBackendKind::X86Avx512 => 4,
    }
}

fn search_backend_from_id(id: u8) -> Option<SearchBackendKind> {
    match id {
        1 => Some(SearchBackendKind::Scalar),
        2 => Some(SearchBackendKind::X86V3),
        3 => Some(SearchBackendKind::Aarch64Simd128),
        4 => Some(SearchBackendKind::X86Avx512),
        5 => Some(SearchBackendKind::Aarch64Simd256),
        6 => Some(SearchBackendKind::Aarch64Simd512),
        _ => None,
    }
}

use crate::board::{
    is_attacked, move_ec, move_er, move_from, move_promotion, move_sc, move_sr, move_to_uci,
    piece_type, BoardState, Move, EMPTY_SQ, MAX_PLY,
};
use crate::movegen::{apply_move, generate_moves};
use crate::tt::SharedTT;
use std::collections::HashSet;

fn malformed_promotion_move(st: &BoardState, mv: Move) -> bool {
    let promo = move_promotion(mv).to_ascii_uppercase();
    let from = move_from(mv);
    let to_rank = move_er(mv);
    let fpi = st.mailbox[from];
    let reaches_back_rank = to_rank == 0 || to_rank == 7;
    let valid_promo = matches!(promo, b'Q' | b'R' | b'B' | b'N');

    if promo != 0 {
        return fpi == EMPTY_SQ || piece_type(fpi) != 0 || !reaches_back_rank || !valid_promo;
    }

    fpi != EMPTY_SQ && piece_type(fpi) == 0 && reaches_back_rank
}

pub fn format_pv_line_uci(st: &BoardState, pv_line: &[Move]) -> String {
    let mut current = *st;
    let mut out = Vec::with_capacity(pv_line.len());

    for &mv in pv_line {
        if malformed_promotion_move(&current, mv) {
            break;
        }

        let legal_moves = generate_moves(&current, current.w, &current.cr, current.ep);
        if !legal_moves.contains(&mv) {
            break;
        }

        out.push(move_to_uci(&current, mv));
        apply_move(
            &mut current,
            move_sr(mv),
            move_sc(mv),
            move_er(mv),
            move_ec(mv),
            move_promotion(mv),
        );
    }

    out.join(" ")
}

pub fn extract_pv_line(shared_tt: &SharedTT, st: &BoardState, first_move: Move) -> Vec<Move> {
    if malformed_promotion_move(st, first_move) {
        return vec![];
    }

    let mut pv = vec![first_move];
    let mut prev_st = *st;
    apply_move(
        &mut prev_st,
        move_sr(first_move),
        move_sc(first_move),
        move_er(first_move),
        move_ec(first_move),
        move_promotion(first_move),
    );

    let moved_king_sq = prev_st.king_sq(!prev_st.w);
    if moved_king_sq == 0 || is_attacked(&prev_st.bb, moved_king_sq, prev_st.w) {
        return pv;
    }

    let mut seen_hashes = HashSet::new();
    seen_hashes.insert(st.hash);
    seen_hashes.insert(prev_st.hash);

    for _ in 0..MAX_PLY.saturating_sub(1) {
        let h = prev_st.hash;
        if let Some((_, _, _, Some(best))) = shared_tt.get_depth(h) {
            let moves = generate_moves(&prev_st, prev_st.w, &prev_st.cr, prev_st.ep);
            if !moves.contains(&best) {
                break;
            }
            if malformed_promotion_move(&prev_st, best) {
                break;
            }
            let promo = move_promotion(best);
            pv.push(best);
            apply_move(
                &mut prev_st,
                move_sr(best),
                move_sc(best),
                move_er(best),
                move_ec(best),
                promo,
            );
            let moved_king_sq = prev_st.king_sq(!prev_st.w);
            if moved_king_sq == 0 || is_attacked(&prev_st.bb, moved_king_sq, prev_st.w) {
                pv.pop();
                break;
            }
            let h_after = prev_st.hash;
            if !seen_hashes.insert(h_after) {
                pv.pop();
                break;
            }
        } else {
            break;
        }
    }
    pv
}
