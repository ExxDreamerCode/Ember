use super::{convert, NNUENet};
use crate::board::{BoardState, BB, BK, BN, BP, BQ, BR, WB, WK, WN, WP, WQ, WR};
use crate::types::{BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};

const NNUE_NUM_PIECE_TYPES: usize = 12;

const CONSENSUS_BUCKETS: [[usize; 8]; 4] = [
    [0, 4, 8, 8, 12, 12, 14, 14],
    [1, 5, 9, 9, 12, 12, 14, 14],
    [2, 6, 10, 10, 13, 13, 15, 15],
    [3, 7, 11, 11, 13, 13, 15, 15],
];

const THREAT_INTERACTION: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
];
const THREAT_TARGETS: [i32; 6] = [6, 10, 8, 8, 10, 8];
const THREAT_COLORED_PIECES: usize = 12;

#[derive(Clone, Copy, Default)]
struct ThreatPair {
    base: i32,
    tracked: bool,
    symmetric: bool,
}

impl ThreatPair {
    #[inline(always)]
    fn skips(self, from: u32, to: u32) -> bool {
        !self.tracked || (self.symmetric && from < to)
    }
}

struct ThreatTables {
    pairs: [[ThreatPair; THREAT_COLORED_PIECES]; THREAT_COLORED_PIECES],
    from_offset: [[i32; 64]; THREAT_COLORED_PIECES],
    ray_rank: [[[u8; 64]; 64]; THREAT_COLORED_PIECES],
    feature_count: usize,
}

static THREAT_TABLES: std::sync::OnceLock<ThreatTables> = std::sync::OnceLock::new();

#[derive(Copy, Clone, Debug)]
pub enum KbLayout {
    Uniform = 0,
    Consensus = 1,
    Reckless = 2,
}

impl KbLayout {
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(KbLayout::Uniform),
            1 => Some(KbLayout::Consensus),
            2 => Some(KbLayout::Reckless),
            _ => None,
        }
    }
}

pub fn compute_king_buckets(layout: KbLayout) -> ([usize; 64], [bool; 64]) {
    let mut kb = [0; 64];
    let mut km = [false; 64];

    for sq in 0..64 {
        let f = sq % 8;
        let r = sq / 8;
        let (mf, mirror) = if f >= 4 { (7 - f, true) } else { (f, false) };

        kb[sq] = match layout {
            KbLayout::Uniform => mf * 4 + r / 2,
            KbLayout::Consensus => CONSENSUS_BUCKETS[mf][r],
            KbLayout::Reckless => {
                let t = [
                    0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 6, 7, 7, 6, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                ];
                t[sq]
            }
        };
        km[sq] = mirror;
    }
    (kb, km)
}

fn threat_tables() -> &'static ThreatTables {
    THREAT_TABLES.get_or_init(build_threat_tables)
}

pub fn threat_feature_count() -> usize {
    threat_tables().feature_count
}

fn build_threat_tables() -> ThreatTables {
    let mut pairs = [[ThreatPair::default(); THREAT_COLORED_PIECES]; THREAT_COLORED_PIECES];
    let mut from_offset = [[0i32; 64]; THREAT_COLORED_PIECES];
    let mut ray_rank = [[[0u8; 64]; 64]; THREAT_COLORED_PIECES];
    let mut slots = [0i32; THREAT_COLORED_PIECES];
    let mut block_base = [0i32; THREAT_COLORED_PIECES];
    let mut next_base = 0i32;

    for color in 0..2 {
        for (pt, targets) in THREAT_TARGETS.iter().copied().enumerate() {
            let cp = color * 6 + pt;
            let mut count = 0i32;
            for sq in 0..64u32 {
                from_offset[cp][sq as usize] = count;
                if pt == PAWN as usize && !(8..56).contains(&sq) {
                    continue;
                }
                count += threat_attacks_empty(cp, sq).count_ones() as i32;
            }
            slots[cp] = count;
            block_base[cp] = next_base;
            next_base += targets * count;
        }
    }

    for (attacker, attacker_pairs) in pairs.iter_mut().enumerate() {
        let attacker_type = attacker % 6;
        let attacker_color = attacker / 6;
        for (victim, pair) in attacker_pairs.iter_mut().enumerate() {
            let victim_type = victim % 6;
            let victim_color = victim / 6;
            let map = THREAT_INTERACTION[attacker_type][victim_type];
            let tracked = map >= 0;
            let symmetric = attacker_type == victim_type
                && (attacker_color != victim_color || attacker_type != 0);
            let color_group = victim_color as i32 * (THREAT_TARGETS[attacker_type] / 2);
            *pair = ThreatPair {
                base: block_base[attacker] + (color_group + map.max(0)) * slots[attacker],
                tracked,
                symmetric,
            };
        }
    }

    for (cp, piece_ranks) in ray_rank.iter_mut().enumerate() {
        for from in 0..64u32 {
            let attacks = threat_attacks_empty(cp, from);
            for to in 0..64u32 {
                let below = if to == 0 { 0 } else { (1u64 << to) - 1 };
                piece_ranks[from as usize][to as usize] = (below & attacks).count_ones() as u8;
            }
        }
    }

    ThreatTables {
        pairs,
        from_offset,
        ray_rank,
        feature_count: next_base as usize,
    }
}

fn threat_attacks_empty(colored_piece: usize, square: u32) -> u64 {
    let piece_type = colored_piece % 6;
    let color = (colored_piece / 6) as u8;
    match piece_type as u8 {
        PAWN => threat_pawn_attacks(color, square),
        KNIGHT => threat_knight_attacks(square),
        BISHOP => threat_slider_attacks_empty(square, &[(1, 1), (1, -1), (-1, 1), (-1, -1)]),
        ROOK => threat_slider_attacks_empty(square, &[(1, 0), (-1, 0), (0, 1), (0, -1)]),
        QUEEN => {
            threat_slider_attacks_empty(square, &[(1, 1), (1, -1), (-1, 1), (-1, -1)])
                | threat_slider_attacks_empty(square, &[(1, 0), (-1, 0), (0, 1), (0, -1)])
        }
        KING => threat_king_attacks(square),
        _ => 0,
    }
}

fn threat_pawn_attacks(color: u8, square: u32) -> u64 {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    const FILE_H: u64 = 0x8080_8080_8080_8080;
    let bb = 1u64 << square;
    if color == WHITE {
        ((bb & !FILE_A) << 7) | ((bb & !FILE_H) << 9)
    } else {
        ((bb & !FILE_H) >> 7) | ((bb & !FILE_A) >> 9)
    }
}

fn threat_knight_attacks(square: u32) -> u64 {
    threat_leaper_attacks(
        square,
        &[
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ],
    )
}

fn threat_king_attacks(square: u32) -> u64 {
    threat_leaper_attacks(
        square,
        &[
            (1, 1),
            (1, 0),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ],
    )
}

fn threat_leaper_attacks(square: u32, deltas: &[(i32, i32)]) -> u64 {
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let mut attacks = 0u64;
    for &(df, dr) in deltas {
        let next_file = file + df;
        let next_rank = rank + dr;
        if (0..8).contains(&next_file) && (0..8).contains(&next_rank) {
            attacks |= 1u64 << (next_rank * 8 + next_file);
        }
    }
    attacks
}

fn threat_slider_attacks_empty(square: u32, deltas: &[(i32, i32)]) -> u64 {
    let file = (square % 8) as i32;
    let rank = (square / 8) as i32;
    let mut attacks = 0u64;
    for &(df, dr) in deltas {
        let mut next_file = file + df;
        let mut next_rank = rank + dr;
        while (0..8).contains(&next_file) && (0..8).contains(&next_rank) {
            attacks |= 1u64 << (next_rank * 8 + next_file);
            next_file += df;
            next_rank += dr;
        }
    }
    attacks
}

#[inline(always)]
fn threat_colored_piece(color: u8, piece_type: u8) -> usize {
    color as usize * 6 + piece_type as usize
}

#[inline(always)]
fn threat_index(
    attacker: usize,
    from: u32,
    victim: usize,
    to: u32,
    mirrored: bool,
    perspective: u8,
) -> Option<usize> {
    let attacker = if perspective == BLACK {
        (attacker + 6) % 12
    } else {
        attacker
    };
    let victim = if perspective == BLACK {
        (victim + 6) % 12
    } else {
        victim
    };

    let tables = threat_tables();
    let pair = tables.pairs[attacker][victim];
    if pair.skips(from, to) {
        return None;
    }

    let flip = (u32::from(mirrored) * 7) ^ (u32::from(perspective) * 56);
    let from = (from ^ flip) as usize;
    let to = (to ^ flip) as usize;
    Some(
        (pair.base
            + tables.from_offset[attacker][from]
            + i32::from(tables.ray_rank[attacker][from][to])) as usize,
    )
}

pub(super) fn halfka_idx(
    kb: &[usize; 64],
    km: &[bool; 64],
    persp: u8,
    ks: u8,
    pc: u8,
    pt: u8,
    ps: u8,
) -> usize {
    let mut k = ks as usize;
    let mut p = ps as usize;
    let mut pi = (pc as usize) * 6 + pt as usize;

    if persp == 1 {
        k ^= 56;
        p ^= 56;
        pi = if pi >= 6 { pi - 6 } else { pi + 6 };
    }
    if km[k] {
        p = (p & !7) | (7 - (p & 7));
    }
    kb[k] * (NNUE_NUM_PIECE_TYPES * 64) + pi * 64 + p
}

pub fn output_bucket(pc: u32) -> usize {
    let b = (pc as i32 - 2) / 4;
    b.clamp(0, 7) as usize
}

pub struct NNUEThreatAccumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    hs: usize,
}

impl Clone for NNUEThreatAccumulator {
    fn clone(&self) -> Self {
        Self {
            white: self.white.clone(),
            black: self.black.clone(),
            hs: self.hs,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.white.clone_from(&source.white);
        self.black.clone_from(&source.black);
        self.hs = source.hs;
    }
}

impl NNUEThreatAccumulator {
    pub fn new(hs: usize) -> Self {
        Self {
            white: vec![0; hs],
            black: vec![0; hs],
            hs,
        }
    }

    pub fn white(&self) -> &[i16] {
        &self.white
    }

    pub fn black(&self) -> &[i16] {
        &self.black
    }

    pub fn refresh(&mut self, net: &NNUENet, st: &BoardState) {
        if self.white.len() != net.hidden_size {
            self.white.resize(net.hidden_size, 0);
        }
        if self.black.len() != net.hidden_size {
            self.black.resize(net.hidden_size, 0);
        }
        self.hs = net.hidden_size;
        Self::refresh_perspective(&mut self.white, net, st, WHITE);
        Self::refresh_perspective(&mut self.black, net, st, BLACK);
    }

    pub fn update_from_parent(
        &mut self,
        parent: &Self,
        net: &NNUENet,
        before: &BoardState,
        after: &BoardState,
    ) -> bool {
        if !net.has_threat_features() {
            self.clone_from(parent);
            return true;
        }
        if before.king_sq(true) != after.king_sq(true)
            || before.king_sq(false) != after.king_sq(false)
        {
            return false;
        }

        self.clone_from(parent);
        if self.white.len() != net.hidden_size {
            self.white.resize(net.hidden_size, 0);
        }
        if self.black.len() != net.hidden_size {
            self.black.resize(net.hidden_size, 0);
        }
        self.hs = net.hidden_size;

        let mut changed_squares = Vec::with_capacity(4);
        let mut before_candidates = [0u64; THREAT_COLORED_PIECES];
        let mut after_candidates = [0u64; THREAT_COLORED_PIECES];

        for piece in 0..THREAT_COLORED_PIECES {
            let diff = before.bb[piece] ^ after.bb[piece];
            let mut squares = diff;
            while squares != 0 {
                let square = squares.trailing_zeros() as usize;
                squares &= squares - 1;
                if !changed_squares.contains(&square) {
                    changed_squares.push(square);
                }
                if before.bb[piece] & (1u64 << square) != 0 {
                    before_candidates[piece] |= 1u64 << square;
                }
                if after.bb[piece] & (1u64 << square) != 0 {
                    after_candidates[piece] |= 1u64 << square;
                }
            }
        }

        if changed_squares.is_empty() {
            return true;
        }

        for &square in &changed_squares {
            collect_threat_attackers(before, square, &mut before_candidates);
            collect_threat_attackers(after, square, &mut after_candidates);
        }

        for piece in 0..THREAT_COLORED_PIECES {
            let mut before_squares = before_candidates[piece] & before.bb[piece];
            while before_squares != 0 {
                let square = before_squares.trailing_zeros() as usize;
                before_squares &= before_squares - 1;
                self.apply_piece_threats(net, before, piece, square, -1);
            }

            let mut after_squares = after_candidates[piece] & after.bb[piece];
            while after_squares != 0 {
                let square = after_squares.trailing_zeros() as usize;
                after_squares &= after_squares - 1;
                self.apply_piece_threats(net, after, piece, square, 1);
            }
        }

        true
    }

    fn refresh_perspective(acc: &mut [i16], net: &NNUENet, st: &BoardState, perspective: u8) {
        acc.fill(0);
        if !net.has_threat_features() {
            return;
        }

        let king_square = convert(st.king_sq(perspective == WHITE) as u8) as u32;
        let mirrored = king_square % 8 >= 4;
        let white = color_occupancy(st, WHITE);
        let black = color_occupancy(st, BLACK);
        let occ = white | black;
        let mailbox = threat_mailbox(st);

        for color in [WHITE, BLACK] {
            for piece_type in 0..6u8 {
                let piece_index = color as usize * 6 + piece_type as usize;
                let mut pieces = st.bb[piece_index];
                while pieces != 0 {
                    let square = pieces.trailing_zeros() as usize;
                    pieces &= pieces - 1;

                    let mut attacks =
                        threat_piece_attacks_on_board(piece_type, color, square, occ) & occ;
                    let attacker = threat_colored_piece(color, piece_type);
                    while attacks != 0 {
                        let target = attacks.trailing_zeros() as usize;
                        attacks &= attacks - 1;

                        let victim = mailbox[target];
                        if victim >= THREAT_COLORED_PIECES {
                            continue;
                        }
                        let Some(index) = threat_index(
                            attacker,
                            u32::from(convert(square as u8)),
                            victim,
                            u32::from(convert(target as u8)),
                            mirrored,
                            perspective,
                        ) else {
                            continue;
                        };
                        if index < net.num_threat_features {
                            Self::add_threat_row(acc, net, index);
                        }
                    }
                }
            }
        }
    }

    fn add_threat_row(acc: &mut [i16], net: &NNUENet, index: usize) {
        let start = index * net.hidden_size;
        let row = &net.threat_weights[start..start + net.hidden_size];
        for (slot, &weight) in acc.iter_mut().zip(row) {
            *slot += i16::from(weight);
        }
    }

    fn sub_threat_row(acc: &mut [i16], net: &NNUENet, index: usize) {
        let start = index * net.hidden_size;
        let row = &net.threat_weights[start..start + net.hidden_size];
        for (slot, &weight) in acc.iter_mut().zip(row) {
            *slot -= i16::from(weight);
        }
    }

    fn apply_piece_threats(
        &mut self,
        net: &NNUENet,
        st: &BoardState,
        piece: usize,
        square: usize,
        sign: i16,
    ) {
        let color = (piece / 6) as u8;
        let piece_type = (piece % 6) as u8;
        let white = color_occupancy(st, WHITE);
        let black = color_occupancy(st, BLACK);
        let occ = white | black;
        let mailbox = threat_mailbox(st);
        let attacker = threat_colored_piece(color, piece_type);
        let mut attacks = threat_piece_attacks_on_board(piece_type, color, square, occ) & occ;

        while attacks != 0 {
            let target = attacks.trailing_zeros() as usize;
            attacks &= attacks - 1;
            let victim = mailbox[target];
            if victim >= THREAT_COLORED_PIECES {
                continue;
            }
            for perspective in [WHITE, BLACK] {
                let king_square = convert(st.king_sq(perspective == WHITE) as u8) as u32;
                let mirrored = king_square % 8 >= 4;
                let Some(index) = threat_index(
                    attacker,
                    u32::from(convert(square as u8)),
                    victim,
                    u32::from(convert(target as u8)),
                    mirrored,
                    perspective,
                ) else {
                    continue;
                };
                if index >= net.num_threat_features {
                    continue;
                }
                let acc = if perspective == WHITE {
                    &mut self.white
                } else {
                    &mut self.black
                };
                if sign > 0 {
                    Self::add_threat_row(acc, net, index);
                } else {
                    Self::sub_threat_row(acc, net, index);
                }
            }
        }
    }
}

fn color_occupancy(st: &BoardState, color: u8) -> u64 {
    let base = if color == WHITE { 0 } else { 6 };
    st.bb[base]
        | st.bb[base + 1]
        | st.bb[base + 2]
        | st.bb[base + 3]
        | st.bb[base + 4]
        | st.bb[base + 5]
}

fn threat_mailbox(st: &BoardState) -> [usize; 64] {
    let mut mailbox = [THREAT_COLORED_PIECES; 64];
    for piece in 0..THREAT_COLORED_PIECES {
        let mut pieces = st.bb[piece];
        while pieces != 0 {
            let square = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            mailbox[square] = piece;
        }
    }
    mailbox
}

fn threat_piece_attacks_on_board(piece_type: u8, color: u8, square: usize, occ: u64) -> u64 {
    let bb = 1u64 << square;
    match piece_type {
        PAWN => {
            const FILE_A: u64 = 0x0101_0101_0101_0101;
            const FILE_H: u64 = 0x8080_8080_8080_8080;
            if color == WHITE {
                ((bb & !FILE_H) >> 7) | ((bb & !FILE_A) >> 9)
            } else {
                ((bb & !FILE_A) << 7) | ((bb & !FILE_H) << 9)
            }
        }
        KNIGHT => crate::board::KNIGHT_ATTACKS[square],
        BISHOP => crate::magic::bishop_attacks(square, occ),
        ROOK => crate::magic::rook_attacks(square, occ),
        QUEEN => {
            crate::magic::bishop_attacks(square, occ) | crate::magic::rook_attacks(square, occ)
        }
        KING => crate::board::KING_ATTACKS[square],
        _ => 0,
    }
}

fn collect_threat_attackers(st: &BoardState, target: usize, candidates: &mut [u64; 12]) {
    let occ = color_occupancy(st, WHITE) | color_occupancy(st, BLACK);
    let bit = 1u64 << target;
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    const FILE_H: u64 = 0x8080_8080_8080_8080;

    let white_pawns = st.bb[WP] & (((bit & !FILE_A) << 7) | ((bit & !FILE_H) << 9));
    let black_pawns = st.bb[BP] & (((bit & !FILE_A) >> 9) | ((bit & !FILE_H) >> 7));
    candidates[WP] |= white_pawns;
    candidates[BP] |= black_pawns;

    candidates[WN] |= st.bb[WN] & crate::board::KNIGHT_ATTACKS[target];
    candidates[BN] |= st.bb[BN] & crate::board::KNIGHT_ATTACKS[target];
    candidates[WK] |= st.bb[WK] & crate::board::KING_ATTACKS[target];
    candidates[BK] |= st.bb[BK] & crate::board::KING_ATTACKS[target];

    let bishop_attackers = crate::magic::bishop_attacks(target, occ);
    candidates[WB] |= st.bb[WB] & bishop_attackers;
    candidates[WQ] |= st.bb[WQ] & bishop_attackers;
    candidates[BB] |= st.bb[BB] & bishop_attackers;
    candidates[BQ] |= st.bb[BQ] & bishop_attackers;

    let rook_attackers = crate::magic::rook_attacks(target, occ);
    candidates[WR] |= st.bb[WR] & rook_attackers;
    candidates[WQ] |= st.bb[WQ] & rook_attackers;
    candidates[BR] |= st.bb[BR] & rook_attackers;
    candidates[BQ] |= st.bb[BQ] & rook_attackers;
}
