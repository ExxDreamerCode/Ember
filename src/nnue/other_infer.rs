use super::other_nets::{OtherNetData, OtherStack};
use crate::board::{BoardState, EMPTY_SQ, KING_ATTACKS, KNIGHT_ATTACKS};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi16,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
};

const PSQ_DIMS: usize = 22_528;
const THREAT_DIMS: usize = 60_720;
const HIDDEN_SIZE: usize = 1024;
const PSQT_BUCKETS: usize = 8;

const OUTPUT_SCALE: i32 = 16;
const WEIGHT_SCALE_BITS: i32 = 6;
const FT_MAX_VAL: i32 = 255;
const HIDDEN_ONE_VAL: i32 = 128;

// FullThreats::numValidTargets in Stockfish piece order, represented with
// Ember's contiguous white-then-black piece numbering.
const NUM_VALID_TARGETS: [u32; 12] = [6, 10, 8, 8, 10, 0, 6, 10, 8, 8, 10, 0];

// FullThreats::map, with Stockfish's 1-based PieceType values normalized to
// Ember's 0-based piece types.
const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

#[inline(always)]
fn ember_to_stockfish_square(square: u32) -> u32 {
    square ^ 56
}

#[inline(always)]
fn halfka_orientation(stockfish_king_square: u32) -> u32 {
    if stockfish_king_square & 4 == 0 {
        7
    } else {
        0
    }
}

#[inline(always)]
fn threat_orientation(stockfish_king_square: u32) -> u32 {
    if stockfish_king_square & 4 == 0 {
        0
    } else {
        7
    }
}

#[inline(always)]
fn swap_piece_color(piece: u32) -> u32 {
    if piece < 6 {
        piece + 6
    } else {
        piece - 6
    }
}

fn halfka_piece_base(perspective: u32, piece: u32) -> u32 {
    let piece = if perspective == 0 {
        piece
    } else {
        swap_piece_color(piece)
    };
    match piece {
        0 => 0,
        1 => 128,
        2 => 256,
        3 => 384,
        4 => 512,
        5 => 640,
        6 => 64,
        7 => 192,
        8 => 320,
        9 => 448,
        10 => 576,
        11 => 640,
        _ => unreachable!("invalid Ember piece index"),
    }
}

fn halfka_index(perspective: u32, square: u32, piece: u32, king_square: u32) -> usize {
    let square = ember_to_stockfish_square(square);
    let king_square = ember_to_stockfish_square(king_square);
    let flip = 56 * perspective;
    let oriented_king = king_square ^ flip;
    let king_rank = oriented_king / 8;
    let king_file = oriented_king % 8;
    let king_bucket = (7 - king_rank) * 4 + core::cmp::min(king_file, 7 - king_file);

    ((square ^ halfka_orientation(king_square) ^ flip)
        + halfka_piece_base(perspective, piece)
        + king_bucket * 11 * 64) as usize
}

fn pawn_attacks_bb(color: u32, square: u32) -> u64 {
    let bb = 1u64 << square;
    const A_FILE: u64 = 0x0101_0101_0101_0101;
    const H_FILE: u64 = 0x8080_8080_8080_8080;
    if color == 0 {
        ((bb & !H_FILE) >> 7) | ((bb & !A_FILE) >> 9)
    } else {
        ((bb & !H_FILE) << 9) | ((bb & !A_FILE) << 7)
    }
}

fn pawn_forward_square(color: u32, square: u32) -> Option<u32> {
    match color {
        0 if square >= 8 => Some(square - 8),
        1 if square < 56 => Some(square + 8),
        _ => None,
    }
}

fn attacks_bb(piece_type: u32, from: u32, occupancy: u64) -> u64 {
    match piece_type {
        1 => KNIGHT_ATTACKS[from as usize],
        2 => crate::magic::bishop_attacks(from as usize, occupancy),
        3 => crate::magic::rook_attacks(from as usize, occupancy),
        4 => {
            crate::magic::bishop_attacks(from as usize, occupancy)
                | crate::magic::rook_attacks(from as usize, occupancy)
        }
        5 => KING_ATTACKS[from as usize],
        _ => 0,
    }
}

fn stockfish_pseudo_attacks(piece: u32, from: u32) -> u64 {
    let piece_type = piece % 6;
    if piece_type == 0 {
        let bb = 1u64 << from;
        const A_FILE: u64 = 0x0101_0101_0101_0101;
        const H_FILE: u64 = 0x8080_8080_8080_8080;
        if piece / 6 == 0 {
            (bb << 8) | ((bb & !H_FILE) << 9) | ((bb & !A_FILE) << 7)
        } else {
            (bb >> 8) | ((bb & !H_FILE) >> 7) | ((bb & !A_FILE) >> 9)
        }
    } else {
        let ember_from = ember_to_stockfish_square(from);
        attacks_bb(piece_type, ember_from, 0).swap_bytes()
    }
}

fn piece_at(state: &BoardState, square: u32) -> u32 {
    debug_assert_ne!(
        state.mailbox[square as usize], EMPTY_SQ,
        "threat target must be occupied"
    );
    state.mailbox[square as usize] as u32
}

struct ThreatLut {
    index_lut1: Box<[u32]>,
    offsets: Box<[u32]>,
    index_lut2: Box<[u16]>,
}

impl ThreatLut {
    fn build() -> Self {
        let mut index_lut2 = vec![0u16; 12 * 64 * 64];
        for piece in 0..12u32 {
            for from in 0..64u32 {
                let attacks = if piece % 6 == 0 && !(1..=6).contains(&(from / 8)) {
                    0
                } else {
                    stockfish_pseudo_attacks(piece, from)
                };
                for to in 0..64u32 {
                    let before = (1u64 << to).wrapping_sub(1);
                    index_lut2[((piece * 64 + from) * 64 + to) as usize] =
                        (before & attacks).count_ones() as u16;
                }
            }
        }

        let mut helper = [(0usize, 0usize); 12];
        let mut offsets = vec![0u32; 12 * 64];
        let mut cumulative = 0usize;
        for piece in 0..12u32 {
            let mut cumulative_piece = 0usize;
            for from in 0..64u32 {
                offsets[(piece * 64 + from) as usize] = cumulative_piece as u32;
                let count = if piece % 6 == 0 && !(1..=6).contains(&(from / 8)) {
                    0
                } else {
                    stockfish_pseudo_attacks(piece, from).count_ones() as usize
                };
                cumulative_piece += count;
            }
            helper[piece as usize] = (cumulative_piece, cumulative);
            cumulative += NUM_VALID_TARGETS[piece as usize] as usize * cumulative_piece;
        }
        assert_eq!(cumulative, THREAT_DIMS);

        let mut index_lut1 = vec![THREAT_DIMS as u32; 12 * 12 * 2];
        for attacker in 0..12u32 {
            for attacked in 0..12u32 {
                let attacker_type = attacker % 6;
                let attacked_type = attacked % 6;
                let map = THREAT_MAP[attacker_type as usize][attacked_type as usize];
                let enemy = attacker / 6 != attacked / 6;
                let semi_excluded = attacker_type == attacked_type && (enemy || attacker_type != 0);
                if map < 0 {
                    continue;
                }

                let (piece_dimensions, piece_offset) = helper[attacker as usize];
                let feature = piece_offset
                    + (attacked / 6 * (NUM_VALID_TARGETS[attacker as usize] / 2) + map as u32)
                        as usize
                        * piece_dimensions;
                let base = (attacker as usize * 12 + attacked as usize) * 2;
                index_lut1[base] = feature as u32;
                if !semi_excluded {
                    index_lut1[base + 1] = feature as u32;
                }
            }
        }

        Self {
            index_lut1: index_lut1.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            index_lut2: index_lut2.into_boxed_slice(),
        }
    }

    fn make_index(
        &self,
        perspective: u32,
        attacker: u32,
        from: u32,
        to: u32,
        attacked: u32,
        king_square: u32,
    ) -> usize {
        let from = ember_to_stockfish_square(from);
        let to = ember_to_stockfish_square(to);
        let king_square = ember_to_stockfish_square(king_square);
        let orientation = threat_orientation(king_square) ^ (56 * perspective);
        let from = from ^ orientation;
        let to = to ^ orientation;
        let attacker = if perspective == 0 {
            attacker
        } else {
            swap_piece_color(attacker)
        };
        let attacked = if perspective == 0 {
            attacked
        } else {
            swap_piece_color(attacked)
        };

        let lut1 = self.index_lut1
            [(attacker as usize * 12 + attacked as usize) * 2 + usize::from(from < to)]
            as usize;
        if lut1 >= THREAT_DIMS {
            return THREAT_DIMS;
        }
        lut1 + self.offsets[(attacker * 64 + from) as usize] as usize
            + self.index_lut2[((attacker * 64 + from) * 64 + to) as usize] as usize
    }
}

fn threat_lut() -> &'static ThreatLut {
    static LUT: OnceLock<ThreatLut> = OnceLock::new();
    LUT.get_or_init(ThreatLut::build)
}

fn active_threat_indices(state: &BoardState, perspective: u32) -> Vec<usize> {
    let lut = threat_lut();
    let king_square = find_king(state, perspective);
    let occupancy = state.bb.iter().copied().fold(0u64, |all, bb| all | bb);
    let all_pawns = state.bb[0] | state.bb[6];
    let mut active = Vec::with_capacity(128);

    for color in 0..2u32 {
        let pawns = state.bb[(color * 6) as usize];
        let mut pieces = pawns;
        while pieces != 0 {
            let from = pieces.trailing_zeros();
            pieces &= pieces - 1;

            let mut attacks = pawn_attacks_bb(color, from) & occupancy;
            while attacks != 0 {
                let to = attacks.trailing_zeros();
                attacks &= attacks - 1;
                let index = lut.make_index(
                    perspective,
                    color * 6,
                    from,
                    to,
                    piece_at(state, to),
                    king_square,
                );
                if index < THREAT_DIMS {
                    active.push(index);
                }
            }

            if let Some(to) = pawn_forward_square(color, from) {
                if all_pawns & (1u64 << to) != 0 {
                    let index = lut.make_index(
                        perspective,
                        color * 6,
                        from,
                        to,
                        piece_at(state, to),
                        king_square,
                    );
                    if index < THREAT_DIMS {
                        active.push(index);
                    }
                }
            }
        }

        for piece_type in 1..5u32 {
            let piece = color * 6 + piece_type;
            let mut pieces = state.bb[piece as usize];
            while pieces != 0 {
                let from = pieces.trailing_zeros();
                pieces &= pieces - 1;
                let mut attacks = attacks_bb(piece_type, from, occupancy) & occupancy;
                while attacks != 0 {
                    let to = attacks.trailing_zeros();
                    attacks &= attacks - 1;
                    let index = lut.make_index(
                        perspective,
                        piece,
                        from,
                        to,
                        piece_at(state, to),
                        king_square,
                    );
                    if index < THREAT_DIMS {
                        active.push(index);
                    }
                }
            }
        }
    }

    active
}

fn find_king(state: &BoardState, perspective: u32) -> u32 {
    state.bb[if perspective == 0 { 5 } else { 11 }].trailing_zeros()
}

fn add_psq_row(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i16],
    psqt_row: &[i32],
) {
    crate::simd::simd_add_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_add(*weight);
    }
}

fn add_threat_row(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i8],
    psqt_row: &[i32],
) {
    crate::simd::simd_add_i8_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_add(*weight);
    }
}

fn remove_psq_row(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i16],
    psqt_row: &[i32],
) {
    crate::simd::simd_sub_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_sub(*weight);
    }
}

fn remove_threat_row(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i8],
    psqt_row: &[i32],
) {
    crate::simd::simd_sub_i8_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_sub(*weight);
    }
}

#[derive(Clone)]
pub(crate) struct OtherAccumulator {
    accumulation: [[i16; HIDDEN_SIZE]; 2],
    psqt: [[i32; PSQT_BUCKETS]; 2],
    threat_indices: [Vec<usize>; 2],
}

impl OtherAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            accumulation: [[0; HIDDEN_SIZE]; 2],
            psqt: [[0; PSQT_BUCKETS]; 2],
            threat_indices: [Vec::new(), Vec::new()],
        }
    }

    pub(crate) fn refresh(&mut self, net: &OtherNetData, state: &BoardState) {
        for perspective in 0..2u32 {
            self.refresh_perspective(net, state, perspective);
        }
    }

    fn refresh_perspective(&mut self, net: &OtherNetData, state: &BoardState, perspective: u32) {
        let side = perspective as usize;
        for (value, bias) in self.accumulation[side].iter_mut().zip(net.ft_bias.iter()) {
            *value = *bias;
        }
        self.psqt[side].fill(0);

        let king_square = find_king(state, perspective);
        for piece in 0..12u32 {
            let mut pieces = state.bb[piece as usize];
            while pieces != 0 {
                let square = pieces.trailing_zeros();
                pieces &= pieces - 1;
                let index = halfka_index(perspective, square, piece, king_square);
                debug_assert!(index < PSQ_DIMS);
                add_psq_row(
                    &mut self.accumulation[side],
                    &mut self.psqt[side],
                    &net.psq_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                    &net.psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
                );
            }
        }

        let mut threat_indices = active_threat_indices(state, perspective);
        threat_indices.sort_unstable();
        for &index in &threat_indices {
            add_threat_row(
                &mut self.accumulation[side],
                &mut self.psqt[side],
                &net.threat_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                &net.threat_psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
            );
        }
        self.threat_indices[side] = threat_indices;
    }

    pub(crate) fn update_from_parent(
        &mut self,
        parent: &Self,
        net: &OtherNetData,
        before: &BoardState,
        after: &BoardState,
    ) {
        self.clone_from(parent);
        for perspective in 0..2u32 {
            if find_king(before, perspective) != find_king(after, perspective) {
                self.refresh_perspective(net, after, perspective);
                continue;
            }

            let side = perspective as usize;
            let king_square = find_king(after, perspective);
            for square in 0..64u32 {
                let before_piece = before.mailbox[square as usize];
                let after_piece = after.mailbox[square as usize];
                if before_piece == after_piece {
                    continue;
                }
                if before_piece != EMPTY_SQ {
                    let index =
                        halfka_index(perspective, square, u32::from(before_piece), king_square);
                    remove_psq_row(
                        &mut self.accumulation[side],
                        &mut self.psqt[side],
                        &net.psq_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                        &net.psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
                    );
                }
                if after_piece != EMPTY_SQ {
                    let index =
                        halfka_index(perspective, square, u32::from(after_piece), king_square);
                    add_psq_row(
                        &mut self.accumulation[side],
                        &mut self.psqt[side],
                        &net.psq_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                        &net.psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
                    );
                }
            }

            let mut added = active_threat_indices(after, perspective);
            added.sort_unstable();
            let removed = &parent.threat_indices[side];
            let (mut before_index, mut after_index) = (0, 0);
            while before_index < removed.len() || after_index < added.len() {
                match (removed.get(before_index), added.get(after_index)) {
                    (Some(&old), Some(&new)) if old == new => {
                        before_index += 1;
                        after_index += 1;
                    }
                    (Some(&old), Some(&new)) if old < new => {
                        remove_threat_row(
                            &mut self.accumulation[side],
                            &mut self.psqt[side],
                            &net.threat_weights[old * HIDDEN_SIZE..(old + 1) * HIDDEN_SIZE],
                            &net.threat_psqt[old * PSQT_BUCKETS..(old + 1) * PSQT_BUCKETS],
                        );
                        before_index += 1;
                    }
                    (_, Some(&new)) => {
                        add_threat_row(
                            &mut self.accumulation[side],
                            &mut self.psqt[side],
                            &net.threat_weights[new * HIDDEN_SIZE..(new + 1) * HIDDEN_SIZE],
                            &net.threat_psqt[new * PSQT_BUCKETS..(new + 1) * PSQT_BUCKETS],
                        );
                        after_index += 1;
                    }
                    (Some(&old), None) => {
                        remove_threat_row(
                            &mut self.accumulation[side],
                            &mut self.psqt[side],
                            &net.threat_weights[old * HIDDEN_SIZE..(old + 1) * HIDDEN_SIZE],
                            &net.threat_psqt[old * PSQT_BUCKETS..(old + 1) * PSQT_BUCKETS],
                        );
                        before_index += 1;
                    }
                    (None, None) => break,
                }
            }
            self.threat_indices[side] = added;
        }
    }
}

fn transformed_feature(accumulator: &[i16; HIDDEN_SIZE], index: usize) -> u8 {
    let a = i32::from(accumulator[index]).clamp(0, FT_MAX_VAL);
    let b = i32::from(accumulator[HIDDEN_SIZE / 2 + index]).clamp(0, FT_MAX_VAL);
    (a * b / 512) as u8
}

fn saturating_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn squared_clipped_relu(value: i32, weight_scale_bits: i32) -> u8 {
    let value = i64::from(saturating_i16(value));
    ((value * value) >> (2 * weight_scale_bits + 7)).min(127) as u8
}
fn clipped_relu(value: i32, weight_scale_bits: i32) -> u8 {
    (value >> weight_scale_bits).clamp(0, 127) as u8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn dot_product_avx2(input: &[u8], weights: &[i8]) -> i32 {
    debug_assert_eq!(input.len(), weights.len());
    debug_assert_eq!(input.len() % 32, 0);
    let ones = _mm256_set1_epi16(1);
    let mut sums = _mm256_setzero_si256();
    for offset in (0..input.len()).step_by(32) {
        let input_bytes =
            unsafe { _mm256_loadu_si256(input.as_ptr().add(offset).cast::<__m256i>()) };
        let weight_bytes =
            unsafe { _mm256_loadu_si256(weights.as_ptr().add(offset).cast::<__m256i>()) };

        let input_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input_bytes));
        let weight_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(weight_bytes));
        let products_low = _mm256_mullo_epi16(input_low, weight_low);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(products_low, ones));

        let input_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(input_bytes, 1));
        let weight_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_bytes, 1));
        let products_high = _mm256_mullo_epi16(input_high, weight_high);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(products_high, ones));
    }

    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums) };
    lanes.into_iter().fold(0i32, i32::wrapping_add)
}

fn dot_product(input: &[u8], weights: &[i8]) -> i32 {
    debug_assert_eq!(input.len(), weights.len());
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: OnceLock<bool> = OnceLock::new();
        if *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2")) {
            // SAFETY: runtime feature detection above guarantees AVX2 support,
            // and all supported layer widths are multiples of 32.
            return unsafe { dot_product_avx2(input, weights) };
        }
    }
    input
        .iter()
        .zip(weights.iter())
        .fold(0i32, |sum, (&input, &weight)| {
            sum.wrapping_add(i32::from(input) * i32::from(weight))
        })
}

fn forward_stack(stack: &OtherStack, transformed: &[u8; HIDDEN_SIZE]) -> i32 {
    let mut fc0 = [0i32; 32];
    for (output, value) in fc0.iter_mut().enumerate() {
        let weights = &stack.fc0_weights[output * HIDDEN_SIZE..(output + 1) * HIDDEN_SIZE];
        *value = stack.fc0_bias[output].wrapping_add(dot_product(transformed, weights));
    }

    let mut activation0 = [0u8; 64];
    for index in 0..32 {
        activation0[index] = squared_clipped_relu(fc0[index], WEIGHT_SCALE_BITS + 1);
        activation0[32 + index] = clipped_relu(fc0[index], WEIGHT_SCALE_BITS + 1);
    }

    let mut fc1 = [0i32; 32];
    for (output, value) in fc1.iter_mut().enumerate() {
        let weights = &stack.fc1_weights[output * 64..(output + 1) * 64];
        *value = stack.fc1_bias[output].wrapping_add(dot_product(&activation0, weights));
    }

    let mut activation1 = [0u8; 64];
    for index in 0..32 {
        activation1[index] = squared_clipped_relu(fc1[index], WEIGHT_SCALE_BITS);
        activation1[32 + index] = clipped_relu(fc1[index], WEIGHT_SCALE_BITS);
    }

    let mut activations = [0u8; 128];
    activations[..64].copy_from_slice(&activation0);
    activations[64..].copy_from_slice(&activation1);
    let output = stack
        .fc2_bias
        .wrapping_add(dot_product(&activations, &stack.fc2_weights));
    output.wrapping_add(fc0[30].wrapping_sub(fc0[31]))
}

fn evaluate_other_net_acc_components(
    net: &OtherNetData,
    accumulator: &OtherAccumulator,
    state: &BoardState,
) -> (i32, i32) {
    debug_assert_eq!(net.hidden_size, HIDDEN_SIZE);
    debug_assert_eq!(net.psq_dims, PSQ_DIMS);
    debug_assert_eq!(net.threat_dims, THREAT_DIMS);
    debug_assert_eq!(net.num_stacks, PSQT_BUCKETS);

    let side_to_move = usize::from(!state.w);
    let opponent = 1 - side_to_move;
    let piece_count: u32 = state.bb.iter().map(|bb| bb.count_ones()).sum();
    let bucket = (((piece_count as i32 - 1) / 4).clamp(0, (PSQT_BUCKETS - 1) as i32)) as usize;
    let perspectives = [side_to_move, opponent];
    let mut transformed = [0u8; HIDDEN_SIZE];
    for (output_side, &perspective) in perspectives.iter().enumerate() {
        for index in 0..HIDDEN_SIZE / 2 {
            transformed[output_side * HIDDEN_SIZE / 2 + index] =
                transformed_feature(&accumulator.accumulation[perspective], index);
        }
    }

    let psqt_value = (accumulator.psqt[side_to_move][bucket] - accumulator.psqt[opponent][bucket])
        .wrapping_div(2)
        / OUTPUT_SCALE;
    let forward = forward_stack(&net.stacks[bucket], &transformed);
    let multiplier = i64::from(600 * OUTPUT_SCALE);
    let denominator = i64::from(HIDDEN_ONE_VAL) * i64::from(1 << WEIGHT_SCALE_BITS) * 2;
    let positional = ((i64::from(forward) * multiplier) / denominator) as i32 / OUTPUT_SCALE;
    (psqt_value, positional)
}

pub(crate) fn evaluate_other_net_acc(
    net: &OtherNetData,
    accumulator: &OtherAccumulator,
    state: &BoardState,
) -> i32 {
    let (psqt, positional) = evaluate_other_net_acc_components(net, accumulator, state);
    psqt + positional
}

pub fn evaluate_other_net(net: &OtherNetData, state: &BoardState) -> i32 {
    let mut accumulator = OtherAccumulator::new();
    accumulator.refresh(net, state);
    evaluate_other_net_acc(net, &accumulator, state)
}

#[cfg(test)]
mod tests {
    use super::{
        active_threat_indices, evaluate_other_net, halfka_index, threat_lut, OtherAccumulator,
        THREAT_DIMS,
    };
    use crate::board::{move_ec, move_er, move_promotion, move_sc, move_sr};
    use crate::movegen::{apply_move, generate_moves};
    use crate::nnue::other_nets::load_other_net;
    use crate::Engine;
    use std::sync::OnceLock;

    fn real_net() -> Option<&'static crate::nnue::other_nets::OtherNetData> {
        static NET: OnceLock<Option<crate::nnue::other_nets::OtherNetData>> = OnceLock::new();
        NET.get_or_init(|| {
            std::fs::read("nn-0ee0657fb25e.nnue")
                .ok()
                .and_then(|data| load_other_net(&data).ok())
        })
        .as_ref()
    }

    fn score_for(fen: &str, net: &crate::nnue::other_nets::OtherNetData) -> i32 {
        let mut engine = Engine::new();
        engine.set_fen(fen);
        evaluate_other_net(net, &engine.st)
    }

    #[test]
    fn halfka_indices_match_stockfish_square_conventions() {
        // These indices exercise Ember's A8=0 board against Stockfish's A1=0
        // feature space. A public move fixture cannot observe this private row
        // selection contract.
        let white_king_e1 = 60;
        assert_eq!(halfka_index(0, 52, 0, white_king_e1), 21_836);
        assert_eq!(halfka_index(0, 12, 6, white_king_e1), 21_940);
        assert_eq!(halfka_index(0, white_king_e1, 5, white_king_e1), 22_468);

        let black_king_e8 = 4;
        assert_eq!(halfka_index(1, black_king_e8, 11, black_king_e8), 22_468);
    }

    #[test]
    fn full_threat_lut_has_stockfish_dimension() {
        let lut = threat_lut();
        assert_eq!(lut.index_lut2.len(), 12 * 64 * 64);
        assert_eq!(THREAT_DIMS, 60_720);
    }

    #[test]
    fn blocked_pawn_push_is_an_active_threat() {
        let mut engine = Engine::new();
        engine.set_fen("7k/8/8/8/8/4p3/4P3/2K5 w - - 0 1");
        let indices = active_threat_indices(&engine.st, 0);
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn real_net_matches_stockfish_oracle() {
        // This is an optional local oracle for the exact network named below.
        // It checks private quantized inference, which a TSV move fixture cannot
        // express. The fixed scores come from Stockfish d91c7f6e.
        let Some(net) = real_net() else {
            return;
        };
        assert_eq!(
            score_for(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                net,
            ),
            10
        );
        assert_eq!(
            score_for(
                "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2Q1RK1 w kq - 0 1",
                net,
            ),
            -482
        );
        assert_eq!(score_for("7k/8/8/8/8/8/8/K7 w - - 0 1", net), 25);

        let additional_cases = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                10,
            ),
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                -72,
            ),
            ("7k/8/8/8/8/4p3/4P3/2K5 w - - 0 1", 52),
            (
                "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8",
                -22,
            ),
            (
                "2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14",
                -74,
            ),
            ("8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40", -61),
            ("8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45", 2314),
            ("8/P4k2/8/8/8/8/8/6K1 w - - 0 1", 3199),
        ];
        for (fen, expected) in additional_cases {
            assert_eq!(score_for(fen, net), expected, "oracle mismatch for {fen}");
        }
    }

    #[test]
    fn incremental_accumulator_matches_full_refresh() {
        // This covers the private accumulator transition contract, including
        // captures, king refreshes, castling, en passant, and promotion.
        let Some(net) = real_net() else {
            return;
        };
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ];

        for fen in fens {
            let mut engine = Engine::new();
            engine.set_fen(fen);
            let before = engine.st;
            let mut parent = OtherAccumulator::new();
            parent.refresh(net, &before);
            for mv in generate_moves(&before, before.w, &before.cr, before.ep) {
                let mut after = before;
                apply_move(
                    &mut after,
                    move_sr(mv),
                    move_sc(mv),
                    move_er(mv),
                    move_ec(mv),
                    move_promotion(mv),
                );
                let mut incremental = OtherAccumulator::new();
                incremental.update_from_parent(&parent, net, &before, &after);
                let mut refreshed = OtherAccumulator::new();
                refreshed.refresh(net, &after);
                assert_eq!(
                    incremental.accumulation, refreshed.accumulation,
                    "incremental feature accumulator mismatch after move {mv} from {fen}"
                );
                assert_eq!(
                    incremental.psqt, refreshed.psqt,
                    "incremental PSQT mismatch after move {mv} from {fen}"
                );
                assert_eq!(
                    incremental.threat_indices, refreshed.threat_indices,
                    "incremental threat-index set mismatch after move {mv} from {fen}"
                );
            }
        }
    }
}
