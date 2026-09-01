use super::ember_v2_net::{EmberV2Data, EmberV2Stack};
use crate::board::{BoardState, EMPTY_SQ, KING_ATTACKS, KNIGHT_ATTACKS};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_max_epi16, _mm256_min_epi16, _mm256_mullo_epi16, _mm256_packus_epi16,
    _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_srli_epi16,
    _mm256_storeu_si256,
};

const PSQ_DIMS: usize = 22_528;
const THREAT_DIMS: usize = 60_720;
const HIDDEN_SIZE: usize = 1024;
const PSQT_BUCKETS: usize = 8;

const OUTPUT_SCALE: i32 = 16;
const WEIGHT_SCALE_BITS: i32 = 6;
const FT_MAX_VAL: i32 = 255;
const HIDDEN_ONE_VAL: i32 = 128;

const NUM_VALID_TARGETS: [u32; 12] = [6, 10, 8, 8, 10, 0, 6, 10, 8, 8, 10, 0];

const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

#[inline(always)]
fn to_v2_square(square: u32) -> u32 {
    square ^ 56
}

#[inline(always)]
fn halfka_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 {
        7
    } else {
        0
    }
}

#[inline(always)]
fn threat_orientation(king_square: u32) -> u32 {
    if king_square & 4 == 0 {
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
    let square = to_v2_square(square);
    let king_square = to_v2_square(king_square);
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

fn pseudo_attacks(piece: u32, from: u32) -> u64 {
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
        let ember_from = to_v2_square(from);
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
                    pseudo_attacks(piece, from)
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
                    pseudo_attacks(piece, from).count_ones() as usize
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
        let from = to_v2_square(from);
        let to = to_v2_square(to);
        let king_square = to_v2_square(king_square);
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

fn collect_active_threat_indices(state: &BoardState, perspective: u32, out: &mut Vec<usize>) {
    let lut = threat_lut();
    let king_square = find_king(state, perspective);
    let occupancy = state.bb.iter().copied().fold(0u64, |all, bb| all | bb);
    let all_pawns = state.bb[0] | state.bb[6];
    out.clear();

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
                    out.push(index);
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
                        out.push(index);
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
                        out.push(index);
                    }
                }
            }
        }
    }
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
pub(crate) struct EmberV2Accumulator {
    accumulation: [[i16; HIDDEN_SIZE]; 2],
    psqt: [[i32; PSQT_BUCKETS]; 2],
    threat_indices: [Vec<usize>; 2],
}

impl EmberV2Accumulator {
    pub(crate) fn new() -> Self {
        Self {
            accumulation: [[0; HIDDEN_SIZE]; 2],
            psqt: [[0; PSQT_BUCKETS]; 2],
            threat_indices: [Vec::new(), Vec::new()],
        }
    }

    pub(crate) fn refresh(&mut self, net: &EmberV2Data, state: &BoardState) {
        for perspective in 0..2u32 {
            self.refresh_perspective(net, state, perspective);
        }
    }

    fn refresh_perspective(&mut self, net: &EmberV2Data, state: &BoardState, perspective: u32) {
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

        let mut threat_indices = std::mem::take(&mut self.threat_indices[side]);
        collect_active_threat_indices(state, perspective, &mut threat_indices);
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
        net: &EmberV2Data,
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

            let mut added = std::mem::take(&mut self.threat_indices[side]);
            collect_active_threat_indices(after, perspective, &mut added);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn transformed_features_avx2(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
    debug_assert_eq!(output.len(), HIDDEN_SIZE / 2);
    let zero = _mm256_setzero_si256();
    let max255 = _mm256_set1_epi16(FT_MAX_VAL as i16);

    for offset in (0..HIDDEN_SIZE / 2).step_by(32) {
        let a_lo =
            unsafe { _mm256_loadu_si256(accumulator.as_ptr().add(offset).cast::<__m256i>()) };
        let a_hi =
            unsafe { _mm256_loadu_si256(accumulator.as_ptr().add(offset + 16).cast::<__m256i>()) };
        let b_lo = unsafe {
            _mm256_loadu_si256(
                accumulator
                    .as_ptr()
                    .add(HIDDEN_SIZE / 2 + offset)
                    .cast::<__m256i>(),
            )
        };
        let b_hi = unsafe {
            _mm256_loadu_si256(
                accumulator
                    .as_ptr()
                    .add(HIDDEN_SIZE / 2 + offset + 16)
                    .cast::<__m256i>(),
            )
        };

        let a_lo = _mm256_min_epi16(_mm256_max_epi16(a_lo, zero), max255);
        let a_hi = _mm256_min_epi16(_mm256_max_epi16(a_hi, zero), max255);
        let b_lo = _mm256_min_epi16(_mm256_max_epi16(b_lo, zero), max255);
        let b_hi = _mm256_min_epi16(_mm256_max_epi16(b_hi, zero), max255);

        let shifted_lo = _mm256_srli_epi16(_mm256_mullo_epi16(a_lo, b_lo), 9);
        let shifted_hi = _mm256_srli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 9);

        let x = _mm256_permute2x128_si256(shifted_lo, shifted_hi, 0x20);
        let y = _mm256_permute2x128_si256(shifted_lo, shifted_hi, 0x31);
        let packed = _mm256_packus_epi16(x, y);
        unsafe {
            _mm256_storeu_si256(output.as_mut_ptr().add(offset).cast::<__m256i>(), packed);
        }
    }
}

fn transformed_features(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
    debug_assert_eq!(output.len(), HIDDEN_SIZE / 2);
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2_FEATURES: OnceLock<bool> = OnceLock::new();
        if *AVX2_FEATURES.get_or_init(|| std::is_x86_feature_detected!("avx2")) {
            // SAFETY: runtime feature detection above guarantees AVX2 support,
            // and output is exactly HIDDEN_SIZE/2 bytes, a multiple of 32.
            return unsafe { transformed_features_avx2(accumulator, output) };
        }
    }
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = transformed_feature(accumulator, index);
    }
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
    // SAFETY: inputs are clamped to [0,127] and weights fit in i8, so every
    // u8*i8 pair product and adjacent-pair sum fits in i16 (max 32258 < 32767);
    // `_mm256_maddubs_epi16` exactly reproduces the mullo+madd reference sum.
    let ones = _mm256_set1_epi16(1);
    let mut sums = _mm256_setzero_si256();
    for offset in (0..input.len()).step_by(32) {
        let input_bytes = _mm256_loadu_si256(input.as_ptr().add(offset).cast::<__m256i>());
        let weight_bytes = _mm256_loadu_si256(weights.as_ptr().add(offset).cast::<__m256i>());
        let pair_sums = _mm256_maddubs_epi16(input_bytes, weight_bytes);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(pair_sums, ones));
    }

    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums) };
    lanes.into_iter().fold(0i32, i32::wrapping_add)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn fc0_forward_avx2(
    transformed: &[u8; HIDDEN_SIZE],
    weights: &[i8],
    biases: &[i32],
    out: &mut [i32; 32],
) {
    debug_assert_eq!(weights.len(), 32 * HIDDEN_SIZE);
    debug_assert_eq!(biases.len(), 32);
    let ones = _mm256_set1_epi16(1);
    for group in 0..4 {
        let mut sums = [_mm256_setzero_si256(); 8];
        for offset in (0..HIDDEN_SIZE).step_by(32) {
            let input_bytes =
                unsafe { _mm256_loadu_si256(transformed.as_ptr().add(offset).cast::<__m256i>()) };
            for (lane, group_acc) in sums.iter_mut().enumerate() {
                let w_ptr = weights
                    .as_ptr()
                    .add((group * 8 + lane) * HIDDEN_SIZE + offset)
                    .cast::<__m256i>();
                let weight_bytes = unsafe { _mm256_loadu_si256(w_ptr) };
                let pair_sums = _mm256_maddubs_epi16(input_bytes, weight_bytes);
                *group_acc = _mm256_add_epi32(*group_acc, _mm256_madd_epi16(pair_sums, ones));
            }
        }
        for lane in 0..8 {
            let mut lanes = [0i32; 8];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums[lane]) };
            out[group * 8 + lane] = biases[group * 8 + lane]
                .wrapping_add(lanes.into_iter().fold(0i32, i32::wrapping_add));
        }
    }
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

fn forward_stack(stack: &EmberV2Stack, transformed: &[u8; HIDDEN_SIZE]) -> i32 {
    let mut fc0 = [0i32; 32];
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2_FC0: OnceLock<bool> = OnceLock::new();
        if *AVX2_FC0.get_or_init(|| std::is_x86_feature_detected!("avx2")) {
            // SAFETY: runtime AVX2 detection above; weights are 32 rows of exactly
            // HIDDEN_SIZE bytes (a multiple of 32), biases have 32 entries.
            unsafe {
                fc0_forward_avx2(transformed, &stack.fc0_weights, &stack.fc0_bias, &mut fc0);
            }
        } else {
            for (output, value) in fc0.iter_mut().enumerate() {
                let weights = &stack.fc0_weights[output * HIDDEN_SIZE..(output + 1) * HIDDEN_SIZE];
                *value = stack.fc0_bias[output].wrapping_add(dot_product(transformed, weights));
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for (output, value) in fc0.iter_mut().enumerate() {
            let weights = &stack.fc0_weights[output * HIDDEN_SIZE..(output + 1) * HIDDEN_SIZE];
            *value = stack.fc0_bias[output].wrapping_add(dot_product(transformed, weights));
        }
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

fn evaluate_ember_v2_acc_components(
    net: &EmberV2Data,
    accumulator: &EmberV2Accumulator,
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
        let base = output_side * HIDDEN_SIZE / 2;
        transformed_features(
            &accumulator.accumulation[perspective],
            &mut transformed[base..base + HIDDEN_SIZE / 2],
        );
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

pub(crate) fn evaluate_ember_v2_acc(
    net: &EmberV2Data,
    accumulator: &EmberV2Accumulator,
    state: &BoardState,
) -> i32 {
    let (psqt, positional) = evaluate_ember_v2_acc_components(net, accumulator, state);
    psqt + positional
}

pub fn evaluate_ember_v2(net: &EmberV2Data, state: &BoardState) -> i32 {
    let mut accumulator = EmberV2Accumulator::new();
    accumulator.refresh(net, state);
    evaluate_ember_v2_acc(net, &accumulator, state)
}

#[cfg(test)]
mod tests {
    use super::{collect_active_threat_indices, halfka_index, threat_lut, THREAT_DIMS};
    use crate::Engine;

    #[test]
    fn halfka_indices_match_v2_square_conventions() {
        // These indices exercise Ember's A8=0 board against the v2 network's
        // A1=0 feature space. A public move fixture cannot observe this private row
        // selection contract.
        let white_king_e1 = 60;
        assert_eq!(halfka_index(0, 52, 0, white_king_e1), 21_836);
        assert_eq!(halfka_index(0, 12, 6, white_king_e1), 21_940);
        assert_eq!(halfka_index(0, white_king_e1, 5, white_king_e1), 22_468);

        let black_king_e8 = 4;
        assert_eq!(halfka_index(1, black_king_e8, 11, black_king_e8), 22_468);
    }

    #[test]
    fn full_threat_lut_has_v2_dimension() {
        let lut = threat_lut();
        assert_eq!(lut.index_lut2.len(), 12 * 64 * 64);
        assert_eq!(THREAT_DIMS, 60_720);
    }

    #[test]
    fn blocked_pawn_push_is_an_active_threat() {
        let mut engine = Engine::new();
        engine.set_fen("7k/8/8/8/8/4p3/4P3/2K5 w - - 0 1");
        let mut indices = Vec::new();
        collect_active_threat_indices(&engine.st, 0, &mut indices);
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn transformed_features_simd_matches_scalar() {
        let mut acc = [0i16; super::HIDDEN_SIZE];
        for (i, v) in acc.iter_mut().enumerate() {
            let x = i as i32;
            *v = ((x * 7919) % 97 - 48) as i16;
        }
        acc[0] = 0;
        acc[1] = 255;
        acc[2] = 256;
        acc[3] = -1;
        acc[4] = i16::MIN;
        acc[5] = i16::MAX;
        acc[super::HIDDEN_SIZE / 2] = 127;
        acc[super::HIDDEN_SIZE / 2 + 1] = -32768;

        let mut expected = [0u8; super::HIDDEN_SIZE / 2];
        for (index, slot) in expected.iter_mut().enumerate() {
            *slot = super::transformed_feature(&acc, index);
        }

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated on runtime AVX2 detection above.
            let mut actual = [0u8; super::HIDDEN_SIZE / 2];
            unsafe { super::transformed_features_avx2(&acc, &mut actual) };
            assert_eq!(actual, expected);
        }
    }
}
