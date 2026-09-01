#[cfg(target_arch = "x86_64")]
use super::backend::Avx512NnueBackend;
use super::backend::{
    NnueBackend, ScalarNnueBackend, Simd128NnueBackend, Simd512NnueBackend, SimdNnueBackend,
};
use super::ember_v2_net::{EmberV2Data, EmberV2Stack};
use crate::board::{BoardState, EMPTY_SQ, KING_ATTACKS, KNIGHT_ATTACKS};
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdInt, SimdUint};
use std::simd::Simd;
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

type I8x128 = Simd<i8, 8>;
type U8x128 = Simd<u8, 8>;
type I16x128 = Simd<i16, 8>;
type I32x128 = Simd<i32, 8>;
type I8x256 = Simd<i8, 16>;
#[allow(dead_code)]
type U8x256 = Simd<u8, 16>;
type I16x256 = Simd<i16, 16>;
#[allow(dead_code)]
type I32x256 = Simd<i32, 16>;
type I8x512 = Simd<i8, 32>;
type U8x512 = Simd<u8, 32>;
type I16x512 = Simd<i16, 32>;
type I32x512 = Simd<i32, 32>;

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

pub(crate) trait EmberV2Backend: NnueBackend {
    fn add_i8_row(accumulator: &mut [i16], row: &[i8]);
    fn sub_i8_row(accumulator: &mut [i16], row: &[i8]);
    fn transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]);
    fn dot(input: &[u8], weights: &[i8]) -> i32;
    fn fc0(transformed: &[u8; HIDDEN_SIZE], weights: &[i8], biases: &[i32], output: &mut [i32; 32]);
}

fn add_psq_row<B: EmberV2Backend>(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i16],
    psqt_row: &[i32],
) {
    B::add_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_add(*weight);
    }
}

fn add_threat_row<B: EmberV2Backend>(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i8],
    psqt_row: &[i32],
) {
    B::add_i8_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_add(*weight);
    }
}

fn remove_psq_row<B: EmberV2Backend>(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i16],
    psqt_row: &[i32],
) {
    B::sub_row(accumulator, row);
    for (value, weight) in psqt.iter_mut().zip(psqt_row.iter()) {
        *value = value.wrapping_sub(*weight);
    }
}

fn remove_threat_row<B: EmberV2Backend>(
    accumulator: &mut [i16; HIDDEN_SIZE],
    psqt: &mut [i32; PSQT_BUCKETS],
    row: &[i8],
    psqt_row: &[i32],
) {
    B::sub_i8_row(accumulator, row);
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

    pub(crate) fn refresh_with_backend<B: EmberV2Backend>(
        &mut self,
        net: &EmberV2Data,
        state: &BoardState,
    ) {
        for perspective in 0..2u32 {
            self.refresh_perspective::<B>(net, state, perspective);
        }
    }

    fn refresh_perspective<B: EmberV2Backend>(
        &mut self,
        net: &EmberV2Data,
        state: &BoardState,
        perspective: u32,
    ) {
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
                add_psq_row::<B>(
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
            add_threat_row::<B>(
                &mut self.accumulation[side],
                &mut self.psqt[side],
                &net.threat_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                &net.threat_psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
            );
        }
        self.threat_indices[side] = threat_indices;
    }

    pub(crate) fn update_from_parent_with_backend<B: EmberV2Backend>(
        &mut self,
        parent: &Self,
        net: &EmberV2Data,
        before: &BoardState,
        after: &BoardState,
    ) {
        self.clone_from(parent);
        for perspective in 0..2u32 {
            if find_king(before, perspective) != find_king(after, perspective) {
                self.refresh_perspective::<B>(net, after, perspective);
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
                    remove_psq_row::<B>(
                        &mut self.accumulation[side],
                        &mut self.psqt[side],
                        &net.psq_weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE],
                        &net.psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS],
                    );
                }
                if after_piece != EMPTY_SQ {
                    let index =
                        halfka_index(perspective, square, u32::from(after_piece), king_square);
                    add_psq_row::<B>(
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
                        remove_threat_row::<B>(
                            &mut self.accumulation[side],
                            &mut self.psqt[side],
                            &net.threat_weights[old * HIDDEN_SIZE..(old + 1) * HIDDEN_SIZE],
                            &net.threat_psqt[old * PSQT_BUCKETS..(old + 1) * PSQT_BUCKETS],
                        );
                        before_index += 1;
                    }
                    (_, Some(&new)) => {
                        add_threat_row::<B>(
                            &mut self.accumulation[side],
                            &mut self.psqt[side],
                            &net.threat_weights[new * HIDDEN_SIZE..(new + 1) * HIDDEN_SIZE],
                            &net.threat_psqt[new * PSQT_BUCKETS..(new + 1) * PSQT_BUCKETS],
                        );
                        after_index += 1;
                    }
                    (Some(&old), None) => {
                        remove_threat_row::<B>(
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

fn transformed_features_scalar(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
    debug_assert_eq!(output.len(), HIDDEN_SIZE / 2);
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = transformed_feature(accumulator, index);
    }
}

macro_rules! define_simd_v2_kernels {
    (
        $add_i8:ident,
        $sub_i8:ident,
        $transform:ident,
        $dot:ident,
        $fc0:ident,
        $i8_vector:ty,
        $u8_vector:ty,
        $i16_vector:ty,
        $i32_vector:ty,
        $lanes:expr
    ) => {
        #[inline(always)]
        fn $add_i8(accumulator: &mut [i16], row: &[i8]) {
            debug_assert_eq!(accumulator.len(), row.len());
            let (accumulator_chunks, accumulator_tail) = accumulator.as_chunks_mut::<$lanes>();
            let (row_chunks, row_tail) = row.as_chunks::<$lanes>();
            for (accumulator_chunk, row_chunk) in accumulator_chunks.iter_mut().zip(row_chunks) {
                let accumulator_values = <$i16_vector>::from_array(*accumulator_chunk);
                let row_values = <$i8_vector>::from_array(*row_chunk).cast::<i16>();
                *accumulator_chunk = (accumulator_values + row_values).to_array();
            }
            crate::simd::scalar_add_i8_row(accumulator_tail, row_tail);
        }

        #[inline(always)]
        fn $sub_i8(accumulator: &mut [i16], row: &[i8]) {
            debug_assert_eq!(accumulator.len(), row.len());
            let (accumulator_chunks, accumulator_tail) = accumulator.as_chunks_mut::<$lanes>();
            let (row_chunks, row_tail) = row.as_chunks::<$lanes>();
            for (accumulator_chunk, row_chunk) in accumulator_chunks.iter_mut().zip(row_chunks) {
                let accumulator_values = <$i16_vector>::from_array(*accumulator_chunk);
                let row_values = <$i8_vector>::from_array(*row_chunk).cast::<i16>();
                *accumulator_chunk = (accumulator_values - row_values).to_array();
            }
            crate::simd::scalar_sub_i8_row(accumulator_tail, row_tail);
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn $transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
            debug_assert_eq!(output.len(), HIDDEN_SIZE / 2);
            let (first, second) = accumulator.split_at(HIDDEN_SIZE / 2);
            let (first_chunks, first_tail) = first.as_chunks::<$lanes>();
            let (second_chunks, second_tail) = second.as_chunks::<$lanes>();
            let (output_chunks, output_tail) = output.as_chunks_mut::<$lanes>();
            let zero = <$i32_vector>::splat(0);
            let maximum = <$i32_vector>::splat(FT_MAX_VAL);
            for ((first_chunk, second_chunk), output_chunk) in
                first_chunks.iter().zip(second_chunks).zip(output_chunks)
            {
                let first_values = <$i16_vector>::from_array(*first_chunk)
                    .cast::<i32>()
                    .simd_clamp(zero, maximum);
                let second_values = <$i16_vector>::from_array(*second_chunk)
                    .cast::<i32>()
                    .simd_clamp(zero, maximum);
                *output_chunk = ((first_values * second_values) / <$i32_vector>::splat(512))
                    .cast::<u8>()
                    .to_array();
            }
            debug_assert!(first_tail.is_empty());
            debug_assert!(second_tail.is_empty());
            debug_assert!(output_tail.is_empty());
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn $dot(input: &[u8], weights: &[i8]) -> i32 {
            debug_assert_eq!(input.len(), weights.len());
            let (input_chunks, input_tail) = input.as_chunks::<$lanes>();
            let (weight_chunks, weight_tail) = weights.as_chunks::<$lanes>();
            let mut sum = 0i32;
            for (input_chunk, weight_chunk) in input_chunks.iter().zip(weight_chunks) {
                let input_values = <$u8_vector>::from_array(*input_chunk).cast::<i32>();
                let weight_values = <$i8_vector>::from_array(*weight_chunk).cast::<i32>();
                sum = sum.wrapping_add((input_values * weight_values).reduce_sum());
            }
            input_tail
                .iter()
                .zip(weight_tail)
                .fold(sum, |sum, (&input, &weight)| {
                    sum.wrapping_add(i32::from(input) * i32::from(weight))
                })
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn $fc0(
            transformed: &[u8; HIDDEN_SIZE],
            weights: &[i8],
            biases: &[i32],
            output: &mut [i32; 32],
        ) {
            debug_assert_eq!(weights.len(), 32 * HIDDEN_SIZE);
            debug_assert_eq!(biases.len(), 32);
            for (index, value) in output.iter_mut().enumerate() {
                let row = &weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE];
                *value = biases[index].wrapping_add($dot(transformed, row));
            }
        }
    };
}

define_simd_v2_kernels!(
    add_i8_row_simd128,
    sub_i8_row_simd128,
    transformed_features_simd128,
    dot_product_simd128,
    fc0_forward_simd128,
    I8x128,
    U8x128,
    I16x128,
    I32x128,
    8
);
define_simd_v2_kernels!(
    add_i8_row_simd256,
    sub_i8_row_simd256,
    transformed_features_simd256,
    dot_product_simd256,
    fc0_forward_simd256,
    I8x256,
    U8x256,
    I16x256,
    I32x256,
    16
);
define_simd_v2_kernels!(
    add_i8_row_simd512,
    sub_i8_row_simd512,
    transformed_features_simd512,
    dot_product_simd512,
    fc0_forward_simd512,
    I8x512,
    U8x512,
    I16x512,
    I32x512,
    32
);

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

fn dot_product_scalar(input: &[u8], weights: &[i8]) -> i32 {
    debug_assert_eq!(input.len(), weights.len());
    input
        .iter()
        .zip(weights.iter())
        .fold(0i32, |sum, (&input, &weight)| {
            sum.wrapping_add(i32::from(input) * i32::from(weight))
        })
}

fn fc0_forward_scalar(
    transformed: &[u8; HIDDEN_SIZE],
    weights: &[i8],
    biases: &[i32],
    output: &mut [i32; 32],
) {
    debug_assert_eq!(weights.len(), 32 * HIDDEN_SIZE);
    debug_assert_eq!(biases.len(), 32);
    for (index, value) in output.iter_mut().enumerate() {
        let row = &weights[index * HIDDEN_SIZE..(index + 1) * HIDDEN_SIZE];
        *value = biases[index].wrapping_add(dot_product_scalar(transformed, row));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn add_i8_row_x86_v3(accumulator: &mut [i16], row: &[i8]) {
    add_i8_row_simd256(accumulator, row);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn sub_i8_row_x86_v3(accumulator: &mut [i16], row: &[i8]) {
    sub_i8_row_simd256(accumulator, row);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn add_i8_row_x86_avx512(accumulator: &mut [i16], row: &[i8]) {
    add_i8_row_simd512(accumulator, row);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn sub_i8_row_x86_avx512(accumulator: &mut [i16], row: &[i8]) {
    sub_i8_row_simd512(accumulator, row);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn transformed_features_x86_avx512(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
    transformed_features_simd512(accumulator, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn dot_product_x86_avx512(input: &[u8], weights: &[i8]) -> i32 {
    dot_product_simd512(input, weights)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
unsafe fn fc0_forward_x86_avx512(
    transformed: &[u8; HIDDEN_SIZE],
    weights: &[i8],
    biases: &[i32],
    output: &mut [i32; 32],
) {
    fc0_forward_simd512(transformed, weights, biases, output);
}

impl EmberV2Backend for ScalarNnueBackend {
    #[inline(always)]
    fn add_i8_row(accumulator: &mut [i16], row: &[i8]) {
        crate::simd::scalar_add_i8_row(accumulator, row);
    }

    #[inline(always)]
    fn sub_i8_row(accumulator: &mut [i16], row: &[i8]) {
        crate::simd::scalar_sub_i8_row(accumulator, row);
    }

    #[inline(always)]
    fn transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
        transformed_features_scalar(accumulator, output);
    }

    #[inline(always)]
    fn dot(input: &[u8], weights: &[i8]) -> i32 {
        dot_product_scalar(input, weights)
    }

    #[inline(always)]
    fn fc0(
        transformed: &[u8; HIDDEN_SIZE],
        weights: &[i8],
        biases: &[i32],
        output: &mut [i32; 32],
    ) {
        fc0_forward_scalar(transformed, weights, biases, output);
    }
}

macro_rules! impl_portable_ember_v2_backend {
    ($backend:ty, $add_i8:ident, $sub_i8:ident, $transform:ident, $dot:ident, $fc0:ident) => {
        impl EmberV2Backend for $backend {
            #[inline(always)]
            fn add_i8_row(accumulator: &mut [i16], row: &[i8]) {
                $add_i8(accumulator, row);
            }

            #[inline(always)]
            fn sub_i8_row(accumulator: &mut [i16], row: &[i8]) {
                $sub_i8(accumulator, row);
            }

            #[inline(always)]
            fn transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
                $transform(accumulator, output);
            }

            #[inline(always)]
            fn dot(input: &[u8], weights: &[i8]) -> i32 {
                $dot(input, weights)
            }

            #[inline(always)]
            fn fc0(
                transformed: &[u8; HIDDEN_SIZE],
                weights: &[i8],
                biases: &[i32],
                output: &mut [i32; 32],
            ) {
                $fc0(transformed, weights, biases, output);
            }
        }
    };
}

impl_portable_ember_v2_backend!(
    Simd128NnueBackend,
    add_i8_row_simd128,
    sub_i8_row_simd128,
    transformed_features_simd128,
    dot_product_simd128,
    fc0_forward_simd128
);

impl EmberV2Backend for SimdNnueBackend {
    #[inline(always)]
    fn add_i8_row(accumulator: &mut [i16], row: &[i8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            add_i8_row_x86_v3(accumulator, row);
        }
        #[cfg(not(target_arch = "x86_64"))]
        add_i8_row_simd256(accumulator, row);
    }

    #[inline(always)]
    fn sub_i8_row(accumulator: &mut [i16], row: &[i8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sub_i8_row_x86_v3(accumulator, row);
        }
        #[cfg(not(target_arch = "x86_64"))]
        sub_i8_row_simd256(accumulator, row);
    }

    #[inline(always)]
    fn transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            transformed_features_avx2(accumulator, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        transformed_features_simd256(accumulator, output);
    }

    #[inline(always)]
    fn dot(input: &[u8], weights: &[i8]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            dot_product_avx2(input, weights)
        }
        #[cfg(not(target_arch = "x86_64"))]
        dot_product_simd256(input, weights)
    }

    #[inline(always)]
    fn fc0(
        transformed: &[u8; HIDDEN_SIZE],
        weights: &[i8],
        biases: &[i32],
        output: &mut [i32; 32],
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc0_forward_avx2(transformed, weights, biases, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        fc0_forward_simd256(transformed, weights, biases, output);
    }
}

impl_portable_ember_v2_backend!(
    Simd512NnueBackend,
    add_i8_row_simd512,
    sub_i8_row_simd512,
    transformed_features_simd512,
    dot_product_simd512,
    fc0_forward_simd512
);

#[cfg(target_arch = "x86_64")]
impl EmberV2Backend for Avx512NnueBackend {
    #[inline(always)]
    fn add_i8_row(accumulator: &mut [i16], row: &[i8]) {
        unsafe {
            add_i8_row_x86_avx512(accumulator, row);
        }
    }

    #[inline(always)]
    fn sub_i8_row(accumulator: &mut [i16], row: &[i8]) {
        unsafe {
            sub_i8_row_x86_avx512(accumulator, row);
        }
    }

    #[inline(always)]
    fn transform(accumulator: &[i16; HIDDEN_SIZE], output: &mut [u8]) {
        unsafe {
            transformed_features_x86_avx512(accumulator, output);
        }
    }

    #[inline(always)]
    fn dot(input: &[u8], weights: &[i8]) -> i32 {
        unsafe { dot_product_x86_avx512(input, weights) }
    }

    #[inline(always)]
    fn fc0(
        transformed: &[u8; HIDDEN_SIZE],
        weights: &[i8],
        biases: &[i32],
        output: &mut [i32; 32],
    ) {
        unsafe {
            fc0_forward_x86_avx512(transformed, weights, biases, output);
        }
    }
}

fn forward_stack<B: EmberV2Backend>(stack: &EmberV2Stack, transformed: &[u8; HIDDEN_SIZE]) -> i32 {
    let mut fc0 = [0i32; 32];
    B::fc0(transformed, &stack.fc0_weights, &stack.fc0_bias, &mut fc0);

    let mut activation0 = [0u8; 64];
    for index in 0..32 {
        activation0[index] = squared_clipped_relu(fc0[index], WEIGHT_SCALE_BITS + 1);
        activation0[32 + index] = clipped_relu(fc0[index], WEIGHT_SCALE_BITS + 1);
    }

    let mut fc1 = [0i32; 32];
    for (output, value) in fc1.iter_mut().enumerate() {
        let weights = &stack.fc1_weights[output * 64..(output + 1) * 64];
        *value = stack.fc1_bias[output].wrapping_add(B::dot(&activation0, weights));
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
        .wrapping_add(B::dot(&activations, &stack.fc2_weights));
    output.wrapping_add(fc0[30].wrapping_sub(fc0[31]))
}

fn evaluate_ember_v2_acc_components<B: EmberV2Backend>(
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
        B::transform(
            &accumulator.accumulation[perspective],
            &mut transformed[base..base + HIDDEN_SIZE / 2],
        );
    }

    let psqt_value = (accumulator.psqt[side_to_move][bucket] - accumulator.psqt[opponent][bucket])
        .wrapping_div(2)
        / OUTPUT_SCALE;
    let forward = forward_stack::<B>(&net.stacks[bucket], &transformed);
    let multiplier = i64::from(600 * OUTPUT_SCALE);
    let denominator = i64::from(HIDDEN_ONE_VAL) * i64::from(1 << WEIGHT_SCALE_BITS) * 2;
    let positional = ((i64::from(forward) * multiplier) / denominator) as i32 / OUTPUT_SCALE;
    (psqt_value, positional)
}

pub(crate) fn evaluate_ember_v2_acc_with_backend<B: EmberV2Backend>(
    net: &EmberV2Data,
    accumulator: &EmberV2Accumulator,
    state: &BoardState,
) -> i32 {
    let (psqt, positional) = evaluate_ember_v2_acc_components::<B>(net, accumulator, state);
    psqt + positional
}

pub fn evaluate_ember_v2(net: &EmberV2Data, state: &BoardState) -> i32 {
    evaluate_ember_v2_with_backend::<ScalarNnueBackend>(net, state)
}

pub(crate) fn evaluate_ember_v2_with_backend<B: EmberV2Backend>(
    net: &EmberV2Data,
    state: &BoardState,
) -> i32 {
    let mut accumulator = EmberV2Accumulator::new();
    accumulator.refresh_with_backend::<B>(net, state);
    evaluate_ember_v2_acc_with_backend::<B>(net, &accumulator, state)
}

#[cfg(test)]
mod tests {
    use super::{collect_active_threat_indices, halfka_index, threat_lut, THREAT_DIMS};
    use crate::Engine;

    fn synthetic_net(states: &[crate::board::BoardState]) -> super::EmberV2Data {
        let mut max_index = 0;
        for state in states {
            for perspective in 0..2u32 {
                let king_square = super::find_king(state, perspective);
                for piece in 0..12u32 {
                    let mut pieces = state.bb[piece as usize];
                    while pieces != 0 {
                        let square = pieces.trailing_zeros();
                        pieces &= pieces - 1;
                        max_index = max_index.max(super::halfka_index(
                            perspective,
                            square,
                            piece,
                            king_square,
                        ));
                    }
                }
                let mut threats = Vec::new();
                super::collect_active_threat_indices(state, perspective, &mut threats);
                assert!(
                    threats.is_empty(),
                    "synthetic position must have no threats"
                );
            }
        }

        let mut psq_weights = vec![0i16; (max_index + 1) * super::HIDDEN_SIZE];
        for (index, value) in psq_weights.iter_mut().enumerate() {
            *value = ((index * 17 + 5) % 31) as i16 - 15;
        }
        let mut psqt = vec![0i32; (max_index + 1) * super::PSQT_BUCKETS];
        for (index, value) in psqt.iter_mut().enumerate() {
            *value = (index as i32 * 97).wrapping_sub(4_001);
        }
        let mut ft_bias = vec![0i16; super::HIDDEN_SIZE];
        for (index, value) in ft_bias.iter_mut().enumerate() {
            *value = ((index * 23 + 3) % 257) as i16 - 1;
        }

        let stacks = (0..super::PSQT_BUCKETS)
            .map(|bucket| {
                let mut fc0_weights = vec![0i8; 32 * super::HIDDEN_SIZE];
                for (index, weight) in fc0_weights.iter_mut().enumerate() {
                    *weight = ((index * 29 + bucket * 11 + 7) % 256) as u8 as i8;
                }
                let mut fc1_weights = vec![0i8; 32 * 64];
                for (index, weight) in fc1_weights.iter_mut().enumerate() {
                    *weight = ((index * 31 + bucket * 13 + 9) % 256) as u8 as i8;
                }
                let mut fc2_weights = vec![0i8; 128];
                for (index, weight) in fc2_weights.iter_mut().enumerate() {
                    *weight = ((index * 37 + bucket * 17 + 1) % 256) as u8 as i8;
                }
                super::EmberV2Stack {
                    fc0_bias: (0..32)
                        .map(|index| index * 10_007 - bucket as i32 * 503)
                        .collect(),
                    fc0_weights,
                    fc1_bias: (0..32)
                        .map(|index| index * 2_003 + bucket as i32 * 101)
                        .collect(),
                    fc1_weights,
                    fc2_bias: bucket as i32 * 997 - 2_011,
                    fc2_weights,
                }
            })
            .collect();

        super::EmberV2Data {
            hidden_size: super::HIDDEN_SIZE,
            psq_dims: super::PSQ_DIMS,
            threat_dims: super::THREAT_DIMS,
            num_stacks: super::PSQT_BUCKETS,
            ft_bias,
            threat_weights: Vec::new(),
            threat_psqt: Vec::new(),
            psq_weights,
            psqt,
            stacks,
            overview: "synthetic backend parity net".into(),
        }
    }

    fn assert_accumulator_backend_matches_scalar<B: super::EmberV2Backend>(
        net: &super::EmberV2Data,
        before: &crate::board::BoardState,
        after: &crate::board::BoardState,
    ) {
        let mut scalar_before = super::EmberV2Accumulator::new();
        scalar_before.refresh_with_backend::<crate::nnue::ScalarNnueBackend>(net, before);
        let expected_before = super::evaluate_ember_v2_acc_with_backend::<
            crate::nnue::ScalarNnueBackend,
        >(net, &scalar_before, before);

        let mut backend_before = super::EmberV2Accumulator::new();
        backend_before.refresh_with_backend::<B>(net, before);
        assert_eq!(backend_before.accumulation, scalar_before.accumulation);
        assert_eq!(backend_before.psqt, scalar_before.psqt);
        assert_eq!(
            super::evaluate_ember_v2_acc_with_backend::<B>(net, &backend_before, before),
            expected_before
        );

        let mut incremental = super::EmberV2Accumulator::new();
        incremental.update_from_parent_with_backend::<B>(&backend_before, net, before, after);
        let mut refreshed = super::EmberV2Accumulator::new();
        refreshed.refresh_with_backend::<B>(net, after);
        assert_eq!(incremental.accumulation, refreshed.accumulation);
        assert_eq!(incremental.psqt, refreshed.psqt);
        assert_eq!(incremental.threat_indices, refreshed.threat_indices);
    }

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

    fn assert_backend_kernels_match_scalar<B: super::EmberV2Backend>() {
        let mut accumulator = [0i16; super::HIDDEN_SIZE];
        for (index, value) in accumulator.iter_mut().enumerate() {
            *value = ((index as i32 * 7919) % 997 - 498) as i16;
        }
        accumulator[..6].copy_from_slice(&[0, 255, 256, -1, i16::MIN, i16::MAX]);

        let mut expected_features = [0u8; super::HIDDEN_SIZE / 2];
        super::transformed_features_scalar(&accumulator, &mut expected_features);
        let mut actual_features = [0u8; super::HIDDEN_SIZE / 2];
        B::transform(&accumulator, &mut actual_features);
        assert_eq!(actual_features, expected_features);

        let mut input = [0u8; super::HIDDEN_SIZE];
        let mut weights = [0i8; super::HIDDEN_SIZE];
        for (index, value) in input.iter_mut().enumerate() {
            *value = ((index * 37 + 11) % 128) as u8;
        }
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = ((index * 53 + 19) % 256) as u8 as i8;
        }
        input[..4].fill(127);
        weights[..4].copy_from_slice(&[127, 127, -128, -128]);
        for width in [64, 128, super::HIDDEN_SIZE] {
            assert_eq!(
                B::dot(&input[..width], &weights[..width]),
                super::dot_product_scalar(&input[..width], &weights[..width]),
                "dot-product mismatch at width {width}"
            );
        }

        let mut fc0_weights = vec![0i8; 32 * super::HIDDEN_SIZE];
        for (index, weight) in fc0_weights.iter_mut().enumerate() {
            *weight = ((index * 29 + index / super::HIDDEN_SIZE * 17 + 7) % 256) as u8 as i8;
        }
        let mut biases = [0i32; 32];
        for (index, bias) in biases.iter_mut().enumerate() {
            *bias = match index {
                0 => i32::MAX,
                1 => i32::MIN,
                _ => (index as i32 * 104_729).wrapping_sub(700_001),
            };
        }
        let mut expected_fc0 = [0i32; 32];
        super::fc0_forward_scalar(&input, &fc0_weights, &biases, &mut expected_fc0);
        let mut actual_fc0 = [0i32; 32];
        B::fc0(&input, &fc0_weights, &biases, &mut actual_fc0);
        assert_eq!(actual_fc0, expected_fc0);

        let mut row = [0i8; super::HIDDEN_SIZE];
        for (index, value) in row.iter_mut().enumerate() {
            *value = ((index * 43 + 23) % 256) as u8 as i8;
        }
        let mut row_accumulator = [0i16; super::HIDDEN_SIZE];
        for (index, value) in row_accumulator.iter_mut().enumerate() {
            *value = (index as i16 % 401) - 200;
        }
        let mut expected_accumulator = row_accumulator;
        crate::simd::scalar_add_i8_row(&mut expected_accumulator, &row);
        let mut actual_accumulator = row_accumulator;
        B::add_i8_row(&mut actual_accumulator, &row);
        assert_eq!(actual_accumulator, expected_accumulator);
        B::sub_i8_row(&mut actual_accumulator, &row);
        assert_eq!(actual_accumulator, row_accumulator);
    }

    #[test]
    fn portable_ember_v2_backends_match_scalar_kernels() {
        assert_backend_kernels_match_scalar::<crate::nnue::Simd128NnueBackend>();
        assert_backend_kernels_match_scalar::<crate::nnue::Simd512NnueBackend>();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_ember_v2_backends_match_scalar_kernels() {
        if crate::backend::x86_v3_available() {
            assert_backend_kernels_match_scalar::<crate::nnue::SimdNnueBackend>();
        }
        if crate::backend::x86_avx512_available() {
            assert_backend_kernels_match_scalar::<crate::nnue::Avx512NnueBackend>();
        }
    }

    #[test]
    fn ember_v2_backends_match_full_refresh_and_incremental_scores() {
        let mut engine = Engine::new();
        engine.set_fen("K7/8/8/8/8/8/8/k7 w - - 0 1");
        let before = engine.st;
        assert!(engine.make_move_uci(0, 0, 0, 1, 0));
        let after = engine.st;
        let net = synthetic_net(&[before, after]);

        assert_accumulator_backend_matches_scalar::<crate::nnue::ScalarNnueBackend>(
            &net, &before, &after,
        );
        assert_accumulator_backend_matches_scalar::<crate::nnue::Simd128NnueBackend>(
            &net, &before, &after,
        );
        assert_accumulator_backend_matches_scalar::<crate::nnue::Simd512NnueBackend>(
            &net, &before, &after,
        );
        #[cfg(target_arch = "x86_64")]
        {
            if crate::backend::x86_v3_available() {
                assert_accumulator_backend_matches_scalar::<crate::nnue::SimdNnueBackend>(
                    &net, &before, &after,
                );
            }
            if crate::backend::x86_avx512_available() {
                assert_accumulator_backend_matches_scalar::<crate::nnue::Avx512NnueBackend>(
                    &net, &before, &after,
                );
            }
        }
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fused_avx2_dot_products_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        // This directly checks the private quantized arithmetic introduced by
        // the fused AVX2 path; a public move fixture cannot isolate it.
        fn scalar_dot(input: &[u8], weights: &[i8]) -> i32 {
            input
                .iter()
                .zip(weights)
                .fold(0i32, |sum, (&input, &weight)| {
                    sum.wrapping_add(i32::from(input) * i32::from(weight))
                })
        }

        let mut input = [0u8; super::HIDDEN_SIZE];
        for (index, value) in input.iter_mut().enumerate() {
            *value = ((index * 37 + 11) % 128) as u8;
        }
        input[..4].fill(127);

        let mut dot_weights = [0i8; super::HIDDEN_SIZE];
        for (index, weight) in dot_weights.iter_mut().enumerate() {
            *weight = ((index * 53 + 19) % 256) as u8 as i8;
        }
        dot_weights[0] = 127;
        dot_weights[1] = 127;
        dot_weights[2] = -128;
        dot_weights[3] = -128;

        for width in [64, 128, super::HIDDEN_SIZE] {
            let expected = scalar_dot(&input[..width], &dot_weights[..width]);
            // SAFETY: runtime detection above guarantees AVX2 support, and all
            // tested widths are multiples of 32.
            let actual = unsafe { super::dot_product_avx2(&input[..width], &dot_weights[..width]) };
            assert_eq!(actual, expected, "dot-product mismatch at width {width}");
        }

        let mut fc0_weights = vec![0i8; 32 * super::HIDDEN_SIZE];
        for (index, weight) in fc0_weights.iter_mut().enumerate() {
            *weight = ((index * 29 + index / super::HIDDEN_SIZE * 17 + 7) % 256) as u8 as i8;
        }
        fc0_weights[..4].copy_from_slice(&[127, 127, -128, -128]);

        let mut biases = [0i32; 32];
        for (index, bias) in biases.iter_mut().enumerate() {
            *bias = match index {
                0 => i32::MAX,
                1 => i32::MIN,
                _ => (index as i32 * 104_729).wrapping_sub(700_001),
            };
        }

        let mut expected = [0i32; 32];
        for (output, value) in expected.iter_mut().enumerate() {
            let weights =
                &fc0_weights[output * super::HIDDEN_SIZE..(output + 1) * super::HIDDEN_SIZE];
            *value = biases[output].wrapping_add(scalar_dot(&input, weights));
        }

        let mut actual = [0i32; 32];
        // SAFETY: runtime detection above guarantees AVX2 support and the
        // buffers have the exact fixed sizes required by the fused kernel.
        unsafe { super::fc0_forward_avx2(&input, &fc0_weights, &biases, &mut actual) };
        assert_eq!(actual, expected);
    }
}
