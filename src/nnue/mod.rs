use crate::backend::{nnue_backend_available, NnueBackendKind};
use crate::board::BoardState;
use crate::types::WHITE;
use std::mem::MaybeUninit;
use std::slice;

pub const PSQ_INPUTS_PER_BUCKET: usize = 768;
pub const NNUE_OUTPUT_BUCKETS: usize = 8;
pub const MAX_HIDDEN_SIZE: usize = 2048;
static ZERO_FEATURE_ROW: [i16; MAX_HIDDEN_SIZE] = [0; MAX_HIDDEN_SIZE];

pub(crate) const QA: i32 = 255;
pub(crate) const QB: i32 = 64;
pub(crate) const QAB: i32 = QA * QB;
pub(crate) const EVAL_SCALE: i32 = 400;
const FT_SHIFT: i32 = 9;

#[inline(always)]
fn uninit_array<T, const N: usize>() -> [MaybeUninit<T>; N] {
    [const { MaybeUninit::uninit() }; N]
}

#[inline(always)]
unsafe fn assume_init_slice<T>(values: &[MaybeUninit<T>]) -> &[T] {
    // Safety: the caller guarantees every element in `values` has previously
    // been initialized, and this produces a shared slice over that prefix only.
    unsafe { slice::from_raw_parts(values.as_ptr() as *const T, values.len()) }
}

mod backend;
mod classic;
mod features;
mod loader;
mod other_infer;
mod other_nets;
#[cfg(target_arch = "x86_64")]
pub(crate) use self::backend::Avx512NnueBackend;
pub(crate) use self::backend::{
    NnueBackend, ScalarNnueBackend, Simd128NnueBackend, Simd512NnueBackend, SimdNnueBackend,
};
#[cfg(test)]
pub(crate) use self::classic::synthetic_test_net_bytes;
pub(crate) use self::classic::ClassicHalfKpNet;
pub use self::features::{
    compute_king_buckets, threat_feature_count, KbLayout, NNUEThreatAccumulator,
};
use self::features::{halfka_idx, output_bucket};
pub(crate) use self::other_infer::{evaluate_other_net, evaluate_other_net_acc, OtherAccumulator};
pub(crate) use self::other_nets::{OtherNetData, OtherNetInfo};

const COMPACT_ZERO_ROW: u16 = u16::MAX;

pub fn convert(sq: u8) -> u8 {
    sq ^ 56
}

pub struct NNUENet {
    pub hidden_size: usize,
    pub input_weights: Vec<i16>,
    pub input_row_map: Vec<u16>,
    pub input_biases: Vec<i16>,
    pub threat_weights: Vec<i8>,
    pub num_threat_features: usize,
    pub output_weights: Vec<i16>,
    pub output_bias: [i32; NNUE_OUTPUT_BUCKETS],
    pub use_screlu: bool,
    pub screlu_i32_output_safe: bool,
    pub screlu_i32_accumulator_safe: bool,
    pub use_pairwise: bool,
    pub l1_size: usize,
    pub l1_per_bucket: usize,
    pub bucketed_hidden: bool,
    pub l1_scale: i32,
    pub l2_size: usize,
    pub l2_per_bucket: usize,
    pub l1_weights: Vec<i16>,
    pub l1_biases: Vec<i16>,
    pub l2_weights_f: Vec<f32>,
    pub l2_biases_f: Vec<f32>,
    pub out_weights_f: Vec<f32>,
    pub out_bias_f: Vec<f32>,
    pub dual_l1: bool,
    pub crelu_hidden: bool,
    pub num_king_buckets: usize,
    pub kb_layout: KbLayout,
    pub king_bucket: [usize; 64],
    pub king_mirror: [bool; 64],
}

impl NNUENet {
    pub fn halfka(&self, persp: u8, ks: u8, pc: u8, pt: u8, ps: u8) -> usize {
        halfka_idx(&self.king_bucket, &self.king_mirror, persp, ks, pc, pt, ps)
    }

    #[inline(always)]
    fn input_row_fast(&self, idx: usize) -> Option<&[i16]> {
        debug_assert!(idx < self.input_row_map.len());
        // Safety: callers pass feature indices produced for this network's
        // HalfKA layout, so the index is within the row map.
        let physical_row = unsafe { *self.input_row_map.get_unchecked(idx) };
        if physical_row == COMPACT_ZERO_ROW {
            return None;
        }

        let start = physical_row as usize * self.hidden_size;
        debug_assert!(start + self.hidden_size <= self.input_weights.len());
        // Safety: non-zero physical rows are produced by
        // `compact_input_weights`, which appends full `hidden_size` rows to
        // `input_weights` and records only those row numbers.
        let row = unsafe {
            std::slice::from_raw_parts(self.input_weights.as_ptr().add(start), self.hidden_size)
        };
        Some(row)
    }

    pub fn has_threat_features(&self) -> bool {
        self.num_threat_features != 0 && !self.threat_weights.is_empty()
    }

    pub fn forward(&self, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        self.forward_with_backend::<ScalarNnueBackend>(acc, stm, piece_count)
    }

    pub fn forward_with_kind(
        &self,
        backend: NnueBackendKind,
        acc: &NNUEAccumulator,
        stm: u8,
        piece_count: u32,
    ) -> i32 {
        debug_assert!(nnue_backend_available(backend));
        match backend {
            NnueBackendKind::Scalar => {
                self.forward_with_backend::<ScalarNnueBackend>(acc, stm, piece_count)
            }
            NnueBackendKind::Simd128 => {
                self.forward_with_backend::<Simd128NnueBackend>(acc, stm, piece_count)
            }
            NnueBackendKind::Simd256 => {
                self.forward_with_backend::<SimdNnueBackend>(acc, stm, piece_count)
            }
            NnueBackendKind::Simd512 => {
                self.forward_with_backend::<Simd512NnueBackend>(acc, stm, piece_count)
            }
            NnueBackendKind::X86Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    self.forward_with_backend::<Avx512NnueBackend>(acc, stm, piece_count)
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.forward_with_backend::<ScalarNnueBackend>(acc, stm, piece_count)
                }
            }
        }
    }

    pub(crate) fn forward_with_threats<B: NnueBackend>(
        &self,
        acc: &NNUEAccumulator,
        threats: &NNUEThreatAccumulator,
        stm: u8,
        piece_count: u32,
    ) -> i32 {
        let bucket = output_bucket(piece_count);
        let (stm_acc, ntm_acc, stm_threat, ntm_threat) = if stm == WHITE {
            (acc.white(), acc.black(), threats.white(), threats.black())
        } else {
            (acc.black(), acc.white(), threats.black(), threats.white())
        };

        if self.l1_size > 0 && self.use_pairwise {
            return self.forward_l1_pairwise_with_threats::<B>(
                stm_acc, ntm_acc, stm_threat, ntm_threat, bucket,
            );
        }

        debug_assert!(
            !self.has_threat_features(),
            "unsupported threat NNUE architecture reached inference"
        );
        self.forward_with_backend::<B>(acc, stm, piece_count)
    }

    #[inline(always)]
    pub(crate) fn forward_with_backend<B: NnueBackend>(
        &self,
        acc: &NNUEAccumulator,
        stm: u8,
        piece_count: u32,
    ) -> i32 {
        let bucket = output_bucket(piece_count);
        let out_w = self.output_weight_row(bucket);

        let (stm_acc, ntm_acc) = if stm == WHITE {
            (acc.white(), acc.black())
        } else {
            (acc.black(), acc.white())
        };

        if self.l1_size > 0 && self.use_pairwise {
            return self.forward_l1_pairwise::<B>(stm_acc, ntm_acc, bucket);
        }
        if self.use_pairwise {
            return self.forward_v6_pairwise(stm_acc, ntm_acc, bucket, out_w);
        }
        self.forward_base::<B>(stm_acc, ntm_acc, bucket, out_w)
    }

    fn output_weight_row(&self, bucket: usize) -> &[i16] {
        let w = if self.l2_per_bucket > 0 {
            self.l2_per_bucket
        } else if self.l1_per_bucket > 0 {
            self.l1_per_bucket
        } else if self.use_pairwise {
            self.hidden_size
        } else {
            2 * self.hidden_size
        };
        &self.output_weights[bucket * w..bucket * w + w]
    }

    #[inline(always)]
    fn forward_base<B: NnueBackend>(
        &self,
        stm: &[i16],
        ntm: &[i16],
        bucket: usize,
        out_w: &[i16],
    ) -> i32 {
        let h = self.hidden_size;
        let mut output = self.output_bias[bucket] as i64;

        if self.use_screlu {
            output += B::forward_base_crelu(
                stm,
                ntm,
                out_w,
                h,
                true,
                self.screlu_i32_output_safe,
                self.screlu_i32_accumulator_safe,
            );
            output /= QA as i64;
        } else {
            output += B::forward_base_crelu(stm, ntm, out_w, h, false, false, false);
        }

        let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
        if self.use_screlu {
            result = result * 4 / 5;
        }
        result
    }

    fn forward_v6_pairwise(&self, stm: &[i16], ntm: &[i16], bucket: usize, out_w: &[i16]) -> i32 {
        let pw = self.hidden_size / 2;
        let mut sum: i64 = 0;

        for i in 0..pw {
            let a = (stm[i] as i32).clamp(0, QA);
            let b = (stm[i + pw] as i32).clamp(0, QA);
            sum += (a * b) as i64 * out_w[i] as i64;
        }
        for i in 0..pw {
            let a = (ntm[i] as i32).clamp(0, QA);
            let b = (ntm[i + pw] as i32).clamp(0, QA);
            sum += (a * b) as i64 * out_w[pw + i] as i64;
        }

        let output = sum / QA as i64 + self.output_bias[bucket] as i64;
        (output * EVAL_SCALE as i64 / QAB as i64) as i32
    }

    #[inline(never)]
    fn forward_l1_pairwise<B: NnueBackend>(&self, stm: &[i16], ntm: &[i16], bucket: usize) -> i32 {
        self.forward_l1_pairwise_inner::<B>(stm, ntm, None, None, bucket)
    }

    #[inline(never)]
    fn forward_l1_pairwise_with_threats<B: NnueBackend>(
        &self,
        stm: &[i16],
        ntm: &[i16],
        stm_threat: &[i16],
        ntm_threat: &[i16],
        bucket: usize,
    ) -> i32 {
        self.forward_l1_pairwise_inner::<B>(stm, ntm, Some(stm_threat), Some(ntm_threat), bucket)
    }

    #[inline(always)]
    fn forward_l1_pairwise_inner<B: NnueBackend>(
        &self,
        stm: &[i16],
        ntm: &[i16],
        stm_threat: Option<&[i16]>,
        ntm_threat: Option<&[i16]>,
        bucket: usize,
    ) -> i32 {
        let pw = self.hidden_size / 2;
        let l1_total = self.l1_size;
        let l1_pb = self.l1_per_bucket;
        let qa_l1 = self.l1_scale;

        let l1_off = if self.bucketed_hidden {
            bucket * l1_pb
        } else {
            0
        };
        let l1 = if self.bucketed_hidden {
            l1_pb
        } else {
            l1_total
        };

        debug_assert!(pw <= MAX_HIDDEN_SIZE / 2);
        debug_assert!(l1 <= MAX_HIDDEN_SIZE);

        let mut sp = uninit_array::<u8, { MAX_HIDDEN_SIZE / 2 }>();
        let mut np = uninit_array::<u8, { MAX_HIDDEN_SIZE / 2 }>();
        if let (Some(stm_threat), Some(ntm_threat)) = (stm_threat, ntm_threat) {
            B::pairwise_pack_with_threats(stm, stm_threat, pw, &mut sp[..pw]);
            B::pairwise_pack_with_threats(ntm, ntm_threat, pw, &mut np[..pw]);
        } else {
            B::pairwise_pack(stm, pw, &mut sp[..pw]);
            B::pairwise_pack(ntm, pw, &mut np[..pw]);
        }
        let sp = unsafe { assume_init_slice(&sp[..pw]) };
        let np = unsafe { assume_init_slice(&np[..pw]) };

        let pw_scale = (QA * QA) >> FT_SHIFT;
        let mut hidden32 = uninit_array::<i32, MAX_HIDDEN_SIZE>();
        self.l1_matmul::<B>(
            sp,
            np,
            l1_total,
            l1,
            l1_off,
            pw,
            pw_scale,
            &mut hidden32[..l1],
        );
        let hidden32 = unsafe { assume_init_slice(&hidden32[..l1]) };

        let mut l1_out = uninit_array::<f32, MAX_HIDDEN_SIZE>();
        let l1_count = if self.dual_l1 { l1 * 2 } else { l1 };
        {
            let qa_f = qa_l1 as f32;
            let qsq = qa_f * qa_f;
            if self.dual_l1 {
                for i in 0..l1 {
                    let v = (hidden32[i] / pw_scale).clamp(0, qa_l1) as f32;
                    l1_out[i].write(v / qa_f); // CReLU half
                    l1_out[l1 + i].write((v * v) / qsq); // SCReLU half
                }
            } else if self.crelu_hidden {
                for i in 0..l1 {
                    let v = (hidden32[i] / pw_scale).clamp(0, qa_l1) as f32;
                    l1_out[i].write(v / qa_f);
                }
            } else {
                Self::screlu_activation::<B>(hidden32, pw_scale, qa_l1, &mut l1_out[..l1]);
            }
        }
        let l1_out = unsafe { assume_init_slice(&l1_out[..l1_count]) };

        if self.l2_per_bucket > 0 {
            self.forward_l2::<B>(l1_out, bucket, l1)
        } else {
            self.forward_l1_output(l1_out, bucket, l1)
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn l1_matmul<B: NnueBackend>(
        &self,
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        out: &mut [MaybeUninit<i32>],
    ) {
        B::l1_matmul(
            sp,
            np,
            l1_total,
            l1,
            l1_off,
            pw,
            pw_scale,
            &self.l1_weights,
            &self.l1_biases,
            out,
        )
    }

    #[inline(always)]
    fn screlu_activation<B: NnueBackend>(
        hidden: &[i32],
        pw_scale: i32,
        qa_l1: i32,
        out: &mut [MaybeUninit<f32>],
    ) {
        B::screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    fn forward_l2<B: NnueBackend>(&self, l1_out: &[f32], bucket: usize, _l1: usize) -> i32 {
        let l2_pb = self.l2_per_bucket;
        let l2_total = self.l2_size;
        let l2_off = if self.bucketed_hidden {
            bucket * l2_pb
        } else {
            0
        };
        let l2 = if self.bucketed_hidden {
            l2_pb
        } else {
            l2_total
        };

        let ow = &self.out_weights_f[bucket * l2_pb..bucket * l2_pb + l2_pb];
        let mut scratch = uninit_array::<f32, MAX_HIDDEN_SIZE>();
        let of = B::forward_l2(
            l1_out,
            &self.l2_weights_f,
            &self.l2_biases_f,
            l2,
            l2_total,
            l2_off,
            ow,
            self.out_bias_f[bucket],
            self.crelu_hidden,
            &mut scratch[..l2],
        );
        (of * EVAL_SCALE as f32) as i32
    }

    fn forward_l1_output(&self, l1_out: &[f32], bucket: usize, l1: usize) -> i32 {
        let l1_pb = self.l1_per_bucket;
        let ow = &self.out_weights_f[bucket * l1_pb..bucket * l1_pb + l1_pb];
        let mut of = self.out_bias_f[bucket];
        for i in 0..l1 {
            of += l1_out[i] * ow[i];
        }
        (of * EVAL_SCALE as f32) as i32
    }
}

#[derive(Clone, Copy)]
struct FeatureChange {
    color: u8,
    piece_type: u8,
    square: u8,
}

#[derive(Clone, Copy)]
struct MoveFeatureChanges {
    removed: [Option<FeatureChange>; 2],
    added: Option<FeatureChange>,
}

pub struct NNUEAccumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    pub hs: usize,
    pub wk: u8,
    pub bk: u8,
}

impl Clone for NNUEAccumulator {
    fn clone(&self) -> Self {
        Self {
            white: self.white.clone(),
            black: self.black.clone(),
            hs: self.hs,
            wk: self.wk,
            bk: self.bk,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.white.clone_from(&source.white);
        self.black.clone_from(&source.black);
        self.hs = source.hs;
        self.wk = source.wk;
        self.bk = source.bk;
    }
}

impl NNUEAccumulator {
    pub fn new(hs: usize) -> Self {
        NNUEAccumulator {
            white: vec![0i16; hs],
            black: vec![0i16; hs],
            hs,
            wk: 0,
            bk: 0,
        }
    }

    pub fn white(&self) -> &[i16] {
        &self.white
    }
    pub fn black(&self) -> &[i16] {
        &self.black
    }

    #[inline(always)]
    fn add_row<B: NnueBackend>(acc: &mut [i16], row: &[i16]) {
        B::add_row(acc, row)
    }

    #[inline(always)]
    fn remove_row<B: NnueBackend>(acc: &mut [i16], row: &[i16]) {
        B::sub_row(acc, row)
    }

    #[inline(always)]
    fn add_feature<B: NnueBackend>(acc: &mut [i16], net: &NNUENet, idx: usize) {
        if let Some(row) = net.input_row_fast(idx) {
            Self::add_row::<B>(acc, row);
        }
    }

    #[inline(always)]
    fn seed_or_add_feature<B: NnueBackend>(
        acc: &mut [i16],
        net: &NNUENet,
        idx: usize,
        seeded: &mut bool,
    ) {
        let Some(row) = net.input_row_fast(idx) else {
            return;
        };

        if *seeded {
            Self::add_row::<B>(acc, row);
        } else {
            let zero = &ZERO_FEATURE_ROW[..acc.len()];
            B::copy_update(acc, &net.input_biases[..acc.len()], zero, zero, row);
            *seeded = true;
        }
    }

    #[inline(always)]
    fn remove_feature<B: NnueBackend>(acc: &mut [i16], net: &NNUENet, idx: usize) {
        if let Some(row) = net.input_row_fast(idx) {
            Self::remove_row::<B>(acc, row);
        }
    }

    pub fn refresh(&mut self, net: &NNUENet, st: &BoardState) {
        self.refresh_with_backend::<ScalarNnueBackend>(net, st)
    }

    pub fn refresh_with_kind(&mut self, backend: NnueBackendKind, net: &NNUENet, st: &BoardState) {
        debug_assert!(nnue_backend_available(backend));
        match backend {
            NnueBackendKind::Scalar => self.refresh_with_backend::<ScalarNnueBackend>(net, st),
            NnueBackendKind::Simd128 => self.refresh_with_backend::<Simd128NnueBackend>(net, st),
            NnueBackendKind::Simd256 => self.refresh_with_backend::<SimdNnueBackend>(net, st),
            NnueBackendKind::Simd512 => self.refresh_with_backend::<Simd512NnueBackend>(net, st),
            NnueBackendKind::X86Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    self.refresh_with_backend::<Avx512NnueBackend>(net, st)
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.refresh_with_backend::<ScalarNnueBackend>(net, st)
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn refresh_with_backend<B: NnueBackend>(&mut self, net: &NNUENet, st: &BoardState) {
        let h = self.hs;
        let wk = convert(st.king_sq(true) as u8);
        let bk = convert(st.king_sq(false) as u8);
        self.wk = wk;
        self.bk = bk;

        let mut white_seeded = false;
        let mut black_seeded = false;

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let mut bb = st.bb[(if color == 0 { 0 } else { 6 }) + pt as usize];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    let csq = convert(sq);

                    Self::seed_or_add_feature::<B>(
                        &mut self.white,
                        net,
                        net.halfka(0, wk, color, pt, csq),
                        &mut white_seeded,
                    );
                    Self::seed_or_add_feature::<B>(
                        &mut self.black,
                        net,
                        net.halfka(1, bk, color, pt, csq),
                        &mut black_seeded,
                    );
                }
            }
        }

        if !white_seeded {
            self.white.copy_from_slice(&net.input_biases[..h]);
        }
        if !black_seeded {
            self.black.copy_from_slice(&net.input_biases[..h]);
        }
    }

    #[inline(always)]
    fn add_piece<B: NnueBackend>(&mut self, net: &NNUENet, color: u8, pt: u8, sq: u8) {
        let csq = convert(sq);

        Self::add_feature::<B>(&mut self.white, net, net.halfka(0, self.wk, color, pt, csq));
        Self::add_feature::<B>(&mut self.black, net, net.halfka(1, self.bk, color, pt, csq));
    }

    #[inline(always)]
    fn remove_piece<B: NnueBackend>(&mut self, net: &NNUENet, color: u8, pt: u8, sq: u8) {
        let csq = convert(sq);

        Self::remove_feature::<B>(&mut self.white, net, net.halfka(0, self.wk, color, pt, csq));
        Self::remove_feature::<B>(&mut self.black, net, net.halfka(1, self.bk, color, pt, csq));
    }

    #[allow(clippy::too_many_arguments)]
    fn move_feature_changes(
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> Option<MoveFeatureChanges> {
        use crate::board::{is_white_piece, piece_on, piece_type, sq, EMPTY_SQ};

        let from = sq(sr, sc);
        let to = sq(er, ec);
        let mover_pi = piece_on(&st_before.bb, from);
        if mover_pi == EMPTY_SQ {
            return Some(MoveFeatureChanges {
                removed: [None, None],
                added: None,
            });
        }

        let mover_type = piece_type(mover_pi);
        let white = is_white_piece(mover_pi);
        let color = if white { 0 } else { 1 };
        if mover_type == 5 {
            return None;
        }

        let mut changes = MoveFeatureChanges {
            removed: [
                Some(FeatureChange {
                    color,
                    piece_type: mover_type,
                    square: from as u8,
                }),
                None,
            ],
            added: None,
        };

        let cap_pi = piece_on(&st_before.bb, to);
        if cap_pi != EMPTY_SQ {
            changes.removed[1] = Some(FeatureChange {
                color: if is_white_piece(cap_pi) { 0 } else { 1 },
                piece_type: piece_type(cap_pi),
                square: to as u8,
            });
        } else if mover_type == 0 && Some(to) == st_before.ep && sc != ec {
            changes.removed[1] = Some(FeatureChange {
                color: if white { 1 } else { 0 },
                piece_type: 0,
                square: if white {
                    (to + 8) as u8
                } else {
                    (to - 8) as u8
                },
            });
        }

        let added_type = if mover_type == 0 && (er == 0 || er == 7) {
            match promotion.to_ascii_uppercase() {
                b'Q' => 4,
                b'R' => 3,
                b'B' => 2,
                b'N' => 1,
                _ => 4,
            }
        } else {
            mover_type
        };
        changes.added = Some(FeatureChange {
            color,
            piece_type: added_type,
            square: to as u8,
        });

        Some(changes)
    }

    #[inline(always)]
    fn feature_row(
        net: &NNUENet,
        perspective: u8,
        king: u8,
        change: Option<FeatureChange>,
    ) -> &[i16] {
        change
            .and_then(|change| {
                net.input_row_fast(net.halfka(
                    perspective,
                    king,
                    change.color,
                    change.piece_type,
                    convert(change.square),
                ))
            })
            .unwrap_or(&ZERO_FEATURE_ROW[..net.hidden_size])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_move(
        &mut self,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        self.update_move_with_backend::<ScalarNnueBackend>(
            net, st_before, sr, sc, er, ec, promotion,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_move_with_kind(
        &mut self,
        backend: NnueBackendKind,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        debug_assert!(nnue_backend_available(backend));
        match backend {
            NnueBackendKind::Scalar => self.update_move_with_backend::<ScalarNnueBackend>(
                net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd128 => self.update_move_with_backend::<Simd128NnueBackend>(
                net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd256 => self.update_move_with_backend::<SimdNnueBackend>(
                net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd512 => self.update_move_with_backend::<Simd512NnueBackend>(
                net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::X86Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    self.update_move_with_backend::<Avx512NnueBackend>(
                        net, st_before, sr, sc, er, ec, promotion,
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.update_move_with_backend::<ScalarNnueBackend>(
                        net, st_before, sr, sc, er, ec, promotion,
                    )
                }
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn update_from_parent_with_kind(
        &mut self,
        backend: NnueBackendKind,
        parent: &Self,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        debug_assert!(nnue_backend_available(backend));
        match backend {
            NnueBackendKind::Scalar => self.update_from_parent_with_backend::<ScalarNnueBackend>(
                parent, net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd128 => self.update_from_parent_with_backend::<Simd128NnueBackend>(
                parent, net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd256 => self.update_from_parent_with_backend::<SimdNnueBackend>(
                parent, net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::Simd512 => self.update_from_parent_with_backend::<Simd512NnueBackend>(
                parent, net, st_before, sr, sc, er, ec, promotion,
            ),
            NnueBackendKind::X86Avx512 => {
                #[cfg(target_arch = "x86_64")]
                {
                    self.update_from_parent_with_backend::<Avx512NnueBackend>(
                        parent, net, st_before, sr, sc, er, ec, promotion,
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.update_from_parent_with_backend::<ScalarNnueBackend>(
                        parent, net, st_before, sr, sc, er, ec, promotion,
                    )
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_from_parent_with_backend<B: NnueBackend>(
        &mut self,
        parent: &Self,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        let Some(changes) = Self::move_feature_changes(st_before, sr, sc, er, ec, promotion) else {
            return false;
        };

        if self.white.len() != parent.white.len() {
            self.white.resize(parent.white.len(), 0);
        }
        if self.black.len() != parent.black.len() {
            self.black.resize(parent.black.len(), 0);
        }

        let white_remove0 = Self::feature_row(net, 0, parent.wk, changes.removed[0]);
        let white_remove1 = Self::feature_row(net, 0, parent.wk, changes.removed[1]);
        let white_add = Self::feature_row(net, 0, parent.wk, changes.added);
        B::copy_update(
            &mut self.white,
            &parent.white,
            white_remove0,
            white_remove1,
            white_add,
        );

        let black_remove0 = Self::feature_row(net, 1, parent.bk, changes.removed[0]);
        let black_remove1 = Self::feature_row(net, 1, parent.bk, changes.removed[1]);
        let black_add = Self::feature_row(net, 1, parent.bk, changes.added);
        B::copy_update(
            &mut self.black,
            &parent.black,
            black_remove0,
            black_remove1,
            black_add,
        );

        self.hs = parent.hs;
        self.wk = parent.wk;
        self.bk = parent.bk;
        true
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_move_with_backend<B: NnueBackend>(
        &mut self,
        net: &NNUENet,
        st_before: &BoardState,
        sr: usize,
        sc: usize,
        er: usize,
        ec: usize,
        promotion: u8,
    ) -> bool {
        let Some(changes) = Self::move_feature_changes(st_before, sr, sc, er, ec, promotion) else {
            return false;
        };

        for change in changes.removed.into_iter().flatten() {
            self.remove_piece::<B>(net, change.color, change.piece_type, change.square);
        }
        if let Some(change) = changes.added {
            self.add_piece::<B>(net, change.color, change.piece_type, change.square);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::{convert, NNUEAccumulator, NNUENet, ScalarNnueBackend};
    use crate::backend::available_nnue_backends;
    use crate::Engine;

    const COMPACT_NET: &[u8] = include_bytes!("../net.compact.nnue");

    fn parse_uci_move(mv: &str) -> (usize, usize, usize, usize, u8) {
        let bytes = mv.as_bytes();
        assert!(matches!(bytes.len(), 4 | 5), "invalid UCI move: {mv}");
        let sc = (bytes[0] - b'a') as usize;
        let sr = 8 - (bytes[1] - b'0') as usize;
        let ec = (bytes[2] - b'a') as usize;
        let er = 8 - (bytes[3] - b'0') as usize;
        let promotion = bytes.get(4).copied().unwrap_or(0).to_ascii_uppercase();
        (sr, sc, er, ec, promotion)
    }

    fn reference_refresh(net: &NNUENet, engine: &Engine) -> NNUEAccumulator {
        let mut acc = NNUEAccumulator::new(net.hidden_size);
        acc.wk = convert(engine.st.king_sq(true) as u8);
        acc.bk = convert(engine.st.king_sq(false) as u8);
        acc.white.copy_from_slice(&net.input_biases);
        acc.black.copy_from_slice(&net.input_biases);

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let mut bb = engine.st.bb[(if color == 0 { 0 } else { 6 }) + pt as usize];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    let csq = convert(sq);
                    NNUEAccumulator::add_feature::<ScalarNnueBackend>(
                        &mut acc.white,
                        net,
                        net.halfka(0, acc.wk, color, pt, csq),
                    );
                    NNUEAccumulator::add_feature::<ScalarNnueBackend>(
                        &mut acc.black,
                        net,
                        net.halfka(1, acc.bk, color, pt, csq),
                    );
                }
            }
        }

        acc
    }

    fn assert_fused_line_matches_reference(net: &NNUENet, fen: &str, moves: &[&str]) {
        for backend in available_nnue_backends() {
            let mut engine = Engine::new();
            engine.try_set_fen(fen).expect("test FEN should parse");

            let expected = reference_refresh(net, &engine);
            let mut parent = NNUEAccumulator::new(net.hidden_size);
            parent.refresh_with_kind(backend, net, &engine.st);
            assert_eq!(parent.white, expected.white);
            assert_eq!(parent.black, expected.black);

            for &uci in moves {
                let (sr, sc, er, ec, promotion) = parse_uci_move(uci);
                let before = engine.st;
                let mut child = NNUEAccumulator::new(net.hidden_size);
                let fused = child.update_from_parent_with_kind(
                    backend, &parent, net, &before, sr, sc, er, ec, promotion,
                );

                let mut incremental_reference = parent.clone();
                let incremental = incremental_reference
                    .update_move_with_kind(backend, net, &before, sr, sc, er, ec, promotion);
                assert_eq!(fused, incremental, "update kind differs for {uci}");

                assert!(
                    engine.make_move_uci(sr, sc, er, ec, promotion),
                    "{uci} should be legal"
                );
                if !fused {
                    child.refresh_with_kind(backend, net, &engine.st);
                    incremental_reference.refresh_with_kind(backend, net, &engine.st);
                }

                assert_eq!(
                    child.white, incremental_reference.white,
                    "white fused update differs after {uci} with {backend:?}"
                );
                assert_eq!(
                    child.black, incremental_reference.black,
                    "black fused update differs after {uci} with {backend:?}"
                );
                assert_eq!(
                    (child.wk, child.bk),
                    (incremental_reference.wk, incremental_reference.bk),
                    "king metadata differs after {uci} with {backend:?}"
                );

                let refreshed = reference_refresh(net, &engine);
                assert_eq!(
                    child.white, refreshed.white,
                    "white refresh differs after {uci} with {backend:?}"
                );
                assert_eq!(
                    child.black, refreshed.black,
                    "black refresh differs after {uci} with {backend:?}"
                );
                parent = child;
            }
        }
    }

    #[test]
    fn accumulator_clone_from_reuses_matching_buffers() {
        let mut source = NNUEAccumulator::new(1024);
        source.white[0] = 17;
        source.black[1023] = -23;
        source.wk = 7;
        source.bk = 56;

        let mut destination = NNUEAccumulator::new(1024);
        let white_ptr = destination.white.as_ptr();
        let black_ptr = destination.black.as_ptr();
        let white_capacity = destination.white.capacity();
        let black_capacity = destination.black.capacity();

        destination.clone_from(&source);

        assert_eq!(destination.white, source.white);
        assert_eq!(destination.black, source.black);
        assert_eq!(destination.hs, source.hs);
        assert_eq!(destination.wk, source.wk);
        assert_eq!(destination.bk, source.bk);
        assert_eq!(destination.white.as_ptr(), white_ptr);
        assert_eq!(destination.black.as_ptr(), black_ptr);
        assert_eq!(destination.white.capacity(), white_capacity);
        assert_eq!(destination.black.capacity(), black_capacity);
    }

    #[test]
    fn fused_parent_updates_and_refreshes_match_reference_paths() {
        let net = NNUENet::load_compact_from_bytes(COMPACT_NET, "<move motifs>")
            .expect("compact NNUE should load");

        assert_fused_line_matches_reference(
            &net,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"],
        );
        assert_fused_line_matches_reference(&net, "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", &["e5d6"]);
        for promotion in ["g7g8q", "g7g8r", "g7g8b", "g7g8n"] {
            assert_fused_line_matches_reference(
                &net,
                "4k3/6P1/8/8/8/8/8/4K3 w - - 0 1",
                &[promotion],
            );
        }
        assert_fused_line_matches_reference(&net, "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1", &["g7h8q"]);
    }
}
