use super::{NNUEAccumulator, NNUENet};
use crate::board::BoardState;
use crate::simd;
use std::mem::MaybeUninit;

pub(crate) trait NnueBackend: Copy {
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32;
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState);
    fn add_row(acc: &mut [i16], row: &[i16]);
    fn sub_row(acc: &mut [i16], row: &[i16]);
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]);
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        screlu_i32_output_safe: bool,
        screlu_i32_accumulator_safe: bool,
    ) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    );
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]);
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32;
}

#[derive(Clone, Copy)]
pub(crate) struct ScalarNnueBackend;

#[derive(Clone, Copy)]
pub(crate) struct Simd128NnueBackend;

#[derive(Clone, Copy)]
pub(crate) struct SimdNnueBackend;

#[derive(Clone, Copy)]
pub(crate) struct Simd512NnueBackend;

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx512NnueBackend;

impl NnueBackend for ScalarNnueBackend {
    #[inline(always)]
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        net.forward_with_backend::<Self>(acc, stm, piece_count)
    }

    #[inline(always)]
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
        acc.refresh_with_backend::<Self>(net, st)
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        simd::scalar_add_row(acc, row)
    }

    #[inline(always)]
    fn sub_row(acc: &mut [i16], row: &[i16]) {
        simd::scalar_sub_row(acc, row)
    }

    #[inline(always)]
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]) {
        simd::scalar_copy_update(dst, src, remove0, remove1, add)
    }

    #[inline(always)]
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        _screlu_i32_output_safe: bool,
        _screlu_i32_accumulator_safe: bool,
    ) -> i64 {
        simd::scalar_forward_base_crelu(stm, ntm, out_w, h, use_screlu)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    ) {
        simd::scalar_l1_matmul(
            sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
        )
    }

    #[inline(always)]
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]) {
        simd::scalar_screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32 {
        simd::scalar_forward_l2(
            l1_out,
            l2_weights,
            l2_biases,
            l2,
            l2_total,
            l2_off,
            out_weights,
            out_bias,
        )
    }
}

impl NnueBackend for Simd128NnueBackend {
    #[inline(always)]
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        net.forward_with_backend::<Self>(acc, stm, piece_count)
    }

    #[inline(always)]
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
        acc.refresh_with_backend::<Self>(net, st)
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        simd::simd128_add_row(acc, row)
    }

    #[inline(always)]
    fn sub_row(acc: &mut [i16], row: &[i16]) {
        simd::simd128_sub_row(acc, row)
    }

    #[inline(always)]
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]) {
        simd::simd128_copy_update(dst, src, remove0, remove1, add)
    }

    #[inline(always)]
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        screlu_i32_output_safe: bool,
        _screlu_i32_accumulator_safe: bool,
    ) -> i64 {
        simd::simd128_forward_base_crelu(stm, ntm, out_w, h, use_screlu, screlu_i32_output_safe)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    ) {
        simd::simd128_l1_matmul(
            sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
        )
    }

    #[inline(always)]
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]) {
        simd::scalar_screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32 {
        simd::simd128_forward_l2(
            l1_out,
            l2_weights,
            l2_biases,
            l2,
            l2_total,
            l2_off,
            out_weights,
            out_bias,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
unsafe fn nnue_forward_x86_v3(
    net: &NNUENet,
    acc: &NNUEAccumulator,
    stm: u8,
    piece_count: u32,
) -> i32 {
    net.forward_with_backend::<SimdNnueBackend>(acc, stm, piece_count)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
unsafe fn nnue_refresh_x86_v3(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
    acc.refresh_with_backend::<SimdNnueBackend>(net, st)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
unsafe fn nnue_forward_x86_avx512(
    net: &NNUENet,
    acc: &NNUEAccumulator,
    stm: u8,
    piece_count: u32,
) -> i32 {
    net.forward_with_backend::<Avx512NnueBackend>(acc, stm, piece_count)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avx512f,avx512bw,avx512dq,avx512vl,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
unsafe fn nnue_refresh_x86_avx512(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
    acc.refresh_with_backend::<Avx512NnueBackend>(net, st)
}

impl NnueBackend for SimdNnueBackend {
    #[inline(always)]
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            nnue_forward_x86_v3(net, acc, stm, piece_count)
        }
        #[cfg(not(target_arch = "x86_64"))]
        net.forward_with_backend::<Self>(acc, stm, piece_count)
    }

    #[inline(always)]
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            nnue_refresh_x86_v3(acc, net, st);
        }
        #[cfg(not(target_arch = "x86_64"))]
        acc.refresh_with_backend::<Self>(net, st)
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_add_row_x86_v3(acc, row);
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_add_row(acc, row)
    }

    #[inline(always)]
    fn sub_row(acc: &mut [i16], row: &[i16]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_sub_row_x86_v3(acc, row);
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_sub_row(acc, row)
    }

    #[inline(always)]
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_copy_update_x86_v3(dst, src, remove0, remove1, add);
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_copy_update(dst, src, remove0, remove1, add)
    }
    #[allow(unused_variables)]
    #[inline(always)]
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        screlu_i32_output_safe: bool,
        screlu_i32_accumulator_safe: bool,
    ) -> i64 {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = screlu_i32_accumulator_safe;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_forward_base_crelu_x86_v3(
                stm,
                ntm,
                out_w,
                h,
                use_screlu,
                screlu_i32_output_safe,
                screlu_i32_accumulator_safe,
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_forward_base_crelu(stm, ntm, out_w, h, use_screlu, screlu_i32_output_safe)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_l1_matmul_x86_v3(
                sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_l1_matmul(
            sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
        )
    }

    #[inline(always)]
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]) {
        simd::scalar_screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::simd_forward_l2_x86_v3(
                l1_out,
                l2_weights,
                l2_biases,
                l2,
                l2_total,
                l2_off,
                out_weights,
                out_bias,
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        simd::simd_forward_l2(
            l1_out,
            l2_weights,
            l2_biases,
            l2,
            l2_total,
            l2_off,
            out_weights,
            out_bias,
        )
    }
}

impl NnueBackend for Simd512NnueBackend {
    #[inline(always)]
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        net.forward_with_backend::<Self>(acc, stm, piece_count)
    }

    #[inline(always)]
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
        acc.refresh_with_backend::<Self>(net, st)
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        simd::simd512_add_row(acc, row)
    }

    #[inline(always)]
    fn sub_row(acc: &mut [i16], row: &[i16]) {
        simd::simd512_sub_row(acc, row)
    }

    #[inline(always)]
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]) {
        simd::simd512_copy_update(dst, src, remove0, remove1, add)
    }

    #[inline(always)]
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        screlu_i32_output_safe: bool,
        _screlu_i32_accumulator_safe: bool,
    ) -> i64 {
        simd::simd512_forward_base_crelu(stm, ntm, out_w, h, use_screlu, screlu_i32_output_safe)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    ) {
        simd::simd512_l1_matmul(
            sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
        )
    }

    #[inline(always)]
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]) {
        simd::scalar_screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32 {
        simd::simd512_forward_l2(
            l1_out,
            l2_weights,
            l2_biases,
            l2,
            l2_total,
            l2_off,
            out_weights,
            out_bias,
        )
    }
}

#[cfg(target_arch = "x86_64")]
impl NnueBackend for Avx512NnueBackend {
    #[inline(always)]
    fn forward(net: &NNUENet, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        unsafe { nnue_forward_x86_avx512(net, acc, stm, piece_count) }
    }

    #[inline(always)]
    fn refresh(acc: &mut NNUEAccumulator, net: &NNUENet, st: &BoardState) {
        unsafe {
            nnue_refresh_x86_avx512(acc, net, st);
        }
    }

    #[inline(always)]
    fn add_row(acc: &mut [i16], row: &[i16]) {
        unsafe {
            simd::simd_add_row_x86_avx512(acc, row);
        }
    }

    #[inline(always)]
    fn sub_row(acc: &mut [i16], row: &[i16]) {
        unsafe {
            simd::simd_sub_row_x86_avx512(acc, row);
        }
    }

    #[inline(always)]
    fn copy_update(dst: &mut [i16], src: &[i16], remove0: &[i16], remove1: &[i16], add: &[i16]) {
        unsafe {
            simd::simd_copy_update_x86_avx512(dst, src, remove0, remove1, add);
        }
    }

    #[inline(always)]
    fn forward_base_crelu(
        stm: &[i16],
        ntm: &[i16],
        out_w: &[i16],
        h: usize,
        use_screlu: bool,
        screlu_i32_output_safe: bool,
        screlu_i32_accumulator_safe: bool,
    ) -> i64 {
        unsafe {
            simd::simd_forward_base_crelu_x86_avx512(
                stm,
                ntm,
                out_w,
                h,
                use_screlu,
                screlu_i32_output_safe,
                screlu_i32_accumulator_safe,
            )
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn l1_matmul(
        sp: &[u8],
        np: &[u8],
        l1_total: usize,
        l1: usize,
        l1_off: usize,
        pw: usize,
        pw_scale: i32,
        l1_weights: &[i16],
        l1_biases: &[i16],
        out: &mut [MaybeUninit<i32>],
    ) {
        unsafe {
            simd::simd_l1_matmul_x86_avx512(
                sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
            );
        }
    }

    #[inline(always)]
    fn screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [MaybeUninit<f32>]) {
        simd::scalar_screlu_activation(hidden, pw_scale, qa_l1, out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn forward_l2(
        l1_out: &[f32],
        l2_weights: &[f32],
        l2_biases: &[f32],
        l2: usize,
        l2_total: usize,
        l2_off: usize,
        out_weights: &[f32],
        out_bias: f32,
    ) -> f32 {
        unsafe {
            simd::simd_forward_l2_x86_avx512(
                l1_out,
                l2_weights,
                l2_biases,
                l2,
                l2_total,
                l2_off,
                out_weights,
                out_bias,
            )
        }
    }
}
