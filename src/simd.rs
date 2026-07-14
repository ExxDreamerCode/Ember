use std::ptr;
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{simd_swizzle, Simd};

const QA: i32 = 255;
const I16_LANES: usize = 16;
const I32_LANES: usize = 8;
const F32_LANES: usize = 8;

type I16x = Simd<i16, I16_LANES>;
type I16x8 = Simd<i16, I32_LANES>;
type I32x = Simd<i32, I32_LANES>;
type F32x = Simd<f32, F32_LANES>;

#[inline(always)]
pub fn scalar_add_row(acc: &mut [i16], row: &[i16]) {
    debug_assert_eq!(acc.len(), row.len());
    for (acc_value, row_value) in acc.iter_mut().zip(row) {
        *acc_value += *row_value;
    }
}

#[inline(always)]
pub fn scalar_sub_row(acc: &mut [i16], row: &[i16]) {
    debug_assert_eq!(acc.len(), row.len());
    for (acc_value, row_value) in acc.iter_mut().zip(row) {
        *acc_value -= *row_value;
    }
}

#[inline(always)]
pub fn scalar_forward_base_crelu(
    stm: &[i16],
    ntm: &[i16],
    out_w: &[i16],
    h: usize,
    use_screlu: bool,
) -> i64 {
    let mut sum = 0i64;

    if use_screlu {
        for j in 0..h {
            let v = stm[j].clamp(0, QA as i16) as i64;
            sum += v * v * out_w[j] as i64;
        }
        for j in 0..h {
            let v = ntm[j].clamp(0, QA as i16) as i64;
            sum += v * v * out_w[h + j] as i64;
        }
        return sum;
    }

    for j in 0..h {
        let v = (stm[j] as i32).clamp(0, QA) as i64;
        sum += v * out_w[j] as i64;
    }
    for j in 0..h {
        let v = (ntm[j] as i32).clamp(0, QA) as i64;
        sum += v * out_w[h + j] as i64;
    }

    sum
}

#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn scalar_l1_matmul(
    sp: &[u8],
    np: &[u8],
    l1_total: usize,
    l1: usize,
    l1_off: usize,
    pw: usize,
    pw_scale: i32,
    l1_weights: &[i16],
    l1_biases: &[i16],
    out: &mut [i32],
) {
    for i in 0..l1 {
        let gi = l1_off + i;
        let mut sum = l1_biases[gi] as i32 * pw_scale;
        for j in 0..pw {
            sum += sp[j] as i32 * l1_weights[j * l1_total + gi] as i32;
            sum += np[j] as i32 * l1_weights[(pw + j) * l1_total + gi] as i32;
        }
        out[i] = sum;
    }
}

#[inline(always)]
pub fn scalar_screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [f32]) {
    let qf = qa_l1 as f32;
    let qsq = qf * qf;
    for (hidden_value, out_value) in hidden.iter().zip(out.iter_mut()) {
        let v = (*hidden_value / pw_scale).clamp(0, qa_l1);
        *out_value = (v * v) as f32 / qsq;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn scalar_forward_l2(
    l1_out: &[f32],
    l2_weights: &[f32],
    l2_biases: &[f32],
    l2: usize,
    l2_total: usize,
    l2_off: usize,
    out_weights: &[f32],
    out_bias: f32,
) -> f32 {
    let mut h2 = vec![0.0f32; l2];
    h2[..l2].copy_from_slice(&l2_biases[l2_off..l2_off + l2]);

    for (i, &l1_value) in l1_out.iter().enumerate() {
        if l1_value == 0.0 {
            continue;
        }
        let w_base = i * l2_total + l2_off;
        let w_row = &l2_weights[w_base..w_base + l2];
        for (h_value, w_value) in h2.iter_mut().zip(w_row) {
            *h_value += l1_value * *w_value;
        }
    }

    let mut result = out_bias;
    for (h_value, w_value) in h2.iter().zip(out_weights) {
        let v = h_value.clamp(0.0, 1.0);
        result += v * v * *w_value;
    }
    result
}

#[inline(always)]
// Safety: `offset + I32_LANES` must be within `slice`. The load is
// unaligned and copies initialized `i16` values before widening to `i32`.
unsafe fn load_i16_i32(slice: &[i16], offset: usize) -> I32x {
    debug_assert!(offset + I32_LANES <= slice.len());
    let ptr = unsafe { slice.as_ptr().add(offset) as *const [i16; I32_LANES] };
    I16x8::from_array(unsafe { ptr::read_unaligned(ptr) }).cast()
}

#[inline(always)]
fn dot_i16(a: I16x, b: I16x) -> i64 {
    let a_lo: I16x8 = simd_swizzle!(a, [0, 1, 2, 3, 4, 5, 6, 7]);
    let a_hi: I16x8 = simd_swizzle!(a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b_lo: I16x8 = simd_swizzle!(b, [0, 1, 2, 3, 4, 5, 6, 7]);
    let b_hi: I16x8 = simd_swizzle!(b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let products =
        a_lo.cast::<i32>() * b_lo.cast::<i32>() + a_hi.cast::<i32>() * b_hi.cast::<i32>();
    products.reduce_sum() as i64
}

#[inline(always)]
fn clamp_crelu_i16(value: I16x) -> I16x {
    value.simd_clamp(I16x::splat(0), I16x::splat(QA as i16))
}

#[inline(always)]
pub fn simd_add_row(acc: &mut [i16], row: &[i16]) {
    debug_assert_eq!(acc.len(), row.len());

    let (acc_chunks, acc_tail) = acc.as_chunks_mut::<I16_LANES>();
    let (row_chunks, row_tail) = row.as_chunks::<I16_LANES>();
    for (acc_chunk, row_chunk) in acc_chunks.iter_mut().zip(row_chunks) {
        let sum = I16x::from_array(*acc_chunk) + I16x::from_array(*row_chunk);
        *acc_chunk = sum.to_array();
    }
    for (acc_value, row_value) in acc_tail.iter_mut().zip(row_tail) {
        *acc_value += *row_value;
    }
}

#[inline(always)]
pub fn simd_sub_row(acc: &mut [i16], row: &[i16]) {
    debug_assert_eq!(acc.len(), row.len());

    let (acc_chunks, acc_tail) = acc.as_chunks_mut::<I16_LANES>();
    let (row_chunks, row_tail) = row.as_chunks::<I16_LANES>();
    for (acc_chunk, row_chunk) in acc_chunks.iter_mut().zip(row_chunks) {
        let sum = I16x::from_array(*acc_chunk) - I16x::from_array(*row_chunk);
        *acc_chunk = sum.to_array();
    }
    for (acc_value, row_value) in acc_tail.iter_mut().zip(row_tail) {
        *acc_value -= *row_value;
    }
}

#[inline(always)]
pub fn simd_forward_base_crelu(
    stm: &[i16],
    ntm: &[i16],
    out_w: &[i16],
    h: usize,
    use_screlu: bool,
) -> i64 {
    let mut sum = 0i64;

    if use_screlu {
        for j in 0..h {
            let v = stm[j].clamp(0, QA as i16) as i64;
            sum += v * v * out_w[j] as i64;
        }
        for j in 0..h {
            let v = ntm[j].clamp(0, QA as i16) as i64;
            sum += v * v * out_w[h + j] as i64;
        }
        return sum;
    }

    let (stm_chunks, stm_tail) = stm[..h].as_chunks::<I16_LANES>();
    let (ntm_chunks, ntm_tail) = ntm[..h].as_chunks::<I16_LANES>();
    let (sw_chunks, sw_tail) = out_w[..h].as_chunks::<I16_LANES>();
    let (nw_chunks, nw_tail) = out_w[h..h + h].as_chunks::<I16_LANES>();
    for (((stm_chunk, ntm_chunk), sw_chunk), nw_chunk) in stm_chunks
        .iter()
        .zip(ntm_chunks)
        .zip(sw_chunks)
        .zip(nw_chunks)
    {
        let sc = clamp_crelu_i16(I16x::from_array(*stm_chunk));
        let nc = clamp_crelu_i16(I16x::from_array(*ntm_chunk));
        let sw = I16x::from_array(*sw_chunk);
        let nw = I16x::from_array(*nw_chunk);
        sum += dot_i16(sc, sw);
        sum += dot_i16(nc, nw);
    }

    for (((stm_value, ntm_value), sw_value), nw_value) in
        stm_tail.iter().zip(ntm_tail).zip(sw_tail).zip(nw_tail)
    {
        let v = (*stm_value as i32).clamp(0, QA) as i64;
        sum += v * *sw_value as i64;
        let v = (*ntm_value as i32).clamp(0, QA) as i64;
        sum += v * *nw_value as i64;
    }

    sum
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn simd_l1_matmul(
    sp: &[u8],
    np: &[u8],
    l1_total: usize,
    l1: usize,
    l1_off: usize,
    pw: usize,
    pw_scale: i32,
    l1_weights: &[i16],
    l1_biases: &[i16],
    out: &mut [i32],
) {
    for i in 0..l1 {
        out[i] = l1_biases[l1_off + i] as i32 * pw_scale;
    }

    let (out_chunks, out_tail) = out.as_chunks_mut::<I32_LANES>();
    for (chunk_idx, out_chunk) in out_chunks.iter_mut().enumerate() {
        let i = chunk_idx * I32_LANES;
        let mut s_acc = I32x::splat(0);
        let mut n_acc = I32x::splat(0);

        for j in 0..pw {
            let sw = unsafe { load_i16_i32(l1_weights, j * l1_total + l1_off + i) };
            let nw = unsafe { load_i16_i32(l1_weights, (pw + j) * l1_total + l1_off + i) };

            s_acc += I32x::splat(sp[j] as i32) * sw;
            n_acc += I32x::splat(np[j] as i32) * nw;
        }

        *out_chunk = (I32x::from_array(*out_chunk) + s_acc + n_acc).to_array();
    }

    for (i, out_value) in (out_chunks.len() * I32_LANES..).zip(out_tail.iter_mut()) {
        let gi = l1_off + i;
        let mut s_sum = 0i32;
        let mut n_sum = 0i32;
        for j in 0..pw {
            s_sum += sp[j] as i32 * l1_weights[j * l1_total + gi] as i32;
            n_sum += np[j] as i32 * l1_weights[(pw + j) * l1_total + gi] as i32;
        }
        *out_value += s_sum + n_sum;
    }
}

#[inline(always)]
pub fn simd_screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [f32]) {
    let len = hidden.len();
    debug_assert!(len <= out.len());
    let qf = qa_l1 as f32;
    let qsq = qf * qf;

    for i in 0..len {
        let v = (hidden[i] / pw_scale).clamp(0, qa_l1);
        out[i] = (v * v) as f32 / qsq;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn simd_forward_l2(
    l1_out: &[f32],
    l2_weights: &[f32],
    l2_biases: &[f32],
    l2: usize,
    l2_total: usize,
    l2_off: usize,
    out_weights: &[f32],
    out_bias: f32,
) -> f32 {
    let mut h2 = vec![0.0f32; l2];
    h2[..l2].copy_from_slice(&l2_biases[l2_off..l2_off + l2]);

    for (i, &l1_value) in l1_out.iter().enumerate() {
        if l1_value == 0.0 {
            continue;
        }
        let l1v = F32x::splat(l1_value);
        let w_base = i * l2_total + l2_off;
        let w_row = &l2_weights[w_base..w_base + l2];
        let (h_chunks, h_tail) = h2.as_chunks_mut::<F32_LANES>();
        let (w_chunks, w_tail) = w_row.as_chunks::<F32_LANES>();
        for (h_chunk, w_chunk) in h_chunks.iter_mut().zip(w_chunks) {
            let h = F32x::from_array(*h_chunk);
            let w = F32x::from_array(*w_chunk);
            *h_chunk = (l1v * w + h).to_array();
        }
        for (h_value, w_value) in h_tail.iter_mut().zip(w_tail) {
            *h_value += l1_value * *w_value;
        }
    }

    let zero = F32x::splat(0.0);
    let one = F32x::splat(1.0);
    let (h_chunks, h_tail) = h2.as_chunks_mut::<F32_LANES>();
    for h_chunk in h_chunks {
        let h = F32x::from_array(*h_chunk).simd_clamp(zero, one);
        *h_chunk = (h * h).to_array();
    }
    for h_value in h_tail {
        *h_value = h_value.clamp(0.0, 1.0);
        *h_value *= *h_value;
    }

    let mut of = F32x::splat(0.0);
    let (h_chunks, h_tail) = h2.as_chunks::<F32_LANES>();
    let (w_chunks, w_tail) = out_weights[..l2].as_chunks::<F32_LANES>();
    for (h_chunk, w_chunk) in h_chunks.iter().zip(w_chunks) {
        let h = F32x::from_array(*h_chunk);
        let w = F32x::from_array(*w_chunk);
        of += h * w;
    }
    let mut result = out_bias + of.reduce_sum();
    for (h_value, w_value) in h_tail.iter().zip(w_tail) {
        result += *h_value * *w_value;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
pub unsafe fn simd_add_row_x86_v3(acc: &mut [i16], row: &[i16]) {
    simd_add_row(acc, row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
pub unsafe fn simd_sub_row_x86_v3(acc: &mut [i16], row: &[i16]) {
    simd_sub_row(acc, row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
pub unsafe fn simd_forward_base_crelu_x86_v3(
    stm: &[i16],
    ntm: &[i16],
    out_w: &[i16],
    h: usize,
    use_screlu: bool,
) -> i64 {
    simd_forward_base_crelu(stm, ntm, out_w, h, use_screlu)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn simd_l1_matmul_x86_v3(
    sp: &[u8],
    np: &[u8],
    l1_total: usize,
    l1: usize,
    l1_off: usize,
    pw: usize,
    pw_scale: i32,
    l1_weights: &[i16],
    l1_biases: &[i16],
    out: &mut [i32],
) {
    simd_l1_matmul(
        sp, np, l1_total, l1, l1_off, pw, pw_scale, l1_weights, l1_biases, out,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
pub unsafe fn simd_screlu_activation_x86_v3(
    hidden: &[i32],
    pw_scale: i32,
    qa_l1: i32,
    out: &mut [f32],
) {
    simd_screlu_activation(hidden, pw_scale, qa_l1, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn simd_forward_l2_x86_v3(
    l1_out: &[f32],
    l2_weights: &[f32],
    l2_biases: &[f32],
    l2: usize,
    l2_total: usize,
    l2_off: usize,
    out_weights: &[f32],
    out_bias: f32,
) -> f32 {
    simd_forward_l2(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_sub_rows_match_scalar_reference() {
        let mut actual: Vec<i16> = (0..67).map(|i| (i * 3 - 91) as i16).collect();
        let original = actual.clone();
        let row: Vec<i16> = (0..67).map(|i| (i * 5 - 83) as i16).collect();

        simd_add_row(&mut actual, &row);
        let expected: Vec<i16> = original
            .iter()
            .zip(&row)
            .map(|(left, right)| left + right)
            .collect();
        assert_eq!(actual, expected);

        simd_sub_row(&mut actual, &row);
        assert_eq!(actual, original);
    }

    #[test]
    fn forward_base_crelu_matches_scalar_reference() {
        let h = 35usize;
        let stm: Vec<i16> = (0..h).map(|i| ((i * 17) as i16) - 160).collect();
        let ntm: Vec<i16> = (0..h).map(|i| 280 - (i * 13) as i16).collect();
        let out_w: Vec<i16> = (0..2 * h).map(|i| ((i * 19 % 101) as i16) - 50).collect();

        for use_screlu in [false, true] {
            let actual = simd_forward_base_crelu(&stm, &ntm, &out_w, h, use_screlu);
            let mut expected = 0i64;
            for i in 0..h {
                let v = stm[i].clamp(0, QA as i16) as i64;
                expected += if use_screlu { v * v } else { v } * out_w[i] as i64;
            }
            for i in 0..h {
                let v = ntm[i].clamp(0, QA as i16) as i64;
                expected += if use_screlu { v * v } else { v } * out_w[h + i] as i64;
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn l1_matmul_matches_scalar_reference() {
        let pw = 7usize;
        let l1_total = 32usize;
        let l1_off = 5usize;
        let l1 = 19usize;
        let pw_scale = (QA * QA) >> 9;

        let sp: Vec<u8> = (0..pw).map(|i| (3 + i * 17) as u8).collect();
        let np: Vec<u8> = (0..pw).map(|i| (5 + i * 19) as u8).collect();
        let weights: Vec<i16> = (0..2 * pw * l1_total)
            .map(|i| ((i as i32 * 37 % 257) - 128) as i16)
            .collect();
        let biases: Vec<i16> = (0..l1_off + l1)
            .map(|i| ((i as i32 * 11 % 53) - 26) as i16)
            .collect();

        let mut actual = vec![0i32; l1];
        simd_l1_matmul(
            &sp,
            &np,
            l1_total,
            l1,
            l1_off,
            pw,
            pw_scale,
            &weights,
            &biases,
            &mut actual,
        );

        let mut expected = vec![0i32; l1];
        for (i, value) in expected.iter_mut().enumerate().take(l1) {
            let gi = l1_off + i;
            *value = biases[gi] as i32 * pw_scale;
            for j in 0..pw {
                *value += sp[j] as i32 * weights[j * l1_total + gi] as i32;
                *value += np[j] as i32 * weights[(pw + j) * l1_total + gi] as i32;
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn screlu_activation_matches_scalar_integer_reference() {
        let pw_scale = (QA * QA) >> 9;
        let qa_l1 = 255;
        let hidden = vec![
            -500,
            -1,
            0,
            1,
            pw_scale - 1,
            pw_scale,
            pw_scale + 1,
            2 * pw_scale - 1,
            2 * pw_scale,
            2 * pw_scale + 1,
            qa_l1 * pw_scale - 1,
            qa_l1 * pw_scale,
            qa_l1 * pw_scale + 1,
            (qa_l1 + 3) * pw_scale,
            12345,
            23456,
            34567,
        ];

        let mut actual = vec![0.0f32; hidden.len()];
        simd_screlu_activation(&hidden, pw_scale, qa_l1, &mut actual);

        let qsq = (qa_l1 as f32) * (qa_l1 as f32);
        let expected: Vec<f32> = hidden
            .iter()
            .map(|value| {
                let v = (*value / pw_scale).clamp(0, qa_l1);
                (v * v) as f32 / qsq
            })
            .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn forward_l2_matches_scalar_reference() {
        let l1 = 13usize;
        let l2 = 19usize;
        let l2_total = 32usize;
        let l2_off = 7usize;
        let l1_out: Vec<f32> = (0..l1)
            .map(|i| {
                if i % 4 == 0 {
                    0.0
                } else {
                    (i as f32 - 5.0) / 17.0
                }
            })
            .collect();
        let l2_weights: Vec<f32> = (0..l1 * l2_total)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 31.0)
            .collect();
        let l2_biases: Vec<f32> = (0..l2_off + l2)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) / 23.0)
            .collect();
        let out_weights: Vec<f32> = (0..l2)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 19.0)
            .collect();
        let out_bias = 0.125f32;

        let actual = simd_forward_l2(
            &l1_out,
            &l2_weights,
            &l2_biases,
            l2,
            l2_total,
            l2_off,
            &out_weights,
            out_bias,
        );

        let mut h2 = vec![0.0f32; l2];
        h2[..l2].copy_from_slice(&l2_biases[l2_off..l2_off + l2]);
        for (i, &l1_value) in l1_out.iter().enumerate() {
            let w_base = i * l2_total + l2_off;
            for k in 0..l2 {
                h2[k] += l1_value * l2_weights[w_base + k];
            }
        }
        let mut expected = out_bias;
        for k in 0..l2 {
            h2[k] = h2[k].clamp(0.0, 1.0);
            h2[k] *= h2[k];
            expected += h2[k] * out_weights[k];
        }

        assert!((actual - expected).abs() < 0.000001);
    }
}
