use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::Simd;

const QA: i32 = 255;
const I16_LANES: usize = 16;
const I32_LANES: usize = 8;
const F32_LANES: usize = 8;

type I16x = Simd<i16, I16_LANES>;
type I32x = Simd<i32, I32_LANES>;
type F32x = Simd<f32, F32_LANES>;

#[inline(always)]
fn load_i16(slice: &[i16], offset: usize) -> I16x {
    I16x::from_slice(&slice[offset..offset + I16_LANES])
}

#[inline(always)]
fn store_i16(slice: &mut [i16], offset: usize, value: I16x) {
    value.copy_to_slice(&mut slice[offset..offset + I16_LANES]);
}

#[inline(always)]
fn load_i32(slice: &[i32], offset: usize) -> I32x {
    I32x::from_slice(&slice[offset..offset + I32_LANES])
}

#[inline(always)]
fn store_i32(slice: &mut [i32], offset: usize, value: I32x) {
    value.copy_to_slice(&mut slice[offset..offset + I32_LANES]);
}

#[inline(always)]
fn load_i16_i32(slice: &[i16], offset: usize) -> I32x {
    I32x::from_array([
        slice[offset] as i32,
        slice[offset + 1] as i32,
        slice[offset + 2] as i32,
        slice[offset + 3] as i32,
        slice[offset + 4] as i32,
        slice[offset + 5] as i32,
        slice[offset + 6] as i32,
        slice[offset + 7] as i32,
    ])
}

#[inline(always)]
fn load_f32(slice: &[f32], offset: usize) -> F32x {
    F32x::from_slice(&slice[offset..offset + F32_LANES])
}

#[inline(always)]
fn store_f32(slice: &mut [f32], offset: usize, value: F32x) {
    value.copy_to_slice(&mut slice[offset..offset + F32_LANES]);
}

#[inline(always)]
fn madd_i16(a: I16x, b: I16x) -> I32x {
    let a = a.to_array();
    let b = b.to_array();
    I32x::from_array([
        a[0] as i32 * b[0] as i32 + a[1] as i32 * b[1] as i32,
        a[2] as i32 * b[2] as i32 + a[3] as i32 * b[3] as i32,
        a[4] as i32 * b[4] as i32 + a[5] as i32 * b[5] as i32,
        a[6] as i32 * b[6] as i32 + a[7] as i32 * b[7] as i32,
        a[8] as i32 * b[8] as i32 + a[9] as i32 * b[9] as i32,
        a[10] as i32 * b[10] as i32 + a[11] as i32 * b[11] as i32,
        a[12] as i32 * b[12] as i32 + a[13] as i32 * b[13] as i32,
        a[14] as i32 * b[14] as i32 + a[15] as i32 * b[15] as i32,
    ])
}

#[inline(always)]
fn clamp_crelu_i16(value: I16x) -> I16x {
    value.simd_clamp(I16x::splat(0), I16x::splat(QA as i16))
}

#[inline(always)]
pub fn simd_add_row(acc: &mut [i16], row: &[i16]) {
    let len = acc.len();
    debug_assert_eq!(len, row.len());

    let mut i = 0;
    while i + I16_LANES <= len {
        let sum = load_i16(acc, i) + load_i16(row, i);
        store_i16(acc, i, sum);
        i += I16_LANES;
    }
    while i < len {
        acc[i] += row[i];
        i += 1;
    }
}

#[inline(always)]
pub fn simd_sub_row(acc: &mut [i16], row: &[i16]) {
    let len = acc.len();
    debug_assert_eq!(len, row.len());

    let mut i = 0;
    while i + I16_LANES <= len {
        let sum = load_i16(acc, i) - load_i16(row, i);
        store_i16(acc, i, sum);
        i += I16_LANES;
    }
    while i < len {
        acc[i] -= row[i];
        i += 1;
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

    let mut i = 0;
    while i + I16_LANES <= h {
        let sc = clamp_crelu_i16(load_i16(stm, i));
        let nc = clamp_crelu_i16(load_i16(ntm, i));
        let sw = load_i16(out_w, i);
        let nw = load_i16(out_w, h + i);

        sum += madd_i16(sc, sw).reduce_sum() as i64;
        sum += madd_i16(nc, nw).reduce_sum() as i64;

        i += I16_LANES;
    }

    while i < h {
        let v = (stm[i] as i32).clamp(0, QA) as i64;
        sum += v * out_w[i] as i64;
        let v = (ntm[i] as i32).clamp(0, QA) as i64;
        sum += v * out_w[h + i] as i64;
        i += 1;
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

    let mut i = 0;
    while i + I32_LANES <= l1 {
        let mut s_acc = I32x::splat(0);
        let mut n_acc = I32x::splat(0);

        for j in 0..pw {
            let sw = load_i16_i32(l1_weights, j * l1_total + l1_off + i);
            let nw = load_i16_i32(l1_weights, (pw + j) * l1_total + l1_off + i);

            s_acc += I32x::splat(sp[j] as i32) * sw;
            n_acc += I32x::splat(np[j] as i32) * nw;
        }

        let existing = load_i32(out, i);
        store_i32(out, i, existing + s_acc + n_acc);
        i += I32_LANES;
    }

    while i < l1 {
        let gi = l1_off + i;
        let mut s_sum = 0i32;
        let mut n_sum = 0i32;
        for j in 0..pw {
            s_sum += sp[j] as i32 * l1_weights[j * l1_total + gi] as i32;
            n_sum += np[j] as i32 * l1_weights[(pw + j) * l1_total + gi] as i32;
        }
        out[i] += s_sum + n_sum;
        i += 1;
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

        let mut k = 0;
        while k + F32_LANES <= l2 {
            let w = load_f32(l2_weights, w_base + k);
            let h = load_f32(&h2, k);
            store_f32(&mut h2, k, l1v * w + h);
            k += F32_LANES;
        }
        while k < l2 {
            h2[k] += l1_value * l2_weights[w_base + k];
            k += 1;
        }
    }

    let zero = F32x::splat(0.0);
    let one = F32x::splat(1.0);
    let mut k = 0;
    while k + F32_LANES <= l2 {
        let h = load_f32(&h2, k).simd_clamp(zero, one);
        store_f32(&mut h2, k, h * h);
        k += F32_LANES;
    }
    while k < l2 {
        h2[k] = h2[k].clamp(0.0, 1.0);
        h2[k] *= h2[k];
        k += 1;
    }

    let mut of = F32x::splat(0.0);
    k = 0;
    while k + F32_LANES <= l2 {
        let h = load_f32(&h2, k);
        let w = load_f32(out_weights, k);
        of += h * w;
        k += F32_LANES;
    }
    let mut result = out_bias + of.reduce_sum();
    while k < l2 {
        result += h2[k] * out_weights[k];
        k += 1;
    }

    result
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
