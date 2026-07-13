#![allow(clippy::missing_safety_doc)]
const QA: i32 = 255;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod platform {
    use super::QA;
    use std::arch::x86_64::*;

    pub const I16_LANES: usize = 16;
    pub const I32_LANES: usize = 8;
    pub const F32_LANES: usize = 8;

    #[inline(always)]
    pub unsafe fn load_i16(ptr: *const i16) -> __m256i {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    #[inline(always)]
    pub unsafe fn store_i16(ptr: *mut i16, v: __m256i) {
        _mm256_storeu_si256(ptr as *mut __m256i, v);
    }

    #[inline(always)]
    pub unsafe fn add_i16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn sub_i16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn splat_i16(v: i16) -> __m256i {
        _mm256_set1_epi16(v)
    }

    #[inline(always)]
    pub unsafe fn min_i16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_min_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn max_i16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_max_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn clamp_crelu_i16(v: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        _mm256_max_epi16(_mm256_min_epi16(v, qa), zero)
    }

    #[inline(always)]
    pub unsafe fn madd_i16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_madd_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn load_i32(ptr: *const i32) -> __m256i {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    #[inline(always)]
    pub unsafe fn store_i32(ptr: *mut i32, v: __m256i) {
        _mm256_storeu_si256(ptr as *mut __m256i, v);
    }

    #[inline(always)]
    pub unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn splat_i32(v: i32) -> __m256i {
        _mm256_set1_epi32(v)
    }

    #[inline(always)]
    pub unsafe fn zero_i32() -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    pub unsafe fn convert_u8_i16_low(a: __m128i) -> __m256i {
        _mm256_cvtepu8_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn convert_u8_i16_high(a: __m128i) -> __m256i {
        _mm256_cvtepu8_epi16(_mm_srli_si128::<8>(a))
    }

    #[inline(always)]
    pub unsafe fn load_u8_widen_i16(ptr: *const u8) -> (__m256i, __m256i) {
        let bytes = _mm_loadu_si128(ptr as *const __m128i);
        let lo = _mm256_cvtepu8_epi16(bytes);
        let hi = _mm256_cvtepu8_epi16(_mm_srli_si128::<8>(bytes));
        (lo, hi)
    }

    #[inline(always)]
    pub unsafe fn zero_f32() -> __m256 {
        _mm256_setzero_ps()
    }

    #[inline(always)]
    pub unsafe fn splat_f32(v: f32) -> __m256 {
        _mm256_set1_ps(v)
    }

    #[inline(always)]
    pub unsafe fn load_f32(ptr: *const f32) -> __m256 {
        _mm256_loadu_ps(ptr)
    }

    #[inline(always)]
    pub unsafe fn store_f32(ptr: *mut f32, v: __m256) {
        _mm256_storeu_ps(ptr, v);
    }

    #[inline(always)]
    pub unsafe fn add_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul_f32(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul_add_f32(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }

    #[inline(always)]
    pub unsafe fn clamp_f32(x: __m256, min: __m256, max: __m256) -> __m256 {
        _mm256_max_ps(_mm256_min_ps(x, max), min)
    }

    #[inline(always)]
    pub unsafe fn convert_i32_f32(a: __m256i) -> __m256 {
        _mm256_cvtepi32_ps(a)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_f32(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps::<1>(v);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps::<1>(sum64, sum64);
        let sum32 = _mm_add_ss(sum64, hi32);
        _mm_cvtss_f32(sum32)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_i32(v: __m256i) -> i64 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let sum = _mm_add_epi32(lo, hi);
        let hi64 = _mm_shuffle_epi32::<0b11101110>(sum);
        let sum2 = _mm_add_epi32(sum, hi64);
        let hi32 = _mm_shuffle_epi32::<0b00000001>(sum2);
        let sum3 = _mm_add_epi32(sum2, hi32);
        _mm_cvtsi128_si32(sum3) as i64
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
mod platform {
    use super::QA;

    pub const I16_LANES: usize = 1;
    pub const I32_LANES: usize = 1;
    pub const F32_LANES: usize = 1;

    #[inline(always)]
    pub unsafe fn load_i16(ptr: *const i16) -> i16 {
        *ptr
    }

    #[inline(always)]
    pub unsafe fn store_i16(ptr: *mut i16, v: i16) {
        *ptr = v;
    }

    #[inline(always)]
    pub unsafe fn add_i16(a: i16, b: i16) -> i16 {
        a + b
    }

    #[inline(always)]
    pub unsafe fn sub_i16(a: i16, b: i16) -> i16 {
        a - b
    }

    #[inline(always)]
    pub unsafe fn splat_i16(v: i16) -> i16 {
        v
    }

    #[inline(always)]
    pub unsafe fn min_i16(a: i16, b: i16) -> i16 {
        a.min(b)
    }

    #[inline(always)]
    pub unsafe fn max_i16(a: i16, b: i16) -> i16 {
        a.max(b)
    }

    #[inline(always)]
    pub unsafe fn clamp_crelu_i16(v: i16) -> i16 {
        v.max(0).min(QA as i16)
    }

    #[inline(always)]
    pub unsafe fn madd_i16(a: i16, b: i16) -> i32 {
        (a as i32) * (b as i32)
    }

    #[inline(always)]
    pub unsafe fn load_i32(ptr: *const i32) -> i32 {
        *ptr
    }

    #[inline(always)]
    pub unsafe fn store_i32(ptr: *mut i32, v: i32) {
        *ptr = v;
    }

    #[inline(always)]
    pub unsafe fn add_i32(a: i32, b: i32) -> i32 {
        a + b
    }

    #[inline(always)]
    pub unsafe fn splat_i32(v: i32) -> i32 {
        v
    }

    #[inline(always)]
    pub unsafe fn zero_i32() -> i32 {
        0
    }

    #[inline(always)]
    pub unsafe fn convert_u8_i16_low(a: u8) -> i16 {
        a as i16
    }

    #[inline(always)]
    pub unsafe fn convert_u8_i16_high(a: u8) -> i16 {
        a as i16
    }

    #[inline(always)]
    pub unsafe fn load_u8_widen_i16(ptr: *const u8) -> (i16, i16) {
        (*ptr as i16, *ptr.add(1) as i16)
    }

    #[inline(always)]
    pub unsafe fn zero_f32() -> f32 {
        0.0
    }

    #[inline(always)]
    pub unsafe fn splat_f32(v: f32) -> f32 {
        v
    }

    #[inline(always)]
    pub unsafe fn load_f32(ptr: *const f32) -> f32 {
        *ptr
    }

    #[inline(always)]
    pub unsafe fn store_f32(ptr: *mut f32, v: f32) {
        *ptr = v;
    }

    #[inline(always)]
    pub unsafe fn add_f32(a: f32, b: f32) -> f32 {
        a + b
    }

    #[inline(always)]
    pub unsafe fn mul_f32(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    pub unsafe fn mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    #[inline(always)]
    pub unsafe fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
        x.max(min).min(max)
    }

    #[inline(always)]
    pub unsafe fn convert_i32_f32(a: i32) -> f32 {
        a as f32
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_f32(v: f32) -> f32 {
        v
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_i32(v: i32) -> i64 {
        v as i64
    }
}

pub use platform::*;

#[inline(always)]
pub unsafe fn simd_add_row(acc: &mut [i16], row: &[i16]) {
    let len = acc.len();
    debug_assert_eq!(len, row.len());
    let lanes = I16_LANES;
    let mut i = 0;
    while i + lanes <= len {
        let a = load_i16(acc.as_ptr().add(i));
        let r = load_i16(row.as_ptr().add(i));
        store_i16(acc.as_mut_ptr().add(i), add_i16(a, r));
        i += lanes;
    }
    while i < len {
        *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn simd_sub_row(acc: &mut [i16], row: &[i16]) {
    let len = acc.len();
    debug_assert_eq!(len, row.len());
    let lanes = I16_LANES;
    let mut i = 0;
    while i + lanes <= len {
        let a = load_i16(acc.as_ptr().add(i));
        let r = load_i16(row.as_ptr().add(i));
        store_i16(acc.as_mut_ptr().add(i), sub_i16(a, r));
        i += lanes;
    }
    while i < len {
        *acc.get_unchecked_mut(i) -= *row.get_unchecked(i);
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn simd_forward_base_crelu(
    stm: &[i16],
    ntm: &[i16],
    out_w: &[i16],
    h: usize,
    use_screlu: bool,
) -> i64 {
    let lanes = I16_LANES;
    let mut sum: i64 = 0;
    let mut i = 0;

    if use_screlu {
        for j in 0..h {
            let v = stm[j].max(0).min(QA as i16) as i64;
            sum += v * v * out_w[j] as i64;
        }
        for j in 0..h {
            let v = ntm[j].max(0).min(QA as i16) as i64;
            sum += v * v * out_w[h + j] as i64;
        }
        sum
    } else {
        while i + lanes <= h {
            let sv = load_i16(stm.as_ptr().add(i));
            let nv = load_i16(ntm.as_ptr().add(i));
            let sw = load_i16(out_w.as_ptr().add(i));
            let nw = load_i16(out_w.as_ptr().add(h + i));

            let sc = clamp_crelu_i16(sv);
            let nc = clamp_crelu_i16(nv);

            let s_partial = madd_i16(sc, sw);
            let n_partial = madd_i16(nc, nw);

            sum += horizontal_sum_i32(s_partial);
            sum += horizontal_sum_i32(n_partial);

            i += lanes;
        }

        while i < h {
            let v = (*stm.get_unchecked(i) as i32).clamp(0, QA) as i64;
            sum += v * *out_w.get_unchecked(i) as i64;
            let v = (*ntm.get_unchecked(i) as i32).clamp(0, QA) as i64;
            sum += v * *out_w.get_unchecked(h + i) as i64;
            i += 1;
        }
        sum
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn simd_l1_matmul(
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
    let lanes = I16_LANES;

    for i in 0..l1 {
        *out.get_unchecked_mut(i) = *l1_biases.get_unchecked(l1_off + i) as i32 * pw_scale;
    }

    let mut i = 0;
    while i + lanes <= l1 {
        let mut s_acc = zero_i32();
        let mut n_acc = zero_i32();

        for j in 0..pw {
            let sp_j = *sp.get_unchecked(j) as i32;
            let np_j = *np.get_unchecked(j) as i32;

            let sw = load_i16(l1_weights.as_ptr().add(j * l1_total + l1_off + i));
            let nw = load_i16(l1_weights.as_ptr().add((pw + j) * l1_total + l1_off + i));

            let sp_splat = splat_i16(sp_j as i16);
            let np_splat = splat_i16(np_j as i16);

            let s_prod = madd_i16(sp_splat, sw);
            let n_prod = madd_i16(np_splat, nw);

            s_acc = add_i32(s_acc, s_prod);
            n_acc = add_i32(n_acc, n_prod);
        }

        let existing = load_i32(out.as_ptr().add(i));
        store_i32(
            out.as_mut_ptr().add(i),
            add_i32(existing, add_i32(s_acc, n_acc)),
        );

        i += lanes;
    }

    while i < l1 {
        let gi = l1_off + i;
        let mut s_sum = 0i32;
        let mut n_sum = 0i32;
        for j in 0..pw {
            s_sum +=
                *sp.get_unchecked(j) as i32 * *l1_weights.get_unchecked(j * l1_total + gi) as i32;
            n_sum += *np.get_unchecked(j) as i32
                * *l1_weights.get_unchecked((pw + j) * l1_total + gi) as i32;
        }
        *out.get_unchecked_mut(i) += s_sum + n_sum;
        i += 1;
    }
}

#[inline(always)]
pub unsafe fn simd_screlu_activation(hidden: &[i32], pw_scale: i32, qa_l1: i32, out: &mut [f32]) {
    let len = hidden.len();
    debug_assert!(len <= out.len());
    let lanes = F32_LANES;
    let mut i = 0;

    let qf = qa_l1 as f32;
    let qsq = qf * qf;
    let inv_qsq = splat_f32(1.0 / qsq);
    let zero = splat_f32(0.0);
    let qa_f = splat_f32(qf);

    while i + lanes <= len {
        let h = load_i32(hidden.as_ptr().add(i));
        let h_f = convert_i32_f32(h);

        let v = mul_f32(h_f, splat_f32(1.0 / pw_scale as f32));
        let clamped = clamp_f32(v, zero, qa_f);
        let result = mul_f32(mul_f32(clamped, clamped), inv_qsq);

        store_f32(out.as_mut_ptr().add(i), result);
        i += lanes;
    }

    while i < len {
        let v = (*hidden.get_unchecked(i) / pw_scale).clamp(0, qa_l1);
        *out.get_unchecked_mut(i) = (v * v) as f32 / qsq;
        i += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn simd_forward_l2(
    l1_out: &[f32],
    l2_weights: &[f32],
    l2_biases: &[f32],
    l2: usize,
    l2_total: usize,
    l2_off: usize,
    out_weights: &[f32],
    out_bias: f32,
) -> f32 {
    let lanes = F32_LANES;

    let mut h2 = vec![0.0f32; l2];
    for k in 0..l2 {
        *h2.get_unchecked_mut(k) = *l2_biases.get_unchecked(l2_off + k);
    }

    for (i, &l1_value) in l1_out.iter().enumerate() {
        if l1_value == 0.0 {
            continue;
        }
        let l1v = splat_f32(l1_value);
        let w_base = l2_weights.as_ptr().add(i * l2_total + l2_off);

        let mut k = 0;
        while k + lanes <= l2 {
            let w = load_f32(w_base.add(k));
            let h = load_f32(h2.as_ptr().add(k));
            store_f32(h2.as_mut_ptr().add(k), mul_add_f32(l1v, w, h));
            k += lanes;
        }
        while k < l2 {
            *h2.get_unchecked_mut(k) += l1_value * *w_base.add(k);
            k += 1;
        }
    }

    let zero = splat_f32(0.0);
    let one = splat_f32(1.0);
    let mut k = 0;
    while k + lanes <= l2 {
        let h = load_f32(h2.as_ptr().add(k));
        let clamped = clamp_f32(h, zero, one);
        let sq = mul_f32(clamped, clamped);
        store_f32(h2.as_mut_ptr().add(k), sq);
        k += lanes;
    }
    while k < l2 {
        let v = h2.get_unchecked_mut(k);
        *v = v.clamp(0.0, 1.0);
        *v *= *v;
        k += 1;
    }

    let mut of = splat_f32(out_bias);
    k = 0;
    while k + lanes <= l2 {
        let h = load_f32(h2.as_ptr().add(k));
        let w = load_f32(out_weights.as_ptr().add(k));
        of = mul_add_f32(h, w, of);
        k += lanes;
    }
    let mut result = horizontal_sum_f32(of);
    while k < l2 {
        result += *h2.get_unchecked(k) * *out_weights.get_unchecked(k);
        k += 1;
    }

    result
}
