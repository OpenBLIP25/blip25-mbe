//! Fixed-point amplitude shape quantizer (`b3..b8`) in the reference's own
//! integer arithmetic.
//!
//! # Why integer arithmetic is load-bearing here
//!
//! The chain is recursive: the previous frame's *reconstructed* log-amplitude
//! vector is resampled onto the current harmonic grid and subtracted before the
//! DCT, and the current frame's reconstruction becomes the next frame's
//! predictor memory. A real-valued model of the same algebra, accurate to ~0.1%
//! per stage, does not track the reference: the residual crosses VQ decision
//! boundaries, a flipped index perturbs the reconstruction by a whole codebook
//! cell, and that perturbation re-enters the predictor. Measured against live
//! reference intermediates, a float model of this exact algebra — fed the
//! reference's own amplitude vector and gain, seeded with the reference's own
//! state — free-runs to 81-90% index agreement with 7% RMS state error, while a
//! model in the reference's integer arithmetic holds 100%. There is no
//! approximate version of this stage.
//!
//! # Stages
//!
//! | stage | what it does |
//! |---|---|
//! | [`predict`] | resample the previous frame's `M_l` onto this frame's `L`, mean-remove, scale by `0x5333` (0.65) |
//! | [`forward`] | residual `M_l - pred`, per-block DCT (`kmax = min(6, J)`), `±sqrt(2)` pair rotation, 8-point DCT |
//! | [`nn_search`] | unweighted Euclidean nearest neighbour, ascending scan, first-wins ties |
//! | [`reconstruct`] | inverse 8-point DCT, pair recombination, per-block inverse DCT, gain-anchored offset |
//! | [`add_pred`] | add the prediction back, clamp to `[-0x7800, 0x77ff]`, pad to 56 |
//!
//! Every intermediate is an `i16` in the Q11 log2 domain, matching
//! [`super::band_decompress`]'s output word format.

use crate::tables::AMBE_BLOCK_LENGTHS;

use super::band_decompress::shift_scale;

/// Predictor coefficient, Q15: `0x5333 / 32768 = 0.65`.
const RHO: i32 = 0x5333;

/// `sqrt(2)` in Q15 (`0xb504`), used by the pair rotation both ways.
const SQRT2_Q15: i32 = 0xb504;

/// Harmonics the predictor memory holds.
const PREV_LEN: usize = 56;

/// Quarter-turn resolution of the shared DCT kernel: 512 points over a full
/// turn, addressed by a Q7 phase.
const COS_LEN: usize = 512;

/// Kernel tables: a 512-point Q15 cosine and the per-length DCT normalizer
/// `trunc(32768 / n)`, both exactly as the reference stores them.
fn kernels() -> &'static ([i16; COS_LEN], [i16; 57]) {
    static K: std::sync::OnceLock<([i16; COS_LEN], [i16; 57])> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let mut cos = [0i16; COS_LEN];
        for (i, slot) in cos.iter_mut().enumerate() {
            let theta = core::f64::consts::TAU * (i as f64) / (COS_LEN as f64);
            let v = (32768.0 * theta.cos() + 0.5).floor();
            *slot = v.clamp(-32767.0, 32767.0) as i16;
        }
        let mut scale = [0i16; 57];
        for (n, slot) in scale.iter_mut().enumerate().skip(1) {
            *slot = (32768usize / n).min(32767) as i16;
        }
        (cos, scale)
    })
}

/// Signed fractional divide with a built-in halving: `sign(a/b) * (|a| / |b| >> 1)`,
/// zero when `|a| < |b|`.
fn frac_div(a: i32, b: i16) -> i32 {
    let neg = ((b as i32) ^ a) < 0;
    let num = if a < 0 {
        if a == i32::MIN {
            i32::MAX
        } else {
            -a
        }
    } else {
        a
    };
    let den: i32 = if b < 0 {
        if b == i16::MIN {
            0x7fff
        } else {
            -(b as i32)
        }
    } else {
        b as i32
    };
    if den == 0 {
        return 0;
    }
    let q = if num < den { 0 } else { (num / den) >> 1 };
    if neg {
        -q
    } else {
        q
    }
}

/// Redundant-sign-bit count of a 40-bit window of a 64-bit accumulator, biased
/// by -9 so the result is directly the block-float exponent.
fn norm40(v: i64) -> i16 {
    let (mut lo, mut hi) = (v as u32, (v >> 32) as u32);
    if lo == 0 && hi == 0 {
        return 0;
    }
    if hi & 0x8000_0000 != 0 {
        lo = !lo;
        hi = !hi;
    }
    let mut mask_lo: u32 = 0;
    let mut mask_hi: u32 = 0x80;
    let mut i: i32 = 0;
    while (mask_lo & lo) == 0 && (mask_hi & hi) == 0 {
        mask_lo = (mask_lo >> 1) | (mask_hi << 31);
        i += 1;
        mask_hi >>= 1;
        if i >= 0x28 {
            break;
        }
    }
    (i - 9) as i16
}

/// Saturating i16 narrowing at the reference's `0x7fff` / `-0x7fff` limits.
/// The negative limit is `-0x7fff`, not `i16::MIN`.
fn sat15(v: i32) -> i16 {
    if v >= 0x8000 {
        0x7fff
    } else if v < -0x7fff {
        -0x7fff
    } else {
        v as i16
    }
}

/// `(a + b)` in Q15 with saturation, narrowed back to Q11 — the shared
/// "add two log-domain words" primitive.
fn add_q11(a: i16, b: i16) -> i16 {
    let sum = ((a as i32) << 15).wrapping_add((b as i32) << 15);
    (shift_scale(sum as u32, 1) >> 16) as u16 as i16
}

/// Forward DCT-II over `x[..n]`, `1/n`-normalized, `min(kmax, n)` bins out.
///
/// Block-float: the 64-bit product accumulator is normalized to a 16-bit
/// mantissa before the `1/n` scaling and denormalized after, so the result
/// carries 16 significant bits regardless of the input's magnitude.
fn forward_dct(x: &[i16], n: usize, kmax: usize) -> Vec<i16> {
    let (cos, scale_tab) = kernels();
    let scale = scale_tab[n] as i32;
    let km = kmax.min(n);
    let mut out = Vec::with_capacity(km);
    let mut phase_step: i32 = 0;
    for _ in 0..km {
        let mut phase = (phase_step >> 1).wrapping_add(0x40);
        let inc = phase_step as i16 as i32;
        let mut acc: i64 = 0;
        for &xj in &x[..n] {
            let idx = ((phase >> 7) & 0x1ff) as usize;
            let term = (xj as i32).wrapping_mul(cos[idx] as i32).wrapping_mul(2);
            acc = acc.wrapping_add(term as i64);
            phase = phase.wrapping_add(inc);
        }
        let sh = norm40(acc);
        let normed = if sh >= 0 {
            ((acc as u64) << (sh as u32 & 0x3f)) as i64
        } else {
            acc >> ((-sh) as u32 & 0x3f)
        };
        let mant = ((normed as u32) >> 16) as u16 as i16;
        let mut val = (mant as i32).wrapping_mul(scale).wrapping_mul(2);
        if sh < 1 {
            val = ((val as u32) << ((-sh) as u32 & 0x1f)) as i32;
        } else if -sh < -0x1e {
            val >>= 31;
        } else {
            val >>= sh as u32 & 0x1f;
        }
        out.push((val.wrapping_add(0x8000) >> 16) as i16);
        phase_step = phase_step.wrapping_add(scale);
    }
    out
}

/// Inverse DCT-II over `x[..min(kmax, n)]` producing `n` samples. `alpha_0 = 1`,
/// `alpha_k = 2` — the doubling is a wrapping `i16` shift, not a saturating one.
fn inverse_dct(x: &[i16], n: usize, kmax: usize) -> Vec<i16> {
    let (cos, scale_tab) = kernels();
    let scale = scale_tab[n];
    let km = kmax.min(n);
    let mut coef = Vec::with_capacity(km);
    coef.push(x[0]);
    for &v in &x[1..km] {
        coef.push(v.wrapping_mul(2));
    }
    let mut phase_step: i16 = scale >> 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut acc: i32 = 0;
        let mut phase: i32 = 0x40;
        for &c in &coef {
            let idx = ((phase >> 7) & 0x1ff) as usize;
            phase = phase.wrapping_add(phase_step as i32);
            acc = acc.wrapping_add((cos[idx] as i32).wrapping_mul(c as i32).wrapping_mul(2));
        }
        phase_step = phase_step.wrapping_add(scale);
        out.push((acc.wrapping_add(0x8000) >> 16) as i16);
    }
    out
}

/// Cross-frame predictor memory: the previous frame's reconstructed `M_l`,
/// padded to 56 harmonics, plus the harmonic count it was reconstructed at.
///
/// The reference seeds this with an all-zero vector at `L = 15`; the loop is
/// contracting, so a run started from that seed re-joins the reference's
/// trajectory immediately.
#[derive(Clone, Debug)]
pub(crate) struct AmpShapeState {
    prev: [i16; PREV_LEN],
    prev_l: i16,
}

impl AmpShapeState {
    pub(crate) fn new() -> Self {
        Self {
            prev: [0; PREV_LEN],
            prev_l: 15,
        }
    }
}

impl Default for AmpShapeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Resample the predictor memory onto `l` harmonics and return the Q11 words
/// plus the full-precision sum the mean-removal needs.
fn resample(prev: &[i16; PREV_LEN], l: usize, prev_l: i16) -> (Vec<i16>, i32) {
    let base = frac_div(0x80000, l as i16) as i16 as i32;
    let span = (prev_l as i32).wrapping_shl(9) as i16 as i32;
    let step = base.wrapping_mul(span).wrapping_mul(2) >> 12;
    let mut out = Vec::with_capacity(l);
    let mut sum: i32 = 0;
    let mut phase = step;
    for _ in 0..l {
        // `step` is positive for every coded `(l, prev_l)`, so `phase` never
        // goes negative; the lower clamp is defensive, not a modelled branch.
        let mut k = ((phase >> 16) as i16).max(0);
        let val = if k == 0 {
            (prev[0] as i32) << 16
        } else {
            if k >= 0x37 {
                k = 0x37;
            }
            let k = k as usize;
            let frac = (phase >> 1) & 0x7fff;
            let lo = frac.wrapping_mul(prev[k - 1] as i32);
            let base = if (lo as u32) & 0x7fff_ffff == 0x4000_0000 {
                i32::MAX
            } else {
                lo.wrapping_mul(-2)
            };
            let hi = (prev[k] as i32)
                .wrapping_mul(frac)
                .wrapping_add((prev[k - 1] as i32).wrapping_mul(0x8000))
                .wrapping_mul(2);
            base.wrapping_add(hi)
        };
        phase = phase.wrapping_add(step);
        out.push((val >> 16) as i16);
        sum = sum.wrapping_add(val >> 6);
    }
    (out, sum)
}

/// The per-harmonic prediction `0.65 * (resampled - mean(resampled))`.
fn predict(prev: &[i16; PREV_LEN], l: usize, prev_l: i16) -> Vec<i16> {
    let (x, sum) = resample(prev, l, prev_l);
    let mean = frac_div(sum, ((l as i32) << 9) as i16) as i16 as i32;
    let scaled = mean.wrapping_mul(RHO);
    let bias = if (scaled as u32) & 0x7fff_ffff == 0x4000_0000 {
        i32::MAX
    } else {
        scaled.wrapping_mul(-2)
    };
    x.iter()
        .map(|&xi| {
            let t = ((xi as i32).wrapping_mul(RHO).wrapping_mul(2) >> 1).wrapping_add(bias >> 1);
            (shift_scale(t as u32, 1) >> 16) as u16 as i16
        })
        .collect()
}

/// Residual, per-block DCT and 8-point DCT. Returns the 8-word PRBA vector and
/// the per-block coefficient array (blocks laid out end to end, bins >= 6
/// zeroed).
fn forward(amps: &[i16], pred: &[i16], blocks: [u8; 4]) -> ([i16; 8], Vec<i16>) {
    let l: usize = blocks.iter().map(|&j| j as usize).sum();
    let mut resid: Vec<i16> = (0..l)
        .map(|i| sat15((amps[i] as i32) - (pred[i] as i32)))
        .collect();
    let mut g = [0i16; 8];
    let mut off = 0usize;
    for (bi, &jb) in blocks.iter().enumerate() {
        let j = jb as usize;
        let c = forward_dct(&resid[off..off + j], j, 6);
        resid[off..off + c.len()].copy_from_slice(&c);
        g[2 * bi] = resid[off];
        g[2 * bi + 1] = resid[off + 1];
        for slot in resid[off..off + j].iter_mut().skip(6) {
            *slot = 0;
        }
        off += j;
    }
    for i in 0..4 {
        let a = ((g[2 * i] as i32) << 16) >> 1;
        let b = g[2 * i + 1] as i32;
        let sum = sat15(b.wrapping_mul(SQRT2_Q15).wrapping_add(a) >> 15);
        let dif = sat15(a.wrapping_add(b.wrapping_mul(-SQRT2_Q15)) >> 15);
        g[2 * i] = sum;
        g[2 * i + 1] = dif;
    }
    let g8 = forward_dct(&g, 8, 8);
    let mut out = [0i16; 8];
    out.copy_from_slice(&g8);
    (out, resid)
}

/// Unweighted Euclidean nearest neighbour over `used` dimensions, ascending
/// scan, first-wins ties, with the reference's saturating accumulation.
fn nn_search(book: &[i16], rows: usize, stride: usize, used: usize, target: &[i16]) -> usize {
    let mut best = i32::MAX;
    let mut best_i = 0usize;
    for i in 0..rows {
        let row = &book[i * stride..i * stride + stride];
        let mut acc: i32 = 0;
        for d in 0..used {
            let diff = ((target[d] as i32) << 16).saturating_sub((row[d] as i32) << 16);
            let dv = (diff >> 16) as i16;
            let sq = if dv == i16::MIN {
                i32::MAX
            } else {
                (dv as i32) * (dv as i32) * 2
            };
            acc = acc.saturating_add(sq);
        }
        if acc < best {
            best = acc;
            best_i = i;
        }
    }
    best_i
}

/// Choose `b3..b8` and write the chosen codebook rows back over the quantized
/// coefficients, exactly as the reference's quantize-and-copy-back does.
fn quantize(g: &[i16; 8], coef: &mut [i16], blocks: [u8; 4]) -> ([u16; 6], [i16; 8]) {
    let prba24 = blip25_codebooks::prba24();
    let prba58 = blip25_codebooks::prba58();
    let b3 = nn_search(prba24.raw, 512, 3, 3, &g[1..4]);
    let b4 = nn_search(prba58.raw, 128, 4, 4, &g[4..8]);
    let mut deq = [0i16; 8];
    deq[1..4].copy_from_slice(prba24.raw_row(b3));
    deq[4..8].copy_from_slice(prba58.raw_row(b4));

    let hoc = [
        blip25_codebooks::hoc_b5(),
        blip25_codebooks::hoc_b6(),
        blip25_codebooks::hoc_b7(),
        blip25_codebooks::hoc_b8(),
    ];
    let mut idx = [b3 as u16, b4 as u16, 0, 0, 0, 0];
    let mut off = 0usize;
    for (i, &jb) in blocks.iter().enumerate() {
        let j = jb as usize;
        let used = j.saturating_sub(2).min(4);
        if used >= 1 {
            let book = &hoc[i];
            let sel = nn_search(book.raw, book.rows, 4, used, &coef[off + 2..off + 2 + used]);
            idx[2 + i] = sel as u16;
            coef[off + 2..off + 2 + used].copy_from_slice(&book.raw_row(sel)[..used]);
        }
        off += j;
    }
    (idx, deq)
}

/// Rebuild `M_l` from the chosen indices: inverse 8-point DCT, pair
/// recombination, per-block inverse DCT, then the gain-anchored DC offset.
fn reconstruct(deq: &[i16; 8], coef: &[i16], blocks: [u8; 4], l: usize, gain: i16) -> Vec<i16> {
    let mut pair = inverse_dct(deq, 8, 8);
    for i in 0..4 {
        let a = ((pair[2 * i] as i32) << 16) >> 1;
        let b = ((pair[2 * i + 1] as i32) << 16) >> 1;
        pair[2 * i] = (b.wrapping_add(a) >> 16) as i16;
        let d = ((a.wrapping_sub(b)) >> 16) as i16;
        pair[2 * i + 1] = ((d as i32).wrapping_mul(SQRT2_Q15) >> 16) as i16;
    }
    let mut blk = coef.to_vec();
    let mut shaped: Vec<i16> = Vec::with_capacity(l);
    let mut off = 0usize;
    for (i, &jb) in blocks.iter().enumerate() {
        let j = jb as usize;
        blk[off] = pair[2 * i];
        blk[off + 1] = pair[2 * i + 1];
        shaped.extend_from_slice(&inverse_dct(&blk[off..off + j], j, 6));
        off += j;
    }
    let log2l = super::loudness_fixed::log2_count_q10(l);
    let anchor = ((((gain as i32) << 16).wrapping_sub((log2l as i32) << 16) >> 16) as i16) as i32;
    let mut acc = anchor.wrapping_mul(l as i32).wrapping_mul(2);
    for (i, &jb) in blocks.iter().enumerate() {
        acc = acc.wrapping_add(
            (jb as i32)
                .wrapping_mul(pair[2 * i] as i32)
                .wrapping_mul(-2),
        );
    }
    let offset = frac_div(acc, l as i16) as i16;
    shaped.iter().map(|&t| add_q11(t, offset)).collect()
}

/// Add the prediction back, clamp into the reference's `M_l` range and pad the
/// unused harmonics with the last one — the vector the next frame predicts from.
fn add_pred(shaped: &[i16], pred: &[i16], l: usize) -> [i16; PREV_LEN] {
    let mut out = [0i16; PREV_LEN];
    let mut last = 0i16;
    for i in 0..l {
        let v = add_q11(pred[i], shaped[i]);
        last = if v >= 0x7800 {
            0x77ff
        } else if v < -0x7800 {
            -0x7800
        } else {
            v
        };
        out[i] = last;
    }
    for slot in out[l..].iter_mut() {
        *slot = last;
    }
    out
}

/// Quantize one frame's amplitude shape into `b3..b8` and advance the predictor
/// memory.
///
/// `amps` is the bias-clamped `band_decompress` output (Q11 log2 per harmonic),
/// `gain` the reconstructed frame gain in the same domain — the value
/// [`super::loudness_fixed::next_bias_raw_exact`] leaves in the gain state after
/// this frame's `b2`. Returns `None` when `l` is outside the coded range or the
/// amplitude vector is short, leaving the state untouched.
pub(crate) fn quantize_shape(
    state: &mut AmpShapeState,
    amps: &[i16],
    l: usize,
    gain: i16,
) -> Option<[u16; 6]> {
    if !(9..=56).contains(&l) || amps.len() < l {
        return None;
    }
    let blocks = AMBE_BLOCK_LENGTHS[l - 9];
    if blocks.iter().map(|&j| j as usize).sum::<usize>() != l {
        return None;
    }
    let pred = predict(&state.prev, l, state.prev_l);
    let (g, mut coef) = forward(amps, &pred, blocks);
    let (idx, deq) = quantize(&g, &mut coef, blocks);
    let shaped = reconstruct(&deq, &coef, blocks, l, gain);
    state.prev = add_pred(&shaped, &pred, l);
    state.prev_l = l as i16;
    Some(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_tables_match_the_reference_words() {
        let (cos, scale) = kernels();
        assert_eq!(cos[0], 32767);
        assert_eq!(cos[1], 32766);
        assert_eq!(cos[128], 0);
        assert_eq!(cos[256], -32767);
        assert_eq!(scale[1], 32767);
        assert_eq!(scale[8], 4096);
        assert_eq!(scale[56], 585);
    }

    #[test]
    fn frac_div_halves_and_carries_the_sign() {
        assert_eq!(frac_div(0x80000, 56), 4681);
        assert_eq!(frac_div(-0x80000, 56), -4681);
        assert_eq!(frac_div(3, 56), 0);
    }

    #[test]
    fn cold_state_is_the_reference_seed() {
        let s = AmpShapeState::new();
        assert_eq!(s.prev_l, 15);
        assert!(s.prev.iter().all(|&v| v == 0));
    }

    #[test]
    fn a_flat_frame_round_trips_through_the_predictor() {
        let mut st = AmpShapeState::new();
        let amps = [1000i16; 56];
        let out = quantize_shape(&mut st, &amps, 56, 0x2000).expect("in range");
        assert!(out.iter().all(|&v| v < 512));
        assert_eq!(st.prev_l, 56);
        // the reconstruction of a constant vector is dominated by its DC term,
        // which the gain anchor carries -- the shape indices must not blow up
        assert!(st.prev.iter().all(|&v| (-0x7800..0x7800).contains(&v)));
    }

    #[test]
    fn out_of_range_l_leaves_the_state_alone() {
        let mut st = AmpShapeState::new();
        let amps = [0i16; 56];
        assert!(quantize_shape(&mut st, &amps, 8, 0).is_none());
        assert_eq!(st.prev_l, 15);
    }
}
