//! The "127-tap" windowed complex correlation, used inside the voicing/pitch
//! chain's `S` angle computation, which calls it with `count=127` as its first
//! step and feeds its 2-word output into
//! [`super::atan2_chain::atan2_dispatch`].
//!
//! ## What this computes
//!
//! Given three per-tap input streams (`w`: a real per-tap window/scale
//! coefficient table; `a`: a complex stream as interleaved `[re,im]` int16
//! pairs; `c`: a second complex stream, same interleaving) and a tap count (127
//! at the call site), this computes a **windowed complex cross-correlation**:
//! for each tap `k`, `A[k]` is first windowed by `w[k]`
//! (`P[k] = (A[k] * w[k]) >> 14`, computed via a `<<2 >>16` shift pair matching
//! the reference instruction sequence), then accumulated into two 64-bit sums:
//!
//! ```text
//! ACC1 = sum_k 2*(C0[k]*P0[k] + C1[k]*P1[k])   // "real" part
//! ACC2 = sum_k 2*(C0[k]*P1[k] - C1[k]*P0[k])   // "imaginary" part
//! ```
//!
//! Both 64-bit sums are then independently normalized to a leading-bit-aligned
//! form ([`super::band_decompress::normalize64`]), combined with the caller's
//! `arg4`/`arg6` scalars into a "common exponent" candidate each
//! (`arg4 - own_exponent + arg6`), and the LARGER of the two candidates is the
//! shared output scale. Both normalized sums are re-shifted
//! ([`super::band_decompress::shift64`]) down to that scale, and the high 16
//! bits of each resulting 32-bit value become the 2-word output. The chosen
//! common exponent is the return value.
//!
//! ## What this does NOT close
//!
//! `w`/`a`/`c`'s upstream construction -- where these 3 streams' *content* comes
//! from in terms of raw PCM -- is a separate, open trace. This module answers
//! "given the per-tap inputs, what does the correlation compute", not "how are
//! the per-tap inputs computed from audio". `a` (`ptr38`) and `c` (`ptr40`) are
//! per-frame-varying, audio-responsive content whose writer is untraced.
//!
//! Separately, the caller's tail AFTER this function returns (and after the
//! atan2 call) has an undecoded second divide plus two more sub-calls before its
//! final `ret` -- see [`super::atan2_chain`]'s module doc.

use super::band_decompress::{normalize64, shift64};

use crate::fixops::acc64::s16;

/// The 127-tap windowed complex correlation. `w`/`a`/`c` must have at least
/// `count` (`w`) / `2*count` (`a`,`c`) entries; only the first
/// `count`/`2*count` are read, matching the reference's tap-count argument.
///
/// Returns `(out0, out2, chosen_exponent)`: `out0` is the "real"/`cx` part
/// (consumed as [`super::atan2_chain::atan2_dispatch`]'s `cx` argument), `out2`
/// is the "imaginary"/`di` part (`atan2_dispatch`'s `di`), and
/// `chosen_exponent` is `atan2_dispatch`'s `corr_exp` argument.
pub(crate) fn windowed_complex_correlation(
    w: &[i16],
    a: &[i16],
    c: &[i16],
    count: usize,
    arg4: i32,
    arg6: i32,
) -> (i16, i16, i16) {
    let mut acc1: i64 = 0;
    let mut acc2: i64 = 0;
    for k in 0..count {
        let wk = w[k] as i32;
        let a0 = a[2 * k] as i32;
        let a1 = a[2 * k + 1] as i32;
        let c0 = c[2 * k] as i32;
        let c1 = c[2 * k + 1] as i32;
        // P = windowed A: (A*w) << 2 >> 16, i.e. (A*w) >> 14, truncated to 16
        // bits then sign-extended -- matching the instruction sequence
        // (imul; shl 2; sar 16).
        let p0 = s16(((a0.wrapping_mul(wk)) << 2) >> 16) as i32;
        let p1 = s16(((a1.wrapping_mul(wk)) << 2) >> 16) as i32;
        acc1 += 2 * ((c0 as i64) * (p0 as i64) + (c1 as i64) * (p1 as i64));
        acc2 += 2 * ((c0 as i64) * (p1 as i64) - (c1 as i64) * (p0 as i64));
    }

    let lo1 = acc1 as u64 as u32;
    let hi1 = (acc1 as u64 >> 32) as u32;
    let exp1 = normalize64(lo1, hi1);
    let (rlo1, rhi1) = shift64(lo1, hi1, exp1);
    let common_exp1 = s16(arg4.wrapping_sub(exp1).wrapping_add(arg6));

    let lo2 = acc2 as u64 as u32;
    let hi2 = (acc2 as u64 >> 32) as u32;
    let exp2 = normalize64(lo2, hi2);
    let (rlo2, rhi2) = shift64(lo2, hi2, exp2);
    let common_exp2 = s16(arg4.wrapping_sub(exp2).wrapping_add(arg6));

    // The LARGER of the two common-exponent candidates is the shared output
    // scale.
    let chosen = if common_exp1 > common_exp2 {
        common_exp1
    } else {
        common_exp2
    };

    let (f1lo, _f1hi) = shift64(rlo1, rhi1, (common_exp1 as i32) - (chosen as i32));
    let (f2lo, _f2hi) = shift64(rlo2, rhi2, (common_exp2 as i32) - (chosen as i32));

    let out0 = s16((f1lo >> 16) as i32);
    let out2 = s16((f2lo >> 16) as i32);
    (out0, out2, chosen)
}
