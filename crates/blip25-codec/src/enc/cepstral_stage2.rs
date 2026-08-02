//! The shared "rescale+accumulate" primitive and the two per-tap combine steps
//! that build array `A`'s final (post-refine) value and array `C` in full.
//!
//! ## `rescale_accumulate`: block-floating-point rescale + exponent accumulate
//!
//! ```text
//! fn rescale_accumulate(buf: &mut [i16], accum: i16) -> i16:
//!     exp = block_exponent(buf)          // super::block_exponent
//!     shift = -exp
//!     rescale_inplace(buf, shift)
//!     return accum.wrapping_add(exp)     // 16-bit add, in place into the
//!                                         // caller's own accumulator cell
//! ```
//!
//! The in-place rescale `rescale_inplace(dst, src, len, shift)`: for
//! `shift == 0`, a plain copy. For `shift != 0`, each element is
//! `((sext32(src[i]) << 16) << shift) >> 16` (positive `shift`, a plain 32-bit
//! wrapping left-shift-then-truncate -- **not** a saturating shift; the `shl`/
//! `sar` pair has no clamp) or `((sext32(src[i]) << 16) >> mag) >> 16` (negative
//! `shift`, `mag = min(-shift, 31)`, arithmetic shifts throughout).
//!
//! ## `combine_scalar`: per-tap complex*real-scalar Q15 multiply, then rescale
//!
//! Called **twice** per parent invocation, the same function both times: once
//! combining array `A` (post-refine) with a scalar weight array in place (the
//! encoder's "`A *= w`" step), once combining `C`'s base array with a second
//! scalar weight array, writing `C`.
//!
//! ```text
//! fn combine_scalar(buf: &mut [i16] /* interleaved re,im pairs, in place */,
//!                    weight: &[i16], seed: i16, count: usize) -> i16:
//!     for k in 0..count:
//!         w = weight[k]
//!         buf[2k]   = round16((buf[2k]   as i32 * w as i32) * 2)   // >>16
//!         buf[2k+1] = round16((buf[2k+1] as i32 * w as i32) * 2)   // >>16
//!     return rescale_accumulate(&mut buf[0..2*count], seed)
//! ```
//! `round16(x) = (x >> 16) as i16`, a plain arithmetic shift -- there is **no**
//! `+0x8000` rounding bias; the instruction pair is `add ecx,ecx; sar ecx,0x10`
//! with nothing in between.
//!
//! `seed` at the call sites is `(arg3 + arg5)` truncated to 16 bits, where
//! `arg3`/`arg5` are the refine step's and the scale step's scalar returns. The
//! refine step unconditionally returns 0 (`xor eax,eax` immediately before its
//! only `ret`), so for the `A` call `seed == scale_ret` alone.
//!
//! ## `selfcorr`: per-tap self-referential complex*complex multiply, then rescale
//!
//! Reads TWO interleaved complex arrays -- `q` (`arg1`==`arg2`, i.e. passed
//! twice, read-only, advancing) and `p` (`arg4`, the true in-place destination;
//! writes land in `p`'s buffer, not `q`'s) -- and computes a conjugate-style
//! cross product:
//!
//! ```text
//! fn selfcorr(p: &mut [i16] /* interleaved, in place */,
//!             q: &[i16] /* interleaved, read-only */,
//!             count: usize) {
//!     for k in 0..count:
//!         (pre,pim) = (p[2k], p[2k+1])
//!         (qre,qim) = (q[2k], q[2k+1])
//!         re = (pre*qre + pim*qim) >> 16   // arithmetic, no rounding bias
//!         im = (pre*qim - pim*qre) >> 16
//!         p[2k] = re; p[2k+1] = im
//!     rescale_accumulate(&mut p[0..2*count], seed)   // same tail as combine_scalar
//! }
//! ```
//!
//! At both call sites feeding `C`, `q`'s buffer is `A`'s post-combine buffer and
//! `p`'s buffer is `A`'s buffer **shifted by exactly one complex pair**
//! (`p_ptr == q_ptr + 4` bytes). So this computes `A[k+1] conj-cross A[k]`, a
//! **lag-1 self-correlation of `A` with itself** -- NOT a fixed
//! coefficient-table "twiddle" multiply. The result is written back into `A`'s
//! own buffer in place at the first call site, so whatever `windowed_complex_correlation` later reads
//! as `A` reflects this self-correlation product, not `combine_scalar`'s
//! earlier combine-with-`w` result alone. `A`'s pipeline is therefore: base
//! build -> refine -> combine-with-`w` -> **self-correlate** -> ...
//!
//! Note the rescale is part of `selfcorr`: comparing the RAW per-tap product
//! against the post-call buffer content, without applying
//! `rescale_accumulate`'s whole-buffer BFP rescale first, leaves an unexplained
//! residual even though the disassembled bytes clearly encode a literal
//! `sar reg,0x10`.
//!
//! ## What this does NOT close
//!
//! The wiring graph feeding these primitives is intricate. Address-matching
//! establishes that `combine_scalar`'s first call's `weight` argument and the
//! normalize step's in-place buffer are the SAME physical stack slot as `w`, and
//! that `combine_scalar`'s first call plus `selfcorr`'s first call both operate
//! in place on the SAME physical stack slot as `A`. But `combine_scalar`'s first
//! call (which reads `w`'s slot) runs BEFORE the normalize step (which writes
//! `w`'s slot) in program order, so the value it reads is NOT the same-frame
//! normalize output -- most likely a previous-frame leftover, not confirmed.
//!
//! `A`'s raw content (post-refine) still requires the unclosed multi-stage FFT
//! core and its butterfly kernels for stages beyond the first -- see
//! [`super::loudness_transform`]. Because `C`'s two `selfcorr` calls both read
//! from `A`'s buffer, `C` inherits the same blocker.

use super::block_exponent::block_exponent;

#[inline]
fn s16(x: i64) -> i16 {
    (x & 0xffff) as u16 as i16
}

/// In-place BFP rescale (`dst == src`, given `len` and `shift`): the reference's
/// literal double-16-bit-shift idiom (a
/// wrapping left shift for positive `shift`, an arithmetic right shift for
/// negative `shift`, an identity copy for `shift == 0`).
fn rescale_inplace(buf: &mut [i16], shift: i16) {
    if shift == 0 {
        return;
    }
    if shift > 0 {
        let sh = (shift as u32) & 0x1f;
        for x in buf.iter_mut() {
            let v: u32 = (*x as i32 as u32).wrapping_shl(16); // sext16 -> hi16, lo16=0
            let v: u32 = v.wrapping_shl(sh); // wrapping 32-bit left shift, NO saturation
            let v: i32 = (v as i32) >> 16; // arithmetic shift right 16
            *x = s16(v as i64);
        }
    } else {
        let mag = if shift <= -31 { 31u32 } else { (-shift) as u32 };
        for x in buf.iter_mut() {
            let v: u32 = (*x as i32 as u32).wrapping_shl(16);
            let v: i32 = (v as i32) >> mag; // arithmetic shift right, 32-bit
            let v: i32 = v >> 16; // arithmetic shift right 16
            *x = s16(v as i64);
        }
    }
}

/// `block_exponent` the buffer, rescale it in place, and return `accum +
/// block_exponent` (wrapping 16-bit add). Shared tail of both
/// [`combine_scalar`] and `selfcorr`.
pub(crate) fn rescale_accumulate(buf: &mut [i16], accum: i16) -> i16 {
    let exp = block_exponent(buf);
    rescale_inplace(buf, s16(-(exp as i64)));
    accum.wrapping_add(exp)
}

/// Per-tap complex*real-scalar Q15 multiply of `buf` (interleaved `re,im`
/// pairs) by `weight` (one real scalar per tap), in place, then a
/// whole-buffer BFP rescale. `seed` is the caller's own pre-summed scalar
/// (`refine_ret + scale_ret` for the `A` call, an analogous pair for the `C`
/// call). Returns the final accumulated exponent (the reference's 16-bit return
/// value). `buf`'s content is pinned bit-exact; the exponent return itself is
/// not independently checked against a captured return value, since the
/// available captures cover array content only.
pub(crate) fn combine_scalar(buf: &mut [i16], weight: &[i16], seed: i16, count: usize) -> i16 {
    for k in 0..count {
        let w = weight[k] as i32;
        for j in 0..2 {
            let v = (buf[2 * k + j] as i32).wrapping_mul(w);
            let v2 = v.wrapping_add(v);
            buf[2 * k + j] = (v2 >> 16) as i16;
        }
    }
    rescale_accumulate(&mut buf[0..2 * count], seed)
}
