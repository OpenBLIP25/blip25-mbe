//! Energy accumulation (`q_builder`) and the 64-bit normalize/shift primitives
//! it rides on.
//!
//! Shared between the encoder's `pq_builder` energy pass and the decoder's
//! unvoiced RMS chain (`dec::unvoiced`). `normalize64`/`shift64` are re-exported
//! back into `enc::band_decompress` for the encoder's other call sites; kept out
//! of `enc/` so the decoder reaches them without the encoder analysis pipeline.

use crate::fixops::u32r::{s32, shl32};

const MASK32: u32 = 0xFFFF_FFFF;

/// Arithmetic shift right on the 32-bit two's-complement value `x`, shift
/// count taken mod 32 (matches x86 `sar`).
#[inline]
fn sar32(x: u32, n: u32) -> u32 {
    ((s32(x)) >> (n & 0x1f)) as u32
}

/// Normalize a 64-bit signed value (given as `lo`/`hi` dwords) to a
/// leading-bit-aligned form, returning the signed shift count (`edi - 9`
/// in the original).
pub(crate) fn normalize64(lo: u32, hi: u32) -> i32 {
    if lo == 0 && hi == 0 {
        return 0;
    }
    let hi_s = s32(hi);
    let (mut lo, mut hi) = (lo, hi);
    if hi_s < 0 {
        lo = !lo;
        hi = !hi;
    }
    // hi_s == 0 case: original asm always treats it as non-negative here
    // (a `test;jae` sequence where `test` unconditionally clears CF) —
    // reproduced literally by simply not negating, same as the `hi_s > 0`
    // case above.
    let mut lo_mask: u32 = 0;
    let mut hi_mask: u32 = 0x80;
    let mut sh: i32 = 0;
    loop {
        let lo_bit = lo_mask & lo;
        let hi_bit = hi_mask & hi;
        if (lo_bit | hi_bit) != 0 {
            break;
        }
        lo_mask = (lo_mask >> 1) | ((hi_mask & 1) << 31);
        sh += 1;
        hi_mask = sar32(hi_mask, 1);
        if sh >= 0x28 {
            break;
        }
    }
    sh - 9
}

/// A saturating 64-bit shift-by-count helper: left shift for `cnt >= 0`,
/// arithmetic right shift for `cnt < 0`. Returns `(new_lo, new_hi)`.
pub(crate) fn shift64(lo: u32, hi: u32, cnt: i32) -> (u32, u32) {
    if cnt >= 0 {
        let sh = cnt as u32;
        if sh >= 0x40 {
            (0, 0)
        } else if sh >= 0x20 {
            (0, shl32(lo, sh & 0x1f))
        } else if sh == 0 {
            (lo, hi)
        } else {
            let new_hi = ((hi << sh) | (lo >> (32 - sh))) & MASK32;
            (shl32(lo, sh), new_hi)
        }
    } else {
        let sh = (-cnt) as u32;
        let hi_s = s32(hi);
        let fill = if hi_s < 0 { MASK32 } else { 0 };
        if sh >= 0x40 {
            (fill, fill)
        } else if sh >= 0x20 {
            (sar32(hi, sh & 0x1f), fill)
        } else if sh == 0 {
            (lo, hi)
        } else {
            let new_lo = ((lo >> sh) | (hi << (32 - sh))) & MASK32;
            (new_lo, sar32(hi, sh))
        }
    }
}

/// The `q_builder` energy accumulator: `sum_k 2 * x[k] * y[k]` over `count`
/// elements, block-float-normalized to `(mantissa, exponent)`.
pub(crate) fn q_builder(x: &[i16], y: &[i16], count: i16) -> (u32, i16) {
    let n = if count >= 0 { count as usize } else { 0 };
    let mut accum: i64 = 0;
    for k in 0..n {
        let a = x[k] as i32;
        let b = y[k] as i32;
        let prod = a.wrapping_mul(b);
        let prod2 = prod.wrapping_add(prod);
        accum = accum.wrapping_add(prod2 as i64);
    }
    let lo = (accum & 0xFFFF_FFFF) as u32;
    let hi = ((accum >> 32) & 0xFFFF_FFFF) as u32;
    let shiftcount = normalize64(lo, hi);
    let (slo, _shi) = shift64(lo, hi, shiftcount);
    let exponent = (-shiftcount) as i16;
    (slo, exponent)
}
