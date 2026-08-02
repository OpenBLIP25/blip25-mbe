//! A block-floating-point normalization-exponent / leading-zero-count
//! primitive, one hop upstream of `fft_bfp_transform`'s `arg2` parameter
//! (`enc::loudness_fixed`'s hardcoded `ARG2_APPROX = -7` placeholder -- see
//! that module's doc, item 2).
//!
//! ```text
//! max_abs = max( saturating_abs(arr[i]) for i in 0..count )  // saturates at 32767
//! if max_abs == 0: return 0
//! shift = 30 - bsr(max_abs << 16)
//! return -shift
//! ```
//!
//! Pinned bit-exact against 600 captured calls.
//!
//! ## What this does NOT close
//!
//! The primitive itself is exact, but `fft_bfp_transform`'s `arg2` is not
//! closed end-to-end. The array this function scans, for the
//! `fft_bfp_transform`-relevant call, is a **stack-local** scratch buffer
//! (three distinct per-frame instances, sizes 199/199/255), not the persistent
//! `encoder_buf`-relative ring array. That stack array's source in terms of raw
//! PCM is unresolved, so `enc::loudness_fixed` cannot replace its
//! `ARG2_APPROX` placeholder with a live formula.

/// `arr[i]`'s saturating absolute value: `i16::MIN` maps to `i16::MAX` (32767),
/// matching the reference's saturating-abs instruction sequence, not a plain
/// two's-complement negate (which would overflow for `i16::MIN`).
#[inline]
fn sat_abs(x: i16) -> u32 {
    if x == i16::MIN {
        i16::MAX as u32
    } else {
        x.unsigned_abs() as u32
    }
}

/// The block-exponent `(arr, count)` formula, given an already-computed
/// `max_abs` (the max over `arr[0..count]` of [`sat_abs`]). Split out from
/// [`block_exponent`] so compact `(max_abs, expected)` capture rows can
/// exercise this half without re-shipping the full arrays.
pub(crate) fn block_exponent_from_max_abs(max_abs: u32) -> i16 {
    if max_abs == 0 {
        return 0;
    }
    // shift = 30 - bsr(max_abs << 16); return -shift
    let v = max_abs << 16;
    let bsr = 31 - v.leading_zeros() as i32;
    let shift = 30 - bsr;
    (-shift) as i16
}

/// The real `(arr, count)` block-floating-point normalization
/// exponent.
pub(crate) fn block_exponent(arr: &[i16]) -> i16 {
    let max_abs = arr.iter().map(|&x| sat_abs(x)).max().unwrap_or(0);
    block_exponent_from_max_abs(max_abs)
}
