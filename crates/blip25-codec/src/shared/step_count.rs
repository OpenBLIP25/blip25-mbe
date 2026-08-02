//! `COUNT`-from-`STEP` derivation (`count_from_step`) and its truncating-divide
//! / shift helpers.
//!
//! Shared between the encoder's `enc::step_recursive_fixed` (which re-imports the
//! helpers for its other stages) and the decoder's site-A pitch path
//! (`crate::dec::sitea_ml`, driven from `lib.rs`): the site-A `L` is derived from
//! the interpolated `omega0` via the same `243100/omega0 -> [9,56]` relation.
//! Kept out of `enc/` so the decoder reaches it without the encoder pipeline.

/// A plain signed truncating divide, with a `|a|<|c| -> 0` special case.
pub(crate) fn trunc_div16(a: i32, c: i32) -> i16 {
    if c == 0 || (a as i64).abs() < (c as i64).abs() {
        0
    } else {
        (a / c) as i16 // Rust's `/` on i32 truncates toward zero, matching IDIV.
    }
}

#[inline]
pub(crate) fn sar32(x: u32, n: u32) -> u32 {
    ((x as i32) >> (n & 31)) as u32
}
#[inline]
pub(crate) fn shl32(x: u32, n: u32) -> u32 {
    x.wrapping_shl(n & 31)
}

/// Derives a matching `COUNT` from a candidate
/// `STEP` value via an approximately-inverse `243100/STEP` relation
/// (a genuinely different, but closely related, magic constant to round
/// 154's own already-documented `237743/COUNT` approximate forward
/// relation), refined once if the product exceeds a threshold, clamped to
/// `[9, 56]` (the same real observed `COUNT` range `step_table.rs`'s own
/// modal table covers).
/// Also reached from the DECODE side: the site-A pitch path uses this same
/// `243100/omega0 -> [9,56]` relation to derive site A's `L` from site A's
/// interpolated `omega0` (see `crate::dec::sitea_ml`). Exposed for that reuse;
/// the encoder's own callers are unaffected.
pub(crate) fn count_from_step(step: i16) -> i16 {
    let k_est = trunc_div16(0x3B39C, step as i32); // 243100 / step
    let factor = ((2i32 * k_est as i32) + 1) as i16 as i32; // lea+movsx, truncate to i16 first
    let product = factor.wrapping_mul(step as i32);
    let result: i16 = if product < 0x80000 {
        k_est
    } else {
        let refined = trunc_div16(0x80000, step as i32); // 524288 / step
        let shl_r = shl32(refined as u16 as u32, 16);
        let dec_r = shl_r.wrapping_sub(1);
        let sar_r = sar32(dec_r, 17);
        sar_r as u16 as i16
    };
    result.clamp(9, 56)
}
