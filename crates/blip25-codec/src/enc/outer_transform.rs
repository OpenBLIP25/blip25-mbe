//! The SCALAR half of the encoder's `outer` combine chain (outer wrapper ->
//! combine function -> the band-decompress function's `*2`), the part of the
//! "loudness" mechanism that sits **downstream** of the transform.
//!
//! `band_decompress`'s `outer` input is NOT the transform's return value
//! forwarded through a thin wrapper. The outer wrapper feeds the transform's
//! *truncated* return into a SECOND function, the combine function, which calls
//! 3 further sub-functions that do a block-floating-point radix-2 butterfly +
//! twiddle-multiply combine on an ARRAY, in place. That array work is a pure
//! side effect: the combine function's scalar return is simply
//! `(int16)(transform return) + 1`, with no further processing -- no stack slot
//! the array code touches is ever read back into the return value. The
//! band-decompress function then doubles that value before treating it as
//! `outer`.
//!
//! **The complete formula**:
//! ```text
//! outer(frame) = 2 * ( (i16)(fft_bfp_transform(a1ptr, a2ptr, arg3-1)) + 1 )
//! ```
//! This module implements everything in that formula EXCEPT
//! `fft_bfp_transform` itself (an iterative radix-2 FFT-style transform with
//! its own twiddle/permutation tables; its accumulator bookkeeping is closed,
//! but its per-iteration `si` input requires the full array transform and has
//! never been assembled end-to-end -- the sole open piece of the whole `outer`
//! mechanism). [`combine_outer`] takes `fft_bfp_transform`'s
//! truncated-to-16-bit return value and produces the exact `outer` value
//! `band_decompress` would receive.

/// The combine-function + band-decompress scalar combine: given the
/// transform's own return value truncated to its low 16 bits (zero-extended
/// to its low 16 bits before it is passed in), produce the exact `outer`
/// value `band_decompress` receives.
///
/// The combine function's array-mutating work (front/back split-combine,
/// butterfly dispatch, twiddle-multiply) never feeds this scalar: nothing
/// writes the stack slot the final `+1` reads from.
pub(crate) fn combine_outer(ret_low16: i16) -> i16 {
    let incremented = (ret_low16 as i32 + 1) as i16;
    (2i32 * incremented as i32) as i16
}
