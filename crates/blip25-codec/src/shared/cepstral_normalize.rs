//! Two reference-vocoder routines: a BFP-shaped saturating
//! "half-reciprocal-ratio" primitive and its per-entry array normalizer.
//!
//! ## The BFP-shaped saturating "half-reciprocal-ratio" primitive
//!
//! A sibling of the atan2 divide primitive:
//!
//! ```text
//! fn ratio(a: i32, b: i16) -> i32:
//!     sign = sign(a) * sign(b)
//!     a_abs = a == i32::MIN ? i32::MAX : a.abs()
//!     b_abs = b == i16::MIN ? 0x7fff : b.unsigned_abs()  // as i32
//!     if a_abs >= b_abs:
//!         ret = (a_abs / b_abs) >> 1     // idiv then sar 1 (halve)
//!     else:
//!         ret = 0
//!     return sign * ret
//! ```
//!
//! **The branch direction is load-bearing and reads backwards.** `jge` jumps
//! INTO the divide path, not away from it: `a_abs >= b_abs` divides, and the
//! else arm returns 0. Inverting it still matches 33.3% of captured tap-words
//! (every zero-result case), so a test that only checks "most taps agree" will
//! not catch it -- every nonzero case fails.
//!
//! ## The per-entry normalizer: applies the primitive per entry to an array
//!
//! ```text
//! fn normalize(dest: &mut [i16], src: &[i16], shift: i16, count: usize):
//!     bias = if shift >= 0: 0x8000i32 << (shift & 31)
//!            elif shift > -31: 0x8000i32 >> (-shift)
//!            else: 0
//!     for i in 0..count:
//!         v = src[i]
//!         dest[i] = if v == 0 { 0x7fff } else { ratio(bias, v) as i16 }
//! ```
//!
//! The reference function also returns a bookkeeping scalar, `16 - shift -
//! arg3`, used only by the caller's exponent tracking. It is not needed to
//! reproduce `dest`, so it is not ported.
//!
//! Pinned bit-exact against 3870 captured tap-words (30 calls x 129 taps),
//! spanning all 6 observed `shift` values (9, 10, 11, 12, 14, 15) and both the
//! zero-sentinel and general-divide paths.
//!
//! ## What this does NOT close
//!
//! This is the normalizer's arithmetic in isolation, given its `(src, shift,
//! count)` inputs. It does not establish where `src` comes from at this call
//! site: the normalizer operates on a LOCAL stack buffer, not `A`/`w`/`C`
//! directly, and that buffer's upstream writer -- most likely the FFT-shaped
//! refine stage, entangled with the open array-transform work -- is unclosed.
//!
//! Nor does it close the sibling complex-multiply stage. That stage is a
//! complex multiply between ADJACENT samples of the SAME in-place array --
//! self-referential, NOT a fixed coefficient-table lookup, so do not model it
//! as a twiddle table. Its exact shift/rounding is unresolved: captured data
//! fits a shift-15 conjugate-multiply on ~64%/57% of taps (exact or +/-1) at
//! the two call sites, while the disassembled bytes unambiguously encode a
//! literal `sar reg,0x10` (=16). Do not ship either reading as the formula.
//! `C`'s full construction needs that stage plus whatever combines its output
//! with the normalizer's before reaching `windowed_complex_correlation`'s `C`.

use crate::fixops::i32r::s16;

/// Given `a` (a BFP-derived rounding/reciprocal bias constant) and `b`
/// (one raw array entry), returns `sign(a)*sign(b) * (|a|/(2*|b|))` if
/// `|a| >= |b|`, else `0`. The branch direction reads backwards from the
/// disassembly -- see the module doc before touching it.
pub(crate) fn bfp_reciprocal_ratio(a: i32, b: i16) -> i32 {
    let sign: i32 = (if a >= 0 { 1 } else { -1 }) * (if b >= 0 { 1 } else { -1 });
    let a_abs: i64 = if a == i32::MIN {
        i32::MAX as i64
    } else {
        (a as i64).abs()
    };
    let b_abs: i64 = if b == i16::MIN {
        0x7fff
    } else {
        (b as i64).abs()
    };
    if a_abs >= b_abs {
        if b_abs == 0 {
            return 0; // guard: hardware would fault (idiv by zero); never observed
        }
        let q = a_abs / b_abs; // both non-negative -> truncating idiv == floor
        (sign as i64 * (q >> 1)) as i32
    } else {
        0
    }
}

/// The normalizer's bias constant, derived from its `shift` argument: the
/// `0x8000<<shift`/`0x8000>>(-shift)` idiom the reference prologue computes
/// before the per-entry loop.
fn normalizer_bias(shift: i16) -> i32 {
    if shift >= 0 {
        (0x8000i32).wrapping_shl((shift as u32) & 31)
    } else if shift > -31 {
        0x8000i32 >> ((-shift) as u32)
    } else {
        0
    }
}

/// Per-entry BFP-shaped normalize with a `0x7fff` zero sentinel.
pub(crate) fn cepstral_normalize(dest: &mut [i16], src: &[i16], shift: i16) {
    let bias = normalizer_bias(shift);
    for (d, &v) in dest.iter_mut().zip(src.iter()) {
        *d = if v == 0 {
            0x7fff
        } else {
            s16(bfp_reciprocal_ratio(bias, v))
        };
    }
}
