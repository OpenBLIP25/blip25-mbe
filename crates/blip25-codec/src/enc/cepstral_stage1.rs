//! **Stage 1** of the cepstral-domain pitch/voicing pipeline, entered from its
//! parent routine.
//!
//! This is the writer of the encoder's array **`A`** as consumed by
//! [`super::windowed_complex_correlation::windowed_complex_correlation`] -- its output buffer is `windowed_complex_correlation`'s `arg3`/`A`
//! pointer.
//!
//! ## What this computes
//!
//! For each of `count` (127 at the call site) raw 32-bit input values (a
//! magnitude-squared spectrum bin), this performs a block-floating-point
//! normalize (`bsr`-based leading-bit scan, dynamic left-shift), derives a
//! 16-bit mantissa and an adjusted exponent, then (for non-degenerate entries)
//! evaluates the SAME fixed-point log2 primitive used by the per-band loudness
//! pipeline ([`super::band_decompress::log2_fn`] /
//! [`super::band_decompress::shift_scale`]). It is one shared, reused log2
//! primitive in the reference codec, not two independently-invented ones.
//!
//! Degenerate entries (zero input, zero mantissa after normalize, or an exponent
//! below -32) are written as the sentinel `0x8000` (-32768). Every entry
//! additionally writes a second, always-zero word immediately after the log
//! value -- the output is a `[log_mag, 0]`-interleaved stream (a real-only
//! "complex" stream, matching [`super::windowed_complex_correlation`]'s `A`/`C` interleaved-pair
//! convention).
//!
//! ```text
//! for k in 0..count:
//!     raw = source[k]
//!     if raw == 0: out[2k]=0x8000; out[2k+1]=0; continue
//!     mag = raw >= 0 ? raw : !raw        // bsr operates on this
//!     shift = 30 - bsr(mag)               // 0 if mag == 0
//!     shifted = raw << shift              // note: shifts the SIGNED raw, not mag
//!     exponent = (scale_arg - shift) as i16
//!     mantissa = (shifted >> 16) as i16   // arithmetic shift
//!     if mantissa == 0 || exponent <= -32:
//!         out[2k] = 0x8000
//!     else:
//!         logres = log2_fn(mantissa, exponent)
//!         out[2k] = round_sat16(shift_scale(logres, 0xf - 5))
//!     out[2k+1] = 0
//! ```
//!
//! `scale_arg` is the caller's forwarded exponent bias (the parent routine's
//! arg5). Only its low 16 bits are used: the caller passes a full 32-bit
//! register whose high word is immaterial, so e.g. `0x00010012` reduces to the
//! same effective `i16` as `0x00010000`.
//!
//! ## What this does NOT close
//!
//! This validates "given the per-tap input, does this formula reproduce the
//! output", not where the input comes from.
//!
//! The `source` array's IDENTITY is known: the parent routine's persistent-context
//! argument (its `arg4`, this module's `source` base) is `encoder_buf + 0x1c`
//! exactly, the same address the confirmed 129-dword `2*(Re²+Im²)` spectrum is
//! anchored to. The CONTENT is not: the fixed-point formula computing
//! `encoder_buf+0x1c` from raw PCM -- the 256-point/129-bin transform itself,
//! downstream of `real_fft32`/`windowed_taper` -- has no known verified assembly
//! formula.
//!
//! Stage 2 of the same pipeline (a three-call chain writing array `w` and a
//! refined array `A`) is also open: `w`'s writer calls a different, undecoded
//! pair of helpers, not this module's log2 path. See `voicing_fixed.rs` for
//! `w_builder`'s `thresh8`/`table8` inputs, which share this base pointer at
//! different offsets.

use super::band_decompress::{log2_fn, shift_scale};

use crate::fixops::acc64::s16;

/// Rounds/saturates a raw 32-bit value via the same
/// add-`0x8000`-then-arithmetic-shift-right-16 idiom used throughout this
/// codebase's already-ported fixed-point primitives (matching
/// `band_decompress.rs`'s inline per-iteration use of the same pattern).
#[inline]
fn round_sat16(shifted: u32) -> i16 {
    let rounded = shifted.wrapping_add(0x8000);
    let ov = (rounded ^ shifted) & rounded;
    let fin = if (ov as i32) >= 0 {
        ((rounded as i32) >> 16) as u32
    } else if (shifted as i32) < 0 {
        ((0x8000_0000u32 as i32) >> 16) as u32
    } else {
        ((0x7FFF_FFFFu32 as i32) >> 16) as u32
    };
    s16(fin as i32)
}

/// Port of stage 1. `source` must have at least `count` entries.
/// Returns a `2*count`-length `[log_mag, 0]`-interleaved `i16` stream
/// (array `A`'s own real per-frame content, before the parent routine's
/// later refinement/combine sub-calls — see module doc for what remains
/// open).
pub(crate) fn cepstral_stage1_log_transform(source: &[i32], scale_arg: i16) -> Vec<i16> {
    let mut out = Vec::with_capacity(source.len() * 2);
    for &raw in source {
        let raw_u32 = raw as u32;
        if raw_u32 == 0 {
            out.push(s16(0x8000));
            out.push(0);
            continue;
        }
        let mag = if (raw_u32 as i32) >= 0 {
            raw_u32
        } else {
            !raw_u32
        };
        // matches this codebase's own bsr(0)=>0 convention, see
        // band_decompress::log2_fn's identical guard.
        let hibit = if mag == 0 {
            0
        } else {
            31 - mag.leading_zeros()
        };
        let shift = (0x1eu32.wrapping_sub(hibit)) & 0xffff;
        let shifted = if (shift & 0x1f) == 0 {
            raw_u32
        } else {
            raw_u32.wrapping_shl(shift & 0x1f)
        };
        // x86 `shl` with a shift count outside 0..31 (after masking to 5
        // bits by the CPU) matches Rust's `wrapping_shl` masking exactly.
        let exp_val = (scale_arg as i32).wrapping_sub(shift as i32) as u32;
        let mantissa = s16((shifted as i32) >> 16);
        let exponent = s16(exp_val as i32);
        if mantissa == 0 || exponent <= (0xffe0u16 as i16) {
            out.push(s16(0x8000));
        } else {
            let logres = log2_fn(mantissa, exponent);
            let scaled = shift_scale(logres, 0xf - 5);
            out.push(round_sat16(scaled));
        }
        out.push(0);
    }
    out
}
