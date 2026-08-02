//! The "final block-consolidate" routine.
//!
//! ## What it does
//!
//! `(dest: *mut i16, src: *const i16, exps: *const i16, count: i32) -> i16` is a
//! block-floating-point re-normalization: given `count` mantissas (`src[i]`)
//! each carrying its OWN exponent (`exps[i]`), it picks ONE shared exponent for
//! the whole block and rescales every mantissa to that scale, writing to `dest`
//! (always `dest == src`, i.e. in place, at all 3 call sites). The shared
//! exponent is the MAX of the `count` per-word exponents -- its sole sub-call is
//! a plain, self-contained `max(arr: &[i16], count) -> i16` (signed comparison,
//! no other logic) -- and is also this routine's return value.
//!
//! Per-word rescale, matching the atan2-core divide/block-float idiom
//! ([`super::atan2_bfp_divide`], [`super::block_exponent`]):
//! ```text
//! shift = exps[i] - shared_exp
//! wide  = (src[i] as i32) << 16
//! wide  = if shift >= 0      { wide << shift }         // no overflow guard on this side (matches the asm)
//!         else if shift > -31 { wide >> (-shift) }      // arithmetic
//!         else                { if wide < 0 {-1} else {0} }  // sar by 31 == sign fill
//! dest[i] = (wide >> 16) as i16   // truncating store, matches `mov [dest],ax`
//! ```
//!
//! The 3 call sites are the P-array consolidate, the Q-array consolidate, and
//! the final `oldgen` consolidate (`oldgen`'s second and final writer).
//!
//! ## What this does NOT close
//!
//! The routine and its `max` sub-call are exact, and the P/Q pre-pass builders'
//! I/O boundary is known: each of their 8 per-word calls returns `ax` = the raw
//! mantissa and separately writes `*dest_ptr` = the raw per-word exponent, and
//! those are exactly the `destPRE`/`exps` arrays fed into this routine
//! immediately afterward.
//!
//! It does NOT close the P/Q builders' INTERNAL arithmetic -- how `ax` and
//! `*dest_ptr` are computed from the raw correlation window. That is comparable
//! in scope to the whole atan2-core closure: a min/max-swap branch, a 64-bit
//! running accumulator, and the same block-float divide the atan2 core uses, all
//! before any value reaches the boundary this module closes. So
//! `Encoder::attempt_b1_fixed` still cannot use real computed P/Q/oldgen arrays
//! end-to-end from raw audio and remains fed `B1_X86_PLACEHOLDER_ABC`.

/// The `max(arr, count)` sub-call formula: plain signed max.
pub(crate) fn shared_exponent(exps: &[i16]) -> i16 {
    exps.iter().copied().max().unwrap_or(0)
}

/// One word's rescale, matching the block-consolidate inner-loop body (see the
/// module doc).
#[inline]
fn rescale_word(mantissa: i16, shift: i32) -> i16 {
    let wide: i64 = (mantissa as i64) << 16;
    let wide = if shift >= 0 {
        // The asm has no overflow guard on this side either. A Rust `<<` on i64
        // with a shift this small -- the codec's per-word exponents are
        // single-digit magnitudes in every captured row -- never overflows i64,
        // so this stays exact without replicating x86's
        // 32-bit-wraparound-then-truncate behavior.
        wide << shift
    } else if shift > -31 {
        wide >> (-shift)
    } else {
        if wide < 0 {
            -1
        } else {
            0
        }
    };
    (wide >> 16) as i16
}

/// The block-consolidate `(dest, src, exps, count)` formula. `dest` and `src`
/// are the same slice in the reference (always in-place at all 3 call sites);
/// they are modeled here as separate `src`/`out` slices so callers do not need
/// an aliased buffer. For an exact in-place simulation, do
/// `out.copy_from_slice(src)` first. Returns the shared exponent.
pub(crate) fn block_consolidate(src: &[i16], exps: &[i16], out: &mut [i16]) -> i16 {
    debug_assert_eq!(src.len(), exps.len());
    debug_assert_eq!(src.len(), out.len());
    let shared = shared_exponent(exps);
    for i in 0..src.len() {
        let shift = exps[i] as i32 - shared as i32;
        out[i] = rescale_word(src[i], shift);
    }
    shared
}
