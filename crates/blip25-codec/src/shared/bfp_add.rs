//! Block-floating-point ADD, one of the pitch-lag "best-candidate tracker's"
//! (gap 2's `clamped_lag` machinery -- see [`super::voicing_fixed`]) two calls
//! to this exact function per iteration.
//!
//! Aligns two `(mantissa, exponent)` block-floating-point pairs to a common
//! scale, adds them, and re-normalizes the sum -- the `bsr`/`shl`/`sar 0x10`
//! normalize idiom used elsewhere in this project (e.g.
//! [`super::atan2_bfp_divide::bfp_divide`], [`super::block_exponent`]) composed
//! with a classic BFP-add alignment step.
//!
//! ## Zero-operand exponents
//!
//! When `m1 == 0` the output exponent is `e2` -- the SURVIVING operand's
//! exponent -- and symmetrically `e1` when `m2 == 0`. Taking the zeroed
//! operand's exponent instead is a natural-looking transcription error that
//! captured calls disprove (200 `m1==0` calls, all with `e1 != e2`).
//!
//! ## Unexercised paths
//!
//! The two large-exponent-difference saturating fast paths (`(e1-e2) >= 0x20`
//! and `(e2-e1) >= 0x20`, both keeping the dominant operand unchanged) are not
//! hit by any captured call. They come from the static read and are not
//! independently verified.

/// Sign-extend the low 16 bits of `x` to `i32` (matches `movsx`).
#[inline]
fn sext16(x: i32) -> i32 {
    (x as i16) as i32
}

/// `bsr` (bit-scan-reverse) of a nonzero 32-bit value: index of the
/// highest set bit.
///
/// `x == 0` is REACHABLE here -- it is exactly the `acc == -1` case, where the
/// reference's `not eax` yields 0 and `bsr eax,eax` then runs with a **zero
/// source**. x86 leaves `bsr`'s destination unchanged in that case, and the
/// destination already holds that same 0, so the idiom yields 0 and
/// `shift = 30`. Do not assert `x != 0`: the reference does not hold it, and a
/// `31 - 32` underflow here returns `(-32768, e-1)` where the reference returns
/// `(-16384, e)`.
#[inline]
fn bsr32(x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    31 - x.leading_zeros()
}

/// Block-floating-point ADD. Returns `(mantissa, exponent)`.
///
/// The `m1==0`/`m2==0`/general-normalize paths are pinned bit-exact against
/// captured calls; the two large-exponent-difference saturating fast paths are
/// static-only -- see the module doc.
pub(crate) fn bfp_add(m1: i16, e1: i16, m2: i16, e2: i16) -> (i16, i16) {
    let (m1, e1, m2, e2) = (m1 as i32, e1 as i32, m2 as i32, e2 as i32);

    // The `m1 == 0` test and the `(e2 - e1) >= 0x20` test BOTH jump to the same
    // exit, so the exponent gap is tested BEFORE `m2 == 0` -- and that exit
    // returns **(0, 0)** when m2 == 0, not (m1, e1). Checking `m2 == 0` first is
    // unreachable on encoder captures and wrong everywhere else.
    if m1 == 0 || (e2 - e1) >= 0x20 {
        return if m2 == 0 {
            (0, 0)
        } else {
            (m2 as i16, e2 as i16)
        };
    }
    // The `m2 == 0` test and the `(e1 - e2) >= 0x20` test both jump to the same
    // exit => return (m1, e1).
    if m2 == 0 || (e1 - e2) >= 0x20 {
        return (m1 as i16, e1 as i16);
    }

    let base = e1.max(e2) + 1;
    // The reference's THREE-way shift (emitted twice as identical copies)
    // includes a total-collapse leg (arithmetic shift right by 31) for
    // `sh <= -31`. The guards above only bound |e1 - e2| <= 31, so `base - e`
    // reaches **32**: a plain `v >> shift` both misses the collapse leg and is a
    // Rust shift-overflow. The reference computes `e - base` and branches on its
    // low 16 bits.
    let align = |m: i32, e: i32| -> i32 {
        let v = sext16(m) << 16;
        let sh = (e - base) as i16 as i32; // e - base, kept to its low 16 bits (signed)
        if sh >= 0 {
            // x86 masks the shift count to 5 bits.
            ((v as u32) << ((sh as u32) & 31)) as i32
        } else if sh <= -31 {
            v >> 31 // arithmetic shift right by 31 -- fill with the sign bit
        } else {
            v >> (-sh) // arithmetic shift right by the alignment amount
        }
    };
    let acc = align(m1, e1).wrapping_add(align(m2, e2));
    if acc == 0 {
        return (0, 0);
    }
    let absacc: u32 = if acc >= 0 { acc as u32 } else { !(acc as u32) };
    let shift: i32 = 30 - bsr32(absacc) as i32;
    let shifted: i32 = if shift >= 0 {
        (acc as i64).wrapping_shl(shift as u32) as i32
    } else {
        acc >> (-shift)
    };
    let mant = sext16(shifted >> 16);
    if mant == 0 {
        return (0, 0);
    }
    (mant as i16, (base - shift) as i16)
}
