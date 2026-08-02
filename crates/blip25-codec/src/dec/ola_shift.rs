//! The OLA band-synth **shift primitives** ([`shift_round`], [`shift_trunc`])
//! and their composer ([`split_hi_res`]).
//!
//! These sit under the band-synth routine, which the placement loop drives.
//!
//! # The shared three-way shift
//!
//! All three (and `fold_incr`, and one further sibling) open with the
//! identical idiom: `ecx = shift - 15` (32-bit), then a branch on **`cx`, the
//! low 16 bits** (`test cx,cx` / `cmp cx,0xffe1`), while the shift itself is
//! applied with **`cl` masked to 5 bits** by the hardware.
//!
//! Reading `cx` rather than `ecx` is load-bearing: `shift` values whose low
//! word is positive while the full i32 is negative (and vice versa) take
//! different legs. Using `cl`'s own 5 bits vs `cx`'s is *not* a
//! distinction: `ecx ≡ cx (mod 65536)` and `32 | 65536`, so `ecx ≡ cx (mod 32)`
//! and `-ecx ≡ -cx (mod 32)`. The two forms are provably identical.
//!
//! **The `-31` vs `-32` boundary is DEGENERATE** (a structural argument). At
//! `cx == -31` the `cmp cx,0xffe1 / jg` guard sends control to the `sar v,0x1f`
//! collapse, while the other arm would compute `sar(v, 31)` -- the same value.
//! No input can separate them. Ported as the bytes read.

use crate::fixops::dec32::{sar, shl, sx16};

/// The three-way shift shared by [`shift_round`], [`shift_trunc`], and
/// [`split_hi_res`]: `sh = sx16(shift - 15)`, branch on `sh`.
///
/// `sh >= 0` -> `shl`; `-31 < sh < 0` -> `sar` by `-sh`; `sh <= -31` -> the
/// `sar(v, 31)` collapse.
#[inline]
fn shift_by(v: i32, shift: i32) -> i32 {
    let sh = sx16(shift.wrapping_sub(15));
    if sh >= 0 {
        shl(v, sh as u32)
    } else if sh > -31 {
        sar(v, (-sh) as u32)
    } else {
        sar(v, 31)
    }
}

/// (71 B, leaf): shift by `shift - 15`, round up by `0xffff` with signed
/// saturation, then mask off the low word.
///
/// `rounded = shifted + 0xffff`; overflow iff `((rounded ^ shifted) & rounded)
/// < 0`; on overflow the result becomes `0x7fffffff + (shifted < 0)`, i.e.
/// `i32::MIN` when `shifted` was negative and `i32::MAX` otherwise. Only
/// `shifted >= 0` can actually overflow a `+0xffff`, so the `sets cl` leg
/// selects `i32::MAX` in every reachable case; ported faithfully regardless.
pub(crate) fn shift_round(v: i32, shift: i32) -> i32 {
    let shifted = shift_by(v, shift);
    // round up by adding 0xffff; the xor/and/sign-test detects signed overflow
    let rounded = shifted.wrapping_add(0xffff);
    let sat = if ((rounded ^ shifted) & rounded) < 0 {
        // on overflow, saturate to i32::MAX (or i32::MIN if the input was negative)
        0x7fff_ffffi32.wrapping_add(i32::from(shifted < 0))
    } else {
        rounded
    };
    // mask off the low word
    sat & 0xffff_0000u32 as i32
}

/// (49 B, leaf): the same shift, **truncating** (no rounding, no saturation),
/// masked to the high word.
pub(crate) fn shift_trunc(v: i32, shift: i32) -> i32 {
    shift_by(v, shift) & 0xffff_0000u32 as i32
}

/// (119 B): splits `v` into a rounded high word and the residual left over, via
/// [`shift_round`].
///
/// * `hi`  = `sar(shift_round(sx16(v) << 16, shift), 16)`
/// * `res` = `((sx16(hi) << 16) - trunc_shift(sx16(v) << 16, shift)) << 15 >> 16`
///
/// i.e. `res` is what the *rounding* shift threw away, re-normalised into a
/// Q15 word. The DLL inlines the truncating shift here rather than calling
/// [`shift_trunc`]; it is the identical three-way branch, so [`shift_by`] serves
/// both.
pub(crate) fn split_hi_res(v: i32, shift: i32) -> (i16, i16) {
    // sign-extend the low word of v and shift it into the high word
    let v_word = shl(sx16(v), 16);
    // rounded shift via shift_round, then take the high word (>> 16) as hi
    let hi = sar(shift_round(v_word, shift), 16) as i16;
    // place hi back into the high word: sx16(hi) << 16
    let hi_word = shl(sx16(hi as i32), 16);
    // the truncating three-way shift of v_word again (sub); the residual is
    // (hi_word - sub) << 15 >> 16
    let sub = shift_by(v_word, shift);
    let res = sar(shl(hi_word.wrapping_sub(sub), 15), 16) as i16;
    (hi, res)
}
