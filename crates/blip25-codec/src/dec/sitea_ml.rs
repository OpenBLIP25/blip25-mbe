//! Site-A (mid-frame) parameter interpolation and its callees.
//!
//! # What site A is
//!
//! Every codec frame the DLL builds a second, "mid-frame" parameter struct
//! (site A) by interpolating between the current frame's decoded params
//! (site B, "cur") and the *previous* frame's params ("prev"). Site A drives
//! the amplitude of the first half of each subframe, so it gates ~half of all
//! output samples.
//!
//! # The struct (272 words at `pstruct`, 128 at the site-A scratch)
//!
//! | byte | word | field |
//! |---|---|---|
//! | `+0x0` | 0 | frametype (== 1 on voice frames) |
//! | `+0x2` | 1 | magic (== 255) |
//! | `+0x4` | 2 | `L` |
//! | `+0x8` | 4,5 | voicing mask (dword) |
//! | `+0xc` | 6 | `omega0` |
//! | `+0x10` | 8.. | `M_l[0..56]` |
//!
//! # Where "prev" lives
//!
//! The per-frame decode routine's tail runs three steps:
//!
//! ```text
//! decode params into `cur`
//! site A <- interp(cur, prev)
//! history <- copy of cur
//! ```
//!
//! The `prev` operand is a **separate history struct**, not `pstruct`. The
//! end-of-frame copy writes `cur` into it at the *end* of every frame, so at
//! frame `s` the "prev" operand holds frame `s-1`'s decoded `M_l` verbatim. The
//! copy is guarded by `src.frametype == 1`, which holds on voice frames, so the
//! history is never stale here.
//!
//! This matters because `pstruct+0x10` *itself* is **not** a usable "prev": it
//! is clobbered between frames by the downstream linear-amp stage. The
//! pre-frame `M_l` never equals the previous frame's post `M_l`, and it is
//! non-negative throughout while the post array is negative on ~53% of words —
//! a different quantity entirely. `L` and `omega0` *do* carry over in that
//! slot, which is what makes it look like a plain carry-over.
//!
//! # The mechanism
//!
//! ```text
//! buf1 = resample(cur.ml,  cur.omega0  -> A.omega0, A.L)
//! buf2 = resample(prev.ml, prev.omega0 -> A.omega0, A.L)
//! for i in 0..A.L:
//!     A.ml[i] = clamp((buf2[i] + buf1[i]) >> 1, -30719, 30719)
//! A.ml[A.L..56] = A.ml[A.L-1]
//! ```
//!
//! The `>>1` is done as a 64-bit `shrd` on the operands shifted left 16, which
//! is exactly an arithmetic (floor) `>>1` of the sum.
//!
//! # Why `(prev + cur) >> 1` on a shared grid is wrong
//!
//! `A.omega0` is the **geometric** mean of `cur` and `prev` — a block-float
//! sqrt of `2*cur.omega0*prev.omega0` with exponents `-8, -4`, computed via a
//! Q15 polynomial sqrt that is off by ~15 even when `cur.omega0 ==
//! prev.omega0`. So the resample ratio is never exactly 1.0 and the grid always
//! drifts slightly; the resampler's exact fixed-point form is what carries the
//! accuracy, not any hidden or stale state operand.
//!
//! The `+-30719` clamp never triggers on real frames. It is ported because the
//! DLL encodes it, not because it is exercised.

/// 16/32-bit truncation helper used throughout.
#[inline]
fn s16(v: i32) -> i16 {
    v as i16
}

/// Re-exported shape of the ratio helper. The lib already owns this primitive in
/// `crate::shared::cepstral_normalize`; it is private there, so the identical
/// 5-line arithmetic is repeated rather than widening that module's API.
fn bfp_reciprocal_ratio(a: i32, b: i16) -> i32 {
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
    if a_abs >= b_abs && b_abs != 0 {
        (sign as i64 * ((a_abs / b_abs) >> 1)) as i32
    } else {
        0
    }
}

/// Q16 resample ratio `a_omega0 / src_omega0`.
///
/// Block-float: normalizes the denominator left so `bsr == 30`, divides via
/// the ratio helper, then re-applies the net exponent. Returns exactly `0x10000`
/// when the two grids match (the DLL short-circuits this).
pub(crate) fn resample_ratio_q16(a_om: i16, src_om: i16) -> i32 {
    if (a_om as u16) == (src_om as u16) {
        return 0x10000;
    }
    let den: i32 = (src_om as i32) << 16;
    let norm_sh: i32 = if den == 0 {
        0
    } else {
        let mag = if den < 0 { !den } else { den };
        let bsr = 31 - (mag as u32).leading_zeros() as i32;
        ((0x1e - bsr) as u16) as i32
    };
    let den_norm: i32 = ((den as u32) << (norm_sh & 31)) as i32 >> 16;
    // Only the low 16 bits of `-4 - norm_sh` survive into the final shift, so
    // the `as u16` wrap is intentional.
    let den_sh: i32 = ((-4i32).wrapping_sub(norm_sh) as u16) as i32;
    let den16: i16 = den_norm as i16;

    let mut num: i32 = (a_om as i32) << 16;
    let mut num_sh: i32 = -4;
    if a_om >= den16 {
        num >>= 1;
        num_sh = -3;
    }
    let r: i32 = (bfp_reciprocal_ratio(num, den16) as i16 as i32) << 16;
    let sh: i16 = (num_sh.wrapping_sub(den_sh).wrapping_sub(0xf)) as i16;
    if sh >= 0 {
        ((r as u32) << (sh & 31)) as i32
    } else if sh > -31 {
        r >> ((-sh) & 31)
    } else {
        r >> 31
    }
}

/// Shift the source envelope right by one slot (the `n = 60` case).
///
/// `dst[0] = src[0]; dst[1..=56] = src[0..56]; dst[57..60] = src[55]`.
/// This one-slot shift is what implements the `(j+1)*ratio - 1` index mapping:
/// the accumulator starts at `ratio` (i.e. `j+1`) and the shift subtracts the 1.
pub(crate) fn shift_magnitudes_right(src: &[i16; 56]) -> [i16; 60] {
    let mut buf = [0i16; 60];
    buf[0] = src[0];
    buf[1..=56].copy_from_slice(&src[..56]);
    for b in buf.iter_mut().take(60).skip(57) {
        *b = src[55];
    }
    buf
}

/// Resample `src_ml` from grid `src_om` onto grid `a_om`,
/// emitting `a_l` samples. Q15 linear interpolation between adjacent slots of
/// the shifted buffer, with the DLL's `-32768 x -32768` and 32-bit
/// signed-overflow saturations reproduced.
pub(crate) fn resample_magnitudes(
    a_om: i16,
    src_om: i16,
    src_ml: &[i16; 56],
    a_l: usize,
) -> Vec<i16> {
    let buf = shift_magnitudes_right(src_ml);
    let ratio = resample_ratio_q16(a_om, src_om);
    let mut acc: i32 = ratio;
    let mut out = Vec::with_capacity(a_l);
    for _ in 0..a_l {
        let idx: i16 = ((acc >> 16) as u16) as i16;
        let frac_q16: i32 = acc.wrapping_sub((idx as i32) << 16);
        let frac: i16 = (((frac_q16 as u32) << 15) as i32 >> 16) as i16;
        let w1: i16 = (0x7fffi32 - frac as i32) as i16;

        let i0 = if idx > 59 { 59usize } else { idx as usize };
        let i1 = if idx + 1 > 59 {
            59usize
        } else {
            (idx + 1) as usize
        };

        let a: i32 = (buf[i0] as i32).wrapping_mul(w1 as i32).wrapping_mul(2);
        let b: i32 = if frac == -32768 && buf[i1] == -32768 {
            0x7fffffff
        } else {
            (buf[i1] as i32).wrapping_mul(frac as i32).wrapping_mul(2)
        };
        // Add the two products with the DLL's explicit signed-overflow probe.
        let sum = match a.checked_add(b) {
            Some(v) => v,
            None => 0x7fffffffi32.wrapping_add(if a < 0 { 1 } else { 0 }),
        };
        out.push((sum >> 16) as i16);
        acc = acc.wrapping_add(ratio);
    }
    out
}

/// Build site A's `M_l` from the current and previous structs.
///
/// `a_om`/`a_l` are site A's own grid (the geometric mean of the two
/// `omega0`s, and the `L` derived from that). Returns the full padded 56-word array.
pub(crate) fn build_sitea_magnitudes(
    a_om: i16,
    a_l: usize,
    cur_om: i16,
    cur_ml: &[i16; 56],
    prev_om: i16,
    prev_ml: &[i16; 56],
) -> [i16; 56] {
    let buf1 = resample_magnitudes(a_om, cur_om, cur_ml, a_l);
    let buf2 = resample_magnitudes(a_om, prev_om, prev_ml, a_l);
    let mut ml = [0i16; 56];
    for i in 0..a_l {
        // 64-bit shrd of both operands <<16 == arithmetic (floor) >>1 of the sum.
        let v = (buf2[i] as i32 + buf1[i] as i32) >> 1;
        ml[i] = s16(v).clamp(-30719, 30719);
    }
    if a_l > 0 {
        for j in a_l..56 {
            ml[j] = ml[a_l - 1];
        }
    }
    ml
}
