//! The voiced-OLA **band oscillator** (`ola_band_oscillator`) and its single callee,
//! a pure cos-interpolation leaf (`osc_cos_interp`).
//!
//! The oscillator is the sample-producing kernel of the OLA generator closure:
//! it walks a phase accumulator across a 256-entry wavetable with linear
//! interpolation, optionally applies a raised-cosine taper, block-shifts the
//! result, and **saturating-accumulates** it into the output through a
//! caller-held cursor.
//!
//! # ABI
//!
//! Derived by tracking `esp` through all five pushes (`ecx`,`edi`,`ebx`,`ebp`,
//! `esi`), so the same `[esp+0x18]` denotes **different** args at different
//! points (arg5 near entry, arg1 after the pushes):
//!
//! ```text
//! void __cdecl ola_band_oscillator(
//!                  int *phase,   // [E+04] in/out, advanced by incr per sample
//!                  short *table, // [E+08] >= 257 words -- see NOTE below
//!                  int freq,     // [E+0c] i16
//!                  int taperseed,// [E+10] BY VALUE (mutated in its own slot)
//!                  int **cursor, // [E+14] in/out, advanced by `count` dwords
//!                  int shift,    // [E+18] i16
//!                  int dir,      // [E+1c] i16; 0 => taper disabled
//!                  int count);   // [E+20] i16
//! ```
//!
//! Two facts that a naive reading gets wrong:
//!
//! * **`taperseed` is passed BY VALUE.** The routine does
//!   `add DWORD PTR [esp+0x24],eax` -- it accumulates into its **own argument
//!   stack slot**, which the caller never reads back. It is a local, not an
//!   in/out parameter. (It likewise reuses the `count` arg slot as a scratch
//!   for `-shift`, after `count` has been copied into `ebp`.)
//! * **`table[idx+1]` is a RAW `+2` read** (`movsx eax,WORD PTR [ecx+eax*2+0x2]`).
//!   It is **NOT** masked `&0xff`. At `idx == 255` the DLL reads `table[256]` --
//!   the word physically following the wavetable -- not `table[0]`. Modelling
//!   it as a period-256 wrap is WRONG; hence the `>= 257` length requirement.
//!
//! # `count <= 0` is a NO-OP, not a guarded call
//!
//! The entry `cmp dx,cx / jge` (with `dx` zeroed) skips the loop but still
//! falls into `mov DWORD PTR [eax],edi` -- the cursor is stored back
//! **unchanged**. The DLL calls the oscillator **unconditionally 3x per callid**
//! and relies on this internal no-op; a nontrivial fraction of real calls have
//! `count == 0`. This port therefore no-ops internally; callers must NOT add a
//! `count > 0` guard, because a call-site guard is a bug.
//!
//! # What real frames cannot distinguish
//!
//! * **The shift legs.** Real frames span `shift` in `[-19 ..= 0]`, the only
//!   non-negative value being `0` -- where `shl esi,cl` is a shift-by-zero
//!   no-op -- and never reach `shift <= -31`, so the `sar esi,0x1f` collapse
//!   leg never executes on real data. Both legs are pinned by a direct sweep
//!   instead.
//! * **The RAW `+2` read is DEGENERATE on real data.** Real frames do reach
//!   `idx == 255`, but every captured wavetable has `table[256] == table[0]`
//!   (the generator writes the wrap word), so the WRONG `&0xff`-wrap model
//!   scores identically there. Only a deliberately non-periodic table separates
//!   the two, which is why the `>= 257` length requirement above is not
//!   negotiable.
//! * **The cos-interp leaf's `cwde` is unpinnable by any input**, proven
//!   exhaustively over all 65536 distinct inputs: its evidence class is
//!   enumeration plus structure, never a score.
//! * **The `-31`/`-32` boundary is degenerate at `-31` exactly**: `neg_shift`
//!   is then 31 and `sar(v, 31 & 31) == sar(v, 0x1f)` -- both arms are the same
//!   expression. Only `shift <= -32` distinguishes them.

use crate::shared::loudness_transform::cos_twiddle_table;

use crate::fixops::dec32::{sar, shl, sx16};
/// Signed-saturating 32-bit add, via the classic `(d^a) & (d^b) < 0` overflow
/// test and `sets dl / add edx,0x7fffffff`.
#[inline]
fn sat_add32(a: i32, b: i32) -> i32 {
    match a.checked_add(b) {
        Some(x) => x,
        None => {
            if b < 0 {
                i32::MIN
            } else {
                i32::MAX
            }
        }
    }
}

/// Interpolated cosine lookup over the `.rdata` cosine table (94 B, leaf).
///
/// `arg` is a Q9.7-ish angle: `sx16(arg) << 16 >> 6` gives an index in the
/// integer high word and a fraction in the low word, then the magnitude is
/// taken (`neg`) and the result linearly interpolated between `T1[idx]` and
/// `T1[(idx+1) & 0x1ff]`.
///
/// **Structural note (evidence class: bytes + range argument, NOT score).**
/// The `angle == i32::MIN` guard (which would clamp to `0x7fffffff`) is
/// **UNREACHABLE**: `angle` is `sar(_, 6)` of a 32-bit value, so
/// `|angle| <= 2^25` and it can never equal `i32::MIN`. Ported faithfully
/// anyway. The `idx` index reaches **512** (not 511) at `sx16(arg) == -32768`,
/// which is why the table must be the 1024-entry generous extraction rather
/// than the nominal 512.
pub(crate) fn osc_cos_interp(arg: i32) -> i32 {
    let cos = cos_twiddle_table();
    // sx16(arg) placed into Q-format: high word = index, low word = fraction
    let mut angle = sar(shl(sx16(arg), 16), 6);
    // take the magnitude (i32::MIN would clamp to 0x7fffffff)
    if angle < 0 {
        angle = if angle == i32::MIN {
            0x7fff_ffff
        } else {
            angle.wrapping_neg()
        };
    }
    let idx = sar(angle, 16) & 0xffff; // high word = table index
    let frac = angle & 0xffff; // low word = interpolation fraction
    let c0 = cos[sx16(idx) as usize] as i32; // T1[idx]
    let c1 = cos[((idx + 1) & 0x1ff) as usize] as i32; // T1[(idx+1) & 0x1ff]
                                                       // linear interpolation: (c0*(0x10000-frac) + c1*frac) >> 16
    let a = c0.wrapping_mul(0x10000 - frac);
    let b = c1.wrapping_mul(frac);
    sar(a.wrapping_add(b), 16)
}

/// The OLA band oscillator (323 B).
///
/// Advances `phase` by `incr = sx16(freq) << 16 >> 11` per sample, reads the
/// wavetable with linear interpolation, optionally applies the `dir`-driven
/// raised-cosine taper `(0x8000 - cos(taperseed)) `, block-shifts by `shift`,
/// and saturating-accumulates into `out[*cursor..]`, advancing `*cursor`.
///
/// `table` must be at least 257 long (see the module doc's RAW `+2` note).
///
/// `count <= 0` is an internal no-op -- **do not add a call-site guard.**
#[allow(clippy::too_many_arguments)]
pub(crate) fn ola_band_oscillator(
    phase: &mut i32,
    table: &[i16],
    freq: i32,
    taperseed: i32,
    out: &mut [i32],
    cursor: &mut usize,
    shift: i32,
    dir: i32,
    count: i32,
) {
    // count is read as i16. The DLL still stores the (unchanged) cursor back
    // after the loop; that is a no-op here.
    let cnt = sx16(count);
    if cnt <= 0 {
        return;
    }
    debug_assert!(table.len() >= 257, "table[idx+1] is a RAW +2 read");

    let sh = sx16(shift); // i16 block-shift amount
    let neg_shift = sh.wrapping_neg(); // -shift, for the right-shift legs
    let incr = sar(shl(sx16(freq), 16), 0xb); // per-sample phase increment
    let mut ts = taperseed; // BY VALUE -- caller never sees the accumulation

    for _ in 0..cnt {
        let ph = *phase;
        let idxw = sar(ph, 16) & 0xffff; // integer part of the 16.16 phase
        *phase = ph.wrapping_add(incr); // advance the phase accumulator

        let idx = (idxw & 0xff) as usize; // wavetable index (0..=255)
        let frac_full = ph.wrapping_sub(shl(sx16(idxw), 16)); // fractional phase
        let frac = sar(shl(frac_full, 0xf), 16) & 0xffff; // fraction scaled to Q15

        let t0 = table[idx] as i32;
        let t1 = table[idx + 1] as i32; // RAW +2 -- NOT (idx+1)&0xff
        let e = sar(shl(t0, 16), 1);
        let a = sar(shl(t1, 16), 1);
        let dt = sx16(sar(a.wrapping_sub(e), 16));
        // linear interpolation between the two table samples
        let mut sample = e.wrapping_add(dt.wrapping_mul(sx16(frac)).wrapping_mul(2));

        // dir is read as its low 16-bit word; 0 disables the taper
        if sx16(dir) != 0 {
            let ta = sar(shl(ts, 0xa), 16);
            let cosv = sx16(osc_cos_interp(ta)); // sign-extend the cos result to i32
            let h = sx16(sar(shl(0x8000 - cosv, 0xf), 16));
            let hi = sx16(sar(sample, 16));
            let lo = sample & 0xffff; // low 16 bits
            sample = sar(lo.wrapping_mul(h), 0xf).wrapping_add(hi.wrapping_mul(h).wrapping_mul(2));
            ts = ts.wrapping_add(shl(sx16(dir), 16)); // advance the taper seed in its own arg slot
        }

        // block-shift: left if sh>=0, arithmetic-right by -sh for sh in (-31,0),
        // else collapse to the sign bit
        let shifted = if sh >= 0 {
            shl(sample, sh as u32)
        } else if sh > -31 {
            sar(sample, neg_shift as u32)
        } else {
            sar(sample, 0x1f)
        };

        out[*cursor] = sat_add32(out[*cursor], shifted);
        *cursor += 1;
    }
}
