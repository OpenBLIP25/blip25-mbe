//! The decoder's voiced-OLA **generator chain** -- the per-harmonic amplitude/
//! shift generator and the placement loop that drives band synthesis.
//!
//! Chain shape:
//!
//! ```text
//!   place_harmonics (154 B)  -> gen_amp_shift_760 (799 B) amp/shift generator
//!                            -> band_synth (430 B)
//!   band_synth               -> split_hi_res (dec::ola_shift, ported)
//!                            -> split_int_frac (186 B) int/frac split
//!                            -> ola_band_oscillator (dec::ola_osc, ported) x3
//!   split_int_frac           -> bfp_add / bfp_divide / shift_trunc
//!   gen_amp_shift_760        -> bfp_add / bfp_divide / gamma_poly_bfp_eval
//! ```
//!
//! **Domain limit:** the DLL faults for `a4 < 0` in the amp/shift generator.
//! `a4` is `>= 0` on every real frame, so this is outside the reachable domain;
//! this port mirrors the DLL only on `a4 >= 0` and makes no claim below it.

use crate::shared::atan2_bfp_divide::bfp_divide;
use crate::shared::bfp_add::bfp_add;
use crate::shared::gamma_poly::gamma_poly_bfp_eval;

#[inline]
pub(crate) fn sx16(v: i32) -> i32 {
    v as i16 as i32
}

#[inline]
pub(crate) fn zx16(v: i32) -> i32 {
    (v as u32 & 0xffff) as i32
}

/// Logical left shift by `s`; x86 masks the shift count to 5 bits.
#[inline]
pub(crate) fn shl(v: i32, s: i32) -> i32 {
    ((v as u32) << ((s as u32) & 31)) as i32
}

/// Arithmetic right shift by `s`; x86 masks the shift count to 5 bits.
#[inline]
pub(crate) fn sar(v: i32, s: i32) -> i32 {
    v >> ((s as u32) & 31)
}

/// The `bsr`-based normalization idiom that appears **nine** times inside
/// `gen_amp_shift_760` and once in `split_int_frac`. Classic `norm_l`: the
/// count of redundant sign bits.
///
/// Note the `bsr` zero quirk: for `v == -1` the `not` yields 0 and `bsr` with a
/// zero source leaves its destination **unchanged** -- and the destination
/// already holds that 0 -- so the idiom yields `0x1e - 0 = 30`. Modelled
/// explicitly below rather than left to chance.
#[inline]
pub(crate) fn norm_l(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let t = if v < 0 { !v } else { v };
    let bsr = if t == 0 {
        0 // `bsr` with src==0: dst unchanged, and dst was the 0 itself.
    } else {
        31 - t.leading_zeros() as i32
    };
    zx16(0x1e - bsr)
}

/// The three-way sign-dependent shift used in `gen_amp_shift_760` and
/// `split_int_frac`: left shift when the count is non-negative, saturate to a
/// full arithmetic right shift of 31 once the count reaches -31, otherwise
/// arithmetic right shift by `-count`.
#[inline]
pub(crate) fn shift_sat(v: i32, sh: i32) -> i32 {
    let sh16 = sx16(sh);
    if sh16 >= 0 {
        shl(v, sh16)
    } else if sh16 <= -31 {
        sar(v, 31)
    } else {
        sar(v, -sh16)
    }
}

/// Faithful port of the 799-byte amp/shift generator: fills `buf_a[k]`
/// (mantissa/amp) and `buf_b[k]` (exponent/shift) for `k` in `0..count`.
///
/// ABI recovered by tracking `esp` through every push of the sole call site in
/// `place_harmonics`: 7 dwords popped => 7 args.
///
/// The DLL's 32-bit exponent scratch slot (which aliases the `a7` argument
/// slot) is modelled faithfully as `scratch: i32` even though `bfp_add`,
/// `bfp_divide`, and `gamma_poly_bfp_eval` each write only its **low 16 bits** --
/// the stale upper half provably cannot reach any consumer, since every
/// subsequent use is an add/sub (no low-from-high carry) followed by a
/// 16-bit-wide read.
pub(crate) fn gen_amp_shift_760(
    buf_a: &mut [i16],
    buf_b: &mut [i16],
    a3: i32,
    a4: i32,
    a5: i32,
    a6: i32,
    count: i32,
) {
    // count is read as i16.
    let n = sx16(count);
    if n <= 0 {
        return;
    }

    // loop-invariant setup.
    let a4s = sx16(a4);
    let a6s = sx16(a6);
    let a5s = sx16(a5);
    let two_a4sq = a4s.wrapping_mul(a4s).wrapping_mul(2);
    let two_a4a6 = a4s.wrapping_mul(a6s).wrapping_mul(2);
    let two_a6sq = a6s.wrapping_mul(a6s).wrapping_mul(2);
    let clamp_ceil = shl(a6s, 16); // the clamp ceiling

    for k in 1..=n {
        // ---- v1 = (k<<16) - a3 -----------------------------
        let v1 = shl(sx16(k), 16).wrapping_sub(a3);
        let n1 = norm_l(v1);
        let m1 = sar(shl(v1, n1), 16);
        let e1 = zx16(0xf - n1);

        // ---- v2 = 2 * m1 * a6 ------------------------------
        let v2 = sx16(m1).wrapping_mul(a6s).wrapping_mul(2);
        let n2 = norm_l(v2);
        let m2 = sar(shl(v2, n2), 16);
        // e2 accumulates (0x10 - n2) into the exponent -- NOT zero-extended.
        let e2 = e1.wrapping_add(0x10 - n2);
        let m2u = zx16(m2);

        // ---- normalize 2*a4^2 ------------------------------
        let n3 = norm_l(two_a4sq);
        let mut ea = zx16(-8 - n3);
        let m3u = zx16(sar(shl(two_a4sq, n3), 16));

        // ---- normalize 2*a6^2 ------------------------------
        let n4 = norm_l(two_a6sq);
        // ea accumulates (0x1e - n4) -- NOT zero-extended.
        ea = ea.wrapping_add(0x1e - n4);
        let m4 = sar(shl(two_a6sq, n4), 16);

        // ---- v5 = 2 * m4 * m3 ------------------------------
        let v5 = sx16(m4).wrapping_mul(sx16(m3u)).wrapping_mul(2);
        let n5 = norm_l(v5);
        ea = ea.wrapping_sub(n5);
        let m5u = zx16(sar(shl(v5, n5), 16));

        // ---- v6 = 2 * a5 * m2 ------------------------------
        let v6 = a5s.wrapping_mul(sx16(m2u)).wrapping_mul(2);
        let n6 = norm_l(v6);
        // the 32-bit scratch store.
        let mut scratch: i32 = ea;
        let m6 = sar(shl(v6, n6), 16);
        let e6 = e2.wrapping_sub(n6).wrapping_sub(4);

        // ---- bfp_add(m5, ea, m6, e6) -----------------------
        let (r1_m, r1_e) = bfp_add(m5u as i16, ea as i16, m6 as i16, e6 as i16);
        scratch = (scratch & !0xffff) | (r1_e as u16 as i32); // WORD write

        // ---- gamma_poly_bfp_eval poly-sqrt, exponent in/out ---------------
        let (r2, new_exp) = gamma_poly_bfp_eval(shl(sx16(r1_m as i32), 16), scratch as i16);
        scratch = (scratch & !0xffff) | (new_exp as u16 as i32); // WORD write
        let sqrt_res = r2 as i32;

        // ---- normalize the sqrt result ---------------------
        let n7 = norm_l(sqrt_res);
        // full 32-bit (DWORD) scratch read, minus the norm.
        let mut exp = scratch.wrapping_sub(n7);
        let m7u = zx16(sar(shl(sqrt_res, n7), 16));
        scratch = exp; // full 32-bit (DWORD) scratch store.

        // ---- bfp_add(m8, 11-n8, m7, exp) -------------------
        let n8 = norm_l(two_a4a6);
        let m8 = sar(shl(two_a4a6, n8), 16);
        let (r3_m, r3_e) = bfp_add(m8 as i16, (0xb - n8) as i16, m7u as i16, exp as i16);
        scratch = (scratch & !0xffff) | (r3_e as u16 as i32); // WORD write

        // ---- bfp_divide(m2<<16, e2, r3, scratch) -----------
        // `None` is the DLL's divide-by-zero; it would fault. Mirror by
        // leaving the (amp,shift) pair at the divide's degenerate 0.
        let (r4_m, r4_e) = bfp_divide(shl(m2, 16), e2, zx16(r3_m as i32), scratch).unwrap_or_default();
        scratch = (scratch & !0xffff) | (r4_e as u16 as i32); // WORD write
        exp = scratch; // full 32-bit (DWORD) scratch read.

        let mut amp_out = zx16(r4_m as i32);
        let mut shift_out = exp;

        // ---- scale and clamp against a6<<16 ----------------
        let scaled = shift_sat(shl(sx16(amp_out), 16), exp.wrapping_sub(0xf));
        // skip the clamp when scaled - clamp_ceil is negative.
        if scaled.wrapping_sub(clamp_ceil) >= 0 {
            let n9 = norm_l(clamp_ceil);
            shift_out = zx16(0xf - n9);
            amp_out = zx16(sar(shl(clamp_ceil, n9), 16));
        }

        // ---- store the pair --------------------------------
        buf_a[(k - 1) as usize] = amp_out as u16 as i16;
        buf_b[(k - 1) as usize] = shift_out as u16 as i16;
    }
}

/// Faithful port of the 186-byte integer/fraction split: splits a BFP value
/// `(a3, a4)`, scaled by `1/a5`, into an integer part and a Q15 fractional
/// remainder.
///
/// ABI from `band_synth`'s call site (5 args; the stack cleanup is shared with
/// the `split_hi_res` call): `(out_int*, out_frac*, a3, a4, a5)`.
///
/// Returns `(out_int, out_frac)`.
///
/// **Domain:** `a5 == 0` makes the DLL divide by zero inside
/// `bfp_divide` and fault; this port returns `(0, 0)` there and makes no claim.
pub(crate) fn split_int_frac(a3: i32, a4: i32, a5: i32) -> (i16, i16) {
    // normalize a5<<16
    let a5_sh = shl(sx16(a5), 16);
    let n = norm_l(a5_sh);
    let m = sar(shl(a5_sh, n), 16);

    // bfp_divide(0x40000000, 1, m, -4 - n)
    let Some((q, qe)) = bfp_divide(0x4000_0000, 1, m, -4 - n) else {
        return (0, 0);
    };

    // bfp_add(a3, a4, q, qe)
    let (s, se) = bfp_add(a3 as i16, a4 as i16, q, qe);

    // sum_sh = s << 16, then shift_trunc(sum_sh, se)
    let sum_sh = shl(sx16(s as i32), 16);
    let t = crate::dec::ola_shift::shift_trunc(sum_sh, se as i32);
    // out_int is the high word of t.
    let out_int = sar(t, 16) as u16 as i16;

    // the same three-way shift, on sum_sh, by (se - 15)
    let scaled = shift_sat(sum_sh, (se as i32).wrapping_sub(0xf));
    // fractional remainder in Q15.
    let diff = scaled.wrapping_sub(shl(sx16(out_int as i32), 16));
    let out_frac = sar(shl(diff, 15), 16) as u16 as i16;
    (out_int, out_frac)
}

/// Faithful port of the 430-byte band synthesiser: synthesises one band into
/// the `out` accumulator via three `ola_band_oscillator` oscillator passes.
///
/// ABI from `place_harmonics`'s loop (6 args): `(out, amp, shift, a4, table,
/// a6)` where `(amp, shift)` is one `(bufA[i], bufB[i])` pair produced by
/// [`gen_amp_shift_760`]. The ABI is recovered by tracking `esp` through every
/// push.
///
/// # Domain limit
///
/// `a4 == 0` makes `split_int_frac` divide by zero inside `bfp_divide` and the
/// DLL **faults**. This port returns early there via [`split_int_frac`]'s
/// `(0, 0)` and makes no claim below it.
///
/// # Unobservable details
///
/// * **The `lb < -16` threshold is UNOBSERVABLE.** Mutating `lb < -16` to
///   `lb <= -16` changes nothing on any input, and this is not a coverage hole
///   -- it is an algebraic identity: at `lb == -16` one arm computes
///   `taper_lo - (-16 << 16)` and the other computes `taper_lo + 0x100000`,
///   **and those are equal**. A search over 600x8x5x2 inputs (5 of which hit
///   `lb == -16` exactly) found **zero** differing inputs. An off-by-one on this
///   threshold is therefore harmless by construction; the clamp behaviour for
///   `lb < -16` IS pinned (the sweep reaches -64/-32768).
/// * **A call-site guard on oscillator call 3 is likewise unobservable:**
///   `ola_band_oscillator` no-ops internally for `count <= 0`, so guarding it is
///   output-equivalent. Kept unguarded to match the bytes.
///
/// The cursor index is always `>= 0`: it is `sx16(lb)` on the `lb >= 0` arm and
/// forced to 0 on the `lb < 0` arm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn band_synth(out: &mut [i32], amp: i32, shift: i32, a4: i32, table: &[i16], a6: i32) {
    // split_hi_res(amp, shift) -> (hi, res)
    let (lb, la) = crate::dec::ola_shift::split_hi_res(amp, shift);
    // split_int_frac(amp, shift, a4) -> (int, frac)
    let (ld, lc) = split_int_frac(amp, shift, a4);

    // pos = min(ld, 0xa6)
    let mut pos = sx16(ld as i32);
    if 0xa6 < pos {
        pos = 0xa6;
    }
    // pos_end = (pos + 0x10 > 0xa6) ? 0xa6 : zx16(pos + 0x10)
    let pos_end = if pos + 0x10 > 0xa6 {
        0xa6
    } else {
        zx16(pos + 0x10)
    };
    // return early when pos_end <= 0
    if sx16(pos_end) <= 0 {
        return;
    }

    let la_s = sx16(la as i32);
    let lb_s = sx16(lb as i32);
    let a4_s = sx16(a4);
    let a4_u = zx16(a4_s);

    // local_a_dw = ((lc<<16)>>15) + 0xf0000
    let local_a_dw = sar(shl(sx16(lc as i32), 16), 0xf).wrapping_add(0xf0000);
    // taper_lo = (la<<16)>>15
    let mut taper_lo = sar(shl(la_s, 16), 0xf);
    // local_c = (2 * a4 * la) >> 11
    let mut local_c = a4_s.wrapping_mul(la_s).wrapping_mul(2);
    local_c = sar(local_c, 0xb);

    let mut hidx;
    let taper_hi;
    if lb_s >= 0 {
        hidx = zx16(lb as i32);
        taper_hi = local_a_dw;
    } else {
        // local_c += (sx16(-lb) * a4) << 5
        local_c = local_c.wrapping_add(shl(sx16(lb_s.wrapping_neg()).wrapping_mul(a4_s), 5));
        // lb < -16 branch (the threshold is algebraically unobservable; see doc)
        if lb_s < -16 {
            taper_lo = taper_lo.wrapping_add(0x100000);
        } else {
            taper_lo = taper_lo.wrapping_sub(shl(lb_s, 16));
        }
        // when pos < 0
        let mut e = local_a_dw;
        if sx16(pos) < 0 {
            e = e.wrapping_add(shl(sx16(pos + 1), 16));
        }
        taper_hi = e;
        hidx = 0;
    }

    // local_a = zx16(a6 - 16)
    let local_a = zx16(a6.wrapping_sub(0x10));
    // cursor index = sx16(hidx) (always >= 0)
    let mut cursor = sx16(hidx) as usize;
    let mut phase = local_c;

    // n1 = zx16(lb - hidx + 16), guarded below
    let n1 = zx16(sx16(lb as i32).wrapping_sub(hidx).wrapping_add(0x10));
    if sx16(n1) > 0 {
        hidx = hidx.wrapping_add(n1);
        crate::dec::ola_osc::ola_band_oscillator(
            &mut phase,
            table,
            a4_u,
            taper_lo,
            out,
            &mut cursor,
            local_a,
            1,
            n1,
        );
    }

    // pos = pos - hidx + 1, guarded below
    pos = pos.wrapping_sub(hidx).wrapping_add(1);
    let n2 = zx16(pos);
    if sx16(n2) > 0 {
        hidx = hidx.wrapping_add(n2);
        crate::dec::ola_osc::ola_band_oscillator(
            &mut phase,
            table,
            a4_u,
            0,
            out,
            &mut cursor,
            local_a,
            0,
            n2,
        );
    }

    // n3 = pos_end - hidx + 1 -- call 3 is UNGUARDED.
    let n3 = pos_end.wrapping_sub(hidx).wrapping_add(1);
    crate::dec::ola_osc::ola_band_oscillator(
        &mut phase,
        table,
        a4_u,
        taper_hi,
        out,
        &mut cursor,
        local_a,
        -1,
        n3,
    );
}

/// Faithful port of the 154-byte placement loop.
///
/// Generates `count` `(amp, shift)` pairs with [`gen_amp_shift_760`], drives
/// [`band_synth`] once per pair, and stores the LAST pair back through
/// the two out-params.
///
/// **The stack-pop trap:** the tail writes twice through the same stack slot,
/// but a `pop` sits between the two reads -- `esp` shifts by 4, so the first
/// write resolves to `arg2` and the second to `arg3`. They are DIFFERENT
/// out-pointers (amp and shift). Read without accounting for the pop, the
/// first write looks dead. Argument names follow the DLL's own slots.
/// Push-order mapping, both recovered by tracking `esp` (7 args to
/// `gen_amp_shift_760`; 6 args to `band_synth`):
///
/// ```text
///   gen_amp_shift_760 <- (bufA, bufB, a7, a9, a8, a10, a11=count)
///   band_synth        <- (a1,   bufA[i], bufB[i], a4, a5=table, a6)
/// ```
///
/// # KNOWN DIVERGENCE at `count <= 0` -- deliberate, and out of domain
///
/// The tail is past the loop's skip-branch target, so **it runs
/// unconditionally**: at `count <= 0` the DLL reads `bufA[sx16(count) - 1]`,
/// which for `count==0` is `bufA[-1]` -- **2 bytes BELOW the local block, in
/// `gen_amp_shift_760`'s dead stack frame**. It is not a function of this fn's
/// arguments (the same call with `a10=80` yields `(238,0)` and with `a10=166`
/// yields `(238,271)`; what it reads is the generator's leftover residue). The
/// guard below therefore leaves the out-params untouched instead of imitating
/// stack garbage. `a11` is in `[1..4]` on every real frame, so `count <= 0` is
/// unreachable in the decoder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_harmonics(
    out: &mut [i32],     // a1
    out_amp: &mut i16,   // a2 (see the stack-pop note above)
    out_shift: &mut i16, // a3
    a4: i32,
    table: &[i16], // a5
    a6: i32,
    a7: i32,
    a8: i32,
    a9: i32,
    a10: i32,
    count: i32, // a11
) {
    // two 20-word locals (buf_a, buf_b).
    let mut buf_a = [0i16; 20];
    let mut buf_b = [0i16; 20];

    // gen_amp_shift_760(bufA, bufB, a7, a9, a8, a10, count)
    gen_amp_shift_760(&mut buf_a, &mut buf_b, a7, a9, a8, a10, count);

    // skip the loop when count <= 0.
    let n = sx16(count);
    for i in 0..n.max(0) {
        // band_synth(a1, bufA[i], bufB[i], a4, a5, a6)
        band_synth(
            out,
            buf_a[i as usize] as i32,
            buf_b[i as usize] as i32,
            a4,
            table,
            a6,
        );
    }

    // store bufA[count-1] / bufB[count-1] back.
    let last = sx16(count) - 1;
    if last >= 0 {
        *out_amp = buf_a[last as usize];
        *out_shift = buf_b[last as usize];
    }
}

/// Faithful port of the 483-byte **spectrum generator + IFFT dispatcher** at
/// the head of the voiced-OLA chain: it turns `(amp[], resid[], phase_step)`
/// into a complex half-spectrum in `a1`, then hands that spectrum to the IFFT
/// ([`crate::dec::unv_frame::array_a_stage2_refine`]).
///
/// # The harmonic-numbering subtlety (`j` vs `m`)
///
/// The DLL keeps **two** counters and they are off by one. A pointer walks the
/// arrays from `&a6[a9]` while the running index is seeded with `zx16(a9 + 1)`.
/// So for array slot `j`:
///
/// ```text
///   amp   = a6[j]      resid = a4[j]      mask = a5[j]
///   m     = j + 1                     <- harmonic number
///   phase = f(a3 * m, a4[j])          <- multiplier is m, NOT j
///   a1[2m], a1[2m+1] = re, im         <- bin is m, NOT j  (bin 0 is DC)
/// ```
///
/// Missing this writes every harmonic into the wrong bin while still looking
/// structurally plausible, so it is called out rather than left implicit.
///
/// # Shape
///
/// 1. `r = block_exp_masked(a6, a5, a9, a8, a10)` -- block exponent of the
///    masked amp array. The 5-arg push order maps exactly onto the
///    already-ported [`crate::dec::ola_leaves::block_exp_masked`]`(arr, mask,
///    lo, hi, mv)`, and `mv` is the same `a10` the in-loop mask test compares
///    against.
/// 2. Word-fill 256 words of `a1` with 0 (inlined here).
/// 3. For `j` in `a9..a8`, **iff `a5[j] == a10`**: compute the phase index and
///    write `(amp*cos, amp*sin)` into bin `m`. `count++`.
/// 4. `*a2 = a7 + 0x17 + r`; if `count > 0`, `*a2 = ` the IFFT of the spectrum,
///    else `*a2 = -32`.
/// 5. `a1[256] = a1[0]` -- the wrap-around copy.
///
/// The phase index is `((m*ps + 0x20 + resid)/64) & 0x1ff` -- the same formula
/// the downstream consumer uses.
///
/// The sine index is `(phase - 128) & 0x1ff`, obtained by the DLL's
/// `shl 16 / sub 0x7f0001 / sar 16 / and 0x1ff` idiom: the `-1` in `0x7f0001`
/// makes the arithmetic shift floor, giving `phase - 0x7f - 1`. A quarter turn
/// of a 512-point table = 128 => `cos` and `sin`. Verified at `phase == 0`:
/// index `384`, `T1[384] == 0 == sin(0)`.
///
/// # Evidence class: **STRUCTURAL ONLY -- unverified**
///
/// The arithmetic is read from the bytes, and its phase-index arm agrees with an
/// independently scored formula, but the function as a whole is unscored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_spectrum(
    a1: &mut [i16], // spectrum out, >= 258 words (bins 0..=128 as re/im pairs)
    a2: &mut i16,   // exponent out
    a3: i32,        // phase step
    a4: &[i16],     // resid
    a5: &[i16],     // mask
    a6: &[i16],     // amp
    a7: i32,        // base exponent
    a8: i32,        // hi (exclusive)
    a9: i32,        // lo (inclusive)
    a10: i32,       // mask value
) -> i32 {
    // r = block_exp_masked(a6, a5, a9, a8, a10)
    let r = crate::dec::ola_leaves::block_exp_masked(a6, a5, a9, a8, a10);
    // the result is kept ZERO-extended.
    let local_r = zx16(r);

    // word-fill a1 with 0, count in WORDS.
    for w in a1[..0x100].iter_mut() {
        *w = 0;
    }

    let mut count: i32 = 0;

    // 16-bit signed compare of a9 vs a8.
    if sx16(a9) < sx16(a8) {
        // sh_cnt = -r (32-bit); the three-way shift count is sx16 of its low
        // word. For a block exponent of -7, that is +7 => shift LEFT 7.
        let sh_cnt = (0i32).wrapping_sub(local_r);

        let cos = crate::shared::loudness_transform::cos_twiddle_table();

        // the running index starts at m = a9 + 1, zero-extended.
        let mut m: i32 = zx16(a9.wrapping_add(1));
        // loop trip count = (a8 - a9) & 0xffff, do-while.
        let n = zx16(a8.wrapping_sub(a9));

        for i in 0..n {
            let j = a9.wrapping_add(i);

            // 16-bit compare of a5[j] against the mask value.
            if a5[j as usize] == (a10 as i16) {
                // the phase-index numerator.
                //   resid_sh = sx16(a4[j]) << 16 >> 6   == resid << 10
                //   pidx = ((a3 * m) + 0x20) << 10
                let resid_sh = sar(shl(sx16(a4[j as usize] as i32), 0x10), 6);
                let mut pidx = sx16(a3).wrapping_mul(sx16(m));
                pidx = pidx.wrapping_add(0x20);
                pidx = shl(pidx, 0xa);
                pidx = pidx.wrapping_add(resid_sh);

                // round_residual(pidx, 6)
                pidx = crate::dec::ola_leaves::round_residual(pidx, 6);
                pidx = sar(pidx, 0x10);
                pidx = sx16(pidx);
                pidx = shl(pidx, 0xa);
                // trunc_toward_zero(pidx, 0xf)
                pidx = crate::dec::ola_leaves::trunc_toward_zero(pidx, 0xf);
                pidx = sar(pidx, 0x10);
                let phase = pidx & 0x1ff;

                let amp = a6[j as usize] as i32;

                // real = (cos[phase] * amp) * 2, shifted, >>16
                let mut re = (cos[phase as usize] as i32).wrapping_mul(amp);
                re = re.wrapping_add(re);
                re = shift_sat(re, sh_cnt);
                re = sar(re, 0x10);
                a1[(2 * m) as usize] = re as i16;

                // sine index = (phase - 128) & 0x1ff
                let sidx = sar(shl(phase, 0x10).wrapping_sub(0x7f0001), 0x10) & 0x1ff;
                let mut im = (cos[sidx as usize] as i32).wrapping_mul(amp);
                im = im.wrapping_add(im);
                im = shift_sat(im, sh_cnt);
                im = sar(im, 0x10);
                a1[(2 * m + 1) as usize] = im as i16;

                count = count.wrapping_add(1);
            }
            // the running index advances every iteration, masked or not.
            m = m.wrapping_add(1);
        }
    }

    // *a2 = a7 + 0x17 + r  (r ZERO-extended; the low 16 bits are what gets
    // stored, so this is equivalent mod 2^16).
    let exp_out = a7.wrapping_add(0x17).wrapping_add(local_r);
    *a2 = exp_out as i16;

    // 16-bit test on the count.
    let res: i32 = if sx16(count) > 0 {
        // array_a_stage2_refine(a1, *a2, 8, 0) -- the IFFT. Use the arg5=arg4 sibling:
        // the transform passes its own arg4 through as the inner routine's arg5,
        // NOT a hardcoded 1. With arg4=0 (this call) that selects the inner
        // routine's arg5==0 head; `array_a_stage2_refine_dispatch` hardcodes arg5=1 and diverges
        // here.
        crate::dec::unv_frame::array_a_stage2_refine(a1, exp_out as i16, 8, 0) as i32
    } else {
        -32
    };
    *a2 = res as i16;

    // a1[256] = a1[0] -- the wrap-around copy.
    a1[0x100] = a1[0];

    // the return is the 16-bit count.
    sx16(count)
}
