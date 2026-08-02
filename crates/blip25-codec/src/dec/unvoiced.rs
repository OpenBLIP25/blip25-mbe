//! The **unvoiced back-end** of the AMBE+2 decoder in the reference vocoder.
//!
//! Five leaves plus the orchestrator:
//!
//! | fn | B | what |
//! |---|---|---|
//! | [`neg_max_unvoiced_amp_norm`] |  91 | `-(norm of max amp over UNVOICED bands)` |
//! | [`scale_shift_array`] | 123 | multiply + three-way shift + `>>16` array writer |
//! | [`array_rms_bfp`] | 200 | RMS `(mantissa, exponent)` of an array |
//! | [`crate::dec::unv_frame::unvoiced_overlap_add_wide`] | 357 | the unvoiced OLA writer |
//! | [`biquad_postfilter`] | 401 | 2-biquad IIR post-filter, 64-bit acc, 6 `i32` state |
//! | [`shape_unvoiced_bands`] | 748 | the unvoiced back-end **orchestrator** (calls all above) |
//!
//! [`shape_unvoiced_bands`]'s **six** scalar/pointer args are pinned by the six
//! pushes at its sole call site; that call site also tears down a ~40 B struct
//! the caller built, **not** this fn's arg block.  It aliases its own arg3/arg4
//! stack slots as scratch (`arg3` -> the `0x11` exponent seed, `arg4` -> a gain
//! mantissa).
//!
//! The unvoiced closure is **self-contained**; every external callee already
//! lives elsewhere in this crate:
//! [`crate::shared::q_energy::q_builder`],
//! [`crate::shared::cepstral_normalize::bfp_reciprocal_ratio`],
//! [`crate::shared::gamma_poly::gamma_poly_bfp_eval`],
//! [`crate::dec::ola_driver::block_denorm`], and the DLL's memcpy / word-fill,
//! inlined here.
//!
//! ## Domain limits and traps
//!
//! * **`bfp_reciprocal_ratio`'s `b == 0` leg is a KNOWN DIVERGENCE, unreachable here.**
//!   The DLL divides by zero and **faults**; the Rust returns 0 behind a guard.
//!   [`array_rms_bfp`] calls it only under an `energy > 0` guard, so `b >= 1`
//!   always at this site.
//! * **[`neg_max_unvoiced_amp_norm`]'s negative-accumulator leg is DEAD CODE,
//!   and that is a proof, not a coverage hole.**  The running max starts at 0
//!   and only ever takes values strictly greater than itself, so the running
//!   max is **>= 0 always** and the non-negative branch always jumps.  No input
//!   can reach the negative leg; a test below asserts it.
//! * **The unvoiced OLA's shift callee is the SATURATING one.**
//!   [`crate::dec::unv_frame::unvoiced_overlap_add_wide`] must use
//!   [`crate::dec::ola_driver::block_denorm`], whose left arm saturates: it
//!   builds `mask = -1 << (31-sh)`, tests the top `sh+1` bits against the sign,
//!   and returns `INT_MIN`/`INT_MAX` on overflow.
//!   [`crate::dec::ola_gen::shift_sat`] has the identical three-way
//!   `test / js / cmp / sar` shape but a plain `shl` left arm, and wiring it
//!   here breaks the OLA while leaving [`scale_shift_array`] -- whose inline
//!   shift really is non-saturating -- exact. Match these by the callee's
//!   bytes, never by shape.
//! * **[`biquad_postfilter`] leaves the 64-bit accumulator RAW across its first
//!   saturation.**  The saturated *high* word is written on all three legs and
//!   **never read**; the accumulator registers are not reloaded.  So the stored
//!   state saturates while the accumulator carries on unsaturated.  Modelled
//!   faithfully.

use crate::dec::ola_gen::{norm_l, sar, shift_sat, shl, sx16, zx16};
use crate::shared::cepstral_normalize::bfp_reciprocal_ratio;
use crate::shared::gamma_poly::gamma_poly_bfp_eval;
use crate::shared::q_energy::q_builder;

// ===========================================================================
// neg_max_unvoiced_amp_norm (91 B) -- leaf, zero callees.
// ===========================================================================

/// Faithful port of the unvoiced max-amp norm: returns `-norm_l(max)` where
/// `max` is the largest `amps[i] << 16` over the bands `i` with
/// `voicing[i] == 0` (UNVOICED), or `0` if there are none.
///
/// The loop runs iff `(i16)l > 0`, and its trip count is `zx16(l)`.
pub(crate) fn neg_max_unvoiced_amp_norm(amps: &[i16], voicing: &[i16], l: i16) -> i32 {
    let mut mx: i32 = 0;
    if l > 0 {
        let count = l as u16 as usize;
        for i in 0..count {
            // skip VOICED bands
            if voicing[i] == 0 {
                // sign-extend the amplitude and shift it left 16
                let v = (amps[i] as i32).wrapping_shl(16);
                // SIGNED running max
                if v > mx {
                    mx = v;
                }
            }
        }
    }
    // the norm_l idiom, then negate
    -norm_l(mx)
}

// ===========================================================================
// scale_shift_array (123 B) -- leaf, zero callees.
// ===========================================================================

/// Faithful port of the scale-shift array writer: for each of `cnt` elements,
/// `dst[i] = (shift3(2 * sx16(src[i]) * sx16(mul), sx16(e1 - e2))) >> 16`.
///
/// The shift is the **three-way** `test / js / cmp / sar` form -- the exact
/// shape whose collapse leg is easy to omit -- omitting it is a real bug.
/// `e1`/`e2` are separate arguments, not a pre-differenced shift.
pub(crate) fn scale_shift_array(
    dst: &mut [i16],
    src: &[i16],
    cnt: i16,
    mul: i16,
    e1: i16,
    e2: i16,
) {
    // guard: cnt <= 0 -> nothing to do
    if cnt <= 0 {
        return;
    }
    let count = cnt as u16 as usize; // zero-extend cnt to the trip count
    let sh = sx16((e1 as i32).wrapping_sub(e2 as i32)); // e1 - e2, sign-extended to the shift amount
    let m = mul as i32; // sign-extend the multiplier
    for i in 0..count {
        // 2 * src[i] * mul
        let mut v = (src[i] as i32).wrapping_mul(m);
        v = v.wrapping_add(v);
        v = if sh >= 0 {
            shl(v, sh) // non-saturating left shift
        } else if sh <= -31 {
            sar(v, 31) // the COLLAPSE leg
        } else {
            sar(v, -sh) // arithmetic right shift
        };
        // >>16, store as i16
        dst[i] = sar(v, 16) as i16;
    }
}

// ===========================================================================
// array_rms_bfp (200 B) -- calls q_builder, bfp_reciprocal_ratio, gamma_poly_bfp_eval.
// ===========================================================================

/// Faithful port of the array RMS: the RMS of `arr` over `2 * cnt` `i16`
/// elements, returned as a `(mantissa, exponent)` BFP pair.
///
/// Chain: `q_builder(arr, arr, 2*cnt)` (energy) -> `bfp_reciprocal_ratio` (divide by the
/// count) -> `gamma_poly_bfp_eval` (the sqrt Horner poly).
///
/// The DLL passes `&a4_slot` as `q_builder`'s `dest_exp`, i.e. it **aliases its
/// own `a4` argument slot** as the exponent scratch.  Returns `(0, 0)` when the
/// energy is non-positive.
pub(crate) fn array_rms_bfp(arr: &[i16], cnt: i16) -> (i16, i16) {
    // q_builder(dest_exp=&a4slot, x=a3, y=a3, count=2*a4)
    let (raw, q_exp) = q_builder(arr, arr, (cnt as i32).wrapping_mul(2) as i16);
    // >>16, then zero-extend to the energy
    let energy_raw = zx16(sar(raw as i32, 16));
    let energy = energy_raw as u16 as i16;
    // non-positive energy => both outs zero
    if energy <= 0 {
        return (0, 0);
    }
    // sign-extend cnt and shift left 16
    let mut cnt_mant = (cnt as i32).wrapping_shl(16);
    // the norm_l idiom on the count mantissa
    let nb = norm_l(cnt_mant);
    cnt_mant = shl(cnt_mant, nb); // normalise
                                  // exponent = 0xf - normbits
    let mut exp = zx16(0xf - nb) as i16;
    let cnt_hi = sar(cnt_mant, 16) as i16;
    // renormalise leg
    if energy <= cnt_hi {
        cnt_mant = sar(cnt_mant, 1);
        exp = exp.wrapping_add(1);
    }
    // bfp_reciprocal_ratio(cnt_mant, energy)
    let r = bfp_reciprocal_ratio(cnt_mant, energy);
    // sign-extend the ratio and shift it into the high word
    let ratio_sh = (r as i16 as i32).wrapping_shl(16);
    // a 16-bit subtract
    exp = exp.wrapping_sub(q_exp);
    // gamma_poly_bfp_eval(ratio_sh, exp) -- exp is in/out
    let (ret, exp2) = gamma_poly_bfp_eval(ratio_sh, exp);
    // >>16, store as i16
    (sar(ret as i32, 16) as i16, exp2)
}

// ===========================================================================
// unvoiced_overlap_add (357 B) -- calls block_denorm x2, memcpy, word-fill.
// ===========================================================================

/// The history-ring length that [`unvoiced_overlap_add`] always zero-fills up
/// to (0x54 words).
pub(crate) const HIST_WORDS_54: usize = 0x54;

// ===========================================================================
// biquad_postfilter (401 B) -- leaf, zero callees.
// ===========================================================================

/// The DLL's 32x16 fractional multiply, open-coded three times inside
/// [`biquad_postfilter`]: `hi*(2c) + ((lo*c) >> 15)`, i.e. `x * c / 32768`.
///
/// The high half is `sx16(x >> 16)` (the sign-extend is a no-op because
/// `x >> 16` already fits in 16 bits); the low half is the *unsigned* low word.
#[inline]
fn mulq(x: i32, c: i32) -> i32 {
    let hi = x >> 16;
    let lo = (x as u32 & 0xffff) as i32;
    hi.wrapping_mul(c.wrapping_mul(2))
        .wrapping_add(lo.wrapping_mul(c) >> 15)
}

/// The DLL's 64->32 saturate, open-coded twice.  Its inner `jg` is dead (it
/// tests the high word `> -1` on a path reached only when the high word `< 0`);
/// read whole, the block is exactly an `i64 -> i32` clamp.
#[inline]
fn sat32_64(acc: i64) -> i32 {
    acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// [`biquad_postfilter`]'s three fixed coefficients, read from the `imul`
/// immediates.  Each appears as a `(2c, c)` pair -- `0xf156 = 2*0x78ab` etc.
pub(crate) const UNVOICED_LPF_C1: i32 = 0x78ab; // 30891
pub(crate) const UNVOICED_LPF_C2: i32 = 0x7c20; // 31776
pub(crate) const UNVOICED_LPF_C3: i32 = 0x78af; // 30895

/// Faithful port of the 2-biquad IIR post-filter over `n` `i32` samples, with a
/// 64-bit accumulator and 6 `i32` state words.
///
/// **The subtle part, modelled faithfully:** after the first saturation the
/// 64-bit accumulator is **NOT** reloaded from the saturated value -- only the
/// *stored state* (`st[1]`, `st[2]`) saturates, while `acc` carries on with the
/// raw 64-bit sum.  The saturated high word is spilled to scratch on all three
/// legs and **never read again**, which is the tell.
pub(crate) fn biquad_postfilter(inp: &[i32], out: &mut [i32], st: &mut [i32; 6], n: i16) {
    // guard: n <= 0 -> nothing to do
    if n <= 0 {
        return;
    }
    let count = n as u16 as usize; // trip count
    for i in 0..count {
        // acc = mulq(st[1], C1) - st[0] + in[i]
        let mut acc: i64 = mulq(st[1], UNVOICED_LPF_C1) as i64;
        acc -= st[0] as i64;
        acc += inp[i] as i64;
        let v1 = sat32_64(acc); // stored saturated; `acc` stays RAW
        st[1] = v1;
        let in_i = inp[i];
        let old2 = st[2];
        st[0] = in_i;
        acc += (st[4] as i64) - 2 * (old2 as i64);
        let old3 = st[3];
        st[2] = v1;
        let t2 = mulq(old3, UNVOICED_LPF_C2);
        st[4] = old2;
        acc += 2 * (t2 as i64);
        let old5 = st[5];
        st[5] = old3;
        let t3 = mulq(old5, UNVOICED_LPF_C3);
        acc -= t3 as i64;
        let v2 = sat32_64(acc);
        st[3] = v2;
        out[i] = v2;
    }
}

// ===========================================================================
// shape_unvoiced_bands (748 B) -- the UNVOICED BACK-END ORCHESTRATOR.
//
// Sole call site.  Flow-walked to the ret; the body is exactly 748 B (0x2ec).
// Six scalar/pointer args (six pushes at the call site); the call site also
// tears down a ~40 B struct the caller built, not this fn's arg block.
//
// Calls, all already ported and reused here:
//   neg_max_unvoiced_amp_norm
//   gamma_poly_bfp_eval   (exp reconstruct)
//   word-fill         (inlined as zero loops, count in WORDS)
//   array_rms_bfp     (per-band RMS, non-mode2 only)
//   scale_shift_array (per-band in-place rescale)
//
// The output buffer `out` (arg1) is IN/OUT: the caller fills it with the
// unvoiced excitation spectrum; this fn reads it (RMS / rescale src) and
// overwrites each unvoiced band in place, zeros voiced bands and the head/tail.
// ===========================================================================

/// The DLL word-fill as invoked from [`shape_unvoiced_bands`]: the count is
/// pushed as a 32-bit value but the callee reads only the low **WORD** and
/// guards on a **signed 16-bit** trip count, so a count whose low 16 bits are
/// `<= 0` fills **nothing** (this is why a negative `cnt2` zero-fills zero
/// words, not the whole buffer).
#[inline]
fn word_fill_zero(out: &mut [i16], start: usize, count: i32) {
    let c = (count as i16) as i32;
    if c > 0 {
        for w in out.iter_mut().skip(start).take(c as usize) {
            *w = 0;
        }
    }
}

/// The fixed fields of [`shape_unvoiced_bands`]'s arg4 struct that the fn reads.
pub(crate) struct UnvoicedBandParams<'a> {
    /// `[edi+0x00]` u16 -- `== 2` selects the "mode2" exponent path.
    pub mode: u16,
    /// `[edi+0x04]` i16 -- band count `L` (loop trip count, signed guard).
    pub l: i16,
    /// `[edi+0x0c]` i16 -- the STEP scale coefficient.
    pub coef: i16,
    /// `[edi+0x10 + k*2]` i16 -- per-band amplitudes, `k in 0..L`.
    pub amps: &'a [i16],
    /// `*[edi+0x80]` i16 -- per-band voicing flags (`!= 0` => voiced).
    pub voicing: &'a [i16],
    /// `[edi+0x84]` u16 -- base exponent added into the output exponent.
    pub base_exp: u16,
}

/// Faithful port of the unvoiced back-end orchestrator.
///
/// Returns the count of unvoiced bands processed; writes the output exponent
/// through `out_exp` (`*arg2 = base_exp + expsum`, 16-bit).
///
/// Transliterated from the stack machine: locals are named by role, every
/// 16/32-bit truncation is explicit (`zx16`/`sx16`/`as i16`).  See the block
/// comment above for the call map.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shape_unvoiced_bands(
    out: &mut [i16],   // arg1: band spectrum buffer, IN/OUT
    out_exp: &mut i16, // arg2: output exponent, IN (mode2) / OUT
    n3: i16,           // arg3: synthesis length
    st: &UnvoicedBandParams,
    arg5: i16, // arg5
    arg6: i16, // arg6
) -> i16 {
    // ---- prologue constants ----
    let mode2 = st.mode == 2;
    let half3 = zx16((n3 >> 1) as i32); // n3 / 2
    let nn = zx16(if n3 == 0x200 { 0xf2 } else { 0x79 });
    let l = st.l;

    // ---- max_unvoiced_norm ----
    let maxn = neg_max_unvoiced_amp_norm(st.amps, st.voicing, l);
    let maxn16 = zx16(maxn); // zero-extend to 16 bits

    // ---- exp reconstruct; expA aliases the arg3 slot ----
    let mut exp_a: i32 = 0x11; // exponent seed
    let var_b: i32; // aliases arg4 slot
    if mode2 {
        let v = sx16(l as i32).wrapping_mul(0x86b6); // L * 0x86b6
        let (ret, ea) = gamma_poly_bfp_eval(v, exp_a as i16);
        exp_a = (ea as u16) as i32; // the poly writes *expA
        let r = sx16(sar(ret as i32, 16)); // >>16, sign-extended
        let prod = r.wrapping_mul(0xaff4); // r * 0xaff4
        let a2v = (*out_exp as u16) as i32; // arg2 (out_exp) input value
        exp_a = (exp_a.wrapping_add(a2v + 1)) & 0xffff; // expA += a2v + 1 (16-bit)
        var_b = zx16(sar(prod, 16)); // high word of the product
    } else {
        let l_arg6 = l.wrapping_mul(arg6); // l * arg6
        let v = sx16(l_arg6 as i32).wrapping_mul(0x86b6); // (l*arg6) * 0x86b6
        let (ret, ea) = gamma_poly_bfp_eval(v, exp_a as i16);
        exp_a = (ea as u16) as i32;
        var_b = zx16(sar(ret as i32, 16)); // high word of the poly result
    }

    // ---- STEP / ACC seed ----
    let expsum = zx16(exp_a + maxn16 + 2);
    let sx_n3 = sx16(n3 as i32);
    let step = sar(sx16(st.coef as i32).wrapping_mul(sx_n3).wrapping_mul(2), 4);
    let mut acc = sar(step.wrapping_add(0x10000), 1);
    let mut bpos = zx16(sar(acc, 16)); // position accumulator seed

    // ---- initial head zero-fill (out[0..2*bpos] = 0), write index starts past it ----
    // the fill returns base + sx16(count) words; the count's low WORD drives both.
    let head16 = ((2 * bpos) as i16) as i32;
    word_fill_zero(out, 0, 2 * bpos);
    let mut out_idx: usize = head16 as usize;

    // ---- shift arg for the inline three-way shift ----
    let neg_maxn = maxn16.wrapping_neg();

    // ---- main band loop ----
    let mut k: i32 = 0;
    let mut count: i32 = 0; // the return value
    if l > 0 {
        loop {
            let prev_pos = zx16(bpos); // OLD position
            acc = acc.wrapping_add(step); // ACC += STEP
            let new_pos = zx16(sar(acc, 16)); // high word of the accumulator
            let pos = if (new_pos as i16) > (half3 as i16) {
                half3 // clamp to n3/2
            } else {
                new_pos
            };
            bpos = pos; // clamped position (prev for next iter)
            let band_width = zx16(pos.wrapping_sub(prev_pos));
            let bw = band_width as usize;

            let voiced = st.voicing[k as usize] != 0;
            if voiced {
                // zero the band (2*bw words), voiced handled elsewhere
                word_fill_zero(out, out_idx, 2 * band_width);
            } else {
                // unvoiced band
                let mant: i32;
                let mut e1: i32;
                if mode2 {
                    mant = zx16(var_b);
                    e1 = zx16(exp_a);
                } else {
                    // rms of the input excitation region (reads the current out slice)
                    let (rms_mant, rms_exp) =
                        array_rms_bfp(&out[out_idx..out_idx + 2 * bw], band_width as i16);
                    // sign-extend varB's low word before the multiply --
                    // so sx16(varB), not the raw zero-extended value.
                    let t = sx16(rms_mant as i32)
                        .wrapping_mul(sx16(var_b))
                        .wrapping_mul(2);
                    mant = zx16(sar(t, 16));
                    e1 = (rms_exp as i32).wrapping_add(exp_a); // += expA
                }
                // amp_prod = 2 * amps[k] * sx16(mant), then inline shift
                let mut amp_prod = sx16(st.amps[k as usize] as i32)
                    .wrapping_mul(sx16(mant))
                    .wrapping_mul(2);
                amp_prod = shift_sat(amp_prod, neg_maxn); // three-way shift by -MAXN
                let scaled = zx16(sar(amp_prod, 16)); // >>16
                e1 = e1.wrapping_add(maxn16);
                // scale_shift_array(dst=src=out slice, cnt=2*bw, mul=scaled, e1, e2=expsum)
                let src: Vec<i16> = out[out_idx..out_idx + 2 * bw].to_vec();
                scale_shift_array(
                    &mut out[out_idx..out_idx + 2 * bw],
                    &src,
                    (2 * bw) as i16,
                    scaled as i16,
                    e1 as i16,
                    expsum as i16,
                );
                count += 1;
            }

            // loop tail
            out_idx += 2 * bw; // advance by 2*bw words
            k += 1;
            if (k as i16) >= l {
                break;
            }
        }
    }

    // ---- epilogue ----
    // tail width = half3 - bpos (last POS or seed); pick clamp against (half3 - NN)
    let mut tail_w = zx16(half3.wrapping_sub(bpos));
    let clamp_thr = sx16(half3).wrapping_sub(sx16(nn)); // half3 - NN
    if clamp_thr > sx16(tail_w) {
        tail_w = zx16(half3.wrapping_sub(nn)); // half3 - NN
    }
    // tail zero-fill: out[sx_n3 - 2*tail_w .. sx_n3] = 0  (2*tail_w words)
    let idx1 = sx_n3.wrapping_sub(sx16(tail_w).wrapping_mul(2));
    word_fill_zero(out, idx1 as usize, 2 * tail_w);
    // head zero-fill #2: out[0..cnt2] = 0  (cnt2 may be <= 0 -> fills nothing)
    let cnt2 = sar(sx16(arg5 as i32).wrapping_mul(sx_n3).wrapping_mul(2), 0x14).wrapping_mul(2);
    word_fill_zero(out, 0, cnt2);
    // *arg2 = base_exp + expsum (16-bit)
    *out_exp = st.base_exp.wrapping_add(expsum as u16) as i16;
    count as i16
}
