//! The P/Q pre-pass builders, the deepest piece of the parent function's
//! internal arithmetic (see [`super::block_consolidate`] and `t_ring`'s module
//! doc for the surrounding chain).
//!
//! ## What P computes
//!
//! `P(dest_exp: *mut i16, x: *const i16, exp_x: i16, y: *const i16,
//! exp_y: i16, count: i16) -> i16`: a block-floating-point **cross energy of a
//! scale-aligned difference** between two raw sample windows `x`/`y`, each
//! carrying its own exponent.
//!
//! 1. **Swap branch** (a SIGNED 16-bit compare-and-branch on the exponents): if
//!    `exp_y > exp_x` (both as signed `i16`), `x` becomes the "ref" array
//!    (walked directly) and `y` becomes "other" (walked via a computed pointer
//!    delta); otherwise the roles swap. This is a genuine min/max-style branch,
//!    not a no-op.
//! 2. **The exponent difference** used for scale alignment is NOT
//!    `|exp_y - exp_x|`. That reduction holds for ordinary small non-negative
//!    exponents and is false in general. The real computation, on the raw
//!    **zero-extended 16-bit** exponent values (`exp_x_u`/`exp_y_u` -- never
//!    sign-reinterpreted, even though the swap branch above IS a signed
//!    compare), is:
//!    `diff = if swap { exp_x_u - exp_y_u } else { exp_y_u - exp_x_u }`, then
//!    `sel = diff - 1` in full 32-bit precision -- but every later use reads
//!    only the low 16-bit sub-register of `sel`, so the EFFECTIVE shift-selector
//!    is `sign_extend_16((diff - 1) & 0xffff)`.
//!
//!    The truncation is invisible for ordinary small exponents and changes the
//!    result completely for rows with a `0xFFFF`-pattern ("looks negative")
//!    exponent: for `exp_x=0, exp_y=0xffff` the real `diff` is `0xffff` (NOT
//!    `-0xffff`) and the real selector is `-2` (NOT the untruncated `-65536`).
//! 3. Per-sample loop (selector constant per call, `2*count` iterations):
//!    `other`'s sample is fixed-shifted right by 1 (`(other<<16)>>1`, exact,
//!    `== other*32768`); `ref`'s sample is shifted by the selector (`sel>=0`:
//!    left by `sel`; `-31<sel<0`: arithmetic right by `-sel`; `sel<=-31`:
//!    sign-fill). `dx = (other_shifted - ref_shifted) >> 16` (arithmetic);
//!    `accum += 2*dx*dx` into a 64-bit running accumulator.
//! 4. Tail: `shiftcount = normalize64(accum)`
//!    ([`super::band_decompress::normalize64`]); `shifted = shift64(accum,
//!    shiftcount)` ([`super::band_decompress::shift64`]); `mantissa =
//!    (shifted.0 as i32 >> 16) as i16` -- the HIGH 16 bits of the shifted low
//!    dword, which differs from Q's tail; `out_exp = if swap { exp_y_u } else
//!    { exp_x_u }`, literally whichever raw exponent slot the swap branch leaves
//!    untouched, NOT a numeric `max()` (the two coincide only for ordinary
//!    non-negative exponents); `exponent = 2*(out_exp+1) - shiftcount`;
//!    `*dest_exp = exponent`; returns `mantissa`.
//!
//! ## What Q computes
//!
//! `Q(dest_exp: *mut i16, x: *const i16, y: *const i16, count: i16) -> u32`: a
//! plain **dot product**. The call sites always pass the same pointer for
//! `x`/`y`, making it a self-energy in practice, but the function supports two
//! distinct arrays: `accum = sum(2*x[i]*y[i] for i in 0..count)`, 64-bit
//! accumulator, same idiom as P. Tail: `shiftcount = normalize64(accum)`;
//! `shifted = shift64(accum, shiftcount)`; the return value is the **full
//! 32-bit** `shifted.0`, not truncated (the caller takes an arithmetic right
//! shift by 16 of it); `*dest_exp = -shiftcount`, with no `out_exp`/`2*(..)+2`
//! term at all, unlike P -- Q has no second array's exponent to combine.
//!
//! ## What this does NOT close
//!
//! This closes P and Q themselves. It does not close what feeds the parent
//! function's incoming args (the raw correlation/window array source, several
//! frames upstream) or array B's separate formula. `Encoder::attempt_b1_fixed`
//! therefore still cannot use real computed arrays end-to-end from raw audio and
//! remains fed `B1_X86_PLACEHOLDER_ABC`.
//!
//! ## Where `x`/`y` come from
//!
//! The P/Q parent has exactly ONE caller (the per-band P/Q dispatcher); the
//! dispatcher has exactly ONE caller (the window-builder body); the body has
//! exactly ONE caller (the spectrum-driver stage, called from the same site
//! `super::loudness_fixed::cepstral_transform_narrow` names, with `arg1 = ebp+0x18` = the
//! d890 stage's 129-dword COMPLEX spectrum). The arg wiring is
//! `x = base_x + si*4`, `y = base_y + si*4`, sliced into 8 bands by harmonic
//! index `si`:
//!
//!   * **`x` (`base_x`)** = an argument passed into the window-builder body
//!     (read as `[esp+0x464]`), rooted in the `ctx` block the spectrum-driver
//!     operates on -- the actual spectrum.
//!   * **`y` (`base_y`)** = a LOCAL stack buffer of the body (`&[esp+0x78]`),
//!     built inside it by a complex-spectrum x real-taper multiply, a normalize,
//!     and a refine stage -- a SYNTHETIC reference spectrum.
//!
//! So P = per-band energy of `(x - y)` and Q = per-band self-energy: the
//! MBE-style per-band voicing measure (actual vs synthetic reference), the
//! direct upstream of b1.
//!
//! ## Composition site-map (cdecl arg order, window-builder body)
//!
//! Prologue arg pre-adjusts, easy to miss: `add [esp+0x14],0x1e` at entry
//! (**arg5 += 30**) and `add [esp+0x42c],0xf` after `sub esp,0x420`
//! (**arg3 += 15**). These are exponent-bias bumps applied BEFORE any site runs.
//!
//! Then `ebp = arg1` (= ctx), and a 2-iteration copy/push loop seeds the T/S
//! ring via [`super::history_ring::HistoryRing::push_half`].
//!
//! Two arms build the two P/Q windows:
//! * **`y` (synthetic reference)** -- sites 1-6, from `arg3`/`arg4`:
//!   1. `cepstral_stage1(&[esp+0x22c], arg3, arg4, 0x81)` -> logmag exp.
//!   2. `yarm_refine(&[esp+0x248], exp=s1.ret, 8, 0x100)`
//!      ([`super::pq_yarm::compose_yarm_reference`]).
//!   3. `site3_transform(&[esp+0x50], arg3, arg4, 0x81)`.
//!   4. `combine(&[esp+0x268], self, s2.ret, &[esp+0x5c], s3.ret, 0x81)`.
//!   5. min-nonzero scan of `[esp+0x28..]` (the `0x7fff` clamp) + normalize64
//!      -> `shift`; `normalize(&[esp+0x30], self, s3.ret, shift, 0x81)`.
//!   6. `selfcorr(&[esp+0x24c], self, s4.ret, &[esp+0x248], s4.ret, 0x80)`.
//! * **`x` (actual spectrum)** -- sites 7-8, from `arg11`/`arg15`:
//!   7. `combine(edi, edi, arg15, &[esp+0x5c], s6.ret, 0x81)`.
//!   8. `selfcorr(edi, edi, s7.ret, &[edi+4], s7.ret, 0x80)` -> x, in place.
//! * `w_builder(&[esp+0x44], arg12 x3)` then `word_fill(&[esp+0x54], 0, 0xc)`.
//! * `dispatch(Pdst, y, s6.ret, x, s8.ret, 0xffffae84, 6, ebp)` -> per-band
//!   P/Q, returning `s_new` (written to `[ebp+0x48]`).
//!
//! At the dispatch call the arg slots are `sp[0]` = P-output dest
//! (`0x4000`-silence init), `sp[1]` = **y, the synthetic reference**, `sp[3]` =
//! **x, the actual spectrum**. `sp[0]` is NOT `y_base`.
//!
//! The `&[esp+N]` offsets shift between sites as pushes move esp; do not read
//! them as fixed, and do not infer equalities like "dstC == y_base" from the
//! literal offsets.
//!
//! ### The post-dispatch per-band repair loop
//!
//! The body's own tail, gated by `[ebp+0x2]>0 && [ebp+0]<0 &&
//! (cmp [ebp+0x12],[ebp+0x14] ...)`: a `count=8` loop over `edi=[ebp+0x28]`
//! (the T ring) that computes, per lane,
//! `dx = (ring[j-8]<<16) shifted by (s_new - [ebp+0x4a]) - (ring[j]<<16)
//! shifted by ([ebp+0x4a]-something)`, and if the difference is negative
//! rewrites `ring[j] = sar(sar(prev<<16, ...),16)`. This is a saturating
//! monotone repair of the T ring against `s_new`: the body post-conditions the
//! ring `a_writer_band` later reads. Wire order for a5 is therefore
//! x,y-from-PCM -> P/Q -> `s_new` -> this repair loop -> t_ring ->
//! `a_writer_band`. The shift arms use the same
//! `sel>=0`/`-31<sel<0`/`sel<=-31` three-way as P/Q (`0xffe1 = -31`).
//!
//! ## Nothing new needs porting
//!
//! Every callee of the window-builder body is already ported:
//!
//! | callee | ported as |
//! |---|---|
//! | ring seed | [`super::history_ring::HistoryRing::push_half`] |
//! | log2 transform | [`super::cepstral_stage1::cepstral_stage1_log_transform`] |
//! | y-arm refine composite (site2) | [`super::pq_yarm::compose_yarm_reference`] |
//! | log-refine sibling | [`super::pq_yarm::site3_transform`] |
//! | combine (sites 4 & 7) | [`super::cepstral_stage2::combine_scalar`] |
//! | normalize | [`super::cepstral_normalize::cepstral_normalize`] |
//! | selfcorr (sites 6 & 8) | `super::cepstral_stage2::selfcorr` |
//! | w_builder gate | [`super::w_builder`] |
//! | word-fill | inlined |
//! | per-band P/Q dispatcher | [`p_builder`] / [`q_builder`] |
//! | transitive FFT/normalize helpers | [`crate::dec::unv_frame::array_forward_transform`], [`super::loudness_transform::fft_bfp_transform`], [`super::array_a_stage2::inverse_fft_butterfly_stage`], `dec::linamp` block-normalize |
//!
//! **The site2 refine callee is NOT [`super::stage2_refine_dispatch::array_a_stage2_refine_dispatch`]
//! alone.** Feeding site2's captured inputs to `array_a_stage2_refine_dispatch` scores 0/200. The
//! callee is a COMPOSITE: `array_a_stage2_refine_dispatch(ptr, arg2, arg3, arg4=1)` -- note
//! `arg4=1`, not the site's `0x100` -- plus a scale-array, a normalize, a
//! word-fill, a forward transform, and a final interpolator
//! ([`super::cos_quadrature_interp::quadrature_interp`]). See
//! [`super::pq_yarm`] for the composed form. `array_a_stage2_refine_dispatch` itself is correct at
//! `arg3=8`; the 0/200 is a wrong function mapping, not a range failure.
//!
//! ## The remaining blocker: the d890 INPUT window
//!
//! `ctx+0x18` (the d890 output complex win) is **not** the x-arm root. It is a
//! distinct buffer with a near-constant imaginary lane (~107..108), and it
//! equals `site7.in.slot0` only on silent frames (24/200). Do not chase it.
//!
//! The x-arm root is the lib transform of the d890 *input*: feeding the
//! captured d890 input (`ctx+0x13dc`) through
//! [`super::loudness_fixed::win_from_gap2`] ->
//! [`super::loudness_transform::fft_bfp_transform`] ->
//! [`super::array_a_stage2::inverse_fft_butterfly_stage`] reproduces both the
//! exponent and `site7.in.slot0` exactly. The transform and the exponent are
//! bit-exact; the only unknown is the input window.
//!
//! And that window is a CONTENT gap, not a scale or alignment one:
//! * A uniform block-exp shift reconciles the from-PCM win in 0 cases -- it was
//!   never an exponent difference.
//! * `transform(gap2_mid/slot1)` scores 0/176 and 0/197 against
//!   `site7.in.slot0`; none of `gap2_mid`/`slot1`/`pre` match on either parity;
//!   and the captured input is absent from the full 383-word ring at EVERY
//!   offset.
//! * The captured input is strongly low-passed and rescaled versus the raw-audio
//!   window (voiced max 3504 vs raw 8926, much smoother; mark is a smooth
//!   32767-saturated tone at a frame whose raw PCM is near-silent).
//!
//! That points at the analysis prefilter / ring feed, upstream of everything the
//! P/Q chain touches. Reproducing `ctx+0x13dc` unblocks x, y, and therefore a5.

use super::band_decompress::{normalize64, shift64};
use super::cepstral_normalize::cepstral_normalize;
use super::cepstral_stage1::cepstral_stage1_log_transform;
use super::cepstral_stage2::{combine_scalar, rescale_accumulate};
use super::pq_yarm::{compose_yarm_reference, site3_transform};

// ===========================================================================
// Full BODY assembly (x-arm superset of y-arm), reproduced from PCM and pinned
// bit-exact against the captured `s6out0`/`s8out0` endpoints on every non-silent
// unit of both conformance clips.
//
// Inputs (both PCM-derived):
//   * `power` = `loudness_fixed::d890_power_arg3(gap2)` (129-i32 |Sw|^2, the
//     d890 power spectrum = arg4 = site1.in.slot1), 394/394 from PCM.
//   * `scale` = `loudness_fixed::cepstral_transform_narrow(gap2).1 + 30` = `ctx[0x60c]`
//     (the d890 return exponent) + the body prologue's arg5 += 30 bias.
//     `0x18+0x60c = 0x624` = the d890-return store.
//   * `xwin` = the d890 COMPLEX window (258 i16, Nyquist-inclusive) = the
//     x-arm root site7.in.slot0 (`loudness_fixed::cepstral_transform_narrow`'s internal
//     `win` + the transform scratch Nyquist pair), 394/394 from PCM.
//
// All buffer content is seed-INDEPENDENT (`rescale_accumulate` renormalizes by
// `block_exponent(buf)` regardless of the seed), so seed = 0 throughout; the
// scalar exponent threading matters only for the (next-round) P/Q dispatch tail.

/// The site5 min-nonzero-scan shift (from the body): the
/// smallest positive value over `w`, then `0xf - (0x1e - hi_bit((min-1)<<16))`
/// (0 if `min == 1`), where `hi_bit` is the highest-set-bit index.
fn minscan_shift(w: &[i16]) -> i16 {
    let mut mn: i16 = 0x7fff;
    for &val in w.iter().take(129) {
        if val != 0 && val < mn {
            mn = val;
        }
    }
    let d = mn.wrapping_sub(1);
    if d == 0 {
        return 0;
    }
    let e = (d as i32) << 16;
    let eb = if e < 0 { !e } else { e };
    let c = 0x1e - (31 - (eb as u32).leading_zeros() as i32);
    (0xf - c) as i16
}

/// In-place lag-1 self-correlation (the selfcorr helper at both x/y call sites):
/// `q` at `buf[q_off..]`, `p` at `buf[p_off..]` (they overlap by one complex
/// pair), writes into `q`, then whole-buffer `rescale_accumulate`.
fn selfcorr_inplace(buf: &mut [i16], q_off: usize, p_off: usize, count: usize) {
    for k in 0..count {
        let qb = q_off + 2 * k;
        let pb = p_off + 2 * k;
        let qre = buf[qb] as i32;
        let qim = buf[qb + 1] as i32;
        let pre = buf[pb] as i32;
        let pim = buf[pb + 1] as i32;
        let re = pre.wrapping_mul(qre).wrapping_add(pim.wrapping_mul(qim));
        let im = pre.wrapping_mul(qim).wrapping_sub(pim.wrapping_mul(qre));
        buf[qb] = (re >> 16) as i16;
        buf[qb + 1] = (im >> 16) as i16;
    }
    let _ = rescale_accumulate(&mut buf[q_off..q_off + 2 * count], 0);
}

/// The x-arm per-tap weight (site5 = `cepstral_normalize(w, w, minscan(w))`,
/// where `w` = site3's output).  Verified `== site7.in.slot3` 200/200.
pub(crate) fn x_arm_weight(power: &[i32], scale: i16) -> Vec<i16> {
    let (w, _) = site3_transform(&power[0..129], scale, 129);
    let shift = minscan_shift(&w);
    let src = w.clone();
    let mut dst = w;
    cepstral_normalize(&mut dst[0..129], &src[0..129], shift);
    dst
}

/// The y-arm site6 selfcorr RETURN exponent (`exp_y = s6.ret`), threaded from
/// PCM. This is the exponent the P/Q dispatch passes as `arg2` and every
/// per-band `p_builder` reads as `exp_y`.
///
/// Seed thread (`combine seed = arg2+arg4`, `selfcorr seed = arg2+arg4+1`):
/// `s1.ret = 4` (const), `s2.ret = 0`, `s3.ret = site3_transform().1`,
/// `s4.ret = combine(y, w, seed=s3.ret)`, `seed6 = 2*s4.ret + 1`,
/// `exp_y = selfcorr(y, seed6)`.
///
/// Returns `(y_content[256], exp_y)`.
pub(crate) fn assemble_y_arm_with_exp(power: &[i32], scale: i16) -> (Vec<i16>, i16) {
    let mut a: Vec<i16> = cepstral_stage1_log_transform(&power[0..129], scale);
    compose_yarm_reference(&mut a, 4, 8, 256);
    let (w, s3ret) = site3_transform(&power[0..129], scale, 129);
    let s4ret = combine_scalar(&mut a[0..258], &w[0..129], s3ret, 129);
    let seed6 = (2i32 * s4ret as i32 + 1) as i16;
    // in-place lag-1 selfcorr with the real seed, capturing the accumulated exp
    for k in 0..128 {
        let (qb, pb) = (2 * k, 2 * k + 2);
        let (qre, qim) = (a[qb] as i32, a[qb + 1] as i32);
        let (pre, pim) = (a[pb] as i32, a[pb + 1] as i32);
        let re = pre.wrapping_mul(qre).wrapping_add(pim.wrapping_mul(qim));
        let im = pre.wrapping_mul(qim).wrapping_sub(pim.wrapping_mul(qre));
        a[qb] = (re >> 16) as i16;
        a[qb + 1] = (im >> 16) as i16;
    }
    let exp_y = rescale_accumulate(&mut a[0..256], seed6);
    (a[0..256].to_vec(), exp_y)
}

/// The x-arm (actual spectrum) `x = site8.out`, from the d890 COMPLEX window
/// (258 i16, Nyquist-inclusive) + power + scale.  site7(combine with the
/// site5 weight) -> site8(lag-1 selfcorr).
///
/// This returns CONTENT only: it does not thread the RETURN exponent
/// (`exp_x = s8.ret`). Use [`assemble_x_arm_with_exp`] when the exponent is
/// needed -- the content is bit-identical either way.
pub(crate) fn assemble_x_arm(xwin: &[i16], power: &[i32], scale: i16) -> Vec<i16> {
    let mut x: Vec<i16> = vec![0i16; 258];
    x[0..258].copy_from_slice(&xwin[0..258]);
    let xw = x_arm_weight(power, scale);
    combine_scalar(&mut x[0..258], &xw[0..129], 0, 129);
    selfcorr_inplace(&mut x, 0, 2, 128);
    x[0..256].to_vec()
}

/// The x-arm site8 selfcorr RETURN exponent (`sc = s8.ret`), threaded from PCM.
/// This is the P/Q dispatch's `arg5` (the value the exp_x stage reads as
/// `selfcorr_exp`) and `windowed_complex_correlation`'s `arg6`.
///
/// **Do not substitute `exp_y` for it.** The two agree on most calls but
/// `sc == exp_y - 1` on some, which is exactly the `exp_x` ±1 residual.
///
/// Seed thread: `arg15 = scale >> 1` (`[esp+0x474]`, `= (dsp1>>1)+15`),
/// `site5.ret = 16 - minscan_shift(w) - s3ret`, `seed7 = arg15 + site5.ret`,
/// `s7ret = combine(x, x_arm_weight, seed7)`, `seed8 = 2*s7ret + 1`,
/// `sc = selfcorr(x, seed8)`.
///
/// Returns `(x_content[256], sc)`; the content is bit-identical to
/// [`assemble_x_arm`].
pub(crate) fn assemble_x_arm_with_exp(xwin: &[i16], power: &[i32], scale: i16) -> (Vec<i16>, i16) {
    let arg15 = (scale as i32) >> 1;
    let (w3, s3ret) = site3_transform(&power[0..129], scale, 129);
    let minscan = minscan_shift(&w3);
    let site5ret = 16i32 - minscan as i32 - s3ret as i32;
    let seed7 = (arg15 + site5ret) as i16;
    let mut x: Vec<i16> = vec![0i16; 258];
    x[0..258].copy_from_slice(&xwin[0..258]);
    let xw = x_arm_weight(power, scale);
    let s7ret = combine_scalar(&mut x[0..258], &xw[0..129], seed7, 129);
    let seed8 = (2i32 * s7ret as i32 + 1) as i16;
    // in-place lag-1 selfcorr with the real seed8, capturing the accumulated exp
    for k in 0..128 {
        let (qb, pb) = (2 * k, 2 * k + 2);
        let (qre, qim) = (x[qb] as i32, x[qb + 1] as i32);
        let (pre, pim) = (x[pb] as i32, x[pb + 1] as i32);
        let re = pre.wrapping_mul(qre).wrapping_add(pim.wrapping_mul(qim));
        let im = pre.wrapping_mul(qim).wrapping_sub(pim.wrapping_mul(qre));
        x[qb] = (re >> 16) as i16;
        x[qb + 1] = (im >> 16) as i16;
    }
    let sc = rescale_accumulate(&mut x[0..256], seed8);
    (x[0..256].to_vec(), sc)
}

use crate::fixops::i32r::{s32, sar32};

/// The `P(dest_exp, x, exp_x, y, exp_y, count) -> mantissa` formula (see the
/// module doc). `x`/`y` must have at least `2*count` valid elements -- the loop
/// length. Returns `(mantissa, exponent)`: the reference's register return and
/// its separate `*dest_exp` word write.
pub(crate) fn p_builder(exp_x: i16, exp_y: i16, x: &[i16], y: &[i16], count: i16) -> (i16, i16) {
    let exp_x_u = (exp_x as u16) as i32;
    let exp_y_u = (exp_y as u16) as i32;
    let swap = exp_y > exp_x; // signed 16-bit compare (SAME width as the real signed exponent compare)

    let loop_limit = 2i32 * (count as i32);
    let mut accum: i64 = 0;

    if loop_limit > 0 {
        let d = exp_y_u - exp_x_u; // raw zero-extended values, no abs()
        let exp_diff = if swap { -d } else { d };
        let sh_full = exp_diff - 1;
        // truncate to the low 16 bits and re-sign-extend (Rust's `as i16`
        // on an i32 does exactly this: keep the low 16 bits, reinterpret as
        // signed) -- reproducing the real 16-bit sub-register read of the
        // shift selector.
        let sh = sh_full as i16 as i32;

        let (ref_arr, other_arr): (&[i16], &[i16]) = if swap { (x, y) } else { (y, x) };

        for k in 0..(loop_limit as usize) {
            let a_ref = ref_arr[k] as i32;
            let b_other = other_arr[k] as i32;

            let term_other = sar32(b_other << 16, 1);

            let ashl = a_ref << 16;
            let term_ref = if sh >= 0 {
                (ashl as u32).wrapping_shl((sh & 0x1f) as u32) as i32
            } else {
                let shamt = (-sh) as u32;
                if shamt > 0x1f {
                    sar32(ashl, 0x1f)
                } else {
                    sar32(ashl, shamt)
                }
            };

            let dx32 = term_other.wrapping_sub(term_ref);
            let dx16 = sar32(dx32, 16);
            let dx = dx16 as i16 as i32;
            let sq = dx.wrapping_mul(dx);
            let sq2 = sq.wrapping_add(sq); // doubling (2*sq)
            accum = accum.wrapping_add(sq2 as i64); // sign-extended 64-bit accumulate; the i64 add matches the real add-with-carry exactly
        }
    }

    let out_exp = if swap { exp_y_u } else { exp_x_u };
    let lo = (accum & 0xFFFF_FFFF) as u32;
    let hi = ((accum >> 32) & 0xFFFF_FFFF) as u32;
    let shiftcount = normalize64(lo, hi);
    let (slo, _shi) = shift64(lo, hi, shiftcount);
    let shifted_lo = s32(slo as i32);
    let hi16 = sar32(shifted_lo, 16);
    let mantissa = hi16 as i16;
    let exponent = (2i32
        .wrapping_mul(out_exp.wrapping_add(1))
        .wrapping_sub(shiftcount)) as i16;
    (mantissa, exponent)
}

/// The `Q(dest_exp, x, y, count) -> eax_raw` formula (see the module doc).
// `q_builder` (the energy accumulator returning `(eax_raw, exponent)` -- the
// reference's full 32-bit register return, caller extracts a second field via a
// >>16) moved to `crate::shared::q_energy` so the decoder can reach it without
// `enc/`. Re-exported so `enc::a5_assemble`'s `super::pq_builder::q_builder`
// call site is unchanged.
pub(crate) use crate::shared::q_energy::q_builder;
