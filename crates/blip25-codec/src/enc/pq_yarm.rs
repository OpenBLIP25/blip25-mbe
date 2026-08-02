//! The y-arm composite (site2 of the y-arm / synthetic-reference spectrum),
//! plus the two leaves it needs as STANDALONE fns.
//!
//! ## Reuse
//!
//! The two leaves ("scale array" and "block-normalize, updates bexp", which
//! calls "-(norm of max|v|)" and "normalize array by shift") already exist
//! verbatim, inlined inside [`crate::dec::linamp`] (`amp_from_mllog`), where
//! they are decode-validated bit-exact whole-file.
//! [`scale_words_q15`]/[`block_normalize_words`] below are the same arithmetic
//! re-expressed for the `[i16]` word buffers the y-arm composition uses
//! (`array_a_stage2_refine_dispatch` / `array_forward_transform` / `quadrature_interp` all take
//! `&mut [i16]`). A test proves the `[i16]` ports are bit-identical to the
//! `[i32]` originals on random data.
//!
//! ## Behaviour of the y-arm composite `compose(dst, exp, arg2, count)`
//!
//! With `H=(count/2)&0xffff` (=128 for count=256) and `S=H+1` (=129):
//!   1. `R1 = array_a_stage2_refine_dispatch(dst, exp, arg2, 1)`
//!   2. `dst[0] = sar(dst[0],1)`; `dst[H] = sar(dst[H],1)` (two in-place
//!      halvings; the SECOND index is `H`, **not** `R1`, the array_a_stage2_refine_dispatch
//!      return).
//!   3. `scale_words_q15(dst, S, 0x70f7)`
//!   4. `exp1 = R1 - 2`; `block_normalize_words(dst, S, &exp1)` (exp1 +=
//!      block-norm shift). The forward-transform exponent is
//!      `R1 - 2 + normshift`, **not** the caller's `exp - 2`.
//!   5. zero-fill `dst[S..count] = 0` (word-fill).
//!   6. `R2 = array_forward_transform(dst, exp1, arg2, 1)`
//!   7. `quadrature_interp(dst, dst+1word, R2, H)` (in-place cos-interp).
//!      returns 0.
//!
//! Both the array_a_stage2_refine_dispatch leaf and the forward transform
//! ([`crate::dec::unv_frame::array_forward_transform`], arg4=1) are ported and
//! validated on their own fixtures. Note the arg5 pitfall: this path uses
//! [`super::stage2_refine_dispatch::array_a_stage2_refine_dispatch`], whose inner sub-args are
//! hardcoded `(1,1)` -- correct here only because `arg4=1`.

use super::cos_quadrature_interp::quadrature_interp;
use super::stage2_refine_dispatch::array_a_stage2_refine_dispatch;
use crate::dec::unv_frame::array_forward_transform;

use crate::fixops::i32r::bsr;

// ---------------------------------------------------------------------------
// Site3 of the y-arm (the "cepstral_stage1 sibling" / `w` producer, ported as
// `site3_transform`). Unlike site1, which does log2 via its own table, site3
// evaluates a DIFFERENT log2 polynomial (`log2_bfp_poly`, its own table) with a
// block-normalize + fractional-lookup tail, wrapped per-tap by `log2_tap_quantize` and
// followed by a whole-buffer `block_normalize_words`.

/// log2_bfp_poly log2 polynomial coefficients (5 signed i16).
const LOG2_POLY_COEF: [i16; 5] = [0xf073u16 as i16, 0x3f26, 0x9292u16 as i16, 0x7b83, 0x4251];
/// log2_bfp_poly tail fractional-lookup table (4 signed i16; `[0]` aliases
/// `LOG2_POLY_COEF[4]`).
const LOG2_POLY_FINAL: [i16; 4] = [0x4251, 0x4c1c, 0x5a82, 0x6ba2];

#[inline]
fn log2_poly_horner_stage(xin: i32, m: i32, coef: i16) -> i16 {
    let prod = xin.wrapping_mul(m);
    let mut acc = prod.wrapping_mul(2).wrapping_add(0x8000);
    acc &= 0xffff_0000u32 as i32;
    acc = acc.wrapping_add((coef as i32) << 16);
    (acc >> 16) as i16
}

/// `log2_bfp_poly`: a `bsr`-normalize + 4-stage Horner log2 polynomial (its own table)
/// with a second block-normalize and an `exp & 3` fractional refinement.
/// Returns `(mantissa_ret, exp_out)`: the reference's `eax` return (a normalized
/// value, mantissa in the high 16 bits) and the exponent it writes back to
/// `*ptr` (`(esi+3)>>2`).  `exp_in` is the value held in `*ptr` on entry (the
/// incoming block exponent / scale).
fn log2_bfp_poly(value: i32, exp_in: i16) -> (i32, i16) {
    if value == 0 {
        return (0, (3 >> 2) as i16); // = 0
    }
    let bsr_in = if value < 0 { !value } else { value };
    let hibit = bsr(bsr_in);
    let shift = 0x1e - hibit;
    let shcnt = (shift as u32) & 0x1f;
    let exp_adj = (exp_in as i32).wrapping_sub(shift & 0xffff) as i16; // sub ax,cx (16-bit)
    let norm = ((value as u32).wrapping_shl(shcnt) as i32) >> 16; // shl edx,cl ; sar edx,0x10
    let mut exp_acc = (exp_adj as u16) as i32; // movzx esi,ax
    let m = (norm as i16) as i32; // movzx eax,dx ; movsx edx,ax

    let x1 = log2_poly_horner_stage(LOG2_POLY_COEF[0] as i32, m, LOG2_POLY_COEF[1]);
    let x2 = log2_poly_horner_stage(x1 as i32, m, LOG2_POLY_COEF[2]);
    let x3 = log2_poly_horner_stage(x2 as i32, m, LOG2_POLY_COEF[3]);
    // Stage D (final, with an extra `sar 1` on both operands).
    let prod = (x3 as i32).wrapping_mul(m);
    let mut mant = prod.wrapping_mul(2).wrapping_add(0x8000);
    let a = (LOG2_POLY_COEF[4] as i32) << 16;
    mant >>= 1;
    let a = a >> 1;
    mant &= 0xffff_8000u32 as i32;
    mant = mant.wrapping_add(a);

    // Tail: renormalize the mantissa and (if `exp_acc & 3 != 0`) apply the
    // fractional table refinement.
    let norm_sh = if mant != 0 {
        let bin = if mant < 0 { !mant } else { mant };
        ((0x1e - bsr(bin)) as u16) as i32
    } else {
        0
    };
    let add = 1i32.wrapping_sub(norm_sh);
    mant = (mant as u32).wrapping_shl((norm_sh as u32) & 0x1f) as i32;
    exp_acc = (exp_acc.wrapping_add(add) as i16) as i32; // add esi,eax  (used as si downstream)

    let e3 = (exp_acc & 3) as usize;
    if e3 != 0 {
        let mant_hi = mant >> 16;
        let c = LOG2_POLY_FINAL[e3] as i32;
        let a = (mant_hi as i16) as i32;
        let prod = c.wrapping_mul(a);
        mant = prod.wrapping_mul(2).wrapping_add(0x8000);
        mant &= 0xffff_0000u32 as i32;
        exp_acc = ((exp_acc + 1) as i16) as i32; // inc esi
    }
    let exp_out = (((exp_acc as i16) as i32 + 3) >> 2) as i16;
    (mant, exp_out)
}

/// `log2_tap_quantize`: per-tap wrapper — `log2_bfp_poly(value, scale)` then shift its mantissa by
/// `(af60_exp - n)` with the codec's standard three-way left/arith-right/
/// sign-fill shift and a `+0x8000` round, truncated to i16.
fn log2_tap_quantize(value: i32, scale: i16, n: i16) -> i16 {
    let (m, exp) = log2_bfp_poly(value, scale);
    let sh = (exp as i32).wrapping_sub((n as u16) as i32) as i16;
    let r = if sh >= 0 {
        ((m as u32).wrapping_shl((sh as u32) & 0x1f) as i32).wrapping_add(0x8000)
    } else if sh > -31 {
        (m >> ((-(sh as i32)) as u32 & 0x1f)).wrapping_add(0x8000)
    } else {
        (m >> 31).wrapping_add(0x8000)
    };
    (r >> 16) as i16
}

/// Port of site3 (the `w`/weight producer of the y-arm): per-tap
/// `log2_tap_quantize(src[i], scale, n)` with `n = (scale+3)>>2`, into a fresh
/// `count`-word i16 buffer, then a whole-buffer `block_normalize_words`.
/// `src` is a 129-dword power spectrum (site3.in.slot1 = site1.in.slot1 =
/// arg4).  `scale = arg5 = ctx[0x60c] = the upstream stage's return
/// exponent`.  Returns `(out, bexp)`; `bexp` is the site's scalar return,
/// consumed as `s3.ret` by site4/site5.
pub(crate) fn site3_transform(src: &[i32], scale: i16, count: usize) -> (Vec<i16>, i16) {
    let n = (((scale as i32) + 3) >> 2) as i16;
    let mut out: Vec<i16> = (0..count).map(|i| log2_tap_quantize(src[i], scale, n)).collect();
    let mut bexp = n;
    block_normalize_words(&mut out, count as i16, &mut bexp);
    (out, bexp)
}

/// `buf[i] = (buf[i]*scale*2) >> 16`, `count` words, in place.
/// Bit-identical to [`crate::dec::linamp`]'s `scale_words_q15`, re-expressed
/// for `[i16]`.
pub(crate) fn scale_words_q15(buf: &mut [i16], count: i16, scale: i16) {
    let n = if count > 0 { count as usize } else { 0 };
    let s = scale as i32;
    for v in buf.iter_mut().take(n) {
        let mut c = (*v as i32).wrapping_mul(s);
        c = c.wrapping_add(c);
        *v = (c >> 16) as i16;
    }
}

/// Returns `-(0x1e - bsr(max|v|<<16))` (a non-positive norm shift), or 0 if
/// the max magnitude is 0. Bit-identical to [`crate::dec::linamp`]'s
/// `neg_max_norm_shift`.
fn neg_max_norm_shift(buf: &[i16], count: i16) -> i16 {
    let n = if count > 0 { count as usize } else { 0 };
    let mut max: i32 = 0;
    for &w in buf.iter().take(n) {
        // movzx eax,[..] then take |.| with -0x8000 -> 0x7fff saturation
        let mut a = (w as u16) as i32;
        let sa = (a as u16) as i16 as i32;
        if sa < 0 {
            a = if sa == -0x8000 {
                0x7fff
            } else {
                ((-sa) as u16) as i32
            };
        }
        let sa = (a as u16) as i16 as i32;
        let se = (max as u16) as i16 as i32;
        if sa > se {
            max = (a as u16) as i32;
        }
    }
    let max_hi = ((max as u16) as i16 as i32) << 16;
    if max_hi == 0 {
        return 0;
    }
    let t = if max_hi < 0 { !max_hi } else { max_hi };
    (-(((0x1e - bsr(t)) as u16) as i32)) as i16
}

/// Arithmetic shift every word by `shift` (left if `>=0`, else right by
/// `|shift|`), in place. Bit-identical to [`crate::dec::linamp`]'s
/// `arith_shift_words`. NB: matches the reference only for the in-place case
/// (`shift==0` is a no-op, as `dst==src`); the y-arm composite uses it
/// exclusively in place.
fn arith_shift_words(buf: &mut [i16], count: i16, shift: i16) {
    let n = if count > 0 { count as usize } else { 0 };
    let shamt = shift as i32;
    if shamt == 0 {
        return;
    }
    if shamt >= 0 {
        let sh = (shamt as u32) & 0x1f;
        for v in buf.iter_mut().take(n) {
            let e = (((*v as i32) << 16) as u32).wrapping_shl(sh) as i32;
            *v = (e >> 16) as i16;
        }
    } else {
        let sh = if shamt <= -31 {
            31u32
        } else {
            ((-shamt) as u16) as u32
        } & 0x1f;
        for v in buf.iter_mut().take(n) {
            let e = ((*v as i32) << 16) >> sh;
            *v = (e >> 16) as i16;
        }
    }
}

/// Block-normalize `count` words (in place, up to full-scale) and add the
/// applied shift to `*bexp`. Bit-identical to [`crate::dec::linamp`]'s
/// `block_normalize_words`.
pub(crate) fn block_normalize_words(buf: &mut [i16], count: i16, bexp: &mut i16) {
    let ret = neg_max_norm_shift(buf, count);
    let norm_shift = (ret as u16) as i32; // zero-extend the 16-bit return
    let shift = ((-norm_shift) as u16) as i16; // sign-extend the negated shift to 16 bits
    arith_shift_words(buf, count, shift);
    *bexp = (*bexp as i32).wrapping_add((norm_shift as u16) as i16 as i32) as i16;
}

/// The y-arm composite (= site2 of the y-arm). `dst` must hold at least `count`
/// words (the reference touches `dst[0..count]`). See the module doc for the
/// 7-step wiring.
pub(crate) fn compose_yarm_reference(dst: &mut [i16], exp: i16, arg2: u32, count: i16) {
    let cnt = count as i32;
    let h = (cnt / 2) as usize; // 128
    let s = h + 1; // 129

    // 1. R1 = array_a_stage2_refine_dispatch(dst, exp, arg2, 1)
    let r1 = array_a_stage2_refine_dispatch(dst, exp, arg2, 1);
    // 2. two in-place halvings
    dst[0] = ((dst[0] as i32) >> 1) as i16;
    dst[h] = ((dst[h] as i32) >> 1) as i16;
    // 3. self-scale by 0x70f7 over S words
    scale_words_q15(&mut dst[..s], s as i16, 0x70f7);
    // 4. exp1 = R1 - 2; block-normalize S words, exp1 += shift
    let mut exp1 = (r1 as i32 - 2) as i16;
    block_normalize_words(&mut dst[..s], s as i16, &mut exp1);
    // 5. zero-fill dst[S..count]
    for v in dst.iter_mut().take(cnt as usize).skip(s) {
        *v = 0;
    }
    // 6. R2 = array_forward_transform(dst, exp1, arg2, 1)
    let r2 = array_forward_transform(dst, exp1, arg2, 1);
    // 7. in-place cos-interp: src = dst + 1 word (clone to satisfy the borrow;
    //    all reads precede their own overwrite, so a clone reproduces the
    //    in-place pass bit-for-bit -- see quadrature_interp's own doc).
    let src: Vec<i16> = dst[1..].to_vec();
    quadrature_interp(dst, &src, r2, h as i16);
}
