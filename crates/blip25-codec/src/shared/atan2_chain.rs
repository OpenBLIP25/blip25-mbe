//! Full block-floating-point `atan2`-shaped chain used by voicing's `S` angle
//! computation. A direct, instruction-by-instruction port -- not a
//! reimplementation -- for the same reason [`super::band_decompress`] and
//! [`super::atan2_bfp_divide`] are: the exact fixed-point rounding / shift /
//! saturation behaviour is not re-derivable from the public spec.
//!
//! ## What this covers
//!
//! Three reference functions, wired together exactly as the caller
//! ([`atan2_dispatch`]) wires them:
//!
//! - [`bfp_gt`]: a block-floating-point "greater than" compare (aligns two
//!   `(mantissa, exponent)` pairs by shift-count, then does a plain signed
//!   compare).
//! - [`bfp_add`] / [`bfp_sub`]: block-floating-point add/subtract with their own
//!   re-normalize.
//! - [`atan2_poly`]: a 3-way range-reduction dispatcher (large ratio ->
//!   reciprocal via [`super::atan2_bfp_divide::bfp_divide`]; medium ratio -> a
//!   half-angle-identity combine via [`bfp_add`]/[`bfp_sub`] then a divide;
//!   small ratio -> direct) feeding one shared 4-stage Horner polynomial in
//!   `u = x^2`.
//! - [`atan2_dispatch`]: the outer entry point -- handles the `cx==0`/`di==0`
//!   special cases, computes the quadrant bias constant/seed, block-floating-point
//!   normalizes both `di` and `cx`, calls the divide once for the normalized
//!   ratio, calls [`atan2_poly`], then combines the polynomial result with the
//!   quadrant bias and re-normalizes for the final return.
//!
//! Pinned exact on both mantissa and exponent against captured `(di, cx)`
//! ground truth spanning all 3 internal range-reduction branches, plus a fuzz
//! sweep over quadrant/threshold edge values (`±32768`, `0`, `±1`, the
//! range-reduction thresholds `19777`/`27146`, the quadrant bias constant
//! `±25736`).
//!
//! ## Two constants that read wrong and must not be "corrected"
//!
//! * The first Horner-stage coefficient is the literal 32-bit immediate
//!   `0xe38e` = **+58254**, used by a full 32-bit `imul`. It is NOT `-7282`:
//!   the encoding is `imul r32,r32,imm32` with a zero-extended upper half, so
//!   reading the 16-bit pattern as sign-extended gives the wrong sign and
//!   magnitude.
//! * The third Horner stage's exponent-tracking subtraction uses the constant
//!   `-2`, set two stages earlier and never touched since -- not the stage's own
//!   running "saved" exponent.
//!
//! Either error silently ships a subtly wrong `S`.
//!
//! ## What this does NOT close
//!
//! `S` itself -- the value written into the voicing context struct -- is the
//! outer `S` function's overall return, and that function's tail (a second
//! divide plus two more undecoded sub-calls) is a separate open track.
//!
//! This chain's `(di, cx)` inputs are [`super::windowed_complex_correlation`]'s output, and
//! `windowed_complex_correlation`'s formula is closed given its 3 input streams. What remains open is
//! one hop further upstream: those streams' (`w`/`A`/`C`) construction from raw
//! PCM. `heap_src` is a SEPARATE gap -- array `B`'s source, not this chain's.
//!
//! So this module is a complete, verified building block that is not yet
//! reachable from raw PCM and cannot move `voicing_b1_accuracy` by itself.

use super::atan2_bfp_divide::bfp_divide;

use crate::fixops::i32r::{s16, s32};

#[inline]
fn u16v(x: i32) -> u16 {
    x as u16
}

/// Shared "shift left by a signed count, saturating shift-right for large
/// negative counts" idiom (`test cx,cx; js ...; shl ...` / `cmp cx,0xffe1;
/// jg ...; sar ...,0x1f`), used pervasively throughout this codec's
/// fixed-point DSP code (also present in [`super::atan2_bfp_divide`]'s own
/// normalize step). `count >= 0` -> left shift (masked to 0..31, matching
/// real x86 `shl` with a `cl` operand); `count` in `-30..=-1` -> arithmetic
/// right shift by `-count`; `count <= -31` -> arithmetic right shift by 31
/// (sign-fill saturate).
fn shl_sar_sat(value: i32, count: i16) -> i32 {
    let count = count as i32;
    if count >= 0 {
        let sh = (count as u32) & 31;
        value.wrapping_shl(sh)
    } else if count > -31 {
        let sh = (-count) as u32;
        value >> sh
    } else {
        value >> 31
    }
}

/// Shared "block-floating-point renormalize via `bsr`" idiom: given a
/// nonzero 32-bit value, returns the shift-count `30 - highest_set_bit`
/// (of the value itself if non-negative, or its bitwise NOT if negative --
/// the one's-complement trick this codec uses throughout to find the
/// leading run length of a negative mantissa, NOT `-value`, which differs
/// from a plain negate for values of the form `-2^k`). Returns `0` for a
/// zero input (matches the real DLL's own `je`-skip-bsr short-circuit).
fn normalize_bsr(value: i32) -> u16 {
    if value == 0 {
        return 0;
    }
    let bsr_input: u32 = if value >= 0 {
        value as u32
    } else {
        !(value as u32)
    };
    // Real x86 `bsr eax,eax` on a zero input leaves `eax` unmodified; the
    // only way `bsr_input` is 0 here is `value == -1` (`!(-1) == 0`), in
    // which case `eax` was already 0 going in (matches the real chain's
    // observed behaviour with zero risk of an undefined shift below).
    let bitpos: i32 = if bsr_input == 0 {
        0
    } else {
        31 - bsr_input.leading_zeros() as i32
    };
    (30i32.wrapping_sub(bitpos)) as u16
}

/// Block-floating-point "greater than" compare. Aligns
/// `(p, e1)` and `(r, e2)` (each a 16-bit mantissa + a shift-count-style
/// exponent) by exponent difference, then does a plain signed 32-bit
/// compare of the aligned mantissas. Returns `true` iff the first pair's
/// value is strictly greater than the second's.
pub(crate) fn bfp_gt(p: i32, e1: i32, r: i32, e2: i32) -> bool {
    let p16 = s16(p);
    let mut p_al: i32 = (p16 as i32).wrapping_shl(16);
    let e_p: i32 = if p_al == 0 { u16v(e2) as i32 } else { e1 };
    let r16 = s16(r);
    let mut r_al: i32 = (r16 as i32).wrapping_shl(16);
    let e_r: i32 = if r_al == 0 { u16v(e_p) as i32 } else { e2 };
    let diff = e_r.wrapping_sub(e_p);
    let sh = s16(diff);
    if sh > 0 {
        let sh_neg = s16((sh as i32).wrapping_neg());
        if sh_neg <= -31 {
            p_al >>= 31;
        } else if sh_neg < 0 {
            p_al >>= (-(sh_neg as i32)) as u32;
        }
        // sh_neg == 0: shl by 0, no-op.
    } else if sh < 0 {
        if sh <= -31 {
            r_al >>= 31;
        } else {
            r_al >>= (-(sh as i32)) as u32;
        }
    }
    p_al > r_al
}

/// Block-floating-point add. `q`/`e_q` and `r`/`e_r` are
/// each a 16-bit mantissa + exponent. Returns `(mantissa, exponent)` of
/// `q + r`.
pub(crate) fn bfp_add(q: i32, e_q: i32, r: i32, e_r: i32) -> (i16, i16) {
    let q16 = s16(q);
    let r16 = s16(r);
    let e_q16 = s16(e_q);
    let e_r16 = s16(e_r);
    if q16 == 0 {
        return if r16 != 0 { (r16, e_r16) } else { (0, 0) };
    }
    let diff_r_minus_q = (e_r16 as i32) - (e_q16 as i32);
    if diff_r_minus_q >= 32 {
        return if r16 != 0 { (r16, e_r16) } else { (0, 0) };
    }
    if r16 == 0 {
        return (q16, e_q16);
    }
    let diff_q_minus_r = (e_q16 as i32) - (e_r16 as i32);
    if diff_q_minus_r >= 32 {
        return (q16, e_q16);
    }
    let common: i32 = if e_q16 > e_r16 {
        e_q16 as i32 + 1
    } else {
        e_r16 as i32 + 1
    };
    let common_u16 = u16v(common) as i32;
    let mut q_al: i32 = (q16 as i32).wrapping_shl(16);
    q_al = shl_sar_sat(q_al, s16((e_q16 as i32) - common_u16));
    let mut r_al: i32 = (r16 as i32).wrapping_shl(16);
    r_al = shl_sar_sat(r_al, s16((e_r16 as i32) - common_u16));
    let total = r_al.wrapping_add(q_al);
    finish_bfp_combine(total, common_u16)
}

/// Block-floating-point subtract. `a`/`b` and `c`/`d` are
/// each a 16-bit mantissa + exponent. Returns `(mantissa, exponent)` of
/// `c - a`.
pub(crate) fn bfp_sub(a: i32, b: i32, c: i32, d: i32) -> (i16, i16) {
    let a16 = s16(a);
    let c16 = s16(c);
    let b16 = s16(b);
    let d16 = s16(d);
    if a16 == 0 {
        return if c16 != 0 { (c16, d16) } else { (0, 0) };
    }
    let diff_d_minus_b = (d16 as i32) - (b16 as i32);
    if diff_d_minus_b >= 32 {
        return if c16 != 0 { (c16, d16) } else { (0, 0) };
    }
    if c16 == 0 {
        return negate_saturating(a16, b16);
    }
    let diff_b_minus_d = (b16 as i32) - (d16 as i32);
    if diff_b_minus_d >= 32 {
        return negate_saturating(a16, b16);
    }
    let common: i32 = if b16 > d16 {
        b16 as i32 + 1
    } else {
        d16 as i32 + 1
    };
    let common_u16 = u16v(common) as i32;
    let mut a_al: i32 = (a16 as i32).wrapping_shl(16);
    a_al = shl_sar_sat(a_al, s16((b16 as i32) - common_u16));
    let mut c_al: i32 = (c16 as i32).wrapping_shl(16);
    c_al = shl_sar_sat(c_al, s16((d16 as i32) - common_u16));
    let total = c_al.wrapping_sub(a_al);
    finish_bfp_combine(total, common_u16)
}

fn negate_saturating(a16: i16, b16: i16) -> (i16, i16) {
    if a16 == i16::MIN {
        (0x7fff, b16)
    } else {
        (s16(-(a16 as i32)), b16)
    }
}

fn finish_bfp_combine(total: i32, common_u16: i32) -> (i16, i16) {
    let shiftcount = normalize_bsr(total);
    let sh = (shiftcount as u32) & 31;
    let total2 = total.wrapping_shl(sh);
    let mant = u16v(total2 >> 16);
    if mant == 0 {
        (0, 0)
    } else {
        let exp_out = s16(u16v(common_u16.wrapping_sub(shiftcount as i32)) as i32);
        (s16(mant as i32), exp_out)
    }
}

/// First Horner-stage coefficient: the literal 32-bit `imul` immediate
/// `0x0000e38e` (= **+58254**, NOT the 16-bit-sign-extended `-7282` an
/// earlier round's static read described -- the real encoding is
/// `imul edx,eax,imm32` with the upper 16 bits of the immediate zero, so
/// the actual multiplier used at runtime is the full positive value).
const COEF0: i32 = 0xe38e;
const COEF1_U: i32 = 0x4925;
const COEF2_U: i32 = 0x6666;
const COEF3_U: i32 = 0x5555;

/// 3-way range-reduction dispatcher + shared 4-stage Horner
/// polynomial (arctan-shaped, evaluated in `u = ratio^2`). `ratio` is the
/// 16-bit input value; `exp_in` is its block-floating-point exponent
/// (from the caller's own prior normalize/divide step). Returns
/// `(mantissa, exponent)`.
pub(crate) fn atan2_poly(ratio: i32, exp_in: i32) -> (i16, i16) {
    let ratio16 = s16(ratio);
    let (ar, sign_neg): (i32, bool) = if ratio16 < 0 {
        if ratio16 == i16::MIN {
            (0x7fff, true)
        } else {
            (-(ratio16 as i32), true)
        }
    } else {
        (ratio16 as i32, false)
    };

    let branch: u8 = if bfp_gt(ar, exp_in, 19777, 2) {
        1
    } else if bfp_gt(ar, exp_in, 27146, -1) {
        2
    } else {
        3
    };

    let (red_ratio, red_exp, bias_mant, bias_exp): (u16, i32, i32, i32) = match branch {
        1 => {
            let (mant, exp) = bfp_divide(0x4000_0000u32 as i32, 1, ar, exp_in).unwrap_or((0, 0));
            let mut neg_recip: i32 = (s16(mant as i32) as i32).wrapping_shl(16);
            neg_recip = if neg_recip == i32::MIN {
                i32::MAX
            } else {
                neg_recip.wrapping_neg()
            };
            let reduced = u16v(neg_recip >> 16);
            (reduced, exp as i32, 0x6488, 1)
        }
        2 => {
            let (add_mant, add_exp) = bfp_add(0x4000, 1, ar, exp_in);
            let (sub_mant, sub_exp) = bfp_sub(0x4000, 1, ar, exp_in);
            let sub_hi: i32 = (sub_mant as i32).wrapping_shl(16);
            let neg_needed = sub_hi < 0;
            let sub_abs = if !neg_needed {
                sub_hi
            } else if sub_hi == i32::MIN {
                i32::MAX
            } else {
                sub_hi.wrapping_neg()
            };
            let (dmant, dexp) =
                bfp_divide(sub_abs, sub_exp as i32, add_mant as i32, add_exp as i32)
                    .unwrap_or((0, 0));
            let reduced = if neg_needed {
                let mut m: i32 = (dmant as i32).wrapping_shl(16);
                m = if m == i32::MIN {
                    i32::MAX
                } else {
                    m.wrapping_neg()
                };
                u16v(m >> 16)
            } else {
                u16v(dmant as i32)
            };
            (reduced, dexp as i32, 0x6488, 0)
        }
        _ => (u16v(ar), exp_in, 0, 0),
    };

    let x: i32 = s16(red_ratio as i32) as i32;
    let u2: i32 = s32(x.wrapping_mul(x)).wrapping_add(x.wrapping_mul(x));
    let sc1 = normalize_bsr(u2);
    let u2n = u2.wrapping_shl((sc1 as u32) & 31);
    let u_exp = u16v((2 * red_exp).wrapping_sub(sc1 as i32) as i32);
    let x_norm: i16 = s16(u16v(u2n >> 16) as i32);

    // stage 0 (coefficient c0)
    let mut acc: i32 = COEF0.wrapping_mul(x_norm as i32);
    acc = shl_sar_sat(acc, s16((u_exp as i32) - 1));
    acc = acc.wrapping_sub(COEF1_U.wrapping_shl(16));

    // stage 1 (coefficient c1)
    let sc2 = normalize_bsr(acc);
    acc = acc.wrapping_shl((sc2 as u32) & 31);
    let tmp1 = s16((-2i32).wrapping_sub(sc2 as i32));
    acc >>= 16;
    acc = s16(u16v(acc) as i32) as i32;
    acc = acc.wrapping_mul(x_norm as i32);
    let mut exp: i32 = (u_exp as i32).wrapping_add(2);
    exp = exp.wrapping_add(u16v(tmp1 as i32) as i32);
    acc = acc.wrapping_add(acc);
    acc = shl_sar_sat(acc, s16(exp));
    acc = acc.wrapping_add(COEF2_U.wrapping_shl(16));

    // stage 2 (coefficient c2) -- exponent tracker uses the constant -2
    // set two stages earlier (freshly loaded, NOT `u_exp`).
    let sc3 = normalize_bsr(acc);
    acc = acc.wrapping_shl((sc3 as u32) & 31);
    let exp3 = (-2i32).wrapping_sub(sc3 as i32);
    exp = (u_exp as i32).wrapping_add(1);
    acc >>= 16;
    acc = s16(u16v(acc) as i32) as i32;
    acc = acc.wrapping_mul(x_norm as i32);
    exp = exp.wrapping_add(u16v(exp3) as i32);
    acc = acc.wrapping_add(acc);
    acc = shl_sar_sat(acc, s16(exp));
    acc = acc.wrapping_sub(COEF3_U.wrapping_shl(16));

    // stage 3 (coefficient c3, final)
    let sc4 = normalize_bsr(acc);
    acc = acc.wrapping_shl((sc4 as u32) & 31);
    let neg1_minus_sc4 = (-1i32).wrapping_sub(sc4 as i32);
    acc >>= 16;
    acc = s16(u16v(acc) as i32) as i32;
    acc = acc.wrapping_mul(x_norm as i32);
    let exp4 = u16v(neg1_minus_sc4) as i32;
    acc = acc.wrapping_add(acc);
    let sc5 = normalize_bsr(acc);
    acc = acc.wrapping_shl((sc5 as u32) & 31);
    exp = (u_exp as i32).wrapping_sub(sc5 as i32);
    let exp5 = exp4.wrapping_add(exp);
    acc >>= 16;
    acc = s16(u16v(acc) as i32) as i32;
    acc = acc.wrapping_mul(x);
    acc = acc.wrapping_add(acc);
    acc = shl_sar_sat(acc, s16(exp5));

    // combine with the original (un-squared) reduced ratio, then with the
    // branch's bias constant.
    let x_hi: i32 = x.wrapping_shl(16);
    acc = acc.wrapping_add(x_hi);
    acc = shl_sar_sat(acc, s16(red_exp.wrapping_sub(1)));

    let mut bias_val: i32 = (s16(bias_mant) as i32).wrapping_shl(16);
    bias_val = shl_sar_sat(bias_val, s16(bias_exp - 1));
    acc = acc.wrapping_add(bias_val);

    let (mant_mag, exp_out): (u16, i16) = if acc == 0 {
        (0, 0)
    } else {
        let sc_f = normalize_bsr(acc);
        let acc_n = acc.wrapping_shl((sc_f as u32) & 31);
        let exp_f = s16(u16v(1i32.wrapping_sub(sc_f as i32)) as i32);
        (u16v(acc_n >> 16), exp_f)
    };

    if mant_mag == 0 {
        return (0, 0);
    }

    let mut mant_final = s16(mant_mag as i32);
    if sign_neg {
        mant_final = if mant_final == i16::MIN {
            0x7fff
        } else {
            s16(u16v(-(mant_final as i32)) as i32)
        };
    }
    (mant_final, exp_out)
}

/// Full `atan2`-shaped chain. `y_di` / `x_cx` are the two 16-bit inputs (`di`,
/// `cx`); `corr_exp` is the shared block-floating-point exponent both inputs are
/// pre-scaled by at the call site, which pushes the same value as both of the
/// divide's exponent seed candidates. Returns `(mantissa, exponent)`, matching
/// the reference's `ax` return value plus its 5th-argument output write.
pub(crate) fn atan2_dispatch(y_di: i32, corr_exp: i32, x_cx: i32) -> (i16, i16) {
    let x = s16(x_cx);
    let y = s16(y_di);

    if x == 0 {
        return match y.cmp(&0) {
            std::cmp::Ordering::Less => (s16(0xffff9b78u32 as i32), 1),
            std::cmp::Ordering::Equal => (0, 0),
            std::cmp::Ordering::Greater => (0x6488, 1),
        };
    }
    if y == 0 {
        return if x > 0 { (0, 0) } else { (0x6488, 2) };
    }

    let quadrant = (if x < 0 { 2 } else { 0 }) | (if y < 0 { 1 } else { 0 });
    let (bias_mant, bias_exp): (i32, i32) = match quadrant {
        2 => (0x6488, 2),
        3 => (s16(0xffff9b78u32 as i32) as i32, 2),
        _ => (0, 0),
    };

    // Normalize x: nx = zero_extend16((x_q16 << sh_x) >> 16).
    let x_q16: i32 = (x as i32).wrapping_shl(16);
    let sh_x = normalize_bsr(x_q16);
    let nx_shifted = x_q16.wrapping_shl((sh_x as u32) & 31);
    let nx_val = u16v(nx_shifted >> 16) as i32;

    // Normalize y: ny stays the wide (un-truncated) shifted value -- it is
    // fed directly as the divide's own 32-bit `X` argument.
    let y_q16: i32 = (y as i32).wrapping_shl(16);
    let sh_y = normalize_bsr(y_q16);
    let ny_val = y_q16.wrapping_shl((sh_y as u32) & 31);

    let e0 = corr_exp.wrapping_sub(sh_y as i32);
    let ebias = corr_exp.wrapping_sub(sh_x as i32);

    let (div_mant, div_exp) = bfp_divide(ny_val, e0, nx_val, ebias).unwrap_or((0, 0));
    let ratio = u16v(div_mant as i32) as i32;

    let (poly_mant, poly_exp) = atan2_poly(ratio, div_exp as i32);

    let mut exp: i32 = (poly_exp as i32).wrapping_sub(bias_exp);
    exp = exp.wrapping_sub(1);
    let mut acc: i32 = (s16(poly_mant as i32) as i32).wrapping_shl(16);
    acc = shl_sar_sat(acc, s16(exp));

    let mut bias_val: i32 = (s16(bias_mant) as i32).wrapping_shl(16);
    bias_val >>= 1;
    acc = acc.wrapping_add(bias_val);

    if acc == 0 {
        let exp_f = s16(u16v(bias_exp + 1) as i32);
        return (0, exp_f);
    }

    let shiftcount_f = normalize_bsr(acc);
    let acc_n = acc.wrapping_shl((shiftcount_f as u32) & 31);
    let exp_f = s16(u16v(bias_exp.wrapping_sub(shiftcount_f as i32).wrapping_add(1)) as i32);
    let mant_out = s16(u16v(acc_n >> 16) as i32);
    (mant_out, exp_f)
}
