// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Linear-amplitude generator for the reference decoder, wire-ready.
//!
//! Composes the DLL's STAGE-A amp chain from the decoded log-amplitudes:
//!   ml_log -> log_to_linear -> spectral_amplitude_enhance
//!          -> low_harmonic_tilt  -> hf_gain_tilt -> amp_out
//!
//! The primitives here also exist elsewhere in the crate (`enc/*`), but those
//! copies carry different signatures and are validated only over ENCODER input
//! ranges, so this module keeps its own decode-validated versions. Do not
//! deduplicate them against the encoder copies.
//!
//! Speech (mode-1) runs all four stages with every stage-enable flag set to 1,
//! which is the fixed configuration this entry point assumes.

use crate::fixops::i32r::bsr;

#[inline]
fn sx16(v: i32) -> i32 {
    (v as u16) as i16 as i32
}
#[inline]
fn zx16(v: i32) -> i32 {
    (v as u16) as i32
}
#[inline]
fn cl_of(v: i32) -> u32 {
    (v as u32) & 0x1f
}

/// norm shift: 0x1e - bsr(x<0 ? !x : x), as u16; 0 if x==0
#[inline]
fn nrm(x: i32) -> i32 {
    if x == 0 {
        return 0;
    }
    let t = if x < 0 { !x } else { x };
    ((0x1e - bsr(t)) as u16) as i32
}

/// x86 `test cx,cx / js ; cmp cx,0xffe1 / jg ; neg/sar` variable shift idiom.
#[inline]
fn vshift(mut edx: i32, ecx: i32) -> i32 {
    let sh = sx16(ecx);
    if sh >= 0 {
        edx = ((edx as u32) << cl_of(ecx)) as i32;
    } else if sh <= -31 {
        edx >>= 31;
    } else {
        edx >>= cl_of(ecx.wrapping_neg());
    }
    edx
}
/// same but with the `+0x8000 >>16` round-and-pack tail
#[inline]
fn vshift_round(mut eax: i32, ecx: i32) -> i32 {
    let sh = sx16(ecx);
    if sh >= 0 {
        eax = ((eax as u32) << cl_of(ecx)) as i32;
    } else if sh <= -31 {
        eax >>= 31;
    } else {
        eax >>= cl_of(ecx.wrapping_neg());
    }
    (eax.wrapping_add(0x8000)) >> 16
}

// --- 2^x core. x is Q16.16; returns (mantissa32, exp) ----------
const P: [i32; 6] = [0x5762, 0x4ecb, 0x71ac, 0x7aff, 0x58b9, 0x4000];
fn pow2_core(x: i32) -> (i32, i32) {
    let int_part = x >> 16;
    let frac = x - (sx16(int_part) << 16);
    let arg = sx16((frac >> 1) as u16 as i32);
    let mut acc = ((((P[0] * arg) * 2 + 0x8000) >> 3) & 0xffffe000u32 as i32) + (P[1] << 16);
    acc >>= 16;
    acc = ((((sx16(acc) * arg) * 2 + 0x8000) >> 3) & 0xffffe000u32 as i32) + ((P[2] << 16) >> 1);
    acc >>= 16;
    acc = ((((sx16(acc) * arg) * 2 + 0x8000) >> 2) & 0xffffc000u32 as i32) + ((P[3] << 16) >> 1);
    acc >>= 16;
    acc = ((((sx16(acc) * arg) * 2 + 0x8000) >> 1) & 0xffff8000u32 as i32) + (P[4] << 16);
    acc >>= 16;
    let mant = ((((sx16(acc) * arg) * 2 + 0x8000) >> 1) & 0xffff8000u32 as i32) + (P[5] << 16);
    (mant, int_part + 1)
}
// --- pow2 to a target block exponent ---------------------------
fn pow2_to_block_exponent(v: i32, a2: i32, a3: i32) -> i32 {
    let arg = vshift(sx16(v) << 16, a2 - 15);
    let (mant, outexp) = pow2_core(arg);
    vshift_round(mant, outexp.wrapping_sub(a3))
}
// --- log -> linear in place, writes block exp ------------------
fn log_to_linear(buf: &mut [i32], l: usize) -> i32 {
    for v in buf.iter_mut().take(l) {
        let a = sx16(*v);
        *v = if a > 0x77ff {
            0x77ff
        } else if a < sx16(0xffff8801u32 as i32) {
            sx16(0xffff8801u32 as i32)
        } else {
            a
        };
    }
    let mut mx = 0x80000000u32 as i32;
    for i in 0..l {
        let e = sx16(buf[i]) << 16;
        if e > mx {
            mx = e;
        }
    }
    mx = (mx >> 0xb).wrapping_add(0x10000) >> 16;
    let exp = (mx as u16) as i32;
    for i in 0..l {
        buf[i] = pow2_to_block_exponent((buf[i] as u16) as i32, 4, exp) as u16 as i32;
    }
    sx16(exp - 0xf)
}

const COS512: [i32; 512] = [
    32767, 32766, 32758, 32746, 32729, 32706, 32679, 32647, 32610, 32568, 32522, 32470, 32413,
    32352, 32286, 32214, 32138, 32058, 31972, 31881, 31786, 31686, 31581, 31471, 31357, 31238,
    31114, 30986, 30853, 30715, 30572, 30425, 30274, 30118, 29957, 29792, 29622, 29448, 29269,
    29086, 28899, 28707, 28511, 28311, 28106, 27897, 27684, 27467, 27246, 27020, 26791, 26557,
    26320, 26078, 25833, 25583, 25330, 25073, 24812, 24548, 24279, 24008, 23732, 23453, 23170,
    22884, 22595, 22302, 22006, 21706, 21403, 21097, 20788, 20475, 20160, 19841, 19520, 19195,
    18868, 18538, 18205, 17869, 17531, 17190, 16846, 16500, 16151, 15800, 15447, 15091, 14733,
    14373, 14010, 13646, 13279, 12910, 12540, 12167, 11793, 11417, 11039, 10660, 10279, 9896, 9512,
    9127, 8740, 8351, 7962, 7571, 7180, 6787, 6393, 5998, 5602, 5205, 4808, 4410, 4011, 3612, 3212,
    2811, 2411, 2009, 1608, 1206, 804, 402, 0, -402, -804, -1206, -1608, -2009, -2411, -2811,
    -3212, -3612, -4011, -4410, -4808, -5205, -5602, -5998, -6393, -6787, -7180, -7571, -7962,
    -8351, -8740, -9127, -9512, -9896, -10279, -10660, -11039, -11417, -11793, -12167, -12540,
    -12910, -13279, -13646, -14010, -14373, -14733, -15091, -15447, -15800, -16151, -16500, -16846,
    -17190, -17531, -17869, -18205, -18538, -18868, -19195, -19520, -19841, -20160, -20475, -20788,
    -21097, -21403, -21706, -22006, -22302, -22595, -22884, -23170, -23453, -23732, -24008, -24279,
    -24548, -24812, -25073, -25330, -25583, -25833, -26078, -26320, -26557, -26791, -27020, -27246,
    -27467, -27684, -27897, -28106, -28311, -28511, -28707, -28899, -29086, -29269, -29448, -29622,
    -29792, -29957, -30118, -30274, -30425, -30572, -30715, -30853, -30986, -31114, -31238, -31357,
    -31471, -31581, -31686, -31786, -31881, -31972, -32058, -32138, -32214, -32286, -32352, -32413,
    -32470, -32522, -32568, -32610, -32647, -32679, -32706, -32729, -32746, -32758, -32766, -32767,
    -32766, -32758, -32746, -32729, -32706, -32679, -32647, -32610, -32568, -32522, -32470, -32413,
    -32352, -32286, -32214, -32138, -32058, -31972, -31881, -31786, -31686, -31581, -31471, -31357,
    -31238, -31114, -30986, -30853, -30715, -30572, -30425, -30274, -30118, -29957, -29792, -29622,
    -29448, -29269, -29086, -28899, -28707, -28511, -28311, -28106, -27897, -27684, -27467, -27246,
    -27020, -26791, -26557, -26320, -26078, -25833, -25583, -25330, -25073, -24812, -24548, -24279,
    -24008, -23732, -23453, -23170, -22884, -22595, -22302, -22006, -21706, -21403, -21097, -20788,
    -20475, -20160, -19841, -19520, -19195, -18868, -18538, -18205, -17869, -17531, -17190, -16846,
    -16500, -16151, -15800, -15447, -15091, -14733, -14373, -14010, -13646, -13279, -12910, -12540,
    -12167, -11793, -11417, -11039, -10660, -10279, -9896, -9512, -9127, -8740, -8351, -7962,
    -7571, -7180, -6787, -6393, -5998, -5602, -5205, -4808, -4410, -4011, -3612, -3212, -2811,
    -2411, -2009, -1608, -1206, -804, -402, 0, 402, 804, 1206, 1608, 2009, 2411, 2811, 3212, 3612,
    4011, 4410, 4808, 5205, 5602, 5998, 6393, 6787, 7180, 7571, 7962, 8351, 8740, 9127, 9512, 9896,
    10279, 10660, 11039, 11417, 11793, 12167, 12540, 12910, 13279, 13646, 14010, 14373, 14733,
    15091, 15447, 15800, 16151, 16500, 16846, 17190, 17531, 17869, 18205, 18538, 18868, 19195,
    19520, 19841, 20160, 20475, 20788, 21097, 21403, 21706, 22006, 22302, 22595, 22884, 23170,
    23453, 23732, 24008, 24279, 24548, 24812, 25073, 25330, 25583, 25833, 26078, 26320, 26557,
    26791, 27020, 27246, 27467, 27684, 27897, 28106, 28311, 28511, 28707, 28899, 29086, 29269,
    29448, 29622, 29792, 29957, 30118, 30274, 30425, 30572, 30715, 30853, 30986, 31114, 31238,
    31357, 31471, 31581, 31686, 31786, 31881, 31972, 32058, 32138, 32214, 32286, 32352, 32413,
    32470, 32522, 32568, 32610, 32647, 32679, 32706, 32729, 32746, 32758, 32766,
];

// ===================== primitives =====================
/// saturating divide, ((|num|/|den|)>>1) signed
fn frac_divide(num: i32, den: i32) -> i32 {
    let mut absden = den;
    let mut absnum = num;
    let sign = sx16(absden) ^ absnum;
    if absnum < 0 {
        absnum = if absnum == i32::MIN {
            0x7fffffff
        } else {
            -absnum
        };
    }
    if sx16(absden) < 0 {
        absden = if sx16(absden) == -0x8000 {
            0x7fff
        } else {
            ((-sx16(absden)) as u16) as i32
        };
    }
    let denw = sx16(absden);
    let mut r = if absnum >= denw {
        (absnum / denw) >> 1
    } else {
        0
    };
    if sign < 0 {
        r = -r;
    }
    r
}
/// block-float divide. returns (mant, exp)
fn blockfloat_divide(a1: i32, a2: i32, a3: i32, a4: i32) -> (i32, i32) {
    let mut num = a1;
    let mut den = a3;
    let sign = sx16(den) ^ num;
    if num < 0 {
        num = if num == i32::MIN { 0x7fffffff } else { -num };
    }
    if sx16(den) < 0 {
        den = if sx16(den) == -0x8000 {
            0x7fff
        } else {
            ((-sx16(den)) as u16) as i32
        };
    }
    let mut exp = a2;
    let denw = sx16(den);
    if (denw << 16) <= num {
        num >>= 1;
        exp += 1;
    }
    let mut q = (num / denw) >> 1;
    q = (q as u16) as i32;
    if sign < 0 {
        q = ((-q) as u16) as i32;
    }
    let mut mant = sx16(q) << 16;
    let sh = nrm(mant);
    mant = ((mant as u32) << (sh as u32 & 0x1f)) as i32;
    mant = (mant >> 16) as u16 as i32;
    if sx16(mant) == 0 {
        return (sx16(mant), 0);
    }
    exp = exp - sh - a4;
    (sx16(mant), sx16(exp))
}
/// block-float sqrt. returns (mant32, newexp)
const SQP: [i32; 4] = [
    0xe676u16 as i16 as i32, // -6538
    0x714b,                  //  29003
    0x283f,                  //  10303
    0x5a82,
]; //  23170
fn blockfloat_sqrt(value: i32, exp_in: i32) -> (i32, i32) {
    let mut mant = value;
    if mant == 0 {
        return (0, exp_in);
    }
    let sh = {
        let t = if mant < 0 { !mant } else { mant };
        ((0x1e - bsr(t)) as u16) as i32
    };
    mant = ((mant as u32) << (sh as u32 & 0x1f)) as i32;
    mant >>= 16;
    let exp = ((exp_in as u16 as i32 - sh) as u16) as i32;
    let m = sx16((mant as u16) as i32);
    let mut acc = SQP[0] * m + (SQP[1] << 15);
    let mut a = (acc.wrapping_mul(2).wrapping_add(0x8000)) >> 16;
    acc = sx16(a) * m + (SQP[2] << 15);
    acc = (acc.wrapping_mul(2).wrapping_add(0x8000)) & 0xffff0000u32 as i32;
    if (exp & 1) != 0 {
        a = acc >> 16;
        acc = sx16(a) * SQP[3];
        acc = acc.wrapping_add(acc);
    }
    acc = (acc.wrapping_add(0x8000)) & 0xffff0000u32 as i32;
    (acc, sx16((sx16(exp) + 1) >> 1))
}
/// 64-bit norm. returns 30 - msb, 0 if zero
fn norm64(v: i64) -> i32 {
    if v == 0 {
        return 0;
    }
    let t = if (v >> 32) < 0 { !v } else { v };
    let mut i = 0i32;
    while i < 0x28 {
        let bit = 39 - i;
        if bit >= 0 && ((t >> bit) & 1) != 0 {
            break;
        }
        i += 1;
    }
    i - 9
}
/// 64-bit dot product, normalized. returns (lo32, exp)
fn normalized_dot_product(a: &[i32], b: &[i32], n: usize) -> (i32, i32) {
    let mut acc: i64 = 0;
    for i in 0..n {
        let p = sx16(a[i]).wrapping_mul(sx16(b[i]));
        let p = p.wrapping_add(p);
        acc = acc.wrapping_add(p as i64);
    }
    let sh = norm64(acc);
    let sv = if sh >= 0 {
        ((acc as u64) << (sh as u32 & 0x3f)) as i64
    } else {
        acc >> ((-sh) as u32 & 0x3f)
    };
    ((sv as u32) as i32, sx16(-sh))
}
/// saturating shift
fn saturating_shift(value: i32, shift: i32) -> i32 {
    let sh = sx16(shift);
    if sh < 0 {
        let cnt = if sh <= -31 { 31 } else { ((-sh) as u16) as i32 };
        return value >> (cnt as u32 & 0x1f);
    }
    let v = value;
    let msh = 0x1f - sh;
    let mask = ((-1i32) as u32).wrapping_shl(msh as u32 & 0x1f) as i32;
    let mut c = mask & v;
    if v < 0 {
        c = c.wrapping_sub(mask);
    }
    if c != 0 {
        return if v < 0 { i32::MIN } else { 0x7fffffff };
    }
    ((v as u32).wrapping_shl(sh as u32 & 0x1f)) as i32
}
/// normalize array by shift
fn arith_shift_words(buf: &mut [i32], n: usize, shift: i32) {
    let sh = sx16(shift);
    if sh == 0 {
        return;
    }
    if sh >= 0 {
        for i in 0..n {
            buf[i] = sx16((((sx16(buf[i]) << 16) as u32) << (sh as u32 & 0x1f)) as i32 >> 16);
        }
    } else {
        let cnt = if sh <= -31 { 31 } else { ((-sh) as u16) as i32 };
        for i in 0..n {
            buf[i] = sx16(((sx16(buf[i]) << 16) >> (cnt as u32 & 0x1f)) >> 16);
        }
    }
}
/// saturating shift array
fn saturating_shift_array(buf: &mut [i32], n: usize, shift: i32) {
    for i in 0..n {
        buf[i] = sx16(saturating_shift(sx16(buf[i]) << 16, shift) >> 16);
    }
}
/// scale array
pub(crate) fn scale_words_q15(buf: &mut [i32], n: usize, scale: i32) {
    for i in 0..n {
        let mut c = sx16(buf[i]) * sx16(scale);
        c = c.wrapping_add(c);
        buf[i] = sx16(c >> 16);
    }
}
/// -(norm of max |v|)
fn neg_max_norm_shift(buf: &[i32], n: usize) -> i32 {
    let mut mx = 0i32;
    for i in 0..n {
        let mut a = (buf[i] as u16) as i32;
        if sx16(a) < 0 {
            a = if sx16(a) == -0x8000 {
                0x7fff
            } else {
                ((-sx16(a)) as u16) as i32
            };
        }
        if sx16(a) > sx16(mx) {
            mx = (a as u16) as i32;
        }
    }
    let mxsh = sx16(mx) << 16;
    if mxsh == 0 {
        return 0;
    }
    let t = if mxsh < 0 { !mxsh } else { mxsh };
    -(((0x1e - bsr(t)) as u16) as i32)
}
/// block-normalize array, updates bexp
pub(crate) fn block_normalize_words(buf: &mut [i32], n: usize, bexp: &mut i32) {
    let sh = (neg_max_norm_shift(buf, n) as u16) as i32;
    arith_shift_words(buf, n, sx16(-sh));
    *bexp = sx16(*bexp + sx16(sh));
}

// ===================== stage functions =====================
/// RM0 / RM1 + cos array
fn compute_comb_moments(
    ptr: &[i32],
    bexp: i32,
    l: usize,
    omega0: i32,
) -> (i32, i32, i32, i32, Vec<i32>) {
    let (lo, mut rm0_exp) = normalized_dot_product(ptr, ptr, l);
    let rm0_mant = sx16(lo >> 16);
    let two_bexp = bexp.wrapping_add(bexp);
    rm0_exp = sx16(rm0_exp + sx16(two_bexp));
    let mut cosarr = vec![0i32; l.max(1)];
    if rm0_mant == 0 {
        return (rm0_mant, rm0_exp, 0, 0, cosarr);
    }
    let step = (sx16(omega0) << 10) >> 4;
    let mut acc: i64 = 0;
    let mut phase = step;
    for i in 0..l {
        let idx = sx16(phase >> 16);
        let cosv = COS512[idx as usize] as u16 as i32;
        cosarr[i] = cosv;
        let a = sx16(ptr[i]);
        let mut t = a.wrapping_mul(a);
        t = t.wrapping_add(t);
        t >>= 16;
        t = sx16(t).wrapping_mul(sx16(cosv));
        t = t.wrapping_add(t);
        acc = acc.wrapping_add(t as i64);
        phase = phase.wrapping_add(step);
    }
    let sh = norm64(acc);
    let sv = if sh >= 0 {
        ((acc as u64) << (sh as u32 & 0x3f)) as i64
    } else {
        acc >> ((-sh) as u32 & 0x3f)
    };
    let rm1_exp = sx16(two_bexp - sh);
    let mut rm1_mant = sx16(((sv as u32) as i32) >> 16);
    if rm1_mant == -0x8000 {
        rm1_mant = -0x7fff;
    }
    (rm0_mant, rm0_exp, rm1_mant, rm1_exp, cosarr)
}

/// persistent RM0 IIR smoother
fn smooth_rm0_iir(st_mant: &mut i32, st_exp: &mut i32, rm0_mant: i32, rm0_exp: i32) {
    let cur_exp = (*st_exp as u16) as i32;
    let new_exp = rm0_exp;
    let max_exp = if sx16(new_exp) > sx16(cur_exp) {
        (new_exp as u16) as i32
    } else {
        cur_exp
    };
    let mut acc = sx16(*st_mant).wrapping_mul(0xf334);
    acc = vshift(acc, cur_exp - max_exp);
    let mut term = sx16(rm0_mant).wrapping_mul(0xcccc);
    term = vshift(term, (new_exp - max_exp) - 4);
    acc = acc.wrapping_add(term);
    let sh = nrm(acc);
    let mut out_exp = max_exp - sh;
    acc = ((acc as u32) << (sh as u32 & 0x1f)) as i32;
    if acc == 0 || sx16(out_exp) <= -17 {
        acc = 0x7fff0000u32 as i32;
        out_exp = -17;
    }
    acc >>= 16;
    *st_mant = sx16(acc);
    *st_exp = sx16(out_exp);
}

/// builds V1 = (RM0^2+RM1^2)*Q and V2 = RM0*RM1*Q
fn build_enhance_coeffs(
    rm0_mant: i32,
    rm0_exp: i32,
    rm1_mant: i32,
    rm1_exp: i32,
    omega0: i32,
) -> (i32, i32, i32, i32) {
    let rm0w = sx16(rm0_mant);
    let mut rm0sq = rm0w.wrapping_mul(rm0w);
    let rm0_e2 = rm0_exp.wrapping_add(rm0_exp);
    let qexp = ((rm0_e2 + 1) as u16) as i32;
    rm0sq = rm0sq.wrapping_add(rm0sq);
    rm0sq = vshift(rm0sq, rm0_e2 - qexp);
    let rm1w = sx16(rm1_mant);
    let sumsq;
    let diffsq;
    if rm1w != 0 {
        let mut rm1sq = rm1w.wrapping_mul(rm1w);
        rm1sq = rm1sq.wrapping_add(rm1sq);
        rm1sq = vshift(rm1sq, (rm1_exp.wrapping_add(rm1_exp)) - qexp);
        let sum = rm1sq.wrapping_add(rm0sq);
        let diff = rm0sq.wrapping_sub(rm1sq);
        sumsq = ((sum >> 16) as u16) as i32;
        let d = ((diff >> 16) as u16) as i32;
        if sx16(d) == 0 {
            return (0x7fff, 0, 0, 0);
        } // give-up path
        diffsq = d;
    } else {
        let e = rm0sq >> 16;
        sumsq = (e as u16) as i32;
        diffsq = sumsq;
    }
    let mut a = sx16(diffsq).wrapping_mul(rm0w);
    a = a.wrapping_add(a);
    a >>= 16;
    let mut den = sx16(a).wrapping_mul(sx16(omega0));
    den = den.wrapping_add(den);
    let sh = if den != 0 { nrm(den) } else { 0 };
    let denw = (((den as u32) << (sh as u32 & 0x1f)) as i32) >> 16;
    let q = frac_divide(0x3cd013a9u32 as i32, denw);
    let mut eadj = sh - qexp - rm0_exp + 4;
    let qw = sx16((q as u16) as i32);
    eadj = (eadj as u16) as i32;
    // V1
    let mut v1 = sx16(sumsq).wrapping_mul(qw);
    v1 = v1.wrapping_add(v1);
    let sh3 = nrm(v1);
    let v1_mant = sx16((((v1 as u32) << (sh3 as u32 & 0x1f)) as i32) >> 16);
    let mut v1_exp = sx16(-sh3);
    if v1_mant != 0 {
        v1_exp = sx16(v1_exp + qexp + eadj);
    }
    // V2
    let mut v2 = rm1w.wrapping_mul(rm0w);
    v2 = v2.wrapping_add(v2);
    v2 >>= 16;
    v2 = sx16(v2).wrapping_mul(qw);
    v2 = v2.wrapping_add(v2);
    v2 >>= 16;
    let v2q = sx16(v2) << 15;
    let sh4 = if v2q != 0 { nrm(v2q) } else { 0 };
    let v2_mant = sx16((((v2q as u32) << (sh4 as u32 & 0x1f)) as i32) >> 16);
    let mut v2_exp = sx16(-sh4);
    if v2_mant != 0 {
        v2_exp = sx16(v2_exp + rm0_exp + eadj + rm1_exp + 2);
    }
    (v1_mant, v1_exp, v2_mant, v2_exp)
}

/// per-harmonic W_l = clamp((V1 - 2*V2*cos)^(1/4), 0.5, 1.2)
// fixed-point chain: parameters are separate mantissa/exponent/length words by nature
#[allow(clippy::too_many_arguments)]
fn apply_harmonic_weights(
    ptr: &mut [i32],
    bexp: &mut i32,
    l: i32,
    v2_mant: i32,
    v2_exp: i32,
    v1_mant: i32,
    v1_exp: i32,
    cosarr: &[i32],
) {
    let start = ((sx16(l) >> 3) as u16) as i32;
    let be = (*bexp as u16) as i32;
    arith_shift_words(ptr, start as usize, sx16(be - (be + 1)));
    let e = if sx16(v1_exp) > sx16(v2_exp) {
        v1_exp + 1
    } else {
        v2_exp + 1
    };
    let e = (e as u16) as i32;
    if sx16(start) >= sx16(l) {
        *bexp = sx16(*bexp + 1);
        return;
    }
    let c_v2 = v2_exp - e;
    let d_v1 = v1_exp - e;
    let v2m = sx16(v2_mant);
    let v1m_sh = sx16(v1_mant) << 16;
    for i in (start as usize)..(l as usize) {
        let mut cterm = sx16(cosarr[i]).wrapping_mul(v2m);
        cterm = cterm.wrapping_add(cterm);
        cterm = vshift(cterm, c_v2);
        let mut arg = vshift(v1m_sh, d_v1);
        arg = arg.wrapping_sub(cterm);
        let w;
        if arg != 0 {
            let (m1, e1) = blockfloat_sqrt(arg, (e as u16) as i32);
            let s1 = sx16(m1 >> 16);
            let amp = sx16(ptr[i]);
            let mut c = s1.wrapping_mul(amp);
            let e2 = sx16(e1 + sx16(*bexp));
            c = c.wrapping_add(c);
            let (m2, e3) = blockfloat_sqrt(c, e2);
            if sx16(e3) > 1 {
                w = 0x4ccd;
            } else {
                let mut a = vshift(m2, e3 - 1);
                a = ((a >> 16) as u16) as i32;
                if sx16(a) > 0x4ccd {
                    w = 0x4ccd;
                } else if sx16(a) < 0x2000 {
                    w = 0x2000;
                } else {
                    w = a;
                }
            }
        } else {
            w = 0x2000;
        }
        let mut c = sx16(ptr[i]).wrapping_mul(sx16(w));
        c = c.wrapping_add(c);
        ptr[i] = sx16(c >> 16);
    }
    *bexp = sx16(*bexp + 1);
}

/// energy renormalization M''_l = M'_l * sqrt(RM0 / sum M'^2)
fn energy_renormalize(ptr: &mut [i32], bexp: &mut i32, l: usize, rm0_mant: i32, rm0_exp: i32) {
    block_normalize_words(ptr, l, bexp);
    let (lo, ex) = normalized_dot_product(ptr, ptr, l);
    let sum_mant = sx16(lo >> 16);
    let sumexp = sx16(ex + sx16(bexp.wrapping_add(*bexp)));
    if sum_mant == 0 {
        return;
    }
    let (qm, qe) = blockfloat_divide(sx16(rm0_mant) << 16, rm0_exp, sum_mant, sumexp);
    let (sm, se) = blockfloat_sqrt(sx16((qm as u16) as i32) << 16, qe);
    let sqrt_mant = ((sm >> 16) as u16) as i32;
    scale_words_q15(ptr, l, sx16(sqrt_mant));
    *bexp = sx16(*bexp + sx16(se));
    if sx16(*bexp) > 0 {
        saturating_shift_array(ptr, l, sx16(*bexp));
        *bexp = 0;
    }
}

/// the mode-1 enhancement (RM0/RM1 comb, W_l shaping, energy renorm)
fn spectral_amplitude_enhance(
    ptr: &mut [i32],
    bexp: &mut i32,
    st_mant: &mut i32,
    st_exp: &mut i32,
    omega0: i32,
    l: usize,
) {
    let (rm0_mant, rm0_exp, rm1_mant, rm1_exp, cosarr) =
        compute_comb_moments(ptr, *bexp, l, omega0);
    smooth_rm0_iir(st_mant, st_exp, rm0_mant, rm0_exp);
    if sx16(rm0_mant) <= 0 {
        return;
    }
    let (v1m, v1e, v2m, v2e) = build_enhance_coeffs(rm0_mant, rm0_exp, rm1_mant, rm1_exp, omega0);
    apply_harmonic_weights(ptr, bexp, l as i32, v2m, v2e, v1m, v1e, &cosarr);
    energy_renormalize(ptr, bexp, l, rm0_mant, rm0_exp);
}

/// low-harmonic tilt.
///   amp[k] *= ((k+1)*omega0 / cutoff)^2  in Q15, while acc < min(cutoff,0x3333)
fn low_harmonic_tilt(amp: &mut [i32], omega0: i32, cutoff_in: i32) {
    let mut cut = zx16(cutoff_in);
    if sx16(cutoff_in) > 0x3333 {
        cut = 0x3333;
    }

    let mut cutn = sx16(cut) << 16;
    let shc = nrm(cutn);
    cutn = ((cutn as u32) << cl_of(shc)) as i32;
    cutn >>= 16;
    let (r, expa) = blockfloat_divide(
        0x40000000,
        1,
        cutn,
        (0xfffffffcu32 as i32).wrapping_sub(shc),
    );

    let mut acc = zx16(omega0);
    if sx16(acc) >= sx16(cut) {
        return;
    }

    let inv = sx16(r);
    let mut k: usize = 0;
    loop {
        let cur = sx16(acc);
        let mut prod = inv.wrapping_mul(cur);
        prod = prod.wrapping_add(prod);
        let sh = nrm(prod);
        let expw = (expa as u16).wrapping_sub(sh as u16).wrapping_sub(4);
        prod = ((prod as u32) << cl_of(sh)) as i32;
        prod >>= 16;
        let e = expw as i32;
        let m = sx16(zx16(prod));
        let mut sq = m.wrapping_mul(m);
        let exp2 = e.wrapping_add(e);
        sq = sq.wrapping_add(sq);
        sq = vshift(sq, exp2);
        sq >>= 16;
        let w = sx16(sq);

        let a = sx16(amp[k]);
        let mut c = w.wrapping_mul(a);
        c = c.wrapping_add(c);
        c >>= 16;
        amp[k] = sx16(c);
        k += 1;

        let cur32 = cur << 16;
        let om32 = sx16(omega0) << 16;
        let sum = om32.wrapping_add(cur32);
        let res = if ((om32 ^ sum) & (sum ^ cur32)) < 0 {
            if cur32 < 0 {
                0x80000000u32 as i32
            } else {
                0x7fffffff
            }
        } else {
            sum
        };
        acc = zx16(res >> 16);
        if sx16(acc) >= sx16(cut) {
            break;
        }
        if k >= amp.len() {
            break;
        } // never fires (verified): cutoff<=0x3333 bounds the loop
    }
}

/// HF gain (5/6 * freq, above 0x4ccd) with per-harmonic
/// block exponents, then re-align every harmonic to the max exponent.
fn hf_gain_tilt(amp: &mut [i32], l: usize, omega0: i32, bexp: &mut i32) {
    let mut exps = vec![0i32; l.max(1)];
    let mut local = vec![0i32; l.max(1)];
    let cnt = l as i32;

    if 0 < sx16(cnt) {
        for i in 0..l {
            exps[i] = zx16(*bexp);
        }
        local[..l].copy_from_slice(&amp[..l]);
    }

    let om = sx16(omega0);
    let mut top = sx16(cnt).wrapping_mul(om);
    top = ((top as u32) << 13) as i32;
    top >>= 16;
    let mut cur = zx16(top);
    let mut idx = zx16(l as i32 - 1);

    if sx16(cur) > 0x4ccd {
        let blockexp = zx16(*bexp);
        let step = (om << 16) >> 3;
        loop {
            let curw = sx16(cur);
            let k = sx16(idx);
            let g = sx16(curw.wrapping_mul(0xd556) >> 16);
            let a = sx16(local[k as usize]);
            let mut prod = g.wrapping_mul(a);
            prod = prod.wrapping_add(prod);
            let sh = nrm(prod);
            idx -= 1;
            let newexp = blockexp.wrapping_sub(sh).wrapping_add(1);
            let mut next = (curw << 16).wrapping_sub(step);
            exps[k as usize] = zx16(newexp);
            prod = ((prod as u32) << cl_of(sh)) as i32;
            prod >>= 16;
            next >>= 16;
            local[k as usize] = sx16(prod);
            cur = zx16(next);
            if sx16(cur) <= 0x4ccd {
                break;
            }
            if idx < 0 {
                break;
            } // never fires (verified): cur crosses 0x4ccd first
        }
    }

    let mut maxe = zx16(exps[0]);
    if 1 < sx16(cnt) {
        for i in 1..l {
            let a = zx16(exps[i]);
            if sx16(maxe) < sx16(a) {
                maxe = a;
            }
        }
    }

    if 0 < sx16(cnt) {
        let mut d: i32 = 0;
        loop {
            let i = sx16(d) as usize;
            let mut v = sx16(local[i]) << 16;
            let sh = zx16((exps[i] as u16).wrapping_sub(maxe as u16) as i32);
            v = vshift(v, sh);
            v >>= 16;
            d += 1;
            amp[i] = sx16(v);
            if sx16(d) >= sx16(cnt) {
                break;
            }
        }
    }
    *bexp = sx16(maxe);
}

/// Compose the reference linear-amp chain from the decoded log-amplitudes.
///
/// Runs all four DLL stages (every stage-enable flag set to 1, the speech
/// configuration): log_to_linear -> spectral_amplitude_enhance -> low_harmonic_tilt -> hf_gain_tilt.
///
/// * `ml_log`       - decoded log-amplitude words M_l (length >= `l`)
/// * `l`            - harmonic count L
/// * `omega0`       - fundamental (word at struct+0xc)
/// * `gain_applied` - low_harmonic_tilt tilt cutoff (the `gain_applied` capture column)
///
/// Returns `(amp_out[0..l], bexp_out)`, bit-exact to the DLL amp array.
pub(crate) fn amp_from_mllog(
    ml_log: &[i16],
    l: usize,
    omega0: i32,
    gain_applied: i32,
) -> (Vec<i16>, i32) {
    let mut buf: Vec<i32> = ml_log.iter().take(l).map(|&v| v as i32).collect();
    // log_to_linear/spectral_amplitude_enhance read/write in place; hf_gain_tilt indexes local scratch of length l.
    let mut bexp = log_to_linear(&mut buf, l);
    // The IIR smoother state (smooth_rm0_iir) does NOT affect the amp array, so a
    // throwaway per-call state reproduces the array exactly.
    let (mut sm, mut se) = (0i32, 0i32);
    spectral_amplitude_enhance(&mut buf, &mut bexp, &mut sm, &mut se, omega0, l);
    low_harmonic_tilt(&mut buf, omega0, gain_applied);
    hf_gain_tilt(&mut buf, l, omega0, &mut bexp);
    let out: Vec<i16> = buf[..l].iter().map(|&v| sx16(v) as i16).collect();
    (out, bexp)
}

/// [`amp_from_mllog`] without the two gain-folding stages — the Annex-T
/// variant.
///
/// The reference's tone branch jumps past `0x1030cb40` and `0x1030c700`,
/// which are exactly [`low_harmonic_tilt`] and [`hf_gain_tilt`] here (this port names them after
/// their RVAs). `low_harmonic_tilt` is where the loudness-gain register enters, so skipping
/// it is what keeps a tone's level a pure function of `A_D` instead of a
/// function of whatever speech preceded it — consistent with the measured
/// 0.711 dB/count over the whole A_D range and with tones not using the
/// `gain_slew` path at all.
pub(crate) fn amp_from_mllog_ungained(ml_log: &[i16], l: usize, omega0: i32) -> (Vec<i16>, i32) {
    let mut buf: Vec<i32> = ml_log.iter().take(l).map(|&v| v as i32).collect();
    let mut bexp = log_to_linear(&mut buf, l);
    let (mut sm, mut se) = (0i32, 0i32);
    spectral_amplitude_enhance(&mut buf, &mut bexp, &mut sm, &mut se, omega0, l);
    let out: Vec<i16> = buf[..l].iter().map(|&v| sx16(v) as i16).collect();
    (out, bexp)
}
