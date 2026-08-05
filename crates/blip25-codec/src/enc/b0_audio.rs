//! AUDIO-ONLY b0 pitch chain: prefiltered PCM -> b0, with NO capture-fed side channels.
//!
//! Scores **195/199 (98.0%) on BOTH conformance clips from audio alone**, with no
//! capture data on disk.
//!
//! # The 12 side channels, derived
//! The reference's per-frame pitch step consumes 12 internal side channels
//! (expo/low/high/b980_a/c62c/c626/arg3/cand/bands/shifts/p4base/e530_seed); fed
//! captured ones it scores 195/199 (voiced) / 197/199 (mark). This module derives
//! all 12 from the prefiltered PCM instead:
//!
//!   * `expo`        = 2 * the inverse-FFT butterfly stage's returned exponent
//!     (that value is doubled before e460 tail-returns it)   3184/3184
//!   * `cand/bands/shifts` = ab80 block exponents per adf0 pass-generation  3152/3152
//!   * `p4base`      = this frame's ab80 pass1 rows[0..10]                  3184/3184
//!   * `c62c`        = octave_halve_decide's own prologue periodicity_ratio() complement at the chosen
//!     octave (halved ? conf5 : conf6)                       199/199
//!   * `c626`        = own raw_r622[f-1]
//!   * `b980_a[8]`   = the ported ctx+0x614 tracker
//!   * `arg3`        = a7 (ctx+0x618) read AFTER the pitch_tracker_update update
//!   * `low`         = the coarse_pitch divisor 4/5/6
//!   * `si`/`arg5`   = const (1,1);  `a11` substitute = 0 (costs 0 vs capture)
//!
//! Cost vs the capture-fed tracker: voiced 195 (unchanged), mark 197 -> 195. This is
//! the whole b0 byte from audio alone, the only column a from-PCM encoder can be
//! scored in.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

// ======================= common int helpers =======================
use crate::fixops::acc64::{
    bf_shift, bshift, bsr, denorm, i16s, m32, norm_edx, norm_sh, norm_shift, s32, sar, sat_add, shl,
};

// ======================= e100 DP (poste100 score) =======================
mod e100 {
    fn sat32_64(x: i64) -> i64 {
        if x > i32::MAX as i64 {
            i32::MAX as i64
        } else if x < i32::MIN as i64 {
            i32::MIN as i64
        } else {
            x
        }
    }
    fn shift_lr(v: i64, sh: i32) -> i64 {
        if sh >= 0 {
            v << (sh as u32)
        } else {
            v >> ((-sh) as u32)
        }
    }
    fn accumulate_aligned(dst: &mut [i64], ea: Option<i32>, src: &[i32], eb: i32) -> i32 {
        let ea = ea.unwrap_or(eb);
        let e = ea.max(eb);
        let sha = e - ea;
        let shb = e - eb;
        for i in 0..dst.len() {
            let a = if sha > 0 {
                dst[i] >> (sha as u32)
            } else {
                dst[i]
            };
            let b = if shb > 0 {
                (src[i] as i64) >> (shb as u32)
            } else {
                src[i] as i64
            };
            dst[i] = a + b;
        }
        e
    }
    fn blockexp(buf: &[i64]) -> i32 {
        let mut maxabs: u64 = 0;
        for &v in buf {
            let a = v.unsigned_abs();
            if a > maxabs {
                maxabs = a;
            }
        }
        if maxabs == 0 {
            return 0;
        }
        (63 - maxabs.leading_zeros() as i32) - 30
    }
    fn pack(acc: &[i64]) -> [i16; 32] {
        let s = blockexp(acc);
        let shift = -s;
        let mut out = [0i16; 32];
        for i in 0..32 {
            let v = sat32_64(shift_lr(acc[i], shift));
            let c = sat32_64(v + 0x8000);
            out[i] = (c >> 16) as i16;
        }
        out
    }
    /// run e100: seq scorevecs (16 x 32 already lag-0,1 masked) + per-seq exponent.
    pub(crate) fn run(
        scorevecs: &[[i32; 32]; 16],
        expo: &[i32; 16],
        st: Option<&[i16; crate::enc::dp_score::ST_WORDS]>,
    ) -> [i16; 32] {
        if crate::enc::dp_score::enabled() {
            return crate::enc::dp_score::run(scorevecs, expo, st);
        }
        let mut acc = vec![0i64; 32];
        let mut eacc: Option<i32> = None;
        for seq in 0..16 {
            let mut a = scorevecs[seq];
            a[0] = 0;
            a[1] = 0;
            eacc = Some(accumulate_aligned(&mut acc, eacc, &a, expo[seq]));
        }
        pack(&acc)
    }
}
// ======================= coarse_pitch (a1 coarse pitch) =======================
mod b7b0m {
    use super::*;
    fn argmax16(sc: &[i32], ptr: i32, cnt: i32) -> i32 {
        if cnt <= 0 {
            return 0;
        }
        let mut best = i16s(sc[ptr as usize] as i64);
        let mut bi = 0i32;
        for i in 1..cnt {
            let v = i16s(sc[(ptr + i) as usize] as i64);
            if v > best {
                best = v;
                bi = i;
            }
        }
        bi
    }
    fn sum16(sc: &[i32], ptr: i32, cnt: i32) -> i32 {
        if cnt <= 0 {
            return 0;
        }
        let mut a = 0i32;
        for i in 0..cnt {
            a = a.wrapping_add(i16s(sc[(ptr + i) as usize] as i64));
        }
        a
    }
    fn frac_divide_sat(num: i32, den: i32) -> i32 {
        let mut n_abs = num;
        let mut den_w = den as i64 & 0xffff_ffff;
        let sign = m32((i16s(den_w) as i64) ^ (num as i64));
        if n_abs < 0 {
            n_abs = if n_abs == i32::MIN {
                0x7fff_ffff
            } else {
                -n_abs
            };
        }
        let d_lo = i16s(den_w);
        if d_lo < 0 {
            den_w = if (den_w & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-d_lo) & 0xffff) as i64
            };
        }
        let divisor = i16s(den_w & 0xffff);
        let mut res;
        if n_abs < divisor {
            res = 0;
        } else if sar(n_abs, 16) >= divisor {
            res = 0x7fff;
        } else {
            res = sar((n_abs as i64 / divisor as i64) as i32, 1);
        }
        if s32(sign) < 0 {
            res = -res;
        }
        res
    }
    fn blockfloat_divide(arg1: i32, arg2: i32, arg3: i32, arg4: i32) -> (i32, i32) {
        let mut dividend = arg1;
        let mut divisor_w = arg3 as i64 & 0xffff_ffff;
        let sign = m32((i16s(divisor_w) as i64) ^ (arg1 as i64));
        if dividend < 0 {
            dividend = if dividend == i32::MIN {
                0x7fff_ffff
            } else {
                -dividend
            };
        }
        let d_lo = i16s(divisor_w);
        if d_lo < 0 {
            divisor_w = if (divisor_w & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-d_lo) & 0xffff) as i64
            };
        }
        let mut exp = arg2;
        let divisor = i16s(divisor_w & 0xffff);
        if shl(divisor, 16) <= dividend {
            dividend = sar(dividend, 1);
            exp = exp.wrapping_add(1);
        }
        let mut q = if divisor != 0 {
            (dividend as i64 / divisor as i64) as i32
        } else {
            0
        };
        q = sar(q, 1) & 0xffff;
        if s32(sign) < 0 {
            q = (-i16s(q as i64)) & 0xffff;
        }
        let q16 = shl(i16s(q as i64), 16);
        let shift = if q16 == 0 {
            0
        } else {
            let t = if q16 >= 0 { q16 } else { s32(!(q16 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let mant = sar(shl(q16, shift), 16) & 0xffff;
        if i16s(mant as i64) == 0 {
            return (0, 0);
        }
        exp = exp.wrapping_sub(shift & 0xffff).wrapping_sub(arg4);
        (mant, i16s(exp as i64))
    }
    fn blockfloat_add(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> (i32, i32) {
        let a_mant = i16s(arg0 as i64);
        let b_exp = arg3;
        if a_mant == 0 {
            if i16s(arg2 as i64) != 0 {
                return (i16s(arg2 as i64) & 0xffff, i16s(arg3 as i64));
            }
            return (0, 0);
        }
        let a_exp = arg1;
        let b_exp_s = i16s(b_exp as i64);
        let a_exp_s = i16s(a_exp as i64);
        if b_exp_s - a_exp_s >= 0x20 {
            if i16s(arg2 as i64) != 0 {
                return (i16s(arg2 as i64) & 0xffff, i16s(arg3 as i64));
            }
            return (0, 0);
        }
        let c_mant = i16s(arg2 as i64);
        if c_mant == 0 {
            return (i16s(arg0 as i64) & 0xffff, i16s(arg1 as i64));
        }
        if i16s(a_exp as i64) - i16s(b_exp as i64) >= 0x20 {
            return (i16s(arg0 as i64) & 0xffff, i16s(arg1 as i64));
        }
        let max_exp = if i16s(arg1 as i64) > i16s(arg3 as i64) {
            i16s(arg1 as i64) + 1
        } else {
            i16s(arg3 as i64) + 1
        };
        let res_exp = max_exp & 0xffff;
        let v1 = bshift(
            shl(i16s(arg0 as i64), 16),
            i16s(arg1 as i64) - i16s(res_exp as i64),
        );
        let v2 = bshift(
            shl(i16s(arg2 as i64), 16),
            i16s(arg3 as i64) - i16s(res_exp as i64),
        );
        let sum = v1.wrapping_add(v2);
        let shift = if sum == 0 {
            0
        } else {
            let t = if sum >= 0 { sum } else { s32(!(sum as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let mant = sar(shl(sum, shift), 16) & 0xffff;
        if i16s(mant as i64) == 0 {
            return (0, 0);
        }
        (mant, i16s(res_exp as i64) - i16s((shift & 0xffff) as i64))
    }
    fn parabolic_interp(k: i32, score: &[i32], count: i32) -> i32 {
        let ki = i16s(k as i64);
        let k16 = shl(ki, 16);
        let c = if i16s(k as i64) == 0 {
            1
        } else {
            let cm1 = i16s(count as i64) - 1;
            if ki == cm1 {
                k - 1
            } else {
                k
            }
        };
        let cc = i16s(c as i64);
        let sl = i16s(score[(cc - 1) as usize] as i64);
        let sr = i16s(score[(cc + 1) as usize] as i64);
        let sc_ = i16s(score[cc as usize] as i64);
        let left = shl(sl, 16);
        let center = shl(sc_, 16);
        let right = shl(sr, 16);
        let mut curv = sar(left, 1).wrapping_sub(center);
        curv = curv.wrapping_add(sar(right, 1));
        curv = sar(curv, 16);
        if i16s((curv & 0xffff) as i64) >= 0 {
            return k16;
        }
        let left_q = sar(left, 2);
        let right_q = sar(right, 2);
        let num = left_q.wrapping_sub(right_q);
        let den = curv & 0xffff;
        let q = frac_divide_sat(num, den);
        let mut peak = i16s((q & 0xffff) as i64).wrapping_add(shl(cc, 15));
        peak = peak.wrapping_add(peak);
        peak
    }
    struct Mem {
        b: [u8; 0x100],
    }
    impl Mem {
        fn new() -> Self {
            Mem { b: [0; 0x100] }
        }
        fn r16u(&self, o: i32) -> i32 {
            let o = o as usize;
            (self.b[o] as i32) | ((self.b[o + 1] as i32) << 8)
        }
        fn r16s(&self, o: i32) -> i32 {
            i16s(self.r16u(o) as i64)
        }
        fn w16(&mut self, o: i32, v: i32) {
            let o = o as usize;
            let v = (v & 0xffff) as u32;
            self.b[o] = (v & 0xff) as u8;
            self.b[o + 1] = ((v >> 8) & 0xff) as u8;
        }
        fn w32(&mut self, o: i32, v: i32) {
            let o = o as usize;
            let v = v as u32;
            for i in 0..4 {
                self.b[o + i] = ((v >> (8 * i)) & 0xff) as u8;
            }
        }
    }
    fn harmonic_bounds(mem: &mut Mem, l: i32, mant: i32, exp: i32, it: i32, count: i32) -> i32 {
        let it_lo = i16s(it as i64);
        let mut acc = s32(shl(i16s(it_lo as i64), 16) as i64 + 0x8000);
        let nsh = if acc != 0 {
            let t = if acc >= 0 { acc } else { s32(!(acc as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let normb = (0xf - nsh) & 0xffff;
        acc = shl(acc, nsh);
        let nexp = normb;
        let edxv = exp - 15 + i16s(nexp as i64);
        let mant_s = i16s(mant as i64);
        let mut prod = i16s(sar(acc, 16) as i64);
        prod = prod.wrapping_mul(mant_s);
        prod = prod.wrapping_add(prod);
        prod = sar(bshift(prod, edxv), 16);
        let prod_lo = i16s((prod & 0xffff) as i64);
        mem.w16(l + 2, prod_lo);
        if prod_lo >= i16s(count as i64) {
            return 1;
        }
        let nexp_neg = i16s((-i16s(nexp as i64)) as i64);
        let e = s32(-(bshift(0x4000_0000u32 as i32, nexp_neg) as i64));
        acc = acc.wrapping_add(2i32.wrapping_mul(e));
        let mut a = i16s(sar(acc, 16) as i64).wrapping_mul(mant_s);
        a = a.wrapping_add(a);
        a = sar(bshift(a, edxv), 16).wrapping_add(1);
        mem.w16(l, a & 0xffff);
        let e2 = bshift(0x2000_0000, nexp_neg);
        acc = acc.wrapping_add(e2);
        let mut a2 = i16s(sar(acc, 16) as i64).wrapping_mul(mant_s);
        a2 = a2.wrapping_add(a2);
        a2 = sar(bshift(a2, edxv), 16).wrapping_add(1);
        mem.w16(l + 4, a2 & 0xffff);
        let step3 = bshift(0x4000_0000u32 as i32, nexp_neg);
        let mut a3 = i16s(sar(step3.wrapping_add(acc), 16) as i64).wrapping_mul(mant_s);
        a3 = a3.wrapping_add(a3);
        a3 = sar(bshift(a3, edxv), 16);
        mem.w16(l + 6, a3 & 0xffff);
        if i16s((a3 & 0xffff) as i64) < mem.r16s(l + 4) {
            return 1;
        }
        0
    }
    fn band_ratio(sum2: i32, sum1: i32, mem: &mut Mem, expslot: i32) -> i32 {
        let s2 = sum2;
        let diff = sum1.wrapping_sub(s2);
        if diff <= 0 {
            let res = if s2 < 0 { 0 } else { 0x6400 };
            mem.w16(expslot, 7);
            return res;
        }
        let shift = if s2 != 0 {
            let t = if s2 >= 0 { s2 } else { s32(!(s2 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let neg_sh = (-shift) & 0xffff;
        let be = bsr(diff);
        let s2n = shl(s2, shift);
        let bexp = 0x1e - be;
        let bexp_lo = bexp & 0xffff;
        let diff_n = sar(shl(diff, bexp_lo), 16);
        let negcx = i16s((-i16s(bexp_lo as i64)) as i64);
        let (mant, ex) = blockfloat_divide(s2n, i16s(neg_sh as i64), diff_n, negcx);
        let mut res = mant & 0xffff;
        let ex_lo = i16s(ex as i64);
        if ex_lo > 7 {
            res = if s2n < 0 { 0 } else { 0x6400 };
            mem.w16(expslot, 7);
            return res;
        }
        if ex_lo == 7 && (res & 0xffff) >= 0x6400 {
            res = 0x6400;
        }
        if s2n < 0 {
            res = 0;
        }
        mem.w16(expslot, ex_lo & 0xffff);
        res
    }
    fn refine_pitch_estimate(mem: &mut Mem, s: i32, it: i32) {
        let itv = i16s(it as i64);
        let mut p1 = s32(mem.r16s(s + 4) as i64 * itv as i64);
        p1 = p1.wrapping_add(p1);
        let sh1 = if p1 != 0 {
            let t = if p1 >= 0 { p1 } else { s32(!(p1 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut x1 = (mem.r16u(s + 6) - (sh1 & 0xffff)) & 0xffff;
        p1 = shl(p1, sh1);
        x1 = (x1 + 0xf) & 0xffff;
        p1 = sar(p1, 16);
        let hi1 = i16s(p1 as i64);
        let exp1 = x1 & 0xffff;
        let mut p2 = s32(mem.r16s(s + 8) as i64 * hi1 as i64);
        p2 = p2.wrapping_add(p2);
        let sh2 = if p2 != 0 {
            let t = if p2 >= 0 { p2 } else { s32(!(p2 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut x2 = (mem.r16u(s + 0xa) - (sh2 & 0xffff)) & 0xffff;
        p2 = shl(p2, sh2);
        x2 = (x2 + exp1) & 0xffff;
        p2 = sar(p2, 16);
        let e2 = x2 & 0xffff;
        let (mant, ex) = blockfloat_add(mem.r16u(s + 0xc), mem.r16u(s + 0xe), p2, e2);
        mem.w16(s + 0xc, mant);
        mem.w16(s + 0xe, ex);
        let mut p3 = s32(hi1 as i64 * itv as i64);
        p3 = p3.wrapping_add(p3);
        let sh3 = if p3 != 0 {
            let t = if p3 >= 0 { p3 } else { s32(!(p3 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut exp3 = (exp1 - (sh3 & 0xffff)) & 0xffff;
        exp3 = (exp3 + 0xf) & 0xffff;
        p3 = sar(shl(p3, sh3), 16);
        let (mant2, ex2) = blockfloat_add(mem.r16u(s + 0x10), mem.r16u(s + 0x12), p3, exp3);
        mem.w16(s + 0x10, mant2);
        mem.w16(s + 0x12, ex2);
        let m2 = i16s(mant2 as i64);
        if m2 > 0 {
            let (r, rex) = blockfloat_divide(
                shl(mem.r16s(s + 0xc), 16),
                mem.r16u(s + 0xe),
                i16s(m2 as i64),
                mem.r16u(s + 0x12),
            );
            mem.w16(s, r);
            mem.w16(s + 2, rex);
        }
    }
    pub(crate) fn coarse_pitch(score: &[i32], low: i32, high: i32) -> i32 {
        let mut mem = Mem::new();
        let l = 0x10;
        let s = 0x1c;
        let cnt0 = high - low;
        let k = low + argmax16(score, low, cnt0);
        let interp = parabolic_interp(k, score, high);
        let (mant, expc) = norm_edx(interp);
        mem.w16(s, mant);
        mem.w16(s + 2, expc);
        mem.w16(s + 8, mant);
        mem.w16(s + 0xa, expc);
        mem.w32(s + 0xc, 0);
        mem.w32(s + 0x10, 0);
        let mut it = 1i32;
        let mut hi = high;
        loop {
            let (a1m, a2e) = (mem.r16u(s), mem.r16u(s + 2));
            let r = harmonic_bounds(&mut mem, l, a1m, a2e, it, hi);
            if r == 1 {
                break;
            }
            let loc6 = mem.r16u(l + 6);
            let loc4 = mem.r16u(l + 4);
            let count2 = (loc6 - loc4 + 1) & 0xffff;
            if i16s(it as i64) > 1 {
                let idx = argmax16(score, i16s(loc4 as i64), i16s(count2 as i64));
                let k2 = idx + i16s(loc4 as i64);
                let e2 = parabolic_interp(k2, score, hi);
                let (m2, x2) = norm_edx(e2);
                mem.w16(s + 0xa, x2);
                mem.w16(s + 8, m2);
            }
            let loc0 = mem.r16s(l);
            let loc2 = mem.r16s(l + 2);
            let cnt3 = (loc2 - loc0 + 1) & 0xffff;
            let s1 = sum16(score, loc0, i16s(cnt3 as i64));
            let s2 = sum16(score, i16s(loc4 as i64), i16s(count2 as i64));
            let ret = band_ratio(s2, s1, &mut mem, s + 6);
            mem.w16(s + 4, ret & 0xffff);
            refine_pitch_estimate(&mut mem, s, it);
            hi = high;
            it += 1;
            if i16s(it as i64) >= 0x10 {
                break;
            }
        }
        denorm(mem.r16u(s), mem.r16u(s + 2))
    }
    pub(crate) fn a1_clamp(preclamp: i32) -> i32 {
        let mut c = i16s(preclamp as i64);
        c = c.clamp(0x666, 0x3e00);
        c
    }
}
// ======================= octave_halve_decide (octave-halve) =======================
mod b980m {
    use super::*;
    fn blockfloat_divide(arg1: i32, arg2: i32, arg3: i32, arg4: i32) -> (i32, i32) {
        let mut dividend = arg1;
        let mut divisor_w = arg3 as i64 & 0xffff_ffff;
        let sign = m32((i16s(divisor_w) as i64) ^ (arg1 as i64));
        if dividend < 0 {
            dividend = if dividend == i32::MIN {
                0x7fff_ffff
            } else {
                -dividend
            };
        }
        let d_lo = i16s(divisor_w);
        if d_lo < 0 {
            divisor_w = if (divisor_w & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-d_lo) & 0xffff) as i64
            };
        }
        let mut exp = arg2;
        let divisor = i16s(divisor_w & 0xffff);
        if shl(divisor, 16) <= dividend {
            dividend = sar(dividend, 1);
            exp = exp.wrapping_add(1);
        }
        let mut q = if divisor != 0 {
            (dividend as i64 / divisor as i64) as i32
        } else {
            0
        };
        q = sar(q, 1) & 0xffff;
        if s32(sign) < 0 {
            q = (-i16s(q as i64)) & 0xffff;
        }
        let q16 = shl(i16s(q as i64), 16);
        let shift = if q16 == 0 {
            0
        } else {
            let t = if q16 >= 0 { q16 } else { s32(!(q16 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let mant = sar(shl(q16, shift), 16) & 0xffff;
        if i16s(mant as i64) == 0 {
            return (0, 0);
        }
        exp = exp.wrapping_sub(shift & 0xffff).wrapping_sub(arg4);
        (mant, i16s(exp as i64))
    }
    fn sum16(sc: &[i32], ptr: i32, cnt: i32) -> i32 {
        if cnt <= 0 {
            return 0;
        }
        let mut a = 0i32;
        for i in 0..cnt {
            let idx = (ptr + i) as usize;
            let v = if idx < sc.len() {
                i16s(sc[idx] as i64)
            } else {
                0
            };
            a = a.wrapping_add(v);
        }
        a
    }
    fn shift_scale(val: i32, sh: i32) -> i32 {
        let shw = i16s(sh as i64);
        if shw < 0 {
            if shw > -31 {
                return sar(val, (-shw) & 31);
            }
            return sar(val, 31);
        }
        let nbits = 0x1f - shw;
        let mask = shl(-1, nbits);
        let mut c = mask & val;
        if val >= 0 {
        } else {
            c = c.wrapping_sub(mask);
        }
        if c != 0 {
            return if val < 0 {
                i32::MIN.wrapping_add(1)
            } else {
                0x7fff_ffff
            };
        }
        shl(val, shw)
    }
    fn comb(score: &[i32], a1: i32, div: i32) -> i32 {
        let limit = 0x20i32;
        let sh = div - 15;
        let mut step = bshift(shl(i16s(a1 as i64), 16), sh);
        let quarter = sar(step, 2);
        let mut pos = step.wrapping_sub(quarter).wrapping_add(0x10000);
        let mut sum = 0i32;
        let mut lag = sar(pos, 16) & 0xffff;
        if i16s(lag as i64) >= limit {
            return sum;
        }
        step = sar(step, 1);
        loop {
            pos = pos.wrapping_add(step);
            let mut nextlag = i16s(sar(pos, 16) as i64);
            if nextlag > limit {
                nextlag = limit;
            }
            let count = nextlag - i16s(lag as i64);
            pos = pos.wrapping_add(step);
            sum = sum.wrapping_add(sum16(score, i16s(lag as i64), count));
            lag = sar(pos, 16) & 0xffff;
            if i16s(lag as i64) >= limit {
                break;
            }
        }
        sum
    }
    fn periodicity_ratio(combmant: i32, blockexp: i32, score: &[i32], a1: i32, arg4: i32) -> i32 {
        let sh0 = arg4 - 16;
        let pos = bshift(shl(i16s(a1 as i64), 16), sh0);
        let lag = sar(pos.wrapping_add(0x10000), 16) & 0xffff;
        let count = 0x20 - i16s(lag as i64);
        let sum = sum16(score, i16s(lag as i64), count);
        if sum <= 0 {
            return 0x7fff;
        }
        let bsrv = bsr(sum);
        let nsh = (0x1e - bsrv) & 0xffff;
        let sum_mant = sar(shl(sum, nsh), 16);
        let negnsh = (-i16s(nsh as i64)) & 0xffff;
        let (mant, ex) = blockfloat_divide(
            shl(i16s(combmant as i64), 16),
            i16s(blockexp as i64),
            i16s(sum_mant as i64),
            i16s(negnsh as i64),
        );
        let _mant_lo = mant & 0xffff;
        if i16s(ex as i64) >= 1 {
            return 0;
        }
        let mut e = shl(i16s(mant as i64), 16);
        e = bshift(e, i16s(ex as i64));
        // BUG2 FIX: result = ((0x7fff0000 - e) >> 16) as u16 (COMPLEMENT of ratio)
        let t = if e == i32::MIN {
            0x7fff_ffffi32.wrapping_add(0x7fff_0000u32 as i32)
        } else {
            e.wrapping_neg().wrapping_add(0x7fff_0000u32 as i32)
        };
        sar(t, 16) & 0xffff
    }
    fn halve(a1: i32) -> i32 {
        let e = ((i16s(a1 as i64)) << 16) >> 1;
        (e.wrapping_add(0x8000) >> 16) & 0xffff
    }
    fn identity(a1: i32) -> i32 {
        i16s(a1 as i64) & 0xffff
    }
    fn abssat(x: i32) -> i32 {
        if x >= 0 {
            x
        } else if x == i32::MIN {
            0x7fff_ffff
        } else {
            -x
        }
    }
    /// (raw_r622, is_halve)
    /// PROBE: the div=6 periodicity-ratio complement computed in octave_halve_decide's prologue.
    /// Candidate identity for the refine-ctx field `c62c`.
    pub(crate) fn conf6(a1: i32, score: &[i32]) -> i32 {
        let comb6 = comb(score, a1, 6);
        let shift6 = norm_sh(comb6);
        let mant6 = i16s(sar(shl(comb6, shift6), 16) as i64);
        let blockexp6 = (-i16s(shift6 as i64)) & 0xffff;
        periodicity_ratio(mant6, blockexp6, score, a1, 6)
    }
    /// Same, at div=5.
    pub(crate) fn conf5(a1: i32, score: &[i32]) -> i32 {
        let comb5 = comb(score, a1, 5);
        let shift5 = norm_sh(comb5);
        let mant5 = i16s(sar(shl(comb5, shift5), 16) as i64);
        let blockexp5 = (-i16s(shift5 as i64)) & 0xffff;
        periodicity_ratio(mant5, blockexp5, score, a1, 5)
    }
    // Port artifacts: the original writes `bsr_local`/`skip_continuity_halve`
    // at points where the values are provably never read again; the dead
    // stores are kept to mirror the reference control flow 1:1.
    #[allow(unused_assignments, unused_variables)]
    pub(crate) fn octave_halve_decide(
        a1: i32,
        score: &[i32],
        a4: i32,
        a5: i32,
        a6: i32,
        a7: i32,
        a8: i32,
        a9: i32,
        a10: i32,
        a11: i32,
    ) -> (i32, bool) {
        let a1s = i16s(a1 as i64);
        let comb6 = comb(score, a1, 6);
        let shift6 = norm_sh(comb6);
        let mant6 = i16s(sar(shl(comb6, shift6), 16) as i64);
        let blockexp6 = (-i16s(shift6 as i64)) & 0xffff;
        let mut bsr_local = if comb6 != 0 {
            bsr(if comb6 >= 0 {
                comb6
            } else {
                s32(!(comb6 as i64))
            })
        } else {
            0
        };
        let r180a = periodicity_ratio(mant6, blockexp6, score, a1, 6);
        if a1s <= 0xccc {
            return (identity(a1), false);
        }
        let comb5 = comb(score, a1, 5);
        let (mant5, blockexp5);
        if comb5 == 0 {
            mant5 = i16s(sar(shl(comb5, 0), 16) as i64);
            blockexp5 = 0i32;
        } else {
            let bsr5 = bsr(if comb5 >= 0 {
                comb5
            } else {
                s32(!(comb5 as i64))
            });
            bsr_local = bsr5;
            let nsh5 = (0x1e - bsr5) & 0xffff;
            mant5 = i16s(sar(shl(comb5, nsh5), 16) as i64);
            blockexp5 = (-i16s(nsh5 as i64)) & 0xffff;
        }
        let m6s15 = (i16s(mant6 as i64) << 16) >> 1;
        let be6 = i16s(blockexp6 as i64);
        let be5 = i16s(blockexp5 as i64);
        let mut skip_continuity_halve = be5 > be6 + 1;
        if !skip_continuity_halve {
            let shift_arg = be5 - be6 - 1;
            let val = i16s(r180a as i64).wrapping_mul(0x8a60);
            let ss = shift_scale(val, shift_arg);
            if ss <= m6s15 {
                skip_continuity_halve = true;
            } else {
                let mut try_a7_halve = i16s(a6 as i64) < 3;
                let mut halve_now = false;
                if !try_a7_halve {
                    let c = i16s(a9 as i64).wrapping_mul(0xe666u32 as i32);
                    let diff_a9 = i16s((sar((i16s(a1 as i64) << 16).wrapping_sub(c), 16)) as i64);
                    if diff_a9 < 0 {
                        try_a7_halve = true;
                    } else {
                        let diff_a9b = i16s(
                            (sar(
                                (i16s(a1 as i64).wrapping_mul(0xe8bau32 as i32))
                                    .wrapping_sub(i16s(a9 as i64) << 16),
                                16,
                            )) as i64,
                        );
                        if diff_a9b <= 0 {
                            skip_continuity_halve = true;
                        } else {
                            try_a7_halve = true;
                        }
                    }
                }
                if !skip_continuity_halve {
                    if try_a7_halve {
                        let a7_scaled = i16s(a7 as i64).wrapping_mul(0xd99au32 as i32);
                        let diff_a7 =
                            i16s((sar((i16s(a1 as i64) << 16).wrapping_sub(a7_scaled), 16)) as i64);
                        if diff_a7 < 0 {
                            halve_now = true;
                        } else {
                            let diff_a7b = i16s(
                                (sar(
                                    (i16s(a1 as i64).wrapping_mul(0xde9cu32 as i32))
                                        .wrapping_sub(i16s(a7 as i64) << 16),
                                    16,
                                )) as i64,
                            );
                            if diff_a7b > 0 {
                                halve_now = true;
                            } else {
                                skip_continuity_halve = true;
                            }
                        }
                    }
                    if halve_now {
                        return (halve(a1), true);
                    }
                }
            }
        }
        let shift_arg2 = be5 - be6 - 1;
        let val2 = i16s(mant5 as i64).wrapping_mul(0xe8bau32 as i32);
        let ss2 = shift_scale(val2, shift_arg2);
        let mut bp = a8;
        let mut go_final = false;
        if ss2 > m6s15 {
            let mut a10_above_branch = false;
            let mut a10_below_branch = false;
            if i16s(a10 as i64) < i16s(r180a as i64) {
                a10_below_branch = true;
            } else if i16s(a10 as i64) >= 0x199a {
                a10_above_branch = true;
            } else {
                a10_below_branch = true;
            }
            if a10_below_branch {
                if a11 == 0 {
                    a10_above_branch = true;
                } else {
                    bp = a8;
                    let d_half =
                        abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(i16s(a8 as i64) << 16));
                    let d_full =
                        abssat((i16s(a1 as i64) << 16).wrapping_sub(i16s(a8 as i64) << 16));
                    if d_half < d_full {
                        return (halve(a1), true);
                    }
                    go_final = true;
                }
            }
            if a10_above_branch && !go_final {
                if i16s(a5 as i64) >= 0xccd {
                    let ss5 = shift_scale(shl(i16s(mant5 as i64), 16), shift_arg2); // BUG1 FIX (overwrite guard)
                    let inner = i16s(
                        (sar(
                            0x599a0000i32.wrapping_sub(i16s(a5 as i64).wrapping_mul(0x2666)),
                            16,
                        )) as i64,
                    );
                    let thresh = inner.wrapping_mul(i16s(mant6 as i64)).wrapping_mul(2);
                    if ss5 <= thresh {
                        bp = a8;
                        go_final = true;
                    } else {
                        let a4_16 = i16s(a4 as i64) << 16;
                        let d_half = abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(a4_16));
                        let d_full = abssat((i16s(a1 as i64) << 16).wrapping_sub(a4_16));
                        if d_half < d_full {
                            return (halve(a1), true);
                        }
                        bp = a8;
                        go_final = true;
                    }
                } else {
                    let val3 = i16s(mant5 as i64).wrapping_mul(0xb6dcu32 as i32);
                    let ss3 = shift_scale(val3, shift_arg2); // BUG1 FIX
                    if ss3 > m6s15 {
                        return (halve(a1), true);
                    }
                    bp = a8;
                    go_final = true;
                }
            }
        } else {
            bp = a8;
            go_final = true;
        }
        if go_final {
            if i16s(a6 as i64) < 3 {
                return (identity(a1), false);
            }
            let ss4 = shift_scale(shl(i16s(mant5 as i64), 16), shift_arg2); // BUG1 FIX
            if ss4 < m6s15 {
                return (identity(a1), false);
            }
            let bp_16 = i16s(bp as i64) << 16;
            let d_half = abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(bp_16));
            let d_full = abssat((i16s(a1 as i64) << 16).wrapping_sub(bp_16));
            if d_half >= d_full {
                return (identity(a1), false);
            }
            return (halve(a1), true);
        }
        (identity(a1), false)
    }
}
// ======================= ctx+0x614 pitch tracker =======================
// The octave_halve_decide side-channel args a4..a10 are NOT DLL-internal magic: they are a 7-word
// struct at refine_ctx+0x614, updated once per frame by the tracker, which is called
// with args (struct, r622, c62c, c626, c630) -- every one of which we already derive.
// Field map (struct offsets, with the values that get stored back):
//   +0x614 w0  \_ block-float (mantissa, exponent) of the tracked pitch delta
//   +0x616 w2  /
//   +0x618 w4  = a7  (fast pitch track, 0.3/0.7 smoother)
//   +0x61a w6  = long-term pitch average (0.98/0.02 smoother)
//   +0x61c w8  = a4  (extrapolated pitch predictor)
//   +0x61e wa  = a5  (confidence: 0x7fff on a good frame, else decays x0.9)
//   +0x620 wc  = a6  (consecutive-stable-pitch counter)
#[derive(Clone, Copy, Debug, PartialEq)]
struct PitchTrackerState {
    w0: i32,
    w2: i32,
    w4: i32,
    w6: i32,
    w8: i32,
    wa: i32,
    wc: i32,
}
impl Default for PitchTrackerState {
    /// Frame-0 entry state, read straight off the ptm2_b980in capture row 0
    /// (base k0 = ctx+0x62c, so k-12..k-6 = ctx+0x614..0x620).
    fn default() -> Self {
        PitchTrackerState {
            w0: 0,
            w2: 0,
            w4: 10923,
            w6: 5243,
            w8: 5243,
            wa: 0,
            wc: 0,
        }
    }
}
/// The once-per-frame tracker update. All args audio-derivable.
fn pitch_tracker_update(t: &mut PitchTrackerState, r622: i32, c62c: i32, c626: i32, c630: i32) {
    let r622_s = i16s(r622 as i64);
    let c62c_s = i16s(c62c as i64);
    let mut lt_avg = t.w6;
    let mut pred_src = t.w8;
    let mut run_ctr = t.wc;
    let mut delta_exp = t.w2;
    let mut fast_trk = t.w4;
    // c62c < 0xccd -> smooth the fast pitch track 0.3*r622 + 0.7*a7.
    if c62c_s < 0xccd {
        let term1 = r622_s.wrapping_mul(0x4ccc);
        let term2 = i16s(t.w4 as i64).wrapping_mul(0xb334);
        fast_trk = (sar(term1.wrapping_add(term2), 16)) & 0xffff;
    }
    // |r622 - c626| vs 0.15*r622 -> stable-run counter.
    let mut delta = shl(r622_s, 16).wrapping_sub(shl(i16s(c626 as i64), 16));
    let mut dtmp = if delta >= 0 {
        delta
    } else if delta == i32::MIN {
        i32::MAX
    } else {
        -delta
    };
    let ecx_thr = sar(r622_s.wrapping_mul(0x999a), 2);
    if dtmp < ecx_thr {
        run_ctr += 1;
    } else {
        run_ctr = 0;
    }
    // good frame = c62c<0x199a AND c630<0x199a AND pitch stable.
    let good = c62c_s < 0x199a && i16s(c630 as i64) < 0x199a && dtmp < ecx_thr;
    #[allow(unused_mut)] // port artifact: the reference re-writes this slot; this path never does
    let mut conf;
    if good {
        pred_src = r622_s & 0xffff; // a4 source := r622
        dtmp = delta;
        let sh = if delta == 0 { 0 } else { norm_shift(dtmp) };
        delta = shl(delta, sh);
        delta_exp = (-sh) & 0xffff;
        let term_avg = i16s(lt_avg as i64).wrapping_mul(0xfae2);
        let mut term_new = i16s(r622_s as i64).wrapping_mul(0xa3d8);
        delta = sar(delta, 16);
        conf = 0x7fff; // a5 := 32767
        term_new = sar(term_new, 5).wrapping_add(term_avg.wrapping_add(0x8000));
        lt_avg = (sar(term_new, 16)) & 0xffff; // 0.98*lt_avg + 0.02*r622
    } else {
        // decay the delta block-float by 0.8 and a5 by 0.9.
        delta = i16s(t.w0 as i64).wrapping_mul(0xcccc);
        let sh = if delta == 0 { 0 } else { norm_shift(delta) };
        delta_exp = delta_exp.wrapping_sub(sh);
        delta = sar(shl(delta, sh & 0xff), 16);
        conf = (sar(i16s(t.wa as i64).wrapping_mul(0xe666), 16)) & 0xffff;
    }
    // a4 = 0.9*pred_src + 0.9*bf(delta) + 0.1*lt_avg, clamped to [0x666,0x3e00].
    let delta_lo = delta & 0xffff;
    let mut d2 = i16s(delta_lo as i64).wrapping_mul(0xe666);
    d2 = bf_shift(d2, delta_exp);
    let ea = i16s(pred_src as i64).wrapping_mul(0xe666);
    let b2 = sat_add(ea, d2);
    let ed = sar(i16s(lt_avg as i64).wrapping_mul(0xcccc), 3);
    let mut res = sar(sat_add(ed, b2), 16) & 0xffff;
    if i16s(res as i64) > 0x3e00 {
        res = 0x3e00;
    } else if i16s(res as i64) < 0x666 {
        res = 0x666;
    }
    t.w8 = res;
    t.wa = conf;
    t.w2 = delta_exp & 0xffff;
    t.w4 = fast_trk;
    t.w0 = delta_lo;
    t.w6 = lt_avg;
    t.wc = run_ctr & 0xffff;
}
/// The divisor coarse_pitch is called with (4, 5 or 6).
fn ring_median_divide(ctr: i32, a7: i32, c626: i32) -> i32 {
    if i16s(ctr as i64) < 3 {
        return 4;
    }
    let a7_s = i16s(a7 as i64);
    let c6 = i16s(c626 as i64);
    if a7_s >= 0x2400 && c6 >= 0x2400 {
        return 6;
    }
    if a7_s < 0x2000 {
        return 4;
    }
    if c6 >= 0x2000 {
        5
    } else {
        4
    }
}
// ======================= octave clamp + bd30 c624 tail =======================
fn octave_clamp(r622_pre: i32) -> i32 {
    let v = r622_pre as i16;
    if v > 0x3759 {
        (v >> 1) as i32
    } else {
        v as i32
    }
}
// continuity_smoother: bit-exact c624/c62e continuity smoother (100/100 both files).
// Q0=r622, Q2=c626, P0=c62c, P2=c630(=prev c62c). Returns (c624, c62e).
fn continuity_smoother(q0: i32, q2: i32, p0: i32, p2: i32) -> (i32, i32) {
    const DD: i32 = 0x3333;
    const BB: i32 = 0x1999;
    let s16 = |x: i32| {
        let x = x & 0xffff;
        if x >= 0x8000 {
            x - 0x10000
        } else {
            x
        }
    };
    let sar = |x: i32, n: u32| x >> n;
    let shl = |x: i32, n: u32| ((x as u32) << n) as i32;
    let sar16_field = |x: i32| s16(x) >> 1;
    let absat = |mut e: i32| -> i32 {
        if e >= 0 {
            return e;
        }
        if (e as u32) == 0x8000_0000 {
            return 0x7fff_ffff;
        }
        e = -(e as i64) as i32;
        e
    };
    let cur = p0;
    if cur > DD && p2 < DD {
        return (s16(q2), s16(p2));
    }
    if cur > BB && p2 < sar16_field(cur) {
        return (s16(q2), s16(p2));
    }
    let prev = p2;
    if prev > DD && cur < DD {
        return (s16(q0), s16(p0));
    }
    if prev > BB && cur < sar16_field(prev) {
        return (s16(q0), s16(p0));
    }
    // blend
    let p_blend = (sar(shl(s16(p2), 16), 1) as i64 + sar(shl(s16(p0), 16), 1) as i64) as i32;
    let p1 = s16(sar(p_blend, 16));
    let q0s = s16(q0);
    let q2s = s16(q2);
    let q0h = sar(shl(q0s, 16), 1);
    let q2h = sar(shl(q2s, 16), 1);
    let mut q1 = s16(sar((q2h as i64 + q0h as i64) as i32, 16));
    if p0 >= 0xccc {
        return (q1, p1);
    }
    let mut e2 = absat((shl(q0s, 16) as i64 - q2h as i64) as i32);
    let q0v = s16(q0);
    e2 = (e2 as i64 - (q0v as i64) * 0x3332) as i32;
    if e2 < 0 {
        q1 = s16(sar((sar(q2h, 1) as i64 + q0h as i64) as i32, 16));
        return (q1, p1);
    }
    let q2_16 = shl(q2s, 16);
    e2 = absat((q2_16 as i64 - q0h as i64) as i32);
    e2 = (e2 as i64 - (q0v as i64) * 0x1998) as i32;
    if e2 < 0 {
        q1 = s16(sar((q2_16 as i64 + q0h as i64) as i32, 16));
    }
    (q1, p1)
}
// ======================= prequant_pitch_candidate =======================
fn pq_s16(x: i32) -> i32 {
    let x = x & 0xffff;
    if x >= 0x8000 {
        x - 0x10000
    } else {
        x
    }
}
fn finalize(val: i32) -> i32 {
    2 * val
}
fn is_subharmonic_peak(q_p1: i32, q_0: i32, q_m1: i32, p_0: i32, p_m1: i32) -> bool {
    let qm1 = pq_s16(q_m1);
    if qm1 >= 0x199a {
        return false;
    }
    if (qm1 << 16) >= ((pq_s16(q_0) << 16) >> 1) {
        return false;
    }
    if (qm1 << 16) >= ((pq_s16(q_p1) << 16) >> 1) {
        return false;
    }
    let half = (pq_s16(p_m1) << 16) >> 1;
    let mut d = (pq_s16(p_0) << 16).wrapping_sub(half);
    if d < 0 {
        d = if d == i32::MIN { i32::MAX } else { -d };
    }
    let eighth = half >> 2;
    if d >= eighth {
        return false;
    }
    true
}
fn prequant_pitch_candidate(p: &[i32; 16], q: &[i32; 16], si: i32, arg3: i32, arg5: i32) -> i32 {
    if si >= 1 {
        let i = si as usize;
        if is_subharmonic_peak(q[i + 1], q[i], q[i - 1], p[i], p[i - 1]) {
            let v = (2 * pq_s16(p[i])) as i16 as i32;
            return finalize(pq_s16(v));
        }
    }
    let idx = si as usize;
    if pq_s16(q[idx]) <= 0x3333 {
        return finalize(pq_s16(p[idx]));
    }
    let sel = arg5;
    let pick: i32;
    if pq_s16(q[sel as usize]) < 0x3333 {
        pick = sel;
    } else {
        let mut d = 0i32;
        for e in 1..5 {
            if pq_s16(q[e as usize]) < pq_s16(q[d as usize]) {
                d = e;
            }
        }
        pick = d;
    }
    if pq_s16(q[pick as usize]) < 0x3333 {
        return finalize(pq_s16(p[pick as usize]));
    }
    let target = pq_s16(arg3);
    let mut bp = 0i32;
    for e in 1..5 {
        if (pq_s16(p[e as usize]) - target).abs() < (pq_s16(p[bp as usize]) - target).abs() {
            bp = e;
        }
    }
    finalize(pq_s16(p[bp as usize]))
}
// ======================= audio front-end: the windowed-taper -> FFT/BFP ->
// block-exponent chain over the shared `enc` primitives, with e460 additionally
// returning its block-exponent candidates so `expo` derivability can be
// measured. =======================
mod fe {
    use crate::enc::array_a_stage2::inverse_fft_butterfly_stage;
    use crate::enc::block_exponent::block_exponent;
    use crate::enc::loudness_fixed::{gamma_poly_scale_pair, gamma_poly_pass_block};
    use crate::enc::loudness_transform::fft_bfp_transform;
    use crate::enc::real_fft32::real_fft32;
    use crate::enc::windowed_taper::windowed_taper;

    const COEF: [i64; 7] = [-4456, 3034, 19608, 29164, 19608, 3034, -4456];
    const NBLK: usize = 25;
    const CFG: [(i64, i64, i64); 2] = [(-48, -28, 108), (32, 52, 108)];
    const TAPER_LUT: [i16; 29] = [
        2644, 2820, 3171, 3692, 4377, 5219, 6206, 7329, 8573, 9924, 11366, 12882, 14454, 16065,
        17695, 19324, 20935, 22508, 24024, 25466, 26817, 28061, 29183, 30171, 31012, 31697, 32219,
        32569, 32746,
    ];
    use crate::fixops::acc64::{s16m, sat16};
    fn shift_scale(val: i32, sh: i32) -> i32 {
        let shw = sh as i16 as i32;
        if shw < 0 {
            if shw > -31 {
                return val >> ((-shw) & 31);
            }
            return val >> 31;
        }
        let nbits = 0x1f - shw;
        let mask = (-1i32).wrapping_shl(nbits as u32);
        let mut c = mask & val;
        if val < 0 {
            c = c.wrapping_sub(mask);
        }
        if c != 0 {
            return if val < 0 {
                i32::MIN.wrapping_add(1)
            } else {
                0x7fff_ffff
            };
        }
        ((val as u32).wrapping_shl(shw as u32)) as i32
    }
    fn renorm1(src: i16, oldbe: i16, newbe: i16) -> i16 {
        (shift_scale((src as i32) << 16, oldbe as i32 - newbe as i32) >> 16) as i16
    }
    fn make_block(pref: &[i16], base: i64, be: i16) -> [i16; 16] {
        let (a2, a3) = gamma_poly_scale_pair(be);
        let sh = -(be as i32);
        let mut w = [0i16; 32];
        for k in 0..32 {
            let idx = base + k as i64;
            let pv = if idx >= 0 && (idx as usize) < pref.len() {
                pref[idx as usize] as i64
            } else {
                0
            };
            w[k] = if sh >= 0 {
                sat16(pv << sh)
            } else {
                sat16(pv >> (-sh))
            };
        }
        let tap = windowed_taper(&w);
        let mut fb = tap;
        real_fft32(&mut fb);
        gamma_poly_pass_block(&fb, a2, a3)
    }
    fn be_for(pref: &[i16], f: i64, be_off: i64, be_len: i64) -> i16 {
        let start = 160 * f + be_off;
        let mut w = Vec::with_capacity(be_len as usize);
        for i in 0..be_len {
            let idx = start + i;
            w.push(if idx >= 0 && (idx as usize) < pref.len() {
                pref[idx as usize]
            } else {
                0
            });
        }
        block_exponent(&w)
    }
    fn pass_block_exponents(blocks: &[[i16; 16]]) -> [[i64; 10]; 16] {
        let mut out = [[0i64; 10]; 16];
        for c in 0..10usize {
            for r in 0..16usize {
                let mut acc = 0i64;
                for k in 0..7 {
                    acc += COEF[k] * blocks[2 * c + k][r] as i64;
                }
                out[r][c] = s16m((acc + 0x8000) >> 16);
            }
        }
        out
    }
    #[derive(Clone)]
    #[derive(Default)]
    pub(crate) struct PassAccumulatorState {
        persist: [[i16; 16]; 5],
        be_state: i16,
    }
    
    impl PassAccumulatorState {
        pub fn run_pass(&mut self, pref: &[i16], f: i64, pass: usize) -> [[i64; 10]; 16] {
            let (win_base_off, be_off, be_len) = CFG[pass];
            let base = 160 * f + win_base_off;
            let new_be = be_for(pref, f, be_off, be_len);
            let mut fresh = Vec::with_capacity(NBLK);
            for m in 0..NBLK {
                fresh.push(make_block(pref, base + 4 * m as i64, new_be));
            }
            let mut blocks = vec![[0i16; 16]; NBLK];
            for j in 0..5 {
                if new_be == self.be_state {
                    blocks[j] = self.persist[j];
                } else {
                    for k in 0..16 {
                        blocks[j][k] = renorm1(self.persist[j][k], self.be_state, new_be);
                    }
                }
            }
            blocks[5..NBLK].copy_from_slice(&fresh[5..NBLK]);
            self.persist.copy_from_slice(&fresh[20..(5 + 20)]);
            self.be_state = new_be;
            pass_block_exponents(&blocks)
        }
    }
    fn shift_scale_worker(dst: &mut [i16], src: &[i16], count: usize, shift: i16) {
        if shift == 0 {
            dst[..count].copy_from_slice(&src[..count]);
            return;
        }
        if shift > 0 {
            for i in 0..count {
                let v = src[i] as i32;
                let e = (v << 16).wrapping_shl((shift as u32) & 31);
                dst[i] = (e >> 16) as i16;
            }
        } else {
            let mut s = -(shift as i32);
            if shift <= -31 {
                s = 31;
            }
            for i in 0..count {
                let v = src[i] as i32;
                let e = (v << 16) >> (s & 31);
                dst[i] = (e >> 16) as i16;
            }
        }
    }
    fn q15_taper(sample: i16, coeff: i16) -> i16 {
        let v: i64 = (sample as i64) * (coeff as i64) * 2 + 0x8000;
        let mut pv = (v >> 16) as i32;
        pv = pv.clamp(-32768, 32767);
        pv as i16
    }
    fn half_cosine_taper(buf: &mut [i16], n: usize) {
        let half = n / 2;
        for i in 0..n {
            let coeff = if i < half {
                TAPER_LUT[i]
            } else {
                TAPER_LUT[n - 1 - i]
            };
            buf[i] = q15_taper(buf[i], coeff);
        }
    }
    fn magsq(out: &mut [i32], src: &[i16], count: usize) {
        for i in 0..count {
            let re = src[2 * i] as i32;
            let im = src[2 * i + 1] as i32;
            out[i] = 2i32.wrapping_mul(re.wrapping_mul(re).wrapping_add(im.wrapping_mul(im)));
        }
    }
    /// Returns (scorevec, expo, blockexp_of_i16_spectrum, blockexp_of_scorevec_i32).
    pub(crate) fn score_vector_transform(
        raw: &[i16; 58],
        cand: i16,
        bands: &[i16; 7],
        shifts: &[i16; 7],
    ) -> ([i32; 32], i32, i32, i32) {
        let mut work = [0i16; 72];
        work[..58].copy_from_slice(raw);
        for b in 0..7 {
            let start = bands[b] as i64;
            let end = if b < 6 { bands[b + 1] as i64 } else { 58 };
            let cnt = (end - start) as usize;
            if cnt == 0 || start < 0 {
                continue;
            }
            let su = start as usize;
            let shift = shifts[b].wrapping_sub(cand);
            let src: Vec<i16> = work[su..su + cnt].to_vec();
            let mut tmp = vec![0i16; cnt];
            shift_scale_worker(&mut tmp, &src, cnt, shift);
            work[su..su + cnt].copy_from_slice(&tmp);
        }
        half_cosine_taper(&mut work, 58);
        for k in 58..72 {
            work[k] = 0;
        }
        let r = fft_bfp_transform(&mut work, 0, cand, 5);
        // The inverse-FFT butterfly stage's returned exponent is doubled before e460
        // tail-returns it (fft_bfp_transform's own return is used only as its input).
        // So expo = 2 * that return.
        let ifft_exp = inverse_fft_butterfly_stage(&mut work, r, 6, 0, 6);
        let mut out = [0i32; 32];
        magsq(&mut out, &work, 32);
        let be_spec = block_exponent(&work[..64]) as i32;
        let mut maxabs: u64 = 0;
        for &v in out.iter() {
            let a = (v as i64).unsigned_abs();
            if a > maxabs {
                maxabs = a;
            }
        }
        let be_sv = if maxabs == 0 {
            0
        } else {
            (63 - maxabs.leading_zeros() as i32) - 30
        };
        (out, 2 * ifft_exp as i32, be_spec, be_sv)
    }
    /// The ab80 per-pass block exponent (`new_be`) for frame `f`, pass 0|1.
    pub(crate) fn pass_be(pref: &[i16], f: i64, pass: usize) -> i32 {
        let (_, be_off, be_len) = CFG[pass];
        be_for(pref, f, be_off, be_len) as i32
    }
}

// ======================= the audio-only b0 chain =======================

const BANDS_CONST: [i16; 7] = [0, 0, 8, 18, 28, 38, 48];
/// Which adf0 pass-generation feeds each e460 band. Band b of the e460 rawwork is
/// exactly one adf0 pass-group (10 harmonic columns) from generation `f + GEN[b]`,
/// pass `PASS[b]`; `shifts[b]` is that pass's ab80 block exponent.
const GEN: [i64; 7] = [0, -2, -2, -1, -1, 0, 0];
const PASS: [usize; 7] = [0, 0, 1, 0, 1, 0, 1];
const K_SLIDE: usize = 20;

/// Streaming audio-only b0 tracker. One instance per encoder stream; call
/// [`push_pcm_frame`](Self::push_pcm_frame) once per 160-sample frame, in order.
pub struct B0Audio {
    ab80: fe::PassAccumulatorState,
    noise: crate::enc::noise_track::NoiseTracker,
    prev_pass1: [[i64; 10]; 16],
    e530_hist: [[i64; 50]; 16],
    r622h: Vec<i32>,
    c624h: Vec<i32>,
    c62eh: Vec<i32>,
    c62ch: Vec<i32>,
    prev_raw_r622: i32,
    trk: PitchTrackerState,
    frame: i64,
}

impl Default for B0Audio {
    fn default() -> Self {
        Self::new()
    }
}

impl B0Audio {
    pub fn new() -> Self {
        B0Audio {
            ab80: fe::PassAccumulatorState::default(),
            noise: crate::enc::noise_track::NoiseTracker::default(),
            prev_pass1: [[0i64; 10]; 16],
            // The e530 frame-0 seed is genuinely all-zero (probe: `seed_zero == true`
            // on both files), so no capture is needed to initialize it.
            e530_hist: [[0i64; 50]; 16],
            r622h: Vec::new(),
            c624h: Vec::new(),
            c62eh: Vec::new(),
            c62ch: Vec::new(),
            prev_raw_r622: 0,
            trk: PitchTrackerState::default(),
            frame: 0,
        }
    }

    /// Advance one frame and return this frame's b0 channel byte, with `a11` supplied.
    ///
    /// `a11` is the PREVIOUS frame's `ctx+0x91a` band-voicing mask. The bd30 tail and
    /// `octave_halve_decide` only ever test it against zero, so only `a11 == 0` vs `a11 != 0` is
    /// load-bearing -- callers may pass any nonzero sentinel for "some band voiced".
    ///
    /// `push_pcm_frame` is exactly `push_pcm_frame_with_prev_mask(pref, 0)`; the substitute `0`
    /// is NOT free (it pins `trk.wc` to 0 every frame, permanently disarming the
    /// stable-run counter). See `b1_audio`-driven callers for an audio-only a11.
    pub fn push_pcm_frame_with_prev_mask(&mut self, pref: &[i16], a11: i32) -> u8 {
        self.push_frame_inner(pref, a11)
    }

    /// Advance one frame and return this frame's b0 channel byte.
    ///
    /// `pref` is the WHOLE prefiltered PCM stream (`enc::audio_prefilter::prefilter`);
    /// the frame index is tracked internally and indexes into it. No captures are read.
    pub fn push_pcm_frame(&mut self, pref: &[i16]) -> u8 {
        self.push_frame_inner(pref, 0)
    }

    /// The frame just pushed's `ctx+0x62c` harmonic-fit error — the `Q[0]` the
    /// `FUN_1030eaa0` tail compares against `0x2001`. `Q[2]` is the previous
    /// frame's value; `32767` before any frame is pushed, per the ring's init.
    pub fn last_c62c(&self) -> i32 {
        *self.c62ch.last().unwrap_or(&32767)
    }

    /// `Q[0]` for frame `f`, for a caller that has advanced this tracker past
    /// `f` and needs the value again. Frames not yet pushed read as the ring's
    /// `32767` seed.
    pub fn c62c_at(&self, f: usize) -> i32 {
        self.c62ch.get(f).copied().unwrap_or(32767)
    }

    /// `Q[2]`: the c62c of the frame before the one just pushed.
    pub fn prev_c62c(&self) -> i32 {
        let n = self.c62ch.len();
        if n >= 2 {
            self.c62ch[n - 2]
        } else {
            32767
        }
    }

    fn push_frame_inner(&mut self, pref: &[i16], a11_in: i32) -> u8 {
        let f = self.frame;
        self.frame += 1;

        let pass0 = self.ab80.run_pass(pref, f, 0);
        let pass1 = self.ab80.run_pass(pref, f, 1);
        let noise_win = crate::enc::noise_track::advance(&mut self.noise, pref, f);
        if f > 0 {
            for b in 0..16usize {
                let mut app = [0i64; 20];
                app[..10].copy_from_slice(&self.prev_pass1[b]);
                app[10..(10 + 10)].copy_from_slice(&pass0[b]);
                let mut nb = [0i64; 50];
                nb[..(49 - K_SLIDE)].copy_from_slice(&self.e530_hist[b][K_SLIDE..((49 - K_SLIDE) + K_SLIDE)]);
                for i in 0..K_SLIDE {
                    nb[49 - K_SLIDE + i] = app[i];
                }
                self.e530_hist[b] = nb;
            }
        }
        self.prev_pass1 = pass1;

        // p4base = this frame's ab80 pass1 rows[0..10] (probe: 3184/3184 vs capture).
        let mut p4: [[i16; 10]; 16] = [[0; 10]; 16];
        for b in 0..16usize {
            p4[b] = std::array::from_fn(|i| pass1[b][i] as i16);
        }

        // per-seq scorevecs + their exponents, all derived.
        let mut svs = [[0i32; 32]; 16];
        let mut expo = [0i32; 16];
        for b in 0..16usize {
            let mut raw = [0i16; 58];
            for i in 0..48 {
                raw[i] = self.e530_hist[b][1 + i] as i16;
            }
            raw[48..(10 + 48)].copy_from_slice(&p4[b]);
            // cand/bands/shifts from the ab80 pass-generation block exponents.
            // Bands whose generation predates frame 0 collapse their start edge to 0,
            // which reproduces the f=0/f=1 warmup band vectors exactly; their scale
            // factor is irrelevant there because the rawwork is zero.
            let mut shifts = [-32768i16; 7];
            let mut bands = [0i16; 7];
            for b2 in 1..7 {
                let ff = f + GEN[b2];
                if ff >= 0 {
                    shifts[b2] = fe::pass_be(pref, ff, PASS[b2]) as i16;
                    bands[b2] = BANDS_CONST[b2];
                }
            }
            let cand = shifts[1..7].iter().copied().max().unwrap();
            let (sv, r, _, _) = fe::score_vector_transform(&raw, cand, &bands, &shifts);
            svs[b] = sv;
            expo[b] = r; // = 2 * inverse-FFT exponent
        }

        let score32 = e100::run(&svs, &expo, noise_win.as_ref());
        let mut score = [0i32; 40];
        for i in 0..32 {
            score[i] = score32[i] as i32;
        }

        let c626 = self.prev_raw_r622;
        // c630 = the previous frame's c62c (bd30's ring shift);
        // seeds to 32767 at frame 0 per the ctx+0x62c..0x634 init.
        let c630 = *self.c62ch.last().unwrap_or(&32767);
        // a11 = the previous frame's ctx+0x91a band-voicing mask. octave_halve_decide only ever tests
        // it ==0, so only zero-vs-nonzero matters. Supplied by the caller;
        // `push_pcm_frame` passes 0. `C62cThresh` reads c630 == the PREVIOUS frame's
        // c62c, which is exactly the frame whose mask a11 denotes.
        let a11 = a11_in;
        // arg6==0 resets the stable-run counter before octave_halve_decide is called.
        if a11 == 0 {
            self.trk.wc = 0;
        }

        let low = ring_median_divide(self.trk.wc, self.trk.w4, c626);
        let high = 32i32;
        let a1 = b7b0m::a1_clamp(b7b0m::coarse_pitch(&score, low, high));
        // a8 = c626 once the run counter is armed, else a7.
        let a8 = if i16s(self.trk.wc as i64) >= 1 {
            c626
        } else {
            self.trk.w4
        };
        let a: [i32; 8] = [
            self.trk.w8,
            self.trk.wa,
            self.trk.wc,
            self.trk.w4,
            a8,
            c626,
            c630,
            a11,
        ];
        let (raw_r622, halved) =
            b980m::octave_halve_decide(a1, &score, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        let r622_post = octave_clamp(raw_r622);

        // c62c = octave_halve_decide's own prologue periodicity-ratio complement at the octave octave_halve_decide
        // selected: div=5 comb when it halved, div=6 otherwise (199/199 both files).
        let c62c = if halved {
            b980m::conf5(a1, &score)
        } else {
            b980m::conf6(a1, &score)
        };

        // the once-per-frame tracker update, on the values just produced.
        pitch_tracker_update(&mut self.trk, raw_r622, c62c, c626, c630);
        // arg3 = the tracker's fast pitch track ctx+0x618 read AFTER the pitch_tracker_update update
        // (prequant_pitch_candidate runs later in the frame than the tracker update) -- 199/199 both files.
        let arg3 = self.trk.w4;
        let (si, arg5) = (1i32, 1i32);

        // continuity_smoother genuinely runs on frame 0 (c624[0] = 3072 = (2048+4096)/2).
        let (c624, c62e) = {
            let (a, b) = continuity_smoother(r622_post, c626, c62c, c630);
            (a & 0xffff, b & 0xffff)
        };

        let nfr = self.r622h.len();
        let rp =
            |v: &Vec<i32>, back: usize, seed: i32| if nfr >= back { v[nfr - back] } else { seed };
        let mut p = [32767i32; 16];
        let mut q = [32767i32; 16];
        p[0] = r622_post;
        p[1] = c624;
        p[2] = rp(&self.r622h, 1, 4096);
        p[3] = rp(&self.c624h, 1, 4096);
        p[4] = rp(&self.r622h, 2, 4096);
        q[0] = c62c;
        q[1] = c62e;
        q[2] = rp(&self.c62ch, 1, 32767);
        q[3] = rp(&self.c62eh, 1, 32767);
        q[4] = rp(&self.c62ch, 2, 32767);
        let pq = if f == 0 {
            8192
        } else {
            prequant_pitch_candidate(&p, &q, si, arg3, arg5)
        };

        self.r622h.push(r622_post);
        self.c624h.push(c624);
        self.c62eh.push(c62e);
        self.c62ch.push(c62c);
        self.prev_raw_r622 = raw_r622;

        crate::enc::pitch::reject_recurrence::reject_b0(pq as u16, 7)
    }
}
