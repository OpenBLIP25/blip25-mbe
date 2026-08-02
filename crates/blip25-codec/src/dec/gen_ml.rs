// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! FROM-BITS `M_l` generator (site-B): the spectral-amplitude decode chain.
//!
//! Packages the band-VQ -> inverse-transform -> predictor chain that turns the
//! 49 info bits into the per-harmonic log-spectral amplitude `M_l`, carrying
//! `prev_ml`/`prev_l`/`gamma` across subframes. Bit-exact for site B.
//!
//! All tables are baked ([`crate::dec::gen_ml_tables`] + the shared cosine table
//! [`crate::shared::loudness_transform`]); this module reads NO files at runtime.
//!
//! The codebook indices come from the info vector `u0..u3` (widths 12/12/11/14,
//! MSB-first) via the `PERM` bit-field map — i.e. the same layout the DLL's
//! deframed-info buffer holds. `u0..u3` come from `frame::decode_bytes`, so a
//! caller drives this from the 9-byte r33 frame alone.

use crate::dec::gen_ml_tables::{cbk, DELTA, LOGT, PART};
use crate::shared::loudness_transform::cos_twiddle_table;
use blip25_codebooks;

use crate::fixops::dec32::{shl, sx16};
use crate::fixops::i32r::i16w;

#[inline]
fn sar(x: i32, n: u32) -> i32 {
    x >> n
}

/// `PERM[k]` = the info-vector bit positions (in the 56-bit `u0||u1||u2||u3`
/// layout) forming codebook index `ci[k]` (b3..b8 residual-band VQ indices).
const PERM: [&[usize]; 6] = [
    &[12, 13, 14, 15, 16, 17, 18, 19, 43],
    &[20, 21, 22, 23, 40, 41, 42],
    &[24, 25, 26, 27, 44],
    &[28, 29, 30, 45],
    &[31, 32, 33, 46],
    &[34, 47, 48],
];

/// Pack the info vector `u0..u3` (widths 12/12/11/14) MSB-first into 56 bits.
fn info_bits(info: &[u16; 4]) -> [u8; 56] {
    const W: [usize; 4] = [12, 12, 11, 14];
    let mut bb = [0u8; 56];
    let mut p = 0usize;
    for i in 0..4 {
        let w = W[i];
        let v = info[i] as u32;
        for j in (0..w).rev() {
            bb[p] = ((v >> j) & 1) as u8;
            p += 1;
        }
    }
    bb
}

fn frac_divide(num: i32, den: i32) -> i32 {
    let sign = (den as i16 as i32) ^ num;
    let mut num_abs = num;
    if num_abs < 0 {
        num_abs = if num_abs == i32::MIN {
            i32::MAX
        } else {
            num_abs.wrapping_neg()
        };
    }
    let den_lo = den as u16 as i16;
    let mut den_abs: i32;
    if den_lo < 0 {
        if den_lo == -32768 {
            den_abs = 0x7fff;
        } else {
            den_abs = ((den).wrapping_neg() as u16) as i32;
        }
        den_abs = den_abs as i16 as i32;
    } else {
        den_abs = den_lo as i32;
    }
    let mut r = if num_abs < den_abs {
        0
    } else {
        (num_abs / den_abs) >> 1
    };
    if sign < 0 {
        r = r.wrapping_neg();
    }
    r
}
fn sat_shift(val: i32, sh: i32) -> i32 {
    let sh16 = sh as i16;
    if sh16 < 0 {
        let n = if sh16 > -31i16 {
            (-(sh16 as i32)) as u32 & 0xff
        } else {
            0x1f
        };
        return val >> (n & 31);
    }
    let pos_sh = sh16 as i32;
    let mask = (-1i32).wrapping_shl((31 - pos_sh) as u32 & 31);
    let mut hibits = mask & val;
    if val < 0 {
        hibits = hibits.wrapping_sub(mask);
    }
    if hibits != 0 {
        return if val < 0 { i32::MIN } else { i32::MAX };
    }
    val.wrapping_shl((pos_sh as u32) & 31)
}
fn predictor_interp(pred: &[i16], l: i32, arg3: i32) -> (Vec<i16>, i32) {
    let inv_l = frac_divide(0x80000, l) as i16 as i32;
    let e = ((arg3).wrapping_shl(0x19) >> 0x10) as i16 as i32;
    let mut prod = inv_l.wrapping_mul(e);
    prod = prod.wrapping_add(prod);
    let step = prod >> 0xc;
    let mut phase = step;
    let mut acc: i32 = 0;
    let mut out = vec![0i16; l as usize];
    for j in 0..(l as usize) {
        let idx = ((phase >> 0x10) as u16) as i32;
        let interp: i32;
        if idx == 0 && idx < 0x37 {
            interp = (pred[0] as i32).wrapping_shl(0x10);
        } else {
            let ei = if idx >= 0x37 { 0x37 } else { idx };
            let frac = (((phase >> 1) & 0x7fff) as i16) as i32;
            let p_lo = pred[(ei - 1) as usize] as i32;
            let mut prod_lo = frac.wrapping_mul(p_lo);
            prod_lo = prod_lo.wrapping_add(prod_lo);
            let neg_lo = if prod_lo == i32::MIN {
                i32::MAX
            } else {
                prod_lo.wrapping_neg()
            };
            let lo_scaled = p_lo.wrapping_shl(0xf);
            let mut hi_acc = pred[ei as usize] as i32;
            hi_acc = hi_acc.wrapping_mul(frac);
            hi_acc = hi_acc.wrapping_add(lo_scaled);
            interp = neg_lo.wrapping_add(hi_acc.wrapping_mul(2));
        }
        out[j] = (interp >> 0x10) as i16;
        phase = phase.wrapping_add(step);
        acc = acc.wrapping_add(interp >> 6);
    }
    (out, acc)
}
fn scale(buf: &mut [i16], l: i32, a6s: i32, arg3: i32) {
    let coef = arg3 as i16 as i32;
    let den = (l as i16 as i32).wrapping_shl(0x19) >> 0x10;
    let a = frac_divide(a6s, den) as i16 as i32;
    let mut nprod = a.wrapping_mul(coef);
    nprod = nprod.wrapping_add(nprod);
    nprod = if nprod == i32::MIN {
        i32::MAX
    } else {
        nprod.wrapping_neg()
    };
    let bias = nprod >> 1;
    for i in 0..(l as usize) {
        let mut v = (buf[i] as i32).wrapping_mul(coef);
        v = v.wrapping_add(v);
        v >>= 1;
        v = v.wrapping_add(bias);
        v = sat_shift(v, 1);
        v >>= 0x10;
        buf[i] = v as i16;
    }
}
fn writer(a0: &[i16], a1: &[i16], l: usize) -> Vec<i16> {
    let mut out = vec![0i16; l];
    for i in 0..l {
        let c = ((a1[i] as i32) << 0x10 >> 1).wrapping_add((a0[i] as i32) << 0x10 >> 1);
        let mut v = (sat_shift(c, 1) >> 0x10) as i16;
        v = v.clamp(-0x7800, 0x77ff);
        out[i] = v;
    }
    out
}
fn log2_poly(t: &[i32], x: i32, e: i32) -> i32 {
    let mut xn = shl(sx16(x), 16);
    let exp: i32;
    if xn == 0 {
        exp = 0;
    } else {
        let mut a = xn;
        if a < 0 {
            a = !a;
        }
        let b = 31 - (a as u32).leading_zeros() as i32;
        exp = ((0x1e - b) as u32 & 0xffff) as i32;
    }
    xn = shl(xn, (exp & 0xff) as u32);
    xn = xn.wrapping_sub(shl(t[0], 16));
    xn = sar(xn, 16);
    let mant = i16w((xn as u16) as i32);
    let mut acc = t[1].wrapping_mul(mant);
    let mut poly = acc.wrapping_mul(2).wrapping_add(0x8000) & (0xffff0000u32 as i32);
    poly = poly.wrapping_add(shl(t[2], 16));
    poly = sar(poly, 16);
    acc = i16w(poly);
    acc = acc.wrapping_mul(mant);
    poly = acc.wrapping_mul(2).wrapping_add(0x8000) & (0xffff0000u32 as i32);
    poly = poly.wrapping_add(shl(t[3], 16));
    poly = sar(poly, 16);
    acc = i16w(poly);
    acc = acc.wrapping_mul(mant);
    poly = acc.wrapping_mul(2).wrapping_add(0x8000);
    poly = sar(poly, 1) & (0xffff8000u32 as i32);
    poly = poly.wrapping_add(shl(t[4], 16));
    poly = sar(poly, 16);
    acc = i16w(poly);
    acc = acc.wrapping_mul(mant);
    acc = acc.wrapping_mul(8).wrapping_add(0x20000) & (0xfffc0000u32 as i32);
    acc = acc.wrapping_add(shl(t[5], 16));
    let exp_term = shl(sx16(e.wrapping_sub(exp)), 16);
    sar(acc, 0xf).wrapping_add(exp_term)
}
fn log2_scaled(t: &[i32], x: i32, n: i32, s: i32) -> i32 {
    let raw = sat_shift(log2_poly(t, x, n), 0xf - s);
    let rounded = raw.wrapping_add(0x8000);
    let ovf = (rounded ^ raw) & rounded;
    if ovf < 0 {
        let mut r = 0x7fffffffi32;
        if raw < 0 {
            r = r.wrapping_add(1);
        }
        return sar(r, 16);
    }
    sar(rounded, 16)
}
fn mean_of(t: &[i32], gamma: i32, l: i32, a5: &[usize; 4], bl: &[i32; 8]) -> i32 {
    let c = log2_scaled(t, l, 0xf, 5);
    let mut acc = i16w(sx16(gamma).wrapping_sub(sx16(c))).wrapping_mul(sx16(l));
    acc = acc.wrapping_add(acc);
    for i in 0..4 {
        let mut term = (a5[i] as i32 as i16 as i32).wrapping_mul(i16w(bl[2 * i]));
        term = term.wrapping_add(term);
        acc = acc.wrapping_sub(term);
    }
    (frac_divide(acc, l) as u16) as i32
}
struct Tabs {
    delta: Vec<i32>,
    cos512: Vec<i32>,
}
fn inverse_dct(t: &Tabs, coef: &[i32], n: usize, ncoef: usize) -> Vec<i32> {
    let ncoef = ncoef.min(n);
    let d = t.delta[n];
    let mut c = vec![i16w(coef[0])];
    for k in 1..ncoef {
        c.push(i16w(coef[k] << 1));
    }
    let mut out = Vec::with_capacity(n);
    let mut step = d >> 1;
    for _ in 0..n {
        let stepm = i16w(step & 0xffff);
        let mut phase: i32 = 0x40;
        let mut acc: i32 = 0;
        for k in 0..ncoef {
            let idx = ((phase >> 7) & 0x1ff) as usize;
            acc = acc.wrapping_add(t.cos512[idx].wrapping_mul(c[k]).wrapping_mul(2));
            phase = phase.wrapping_add(stepm);
        }
        out.push(sar(acc.wrapping_add(0x8000), 16));
        step = step.wrapping_add(d);
    }
    out
}
const SQRT2: i32 = 0xb504;
fn transform(t: &Tabs, a3: &[i32], a5: &[usize; 4], bands: &[[i32; 4]; 4]) -> (Vec<i32>, [i32; 8]) {
    let mut bl = inverse_dct(t, a3, 8, 8);
    for j in 0..4 {
        let d0 = bl[2 * j];
        let c0 = bl[2 * j + 1];
        bl[2 * j] = sar((c0 + d0).wrapping_mul(32768), 16);
        let dd = sar((d0 - c0).wrapping_mul(32768), 16);
        bl[2 * j + 1] = i16w(sar(dd.wrapping_mul(SQRT2), 16));
    }
    let mut a4 = Vec::new();
    for i in 0..4 {
        let mut coef = vec![bl[2 * i], bl[2 * i + 1]];
        coef.extend_from_slice(&bands[i]);
        a4.extend(inverse_dct(t, &coef, a5[i], 6));
    }
    let mut blo = [0i32; 8];
    blo.copy_from_slice(&bl[..8]);
    (a4, blo)
}

/// Cross-subframe `M_l` decoder state (site B), carried from cold init.
pub(crate) struct MlGen {
    logt: Vec<i32>,
    tabs: Tabs,
    gain_o: Vec<i32>,
    prev_ml: [i16; 56],
    prev_l: i32,
    gamma_prev: i32,
}
impl Default for MlGen {
    fn default() -> Self {
        Self::new()
    }
}
impl MlGen {
    pub fn new() -> Self {
        let cos1024 = cos_twiddle_table();
        let tabs = Tabs {
            delta: DELTA.to_vec(),
            cos512: cos1024[..512].iter().map(|&x| x as i32).collect(),
        };
        let gain_o: Vec<i32> = blip25_codebooks::gain_o()
            .iter()
            .map(|&v| v as i32)
            .collect();
        MlGen {
            logt: LOGT.to_vec(),
            tabs,
            gain_o,
            prev_ml: [0i16; 56],
            prev_l: 15,
            gamma_prev: 0,
        }
    }

    /// Site-B `M_l` for one subframe from bits; advances carried state.
    ///
    /// `b` = deprioritized b0..b8 (needs `b[2]` = gain index); `info` = the
    /// 49-bit info vector `u0..u3` (codebook-index source); `l` = the from-bits
    /// harmonic count (`l_step_from_b0_b1`). Returns `None` on `L<=0`.
    pub fn next_subframe(&mut self, b: &[u16; 9], info: &[u16; 4], l: i32) -> Option<Vec<i16>> {
        if l <= 0 {
            return None;
        }
        let bb = info_bits(info);
        let fld = |b: &[u8], idxs: &[usize]| -> i32 {
            idxs.iter().fold(0i32, |v, &s| (v << 1) | b[s] as i32)
        };
        let ci: Vec<i32> = (0..6).map(|i| fld(&bb, PERM[i])).collect();
        let e = (PART[(l - 9) as usize] & 0xffff) as u32;
        let a5: [usize; 4] = core::array::from_fn(|i| 2 + ((e >> (4 * i)) & 0xf) as usize);
        let mut a3 = vec![0i32];
        a3.extend_from_slice(&cbk(0, ci[0] as usize)[..3]);
        a3.extend_from_slice(&cbk(1, ci[1] as usize)[..4]);
        let mut bands = [[0i32; 4]; 4];
        for i in 0..4 {
            bands[i].copy_from_slice(&cbk(2 + i, ci[2 + i] as usize)[..4]);
        }
        let (a4, bl) = transform(&self.tabs, &a3, &a5, &bands);
        if a4.len() < l as usize {
            return None;
        }
        let gamma = self.gain_o[b[2] as usize] + (self.gamma_prev >> 1);
        self.gamma_prev = gamma;
        let m = sx16(mean_of(&self.logt, gamma, l, &a5, &bl));
        let xo: Vec<i32> = (0..l as usize)
            .map(|j| {
                sar(
                    sat_shift(
                        sar(shl(i16w(a4[j]), 16), 1).wrapping_add(sar(shl(m, 16), 1)),
                        1,
                    ),
                    16,
                )
            })
            .collect();
        let (mut a1, a6s) = predictor_interp(&self.prev_ml, l, self.prev_l);
        scale(&mut a1, l, a6s, 21299);
        let a0: Vec<i16> = xo.iter().map(|&v| v as i16).collect();
        let ml = writer(&a0, &a1, l as usize);
        let fill = *ml.last().unwrap_or(&0);
        for i in 0..56 {
            self.prev_ml[i] = if i < ml.len() { ml[i] } else { fill };
        }
        self.prev_l = l;
        Some(ml)
    }
}
