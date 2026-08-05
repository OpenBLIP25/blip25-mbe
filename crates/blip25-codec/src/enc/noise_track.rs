//! The `ctx+0x938` noise tracker (`FUN_1031d810` and its call closure).
//!
//! One `d810` call runs per analysis pass, twice per frame, immediately before the
//! DP score accumulation. It maintains a 0x180-byte block of per-band noise levels
//! and short histories that [`super::dp_score`] reads to gate and scale the sixteen
//! row vectors.
//!
//! Its three inputs all come from the same pass:
//!
//! * the sixteen-word band-energy vector (`FUN_1031b110`'s accumulator: the sum over
//!   the pass's twenty fresh blocks of each block's `FUN_10317610` magnitudes, shifted
//!   right by 7),
//! * `2 * pass_block_exponent + 7`,
//! * the pass's block count times two, a constant 20.
//!
//! Call order inside the block: history shift, `FUN_1031e8c0` (frame level),
//! `FUN_1031e1d0` (the per-band trackers), `FUN_1031ea20` (smoothed level),
//! `FUN_1031e080` (activity decision), `FUN_1031ebf0` (noise floor, only when the
//! frame reads as inactive).
#![allow(clippy::needless_range_loop)]

use super::dp_score::{a310, a4f0, a7a0, b100, norm_shift, s16, shift_sat, shr_dsp, ST_WORDS};
use crate::enc::loudness_fixed::gamma_poly_magsq16;
use crate::enc::real_fft32::real_fft32;
use crate::enc::windowed_taper::windowed_taper;
use crate::fixops::acc64::sat16;

/// The two analysis passes' window bases, in samples relative to the frame start,
/// and the offset/length of the run each pass takes its block exponent over.
const PASS_BASE: [i64; 2] = [-48, 32];
const PASS_BE: [(i64, i64); 2] = [(-28, 108), (52, 108)];

/// A pass's block exponent, the same quantity the row vectors are scaled by.
pub(crate) fn pass_be(pref: &[i16], f: i64, pass: usize) -> i16 {
    let (off, len) = PASS_BE[pass];
    let start = 160 * f + off;
    let mut w = Vec::with_capacity(len as usize);
    for i in 0..len {
        let idx = start + i;
        w.push(if idx >= 0 && (idx as usize) < pref.len() {
            pref[idx as usize]
        } else {
            0
        });
    }
    crate::enc::block_exponent::block_exponent(&w)
}

/// Advance the tracker over both of frame `f`'s analysis passes.
///
/// Frame `f` reads `pref` over `[160f - 28, 160f + 160)`, so the tracker is
/// frame-serial and causal with no look-ahead; out-of-range samples read as zero.
pub(crate) fn push_frame(nt: &mut NoiseTracker, pref: &[i16], f: i64) {
    for pass in 0..2 {
        let be = pass_be(pref, f, pass);
        let bv = pass_band_vector(pref, f, pass, be);
        nt.push_pass(&bv, 2 * be as i32 + 7, 20);
    }
}

/// Advance the tracker over both of frame `f`'s analysis passes and return the
/// window the DP score reads, or `None` when the weighted score is switched off.
pub(crate) fn advance(nt: &mut NoiseTracker, pref: &[i16], f: i64) -> Option<[i16; ST_WORDS]> {
    if !crate::enc::dp_score::enabled() {
        return None;
    }
    push_frame(nt, pref, f);
    Some(nt.window())
}
/// Blocks a pass contributes to the band vector, and their four-sample stride.
const FRESH: std::ops::Range<usize> = 5..25;

/// `FUN_1031b110`'s accumulator: the pass's band-energy vector, the sum over the
/// pass's fresh blocks of `FUN_10317610`'s magnitudes shifted right by 7.
pub(crate) fn pass_band_vector(pref: &[i16], f: i64, pass: usize, be: i16) -> [i32; 16] {
    let base = 160 * f + PASS_BASE[pass];
    let sh = -(be as i32);
    let mut acc = [0i32; 16];
    for m in FRESH {
        let mut w = [0i16; 32];
        for k in 0..32 {
            let idx = base + 4 * m as i64 + k as i64;
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
        let mut fb = windowed_taper(&w);
        real_fft32(&mut fb);
        let ms = gamma_poly_magsq16(&fb);
        for i in 0..16 {
            acc[i] = acc[i].wrapping_add(ms[i] >> 7);
        }
    }
    acc
}

/// The block as 16-bit words: byte offsets 0x00..0x190.
const NW: usize = 200;

/// `FUN_1031d810`'s persistent state.
#[derive(Clone)]
pub(crate) struct NoiseTracker {
    w: [i16; NW],
}

/// The block's content at encoder open: a descending per-band noise-floor template
/// with matching exponents, unit band gains, and the level/threshold seeds.
fn initial_block() -> [i16; NW] {
    let mut w = [0i16; NW];
    let mut set = |off: usize, v: i32| w[off / 2] = v as i16;
    set(0x40, -24576);
    set(0x42, 15);
    set(0x48, -7679);
    set(0x4e, 300);
    set(0x50, -11519);
    for k in 0..16 {
        let ramp = (-23551 - 1024 * k as i32).max(-29695);
        set(0x52 + 2 * k, ramp);
        set(0x72 + 2 * k, 5);
        set(0x92 + 2 * k, 32767);
        set(0xb2 + 2 * k, 5);
        set(0xd2 + 2 * k, 300);
        set(0x160 + 2 * k, -30);
    }
    set(0x13a, -7680);
    set(0x13c, 20000);
    set(0x13e, -16);
    w
}

impl Default for NoiseTracker {
    fn default() -> Self {
        NoiseTracker { w: initial_block() }
    }
}

impl NoiseTracker {
    #[inline]
    fn g(&self, off: usize) -> i32 {
        self.w[off / 2] as i32
    }
    #[inline]
    fn s(&mut self, off: usize, v: i32) {
        self.w[off / 2] = v as i16;
    }
    /// The block's first 0x40 bytes hold the previous pass's band vector as dwords.
    #[inline]
    fn dw(&self, i: usize) -> i32 {
        (self.w[2 * i] as u16 as i32) | ((self.w[2 * i + 1] as u16 as i32) << 16)
    }
    #[inline]
    fn set_dw(&mut self, i: usize, v: i32) {
        self.w[2 * i] = v as i16;
        self.w[2 * i + 1] = (v >> 16) as i16;
    }

    /// Block offset `0x44`: the voicing shift register, two bits per frame.
    ///
    /// `FUN_1030fe40` copies it into the quantiser descriptor at `+0x30`, where
    /// `FUN_1030c3f0` reads it as the `hdr30` word and `FUN_10310c50` tests bit
    /// `s + 2` to gate the a4 band it produces.
    pub(crate) fn voicing_register(&self) -> i32 {
        self.g(0x44) & 0xffff
    }

    /// The window [`super::dp_score`] reads: block offsets 0x40..0x190.
    pub(crate) fn window(&self) -> [i16; ST_WORDS] {
        let mut out = [0i16; ST_WORDS];
        out.copy_from_slice(&self.w[0x20..0x20 + ST_WORDS]);
        out
    }

    /// Advance one analysis pass. `bandvec` is the pass's sixteen-word band-energy
    /// accumulator, `exp` its `2 * block_exponent + 7`, `nblk2` its block count
    /// times two.
    pub(crate) fn push_pass(&mut self, bandvec: &[i32; 16], exp: i32, nblk2: i32) {
        // history shift: the eight-word lag-0 row and its two lags, plus the two
        // scalar lags at 0x132.
        for off in (0xf8..=0x100).rev().step_by(2) {
            self.w[off / 2] = self.w[off / 2 - 1];
        }
        for (dst, src, tail) in [(0x122usize, 0x112usize, 0x136usize), (0x112, 0x102, 0x134)] {
            for k in 0..8 {
                self.w[dst / 2 + k] = self.w[src / 2 + k];
            }
            self.w[tail / 2] = self.w[tail / 2 - 1];
        }

        let lvl = e8c0(bandvec, exp, nblk2);
        self.s(0x4a, lvl);

        let gain = if self.g(0x48) < -0x1400 {
            let t = s16(self.g(0x48) + 0x1400).wrapping_mul(0x550a) >> 16;
            let t = if s16(t) < -0x352 { -0x352 } else { s16(t) };
            let v = a890(t, 7, 0);
            s16(v).wrapping_mul(0x8b34) >> 16
        } else {
            0x459a
        };

        let mut side: i32 = 0;
        let ret = self.e1d0(bandvec, exp, nblk2, gain, &mut side);
        self.ea20(ret);
        let mut act = if s16(ret) < s16(side) { side } else { ret };
        let voiced = self.e080(&mut act);
        self.s(0xf6, act);
        if voiced == 0 {
            self.ebf0();
        }
        if self.g(0x138) < 8 {
            self.s(0x138, self.g(0x138) + 1);
        }
        self.s(0x44, s16(self.g(0x44).wrapping_mul(2) | voiced));
    }

    /// `FUN_1031e1d0`: the sixteen per-band noise trackers, plus the frame-level
    /// aggregates. Returns the frame's activity measure; `side` receives its
    /// companion.
    fn e1d0(&mut self, bandvec: &[i32; 16], p3: i32, p4: i32, p7: i32, side: &mut i32) -> i32 {
        let p5 = self.g(0x4e);
        let (mut l74, mut l7c) = (0i32, 0i32);
        let (mut l54, mut l80) = (0i32, 0i32);
        let (mut l5c, mut l64) = (0i32, 0i32);
        let (mut l28, mut l30) = (0i32, 0i32);
        let (mut l44, mut l40) = (0i32, 0i32);
        let (mut l50, mut l4c) = (32000i32, 10i32);
        let mut l68 = 0i32;
        *side = 0;
        let mut lag_ix = 0usize;

        let u = s16(p4).wrapping_mul(0x10000);
        let n = if u == 0 { 0 } else { norm_shift(u) };
        let (ref_m, ref_e) = a660(
            0x5b95_0000,
            0,
            ((u as u32) << (n & 0x1f)) as i32 >> 16,
            0xf - n,
        );

        let mut amant = [0i16; 16];
        let mut aexp = [0i16; 16];
        for k in 0..16 {
            amant[k] = self.w[(0x52 + 2 * k) / 2];
            aexp[k] = self.w[(0x72 + 2 * k) / 2];
        }
        let wgt = d4f0(&amant, &aexp);

        let cap = if self.g(0x48) < -0x18ff {
            0x200
        } else {
            let d = s16(self
                .g(0x48)
                .wrapping_mul(0x10000)
                .wrapping_add(self.g(0x50).wrapping_mul(-0x10000))
                >> 16);
            let v = s16(d.wrapping_mul(0x3332) >> 16);
            if v < 0x201 {
                if v < 0x101 {
                    0x100
                } else {
                    v
                }
            } else {
                0x200
            }
        };

        let mut acc: i64 = 0;
        for k in 0..16 {
            let prev_exp = self.g(0xf2);
            let cur = bandvec[k];
            let old = self.dw(k);
            let nc = if cur == 0 { 0 } else { norm_shift(cur) };
            let no = if old == 0 { 0 } else { norm_shift(old) };
            let (dm, de) = a310(
                ((cur as u32) << (nc & 0x1f)) as i32 >> 16,
                p3 - nc,
                ((old as u32) << (no & 0x1f)) as i32 >> 16,
                prev_exp - no,
            );
            let de = s16(de) - 1;

            let mut a_m = self.g(0x52 + 2 * k);
            let mut a_e = self.g(0x72 + 2 * k);
            let mut b_m = self.g(0x92 + 2 * k);
            let mut b_e = self.g(0xb2 + 2 * k);

            let (lm, le) = d3f0(k as i32, dm, de, ref_m, ref_e);
            let (cm, ce) = a310(a_m, a_e, cap, 7);

            let hold = if self.g(0x138) < 8 {
                let count = self.g(0x138);
                dbc0(&mut a_m, &mut a_e, &mut b_m, &mut b_e, lm, le, p5, count)
            } else if a4f0(lm, le, cm, ce) {
                dcb0(&mut a_m, &mut a_e, &mut b_m, &mut b_e, lm, le, p5)
            } else {
                let h = self.g(0xd2 + 2 * k);
                if h < 1 {
                    a_m = b_m;
                    a_e = b_e;
                    b_m = 0x7fff;
                    b_e = 5;
                    s16(p5)
                } else {
                    da80(&mut b_m, &mut b_e, lm, le, h, cap)
                }
            };
            self.s(0xd2 + 2 * k, hold);
            self.s(0x92 + 2 * k, b_m);
            self.s(0xb2 + 2 * k, b_e);
            self.s(0x52 + 2 * k, a_m);
            self.s(0x72 + 2 * k, a_e);
            self.set_dw(k, cur);

            let (mut dm2, mut de2) = (0i32, 0i32);
            d970(
                &mut l44, &mut l40, &mut l50, &mut l4c, &mut dm2, &mut de2, a_m, a_e, lm, le, dm,
                de, k as i32,
            );
            dde0(
                &mut l54,
                &mut l80,
                &mut l5c,
                &mut l64,
                dm,
                de,
                lm,
                le,
                a_m,
                a_e,
                &mut l28,
                &mut l30,
                side,
                &mut l68,
                k as i32,
                p7,
                wgt[k] as i32,
                self.g(0x48),
                self.g(0x50),
            );

            if k % 2 == 0 {
                l74 = dm2;
                l7c = de2;
            } else {
                let (m, e) = a310(l74, l7c, dm2, de2);
                l74 = m;
                l7c = e;
                let v = shr_dsp(s16(l74).wrapping_mul(0x10000), s16(l7c - 7)) >> 16;
                self.s(0x102 + 2 * lag_ix, v);
                lag_ix += 1;
            }
            if k == 0 {
                self.s(
                    0x132,
                    shr_dsp(s16(dm2).wrapping_mul(0x10000), s16(de2 - 5)) >> 16,
                );
            }

            acc = acc.wrapping_add(shr_dsp(s16(lm).wrapping_mul(0x10000), s16(le - 7)) as i64);
        }

        let low = ((acc as u64 >> 4) & 0xffff_ffff) as u32 as i32;
        let t = s16(low.wrapping_add(0x700_0000) >> 16);
        self.s(
            0x13a,
            t.wrapping_mul(0x9a1c)
                .wrapping_add(self.g(0x13a).wrapping_mul(0xcccc))
                >> 16,
        );

        let (m, e) = a660(s16(l44).wrapping_mul(0x10000), l40, l50, l4c);
        self.s(
            0xf4,
            shift_sat(s16(m).wrapping_mul(0x10000), s16(e - 2)) >> 16,
        );

        let v = shift_sat(s16(*side).wrapping_mul(0x3332), s16(l68 - 3));
        *side = super::dp_score::sat32(v as i64 + 0x8000) >> 16;

        let (dm, de) = if s16(l64) < 3 {
            (0x4000, 3)
        } else {
            (s16(l5c), s16(l64))
        };
        let (m, e) = a660(s16(l54).wrapping_mul(0x10000), l80, dm, de);
        let v = shift_sat(s16(m).wrapping_mul(0x10000), s16(e - 3));
        self.s(0xf2, p3);
        s16(super::dp_score::sat32(v as i64 + 0x8000) >> 16)
    }

    /// `FUN_1031ea20`: the smoothed frame level at `+0x48`.
    fn ea20(&mut self, act: i32) {
        let cur = self.g(0x4a);
        let prev = self.g(0x48);
        let (ca, ea, cb, eb);
        if s16(act) < 0x4001 || cur < -0x27ff {
            if cur <= prev {
                if s16(act) > 0x2000 {
                    ca = 0xa3d8;
                    ea = 1;
                    cb = 0xfd70;
                    eb = 7;
                } else if prev < -0x27ff {
                    return;
                } else {
                    ca = 0x8312;
                    ea = -2;
                    cb = 0xffbe;
                    eb = 7;
                }
            } else {
                ca = 0xa3d8;
                ea = 1;
                cb = 0xfd70;
                eb = 7;
            }
        } else if prev < cur {
            ca = 0xcccc;
            ea = 4;
            cb = 0xe666;
            eb = 7;
        } else if s16(act) > 0x2000 {
            ca = 0xa3d8;
            ea = 1;
            cb = 0xfd70;
            eb = 7;
        } else if prev < -0x27ff {
            return;
        } else {
            ca = 0x8312;
            ea = -2;
            cb = 0xffbe;
            eb = 7;
        }
        let v = self.mix(cur, ca, ea, prev, cb, eb);
        self.s(0x48, v);
    }

    /// `FUN_1031ebf0`: the noise floor at `+0x50`, a 0.1/0.9 one-pole on `+0x4a`.
    fn ebf0(&mut self) {
        let v = self.mix(self.g(0x4a), 0xcccc, 4, self.g(0x50), 0xe666, 7);
        self.s(0x50, v);
    }

    /// The shared "scale both terms, block-float add, rescale to Q7" tail of
    /// `ea20` and `ebf0`.
    fn mix(&self, x: i32, cx: i32, ex: i32, y: i32, cy: i32, ey: i32) -> i32 {
        let ux = x.wrapping_mul(cx);
        let nx = if ux == 0 { 0 } else { norm_shift(ux) };
        let uy = y.wrapping_mul(cy);
        let ny = if uy == 0 { 0 } else { norm_shift(uy) };
        let (m, e) = a310(
            ((ux as u32) << (nx & 0x1f)) as i32 >> 16,
            ex - nx,
            ((uy as u32) << (ny & 0x1f)) as i32 >> 16,
            ey - ny,
        );
        shr_dsp(s16(m).wrapping_mul(0x10000), s16(e - 7)) >> 16
    }

    /// `FUN_1031e080`: the activity decision, adjusting the measure in place.
    fn e080(&self, act: &mut i32) -> i32 {
        let d = self
            .g(0x4a)
            .wrapping_mul(0x10000)
            .wrapping_add(self.g(0x48).wrapping_mul(-0x10000));
        let lim = s16(self.g(0x4c).wrapping_mul(0x100_0000) >> 16);
        let mut v = s16(*act).wrapping_mul(0x10000);
        if lim > -0x5a00 && self.g(0x4a) <= lim {
            if self.g(0x138) < 8 {
                v = v.wrapping_sub(0x3800_0000);
            } else if d < -0x13ff_faff
                || ((self.g(0x44) & 1) == 0 && (self.g(0x48) - self.g(0x50)) > 0x7ff)
            {
                if d >= -0x13ff_ffff {
                    v = v.wrapping_sub(0x1000_0000);
                } else if d >= -0x1dff_ffff {
                    v = v.wrapping_sub(0x1800_0000).wrapping_add(
                        s16((d.wrapping_add(0x1e00_0000) as u32 >> 16) as i32).wrapping_mul(0xccca),
                    );
                } else if d >= -0x27ff_ffff {
                    v = v.wrapping_add((-0x1c00_0000i32).wrapping_sub(
                        s16((d.wrapping_add(0x2800_0000) as u32 >> 16) as i32).wrapping_mul(-0xcccc)
                            >> 1,
                    ));
                } else {
                    let mut done = false;
                    if d > -0x3200_0000 && d640(self.g(0x44), 0x12) > 1 {
                        v = v.wrapping_sub(0x3c00_0000).wrapping_add(
                            s16((d.wrapping_add(0x3200_0000) as u32 >> 16) as i32)
                                .wrapping_mul(0x33328),
                        );
                        done = true;
                    }
                    if !done {
                        v = v.wrapping_sub(0x1c00_0000);
                    }
                }
            } else {
                v = v.wrapping_sub(0x0999_8000);
            }
        }
        *act = s16((v as u32 >> 16) as i32);
        i32::from(*act != 0 && v >= 0)
    }
}

/// `FUN_1031e8c0`: the frame's level from the band vector.
fn e8c0(bandvec: &[i32; 16], p2: i32, p3: i32) -> i32 {
    let first = bandvec[0];
    let neg = if first == i32::MIN {
        i32::MAX
    } else {
        first.wrapping_neg()
    };
    let mut acc: i64 = (neg >> 1) as i64;
    for k in 0..16 {
        acc = acc.wrapping_add(bandvec[k] as i64);
    }
    let sh = s16(norm64(acc));
    let shifted = if sh >= 0 {
        ((acc as u64) << (sh.min(63) as u32)) as i64
    } else {
        let s = (-sh).min(63) as u32;
        acc >> s
    };
    let lo = shifted as i32;
    let e = s16(p2 - (sh & 0xffff) + 1);
    let u = s16(p3).wrapping_mul(0x10000);
    let n = if u == 0 { 0 } else { norm_shift(u) };
    let (m, e2) = a660(lo, e, ((u as u32) << (n & 0x1f)) as i32 >> 16, 0xf - n);
    if m == 0 || s16(e2) < -0x1f {
        return -0x59ff;
    }
    let lg = b100(m, e2);
    let nn = if lg == 0 { 0 } else { norm_shift(lg) };
    let mant = s16(((lg as u32) << (nn & 0x1f)) as i32 >> 16);
    let sh2 = s16((0xf - nn) - 5);
    let prod = mant.wrapping_mul(0xc0a8);
    let scaled = if sh2 >= 0 {
        ((prod as u32) << (sh2 & 0x1f)) as i32
    } else if sh2 < -0x1e {
        prod / 0xaa16 + (prod >> 31)
    } else {
        prod >> ((-sh2) & 0x1f)
    };
    scaled.wrapping_add(0x4ba_7000) >> 16
}

/// `FUN_10310150`: the 64-bit normalising shift.
fn norm64(v: i64) -> i32 {
    let (mut lo, mut hi) = (v as u32, (v >> 32) as u32);
    if lo == 0 && hi == 0 {
        return 0;
    }
    if (hi as i32) < 0 {
        lo = !lo;
        hi = !hi;
    }
    let mut m_lo: u32 = 0;
    let mut m_hi: u32 = 0x80;
    let mut i = 0i32;
    loop {
        if (m_lo & lo) != 0 || (m_hi & hi) != 0 {
            break;
        }
        m_lo = (m_lo >> 1) | (m_hi << 31);
        i += 1;
        m_hi = ((m_hi as i32) >> 1) as u32;
        if (i as i16) >= 0x28 {
            break;
        }
    }
    i - 9
}

/// `FUN_1030a660`: block-float divide, returns `(mantissa, exponent)`.
fn a660(p1: i32, p2: i32, p3: i32, p4: i32) -> (i32, i32) {
    let mut num = p1;
    let mut den = s16(p3);
    let sgn = den ^ p1;
    if num < 0 {
        num = if num == i32::MIN { i32::MAX } else { -num };
    }
    if den < 0 {
        den = if den == -0x8000 { 0x7fff } else { -den };
    }
    if den == 0 {
        return (0, 0);
    }
    let mut e = s16(p2);
    if den.wrapping_mul(0x10000) <= num {
        num >>= 1;
        e = s16(e + 1);
    }
    let mut q = s16((num / den) >> 1);
    if sgn < 0 {
        q = s16(-q);
    }
    let u = q.wrapping_mul(0x10000);
    let n = if u == 0 { 0 } else { norm_shift(u) };
    let m = s16(((u as u32) << (n & 0x1f)) as i32 >> 16);
    if m == 0 {
        return (0, 0);
    }
    (m, s16(s16(e - n) - s16(p4)))
}

/// `FUN_1030a600`: fractional divide.
fn a600(p1: i32, p2: i32) -> i32 {
    let den0 = s16(p2);
    let sgn = den0 ^ p1;
    let num = if p1 < 0 {
        if p1 == i32::MIN {
            i32::MAX
        } else {
            -p1
        }
    } else {
        p1
    };
    let den = if den0 < 0 {
        if den0 == -0x8000 {
            0x7fff
        } else {
            -den0
        }
    } else {
        den0
    };
    if den == 0 {
        return 0;
    }
    let mut r = if num < den { 0 } else { (num / den) >> 1 };
    if sgn < 0 {
        r = -r;
    }
    r
}

/// `FUN_1030a890`: `2^x` with a block-float rescale.
fn a890(p1: i32, p2: i32, p3: i32) -> i32 {
    let v = shr_dsp(s16(p1).wrapping_mul(0x10000), s16(p2 - 0xf));
    let (val, exp) = a7a0(v);
    shr_dsp(val, s16(exp - p3)).wrapping_add(0x8000) >> 16
}

/// `FUN_1030b490`: block-float `(m2, e2) - (m1, e1)`.
fn b490(m1: i32, e1: i32, m2: i32, e2: i32) -> (i32, i32) {
    let (m1, e1, m2, e2) = (s16(m1), s16(e1), s16(m2), s16(e2));
    if m1 != 0 && e2 - e1 < 0x20 {
        if m2 == 0 || e1 - e2 > 0x1f {
            if m1 == -0x8000 {
                return (0x7fff, e1);
            }
            return (-m1, e1);
        }
        let top = (if e1 <= e2 { e2 } else { e1 }) + 1;
        let va = shr_dsp(m1.wrapping_mul(0x10000), e1 - top);
        let vb = shr_dsp(m2.wrapping_mul(0x10000), e2 - top);
        let t = vb.wrapping_sub(va);
        let n = if t == 0 { 0 } else { norm_shift(t) };
        let m = s16(((t as u32) << (n & 0x1f)) as i32 >> 16);
        if m == 0 {
            return (0, 0);
        }
        return (m, s16(top - n));
    }
    if m2 == 0 {
        return (0, 0);
    }
    (m2, e2)
}

/// `FUN_1031d640`: population count over the low `n` bits.
fn d640(v: i32, n: i32) -> i32 {
    let mut cnt = 0;
    if s16(n) > 0 {
        let mut p = v;
        let mut u = p as i16;
        for _ in 0..(n as u16) {
            u >>= 1;
            cnt += p & 1;
            p = u as u16 as i32;
        }
    }
    cnt
}

/// `FUN_1031d4f0`: the sixteen per-band weights the frame aggregate uses, a tilted
/// template offset by the band levels' mean and normalised to a fixed total.
fn d4f0(amant: &[i16; 16], aexp: &[i16; 16]) -> [i16; 16] {
    let mut acc: i64 = 0;
    for k in 0..16 {
        let v = shr_dsp(
            (amant[k] as i32).wrapping_mul(0x10000),
            s16(aexp[k] as i32 - 5),
        );
        acc = acc.wrapping_add(v as i64);
    }
    let mut mean = s16(((acc as u64 >> 20) & 0xffff) as i32);
    if mean < -0x5a00 {
        mean = -0x5a00;
    }
    let mut base = (mean + 0x2800).wrapping_mul(0x10000);
    let mut out = [0i16; 16];
    let mut sum: i32 = 0;
    for k in 0..16 {
        let neg = s16(-(amant[k] as i32)) as i64;
        let v64 = neg << 16;
        let sh = s16(aexp[k] as i32 - 5);
        let shifted = if sh >= 0 {
            ((v64 as u64) << (sh.min(63) as u32)) as i64
        } else {
            v64 >> ((-sh).min(63) as u32)
        };
        let x = s16((shifted as i32).wrapping_add(base) >> 16).clamp(0x400, 0x2800);
        base = base.wrapping_sub(0x266_6000);
        out[k] = x as i16;
        sum = sum.wrapping_add(x);
    }
    let g = s16(a600(0x3ff_ffff, s16(sum >> 4)));
    for k in 0..16 {
        out[k] = (((out[k] as i32).wrapping_mul(g).wrapping_mul(2)) >> 16) as i16;
    }
    out
}

/// `FUN_1031d3f0`: the band's log level relative to the frame reference.
fn d3f0(row: i32, dm: i32, de: i32, ref_m: i32, ref_e: i32) -> (i32, i32) {
    let u = s16(dm).wrapping_mul(s16(ref_m)).wrapping_mul(2);
    let n = if u == 0 { 0 } else { 0x1e - bsr(u) };
    let e = s16(de) - n + s16(ref_e);
    let m = s16(((u as u32) << (n & 0x1f)) as i32 >> 16);
    let u2 = (0x20 - s16(row)).wrapping_mul(0x200_0000);
    let n2 = if u2 == 0 { 0 } else { norm_shift(u2) };
    let (sm, se) = a310(m, e, ((u2 as u32) << (n2 & 0x1f)) as i32 >> 16, -0x1e - n2);
    if sm == 0 {
        return (-0x8000, 0xf);
    }
    let lg = b100(sm, se);
    let n3 = if lg == 0 { 0 } else { 0x1e - bsr(lg) };
    (
        s16(((lg as u32) << (n3 & 0x1f)) as i32 >> 16),
        s16(0xf - n3),
    )
}

#[inline]
fn bsr(v: i32) -> i32 {
    let u = (if v < 0 { !v } else { v }) as u32;
    if u == 0 {
        0x1f
    } else {
        31 - u.leading_zeros() as i32
    }
}

/// `FUN_1031dbc0`: the warm-up band update, a running mean over the first frames.
#[allow(clippy::too_many_arguments)]
fn dbc0(
    a_m: &mut i32,
    a_e: &mut i32,
    b_m: &mut i32,
    b_e: &mut i32,
    lm: i32,
    le: i32,
    p7: i32,
    count: i32,
) -> i32 {
    let c = s16(count);
    let u = s16(*a_m).wrapping_mul(c).wrapping_mul(2);
    let n = if u == 0 { 0 } else { norm_shift(u) };
    let e = s16(*a_e) + (0xf - n);
    let (m, e) = a310(((u as u32) << (n & 0x1f)) as i32 >> 16, e, lm, le);
    let u2 = (c + 1).wrapping_mul(0x10000);
    let n2 = if u2 == 0 { 0 } else { norm_shift(u2) };
    let (m2, e2) = a660(
        s16(m).wrapping_mul(0x10000),
        e,
        ((u2 as u32) << (n2 & 0x1f)) as i32 >> 16,
        0xf - n2,
    );
    *a_m = m2;
    *a_e = e2;
    *b_m = 0x7fff;
    *b_e = 5;
    s16(p7)
}

/// `FUN_1031dcb0`: the steady-state band update when the level sits below the cap.
fn dcb0(
    a_m: &mut i32,
    a_e: &mut i32,
    b_m: &mut i32,
    b_e: &mut i32,
    lm: i32,
    le: i32,
    p7: i32,
) -> i32 {
    let mut ca = 0x6666i32;
    let mut cb = 0x7333i32;
    let mut ea = s16(le - 3);
    let cur_m = *a_m;
    let cur_e = *a_e;
    if a4f0(lm, le, cur_m, cur_e) && a4f0(lm, le, -0x7000, 5) {
        ca = 0x4000;
        ea = s16(le);
        cb = 0x4000;
    }
    let ua = ca.wrapping_mul(s16(lm)).wrapping_mul(2);
    let na = if ua == 0 { 0 } else { norm_shift(ua) };
    let ub = s16(cur_m).wrapping_mul(cb).wrapping_mul(2);
    let nb = if ub == 0 { 0 } else { norm_shift(ub) };
    let (m, e) = a310(
        ((ua as u32) << (na & 0x1f)) as i32 >> 16,
        ea - na,
        ((ub as u32) << (nb & 0x1f)) as i32 >> 16,
        cur_e - nb,
    );
    *a_m = m;
    *a_e = e;
    *b_m = 0x7fff;
    *b_e = 5;
    s16(p7)
}

/// `FUN_1031da80`: the hold-branch update of the band's secondary level.
fn da80(b_m: &mut i32, b_e: &mut i32, lm: i32, le: i32, hold: i32, cap: i32) -> i32 {
    let mut m = *b_m;
    let mut e = *b_e;
    let (dm, de) = b490(0x4000, 2, m, e);
    let (sm, se) = a310(cap, 7, m, e);
    if a4f0(lm, le, dm, de) {
        *b_m = s16(lm);
        *b_e = s16(le);
    } else {
        if a4f0(lm, le, sm, se) {
            let u = s16(m).wrapping_mul(0xe666);
            let n = if u == 0 { 0 } else { norm_shift(u) };
            let e1 = s16(e - n);
            let m1 = s16(((u as u32) << (n & 0x1f)) as i32 >> 16);
            let u2 = s16(lm).wrapping_mul(0xcccc);
            let n2 = if u2 == 0 { 0 } else { norm_shift(u2) };
            let e2 = s16(s16(le) - n2 - 3);
            let (nm, ne) = a310(m1, e1, ((u2 as u32) << (n2 & 0x1f)) as i32 >> 16, e2);
            m = nm;
            e = ne;
        }
        *b_m = m;
        *b_e = e;
    }
    s16(hold - 1)
}

/// `FUN_1031d970`: the per-band deviation aggregate.
#[allow(clippy::too_many_arguments)]
fn d970(
    s_m: &mut i32,
    s_e: &mut i32,
    n_m: &mut i32,
    n_e: &mut i32,
    out_m: &mut i32,
    out_e: &mut i32,
    a_m: i32,
    a_e: i32,
    lm: i32,
    le: i32,
    dm: i32,
    de: i32,
    row: i32,
) {
    let (d1m, d1e) = b490(a_m, a_e, lm, le);
    *out_m = d1m;
    *out_e = d1e;
    let (d2m, d2e) = b490(lm, le, a_m, a_e);
    let (mut mm, mut ee) = (s16(d1m), d1e);
    if mm < 0 {
        ee = d2e;
        mm = d2m;
    }
    if s16(row) < 6 {
        let u = s16(mm).wrapping_mul(s16(dm)).wrapping_mul(2);
        let n = if u == 0 { 0 } else { norm_shift(u) };
        let e = s16(ee - n + de);
        let (m, e2) = a310(*s_m, *s_e, ((u as u32) << (n & 0x1f)) as i32 >> 16, e);
        *s_m = m;
        *s_e = e2;
        let (m2, e3) = a310(*n_m, *n_e, dm, de);
        *n_m = m2;
        *n_e = e3;
    }
}

/// `FUN_1031dde0`: the frame activity aggregate and the running band-deviation peak.
#[allow(clippy::too_many_arguments)]
fn dde0(
    p1_m: &mut i32,
    p1_e: &mut i32,
    p3_m: &mut i32,
    p3_e: &mut i32,
    dm: i32,
    de: i32,
    lm: i32,
    le: i32,
    a_m: i32,
    a_e: i32,
    cur_m: &mut i32,
    cur_e: &mut i32,
    peak_m: &mut i32,
    peak_e: &mut i32,
    row: i32,
    gain: i32,
    weight: i32,
    lvl48: i32,
    lvl50: i32,
) {
    if s16(lm) == 0 {
        let (d, _) = b490(*peak_m, *peak_e, *cur_m, *cur_e);
        if s16(d) > 0 {
            *peak_m = *cur_m;
            *peak_e = *cur_e;
        }
        *cur_m = 0;
        *cur_e = 0;
        return;
    }
    let t = (s16(row).wrapping_mul(0x10000)) >> 5;
    let t = if t == i32::MIN { i32::MAX } else { -t };
    let u = s16(t.wrapping_add(0x10000).wrapping_mul(0x4000) >> 16)
        .wrapping_mul(s16(gain))
        .wrapping_mul(2);
    let n = if u == 0 { 0 } else { norm_shift(u) };
    let (sm, se) = a310(dm, de, ((u as u32) << (n & 0x1f)) as i32 >> 16, -0x13 - n);
    let (qm, qe) = a660(s16(dm).wrapping_mul(0x10000), de, sm, se);
    let mut exp = s16(qe + 5);
    let scaled = s16(s16(qm).wrapping_mul(s16(weight)).wrapping_mul(2) >> 16);
    let (m3, e3) = a310(*p3_m, *p3_e, scaled, exp);
    *p3_m = m3;
    *p3_e = e3;

    let (dv, dve) = b490(lm, le, a_m, a_e);
    let mut mag = s16(dv);
    if mag < 0 {
        mag = if mag == -0x8000 { 0x7fff } else { -mag };
    }
    let u2 = scaled.wrapping_mul(mag).wrapping_mul(2);
    let n2 = if u2 == 0 { 0 } else { 0x1e - bsr(u2) };
    exp = s16(exp + (s16(dve) - n2));
    let (m1, e1) = a310(*p1_m, *p1_e, ((u2 as u32) << (n2 & 0x1f)) as i32 >> 16, exp);
    *p1_m = m1;
    *p1_e = e1;

    let mut contrib = mag;
    if (lvl48 - s16(lvl50)) > 0x800 || lvl48 < -0x1900 {
        if s16(row) == 0 {
            contrib = s16((mag.wrapping_mul(0x8000)) >> 16);
        } else if s16(row) == 1 {
            contrib = s16((mag.wrapping_mul(0xc000)) >> 16);
        }
    }
    let (nm, ne) = a310(*cur_m, *cur_e, contrib, dve);
    let (d, _) = b490(*peak_m, *peak_e, nm, ne);
    if s16(d) > 0 {
        *peak_m = nm;
        *peak_e = ne;
    }
    *cur_m = s16(contrib);
    *cur_e = s16(dve);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One captured `FUN_1031d810` call from the `noisy` conformance vector: the
    /// block before, the pass's band vector, its two scalar arguments, and the
    /// block after. 194 + 16 + 2 + 194 words.
    const D810_CASE: &str = concat!(
        "18626 1 26900 7 26176 49 -1720 105 24920 55 8913 38 28744 11 30735 14 22152 22 ",
        "-8955 10 -22455 12 -9843 4 21382 3 -5206 0 4800 0 552 0 -24576 15 0 0 -7708 -8748 ",
        "0 300 -9515 -22565 -19829 -17862 -17388 -17867 -18498 -20142 -19894 -19319 -19491 ",
        "-19869 -20914 -21645 -24228 -27742 -30538 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 32767 ",
        "32767 32767 32767 32767 32767 32767 32767 32767 32767 32767 32767 32767 32767 ",
        "32767 32767 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 300 300 300 300 300 300 300 300 300 ",
        "300 300 300 300 300 300 300 -3 0 -1035 -668 -1026 -2622 -2057 -1931 271 226 128 ",
        "151 143 -47 71 154 493 251 -155 177 118 -43 175 38 231 401 -152 61 -78 77 209 39 ",
        "618 1202 882 8 -10292 22654 -16 0 0 0 0 0 0 0 0 32380 19778 21536 20235 20382 ",
        "28666 27004 27608 -30 -30 -30 -30 -30 -30 -30 -30 -23 -23 -23 -24 -25 -29 -33 -34 ",
        "0 63 132361 647116 1936394 3560032 3214317 2951310 1407975 1404006 1635056 2055735 ",
        "974340 363662 250444 38999 4737 512 -3 20 1289 2 -8244 9 -29686 29 21088 54 3053 ",
        "49 2190 45 31719 21 27750 21 -3344 24 24119 31 -8700 14 -29554 5 -11700 3 -26537 0 ",
        "4737 0 512 0 -24576 15 0 0 -7710 -8835 0 300 -9447 -22570 -19863 -17869 -17338 ",
        "-17832 -18434 -20050 -19814 -19255 -19427 -19831 -20915 -21628 -24182 -27687 ",
        "-30502 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 32767 32767 32767 32767 32767 32767 32767 ",
        "32767 32767 32767 32767 32767 32767 32767 32767 32767 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ",
        "5 5 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 -3 0 -968 ",
        "-1035 -668 -1026 -2622 -2057 -86 102 227 390 288 87 145 209 271 226 128 151 143 ",
        "-47 71 154 493 251 -155 177 118 -43 175 38 -39 618 1202 8 -10276 22654 -16 0 0 0 0 ",
        "0 0 0 0 32380 19778 21536 20235 20382 28666 27004 27608 -30 -30 -30 -30 -30 -30 ",
        "-30 -30 -23 -23 -23 -24 -25 -29 -33 -34 0 63"
    );

    fn ints(s: &str) -> Vec<i32> {
        s.split_whitespace().map(|t| t.parse().unwrap()).collect()
    }

    #[test]
    fn one_pass_reproduces_the_reference_block() {
        let v = ints(D810_CASE);
        assert_eq!(v.len(), 406);
        let mut t = NoiseTracker { w: [0i16; NW] };
        for i in 0..194 {
            t.w[i] = v[i] as i16;
        }
        let mut bv = [0i32; 16];
        bv.copy_from_slice(&v[194..210]);
        t.push_pass(&bv, v[210], v[211]);
        for i in 0..194 {
            assert_eq!(t.w[i] as i32, v[212 + i], "block word {:#x}", 2 * i);
        }
    }
}
