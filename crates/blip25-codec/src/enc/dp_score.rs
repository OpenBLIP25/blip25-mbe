//! The e100 DP score chain: block-float accumulation of the sixteen per-row score
//! vectors, with the per-row spectral weighting that precedes it.
//!
//! The reference (`FUN_1030e100`) accumulates sixteen 32-word vectors in block float
//! and normalises the result into the 32-word DP score array that drives coarse pitch
//! (`FUN_1031b7b0`) and the octave decision (`FUN_1031b980`). Before each vector is
//! added it may be scaled by a per-row gain, gated by a noise classifier
//! (`FUN_1030dd30`) reading the `ctx+0x938` noise-tracker block:
//!
//! * gate 0 - the vector is accumulated verbatim.
//! * gate 1 - rows 0 and 1 take their gain from `FUN_1030eed0`, rows 2..15 from
//!   `FUN_1030ed30`; both return a `(mantissa, exponent-bump)` pair applied as a Q15
//!   multiply plus an exponent adjust.
//!
//! `st` is the 168-word window of the noise-tracker block covering byte offsets
//! `0x40..0x190`, the only part any of the three functions reads. Passing `None`
//! selects the gate-0 path for every row, which is the unweighted accumulation.
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

/// Words of the `ctx+0x938` noise-tracker block that the weighting reads,
/// starting at block offset `0x40`.
pub(crate) const ST_WORDS: usize = 168;

/// Opt-in switch (`BLIP25_DP_SCORE=1`, OFF by default) for the weighted DP score
/// accumulation. Read once.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("BLIP25_DP_SCORE").as_deref() == Ok("1"))
}

// ---------------------------------------------------------------- primitives

#[inline]
pub(crate) fn s16(x: i32) -> i32 {
    x as i16 as i32
}

/// `sh >= 0` shifts left, otherwise right, saturating the shift count at 31.
#[inline]
pub(crate) fn shr_dsp(v: i32, sh: i32) -> i32 {
    if sh < 0 {
        if sh < -0x1e {
            v >> 31
        } else {
            v >> ((-sh) & 0x1f)
        }
    } else {
        ((v as u32) << (sh & 0x1f)) as i32
    }
}

/// `0x1e - bsr(v < 0 ? !v : v)`, with `bsr(0)` leaving 0x1f as the reference does.
#[inline]
pub(crate) fn norm_shift(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let u = (if v < 0 { !v } else { v }) as u32;
    let top = if u == 0 {
        0x1f
    } else {
        31 - u.leading_zeros() as i32
    };
    0x1e - top
}

#[inline]
pub(crate) fn sat32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// `FUN_103101b0`: shift left with overflow detection, else shift right.
pub(crate) fn shift_sat(v: i32, sh: i32) -> i32 {
    if sh < 0 {
        return if sh < -0x1e {
            v >> 31
        } else {
            v >> ((-sh) & 0x1f)
        };
    }
    let mask = ((-1i32) as u32).wrapping_shl(((0x1f - sh) & 0x1f) as u32) as i32;
    let mut t = mask & v;
    if v < 0 {
        t = t.wrapping_sub(mask);
    }
    if t != 0 {
        return if v >= 0 { i32::MAX } else { i32::MIN };
    }
    ((v as u32) << (sh & 0x1f)) as i32
}

/// `FUN_1030a440`: block-float `(m1, e1) > (m2, e2)`.
pub(crate) fn a440(m1: i32, e1: i32, m2: i32, e2: i32) -> bool {
    let i1 = s16(m1);
    let i2 = i1.wrapping_mul(0x10000);
    let e = if i2 == 0 { s16(e2) } else { s16(e1) };
    let i3 = s16(m2);
    let i4 = i3.wrapping_mul(0x10000);
    let d = if i4 != 0 { s16(e2) } else { e };
    let d = s16(d - e);
    if d > 0 {
        let d = -d;
        if d > -0x21 {
            if d >= 0 {
                return i4 < ((i2 as u32) << (d & 0x1f)) as i32;
            }
            if d > -0x1f {
                return i4 < i2 >> ((-d) & 0x1f);
            }
        }
        return i4 < i1 / 0x8001 + (i1 >> 31);
    }
    if d > -0x21 {
        if d >= 0 {
            return (((i4 as u32) << (d & 0x1f)) as i32) < i2;
        }
        if d > -0x1f {
            return i4 >> ((-d) & 0x1f) < i2;
        }
    }
    i3 / 0x8001 + (i3 >> 31) < i2
}

/// `FUN_1030a4f0`: block-float `(m1, e1) < (m2, e2)`.
pub(crate) fn a4f0(m1: i32, e1: i32, m2: i32, e2: i32) -> bool {
    let i1 = s16(m1);
    let i2 = i1.wrapping_mul(0x10000);
    let e = if i2 == 0 { s16(e2) } else { s16(e1) };
    let i3 = s16(m2);
    let i4 = i3.wrapping_mul(0x10000);
    let d = if i4 != 0 { s16(e2) } else { e };
    let d = s16(d - e);
    if d > 0 {
        let d = -d;
        if d > -0x21 {
            if d >= 0 {
                return (((i2 as u32) << (d & 0x1f)) as i32) < i4;
            }
            if d > -0x1f {
                return i2 >> ((-d) & 0x1f) < i4;
            }
        }
        return i1 / 0x8001 + (i1 >> 31) < i4;
    }
    if d > -0x21 {
        if d >= 0 {
            return i2 < ((i4 as u32) << (d & 0x1f)) as i32;
        }
        if d > -0x1f {
            return i2 < i4 >> ((-d) & 0x1f);
        }
    }
    i2 < i3 / 0x8001 + (i3 >> 31)
}

/// `FUN_1030a310`: block-float add.
pub(crate) fn a310(am: i32, ae: i32, bm: i32, be: i32) -> (i32, i32) {
    let (am, ae, bm, be) = (s16(am), s16(ae), s16(bm), s16(be));
    if am != 0 && be - ae < 0x20 {
        if bm == 0 || ae - be > 0x1f {
            return (am, ae);
        }
        let e = if ae > be { ae } else { be } + 1;
        let va = shr_dsp(am.wrapping_mul(0x10000), ae - e);
        let vb = shr_dsp(bm.wrapping_mul(0x10000), be - e);
        let t = va.wrapping_add(vb);
        let n = norm_shift(t);
        let m = s16(((t as u32) << (n & 0x1f)) as i32 >> 16);
        if m == 0 {
            return (0, 0);
        }
        return (m, s16(e - n));
    }
    if bm == 0 {
        return (0, 0);
    }
    (bm, be)
}

const A7A0_C: [i32; 6] = [22370, 20171, 29100, 31487, 22713, 16384];

/// `FUN_1030a7a0`: exponential polynomial, returns `(value, exponent)`.
pub(crate) fn a7a0(p1: i32) -> (i32, i32) {
    #[inline]
    fn step(t: i64, x: i64, sh: u32, mask: u32) -> i32 {
        ((((t * x * 2 + 0x8000) >> sh) as i32) as u32 & mask) as i32
    }
    let hi = s16(p1 >> 16);
    let x = s16(p1.wrapping_sub(hi.wrapping_mul(0x10000)) >> 1) as i64;
    let exp = hi + 1;
    let mut r = step(A7A0_C[0] as i64, x, 3, 0xffffe000);
    let mut t = s16(r.wrapping_add(A7A0_C[1] * 0x10000) >> 16) as i64;
    r = step(t, x, 3, 0xffffe000);
    t = s16(r.wrapping_add((A7A0_C[2] * 0x10000) >> 1) >> 16) as i64;
    r = step(t, x, 2, 0xffffc000);
    t = s16(r.wrapping_add((A7A0_C[3] * 0x10000) >> 1) >> 16) as i64;
    r = step(t, x, 1, 0xffff8000);
    t = s16(r.wrapping_add(A7A0_C[4] * 0x10000) >> 16) as i64;
    r = step(t, x, 1, 0xffff8000);
    (r.wrapping_add(A7A0_C[5] * 0x10000), exp)
}

const B100_C: [i32; 6] = [23170, -23636, 22285, -23636, 16714, -16384];

/// `FUN_1030b100`: log2, returns a packed value whose high half is the exponent.
pub(crate) fn b100(m: i32, e: i32) -> i32 {
    #[inline]
    fn step(t: i64, x: i64, mul: i64, add: i64, sh: u32, mask: u32) -> i32 {
        ((((t * x * mul + add) >> sh) as i32) as u32 & mask) as i32
    }
    let u = s16(m).wrapping_mul(0x10000);
    let n = if u == 0 { 0 } else { norm_shift(u) };
    let v = (((u as u32) << (n & 0x1f)) as i32).wrapping_add(B100_C[0] * -0x10000);
    let x = s16(v >> 16) as i64;
    let mut r = step(B100_C[1] as i64, x, 2, 0x8000, 0, 0xffff0000);
    let mut t = s16(r.wrapping_add(B100_C[2] * 0x10000) >> 16) as i64;
    r = step(t, x, 2, 0x8000, 0, 0xffff0000);
    t = s16(r.wrapping_add(B100_C[3] * 0x10000) >> 16) as i64;
    r = step(t, x, 2, 0x8000, 1, 0xffff8000);
    t = s16(r.wrapping_add(B100_C[4] * 0x10000) >> 16) as i64;
    r = step(t, x, 8, 0x20000, 0, 0xfffc0000);
    let acc = r.wrapping_add(B100_C[5] * 0x10000);
    (acc >> 0xf).wrapping_add(s16(s16(e) - n).wrapping_mul(0x10000))
}

// ------------------------------------------------------- the weighting stage

/// Word accessors into the `ctx+0x938` window. `A` is the sixteen band levels at
/// block `+0x52`/`+0x72`, `B` the sixteen at `+0x140`/`+0x160`.
#[inline]
fn a_mant(st: &[i16; ST_WORDS], i: usize) -> i32 {
    st[9 + i] as i32
}
#[inline]
fn a_exp(st: &[i16; ST_WORDS], i: usize) -> i32 {
    st[25 + i] as i32
}
#[inline]
fn b_mant(st: &[i16; ST_WORDS], i: usize) -> i32 {
    st[128 + i] as i32
}
#[inline]
fn b_exp(st: &[i16; ST_WORDS], i: usize) -> i32 {
    st[144 + i] as i32
}

/// `FUN_1030dd30`: the weighting gate, plus the noise level and flag the row
/// weights consume. Returns `(gate, level_mant, level_exp, flag)`.
fn dd30(st: &[i16; ST_WORDS]) -> (i32, i32, i32, i32) {
    let mut best_m = a_mant(st, 0);
    let mut best_e = a_exp(st, 0);
    for i in 1..16 {
        if a440(a_mant(st, i), a_exp(st, i), best_m, best_e) {
            best_m = a_mant(st, i);
            best_e = a_exp(st, i);
        }
    }

    let mut sm = 0i32;
    let mut se = 0i32;
    for i in 0..8 {
        let (m, e) = a310(sm, se, b_mant(st, 8 + i), b_exp(st, 8 + i));
        sm = m;
        se = e;
    }

    let mut flag = st[160] as i32;
    if a440(sm, se, 0x7a12, -10) {
        flag = 1;
    } else {
        if a440(sm, se, 0xc35, -10)
            && a440(
                sm,
                se,
                (st[126] as i32).wrapping_mul(0x8ccc) >> 16,
                st[127] as i32,
            )
        {
            return dd30_tail(st, 1);
        }
        let v50 = st[8] as i32;
        if v50 <= -0x1e00
            && !a440(best_m, best_e, s16(0xb000), 5)
            && (st[4] as i32 - v50) - 0xf00 >= 0
        {
            return (0, 0, 0, flag);
        }
    }
    dd30_tail(st, flag)
}

fn dd30_tail(st: &[i16; ST_WORDS], flag: i32) -> (i32, i32, i32, i32) {
    let mut acc: i64 = 0;
    for i in 0..8 {
        acc += shr_dsp(a_mant(st, i).wrapping_mul(0x10000), a_exp(st, i) - 5) as i64;
    }
    let mut v = s16(((acc as u64 >> 19) & 0xffff) as i32);
    if v < -0x7000 {
        v = -0x7000;
    }
    (1, v, 5, flag)
}

/// Normalise `a7a0`'s output into a `(mantissa, exponent)` gain, optionally floored
/// at `(cm, ce)`.
fn finish(t: i32, floor: Option<(i32, i32)>) -> (i32, i32) {
    let (val, exp) = a7a0(t);
    let n = if val == 0 { 0 } else { norm_shift(val) };
    let e = s16(exp - n);
    let v = ((val as u32) << (n & 0x1f)) as i32;
    if let Some((cm, ce)) = floor {
        if a4f0(s16(v >> 16), e, cm, ce) {
            return (cm, ce);
        }
    }
    (s16(v >> 16), e)
}

/// `FUN_1030eed0`: the gain for rows 0 and 1.
fn eed0(st: &[i16; ST_WORDS], inner: i32) -> (i32, i32) {
    let v112 = st[105] as i32;
    let v122 = st[113] as i32;
    let v134 = st[122] as i32;
    let v136 = st[123] as i32;
    let mut a = (v136.wrapping_mul(0x10000) >> 1).wrapping_add(v134.wrapping_mul(0x10000) >> 1);
    if inner > 0 {
        a = (v122
            .wrapping_add(v112)
            .wrapping_mul(0x10000)
            .wrapping_sub(a >> 1))
        .wrapping_mul(2);
    }
    let u = a.wrapping_add(0xee00_0000u32 as i32);
    if u >= 0 {
        return (0x4000, 1);
    }
    let t = (s16(u >> 16)
        .wrapping_mul(0xaac0u32 as i32)
        .wrapping_add((((u as u32 & 0xffff) as i32).wrapping_mul(0x5560)) >> 15))
        >> 9;
    finish(t, Some((0x51eb, -6)))
}

/// `FUN_1030ed30`: the gain for rows 2..15.
fn ed30(st: &[i16; ST_WORDS], nf_mant: i32, nf_exp: i32, row: i32, flag: i32) -> (i32, i32) {
    let kk = s16(row << 11);
    let sh = s16(5 - nf_exp);
    let v6 = shr_dsp(kk.wrapping_mul(-0x2222).wrapping_add(0x1000_0000), sh)
        .wrapping_add(s16(nf_mant).wrapping_mul(0x10000));
    let mut thr = shr_dsp(kk.wrapping_mul(-0x4444).wrapping_sub(0x3000_0000), sh);
    if v6 < thr {
        thr = v6;
    }

    let k = row as usize;
    let mut lvl = shr_dsp(
        a_mant(st, k).wrapping_mul(0x10000),
        a_exp(st, k) - s16(nf_exp),
    );
    if flag != 0 && row > 8 {
        let u = b100(b_mant(st, k), b_exp(st, k));
        let n = if u == 0 { 0 } else { norm_shift(u) };
        let floor = s16(shift_sat(
            ((u as u32) << (n & 0x1f)) as i32,
            s16((0xf - n) - nf_exp),
        ));
        if s16(lvl >> 16) < floor {
            lvl = floor.wrapping_mul(0x10000);
        }
    }

    let d = thr.wrapping_sub(lvl);
    if d < 0 {
        let lo = (((d as u32 & 0xffff) as i32).wrapping_shl(14)) >> 15;
        let t = s16(d >> 16).wrapping_mul(0x8000).wrapping_add(lo) >> 9;
        return finish(t, None);
    }
    (0x4000, 1)
}

// -------------------------------------------------------------- accumulation

/// The Q15 scale the reference applies to each 32-bit word of a row vector.
#[inline]
fn q15mul(v: i32, g: i32) -> i32 {
    let lo = (v as u32 & 0xffff) as i64;
    let hi = s16(v >> 16) as i64;
    let a = ((lo * g as i64) >> 15) as i32;
    let b = (hi * g as i64 * 2) as i32;
    a.wrapping_add(b)
}

/// `FUN_1030f820`: saturating block-float accumulate of `src` into `dst`.
fn f820(dst: &mut [i32; 32], ed: i32, src: &[i32; 32], es: i32) -> i32 {
    let e = ed.max(es);
    let sa = ed - e;
    let sb = es - e;
    for i in 0..32 {
        let a = shr_dsp(dst[i], sa) as i64;
        let b = shr_dsp(src[i], sb) as i64;
        dst[i] = sat32(a + b);
    }
    e
}

/// `FUN_103173a0`: normalise the accumulator and round to 16 bits.
fn pack(acc: &[i32; 32]) -> [i16; 32] {
    let mx = acc.iter().map(|&v| (v as i64).abs()).max().unwrap_or(0);
    let be = if mx == 0 {
        0
    } else {
        (63 - (mx as u64).leading_zeros() as i32) - 30
    };
    let mut out = [0i16; 32];
    for i in 0..32 {
        let x = shift_sat(acc[i], -be);
        let c = sat32(x as i64 + 0x8000);
        out[i] = (c >> 16) as i16;
    }
    out
}

/// Run the DP score chain over the sixteen row vectors and their exponents.
///
/// `st` supplies the noise-tracker window that gates and scales the rows; `None`
/// accumulates every row verbatim.
pub(crate) fn run(
    scorevecs: &[[i32; 32]; 16],
    expo: &[i32; 16],
    st: Option<&[i16; ST_WORDS]>,
) -> [i16; 32] {
    let side = st.map(|s| (s, dd30(s)));
    let mut acc = [0i32; 32];
    let mut e: i32 = -30;
    for k in 0..16 {
        let mut v = scorevecs[k];
        v[0] = 0;
        v[1] = 0;
        let mut ex = expo[k];
        if let Some((s, (gate, nf_mant, nf_exp, flag))) = side {
            if gate != 0 {
                let (g, adj) = if k < 2 {
                    eed0(s, k as i32)
                } else {
                    ed30(s, nf_mant, nf_exp, k as i32, flag)
                };
                for i in 0..32 {
                    v[i] = q15mul(v[i], g);
                }
                ex = s16(ex + adj);
            }
        }
        e = f820(&mut acc, e, &v, ex);
    }
    pack(&acc)
}

/// The half of `FUN_1030e100` that survives the call: rows 0 and 1 — the pair the
/// per-band combine folds into band 0 of the spectrum `FUN_1031c960` reads — are
/// scaled by `FUN_1030eed0`'s gain **in the caller's own buffer**, and their
/// exponents take its bump. Rows 2..15 are scaled only into the accumulator's
/// scratch copy, so the caller's buffer keeps them verbatim. `None`, or a zero
/// gate, leaves every row alone.
pub(crate) fn weight_rows01_in_place(
    scorevecs: &mut [[i32; 32]; 16],
    expo: &mut [i32; 16],
    st: Option<&[i16; ST_WORDS]>,
) {
    let Some(s) = st else { return };
    if dd30(s).0 == 0 {
        return;
    }
    for k in 0..2 {
        let (g, adj) = eed0(s, k as i32);
        scorevecs[k][0] = 0;
        scorevecs[k][1] = 0;
        for i in 0..32 {
            scorevecs[k][i] = q15mul(scorevecs[k][i], g);
        }
        expo[k] = s16(expo[k] + adj);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One captured `FUN_1030e100` call from the `noisy` conformance vector, whose
    /// gate is 1 so every weighted path runs: the noise window (168), the gate, the
    /// sixteen raw row vectors (16 x 32), their exponents (16) and the emitted
    /// 32-word score.
    const E100_CASE: &str = concat!(
        "-24576 15 0 0 -7710 -8835 0 300 -9447 -22570 -19863 -17869 -17338 -17832 -18434 ",
        "-20050 -19814 -19255 -19427 -19831 -20915 -21628 -24182 -27687 -30502 5 5 5 5 5 5 ",
        "5 5 5 5 5 5 5 5 5 5 32767 32767 32767 32767 32767 32767 32767 32767 32767 32767 ",
        "32767 32767 32767 32767 32767 32767 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 300 300 300 ",
        "300 300 300 300 300 300 300 300 300 300 300 300 300 -3 0 -968 -1035 -668 -1026 ",
        "-2622 -2057 -86 102 227 390 288 87 145 209 271 226 128 151 143 -47 71 154 493 251 ",
        "-155 177 118 -43 175 38 -39 618 1202 8 -10276 22654 -16 0 0 0 0 0 0 0 0 32380 ",
        "19778 21536 20235 20382 28666 27004 27608 -30 -30 -30 -30 -30 -30 -30 -30 -23 -23 ",
        "-23 -24 -25 -29 -33 -34 0 63 1 255 38 1500 0 0 1 0 0 1343090 603200 513956 1151162 ",
        "953794 132500 90116 31250 8424 426240 707890 98696 685346 10783364 22141480 ",
        "6632722 1707514 1170322 3340450 2695346 1376660 911188 381544 18698 50674 177970 ",
        "183188 5954 44944 320840 0 0 8893474 3032840 9671834 18509290 11191050 2374452 ",
        "1240480 926356 2211010 857480 15876 567524 1355138 386020 555268 929736 352114 ",
        "483850 75400 603242 2276368 1043810 438290 325354 169874 76232 151610 84052 35456 ",
        "1850 0 0 7612786 1387170 4068466 9053098 1596370 6155344 1360298 438538 418826 ",
        "623752 1734122 1335298 1532052 9026 2064530 2831560 919730 614602 296146 313274 ",
        "1062162 428852 249938 295250 33680 24400 18212 33140 18850 10706 0 0 2958436 ",
        "1983970 1704040 9483058 4113940 11722418 1476688 283546 2116130 1555762 2700218 ",
        "4646500 5311476 3457442 2788100 1942900 740402 480328 256946 1006538 527616 163106 ",
        "487528 125738 57538 15210 936 9556 1250 1186 0 0 6168532 2320000 3108386 260836 ",
        "9189034 16416370 5274756 2122682 17299720 14599602 4074640 1245924 7457480 5116660 ",
        "1495112 1696850 1162394 976570 633060 12308816 10238738 366820 321490 244936 ",
        "256834 373570 193780 98482 84058 292672 0 0 2667092 518858 85370 4165200 7998778 ",
        "7644628 2005650 4130388 2358250 10752628 2145680 3177512 1702850 1701650 807066 ",
        "1426114 4851236 3875450 1200712 936520 701770 510696 367336 28618 103066 20180 ",
        "68962 25330 46792 74600 0 0 1256292 151300 1008674 1629728 520250 3034354 2985658 ",
        "1437554 76564 1290274 695754 1550996 6627620 7993924 2032538 1164290 969416 ",
        "2380820 2448034 521650 152018 279458 175528 140650 323370 105448 178810 27040 ",
        "111994 22048 0 0 4296890 7483034 8257492 1091842 1472362 1306418 760072 46834 ",
        "623492 1440116 1268020 1243106 2035674 2386240 357290 791056 205796 422932 877682 ",
        "524420 568586 164930 9970 92650 191458 28628 60680 342548 122434 20978 0 0 2384420 ",
        "7292178 8121448 412546 4417312 734810 2411620 6890850 8828210 2141858 497250 43528 ",
        "73034 645170 452498 328402 744680 1842602 1306810 172778 261154 294290 372730 ",
        "77200 261746 283460 72848 19664 37280 68042 0 0 1873834 517570 865476 417992 ",
        "2802760 1971874 3306004 5304436 4993586 651482 526144 3955464 4082258 5254804 ",
        "5202152 3846772 661536 656960 307666 939690 1421522 539540 191556 19720 100746 ",
        "160298 146890 3866 19796 340 0 0 229850 1138036 1654736 583090 1853136 328648 ",
        "2961194 2502418 46330 809588 648050 4369370 2875720 315044 650740 823912 69172 ",
        "71514 47700 128970 95120 170570 248202 1124 1768 53024 29284 19058 1394 3464 0 0 ",
        "6550480 4012786 8556808 8303464 3587092 170644 1833770 498100 118196 2397668 ",
        "4588884 11505096 2454736 3727226 4177000 994850 2250536 941224 78724 938320 356882 ",
        "338042 2822058 2105316 327114 328354 468938 246650 69642 116944 0 0 7142536 477200 ",
        "8732960 5491124 211298 1511362 614672 1065472 1873540 488450 472018 1465168 26618 ",
        "754960 986180 1654778 1112072 645184 20932 152200 18730 55652 63880 500890 101690 ",
        "612 5536 29448 2050 7930 0 0 513530 1147922 505474 430586 997930 1018984 1471552 ",
        "221428 295290 126970 353800 345640 389210 169168 132194 324176 180954 133348 54468 ",
        "63890 62420 20498 9544 91412 74330 11714 15760 2906 5706 7930 0 0 64778 31610 ",
        "37780 72410 81770 11234 16162 16936 30474 21618 32090 18778 44420 72298 31466 784 ",
        "16936 27460 30146 17082 9754 1156 3626 360 260 98 1394 320 122 116 0 0 5536 9872 ",
        "6292 1924 788 20 2624 5140 788 4090 5972 1258 634 1970 6740 2330 482 2248 2260 ",
        "1296 1492 1602 1828 1258 356 1210 1042 90 2 2 -8 -6 -4 -4 -6 -6 -6 -6 -6 -6 -6 -8 ",
        "-8 -8 -8 -8 0 0 15866 8402 12432 21190 12849 25315 7830 6299 10988 10056 7071 ",
        "10465 12940 9416 7801 7325 3960 3679 2225 5162 4879 1180 1313 726 424 311 234 190 ",
        "128 138"
    );

    fn ints(s: &str) -> Vec<i32> {
        s.split_whitespace().map(|t| t.parse().unwrap()).collect()
    }

    fn case() -> ([i16; ST_WORDS], [[i32; 32]; 16], [i32; 16], Vec<i32>) {
        let v = ints(E100_CASE);
        assert_eq!(v.len(), 729);
        let mut st = [0i16; ST_WORDS];
        for i in 0..ST_WORDS {
            st[i] = v[i] as i16;
        }
        let mut pre = [[0i32; 32]; 16];
        for k in 0..16 {
            pre[k].copy_from_slice(&v[169 + 32 * k..169 + 32 * (k + 1)]);
        }
        let mut expo = [0i32; 16];
        expo.copy_from_slice(&v[681..697]);
        (st, pre, expo, v)
    }

    #[test]
    fn weighted_accumulation_reproduces_the_reference_score() {
        let (st, pre, expo, v) = case();
        assert_eq!(dd30(&st).0, v[ST_WORDS], "gate");
        let got = run(&pre, &expo, Some(&st));
        for i in 0..32 {
            assert_eq!(got[i] as i32, v[697 + i], "score word {i}");
        }
    }

    /// Without a noise window every row is accumulated verbatim, which is what a
    /// gate-0 window must also produce.
    #[test]
    fn no_window_is_the_unweighted_accumulation() {
        let (st, pre, expo, _) = case();
        let mut off = st;
        // drive the band levels to the floor and the noise floor below its
        // threshold, the combination `dd30` answers 0 to.
        for i in 9..25 {
            off[i] = -0x7000;
        }
        off[8] = -0x2000;
        off[4] = 0;
        assert_eq!(dd30(&off).0, 0);
        assert_eq!(run(&pre, &expo, Some(&off)), run(&pre, &expo, None));
    }

    /// The write-back reaches rows 0 and 1 and stops there. Widening it to the
    /// rows `FUN_1030ed30` scales would corrupt bands 1..7 of the spectrum the
    /// band-voicing row is built from, which the reference leaves verbatim.
    #[test]
    fn only_rows_0_and_1_are_written_back() {
        let (st, pre, expo, _) = case();
        assert_ne!(dd30(&st).0, 0, "fixture gate must fire");
        let (mut svs, mut ex) = (pre, expo);
        weight_rows01_in_place(&mut svs, &mut ex, Some(&st));
        for k in 0..2 {
            let (g, adj) = eed0(&st, k as i32);
            assert_eq!(ex[k], s16(expo[k] + adj), "row {k} exponent");
            for i in 2..32 {
                assert_eq!(svs[k][i], q15mul(pre[k][i], g), "row {k} word {i}");
            }
            assert_eq!((svs[k][0], svs[k][1]), (0, 0), "row {k} head");
        }
        for k in 2..16 {
            assert_eq!(svs[k], pre[k], "row {k} must be untouched");
            assert_eq!(ex[k], expo[k], "row {k} exponent must be untouched");
        }
    }

    /// A zero gate, or no window at all, leaves every row alone.
    #[test]
    fn gate_zero_writes_nothing_back() {
        let (st, pre, expo, _) = case();
        let mut off = st;
        for i in 9..25 {
            off[i] = -0x7000;
        }
        off[8] = -0x2000;
        off[4] = 0;
        assert_eq!(dd30(&off).0, 0);
        for w in [Some(&off), None] {
            let (mut svs, mut ex) = (pre, expo);
            weight_rows01_in_place(&mut svs, &mut ex, w);
            assert_eq!(svs, pre);
            assert_eq!(ex, expo);
        }
    }
}
