//! FROM-BITS voiced-phase residual generator.
//!
//! `resid[o]` is the per-harmonic phase residual the voiced OLA reads. It is
//! produced from `M_l` (the per-harmonic log-spectral amplitude) by:
//!   1. [`build_s`] — lay `M_l` into a symmetric spectrum `s`.
//!   2. [`hilbert_resid`] — a 10-tap Hilbert transform of `s`.
//!
//! Bit-exact.

const RESID_STEP: i64 = 0x5c50000;
const RESID_COEF: [i64; 10] = [18411, 6137, 3682, 2630, 2046, 1674, 1416, 1227, 1083, 969];

#[inline]
fn sat16_64(x: i64) -> i32 {
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x as i32
    }
}
#[inline]
fn i16w64(x: i64) -> i32 {
    let v = (x as u32) & 0xffff;
    if v >= 0x8000 {
        (v as i32) - 0x10000
    } else {
        v as i32
    }
}
#[inline]
fn resid_h1(x: i64, sh: u32) -> i64 {
    let m = 31 - sh;
    let mask = (1i64 << m) - 1;
    let r = if x >= 0 { x & mask } else { -((-x) & mask) };
    r << sh
}

/// Lay `M_l[0..L]` into the symmetric working spectrum `s[128]`.
pub(crate) fn build_s(amp: &[i32; 128], l: usize) -> [i32; 128] {
    let mut s = [0i32; 128];
    s[20..(l + 20)].copy_from_slice(&amp[..l]);
    let last = amp[l - 1] as i64;
    let mut k = 0i64;
    let mut idx = 20 + l;
    while idx < 84 {
        s[idx] = sat16_64(((last << 16) - (k + 1) * RESID_STEP) >> 16);
        k += 1;
        idx += 1;
    }
    for kk in 1..(128 - 83) {
        s[83 + kk] = s[83 - kk];
    }
    for kk in 1..20 {
        s[19 - kk] = s[19 + kk];
    }
    s
}

/// 10-tap Hilbert transform of `s` -> `resid[0..L]`.
pub(crate) fn hilbert_resid(s: &[i32; 128], l: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(l);
    for o in 0..l {
        let mut acc = 0i64;
        for j in 0..10 {
            let f = s[20 + o + (2 * j + 1)] as i64;
            let b = s[20 + o - (2 * j + 1)] as i64;
            acc += 2 * RESID_COEF[j] * (f - b);
        }
        out.push(i16w64(resid_h1(acc, 1) >> 16));
    }
    out
}

/// Convenience: `M_l` (i16 per-harmonic log amplitude) -> `resid` (i16).
pub(crate) fn resid_from_ml(ml: &[i16], l: usize) -> Vec<i16> {
    if l == 0 {
        return Vec::new();
    }
    let mut amp = [0i32; 128];
    for i in 0..l.min(ml.len()) {
        amp[i] = ml[i] as i32;
    }
    hilbert_resid(&build_s(&amp, l), l)
        .into_iter()
        .map(|v| v as i16)
        .collect()
}
