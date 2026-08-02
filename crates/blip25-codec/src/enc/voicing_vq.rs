//! Voicing quantizer (b1): bit-exact VQ port of the reference's
//! candidate-search, metric, and 64-bit-normalize routines.
//!
//! Given the analysis-stage per-band voicing measurement arrays `a4/a5/a6`
//! (16 bands each, the three sub-arrays of the voicing param struct at
//! offsets +0/+0x20/+0x40), produces the transmitted 5-bit `b1` index.
//! Validated bit-exact against the reference on both conformance clips
//! (199/199 each, whole file).
//!
//! Integration boundary: the `a4/a5/a6` measurement itself is produced by the
//! analysis leaf functions over the already-bit-exact bins — that leaf port is
//! the remaining PCM->b1 step; this module is the (bit-exact) quantizer/decision
//! that consumes it.

// The voicing codebook `VOICING_CB`, the `L = 56` override gate
// (`l56_gate_from_b1` + `L56_STEP` / `L56_L`), and the `(L, step)` derivation
// `l_step_from_b0_b1` moved to `crate::shared::voicing_map` so the decoder can
// reach them without `enc/`. `VOICING_CB` is re-exported for `voicing_b1_vq`.
pub(crate) use crate::shared::voicing_map::VOICING_CB;

#[inline]
fn s16(x: i64) -> i64 {
    let x = x & 0xffff;
    if x >= 0x8000 {
        x - 0x10000
    } else {
        x
    }
}
#[inline]
fn s32(x: i64) -> i64 {
    let x = x & 0xffff_ffff;
    if x >= 0x8000_0000 {
        x - 0x1_0000_0000
    } else {
        x
    }
}
#[inline]
fn mul16pow(v: i64) -> i64 {
    let t = s32(s32(v) * s32(v));
    let t = s32(t * 2);
    s16((t >> 16) & 0xffff)
}

/// Build the per-band hypothesis-energy arrays `A1/A2/A3` (code 1/2/0-or-3)
/// from `a4` (norm) and `a5`. Read in reverse (word `15-k` for band `k`).
fn build_arrays(a4: &[i16; 16], a5: &[i16; 16]) -> ([i64; 16], [i64; 16], [i64; 16]) {
    let mut a1 = [0i64; 16];
    let mut a2 = [0i64; 16];
    let mut a3 = [0i64; 16];
    for k in 0..16 {
        let wi = 15 - k;
        let c = a4[wi] as i64;
        let d = a5[wi] as i64;
        let cc = mul16pow(c);
        let omc = s16(0x7fff - c);
        let e = mul16pow(omc);
        let dd = mul16pow(d);
        let ome = s16(0x7fff - e);
        let mut acc = s32(s16(ome) * s16(dd));
        acc = s32(acc * 2);
        acc >>= 1;
        acc = s32(acc + s32((e << 16) & 0xffff_ffff));
        a1[k] = s16((acc >> 16) & 0xffff);
        let omcc = s16(0x7fff - cc);
        let omd = s16(0x7fff - d);
        let t = mul16pow(omd);
        let mut acc = s32(s16(t) * s16(omcc));
        acc = s32(acc * 2);
        acc >>= 1;
        acc = s32(acc + s32((cc << 16) & 0xffff_ffff));
        a2[k] = s16((acc >> 16) & 0xffff);
        let mut acc = s32(s16(omcc) * s16(dd));
        acc = s32(acc * 2);
        acc >>= 1;
        acc = s32(acc + s32((cc << 16) & 0xffff_ffff));
        a3[k] = s16((acc >> 16) & 0xffff);
    }
    (a1, a2, a3)
}

/// Leading-sign 64-bit normalize -> shift (`count - 9`).
/// Load-bearing fix: the two's-complement of the operand happens only when the
/// HIGH dword is negative; the negation is skipped whenever `hi` is non-negative
/// (in particular when `hi == 0`).
fn norm64(lo: i64, hi: i64) -> i64 {
    let mut low = lo & 0xffff_ffff;
    let mut high = hi & 0xffff_ffff;
    if s32(high) < 0 {
        low = (!low) & 0xffff_ffff;
        high = (!high) & 0xffff_ffff;
    }
    let mut lo_mask = 0i64;
    let mut hi_mask = 0x80i64;
    let mut count = 0i64;
    loop {
        let lo_hit = lo_mask & low;
        let hi_hit = hi_mask & high;
        if (lo_hit | hi_hit) != 0 {
            break;
        }
        lo_mask = ((lo_mask >> 1) | ((hi_mask & 1) << 31)) & 0xffff_ffff;
        count += 1;
        hi_mask = s32(hi_mask) >> 1;
        if count >= 0x28 {
            break;
        }
    }
    count - 9
}
// Register-pair signature mirrored from the reference; the `hi` lane is
// intentionally unread in this low-word helper.
#[allow(unused_variables)]
fn shl64_lo(lo: i64, hi: i64, cl: i64) -> i64 {
    let cl = cl & 63;
    if cl == 0 {
        return lo & 0xffff_ffff;
    }
    if cl >= 32 {
        return 0;
    }
    (lo << cl) & 0xffff_ffff
}
fn sar64_lo(lo: i64, hi: i64, cl: i64) -> i64 {
    let cl = cl & 63;
    if cl == 0 {
        return lo & 0xffff_ffff;
    }
    if cl >= 32 {
        return (s32(hi) >> (cl - 32)) & 0xffff_ffff;
    }
    ((lo >> cl) | (hi << (32 - cl))) & 0xffff_ffff
}

/// Compute the transmitted 5-bit voicing index `b1` from the analysis voicing
/// measurement (`a4/a5/a6`, 16 bands each). Bit-exact against the reference.
pub(crate) fn voicing_b1_vq(a4: &[i16; 16], a5: &[i16; 16], a6: &[i16; 16]) -> u16 {
    voicing_b1_vq_cb(a4, a5, a6, &VOICING_CB, 32, 4)
}

/// Score ONE codebook word: the 16-band weighted accumulate, then the 64-bit
/// leading-sign normalize. Returns `(neg_exp, mant)`. This is the SINGLE source of
/// truth for candidate scoring -- [`voicing_b1_vq_cb`] calls it, so a diagnostic that
/// calls it cannot silently fork from the scoring the headline chain actually runs.
fn score_one(
    aa1: &[i64; 16],
    aa2: &[i64; 16],
    aa3: &[i64; 16],
    a6: &[i16; 16],
    word: u32,
) -> (i64, i64) {
    let mut codeword = word as i64 & 0xffff_ffff;
    let mut acc: i64 = 0;
    for j in 0..16 {
        let mword = a6[15 - j] as i64;
        let code = codeword & 3;
        let sel = match code {
            1 => s16(aa1[j]),
            2 => s16(aa2[j]),
            _ => s16(aa3[j]),
        };
        let mut prod = s32((mword * sel) & 0xffff_ffff);
        prod = s32((prod * 2) & 0xffff_ffff);
        acc += prod;
        codeword = s32(codeword) >> 2;
    }
    let lo = acc & 0xffff_ffff;
    let hi = (acc >> 32) & 0xffff_ffff;
    let shift = norm64(lo, hi);
    let sh = s16(shift & 0xffff);
    let shifted = if sh >= 0 {
        shl64_lo(lo, hi, sh & 0xff)
    } else {
        sar64_lo(lo, hi, (-sh) & 0xff)
    };
    let mant = (s32(shifted) >> 16) & 0xffff;
    let neg_exp = (-sh) & 0xffff;
    (neg_exp, mant)
}

/// Same as [`voicing_b1_vq`] but with an explicit codebook / candidate config,
/// exposed for cross-checking against captured data.
pub(crate) fn voicing_b1_vq_cb(
    a4: &[i16; 16],
    a5: &[i16; 16],
    a6: &[i16; 16],
    cb: &[u32],
    ncand: usize,
    stride: usize,
) -> u16 {
    let (aa1, aa2, aa3) = build_arrays(a4, a5);
    let mut best_exp: i64 = 0x7fff;
    let mut best_mant: i64 = 0x7fff;
    let mut best_idx = 0u16;
    for ci in 0..ncand {
        let (neg_exp, mant) = score_one(&aa1, &aa2, &aa3, a6, cb[ci * stride]);
        if mant == 0 {
            return ci as u16;
        }
        let exp_s = s16(neg_exp);
        let mant_s = s16(mant);
        let better = exp_s < s16(best_exp) || (exp_s == s16(best_exp) && mant_s < s16(best_mant));
        if better {
            best_exp = neg_exp;
            best_mant = mant;
            best_idx = ci as u16;
        }
    }
    best_idx
}
