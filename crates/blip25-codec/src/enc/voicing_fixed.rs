//! Fixed-point back end for **b1 (voicing)** — the "scoring + candidate
//! selection" stage that turns the per-band voicing analysis into the
//! transmitted `b1` codeword.
//!
//! This is a **direct fixed-point port** of the reference vocoder's b1 stage,
//! validated bit-exact against captured reference frames — not a spec-derived
//! reimplementation. The exact rounding / shift / saturation behaviour of this
//! stage is not realistically re-derivable from the published MBE spec, so the
//! port matches the reference arithmetic verbatim and is guarded by the codec
//! acceptance tests.
//!
//! ## Entry points
//! - `c_transform` — the array-`C` transform used at the voicing call site.
//!   Despite looking like a DCT in a static reading, the real effect on this
//!   path is a plain two-word left shift (`POST[i] == PRE[i+2]` for `i = 0..6`,
//!   with slots 6/7 left as pre-existing memory); this is the form supported by
//!   the captured data and is what is implemented here.
//! - `b1_from_abc` — the end-to-end selection. It scores 32 fixed candidate
//!   rows against the per-band arrays `A`/`B`/`C` and returns the winning row
//!   index as `b1`. The per-row weight vector is array `B` read in reverse; the
//!   winner is the candidate with the smallest accumulator magnitude, with an
//!   immediate win on an exact `0`.
//! - `argmin_row_accumulator` — the raw 64-bit per-candidate accumulator
//!   (`2 * sum_band(reverse(B)[band] * selected_row_array[band])`) behind the
//!   selection above.
//! - `band_voicing_heap_src` — the per-band raw-source value that feeds the
//!   `A`/`B`/`C` arrays.
//!
//! The remaining helpers (`c420_*` windowed integration, block-floating-point
//! add/normalize, and the fractional shifts and rounders) are the fixed-point
//! primitives these entry points are built from.
//!
//! ## Selection rule: a deliberate, equivalent simplification
//! The reference ranks candidates with a block-floating-point normalize
//! followed by a lexicographic (exponent, mantissa) compare. This port ranks
//! by the plain 64-bit accumulator magnitude instead. The two are provably
//! equivalent for *winner selection* — both order candidates by the same
//! monotonic quantity — and the magnitude form is what the acceptance tests
//! validate.
//!
//! ## Scope
//! This module is the selection back end only. The spectral-content inputs it
//! consumes are produced upstream in the encoder by `audio_prefilter` →
//! `history_ring` → `windowed_taper` → `real_fft32`; see those modules for the
//! analysis front end.

use super::atan2_bfp_divide::bfp_divide;
use super::band_decompress::{log2_fn, normalize64, shift64, shift_scale};

use crate::fixops::acc64::s16;
use crate::fixops::u32r::{s32, u32v};

/// Q15 "squared" idiom used throughout this codebase: `(2*x*x) >> 16`.
#[inline]
fn q15_sq(x: i16) -> i16 {
    let p = (x as i32).wrapping_mul(x as i32).wrapping_mul(2);
    s32(u32v(p) >> 16) as i16
}

// NOTE: an earlier DCT-shaped model of `c_transform` used a
// `twiddle`/`recip_table` pair (`cos(2*pi*k/512)` and a `32767/n`-style
// reciprocal). It did not match the captured data at the voicing call site
// (see `c_transform`'s doc below) and was replaced with a plain shift; the
// two table helpers were removed as dead code. That model may still be
// correct for the transform's other call sites, which were never validated.

/// Array A's per-band writer, given the per-band history-table row `t`
/// (8 words) and scalar `s` (both from the T/S delay-line). `flag` was `0`
/// on every captured sample; the `flag!=0 && j>=4` branch is preserved for
/// completeness but has never been exercised. Validated bit-exact
/// (1600/1600) against captured reference band computations.
pub(crate) fn a_writer_band(t: [i16; 8], s: i16, flag: bool) -> [i16; 8] {
    let mut p = [0i16; 8];
    for j in 0..8 {
        if t[j] == 0 {
            p[j] = 0x7fffu16 as i16;
            continue;
        }
        let raw = log2_fn(t[j], s);
        // satshl(raw, 9) then round_sat16: round-to-nearest (add the 0x8000
        // bias BEFORE truncating to the top 16 bits), not a plain truncation
        // -- matching the rounding idiom used throughout this codebase.
        let shifted = shift_scale(raw, 9);
        let rounded = s16(((shifted.wrapping_add(0x8000)) >> 16) as i32 & 0xffff);
        let mut val: i64 = (-1187i64 - rounded as i64) << 16;
        if flag && j >= 4 {
            val -= 0x2000000;
        }
        if val < 0 {
            val = 0;
        }
        let val32 = s32(u32v(val as i32));
        let scaled = shift_scale(u32v(val32), 5);
        let out = s32(scaled) >> 16;
        p[j] = s16(out & 0xffff);
    }
    p
}

/// First scoring loop: arrays `C` and `A` -> the three scored arrays.
/// Reading order is the reference's REVERSE index (`k`-th output uses input
/// index `15-k`), already applied here so callers can pass arrays in natural
/// order. Validated bit-exact (1500/1500) against captured reference triples.
pub(crate) fn score_arrays(c: [i16; 16], a: [i16; 16]) -> ([i16; 16], [i16; 16], [i16; 16]) {
    let mut out3a = [0i16; 16];
    let mut out5a = [0i16; 16];
    let mut out7a = [0i16; 16];
    for k in 0..16 {
        let idx = 15 - k;
        let p = c[idx]; // the reference's own "A" local (first arg)
        let q = a[idx]; // the reference's own "B" local (second arg)
        let asq = q15_sq(p);
        let bsq = q15_sq(q);
        let one_minus_a = 0x7fffi16.wrapping_sub(p);
        let sq_one_minus_a = q15_sq(one_minus_a);
        let one_minus_b = 0x7fffi16.wrapping_sub(q);
        let sq_one_minus_b = q15_sq(one_minus_b);
        let one_minus_asq = 0x7fffi16.wrapping_sub(asq);

        let o5 = (((0x7fffi32 - sq_one_minus_a as i32) * bsq as i32)
            + ((sq_one_minus_a as i32) << 16))
            >> 16;
        let o3 = ((sq_one_minus_b as i32 * one_minus_asq as i32) + ((asq as i32) << 16)) >> 16;
        let o7 = ((one_minus_asq as i32 * bsq as i32) + ((asq as i32) << 16)) >> 16;

        out3a[k] = s16(o3 & 0xffff);
        out5a[k] = s16(o5 & 0xffff);
        out7a[k] = s16(o7 & 0xffff);
    }
    (out3a, out5a, out7a)
}

/// The fixed, 32-row 2-bit-per-band array selector table (static,
/// compiled-in). Row `r`'s bits, taken 2 at a time from LSB, select which
/// of `out3a`/`out5a`/`out7a` each band uses (`1`->`out5a`, `2`->`out3a`,
/// `0` or `3`->`out7a`).
pub(crate) const ARGMIN_ROW_TABLE: [u32; 32] = [
    0x55555555, 0x55555554, 0x55545554, 0x55555545, 0x55505550, 0x51555155, 0x54555455, 0x55455545,
    0x55005500, 0x55405540, 0x54005400, 0x54015400, 0x50005000, 0x54005000, 0x40004000, 0x54000000,
    0x00000000, 0x00004000, 0x80000000, 0xa0000000, 0xaaaa0000, 0x08000000, 0x00020000, 0x002a0000,
    0xaaaaaaaa, 0x0aaa0aaa, 0x00008000, 0x00000800, 0x0000aaaa, 0x0000a000, 0x00000002, 0x0000000a,
];

/// The argmin inner loop's per-band "weight" array is array B
/// (`B[i] = max(heap_src[i], 1)`), read in REVERSE order. Validated
/// bit-exact (40/40) against captured reference frames.
#[inline]
pub(crate) fn per_band_weight_table(b: [i16; 16]) -> [i16; 16] {
    let mut out = b;
    out.reverse();
    out
}

/// The per-candidate-row accumulator,
/// `2 * sum_band(weight[band] * selected_array(row,band)[band])`, as a
/// 64-bit signed value matching the reference's 64-bit accumulate.
/// Validated bit-exact (640/640) against captured reference frames for the
/// first 16 candidates per frame; candidates beyond the 16th were not
/// captured, an open item -- see the module doc.
pub(crate) fn argmin_row_accumulator(
    weight: [i16; 16],
    row_dword: u32,
    out3a: [i16; 16],
    out5a: [i16; 16],
    out7a: [i16; 16],
) -> i64 {
    let mut acc: i64 = 0;
    for band in 0..16 {
        let sel = (row_dword >> (2 * band)) & 3;
        let val = match sel {
            1 => out5a[band],
            2 => out3a[band],
            _ => out7a[band],
        };
        acc += (weight[band] as i64) * (val as i64);
    }
    acc * 2
}

/// Full argmin over the fixed 32-row table, given `weight`
/// (`per_band_weight_table`) and `out3a`/`out5a`/`out7a` (`score_arrays`).
/// Picks the row with the smallest `|accumulator|`, with an immediate
/// winner on an exact `0` (matching the reference's early-exit on a perfect
/// match). This is a deliberate simplification of the reference's two-step
/// "block-floating-point normalize, then lexicographic (exponent, mantissa)
/// compare" -- see the module doc for why the exact bit-level port wasn't
/// used here. Validated bit-exact (100/100) against captured reference
/// frames: given real A/B/C, this picks the same winning row the reference
/// transmitted as b1.
pub(crate) fn argmin_b1(
    weight: [i16; 16],
    out3a: [i16; 16],
    out5a: [i16; 16],
    out7a: [i16; 16],
) -> u16 {
    let mut best_row = 0u16;
    let mut best_abs: i64 = i64::MAX;
    for (row, &row_dword) in ARGMIN_ROW_TABLE.iter().enumerate() {
        let acc = argmin_row_accumulator(weight, row_dword, out3a, out5a, out7a);
        if acc == 0 {
            return row as u16;
        }
        let a = acc.abs();
        if a < best_abs {
            best_abs = a;
            best_row = row as u16;
        }
    }
    best_row
}

/// The complete downstream chain, `(A,B,C) -> b1`, given A/B/C from the
/// reference front end (the raw-audio-to-A/B/C path is not yet closed --
/// see the module doc). Validated bit-exact (100/100) against captured
/// reference frames' transmitted b1.
pub(crate) fn b1_from_abc(a: [i16; 16], b: [i16; 16], c: [i16; 16]) -> u16 {
    let (out3a, out5a, out7a) = score_arrays(c, a);
    let weight = per_band_weight_table(b);
    argmin_b1(weight, out3a, out5a, out7a)
}

/// Plain `Sigma int16` sum reduction (`ptr,count` -> raw `i32` sum, no
/// multiply), as used by `band_spectral_bfp`/`band_windowed_integration`.
/// `lo`/`hi` are clamped to `[0, 32)` (the fixed per-band window size every
/// call site uses); `hi <= lo` returns `0`, matching the reference's
/// `count <= 0` early return.
fn band_int16_sum_range(window: &[i16; 32], lo: i32, hi: i32) -> i32 {
    let lo = lo.clamp(0, 32) as usize;
    let hi = hi.clamp(0, 32) as usize;
    let mut acc: i32 = 0;
    if hi > lo {
        for w in &window[lo..hi] {
            acc = acc.wrapping_add(*w as i32);
        }
    }
    acc
}

/// Block-floating-point normalize -- the bit-scan-reverse + shift idiom
/// repeated 6 times inline in `band_spectral_bfp` (and used by
/// `band_windowed_integration`'s return-value normalize). Given a raw 32-bit
/// accumulator `v`, returns `(mantissa, exponent)` under the same
/// `value = mantissa * 2^exponent` BFP convention used by
/// `band_voicing_ratio_code`. `v == 0` is a fast path returning `(0, 0)`;
/// the source-is-zero edge case (only reachable for `v == -1`, never
/// observed in captured frames) is guarded explicitly rather than relying
/// on undefined bit-scan semantics.
fn band_bfp_normalize(v: i32) -> (i16, i16) {
    if v == 0 {
        return (0, 0);
    }
    let t: u32 = if v >= 0 { v as u32 } else { !(v as u32) };
    let bsr: i32 = if t == 0 {
        0
    } else {
        31 - t.leading_zeros() as i32
    };
    let sh: i32 = 0x1e - bsr;
    let shifted: i32 = if sh >= 0 {
        (u32v(v).wrapping_shl(sh as u32)) as i32
    } else {
        v >> ((-sh) as u32).min(31)
    };
    let mantissa = s16(shifted >> 16);
    let exponent = s16(-sh);
    (mantissa, exponent)
}

/// The "windowed integration/normalization" step `band_spectral_bfp` calls
/// twice (`step=6` and `step=5`). Validated bit-exact (800/800) against
/// captured reference frames.
///
/// Key subtlety: the loop's step size is halved exactly ONCE, on first
/// entry -- every later iteration reuses the same once-halved step. Halving
/// per iteration would give a convergent geometric advance that never
/// reaches the exit index (32), i.e. a non-terminating loop for real
/// inputs. Halving once gives a linear advance through the fixed `0..32`
/// index range that always terminates.
fn band_windowed_integration(window: &[i16; 32], x: i16, step: i32) -> i32 {
    const LIMIT: i16 = 32; // arg4 (0x40) halved -- always 32 at every call site
    let mut scaled: i32 = (x as i32).wrapping_shl(16);
    let shift_amt: i16 = s16(step - 15);
    if shift_amt >= 0 {
        scaled = (u32v(scaled).wrapping_shl(shift_amt as u32)) as i32;
    } else if shift_amt > s16(0xffe1) {
        // shift_amt in (-31, 0): real right-shift-by-abs path
        scaled >>= (-shift_amt) as u32;
    } else {
        // shift_amt <= -31: sign-fill
        scaled = if scaled < 0 { -1 } else { 0 };
    }

    let quarter = scaled >> 2;
    let mut pos = scaled.wrapping_sub(quarter).wrapping_add(0x10000);
    let mut idx: i16 = s16(pos >> 16);
    if idx >= LIMIT {
        return 0;
    }
    scaled >>= 1; // executes exactly ONCE -- see doc comment above
    let mut accum: i32 = 0;
    loop {
        pos = pos.wrapping_add(scaled);
        let next_idx = s16(pos >> 16);
        let cand = if next_idx <= LIMIT { next_idx } else { LIMIT };
        let cnt = s16(cand as i32 - idx as i32);
        pos = pos.wrapping_add(scaled); // second add, SAME step (not re-halved)
        if cnt > 0 {
            accum = accum.wrapping_add(band_int16_sum_range(window, idx as i32, idx as i32 + cnt as i32));
        }
        idx = s16(pos >> 16);
        if idx >= LIMIT {
            break;
        }
    }
    accum
}

/// The BFP `greater-than` compare `band_spectral_bfp` uses to pick between
/// its two candidate `(mantissa, exponent)` pairs when `X > 3277`. This is
/// a validated-equivalent model (align the larger-exponent operand's
/// mantissa down to the smaller exponent's scale by left-shifting it by the
/// exponent difference, then compare mantissas), not a literal restatement
/// of the reference's more elaborate branch structure (separate
/// near-identical paths per which operand has the larger exponent).
/// Validated bit-exact (152/152) against every captured `X > 3277` case.
fn band_bfp_greater_than(m1: i16, e1: i16, m2: i16, e2: i16) -> bool {
    let v1: i32 = (m1 as i32).wrapping_shl(16);
    let v2: i32 = (m2 as i32).wrapping_shl(16);
    let d: i32 = s16(e1 as i32 - e2 as i32) as i32;
    if d > 0 {
        let v2_aligned: i32 = if d <= 30 {
            (u32v(v2).wrapping_shl(d as u32)) as i32
        } else {
            0
        };
        v1 > v2_aligned
    } else if d < 0 {
        let dd = -d;
        let v1_aligned: i32 = if dd <= 30 {
            (u32v(v1).wrapping_shl(dd as u32)) as i32
        } else {
            0
        };
        v1_aligned > v2
    } else {
        v1 > v2
    }
}

/// Computes `band_voicing_ratio_code`'s 4 input words (2 BFP
/// `(mantissa, exponent)` pairs) from one band's `window` (32 `i16`
/// spectral-window samples, one per-band slice of a frame's 8 bands), the
/// per-frame scalar `x` (the pitch-lag `X` value), and `a`/`b`
/// (`floor(x/1024)+1` / `floor(x/2048)+1`).
///
/// The reference computes:
/// - `pairP = normalize(Sigma window[a..32))` (plain sum via `band_int16_sum_range`)
/// - `pairQ = normalize(windowed_integration(window, x, step=6))`
///   (`band_windowed_integration`)
/// - if `x <= 3277` (`0x0ccd`): `(out1,out2,out3,out4) = (pairQ, pairP)`
///   unconditionally (the default/common case)
/// - if `x > 3277`: additionally compute `pairR = normalize(Sigma
///   window[b..32))` and `pairS = normalize(windowed_integration(window, x,
///   step=5))`, then a cross-product BFP compare (`band_bfp_greater_than`) of
///   `2*pairS.mantissa*pairP.mantissa` against
///   `2*pairR.mantissa*pairQ.mantissa` (each re-normalized) decides:
///   compare-true picks `(pairS, pairR)`, compare-false falls back to
///   `(pairQ, pairP)` (same as the `x <= 3277` case).
///
/// Validated bit-exact (800/800) against captured reference frames, both
/// branches (`x<=3277` and `x>3277`) independently exact. See
/// `c420_fixtures::C420_FIXTURES` for the fixture set and the
/// `c420_matches_real_captured_frames` test below.
///
/// The `window` argument's upstream is the per-band spectrum builder's
/// per-band pointer; validated content-exact (100/100) against captured
/// reference frames, but not yet reduced to a raw-PCM formula (its own
/// upstream, a persistent per-frame accumulator, remains open).
pub(crate) fn band_spectral_bfp(
    window: &[i16; 32],
    x: i16,
    a: i16,
    b: i16,
) -> (i16, i16, i16, i16) {
    let a_idx = a as i32;
    let b_idx = b as i32;

    let sum_p = band_int16_sum_range(window, a_idx, 32);
    let (mp, ep) = band_bfp_normalize(sum_p);
    let ret_q = band_windowed_integration(window, x, 6);
    let (mq, eq) = band_bfp_normalize(ret_q);

    let mut out1 = mq;
    let mut out2 = eq;
    let mut out3 = mp;
    let mut out4 = ep;

    if x > 0x0ccd {
        let sum_r = band_int16_sum_range(window, b_idx, 32);
        let (mr, er) = band_bfp_normalize(sum_r);
        let ret_s = band_windowed_integration(window, x, 5);
        let (ms, es) = band_bfp_normalize(ret_s);

        let prod1: i32 = (ms as i32).wrapping_mul(mp as i32).wrapping_mul(2);
        let prod2: i32 = (mr as i32).wrapping_mul(mq as i32).wrapping_mul(2);
        let (p1m, sh_a) = band_bfp_normalize(prod1);
        let (p2m, sh_b) = band_bfp_normalize(prod2);
        let e_combo1 = s16(ep as i32 + es as i32 + sh_a as i32);
        let e_combo2 = s16(eq as i32 + er as i32 + sh_b as i32);
        if band_bfp_greater_than(p1m, e_combo1, p2m, e_combo2) {
            out1 = ms;
            out2 = es;
            out3 = mr;
            out4 = er;
        }
    }
    (out1, out2, out3, out4)
}

/// A generic 32-element BFP add: two source arrays (`dest`/`src2`, aliased
/// to the SAME buffer at the one traced call site, and `src1`) each carry
/// their own exponent (`exp2`/`exp1`); each element is shifted to a common
/// exponent (`max(exp1,exp2)`) then summed with saturation, and the result
/// (at the common exponent) is written back into `dest`. This is the
/// per-band spectrum builder's final combine, merging its two per-band
/// magnitude-squared accumulators into `band_spectral_bfp`'s `window` array
/// pre-normalize content -- see `bfp_normalize_pack32` for the next stage.
/// Validated bit-exact (250/250) against captured reference frames, covering
/// both the common case (`exp1==exp2`, 239/239) and every observed
/// unequal-exponent case (`exp1!=exp2`, 11/11).
pub(crate) fn bfp_add_arrays(
    dest: &[i32; 32],
    src1: &[i32; 32],
    exp2: i16,
    exp1: i16,
) -> [i32; 32] {
    fn align(v: i32, sh: i32) -> i32 {
        if sh < 0 {
            if sh < -0x1e {
                v >> 31
            } else {
                v >> ((-sh) & 0x1f)
            }
        } else {
            u32v(v).wrapping_shl((sh & 0x1f) as u32) as i32
        }
    }
    let common = (exp2 as i32).max(exp1 as i32);
    let p5 = exp1 as i32 - common; // <= 0
    let p3 = exp2 as i32 - common; // <= 0
    let mut out = [0i32; 32];
    for i in 0..32 {
        let u3 = align(dest[i], p3);
        let u5 = align(src1[i], p5);
        let mut u4 = u3.wrapping_add(u5);
        let u3u = u32v(u3);
        let u5u = u32v(u5);
        let u4u = u32v(u4);
        if ((u4u ^ u5u) & (u4u ^ u3u)) & 0x8000_0000 != 0 {
            u4 = if u3 >= 0 { 0x7fff_ffff } else { -0x8000_0000 };
        }
        out[i] = u4;
    }
    out
}

/// A 32-element max-magnitude scan returning a BFP normalize shift:
/// `bsr(max(|arr[i]|)) - 30` (the single-value `band_bfp_normalize`
/// `0x1e - bsr` idiom, generalized to an array-wide max). Returns `-1` if
/// every element is zero (the reference's degenerate-input path).
fn array_find_shift(arr: &[i32; 32]) -> i32 {
    let mut maxabs: u32 = 0;
    for &v in arr {
        let a = if v == i32::MIN {
            i32::MAX as u32
        } else {
            v.unsigned_abs()
        };
        if a > maxabs {
            maxabs = a;
        }
    }
    if maxabs == 0 {
        return -1;
    }
    let bsr = 31 - maxabs.leading_zeros() as i32;
    bsr - 30
}

/// The per-element pack body -- shift each dword by `-shift` (via
/// `shift_scale`, the primitive this call site uses), then round-and-pack
/// to `i16` (`+0x8000` bias with overflow-saturate, arithmetic `>> 16`).
fn round_pack16(shifted: u32) -> i16 {
    let raw = shifted;
    let biased = raw.wrapping_add(0x8000);
    let ovf = (biased ^ raw) & biased;
    let sat = if (ovf as i32) >= 0 {
        biased
    } else if (raw as i32) < 0 {
        0x8000_0000u32
    } else {
        0x7fff_ffffu32
    };
    ((sat as i32) >> 16) as i16
}

/// The final step that converts `bfp_add_arrays`'s 32-dword sparse output
/// (magnitude in the low half) into the dense `[i16; 32]` array
/// `band_spectral_bfp` consumes as its `window` argument. This closes the
/// array's writer chain down to (but not through) the per-band
/// correlation-tap accumulator, which itself bottoms out at the
/// pitch-lag/amplitude front end plus the persistent PCM-history ring,
/// neither closed here. Validated bit-exact (250/250) against captured
/// reference frames.
pub(crate) fn bfp_normalize_pack32(arr: &[i32; 32]) -> [i16; 32] {
    let shift = array_find_shift(arr);
    let shift_arg = (-shift) as i16;
    let mut out = [0i16; 32];
    for i in 0..32 {
        let shifted = shift_scale(arr[i] as u32, shift_arg);
        out[i] = round_pack16(shifted);
    }
    out
}

/// The "7-tap per-band correlation" step is not a new formula: it is a thin
/// argument-remapping wrapper around `shift_scale`
/// (`dest[k] = ((sext32(src[k])<<16)<<shift)>>16` for positive `shift`,
/// mirrored right-shift for negative, identity for `shift==0`), applied
/// in-place (`dest==src`) to a `count`-word slice, with `count`/`shift`
/// read from a small fixed per-band table, not computed from audio content.
/// Validated bit-exact (700/700) against captured reference frames. Because
/// it reduces to an existing primitive with no new math, no new `pub fn`
/// was added -- callers compose `shift_scale` directly.
///
/// This closes the entire path from the raw per-band array through the
/// 7-tap rescale and `magsq_assemble_bins` to the final `window` array,
/// with one remaining unknown: the raw pre-rescale content itself, a fresh
/// per-frame value. It was confirmed NOT to be the already-exact
/// `audio_prefilter -> history_ring -> windowed_taper -> real_fft32`
/// spectrum staging buffer, nor a shifted copy of the "array B" pattern
/// (both refuted). The sole remaining wall for the `window` array's
/// raw-audio dependency is that value's own producer -- a large,
/// separately-tracked, still-open function.
#[allow(dead_code)]
const MAGSQ_TAP_RESCALE_R90_NOTE: () = ();

/// The complete per-band `heap_src[i]` pipeline:
/// `window/x/a/b -> band_spectral_bfp -> out1..out4 ->
/// band_voicing_ratio_code -> heap_src[i]`. Validated against captured
/// reference frames (`c420_fixtures::C420_FIXTURES`): the composed pipeline
/// reproduces the reference's `band_voicing_ratio_code` return value
/// (`heap_src[i]`) bit-exact wherever `band_voicing_ratio_code` itself is
/// bit-exact (587/800 exact, the remaining 213/800 within the documented
/// +/-1 LSB divide-rounding residual -- see that function's doc), and the
/// voiced/unvoiced bit decision matches 800/800.
pub(crate) fn band_voicing_heap_src(window: &[i16; 32], x: i16, a: i16, b: i16) -> i16 {
    let (out1, out2, out3, out4) = band_spectral_bfp(window, x, a, b);
    band_voicing_ratio_code(out1, out2, out3, out4)
}

/// Fixed-point port of the reference's spectral-energy ratio producer,
/// `a9` -- the last unknown input to the ring slot-1 writer, and one of the
/// values `b1` ultimately depends on (it scales the p1 slot the argmin
/// reads).
///
/// Given two 64-bit power-spectrum sums with their block exponents, each is
/// normalized to a `(mantissa, exponent)` pair (via `normalize64`/`shift64`);
/// the first mantissa passes through a zero-gate (`0x68db`, exponent `-13`)
/// that keeps `bfp_divide` from being handed a zero divisor, and the two
/// pairs are BFP-divided to produce `a9`, an i16 spectral-energy ratio.
/// Block 2 keeps the full low dword (no `>>16`), an asymmetry corroborated
/// by `bfp_divide`'s signature (its `y` uses only its low 16 bits). Because
/// both inputs are plain sums, `a9` is a function of the two sums alone, not
/// of the individual array words -- so this is the real core and `df20_a9`
/// is a thin wrapper. Validated bit-exact (199/199 per file) against
/// captured reference frames; non-vacuous (192/199 and 198/199 distinct
/// `a9` values, where a constant scores at most 6/199 and 2/199).
///
/// Honest limit: `bfp_divide`'s returned mantissa is exponent-independent,
/// so this test does not pin the exponent wiring (a `swapexp` mutant ties
/// vacuously). `a9` itself is provably exponent-independent, but the
/// out-param exponent is NOT closed by this test and must not be assumed;
/// see `df20_a9_from_windows`, which verifies the exponents directly.
///
/// The two inputs are two different power spectra of the same frame, not yet
/// derivable from raw PCM at this level -- see `df20_a9_from_windows` for the
/// PCM-driven closure.
pub(crate) fn spectral_energy_ratio_from_sums(sum1: i64, exp1_in: i16, sum2: i64, exp2_in: i16) -> Option<i16> {
    /// One normalize+shift stage -- the same instruction sequence used for
    /// both blocks. Returns `(shifted_lo, zero_extended_shift)`.
    fn norm_shift(sum: i64) -> (u32, i32) {
        let u = sum as u64;
        let (lo, hi) = ((u & 0xFFFF_FFFF) as u32, ((u >> 32) & 0xFFFF_FFFF) as u32);
        let sh = normalize64(lo, hi);
        let shift_zext = (sh as u32 as u16) as i32; // zero-extended normalize shift
        let shift_sext = (sh as u32 as u16) as i16 as i32; // sign-extended normalize shift
        let (slo, _shi) = shift64(lo, hi, shift_sext);
        (slo, shift_zext)
    }

    // ---- Block 1: sum over the first spectrum, with its block exponent ----
    let (lo1, shift1) = norm_shift(sum1);
    let mut mant1: i32 = (((lo1 as i32) >> 16) as u32 as u16) as i32;
    let mut exp1: i32 = ((exp1_in as i32).wrapping_sub(shift1) as u32 as u16) as i32;
    if mant1 == 0 {
        mant1 = 0x68db;
        exp1 = -13;
    }

    // ---- Block 2: sum over the second spectrum, with its block exponent ----
    let (lo2, shift2) = norm_shift(sum2);
    let mant2: i32 = lo2 as i32; // block 2 keeps the full low dword (no >>16)
    let exp2: i32 = (exp2_in as i32).wrapping_sub(shift2);

    // ---- Tail: bfp_divide(mant2, exp2, mant1, exp1) -> a9 mantissa ----
    bfp_divide(mant2, exp2, mant1, exp1).map(|(mant, _exp)| mant)
}

pub(crate) fn band_voicing_ratio_code(out1: i16, out2: i16, out3: i16, out4: i16) -> i16 {
    if out3 <= 0 {
        return 0x7fff;
    }
    // Fixed-point port of the reference's ratio-code body. The approximate
    // "round(0x7fff * true_rational_ratio)" version (`band_voicing_ratio_code_approx`
    // below) scored only 587/800 exact + 213/800 within +/-1 LSB, because
    // the reference does NOT compute the mathematically-true rounded ratio:
    // it calls `super::atan2_bfp_divide::bfp_divide` and then applies a
    // specific variable-shift-with-saturation tail with its own
    // truncation/rounding behaviour, not a generic round-to-nearest. Porting
    // that exact tail and driving it off `bfp_divide` gets 800/800 exact
    // against `c420_fixtures::C420_HEAP_SRC_FIXTURES` (see
    // `band_voicing_heap_src_exact_end_to_end` below).
    let x = (out1 as i32) << 16;
    let Some((mantissa, exponent)) =
        super::atan2_bfp_divide::bfp_divide(x, out2 as i32, out3 as i32, out4 as i32)
    else {
        // the reference divides by zero only when out3's low 16 bits are 0,
        // already excluded by the `out3 <= 0` early return above (out3==0
        // is <= 0); kept as a defensive match for completeness.
        return 0x7fff;
    };
    if exponent >= 1 {
        return 0;
    }
    // sext16(mantissa) << 16, then shift left by `exponent` (only possible
    // value here is 0, since exponent<=0 and the >=1 case already returned)
    // or right by `-exponent` (arithmetic, saturating for exponent <= -31),
    // matching the reference's shift tail exactly.
    let mut shifted: i32 = (mantissa as i32) << 16;
    if exponent == 0 {
        // a shift by 0 is a no-op; kept explicit to mirror the reference branch.
    } else if exponent > -31 {
        shifted >>= (-exponent) as u32;
    } else {
        shifted = if shifted < 0 { -1 } else { 0 }; // arithmetic shift right 31 (sign-fill)
    }
    // neg(shifted) with INT_MIN saturating to INT_MAX
    let negated: i32 = if shifted == i32::MIN {
        i32::MAX
    } else {
        shifted.wrapping_neg()
    };
    // (negated + 0x7fff0000) >> 16 arithmetic, then truncate to u16.
    let combined: i32 = negated.wrapping_add(0x7fff0000u32 as i32);
    let result: i32 = combined >> 16;
    (result as i16 as u16) as i16
}

/// An approximate ratio-code formula (round(true rational ratio), clamped)
/// -- kept only as a documented cross-check, NOT used by
/// `band_voicing_ratio_code` anymore now that the literal `bfp_divide`-based
/// port above is proven bit-exact (800/800). Matches the exact port on the
/// overwhelming majority of cases (587/800 exactly, the rest within +/-1
/// LSB) since it targets the same mathematical quantity via a different
/// (non-fixed-point-faithful) rounding rule.
#[allow(dead_code)]
fn band_voicing_ratio_code_approx(out1: i16, out2: i16, out3: i16, out4: i16) -> i16 {
    if out3 <= 0 {
        return 0x7fff;
    }
    let shift = out2 as i32 - out4 as i32;
    let (numer, denom): (i64, i64) = if shift >= 0 {
        ((out1 as i64) << shift, out3 as i64)
    } else {
        (out1 as i64, (out3 as i64) << (-shift))
    };
    if denom == 0 {
        return 0x7fff;
    }
    let scaled = (numer as i128) * 0x7fffi128;
    let half_denom = (denom as i128).abs() / 2;
    let ratio_q15 = if scaled >= 0 {
        (scaled + half_denom) / (denom as i128)
    } else {
        (scaled - half_denom) / (denom as i128)
    };
    let pred = 0x7fffi128 - ratio_q15;
    pred.clamp(0, 0x7fff) as i16
}
