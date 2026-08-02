//! The reference vocoder's array-transform routine ([`array_transform_with_mode`], reached
//! via [`fft_bfp_transform`] and [`loudness_array_transform`]) -- the array-transform half
//! of the encoder's `outer` (loudness) mechanism.
//!
//! ## Two call sites, two array-mutation behaviours
//!
//! The same routine serves the voicing and loudness call sites, and they differ
//! in how far the ARRAY-mutating group loop runs. This is the single easiest
//! thing to get wrong here.
//!
//! * **Voicing site** ([`fft_bfp_transform`], `multi_stage=true`, `arg2=5`,
//!   `arg3=7`): the group loop fires on **every** stage -- 63 butterfly-combine
//!   hits per call, `63 = 32+16+8+4+2+1`, the all-6-stage radix-2 pattern.
//! * **Loudness site** ([`loudness_array_transform`], `multi_stage=false`): the group loop
//!   fires on **stage 1 only** -- exactly 32 group calls per top-level call,
//!   with `step_val` always 128, never halving. Verified across 522 top-level
//!   calls on two recordings with zero exceptions, from silence through loud
//!   voiced speech. Running the multi-stage formula here reproduces the real
//!   POST array on ~2% of calls and the scalar `ret` on ~31%.
//!
//! Do not unify the two modes in either direction. The routine has no
//! `si_test`/magnitude early-exit that could explain the split from the inside:
//! its only conditional jumps are the odd/even estimate skip, the
//! butterfly-variant dispatch, the `n_groups <= 0` guard (`n_groups` is purely
//! `n_pairs`-derived), and the two `stage >= arg3` exits. So for a fixed `arg3`
//! the group loop is structurally deterministic and the difference cannot be a
//! window-value branch inside this function. The branch that selects it has not
//! been traced.
//!
//! ## Stage geometry (when the group loop fires)
//!
//! Per stage `s` (1-indexed), `span = 2^s` (2/4/8/16/32/64); the number of outer
//! butterfly-combine CALLS is `n_groups = n_pairs/(2*span)` (32/16/8/4/2/1 for
//! `n_pairs=128`); each call does `count2 = span/2` INTERNAL butterfly
//! iterations (the butterfly-combine's own `((span-1)>>1)+1` trip-count). Within
//! one call, internal iteration `k` (`0..count2`) operates on `pk = p0+4*k`,
//! `qk = q0+4*k` (`q0 = p0+2*span`), with twiddle index `2*step_val*k`, where
//! `step_val` is the passed 4th arg, `128>>(stage-1)`.
//!
//! Each top-level call does exactly ONE `bitrev_permute` plus ONE twiddle-group
//! pass at a given size, not several at growing sizes -- so the
//! `bitrev_permute` call-count sequence is not expected to follow a monotonic
//! doubling loop.
//!
//! ## The two butterfly variants share the SAME position mapping
//!
//! They are not an "asymmetric mix" of twiddle pointers: both read and write the
//! same 8 relative positions per iteration, with the same twiddle-index
//! progression (`tw_idx`, `tw_idx+step` per iteration, advancing by `2*step`
//! between iterations). The only difference is that one variant computes
//! `2*(sum)` via a genuine 32-bit doubling add before the final `>>16`, and the
//! other does a plain `>>16` on the un-doubled sum.
//!
//! `2*x >> 16 == x >> 15` only while `x` stays clear of the `i32` overflow
//! boundary on the doubling step, and captured data crosses it: 13 of 300
//! captures need the literal doubling semantics. Keep the wrapping arithmetic --
//! collapsing it to a plain `>>15` is wrong.
//!
//! Variant dispatch is by `si_test`: `<= 1` selects the plain-`>>16` variant,
//! `> 1` the doubling variant.
//!
//! ## Exponent bookkeeping
//!
//! The per-stage `si_test`/`need_extra_scale` SCALAR bookkeeping runs every
//! stage in **both** modes -- it never touches the array. It carries `carry_si`
//! on odd stages and re-scans via a fresh `si_estimate` call on even stages;
//! that re-scan's `n` argument is the CONSTANT `n_pairs=128` and does not shrink
//! per stage.
//!
//! `si_estimate`, `bitrev_permute` and `prescale` are each confirmed exact in
//! isolation.

// The DLL's `cos(2*pi*k/512)` Q15 twiddle table -- extracted generously
// (1024 entries, not just the nominal 512-entry period) because the real
// code's own twiddle-pointer advance (`TWIDSTEP = 2*step`, `step` up to 128)
// can walk PAST the table's own nominal end into whatever follows it in
// `.rdata` -- confirmed this is real DLL behavior (not a bug to work
// around), so the extra entries are the REAL adjacent `.rdata` bytes, not
// padding.
include!("loudness_transform_tables.rs");

/// The DLL's `.rdata` `cos(2*pi*k/512)` Q15 table (plus the real adjacent
/// `.rdata` words past the nominal 512-entry period).
///
/// Exposed because the DECODE side needs the identical table: the OLA
/// oscillator's cos-interp leaf (see [`crate::dec::ola_osc`]) indexes this
/// same table. Re-verified against the DLL's own `.rdata` bytes:
/// **1024/1024 words identical**.
pub(crate) fn cos_twiddle_table() -> &'static [i16; 1024] {
    &T1
}

use crate::fixops::i16r::{i16t, s32};

/// Peak-power block-floating-point exponent probe over `n_pairs` complex
/// `(re, im)` `i16` samples starting at `arr[base..]`. Returns a small signed
/// "headroom" count: high when the signal is quiet, dropping toward/below 1
/// as it gets loud.
fn si_estimate(arr: &[i16], base: usize, n_pairs: usize) -> i16 {
    let mut peak: i32 = 0x0100_0000;
    for i in 0..n_pairs {
        let re = arr[base + 2 * i] as i32;
        let im = arr[base + 2 * i + 1] as i32;
        let p = re.wrapping_mul(re).wrapping_add(im.wrapping_mul(im));
        if p > peak {
            peak = p;
        }
    }
    if peak == 0 {
        return -1; // dead code in practice (peak is seeded >= 0x1000000)
    }
    let positive: u32 = if peak >= 0 {
        peak as u32
    } else {
        !(peak as u32)
    };
    let bit = 31 - positive.leading_zeros(); // bsr
    let hdr = (30i32 - bit as i32) as u16; // movzx ax
    i16t(hdr as i32 - 1)
}

/// Table-driven in-place pairwise swap/reorder pass. Selects one of 5
/// size-bucket tables via the `BITREV_BUCKET_LUT` byte lookup table
/// (`idx = n_pairs - 8`, clamped/defaulted to bucket 4 outside `[0,248]`),
/// then walks two DRIFTING cursors (cumulative, never reset per swap) through
/// `count` `(d1,d2)` element deltas, swapping the 2-element group at each
/// landing pair of positions.
fn bitrev_permute(arr: &mut [i16], base: usize, n_pairs: usize) {
    let idx = n_pairs as i32 - 8;
    let bucket = if !(0..=248).contains(&idx) {
        4u8
    } else {
        BITREV_BUCKET_LUT[idx as usize]
    };
    let table: &[(i16, i16)] = match bucket {
        0 => &BITREV_SWAP_DELTAS_0,
        1 => &BITREV_SWAP_DELTAS_1,
        2 => &BITREV_SWAP_DELTAS_2,
        3 => &BITREV_SWAP_DELTAS_3,
        _ => &BITREV_SWAP_DELTAS_4,
    };
    let mut pos1 = base as isize;
    let mut pos2 = base as isize;
    for &(idx1, idx2) in table {
        pos1 += idx1 as isize;
        pos2 += idx2 as isize;
        arr.swap(pos1 as usize, pos2 as usize);
        pos1 += 1;
        pos2 += 1;
        arr.swap(pos1 as usize, pos2 as usize);
    }
}

/// One-shot butterfly pre-scale (the trivial, twiddle=1 "stage 0" combine of
/// a radix-2 DIT FFT run over already bit-reversed data), applying
/// `edi_init`'s own right-shift
/// (2/1/0 bits) to the whole array via a plain (non-twiddle) sum/
/// difference combine, processing groups of 4 consecutive elements
/// `(a,b,c,d) = (4i, 4i+1, 4i+2, 4i+3)`:
/// `new[a]=(a+c)>>s`, `new[c]=(a-c)>>s`, `new[b]=(b+d)>>s`, `new[d]=(b-d)>>s`.
/// `edi_init=0` (`s=0`) uses NO shift at all (confirmed directly against
/// raw bytes: this branch has zero `sar` instructions anywhere in its
/// body, unlike the other two branches).
fn prescale(arr: &mut [i16], base: usize, n_pairs: usize, edi_init: i32) {
    if n_pairs == 0 {
        return;
    }
    let shift = match edi_init {
        -2 => 2,
        -1 => 1,
        0 => 0,
        _ => unreachable!("edi_init must be -2, -1, or 0"),
    };
    let count = ((n_pairs as i32 - 1) >> 1) + 1;
    for i in 0..count as usize {
        let e = base + 4 * i;
        let b = e + 1;
        let c = e + 2;
        let d = e + 3;
        let a_ = arr[e] as i32;
        let c_ = arr[c] as i32;
        let b_ = arr[b] as i32;
        let d_ = arr[d] as i32;
        arr[e] = i16t((a_ + c_) >> shift);
        arr[c] = i16t((a_ - c_) >> shift);
        arr[b] = i16t((b_ + d_) >> shift);
        arr[d] = i16t((b_ - d_) >> shift);
    }
}

#[inline]
fn cos_at(off: i32) -> i32 {
    T1[off as usize] as i32
}
#[inline]
fn sin_at(off: i32) -> i32 {
    T2[off as usize] as i32
}

/// Shared 8-write twiddle-butterfly combine used by both butterfly variants
/// (they are the SAME position mapping / same combine shape -- see module
/// doc). `p`/`q` are ELEMENT indices into `arr` with `q == p+4` always (one
/// "group" is 2 complex pairs at `p` -- the `a1ptr`-side pair -- and 2 more
/// at `q` -- the `arg2`-side pair, `arg2 == a1ptr + 2*span` elements in the
/// caller's own addressing, always `+4` within one group here since
/// `bitrev_permute`+`prescale` already collapsed the group-internal
/// addressing to this fixed shape). `tw_idx0` is the starting ELEMENT offset
/// into `T1`/`T2` for this call (always 0 for the real, single stage-1 call
/// this project has ever observed exercising this function, but threaded
/// through generally). `step` is the per-iteration twiddle-index advance (128
/// for the one real stage this function is ever called at).
/// `double_before_shift` selects the plain `>>16` variant (`false`) vs the
/// `2*sum >> 16` variant (`true`), with a REAL (not assumed-equivalent-to-
/// >>15) 32-bit wraparound on the doubling step -- this literal distinction
/// > >     matters for large sums.
fn twiddle_pair(
    arr: &mut [i16],
    p: usize,
    q: usize,
    tw_idx0: i32,
    step: i32,
    double_before_shift: bool,
) {
    let val0 = s32(arr[p]);
    let y1 = s32(arr[p + 1]);
    let val0pp = s32(arr[p + 2]);
    let val1pp = s32(arr[p + 3]);
    let xim = s32(arr[q]); // "X_m1"
    let xre = s32(arr[q + 1]); // "X0"
    let x1 = s32(arr[q + 2]);
    let x2 = s32(arr[q + 3]);

    let tc = cos_at(tw_idx0);
    let ts = sin_at(tw_idx0);
    let tc2 = cos_at(tw_idx0 + step);
    let ts2 = sin_at(tw_idx0 + step);

    let d1 = tc.wrapping_mul(xim).wrapping_sub(ts.wrapping_mul(xre));
    let (w1, w2) = combine_pair(val0, d1, double_before_shift);
    let d2 = tc.wrapping_mul(xre).wrapping_add(ts.wrapping_mul(xim));
    let (w3, w4) = combine_pair(y1, d2, double_before_shift);

    let d3 = tc2.wrapping_mul(x1).wrapping_sub(ts2.wrapping_mul(x2));
    let (w5, w6) = combine_pair(val0pp, d3, double_before_shift);
    let d4 = tc2.wrapping_mul(x2).wrapping_add(ts2.wrapping_mul(x1));
    let (w7, w8) = combine_pair(val1pp, d4, double_before_shift);

    arr[p] = w1;
    arr[q] = w2;
    arr[p + 1] = w3;
    arr[q + 1] = w4;
    arr[p + 2] = w5;
    arr[q + 2] = w6;
    arr[p + 3] = w7;
    arr[q + 3] = w8;
}

/// One `(sum, diff)` butterfly write pair, matching the real DLL's own
/// per-instruction 32-bit arithmetic exactly (`imul`/`add`/`sub`/`shl`/
/// `sar` each wrap to 32 bits on real hardware -- emulated here via
/// `wrapping_*` at every step rather than trusting Rust's default
/// overflow-checked/panicking `i32` arithmetic or an unbounded
/// wider-precision shortcut).
#[inline]
fn combine_pair(v: i32, d: i32, double_before_shift: bool) -> (i16, i16) {
    let vs = v.wrapping_shl(15);
    let sum = vs.wrapping_add(d);
    let diff = vs.wrapping_sub(d);
    if double_before_shift {
        let sum2 = sum.wrapping_mul(2);
        let diff2 = diff.wrapping_mul(2);
        (i16t(sum2 >> 16), i16t(diff2 >> 16))
    } else {
        (i16t(sum >> 16), i16t(diff >> 16))
    }
}

/// Port of the DLL's array-transform routine `(a1ptr, arg2, arg3)`. `arr`
/// must be at least `2 * 2^arg3` `i16` words long starting at `base`; mutated
/// in place exactly like the real function (the array's final content beyond
/// what `bitrev_permute`+`prescale`+the one twiddle-group pass touch is
/// whatever `prescale` left it at -- matches real DLL behavior, this project
/// never found a caller that reads `fft_bfp_transform`'s array output back
/// out, only its scalar return). Returns the (already `i16`-truncated) scalar
/// the real DLL returns.
pub(crate) fn fft_bfp_transform(arr: &mut [i16], base: usize, arg2: i16, arg3: u32) -> i16 {
    array_transform_with_mode(arr, base, arg2, arg3, true)
}

pub(crate) fn loudness_array_transform(arr: &mut [i16], base: usize, arg2: i16, arg3: u32) -> i16 {
    array_transform_with_mode(arr, base, arg2, arg3, false)
}

/// The array-transform routine `(a1ptr, arg2, arg3)`, parameterized by
/// `multi_stage` -- see [`fft_bfp_transform`] (voicing call site,
/// `multi_stage=true`) and [`loudness_array_transform`] (loudness call site,
/// `multi_stage=false`).
///
/// `multi_stage` selects whether the ARRAY-mutating group loop keeps firing past
/// stage 1; the `si_test`/`carry_si`/`need_extra_scale` SCALAR bookkeeping runs
/// every stage either way. See the module doc for why the two modes exist and
/// why neither may be collapsed into the other.
///
/// For the loudness site the observed stage-1-vs-6 split is POSITIONAL
/// (end-of-stream flush, last 10 frames), not derivable from the window values
/// -- see [`super::loudness_fixed::d890_arg2_from_window`].
fn array_transform_with_mode(arr: &mut [i16], base: usize, arg2: i16, arg3: u32, multi_stage: bool) -> i16 {
    let n_pairs = 1usize << arg3;
    let si0 = si_estimate(arr, base, n_pairs);
    let edi_init: i32 = if si0 < 0 {
        -2
    } else if si0 < 2 {
        -1
    } else {
        0
    };

    bitrev_permute(arr, base, n_pairs);
    prescale(arr, base, n_pairs, edi_init);

    if arg3 <= 1 {
        return i16t(arg2 as i32 - edi_init);
    }

    let mut acc: i32 = edi_init;
    // "current si estimate carried across stages" -- starts at si0-2 (a
    // ONE-TIME adjustment applied before the loop), then gets an
    // UNCONDITIONAL further -2 at the tail of every stage (even the ones
    // that also got a fresh re-estimate that stage).
    let mut carry_si: i32 = si0 as i32 - 2;

    let mut stage: u32 = 1;
    // The array-mutating group loop is gated to stage 1 when `multi_stage` is
    // false (the loudness call site) and fires every stage when it is true (the
    // voicing call site). See this function's doc and the module doc.
    let mut span: usize = 2;
    let mut step_val: i32 = 128;

    while stage < arg3 {
        // Stages alternate: odd stage numbers reuse the carried estimate,
        // even stage numbers re-scan the WHOLE (by-now fully-transformed,
        // not just stage-1-transformed) array fresh via si_estimate again.
        let si_test: i32 = if stage % 2 == 1 {
            carry_si
        } else {
            let fresh = si_estimate(arr, base, n_pairs) as i32;
            carry_si = fresh;
            fresh
        };
        let need_extra_scale = si_test <= 1;
        if need_extra_scale {
            acc -= 1;
        }

        // `n_groups = n_pairs / (2*span)` outer groups (32 at span=2/stage=1),
        // each covering `2*span` elements on the `p` side and `2*span` on the
        // `q` side; each group internally does `count2 = span/2` `twiddle_pair`
        // calls, stepping `p`/`q` by 4 elements and the twiddle index by
        // `2*step_val` per internal iteration.
        if stage == 1 || multi_stage {
            let n_groups = n_pairs / (2 * span);
            let count2 = span / 2;
            for g in 0..n_groups {
                let p0 = base + g * 4 * span;
                let q0 = p0 + 2 * span;
                for k in 0..count2 {
                    let pk = p0 + 4 * k;
                    let qk = q0 + 4 * k;
                    let tw_idx0 = 2 * step_val * (k as i32);
                    twiddle_pair(arr, pk, qk, tw_idx0, step_val, !need_extra_scale);
                }
            }
        }

        carry_si -= 2;
        span *= 2;
        step_val >>= 1;
        stage += 1;
    }

    i16t(arg2 as i32 - acc)
}
