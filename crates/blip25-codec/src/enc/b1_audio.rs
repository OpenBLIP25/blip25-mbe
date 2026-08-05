//! **b1 (voicing) FROM AUDIO** -- the `band_voicing_mask` / `refine_ring_p0` / `bfp_scale` ring state
//! machine, wired frame-serial.
//!
//! # Chain (per frame, ring carried across frames)
//! ```text
//!   PCM -> prefilter -> ab80/adf0/e530 -> scorevecs `svs` + `expo`
//!     |-> a3[8][32] + bandexp[8]   (accumulate_aligned combine + pack)
//!     |-> x = octave_halve_decide's raw_r622
//!   history_ring (lib, bit-exact) -> d890 two windows @ gap f-1 -> df20 -> a9
//!   band_voicing_mask(ring, exp_hist, a3, x, bandexp)          [shift + p0/p1 slot0 + mask]
//!   refine_ring_p0(ring, halved_x, wide_bins)               [conditional min-refine, see below]
//!   bfp_scale(p0 s0/s1/s2, p1 s0/s1/s2, a7, a8, a9)    [both rings' slot 1]
//!   a4 = compand(min(max(R0,R2),R1)) over p0 slots 0..4
//!   a6 = max(1, p1[8+k]) over p1 slots 1 AND 2 ; a5 = 0
//!   -> enc::voicing_vq::voicing_b1_vq -> b1
//! ```
//!
//! # Accuracy, audio-only
//! voiced **190/199**, mark **195/199**, against 68/199 and 41/199 for a
//! constant-b1 placeholder. Zeroing the audio collapses this to 67/199 and
//! 40/199 -- the decisive test that the chain is genuinely audio-driven, since a
//! placeholder is immune to it.
//!
//! # `C820` is OFF by default, and that is a measured choice
//!
//! Read this before re-attacking refine_ring_p0.
//!
//! `refine_ring_p0`'s `arg3` (a 129-i32 buffer from the d890) IS derived from audio
//! bit-exactly -- 151/151 rows, 19328/19328 words, with `MUT=zeroaudio` at 0/151
//! and sharp alignment (gapoff -1 = 151/151; 0 and +1 = ~250/16256). A transform
//! that omits the in-place normalize and the `arr[1]=0` store scores 3928/16256
//! and looks like a wrong source; it is not. See `cepstral_transform_wide`.
//!
//! With the correct arg3 wired in, `Gate` makes every bfp_scale ring intermediate
//! near-exact -- voiced p0 slot0 105 -> 197, p0 slot2 104 -> 196, p0 slot1 out
//! 94 -> 194, i.e. the ring state matches the reference capture. **And b1 gets
//! WORSE: voiced 190 -> 166** (mark unchanged at 195).
//!
//! The conclusion is not "arg3 is wrong". The shipped `Off` 190 is substantially
//! **error cancellation**: a wrong ring lands on the right VQ cell more often
//! than a right ring does. The real defect is DOWNSTREAM of bfp_scale, in
//! `pitch_predictor_from_ring` / `stable_counter_from_ring`: against the reference's captured a4/a6,
//! **a6 = 17/199 voiced and 11/199 mark**, versus a zeroed-audio floor of 3/199
//! and 1/199 -- essentially chance; a4 = 105/199 voiced against a floor of 50.
//! That is the b1 lever; refine_ring_p0 is not.
//!
//! `c820_gate_regresses_b1_until_a4_a6_are_fixed` pins this. When a4/a6 are
//! fixed and `Gate` overtakes `Off`, flip the default.
//!
//! # a4 -- and a4 alone with a5 -- is the whole b1 gap. a6 is worth ZERO.
//!
//! Substituting ground-truth a4 and a5 gives b1 **199/199** and whole-frame
//! **195 voiced / 197 mark**, exactly the b1-ground-truth ceiling: nothing but
//! a4 and a5 is missing. They fix DISJOINT frames -- a5 fixes voiced
//! [138,195,196,197], a4 fixes voiced [0,15,23,128,173]. Ground-truth a6 moves
//! b1 on **0/398 frames**. Do not spend time on a6.
//!
//! a4's error has TWO independent halves, separable by feeding the reference's
//! captured `in0` (the p0 plane, flat) into [`pitch_predictor_flat`], the same code
//! this chain runs:
//!   * **the p0 ring is wrong** -- 85/199 voiced vectors match the captured
//!     `in0`; and
//!   * **the formula is wrong even on perfect input** -- 145/199 voiced.
//!
//! ## The second half is a **GATE**, not a nonzero `num`
//!
//! A live capture of compand's arguments shows:
//!   * compand's shift `num` (the `count` operand of `vshift_scale`) is **0** on
//!     the default path; `numv = word<<16` (the `value` operand). The Rust roles
//!     are transposed in name but arithmetically equivalent for the common path.
//!   * The a4 producer **GATES the whole thing**: it tests
//!     `(hdr30 >> (band+3)) & 1` and, when clear, writes the band's 8 a4 words
//!     as **ZERO** without calling compand. `hdr30` = `[ctx+0x30]` = the vmeas
//!     `hdr30` column. **a4 band b (b in 0,1) is emitted iff bit(b+3) of hdr30
//!     is set; else it is 0.** Verified 199/199 on both files (compand-fired ==
//!     gate-bit-set): see [`pitch_predictor_gate`].
//!
//! Frames where `den=3, num=0` would give `0x7fff` but the reference emits 0 are
//! gated-off frames, not evidence of a shift: f0 in [0,15] has hdr30 bits 3,4
//! clear, so a4=0. Solving for a nonzero `num` here is a many-to-one trap --
//! it fits a huge shift to turn `0x7fff` into 0 on bands the reference never
//! computes. Do not re-derive it that way.
//!
//! With the true (captured) hdr30 and count=0, a4-from-in0 goes 145 -> 187
//! voiced / 180 -> 190 mark frame-exact, and feeding the gated a4 into
//! `voicing_b1_vq` gives b1 190 -> 194 voiced (fixes [0,15,128,173]) and
//! 195 -> 197 mark (fixes [0,76]), with zero frames broken -- no refine_ring_p0-style
//! error cancellation.
//!
//! ## The remaining blocker: hdr30 is a STATEFUL voicing-onset envelope
//!
//! hdr30 is a shift register: onset ramps `0,1,7,31,127,...`, 2 per-band voicing
//! bits shifted in per frame. a4 is emitted only in the voiced regions
//! ([16-117],[149-198] voiced) and zeroed during onset (0-15) and the inter-word
//! gap (118-148).
//!
//! A per-frame content threshold on `den` replicates the gate only 77%/95%
//! (`maxden>=4`), so it REGRESSES: the gate needs the multi-frame voicing state,
//! not a scalar test. Deriving hdr30 audio-only is the next lever; until then
//! [`pitch_predictor_gate`] is unwired, since there is no capture-fed hdr30 in the live path.
//!
//! # Two late writers exist -- do not assume otherwise
//! The VQ leaf runs late in the call graph, after the main p0/p1 writer. A second
//! gain writer rescales p1 slot1 (**costs b1 exactly zero** -- a6 is a VQ weight,
//! and a proportional rescale does not move the argmin), and a late p0
//! slot0/slot1 writer makes sparse per-band revisions (costs voiced 195->178).
//! Neither is modelled here.

#![allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::enc::block_exponent::block_exponent;
use crate::enc::history_ring::{GAP2_LEN, GAP2_LEN_WIDE};
use crate::enc::loudness_fixed::{assemble_bins_from_win, win_from_gap2};
use crate::enc::voicing_fixed::{
    a_writer_band, band_spectral_bfp, band_voicing_heap_src, band_voicing_ratio_code,
    bfp_add_arrays, bfp_normalize_pack32, spectral_energy_ratio_from_sums,
};
use crate::enc::voicing_vq::voicing_b1_vq;
use crate::enc::win_taper_wide::win_from_gap2_slot2;

/// Whether to run `refine_ring_p0`'s conditional min-refine of p0 slot 0.
///
/// `Off` is the shipping default and the configuration the 190/195 numbers were
/// measured in -- see the module header for why feeding refine_ring_p0 currently HURTS.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RingRefineMode {
    /// Never call refine_ring_p0. **The measured-best default.**
    #[default]
    Off,
    /// Call iff `x_halved > 0x1000` (the `[esi] != 0`
    /// arm of the real gate is the caller's own arg and is not modelled).
    Gate,
    /// Always call. Panics the port's own `count1 <= 55` assert on real audio --
    /// which is independent confirmation that the `> 0x1000` gate is real.
    Always,
}

/// Everything one frame of the machine produces. The bins score these fields
/// against their clean-15 keys; the encoder only needs `b1`.
#[derive(Clone, Debug)]
pub struct B1Frame {
    pub a3: [[i16; 32]; 8],
    pub bandexp: [i16; 8],
    /// band_voicing_mask's `arg2` (the pitch tracker's return). 199/199 voiced, 194/199 mark.
    pub x: i16,
    /// `df20`'s return. 199/199 both files.
    pub a9: i16,
    /// `band_voicing_mask`'s RETURN = the 8-bit band-voicing mask the DLL stores to
    /// `ctx+0x91a`. Bit k set iff band k's ratio code < 0x199a.
    /// The next frame reads it back and passes it to the pitch tracker
    /// as its `a11` -- see `b0_audio::push_pcm_frame_with_prev_mask`.
    pub mask: u16,
    pub a7: i16,
    pub a8: i16,
    /// p0 slot 0 as bfp_scale sees it (after band_voicing_mask, and refine_ring_p0 when it fires) = bfp_scale's `a1`.
    pub p0s0: [i16; 8],
    /// p1 slot 0 after band_voicing_mask = bfp_scale's `a4`.
    pub p1s0: [i16; 8],
    pub p0s2: [i16; 8],
    pub p1s2: [i16; 8],
    /// bfp_scale's `a2` out = p0 slot 1.
    pub o2: [i16; 8],
    /// bfp_scale's `a5` out = p1 slot 1.
    pub o5: [i16; 8],
    /// The VQ's compand measurement. NOTE the name collision:
    /// this is NOT band_voicing_mask's own `a4[8]` (that one is `bandexp`).
    pub a4: [i16; 16],
    pub a6: [i16; 16],
    /// **The voicing byte, from audio alone.**
    pub b1: u16,
    /// The p0 plane, flat (`p0_flat[s * 8 + k]` = ring slot `s`, element `k`), as it
    /// stands at the moment a4 is taken. This is the DLL's `in0` array (its leaf
    /// input, the `in0_00..in0_39` columns of the measurement capture), exposed
    /// so a4's error can be split into "wrong ring" vs "wrong formula".
    pub p0_flat: [i16; 40],
    /// The p1 plane, flat -- the DLL's `in1` array. Same purpose, for a6.
    pub p1_flat: [i16; 40],
    /// The f0 ring's two previous entries as the packer reads them (`ring[1]`,
    /// `ring[2]`): this frame's first analysis call's
    /// [`repair_gate_f0`](super::t_ring::repair_gate_f0) and the previous
    /// frame's second call's. The packer's tone branch quantises these two.
    pub f0_ring: (i16, i16),
}

// ============================================================================
// a5 (array A) FROM PCM -- the r54..r64 chain, assembled STANDALONE (audio-only,
// NO captured CSV/txt at runtime) and voicing-gated. See `enc::a5_assemble` for
// the driver/ring lib fns for the port.
// ============================================================================

/// gate tail: given windowed_complex_correlation's `(out0, out1, r)`, the x-arm self-corr
/// exponent `sc`, and the x-arm complex buffer, return `(ph0, ph1, exp_x)`.
fn a5_gate_tail(out0: i16, out1: i16, r: i16, sc: i32, cbuf: &[i16]) -> (i16, i16, i16) {
    use crate::enc::atan2_bfp_divide::bfp_divide;
    use crate::enc::band_decompress::{normalize64, shift64};
    use crate::enc::loudness_fixed::gamma_poly_bfp_eval;
    let c0 = (out0 as i32).max(-32767);
    let c1 = (out1 as i32).max(-32767);
    let z2: i64 = 2 * (c0 as i64) * (c0 as i64) + 2 * (c1 as i64) * (c1 as i64);
    let lo = (z2 as u64 & 0xffff_ffff) as u32;
    let hi = ((z2 as u64 >> 32) & 0xffff_ffff) as u32;
    let m = normalize64(lo, hi);
    let mag = shift64(lo, hi, m).0 as i32;
    let (seed, ph0, ph1);
    if mag <= 0 {
        seed = 0i32;
        ph0 = c0;
        ph1 = c1;
    } else {
        let exp_state = ((2 * (r as i32) - m) & 0xffff) as u16 as i16;
        let (pc, exp_state2) = gamma_poly_bfp_eval(mag, exp_state);
        let (acc, oute) =
            bfp_divide(0x4000_0000, 1, (pc >> 16) as i32, exp_state2 as i32).unwrap_or((0, 0));
        seed = (r as i32) + oute as i32;
        ph0 = ((2 * (acc as i32) * c0) >> 16) as i16 as i32;
        ph1 = ((2 * (acc as i32) * c1) >> 16) as i16 as i32;
    }
    let sv = (sc + seed + 1) & 0xffff;
    let (cos, sin) = (ph0, ph1);
    let mut out = [0i16; 254];
    for k in 0..127 {
        let xre = cbuf[2 * k] as i32;
        let xim = cbuf[2 * k + 1] as i32;
        out[2 * k] = ((cos * xre - sin * xim) >> 16) as i16;
        out[2 * k + 1] = ((xre * sin + cos * xim) >> 16) as i16;
    }
    let be = block_exponent(&out) as i32;
    let expx = (sv as u16 as i16 as i32 + be) as i16;
    (ph0 as i16, ph1 as i16, expx)
}

/// Opt-in switch (`BLIP25_TS_NEXT=1`, OFF by default) for the a5 analysis
/// window's spectral transform: the FULL multistage `fft_bfp_transform` on
/// every call rather than the stage-1-only `loudness_array_transform` that the
/// `FLUSH_LOOKAHEAD` rule in [`a5_pcm_arms`] otherwise selects.
///
/// Read out of the live Wave7k DLL by hooking the T/S producer
/// (`0x10314070`, the sole caller of the (T,S) delay line): at the unchanged
/// window start `s = (2*f + c) * 80 - 275`, the multistage
/// `(power[129], scale)` is bit-exact against the oracle's own arguments on
/// 236/236 `mark` calls and 294/294 `clean` calls; the stage-1-only transform
/// matches 0/236 and 0/294 (`scale` alone 70/236).
pub(crate) fn ts_next_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BLIP25_TS_NEXT")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Per-(frame,call) PCM arms + exp_y + x-arm self-corr exponent `sc`.
/// `pre` = the encoder's prefiltered stream (`prefiltered_log`), the same
/// signal the a5 window was validated against. `seq = 2*f + c`, window start
/// `seq*80 - 275`. Returns `(y[254], x[254], exp_y, sc)`.
fn a5_pcm_arms(pre: &[i16], f: usize, c: usize) -> Option<(Vec<i16>, Vec<i16>, i16, i16)> {
    use crate::enc::array_a_stage2::inverse_fft_butterfly_stage;
    use crate::enc::loudness_transform::{loudness_array_transform, fft_bfp_transform};
    use crate::enc::pq_builder::{
        assemble_x_arm, assemble_x_arm_with_exp, assemble_y_arm_with_exp,
    };
    let seq = 2 * f as i64 + c as i64;
    let s = seq * 80 - 275;
    if s < 0 || (s as usize + GAP2_LEN) > pre.len() {
        return None;
    }
    let mut g = [0i16; GAP2_LEN];
    for t in 0..GAP2_LEN {
        g[t] = pre[s as usize + t];
    }
    // d890 complex window (258 words, Nyquist-inclusive). The a5 x-arm (arg2) is
    // d890's WINDOWED FFT of the loudness window
    // (`loudness_fixed::d890_arg2_from_window`), retaining the Nyquist pair
    // [256]/[257] that `assemble_x_arm` reads.
    //
    // The fft_bfp_transform stage count k is an END-OF-INPUT FLUSH, not a
    // data-dependent branch: k=6 (full multistage, the voicing
    // `fft_bfp_transform`) on frames whose forward analysis context (window start
    // + GAP2_LEN + FLUSH_LOOKAHEAD) runs past the last input sample; k=1
    // (stage-1-only, `loudness_array_transform`) otherwise.
    //
    // Do not look for a window or FFT-intermediate feature that selects k. There
    // is none: the transform has no si_test or magnitude early-exit, the group
    // loop is structurally all-6-stages for arg3=7, and a full separation battery
    // over every window feature fails to separate the k=6 frames (only the flush
    // condition does). The flip point moves with input length and is identical
    // across two independent clips.
    //
    // FLUSH_LOOKAHEAD ~= the analysis forward context (~12 frames), calibrated to
    // reproduce the measured boundary on the 200-frame clips (first flush call =
    // frame 189) while generalizing to any input length.
    //
    // Direct DLL readout contradicts the whole flush rule: the oracle takes the
    // multistage transform on EVERY call. `ts_next_enabled` is that reading.
    const FLUSH_LOOKAHEAD: usize = 1876;
    let multi = ts_next_enabled() || (s as usize + GAP2_LEN + FLUSH_LOOKAHEAD) > pre.len();
    // The y-arm POWER + SCALE share the SAME per-frame fft_bfp_transform stage
    // count k as the x-arm: both are one d890 fft_bfp_transform+a1b0 pass over the
    // same loudness window. Pinning power/scale to a fixed k=6 while latching the
    // x-arm leaves the y-arm right only on the flush frames -- latch both.
    let (power, scale) = crate::enc::loudness_fixed::frame_power_and_scale(&g, multi);
    let mut w = win_from_gap2(&g);
    let a2 = block_exponent(&g).saturating_sub(7);
    let fft_ret = if multi {
        fft_bfp_transform(&mut w, 0, a2, 7)
    } else {
        loudness_array_transform(&mut w, 0, a2, 7)
    };
    let mut xwin = [0i16; 258];
    xwin[..256].copy_from_slice(&w[..256]);
    let _ = inverse_fft_butterfly_stage(&mut xwin, fft_ret, 8, 0, 1);
    let x = assemble_x_arm(&xwin, &power, scale);
    let (y, exp_y) = assemble_y_arm_with_exp(&power, scale);
    let (_xc, sc) = assemble_x_arm_with_exp(&xwin, &power, scale);
    Some((y, x, exp_y, sc))
}

/// Assemble `a5` (the 16-word array A, two 8-word `a_writer_band` halves) for
/// every frame. This is the standalone realization of the r54..r64 chain:
/// prefiltered audio -> arms -> windowed_complex_correlation (w from the lib's
/// own band_voicing_mask) -> gate -> `pq_driver` -> cross-frame T/S ring (with
/// the `repair_gate_f0` reference) -> `a_writer_band`.
///
/// `pre` is the caller's prefiltered stream and `fr` its front end, both
/// starting at the same sample as the frame index. The chain reads no raw PCM
/// and starts no filter of its own: every stage here is a block-floating-point
/// cascade over `pre`, and the prefilter is an integer IIR whose state error
/// does not decay to zero, so re-deriving `pre` from a window of raw samples
/// perturbs `a5` an unbounded distance into the window.
///
/// The raw `a_writer` output is nonzero on ~85 voiced / ~48 mark NON-lever
/// frames (the zeros-trap: the off-lever T/S ring is only lever-validated).
/// `a5` is emitted verbatim here; the caller applies whatever voicing gate it
/// wants. Returns one `[i16;16]` per frame (index 0..nframes) alongside the
/// per-call `repair_gate_f0` log (index 0..2*nframes).
fn a5_track_from_pre(pre: &[i16], fr: &[FrameIn], nframes: usize) -> (Vec<[i16; 16]>, Vec<i16>) {
    use crate::enc::a5_assemble::{pq_driver, RingState};
    use crate::enc::windowed_complex_correlation::windowed_complex_correlation;
    use crate::enc::t_ring::repair_gate_f0;
    use crate::enc::w_builder::w_builder;

    // Per-frame w_builder inputs (thresh8, exp_arg, table8) via the lib's own
    // band_voicing_mask ring -- NO captured band_voicing_mask CSV. table8 = the
    // FRESH heap_src this frame (`table8[N] == heap_src[N+1]`, r54 `w_builder`
    // doc).
    let mut win_in: Vec<([i16; 8], i16, [i16; 8])> = Vec::with_capacity(nframes);
    {
        let mut ring = Ring90 { w: [0i16; 90] };
        let mut exp_hist = [0i16; 4];
        for f in 0..nframes {
            let fi = &fr[f];
            let _mask = band_voicing_mask(&mut ring, &mut exp_hist, &fi.a3, fi.x, &fi.bandexp);
            let thresh8 = ring.slot(0, 0);
            let exp_arg = exp_hist[0];
            let x = fi.x;
            let x_hi: i32 = (x as i32) << 16;
            let v1 = ((((x_hi >> 10).wrapping_add(0x10000)) >> 16) & 0xffff) as u16 as i16;
            let v2 = ((((x_hi >> 11).wrapping_add(0x10000)) >> 16) & 0xffff) as u16 as i16;
            let mut table8 = [0i16; 8];
            for k in 0..8 {
                table8[k] = band_voicing_heap_src(&fi.a3[k], x, v1, v2);
            }
            win_in.push((thresh8, exp_arg, table8));
        }
    }

    // Per-call (oldgen, s_new) and the repair-gate reference f0, fully from PCM.
    let ncalls = nframes * 2;
    let mut og: Vec<[i16; 8]> = vec![[0; 8]; ncalls];
    let mut sn: Vec<i16> = vec![0; ncalls];
    let mut valid: Vec<bool> = vec![false; ncalls];
    let mut f0v: Vec<i16> = vec![0; ncalls];
    for f in 0..nframes {
        for c in 0..2 {
            let k = f * 2 + c;
            let Some((y, x, exp_y, sc)) = a5_pcm_arms(pre, f, c) else {
                continue;
            };
            // w with the one-call delay: c=1 uses this frame's band_voicing_mask; c=0 the
            // previous frame's.
            let g = if c == 0 {
                if f == 0 {
                    continue;
                }
                f - 1
            } else {
                f
            };
            let (th, ea, tb) = win_in[g];
            let w = w_builder(ea, th, tb);
            let (o0, o1, rr) = windowed_complex_correlation(
                &w[..127],
                &y[..254],
                &x[..254],
                127,
                exp_y as i32,
                sc as i32,
            );
            f0v[k] = repair_gate_f0(o0, o1, rr);
            let (ph0, ph1, expx) = a5_gate_tail(o0, o1, rr, sc as i32, &x[..254]);
            let (o, s) = pq_driver(&y[..254], &x[..254], ph0, ph1, expx, exp_y);
            og[k] = o;
            sn[k] = s;
            valid[k] = true;
        }
    }

    // Cross-frame T/S ring; repair gate = (prev-call f0 > 0) && (this-call f0 < 0).
    let mut rs = RingState::default();
    let mut a5_out: Vec<[i16; 16]> = vec![[0i16; 16]; nframes];
    for f in 0..nframes {
        for c in 0..2 {
            let k = f * 2 + c;
            let prev_f0 = if k == 0 { 0 } else { f0v[k - 1] };
            let gate = valid[k] && prev_f0 > 0 && f0v[k] < 0;
            rs.push_call(og[k], sn[k], gate);
            if c == 1 {
                let ((g1, sa), (g2, sc2)) = rs.a_writer_inputs();
                let p1 = a_writer_band(g1, sa, false);
                let p2 = a_writer_band(g2, sc2, false);
                let mut a = [0i16; 16];
                a[..8].copy_from_slice(&p1);
                a[8..].copy_from_slice(&p2);
                a5_out[f] = a;
            }
        }
    }
    (a5_out, f0v)
}

/// Source of the r34 encoder gap2 logs `precompute_from_logs` needs.
/// `FromPcm` runs the encoder internally; `Supplied` reuses logs a caller
/// already captured from its own encoder pass, so no encoder run is repeated.
#[derive(Clone, Copy)]
enum EncLogs<'a> {
    FromPcm,
    Supplied {
        gap0: &'a [[i16; GAP2_LEN]],
        gap1: &'a [[i16; GAP2_LEN]],
        gap2w: &'a [[i16; GAP2_LEN_WIDE]],
    },
}

/// Run the whole machine and return one `B1Frame` per frame. Takes an
/// already-prefiltered signal alongside the raw pcm (the history ring wants
/// the raw samples). Frame-serial and causal, so asking for more frames never
/// changes an earlier frame's answer.
pub(crate) fn b1_track(
    pref: &[i16],
    raw_pcm: &[i16],
    nframes: usize,
    c820mode: RingRefineMode,
) -> Vec<B1Frame> {
    b1_track_core(pref, raw_pcm, nframes, c820mode, EncLogs::FromPcm, None)
}

/// As [`b1_track`], but with a caller-supplied hdr30 VAD (one `i32` per frame).
/// A streaming caller tracks the VAD's adaptive noise floor persistently — the
/// single slow-converging state a bounded analysis window can't reproduce — so
/// a short window plus this override is byte-exact for the finalised frames.
pub fn b1_track_hdr30(
    pref: &[i16],
    raw_pcm: &[i16],
    nframes: usize,
    c820mode: RingRefineMode,
    hdr30: &[i32],
) -> Vec<B1Frame> {
    b1_track_core(
        pref,
        raw_pcm,
        nframes,
        c820mode,
        EncLogs::FromPcm,
        Some(hdr30),
    )
}

/// As [`b1_track`], but reuses the r34 encoder gap2 logs a caller already
/// captured from its own encoder pass (`gap2_mid_log` / `gap2_slot1_log` /
/// `gap2_slot2_log`) instead of running the encoder again internally
/// (`precompute_from_ring`). The emitted-bits encoder pass and the b1 chain
/// then share ONE analysis pass. The r33 and r34 encoders share the same
/// analysis / prefilter / gap-log writers and differ only in the final
/// bit-packing, so r33-captured logs are byte-identical here.
///
/// `pre` is the same pass's `prefiltered_log`. The encoder prefilters each
/// frame in two 80-sample halves off one persistent state, so for a whole
/// number of frames it is the same stream as prefiltering `raw_pcm` in one
/// call; it is taken here as a checked redundancy rather than a second input.
#[allow(clippy::too_many_arguments)]
pub fn b1_track_from_logs(
    gap0: &[[i16; GAP2_LEN]],
    gap1: &[[i16; GAP2_LEN]],
    gap2w: &[[i16; GAP2_LEN_WIDE]],
    pre: &[i16],
    raw_pcm: &[i16],
    nframes: usize,
    c820mode: RingRefineMode,
) -> Vec<B1Frame> {
    let (pref, _) = crate::enc::audio_prefilter::prefilter(
        &crate::enc::audio_prefilter::PrefilterState::default(),
        raw_pcm,
    );
    debug_assert!(
        {
            let n = pre.len().min(pref.len());
            pre[..n] == pref[..n]
        },
        "prefiltered_log disagrees with the prefilter over raw_pcm"
    );
    b1_track_core(
        &pref,
        raw_pcm,
        nframes,
        c820mode,
        EncLogs::Supplied { gap0, gap1, gap2w },
        None,
    )
}

fn b1_track_core(
    pref: &[i16],
    raw_pcm: &[i16],
    nframes: usize,
    c820mode: RingRefineMode,
    logs: EncLogs<'_>,
    hdr30_override: Option<&[i32]>,
) -> Vec<B1Frame> {
    let fr = front_end(pref, nframes);
    let (a9v, wide) = match logs {
        EncLogs::FromPcm => precompute_from_ring(raw_pcm, nframes),
        EncLogs::Supplied { gap0, gap1, gap2w } => precompute_from_logs(gap0, gap1, gap2w, nframes),
    };
    let mut ring = Ring90 { w: [0i16; 90] };
    let mut exp_hist = [0i16; 4];
    let mut out: Vec<B1Frame> = Vec::with_capacity(nframes);

    // r51: the a4 GATE reads the `hdr30` onset envelope. hdr30 is a 16-bit
    // shift register `hdr30 = (hdr30<<2) | fill`, fill = (Vcur | Vprev<<1) a 2-bit voicing
    // edge code. r51 PROVED the driving flag V is NOT the voicing byte (GT b1 fed in fails
    // 175/199) -- it is an ENERGY VAD (b1==16 unvoiced-byte frames stay envelope-VOICED mid
    // word). A file-independent adaptive noise-floor VAD reproduces the gate bits 396/395 of
    // 398 and lifts b1 +3 voiced / +2 mark. Default ON (r51 measured +3 voiced / +2 mark
    // ALL-9, zero regressions, robust across MARGIN [2,5]).
    // A caller streaming frame-by-frame supplies a persistently-tracked hdr30
    // VAD (its adaptive noise floor is the one slow-converging state that a
    // bounded analysis window cannot reproduce); otherwise derive it here.
    let hdr30_deriv: Vec<i32> = if let Some(h) = hdr30_override {
        h.to_vec()
    } else {
        derive_hdr30_vad(raw_pcm, nframes)
    };

    // a5 (array A), the r64 chain, gated by the hdr30 VAD (a5=[0;16] off voiced
    // frames) to suppress the off-lever zeros-trap false-nonzeros. See
    // `a5_track_from_pre`. It reads the caller's own prefiltered stream — never
    // a re-derived one — truncated to the whole frames the front end covers.
    // r93: a5 is ON BY DEFAULT. It is nonzero only on the ~9 voiced VQ levers +
    // the first 3 startup-warmup frames (identical 0/1/2 on both independent
    // clips = a real encoder startup transient, not a fit). Zeroing that warmup
    // (A5_WARMUP) reproduces the +22-voiced ceiling (voiced 171->193, mark
    // 195->197).
    let a5_pre_len = ((raw_pcm.len() / crate::synth::FRAME_SAMPLES) * crate::synth::FRAME_SAMPLES)
        .min(pref.len());
    let (a5_all, f0v): (Vec<[i16; 16]>, Vec<i16>) =
        a5_track_from_pre(&pref[..a5_pre_len], &fr, nframes);
    /// Startup-transient frames where the analysis chain is not yet warmed up
    /// and the real encoder emits a5=[0;16] (r40 f0/1/2 = 0; the chain is
    /// spuriously nonzero there). Zeroed to match. Principled (input-independent
    /// startup).
    const A5_WARMUP: usize = 3;
    let a5_vad: Vec<i32> = if a5_all.is_empty() {
        Vec::new()
    } else if hdr30_deriv.is_empty() {
        derive_hdr30_vad(raw_pcm, nframes)
    } else {
        hdr30_deriv.clone()
    };
    // The 3 onset levers [138,143,145] are a capture-run disagreement (r64):
    // unreachable, so `onset` mode zeros them to avoid a wrong-a5 regression.
    const A5_ONSET_ZERO: [usize; 3] = [138, 143, 145];

    for f in 0..nframes {
        let fi = &fr[f];
        let mask = band_voicing_mask(&mut ring, &mut exp_hist, &fi.a3, fi.x, &fi.bandexp);
        // refine_ring_p0 gate: halve the pitch word if > 0x3759,
        // then call iff (x > 0x1000) || ([esi] != 0). `[esi]` is the gate's own arg
        // and is NOT modelled -- hence RingRefineMode.
        let xh: i16 = if fi.x > 0x3759 { fi.x >> 1 } else { fi.x };
        let fire = match c820mode {
            RingRefineMode::Off => false,
            RingRefineMode::Always => true,
            RingRefineMode::Gate => xh > 0x1000,
        };
        if fire {
            c820m::refine_ring_p0(&mut ring.w, xh, &wide[f]);
        }
        let a9 = a9v[f].unwrap_or(0);
        let (p0s0, p0s2) = (ring.slot(0, 0), ring.slot(0, 2));
        let (p1s0, p1s2) = (ring.slot(1, 0), ring.slot(1, 2));
        let (mut o2, mut o5) = ([0i16; 8], [0i16; 8]);
        let (a7, a8) = (exp_hist[0], exp_hist[1]);
        bfp_scale(&p0s0, &mut o2, &p0s2, &p1s0, &mut o5, &p1s2, a7, a8, a9);
        for k in 0..8 {
            ring.set_p0(1, k, o2[k]);
            ring.set_p1(1, k, o5[k]);
        }
        let mut a4 = pitch_predictor_from_ring(&ring);
        if !hdr30_deriv.is_empty() {
            a4 = pitch_predictor_gate(a4, hdr30_deriv[f]);
        }
        let a6 = stable_counter_from_ring(&ring);
        let a5: [i16; 16] = if a5_all.is_empty() || f < A5_WARMUP {
            // warmup startup transient -> a5=0 (matches r40 f0/1/2 = 0).
            [0i16; 16]
        } else {
            a5_all[f]
        };
        let _ = (&a5_vad, &A5_ONSET_ZERO); // retained for reference; unused in the default path
        let b1 = voicing_b1_vq(&a4, &a5, &a6);
        let p0_flat: [i16; 40] = std::array::from_fn(|j| ring.p0(j / 8, j % 8));
        let p1_flat: [i16; 40] = std::array::from_fn(|j| ring.p1(j / 8, j % 8));
        out.push(B1Frame {
            a3: fi.a3,
            bandexp: fi.bandexp,
            x: fi.x,
            a9,
            a7,
            a8,
            p0s0,
            p1s0,
            p0s2,
            p1s2,
            o2,
            o5,
            a4,
            a6,
            b1,
            mask,
            p0_flat,
            p1_flat,
            f0_ring: (
                f0v.get(2 * f).copied().unwrap_or(0),
                if f == 0 {
                    0
                } else {
                    f0v.get(2 * f - 1).copied().unwrap_or(0)
                },
            ),
        });
    }
    out
}

use crate::fixops::acc64::{
    bf_shift, bshift, bsr, i16s, m32, norm_sh, norm_shift, s32, sar, sat_add, shl,
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
        let mut ecx = num;
        let mut eax = den as i64 & 0xffff_ffff;
        let esi = m32((i16s(eax) as i64) ^ (num as i64));
        if ecx < 0 {
            ecx = if ecx == i32::MIN { 0x7fff_ffff } else { -ecx };
        }
        let ax = i16s(eax);
        if ax < 0 {
            eax = if (eax & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-ax) & 0xffff) as i64
            };
        }
        let edi = i16s(eax & 0xffff);
        let mut res;
        if ecx < edi {
            res = 0;
        } else if sar(ecx, 16) >= edi {
            res = 0x7fff;
        } else {
            res = sar((ecx as i64 / edi as i64) as i32, 1);
        }
        if s32(esi) < 0 {
            res = -res;
        }
        res
    }
    fn blockfloat_divide(arg1: i32, arg2: i32, arg3: i32, arg4: i32) -> (i32, i32) {
        let mut ecx = arg1;
        let mut eax = arg3 as i64 & 0xffff_ffff;
        let edi = m32((i16s(eax) as i64) ^ (arg1 as i64));
        if ecx < 0 {
            ecx = if ecx == i32::MIN { 0x7fff_ffff } else { -ecx };
        }
        let ax = i16s(eax);
        if ax < 0 {
            eax = if (eax & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-ax) & 0xffff) as i64
            };
        }
        let mut esi = arg2;
        let ebx = i16s(eax & 0xffff);
        if shl(ebx, 16) <= ecx {
            ecx = sar(ecx, 1);
            esi = esi.wrapping_add(1);
        }
        let mut q = if ebx != 0 {
            (ecx as i64 / ebx as i64) as i32
        } else {
            0
        };
        q = sar(q, 1) & 0xffff;
        if s32(edi) < 0 {
            q = (-i16s(q as i64)) & 0xffff;
        }
        let edx = shl(i16s(q as i64), 16);
        let shift = if edx == 0 {
            0
        } else {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let edx2 = sar(shl(edx, shift), 16) & 0xffff;
        if i16s(edx2 as i64) == 0 {
            return (0, 0);
        }
        esi = esi.wrapping_sub(shift & 0xffff).wrapping_sub(arg4);
        (edx2, i16s(esi as i64))
    }
    fn blockfloat_add(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> (i32, i32) {
        let di = i16s(arg0 as i64);
        let ebx = arg3;
        if di == 0 {
            if i16s(arg2 as i64) != 0 {
                return (i16s(arg2 as i64) & 0xffff, i16s(arg3 as i64));
            }
            return (0, 0);
        }
        let edx = arg1;
        let esi = i16s(ebx as i64);
        let ecx = i16s(edx as i64);
        if esi - ecx >= 0x20 {
            if i16s(arg2 as i64) != 0 {
                return (i16s(arg2 as i64) & 0xffff, i16s(arg3 as i64));
            }
            return (0, 0);
        }
        let bp = i16s(arg2 as i64);
        if bp == 0 {
            return (i16s(arg0 as i64) & 0xffff, i16s(arg1 as i64));
        }
        if i16s(edx as i64) - i16s(ebx as i64) >= 0x20 {
            return (i16s(arg0 as i64) & 0xffff, i16s(arg1 as i64));
        }
        let eax = if i16s(arg1 as i64) > i16s(arg3 as i64) {
            i16s(arg1 as i64) + 1
        } else {
            i16s(arg3 as i64) + 1
        };
        let esil = eax & 0xffff;
        let v1 = bshift(
            shl(i16s(arg0 as i64), 16),
            i16s(arg1 as i64) - i16s(esil as i64),
        );
        let v2 = bshift(
            shl(i16s(arg2 as i64), 16),
            i16s(arg3 as i64) - i16s(esil as i64),
        );
        let edxr = v1.wrapping_add(v2);
        let shift = if edxr == 0 {
            0
        } else {
            let t = if edxr >= 0 { edxr } else { s32(!(edxr as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let dx = sar(shl(edxr, shift), 16) & 0xffff;
        if i16s(dx as i64) == 0 {
            return (0, 0);
        }
        (dx, i16s(esil as i64) - i16s((shift & 0xffff) as i64))
    }
    fn parabolic_interp(k: i32, score: &[i32], count: i32) -> i32 {
        let ki = i16s(k as i64);
        let edx0 = shl(ki, 16);
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
        let esi = shl(sl, 16);
        let eaxc = shl(sc_, 16);
        let edi = shl(sr, 16);
        let mut ecx = sar(esi, 1).wrapping_sub(eaxc);
        ecx = ecx.wrapping_add(sar(edi, 1));
        ecx = sar(ecx, 16);
        if i16s((ecx & 0xffff) as i64) >= 0 {
            return edx0;
        }
        let esi2 = sar(esi, 2);
        let edi2 = sar(edi, 2);
        let num = esi2.wrapping_sub(edi2);
        let den = ecx & 0xffff;
        let q = frac_divide_sat(num, den);
        let mut eax = i16s((q & 0xffff) as i64).wrapping_add(shl(cc, 15));
        eax = eax.wrapping_add(eax);
        eax
    }
    fn norm_edx(edx: i32) -> (i32, i32) {
        let ecx = if edx == 0 {
            0
        } else {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let expc = i16s(((0xf - ecx) & 0xffff) as i64);
        let mant = i16s((sar(shl(edx, ecx), 16) & 0xffff) as i64);
        (mant, expc)
    }
    fn denorm(mant: i32, expc: i32) -> i32 {
        let sh = i16s(((expc - 6) & 0xffff) as i64);
        let edx = shl(i16s(mant as i64), 16);
        let edx = sar(bshift(edx, sh), 16);
        i16s((edx & 0xffff) as i64)
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
        let eax = i16s(it as i64);
        let mut edi = s32(shl(i16s(eax as i64), 16) as i64 + 0x8000);
        let edx = if edi != 0 {
            let t = if edi >= 0 { edi } else { s32(!(edi as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let normb = (0xf - edx) & 0xffff;
        edi = shl(edi, edx);
        let ebx = normb;
        let edxv = exp - 15 + i16s(ebx as i64);
        let mant_s = i16s(mant as i64);
        let mut esi = i16s(sar(edi, 16) as i64);
        esi = esi.wrapping_mul(mant_s);
        esi = esi.wrapping_add(esi);
        esi = sar(bshift(esi, edxv), 16);
        let si = i16s((esi & 0xffff) as i64);
        mem.w16(l + 2, si);
        if si >= i16s(count as i64) {
            return 1;
        }
        let ebx2 = i16s((-i16s(ebx as i64)) as i64);
        let e = s32(-(bshift(0x4000_0000u32 as i32, ebx2) as i64));
        edi = edi.wrapping_add(2i32.wrapping_mul(e));
        let mut a = i16s(sar(edi, 16) as i64).wrapping_mul(mant_s);
        a = a.wrapping_add(a);
        a = sar(bshift(a, edxv), 16).wrapping_add(1);
        mem.w16(l, a & 0xffff);
        let e2 = bshift(0x2000_0000, ebx2);
        edi = edi.wrapping_add(e2);
        let mut a2 = i16s(sar(edi, 16) as i64).wrapping_mul(mant_s);
        a2 = a2.wrapping_add(a2);
        a2 = sar(bshift(a2, edxv), 16).wrapping_add(1);
        mem.w16(l + 4, a2 & 0xffff);
        let esi2 = bshift(0x4000_0000u32 as i32, ebx2);
        let mut a3 = i16s(sar(esi2.wrapping_add(edi), 16) as i64).wrapping_mul(mant_s);
        a3 = a3.wrapping_add(a3);
        a3 = sar(bshift(a3, edxv), 16);
        mem.w16(l + 6, a3 & 0xffff);
        if i16s((a3 & 0xffff) as i64) < mem.r16s(l + 4) {
            return 1;
        }
        0
    }
    fn band_ratio(sum2: i32, sum1: i32, mem: &mut Mem, expslot: i32) -> i32 {
        let esi0 = sum2;
        let edi = sum1.wrapping_sub(esi0);
        if edi <= 0 {
            let eax = if esi0 < 0 { 0 } else { 0x6400 };
            mem.w16(expslot, 7);
            return eax;
        }
        let shift = if esi0 != 0 {
            let t = if esi0 >= 0 { esi0 } else { s32(!(esi0 as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let edx = (-shift) & 0xffff;
        let be = bsr(edi);
        let esi = shl(esi0, shift);
        let ebx = 0x1e - be;
        let ecx = ebx & 0xffff;
        let edi_s = sar(shl(edi, ecx), 16);
        let negcx = i16s((-i16s(ecx as i64)) as i64);
        let (mant, ex) = blockfloat_divide(esi, i16s(edx as i64), edi_s, negcx);
        let mut eax = mant & 0xffff;
        let dx = i16s(ex as i64);
        if dx > 7 {
            eax = if esi < 0 { 0 } else { 0x6400 };
            mem.w16(expslot, 7);
            return eax;
        }
        if dx == 7 && (eax & 0xffff) >= 0x6400 {
            eax = 0x6400;
        }
        if esi < 0 {
            eax = 0;
        }
        mem.w16(expslot, dx & 0xffff);
        eax
    }
    fn refine_pitch_estimate(mem: &mut Mem, s: i32, it: i32) {
        let itv = i16s(it as i64);
        let mut edx = s32(mem.r16s(s + 4) as i64 * itv as i64);
        edx = edx.wrapping_add(edx);
        let sh1 = if edx != 0 {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut ax = (mem.r16u(s + 6) - (sh1 & 0xffff)) & 0xffff;
        edx = shl(edx, sh1);
        ax = (ax + 0xf) & 0xffff;
        edx = sar(edx, 16);
        let edi = i16s(edx as i64);
        let ebp = ax & 0xffff;
        let mut edx = s32(mem.r16s(s + 8) as i64 * edi as i64);
        edx = edx.wrapping_add(edx);
        let sh2 = if edx != 0 {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut ax = (mem.r16u(s + 0xa) - (sh2 & 0xffff)) & 0xffff;
        edx = shl(edx, sh2);
        ax = (ax + ebp) & 0xffff;
        edx = sar(edx, 16);
        let e2 = ax & 0xffff;
        let (mant, ex) = blockfloat_add(mem.r16u(s + 0xc), mem.r16u(s + 0xe), edx, e2);
        mem.w16(s + 0xc, mant);
        mem.w16(s + 0xe, ex);
        let mut edi = s32(edi as i64 * itv as i64);
        edi = edi.wrapping_add(edi);
        let sh3 = if edi != 0 {
            let t = if edi >= 0 { edi } else { s32(!(edi as i64)) };
            (0x1e - bsr(t)) & 0xffff
        } else {
            0
        };
        let mut ebp2 = (ebp - (sh3 & 0xffff)) & 0xffff;
        ebp2 = (ebp2 + 0xf) & 0xffff;
        edi = sar(shl(edi, sh3), 16);
        let (mant2, ex2) = blockfloat_add(mem.r16u(s + 0x10), mem.r16u(s + 0x12), edi, ebp2);
        mem.w16(s + 0x10, mant2);
        mem.w16(s + 0x12, ex2);
        let di = i16s(mant2 as i64);
        if di > 0 {
            let (r, rex) = blockfloat_divide(
                shl(mem.r16s(s + 0xc), 16),
                mem.r16u(s + 0xe),
                i16s(di as i64),
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
        let edx = parabolic_interp(k, score, high);
        let (mant, expc) = norm_edx(edx);
        mem.w16(s, mant);
        mem.w16(s + 2, expc);
        mem.w16(s + 8, mant);
        mem.w16(s + 0xa, expc);
        mem.w32(s + 0xc, 0);
        mem.w32(s + 0x10, 0);
        let mut ebx = 1i32;
        let mut esi = high;
        loop {
            let (a1m, a2e) = (mem.r16u(s), mem.r16u(s + 2));
            let r = harmonic_bounds(&mut mem, l, a1m, a2e, ebx, esi);
            if r == 1 {
                break;
            }
            let loc6 = mem.r16u(l + 6);
            let loc4 = mem.r16u(l + 4);
            let count2 = (loc6 - loc4 + 1) & 0xffff;
            if i16s(ebx as i64) > 1 {
                let idx = argmax16(score, i16s(loc4 as i64), i16s(count2 as i64));
                let k2 = idx + i16s(loc4 as i64);
                let e2 = parabolic_interp(k2, score, esi);
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
            refine_pitch_estimate(&mut mem, s, ebx);
            esi = high;
            ebx += 1;
            if i16s(ebx as i64) >= 0x10 {
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
        let mut ecx = arg1;
        let mut eax = arg3 as i64 & 0xffff_ffff;
        let edi = m32((i16s(eax) as i64) ^ (arg1 as i64));
        if ecx < 0 {
            ecx = if ecx == i32::MIN { 0x7fff_ffff } else { -ecx };
        }
        let ax = i16s(eax);
        if ax < 0 {
            eax = if (eax & 0xffff) == 0x8000 {
                0x7fff
            } else {
                ((-ax) & 0xffff) as i64
            };
        }
        let mut esi = arg2;
        let ebx = i16s(eax & 0xffff);
        if shl(ebx, 16) <= ecx {
            ecx = sar(ecx, 1);
            esi = esi.wrapping_add(1);
        }
        let mut q = if ebx != 0 {
            (ecx as i64 / ebx as i64) as i32
        } else {
            0
        };
        q = sar(q, 1) & 0xffff;
        if s32(edi) < 0 {
            q = (-i16s(q as i64)) & 0xffff;
        }
        let edx = shl(i16s(q as i64), 16);
        let shift = if edx == 0 {
            0
        } else {
            let t = if edx >= 0 { edx } else { s32(!(edx as i64)) };
            (0x1e - bsr(t)) & 0xffff
        };
        let edx2 = sar(shl(edx, shift), 16) & 0xffff;
        if i16s(edx2 as i64) == 0 {
            return (0, 0);
        }
        esi = esi.wrapping_sub(shift & 0xffff).wrapping_sub(arg4);
        (edx2, i16s(esi as i64))
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
        let ecx = 0x1f - shw;
        let mask = shl(-1, ecx);
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
        let ebx = 0x20i32;
        let sh = div - 15;
        let mut esi = bshift(shl(i16s(a1 as i64), 16), sh);
        let eax = sar(esi, 2);
        let mut edi = esi.wrapping_sub(eax).wrapping_add(0x10000);
        let mut ebp = 0i32;
        let mut ecx = sar(edi, 16) & 0xffff;
        if i16s(ecx as i64) >= ebx {
            return ebp;
        }
        esi = sar(esi, 1);
        loop {
            edi = edi.wrapping_add(esi);
            let mut nextlag = i16s(sar(edi, 16) as i64);
            if nextlag > ebx {
                nextlag = ebx;
            }
            let count = nextlag - i16s(ecx as i64);
            edi = edi.wrapping_add(esi);
            ebp = ebp.wrapping_add(sum16(score, i16s(ecx as i64), count));
            ecx = sar(edi, 16) & 0xffff;
            if i16s(ecx as i64) >= ebx {
                break;
            }
        }
        ebp
    }
    fn periodicity_ratio(combmant: i32, blockexp: i32, score: &[i32], a1: i32, arg4: i32) -> i32 {
        let ecx0 = arg4 - 16;
        let edx = bshift(shl(i16s(a1 as i64), 16), ecx0);
        let lag = sar(edx.wrapping_add(0x10000), 16) & 0xffff;
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
        let _aax = mant & 0xffff;
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
        let esi = i16s(a1 as i64);
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
        if esi <= 0xccc {
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
            let ecx5 = (0x1e - bsr5) & 0xffff;
            mant5 = i16s(sar(shl(comb5, ecx5), 16) as i64);
            blockexp5 = (-i16s(ecx5 as i64)) & 0xffff;
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
                    let cx = i16s((sar((i16s(a1 as i64) << 16).wrapping_sub(c), 16)) as i64);
                    if cx < 0 {
                        try_a7_halve = true;
                    } else {
                        let ax = i16s(
                            (sar(
                                (i16s(a1 as i64).wrapping_mul(0xe8bau32 as i32))
                                    .wrapping_sub(i16s(a9 as i64) << 16),
                                16,
                            )) as i64,
                        );
                        if ax <= 0 {
                            skip_continuity_halve = true;
                        } else {
                            try_a7_halve = true;
                        }
                    }
                }
                if !skip_continuity_halve {
                    if try_a7_halve {
                        let eax = i16s(a7 as i64).wrapping_mul(0xd99au32 as i32);
                        let cx = i16s((sar((i16s(a1 as i64) << 16).wrapping_sub(eax), 16)) as i64);
                        if cx < 0 {
                            halve_now = true;
                        } else {
                            let ax = i16s(
                                (sar(
                                    (i16s(a1 as i64).wrapping_mul(0xde9cu32 as i32))
                                        .wrapping_sub(i16s(a7 as i64) << 16),
                                    16,
                                )) as i64,
                            );
                            if ax > 0 {
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
                    let eax =
                        abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(i16s(a8 as i64) << 16));
                    let ecx = abssat((i16s(a1 as i64) << 16).wrapping_sub(i16s(a8 as i64) << 16));
                    if eax < ecx {
                        return (halve(a1), true);
                    }
                    go_final = true;
                }
            }
            if a10_above_branch && !go_final {
                if i16s(a5 as i64) >= 0xccd {
                    let ss5 = shift_scale(shl(i16s(mant5 as i64), 16), shift_arg2); // BUG1 FIX (dropped a stale overwrite)
                    let inner = i16s(
                        (sar(
                            0x599a0000i32.wrapping_sub(i16s(a5 as i64).wrapping_mul(0x2666)),
                            16,
                        )) as i64,
                    );
                    let ecxc = inner.wrapping_mul(i16s(mant6 as i64)).wrapping_mul(2);
                    if ss5 <= ecxc {
                        bp = a8;
                        go_final = true;
                    } else {
                        let edx = i16s(a4 as i64) << 16;
                        let eax = abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(edx));
                        let ecx = abssat((i16s(a1 as i64) << 16).wrapping_sub(edx));
                        if eax < ecx {
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
            let edx = i16s(bp as i64) << 16;
            let eax = abssat((i16s(a1 as i64) << 16 >> 1).wrapping_sub(edx));
            let ecx = abssat((i16s(a1 as i64) << 16).wrapping_sub(edx));
            if eax >= ecx {
                return (identity(a1), false);
            }
            return (halve(a1), true);
        }
        (identity(a1), false)
    }
}

// ======================= ctx+0x614 pitch tracker =======================
// The octave_halve_decide side-channel args a4..a10 are NOT DLL-internal magic: they are a 7-word
// struct at refine_ctx+0x614, updated once per frame by the tracker update, which is
// called from the pitch tracker with args (struct, r622, c62c, c626, c630) -- every
// one of which we already derive. Field map (from the pitch-tracker call site and the
// tracker's own field stores):
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
    let di = i16s(r622 as i64);
    let bp = i16s(c62c as i64);
    let mut l20 = t.w6; // long-term pitch average
    let mut l14 = t.w8; // a4 predictor source
    let mut l10 = t.wc; // stable-run counter
    let mut ebx = t.w2;
    let mut l18 = t.w4; // fast pitch track
                        // c62c < 0xccd -> smooth the fast pitch track 0.3*r622 + 0.7*a7.
    if bp < 0xccd {
        let ecx = di.wrapping_mul(0x4ccc);
        let eax = i16s(t.w4 as i64).wrapping_mul(0xb334);
        l18 = (sar(ecx.wrapping_add(eax), 16)) & 0xffff;
    }
    // |r622 - c626| vs 0.15*r622 -> stable-run counter.
    let mut edx = shl(di, 16).wrapping_sub(shl(i16s(c626 as i64), 16));
    let mut eax = if edx >= 0 {
        edx
    } else if edx == i32::MIN {
        i32::MAX
    } else {
        -edx
    };
    let ecx_thr = sar(di.wrapping_mul(0x999a), 2);
    if eax < ecx_thr {
        l10 += 1;
    } else {
        l10 = 0;
    }
    // good frame = c62c<0x199a AND c630<0x199a AND pitch stable.
    let good = bp < 0x199a && i16s(c630 as i64) < 0x199a && eax < ecx_thr;
    #[allow(unused_mut)] // port artifact: the reference re-writes this slot; this path never does
    let mut l24;
    if good {
        l14 = di & 0xffff; // a4 source := r622
        eax = edx;
        let sh = if edx == 0 { 0 } else { norm_shift(eax) };
        edx = shl(edx, sh);
        ebx = (-sh) & 0xffff;
        let ecx = i16s(l20 as i64).wrapping_mul(0xfae2);
        let mut ax = i16s(di as i64).wrapping_mul(0xa3d8);
        edx = sar(edx, 16);
        l24 = 0x7fff; // a5 := 32767
        ax = sar(ax, 5).wrapping_add(ecx.wrapping_add(0x8000));
        l20 = (sar(ax, 16)) & 0xffff; // 0.98*l20 + 0.02*r622
    } else {
        // decay the delta block-float by 0.8 and a5 by 0.9.
        edx = i16s(t.w0 as i64).wrapping_mul(0xcccc);
        let sh = if edx == 0 { 0 } else { norm_shift(edx) };
        ebx = ebx.wrapping_sub(sh);
        edx = sar(shl(edx, sh & 0xff), 16);
        l24 = (sar(i16s(t.wa as i64).wrapping_mul(0xe666), 16)) & 0xffff;
    }
    // a4 = 0.9*l14 + 0.9*bf(delta) + 0.1*l20, clamped to [0x666,0x3e00].
    let ebp = edx & 0xffff;
    let mut d2 = i16s(ebp as i64).wrapping_mul(0xe666);
    d2 = bf_shift(d2, ebx);
    let ea = i16s(l14 as i64).wrapping_mul(0xe666);
    let b2 = sat_add(ea, d2);
    let ed = sar(i16s(l20 as i64).wrapping_mul(0xcccc), 3);
    let mut res = sar(sat_add(ed, b2), 16) & 0xffff;
    if i16s(res as i64) > 0x3e00 {
        res = 0x3e00;
    } else if i16s(res as i64) < 0x666 {
        res = 0x666;
    }
    t.w8 = res;
    t.wa = l24;
    t.w2 = ebx & 0xffff;
    t.w4 = l18;
    t.w0 = ebp;
    t.w6 = l20;
    t.wc = l10 & 0xffff;
}
/// The divisor coarse_pitch is called with (4, 5 or 6).
fn ring_median_divide(ctr: i32, a7: i32, c626: i32) -> i32 {
    if i16s(ctr as i64) < 3 {
        return 4;
    }
    let cx = i16s(a7 as i64);
    let c6 = i16s(c626 as i64);
    if cx >= 0x2400 && c6 >= 0x2400 {
        return 6;
    }
    if cx < 0x2000 {
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
    let si = p0;
    if si > DD && p2 < DD {
        return (s16(q2), s16(p2));
    }
    if si > BB && p2 < sar16_field(si) {
        return (s16(q2), s16(p2));
    }
    let cx = p2;
    if cx > DD && si < DD {
        return (s16(q0), s16(p0));
    }
    if cx > BB && si < sar16_field(cx) {
        return (s16(q0), s16(p0));
    }
    // blend
    let ecx = (sar(shl(s16(p2), 16), 1) as i64 + sar(shl(s16(p0), 16), 1) as i64) as i32;
    let p1 = s16(sar(ecx, 16));
    let ec0 = s16(q0);
    let eb0 = s16(q2);
    let edi = sar(shl(ec0, 16), 1);
    let esi = sar(shl(eb0, 16), 1);
    let mut q1 = s16(sar((esi as i64 + edi as i64) as i32, 16));
    if p0 >= 0xccc {
        return (q1, p1);
    }
    let mut e2 = absat((shl(ec0, 16) as i64 - esi as i64) as i32);
    let edx = s16(q0);
    e2 = (e2 as i64 - (edx as i64) * 0x3332) as i32;
    if e2 < 0 {
        q1 = s16(sar((sar(esi, 1) as i64 + edi as i64) as i32, 16));
        return (q1, p1);
    }
    let ebx = shl(eb0, 16);
    e2 = absat((ebx as i64 - edi as i64) as i32);
    e2 = (e2 as i64 - (edx as i64) * 0x1998) as i32;
    if e2 < 0 {
        q1 = s16(sar((ebx as i64 + edi as i64) as i32, 16));
    }
    (q1, p1)
}
// bd30 ring-median tail -> c624
#[allow(dead_code)]
fn ring_median_tail(c62c: i32, prev_voicing_score: i32, c626: i32, rr: i32) -> i32 {
    let k = 0x3333i32;
    let l = 0x199ai32;
    let u16 = |x: i32| x & 0xffff;
    let s16 = |x: i32| {
        let x = x & 0xffff;
        if x >= 0x8000 {
            x - 0x10000
        } else {
            x
        }
    };
    let (si, c630) = (u16(c62c), u16(prev_voicing_score));
    if si > k && c630 < k {
        return u16(c626);
    }
    let skip_pitch_average = if c630 <= k {
        !(c630 <= l || si >= (c630 >> 1))
    } else if si < k {
        true
    } else {
        !(c630 <= l || si >= (c630 >> 1))
    };
    if skip_pitch_average {
        return u16(rr);
    }
    u16((s16(rr) + s16(c626)) >> 1)
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
    let cx = arg5;
    let di: i32;
    if pq_s16(q[cx as usize]) < 0x3333 {
        di = cx;
    } else {
        let mut d = 0i32;
        for e in 1..5 {
            if pq_s16(q[e as usize]) < pq_s16(q[d as usize]) {
                d = e;
            }
        }
        di = d;
    }
    if pq_s16(q[di as usize]) < 0x3333 {
        return finalize(pq_s16(p[di as usize]));
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
        let ecx = 0x1f - shw;
        let mask = (-1i32).wrapping_shl(ecx as u32);
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
    /// Returns (scorevec, transform exponent, blockexp_of_i16_spectrum, blockexp_of_scorevec_i32).
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
        // a2f0 returns a1b0's value, not fft_bfp_transform's, and d810
        // doubles it before e460 tail-returns it. So expo = 2*stage_ret.
        let stage_ret = inverse_fft_butterfly_stage(&mut work, r, 6, 0, 6);
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
        (out, 2 * stage_ret as i32, be_spec, be_sv)
    }
    /// The ab80 per-pass block exponent (`new_be`) for frame `f`, pass 0|1.
    pub(crate) fn pass_be(pref: &[i16], f: i64, pass: usize) -> i32 {
        let (_, be_off, be_len) = CFG[pass];
        be_for(pref, f, be_off, be_len) as i32
    }
}

const BANDS_CONST: [i16; 7] = [0, 0, 8, 18, 28, 38, 48];

mod c820m {
    #![allow(clippy::needless_range_loop)]
    use crate::enc::atan2_bfp_divide::bfp_divide;
    use crate::enc::voicing_fixed::band_voicing_ratio_code;
    // ============================================================ primitives
    // `normalize64` / `shift64` are `pub(crate)` in `enc::band_decompress`, so a
    // bin crate cannot name them. These are byte-identical re-derivations of the
    // same two functions from the same disassembly; `same_as_lib_*` tests below pin
    // them against the lib's own behaviour indirectly via the real captured data.

    /// Normalize a 64-bit signed value -> signed shift count.
    fn normalize64(lo: u32, hi: u32) -> i32 {
        if lo == 0 && hi == 0 {
            return 0;
        }
        let (mut lo, mut hi) = (lo, hi);
        if (hi as i32) < 0 {
            lo = !lo;
            hi = !hi;
        }
        let mut esi: u32 = 0;
        let mut edx: u32 = 0x80;
        let mut edi: i32 = 0;
        loop {
            if ((esi & lo) | (edx & hi)) != 0 {
                break;
            }
            esi = (esi >> 1) | ((edx & 1) << 31);
            edi += 1;
            edx = ((edx as i32) >> 1) as u32;
            if edi >= 0x28 {
                break;
            }
        }
        edi - 9
    }

    /// Left shift (cnt >= 0) / arithmetic right shift (cnt < 0).
    fn shift64(lo: u32, hi: u32, cnt: i32) -> (u32, u32) {
        if cnt >= 0 {
            let cl = cnt as u32;
            if cl >= 0x40 {
                (0, 0)
            } else if cl >= 0x20 {
                (0, lo.wrapping_shl(cl & 0x1f))
            } else if cl == 0 {
                (lo, hi)
            } else {
                (lo.wrapping_shl(cl), (hi << cl) | (lo >> (32 - cl)))
            }
        } else {
            let cl = (-cnt) as u32;
            let fill = if (hi as i32) < 0 { u32::MAX } else { 0 };
            if cl >= 0x40 {
                (fill, fill)
            } else if cl >= 0x20 {
                ((((hi as i32) >> (cl & 0x1f)) as u32), fill)
            } else if cl == 0 {
                (lo, hi)
            } else {
                ((lo >> cl) | (hi << (32 - cl)), ((hi as i32) >> cl) as u32)
            }
        }
    }

    /// Shared arithmetic variable-shift: left by `cx` if non-negative, else right
    /// by `-cx`, saturating to a full sign shift for cx <= -31. Used in band_min_reduce, renormalize_by_exponents
    /// and inside the band walkers.
    fn var_shift(eax: i32, cx: i16) -> i32 {
        if cx >= 0 {
            ((eax as u32).wrapping_shl((cx as u32) & 0x1f)) as i32
        } else if cx > -31 {
            eax >> (((-(cx as i32)) as u32) & 0x1f)
        } else {
            eax >> 31
        }
    }

    /// The 32x16 fractional multiply both band walkers inline: split `v` into
    /// low/high 16-bit halves, multiply each by `frac`, and recombine.
    fn mulfrac(v: i32, frac: i32) -> i32 {
        let lo = ((v as u32) & 0xffff) as i32;
        let hi = ((v >> 16) as i16) as i32;
        let a = lo.wrapping_mul(frac) >> 15;
        let c = hi.wrapping_mul(frac);
        a.wrapping_add(c.wrapping_mul(2))
    }

    /// Saturating negate: `-v`, mapping `i32::MIN` to `i32::MAX`.
    fn sat_neg(v: i32) -> i32 {
        if v == i32::MIN {
            i32::MAX
        } else {
            v.wrapping_neg()
        }
    }

    /// The `(sext16(hi) << 16) | u16(lo)` saturating 32-bit reconstruct both
    /// walkers perform before each `+= STEP`.
    fn sat_reconstruct(hi: u16, lo: u16) -> i32 {
        (((hi as i16) as i32) << 16) | (lo as i32)
    }

    // ============================================================ full-step band integrate
    /// `(out_mant, out_exp, x, src)` -> band count.
    /// STEP = `(sext16(x) << 9) >> 3` = x*64. Walks until `hi >= 0x80`.
    fn integrate_spectral_bands_full_step(
        out_mant: &mut [i32],
        out_exp: &mut [i16],
        x: i16,
        src: &[i32],
    ) -> u16 {
        let step: i32 = ((x as i32) << 9) >> 3;
        let edx0: i32 = (step >> 1).wrapping_add(0x8000);

        let mut frac_src: u16 = (edx0 as u32 & 0xffff) as u16; // frac source
        let pos0: u16 = ((edx0 >> 16) as u32 & 0xffff) as u16;
        let mut esi: i32 = sat_reconstruct(pos0, (edx0 as u32 & 0xffff) as u16).wrapping_add(step);
        let mut band_pos: u16 = pos0; // pos
        let mut hi: u16 = ((esi >> 16) as u32 & 0xffff) as u16;
        let mut esi_saved: i32 = esi;
        let mut count: u16 = ((hi as i32) - (band_pos as i32)) as u32 as u16;
        let mut i: i32 = 0;

        if (hi as i16) >= 0x80 {
            return 0;
        }
        loop {
            let pos = (band_pos as i16) as i32;
            let base = src[pos as usize];

            let frac1 = ((((frac_src as i32) >> 1) & 0x7fff) as i16) as i32;
            let t = sat_neg(mulfrac(base, frac1)).wrapping_add(base);
            let mut acc: i64 = t as i64;

            let esi_lo: u16 = (esi as u32 & 0xffff) as u16;
            let cnt_i = (count as i16) as i32;
            let frac2 = (((esi >> 1) & 0x7fff) as i16) as i32;
            acc += mulfrac(src[(pos + cnt_i) as usize], frac2) as i64;

            // middle sum runs when count > 1
            if 1 < (count as i16) {
                for j in 0..((count as i16) as i32 - 1) {
                    acc += src[(pos + 1 + j) as usize] as i64;
                }
            }

            let alo = (acc as u64 & 0xffff_ffff) as u32;
            let ahi = ((acc as u64) >> 32) as u32;
            let sh = normalize64(alo, ahi);
            let (rlo, _rhi) = shift64(alo, ahi, sh);

            out_exp[i as usize] = (-sh) as i16;
            out_mant[i as usize] = rlo as i32;
            i += 1;

            // advance
            esi = sat_reconstruct(hi, (esi_saved as u32 & 0xffff) as u16).wrapping_add(step);
            band_pos = hi;
            esi_saved = esi;
            let new_hi: u16 = ((esi >> 16) as u32 & 0xffff) as u16;
            count = ((new_hi as i32) - (band_pos as i32)) as u32 as u16;
            frac_src = esi_lo;
            hi = new_hi;
            if (new_hi as i16) >= 0x80 {
                break;
            }
        }
        i as u16
    }

    // ============================================================ half-step band integrate
    /// `(out_mant, out_exp, x, n, src)`.
    /// STEP2 = `(sext16(x) << 9) >> 4` = x*32; runs exactly `n` bands; the band
    /// hi edge is hard-capped at 0x7f (min with 0x7f).
    fn integrate_spectral_bands_half_step(
        out_mant: &mut [i32],
        out_exp: &mut [i16],
        x: i16,
        n: i16,
        src: &[i32],
    ) {
        let step2: i32 = ((x as i32) << 9) >> 4;
        let mut acc_pos: i32 = (step2 >> 1).wrapping_add(0x8000).wrapping_add(step2);
        if 0 >= n {
            return;
        }
        let mut band_pos: u16 = ((acc_pos >> 16) as u32 & 0xffff) as u16; // pos
        let mut frac_src: u16 = (acc_pos as u32 & 0xffff) as u16; // frac source

        for (w, _) in (0..n).enumerate() {
            let pos = (band_pos as i16) as i32;
            let edi: i32 = sat_reconstruct(
                ((pos as u32) & 0xffff) as u16,
                (acc_pos as u32 & 0xffff) as u16,
            )
            .wrapping_add(step2);

            let raw_hi: u16 = ((edi >> 16) as u32 & 0xffff) as u16;
            // min_i16(0x7f, hi) — hi edge capped at 0x7f
            let capped_hi: i32 = if 0x7fi16 < (raw_hi as i16) {
                0x7f
            } else {
                raw_hi as i32
            };
            let base = src[pos as usize];
            let count: u16 = ((capped_hi - (band_pos as i32)) as u32) as u16;

            let frac1 = ((((frac_src as i32) >> 1) & 0x7fff) as i16) as i32;
            let t = sat_neg(mulfrac(base, frac1)).wrapping_add(base);
            let mut acc: i64 = t as i64;

            let cnt_i = (count as i16) as i32;
            let frac2 = (((edi >> 1) & 0x7fff) as i16) as i32;
            acc += mulfrac(src[(pos + cnt_i) as usize], frac2) as i64;

            if 1 < (count as i16) {
                for j in 0..((count as i16) as i32 - 1) {
                    acc += src[(pos + 1 + j) as usize] as i64;
                }
            }

            let alo = (acc as u64 & 0xffff_ffff) as u32;
            let ahi = ((acc as u64) >> 32) as u32;
            let sh = normalize64(alo, ahi);
            let (rlo, _rhi) = shift64(alo, ahi, sh);

            out_exp[w] = (-sh) as i16;
            out_mant[w] = rlo as i32;

            // advance: uses the CAPPED hi (unlike the full-step walk, which uses raw hi)
            acc_pos = sat_reconstruct(
                ((capped_hi as u32) & 0xffff) as u16,
                (edi as u32 & 0xffff) as u16,
            )
            .wrapping_add(step2);
            frac_src = (acc_pos as u32 & 0xffff) as u16;
            band_pos = ((acc_pos >> 16) as u32 & 0xffff) as u16;
        }
    }

    // ============================================================ renormalize_by_exponents
    /// `(dst, src, exps, n)`: renormalize `src[i]` by `exps[i]-max` into
    /// `dst[i]` (refine_ring_p0 always passes dst == src), returning `max` over `exps[..n]`.
    fn renormalize_by_exponents(arr: &mut [i32], exps: &[i16], n: i16) -> i16 {
        let mut max = exps[0];
        for k in 1..(n as usize) {
            if exps[k] > max {
                max = exps[k];
            }
        }
        if 0 >= n {
            return max;
        }
        for k in 0..(n as usize) {
            let sh = exps[k].wrapping_sub(max);
            arr[k] = var_shift(arr[k], sh);
        }
        max
    }

    // ============================================================ band_min_reduce
    /// `(out, arr2, exp2, arr4, exp4, x, cap)`: 8 bands, each
    /// `out[k] = min(out[k], c390(...))`. `arr2`/`arr4` are the GUARDED arrays
    /// (index 0 is the zero guard word the caller places at `E-0xe0`/`E-0x1c0`).
    #[allow(clippy::too_many_arguments)]
    fn band_min_reduce(out: &mut [i16], arr2: &[i32], exp2: i16, arr4: &[i32], exp4: i16, x: i16, cap: i16) {
        // bfp_divide(0x20000000, -3, x, -3) -> the band-walk step
        let (mant, expo) =
            bfp_divide(0x2000_0000, -3, x as i32, -3).expect("divide-by-zero: x low 16 bits are 0");
        let sh = expo.wrapping_sub(0xf);
        let step: i32 = var_shift((mant as i32) << 16, sh);
        let step2: i32 = step.wrapping_add(step);

        let mut step_acc: i32 = 0; // running step accumulator
        let mut prev_edge: i16 = 0; // previous band hi edge

        for k in 0..8usize {
            step_acc = step_acc.wrapping_add(step2);
            let rounded = (step_acc.wrapping_add(0x8000) >> 16) as u32 as u16;
            let mut edge = rounded as i16;
            if edge > cap {
                edge = cap;
            }
            let n = ((edge as i32) - (prev_edge as i32)) as u32 as u16 as i16;

            // --- accumulate 1: from arr4 (+prev_edge), exponent exp4 ---
            let mut acc1: i64 = 0;
            if 0 <= n {
                for j in 0..=(n as i32) {
                    acc1 += arr4[((prev_edge as i32) + j) as usize] as i64;
                }
            }
            let (l1, h1) = (
                (acc1 as u64 & 0xffff_ffff) as u32,
                ((acc1 as u64) >> 32) as u32,
            );
            let sh1 = normalize64(l1, h1);
            let (r1, _) = shift64(l1, h1, sh1);
            let exp1: i16 = ((exp4 as i32) - sh1) as u32 as u16 as i16;
            let mant1: i16 = (((r1 as i32) >> 16) as u32 as u16) as i16;

            // --- accumulate 2: from arr2 (+prev_edge), exponent exp2 ---
            let mut acc2: i64 = 0;
            if 0 <= n {
                for j in 0..=(n as i32) {
                    acc2 += arr2[((prev_edge as i32) + j) as usize] as i64;
                }
            }
            let (l2, h2) = (
                (acc2 as u64 & 0xffff_ffff) as u32,
                ((acc2 as u64) >> 32) as u32,
            );
            let sh2 = normalize64(l2, h2);
            let (r2, _) = shift64(l2, h2, sh2);
            // NOTE: these two are pushed FULL 32-bit (no movzx), but the ratio-code
            // helper reads both as 16-bit (`test ax,ax`, `cwde`) and the divide's own
            // exp_seed only ever reaches a `as i16` truncation -- so narrowing
            // here is exact, not an approximation.
            let exp2v: i16 = ((exp2 as i32) - sh2) as i16;
            let mant2: i16 = ((r2 as i32) >> 16) as i16;

            let code = band_voicing_ratio_code(mant1, exp1, mant2, exp2v);
            if code < out[k] {
                out[k] = code;
            }

            prev_edge =
                ((step_acc.wrapping_sub(step).wrapping_add(0x8000) >> 16) as u32 as u16) as i16;
        }
    }

    // ============================================================ refine_ring_p0
    /// Full port of `refine_ring_p0`. Mutates the 90-word ring in place (it only ever
    /// touches p0 slot 0 = ring[10..18]).
    pub(crate) fn refine_ring_p0(ring: &mut [i16; 90], x: i16, src: &[i32; 129]) {
        // The real frame's local arrays: exp array at E-0x22e has 0x6e bytes of
        // room (55 i16) before the E-0x1c0 guard; both mant arrays have 0xdc bytes
        // (55 i32). Sized generously here; the assert pins the real bound.
        let mut mant1 = [0i32; 64];
        let mut mant2 = [0i32; 64];
        let mut exps = [0i16; 64];

        let count1 = integrate_spectral_bands_full_step(&mut mant1, &mut exps, x, src);
        assert!(
            count1 as usize <= 55,
            "returned {count1} bands; the real frame only has room for 55"
        );
        let max1 = renormalize_by_exponents(&mut mant1, &exps, count1 as i16);

        // step 3 REUSES the same exp array, overwriting step 1's exponents
        integrate_spectral_bands_half_step(&mut mant2, &mut exps, x, count1 as i16, src);
        let max2 = renormalize_by_exponents(&mut mant2, &exps, count1 as i16);

        // guarded pointers: E-0xe0 = &mant1[-1], E-0x1c0 = &mant2[-1], guard == 0
        let mut g1 = [0i32; 65];
        let mut g2 = [0i32; 65];
        g1[1..].copy_from_slice(&mant1);
        g2[1..].copy_from_slice(&mant2);

        let mut out = [0i16; 8];
        out.copy_from_slice(&ring[10..18]);
        band_min_reduce(&mut out, &g1, max1, &g2, max2, x, count1 as i16);
        ring[10..18].copy_from_slice(&out);
    }
}

// ==================================================================
// a3/bandexp, band_voicing_mask, bfp_scale, a9, the VQ measurement, and the frame-serial
// ring driver.
// ==================================================================

// ---------------------------------------------------------------- a3 + bandexp
/// 32-element max-magnitude scan -> BFP normalize shift. Local re-derivation,
/// because `voicing_fixed::array_find_shift` is private to that module.
fn array_find_shift32(arr: &[i32; 32]) -> i32 {
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
    (31 - maxabs.leading_zeros() as i32) - 30
}

/// The per-band tail: mask lags 0/1, accumulate_aligned-combine the two subpos
/// scorevecs, then pack to i16. Returns (a3[8][32], bandexp[8]).
/// `bandexp[c] = common + shift`, `common = max(expo[s0], expo[s1])`
/// (accumulate_aligned's return), `shift` = the pack's own in-place exponent
/// update.
fn amp_scores_and_bandexp(svs: &[[i32; 32]; 16], expo: &[i32; 16]) -> ([[i16; 32]; 8], [i16; 8]) {
    let mut a3 = [[0i16; 32]; 8];
    let mut bandexp = [0i16; 8];
    for c in 0..8usize {
        let (s0, s1) = (2 * c, 2 * c + 1);
        let mut a = svs[s0];
        a[0] = 0;
        a[1] = 0;
        let mut b = svs[s1];
        b[0] = 0;
        b[1] = 0;
        let acc = bfp_add_arrays(&a, &b, expo[s0] as i16, expo[s1] as i16);
        a3[c] = bfp_normalize_pack32(&acc);
        let common = expo[s0].max(expo[s1]);
        bandexp[c] = (common + array_find_shift32(&acc)) as i16;
    }
    (a3, bandexp)
}

// ---------------------------------------------------------------- d890 / a9
/// The `d890` call (table pointer, expbias=-7, n=199, plus1=1).
///
/// a2f0 returns **a1b0's** value, NOT fft_bfp_transform's, so
/// `outer_transform::combine_outer` is the WRONG exponent here.
fn cepstral_transform_narrow(gap2: &[i16; GAP2_LEN]) -> ([i32; 129], i16) {
    use crate::enc::array_a_stage2::inverse_fft_butterfly_stage;
    use crate::enc::loudness_transform::fft_bfp_transform;
    let mut win = win_from_gap2(gap2);
    let exp0 = block_exponent(gap2);
    let fft_ret = fft_bfp_transform(&mut win, 0, exp0.saturating_sub(7), 7);
    let mut scratch = [0i16; 258];
    scratch[..256].copy_from_slice(&win[..256]);
    let stage_ret = inverse_fft_butterfly_stage(&mut scratch, fft_ret, 8, 0, 1);
    win[..256].copy_from_slice(&scratch[..256]);
    (assemble_bins_from_win(&win), stage_ret.wrapping_mul(2))
}

/// The OTHER d890 call, inside the wide-window path:
/// `d890(dst, src, table pointer, expbias=-7, n=0xff=255,
/// plus1=0)`. Its output buffer is the refine_ring_p0 arg3
/// (129 i32; the walk caps at index 0x80).
/// Bit-exact against capture (151/151 rows, 19328/19328 words; zeroed audio
/// gives 0/151).
///
/// Two steps are easy to omit here, and omitting them scores ~3928/16256, which
/// looks like a wrong `src` rather than a wrong transform:
///  1. the in-place normalize. `win_taper::from_raw_gap2` folds it in, but
///     `win_taper_wide::win_from_gap2_slot2` does NOT (it takes raw,
///     un-normalized data), so it must be applied here.
///  2. a2f0's tail stores `arr[1] = 0` when `plus1 == 0`. The usual shorthand
///     for a2f0 ("fft_bfp_transform then a1b0, returns a1b0's value") omits this;
///     it does not matter for the a3 chain, which passes `plus1=1` and skips the
///     store.
fn cepstral_transform_wide(gap2w: &[i16; GAP2_LEN_WIDE]) -> [i32; 129] {
    use crate::enc::array_a_stage2::inverse_fft_butterfly_stage;
    use crate::enc::cepstral_stage2::rescale_accumulate;
    use crate::enc::loudness_transform::fft_bfp_transform;
    let mut localbuf = *gap2w;
    let exp0 = rescale_accumulate(&mut localbuf, 0); // in place
    let mut win = win_from_gap2_slot2(&localbuf);
    let fft_ret = fft_bfp_transform(&mut win, 0, exp0.saturating_sub(7), 7);
    let mut scratch = [0i16; 258];
    scratch[..256].copy_from_slice(&win[..256]);
    let stage_ret = inverse_fft_butterfly_stage(&mut scratch, fft_ret, 8, 0, 0); // plus1 = 0 (static trace)
    let _ = stage_ret;
    win[..256].copy_from_slice(&scratch[..256]);
    win[1] = 0; // a2f0's own zero store, taken because plus1 == 0
    assemble_bins_from_win(&win)
}

fn sum128(bins: &[i32]) -> i64 {
    bins[..128].iter().map(|&v| v as i64).sum()
}

/// Precompute, from PCM alone, the per-frame `a9` and the wide refine_ring_p0 source.
/// `a9[f]` uses the ring's two 199-word windows at gap index `f-1`
/// (`arr1` = slot 1 = `ring[108..307]`, `arr2` = slot 0 = `ring[28..227]`).
fn precompute_from_ring(pcm: &[i16], nframes: usize) -> (Vec<Option<i16>>, Vec<[i32; 129]>) {
    use crate::synth::FRAME_SAMPLES;
    use crate::Encoder;
    let mut enc = Encoder::new();
    for ch in pcm.chunks(FRAME_SAMPLES) {
        if ch.len() < FRAME_SAMPLES {
            break;
        }
        let mut fr = [0i16; FRAME_SAMPLES];
        fr.copy_from_slice(ch);
        let _ = enc.encode_frame_r34(&fr);
    }
    let _ = enc.flush_r34();
    precompute_from_logs(
        enc.gap2_mid_log(),
        enc.gap2_slot1_log(),
        enc.gap2_slot2_log(),
        nframes,
    )
}

/// The audio-independent half of [`precompute_from_ring`]: given the encoder's
/// already-captured gap2 logs (`gap2_mid_log` / `gap2_slot1_log` /
/// `gap2_slot2_log`), derive the per-frame `a9` and refine_ring_p0 `wide` inputs WITHOUT
/// re-running the r34 encoder. `precompute_from_ring` runs the encoder once and
/// forwards its logs here; `b1_track_from_logs` forwards logs captured by a
/// caller's own encoder pass, so the pass is shared instead of repeated.
fn precompute_from_logs(
    gap0: &[[i16; GAP2_LEN]],
    gap1: &[[i16; GAP2_LEN]],
    gap2w: &[[i16; GAP2_LEN_WIDE]],
    nframes: usize,
) -> (Vec<Option<i16>>, Vec<[i32; 129]>) {
    if gap0.is_empty() || gap1.is_empty() || gap2w.is_empty() {
        panic!("TRAP#1 GUARD: encoder produced EMPTY gap logs");
    }
    let mut a9 = vec![None; nframes];
    let mut wide = vec![[0i32; 129]; nframes];
    for f in 0..nframes {
        let gi = f as i64 - 1; // measured alignment
        if gi >= 0 && (gi as usize) < gap0.len() && (gi as usize) < gap1.len() {
            let (b1, e1) = cepstral_transform_narrow(&gap1[gi as usize]);
            let (b2, e2) = cepstral_transform_narrow(&gap0[gi as usize]);
            a9[f] = spectral_energy_ratio_from_sums(sum128(&b1), e1, sum128(&b2), e2);
        }
        // The wide (refine_ring_p0 arg3) source uses the SAME `f-1` gap alignment as a9,
        // and the alignment is sharp: gapoff -1 = 151/151 rows, 0 = 251/16256
        // words, +1 = 241/16256. Frame 0 has no predecessor, so `wide[0]` stays
        // zero -- the same cold start a9 takes.
        if gi >= 0 && (gi as usize) < gap2w.len() {
            wide[f] = cepstral_transform_wide(&gap2w[gi as usize]);
        }
    }
    (a9, wide)
}

// ---------------------------------------------------------------- band_voicing_mask
fn i16_array_max(arr: &[i16]) -> i16 {
    let mut m = arr[0];
    for &v in &arr[1..] {
        if v > m {
            m = v;
        }
    }
    m
}
/// The inline per-element renormalize.
fn renorm16(v: i16, sh: i16) -> i16 {
    let eax = (v as i32) << 16;
    let out = if sh >= 0 {
        (eax as u32).wrapping_shl((sh as u32) & 0x1f) as i32
    } else if sh <= -0x1f {
        eax >> 31
    } else {
        eax >> ((-(sh as i32)) & 0x1f)
    };
    (out >> 16) as i16
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Ring90 {
    w: [i16; 90],
}
impl Ring90 {
    fn p0(&self, s: usize, k: usize) -> i16 {
        self.w[10 + s * 8 + k]
    }
    fn set_p0(&mut self, s: usize, k: usize, v: i16) {
        self.w[10 + s * 8 + k] = v;
    }
    fn p1(&self, s: usize, k: usize) -> i16 {
        self.w[50 + s * 8 + k]
    }
    fn set_p1(&mut self, s: usize, k: usize, v: i16) {
        self.w[50 + s * 8 + k] = v;
    }
    fn slot(&self, which: usize, s: usize) -> [i16; 8] {
        std::array::from_fn(|k| {
            if which == 0 {
                self.p0(s, k)
            } else {
                self.p1(s, k)
            }
        })
    }
}

/// `band_voicing_mask`.
fn band_voicing_mask(
    ring: &mut Ring90,
    exp_hist: &mut [i16; 4],
    windows: &[[i16; 32]; 8],
    x: i16,
    bandexp: &[i16; 8],
) -> u16 {
    for k in 0..3usize {
        let src = 2 - k;
        let dst = src + 2;
        for j in 0..8 {
            let a = ring.p0(src, j);
            ring.set_p0(dst, j, a);
            let b = ring.p1(src, j);
            ring.set_p1(dst, j, b);
        }
    }
    exp_hist[3] = exp_hist[2];
    exp_hist[2] = exp_hist[1];
    exp_hist[1] = exp_hist[0];
    let x_hi: i32 = (x as i32) << 16;
    let v1 = ((((x_hi >> 10).wrapping_add(0x10000)) >> 16) & 0xffff) as u16 as i16;
    let v2 = ((((x_hi >> 11).wrapping_add(0x10000)) >> 16) & 0xffff) as u16 as i16;
    let mut mask: u16 = 0;
    let mut local_a = [0i16; 8];
    for k in 0..8usize {
        let (o1, o2, o3, o4) = band_spectral_bfp(&windows[k], x, v1, v2);
        ring.set_p1(0, k, o3);
        local_a[k] = bandexp[k].wrapping_add(o4);
        let code = band_voicing_ratio_code(o1, o2, o3, o4);
        ring.set_p0(0, k, code);
        mask = mask.wrapping_shl(1);
        if code < 0x199a {
            mask |= 1;
        }
    }
    let e = i16_array_max(&local_a);
    for k in 0..8usize {
        let v = renorm16(ring.p1(0, k), local_a[k].wrapping_sub(e));
        ring.set_p1(0, k, v);
    }
    exp_hist[0] = e;
    if (0..8).all(|k| ring.p1(0, k) == 0) {
        for k in 0..8 {
            ring.set_p1(0, k, 1);
        }
        exp_hist[0] = -30;
    }
    mask
}

// ---------------------------------------------------------------- bfp_scale
fn bfp_scale_shl32(v: u32, n: u32) -> u32 {
    if n >= 32 {
        0
    } else {
        v << n
    }
}
fn bfp_scale_sar32(v: u32, n: u32) -> u32 {
    ((v as i32) >> n.min(31)) as u32
}
fn bfp_scale_shift(value: u32, count: i16) -> u32 {
    if count >= 0 {
        let esi = count as u32;
        let ecx = 31u32.wrapping_sub(esi);
        let eax = bfp_scale_shl32(0xFFFF_FFFF, ecx & 0x1f);
        let mut ecx2 = eax & value;
        if (value as i32) < 0 {
            ecx2 = ecx2.wrapping_sub(eax);
        }
        if ecx2 != 0 {
            if (value as i32) < 0 {
                0x8000_0000
            } else {
                0x7FFF_FFFF
            }
        } else {
            bfp_scale_shl32(value, esi & 0x1f)
        }
    } else {
        let mag = (-(count as i32)) as u32;
        if mag < 31 {
            bfp_scale_sar32(value, mag & 0x1f)
        } else {
            bfp_scale_sar32(value, 31)
        }
    }
}
fn bfp_scale_norm32(v: i32) -> i32 {
    if v == 0 {
        return 0;
    }
    let c: u32 = if v >= 0 { v as u32 } else { !(v as u32) };
    if c == 0 {
        panic!("norm32: bsr(0) undefined, v={v}");
    }
    let bsr = 31 - c.leading_zeros() as i32;
    ((30 - bsr) as u16) as i32
}
fn bfp_scale_bfp_shift(v: i32, cx: i32) -> i32 {
    let c16 = (cx as u32 as u16) as i16;
    if c16 >= 0 {
        bfp_scale_shl32(v as u32, (cx as u32) & 0x1f) as i32
    } else if c16 > -31 {
        bfp_scale_sar32(v as u32, cx.wrapping_neg() as u32 & 0x1f) as i32
    } else {
        bfp_scale_sar32(v as u32, 31) as i32
    }
}
fn min_s16(a: i16, b: i16) -> i16 {
    if a < b {
        a
    } else {
        b
    }
}
fn max_s16(a: i16, b: i16) -> i16 {
    if a > b {
        a
    } else {
        b
    }
}

/// `bfp_scale` (the formula only).
#[allow(clippy::too_many_arguments)]
fn bfp_scale(
    a1: &[i16; 8],
    a2: &mut [i16; 8],
    a3: &[i16; 8],
    a4: &[i16; 8],
    a5: &mut [i16; 8],
    a6: &[i16; 8],
    a7: i16,
    a8: i16,
    a9: i16,
) {
    use crate::enc::atan2_bfp_divide::bfp_divide;
    let gain = a9 as i32;
    let (e7, e8) = ((a7 as u16) as i32, (a8 as u16) as i32);
    let gain_of = |b: usize| -> i16 {
        let mut eax = (a4[b] as i32).wrapping_mul(gain);
        eax = eax.wrapping_add(eax);
        (eax >> 16) as i16
    };
    for b in 0..3usize {
        a2[b] = min_s16(a1[b], a3[b]);
        a5[b] = gain_of(b);
    }
    for b in 3..8usize {
        let (w_a, w_b) = (a4[b], a6[b]);
        let edi = w_a as i32;
        let mut eax = bfp_scale_shl32(edi as u32, 16) as i32;
        let sh_a = bfp_scale_norm32(eax);
        eax = bfp_scale_shl32(eax as u32, (sh_a as u32) & 0x1f) as i32;
        let ea = ((e7 - sh_a) as u32 as u16) as i32;
        let mut esi = bfp_scale_shl32(w_b as i32 as u32, 16) as i32;
        let sh_b = bfp_scale_norm32(esi);
        esi = bfp_scale_shl32(esi as u32, (sh_b as u32) & 0x1f) as i32;
        let eb = ((e8 - sh_b) as u32 as u16) as i32;
        let big = max_s16(ea as u16 as i16, eb as u16 as i16);
        let e_common = ((big as i32).wrapping_add(1) as u32 as u16) as i32;
        eax = bfp_scale_bfp_shift(eax, ea - e_common);
        esi = bfp_scale_bfp_shift(esi, eb - e_common);
        let sum = eax.wrapping_add(esi);
        let sh_s = bfp_scale_norm32(sum);
        let exp_sum = e_common - sh_s;
        let mant_sum = ((bfp_scale_shl32(sum as u32, (sh_s as u32) & 0x1f) as i32) >> 16) as u32 as u16;
        let mut p1 = (a1[b] as i32).wrapping_mul(edi);
        p1 = p1.wrapping_add(p1);
        let sh_p1 = bfp_scale_norm32(p1);
        let ep1 = ((e7 - sh_p1) as u32 as u16) as i32;
        p1 = bfp_scale_shl32(p1 as u32, (sh_p1 as u32) & 0x1f) as i32;
        let mut p2 = (a3[b] as i32).wrapping_mul(w_b as i32);
        p2 = p2.wrapping_add(p2);
        let sh_p2 = bfp_scale_norm32(p2);
        let ep2 = ((e8 - sh_p2) as u32 as u16) as i32;
        p2 = bfp_scale_shl32(p2 as u32, (sh_p2 as u32) & 0x1f) as i32;
        let big2 = max_s16(ep1 as u16 as i16, ep2 as u16 as i16);
        let e2 = ((big2 as i32).wrapping_add(1) as u32 as u16) as i32;
        p1 = bfp_scale_bfp_shift(p1, ep1 - e2);
        p2 = bfp_scale_bfp_shift(p2, ep2 - e2);
        let sum2 = p1.wrapping_add(p2);
        let sh_s2 = bfp_scale_norm32(sum2);
        if mant_sum as i16 == 0 {
            a2[b] = 0x7fffu16 as i16;
        } else {
            let exp2 = e2 - sh_s2;
            let mant2 = bfp_scale_shl32(sum2 as u32, (sh_s2 as u32) & 0x1f) as i32;
            let Some((m_ret, e_ret)) = bfp_divide(mant2, exp2, mant_sum as i32, exp_sum) else {
                panic!("divide-by-zero: mant_sum checked non-zero");
            };
            let scaled = bfp_scale_shift(bfp_scale_shl32(m_ret as i32 as u32, 16), e_ret);
            a2[b] = (bfp_scale_sar32(scaled, 16) as i32) as i16;
        }
        a5[b] = gain_of(b);
    }
    let mut or_all: u16 = 0;
    for &v in a5.iter() {
        or_all |= v as u16;
    }
    if or_all == 0 {
        *a5 = [1i16; 8];
    }
}

// ---------------------------------------------------------------- the VQ measurement
const VMASK32: u32 = 0xffff_ffff;
fn vs16(x: i64) -> i64 {
    let x = x & 0xffff;
    if x >= 0x8000 {
        x - 0x10000
    } else {
        x
    }
}
fn vs32(x: u32) -> i32 {
    x as i32
}
fn vsar32(x: u32, n: u32) -> u32 {
    ((vs32(x)) >> (n & 31)) as u32
}
fn vshl32(x: u32, n: u32) -> u32 {
    x.wrapping_shl(n & 31)
}
const VCOEF: [i32; 6] = [0x5a82, 0xa3ac, 0x570d, 0xa3ac, 0x414a, 0xc000];
fn vsw(v: i32) -> i32 {
    if v >= 0x8000 {
        v - 0x10000
    } else {
        v
    }
}
/// Fixed-point base-2 logarithm.
fn vlog2_fn(mant16: i32, exp_arg: i32) -> u32 {
    let c: Vec<i32> = VCOEF.iter().map(|&x| vsw(x)).collect();
    let edx0 = vshl32((mant16 as u32) & VMASK32, 16);
    let ebx_shift: i32 = if edx0 == 0 {
        0
    } else {
        let bsr_in = if vs32(edx0) >= 0 { edx0 } else { !edx0 };
        0x1e - (31 - bsr_in.leading_zeros() as i32)
    };
    let cl = (ebx_shift as u32) & 0xff;
    let eax0 = vshl32(VCOEF[0] as u32 & VMASK32, 16);
    let mut edx2 = (vs32(vshl32(edx0, cl & 0x1f)) as i64 - vs32(eax0) as i64) as u32;
    edx2 = vsar32(edx2, 16);
    let edx = vs16((edx2 & 0xffff) as i64) as i32;
    let stage = |edx: i32, xin: i32, cadd: i32, extra_sar: bool| -> i32 {
        let eax = (edx.wrapping_mul(xin)) as u32 & VMASK32;
        let mut ecx = ((vs32(eax) as i64 * 2) as u32).wrapping_add(0x8000) & VMASK32;
        if extra_sar {
            ecx = vsar32(ecx, 1);
            ecx &= 0xffff_8000;
        } else {
            ecx &= 0xffff_0000;
        }
        ecx = ecx.wrapping_add(vshl32(cadd as u32 & VMASK32, 16));
        ecx = vsar32(ecx, 16);
        vs16((ecx & 0xffff) as i64) as i32
    };
    let x1 = stage(edx, c[1], VCOEF[2], false);
    let x2 = stage(edx, x1, VCOEF[3], false);
    let x3 = stage(edx, x2, VCOEF[4], true);
    let eax = (edx.wrapping_mul(x3)) as u32 & VMASK32;
    let ecx2 = vshl32(VCOEF[5] as u32 & VMASK32, 16);
    let mut eax2 = ((vs32(eax) as i64 * 8) as u32).wrapping_add(0x2_0000) & VMASK32;
    eax2 &= 0xfffc_0000;
    eax2 = eax2.wrapping_add(ecx2);
    let frac = vsar32(eax2, 15);
    let exp_part = vshl32((((exp_arg - ebx_shift) as i16) as u16) as u32, 16);
    ((vs32(frac) as i64 + vs32(exp_part) as i64) as u32) & VMASK32
}
/// Saturating variable shift-scale.
fn vshift_scale(value: u32, count: i16) -> u32 {
    if count >= 0 {
        let ecx = (31u32).wrapping_sub(count as u32);
        let eax = vshl32(0xffff_ffff, ecx & 0x1f);
        let mut ecx2 = eax & value;
        if vs32(value) < 0 {
            ecx2 = ecx2.wrapping_sub(eax);
        }
        if ecx2 != 0 {
            if vs32(value) < 0 {
                0x8000_0000
            } else {
                0x7fff_ffff
            }
        } else {
            vshl32(value, count as u32 & 0x1f)
        }
    } else {
        let mag = (-(count as i32)) as u32;
        if mag < 31 {
            vsar32(value, mag & 0x1f)
        } else {
            vsar32(value, 31)
        }
    }
}
/// The compand per-element body.
fn compand(den_dword: u32, num_word: i16) -> i64 {
    let sc = vshift_scale(den_dword, num_word);
    let ax = vs16(((sc >> 16) & 0xffff) as i64);
    if ax == 0 {
        return 0x7fff;
    }
    let mut e = (vs32(vlog2_fn(ax as i32, 0)) >> 1) as i64;
    if (e as u32 & VMASK32) == 0x8000_0000 {
        e = 0x7fff_ffff;
    } else {
        e = -e;
    }
    e -= 0xa934;
    if e < 0 {
        return 0;
    }
    if e > 0xfffe {
        return 0x7fff;
    }
    e += 1;
    vs16((((e << 15) >> 16) & 0xffff) as i64) & 0xffff
}
/// a6[k] = max(1, i16(p1[8+k])) -- p1 slots 1 AND 2 (the look-back).
fn stable_counter_from_ring(ring: &Ring90) -> [i16; 16] {
    std::array::from_fn(|k| std::cmp::max(1, ring.p1(1 + k / 8, k % 8)) as i16)
}
/// **a4 from a FLAT p0 plane** -- `p0_flat[s * 8 + k]` = `ring.p0(s, k)`, slots 0..4.
///
/// This is the single source of truth for a4's arithmetic: [`pitch_predictor_from_ring`] is a thin
/// adapter over it, so a diagnostic that feeds this function the DLL's captured `in0`
/// array (the `in0_00..` columns, which are exactly p0 flat from slot 0) is testing
/// the SAME code the live b1 chain runs -- not a drifting copy. `num[j]` is compand's per-element shift count; the live path
/// passes all-zero (see `pitch_predictor_from_ring`), and this probe exists so a future round can
/// measure what a real `num` buys without forking the formula.
pub(crate) fn pitch_predictor_flat(p0_flat: &[i16; 40], num: &[i16; 16]) -> [i16; 16] {
    let mut a4 = [0i16; 16];
    for b in 0..2usize {
        let s = b + 1;
        let mode = if s > 1 { 2usize } else { 1usize };
        for k in 0..8usize {
            let r0 = p0_flat[(s - mode) * 8 + k] as i64;
            let r1 = p0_flat[s * 8 + k] as i64;
            let r2 = p0_flat[(s + mode) * 8 + k] as i64;
            let den = std::cmp::min(std::cmp::max(r0, r2), r1);
            a4[b * 8 + k] = compand(vshl32((den as u32) & VMASK32, 16), num[b * 8 + k]) as i16;
        }
    }
    a4
}

/// a4 over p0 slots 0..4 (num = 0; gate NOT applied in the live chain -- see the
/// r47 `hdr30` note in the module header for why hdr30 is not yet audio-derivable).
fn pitch_predictor_from_ring(ring: &Ring90) -> [i16; 16] {
    let p0_flat: [i16; 40] = std::array::from_fn(|j| ring.p0(j / 8, j % 8));
    pitch_predictor_flat(&p0_flat, &[0i16; 16])
}

/// **The a4 GATE** (proven r47 vs `compand_*.log`).
///
/// a4 band `b` (b in {0,1}) is emitted iff bit `b+3` of `hdr30` (`[ctx+0x30]`, the vmeas
/// `hdr30` column) is set; otherwise those 8 a4 words are forced to **0** and the DLL does
/// not call compand for that band. Verified band-firing == gate-bit on 199/199 both files.
///
/// Wiring this with the true hdr30 lifts b1 +4 voiced / +2 mark with zero
/// regressions. It is NOT called by the live chain, because hdr30 is a stateful
/// voicing-onset envelope and is not audio-derivable -- feeding a captured hdr30
/// would make the chain capture-fed rather than audio-only.
pub(crate) fn pitch_predictor_gate(mut a4: [i16; 16], hdr30: i32) -> [i16; 16] {
    let h = (hdr30 as u32) & 0xffff;
    for b in 0..2usize {
        if (h >> (b + 3)) & 1 == 0 {
            for k in 0..8 {
                a4[b * 8 + k] = 0;
            }
        }
    }
    a4
}

/// **AUDIO-ONLY `hdr30` (a4 gate envelope), derived per frame from PCM energy.** (r51)
///
/// hdr30 is a 16-bit shift register `hdr30 = (hdr30<<2) | fill`, where `fill` is a 2-bit
/// per-frame voicing edge code `Vcur | (Vprev<<1)` (`11`=steady voiced, `01`=onset,
/// `10`=offset, `00`=unvoiced). Verified against the captured `enc_vmeas` hdr30 column: the
/// ramp `0,1,7,31,127,511,…` and the offset ramp are exactly this recurrence.
///
/// The driving flag `V` is a **voice-activity (energy) decision**, NOT the voicing byte:
/// r51 proved 22 voiced / 31 mark frames have voicing-byte `b1==16` (unvoiced) yet the
/// hdr30 envelope is VOICED (onset-ramp frames + mid-word dropouts), so feeding the voicing
/// byte -- even the GT byte -- reproduces the gate on only 175/199 frames and REGRESSES the
/// gate below no-gate. An adaptive noise-floor VAD (below) reproduces the two gate bits
/// (bit3/bit4) on **396/398 voiced, 395/398 mark** and recovers the gate value.
///
/// VAD = `log2(mean-square over the 160-sample frame) > noise_floor + MARGIN`, where the
/// noise floor tracks the running minimum with a slow upward `LEAK`, plus 1-frame hangover.
/// `MARGIN`/`LEAK`/`HANG` are a single shared config (same for both corpus files); they were
/// tuned against the captured hdr30 on those two files, so this is a 1-parameter fit on a
/// 2-file corpus -- audio-only (reads only PCM) but NOT yet validated to generalise.
/// hdr30 VAD decision margin, in log2 power units above the tracked noise floor.
pub const HDR30_MARGIN: f64 = 2.75;
/// Upward leak applied to the hdr30 noise floor on every non-minimum frame.
pub const HDR30_LEAK: f64 = 0.05;
/// Frames of hangover after the hdr30 VAD releases.
pub const HDR30_HANG: usize = 1;

/// Incremental, frame-serial form of [`derive_hdr30_vad`] for streaming callers.
///
/// The adaptive noise floor is the one slow-converging state a bounded analysis
/// window cannot reproduce, so a streamer carries this across pumps and hands the
/// accumulated register to [`b1_track_hdr30`]. Pushing frames `0..n` in order
/// yields exactly `derive_hdr30_vad(pcm, n)`.
#[derive(Clone, Debug, Default)]
pub struct Hdr30Vad {
    nf: Option<f64>,
    cd: usize,
    h: i32,
    prev: bool,
}

impl Hdr30Vad {
    /// A cold tracker, positioned before frame 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume the next 160-sample frame and return its shift-register value.
    /// `frame` shorter than 160 samples is treated as zero-padded, matching
    /// `derive_hdr30_vad`'s out-of-range read.
    pub fn push_frame(&mut self, frame: &[i16]) -> i32 {
        const FRAME: usize = 160;
        let mut acc = 0.0f64;
        for i in 0..FRAME {
            let v = f64::from(frame.get(i).copied().unwrap_or(0));
            acc += v * v;
        }
        let e = (acc / FRAME as f64 + 1.0).log2();
        let nf = self.nf.unwrap_or(e);
        let nf = if e < nf { e } else { nf + HDR30_LEAK };
        self.nf = Some(nf);
        let v = if e > nf + HDR30_MARGIN {
            self.cd = HDR30_HANG;
            true
        } else if self.cd > 0 {
            self.cd -= 1;
            true
        } else {
            false
        };
        let fill = (v as i32) | ((self.prev as i32) << 1);
        self.h = ((self.h << 2) | fill) & 0xffff;
        self.prev = v;
        self.h
    }
}

pub(crate) fn derive_hdr30_vad(raw_pcm: &[i16], nframes: usize) -> Vec<i32> {
    const FRAME: usize = 160;
    let margin: f64 = HDR30_MARGIN;
    const LEAK: f64 = HDR30_LEAK;
    const HANG: usize = HDR30_HANG;
    let energy: Vec<f64> = (0..nframes)
        .map(|f| {
            let s = f * FRAME;
            let mut acc = 0.0f64;
            for i in 0..FRAME {
                if s + i < raw_pcm.len() {
                    let v = raw_pcm[s + i] as f64;
                    acc += v * v;
                }
            }
            (acc / FRAME as f64 + 1.0).log2()
        })
        .collect();
    // adaptive noise-floor VAD + hangover
    let mut nf = energy.first().copied().unwrap_or(0.0);
    let mut base = Vec::with_capacity(nframes);
    for &e in &energy {
        if e < nf {
            nf = e;
        } else {
            nf += LEAK;
        }
        base.push(e > nf + margin);
    }
    let mut vad = vec![false; nframes];
    let mut cd = 0usize;
    for i in 0..nframes {
        if base[i] {
            cd = HANG;
            vad[i] = true;
        } else if cd > 0 {
            cd -= 1;
            vad[i] = true;
        }
    }
    // shift register
    let mut out = Vec::with_capacity(nframes);
    let mut h: i32 = 0;
    let mut prev = false;
    for &v in &vad {
        let fill = (v as i32) | ((prev as i32) << 1);
        h = ((h << 2) | fill) & 0xffff;
        out.push(h);
        prev = v;
    }
    out
}

// ---------------------------------------------------------------- the driver

struct FrameIn {
    a3: [[i16; 32]; 8],
    bandexp: [i16; 8],
    x: i16,
}

/// The AUDIO-ONLY front end, additionally emitting a3/bandexp. No capture
/// branches: this is the arithmetic that scores x at 199/199 and 194/199.
fn front_end(pref: &[i16], nframes: usize) -> Vec<FrameIn> {
    let mut ab80 = fe::PassAccumulatorState::default();
    let mut prev_pass1 = [[0i64; 10]; 16];
    let mut e530_hist = [[0i64; 50]; 16];
    let (mut r622h, mut c624h, mut c62eh, mut c62ch): (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) =
        (vec![], vec![], vec![], vec![]);
    let mut prev_raw_r622 = 0i32;
    let mut trk = PitchTrackerState::default();
    let mut prev_voicing_mask: i32 = 0;
    const K: usize = 20;
    let mut out: Vec<FrameIn> = Vec::new();
    for f in 0..nframes as i64 {
        let pass0 = ab80.run_pass(pref, f, 0);
        let pass1 = ab80.run_pass(pref, f, 1);
        if f == 0 {
            e530_hist = [[0i64; 50]; 16];
        } else {
            for b in 0..16usize {
                let mut app = [0i64; 20];
                app[..10].copy_from_slice(&prev_pass1[b]);
                app[10..(10 + 10)].copy_from_slice(&pass0[b]);
                let mut nb = [0i64; 50];
                nb[..(49 - K)].copy_from_slice(&e530_hist[b][K..((49 - K) + K)]);
                for i in 0..K {
                    nb[49 - K + i] = app[i];
                }
                e530_hist[b] = nb;
            }
        }
        prev_pass1 = pass1;
        let mut p4: [[i16; 10]; 16] = [[0; 10]; 16];
        for b in 0..16usize {
            p4[b] = std::array::from_fn(|i| pass1[b][i] as i16);
        }
        let mut svs = [[0i32; 32]; 16];
        let mut expo = [0i32; 16];
        for b in 0..16usize {
            let mut raw = [0i16; 58];
            for i in 0..48 {
                raw[i] = e530_hist[b][1 + i] as i16;
            }
            raw[48..(10 + 48)].copy_from_slice(&p4[b]);
            // CAND_MODE 3, verbatim: band b of the e460 rawwork is one adf0
            // pass-group from generation gen(b); shifts[b] = that pass's ab80
            // block exponent; cand = max(shifts[1..7]).
            const GEN: [i64; 7] = [0, -2, -2, -1, -1, 0, 0];
            const PASS: [usize; 7] = [0, 0, 1, 0, 1, 0, 1];
            let mut sh = [-32768i16; 7];
            let mut bd = [0i16; 7];
            for bb in 1..7 {
                let ff = f + GEN[bb];
                if ff >= 0 {
                    sh[bb] = fe::pass_be(pref, ff, PASS[bb]) as i16;
                    bd[bb] = BANDS_CONST[bb];
                }
            }
            let cd = sh[1..7].iter().copied().max().unwrap();
            let (sv, r, _, _) = fe::score_vector_transform(&raw, cd, &bd, &sh);
            svs[b] = sv;
            expo[b] = r;
        }
        let (a3, bandexp) = amp_scores_and_bandexp(&svs, &expo);
        // No noise window here: this chain's `x` / a3 / bandexp consumers were
        // derived against the unweighted score, and feeding them the weighted one
        // costs b1 on every vector. The b0 chain takes the weighted score.
        let score32 = e100::run(&svs, &expo, None);
        let mut score = [0i32; 40];
        for i in 0..32 {
            score[i] = score32[i] as i32;
        }
        let c626 = prev_raw_r622;
        let c630 = *c62ch.last().unwrap_or(&32767);
        let a11 = prev_voicing_mask;
        if a11 == 0 {
            trk.wc = 0;
        }
        let div = ring_median_divide(trk.wc, trk.w4, c626);
        let (low, high) = (div, 32);
        let a1 = b7b0m::a1_clamp(b7b0m::coarse_pitch(&score, low, high));
        let a8_derived = if i16s(trk.wc as i64) >= 1 {
            c626
        } else {
            trk.w4
        };
        let a: [i32; 8] = [trk.w8, trk.wa, trk.wc, trk.w4, a8_derived, c626, c630, a11];
        let (raw_r622, halved) =
            b980m::octave_halve_decide(a1, &score, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        let r622_post = octave_clamp(raw_r622);
        let c62c = if halved {
            b980m::conf5(a1, &score)
        } else {
            b980m::conf6(a1, &score)
        };
        pitch_tracker_update(&mut trk, raw_r622, c62c, c626, c630);
        let (c624, c62e) = {
            let (aa, bb) = continuity_smoother(r622_post, c626, c62c, c630);
            (aa & 0xffff, bb & 0xffff)
        };
        r622h.push(r622_post);
        c624h.push(c624);
        c62eh.push(c62e);
        c62ch.push(c62c);
        prev_raw_r622 = raw_r622;
        prev_voicing_mask = 0; // overwritten by the caller's real band_voicing_mask mask (see drive_b1)
        out.push(FrameIn {
            a3,
            bandexp,
            x: (raw_r622 & 0xffff) as u16 as i16,
        });
    }
    out
}
