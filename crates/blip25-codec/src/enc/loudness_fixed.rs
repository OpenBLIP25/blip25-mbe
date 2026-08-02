// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! EXPERIMENTAL comparison path: assembles a mechanism-matched `step`
//! (harmonic-count lookup, [`super::step_table`]) + `outer`
//! ([`super::loudness_transform`]/[`super::outer_transform`]) +
//! `band_decompress` ([`super::band_decompress`]) chain into a candidate
//! per-frame gain (`gamma`), for **diagnostic comparison only**.
//!
//! It is exposed alongside the shipped path, never silently replacing it, so
//! `encode_frame_r33`/`r34`'s output cannot be regressed by an unverified
//! formula here. An existing diagnostic path is likewise never rewired in
//! place: a newly-closed mechanism gets its OWN new path so earlier
//! measurements stay comparable. That is why several near-duplicate
//! `candidate_gamma_*` functions coexist below.
//!
//! [`candidate_gamma_real_bins`] is the most mechanism-matched path and the one
//! to read first. [`candidate_gamma`] keeps the older synthesized-spectrum
//! approximation described below.
//!
//! ## Scope of `candidate_gamma` (the OLDER path, kept for comparison)
//!
//! **Recovered and tested** (see each module's doc for accuracy):
//! [`super::step_table`] (47/48 captured `COUNT`->`STEP` entries, 71.5% exact /
//! 100% within 5%), [`super::loudness_transform`] +
//! [`super::outer_transform`] (the array-transform and scalar-combine halves of
//! `outer`), and [`super::band_decompress`] itself (bit-exact for its
//! `(step, count, bins, outer) -> raw16` computation).
//!
//! **Two gaps this path does NOT close**:
//!
//! 1. **`bins`/`win` sourcing.** The reference's `bins` (129-entry
//!    magnitude-squared spectrum) and `win` (the 256-`i16`-word array
//!    `loudness_transform::fft_bfp_transform` transforms) come from its internal
//!    fixed-point spectral pipeline (`history_ring` -> `windowed_taper` ->
//!    `real_fft32`, or a related buffer), never closed end-to-end for
//!    encode-side use -- see `band_decompress.rs`'s "what is NOT yet wired up".
//!    This path substitutes the synthesized `analysis::Spectrum`
//!    (`spec.re`/`spec.im`) scaled by a fixed constant (`spec_scale()`), which
//!    is not verified against the reference's internal fixed-point spectrum.
//! 2. **`fft_bfp_transform`'s `arg2` parameter.** The FORMULA half is closed:
//!    `esiVal == `[`super::block_exponent::block_exponent`]`(arr[0:count])` and
//!    the FFT stage's `arg2 == esiVal - 7`, confirmed by two independent
//!    captures on two recordings.
//!
//!    **What remains open** is the array itself. `arr`/`count` is a
//!    **stack-local** scratch buffer (sizes 199/199/255 across the 3 per-frame
//!    calls) at a DIFFERENT address from `fft_bfp_transform`'s `a1ptr`, and it is
//!    NOT a raw slice of [`super::history_ring::HistoryRing`]: fitting the best
//!    contiguous ring slice against captured arrays collapses to 1-5% per-tap
//!    accuracy past the first ~9 frames. Do not re-test that hypothesis. Since
//!    the array cannot be derived, this module uses the literal `-7`
//!    (`block_exponent` of a silent array, minus 7 -- the value the spectrum
//!    stage is observed to push before its content-dependent adjustment) as a
//!    zeroth-order approximation.
//!
//! Because of those gaps, `candidate_gamma` is an **approximate, calibrated**
//! quantity, not a first-principles one: `gamma_calib_a()`/`gamma_calib_b()` are
//! an affine fit against the reference's `gamma` trajectory (recovered by replaying
//! the anchor files' `b2` bits through [`super::quantize::quantize_gain`]'s
//! inverse, [`crate::dequantize::decode_gain`]), mapping the mean of
//! `band_decompress`'s bias-clamped per-band output onto the `gain_O`
//! codebook's numeric range. A statistical calibration, not a recovered formula.

use super::analysis::Spectrum;
use super::band_decompress;
use super::step_table;

// The block-float word-rescale primitive and its BFP-polynomial sub-call moved
// to `crate::shared::gamma_poly` so the decoder can reach them without `enc/`.
// Re-exported here so this module's `ptr63c` assembler and `enc::b1_audio`'s
// `crate::enc::loudness_fixed::gamma_poly_bfp_eval` call sites are unchanged.
pub(crate) use crate::shared::gamma_poly::{block_float_word_rescale, gamma_poly_bfp_eval};

/// The DERIVED (not fitted) slope converting `band_decompress`'s
/// `raw16`/`biased` output back into a log2-amplitude domain -- see
/// [`candidate_gamma_real_bins`] for the numeric derivation. Substituting this
/// exact constant for a fitted slope does NOT improve the `b2` match rate.
pub(crate) const RAW16_TO_LOG2_SCALE: f64 = 2048.0;

/// Fixed scale applied to the synthesized `spec.re`/`spec.im` values before
/// quantizing to the `i16`/`i32` fixed-point domain
/// `band_decompress`/`loudness_transform` expect. **Not derived from the reference**
/// -- picked by a grid search (spec scale x the gamma-calib-a coefficient, both
/// swept over powers of 2) keeping whichever combination gave the highest raw
/// `b2` match rate.
///
/// Do not re-run that search expecting a better constant. The best value found
/// (`0.25`) scores 3.8% overall, against the ~3.1% a uniformly-random guess
/// scores on the 32-entry `gain_O` codebook, and no combination has a stable
/// above-chance peak. That is evidence the `bins`/`win` sourcing gap (module doc
/// item 1) dominates this path's error, not the calibration constants. Closing
/// that gap would replace this with the reference's own fixed-point spectrum and
/// retire the constant.
fn spec_scale() -> f64 {
    0.25
}

/// Approximation for `fft_bfp_transform`'s `arg2` (module doc item 2, silence
/// case): the value the spectrum stage is observed to push before its
/// content-dependent conditional adjustment. The fallback when no audio window
/// is available -- the first frames, or unit tests.
const ARG2_APPROX_SILENCE: i16 = -7;

/// **Content-responsive** approximation for `fft_bfp_transform`'s `arg2`, using
/// the closed formula
/// (`arg2 == `[`super::block_exponent::block_exponent`]`(arr) - 7`) applied to a
/// PROXY array rather than the reference's upstream `arr` -- a stack-local array
/// sourced from `encoder_buf+0x13a8`/`+0x13e0`, 199/199/255 samples per frame.
/// The copy write-site and source address are confirmed, but three independent
/// content searches found it does not match `HistoryRing` or raw PCM at any
/// tested offset or cadence; its semantic identity is open.
///
/// **A constant `arg2` is not an acceptable substitute.** [`WinState`] is a
/// self-referential accumulator starting all-zero with no external input, and
/// `fft_bfp_transform` is zero-preserving (`bitrev_permute`/`prescale`/
/// `twiddle_pair` all map zero to zero), so a constant `arg2` freezes `win`
/// permanently and makes the whole `candidate_gamma` chain
/// content-INDEPENDENT for the entire file.
///
/// The reference's upstream array is audio-driven -- growing in magnitude through
/// frames 7-9 and hard-saturating on speech onset -- so the proxy is the most
/// recent window of bit-exact audio (raw or prefiltered PCM, caller's choice)
/// through the same `block_exponent` formula. `Encoder::analyze_frame` passes
/// [`super::history_ring::HistoryRing::gap2_window`], since a write-watchpoint
/// on the real array's address proves its writer IS the ring's shift mechanism,
/// making the ring more principled than an arbitrary recency window.
///
/// **This is a proxy, not a recovered formula.** It gives `arg2` the right order
/// of magnitude and the right qualitative behaviour (silence -> `-7`, loud
/// speech -> less negative or positive) and does not reach a
/// statistically-significant above-chance result.
pub(crate) fn arg2_from_recent_audio(recent: &[i16]) -> i16 {
    if recent.is_empty() {
        return ARG2_APPROX_SILENCE;
    }
    super::block_exponent::block_exponent(recent).saturating_sub(7)
}

/// `arg3` is fixed at `7` for every capture on record (`n_pairs = 1 << 7 = 128`
/// complex pairs = 256 `i16` words).
pub(crate) const ARG3: u32 = 7;

/// Affine calibration mapping `band_decompress`'s mean bias-clamped raw output
/// onto the `gain_O` codebook's numeric range. A statistical fit, not a
/// recovered formula (module doc). `1/4096` is the best of the grid search
/// [`spec_scale`] describes, giving **3.8%** overall `b2` match -- close to, not
/// clearly above, the 32-entry codebook's ~3.1% random-guess baseline.
fn gamma_calib_a() -> f64 {
    1.0 / 4096.0
}
fn gamma_calib_b() -> f64 {
    0.0
}

/// Assemble `bins` (129-entry magnitude-squared spectrum) from the synthesized
/// `analysis::Spectrum`, per `spec_scale()`. **Approximate input** -- see module
/// doc item 1. `win` is not built here; see [`WinState`].
///
/// **Superseded by [`assemble_bins_from_win`]**, which is mechanism-matched.
/// This is kept only because the three older diagnostic paths
/// (`candidate_gamma`/`band_loudness_fixed`, feeding
/// `b2_x86_log`/`b2_x86_pf_log`/`b2_x86_wintaper_log`) call it, and rewiring
/// them in place would invalidate their recorded measurements.
fn assemble_bins(spec: &Spectrum) -> [i32; 129] {
    let mut bins = [0i32; 129];
    for k in 0..129 {
        let re = spec.re[k] * spec_scale();
        let im = spec.im[k] * spec_scale();
        let mag2 = 2.0 * (re * re + im * im);
        bins[k] = mag2.clamp(0.0, i32::MAX as f64) as i32;
    }
    bins
}

/// The mechanism-matched `bins` (129-entry magnitude-squared spectrum),
/// computed directly from `win`'s own content -- **not** a DFT.
///
/// ```text
/// bins[k] = 2*((win[2k] as i64)^2 + (win[2k+1] as i64)^2)
/// ```
/// truncated to `i32`, matching the reference's 32-bit store (no saturation is
/// observed or needed on any capture).
///
/// The magnitude-squared assembler's `input_ptr` at this call site is
/// byte-identical to `a1ptr`/`win`'s address, and its `output_ptr` is
/// `encoder_buf+0x1c` -- the long-confirmed primary spectrum write. So the same
/// 256-word array [`super::loudness_transform::fft_bfp_transform`] mutates in
/// place to produce `outer`'s scalar return IS the FFT-domain buffer the reference
/// reads to build its magnitude-squared spectrum. There is no separate
/// "20-block real_fft32 assembly": that hypothesis chases a different call site
/// (the per-band x2-pass loop, a genuinely different consumer feeding a separate
/// energy accumulator, not `bins`).
///
/// ## Two flagged gaps
///
/// **Which of the three spectrum-stage invocations.** Exactly 2 of the 3
/// per-frame invocations write `encoder_buf+0x1c`, overwriting each other in
/// program order, so the SECOND is what persists -- and their `win` content
/// genuinely differs on non-silent frames. This function reuses the already-wired
/// `win` built from `HistoryRing::gap2_mid` ("this frame's SECOND `count=199`
/// reader"), which models that instant, but that `gap2_mid` is exactly slot 2's
/// localbuf source is carried over from the `compute_outer` finding, not proven
/// for this link.
///
/// **Bin 128.** The reference's array extends 2 `i16` words past `win`'s modeled
/// 256-word bound (indices 256/257, the 129th bin's real/imaginary parts).
/// `im` is consistently 0, matching every other bin's Nyquist-style convention,
/// but `re` is a small content-correlated value with no known raw-PCM formula.
/// `win`'s 256-word structural bound applies to its declaring frame, not to this
/// adjacent memory one hop further into the enclosing function's stack. Bin 128
/// is approximated as zero here rather than fabricated.
pub(crate) fn assemble_bins_from_win(win: &[i16; 256]) -> [i32; 129] {
    let mut bins = [0i32; 129];
    for (k, slot) in bins.iter_mut().enumerate().take(128) {
        let re = win[2 * k] as i64;
        let im = win[2 * k + 1] as i64;
        let mag2 = 2 * (re * re + im * im);
        *slot = mag2 as i32; // truncating cast, matches the real 32-bit store
    }
    // bins[128] left at 0 -- see this function's doc, "Bin 128".
    bins
}

/// (`be_wire_bins_amps`) build per-harmonic LINEAR amplitudes `M_l` from the
/// fixed-point 129-bin `|Sw|^2` spectrum + the bit-exact
/// [`super::band_decompress::band_decompress`] resampling chain (the reference's
/// actual method). `band_decompress` produces `count`=L log2-domain per-band
/// loudness words; converting each back to linear via `2^(word/2048)` (the
/// derived [`RAW16_TO_LOG2_SCALE`]) yields an `M_l` envelope the quantizer's
/// own `log2` forward transform then re-linearizes. The `outer` exponent only
/// adds a CONSTANT to every band's log level (it cancels in the quantizer's
/// mean/gain removal), so it is a fixed constant here (the per-harmonic SHAPE,
/// which is what b3..b8 encode, is unaffected by it).
///
/// This keys the `band_decompress` `step` on the integer-`L` modal lookup
/// ([`step_table::step_for_count`]). IMBE non-live (streaming) amplitude path;
/// the whole-buffer IMBE path uses the pitch-aligned [`amps_from_mechanism_bins`].
pub(crate) fn amps_from_bins(spec: &Spectrum, l: usize) -> Vec<f32> {
    let count = l.max(1);
    let l_c = l.clamp(1, 56);
    let bins = assemble_bins(spec);
    let step = step_table::step_for_count(l_c as u32);
    let outer_const: i16 = -10;
    let raw = band_decompress::band_decompress(step, count, &bins, outer_const);
    band_to_amps(&raw)
}

/// As [`amps_from_bins`], but keys the `band_decompress` `step` on the continuous
/// `omega_0`-refined count ([`step_table::step_for_omega`]) rather than the
/// integer-`L` modal lookup. AMBE+2 encode path only (`enc::mod`), used as the
/// non-live-gap2 fallback.
pub(crate) fn amps_from_bins_omega(spec: &Spectrum, l: usize, omega_0: f64) -> Vec<f32> {
    let count = l.max(1);
    let l_c = l.clamp(1, 56);
    let bins = assemble_bins(spec);
    let step = step_table::step_for_omega(l_c as u32, omega_0);
    let outer_const: i16 = -10;
    let raw = band_decompress::band_decompress(step, count, &bins, outer_const);
    band_to_amps(&raw)
}

/// Pitch-aligned mechanism amplitudes. Sources the per-band envelope from the
/// reference's own fixed-point spectrum mechanism (`frame_power_and_scale` ->
/// `band_decompress`) rather than the synthesized-DFT `assemble_bins` proxy
/// [`amps_from_bins_omega`] uses. `gap2` is the analysis window aligned to the
/// SAME frame as the pitch (`omega_0`/`l`); `outer` is the mechanism's own
/// block-exponent-derived scale (`scale - 30`), and `step` is keyed on the
/// continuous `omega_0` like [`amps_from_bins_omega`].
///
/// Returns `(M_l, raw)`: the linear amplitudes plus the raw `band_decompress`
/// log2-domain words (the latter for callers that want the pre-relinearize
/// vector). AMBE+2 Route A encode path (`enc::mod`'s aligned override site);
/// see that call site and `Encoder::set_live_gap2_amps`.
pub(crate) fn amps_from_mechanism_bins(
    gap2: &[i16; super::history_ring::GAP2_LEN],
    l: usize,
    omega_0: f64,
) -> (Vec<f32>, Vec<i16>) {
    let (bins, scale) = frame_power_and_scale(gap2, true);
    let outer = scale - 30;
    let l_c = l.clamp(1, 56);
    let step = step_table::step_for_omega(l_c as u32, omega_0);
    let raw = band_decompress::band_decompress(step, l.max(1), &bins, outer);
    (band_to_amps(&raw), raw)
}

/// Identical to [`amps_from_mechanism_bins`] but takes the `band_decompress`
/// `step` directly instead of deriving it from a continuous `omega_0` via
/// `step_table::step_for_omega`.
///
/// The IMBE mechanism path uses this so it can supply the reference-exact per-frame
/// step derived from the quantized IMBE pitch index (`fund_freq(b0) >> 13`),
/// which reproduces the reference DLL's captured step 45/45 on the mark
/// capture — where `step_for_omega`, fed the unquantized continuous pitch
/// estimate, is essentially never exact. `step_for_omega` (keyed on `omega_0`)
/// is shared with the frozen AMBE+2 Route A amplitude path and stays untouched.
pub(crate) fn amps_from_mechanism_bins_step(
    gap2: &[i16; super::history_ring::GAP2_LEN],
    l: usize,
    step: i16,
) -> (Vec<f32>, Vec<i16>) {
    let (bins, scale) = frame_power_and_scale(gap2, true);
    let outer = scale - 30;
    let raw = band_decompress::band_decompress(step, l.max(1), &bins, outer);
    (band_to_amps(&raw), raw)
}

/// The same `bins`/`scale`/`outer`/`step`/`band_decompress` mechanism chain as
/// [`amps_from_mechanism_bins`], but returns the `band_decompress` raw words
/// bias-clamped (the `[i16]` shape [`gamma_ref_exact`] consumes) instead of the
/// relinearized `M_l`. Used by the whole-buffer path's reference-exact `b2` gain
/// override.
pub(crate) fn biased_from_mechanism_bins(
    gap2: &[i16; super::history_ring::GAP2_LEN],
    l: usize,
    omega_0: f64,
) -> Vec<i16> {
    let (bins, scale) = frame_power_and_scale(gap2, true);
    let outer = scale - 30;
    let l_c = l.clamp(1, 56);
    let step = step_table::step_for_omega(l_c as u32, omega_0);
    let raw = band_decompress::band_decompress(step, l.max(1), &bins, outer);
    raw.iter()
        .map(|&v| band_decompress::bias_clamp_one(v))
        .collect()
}

/// Shared tail of the `amps_from_bins*` family: relinearize each
/// `band_decompress` log2-domain word (`2^(word/2048)`) into a linear `M_l`.
fn band_to_amps(raw: &[i16]) -> Vec<f32> {
    raw.iter()
        .map(|&v| {
            let lambda = f64::from(v) / RAW16_TO_LOG2_SCALE;
            2f64.powf(lambda) as f32
        })
        .collect()
}

/// The mechanism-matched `bins`/`step`/`outer`/
/// `band_decompress` chain's per-band output, bias-clamped, using
/// [`assemble_bins_from_win`] instead of [`assemble_bins`]'s synthesized
/// DFT. `win` is the caller-owned `WinState`, already mutated in place by
/// a prior [`band_decompress::compute_outer`] call this same frame (the
/// caller must call `compute_outer` BEFORE this function, exactly once,
/// with the SAME `win` -- matching the real reference's own program order, the
/// spectrum stage's `fft_bfp_transform` call before its own magnitude-squared
/// assembler call).
pub(crate) fn band_loudness_real_bins(win: &WinState, l: usize, step_outer: i16) -> Vec<i16> {
    let l_clamped = l.clamp(1, 56);
    let bins = assemble_bins_from_win(win);
    let step = step_table::step_for_count(l_clamped as u32);
    let raw = band_decompress::band_decompress(step, l_clamped, &bins, step_outer);
    raw.iter()
        .map(|&v| band_decompress::bias_clamp_one(v))
        .collect()
}

/// Candidate `gamma` from the real-`bins` chain above, via its OWN
/// independently tuned affine calibration
/// ([`realbins_calib_a`]/[`realbins_calib_b`]).
///
/// Those constants are deliberately NOT shared with [`candidate_gamma`]'s
/// `gamma_calib_a`/`b`, which were tuned for the synthesized-spectrum input's
/// completely different numeric scale; sharing them would silently perturb that
/// separately-tracked path.
///
/// ## Accuracy, and why the headline number is misleading
///
/// Pooled over all 3 anchor files (525 frames, the project's standing
/// convention): **20.38% (107/525) `b2` match**, versus the ~3.125% uniform
/// chance floor of the 32-entry `gain_O` codebook.
///
/// **That pooled number is dominated by one easy file.** `dtone_10` (a synthetic
/// dual-tone file, 125 of the 525 frames) has only 2 distinct `b2` values across
/// all 125 frames, so "always guess the most common value" alone scores 98.4% on
/// it. This calibration scores 78.4% there -- below that trivial baseline, but a
/// large absolute count that dominates the pool. On the two real-speech files
/// (`mark.pcm`/`cpvbad`, 400 frames combined) the SAME calibration scores
/// **3.0%/1.5%**, both below their own trivial baselines (9.5% / 15.5%) and at
/// or below the uniform chance floor.
///
/// Re-optimizing only on `mark`+`cpvbad` reaches 31/400 = 7.75% on speech alone
/// (`mark` 10.0%, `cpvbad` 5.5%) -- still below `cpvbad`'s trivial baseline. And
/// every calibration region with `a` small enough to make `gamma` nearly
/// frame-independent degenerates to "always guess one fixed index", reproducing
/// rather than exceeding whichever file's trivial baseline it lands near.
///
/// **The right baseline is not the uniform floor.** Judge against strategies
/// with zero understanding of the sound that exploit how skewed and
/// autocorrelated the `b2` target sequence is. Pooled over the same 525 frames:
/// "always guess this file's most common `b2`" scores **33.0%** (173/525), and
/// "guess the previous frame's value" scores **36.0%** (189/525). Both beat this
/// function's 20.38%.
///
/// So: `win`, `arg2`, `bins` and `outer` are all provably exact, and this is
/// still not a working loudness predictor. Two downstream pieces remain
/// approximate rather than derived -- `step_table` (71.5% exact) and this affine
/// calibration -- and they eat whatever signal the exact ingredients carry.
///
/// ## The slope is derivable, and deriving it does not help
///
/// The MULTIPLICATIVE half of the calibration is exactly derivable.
/// [`super::band_decompress::band_decompress`]'s `raw16` increases by exactly
/// `1024` per unit of `outer` and per doubling of the summed per-band bin value:
/// `raw16 ~= 1024*(outer + log2(big)) - 1023`. Since `bins` are
/// magnitude-SQUARED (`2*(re^2+im^2)`), `log2(big) = 2*log2(amplitude)` up to
/// the summation spanning multiple bins per band, giving a derived slope of
/// `1/2048` back to a log2-amplitude domain -- see [`RAW16_TO_LOG2_SCALE`]. The
/// same constant (`~=0.00048828`) falls out independently from `log2_fn`'s
/// Q16.16 output and the `shift_scale(_, 0xf-5)` rescale.
///
/// **Substituting the derived slope makes the match rate WORSE**: 12.57% pooled
/// (66/525), down from 20.4% (real speech: `mark` 2.0%, `cpvbad` 2.5%, both at
/// chance; `dtone_10` 45.6%, far below its 98.4% trivial baseline). A per-BAND
/// regression of `biased` against ground-truth `log2(M_l)` over 13,526 real
/// `(band, frame)` pairs finds essentially zero correlation (`r^2=0.0018`), with
/// the empirical slope even carrying the wrong sign.
///
/// `mean(biased)` simply does not carry a linear log-amplitude signal on real
/// speech through this mechanism, whatever affine transform is applied. The
/// likely reason: a wrong `STEP` for a frame's `L` does not add a small scale
/// error -- it changes WHICH of the 129 spectral bins are summed into which
/// output band, scrambling exactly the frames where `L` varies fastest (real
/// speech) while leaving near-constant-`L` content (`dtone_10`) alone. That
/// matches the observed pattern.
///
/// The derived-slope variant is deliberately NOT wired into `Encoder` as a live
/// diagnostic path, unlike every other candidate here: it is a measured
/// regression against the fitted-slope diagnostic with no offsetting benefit.
pub(crate) fn candidate_gamma_real_bins(win: &mut WinState, l: usize, arg2: i16) -> f64 {
    let mean_raw = mean_raw_real_bins(win, l, arg2);
    realbins_calib_a() * mean_raw + realbins_calib_b()
}

/// The exact reference formula computing the gain (`b2`) pre-quantization scalar from
/// the bit-exact [`super::band_decompress::band_decompress`] +
/// [`super::band_decompress::bias_clamp_one`] per-band output array, the
/// harmonic count `l`, and a `prev_gamma`-shaped bias term. A direct port, not a
/// fit.
///
/// ```text
/// mean       = trunc_div(sum_{i=0}^{l-1} sext16(biased[i]), l)     // signed truncating divide
/// log2l      = round16(shift_scale(log2_fn(l as i16, 15), 0xf-5))  // applied to the harmonic COUNT
///                                                                     itself, NOT a second spectral array
/// combined   = sat_add32(mean<<16, log2l<<16)
/// pre_shift  = sat_sub32(combined, prev_gamma_raw<<15)             // <<15 not <<16: HALF weight, matching
///                                                                     quantize_gain's `- 0.5*prev_gamma`
/// after      = shift_scale(pre_shift, 0)                           // count=0: pass-through/saturate
/// gamma_raw  = round16(sat_add32(after, 0x8000))
/// ```
///
/// Given the REAL `biased`/`l`/`prev_gamma_raw` a frame used, this reproduces
/// the reference's pre-quantization gamma bit-exactly on captured ground truth from
/// two files.
///
/// `prev_gamma_raw` is the one input this function does not derive. It is NOT
/// `state.prev_gamma() * 2048.0` (this project's recursive
/// `gamma - 0.5*prev_gamma` / `gain_o[b2] + 0.5*prev_gamma` state), and it is not
/// a plain previous-frame carry -- both are tested and refuted. It follows some
/// deterministic multi-frame recursion. See [`next_bias_raw_exact`] for the
/// traced update, and `prev_gamma_raw_candidate` for the fallback
/// [`candidate_gamma_ref_exact`] uses.
pub(crate) fn gamma_ref_exact(biased: &[i16], l: usize, prev_gamma_raw: i16) -> i16 {
    gamma_ref_exact_intermediates(biased, l, prev_gamma_raw).5
}

/// Same computation as [`gamma_ref_exact`], but returns every named
/// intermediate (`mean16`, `log2l16`, `combined`, `pre_shift`, `after`,
/// `gamma_raw`) instead of just the final value, so an offline tool can test
/// which intermediate matches a newly-found quantity without duplicating this
/// bit-exact arithmetic.
pub(crate) fn gamma_ref_exact_intermediates(
    biased: &[i16],
    l: usize,
    prev_gamma_raw: i16,
) -> (i16, i16, i32, i32, i32, i16) {
    let l = l.max(1);
    let take = l.min(biased.len());
    let total: i64 = biased.iter().take(take).map(|&v| v as i64).sum();
    // signed truncating divide; |sum| < |count| short-circuits to 0, matching
    // the reference's small-sum-vs-count edge case.
    let c = l as i64;
    let mean_i32: i32 = if total.abs() < c.abs() {
        0
    } else {
        (total / c) as i32
    };
    let mean16 = mean_i32 as i16;

    let l_ret = band_decompress::log2_fn(l as i16, 15);
    let shifted = band_decompress::shift_scale(l_ret, 0xf - 5);
    let rounded = sat_add32(shifted, 0x8000);
    let log2l16 = (rounded >> 16) as u16 as i16;

    let combined = sat_add32(
        ((mean16 as i32) << 16) as u32,
        ((log2l16 as i32) << 16) as u32,
    );
    let bias_shifted = ((prev_gamma_raw as i32) << 15) as u32;
    let pre_shift = sat_sub32(combined, bias_shifted);

    let after = band_decompress::shift_scale(pre_shift, 0);
    let rounded2 = sat_add32(after, 0x8000);
    let gamma_raw = (rounded2 >> 16) as u16 as i16;
    (
        mean16,
        log2l16,
        combined as i32,
        pre_shift as i32,
        after as i32,
        gamma_raw,
    )
}

#[inline]
fn sat_add32(a: u32, b: u32) -> u32 {
    let r = a.wrapping_add(b);
    let ov = (a ^ r) & (b ^ r) & 0x8000_0000;
    if ov != 0 {
        if (a as i32) < 0 {
            0x8000_0000
        } else {
            0x7FFF_FFFF
        }
    } else {
        r
    }
}
#[inline]
fn sat_sub32(a: u32, b: u32) -> u32 {
    let r = a.wrapping_sub(b);
    let ov = (a ^ b) & (a ^ r) & 0x8000_0000;
    if ov != 0 {
        if (a as i32) < 0 {
            0x8000_0000
        } else {
            0x7FFF_FFFF
        }
    } else {
        r
    }
}

/// The bit-exact, fully-derived (zero fitted parameters) `arg5`/`bias` update
/// formula -- the input [`gamma_ref_exact`] needs as `prev_gamma_raw`.
///
/// The gain routine's `arg5` (the bias pointer) is its caller's `arg1 + 0x84`, a
/// fixed per-run persistent context pointer, written once per frame. The write
/// instructions in the gain routine's tail, right after the gain-quantizer call:
///
/// ```text
/// ecx = sext16(*edi)              // *edi (outptr), AFTER the gain quantizer's
///                                    call. The quantizer overwrites *edi with
///                                    the DEQUANTIZED reconstruction, gain_o[b2]
///                                    -- NOT deref_arg0, the pre-quantization
///                                    value captured BEFORE that call runs.
/// eax = sext16(*esi)              // *esi = current bias (esi == arg5 throughout)
/// eax = eax + 2*ecx                // lea eax,[eax+ecx*2]
/// eax = (eax << 15) sar 16         // == eax >> 1 arithmetic, over the observed
///                                     value range (no 32-bit overflow on real
///                                     content)
/// ax  = min(eax as i16, 0x6800)    // signed clamp, ceiling 26624 -- there is no
///                                     floor clamp
/// *esi = ax                        // THE WRITE, gated on mode != 2 (mode==2 is
///                                     never observed at this call site)
/// ```
///
/// i.e. `bias_next = min(gain_o_q11_raw[b2] + (bias_prev >> 1), 0x6800)`.
///
/// **This is exactly `prev_gamma_raw = prev_gamma * 2048`** -- the same recursive
/// quantity [`crate::dequantize::decode_gain`] /
/// [`super::quantize::quantize_gain`] already track
/// (`gamma[n] = gain_o[b2][n] + 0.5*gamma[n-1]`), in the raw Q11 integer domain.
///
/// **The `b2` fed in must come from the same execution being modelled.** Chaining
/// `DecoderState::prev_gamma()` from a different `b2` source than the one that
/// actually produced the frame gives a mean absolute error around 7900 with no
/// resemblance to the real sequence -- a cross-source alignment mismatch, not a
/// wrong formula.
pub(crate) fn next_bias_raw_exact(prev_bias_raw: i16, b2: u8) -> i16 {
    let table = blip25_codebooks::gain_o();
    let gain_raw = table[b2 as usize] as i32;
    let combo: i32 = prev_bias_raw as i32 + 2 * gain_raw;
    // shl eax,0xf ; sar eax,0x10 -- exact 32-bit truncating-shift-left then
    // arithmetic-shift-right, matching x86 semantics bit-for-bit, including the
    // 32-bit wraparound (never exercised on real content, but load-bearing).
    let shifted = (combo as i64) << 15;
    let wrapped = (shifted & 0xFFFF_FFFF) as u32;
    let bias_i32 = (wrapped as i32) >> 16;
    let bias_i16 = bias_i32 as i16;
    const CEIL: i16 = 0x6800; // 26624 -- the reference's clamp ceiling
    if bias_i16 < CEIL {
        bias_i16
    } else {
        CEIL
    }
}

/// Direct, zero-calibration nearest-neighbor search against the `gain_o`
/// codebook's RAW Q11 integer table -- shared by both
/// [`candidate_gamma_ref_exact`] and
/// [`candidate_gamma_ref_exact_recursive_bias`].
pub(crate) fn nearest_gain_o_index(gamma_raw: i16) -> u8 {
    let table = blip25_codebooks::gain_o();
    let mut best_idx = 0usize;
    let mut best_dist = i32::MAX;
    for (i, &g) in table.iter().enumerate() {
        let d = (g as i32 - gamma_raw as i32).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx as u8
}

/// Same as [`candidate_gamma_ref_exact`], but drives `prev_gamma_raw` from this
/// function's own PREVIOUS output via [`next_bias_raw`]'s self-referential
/// recursion, rather than `prev_gamma_raw_candidate`'s decoder-state
/// approximation. The caller threads `bias_state` across frames, starting at
/// `0` to match the reference's observed frame-0 `bias=0`.
///
/// Returns `(b2, gamma_raw)`; `gamma_raw` becomes the next call's `bias_state`.
pub(crate) fn candidate_gamma_ref_exact_recursive_bias(
    biased: &[i16],
    l: usize,
    bias_state: i16,
) -> (u8, i16) {
    let gamma_raw = gamma_ref_exact(biased, l, bias_state);
    (nearest_gain_o_index(gamma_raw), gamma_raw)
}

/// Self-contained composition of
/// [`band_decompress::compute_outer`] + [`band_loudness_real_bins`] +
/// [`candidate_gamma_ref_exact_recursive_bias`] for [`super::Encoder`]'s
/// live diagnostic path (`Encoder::b2_x86_refexact_log`). It self-contains
/// `compute_outer`, like [`candidate_gamma_outer_only`], so callers do not need
/// this module's private `ARG3`. `win`
/// must be a freshly-built [`WinState`] for this frame (e.g.
/// [`win_from_gap2`]), not yet mutated by another `compute_outer` call.
/// Returns `(b2, gamma_raw)` -- `gamma_raw` is the caller's next
/// `bias_state` input.
pub(crate) fn candidate_gamma_ref_exact_from_win(
    win: &mut WinState,
    l: usize,
    arg2: i16,
    bias_state: i16,
) -> (u8, i16) {
    let outer = band_decompress::compute_outer(win, 0, arg2, ARG3);
    let biased = band_loudness_real_bins(win, l, outer);
    candidate_gamma_ref_exact_recursive_bias(&biased, l, bias_state)
}

/// Same per-band chain as [`band_loudness_real_bins`], but takes `step` directly
/// from the caller instead of [`super::step_table::step_for_count`]'s
/// `COUNT`-keyed modal lookup -- see [`super::step_recursive_fixed`] for the
/// recursive `STEP` update law a caller can thread in. `step_table.rs` itself is
/// untouched; this supersedes the modal table for this one path only.
pub(crate) fn band_loudness_real_bins_with_step(
    win: &WinState,
    l: usize,
    step_outer: i16,
    step_in: i16,
) -> Vec<i16> {
    let l_clamped = l.clamp(1, 56);
    let bins = assemble_bins_from_win(win);
    let raw = band_decompress::band_decompress(step_in, l_clamped, &bins, step_outer);
    raw.iter()
        .map(|&v| band_decompress::bias_clamp_one(v))
        .collect()
}

/// Same composition as [`candidate_gamma_ref_exact_from_win`] but via
/// [`band_loudness_real_bins_with_step`] -- see that function's doc. Returns
/// `(b2, gamma_raw)`, the same convention.
pub(crate) fn candidate_gamma_ref_exact_from_win_recursive_step(
    win: &mut WinState,
    l: usize,
    arg2: i16,
    bias_state: i16,
    step_in: i16,
) -> (u8, i16) {
    let outer = band_decompress::compute_outer(win, 0, arg2, ARG3);
    let biased = band_loudness_real_bins_with_step(win, l, outer, step_in);
    candidate_gamma_ref_exact_recursive_bias(&biased, l, bias_state)
}

/// The affine calibration for [`candidate_gamma_real_bins`] -- see that
/// function's doc for the grid search these defaults come from.
///
/// Tied to `band_decompress::compute_outer`'s exact behaviour: it applies
/// `array_a_stage2::inverse_fft_butterfly_stage`'s array mutation, which the
/// magnitude-squared assembler reads AFTER, and that sets `mean_raw`'s numeric
/// scale. Changing `compute_outer` invalidates these constants.
///
/// Coarse grid only (power-of-2 x1.5); the sibling
/// [`realbins_realstep_calib_a`] got the finer search. Scores 149/525 (28.38%)
/// pooled.
fn realbins_calib_a() -> f64 {
    0.00001144
}
/// See [`realbins_calib_a`].
fn realbins_calib_b() -> f64 {
    8.000
}

/// The raw, un-calibrated mean bias-clamped `band_decompress` output for the
/// real-`bins` chain -- exposed separately so a calibration grid search can
/// compute it ONCE per frame and cheaply
/// replay just the affine-fit + `quantize_gain` recursion for many `(a,
/// b)` candidates, instead of re-running the whole encoder per grid point
/// (mirrors [`raw_outer`]'s own reason for existing).
pub(crate) fn mean_raw_real_bins(win: &mut WinState, l: usize, arg2: i16) -> f64 {
    let outer = band_decompress::compute_outer(win, 0, arg2, ARG3);
    let biased = band_loudness_real_bins(win, l, outer);
    let n = biased.len().max(1) as f64;
    biased.iter().map(|&v| v as f64).sum::<f64>() / n
}

/// Same per-band chain as
/// [`mean_raw_real_bins`]/[`candidate_gamma_real_bins`], but takes `step`
/// directly from the caller via [`band_loudness_real_bins_with_step`] instead of
/// [`super::step_table::step_for_count`]'s `COUNT`-keyed modal lookup, which is
/// only ~57-59% exact on real speech.
///
/// This pairs the recursive `STEP` law with REAL_BINS's separately-tuned affine
/// calibration + [`super::quantize::quantize_gain`] recursion. The other
/// recursive-`STEP` paths pair it instead with
/// [`candidate_gamma_ref_exact_from_win_recursive_step`]'s different final
/// chain ([`gamma_ref_exact`] + the approximate `prev_gamma_raw` IIR echo + a
/// raw-codebook nearest-neighbor search).
///
/// ## Accuracy
///
/// With REAL_BINS's calibration constants unchanged (tuned for the
/// step_table-driven signal's different scale): **24.0% pooled** over 525 anchor
/// frames -- `mark` 3.0%, `cpvbad` 1.5%, `dtone_10` 93.6%. The entire gain over
/// REAL_BINS's 20.4% is in `dtone_10` (78.4% -> 93.6%); `mark`/`cpvbad` are
/// byte-for-byte unchanged, because the recursive `STEP` law beats the modal
/// table specifically on that file's near-constant, low-entropy content.
///
/// With a dedicated re-calibration for this signal's scale
/// ([`realbins_realstep_calib_a`]/[`realbins_realstep_calib_b`]): **30.1%
/// pooled** (158/525) -- `mark` 3.0% (still at chance), `cpvbad` 14.5% (above
/// the ~3.1% chance floor but below `cpvbad`'s 15.5% trivial baseline),
/// `dtone_10` 98.4% (exactly its trivial "always guess the modal value"
/// baseline).
///
/// So this is the best mechanism-level pooled score, and it still does not beat
/// either pooled trivial baseline (33.0%/36.0%), with `mark` at chance. The
/// calibration constants are fit in-sample on the same 3 anchor files
/// `validate_encode` scores against and are not verified to generalize.
pub(crate) fn mean_raw_real_bins_with_step(
    win: &mut WinState,
    l: usize,
    arg2: i16,
    step_in: i16,
) -> f64 {
    let outer = band_decompress::compute_outer(win, 0, arg2, ARG3);
    let biased = band_loudness_real_bins_with_step(win, l, outer, step_in);
    let n = biased.len().max(1) as f64;
    biased.iter().map(|&v| v as f64).sum::<f64>() / n
}

/// [`realbins_realstep_calib_a`]/[`realbins_realstep_calib_b`] -- this path's
/// own separately tuned affine calibration, see
/// [`mean_raw_real_bins_with_step`] -- applied to that function's output.
pub(crate) fn candidate_gamma_real_bins_with_step(
    win: &mut WinState,
    l: usize,
    arg2: i16,
    step_in: i16,
) -> f64 {
    let mean_raw = mean_raw_real_bins_with_step(win, l, arg2, step_in);
    realbins_realstep_calib_a() * mean_raw + realbins_realstep_calib_b()
}

/// The affine calibration for [`candidate_gamma_real_bins_with_step`] -- see
/// that function's doc for the grid search these defaults come from.
///
/// Kept separate from [`realbins_calib_a`], which is tuned for the
/// step_table-driven raw signal's different numeric scale. Reusing that one
/// unchanged still scores 24.0%, but a fresh fit does better. Each experimental
/// path gets its own calibration rather than sharing one tuned for a different
/// raw signal.
///
/// Same dependency on `band_decompress::compute_outer`'s exact behaviour as
/// [`realbins_calib_a`]. Three-stage grid search converging to a=0.00000785,
/// b=8.020: **158/525 (30.10%) pooled** -- mark 3.0%, cpvbad 14.5%,
/// dtone_10 98.4%.
fn realbins_realstep_calib_a() -> f64 {
    0.00000785
}
/// See [`realbins_realstep_calib_a`].
fn realbins_realstep_calib_b() -> f64 {
    8.020
}

/// Identical composition to [`mean_raw_real_bins_with_step`], EXCEPT `bins` is
/// assembled from a SEPARATE, already-mutated `win` array: `win_slot1`, the
/// caller's `win_from_gap2(&gap2_slot1)` already run through
/// [`band_decompress::compute_outer`] for its array-mutation side effect and its
/// own slot-1-appropriate `arg2` (see [`arg2_from_recent_audio`]).
///
/// The reference builds a SECOND, genuinely different window ("slot 1") every frame,
/// at the SAME `gap2_mid` timing as slot 0's, immediately adjacent in the ring
/// (offset 108 vs 28) -- see `history_ring::GAP2_OFFSET_SLOT1`.
///
/// `outer` is NOT recomputed from `win_slot1`: it is passed in from the caller's
/// slot-0 computation, because `combine_outer`'s formula depends on slot 0's
/// `fft_bfp_transform` return alone. Only `bins`' source window is substituted.
pub(crate) fn mean_raw_real_bins_slot1_with_step(
    win_slot1_mutated: &WinState,
    l: usize,
    outer: i16,
    step_in: i16,
) -> f64 {
    let biased = band_loudness_real_bins_with_step(win_slot1_mutated, l, outer, step_in);
    let n = biased.len().max(1) as f64;
    biased.iter().map(|&v| v as f64).sum::<f64>() / n
}

/// [`realbins_slot1_realstep_calib_a`]/
/// [`realbins_slot1_realstep_calib_b`] applied to
/// [`mean_raw_real_bins_slot1_with_step`]'s output.
pub(crate) fn candidate_gamma_real_bins_slot1_with_step(
    win_slot1_mutated: &WinState,
    l: usize,
    outer: i16,
    step_in: i16,
) -> f64 {
    let mean_raw = mean_raw_real_bins_slot1_with_step(win_slot1_mutated, l, outer, step_in);
    realbins_slot1_realstep_calib_a() * mean_raw + realbins_slot1_realstep_calib_b()
}

/// REALBINS_SLOT1_REALSTEP's spectrum (`bins` from slot 1's window) plus the
/// bit-exact recursive `STEP` law, fed through [`gamma_ref_exact`]'s
/// fully-derived, zero-fitted-parameter formula (with
/// [`next_bias_raw_exact`] supplying its bias input) instead of
/// [`candidate_gamma_real_bins_slot1_with_step`]'s 2-parameter grid-fitted
/// affine map.
///
/// Returns `(b2, gamma_raw)`, the same convention as
/// [`candidate_gamma_ref_exact_recursive_bias`]: the caller threads the
/// resulting `b2` into [`next_bias_raw_exact`] for the next frame's
/// `bias_state`, exactly as the REF_EXACT path does.
pub(crate) fn candidate_gamma_ref_exact_slot1_with_step(
    win_slot1_mutated: &WinState,
    l: usize,
    outer: i16,
    step_in: i16,
    bias_state: i16,
) -> (u8, i16) {
    let biased = band_loudness_real_bins_with_step(win_slot1_mutated, l, outer, step_in);
    let gamma_raw = gamma_ref_exact(&biased, l, bias_state);
    (nearest_gain_o_index(gamma_raw), gamma_raw)
}

/// The affine calibration for [`candidate_gamma_real_bins_slot1_with_step`].
/// `bins`' raw numeric scale differs from [`realbins_realstep_calib_a`]'s signal
/// (a different window, same formula), so this gets its own tuned constants
/// rather than reusing those.
///
/// Three-stage grid search: a=0.00000915, b=8.000, **159/525 (30.29%) pooled**
/// -- mark 4.0% (8/200), cpvbad 14.0% (28/200), dtone_10 98.4% (123/125). A
/// marginal, NOT decisive gain over REALBINS_REALSTEP's 158/525 (mark +1,
/// cpvbad -1, net wash), still below both pooled trivial baselines
/// (33.0%/36.0%).
fn realbins_slot1_realstep_calib_a() -> f64 {
    0.00000915
}
/// See [`realbins_slot1_realstep_calib_a`].
fn realbins_slot1_realstep_calib_b() -> f64 {
    8.000
}

/// Identical composition to [`mean_raw_real_bins_slot1_with_step`], except
/// `bins` is assembled from SLOT 2's window: `win_slot2`, the caller's
/// [`super::win_taper_wide::win_from_gap2_slot2`]`(&self.gap2_slot2)` output run
/// through its own [`band_decompress::compute_outer`] call for its
/// content-appropriate `arg2` (see [`arg2_from_recent_audio`]).
///
/// `outer` is NOT recomputed from `win_slot2`: it is passed in from the caller's
/// slot-0 computation, same as slot 1's path, because `combine_outer`'s formula
/// depends on slot 0's `fft_bfp_transform` return alone.
pub(crate) fn mean_raw_real_bins_slot2_with_step(
    win_slot2_mutated: &WinState,
    l: usize,
    outer: i16,
    step_in: i16,
) -> f64 {
    let biased = band_loudness_real_bins_with_step(win_slot2_mutated, l, outer, step_in);
    let n = biased.len().max(1) as f64;
    biased.iter().map(|&v| v as f64).sum::<f64>() / n
}

/// [`realbins_slot2_realstep_calib_a`]/
/// [`realbins_slot2_realstep_calib_b`] applied to
/// [`mean_raw_real_bins_slot2_with_step`]'s output.
pub(crate) fn candidate_gamma_real_bins_slot2_with_step(
    win_slot2_mutated: &WinState,
    l: usize,
    outer: i16,
    step_in: i16,
) -> f64 {
    let mean_raw = mean_raw_real_bins_slot2_with_step(win_slot2_mutated, l, outer, step_in);
    realbins_slot2_realstep_calib_a() * mean_raw + realbins_slot2_realstep_calib_b()
}

/// The affine calibration for [`candidate_gamma_real_bins_slot2_with_step`].
/// `bins`' raw numeric scale differs from slot 0/1's signal (a wider window, 255
/// words not 199, and a different ramp table), so this gets its own tuned
/// constants.
///
/// Four-stage grid search converging to a=0.00000517, b=8.060: **156/525
/// (29.71%) pooled** -- mark 3.0% (6/200), cpvbad 13.5% (27/200), dtone_10
/// 98.4% (123/125).
fn realbins_slot2_realstep_calib_a() -> f64 {
    0.00000517
}
/// See [`realbins_slot2_realstep_calib_a`].
fn realbins_slot2_realstep_calib_b() -> f64 {
    8.060
}

/// [`realbins_realstep_gateref_calib_a`]/
/// [`realbins_realstep_gateref_calib_b`] applied to
/// [`mean_raw_real_bins_with_step`]'s output, for the
/// REALBINS_REALSTEP_GATEREF path: REALBINS_REALSTEP with the captured
/// ACCEPT/REJECT gate reference driving `STEP` instead of always assuming
/// REJECT-FORMULA.
///
/// This gets its OWN function and calibration pair, separate from
/// [`candidate_gamma_real_bins_with_step`], because gate-corrected `STEP` has a
/// materially different numeric distribution (occasional literal `4217` jumps on
/// ACCEPT frames). Reusing the always-recursive path's calibration unchanged
/// regresses `dtone_10` from 98.4% to 0.8%.
pub(crate) fn candidate_gamma_real_bins_with_gated_step(
    win: &mut WinState,
    l: usize,
    arg2: i16,
    step_in: i16,
) -> f64 {
    let mean_raw = mean_raw_real_bins_with_step(win, l, arg2, step_in);
    realbins_realstep_gateref_calib_a() * mean_raw + realbins_realstep_gateref_calib_b()
}

/// The affine calibration for [`candidate_gamma_real_bins_with_gated_step`],
/// grid-searched against `realbins_realstep_gateref_meanraw_log` with the
/// gate reference applied. A grid search that forgets to call
/// `set_step_gate_ref` silently fits always-REJECT-FORMULA data instead.
///
/// Converges to the same 158/525 (30.10%) pooled score as REALBINS_REALSTEP,
/// with an identical mark/cpvbad/dtone_10 breakdown (3.0%/14.5%/98.4%) -- a tie,
/// not an improvement.
fn realbins_realstep_gateref_calib_a() -> f64 {
    0.00000737
}
/// See [`realbins_realstep_gateref_calib_a`].
fn realbins_realstep_gateref_calib_b() -> f64 {
    8.020
}

/// Cross-frame `win`/`a1ptr` state for
/// [`super::loudness_transform::fft_bfp_transform`], zero-seeded.
///
/// **`a1ptr` is FRESH SCRATCH each active call, not a persistent accumulator.**
/// `fft_bfp_transform`'s in-place recursive transform is the array's only writer
/// of meaningful content; there is no carry-over from the previous frame's end
/// state, and the array is never reseeded from raw audio, prefiltered audio, or
/// any other external buffer. Nothing in this crate's already-bit-exact chain
/// (`audio_prefilter`/`HistoryRing`/`windowed_taper`/`real_fft32`) is its source.
///
/// Two traps when re-measuring that:
/// * **`a1ptr`'s true extent is 246 `i16` words, not 256 or 512.** The owning
///   per-frame encode driver's prologue is `sub esp,0x208` with `a1ptr` at frame
///   offset `0x1c`, so `(0x208-0x1c)/2 = 246`. A 512-word capture window reads up
///   to 266 words of unrelated CALLER-frame memory that sits untouched between
///   nearby snapshots regardless of what `a1ptr` does.
/// * **Some `a1ptr` calls are entirely zero** -- a real silent-cycle behaviour,
///   not an artifact. Comparing two inactive cycles gives a near-total
///   zero-equals-zero "match" that looks like persistence. Restricted to the true
///   246-word bound and to real-to-real transitions (both sides nonzero,
///   magnitudes up to ~6100), the match rate is 0-2 out of 246 -- at or below
///   chance for small-magnitude signed data.
///
/// Since the per-call content is undefined leftover stack on the test process
/// and there is no raw-PCM formula for a seed, this port seeds `WinState` at
/// all-zero: the only deterministic choice available, and the one matching
/// [`super::audio_prefilter`]'s documented `Reset()` convention for the other
/// persistent reference state.
///
/// **An all-zero `win` is a permanent fixed point of `fft_bfp_transform`.** Every
/// array-mutating step is zero-preserving (`bitrev_permute` is a pure
/// permutation; `prescale`/`twiddle_pair` are pure scale/combine with no additive
/// constant), and `arg2` is used only in the final scalar return
/// (`i16t(arg2 - acc)`), never written into the array -- so a zero-seeded `win`
/// stays exactly zero forever, for any sequence of `arg2` values. See the
/// `zero_seeded_window_is_permanent_fixed_point` test.
///
/// **Do not rebuild `win` from `spec` each frame.** Injecting spectrum-derived
/// content into `win` has no basis in the reference's behaviour, and it destroys
/// the state that should carry over. `win` is owned by the caller
/// ([`super::Encoder`]'s `win_fixed`/`win_x86_pf` fields), zero-seeded, threaded
/// through, and mutated in place by `fft_bfp_transform` each frame.
///
/// For the paths that DO have a real per-frame `win`, see [`win_from_gap2`],
/// which supersedes the zero-fixed-point analysis above.
pub(crate) type WinState = [i16; 256];

/// Fresh, all-zero `WinState` -- see [`WinState`] for why zero, not the x86
/// harness's observed garbage, is the correct seed.
pub(crate) fn new_win_state() -> WinState {
    [0i16; 256]
}

/// Build `win` from `gap2`. This supersedes [`WinState`]'s "zero forever, `win`
/// is a structural no-op" analysis for this candidate path.
///
/// `a1ptr`/`win` has a real per-frame write site: a 6-argument taper driver
/// called directly inside the spectrum stage, with `a1ptr` as its `dest` and the
/// same `localbuf`/`gap2` array [`arg2_from_recent_audio`] consumes (via the
/// block-exponent normalize) as its `src`. So **`win` is not a zero-forever
/// no-op** -- it is freshly computed on every active call from the same
/// content-derived `gap2` array this module already threads through for `arg2`.
///
/// The driver writes two arms and leaves a gap:
/// * the ascending arm at indices `1..=99` plus the special center sample at
///   index `0`;
/// * the descending arm at the array's LAST `HALFCOUNT` words, absolute indices
///   `256-HALFCOUNT..256` (`157..256` for `count=199`/`HALFCOUNT=99`) -- **not**
///   indices `~100-198`;
/// * a genuine structural 57-word gap at indices `100..157` that the driver never
///   writes on this call.
///
/// See [`super::win_taper`] for the derivation. `win_from_gap2` assembles both
/// arms plus the zero gap.
///
/// Because `win` is fresh scratch on each active call rather than a persistent
/// accumulator, this function takes and returns no cross-frame state: callers
/// must rebuild `win` from it EVERY frame (see [`super::Encoder`]'s
/// `b2_x86_wintaper_log` wiring), not carry a `mut WinState` across calls the way
/// the other two experimental paths do.
///
/// `gap2` is the same `history_ring::HistoryRing::gap2_window()` output
/// [`arg2_from_recent_audio`] consumes. This function's transform is bit-exact
/// GIVEN a correct `gap2` and inherits whatever error `gap2` itself carries --
/// see `history_ring::GAP2_OFFSET` for those caveats.
pub(crate) fn win_from_gap2(gap2: &[i16; super::history_ring::GAP2_LEN]) -> WinState {
    let win = super::win_taper::WinArray::from_raw_gap2(gap2);
    let mut out = [0i16; 256];
    out[0..=super::win_taper::HALFCOUNT].copy_from_slice(&win.first_half);
    // The gap between the two arms (indices `HALFCOUNT+1..256-HALFCOUNT`) is a
    // structural fact -- the driver never writes there on this call -- and is
    // left at zero deliberately, not a missing finding.
    let desc_start = 256 - super::win_taper::HALFCOUNT;
    out[desc_start..256].copy_from_slice(&win.second_half);
    out
}

/// Frame-aware power + scale sharing the a5/loudness fft_bfp_transform per-frame stage count
/// `k` with the x-arm. `multi_stage=false` is the round-164 stage-1-only fft_bfp_transform
/// (k=1, the window-derivable transform for the 378 non-flush calls);
/// `multi_stage=true` is the full multi-stage fft_bfp_transform (k=6, the end-of-input FLUSH
/// frames 189-198). Returns (`[i32;129]` Nyquist-inclusive power = the y-arm's
/// `arg3`, `scale` = butterfly-stage return * 2 + 30 = the arm block-float
/// scale) so BOTH the power AND its scale track the same k (they are one
/// `fft_bfp_transform` + butterfly-stage pass).
///
/// ## Root cause of the per-frame k
///
/// The `fft_bfp_transform` group loop has NO data-dependent early-exit: for a
/// fixed `arg3` it is structurally all-6-stages
/// (`dx = n_pairs>>(stage+1)`, `>0` for stages 1..6). The a5/loudness
/// FFT-stage call passes a **constant** `arg3=8` (=> `fft_bfp_transform`
/// `arg3=7`) on every frame (the FFT-stage arg capture shows a3=8, 400/400).
/// Yet the DLL's OWN captured `fft_bfp_transform` pre/post
/// arrays (`[outer-fft_bfp_transform-top]`/`[outer-fft_bfp_transform-top-post]`, harness_a370tail) show
/// the output is **k=1 (stage-1-only) on frames <189 and k=6 (all-6-stages) on
/// frames 189-198** -- exactly binary, never intermediate, and the flip lands at
/// frame 188.call3/189 **IDENTICALLY in voiced.pcm AND mark.pcm** (two different
/// recordings). An audio threshold cannot coincide across two clips; and si0 /
/// input block-exp / peak / maxabs all OVERLAP between k=1 and k=6 frames (e.g.
/// f185 k=1 peakbits=30 maxabs=21371 vs f189 k=6 peakbits=30 maxabs=22791) --
/// so k is NOT window-magnitude-derivable.
///
/// **Decisive experiment**: re-running with `nframes=150` (the file still has
/// audio past frame 150) leaves frames 140-149 **k=1** -- no flush appears. So
/// k=6 is NOT "last-N-processed"; it is triggered when the frame's window
/// LOOK-AHEAD (`+160` GAP, taps 115..198) runs **past the end of the input
/// audio**. The 200-frame file processed at 199 makes frames ~189-198's
/// look-ahead exhaust the input => the encoder flushes those final frames and
/// fft_bfp_transform runs the full multi-stage transform. This is an **end-of-input flush**,
/// positional relative to the input's last sample: it moves with input length
/// and is NOT a function of the current fft_bfp_transform window content. The
/// `latch=189` callers use is that flush boundary for these 200-frame files; a
/// general encoder sets it from `frame_lookahead_end > input_len`.
pub(crate) fn frame_power_and_scale(
    gap2: &[i16; super::history_ring::GAP2_LEN],
    multi_stage: bool,
) -> ([i32; 129], i16) {
    let mut win = win_from_gap2(gap2);
    let exp0 = super::block_exponent::block_exponent(gap2);
    let a2 = exp0.saturating_sub(7);
    let fft_exp = if multi_stage {
        super::loudness_transform::fft_bfp_transform(&mut win, 0, a2, 7)
    } else {
        super::loudness_transform::loudness_array_transform(&mut win, 0, a2, 7)
    };
    let mut scratch = [0i16; 258];
    scratch[..256].copy_from_slice(&win[..256]);
    let scale_ret =
        super::array_a_stage2::inverse_fft_butterfly_stage(&mut scratch, fft_exp, 8, 0, 1);
    let mut win256 = [0i16; 256];
    win256.copy_from_slice(&scratch[..256]);
    let mut bins = assemble_bins_from_win(&win256);
    let re = scratch[256] as i64;
    let im = scratch[257] as i64;
    bins[128] = (2 * (re * re + im * im)) as i32;
    let scale = (scale_ret.wrapping_mul(2) as i32 + 30) as i16;
    (bins, scale)
}

/// The mechanism-matched (`step`/`outer`/`band_decompress`) chain's per-band
/// output, bias-clamped -- for diagnostics. `l` is the harmonic count (`COUNT`,
/// [`step_table::step_for_count`]'s input; `l_us` from `pitch::estimate_pitch`
/// shares `step`'s observed `COUNT` range, `9..=56`). `win` is the caller-owned,
/// cross-frame-persistent `WinState` (see its doc), mutated in place by this
/// call to match the reference's self-referential accumulator. `arg2` is
/// `fft_bfp_transform`'s second argument -- see [`arg2_from_recent_audio`] for
/// how to derive a content-responsive value.
pub(crate) fn band_loudness_fixed(
    spec: &Spectrum,
    l: usize,
    win: &mut WinState,
    arg2: i16,
) -> Vec<i16> {
    let l_clamped = l.clamp(1, 56);
    let bins = assemble_bins(spec);
    let step = step_table::step_for_count(l_clamped as u32);
    let outer = band_decompress::compute_outer(win, 0, arg2, ARG3);
    let raw = band_decompress::band_decompress(step, l_clamped, &bins, outer);
    raw.iter()
        .map(|&v| band_decompress::bias_clamp_one(v))
        .collect()
}

/// Candidate `gamma` (overall frame gain, the quantity [`super::quantize::
/// quantize_gain`] needs) from the mechanism-matched chain above, via the
/// documented affine calibration. **Experimental, approximate** -- see
/// module doc. `win` is the caller-owned, cross-frame-persistent
/// `WinState` -- see its own doc. `arg2` -- see
/// [`arg2_from_recent_audio`]'s doc.
pub(crate) fn candidate_gamma(spec: &Spectrum, l: usize, win: &mut WinState, arg2: i16) -> f64 {
    let biased = band_loudness_fixed(spec, l, win, arg2);
    let n = biased.len().max(1) as f64;
    let mean_raw: f64 = biased.iter().map(|&v| v as f64).sum::<f64>() / n;
    gamma_calib_a() * mean_raw + gamma_calib_b()
}

/// A structurally simpler candidate `gamma` that skips
/// `bins`/`step`/`band_decompress` entirely and uses ONLY the `outer` scalar
/// (`band_decompress::compute_outer`, i.e.
/// [`super::loudness_transform::fft_bfp_transform`] composed with
/// [`super::outer_transform::combine_outer`]).
///
/// `outer` is composed entirely of exact pieces -- `win`/`arg2` (bit-exact over
/// 34,000+ captured samples, both content and the derived `arg2` value),
/// `fft_bfp_transform`, and `combine_outer` -- so it is a fully
/// mechanism-matched per-frame scalar, unlike `candidate_gamma` above, which
/// still depends on `bins`'s synthesized-spectrum sourcing and `step`'s
/// 71.5%-exact lookup table.
///
/// This tests whether `outer` alone (a small signed exponent, observed range
/// roughly `-2..-42`) carries enough loudness signal to beat the chance floor,
/// via an affine fit with the same status as `gamma_calib_a`/`b`: a statistical
/// calibration, not a recovered formula for the bins/step machinery it bypasses.
pub(crate) fn candidate_gamma_outer_only(win: &mut WinState, arg2: i16) -> f64 {
    let outer = raw_outer(win, arg2);
    outer_calib_a() * outer as f64 + outer_calib_b()
}

/// The raw, un-calibrated `outer` scalar itself -- exposed separately so a
/// calibration grid search can compute it ONCE per frame and cheaply replay just
/// the affine-fit +
/// `quantize_gain` recursion for many `(a, b)` candidates, instead of
/// re-running the whole encoder per grid point.
pub(crate) fn raw_outer(win: &mut WinState, arg2: i16) -> i16 {
    band_decompress::compute_outer(win, 0, arg2, ARG3)
}

fn outer_calib_a() -> f64 {
    0.1
}
fn outer_calib_b() -> f64 {
    0.0
}


// ---------------------------------------------------------------------
// The full end-to-end `ptr63c` ASSEMBLY. See `gamma_poly_generate_approx` below for
// the scope: what is verified against capture versus approximated.

/// The magnitude-squared assembler's `dest[i] = 2*(re^2+im^2)` formula, applied
/// to a [`super::real_fft32::real_fft32`] output buffer's first 16 complex pairs
/// (bins 0..15).
///
/// A local re-derivation, NOT a reuse of
/// [`super::voicing_fixed::magsq_assemble_bins`]: that one's `src` array is 64
/// `i16` wide, a different call site's wider window. The `ptr63c` call site
/// consumes a 32-`i16` `real_fft32` output directly.
fn gamma_poly_magsq16(fft_out: &[i16; 32]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        let re = i64::from(fft_out[2 * i]);
        let im = i64::from(fft_out[2 * i + 1]);
        let mag2 = 2 * (re * re + im * im);
        *slot = mag2 as i32; // truncating cast, matches the real 32-bit store
    }
    out
}

/// One outer*sub PASS's fresh 16-word block (`local_90[0..15]`):
/// - `word[0] = max(0, real_fft32_output[0])` -- bin 0's real/DC part, clamped
///   non-negative. It is NOT a clamped magsq bin, and it reads `buf[0]`/bin0,
///   not `buf[2]`/bin1: the push-order/esp-drift arithmetic makes that
///   off-by-one easy to reach from the disassembly alone.
/// - `word[1..=15] = `[`block_float_word_rescale`]`(magsq16[i], a2, a3)` where
///   `magsq16` is [`gamma_poly_magsq16`] applied to the SAME pass's `real_fft32`
///   output.
pub(crate) fn gamma_poly_pass_block(fft_out: &[i16; 32], a2: i16, a3: i16) -> [i16; 16] {
    let mut out = [0i16; 16];
    out[0] = fft_out[0].max(0);
    let magsq = gamma_poly_magsq16(fft_out);
    for i in 1..16 {
        out[i] = block_float_word_rescale(magsq[i], a2, a3);
    }
    out
}

/// The `a2`/`a3` scalar pair fed to every one of a pass's 15
/// [`block_float_word_rescale`] calls.
///
/// From the pass function's entry: `ecx=[esp+0x2c]`, read after its 4 register
/// pushes -- this is **`param_5`**, not `param_9`/band-count. Then
/// `[esp+0x18] = zero_extend16(2*param_5)`, computed ONCE before the outer*sub
/// loop nest rather than per-pass, becomes `a2`, and `param_5` itself,
/// sign-extended, becomes `a3`.
///
/// **`a3` is not an independent "half of a2" quantity.** `a3 == a2/2` holds only
/// because both derive from the same `param_5`; they are not two separately
/// computed values.
///
/// **What this does NOT close**: `param_5`'s own raw-audio derivation.
/// `param_5` is read from the pass function's caller (`push ecx` where
/// `ecx=[esp+0x2c]`) inside the voicing inner driver, and `[esp+0x2c]` is never
/// written anywhere in that driver's body up to that point -- so it is itself an
/// incoming argument of the voicing inner driver, one hop further up a caller
/// chain that is a separate open investigation.
/// [`ptr63c_a2a3_silence`] is the flagged fallback until that closes.
pub(crate) fn gamma_poly_scale_pair(param5: i16) -> (i16, i16) {
    let a3 = param5;
    let a2_u = (2i32.wrapping_mul(i32::from(param5)) as u32 & 0xffff) as u16;
    let a2 = a2_u as i16;
    (a2, a3)
}

/// Silence-value fallback for [`gamma_poly_scale_pair`]'s open `param5` input.
/// `param5=0` gives `(a2,a3)=(0,0)`, i.e. no shift or scale inside
/// [`block_float_word_rescale`] -- the reference's natural "no adjustment" case,
/// not an arbitrary zero. Same convention as `ARG2_APPROX_SILENCE`.
pub(crate) const fn ptr63c_a2a3_silence() -> (i16, i16) {
    (0, 0)
}

/// The full `ptr63c` ring ASSEMBLY: concatenate 20 passes'
/// [`gamma_poly_pass_block`] output (320 words) and slice out words `[240..280)`,
/// the 40-word window that becomes `ptr63c`'s final content on the LAST pass's
/// snapshot. Given the reference's 20 captured `local_90[0..15]` blocks for one
/// pass-function call, this reproduces the reference's final `ptr63c` content
/// bit-exactly on every non-near-silent captured call across both files.
///
/// **Scope of this function specifically**: `fft_windows` -- 20 `[i16; 32]`
/// buffers, one per pass, already run through
/// [`super::windowed_taper::windowed_taper`] +
/// [`super::real_fft32::real_fft32`] by the caller -- is the one input not
/// independently re-derived from raw PCM for this call site.
/// [`gamma_poly_generate_approx`] uses `analysis_stage_window::analysis_windows`
/// anyway, as a flagged approximation: the geometry is closed, the content is
/// not proven bit-exact at this call site.
pub(crate) fn gamma_poly_assemble(fft_windows: &[[i16; 32]; 20], a2: i16, a3: i16) -> [i16; 40] {
    let mut stream = [0i16; 320];
    for (p, win) in fft_windows.iter().enumerate() {
        let block = gamma_poly_pass_block(win, a2, a3);
        stream[p * 16..p * 16 + 16].copy_from_slice(&block);
    }
    let mut out = [0i16; 40];
    out.copy_from_slice(&stream[240..280]);
    out
}

/// EXPERIMENTAL end-to-end candidate: [`gamma_poly_assemble`] fed with
/// [`super::analysis_stage_window::analysis_windows`]'s 20 windows -- built from
/// `stage`, the bit-exact 108-word staging buffer -- run through
/// `windowed_taper` + `real_fft32`, with [`ptr63c_a2a3_silence`] for the open
/// `a2`/`a3` input.
///
/// This tests whether `ptr63c`'s content is the missing spectral-richness signal
/// both the loudness and voicing chains point at. See [`gamma_poly_assemble`] for
/// the scope of what is and is not independently re-derived from raw PCM.
/// Returns the raw 40-word candidate array; [`candidate_gamma_poly`] turns it
/// into a `b2` guess.
pub(crate) fn gamma_poly_generate_approx(
    stage: &[i16; super::analysis_stage_window::STAGE_LEN],
) -> [i16; 40] {
    let windows = super::analysis_stage_window::analysis_windows(stage);
    let mut fft_windows = [[0i16; 32]; 20];
    for (i, w) in windows.iter().enumerate() {
        let tapered = super::windowed_taper::windowed_taper(w);
        let mut buf = tapered;
        super::real_fft32::real_fft32(&mut buf);
        fft_windows[i] = buf;
    }
    let (a2, a3) = ptr63c_a2a3_silence();
    gamma_poly_assemble(&fft_windows, a2, a3)
}

/// Affine calibration mapping `gamma_poly_generate_approx`'s mean-abs content onto
/// the `gain_O` codebook's numeric range. **Not fitted** -- a round-number
/// starting point, the same status `spec_scale`/`gamma_calib_a` had before their
/// grid searches.
fn gamma_poly_calib_a() -> f64 {
    1.0 / 64.0
}
fn gamma_poly_calib_b() -> f64 {
    0.0
}

/// [`gamma_poly_generate_approx`]'s mean-abs content, affine-calibrated
/// ([`gamma_poly_calib_a`]/`b`) and matched to the nearest `gain_O` codebook entry
/// via [`nearest_gain_o_index`] -- the candidate `b2` signal from the `ptr63c`
/// chain. Judge it against the trivial baselines
/// [`candidate_gamma_real_bins`] documents, not raw percentages.
pub(crate) fn candidate_gamma_poly(stage: &[i16; super::analysis_stage_window::STAGE_LEN]) -> u8 {
    let words = gamma_poly_generate_approx(stage);
    let mean_abs: f64 = words.iter().map(|&w| f64::from(w).abs()).sum::<f64>() / words.len() as f64;
    let gamma_raw = gamma_poly_calib_a() * mean_abs + gamma_poly_calib_b();
    nearest_gain_o_index(gamma_raw.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
}

/// The formula for the pass function's `param5` input:
/// `param5 == int16(ebx_seed + block_exponent(bufA[0..108]))`, given a
/// caller-supplied `stage` (the bit-exact 108-word staging window).
///
/// **`bufA` is a substitution, not an independent re-derivation.** `bufA`'s
/// source ADDRESS is exactly `encoder_buf+0x15ce`, bit-for-bit the same address
/// [`super::history_ring::STAGE_OFFSET`] reads (`stage`'s address), on 400 live
/// hits across both files with zero exceptions. But `bufA`'s CONTENT is not
/// confirmed against a from-scratch offline raw-PCM replay of this crate's
/// `HistoryRing`: a brute-force search across all 200 half-push states finds
/// zero non-trivial exact matches on either file. That is the same staging-buffer
/// cadence anomaly the `*_STAGE_AT_FRAMEn` fixtures show, and it blocks live
/// `bufA` content reconstruction too.
///
/// `stage`/`current_stage()` is independently proven bit-exact by same-instant
/// reads and cross-validated against 3 other closed consumers of the identical
/// ring+prefilter chain (`gap2_window`/`gap2_window_slot1`/`gap2_window_slot2`),
/// so using it here is a justified best-available substitution -- but do not read
/// it as "bufA fully closed".
///
/// `ebx_seed` is [`GAMMA_POLY_SEED_CONSTANT`], a CONSTANT `0` on every one of 800
/// captured voicing-inner-driver calls across both files and two independent
/// harnesses. That is a structural artifact, **not** the `ctx+0x6dc`-derived
/// pitch-lag quotient (`local_1cc`): `ebx` is repurposed twice as a plain loop
/// counter between the point where it briefly equals `local_1cc` and the read
/// site, and the second reuse (`dec ebx; jne`) is a counted loop that by
/// construction always leaves `ebx==0` on normal exit, whatever the content.
/// `local_1cc` (confirmed always `10` for real audio) is still relevant -- but to
/// `bufA`'s ADDRESS/offset (`STAGE_OFFSET`), not to this add term.
pub(crate) const GAMMA_POLY_SEED_CONSTANT: i16 = 0;

/// `param5` per the formula on [`GAMMA_POLY_SEED_CONSTANT`]:
/// `int16(ebx_seed + block_exponent(bufA))`, with `bufA` substituted by `stage`
/// -- see that doc for the scope of the substitution.
pub(crate) fn gamma_poly_param5_from_stage(
    stage: &[i16; super::analysis_stage_window::STAGE_LEN],
) -> i16 {
    let be = super::block_exponent::block_exponent(stage);
    (i32::from(GAMMA_POLY_SEED_CONSTANT) + i32::from(be)) as i16
}

/// Variant of [`gamma_poly_generate_approx`]: same pipeline, but derives `a2`/`a3`
/// from [`gamma_poly_param5_from_stage`]'s `param5` via
/// [`gamma_poly_scale_pair`], instead of the [`ptr63c_a2a3_silence`]
/// placeholder.
pub(crate) fn ptr63c_generate_real_param5(
    stage: &[i16; super::analysis_stage_window::STAGE_LEN],
) -> [i16; 40] {
    let windows = super::analysis_stage_window::analysis_windows(stage);
    let mut fft_windows = [[0i16; 32]; 20];
    for (i, w) in windows.iter().enumerate() {
        let tapered = super::windowed_taper::windowed_taper(w);
        let mut buf = tapered;
        super::real_fft32::real_fft32(&mut buf);
        fft_windows[i] = buf;
    }
    let param5 = gamma_poly_param5_from_stage(stage);
    let (a2, a3) = gamma_poly_scale_pair(param5);
    gamma_poly_assemble(&fft_windows, a2, a3)
}

/// Variant of [`candidate_gamma_poly`] using [`ptr63c_generate_real_param5`]
/// (real `ebx_seed`, substituted `bufA`) instead of the `(0,0)` silence
/// placeholder. Uses the SAME calibration constants as
/// [`candidate_gamma_poly`]; they are not re-tuned for this input.
pub(crate) fn candidate_gamma_ptr63c_r185(
    stage: &[i16; super::analysis_stage_window::STAGE_LEN],
) -> u8 {
    let words = ptr63c_generate_real_param5(stage);
    let mean_abs: f64 = words.iter().map(|&w| f64::from(w).abs()).sum::<f64>() / words.len() as f64;
    let gamma_raw = gamma_poly_calib_a() * mean_abs + gamma_poly_calib_b();
    nearest_gain_o_index(gamma_raw.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
}

// ============ FLOOR-THEN-GAIN ORDERING (the floor call before the gain call) ============
//
// The per-frame driver calls, IN THIS ORDER:
//
//   floor call  -> band decompress -> { decompress,
//                                       bias,
//                                       GATED LOW-BAND FLOOR }
//   (intermediate call)
//   gain call   -> gain driver -> { bias write,
//                                   GAIN QUANTIZER }
//
// Boundaries verified against the `ret`, per the objdump-overshoot trap:
// the band-decompress call returns only after its own inner decompress call
// (the gated floor is the LAST thing it does); the gain-driver call returns
// only after its inner gain-quantizer call, which is the last call before
// the ret.
//
// So the gain index is quantized from the **POST-floor** M_l. Computing b2 from
// the PRE-floor M_l, and applying the floor only to the vector handed to the
// amplitude quantizer, is wrong.
//
// **`voiced.pcm` cannot detect that error.** Not because the gate rarely fires
// there -- it fires on 130/199 voiced frames -- but because the floor rarely
// CHANGES a value: it only raises low-band amplitudes that sit below it, and on
// loud harmonic frames they already exceed it. The floor actually modifies M_l on
// 2/199 voiced frames (161, 191) versus 58/199 mark frames. `voiced`'s b2 is
// 199/199 and bit-identical under BOTH orderings; `mark` is the only file that
// discriminates. That is the "voiced b2 199/199 but mark b2 188/199" asymmetry.
//
// Getting the order right is worth, on mark: b2 188 -> 197, whole-frame 184 ->
// 193. It repairs exactly mark's 9 b2 sole-blocker frames
// [45,65,116,117,144,145,174,175,198] and breaks nothing -- b2's measured causal
// ceiling. 8 of those are frames where the floor bites directly; frame 117
// inherits via `prev_bias` from 116, since `next_bias_raw_exact` feeds the next
// frame's gamma.
