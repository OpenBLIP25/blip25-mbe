// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Encoder stage 1: **pitch estimation** (MBE §0.3 initial estimate).
//!
//! Clean-room implementation of the Multi-Band Excitation pitch likelihood
//! (Griffin & Lim, IEEE TASSP 1988 / IMBE §0.3). For each candidate ω₀ this
//! fits ONE window-shaped peak per harmonic band and scores the fraction of
//! signal energy explained — the proper matched-filter error, NOT a raw
//! energy-coverage count.
//!
//! ## Where this runs
//! This is NOT the pitch path of `Vocoder::encode`, which overrides `b0` with
//! the reverse-engineered reference chain via `forced_b0`. It IS the transmitted
//! pitch for `Vocoder::encode_pcm` / `encode_stream`, and for `LiveEncoder` on
//! the two IMBE rates. Those paths carry this estimator's ~25 % gross
//! subharmonic tail, which is audible as "axe" distortion; that tail is a
//! property of single-frame clean-room estimation here, not a bug to tune out.
//!
//! ## Window-matched scoring is required
//! Scoring by summed |S|² inside harmonic bands (maximising coverage) is
//! biased: a denser comb (lower ω₀) grabs more energy regardless of where its
//! teeth land, and the estimate collapses to the lowest pitch (b0≈119) on
//! essentially every frame, including pure tones. [`comb_score_n`] instead
//! projects each band onto the analysis-window mainlobe shape [`window_ft`]
//! with the random-phase noise floor subtracted: a halved/doubled candidate,
//! whose band holds a real harmonic OFF the mainlobe centre, fits poorly and is
//! rejected. Do not replace it with an energy-coverage score.
//!
//! The analysis window is the firmware's bit-exact **250-pt reference window**
//! (`fw_tables::reference_win250`, the interior of a 256-pt Hann, extracted from the
//! firmware image) — NOT a 221-pt Hamming. [`analysis_window`] keys off the
//! sample count (250 → the reference, else Hamming for tests); [`window_ft`] must match
//! whatever [`analysis_window`] returns.
//!
//! ## YIN octave anchor
//! The bare matched filter, even with noise-floor subtraction, floors at the
//! lowest pitch on a large fraction of voiced frames: a dense low-ω₀ comb has
//! ~60 free harmonic amplitudes and overfits almost any window.
//! [`yin_omega_conf`] (time-domain CMNDF) picks the OCTAVE — it counts a
//! period, it does not fit amplitudes, so comb density cannot fool it — and
//! `estimate_pitch_f64` then refines the spectral matched filter within
//! ±½ octave of that anchor. The anchor is load-bearing; removing it
//! reinstates the low-octave collapse.
//!
//! ## The residual error is not reachable from this side
//! The reference TRACKS and refines pitch across frames (§0.4) inside a
//! kernel that cannot be observed, and its own b0 contour jumps ~19 steps per
//! frame with octave flips. No single-frame clean-room estimator reproduces
//! that contour. These levers are all flat or negative and should not be
//! re-attempted: window shape/length, score form (incoherent /
//! coherent-excess / matched-filter / +noise-floor / full reconstruction
//! error), the pitch-band ceiling ([`pitch_fmax`]), frequency weighting
//! ([`pitch_wexp`]), window placement, DP tracking, and frame-alignment
//! sweeps. This is the best clean-room estimate available:
//! octave-anchored and firmware-window-exact.
//!
//! The final ω₀ is quantised to the nearest Annex-L voice entry
//! (`dequantize::decode_pitch` over b0 = 0..120) so the encoder stays
//! decoder-consistent. Internals are `f64`; a fixed-point port comes later.

use core::f64::consts::PI;

use crate::dequantize::{decode_pitch, PITCH_INDEX_MAX};

/// Sample rate of the half-rate AMBE+2 codec (Hz).
pub(crate) const SAMPLE_RATE: f64 = 8000.0;

/// Frequency resolution of the analysis power spectrum (number of bins
/// across [0, π)). 512 gives ~7.8 Hz/bin at 8 kHz, finer than the
/// Annex-L pitch quantiser spacing.
const SPECTRUM_BINS: usize = 512;

/// Number of candidate fundamentals swept in the coarse §0.3 search.
const PITCH_CANDIDATES: usize = 600;

/// Result of pitch estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PitchEstimate {
    /// Raw, continuous fundamental from the §0.3 search + parabolic
    /// interpolation + sub-harmonic escape (rad/sample), before Annex-L
    /// quantisation.
    pub omega_0_raw: f32,
    /// Quantised fundamental: the Annex-L table value for `b0`.
    pub omega_0: f32,
    /// Number of harmonics L for `b0` (Annex-L).
    pub l: u8,
    /// 7-bit pitch index (0..=119).
    pub b0: u8,
}

/// Estimate pitch from a frame of 16-bit PCM.
///
/// `pcm` should be the analysis window centred on the current 20 ms frame
/// (e.g. 160..256 samples). The window is applied internally; the caller
/// must NOT pre-window. The window must be centred — do not hand it one that
/// starts ~45 samples early.
///
/// Returns `(omega_0, L, b0)` where `omega_0`/`L` are the *quantised*
/// Annex-L values for `b0` (so the encoder agrees with the decoder).
pub(crate) fn estimate_pitch(pcm: &[i16]) -> (f32, u8, u8) {
    let samples: Vec<f64> = pcm.iter().map(|&s| f64::from(s)).collect();
    let e = estimate_pitch_f64(&samples);
    (e.omega_0, e.l, e.b0)
}

/// Core estimator over `f64` samples. See [`estimate_pitch`].
pub(crate) fn estimate_pitch_f64(samples: &[f64]) -> PitchEstimate {
    estimate_pitch_f64_ctx(samples, samples)
}

/// Same as [`estimate_pitch_f64`], but the YIN octave anchor is computed
/// from a SEPARATE (typically wider) context buffer `yin_ctx`, while the
/// spectral matched-filter analysis still uses the exact `samples` window
/// (must stay the firmware's 250-pt window for bit-matched behaviour).
/// `yin_ctx` must be centred on the same point as `samples`. Giving YIN more
/// periods to average over changes only the octave pick, never the spectral
/// analysis window.
pub(crate) fn estimate_pitch_f64_ctx(samples: &[f64], yin_ctx: &[f64]) -> PitchEstimate {
    // Search bounds straight from the Annex-L voice table, with a little
    // slack so the parabolic step can reach the endpoint entries.
    let w_lo = decode_pitch(PITCH_INDEX_MAX).unwrap().omega_0 as f64 * 0.97;
    let w_hi = decode_pitch(0).unwrap().omega_0 as f64 * 1.03;

    // Degenerate input: nothing to analyse. Park at the lowest pitch
    // (highest L) entry; the voicing stage will mark it unvoiced/silence.
    if samples.len() < 4 {
        return quantize(w_lo as f32);
    }

    let n_win = samples.len();
    let (re, im) = complex_spectrum(samples);
    // |S(ω_k)|² per bin, computed once and shared with every comb_score_n
    // visit rather than recomputed per candidate ω₀.
    let mag2: Vec<f64> = re.iter().zip(&im).map(|(r, i)| r * r + i * i).collect();
    let total: f64 = mag2.iter().sum();
    if total <= f64::MIN_POSITIVE {
        return quantize(w_lo as f32);
    }

    // ---- octave anchor: YIN/CMNDF on the raw window -----------------------
    // The pure spectral matched-filter (comb_score_n) OVERFITS a dense low-ω₀
    // comb: a 65 Hz fundamental spreads ~60 free harmonic amplitudes across the
    // band and "explains" almost any window, so the bare argmax collapses to
    // the pitch floor on a large fraction of voiced frames. A time-domain YIN
    // estimate is immune to that bias (it counts a period, it doesn't fit
    // amplitudes), so it picks the OCTAVE and the spectral search is restricted
    // to a half-octave band around it — keeping the spectral mainlobe precision
    // while killing the low-octave collapse. If YIN is unsure the search falls
    // back to the full band.
    //
    // The single best-CMNDF-dip anchor is wrong on a large minority of frames,
    // and when it is wrong the refined answer is no better than a uniform
    // random b0 — the anchor pick, not the spectral refine, is the dominant
    // error source. Multi-candidate YIN does NOT fix that: taking the top-4
    // CMNDF local minima as extra candidate octaves and arbitrating between
    // their refined ω₀ (by reconstruction error or by comb_score_n) is
    // net-negative, because when the primary anchor is wrong the other CMNDF
    // minima are usually not the right octave either, while the arbitration
    // score sometimes overrides an already-correct primary pick. Do not
    // re-add it without a genuinely different octave-disambiguation signal.
    let step = (w_hi - w_lo) / (PITCH_CANDIDATES - 1) as f64;
    let seed_conf = yin_omega_conf(yin_ctx);
    let seed = seed_conf.map(|(w, _)| w);
    let hit_dip = seed_conf.map(|(_, hd)| hd).unwrap_or(false);
    let (slack_lo, slack_hi) = yin_slack();
    let (lo_w, hi_w) = match seed {
        Some(ws) => ((ws * slack_lo).max(w_lo), (ws * slack_hi).min(w_hi)),
        None => (w_lo, w_hi),
    };
    let omega = refine_in_window(&re, &im, &mag2, total, n_win, w_lo, step, lo_w, hi_w);
    let est = quantize_rescored(&re, &im, &mag2, total, n_win, omega as f32);
    // Spectral confidence at the FINAL raw omega (reuses the already-
    // computed re/im/total/n_win, no extra spectrum work) -- the second,
    // independent confidence signal in conf_gated_calibrate's gate (see that
    // function's doc).
    let spec_conf = comb_score_n(&re, &im, &mag2, total, f64::from(est.omega_0_raw), n_win);
    let bzcr_omega = band_limited_zcr_omega(samples);
    let hps_omega = harmonic_product_spectrum_omega(&re, &im, w_lo, w_hi);
    conf_gated_calibrate(est, hit_dip, spec_conf, bzcr_omega, hps_omega)
}

/// Confidence-gated bias correction. Disabled by [`conf_calib_enabled`] —
/// see that function for why.
///
/// Gates on TWO independent, fully causal per-frame confidence signals;
/// neither requires knowing the reference's answer:
/// - `hit_dip`: did the YIN octave anchor find a genuine CMNDF dip below
///   threshold, or did it fall back to the (much less reliable) global
///   minimum / find no anchor at all? See [`yin_omega_conf`].
/// - `spec_conf`: the spectral matched-filter's own explained-energy
///   fraction ([`comb_score_n`]) at the FINAL chosen ω₀ — low even when YIN
///   found a confident dip means the spectral refine itself isn't a good
///   fit either.
///
/// The gate is `!hit_dip || spec_conf < `[`conf_calib_spec_cutoff`]` — i.e.
/// "apply the correction unless BOTH signals say the frame is confident".
/// When gated, it applies a shrink-toward-the-mean affine over `b0` and the
/// band-limited-ZCR predictor ([`conf_calib_abc`]). `hit_dip` stays a hard
/// binary override rather than being folded into one continuous CMNDF
/// threshold: a tighter single-signal cutoff looks better in aggregate but
/// badly regresses confident tonal content.
///
/// An UNGATED global correction is not an option: it also nudges the
/// already-correct confident answers, which costs more than it gains.
///
/// The second predictor is the zero-crossing rate of the analysis window
/// after a 4-pole Butterworth low-pass ([`band_limited_zcr_omega`], cutoff
/// [`bzcr_cutoff_hz`]). Raw or lightly-smoothed ZCR is useless on real speech
/// (dominated by fricative/formant energy above the fundamental); band-limiting
/// below the fundamental band BEFORE counting crossings is what makes it
/// correlate at all — weaker than `b0` alone, but genuinely additive. The
/// blended constants are baked into [`conf_calib_abc`]; there is no runtime
/// blending. Filtering is PER-WINDOW with zero cross-frame state, the only
/// thing the 250-sample call site can do.
fn conf_gated_calibrate(
    est: PitchEstimate,
    hit_dip: bool,
    spec_conf: f64,
    bzcr_omega: f64,
    hps_omega: f64,
) -> PitchEstimate {
    if !conf_calib_enabled() {
        return est;
    }
    let gated = !hit_dip || spec_conf < conf_calib_spec_cutoff();
    if !gated {
        return est;
    }
    let new_b0 = if hps_extreme_enabled() && is_extreme_b0(est.b0) {
        // Extreme-pitch-range gate (see `conf_calib_extreme_achb`'s doc): a
        // SEPARATE 4-constant affine that adds the harmonic-product-spectrum
        // predictor, fit only on the gated+extreme subset. Everywhere else
        // (the common case) uses the plain 3-constant
        // `(raw_b0, bzcr_omega)` affine below.
        let (a, c, h, b) = conf_calib_extreme_achb();
        let v = a * f64::from(est.b0) + c * bzcr_omega + h * hps_omega + b;
        v.round().clamp(0.0, PITCH_INDEX_MAX as f64) as u8
    } else {
        let (a, c, b) = conf_calib_abc();
        let v = a * f64::from(est.b0) + c * bzcr_omega + b;
        v.round().clamp(0.0, PITCH_INDEX_MAX as f64) as u8
    };
    if new_b0 == est.b0 {
        return est;
    }
    match decode_pitch(new_b0) {
        Some(e) => PitchEstimate {
            omega_0_raw: est.omega_0_raw,
            omega_0: e.omega_0,
            l: e.l,
            b0: new_b0,
        },
        None => est,
    }
}

fn conf_calib_enabled() -> bool {
    // OFF, because the bit-match objective wins: this correction is a
    // PESQ-tuned clean-room heuristic that pushes b0 AWAY from the reference
    // bits. Off, b0 median |diff| is 2 steps rather than 7, and b0 bit-match
    // is higher.
    false
}

/// 3-constant affine `(A, C, B)` for [`conf_gated_calibrate`]:
/// `b0' = round(clamp(A*raw_b0 + C*bzcr_omega + B, 0, 119))`. A regularized
/// blend of a single-variable `raw_b0` affine and a `(raw_b0, bzcr_omega)`
/// multivariate affine, both least-squares fit on the gated subset of a
/// real-reference corpus. That corpus and the reference that produced it are retired:
/// these constants cannot be re-derived, so do not re-tune them.
fn conf_calib_abc() -> (f64, f64, f64) {
    (0.301070, -41.563144, 65.083346)
}

/// Whether the extreme-pitch-range HPS gate ([`conf_calib_extreme_achb`]) is
/// active. When inactive, the plain 3-constant `(raw_b0, bzcr_omega)` affine is
/// used everywhere.
fn hps_extreme_enabled() -> bool {
    true
}

/// `[lo, hi)` raw-`b0` range considered "extreme" for [`conf_calib_extreme_achb`].
/// 15/105 is the widest pair that regresses no corpus file; see that function's
/// doc before moving either bound.
fn hps_extreme_range() -> (u8, u8) {
    (15u8, 105u8)
}

fn is_extreme_b0(b0: u8) -> bool {
    let (lo, hi) = hps_extreme_range();
    b0 < lo || b0 >= hi
}

/// 4-constant affine `(A, C, H, B)` for [`conf_gated_calibrate`]'s
/// extreme-pitch-range branch: `b0' = round(clamp(A*raw_b0 + C*bzcr_omega +
/// H*hps_omega + B, 0, 119))`. Adds a THIRD predictor, the harmonic-product-
/// spectrum fundamental estimate ([`harmonic_product_spectrum_omega`],
/// `n_harm=4`), on top of the shipped `(raw_b0, bzcr_omega)` pair — but
/// ONLY on frames whose raw `b0` falls in the extreme range
/// ([`is_extreme_b0`]); everywhere else the plain 3-constant
/// [`conf_calib_abc`] affine is used unchanged.
///
/// ## Why the gate is range-restricted, not a corpus-wide 4-constant affine
/// HPS is a genuinely new, correctly-signed signal, but folding it into one
/// corpus-wide multivariate refit dilutes and destabilises the mid-range fit
/// that the 3-constant affine already handles well, and measurably regresses
/// part of the corpus. Restricting the extra predictor to the extreme pitch
/// range — where the other two predictors, raw `b0` and
/// [`band_limited_zcr_omega`], are weakest — is what makes it additive rather
/// than net-negative.
///
/// ## Do not widen the extreme range
/// [`hps_extreme_range`] is (15, 105) because it is the widest bound pair
/// under which no corpus file shows a statistically real regression. Tighter
/// 20/100 scores marginally better in aggregate but regresses an individual
/// file; tighter 25/95 is a clear regression on that file; wider (10/110)
/// regresses a large, reliable file outright. Small per-file deltas in this
/// region are within sampling noise and are not evidence either way.
///
/// Fit on the gated subset of a real-reference corpus, restricted further to
/// [`is_extreme_b0`]. That corpus and the reference that produced it are retired:
/// these constants cannot be re-derived, so do not re-tune them.
fn conf_calib_extreme_achb() -> (f64, f64, f64, f64) {
    (0.154458, -48.619023, -20.009836, 88.859147)
}

/// `spec_conf` cutoff for [`conf_gated_calibrate`]'s widened gate (0.60 —
/// picked from the corpus's own spec_conf distribution, which has a natural
/// break point around its 10th percentile (~0.71) with the bulk of frames
/// clustered at 0.87-0.88; 0.60 sits safely in the low tail so it doesn't
/// accidentally catch genuinely-confident frames).
fn conf_calib_spec_cutoff() -> f64 {
    0.60
}

/// Low-pass cutoff (Hz) for [`band_limited_zcr_omega`]. 800 Hz is the best
/// point over a 400–1000 Hz range against the retired reference; it
/// cannot be re-derived, so do not re-tune it.
fn bzcr_cutoff_hz() -> f64 {
    800.0
}

/// One RBJ-cookbook lowpass biquad (bilinear transform of a 2-pole analog
/// prototype at cutoff `fc` with pole quality `q`). Direct form 1.
#[derive(Clone, Copy)]
struct BzcrBiquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl BzcrBiquad {
    fn lowpass(fc: f64, fs: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * fc / fs;
        let alpha = w0.sin() / (2.0 * q);
        let cosw0 = w0.cos();
        let b0 = (1.0 - cosw0) / 2.0;
        let b1 = 1.0 - cosw0;
        let b2 = (1.0 - cosw0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cosw0;
        let a2 = 1.0 - alpha;
        BzcrBiquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Filters `x` forward, zero initial state (direct form 1).
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0f64; x.len()];
        let (mut x1, mut x2, mut y1, mut y2) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..x.len() {
            let xi = x[i];
            let yi = self.b0 * xi + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            y[i] = yi;
            x2 = x1;
            x1 = xi;
            y2 = y1;
            y1 = yi;
        }
        y
    }
}

/// Zero-phase (filtfilt: forward pass, then a reverse-direction pass) 4-pole
/// Butterworth lowpass of `x` at `fc` Hz. Applied PER-WINDOW with zero
/// cross-frame state — the only thing the per-frame call site can do, since it
/// only ever sees the current analysis window, not a continuously-filtered
/// whole-file stream. [`conf_calib_abc`] is fit in exactly this configuration,
/// so carrying filter state across frames would invalidate those constants.
fn butter_lowpass_filtfilt(x: &[f64], fc: f64, fs: f64) -> Vec<f64> {
    // Standard per-stage pole Q values for a maximally-flat 4th-order
    // Butterworth lowpass, cascaded as 2 biquads.
    const QS: [f64; 2] = [0.541_196_1, 1.306_562_9];
    let mut y = x.to_vec();
    for &q in &QS {
        y = BzcrBiquad::lowpass(fc, fs, q).apply(&y);
    }
    y.reverse();
    for &q in &QS {
        y = BzcrBiquad::lowpass(fc, fs, q).apply(&y);
    }
    y.reverse();
    y
}

/// Band-limited zero-crossing rate, converted to an ω (rad/sample) estimate:
/// low-pass `samples` to [`bzcr_cutoff_hz`], count sign crossings of the
/// mean-removed filtered window, then convert crossings-per-sample to ω
/// assuming ~2 crossings per period. Returns `0.0` on a degenerate
/// (silent/too-short) window. See [`conf_calib_abc`] for how this feeds the
/// correction.
///
/// Two properties this depends on:
/// - The low-pass is mandatory. Unfiltered zero-crossing counting is useless
///   on real speech, because fricative/formant energy above the fundamental
///   dominates the count.
/// - No refractory gap. The low-pass has already removed the sub-period
///   chatter a gap would target, and adding one reintroduces a wrong-sign
///   bias on the low-`b0` / high-pitch extreme.
fn band_limited_zcr_omega(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return 0.0;
    }
    let filtered = butter_lowpass_filtfilt(samples, bzcr_cutoff_hz(), SAMPLE_RATE);
    let mean: f64 = filtered.iter().sum::<f64>() / n as f64;
    let mut crossings = 0u32;
    let mut prev_sign = (filtered[0] - mean) >= 0.0;
    for &s in &filtered[1..] {
        let sign = (s - mean) >= 0.0;
        if sign != prev_sign {
            crossings += 1;
        }
        prev_sign = sign;
    }
    let z = crossings as f64 / (n - 1) as f64;
    if z <= 1e-6 {
        return 0.0;
    }
    let period = 2.0 / z;
    2.0 * PI / period
}

/// Harmonic Product Spectrum (HPS) fundamental estimate, `n_harm=4`; see
/// [`conf_calib_extreme_achb`] for how this feeds the correction. Classical
/// frequency-domain pitch technique: a candidate fundamental ω scores well
/// only if ALL its first `n_harm` harmonics (ω, 2ω, 3ω, 4ω) carry real energy
/// in the magnitude spectrum — the log-magnitude product (equivalently, mean
/// of log-magnitudes) across harmonics, argmax over the same
/// [`PITCH_CANDIDATES`]-point grid the primary spectral refine searches.
/// Reuses the caller's ALREADY-COMPUTED `re`/`im` from [`complex_spectrum`],
/// so it costs no extra spectrum computation. Its value is that its bias/noise
/// profile differs from both `raw_b0` (a window-matched-filter comb search)
/// and [`band_limited_zcr_omega`] (a time-domain zero-crossing count): it is
/// correctly signed and informative specifically on the low-pitch extreme,
/// where the other two predictors are weakest.
fn harmonic_product_spectrum_omega(re: &[f64], im: &[f64], w_lo: f64, w_hi: f64) -> f64 {
    const N_HARM: usize = 4;
    let power: Vec<f64> = re.iter().zip(im).map(|(r, i)| r * r + i * i).collect();
    let bins_per_rad = SPECTRUM_BINS as f64 / PI;
    let interp = |omega: f64| -> f64 {
        if omega < 0.0 {
            return 0.0;
        }
        let x = omega * bins_per_rad;
        let i = x.floor() as usize;
        if i + 1 >= power.len() {
            return *power.last().unwrap_or(&0.0);
        }
        let frac = x - i as f64;
        power[i] * (1.0 - frac) + power[i + 1] * frac
    };
    let step = (w_hi - w_lo) / (PITCH_CANDIDATES - 1) as f64;
    let mut best_w = w_lo;
    let mut best_score = f64::NEG_INFINITY;
    for i in 0..PITCH_CANDIDATES {
        let w = w_lo + step * i as f64;
        let max_h = ((PI * 0.97) / w).floor().max(1.0) as usize;
        let h_cap = N_HARM.min(max_h);
        let mut score = 0f64;
        for h in 1..=h_cap {
            let p = interp(w * h as f64).max(1e-9);
            score += p.ln();
        }
        score /= h_cap as f64;
        if score > best_score {
            best_score = score;
            best_w = w;
        }
    }
    best_w
}

/// Discrete Annex-L rescore radius around the nearest-omega snap; 0 disables
/// it and leaves the plain nearest-omega [`quantize`] result.
///
/// `quantize`'s nearest-omega snap can land on a different `b0` than the one
/// [`comb_score_n`] itself prefers, because the continuous parabolic refine's
/// vertex and the DISCRETE Annex-L table spacing are not aligned (table
/// entries are non-uniformly spaced in ω, denser at low `b0`). Re-scoring the
/// few `b0` indices nearest the initial snap with the SAME `comb_score_n`
/// (no extra spectrum work; it reuses the caller's `re`/`im`/`total`) picks
/// whichever discrete table entry the matched-filter score favours. Gains
/// flatten out past radius 5.
fn rescore_radius() -> i64 {
    // Disabled, because the bit-match objective wins: the discrete rescore is
    // a PESQ-oriented heuristic that re-snaps b0 to the comb_score-preferred
    // Annex-L entry, lowering mean|Δb0| but pushing the INDEX BITS away from
    // the reference.
    0
}

fn quantize_rescored(
    re: &[f64],
    im: &[f64],
    mag2: &[f64],
    total: f64,
    n_win: usize,
    omega_raw: f32,
) -> PitchEstimate {
    let base = quantize(omega_raw);
    let radius = rescore_radius();
    if radius <= 0 {
        return base;
    }
    let lo = base.b0.saturating_sub(radius.max(0) as u8);
    let hi = (base.b0 as i64 + radius).min(PITCH_INDEX_MAX as i64) as u8;
    let mut best_b0 = base.b0;
    let mut best_score = f64::NEG_INFINITY;
    for b0 in lo..=hi {
        let e = match decode_pitch(b0) {
            Some(e) => e,
            None => continue,
        };
        let s = comb_score_n(re, im, mag2, total, f64::from(e.omega_0), n_win);
        if s > best_score {
            best_score = s;
            best_b0 = b0;
        }
    }
    if best_b0 == base.b0 {
        base
    } else {
        let e = decode_pitch(best_b0).unwrap();
        PitchEstimate {
            omega_0_raw: omega_raw,
            omega_0: e.omega_0,
            l: e.l,
            b0: best_b0,
        }
    }
}

/// Grid-search + parabolic-interpolation + local spectral refine of ω₀
/// within `[lo_w, hi_w]` (the YIN-anchored half-octave window, or the full
/// band if YIN found no confident anchor).
// spectral refine takes the re/im/mag2 arrays plus the search-window scalars by nature
#[allow(clippy::too_many_arguments)]
fn refine_in_window(
    re: &[f64],
    im: &[f64],
    mag2: &[f64],
    total: f64,
    n_win: usize,
    w_lo: f64,
    step: f64,
    lo_w: f64,
    hi_w: f64,
) -> f64 {
    let mut best_i = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut scores = [0f64; PITCH_CANDIDATES];
    for i in 0..PITCH_CANDIDATES {
        let w = w_lo + step * i as f64;
        if w < lo_w || w > hi_w {
            scores[i] = f64::NEG_INFINITY;
            continue;
        }
        let s = comb_score_n(re, im, mag2, total, w, n_win);
        scores[i] = s;
        if s > best_score {
            best_score = s;
            best_i = i;
        }
    }
    let mut omega = w_lo + step * best_i as f64;

    // ---- parabolic interpolation about the discrete minimum ---------------
    if best_i > 0
        && best_i < PITCH_CANDIDATES - 1
        && scores[best_i - 1].is_finite()
        && scores[best_i + 1].is_finite()
    {
        omega = parabolic_vertex(
            w_lo + step * (best_i - 1) as f64,
            w_lo + step * best_i as f64,
            w_lo + step * (best_i + 1) as f64,
            scores[best_i - 1],
            scores[best_i],
            scores[best_i + 1],
        );
    }

    // No sub-harmonic-trap escape is needed here: the window-matched score
    // (comb_score_n) already rejects halved/doubled pitch.

    // Re-refine around the fundamental for sub-grid accuracy.
    local_refine(re, im, mag2, total, omega, step, n_win)
}

/// Lowest / highest pitch *period* (samples) admitted by the YIN octave
/// anchor: ~444 Hz down to ~62 Hz at 8 kHz. Spans the Annex-L voice range
/// with a little slack so the spectral refine can reach the table endpoints.
const YIN_PMIN: usize = 18;
const YIN_PMAX: usize = 128;
/// CMNDF acceptance threshold: the first dip below this is taken as the period
/// (the classic YIN absolute threshold). Above it we fall back to the global
/// minimum, and if even that is weak the caller does a full-band search.
const YIN_THRESH_DEFAULT: f64 = 0.15;

/// CMNDF acceptance threshold ([`YIN_THRESH_DEFAULT`]).
fn yin_thresh() -> f64 {
    YIN_THRESH_DEFAULT
}

/// Half-octave search-window slack around the YIN anchor (`(0.71, 1.41)` — a
/// true half octave is `(0.7071, 1.4142)`).
fn yin_slack() -> (f64, f64) {
    (0.71, 1.41)
}

/// YIN/CMNDF octave anchor: returns `(ω, confident)`, where `confident` is
/// `true` when the pick came from a genuine CMNDF dip below [`yin_thresh`]
/// and `false` when it came from the global-minimum fallback because no dip
/// cleared the threshold. The flag matters because fallback frames carry
/// materially more `b0` error than confident-dip frames; see
/// [`estimate_pitch_f64_ctx`] and [`conf_gated_calibrate`] for its use.
fn yin_omega_conf(samples: &[f64]) -> Option<(f64, bool)> {
    let n = samples.len();
    if n < 2 * YIN_PMIN + 2 {
        return None;
    }
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;
    let pmax = YIN_PMAX.min(n / 2);
    if pmax <= YIN_PMIN {
        return None;
    }
    // squared difference function d(τ)
    let mut d = vec![0f64; pmax + 1];
    for tau in YIN_PMIN..=pmax {
        let mut s = 0f64;
        for i in 0..n - tau {
            let diff = (samples[i] - mean) - (samples[i + tau] - mean);
            s += diff * diff;
        }
        d[tau] = s;
    }
    // cumulative-mean normalisation
    let mut cmndf = vec![1f64; pmax + 1];
    let mut running = 0f64;
    for tau in YIN_PMIN..=pmax {
        running += d[tau];
        cmndf[tau] = d[tau] * (tau - YIN_PMIN + 1) as f64 / running.max(1e-9);
    }
    // first local minimum below threshold, else global minimum
    let mut best = 0usize;
    let mut best_v = f64::MAX;
    let mut hit_dip = false;
    for tau in YIN_PMIN + 1..pmax {
        if cmndf[tau] < yin_thresh() && cmndf[tau] < cmndf[tau - 1] && cmndf[tau] <= cmndf[tau + 1]
        {
            best = tau;
            hit_dip = true;
            break;
        }
        if cmndf[tau] < best_v {
            best_v = cmndf[tau];
            best = tau;
        }
    }
    if best < YIN_PMIN {
        return None;
    }
    // a totally aperiodic frame (cmndf never dips) carries no octave info
    if best_v > 0.85 && cmndf[best] > 0.85 {
        return None;
    }
    // parabolic interpolation of the period for a smoother seed
    let period = if best > YIN_PMIN && best < pmax {
        let (a, b, c) = (cmndf[best - 1], cmndf[best], cmndf[best + 1]);
        let den = a - 2.0 * b + c;
        if den.abs() > 1e-12 {
            best as f64 + 0.5 * (a - c) / den
        } else {
            best as f64
        }
    } else {
        best as f64
    };
    Some((2.0 * PI / period, hit_dip))
}

/// Quantise a continuous ω₀ to the nearest Annex-L voice entry.
fn quantize(omega_raw: f32) -> PitchEstimate {
    let metric = quantize_metric();
    let map = |w: f32| -> f64 {
        match metric {
            // nearest fundamental frequency
            0 => f64::from(w),
            // nearest pitch PERIOD (2pi/omega) -- the spec quantizes pitch period
            1 => 2.0 * PI / f64::from(w).max(1e-9),
            // nearest log-frequency
            _ => f64::from(w).max(1e-9).ln(),
        }
    };
    let target = map(omega_raw);
    let b0max = quantize_b0max();
    let mut best_b0 = 0u8;
    let mut best_d = f64::INFINITY;
    let mut best = decode_pitch(0).unwrap();
    for b0 in 0..=b0max {
        let e = decode_pitch(b0).unwrap();
        let d = (map(e.omega_0) - target).abs();
        if d < best_d {
            best_d = d;
            best_b0 = b0;
            best = e;
        }
    }
    PitchEstimate {
        omega_0_raw: omega_raw,
        omega_0: best.omega_0,
        l: best.l,
        b0: best_b0,
    }
}

fn quantize_metric() -> u8 {
    // 1 = nearest PERIOD (0 = omega, 2 = log). The Annex-L pitch table is a
    // geometric progression (constant period ratio ~1.0153), so the reference
    // quantizer's native domain is pitch period / log-frequency, NOT linear
    // frequency. Snapping to the nearest table PERIOD raises b0 bit-match
    // relative to snapping to the nearest omega.
    1u8
}

fn quantize_b0max() -> u8 {
    PITCH_INDEX_MAX
}

/// COMPLEX spectrum S_w(ω) on a uniform grid ω_k = π·k/SPECTRUM_BINS,
/// k = 0..SPECTRUM_BINS.
///
/// The frame mean is removed (kills DC bias in the low bands) and
/// [`analysis_window`] is applied, centred on the supplied samples.
///
/// The time origin is centred on the window so a true harmonic has
/// near-constant phase across its mainlobe — required for the coherent
/// harmonic-fit score (the discriminator that defeats sub-harmonic traps).
fn complex_spectrum(samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;
    let center = (n as f64 - 1.0) / 2.0;

    let win = analysis_window(n);
    let mut x = vec![0f64; n];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (samples[i] - mean) * win[i];
    }

    let mut re = vec![0f64; SPECTRUM_BINS + 1];
    let mut im = vec![0f64; SPECTRUM_BINS + 1];
    if n == crate::fw_tables::REFERENCE_WIN250_N {
        // Production reference window: cos/sin depend only on (k, n_i), not on the
        // data, so read them from a precomputed twiddle table. The per-bin sum
        // (mean-subtracted, windowed `x`) stays in the loop and is unchanged.
        let (cos_t, sin_t) = spectrum_twiddles();
        for k in 0..=SPECTRUM_BINS {
            let base = k * n;
            let (mut r, mut ii) = (0f64, 0f64);
            for (n_i, &xn) in x.iter().enumerate() {
                r += xn * cos_t[base + n_i];
                ii -= xn * sin_t[base + n_i];
            }
            re[k] = r;
            im[k] = ii;
        }
    } else {
        for k in 0..=SPECTRUM_BINS {
            let omega = PI * k as f64 / SPECTRUM_BINS as f64;
            let (mut r, mut ii) = (0f64, 0f64);
            for (n_i, &xn) in x.iter().enumerate() {
                let phase = omega * (n_i as f64 - center);
                r += xn * phase.cos();
                ii -= xn * phase.sin();
            }
            re[k] = r;
            im[k] = ii;
        }
    }
    (re, im)
}

/// Precomputed cos/sin twiddles for `complex_spectrum` at the production the reference
/// window length (n = 250). `COS[k*n + n_i] = cos(omega*(n_i - center))` and
/// `SIN` likewise, with `omega = PI*k/SPECTRUM_BINS` and `center = (n-1)/2` —
/// the exact same scalar expression the runtime path evaluates, so the twiddle
/// values (and therefore the emitted bits) are bit-identical.
fn spectrum_twiddles() -> &'static (Vec<f64>, Vec<f64>) {
    use std::sync::OnceLock;
    static T: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    T.get_or_init(|| {
        let n = crate::fw_tables::REFERENCE_WIN250_N;
        let center = (n as f64 - 1.0) / 2.0;
        let len = (SPECTRUM_BINS + 1) * n;
        let mut cos_t = vec![0f64; len];
        let mut sin_t = vec![0f64; len];
        for k in 0..=SPECTRUM_BINS {
            let omega = PI * k as f64 / SPECTRUM_BINS as f64;
            let base = k * n;
            for n_i in 0..n {
                let phase = omega * (n_i as f64 - center);
                cos_t[base + n_i] = phase.cos();
                sin_t[base + n_i] = phase.sin();
            }
        }
        (cos_t, sin_t)
    })
}

/// Map a radian frequency to a fractional power-spectrum bin index.
#[inline]
fn omega_to_bin(omega: f64) -> f64 {
    omega * SPECTRUM_BINS as f64 / PI
}

/// Analysis-window coefficients (normalised, peak ~1.0) for a length-`n`
/// window. For n=250 this is the firmware's bit-exact reference window
/// (`fw_tables::reference_win250`, the interior of a 256-pt Hann) — the window the
/// reference actually analyses with. Other lengths fall back to Hamming (synthetic
/// tone tests etc.). Matched in `window_ft` so the matched filter uses the same
/// shape as the spectrum.
fn analysis_window(n: usize) -> std::borrow::Cow<'static, [f64]> {
    use std::borrow::Cow;
    use std::sync::OnceLock;
    if n == crate::fw_tables::REFERENCE_WIN250_N {
        // The reference 250-pt window is a frame-invariant constant; caching it
        // avoids re-dividing the firmware table (and re-allocating a Vec) on
        // every `complex_spectrum` call (per-frame, run several times per
        // encode).
        static W250: OnceLock<Vec<f64>> = OnceLock::new();
        Cow::Borrowed(W250.get_or_init(|| {
            crate::fw_tables::reference_win250()
                .iter()
                .map(|&w| f64::from(w) / 32768.0)
                .collect()
        }))
    } else {
        Cow::Owned(
            (0..n)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
                .collect(),
        )
    }
}

/// Cached spectral transform E_w(Δ) of the reference 250-pt window, sampled on a
/// fine Δ grid for fast matched-filter lookup. E_w(Δ)=Σ_n w(n)·cos(Δ(n−centre))
/// (real, the window is symmetric). Covers |Δ| ≤ REFERENCE_FT_MAX.
const REFERENCE_FT_MAX: f64 = 0.30;
const REFERENCE_FT_STEP: f64 = 0.0005;
fn reference_win_ft_table() -> &'static Vec<f64> {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| {
        let w = analysis_window(crate::fw_tables::REFERENCE_WIN250_N);
        let center = (w.len() as f64 - 1.0) / 2.0;
        let m = (REFERENCE_FT_MAX / REFERENCE_FT_STEP) as usize;
        (0..=m)
            .map(|j| {
                let delta = j as f64 * REFERENCE_FT_STEP;
                w.iter()
                    .enumerate()
                    .map(|(n, &wn)| wn * (delta * (n as f64 - center)).cos())
                    .sum()
            })
            .collect()
    })
}

#[inline]
fn reference_win_ft(delta: f64) -> f64 {
    let t = reference_win_ft_table();
    let a = delta.abs();
    if a >= REFERENCE_FT_MAX {
        return 0.0;
    }
    let x = a / REFERENCE_FT_STEP;
    let i = x.floor() as usize;
    let frac = x - i as f64;
    t[i] * (1.0 - frac) + t[i + 1] * frac
}

/// Spectral shape E_w(Δ) of the centred length-`n` Hamming analysis window,
/// evaluated at frequency offset `Δ` (rad) from a peak. Closed form via the
/// Dirichlet kernel: a Hamming window is 0.54·rect + 0.23·(two shifted rects),
/// and the centred rect transform is Dir_n(Δ)=sin(Δn/2)/sin(Δ/2). Real because
/// the window is symmetric. This is the matched-filter template for one
/// harmonic peak — narrow (mainlobe only), so a band that holds energy OFF the
/// peak (the tell-tale of a halved/doubled pitch) fits it poorly.
#[inline]
fn window_ft(delta: f64, n: usize) -> f64 {
    if n == crate::fw_tables::REFERENCE_WIN250_N {
        return reference_win_ft(delta);
    }
    // Hamming fallback (synthetic-tone tests): closed-form Dirichlet sum.
    let nf = n as f64;
    let dir = |d: f64| -> f64 {
        let s = (d * 0.5).sin();
        if s.abs() < 1e-12 {
            nf
        } else {
            (d * nf * 0.5).sin() / s
        }
    };
    let d0 = 2.0 * PI / (nf - 1.0);
    0.54 * dir(delta) + 0.23 * dir(delta - d0) + 0.23 * dir(delta + d0)
}

/// Upper frequency limit (fraction of π) for the pitch harmonic-fit sum
/// (1.0 = full band). Limiting to the band where speech has clean harmonic
/// structure removes the residual low-pitch bias (a lower ω₀ otherwise gains
/// extra weakly-fitting harmonics).
fn pitch_wexp() -> f64 {
    0.0
}

fn pitch_fmax() -> f64 {
    1.0
}

/// Proper MBE (Griffin-Lim) harmonic-fit score: the fraction of signal energy
/// explained by fitting ONE window-shaped peak per harmonic band. For each
/// harmonic l, least-squares-fit a single complex amplitude A_l so the model
/// A_l·E_w(ω−lω₀) matches S_w over the band [(l−½)ω₀,(l+½)ω₀]; the explained
/// energy is |Σ_k S_k·E_w(ω_k−lω₀)|² / Σ_k E_w². Summed over harmonics and
/// normalised by total energy. Because E_w is the NARROW window mainlobe, a
/// halved/doubled pitch (whose wide bands hold a real harmonic off-centre)
/// fits poorly and is rejected — the discriminator the crude coherent sum
/// lacked. Higher = better. `n_win` is the analysis-window length.
fn comb_score_n(re: &[f64], im: &[f64], mag2: &[f64], total: f64, omega: f64, n_win: usize) -> f64 {
    if omega <= 0.0 || total <= 0.0 {
        return 0.0;
    }
    let fmax = pitch_fmax();
    // Hoisted out of the per-harmonic loop: the weight exponent is frame-
    // invariant. `wexp == 0.0` (the shipped default) makes the per-harmonic
    // weight `(1/l)^0 == 1.0` exactly, so the multiply is skipped entirely
    // (byte-identical: IEEE pow(base,+0.0)==1.0 for base>0 and x*1.0==x).
    let wexp = pitch_wexp();
    let max_l = ((fmax * PI) / omega).floor() as usize;
    // ω_k = k / bins_per_rad tabled once (byte-identical to the division).
    let bin_omega = bin_omega();
    let mut explained = 0f64;
    for l in 1..=max_l {
        let center = l as f64 * omega;
        let lo = omega_to_bin(center - 0.5 * omega).floor().max(0.0) as usize;
        let hi = (omega_to_bin(center + 0.5 * omega).ceil() as usize).min(SPECTRUM_BINS);
        if hi <= lo {
            continue;
        }
        // matched filter against the window mainlobe centred at l·ω₀, with the
        // random-phase NOISE FLOOR subtracted. p=Σ S·E_w, q=Σ|S|²·E_w² (the
        // diagonal = E[|p|²] under random phase), g=Σ E_w². The coherent excess
        // (|p|²−q)/g is ~0 for noise (so a long low-pitch comb accumulates no
        // bias) and large for a real phase-coherent peak.
        let (mut pr, mut pi, mut q, mut gg) = (0f64, 0f64, 0f64, 0f64);
        for k in lo..=hi {
            let omega_k = bin_omega[k];
            let g = window_ft(omega_k - center, n_win);
            pr += re[k] * g;
            pi += im[k] * g;
            // mag2[k] == re[k]*re[k] + im[k]*im[k], precomputed once per frame.
            // Do NOT fold g*g into a shared temporary here: it changes the
            // rounding and therefore the emitted bits.
            q += mag2[k] * g * g;
            gg += g * g;
        }
        if gg > 1e-12 {
            let excess = (pr * pr + pi * pi - q) / gg;
            if excess > 0.0 {
                if wexp == 0.0 {
                    explained += excess;
                } else {
                    explained += excess * (1.0 / l as f64).powf(wexp);
                }
            }
        }
    }
    explained / total
}

/// Precomputed `ω_k = k / bins_per_rad` for `k = 0..=SPECTRUM_BINS`, where
/// `bins_per_rad = SPECTRUM_BINS/π`. Built with the identical expression the
/// per-bin division evaluates, so the tabled value is bit-for-bit the same.
fn bin_omega() -> &'static [f64; SPECTRUM_BINS + 1] {
    use std::sync::OnceLock;
    static T: OnceLock<[f64; SPECTRUM_BINS + 1]> = OnceLock::new();
    T.get_or_init(|| {
        let bins_per_rad = SPECTRUM_BINS as f64 / PI;
        let mut t = [0f64; SPECTRUM_BINS + 1];
        for (k, e) in t.iter_mut().enumerate() {
            *e = k as f64 / bins_per_rad;
        }
        t
    })
}

/// Vertex of the parabola through three (x, y) points; clamped to
/// [x0, x2]. Used for sub-grid interpolation of the score peak.
fn parabolic_vertex(x0: f64, x1: f64, x2: f64, y0: f64, y1: f64, y2: f64) -> f64 {
    let denom = (y0 - 2.0 * y1 + y2) * 2.0;
    if denom.abs() < 1e-18 {
        return x1;
    }
    let dx = x1 - x0; // uniform spacing assumed (x2 - x1 == x1 - x0)
    let offset = dx * (y0 - y2) / (2.0 * (y0 - 2.0 * y1 + y2));
    (x1 + offset).clamp(x0, x2)
}

/// Three-point parabolic refine of the score peak around `omega`.
fn local_refine(
    re: &[f64],
    im: &[f64],
    mag2: &[f64],
    total: f64,
    omega: f64,
    step: f64,
    n_win: usize,
) -> f64 {
    let d = step;
    let s0 = comb_score_n(re, im, mag2, total, omega - d, n_win);
    let s1 = comb_score_n(re, im, mag2, total, omega, n_win);
    let s2 = comb_score_n(re, im, mag2, total, omega + d, n_win);
    parabolic_vertex(omega - d, omega, omega + d, s0, s1, s2)
}

/// Reference-matched `b0` **reject-path recurrence**.
///
/// The reference encoder's `b0` field is written by one of two code paths
/// inside its per-frame `b0` writer, chosen once per frame by a gate:
/// - **reject** (~⅔ of frames on real speech): does **not** examine the audio
///   at all. It is a self-referential recurrence over one persistent 16-bit
///   state cell (`subframe_ctx[0xc]`), covering both the transmitted `b0`
///   value and that cell's own next-frame update.
/// - **accept** (~⅓ of frames): a different mechanism — the gate computes and
///   writes a fresh candidate itself; it is not a read of the candidate
///   buffer.
///
/// This module implements the reject path only. The accept path's numeric
/// refinement and the gate's trigger condition are not closed, so
/// [`reject_recurrence::reject_b0`] is a validated building block rather than
/// a drop-in replacement for [`estimate_pitch`]: the caller must supply the
/// accept/reject decision itself.
///
/// The arithmetic is a literal, instruction-by-instruction port of the reject
/// recurrence and its shared ratio helper, not a numerical approximation. Do
/// not "simplify" it into equivalent-looking float or wider-integer math —
/// the truncations and saturations are the behaviour. It was validated
/// bit-exact against the reference vocoder, which is retired: it cannot be
/// re-derived, so do not rewrite it. It reuses the
/// [`crate::enc::band_decompress::log2_fn`] port of the shared log2 primitive.
pub mod reject_recurrence {
    use crate::enc::band_decompress::log2_fn;

    /// `sign(a)·sign(b) · (|a| ≥ |b| ? |a|/|b| : 0)` — a saturating,
    /// truncating fixed-point ratio helper reused all over this codec.
    fn bounded_ratio(a: i32, b: i16) -> i32 {
        let sign_diff = ((b as i32) ^ a) < 0;
        let abs_a: i32 = if a == i32::MIN { i32::MAX } else { a.abs() };
        let abs_b: i32 = if b == i16::MIN {
            0x7fff
        } else {
            (b as i32).abs()
        };
        let result = if abs_a < abs_b { 0 } else { abs_a / abs_b };
        if sign_diff {
            -result
        } else {
            result
        }
    }

    /// The reject recurrence's `mode==1` body: given the persistent state
    /// cell's current value, returns the `b0` value this frame transmits on
    /// a reject. `width` is always 7 for `b0`, so the `60<<(width-6)` scale
    /// always resolves to 120.
    pub(crate) fn reject_b0(subctx_c_before: u16, width: i32) -> u8 {
        let log2res = log2_fn(subctx_c_before as i16, -4) as i32;
        let mut t: i32 = if (log2res as u32) == 0x8000_0000 {
            i32::MAX
        } else {
            -log2res
        };
        t = t.wrapping_shl(12) >> 16; // arithmetic (net: negate then /16, matching shl12/sar16)
        let t = t as i16 as i32; // cwde truncate to 16 significant bits
        let diff = (t - 0x44fd) as i16 as i32; // shl16/sar16 truncate
        let scale2 = 2 * (60 << (width - 6)); // == 240 for width=7
        let scaled = diff * scale2;
        let ratio = bounded_ratio(scaled, 21668);
        let scale = 60 << (width - 6); // == 120 for width=7
        if ratio < 0 {
            0
        } else if ratio >= scale {
            (scale - 1) as u8
        } else {
            ratio as u8
        }
    }
}

/// Accept-path "candidate refinement" of the reference encoder's `b0`.
///
/// Intentionally empty. The refinement arithmetic itself is understood — it is
/// a block-floating-point evaluation of
/// `candidate ≈ (prevValue*32768 + 0x20800000) * step / 0x41000000`, not a
/// plain multiply, and it was validated bit-exact against the reference
/// encoder, which is retired: it cannot be re-derived. But the gate deciding
/// which frames take the accept path is driven by `trackA_ctx[8]` /
/// `trackB_ctx[8]`, state written outside the reference vocoder. There is no
/// faithful way to decide from live audio when the accept path would fire, so
/// implementing the refinement here would produce a routine nothing may call.
/// Do not re-open this without a new source for that gate; the counterpart
/// [`reject_recurrence`] is the usable half.
pub mod accept_candidate {}
