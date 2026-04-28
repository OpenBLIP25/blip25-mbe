//! Spectral subtraction denoiser — opt-in input-side noise suppression
//! applied to `signal_spectrum()` output before §0.5 amplitude
//! estimation.
//!
//! Different from the retired `--harmonic-bin-pad` and
//! `--noise-floor-beta` directions: those operated on per-bin /
//! per-harmonic math inside the §0.5 estimator. This operates on the
//! input spectrum BEFORE the harmonic estimator sees it.
//!
//! Reference: Boll, *Suppression of Acoustic Noise in Speech Using
//! Spectral Subtraction*, IEEE TASSP 1979 (classic spectral
//! subtraction); Martin, *Noise Power Spectral Density Estimation
//! Based on Optimal Smoothing and Minimum Statistics*, IEEE TSAP
//! 2001 (per-bin running minimum noise tracker).
//!
//! Design:
//! - Per-bin running noise estimate `N(m)` updated via EMA on
//!   silence-flagged frames (the §0.8.4 silence detector is the
//!   speech-activity gate).
//! - On every frame, compute suppression gain
//!   `G(m) = sqrt(max(|S(m)|² - β·N(m), α·|S(m)|²) / |S(m)|²)`
//!   with `α` an over-subtraction floor preventing musical noise.
//! - Apply phase-preserving: `S'(m) = G(m) · S(m)`.

use super::basis::{Complex64, DFT_SIZE};

/// Over-subtraction multiplier `β`. Boll 1979 used `β = 1`. β-sweep
/// 2026-04-28 across the 5-vector set: knox_1 (noisy tone) peaks at
/// β=0.1 (+0.638 PESQ); β=0.05 is too gentle and *hurts* knox_1
/// (-0.122 PESQ) by introducing a subtractive bias without removing
/// noise; β≥0.5 over-subtracts speech (mark drops 0.36+ PESQ). 0.1
/// is the empirical sweet spot for the existing silence-detector
/// gating.
const SUBTRACTION_BETA: f64 = 0.1;

/// Suppression floor `α` — the gain `G(m)` is never allowed to drop
/// below `sqrt(α)`. Prevents nulls in the output spectrum (musical
/// noise). Berouti et al. 1979 recommend 0.01–0.05.
const SPECTRAL_FLOOR_ALPHA: f64 = 0.02;

/// EMA forget factor for noise-spectrum updates during silence frames.
/// `λ ∈ [0, 1)`; smaller = faster adaptation. Martin 2001 uses
/// long-window minimum statistics; we approximate with a slow EMA
/// (≈ 1 s effective window at 50 fps).
const NOISE_EMA_LAMBDA: f64 = 0.95;

/// Cold-start noise-spectrum value (per bin, in `|S|²` units).
const NOISE_INIT: f64 = 0.0;

/// Frame-energy / noise-floor ratio gate: a frame must satisfy
/// `frame_energy < STRICT_ENERGY_RATIO · η` to be eligible for a
/// noise-estimate update. Kept as a permissive secondary check —
/// the dominant filter is now spectral stationarity (below).
const STRICT_ENERGY_RATIO: f64 = 4.0;

/// Minimum consecutive eligible frames before any noise-estimate
/// update fires. Two frames is enough to reject single-frame
/// outliers without forcing the §0.8.4 silence-detector hysteresis
/// (which doesn't fire on continuous stationary tones like knox_1).
const STRICT_CONSECUTIVE_FRAMES: u32 = 2;

/// Cosine-similarity threshold between the current frame's
/// `|S(m)|²` spectrum and the running EMA of recent `|S(m)|²`
/// spectra. Above this → spectrum is stationary frame-to-frame →
/// eligible for noise-estimate update. Sweep 2026-04-28 across the
/// 5-vector set: 0.99 leaks into clean/alert; 0.995 leaks into
/// alert (formant of the tonal sweep momentarily stationary);
/// **0.999 hits all five vectors within the 0.05 PESQ goal** while
/// retaining +0.585 PESQ on knox_1.
///
/// The 2026-04-28 strict-gate follow-up showed energy alone can't
/// distinguish stationary tones (high energy, want to update) from
/// speech pauses (high residual energy, do NOT want to update).
/// Spectral stationarity targets what energy can't see: speech
/// changes the spectrum every frame (formant motion); stationary
/// tones / hum / fan noise hold the spectrum constant.
const STATIONARITY_COS_THRESHOLD: f64 = 0.999;

/// EMA forget factor for the magnitude-spectrum reference used by
/// the stationarity gate. Faster than the noise-PSD EMA — the
/// reference needs to track recent spectra so the cosine-similarity
/// test reflects "this frame matches the *recent* spectrum", not
/// "this frame matches the long-run average." 0.7 = ~3-frame
/// effective window.
const STATIONARITY_REF_LAMBDA: f64 = 0.7;

/// Per-bin noise-PSD estimate covering the same DFT layout as
/// `signal_spectrum` (256-entry, half-spectrum used).
#[derive(Clone, Debug)]
pub struct NoiseSpectrum {
    /// `n_psd[m]` = running estimate of `E[|N(m)|²]` for bin `m`.
    /// Indexed identically to `signal_spectrum`'s output.
    pub n_psd: [f64; DFT_SIZE],
    /// `true` after the first eligible-frame EMA update; before that
    /// `n_psd` is all-zero and `apply_subtraction` is a no-op (no
    /// noise model yet).
    pub primed: bool,
    /// Consecutive eligible-frame counter — saturates at
    /// `STRICT_CONSECUTIVE_FRAMES`. Updates only fire once this
    /// counter reaches the threshold, after which it stays armed
    /// until ineligibility resets it to zero.
    eligible_run: u32,
    /// Running EMA reference of `|S(m)|²` for the spectral-stationarity
    /// gate. Updated every frame (regardless of eligibility) so the
    /// gate has a recent baseline to compare against.
    spectrum_ref: [f64; DFT_SIZE],
    /// `true` after the first frame populates `spectrum_ref`. Before
    /// that the cosine-similarity test is undefined, so the gate
    /// treats it as ineligible.
    spectrum_ref_primed: bool,
}

impl NoiseSpectrum {
    /// Cold-start state — zeros, not primed, no eligible run.
    pub const fn new() -> Self {
        Self {
            n_psd: [NOISE_INIT; DFT_SIZE],
            primed: false,
            eligible_run: 0,
            spectrum_ref: [0.0; DFT_SIZE],
            spectrum_ref_primed: false,
        }
    }

    /// Reset to cold start (mirrors `Vocoder::reset`).
    pub fn reset(&mut self) {
        self.n_psd = [NOISE_INIT; DFT_SIZE];
        self.primed = false;
        self.eligible_run = 0;
        self.spectrum_ref = [0.0; DFT_SIZE];
        self.spectrum_ref_primed = false;
    }

    /// Cosine similarity between `|sw|²` and the running spectrum
    /// reference. Returns `1.0` when both vectors are identical (in
    /// direction); near `0` when orthogonal. Returns `0.0` if the
    /// reference isn't yet primed or either vector is all-zero.
    fn stationarity_cos_sim(&self, sw: &[Complex64; DFT_SIZE]) -> f64 {
        if !self.spectrum_ref_primed {
            return 0.0;
        }
        let mut dot = 0.0f64;
        let mut a_norm = 0.0f64;
        let mut b_norm = 0.0f64;
        for m in 0..DFT_SIZE {
            let a = sw[m].norm_sqr();
            let b = self.spectrum_ref[m];
            dot += a * b;
            a_norm += a * a;
            b_norm += b * b;
        }
        let denom = (a_norm * b_norm).sqrt();
        if denom <= 1e-30 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Update the noise estimate from the current frame's
    /// `signal_spectrum`. Three-stage gate:
    ///
    /// 1. **Spectral stationarity** — cosine similarity between
    ///    `|sw|²` and the running EMA reference must be ≥
    ///    `STATIONARITY_COS_THRESHOLD`. This is the dominant filter:
    ///    speech changes formant peaks every frame (low cos-sim);
    ///    stationary tones / hum / fan noise hold (high cos-sim).
    /// 2. **Permissive energy / silence** — `silent OR frame_energy <
    ///    STRICT_ENERGY_RATIO · η`. Either signal-detector path
    ///    suffices; the spectral-stationarity test does most of the
    ///    work.
    /// 3. **Consecutive-frame counter** — at least
    ///    `STRICT_CONSECUTIVE_FRAMES` eligible frames in a row to
    ///    reject single-frame outliers.
    ///
    /// The reference spectrum is updated every frame regardless of
    /// eligibility so the cos-sim test always has a recent baseline.
    pub fn update(&mut self, sw: &[Complex64; DFT_SIZE], silent: bool, frame_energy: f64, noise_floor_eta: f64) {
        // Stage 1: spectral stationarity test (uses prior reference).
        let cos_sim = self.stationarity_cos_sim(sw);
        let stationary = cos_sim >= STATIONARITY_COS_THRESHOLD;

        // Update the spectrum reference EVERY frame. A fast EMA so the
        // reference reflects the recent past, not the long-run mean.
        if !self.spectrum_ref_primed {
            for (m, slot) in self.spectrum_ref.iter_mut().enumerate() {
                *slot = sw[m].norm_sqr();
            }
            self.spectrum_ref_primed = true;
        } else {
            for (m, slot) in self.spectrum_ref.iter_mut().enumerate() {
                let cur = sw[m].norm_sqr();
                *slot = STATIONARITY_REF_LAMBDA * *slot
                    + (1.0 - STATIONARITY_REF_LAMBDA) * cur;
            }
        }

        // Stage 2: secondary energy / silence permissive check.
        let energy_ok = silent || frame_energy < STRICT_ENERGY_RATIO * noise_floor_eta;
        if !stationary || !energy_ok {
            self.eligible_run = 0;
            return;
        }

        // Stage 3: consecutive-eligible-frame counter.
        self.eligible_run = self.eligible_run.saturating_add(1);
        if self.eligible_run < STRICT_CONSECUTIVE_FRAMES {
            return;
        }
        if !self.primed {
            for (m, slot) in self.n_psd.iter_mut().enumerate() {
                *slot = sw[m].norm_sqr();
            }
            self.primed = true;
            return;
        }
        for (m, slot) in self.n_psd.iter_mut().enumerate() {
            let cur = sw[m].norm_sqr();
            *slot = NOISE_EMA_LAMBDA * *slot + (1.0 - NOISE_EMA_LAMBDA) * cur;
        }
    }
}

impl Default for NoiseSpectrum {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Apply spectral subtraction to `sw` using the running noise estimate.
/// No-op when `state.primed == false` (no noise model yet — happens
/// during the first speech burst before any silence frame is seen).
///
/// Returns a new `[Complex64; DFT_SIZE]` array; the caller chooses
/// whether to feed it to §0.5 (amplitude only) or substitute it
/// throughout (also affecting §0.4 refinement / §0.7 V/UV — see
/// `mod.rs` for the policy choice). Phase is preserved.
pub fn apply_subtraction(
    sw: &[Complex64; DFT_SIZE],
    state: &NoiseSpectrum,
) -> [Complex64; DFT_SIZE] {
    if !state.primed {
        return *sw;
    }
    let mut out = *sw;
    for m in 0..DFT_SIZE {
        let s_psd = sw[m].norm_sqr();
        if s_psd <= 1e-30 {
            continue;
        }
        let suppressed = (s_psd - SUBTRACTION_BETA * state.n_psd[m])
            .max(SPECTRAL_FLOOR_ALPHA * s_psd);
        let gain = (suppressed / s_psd).sqrt();
        out[m] = Complex64::new(sw[m].re * gain, sw[m].im * gain);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_spectrum(magnitude: f64) -> [Complex64; DFT_SIZE] {
        let mut sw = [Complex64::new(0.0, 0.0); DFT_SIZE];
        for s in &mut sw {
            *s = Complex64::new(magnitude, 0.0);
        }
        sw
    }

    #[test]
    fn unprimed_noise_state_is_passthrough() {
        let sw = flat_spectrum(2.0);
        let state = NoiseSpectrum::new();
        let out = apply_subtraction(&sw, &state);
        for m in 0..DFT_SIZE {
            assert!((out[m].re - sw[m].re).abs() < 1e-12);
        }
    }

    /// Helper: drive enough eligible frames so the strict gate primes.
    /// Needs `STRICT_CONSECUTIVE_FRAMES + 1` calls because the first
    /// call only seeds the spectrum reference (the cos-sim test fails
    /// on an unprimed reference). All calls pass `silent=true` and
    /// `frame_energy=0` so the secondary gate also passes.
    fn prime_with(state: &mut NoiseSpectrum, sw: &[Complex64; DFT_SIZE]) {
        for _ in 0..(STRICT_CONSECUTIVE_FRAMES + 1) {
            state.update(sw, true, /*frame_energy*/ 0.0, /*η*/ 1.0);
        }
    }

    #[test]
    fn strict_gate_requires_consecutive_stationary_frames() {
        let mut state = NoiseSpectrum::new();
        let noise = flat_spectrum(0.5);
        // First call only seeds the spectrum reference. Subsequent
        // STRICT_CONSECUTIVE_FRAMES - 1 calls don't yet prime the
        // noise estimator.
        for _ in 0..STRICT_CONSECUTIVE_FRAMES {
            state.update(&noise, true, 0.0, 1.0);
        }
        assert!(!state.primed);
        // One more eligible frame trips the gate.
        state.update(&noise, true, 0.0, 1.0);
        assert!(state.primed);
        for m in 0..DFT_SIZE {
            assert!((state.n_psd[m] - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn non_stationary_frame_breaks_eligible_run() {
        let mut state = NoiseSpectrum::new();
        let noise = flat_spectrum(0.5);
        // Drive a few stationary frames (seeds reference + builds run).
        for _ in 0..STRICT_CONSECUTIVE_FRAMES {
            state.update(&noise, true, 0.0, 1.0);
        }
        // A spectrum-shifted frame fails the stationarity test.
        let mut shifted = flat_spectrum(0.0);
        // Make spectrum drastically different from the all-0.5 reference:
        // concentrate all energy at one bin, the reference at all bins
        // → cos_sim ≈ 1/sqrt(N), well below the 0.99 threshold.
        shifted[10] = Complex64::new(100.0, 0.0);
        state.update(&shifted, true, 0.0, 1.0);
        // Run resets, primed not advanced.
        assert!(!state.primed);
    }

    #[test]
    fn voiced_frames_hold_noise_estimate() {
        let mut state = NoiseSpectrum::new();
        let noise = flat_spectrum(0.5);
        prime_with(&mut state, &noise);
        let baseline = state.n_psd[0];
        // A spectrum-shifted "voiced" frame fails the stationarity
        // gate AND the silent gate — must NOT change n_psd.
        let mut voice = flat_spectrum(0.0);
        voice[10] = Complex64::new(50.0, 0.0);
        state.update(&voice, false, 100.0, 1.0);
        for m in 0..DFT_SIZE {
            assert!((state.n_psd[m] - baseline).abs() < 1e-9,
                "voiced update leaked: bin {m} = {}", state.n_psd[m]);
        }
    }

    #[test]
    fn subtraction_attenuates_signal_at_or_below_noise_psd() {
        let mut state = NoiseSpectrum::new();
        // Train the noise estimate to |0.5|² = 0.25.
        let train = flat_spectrum(0.5);
        prime_with(&mut state, &train);
        // Apply to a signal at the same level. After β·N subtraction
        // the residual is max((1-β)·s_psd, α·s_psd); gain is
        // sqrt(max(1-β, α)). At β=0.1 → sqrt(0.9) ≈ 0.949;
        // at β=1.5 → sqrt(α) ≈ 0.141. Both regimes must yield gain
        // strictly less than 1 (the signal IS attenuated).
        let probe = flat_spectrum(0.5);
        let out = apply_subtraction(&probe, &state);
        let expected_gain = ((1.0 - SUBTRACTION_BETA).max(SPECTRAL_FLOOR_ALPHA)).sqrt();
        let expected_max = expected_gain * 0.5 + 1e-9;
        for m in 0..DFT_SIZE {
            assert!(out[m].re.abs() <= expected_max,
                "bin {m} expected ≤ {expected_max:.4}, got {}", out[m].re);
            assert!(out[m].re.abs() < 0.5,
                "bin {m} not attenuated at all: {}", out[m].re);
        }
    }

    #[test]
    fn subtraction_passes_strong_signal_through() {
        let mut state = NoiseSpectrum::new();
        let noise = flat_spectrum(0.1);
        prime_with(&mut state, &noise);
        // |signal|² = 100; after β·N = 1.5·0.01 = 0.015 subtraction,
        // gain ≈ sqrt((100 - 0.015) / 100) ≈ 0.99992 ≈ 1.
        let signal = flat_spectrum(10.0);
        let out = apply_subtraction(&signal, &state);
        for m in 0..DFT_SIZE {
            assert!((out[m].re - 10.0).abs() < 0.01,
                "bin {m} over-attenuated: {}", out[m].re);
        }
    }
}
