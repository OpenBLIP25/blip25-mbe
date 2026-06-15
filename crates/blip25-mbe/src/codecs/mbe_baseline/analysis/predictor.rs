//! §0.6 — Log-magnitude prediction residual.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.6 (BABA-A §6.3
//! Eq. 52–57, Figure 16). Produces T̂_l for 1 ≤ l ≤ L̂(0) — the log₂-
//! domain prediction residuals that feed the quantizer chain. Consumes
//! M̂_l(0) from §0.5, L̂(0) from §0.4, and matched-decoder state
//! M̃_l(−1)/L̃(−1) from the prior frame (closed-loop feedback per
//! §0.6.6: encoder runs the decoder internally).

use super::{band_for_harmonic, SpectralAmplitudes, VuvResult, L_HAT_MAX};

/// Cold-start value of `L̃(−1)` per BABA-A §6.3 (page 25): mid-range
/// harmonic count that keeps `k̂_l` well-distributed on frame 0.
pub const L_TILDE_COLD_START: u8 = 30;

/// Half-rate cold-start value of `L̃(−1)` per BABA-A §14.3 (page 60)
/// and addendum §0.6.10. Distinct from the full-rate init (30).
pub const L_TILDE_COLD_START_HALFRATE: u8 = 15;

/// Half-rate prediction coefficient per BABA-A Eq. 155 (literal 0.65,
/// independent of `L̂`). See addendum §0.6.10 #3.
pub const RHO_HALFRATE: f64 = 0.65;

/// Half-rate Eq. 150's unvoiced adjustment constant:
/// `Λ̂_l = log₂ M̂_l + 0.5·log₂(ω̃_0·L̂) + 2.289` for unvoiced harmonics.
pub const LAMBDA_HAT_UNVOICED_BIAS: f64 = 2.289;

/// Log₂-domain prediction residuals `T̂_l`, 1-indexed; `[0]` unused.
pub type PredictionResidual = [f64; L_HAT_MAX as usize + 1];

/// Closed-loop predictor state per §0.6.7. Holds the matched-decoder
/// reconstruction `M̃_l(0)` from the previous frame (stored as
/// `M̃_l(−1)` here) together with its harmonic count `L̃(−1)`.
/// `m_tilde_prev[0]` is always `1.0` per Eq. 56.
#[derive(Clone, Debug)]
pub struct PredictorState {
    /// `M̃_l(−1)` in linear units. Index `0 = 1.0` (Eq. 56 DC boundary),
    /// indices `1..=l_tilde_prev` hold reconstructed amplitudes,
    /// higher slots are Eq. 57 constant-extrapolation placeholders
    /// (left as the final amplitude during commit).
    m_tilde_prev: [f64; L_HAT_MAX as usize + 1],
    /// `L̃(−1)` — previous frame's harmonic count.
    l_tilde_prev: u8,
}

impl PredictorState {
    /// Cold-start: `M̃_l(−1) = 1.0` for all l, `L̃(−1) = 30` per
    /// addendum §0.6.7 (BABA-A §6.3 prose verbatim).
    pub const fn cold_start() -> Self {
        Self {
            m_tilde_prev: [1.0; L_HAT_MAX as usize + 1],
            l_tilde_prev: L_TILDE_COLD_START,
        }
    }

    /// Read `M̃_l(−1)` with Eq. 56/57 boundary handling:
    /// - `l = 0` → `1.0` (Eq. 56 DC).
    /// - `l > L̃(−1)` → `M̃_{L̃(−1)}(−1)` (Eq. 57 constant extrapolation).
    pub(super) fn read(&self, l: u32) -> f64 {
        if l == 0 {
            return 1.0;
        }
        let clamped = (l as usize).min(self.l_tilde_prev as usize);
        self.m_tilde_prev[clamped]
    }

    /// 0-indexed slice of the past-frame amplitudes for `l = 1..=l_tilde_prev`.
    /// Entry `i` is `M̃_{i+1}(−1)`. Length equals `l_tilde_prev`.
    /// Used by the §0.10 matched-decoder roundtrip to seed a
    /// `DecoderState` via [`crate::imbe7200::dequantize::DecoderState::from_amplitudes`].
    pub fn m_tilde_prev_slice(&self) -> Vec<f32> {
        let n = self.l_tilde_prev as usize;
        (1..=n).map(|i| self.m_tilde_prev[i] as f32).collect()
    }

    /// Previous frame's harmonic count `L̃(−1)`.
    pub fn l_tilde_prev(&self) -> u8 {
        self.l_tilde_prev
    }

    /// Commit `M̃_l(0), L̃(0)` from the matched-decoder pass as the
    /// next frame's `M̃_l(−1), L̃(−1)`. Called by the §0.10
    /// end-to-end pipeline after the quantizer + full-rate inverse
    /// pipeline reconstructs `M̃_l(0)`.
    ///
    /// `m_tilde_curr` is 1-indexed with `[0]` unused; the caller
    /// supplies at least `l_tilde_curr + 1` entries.
    pub fn commit(&mut self, m_tilde_curr: &[f64], l_tilde_curr: u8) {
        assert!(l_tilde_curr as usize <= L_HAT_MAX as usize);
        assert!(m_tilde_curr.len() > l_tilde_curr as usize);
        self.m_tilde_prev[0] = 1.0;
        for l in 1..=l_tilde_curr as usize {
            self.m_tilde_prev[l] = m_tilde_curr[l];
        }
        // Eq. 57 is applied at read time via clamping; no need to
        // fill slots past `l_tilde_curr` here.
        self.l_tilde_prev = l_tilde_curr;
    }
}

impl Default for PredictorState {
    fn default() -> Self {
        Self::cold_start()
    }
}

/// Full-rate prediction coefficient `ρ` per BABA-A Eq. 55.
///
/// Promoted from [`crate::imbe7200::dequantize::imbe_rho`]'s
/// `f32` helper into `f64` for the analysis pipeline. Values are
/// identical; the `as f64` cast is lossless for these schedule points.
#[inline]
pub fn imbe_rho_f64(l_hat: u8) -> f64 {
    f64::from(crate::imbe7200::dequantize::imbe_rho(l_hat))
}

/// Compute the log₂-domain prediction residual `T̂_l` per Eq. 54 in
/// its mean-removed form:
///
/// ```text
/// T̂_l = log₂ M̂_l(0) − ρ · ( P_l − mean_λ P_λ )
/// ```
///
/// where `P_l` is the linearly-interpolated past-frame log-amplitude
/// at harmonic `l` using Eq. 52/53 indexing and Eq. 56/57 boundary
/// handling. The mean-removal keeps `T̂` DC-free so the gain vector
/// `G_1` carries the absolute log-envelope level.
///
/// For harmonics where `M̂_l(0) ≤ 0` (degenerate / silent band), the
/// log₂ is not defined; the canonical convention (per addendum §0.5.6
/// "Near-zero denominators") is to emit `T̂_l = 0`, which the decoder
/// tolerates via its own floor. We implement that here by routing
/// non-positive amplitudes to `log₂ = 0`.
pub fn compute_prediction_residual(
    m_hat: &SpectralAmplitudes,
    l_hat: u8,
    state: &PredictorState,
) -> PredictionResidual {
    let mut t_hat = [0.0f64; L_HAT_MAX as usize + 1];
    if l_hat == 0 {
        return t_hat;
    }
    let l_hat_u32 = u32::from(l_hat);
    let l_prev_u32 = u32::from(state.l_tilde_prev.max(1));
    let ratio = f64::from(state.l_tilde_prev) / f64::from(l_hat);
    let rho = imbe_rho_f64(l_hat);

    // Eq. 52/53 + Eq. 56/57: build P_l for all l = 1..=L̂.
    let mut p = [0.0f64; L_HAT_MAX as usize + 1];
    let mut p_sum = 0.0;
    for l in 1..=l_hat_u32 {
        let k_hat = ratio * f64::from(l);
        let k_floor = k_hat.floor();
        let delta = k_hat - k_floor;
        let kf = k_floor as i64;

        let lo = if kf < 0 {
            1.0 // Guard (shouldn't happen for non-negative ratio · l).
        } else if (kf as u32) <= l_prev_u32 {
            state.read(kf as u32)
        } else {
            state.read(l_prev_u32) // Eq. 57.
        };
        let hi = if kf < 0 {
            state.read(0)
        } else if (kf as u32 + 1) <= l_prev_u32 {
            state.read(kf as u32 + 1)
        } else {
            state.read(l_prev_u32) // Eq. 57.
        };

        let log_lo = if lo > 0.0 { lo.log2() } else { 0.0 };
        let log_hi = if hi > 0.0 { hi.log2() } else { 0.0 };
        let p_l = (1.0 - delta) * log_lo + delta * log_hi;
        p[l as usize] = p_l;
        p_sum += p_l;
    }
    let p_mean = p_sum / f64::from(l_hat);

    // Eq. 54 mean-removed form.
    for l in 1..=l_hat_u32 {
        let m_l = m_hat[l as usize];
        let log_m = if m_l > 0.0 { m_l.log2() } else { 0.0 };
        t_hat[l as usize] = log_m - rho * (p[l as usize] - p_mean);
    }
    t_hat
}

#[cfg(test)]
mod imbe_tests {
    use super::*;

    #[test]
    fn predictor_state_cold_start_matches_spec() {
        let st = PredictorState::cold_start();
        assert_eq!(st.l_tilde_prev(), L_TILDE_COLD_START);
        // M̃_l(−1) = 1.0 for all l per §6.3 prose; log₂ 1 = 0 makes the
        // first-frame predictor contribution vanish.
        for l in 0..=L_HAT_MAX as u32 {
            assert_eq!(st.read(l), 1.0, "M̃_{l}(−1) = {}", st.read(l));
        }
    }

    #[test]
    fn predictor_state_read_applies_eq56_eq57_boundaries() {
        let mut st = PredictorState::cold_start();
        // Commit a short vector with distinct values to see the boundaries.
        let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
        m_tilde[1] = 2.0;
        m_tilde[2] = 4.0;
        m_tilde[3] = 8.0;
        st.commit(&m_tilde, 3);
        // Eq. 56: M̃_0 = 1.0.
        assert_eq!(st.read(0), 1.0);
        // Interior values.
        assert_eq!(st.read(1), 2.0);
        assert_eq!(st.read(3), 8.0);
        // Eq. 57: beyond L̃(−1), clamp to last.
        assert_eq!(st.read(4), 8.0);
        assert_eq!(st.read(30), 8.0);
    }

    /// On cold start, `M̃_l(−1) = 1.0` for all l, so `log₂` of every
    /// past value is 0 and the predictor contributes zero. The
    /// residual `T̂_l` equals `log₂ M̂_l(0)` exactly.
    #[test]
    fn compute_prediction_residual_cold_start_equals_log2_amplitude() {
        let st = PredictorState::cold_start();
        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        // Fill with distinct positive values.
        for l in 1..=12u32 {
            m_hat[l as usize] = f64::from(l) * 0.5 + 1.0;
        }
        let t_hat = compute_prediction_residual(&m_hat, 12, &st);
        for l in 1..=12u32 {
            let expected = m_hat[l as usize].log2();
            assert!(
                (t_hat[l as usize] - expected).abs() < 1e-12,
                "T̂_{l} = {}, expected log₂({}) = {}",
                t_hat[l as usize],
                m_hat[l as usize],
                expected
            );
        }
    }

    /// Mean-preservation invariant: `mean_l T̂_l = mean_l log₂ M̂_l(0)`
    /// by construction (the mean-removed predictor term has mean 0).
    #[test]
    fn compute_prediction_residual_preserves_mean_of_log2_amplitude() {
        let mut st = PredictorState::cold_start();
        // Seed past state with non-trivial values so ρ·P has teeth.
        let mut past = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=20u32 {
            past[l as usize] = 1.0 + f64::from(l) * 0.1; // M̃ ∈ [1.1, 3.0]
        }
        st.commit(&past, 20);

        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=18u32 {
            m_hat[l as usize] = 0.5 + f64::from(l) * 0.2;
        }
        let l_hat = 18u8;
        let t_hat = compute_prediction_residual(&m_hat, l_hat, &st);

        let t_mean: f64 = (1..=u32::from(l_hat))
            .map(|l| t_hat[l as usize])
            .sum::<f64>()
            / f64::from(l_hat);
        let log_m_mean: f64 = (1..=u32::from(l_hat))
            .map(|l| m_hat[l as usize].log2())
            .sum::<f64>()
            / f64::from(l_hat);
        assert!(
            (t_mean - log_m_mean).abs() < 1e-12,
            "mean(T̂) = {t_mean}, mean(log₂ M̂) = {log_m_mean}"
        );
    }

    /// When `M̂_l(0) = M̃_l(−1)` exactly (perfect prediction),
    /// `T̂_l = (1−ρ)·log₂ M̂_l + ρ·mean(log₂ M̂)` — a partial blend
    /// toward the mean weighted by the predictor coefficient.
    #[test]
    fn compute_prediction_residual_partial_blend_when_prediction_perfect() {
        let mut st = PredictorState::cold_start();
        let mut m_same = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=16u32 {
            m_same[l as usize] = 1.0 + f64::from(l) * 0.3;
        }
        st.commit(&m_same, 16);
        let t_hat = compute_prediction_residual(&m_same, 16, &st);

        let log_mean: f64 = (1..=16u32).map(|l| m_same[l as usize].log2()).sum::<f64>() / 16.0;
        let rho = imbe_rho_f64(16);
        for l in 1..=16u32 {
            let log_m = m_same[l as usize].log2();
            let expected = (1.0 - rho) * log_m + rho * log_mean;
            assert!(
                (t_hat[l as usize] - expected).abs() < 1e-9,
                "T̂_{l} = {}, expected (1-ρ)·log + ρ·mean = {}",
                t_hat[l as usize],
                expected
            );
        }
    }

    /// `L̂ ≠ L̃(−1)` mismatch: Eq. 52/53 linear interpolation handles
    /// the harmonic-index mapping. Sanity-check that the residual is
    /// well-defined (finite) across a rising/falling L̂ transition.
    #[test]
    fn compute_prediction_residual_handles_l_hat_mismatch() {
        let mut st = PredictorState::cold_start();
        let mut past = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=10u32 {
            past[l as usize] = 2.0_f64.powf(f64::from(l) * 0.1);
        }
        st.commit(&past, 10); // L̃(−1) = 10

        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=25u32 {
            m_hat[l as usize] = 2.0_f64.powf(f64::from(l) * 0.05);
        }
        // Current frame has more harmonics than the past frame.
        let t_hat = compute_prediction_residual(&m_hat, 25, &st);
        for l in 1..=25u32 {
            assert!(
                t_hat[l as usize].is_finite(),
                "T̂_{l} = {}",
                t_hat[l as usize]
            );
        }
        // And the other direction: current frame has fewer harmonics.
        let t_hat_shrink = compute_prediction_residual(&m_hat, 5, &st);
        for l in 1..=5u32 {
            assert!(
                t_hat_shrink[l as usize].is_finite(),
                "T̂_{l} (shrink) = {}",
                t_hat_shrink[l as usize]
            );
        }
    }

    /// `imbe_rho_f64` matches the Eq. 55 schedule (within the
    /// f32→f64 cast precision of the underlying helper).
    #[test]
    fn imbe_rho_matches_eq55_schedule_points() {
        let tol = 1e-5;
        // L̂ ≤ 15 → 0.40.
        assert!((imbe_rho_f64(9) - 0.40).abs() < tol);
        assert!((imbe_rho_f64(15) - 0.40).abs() < tol);
        // 15 < L̂ ≤ 24 → 0.03·L − 0.05.
        assert!((imbe_rho_f64(16) - 0.43).abs() < tol);
        assert!((imbe_rho_f64(20) - 0.55).abs() < tol);
        assert!((imbe_rho_f64(24) - 0.67).abs() < tol);
        // L̂ > 24 → 0.70.
        assert!((imbe_rho_f64(25) - 0.70).abs() < tol);
        assert!((imbe_rho_f64(56) - 0.70).abs() < tol);
    }
}

// ---------------------------------------------------------------------------
// Half-rate predictor (addendum §0.6.10 / BABA-A §14.3, Eq. 150–157).
//
// Structural deltas from full-rate:
// - State stored in log₂ domain (`Λ̃_l(−1)`) rather than linear.
// - `L̃(−1) = 15` cold-start, not 30.
// - Additional `γ̃(−1)` differential-gain scalar (Eq. 152 state).
// - Voicing-dependent Eq. 150 `M̂_l → Λ̂_l` conversion before predictor.
// - `ρ = 0.65` literal (per §0.6.2 / Eq. 155), not the full-rate schedule.
// - DC boundary `Λ̃_0(−1) = Λ̃_1(−1)` (Eq. 156), not `M̃_0 = 1` (Eq. 56).
// ---------------------------------------------------------------------------

/// Half-rate predictor state per addendum §0.6.10 (log₂-domain Λ̃,
/// L̃ init 15, plus an Eq. 152 differential-gain `γ̃(−1)` scalar).
///
/// Distinct from [`PredictorState`] — the two are **not**
/// interchangeable (state domain differs, init L̃ differs, γ̃ is
/// half-rate-only). An encoder that supports both rates keeps
/// separate instances per §0.6.10's "split state per rate" guidance.
#[derive(Clone, Debug)]
pub struct HalfratePredictorState {
    /// `Λ̃_l(−1)` in log₂ domain, 1-indexed. Cold start is `1.0` (not
    /// `0.0` — see addendum §0.6.10 "unusual half-rate init"). Slot
    /// `[0]` is unused and reads via [`Self::read`] apply Eq. 156.
    lambda_tilde_prev: [f64; L_HAT_MAX as usize + 1],
    /// `L̃(−1)` — previous frame's harmonic count. Init 15 per
    /// BABA-A §14.3.
    l_tilde_prev: u8,
    /// `γ̃(−1)` — Eq. 152 differential-gain state. Init 0.0 per
    /// BABA-A §14.3. Only voice frames advance this (tone / silence /
    /// erasure frames leave it unchanged per §13.3 parallel).
    gamma_tilde_prev: f64,
}

impl HalfratePredictorState {
    /// Cold-start: `Λ̃_l(−1) = 1.0` for all l, `L̃(−1) = 15`,
    /// `γ̃(−1) = 0.0` per BABA-A §14.3 and addendum §0.6.10
    /// recommended shape.
    pub const fn cold_start() -> Self {
        Self {
            lambda_tilde_prev: [1.0; L_HAT_MAX as usize + 1],
            l_tilde_prev: L_TILDE_COLD_START_HALFRATE,
            gamma_tilde_prev: 0.0,
        }
    }

    /// Read `Λ̃_l(−1)` with Eq. 156/157 boundary handling:
    /// - `l = 0` → `Λ̃_1(−1)` (Eq. 156 — DC couples to first harmonic;
    ///   differs from full-rate Eq. 56's `M̃_0 = 1.0`).
    /// - `l > L̃(−1)` → `Λ̃_{L̃(−1)}(−1)` (Eq. 157 constant extrapolation).
    pub(super) fn read(&self, l: u32) -> f64 {
        let l_prev = self.l_tilde_prev as usize;
        if l == 0 {
            // Eq. 156: Λ̃_0 tracks Λ̃_1. Index 1 is always in range (the
            // array holds L_HAT_MAX+1 ≥ 2 entries) and is the spec target
            // regardless of L̃(−1), so read it directly.
            return self.lambda_tilde_prev[1];
        }
        let clamped = (l as usize).min(l_prev.max(1));
        self.lambda_tilde_prev[clamped]
    }

    /// Previous frame's harmonic count `L̃(−1)`.
    pub fn l_tilde_prev(&self) -> u8 {
        self.l_tilde_prev
    }

    /// Previous frame's `γ̃(−1)` differential-gain state.
    pub fn gamma_tilde_prev(&self) -> f64 {
        self.gamma_tilde_prev
    }

    /// Borrow the stored `Λ̃_l(−1)` vector. Slot `[0]` is unused;
    /// meaningful data lives in `[1..=l_tilde_prev]`. Used by the
    /// matched-decoder roundtrip to seed the half-rate wire decoder.
    pub fn lambda_tilde_prev(&self) -> &[f64; L_HAT_MAX as usize + 1] {
        &self.lambda_tilde_prev
    }

    /// Commit `Λ̃_l(0)`, `L̃(0)`, `γ̃(0)` from the matched-decoder
    /// roundtrip as next frame's state. Called by the §0.10
    /// half-rate pipeline after
    /// [`crate::rate33::dequantize::dequantize`] advances the
    /// wire decoder's own `DecoderState`.
    ///
    /// `lambda_tilde_curr` is 1-indexed with `[0]` ignored; the
    /// caller supplies at least `l_tilde_curr + 1` entries.
    pub fn commit(&mut self, lambda_tilde_curr: &[f64], l_tilde_curr: u8, gamma_tilde_curr: f64) {
        assert!(l_tilde_curr as usize <= L_HAT_MAX as usize);
        assert!(lambda_tilde_curr.len() > l_tilde_curr as usize);
        for l in 1..=l_tilde_curr as usize {
            self.lambda_tilde_prev[l] = lambda_tilde_curr[l];
        }
        self.l_tilde_prev = l_tilde_curr;
        self.gamma_tilde_prev = gamma_tilde_curr;
    }
}

impl Default for HalfratePredictorState {
    fn default() -> Self {
        Self::cold_start()
    }
}

/// Convert the rate-agnostic per-harmonic magnitude `M̂_l(0)` (from
/// §0.5) into the half-rate predictor's log-domain input `Λ̂_l(0)` per
/// BABA-A Eq. 150 (page 60). This is the voicing-dependent bridge:
///
/// ```text
/// Λ̂_l = log₂ M̂_l + 0.5·log₂ L̂                  if ṽ_l = 1  (voiced)
/// Λ̂_l = log₂ M̂_l + 0.5·log₂(ω̃_0·L̂) + 2.289    otherwise    (unvoiced)
/// ```
///
/// `m_hat[l]` for `l = 1..=l_hat` is the linear-domain magnitude;
/// returns a 1-indexed `Λ̂` array.
///
/// Per §0.6.10 #1: the voicing-dependent adjustment compensates for
/// the decoder's Eq. 188 inverse scaling (voiced vs unvoiced
/// amplitude reconstruction); storing `Λ̂` avoids the encoder and
/// decoder disagreeing on which side of the factor the value lives.
///
/// `M̂_l ≤ 0` routes to `log₂ M̂_l = 0` (the §0.5.6 convention).
pub fn lambda_hat_from_m_hat(
    m_hat: &SpectralAmplitudes,
    vuv: &VuvResult,
    omega_hat_0: f64,
    l_hat: u8,
) -> [f64; L_HAT_MAX as usize + 1] {
    let mut lambda_hat = [0.0f64; L_HAT_MAX as usize + 1];
    if l_hat == 0 {
        return lambda_hat;
    }
    let l_hat_f = f64::from(l_hat);
    let log2_l_hat = l_hat_f.log2();
    let log2_wl = (omega_hat_0 * l_hat_f).log2();
    let k_hat = vuv.k_hat;
    for l in 1..=u32::from(l_hat) {
        let m_l = m_hat[l as usize];
        let log_m = if m_l > 0.0 { m_l.log2() } else { 0.0 };
        let k = band_for_harmonic(l, k_hat);
        let is_voiced = k >= 1 && vuv.vuv[k as usize] == 1;
        lambda_hat[l as usize] = if is_voiced {
            log_m + 0.5 * log2_l_hat
        } else {
            log_m + 0.5 * log2_wl + LAMBDA_HAT_UNVOICED_BIAS
        };
    }
    lambda_hat
}

/// Compute half-rate prediction residual `T̂_l` per BABA-A Eq. 155
/// (page 60). Structurally identical to full-rate Eq. 54 with
/// `log₂ M̂ → Λ̂`, `log₂ M̃ → Λ̃`, and `ρ = 0.65` literal.
///
/// `lambda_hat[l]` for `l = 1..=l_hat` is the Eq. 150-adjusted
/// log-magnitude (produced by [`lambda_hat_from_m_hat`]).
///
/// The mean-removal matches full-rate — `+ (ρ/L̂)·Σ Λ̃` inside the
/// bracket, not `-`. The sign is per Eq. 155 direct reading.
pub fn compute_prediction_residual_ambe_plus2(
    lambda_hat: &[f64; L_HAT_MAX as usize + 1],
    l_hat: u8,
    state: &HalfratePredictorState,
) -> PredictionResidual {
    let mut t_hat = [0.0f64; L_HAT_MAX as usize + 1];
    if l_hat == 0 {
        return t_hat;
    }
    let l_hat_u32 = u32::from(l_hat);
    let l_prev_u32 = u32::from(state.l_tilde_prev.max(1));
    let ratio = f64::from(state.l_tilde_prev) / f64::from(l_hat);

    // Eq. 153/154 + Eq. 156/157: build P_l for all l = 1..=L̂ from
    // log-domain Λ̃ directly (no log₂ call per access — we already
    // store the log form).
    let mut p = [0.0f64; L_HAT_MAX as usize + 1];
    let mut p_sum = 0.0;
    for l in 1..=l_hat_u32 {
        let k_hat_f = ratio * f64::from(l);
        let k_floor = k_hat_f.floor();
        let delta = k_hat_f - k_floor;
        let kf = k_floor as i64;

        let lo = if kf < 0 {
            state.read(0)
        } else if (kf as u32) <= l_prev_u32 {
            state.read(kf as u32)
        } else {
            state.read(l_prev_u32) // Eq. 157.
        };
        let hi = if kf < 0 {
            state.read(0)
        } else if (kf as u32 + 1) <= l_prev_u32 {
            state.read(kf as u32 + 1)
        } else {
            state.read(l_prev_u32)
        };

        let p_l = (1.0 - delta) * lo + delta * hi;
        p[l as usize] = p_l;
        p_sum += p_l;
    }
    let p_mean = p_sum / f64::from(l_hat);

    // Eq. 155 mean-removed form (sign convention matches Eq. 54).
    for l in 1..=l_hat_u32 {
        t_hat[l as usize] = lambda_hat[l as usize] - RHO_HALFRATE * (p[l as usize] - p_mean);
    }
    t_hat
}

#[cfg(test)]
mod ambe_plus2_tests {
    use super::*;

    fn zero_vuv(k_hat: u8) -> VuvResult {
        VuvResult {
            k_hat,
            vuv: [0; super::super::K_HAT_MAX as usize + 1],
            d_k: [0.0; super::super::K_HAT_MAX as usize + 1],
            theta_k: [0.0; super::super::K_HAT_MAX as usize + 1],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: 0.0,
        }
    }

    fn all_voiced(k_hat: u8) -> VuvResult {
        let mut v = [0u8; super::super::K_HAT_MAX as usize + 1];
        for k in 1..=k_hat as usize {
            v[k] = 1;
        }
        VuvResult {
            k_hat,
            vuv: v,
            d_k: [0.0; super::super::K_HAT_MAX as usize + 1],
            theta_k: [0.0; super::super::K_HAT_MAX as usize + 1],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: 0.0,
        }
    }

    #[test]
    fn ambe_plus2_predictor_cold_start_values() {
        let st = HalfratePredictorState::cold_start();
        assert_eq!(st.l_tilde_prev(), 15);
        assert_eq!(st.gamma_tilde_prev(), 0.0);
        for l in 0..=L_HAT_MAX as usize {
            assert_eq!(st.lambda_tilde_prev[l], 1.0);
        }
    }

    #[test]
    fn ambe_plus2_predictor_read_applies_eq156_eq157() {
        let mut st = HalfratePredictorState::cold_start();
        // Seed distinct values into lambda_tilde_prev[1..=5]; rest is
        // cold-start 1.0 but we'll lower L̃ to 5 so Eq. 157 fires
        // beyond that.
        st.lambda_tilde_prev[1] = 0.1;
        st.lambda_tilde_prev[2] = 0.2;
        st.lambda_tilde_prev[3] = 0.3;
        st.lambda_tilde_prev[4] = 0.4;
        st.lambda_tilde_prev[5] = 0.5;
        st.l_tilde_prev = 5;

        // Eq. 156: read(0) == Λ̃_1.
        assert_eq!(st.read(0), 0.1);
        // In-range reads pass through.
        assert_eq!(st.read(1), 0.1);
        assert_eq!(st.read(5), 0.5);
        // Eq. 157: l > L̃(−1) clamps to Λ̃_{L̃(−1)}.
        assert_eq!(st.read(6), 0.5);
        assert_eq!(st.read(999), 0.5);
    }

    #[test]
    fn lambda_hat_voiced_adjustment_matches_eq150() {
        // Single-harmonic test: L̂ = 10, ω̂_0 = 0.3, M̂_1 = 16.
        // Voiced case: Λ̂ = log₂(16) + 0.5·log₂(10) = 4 + 1.66096 ≈ 5.66.
        let l_hat = 10u8;
        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        m_hat[1] = 16.0;
        let vuv = all_voiced(4); // K̂ = 4 so band 1 covers l = 1..=3
        let lambda_hat = lambda_hat_from_m_hat(&m_hat, &vuv, 0.3, l_hat);
        let expected = (16.0_f64).log2() + 0.5 * (10.0_f64).log2();
        assert!(
            (lambda_hat[1] - expected).abs() < 1e-12,
            "lambda_hat[1] = {} vs expected {}",
            lambda_hat[1],
            expected
        );
    }

    #[test]
    fn lambda_hat_unvoiced_adds_2289_plus_omega_l_term() {
        // Unvoiced case: Λ̂ = log₂(16) + 0.5·log₂(ω̃_0·L̂) + 2.289.
        // With ω̃_0 = 0.3, L̂ = 10: 0.5·log₂(3.0) + 2.289 ≈ 0.7925 + 2.289 = 3.0815.
        // Plus log₂(16) = 4. Total ≈ 7.0815.
        let l_hat = 10u8;
        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        m_hat[1] = 16.0;
        let vuv = zero_vuv(4);
        let lambda_hat = lambda_hat_from_m_hat(&m_hat, &vuv, 0.3, l_hat);
        let expected = (16.0_f64).log2() + 0.5 * (0.3_f64 * 10.0).log2() + LAMBDA_HAT_UNVOICED_BIAS;
        assert!(
            (lambda_hat[1] - expected).abs() < 1e-12,
            "lambda_hat[1] = {} vs expected {}",
            lambda_hat[1],
            expected
        );
    }

    #[test]
    fn lambda_hat_zero_magnitude_uses_zero_log_not_neg_infinity() {
        // M̂_l = 0 → log₂ = 0 (addendum §0.5.6 convention).
        let l_hat = 5u8;
        let m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        let vuv = all_voiced(2);
        let lambda_hat = lambda_hat_from_m_hat(&m_hat, &vuv, 0.2, l_hat);
        for l in 1..=l_hat as usize {
            // All voiced → Λ̂_l = 0 + 0.5·log₂(5) ≈ 1.161.
            let expected = 0.5 * (5.0_f64).log2();
            assert!(
                (lambda_hat[l] - expected).abs() < 1e-12,
                "l={l}: {} vs {}",
                lambda_hat[l],
                expected
            );
        }
    }

    #[test]
    fn prediction_residual_ambe_plus2_cold_start_equals_lambda_hat_minus_rho_residual() {
        // Cold-start state: all Λ̃ = 1.0, L̃(−1) = 15. Feed a flat Λ̂
        // = 3.0 for L̂ = 12 harmonics. Predictor term P_l is flat at
        // 1.0 (k̂_l varies 1..1.25, but reads all land on cold-start
        // 1.0 regardless), so mean equals P_l for every l and the
        // mean-removal cancels. T̂_l = Λ̂_l = 3.0.
        let l_hat = 12u8;
        let mut lambda_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=l_hat as usize {
            lambda_hat[l] = 3.0;
        }
        let st = HalfratePredictorState::cold_start();
        let t_hat = compute_prediction_residual_ambe_plus2(&lambda_hat, l_hat, &st);
        for l in 1..=l_hat as usize {
            assert!((t_hat[l] - 3.0).abs() < 1e-12, "l={l}: {} vs 3.0", t_hat[l]);
        }
    }

    #[test]
    fn prediction_residual_ambe_plus2_perfect_prediction_gives_zero_mean() {
        // Perfect-prediction check: Λ̂_l matches a rescaled Λ̃_l
        // exactly. After interpolation and mean-removal the residual
        // should be mean-zero (Σ T̂_l = 0) because Eq. 155's structure
        // is mean-preserving from Λ̂ to Λ̂ − ρ·(P − P̄).
        let l_hat = 20u8;
        let mut lambda_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=l_hat as usize {
            lambda_hat[l] = (l as f64) * 0.1;
        }
        let mut st = HalfratePredictorState::cold_start();
        st.l_tilde_prev = 20;
        for l in 1..=20 {
            st.lambda_tilde_prev[l] = (l as f64) * 0.1;
        }
        let t_hat = compute_prediction_residual_ambe_plus2(&lambda_hat, l_hat, &st);
        // Σ T̂_l = Σ Λ̂_l − ρ·Σ (P_l − P̄) = Σ Λ̂_l − 0 = Σ Λ̂_l.
        let sum_t: f64 = t_hat[1..=l_hat as usize].iter().sum();
        let sum_lambda: f64 = lambda_hat[1..=l_hat as usize].iter().sum();
        assert!(
            (sum_t - sum_lambda).abs() < 1e-10,
            "Σ T̂ = {} vs Σ Λ̂ = {} (mean-removal should cancel on perfect prediction)",
            sum_t,
            sum_lambda,
        );
    }

    #[test]
    fn ambe_plus2_predictor_commit_advances_state() {
        let mut st = HalfratePredictorState::cold_start();
        let mut lambda_curr = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=12 {
            lambda_curr[l] = l as f64 * 0.25;
        }
        st.commit(&lambda_curr, 12, 5.0);
        assert_eq!(st.l_tilde_prev(), 12);
        assert_eq!(st.gamma_tilde_prev(), 5.0);
        for l in 1..=12 {
            assert_eq!(st.lambda_tilde_prev[l], l as f64 * 0.25);
        }
    }
}
