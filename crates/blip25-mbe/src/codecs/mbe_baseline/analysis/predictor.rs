//! §0.6 — Log-magnitude prediction residual.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.6 (BABA-A §6.3
//! Eq. 52–57, Figure 16). Produces T̂_l for 1 ≤ l ≤ L̂(0) — the log₂-
//! domain prediction residuals that feed the quantizer chain. Consumes
//! M̂_l(0) from §0.5, L̂(0) from §0.4, and matched-decoder state
//! M̃_l(−1)/L̃(−1) from the prior frame (closed-loop feedback per
//! §0.6.6: encoder runs the decoder internally).

use super::{L_HAT_MAX, SpectralAmplitudes};

/// Cold-start value of `L̃(−1)` per BABA-A §6.3 (page 25): mid-range
/// harmonic count that keeps `k̂_l` well-distributed on frame 0.
pub const L_TILDE_COLD_START: u8 = 30;

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
    /// `DecoderState` via [`crate::p25_fullrate::dequantize::DecoderState::from_amplitudes`].
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
/// Promoted from [`crate::p25_fullrate::dequantize::fullrate_rho`]'s
/// `f32` helper into `f64` for the analysis pipeline. Values are
/// identical; the `as f64` cast is lossless for these schedule points.
#[inline]
pub fn fullrate_rho_f64(l_hat: u8) -> f64 {
    f64::from(crate::p25_fullrate::dequantize::fullrate_rho(l_hat))
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
    let rho = fullrate_rho_f64(l_hat);

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
