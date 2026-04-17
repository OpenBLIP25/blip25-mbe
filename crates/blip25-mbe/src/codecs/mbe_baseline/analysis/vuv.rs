//! §0.7 — Voiced/Unvoiced determination.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.7 (BABA-A §5.2
//! Eq. 34–42, Figure 11). Per-band voicing decisions v̂_k ∈ {0, 1} for
//! 1 ≤ k ≤ K̂, where K̂ is derived from L̂ via Eq. 34. Consumes
//! precomputed S_w(m) from §0.2, ω̂_0/L̂/â_l/b̂_l from §0.4, and
//! E(P̂_I) from §0.3.

use super::{Complex64, DFT_SIZE, HarmonicBasis, PitchRefinement, packed_index};

/// Maximum band count `K̂` per Eq. 34 (cap).
pub const K_HAT_MAX: u8 = 12;

/// Minimum `ξ_max` floor per Eq. 41 Case 3.
pub const XI_MAX_FLOOR: f64 = 20000.0;

/// Threshold for Eq. 37 Case 1 (poor pitch match).
const E_P_POOR_MATCH_THRESHOLD: f64 = 0.5;

/// `Θ_ξ` base when the band was voiced last frame (hysteresis up).
const THETA_BASE_VOICED_HISTORY: f64 = 0.5625;

/// `Θ_ξ` base when the band was unvoiced last frame (hysteresis down).
const THETA_BASE_UNVOICED_HISTORY: f64 = 0.45;

/// Coefficient of `(k−1)·ω̂_0` in the `Θ_ξ` pitch/band modulation.
const THETA_PITCH_BAND_COEF: f64 = 0.3096;

/// Cross-frame V/UV encoder state per §0.7.8. `v̂_k(−1)` is stored
/// 1-indexed in `vuv_prev[1..=K_HAT_MAX]`; slot 0 is unused.
#[derive(Clone, Debug)]
pub struct VuvState {
    /// `ξ_max(−1)` — previous frame's tracked peak envelope.
    pub(super) xi_max: f64,
    /// `v̂_k(−1)` — previous frame's decisions, 1-indexed.
    pub(super) vuv_prev: [u8; (K_HAT_MAX + 1) as usize],
    /// `K̂(−1)` — previous frame's band count (unused slots past this
    /// index are 0).
    pub(super) k_prev: u8,
}

impl VuvState {
    /// Cold-start initialization per §0.7.8: `ξ_max = XI_MAX_FLOOR`,
    /// `v̂_k(−1) = 0` for all `k`, `K̂(−1) = 0`.
    pub const fn cold_start() -> Self {
        Self {
            xi_max: XI_MAX_FLOOR,
            vuv_prev: [0; (K_HAT_MAX + 1) as usize],
            k_prev: 0,
        }
    }

    /// Current tracked peak-envelope value (for inspection / tests).
    pub fn xi_max(&self) -> f64 {
        self.xi_max
    }

    /// Previous-frame decision for band `k` (1-indexed). Returns 0 for
    /// out-of-range `k` (consistent with new-index cold-start behavior
    /// described in §0.7.8).
    pub fn vuv_prev(&self, k: u8) -> u8 {
        if (1..=K_HAT_MAX).contains(&k) {
            self.vuv_prev[k as usize]
        } else {
            0
        }
    }

    /// Override the per-band V/UV history with external decisions.
    /// Used for conformance diagnostics that inject the chip's V/UV
    /// trajectory to measure feedback-divergence contribution.
    /// `decisions` is 1-indexed (`decisions[k]` for band `k`);
    /// `k_hat` is the band count to write.
    pub fn override_vuv_prev(&mut self, decisions: &[u8], k_hat: u8) {
        self.vuv_prev.fill(0);
        let n = (k_hat as usize).min(decisions.len()).min(K_HAT_MAX as usize);
        for k in 1..=n {
            self.vuv_prev[k] = decisions[k];
        }
        self.k_prev = k_hat;
    }

    /// Override ξ_max with an external value.
    pub fn override_xi_max(&mut self, xi_max: f64) {
        self.xi_max = xi_max;
    }
}

impl Default for VuvState {
    fn default() -> Self {
        Self::cold_start()
    }
}

/// Output of [`determine_vuv`].
#[derive(Clone, Debug)]
pub struct VuvResult {
    /// Band count `K̂` per Eq. 34.
    pub k_hat: u8,
    /// Per-band decisions `v̂_k ∈ {0, 1}`, 1-indexed in `vuv[1..=k_hat]`.
    /// Slots past `k_hat` are zero.
    pub vuv: [u8; (K_HAT_MAX + 1) as usize],
    /// Values of `D_k` per band (useful for debugging / DVSI diffing).
    /// 1-indexed; slots past `k_hat` are zero.
    pub d_k: [f64; (K_HAT_MAX + 1) as usize],
    /// Per-band Θ_ξ thresholds (Eq. 37). 1-indexed; slots past `k_hat`
    /// are zero. `D_k < Θ_ξ(k)` → voiced.
    pub theta_k: [f64; (K_HAT_MAX + 1) as usize],
    /// Frame energy ratio `M(ξ)` (Eq. 42).
    pub m_xi: f64,
    /// Total spectral energy `ξ_0 = ξ_LF + ξ_HF` (Eq. 38–39).
    pub xi_0: f64,
    /// `ξ_max(0)` committed to state (same as `state.xi_max` after
    /// this call returns).
    pub xi_max_after: f64,
}

/// Band count `K̂` per Eq. 34.
#[inline]
pub fn band_count_for(l_hat: u8) -> u8 {
    if l_hat <= 36 {
        (l_hat + 2) / 3
    } else {
        K_HAT_MAX
    }
}

/// Voiced/unvoiced determination for one frame per Eq. 34–42.
///
/// Mutates `state`: `ξ_max` and `v̂_k(−1)` are advanced to the current
/// frame's values on return.
pub fn determine_vuv(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    refinement: &PitchRefinement,
    e_p_hat_i: f64,
    state: &mut VuvState,
) -> VuvResult {
    let l_hat = refinement.l_hat;
    let omega_hat = refinement.omega_hat;
    let k_hat = band_count_for(l_hat);

    // Eq. 38–40: one-sided energy statistics normalized by |W_R(0)|².
    let wr_dc = basis.w_r(0);
    let wr_dc_sq = wr_dc * wr_dc;
    let inv_wr_dc_sq = if wr_dc_sq > 0.0 { 1.0 / wr_dc_sq } else { 0.0 };
    let mut xi_lf = 0.0;
    for m in 0..=63 {
        xi_lf += sw[m].norm_sqr();
    }
    xi_lf *= inv_wr_dc_sq;
    let mut xi_hf = 0.0;
    for m in 64..=128 {
        xi_hf += sw[m].norm_sqr();
    }
    xi_hf *= inv_wr_dc_sq;
    let xi_0 = xi_lf + xi_hf;

    // Eq. 41: ξ_max update (attack / slow decay / floor).
    let xi_max_new = if xi_0 > state.xi_max {
        0.5 * state.xi_max + 0.5 * xi_0
    } else {
        let decay = 0.99 * state.xi_max + 0.01 * xi_0;
        decay.max(XI_MAX_FLOOR)
    };

    // Eq. 42: M(ξ) — HF-dominant branch multiplies by √(ξ_LF / 5·ξ_HF).
    let ratio_den = 0.01 * xi_max_new + xi_0;
    let ratio = if ratio_den > 0.0 {
        (0.0025 * xi_max_new + xi_0) / ratio_den
    } else {
        0.0
    };
    let m_xi = if xi_lf >= 5.0 * xi_hf {
        ratio
    } else if xi_hf > 0.0 {
        ratio * (xi_lf / (5.0 * xi_hf)).sqrt()
    } else {
        // Degenerate: ξ_HF ≈ 0 AND ξ_LF < 5·ξ_HF implies ξ_LF ≈ 0 too;
        // entire frame is silent. M(ξ) → 0.25 via the first-branch
        // limit (ratio → 0.25 when ξ_0 = 0 and ξ_max = floor). Route
        // through branch 1 instead.
        ratio
    };

    // Per-band D_k (Eq. 35/36) and decision (Eq. 37 + strict-less-than).
    let mut vuv = [0u8; (K_HAT_MAX + 1) as usize];
    let mut d_k = [0f64; (K_HAT_MAX + 1) as usize];
    let mut theta_k = [0f64; (K_HAT_MAX + 1) as usize];
    for k in 1..=k_hat {
        let l_lo = 3 * u32::from(k) - 2;
        let l_hi = if k < k_hat {
            3 * u32::from(k)
        } else {
            u32::from(l_hat)
        };

        // Numerator Σ|S_w(m) − S_w(m, ω̂_0)|² and denominator Σ|S_w(m)|²
        // accumulated over the contiguous harmonics l_lo..=l_hi.
        let mut num = 0.0;
        let mut den = 0.0;
        for l in l_lo..=l_hi {
            let a_l = basis.harmonic_amplitude(sw, l, omega_hat);
            let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(l, omega_hat);
            for m in m_lo..m_hi {
                let synth = basis.synthetic_bin(m, l, omega_hat, a_l);
                let observed = sw[packed_index(m)];
                let dre = observed.re - synth.re;
                let dim = observed.im - synth.im;
                num += dre * dre + dim * dim;
                den += observed.norm_sqr();
            }
        }
        let d = if den > 0.0 { num / den } else { 1.0 };
        d_k[k as usize] = d;

        // Eq. 37: Θ_ξ(k, ω̂_0).
        let theta = if e_p_hat_i > E_P_POOR_MATCH_THRESHOLD && k >= 2 {
            0.0
        } else {
            let base = if state.vuv_prev[k as usize] == 1 {
                THETA_BASE_VOICED_HISTORY
            } else {
                THETA_BASE_UNVOICED_HISTORY
            };
            let modulation = 1.0 - THETA_PITCH_BAND_COEF * f64::from(k - 1) * omega_hat;
            (base * modulation * m_xi).max(0.0)
        };

        theta_k[k as usize] = theta;
        vuv[k as usize] = if d < theta { 1 } else { 0 };
    }

    // Commit state for next frame.
    state.xi_max = xi_max_new;
    state.vuv_prev.fill(0);
    for k in 1..=k_hat {
        state.vuv_prev[k as usize] = vuv[k as usize];
    }
    state.k_prev = k_hat;

    VuvResult {
        k_hat,
        vuv,
        d_k,
        theta_k,
        m_xi,
        xi_0,
        xi_max_after: xi_max_new,
    }
}
