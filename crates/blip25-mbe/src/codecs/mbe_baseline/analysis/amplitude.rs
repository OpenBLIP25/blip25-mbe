//! §0.5 — Spectral Amplitude Estimation.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.5 (BABA-A §5.3
//! Eq. 43 voiced / Eq. 44 unvoiced, Figure 13). Produces M̂_l for
//! 1 ≤ l ≤ L̂ — magnitude-only, no complex decomposition. Runs AFTER
//! §0.7 V/UV determination (per Figure 5 on PDF p. 9): each harmonic
//! inherits its band's v̂_k bit, which picks between Eq. 43 and Eq. 44.

use super::{
    Complex64, DFT_SIZE, HarmonicBasis, L_HAT_MAX, PitchRefinement, VuvResult,
    WINDOW_DFT_SIZE, floor_i32, packed_index,
};

/// Per-harmonic spectral amplitudes. Index 0 unused; amplitudes live
/// at indices `1..=L̂` (1-indexed per BABA-A notation). Slots past `L̂`
/// are zero.
pub type SpectralAmplitudes = [f64; L_HAT_MAX as usize + 1];

/// Map harmonic index `l` (1-indexed) to the containing V/UV band
/// `k = ⌈l / 3⌉` for `l ≤ 3(K̂−1)`; harmonics above that all fall into
/// the highest band `K̂`. Returns 0 for `l = 0` (reserved/unused).
#[inline]
pub fn band_for_harmonic(l: u32, k_hat: u8) -> u8 {
    if l == 0 || k_hat == 0 {
        return 0;
    }
    let k = ((l + 2) / 3) as u8;
    k.min(k_hat)
}

/// Estimate per-harmonic spectral amplitudes `M̂_l` per Eq. 43 / Eq. 44.
///
/// `sw` is the precomputed signal DFT (§0.2, once per frame). `basis`
/// supplies `W_R(k)` for the Eq. 43 denominator and `Σ w_R(n) = W_R(0)`
/// for the Eq. 44 external factor. `refinement` carries `ω̂_0` and `L̂`.
/// `vuv` holds the per-band bits from §0.7.
///
/// Returns a 1-indexed array; `amplitudes[l] = M̂_l` for `1 ≤ l ≤ L̂`,
/// zero elsewhere. `M̂_0` is intentionally not written (PDF §5.3:
/// "the D.C. spectral amplitude is ignored").
pub fn estimate_spectral_amplitudes(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    refinement: &PitchRefinement,
    vuv: &VuvResult,
) -> SpectralAmplitudes {
    let mut out = [0.0f64; L_HAT_MAX as usize + 1];
    let l_hat = refinement.l_hat;
    if l_hat == 0 {
        return out;
    }
    let omega_hat = refinement.omega_hat;
    let k_hat = vuv.k_hat;
    let bin_scale_16384 = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
    // Σ w_R(n) = W_R(0) per Eq. 30 at m = 0 — reuse the precomputed basis table.
    let wr_sum = basis.w_r(0);

    for l in 1..=u32::from(l_hat) {
        let k = band_for_harmonic(l, k_hat);
        let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(l, omega_hat);
        if m_hi <= m_lo {
            continue;
        }

        // Numerator Σ |S_w(m)|² — shared between Eq. 43 and Eq. 44.
        let mut s_energy = 0.0;
        for m in m_lo..m_hi {
            s_energy += sw[packed_index(m)].norm_sqr();
        }

        let is_voiced = k >= 1 && vuv.vuv[k as usize] == 1;
        let m_l = if is_voiced {
            // Eq. 43: M̂_l = sqrt(Σ|S_w(m)|² / Σ|W_R(k_m)|²).
            let l_offset = bin_scale_16384 * f64::from(l) * omega_hat;
            let mut window_energy = 0.0;
            for m in m_lo..m_hi {
                let k_bin = floor_i32(64.0 * f64::from(m) - l_offset + 0.5);
                let wr = basis.w_r(k_bin);
                window_energy += wr * wr;
            }
            if window_energy > 0.0 {
                (s_energy / window_energy).sqrt()
            } else {
                0.0
            }
        } else {
            // Eq. 44: M̂_l = (1/Σw_R) · sqrt(Σ|S_w(m)|² / bins).
            let bins = (m_hi - m_lo) as f64;
            if bins > 0.0 && wr_sum != 0.0 {
                (s_energy / bins).sqrt() / wr_sum
            } else {
                0.0
            }
        };
        out[l as usize] = m_l;
    }
    out
}
