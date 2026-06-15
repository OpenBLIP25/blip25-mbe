//! §0.5 — Spectral Amplitude Estimation.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.5 (BABA-A §5.3
//! Eq. 43 voiced / Eq. 44 unvoiced, Figure 13). Produces M̂_l for
//! 1 ≤ l ≤ L̂ — magnitude-only, no complex decomposition. Runs AFTER
//! §0.7 V/UV determination (per Figure 5 on PDF p. 9): each harmonic
//! inherits its band's v̂_k bit, which picks between Eq. 43 and Eq. 44.

use super::basis::floor_i32;
use super::{
    packed_index, Complex64, HarmonicBasis, PitchRefinement, VuvResult, DFT_SIZE, L_HAT_MAX,
    WINDOW_DFT_SIZE,
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
    frac_edges: bool,
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
        // Integration range + per-bin coverage weight. `frac_edges`
        // (BLIP25_BIN_EDGE=frac) weights boundary bins by their overlap
        // with the continuous band `[a_l, b_l)` (each DFT bin m owns
        // `[m−0.5, m+0.5)`), recovering the band energy the spec's
        // ceil-to-ceil integer rule loses unevenly across `l` (the
        // L-dependent gain bias). `frac_edges = false` is byte-identical
        // to the spec: weight 1.0 over `[m_lo, m_hi)`.
        let (a_f, b_f) = HarmonicBasis::bin_edges_frac(l, omega_hat);
        let (m_first, m_last) = if frac_edges {
            ((a_f - 0.5).floor() as i32, (b_f + 0.5).ceil() as i32)
        } else {
            (m_lo, m_hi)
        };
        if m_last <= m_first {
            continue;
        }
        let bin_w = |m: i32| -> f64 {
            if !frac_edges {
                return 1.0;
            }
            let lo = (f64::from(m) - 0.5).max(a_f);
            let hi = (f64::from(m) + 0.5).min(b_f);
            (hi - lo).clamp(0.0, 1.0)
        };

        // Numerator Σ w(m)·|S_w(m)|² — shared between Eq. 43 and Eq. 44.
        let mut s_energy = 0.0;
        for m in m_first..m_last {
            s_energy += bin_w(m) * sw[packed_index(m)].norm_sqr();
        }

        let is_voiced = k >= 1 && vuv.vuv[k as usize] == 1;
        let m_l = if is_voiced {
            // Eq. 43: M̂_l = sqrt(Σ w(m)|S_w(m)|² / Σ w(m)|W_R(k_m)|²).
            let l_offset = bin_scale_16384 * f64::from(l) * omega_hat;
            let mut window_energy = 0.0;
            for m in m_first..m_last {
                let k_bin = floor_i32(64.0 * f64::from(m) - l_offset + 0.5);
                let wr = basis.w_r(k_bin);
                window_energy += bin_w(m) * wr * wr;
            }
            if window_energy > 0.0 {
                (s_energy / window_energy).sqrt()
            } else {
                0.0
            }
        } else {
            // Eq. 44: M̂_l = (1/Σw_R) · sqrt(Σ w(m)|S_w(m)|² / bins),
            // where `bins` is the coverage sum (= m_hi − m_lo in the
            // spec/integer path).
            let bins: f64 = if frac_edges {
                (m_first..m_last).map(bin_w).sum()
            } else {
                (m_hi - m_lo) as f64
            };
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

#[cfg(test)]
mod tests {
    use super::super::{
        band_count_for, determine_vuv, refine_pitch, signal_spectrum, VuvState, K_HAT_MAX,
        W_R_HALF, XI_MAX_FLOOR,
    };
    use super::*;

    fn broadband_periodic(period: f64, max_h: u32) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let omega = 2.0 * core::f64::consts::PI / period;
        for (idx, slot) in signal.iter_mut().enumerate() {
            let n = idx as i32 - W_R_HALF;
            let nf = f64::from(n);
            let mut s = 0.0;
            for h in 1..=max_h {
                s += (f64::from(h) * omega * nf).cos() / f64::from(h);
            }
            *slot = s;
        }
        signal
    }

    /// A helper returning a builder-assembled `VuvResult` with all
    /// bands voiced (for §0.5 tests that bypass §0.7).
    fn vuv_all_voiced(k_hat: u8) -> VuvResult {
        let mut vuv = [0u8; (K_HAT_MAX + 1) as usize];
        for k in 1..=k_hat {
            vuv[k as usize] = 1;
        }
        VuvResult {
            k_hat,
            vuv,
            d_k: [0.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    fn vuv_all_unvoiced(k_hat: u8) -> VuvResult {
        VuvResult {
            k_hat,
            vuv: [0; (K_HAT_MAX + 1) as usize],
            d_k: [1.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    #[test]
    fn band_for_harmonic_follows_ceil_l_div_3() {
        // K̂ = 12 (high-harmonic case): standard bands are l∈{1,2,3}→k=1, {4,5,6}→k=2, etc.
        assert_eq!(band_for_harmonic(1, 12), 1);
        assert_eq!(band_for_harmonic(2, 12), 1);
        assert_eq!(band_for_harmonic(3, 12), 1);
        assert_eq!(band_for_harmonic(4, 12), 2);
        assert_eq!(band_for_harmonic(33, 12), 11);
        assert_eq!(band_for_harmonic(34, 12), 12);
        assert_eq!(band_for_harmonic(56, 12), 12); // capped at K̂
                                                   // Smaller K̂: everything above 3(K̂−1) goes to the highest band.
        assert_eq!(band_for_harmonic(9, 3), 3);
        assert_eq!(band_for_harmonic(10, 3), 3);
        assert_eq!(band_for_harmonic(5, 3), 2);
        // Edge: l=0 or K̂=0 returns 0.
        assert_eq!(band_for_harmonic(0, 12), 0);
        assert_eq!(band_for_harmonic(1, 0), 0);
    }

    /// Spectral amplitudes on silent input are zero (or numerically
    /// negligible) for every harmonic.
    #[test]
    fn estimate_amplitudes_zero_on_silence() {
        let basis = HarmonicBasis::new();
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        let refinement = PitchRefinement {
            omega_hat: 2.0 * core::f64::consts::PI / 50.0,
            l_hat: 23,
            b0: 41,
            e_r: 0.0,
        };
        let vuv = vuv_all_voiced(band_count_for(refinement.l_hat));
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv, false);
        for l in 1..=refinement.l_hat {
            assert!(
                amps[l as usize].abs() < 1e-12,
                "M̂_{l} = {} on silence",
                amps[l as usize]
            );
        }
    }

    /// Eq. 43 recovers the amplitude of a pure cosine within its
    /// fundamental harmonic. For `s(n) = A·cos(ω·n)` with `ω = 2π/P`,
    /// `|S_w(m)|²` on the l=1 support has energy `(A/2)² · |W_R(k)|²`
    /// for each m (the main-lobe term), and Eq. 43's ratio gives
    /// `sqrt(Σ (A/2)² · |W_R|²  / Σ |W_R|²) = A/2`.
    #[test]
    fn estimate_amplitudes_voiced_recovers_half_cosine_amplitude() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let period = 50.0;
        let amplitude = 1.0;
        let omega = 2.0 * PI / period;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = amplitude * (omega * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, period);
        let vuv = vuv_all_voiced(band_count_for(refinement.l_hat));
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv, false);
        // Expected: M̂_1 ≈ A/2 = 0.5. Allow 10% slack for quarter-sample
        // refinement offset and window sidelobe spillover.
        let m1 = amps[1];
        assert!(
            (m1 - 0.5).abs() < 0.05,
            "M̂_1 = {m1}, expected ≈ 0.5 (A/2 for pure cosine)"
        );
        // Higher harmonics should be near zero (no energy there).
        for l in 3..=refinement.l_hat {
            assert!(
                amps[l as usize].abs() < 0.05,
                "M̂_{l} = {} (spurious on pure-tone input)",
                amps[l as usize]
            );
        }
    }

    /// Eq. 43 and Eq. 44 produce different values for the same input.
    /// This validates the code path differentiation — if V/UV were
    /// ignored, all amplitudes would be identical between the two.
    #[test]
    fn estimate_amplitudes_voiced_and_unvoiced_differ() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let k_hat = band_count_for(refinement.l_hat);
        let v = vuv_all_voiced(k_hat);
        let u = vuv_all_unvoiced(k_hat);
        let amps_v = estimate_spectral_amplitudes(&sw, &basis, &refinement, &v, false);
        let amps_u = estimate_spectral_amplitudes(&sw, &basis, &refinement, &u, false);
        let mut any_diff = false;
        for l in 1..=refinement.l_hat {
            if (amps_v[l as usize] - amps_u[l as usize]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Eq. 43 and Eq. 44 produced identical amplitudes");
    }

    /// Amplitudes are non-negative (both Eq. 43 and Eq. 44 are `sqrt` outputs).
    #[test]
    fn estimate_amplitudes_are_nonnegative_and_finite() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let vuv = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv, false);
        for l in 1..=refinement.l_hat {
            let a = amps[l as usize];
            assert!(a.is_finite() && a >= 0.0, "M̂_{l} = {a}");
        }
        // Slots past L̂ are zero.
        for l in (refinement.l_hat + 1)..=L_HAT_MAX {
            assert_eq!(amps[l as usize], 0.0);
        }
        // DC is always zero (not written).
        assert_eq!(amps[0], 0.0);
    }

    /// Highest-band span is `L̂ − 3K̂ + 3` harmonics, all sharing the
    /// same band index `K̂`. For `L̂=23, K̂=8`, the top band covers
    /// `l ∈ {22, 23}` (2 harmonics). Verify band_for_harmonic agrees
    /// and that estimate_spectral_amplitudes uses the same V/UV bit
    /// across that span.
    #[test]
    fn estimate_amplitudes_highest_band_spans_multiple_harmonics_consistently() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let k_hat = band_count_for(refinement.l_hat);
        // Build a VuvResult where every band is voiced EXCEPT the
        // highest — which is unvoiced.
        let mut vuv_bits = [1u8; (K_HAT_MAX + 1) as usize];
        vuv_bits[0] = 0;
        vuv_bits[k_hat as usize] = 0;
        // Zero out above k_hat so state is clean.
        for k in (k_hat + 1)..=K_HAT_MAX {
            vuv_bits[k as usize] = 0;
        }
        let vuv = VuvResult {
            k_hat,
            vuv: vuv_bits,
            d_k: [0.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        };
        let amps_split = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv, false);
        // Redo with all voiced — highest-band amplitudes should differ.
        let amps_all_v =
            estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv_all_voiced(k_hat), false);
        // Harmonics in the highest band: l ∈ [3·(K̂−1)+1, L̂].
        let l_hi_lo = 3 * u32::from(k_hat - 1) + 1;
        for l in l_hi_lo..=u32::from(refinement.l_hat) {
            assert_eq!(
                band_for_harmonic(l, k_hat),
                k_hat,
                "harmonic {l} should map to highest band"
            );
            // Eq. 43 (voiced) vs Eq. 44 (unvoiced) should diverge.
            let diff = (amps_split[l as usize] - amps_all_v[l as usize]).abs();
            assert!(
                diff > 1e-9,
                "harmonic {l}: voiced/unvoiced produced same M̂_l (diff = {diff})"
            );
        }
    }
}
