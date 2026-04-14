//! MBE baseline codec — TIA-102.BABA-A §1.10–§1.13 (synthesis pipeline)
//! and §1.8 (analysis is shared with the upstream pipeline in
//! [`crate::imbe_frames::dequantize`]).
//!
//! This module hosts the synthesizer that turns [`MbeParams`] into
//! 8 kHz 16-bit PCM. The synthesis algorithm follows the 1993 P25
//! vocoder specification:
//!
//! - **§1.10 enhancement** — `M̃_l → M̄_l` via the W_l psycho-acoustic
//!   weighting.
//! - **§1.11 smoothing** — frame repeat / mute / V/UV smoothing.
//! - **§1.12.1 unvoiced** — LCG noise → DFT → per-band spectral shaping
//!   → IDFT → weighted overlap-add.
//! - **§1.12.2 voiced** — sinusoidal overlap-add with phase tracking
//!   across the four V/UV transition cases.
//!
//! Per TIA-102.BABG this baseline generation explicitly cannot pass the
//! enhanced-vocoder performance tests. It is implemented here for spec
//! completeness and as the reference point against which AMBE/+/+2
//! generations' improvements are measured.
//!
//! [`MbeParams`]: crate::mbe_params::MbeParams

use crate::mbe_params::L_MAX;

include!(concat!(env!("OUT_DIR"), "/annex_i_synth_window.rs"));

/// Lookup `wS(n)` for `n ∈ [-105, 105]`. Returns 0.0 outside that range
/// (matching the spec — `wS` is treated as zero outside its support).
#[inline]
pub fn synth_window(n: i32) -> f32 {
    if !(-105..=105).contains(&n) {
        return 0.0;
    }
    IMBE_SYNTH_WINDOW[(n + 105) as usize]
}

// ---------------------------------------------------------------------------
// §1.10 — Spectral amplitude enhancement (Eq. 105–111)
// ---------------------------------------------------------------------------

/// Initial value for the local-energy state `S_E` per BABA-A §10
/// Annex A. Used when the synthesizer is cold-started.
pub const INIT_S_E: f64 = 75000.0;

/// Lower bound on `S_E` per Eq. 111. Floors out the recurrence so very
/// quiet frames don't drive the predictor toward zero.
pub const S_E_FLOOR: f64 = 10000.0;

/// Apply BABA-A §1.10 spectral amplitude enhancement to the
/// reconstructed magnitudes `M̃_l`, producing the synthesizer-side
/// magnitudes `M̄_l` and the updated local-energy state `S_E(0)`.
///
/// Per §1.10:
/// 1. Compute energy moments `R_M0`, `R_M1` (Eq. 105–106).
/// 2. Per-harmonic weight `W_l` (Eq. 107) with `8·l ≤ L̃` bypass and
///    `[0.5, 1.2]` clamp on `W_l` (Eq. 108).
/// 3. Energy-preserving rescale by `γ = sqrt(R_M0 / Σ M̄_l²)` (Eq. 109–110).
/// 4. Update `S_E(0) = max(0.95·S_E(−1) + 0.05·R_M0, 10000)` (Eq. 111).
///
/// All internal arithmetic is `f64` per the spec's reference C
/// implementation. Inputs and outputs at the `MbeParams` boundary
/// stay in `f32`.
///
/// `m_tilde` must have length `L̃ ≥ 1`; the returned array's entries
/// `[0..L̃]` are populated, the rest are zero.
///
/// **Edge cases.** If `R_M0` is zero (silence frame) the enhancement
/// reduces to a passthrough — `W_l` is undefined (denominator = 0)
/// and `γ` is also undefined; we return `M̄_l = M̃_l` and only update
/// `S_E`. Similarly for `R_M0 == R_M1` (denominator zero by another
/// path).
pub fn enhance_spectral_amplitudes(
    m_tilde: &[f32],
    omega_0: f32,
    s_e_prev: f64,
) -> ([f32; L_MAX as usize], f64) {
    let l = m_tilde.len();
    debug_assert!(l > 0 && l <= L_MAX as usize);
    let omega_0 = f64::from(omega_0);

    // Eq. 105–106: energy moments.
    let mut r_m0 = 0.0f64;
    let mut r_m1 = 0.0f64;
    for (i, &m) in m_tilde.iter().enumerate() {
        let l_one = (i + 1) as f64;
        let m2 = f64::from(m) * f64::from(m);
        r_m0 += m2;
        r_m1 += m2 * (omega_0 * l_one).cos();
    }

    let mut m_bar_f64 = [0f64; L_MAX as usize];
    let denom = omega_0 * r_m0 * (r_m0 - r_m1);

    if r_m0 <= 0.0 || denom.abs() < 1e-30 {
        // Silence-frame degenerate path — bypass enhancement.
        for (i, &m) in m_tilde.iter().enumerate() {
            m_bar_f64[i] = f64::from(m);
        }
    } else {
        // Eq. 107–108: per-harmonic weight with bypass + clamp.
        for (i, &m) in m_tilde.iter().enumerate() {
            let l_one = (i + 1) as f64;
            let bar = if 8 * (i + 1) <= l {
                f64::from(m) // first branch — bypass for the lowest ⌊L̃/8⌋
            } else {
                let num = r_m0 * r_m0 + r_m1 * r_m1
                    - 2.0 * r_m0 * r_m1 * (omega_0 * l_one).cos();
                if num <= 0.0 {
                    f64::from(m) // numerator pathology → passthrough
                } else {
                    let w_l = f64::from(m).sqrt() * (0.96 * num / denom).powf(0.25);
                    if w_l > 1.2 {
                        1.2 * f64::from(m)
                    } else if w_l < 0.5 {
                        0.5 * f64::from(m)
                    } else {
                        w_l * f64::from(m)
                    }
                }
            };
            m_bar_f64[i] = bar;
        }

        // Eq. 109–110: energy-preserving rescale.
        let mut sum_sq = 0.0f64;
        for &v in m_bar_f64.iter().take(l) {
            sum_sq += v * v;
        }
        if sum_sq > 1e-30 {
            let gamma = (r_m0 / sum_sq).sqrt();
            for v in m_bar_f64.iter_mut().take(l) {
                *v *= gamma;
            }
        }
    }

    // Eq. 111: S_E recurrence with floor.
    let s_e = (0.95 * s_e_prev + 0.05 * r_m0).max(S_E_FLOOR);

    let mut m_bar = [0f32; L_MAX as usize];
    for (i, &v) in m_bar_f64.iter().take(l).enumerate() {
        m_bar[i] = v as f32;
    }
    (m_bar, s_e)
}

// ---------------------------------------------------------------------------
// §1.11 — Adaptive smoothing, frame repeat / mute, V/UV smoothing
// ---------------------------------------------------------------------------

/// Initial value for the amplitude-smoothing state `τ_M` per BABA-A
/// §10 Annex A.
pub const INIT_TAU_M: f64 = 20480.0;

/// Smoothed-error-rate threshold above which the frame is muted
/// (Eq. mute condition in §1.11.2).
pub const MUTE_EPSILON_R_THRESHOLD: f64 = 0.0875;

/// Per-frame error context derived from FEC decoding (§1.5) plus the
/// pitch-validity check (§1.3.1). Drives the §1.11 smoothing
/// decisions.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameErrorContext {
    /// Bit errors corrected in û₀ (Golay).
    pub epsilon_0: u8,
    /// Bit errors corrected in û₄ specifically (used in V_M branch 2).
    pub epsilon_4: u8,
    /// Total bit errors across all 7 FEC-protected vectors `û₀..û₆`.
    pub epsilon_t: u8,
    /// Set when the decoded pitch index `b̂₀` is in the reserved range
    /// `[208, 255]` (or otherwise invalid) — forces a frame repeat
    /// regardless of the error counts.
    pub bad_pitch: bool,
}

/// Action the synthesizer should take for the current frame, per
/// BABA-A §1.11.1 / §1.11.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDisposition {
    /// Synthesize from the current frame's parameters.
    Use,
    /// Substitute the previous frame's parameters (Eq. 99–104) and
    /// then synthesize.
    Repeat,
    /// Bypass the synthesizer and emit silence / low-amplitude noise.
    Mute,
}

/// Update the smoothed error rate `ε_R(0) = 0.95·ε_R(−1) + 0.05·(ε_T/144)`
/// and decide the frame disposition.
///
/// Per §1.11.1 the repeat trigger is:
/// `b̂₀ invalid` **OR** (`ε₀ ≥ 2` AND `ε_T ≥ 10 + 40·ε_R(0)`).
///
/// Per §1.11.2 mute supersedes repeat when `ε_R(0) > 0.0875`.
pub fn frame_disposition(
    err: &FrameErrorContext,
    epsilon_r_prev: f64,
) -> (FrameDisposition, f64) {
    let epsilon_r =
        0.95 * epsilon_r_prev + 0.05 * (f64::from(err.epsilon_t) / 144.0);
    let disp = if epsilon_r > MUTE_EPSILON_R_THRESHOLD {
        FrameDisposition::Mute
    } else if err.bad_pitch
        || (err.epsilon_0 >= 2
            && f64::from(err.epsilon_t) >= 10.0 + 40.0 * epsilon_r)
    {
        FrameDisposition::Repeat
    } else {
        FrameDisposition::Use
    };
    (disp, epsilon_r)
}

/// Output of the §1.11.3 V/UV-and-amplitude smoothing step.
#[derive(Clone, Debug)]
pub struct SmoothedFrame {
    /// Smoothed per-harmonic spectral amplitudes (`M̄_l` after γ_M scaling).
    /// Entries `[0..L]` are populated.
    pub m_bar: [f32; L_MAX as usize],
    /// Smoothed per-harmonic V/UV decisions (`v̄_l`). Entries `[0..L]`
    /// are populated; higher indices are `false`.
    pub v_bar: [bool; L_MAX as usize],
    /// Updated amplitude-smoothing state `τ_M(0)` for the next frame.
    pub tau_m: f64,
}

/// Apply the §1.11.3 V/UV-and-amplitude smoothing pipeline.
///
/// Inputs:
/// - `m_bar`: enhanced spectral amplitudes from §1.10 (`M̄_l`).
/// - `v_tilde`: per-harmonic V/UV from the §1.3.2 expansion (`ṽ_l`).
/// - `s_e`: current-frame `S_E(0)` from §1.10.
/// - `epsilon_r`: smoothed error rate `ε_R(0)` from
///   [`frame_disposition`].
/// - `epsilon_t`, `epsilon_4`: per-frame error counts (raw, not smoothed).
/// - `tau_m_prev`: previous frame's `τ_M(−1)`.
///
/// Returns the smoothed `M̄_l`, the smoothed per-harmonic V/UV `v̄_l`,
/// and the updated `τ_M(0)`.
pub fn apply_smoothing(
    m_bar: &[f32],
    v_tilde: &[bool],
    s_e: f64,
    epsilon_r: f64,
    epsilon_t: u8,
    epsilon_4: u8,
    tau_m_prev: f64,
) -> SmoothedFrame {
    let l = m_bar.len();
    debug_assert_eq!(v_tilde.len(), l);

    // Eq. 112: V_M threshold (3-branch).
    let v_m: f64 = if epsilon_r <= 0.005 && epsilon_t <= 4 {
        f64::INFINITY
    } else if epsilon_r <= 0.0125 && epsilon_4 == 0 {
        45.255 * s_e.powf(0.375) * (-277.26 * epsilon_r).exp()
    } else {
        1.414 * s_e.powf(0.375)
    };

    // Eq. 113: per-harmonic V/UV override (force voiced if M̄_l > V_M).
    let mut v_bar = [false; L_MAX as usize];
    for (i, &m) in m_bar.iter().enumerate() {
        v_bar[i] = if f64::from(m) > v_m { true } else { v_tilde[i] };
    }

    // Eq. 114: total amplitude.
    let a_m: f64 = m_bar.iter().take(l).map(|&v| f64::from(v)).sum();

    // Eq. 115: τ_M recurrence.
    let tau_m = if epsilon_r <= 0.005 && epsilon_t <= 6 {
        20480.0
    } else {
        6000.0 - 300.0 * f64::from(epsilon_t) + tau_m_prev
    };

    // Eq. 116: γ_M scaling.
    let gamma_m: f64 = if tau_m > a_m { 1.0 } else if a_m > 0.0 { tau_m / a_m } else { 1.0 };

    let mut m_bar_out = [0f32; L_MAX as usize];
    for (i, &m) in m_bar.iter().enumerate() {
        m_bar_out[i] = ((f64::from(m)) * gamma_m) as f32;
    }

    SmoothedFrame { m_bar: m_bar_out, v_bar, tau_m }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_window_endpoints_are_zero() {
        // Per Annex I header, wS(±105) = 0.
        assert_eq!(synth_window(-105), 0.0);
        assert_eq!(synth_window(105), 0.0);
    }

    #[test]
    fn synth_window_symmetric() {
        for n in 1..=105 {
            assert_eq!(
                synth_window(-n),
                synth_window(n),
                "wS(-{n}) != wS({n})"
            );
        }
    }

    #[test]
    fn synth_window_peaks_near_zero() {
        // The window is a tapered shape that peaks near n=0. Verify
        // the centre is the maximum (or among the max set).
        let centre = synth_window(0);
        for n in -105..=105 {
            assert!(
                synth_window(n) <= centre + 1e-6,
                "wS({n}) = {} exceeds wS(0) = {centre}",
                synth_window(n)
            );
        }
    }

    #[test]
    fn synth_window_out_of_range_is_zero() {
        // Spec treats wS outside [-105, 105] as zero (used in the OLA
        // formula in §1.12).
        assert_eq!(synth_window(-106), 0.0);
        assert_eq!(synth_window(106), 0.0);
        assert_eq!(synth_window(1000), 0.0);
        assert_eq!(synth_window(-1000), 0.0);
    }

    #[test]
    fn synth_window_table_length_matches_constant() {
        assert_eq!(IMBE_SYNTH_WINDOW.len(), SYNTH_WINDOW_LEN);
        assert_eq!(SYNTH_WINDOW_LEN, 211);
    }

    // ---- §1.10 enhancement -----------------------------------------------

    fn sum_sq(arr: &[f32]) -> f64 {
        arr.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    }

    #[test]
    fn enhance_silence_passes_through_and_floors_s_e() {
        // All-zero M̃_l → R_M0 = 0 → silence-frame degenerate path.
        let m_tilde = vec![0.0f32; 30];
        let (m_bar, s_e) = enhance_spectral_amplitudes(&m_tilde, 0.1, INIT_S_E);
        for i in 0..30 {
            assert_eq!(m_bar[i], 0.0, "M̄_{i} = {}", m_bar[i]);
        }
        // S_E recurrence with R_M0 = 0: 0.95·75000 = 71250 (above floor).
        assert!((s_e - 71250.0).abs() < 1e-6, "S_E = {s_e}");
    }

    #[test]
    fn enhance_preserves_energy_within_a_few_percent() {
        // The Eq. 109–110 rescale is *exact* energy-preserving in
        // double precision before the f64 → f32 cast; we just check
        // the round-tripped energy matches within a small tolerance.
        let l = 24usize;
        let m_tilde: Vec<f32> = (1..=l).map(|i| (i as f32 * 0.3).sin().abs() + 0.1).collect();
        let omega_0 = std::f32::consts::PI / 12.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        let energy_in = sum_sq(&m_tilde);
        let energy_out = sum_sq(&m_bar[..l]);
        let rel_err = (energy_out - energy_in).abs() / energy_in;
        assert!(rel_err < 1e-4, "energy drift {rel_err} (in={energy_in}, out={energy_out})");
    }

    #[test]
    fn enhance_s_e_floor_engages() {
        // After many quiet frames, S_E should converge to the floor.
        let m_tilde = vec![0.0f32; 30];
        let mut s_e = INIT_S_E;
        for _ in 0..200 {
            let (_, new_s_e) = enhance_spectral_amplitudes(&m_tilde, 0.1, s_e);
            s_e = new_s_e;
        }
        assert!((s_e - S_E_FLOOR).abs() < 1e-6, "S_E = {s_e}");
    }

    #[test]
    fn enhance_s_e_recurrence_responds_to_input_energy() {
        // Pumping in higher-energy frames should drive S_E up from
        // its initial value (with the 0.05 mixing factor).
        let m_tilde: Vec<f32> = (1..=20).map(|_| 100.0).collect(); // R_M0 ≈ 200000
        let (_, s_e1) = enhance_spectral_amplitudes(&m_tilde, 0.1, INIT_S_E);
        // S_E = 0.95·75000 + 0.05·200000 = 71250 + 10000 = 81250
        assert!((s_e1 - 81250.0).abs() < 1e-2, "S_E = {s_e1}");
    }

    #[test]
    fn enhance_low_harmonic_bypass_holds_per_section_1_10() {
        // For l with 8·l ≤ L̃, the W_l weighting is bypassed (Eq. 108
        // first branch). The bypass gives M̄_l = M̃_l *before* the
        // global γ rescale. Verify the per-harmonic ratio M̄_l / M̃_l
        // is identical across all bypassed l (i.e., they all get the
        // same γ scaling and no per-l W weighting).
        let l = 56usize; // many harmonics so 8·l ≤ L̃ for several values
        let m_tilde: Vec<f32> = (1..=l).map(|i| 1.0 + (i as f32 * 0.05).sin()).collect();
        let omega_0 = std::f32::consts::PI / 28.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        // Bypassed harmonics: l = 1..=⌊L/8⌋ = 1..=7. They all share
        // the same final γ multiplier.
        let r0 = m_bar[0] / m_tilde[0];
        for i in 1..7 {
            let r = m_bar[i] / m_tilde[i];
            assert!(
                (r - r0).abs() < 1e-4,
                "bypassed harmonic l={}: ratio {r} vs {r0}",
                i + 1
            );
        }
    }

    // ---- §1.11 smoothing / disposition -----------------------------------

    fn err(epsilon_0: u8, epsilon_t: u8, epsilon_4: u8) -> FrameErrorContext {
        FrameErrorContext { epsilon_0, epsilon_4, epsilon_t, bad_pitch: false }
    }

    #[test]
    fn frame_disposition_clean_frame_uses() {
        let (d, er) = frame_disposition(&err(0, 0, 0), 0.0);
        assert_eq!(d, FrameDisposition::Use);
        assert_eq!(er, 0.0);
    }

    #[test]
    fn frame_disposition_bad_pitch_repeats() {
        let mut e = err(0, 0, 0);
        e.bad_pitch = true;
        let (d, _) = frame_disposition(&e, 0.0);
        assert_eq!(d, FrameDisposition::Repeat);
    }

    #[test]
    fn frame_disposition_joint_error_threshold_repeats() {
        // ε₀ ≥ 2 AND ε_T ≥ 10 + 40·ε_R(0). With ε_R(prev) = 0 and ε_T = 12,
        // ε_R(0) = 0.05·12/144 ≈ 0.00417. Threshold = 10 + 40·0.00417 ≈ 10.17.
        // ε_T = 12 ≥ 10.17 → repeat.
        let (d, _) = frame_disposition(&err(2, 12, 0), 0.0);
        assert_eq!(d, FrameDisposition::Repeat);
        // ε₀ = 1 below threshold → use.
        let (d, _) = frame_disposition(&err(1, 12, 0), 0.0);
        assert_eq!(d, FrameDisposition::Use);
    }

    #[test]
    fn frame_disposition_high_error_rate_mutes() {
        // ε_R prev high enough that the smoothed value crosses 0.0875.
        let (d, _) = frame_disposition(&err(0, 50, 0), 0.5);
        assert_eq!(d, FrameDisposition::Mute);
    }

    #[test]
    fn smoothing_low_error_uses_infinite_v_m() {
        // Branch 1: ε_R ≤ 0.005 AND ε_T ≤ 4 → V_M = ∞. Even very tall
        // M̄_l should not flip an unvoiced harmonic to voiced.
        let m_bar = vec![1e6f32; 9];
        let v_tilde = vec![false; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, 75000.0, 0.001, 2, 0, INIT_TAU_M);
        for i in 0..9 {
            assert!(!s.v_bar[i], "v̄_{i} flipped to voiced under V_M = ∞");
        }
    }

    #[test]
    fn smoothing_v_uv_override_when_amplitude_exceeds_threshold() {
        // Force branch 3 (V_M = 1.414 · S_E^0.375). With S_E small, V_M
        // is small; large M̄ should flip to voiced.
        let s_e = 100.0; // V_M = 1.414 · 100^0.375 ≈ 1.414 · 5.84 ≈ 8.27
        let m_bar = vec![1.0f32, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0];
        let v_tilde = vec![false; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, s_e, 0.05, 10, 1, INIT_TAU_M);
        for i in 0..9 {
            let expected = m_bar[i] > 8.27;
            assert_eq!(s.v_bar[i], expected, "v̄_{i}");
        }
    }

    #[test]
    fn smoothing_amplitude_low_error_resets_tau_m() {
        // Branch 1 of Eq. 115: ε_R ≤ 0.005 AND ε_T ≤ 6 → τ_M = 20480.
        let m_bar = vec![10.0f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.001, 3, 0, 5000.0);
        assert_eq!(s.tau_m, 20480.0);
    }

    #[test]
    fn smoothing_amplitude_recurrence_under_errors() {
        // Branch 2: τ_M(0) = 6000 − 300·ε_T + τ_M(−1)
        let m_bar = vec![10.0f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.05, 8, 1, 10000.0);
        // expected τ_M = 6000 − 300·8 + 10000 = 13600
        assert!((s.tau_m - 13600.0).abs() < 1e-6);
    }

    #[test]
    fn smoothing_gamma_m_is_one_when_tau_m_exceeds_amplitude_total() {
        // A_M = Σ M̄_l. Tiny amplitudes → A_M < τ_M → γ_M = 1 (no scaling).
        let m_bar = vec![0.01f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.001, 0, 0, INIT_TAU_M);
        for i in 0..9 {
            assert!((s.m_bar[i] - 0.01).abs() < 1e-6);
        }
    }

    #[test]
    fn smoothing_gamma_m_clamps_loud_frames() {
        // Loud amplitudes → A_M > τ_M → γ_M = τ_M / A_M < 1, all M̄_l scaled
        // by the same γ_M.
        let m_bar = vec![10000.0f32; 9];
        let v_tilde = vec![true; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, 1000.0, 0.001, 0, 0, INIT_TAU_M);
        // A_M = 90000, τ_M = 20480, γ_M ≈ 0.2276.
        let expected_gamma = 20480.0 / 90000.0;
        for i in 0..9 {
            let ratio = s.m_bar[i] / 10000.0;
            assert!(
                (ratio - expected_gamma as f32).abs() < 1e-3,
                "γ_M ratio at l={}: {ratio} vs {expected_gamma}",
                i + 1
            );
        }
    }

    #[test]
    fn enhance_clamps_w_l_within_brackets() {
        // For an extreme spectral shape we expect some W_l values to
        // hit the clamp (1.2 high, 0.5 low). Non-bypassed M̄_l / M̃_l
        // ratios must lie in [0.5·γ, 1.2·γ] for some shared γ.
        let l = 12usize; // small enough that 8·l > L̃ for all l ≥ 2
        let m_tilde: Vec<f32> = vec![1e-3, 10.0, 0.01, 5.0, 0.05, 3.0, 0.1, 2.0, 0.2, 1.5, 0.3, 1.0];
        let omega_0 = std::f32::consts::PI / 6.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        // All non-bypassed (l ≥ 2 since L=12 means only l=1 is bypassed)
        // ratios M̄/M̃ before γ would be in [0.5, 1.2]. After γ scaling
        // they all share the same γ. So the ratio of any two should be
        // bounded by (1.2/0.5) = 2.4.
        let mut ratios: Vec<f32> = (1..l).map(|i| m_bar[i] / m_tilde[i]).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let span = ratios[ratios.len() - 1] / ratios[0];
        assert!(span <= 2.4 + 1e-4, "ratio span {span} exceeds 2.4");
    }
}
