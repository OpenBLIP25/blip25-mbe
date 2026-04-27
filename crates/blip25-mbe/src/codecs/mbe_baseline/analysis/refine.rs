//! §0.4 — Pitch refinement and pitch quantization.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.4 (BABA-A §5.1.5
//! Eq. 24–33, §6.1 Eq. 45). Takes P̂_I from §0.3 and refines to
//! quarter-sample resolution ω̂_0 by minimizing E_R(ω_0) over ten
//! candidates at offsets ±1/8, ±3/8, ±5/8, ±7/8, ±9/8 samples. Also
//! computes L̂ (Eq. 31) and the quantized pitch index b̂_0 (Eq. 45).

use super::{Complex64, DFT_SIZE, HarmonicBasis, packed_index};

/// Number of pitch-refinement candidates (BABA-A §5.1.5).
pub const N_REFINE_CANDIDATES: usize = 10;

/// Quarter-sample offsets applied to `P̂_I` to form the ten candidates.
/// `P̂_I` itself is excluded; smallest offset is `±1/8`.
pub const REFINE_OFFSETS: [f64; N_REFINE_CANDIDATES] = [
    -9.0 / 8.0,
    -7.0 / 8.0,
    -5.0 / 8.0,
    -3.0 / 8.0,
    -1.0 / 8.0,
    1.0 / 8.0,
    3.0 / 8.0,
    5.0 / 8.0,
    7.0 / 8.0,
    9.0 / 8.0,
];

/// Minimum harmonic count per Eq. 31 over the admissible ω̂_0 range.
pub const L_HAT_MIN: u8 = 9;

/// Maximum harmonic count per Eq. 31 over the admissible ω̂_0 range.
pub const L_HAT_MAX: u8 = 56;

/// Maximum quantized pitch index per Eq. 45. Values `[208, 255]` are
/// reserved per §6.1 prose.
pub const B0_MAX: u8 = 207;

/// Lower bound of the Eq. 24 residual sum (fixed at `m = 50`).
const E_R_LOWER_M: i32 = 50;

/// Output of [`refine_pitch`].
#[derive(Clone, Copy, Debug)]
pub struct PitchRefinement {
    /// Refined fundamental frequency `ω̂_0` (radians/sample).
    pub omega_hat: f64,
    /// Number of harmonics `L̂` per Eq. 31 (clamped to `[L_HAT_MIN, L_HAT_MAX]`).
    pub l_hat: u8,
    /// Quantized pitch index `b̂_0` per Eq. 45 (`[0, B0_MAX]`).
    pub b0: u8,
    /// Residual `E_R(ω̂_0)` at the winning candidate. Smaller = better fit.
    pub e_r: f64,
}

/// Number of harmonics `L̂` from `ω̂_0` per Eq. 31:
/// `L̂ = ⌊0.9254 · ⌊π/ω̂_0 + 0.25⌋⌋`. The inner floor is load-bearing
/// — it rounds π/ω̂_0 to the nearest integer (pitch period in samples)
/// before the `0.9254` scale. Omitting it biases `L̂` by +1 on ~40% of
/// admissible ω̂_0 (addendum §0.4.3, specs commit `d20f5ab`).
#[inline]
pub fn harmonic_count_for(omega_hat: f64) -> u8 {
    let inner = (core::f64::consts::PI / omega_hat + 0.25).floor();
    let raw = (0.9254 * inner).floor();
    raw.clamp(f64::from(L_HAT_MIN), f64::from(L_HAT_MAX)) as u8
}

/// Quantize `ω̂_0` to the 8-bit pitch index `b̂_0` per Eq. 45. Encoder
/// floor is `−39`, not `−39.5` (Eq. 46's `+39.5` is decoder-side).
///
/// Returns `None` when the computed index falls outside `[0, B0_MAX]`;
/// callers should dispatch to silence/erasure rather than clamp, per
/// addendum §0.4.5.
#[inline]
pub fn quantize_pitch_index(omega_hat: f64) -> Option<u8> {
    let raw = (4.0 * core::f64::consts::PI / omega_hat - 39.0).floor();
    if (0.0..=f64::from(B0_MAX)).contains(&raw) {
        Some(raw as u8)
    } else {
        None
    }
}

/// Upper limit of the Eq. 24 residual sum as a function of `ω_0`.
/// Simplified form: `⌊0.9254·128 − 64·ω_0/π⌋ = ⌊118.45 − 64·ω_0/π⌋`.
#[inline]
fn e_r_upper_m(omega0: f64) -> i32 {
    (0.9254 * 128.0 - 64.0 * omega0 / core::f64::consts::PI).floor() as i32
}

/// Compute `E_R(ω_0)` per Eq. 24 — the spectral residual between the
/// observed `S_w(m)` and the synthetic `S_w(m, ω_0)` built from the
/// harmonic projections `A_l(ω_0)` of Eq. 28.
pub(super) fn residual_e_r(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    omega0: f64,
) -> f64 {
    let m_max = e_r_upper_m(omega0);
    if m_max < E_R_LOWER_M {
        return 0.0;
    }
    let mut e_r = 0.0;
    // Iterate harmonics in ascending order. Supports are contiguous
    // and non-overlapping (Eq. 26/27: b_l = a_{l+1}). Stop once the
    // next harmonic starts above m_max.
    let mut l = 0u32;
    loop {
        let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(l, omega0);
        if m_lo > m_max {
            break;
        }
        // Early-exit harmonics that end before the residual's lower bound.
        if m_hi <= E_R_LOWER_M {
            l += 1;
            continue;
        }
        let a_l = basis.harmonic_amplitude(sw, l, omega0);
        let m_start = m_lo.max(E_R_LOWER_M);
        let m_end = m_hi.min(m_max + 1);
        for m in m_start..m_end {
            let synth = basis.synthetic_bin(m, l, omega0, a_l);
            let observed = sw[packed_index(m)];
            let dre = observed.re - synth.re;
            let dim = observed.im - synth.im;
            e_r += dre * dre + dim * dim;
        }
        l += 1;
        // Hard safety cap (shouldn't trigger for admissible ω_0).
        if l > u32::from(L_HAT_MAX) + 2 {
            break;
        }
    }
    e_r
}

/// Refine `P̂_I` to quarter-sample resolution.
///
/// `sw` is the pre-computed Eq. 29 signal DFT for the current frame
/// (one per frame, shared across the ten candidates per addendum
/// §0.4.2). `basis` is the shared `W_R(k)` table. `p_hat_i` is the
/// §0.3 initial pitch estimate in samples.
///
/// Tie-breaking: when two candidates tie on `E_R`, the one closer to
/// `P̂_I` wins (equivalent to lower quantizer-step index via Eq. 45),
/// minimizing bit churn on borderline frames. For deterministic
/// bit-exactness against DVSI test vectors the tie rule matters only
/// on floating-point collisions, which are rare in practice.
pub fn refine_pitch(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    p_hat_i: f64,
) -> PitchRefinement {
    let mut best_e_r = f64::INFINITY;
    let mut best_omega = 2.0 * core::f64::consts::PI / p_hat_i;
    let mut best_abs_offset = f64::INFINITY;
    for &offset in &REFINE_OFFSETS {
        let p = p_hat_i + offset;
        let omega0 = 2.0 * core::f64::consts::PI / p;
        let e_r = residual_e_r(sw, basis, omega0);
        let abs_off = offset.abs();
        // Strict-less-than keeps the first seen winner; on an exact
        // tie the tie-break kicks in via `abs_off < best_abs_offset`.
        let take = e_r < best_e_r
            || (e_r == best_e_r && abs_off < best_abs_offset);
        if take {
            best_e_r = e_r;
            best_omega = omega0;
            best_abs_offset = abs_off;
        }
    }
    let l_hat = harmonic_count_for(best_omega);
    let b0 = quantize_pitch_index(best_omega).unwrap_or(B0_MAX);
    PitchRefinement {
        omega_hat: best_omega,
        l_hat,
        b0,
        e_r: best_e_r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{W_R_HALF, signal_spectrum};

    #[test]
    fn refine_offsets_are_symmetric_and_quarter_sample_spaced() {
        // Spacing 0.25, symmetric around 0, excludes 0.
        assert!(!REFINE_OFFSETS.contains(&0.0));
        for i in 1..N_REFINE_CANDIDATES {
            let step = REFINE_OFFSETS[i] - REFINE_OFFSETS[i - 1];
            assert!((step - 0.25).abs() < 1e-12, "step at i={i}: {step}");
        }
        let sum: f64 = REFINE_OFFSETS.iter().sum();
        assert!(sum.abs() < 1e-12, "offsets not symmetric, sum = {sum}");
        // Range is ±9/8.
        assert_eq!(REFINE_OFFSETS[0], -9.0 / 8.0);
        assert_eq!(REFINE_OFFSETS[9], 9.0 / 8.0);
    }

    #[test]
    fn harmonic_count_respects_admissible_range() {
        // At ω̂_0 = 2π/123.125: π/ω̂_0 = 61.5625, inner floor = 61,
        //                      ·0.9254 = 56.4494, floor = 56.
        let omega_lo = 2.0 * core::f64::consts::PI / 123.125;
        assert_eq!(harmonic_count_for(omega_lo), 56);
        // At ω̂_0 = 2π/19.875: π/ω̂_0 = 9.9375, inner floor = 10,
        //                      ·0.9254 = 9.254, floor = 9.
        let omega_hi = 2.0 * core::f64::consts::PI / 19.875;
        assert_eq!(harmonic_count_for(omega_hi), 9);
        // Mid-range: ω̂_0 = 2π/50 → π/ω̂_0 = 25, inner floor = 25,
        //                           ·0.9254 = 23.135, floor = 23.
        let omega_mid = 2.0 * core::f64::consts::PI / 50.0;
        assert_eq!(harmonic_count_for(omega_mid), 23);
    }

    #[test]
    fn harmonic_count_applies_inner_floor() {
        // π/ω̂_0 = 13.8: buggy form = ⌊0.9254·14.05⌋ = 13, correct form
        // = ⌊0.9254·⌊14.05⌋⌋ = ⌊0.9254·14⌋ = 12. Regression guard for
        // specs commit d20f5ab (addendum §0.4.3).
        let omega_hat = core::f64::consts::PI / 13.8;
        assert_eq!(harmonic_count_for(omega_hat), 12);
    }

    #[test]
    fn quantize_pitch_index_covers_full_range() {
        // Lowest ω̂_0 → b̂_0 = 207.
        let omega_lo = 2.0 * core::f64::consts::PI / 123.125;
        assert_eq!(quantize_pitch_index(omega_lo), Some(207));
        // Highest ω̂_0 → b̂_0 = 0.
        let omega_hi = 2.0 * core::f64::consts::PI / 19.875;
        assert_eq!(quantize_pitch_index(omega_hi), Some(0));
        // Out of range returns None.
        let omega_too_hi = 2.0 * core::f64::consts::PI / 10.0; // P=10, very high ω
        assert_eq!(quantize_pitch_index(omega_too_hi), None);
        let omega_too_lo = 2.0 * core::f64::consts::PI / 250.0; // P=250, very low
        assert_eq!(quantize_pitch_index(omega_too_lo), None);
    }

    /// `quantize_pitch_index` and the imbe decoder's pitch_decode
    /// must round-trip within one quantization step — the encoder's
    /// `−39` floor pairs with the decoder's `+39.5` offset to
    /// implement midtread quantization per addendum §0.4.4.
    #[test]
    fn quantize_pitch_index_roundtrips_through_decoder() {
        use core::f64::consts::PI;
        // Sweep P and check that quantize(2π/P) reconstructs to within
        // one quantization step of the input ω.
        for p_times_2 in 42u32..=244 {
            // P in half-sample units: 21.0 to 122.0.
            let p = f64::from(p_times_2) * 0.5;
            let omega_hat = 2.0 * PI / p;
            let b0 = quantize_pitch_index(omega_hat).expect("in range");
            // Decoder: ω̃_0 = 4π / (b̂_0 + 39.5).
            let omega_tilde = 4.0 * PI / (f64::from(b0) + 39.5);
            // One quantization step ≈ |dω̃/db̂| = 4π / (b̂+39.5)².
            let step = 4.0 * PI / (f64::from(b0) + 39.5).powi(2);
            assert!(
                (omega_hat - omega_tilde).abs() <= step,
                "P={p}: ω̂={omega_hat:.6}, ω̃={omega_tilde:.6}, step={step:.6}"
            );
        }
    }

    /// Build a broadband periodic signal: harmonics 1..=max_h at
    /// amplitudes `1/h` (natural glottal-like rolloff). Populates
    /// spectral bins across the full range `m ∈ [0, 128]`, so the
    /// Eq. 24 residual (summed from m=50 upward) has non-trivial
    /// signal content to fit.
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

    /// `refine_pitch` returns a refinement within ±1.125 samples of
    /// `P̂_I` (quarter-sample resolution). Broadband signal so E_R
    /// above m=50 has non-trivial content.
    #[test]
    fn refine_pitch_returns_candidate_within_offset_range() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        let p_refined = 2.0 * PI / refined.omega_hat;
        assert!(
            (p_refined - 50.0).abs() <= 9.0 / 8.0 + 1e-6,
            "P refined = {p_refined}, out of ±1.125 around 50"
        );
        let matches_a_candidate = REFINE_OFFSETS
            .iter()
            .any(|&offset| (p_refined - (50.0 + offset)).abs() < 1e-9);
        assert!(matches_a_candidate, "P refined = {p_refined} not a candidate");
    }

    /// For a broadband harmonic signal at period 50 with `P̂_I = 50`,
    /// the refinement should pick a candidate close to 50. Allow ±3/8
    /// tolerance — the residual landscape around the truth is shallow
    /// and finite-precision DFT evaluation can push the minimum one
    /// or two steps off the nearest-to-truth candidate.
    #[test]
    fn refine_pitch_picks_close_candidate_on_broadband_signal() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        let p_refined = 2.0 * PI / refined.omega_hat;
        assert!(
            (p_refined - 50.0).abs() <= 3.0 / 8.0 + 1e-6,
            "P refined = {p_refined}, expected within ±3/8 of 50"
        );
    }

    /// L̂, â_l, b̂_l derived parameters are consistent with the refined
    /// ω̂_0 (matches `harmonic_count_for` and `HarmonicBasis::bin_endpoints`).
    #[test]
    fn refine_pitch_derived_parameters_are_consistent() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        assert_eq!(refined.l_hat, harmonic_count_for(refined.omega_hat));
        assert_eq!(refined.b0, quantize_pitch_index(refined.omega_hat).unwrap());
        assert!((L_HAT_MIN..=L_HAT_MAX).contains(&refined.l_hat));
        assert!(refined.b0 <= B0_MAX);
    }

    /// Diagnostic: dump E_R for each candidate.
    #[test]
    #[ignore]
    fn probe_dump_residual_e_r() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let omega_truth = 2.0 * PI / 50.0;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega_truth * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        for &offset in &REFINE_OFFSETS {
            let p = 50.0 + offset;
            let omega0 = 2.0 * PI / p;
            let e_r = residual_e_r(&sw, &basis, omega0);
            println!("  P = {p:6.3} (offset {offset:+.3})  ω₀ = {omega0:.6}  E_R = {e_r:.6e}");
        }
    }

    /// Residual is nonnegative (sum of squared magnitudes).
    #[test]
    fn refine_pitch_residual_is_nonnegative() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(40.0, 15);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 40.0);
        assert!(refined.e_r >= 0.0, "E_R = {}", refined.e_r);
        assert!(refined.e_r.is_finite());
    }
}
