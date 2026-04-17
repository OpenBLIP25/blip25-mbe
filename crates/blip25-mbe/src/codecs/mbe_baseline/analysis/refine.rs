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

/// Number of harmonics `L̂` from `ω̂_0` per Eq. 31.
#[inline]
pub fn harmonic_count_for(omega_hat: f64) -> u8 {
    let raw = (0.9254 * (core::f64::consts::PI / omega_hat + 0.25)).floor();
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
