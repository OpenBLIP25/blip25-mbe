//! AMBE+2 phase regeneration (US5701390, §5 of the AMBE-3000 decoder
//! implementation spec).
//!
//! Replaces BABA-A §1.12.2 Eq. 141 (noise-perturbed phase) for the
//! half-rate / AMBE+2 decoder. Full-rate IMBE continues to use the
//! baseline noise-based phase.
//!
//! Algorithm summary (see
//! `~/blip25-specs/DVSI/AMBE-3000/AMBE-3000_Decoder_Implementation_Spec.md`
//! §5 and `analysis/ambe_phase_regen_hilbert.md`):
//!
//! ```text
//! B_l        = log₂(M̄_l)                 for 1 ≤ l ≤ L̃, else 0
//! h(m)       = 2 / (π · m)                for m ≠ 0
//! h(0)       = 0
//! D          = 19                         (kernel half-width)
//! γ          = 0.44                       (scaling, patent-empirical)
//! φ_regen,l  = γ · Σ_{m=−D, m≠0}^{+D} h(m) · B_{l+m}
//! ```
//!
//! Kernel `h(m) = 2/(π·m)` is the discrete Hilbert transform impulse
//! response; the regenerated phase equals the Hilbert transform of the
//! log-spectrum, i.e. Kolmogorov's minimum-phase identity applied
//! across the harmonic index.

use crate::mbe_params::L_MAX;

/// Kernel half-width D per US5701390 col. 7 (patent-recommended).
pub const PHASE_KERNEL_D: usize = 19;

/// Scaling γ per US5701390 col. 7 line 55. The patent notes only that
/// "γ = 0.44 has been found to work well" and offers no derivation.
pub const GAMMA_PHASE: f64 = 0.44;

/// Return the precomputed kernel `γ · h(m)` for `m = 0..=PHASE_KERNEL_D`.
/// Entry `0` is zero (`h(0) = 0`). Entry `m` for `m > 0` is
/// `γ · 2/(π·m)`. The kernel is antisymmetric, so
/// `γ · h(−m) = −γ · h(m)` and this table suffices for the full support.
fn phase_kernel() -> &'static [f64; PHASE_KERNEL_D + 1] {
    use std::sync::OnceLock;
    static KERNEL: OnceLock<[f64; PHASE_KERNEL_D + 1]> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let mut k = [0.0_f64; PHASE_KERNEL_D + 1];
        for m in 1..=PHASE_KERNEL_D {
            k[m] = GAMMA_PHASE * 2.0 / (core::f64::consts::PI * m as f64);
        }
        k
    })
}

/// Compute `φ_regen,l` for `l = 1..=L̃` per US5701390 Eq. 8.
///
/// `m_bar[l-1]` carries `M̄_l(0)` for `1 ≤ l ≤ L̃`. The `phi_regen`
/// output is 1-indexed (slot `[0]` unused, slots `[1..=L̃]` written),
/// matching the convention used elsewhere in the synthesizer.
/// Slots beyond `L̃` are left at zero.
///
/// Harmonic indices outside `[1, L̃]` contribute `B = 0` to the sum
/// (zero-padding per §5.2 step 1).
///
/// `M̄_l = 0` is handled by clamping `B_l` to `log2(1e-10) ≈ −33` per
/// §5.4. (The spec leaves this unspecified between "floor" and "zero";
/// the floor is chosen to match observed OSS-MBE behavior, pending
/// empirical chip validation.)
pub fn ambe_phase_regen(m_bar: &[f32], phi_regen: &mut [f64; L_MAX as usize + 1]) {
    let l_hat = m_bar.len();
    debug_assert!(l_hat <= L_MAX as usize, "phase_regen: L̃ > L_MAX");

    // Step 1: B_l = log₂(M̄_l).
    // Use M̄_l clamped by log2(1e-10) ≈ −33.22 for numerical safety
    // when M̄_l is zero (unvoiced-past-enhancement), per §5.4.
    let mut b = [0.0_f64; L_MAX as usize + 1];
    for l in 1..=l_hat {
        let m = f64::from(m_bar[l - 1]);
        b[l] = if m > 1e-10 {
            m.log2()
        } else {
            (1e-10_f64).log2()
        };
    }
    // b[0] and b[l_hat + 1..] stay zero — matches the spec's
    // zero-padding rule for indices outside [1, L̃].

    let kernel = phase_kernel();
    for l in 1..=l_hat {
        let mut acc = 0.0_f64;
        for m in 1..=PHASE_KERNEL_D {
            // Antisymmetric sum: γ·h(m) · (B_{l+m} − B_{l−m}).
            let b_plus = if l + m <= L_MAX as usize {
                b[l + m]
            } else {
                0.0
            };
            // l ≥ 1, so l - m is either >= 1 (use b[l-m]) or <= 0 (use 0).
            let b_minus = if l > m { b[l - m] } else { 0.0 };
            acc += kernel[m] * (b_plus - b_minus);
        }
        phi_regen[l] = acc;
    }
    // phi_regen[l] for l > l_hat stays at caller-supplied value (we
    // don't touch it; caller must zero or manage appropriately).
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn phase_kernel_first_entry_is_zero() {
        assert_eq!(phase_kernel()[0], 0.0);
    }

    #[test]
    fn phase_kernel_first_nonzero_matches_patent_constant() {
        // γ · 2/π = 0.44 · 2/π ≈ 0.28011
        let expected = 0.44 * 2.0 / core::f64::consts::PI;
        assert!((phase_kernel()[1] - expected).abs() < TOLERANCE);
    }

    #[test]
    fn phase_kernel_decay_is_one_over_m() {
        // PHASE_KERNEL[m] · m should be constant = γ · 2/π.
        let kernel = phase_kernel();
        let target = kernel[1];
        for m in 2..=PHASE_KERNEL_D {
            let scaled = kernel[m] * m as f64;
            assert!(
                (scaled - target).abs() < TOLERANCE,
                "m={m}: {scaled} vs {target}",
            );
        }
    }

    #[test]
    fn regen_of_constant_log_spectrum_is_zero() {
        // If B_l is constant across l, the antisymmetric kernel
        // integrates to zero. But note: the kernel's truncation + the
        // zero-padding at the edges break perfect cancellation near
        // the boundaries, so test with a fully-supported middle index.
        let l_hat = 2 * PHASE_KERNEL_D + 1; // harmonics [1..=39]
        let m_bar: Vec<f32> = vec![4.0; l_hat]; // log2(4) = 2
        let mut phi = [0.0_f64; L_MAX as usize + 1];
        ambe_phase_regen(&m_bar, &mut phi);
        // Center harmonic has full ±D support in-range → zero.
        let l_center = PHASE_KERNEL_D + 1; // l = 20, both l±19 in [1, 39]
        assert!(
            phi[l_center].abs() < TOLERANCE,
            "center regen not zero: phi[{l_center}] = {}",
            phi[l_center],
        );
    }

    #[test]
    fn regen_of_monotonic_ramp_is_positive() {
        // B_l linearly increasing → regen should be positive (the
        // Hilbert transform of a ramp has a positive sign in the
        // forward direction). Specifically for a uniform ramp
        // B_l = α·l, the sum γ·Σ h(m)·α·(l+m) − h(m)·α·(l−m)
        //       = γ · α · Σ 2·m · h(m) = γ · α · Σ 4/π > 0.
        let l_hat = 2 * PHASE_KERNEL_D + 1;
        let m_bar: Vec<f32> = (1..=l_hat).map(|l| (2.0_f32).powi(l as i32 - 20)).collect();
        let mut phi = [0.0_f64; L_MAX as usize + 1];
        ambe_phase_regen(&m_bar, &mut phi);
        let l_center = PHASE_KERNEL_D + 1;
        assert!(
            phi[l_center] > 0.0,
            "ramp regen not positive: phi[{l_center}] = {}",
            phi[l_center],
        );
    }

    #[test]
    fn regen_antisymmetric_kernel_zero_on_symmetric_log_spectrum() {
        // A spectrum symmetric about l = l_center should produce zero
        // regenerated phase at l_center regardless of shape — the
        // antisymmetric kernel cancels symmetric contributions.
        let l_hat = 2 * PHASE_KERNEL_D + 1; // 39 harmonics
        let l_center = PHASE_KERNEL_D + 1; // l = 20
        let m_bar: Vec<f32> = (1..=l_hat)
            .map(|l| {
                let d = (l as i32 - l_center as i32).abs() as f32;
                (2.0_f32).powf(-d)
            })
            .collect();
        let mut phi = [0.0_f64; L_MAX as usize + 1];
        ambe_phase_regen(&m_bar, &mut phi);
        assert!(
            phi[l_center].abs() < TOLERANCE,
            "symmetric regen not zero: phi[{l_center}] = {}",
            phi[l_center],
        );
    }

    #[test]
    fn regen_zero_amplitude_treated_as_floor_not_neg_infinity() {
        // A single harmonic at M̄_l = 0 should not NaN or blow up.
        let m_bar: Vec<f32> = vec![0.0; 40];
        let mut phi = [0.0_f64; L_MAX as usize + 1];
        ambe_phase_regen(&m_bar, &mut phi);
        // All regen values should be finite (zero kernel·zero input,
        // but floor kicks in so no −∞).
        for l in 1..=40 {
            assert!(phi[l].is_finite(), "phi[{l}] = {} not finite", phi[l]);
        }
    }

    #[test]
    fn regen_writes_only_up_to_l_hat() {
        // Caller provides l_hat = 10; slots [11..=L_MAX] untouched.
        let m_bar: Vec<f32> = vec![1.0; 10];
        let mut phi = [1.0_f64; L_MAX as usize + 1];
        ambe_phase_regen(&m_bar, &mut phi);
        // phi[0] untouched (sentinel), phi[11..] untouched.
        assert_eq!(phi[0], 1.0);
        for l in 11..=L_MAX as usize {
            assert_eq!(phi[l], 1.0, "phi[{l}] should be untouched");
        }
    }
}
