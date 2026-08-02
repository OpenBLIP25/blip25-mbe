// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! AMBE+2 phase regeneration (US5701390 §5): the discrete Hilbert
//! transform of the log-magnitude spectrum across the harmonic index.
//!
//! The *baseline* magnitude-driven phase used by the half-rate synthesizer;
//! the reference's exact per-harmonic phase back-end is the documented proprietary
//! frontier and is not reproduced (see README).

use crate::dequantize::L_MAX;

/// Kernel half-width D (US5701390 col. 7).
pub(crate) const PHASE_KERNEL_D: usize = 19;
/// Scaling gamma (patent-empirical).
pub(crate) const GAMMA_PHASE: f64 = 0.44;

fn phase_kernel() -> &'static [f64; L_MAX + 1] {
    use std::sync::OnceLock;
    static KERNEL: OnceLock<[f64; L_MAX + 1]> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let mut k = [0.0_f64; L_MAX + 1];
        for m in 1..=PHASE_KERNEL_D {
            k[m] = GAMMA_PHASE * 2.0 / (core::f64::consts::PI * m as f64);
        }
        k
    })
}

/// Compute phi_regen,l for l = 1..=L (1-indexed output; slot 0 unused).
pub(crate) fn ambe_phase_regen(m_bar: &[f32], phi_regen: &mut [f64; L_MAX + 1]) {
    let l_hat = m_bar.len();
    let mut b = [0.0_f64; L_MAX + 1];
    for l in 1..=l_hat {
        let m = f64::from(m_bar[l - 1]);
        b[l] = if m > 1e-10 {
            m.log2()
        } else {
            (1e-10_f64).log2()
        };
    }
    let kernel = phase_kernel();
    let d = PHASE_KERNEL_D;
    for l in 1..=l_hat {
        let mut acc = 0.0_f64;
        for m in 1..=d {
            let b_plus = if l + m <= L_MAX { b[l + m] } else { 0.0 };
            let b_minus = if l > m { b[l - m] } else { 0.0 };
            acc += kernel[m] * (b_plus - b_minus);
        }
        phi_regen[l] = acc;
    }
}
