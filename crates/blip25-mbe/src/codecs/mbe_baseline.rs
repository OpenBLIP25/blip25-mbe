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
}
