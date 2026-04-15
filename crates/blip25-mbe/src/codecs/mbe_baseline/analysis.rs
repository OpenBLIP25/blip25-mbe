//! MBE baseline analysis encoder — PCM → [`MbeParams`].
//!
//! The forward direction of the vocoder: 8 kHz 16-bit PCM in, one
//! [`MbeParams`] out per 20 ms (160-sample) frame. The result is
//! consumed by any of the wire-side `quantize` pipelines
//! ([`crate::p25_fullrate::quantize`],
//! [`crate::p25_halfrate::quantize`],
//! [`crate::dvsi_3000::quantize`]) and is independent of wire format.
//!
//! # Status: scaffolding only
//!
//! The derived implementation spec
//! (`~/blip25-specs/standards/TIA-102.BABA-A/P25_Vocoder_Implementation_Spec.md`)
//! is decode-only. The analysis-side equations — BABA-A §§2–5:
//!
//! - initial pitch estimation (Eq. 5–9) with look-back / look-ahead
//!   tracking,
//! - pitch refinement (Eq. 24–31) minimizing the residual `E_R(ω₀)`
//!   over the half-sample grid,
//! - the harmonic basis function `S_w(m, l)` (Eq. 25–28),
//! - the spectral amplitude estimator (Eq. 32–33),
//! - the V/UV discriminant `D_k` and the `θ_G(k, ω̂₀)` threshold
//!   (Eq. 34–42), and
//! - the log-magnitude prediction residual and its cross-frame
//!   predictor state,
//!
//! — are not rendered as code-ready derived material anywhere in the
//! implementation spec. The gap is enumerated in
//! `~/blip25-specs/analysis/vocoder_analysis_encoder_gap.md`. Per the
//! project's spec-first policy (`CLAUDE.md`), the DSP math cannot be
//! written here until the implementation spec has been extended.
//!
//! This module therefore currently provides only the scaffolding that
//! does *not* require algorithmic choices: the entry-point signature,
//! the state-struct shape at the level the spec does constrain, and
//! access to the three annex tables the analysis encoder needs. Every
//! function whose body would require a design choice unsupported by
//! the implementation spec returns [`AnalysisError::SpecGap`] or
//! `unimplemented!()` with a pointer to the gap report.
//!
//! # Annex tables available here
//!
//! - **Annex B** — analysis window `wI(n)`, 301 values, used by initial
//!   pitch estimation. [`analysis_window`], [`IMBE_ANALYSIS_WINDOW`].
//! - **Annex C** — pitch refinement window `wR(n)`, 221 values, used by
//!   pitch refinement and spectral amplitude estimation.
//!   [`refinement_window`], [`IMBE_REFINEMENT_WINDOW`].
//! - **Annex D** — FIR LPF `h_LPF(n)`, 21 taps, used by the initial
//!   pitch autocorrelation. [`lpf_tap`], [`IMBE_ANNEX_D_LPF`].

use crate::mbe_params::{MbeParams, SAMPLES_PER_FRAME};

include!(concat!(env!("OUT_DIR"), "/annex_b_analysis_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_c_refinement_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_d_lpf.rs"));

/// Lookup `wI(n)` for `n ∈ [−150, 150]`. Returns `0.0` outside that
/// range (the spec treats `wI` as zero outside its support).
#[inline]
pub fn analysis_window(n: i32) -> f32 {
    if !(-150..=150).contains(&n) {
        return 0.0;
    }
    IMBE_ANALYSIS_WINDOW[(n + 150) as usize]
}

/// Lookup `wR(n)` for `n ∈ [−110, 110]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn refinement_window(n: i32) -> f32 {
    if !(-110..=110).contains(&n) {
        return 0.0;
    }
    IMBE_REFINEMENT_WINDOW[(n + 110) as usize]
}

/// Lookup `h_LPF(n)` for `n ∈ [−10, 10]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn lpf_tap(n: i32) -> f32 {
    if !(-10..=10).contains(&n) {
        return 0.0;
    }
    IMBE_ANNEX_D_LPF[(n + 10) as usize]
}

/// Errors reported by the analysis encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisError {
    /// The input PCM slice was not exactly [`SAMPLES_PER_FRAME`] samples.
    WrongFrameLength,
    /// An algorithm required by the encoder is not yet implementable
    /// because the derived implementation spec does not cover it. See
    /// `~/blip25-specs/analysis/vocoder_analysis_encoder_gap.md`.
    SpecGap,
}

/// Cross-frame state carried by the analysis encoder.
///
/// The specific fields and their initialization values are a spec gap.
/// Section 8 of the gap report lists the state pitch history, residual
/// energy history, `ξ_max` V/UV threshold state, and the `R̃_l(−1)`
/// log-magnitude predictor — but initial values and reset policies are
/// not specified. The type is kept deliberately opaque until the
/// addendum lands.
#[derive(Clone, Debug, Default)]
pub struct AnalysisState {
    _private: (),
}

impl AnalysisState {
    /// Construct a cold-start encoder state.
    ///
    /// The spec does not yet define encoder-side Annex A
    /// initialization values; this simply marks the state as uninitialized.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

/// Analyze a 20 ms (160-sample) frame of 8 kHz 16-bit PCM and produce
/// the MBE parameters for that frame.
///
/// # Spec gap
///
/// This function cannot be implemented against the current
/// implementation spec. See
/// `~/blip25-specs/analysis/vocoder_analysis_encoder_gap.md` for the
/// enumeration of missing algorithms. The signature is stable; the
/// body will be filled in once the implementation-spec addendum lands.
///
/// # Arguments
///
/// - `pcm` — exactly `SAMPLES_PER_FRAME` (160) `i16` samples, the
///   20 ms frame being encoded. Per BABA-A §1.2 the algorithmic delay
///   is 80 ms; when that lookahead model is spec'd, this signature
///   will change to accept a buffer covering lookahead + current +
///   look-back.
/// - `state` — cross-frame encoder state. Will be mutated in place.
#[allow(unused_variables)]
pub fn encode(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<MbeParams, AnalysisError> {
    if pcm.len() != SAMPLES_PER_FRAME as usize {
        return Err(AnalysisError::WrongFrameLength);
    }
    Err(AnalysisError::SpecGap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_window_symmetry_and_endpoints() {
        assert_eq!(ANALYSIS_WINDOW_LEN, 301);
        // Outside support is zero.
        assert_eq!(analysis_window(-151), 0.0);
        assert_eq!(analysis_window(151), 0.0);
        // Even symmetry.
        for n in 1..=150 {
            assert!(
                (analysis_window(-n) - analysis_window(n)).abs() < 1e-5,
                "wI(-{n}) != wI({n})"
            );
        }
    }

    #[test]
    fn refinement_window_peak_and_symmetry() {
        assert_eq!(REFINEMENT_WINDOW_LEN, 221);
        assert!((refinement_window(0) - 1.0).abs() < 1e-6);
        assert_eq!(refinement_window(-111), 0.0);
        assert_eq!(refinement_window(111), 0.0);
        for n in 1..=110 {
            assert!(
                (refinement_window(-n) - refinement_window(n)).abs() < 1e-5,
                "wR(-{n}) != wR({n})"
            );
        }
    }

    #[test]
    fn lpf_symmetry_and_support() {
        assert_eq!(ANNEX_D_LPF_LEN, 21);
        for n in 1..=10 {
            assert!((lpf_tap(-n) - lpf_tap(n)).abs() < 1e-6, "h_LPF(-{n}) != h_LPF({n})");
        }
        assert_eq!(lpf_tap(-11), 0.0);
        assert_eq!(lpf_tap(11), 0.0);
    }

    #[test]
    fn encode_rejects_wrong_length() {
        let mut s = AnalysisState::new();
        assert_eq!(
            encode(&[0i16; 159], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
        assert_eq!(
            encode(&[0i16; 161], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
    }

    #[test]
    fn encode_reports_spec_gap_on_well_formed_input() {
        let mut s = AnalysisState::new();
        let pcm = [0i16; SAMPLES_PER_FRAME as usize];
        assert_eq!(encode(&pcm, &mut s), Err(AnalysisError::SpecGap));
    }
}
