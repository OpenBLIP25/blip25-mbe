//! AMBE+ — AMBE-2000, Generation 2 of the DVSI MBE codec family.
//!
//! Covers rate indices `r16..r31` in the AMBE-3000 rate table. Adds to the
//! Generation 1 algorithms:
//!
//! - US5701390 — phase regeneration from spectral-envelope shape,
//!   replacing the random-phase restarts of the baseline voiced synthesis
//!   path. Eliminates the "buzzy" artifact characteristic of baseline MBE.
//! - US6199037 — joint quantization of voicing metrics and pitch across
//!   subframes.
//!
//! ## Consumer entry points
//!
//! Exposes `synthesize_frame` / `synthesize_tone` that delegate to
//! [`super::mbe_baseline::synthesize_frame_ambe_plus`] — the US5701390
//! phase-regen variant of the §1.10–§1.12 synth.

use crate::codecs::mbe_baseline::{
    synthesize_frame_ambe_plus as baseline_synthesize_ambe_plus,
    synthesize_repeat_ambe_plus as baseline_synthesize_repeat, FRAME_SAMPLES,
};
use crate::mbe_params::MbeParams;

pub use crate::codecs::mbe_baseline::SynthState;

/// Number of 8 kHz samples in one 20 ms synthesis frame.
pub const SAMPLES_PER_FRAME: usize = FRAME_SAMPLES;

/// Synthesize one voice frame of PCM from `params`.
///
/// Generation-2 (AMBE+) phase regeneration per US5701390, replacing
/// BABA-A Eq. 141. Frame-error handling and `γ_w` read from
/// [`SynthState`] (see module-level docs for defaults).
pub fn synthesize_frame(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    let err = state.err;
    let gamma_w = state.gamma_w;
    baseline_synthesize_ambe_plus(params, &err, gamma_w, state)
}

/// Synthesize one tone frame of PCM from `params`. Delegates to
/// [`synthesize_frame`] pending a tone-specific renderer.
pub fn synthesize_tone(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    synthesize_frame(params, state)
}

/// Synthesize a repeated frame in response to a wire-layer erasure.
/// AMBE+ uses US5701390 phase regeneration on the repeated frame.
pub fn synthesize_repeat(state: &mut SynthState) -> [i16; SAMPLES_PER_FRAME] {
    baseline_synthesize_repeat(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbe_params::L_MIN;

    #[test]
    fn synthesize_frame_returns_expected_sample_count() {
        let voiced = vec![false; L_MIN as usize];
        let amps = vec![0.0f32; L_MIN as usize];
        let params = MbeParams::new(
            2.0 * core::f32::consts::PI / 50.0,
            L_MIN,
            &voiced,
            &amps,
        )
        .unwrap();
        let mut state = SynthState::new();
        let pcm = synthesize_frame(&params, &mut state);
        assert_eq!(pcm.len(), SAMPLES_PER_FRAME);
    }
}
