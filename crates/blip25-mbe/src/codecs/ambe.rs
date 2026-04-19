//! AMBE — AMBE-1000, Generation 1 of the DVSI MBE codec family.
//!
//! Covers rate indices `r0..r15` in the AMBE-3000 rate table.
//!
//! Primary references:
//!
//! - US5054072 — MBE vocoder (expired ~2009)
//! - US5226084 — MBE parameter quantization (expired ~2010)
//!
//! ## Consumer entry points
//!
//! This module exposes `synthesize_frame` and `synthesize_tone` so a
//! consumer can import one codec generation's synthesis API without
//! reaching into [`super::mbe_baseline`]. The initial Generation 1
//! implementation delegates to the baseline §1.10–§1.12 synthesizer
//! with BABA-A Eq. 141 phase (random-phase restart on voiced onsets).
//! The two functions are currently identical; the split exists so
//! that tone-specific rendering (e.g. dual-sinusoid DTMF) can be
//! layered onto [`synthesize_tone`] without perturbing the voice
//! API. Re-exports [`SynthState`] and [`SAMPLES_PER_FRAME`] so the
//! consumer gets the state type from the same module as the synth.

use crate::codecs::mbe_baseline::{synthesize_frame as baseline_synthesize, FRAME_SAMPLES};
use crate::mbe_params::MbeParams;

pub use crate::codecs::mbe_baseline::SynthState;

/// Number of 8 kHz samples in one 20 ms synthesis frame.
pub const SAMPLES_PER_FRAME: usize = FRAME_SAMPLES;

/// Synthesize one voice frame of PCM from `params`.
///
/// Generation-1 (AMBE) phase: random restart on voiced onsets per
/// BABA-A Eq. 141. Frame-error handling (§1.11.1 repeat / §1.11.2
/// mute) reads [`SynthState::err`]; calibration scale reads
/// [`SynthState::gamma_w`]. Both default to "error-free" and the
/// committed `γ_w` on a fresh [`SynthState::new`].
pub fn synthesize_frame(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    let err = state.err;
    let gamma_w = state.gamma_w;
    baseline_synthesize(params, &err, gamma_w, state)
}

/// Synthesize one tone frame of PCM from `params`.
///
/// Generation 1 has no tone-specific logic; this delegates to
/// [`synthesize_frame`]. Consumers with tone frames (a half-rate
/// concept that AMBE-1000 does not carry on the wire) should still
/// prefer this entry point for dispatch symmetry — a future
/// dual-sinusoid renderer could land here without touching the voice
/// path.
pub fn synthesize_tone(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    synthesize_frame(params, state)
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
