//! AMBE+2 — AMBE-3000, Generation 3 of the DVSI MBE codec family.
//!
//! Covers rate indices `r32..r63` in the AMBE-3000 rate table. The current
//! DVSI generation; the primary target of this project's chip-conformance
//! harness. Builds on AMBE+ with:
//!
//! - US8595002 — half-rate AMBE+2 split vector quantization of spectral
//!   magnitudes with DCT-domain codebooks; per-harmonic (rather than
//!   per-band) voicing decisions; data-dependent scrambling for error
//!   resilience.
//! - US8315860 — enhanced full-rate encoding: three-state voicing model
//!   (voiced / unvoiced / pulsed), fundamental-frequency-field repurposing
//!   when no voiced bands exist, tone detection, spectral sidelobe
//!   suppression, and noise suppression via spectral subtraction.
//!
//! ## Consumer entry points
//!
//! Exposes `synthesize_frame` / `synthesize_tone` that delegate to
//! the phase-regen variant of the §1.10–§1.12 synth. US8595002 and
//! US8315860 are predominantly encode-side; the decode side re-uses
//! the AMBE+ phase-regen synth, which is the current deployment
//! pattern for P25 full-rate + half-rate when the consumer has
//! committed to AMBE+2-quality decoding regardless of wire
//! generation (the SCBA-mask pattern — see `INTEGRATION.md`
//! §"Spectral enhancement").

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
/// Generation-3 (AMBE+2) decode-side synth = AMBE+'s US5701390 phase
/// regeneration. The AMBE+2-specific bits (US8595002's per-harmonic
/// voicing, US8315860's pulsed-voicing / tone detection / spectral
/// subtraction) live on the **encode** side; decode-side this module
/// mirrors AMBE+.
pub fn synthesize_frame(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    let err = state.err;
    let gamma_w = state.gamma_w;
    baseline_synthesize_ambe_plus(params, &err, gamma_w, state)
}

/// Synthesize one tone frame of PCM from `params`. Delegates to
/// [`synthesize_frame`]; a future dedicated DTMF / ringback renderer
/// for P25 Phase 2 half-rate tone frames can land here.
pub fn synthesize_tone(
    params: &MbeParams,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    synthesize_frame(params, state)
}

/// Synthesize a repeated frame in response to a wire-layer erasure.
/// AMBE+2 decode-side shares AMBE+'s US5701390 phase regeneration.
pub fn synthesize_repeat(state: &mut SynthState) -> [i16; SAMPLES_PER_FRAME] {
    baseline_synthesize_repeat(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbe_params::L_MIN;

    fn silence_params() -> MbeParams {
        let voiced = vec![false; L_MIN as usize];
        let amps = vec![0.0_f32; L_MIN as usize];
        MbeParams::new(
            2.0 * core::f32::consts::PI / 50.0,
            L_MIN,
            &voiced,
            &amps,
        )
        .unwrap()
    }

    #[test]
    fn synthesize_frame_returns_expected_sample_count() {
        let params = silence_params();
        let mut state = SynthState::new();
        let pcm = synthesize_frame(&params, &mut state);
        assert_eq!(pcm.len(), SAMPLES_PER_FRAME);
    }

    #[test]
    fn synthesize_frame_matches_baseline_ambe_plus_path() {
        // The codec entry point must produce the same audio as a direct
        // call to the underlying baseline `synthesize_frame_ambe_plus`
        // with the same inputs — that is how consumers know the split
        // is a transparent alias, not a second implementation.
        let params = silence_params();
        let mut a = SynthState::new();
        let mut b = SynthState::new();
        let out_codec = synthesize_frame(&params, &mut a);
        let err = b.err;
        let gamma_w = b.gamma_w;
        let out_direct = baseline_synthesize_ambe_plus(&params, &err, gamma_w, &mut b);
        assert_eq!(out_codec, out_direct);
    }

    #[test]
    fn synthesize_tone_returns_expected_sample_count() {
        let params = silence_params();
        let mut state = SynthState::new();
        let pcm = synthesize_tone(&params, &mut state);
        assert_eq!(pcm.len(), SAMPLES_PER_FRAME);
    }

    #[test]
    fn synthesize_repeat_on_cold_start_emits_silence() {
        let mut state = SynthState::new();
        let pcm = synthesize_repeat(&mut state);
        assert_eq!(pcm.len(), SAMPLES_PER_FRAME);
        // First-call repeat with no `last_good` goes through the
        // cold-start fallback (silence MbeParams).
        let energy: i64 = pcm.iter().map(|&s| i64::from(s).pow(2)).sum();
        assert!(energy < 1_000_000, "cold-start repeat should be near-silent, got energy {energy}");
    }

    #[test]
    fn synthesize_repeat_after_voice_does_not_reset_state() {
        // After a voice frame, state.last_good is populated. Calling
        // synthesize_repeat should produce a non-empty output without
        // panicking and without returning state to cold-start.
        let params = silence_params();
        let mut state = SynthState::new();
        let _ = synthesize_frame(&params, &mut state);
        let s_e_after_voice = state.s_e;
        let _ = synthesize_repeat(&mut state);
        // s_e evolves on every frame; just verify it's still finite and
        // in a plausible range (the §1.10 recursion never makes it
        // negative or infinite on well-formed input).
        assert!(state.s_e.is_finite());
        assert!(state.s_e > 0.0);
        // We synthesized twice so epsilon_r has advanced; s_e may have
        // decayed. Sanity check: both are distinct from cold start.
        assert!(
            state.s_e != crate::codecs::mbe_baseline::INIT_S_E
                || s_e_after_voice != crate::codecs::mbe_baseline::INIT_S_E,
            "state should have evolved from cold start"
        );
    }
}
