//! MBE **synthesis** — parameters → PCM — over the vendored [`blip25_codec`] codec.
//!
//! This is the params→PCM companion to the wire/parameter layer
//! ([`crate::rate33::dequantize`], [`crate::imbe7200::dequantize`]): a consumer
//! that has already turned bits into [`MbeParams`] (and inspected the frame
//! classification, FEC error counts, DUID, etc. for its own control flow) hands
//! the params here to render audio.
//!
//! # This is NOT sample-identical to [`crate::vocoder::Vocoder::decode_bits`]
//!
//! The crate ships **two** synthesis back-ends and this module is the float one:
//!
//! | path | back-end |
//! |---|---|
//! | [`crate::vocoder::Vocoder::decode_bits`] (both codecs) | fixed-point OLA (`blip25_codec::dec`) |
//! | this module, and AMBE+2 Annex-T tone frames | float [`blip25_codec::synth`] |
//!
//! They are close — per-frame energy envelopes track to within ±5%, which is all
//! this module's own test asserts — but they are **not** the same samples, and
//! the constants that govern this path (`GAMMA_W`, `LCG_SEED`, the soft limiter)
//! do not affect `decode_bits` at all.
//!
//! For byte-for-byte parity with the shipped decoder, use `decode_bits`. There is
//! no params→PCM entry point into the fixed-point back-end, because that path
//! reconstructs its amplitudes from the wire bits with its own cross-frame state
//! rather than from a caller-supplied [`MbeParams`].
//!
//! Two-stage decode with this module:
//! ```no_run
//! use blip25_mbe::rate33::dequantize::{decode_to_params, Decoded, DecoderState};
//! use blip25_mbe::synth::{self, SynthState};
//! let mut dstate = DecoderState::default();
//! let mut sstate = SynthState::new();            // AMBE+2; `new_imbe()` for IMBE
//! # let info = [0u16; 4];
//! let pcm = match decode_to_params(&info, &mut dstate) {
//!     Ok(Decoded::Voice(p))       => synth::synthesize_frame(&p, &mut sstate),
//!     Ok(Decoded::Tone { params, .. }) => synth::synthesize_tone(&params, &mut sstate),
//!     _ /* erasure */             => synth::synthesize_repeat(&mut sstate),
//! };
//! ```
//!
//! Concealment: the consumer drives it. On a frame IT judges lost (FEC over
//! threshold, dropped burst, AMBE reject), call [`synthesize_repeat`] to
//! re-render the last good frame; mute (emit silence) after however many
//! consecutive repeats the caller's policy allows.

use crate::mbe_params::MbeParams;

/// Samples produced per synthesized frame (20 ms at 8 kHz).
pub(crate) const SAMPLES_PER_FRAME: usize = blip25_codec::synth::FRAME_SAMPLES;

/// Phase-handling selector. Advisory only: the reference vocoder synth picks its phase
/// model internally per rate, so this selector does not change the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PhaseMode {
    /// Baseline phase (IMBE / Phase 1).
    #[default]
    Baseline,
    /// AMBE+ phase regeneration (Phase 2 half-rate).
    AmbePlus,
}

/// Cross-frame synthesis state (overlap-add continuity, previous-frame
/// parameters for repeat concealment). One per decode channel; carry it across
/// frames. Construct with [`SynthState::new`] (AMBE+2 half-rate) or
/// [`SynthState::new_imbe`] (IMBE full-rate).
#[derive(Clone)]
pub struct SynthState {
    inner: blip25_codec::synth::SynthState,
    /// Last successfully synthesized voice/tone params — the repeat source.
    last: Option<blip25_codec::dequantize::MbeParams>,
}

impl SynthState {
    /// AMBE+2 (Phase 2 half-rate) synthesis state, cold.
    pub fn new() -> Self {
        Self {
            inner: blip25_codec::synth::SynthState::new(),
            last: None,
        }
    }

    /// IMBE (Phase 1 full-rate) synthesis state, cold.
    pub fn new_imbe() -> Self {
        Self {
            inner: blip25_codec::synth::SynthState::new_imbe(),
            last: None,
        }
    }
}

impl Default for SynthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge a crate [`MbeParams`] to the reference vocoder synth's parameter struct.
fn to_codec(p: &MbeParams) -> blip25_codec::dequantize::MbeParams {
    let l = p.harmonic_count();
    let n = l as usize;
    blip25_codec::dequantize::MbeParams {
        b0: 0, // pitch index; unused by synthesis (classification only)
        omega_0: p.omega_0(),
        l,
        voiced: p.voiced_slice()[..n].to_vec(),
        amplitudes: p.amplitudes_slice()[..n].to_vec(),
    }
}

/// Synthesize one **voice** frame from `params`, advancing `state`.
pub fn synthesize_frame(params: &MbeParams, state: &mut SynthState) -> [i16; SAMPLES_PER_FRAME] {
    state.inner.set_tone_frame(false);
    let wp = to_codec(params);
    let out = blip25_codec::synth::synthesize_frame(&wp, &mut state.inner);
    state.last = Some(wp);
    out
}

/// Synthesize one **tone** frame (Annex-T) from `params`. The harmonics are
/// rendered as pure sinusoids (no voiced phase continuity) — the reference vocoder tone
/// path — so a tone doesn't ring against a preceding voice frame.
pub fn synthesize_tone(params: &MbeParams, state: &mut SynthState) -> [i16; SAMPLES_PER_FRAME] {
    state.inner.set_tone_frame(true);
    let wp = to_codec(params);
    let out = blip25_codec::synth::synthesize_frame(&wp, &mut state.inner);
    state.last = Some(wp);
    out
}

/// Re-render the last successfully synthesized frame — **erasure/repeat
/// concealment**. Emits silence on a cold start (no prior good frame). The
/// caller decides when to invoke this (FEC over threshold, dropped burst) and
/// when to stop repeating and mute.
pub fn synthesize_repeat(state: &mut SynthState) -> [i16; SAMPLES_PER_FRAME] {
    match state.last.clone() {
        Some(wp) => blip25_codec::synth::synthesize_frame(&wp, &mut state.inner),
        None => [0i16; SAMPLES_PER_FRAME],
    }
}

/// [`synthesize_repeat`] with an explicit [`PhaseMode`]. The mode is advisory
/// and does not change the output.
pub fn synthesize_repeat_with_mode(
    _mode: PhaseMode,
    state: &mut SynthState,
) -> [i16; SAMPLES_PER_FRAME] {
    synthesize_repeat(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate33::dequantize::{decode_to_params, Decoded, DecoderState};

    /// Frame energy from the two-stage decode (dequantize → synth) tracks
    /// `Vocoder::decode_bits` within ±5% on a reference AMBE+2 vector.
    #[test]
    fn synth_matches_vocoder_env() {
        // Resolved from `CARGO_MANIFEST_DIR`, not from the working directory:
        // cargo runs a test binary with the *package* root as its cwd, so a
        // plain relative path looks for `crates/blip25-mbe/reference-material/`
        // and never resolves — which silently disabled this test entirely.
        const CLEAN_BIT: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../reference-material/Vectors/tv-rc/r33/clean.bit"
        );
        let bits = match std::fs::read(CLEAN_BIT) {
            Ok(b) => b,
            // The vectors are not redistributable and are gitignored, so their
            // absence is the normal state of a fresh clone. Announce the skip:
            // cargo reports a quiet `return` as a pass, indistinguishable from
            // a test that verified something.
            Err(_) => {
                eprintln!(
                    "SKIP synth_matches_vocoder_env: no reference vectors at {CLEAN_BIT}. \
                     This is the only coverage `blip25_mbe::synth` has."
                );
                return;
            }
        };
        use crate::vocoder::{Rate, Vocoder};
        let mut voc = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut dstate = DecoderState::default();
        let mut sstate = SynthState::new();
        let (mut e_ref, mut e_ours, mut n) = (0.0f64, 0.0f64, 0usize);
        for fr in bits.chunks_exact(9) {
            let refpcm = voc.decode_bits(fr).unwrap();
            let info = blip25_codec::frame::decode_bytes(fr).info;
            let ours = match decode_to_params(&info, &mut dstate) {
                Ok(Decoded::Voice(p)) => synthesize_frame(&p, &mut sstate),
                Ok(Decoded::Tone { params, .. }) => synthesize_tone(&params, &mut sstate),
                _ => synthesize_repeat(&mut sstate),
            };
            let er: f64 = refpcm
                .iter()
                .map(|&s| (s as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let eo: f64 = ours.iter().map(|&s| (s as f64).powi(2)).sum::<f64>().sqrt();
            e_ref += er;
            e_ours += eo;
            n += 1;
        }
        assert!(n > 0);
        // Frame energies must track closely (same synthesis, same content).
        let ratio = e_ours / e_ref.max(1.0);
        assert!(
            (0.95..=1.05).contains(&ratio),
            "energy ratio {ratio} out of range"
        );
    }
}
