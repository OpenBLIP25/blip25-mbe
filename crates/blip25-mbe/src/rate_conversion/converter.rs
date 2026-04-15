//! Cross-rate conversion — parse at rate A, requantize, pack at rate B.
//!
//! Parameters flow through:
//!
//! ```text
//!   A-rate frame bits
//!     → parse via the source rate's wire module (decode + dequantize)
//!     → MbeParams
//!     → quantize at the destination rate
//!     → pack via the destination rate's wire module (prioritize + encode)
//!     → B-rate frame bits
//! ```
//!
//! Each converter holds two state objects: the source-rate decoder
//! state (which advances as input frames are consumed) and the
//! destination-rate encoder state (which advances as output frames are
//! produced). They are independent because the upstream radio's encoder
//! and the downstream radio's decoder evolve in lockstep with the
//! converter's encoder and decoder respectively, not with each other.
//!
//! ## What this first cut does not do
//!
//! - **Voicing-band normalization** and **magnitude interpolation across
//!   L mismatches** described in US7634399. The current path relies on
//!   the destination quantizer's own pitch table to pick a representable
//!   `(ω₀, L)` and resamples per-harmonic amplitudes only via that
//!   choice. Bridging IMBE full-rate (L = 9..56 from a 208-entry table)
//!   to AMBE+2 half-rate (L = 9..56 from a 120-entry Annex L table) is
//!   well-behaved in practice because both share the same `(ω₀, L)`
//!   relationship; large excursions are rare.
//! - **Tone / silence / erasure handling.** Half-rate tone frames
//!   (`b̂₀ ∈ [126, 127]`) and reserved values surface as
//!   [`crate::p25_fullrate::dequantize::DecodeError::BadPitch`]. A
//!   future revision should detect these on the input side and emit a
//!   matching frame on the output side instead of erroring.

use crate::p25_fullrate::dequantize::{
    DecodeError, DecoderState, EncodeError, dequantize, quantize,
};
use crate::p25_fullrate::frame::{decode_frame as decode_full, encode_frame as encode_full};
use crate::p25_fullrate::priority::prioritize as prioritize_full;
use crate::p25_halfrate::dequantize::{
    DecoderState as HalfDecoderState, dequantize as dequantize_half, quantize as quantize_half,
};
use crate::p25_halfrate::frame::{
    DIBITS_PER_FRAME, decode_frame as decode_half, encode_frame as encode_half,
};

/// Error from a cross-rate conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertError {
    /// The source-rate dequantizer rejected the input frame.
    Decode(DecodeError),
    /// The destination-rate quantizer rejected the parameters. Only
    /// the full-rate encoder produces this; the half-rate encoder
    /// uses [`DecodeError`] for its own pitch-out-of-range case.
    Encode(EncodeError),
    /// Parameters fell outside the half-rate pitch table when packing
    /// to half rate. Re-uses [`DecodeError::BadPitch`] for symmetry
    /// with the half-rate encoder's signature.
    HalfPitchOutOfRange,
}

impl From<DecodeError> for ConvertError {
    fn from(e: DecodeError) -> Self { Self::Decode(e) }
}

impl From<EncodeError> for ConvertError {
    fn from(e: EncodeError) -> Self { Self::Encode(e) }
}

// ---------------------------------------------------------------------------
// P25 Phase 1 full-rate (IMBE) → P25 Phase 2 half-rate (AMBE+2)
// ---------------------------------------------------------------------------

/// Full-rate (144-bit IMBE) → half-rate (72-bit AMBE+2) converter.
///
/// Holds the source decoder state and the destination encoder state.
/// Call [`Self::convert`] once per 20 ms frame.
#[derive(Clone, Debug, Default)]
pub struct FullToHalfConverter {
    decoder: DecoderState,
    encoder: HalfDecoderState,
}

impl FullToHalfConverter {
    /// Cold-start converter.
    pub fn new() -> Self { Self::default() }

    /// Convert one full-rate frame (72 dibits) into one half-rate
    /// frame (36 dibits).
    pub fn convert(&mut self, dibits: &[u8; 72]) -> Result<[u8; DIBITS_PER_FRAME], ConvertError> {
        let frame = decode_full(dibits);
        let params = dequantize(&frame.info, &mut self.decoder)?;
        let u = quantize_half(&params, &mut self.encoder).map_err(|e| match e {
            DecodeError::BadPitch => ConvertError::HalfPitchOutOfRange,
            other => ConvertError::Decode(other),
        })?;
        Ok(encode_half(&u))
    }

    /// Borrow the source decoder state.
    pub fn decoder_state(&self) -> &DecoderState { &self.decoder }

    /// Borrow the destination encoder state.
    pub fn encoder_state(&self) -> &HalfDecoderState { &self.encoder }
}

// ---------------------------------------------------------------------------
// P25 Phase 2 half-rate (AMBE+2) → P25 Phase 1 full-rate (IMBE)
// ---------------------------------------------------------------------------

/// Half-rate (72-bit AMBE+2) → full-rate (144-bit IMBE) converter.
#[derive(Clone, Debug, Default)]
pub struct HalfToFullConverter {
    decoder: HalfDecoderState,
    encoder: DecoderState,
}

impl HalfToFullConverter {
    /// Cold-start converter.
    pub fn new() -> Self { Self::default() }

    /// Convert one half-rate frame (36 dibits) into one full-rate
    /// frame (72 dibits).
    pub fn convert(&mut self, dibits: &[u8; DIBITS_PER_FRAME]) -> Result<[u8; 72], ConvertError> {
        let frame = decode_half(dibits);
        let params = dequantize_half(&frame.info, &mut self.decoder)?;
        let l = params.harmonic_count();
        let b = quantize(&params, &mut self.encoder)?;
        let u = prioritize_full(&b, l);
        Ok(encode_full(&u))
    }

    /// Borrow the source decoder state.
    pub fn decoder_state(&self) -> &HalfDecoderState { &self.decoder }

    /// Borrow the destination encoder state.
    pub fn encoder_state(&self) -> &DecoderState { &self.encoder }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbe_params::MbeParams;

    /// Build a valid voice MbeParams that lies inside both pitch tables.
    fn sample_params(omega_0: f32) -> MbeParams {
        let l = MbeParams::harmonic_count_for(omega_0);
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        MbeParams::new(omega_0, l, &voiced, &amps).unwrap()
    }

    fn fullrate_dibits_from(params: &MbeParams, state: &mut DecoderState) -> [u8; 72] {
        let l = params.harmonic_count();
        let b = quantize(params, state).expect("quantize fullrate");
        let u = prioritize_full(&b, l);
        encode_full(&u)
    }

    fn halfrate_dibits_from(
        params: &MbeParams,
        state: &mut HalfDecoderState,
    ) -> [u8; DIBITS_PER_FRAME] {
        let u = quantize_half(params, state).expect("quantize halfrate");
        encode_half(&u)
    }

    fn assert_valid_dibits<const N: usize>(dibits: &[u8; N]) {
        for (i, d) in dibits.iter().enumerate() {
            assert!(*d < 4, "dibit {i} = {d} is out of range");
        }
    }

    #[test]
    fn converters_construct() {
        let _ = FullToHalfConverter::new();
        let _ = HalfToFullConverter::new();
    }

    #[test]
    fn full_to_half_emits_valid_dibits() {
        let mut src_state = DecoderState::new();
        let mut conv = FullToHalfConverter::new();
        let params = sample_params(0.20);
        let input = fullrate_dibits_from(&params, &mut src_state);
        let out = conv.convert(&input).expect("convert");
        assert_valid_dibits(&out);
    }

    #[test]
    fn half_to_full_emits_valid_dibits() {
        let mut src_state = HalfDecoderState::new();
        let mut conv = HalfToFullConverter::new();
        let params = sample_params(0.18);
        let input = halfrate_dibits_from(&params, &mut src_state);
        let out = conv.convert(&input).expect("convert");
        assert_valid_dibits(&out);
    }

    #[test]
    fn cross_rate_round_trip_preserves_pitch_within_quantizer_grid() {
        let mut src_state = DecoderState::new();
        let mut a_to_b = FullToHalfConverter::new();
        let mut b_to_a = HalfToFullConverter::new();
        let mut sink_state = DecoderState::new();

        let params = sample_params(0.20);

        let mut last_omega = 0f32;
        for _ in 0..6 {
            let a = fullrate_dibits_from(&params, &mut src_state);
            let b = a_to_b.convert(&a).expect("A→B");
            let a2 = b_to_a.convert(&b).expect("B→A");

            let frame = decode_full(&a2);
            let back = dequantize(&frame.info, &mut sink_state).expect("decode");
            last_omega = back.omega_0();
        }

        let rel = (last_omega - params.omega_0()).abs() / params.omega_0();
        assert!(rel < 0.05, "ω₀ drift {rel:.4} exceeds 5%");
    }
}
