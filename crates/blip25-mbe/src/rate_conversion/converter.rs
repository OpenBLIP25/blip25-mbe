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
//! ## Boundary behaviour
//!
//! The full-rate grid (b̂₀ ∈ [0, 207] → ω̃₀ ∈ [4π/246.5, 4π/39.5] ≈
//! [0.05098, 0.31819]) and the half-rate Annex L grid (120 entries,
//! ω̃₀ ∈ [0.05105, 0.31398]) almost but don't quite coincide at the
//! edges — full-rate extends ~0.14% past half-rate's minimum and
//! ~1.3% past its maximum. A half-rate frame at the grid endpoint,
//! round-tripped through full-rate and back, can pick up enough drift
//! at the first hop to fall outside half-rate's grid at the second.
//! We snap the source parameters' ω̃₀ to the destination grid's
//! representable range (keeping L unchanged — the boundary entries
//! share L across rates) before calling the destination quantizer,
//! so boundary frames encode to the nearest grid endpoint instead of
//! rejecting. See `conformance-vectors rate-convert-roundtrip` for
//! the diagnostic that surfaces this case on DVSI speech.
//!
//! ## Frame-kind handling
//!
//! Half-rate defines four pitch-index regions (§13.1 Table 14):
//! voice `[0, 119]`, erasure `[120, 123]`, silence `[124, 125]`, and
//! tone `[126, 127]`. Full-rate IMBE has no tone/silence signaling —
//! any `b̂₀ ∉ [0, 207]` is a frame-repeat trigger at the decoder
//! (§1.11). Converters map between these worlds as follows:
//!
//! | Source kind            | Emitted frame               | Encoder state |
//! |------------------------|-----------------------------|---------------|
//! | Voice                  | Voice at destination rate   | advances      |
//! | Half-rate tone         | Full-rate voice from Eq. 206–209 MBE bridge params (tone rendered as a 1–2 harmonic voiced sinusoid) | advances |
//! | Half-rate silence      | Destination-rate erasure signal | does **not** advance |
//! | Half-rate erasure      | Destination-rate erasure signal | does **not** advance |
//! | Full-rate erasure/reserved | Half-rate erasure (b̂₀=120) | does **not** advance |
//!
//! The "does not advance" rule matches §1.11 / §2.8: repeat and mute
//! preserve prior-frame parameters in the receiver, and the encoder
//! has nothing new to quantize, so its predictor state must not
//! drift.
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

use core::f32::consts::PI;

use crate::mbe_params::MbeParams;
use crate::p25_fullrate::dequantize::{
    DecodeError, DecoderState, EncodeError, PITCH_INDEX_MAX, dequantize, quantize,
};
use crate::p25_fullrate::frame::{decode_frame as decode_full, encode_frame as encode_full};
use crate::p25_fullrate::priority::{
    IMBE_B_MAX, L_MIN as FULL_L_MIN, prioritize as prioritize_full,
};
use crate::p25_halfrate::dequantize::{
    Decoded, DecoderState as HalfDecoderState, decode_to_params, quantize as quantize_half,
};
use crate::p25_halfrate::frame::{
    AMBE_PITCH_TABLE, DIBITS_PER_FRAME, decode_frame as decode_half, encode_frame as encode_half,
};
use crate::p25_halfrate::priority::{AMBE_B_COUNT, prioritize as prioritize_half};

/// Lowest ω̃₀ in the half-rate Annex L table (entry 119, L = 56).
/// Used to clamp source parameters to the half-rate grid's range
/// before handing them to the half-rate quantizer.
fn halfrate_omega_min() -> f32 {
    AMBE_PITCH_TABLE[AMBE_PITCH_TABLE.len() - 1].omega_0
}

/// Highest ω̃₀ in the half-rate Annex L table (entry 0, L = 9).
fn halfrate_omega_max() -> f32 {
    AMBE_PITCH_TABLE[0].omega_0
}

/// Lowest ω̃₀ emittable by the full-rate encoder (b̂₀ = 207): `4π/246.5`.
fn fullrate_omega_min() -> f32 {
    4.0 * PI / (f32::from(PITCH_INDEX_MAX) + 39.5)
}

/// Highest ω̃₀ emittable by the full-rate encoder (b̂₀ = 0): `4π/39.5`.
fn fullrate_omega_max() -> f32 {
    4.0 * PI / 39.5
}

/// If `params.omega_0()` falls outside `[min, max]`, return a copy with
/// ω̃₀ snapped to the nearest boundary — L, voicing, and amplitudes
/// unchanged. Returns the input cloned otherwise. Cheap; avoids an
/// allocation path on the common in-range case.
fn clamp_omega_to(params: &MbeParams, min: f32, max: f32) -> MbeParams {
    let w = params.omega_0();
    if w >= min && w <= max {
        return params.clone();
    }
    let clamped = w.clamp(min, max);
    MbeParams::new(
        clamped,
        params.harmonic_count(),
        params.voiced_slice(),
        params.amplitudes_slice(),
    )
    .expect("clamped ω₀ stays inside (0, π)")
}

/// First `b̂₀` value in the half-rate erasure range (§13.1 Table 14,
/// `[120, 123]`). Emitted by converters when the source-rate frame
/// signals an erasure / silence / reserved pitch condition.
const HALFRATE_ERASURE_B0: u16 = 120;

/// First `b̂₀` value in the full-rate reserved range (§6.1,
/// `[208, 255]`). The full-rate decoder treats any `b̂₀ ∉ [0, 207]`
/// as a frame-repeat trigger (§1.11), so a single sentinel is
/// sufficient.
const FULLRATE_ERASURE_B0: u16 = PITCH_INDEX_MAX as u16 + 1;

/// Emit a half-rate erasure frame: 36 dibits whose deprioritized
/// `b̂₀ = 120` (other parameters zero). After the channel codec pass
/// these will round-trip through `classify_halfrate_frame` as
/// [`crate::p25_halfrate::dequantize::FrameKind::Erasure`].
fn halfrate_erasure_dibits() -> [u8; DIBITS_PER_FRAME] {
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = HALFRATE_ERASURE_B0;
    let u = prioritize_half(&b);
    encode_half(&u)
}

/// Emit a full-rate erasure frame: 72 dibits whose deprioritized
/// `b̂₀ = 208` (just past `PITCH_INDEX_MAX`). The full-rate decoder's
/// §1.11 repeat logic will substitute the previous voice frame.
fn fullrate_erasure_dibits() -> [u8; 72] {
    let mut b = [0u16; IMBE_B_MAX];
    b[0] = FULLRATE_ERASURE_B0;
    // Prioritize uses the L-indexed map; at L=L_MIN only the fixed
    // b̂₀/b̂₁ positions receive bits (the rest of b̂ is zero), which is
    // what we want — the decoder rejects on pitch before the rest
    // matters.
    let u = prioritize_full(&b, FULL_L_MIN);
    encode_full(&u)
}

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
    /// frame (36 dibits). On a full-rate erasure / reserved pitch
    /// (`b̂₀ ∈ [208, 255]`) the output is a half-rate erasure signal
    /// and encoder state is preserved (§2.8 frame-repeat semantics).
    pub fn convert(&mut self, dibits: &[u8; 72]) -> Result<[u8; DIBITS_PER_FRAME], ConvertError> {
        let frame = decode_full(dibits);
        let params = match dequantize(&frame.info, &mut self.decoder) {
            Ok(p) => p,
            Err(DecodeError::BadPitch) => return Ok(halfrate_erasure_dibits()),
            Err(other) => return Err(ConvertError::Decode(other)),
        };
        let clamped = clamp_omega_to(&params, halfrate_omega_min(), halfrate_omega_max());
        let u = quantize_half(&clamped, &mut self.encoder).map_err(|e| match e {
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
    /// frame (72 dibits). Handles all four half-rate frame kinds:
    ///
    /// - Voice → full-rate voice (normal requantize path).
    /// - Tone → full-rate voice encoded from the Eq. 206–209 MBE
    ///   bridge parameters. Renders a tone as a one- or two-harmonic
    ///   voiced sinusoid; the full-rate receiver synthesizes it as
    ///   voice.
    /// - Silence / Erasure → full-rate erasure signal. Encoder state
    ///   is preserved so the downstream receiver's §1.11 repeat logic
    ///   stays consistent.
    pub fn convert(&mut self, dibits: &[u8; DIBITS_PER_FRAME]) -> Result<[u8; 72], ConvertError> {
        let frame = decode_half(dibits);
        let params = match decode_to_params(&frame.info, &mut self.decoder) {
            Ok(Decoded::Voice(p)) => p,
            Ok(Decoded::Tone { params, .. }) => params,
            Ok(Decoded::Erasure) => return Ok(fullrate_erasure_dibits()),
            Err(_) => return Ok(fullrate_erasure_dibits()),
        };
        let clamped = clamp_omega_to(&params, fullrate_omega_min(), fullrate_omega_max());
        let l = clamped.harmonic_count();
        let b = quantize(&clamped, &mut self.encoder)?;
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

    #[test]
    fn half_grid_minimum_round_trips_without_rejection() {
        // Regression: before the boundary clamp, a half-rate frame at
        // the Annex L minimum (b̂₀=119, ω₀=0.051051, L=56) round-tripped
        // through full-rate → full-rate decoded it as b̂₀=207 →
        // ω₀=0.050979, which was just below the half-rate grid
        // minimum, so FullToHalfConverter rejected with
        // HalfPitchOutOfRange. The clamp snaps ω₀ to the half-rate
        // grid edge before quantizing.
        let mut src_state = HalfDecoderState::new();
        let l = 56u8;
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        let params =
            MbeParams::new(halfrate_omega_min(), l, &voiced, &amps).unwrap();

        let mut a_to_b = HalfToFullConverter::new();
        let mut b_to_a = FullToHalfConverter::new();
        for _ in 0..4 {
            let a = halfrate_dibits_from(&params, &mut src_state);
            let b = a_to_b.convert(&a).expect("half → full");
            let _ = b_to_a.convert(&b).expect("full → half (was failing)");
        }
    }

    #[test]
    fn full_grid_maximum_round_trips_without_rejection() {
        // Symmetric: full-rate b̂₀=0 → ω₀=0.318194 is above the
        // half-rate grid maximum (0.313977). FullToHalfConverter
        // should clamp down to the half-rate max rather than reject.
        let mut src_state = DecoderState::new();
        let params = MbeParams::silence(); // ω₀ = 4π/39.5 = full-rate max
        let mut conv = FullToHalfConverter::new();
        for _ in 0..4 {
            let a = fullrate_dibits_from(&params, &mut src_state);
            let _ = conv.convert(&a).expect("full → half at grid top");
        }
    }

    #[test]
    fn clamp_is_no_op_for_in_range_omega() {
        let params = sample_params(0.20);
        let clamped = clamp_omega_to(&params, 0.1, 0.3);
        assert_eq!(clamped.omega_0(), params.omega_0());
        assert_eq!(clamped.harmonic_count(), params.harmonic_count());
    }

    #[test]
    fn clamp_snaps_to_the_nearest_boundary() {
        let below = sample_params(0.10);
        let snapped = clamp_omega_to(&below, 0.15, 0.25);
        assert!((snapped.omega_0() - 0.15).abs() < 1e-6);
        assert_eq!(snapped.harmonic_count(), below.harmonic_count());

        let above = sample_params(0.30);
        let snapped = clamp_omega_to(&above, 0.15, 0.25);
        assert!((snapped.omega_0() - 0.25).abs() < 1e-6);
    }

    // ---- Tone / silence / erasure handling -----------------------------

    use crate::p25_halfrate::dequantize::{FrameKind, classify_halfrate_frame};

    /// Build a half-rate erasure frame's 36 dibits by deliberately
    /// prioritizing `b̂₀ = 120` with everything else zero — same path
    /// the converter uses to emit erasure on its output side.
    fn halfrate_erasure_input() -> [u8; DIBITS_PER_FRAME] {
        halfrate_erasure_dibits()
    }

    /// Build a half-rate tone frame's 36 dibits for tone ID 5 (first
    /// valid Annex T entry, 156.25 Hz with l₁=l₂=1). Tone frames use
    /// a different bit layout from voice (§2.10.1 Table 20), so we
    /// set the signature bits directly rather than going through the
    /// voice-path prioritizer.
    fn halfrate_tone_input() -> [u8; DIBITS_PER_FRAME] {
        // Signature: û₀(11..6) = 0x3F, û₃(3..0) = 0.
        // Tone ID copy 4 in û₃(12..5) = 5 (the primary ID the
        // parser reads).
        let mut u = [0u16; 4];
        u[0] = 0x3F << 6; // signature
        u[3] = 5u16 << 5; // tone ID = 5 in copy 4 position
        encode_half(&u)
    }

    /// Build a full-rate "erasure" frame by prioritizing `b̂₀ = 208`
    /// into û and encoding. The decoder will reject with
    /// `DecodeError::BadPitch`, which the converter maps to a
    /// half-rate erasure on its output.
    fn fullrate_erasure_input() -> [u8; 72] {
        fullrate_erasure_dibits()
    }

    #[test]
    fn halfrate_erasure_dibits_round_trip_as_erasure() {
        // Self-check: the dibits we emit must classify as Erasure
        // after the channel codec pass (otherwise receivers would
        // interpret the converter's output as voice).
        let dibits = halfrate_erasure_dibits();
        let frame = decode_half(&dibits);
        assert_eq!(classify_halfrate_frame(&frame.info), FrameKind::Erasure);
    }

    #[test]
    fn fullrate_erasure_dibits_round_trip_as_bad_pitch() {
        // Self-check: the dibits we emit must decode to a reserved
        // `b̂₀`, which the full-rate dequantizer rejects as BadPitch —
        // the §1.11 frame-repeat trigger at the receiver.
        let dibits = fullrate_erasure_dibits();
        let frame = decode_full(&dibits);
        let mut state = DecoderState::new();
        match dequantize(&frame.info, &mut state) {
            Err(DecodeError::BadPitch) => {}
            other => panic!("expected BadPitch, got {other:?}"),
        }
    }

    #[test]
    fn full_to_half_emits_erasure_on_reserved_pitch() {
        // Full-rate input signaling b̂₀ = 208 → half-rate output must
        // classify as Erasure at the receiver (not Voice, which would
        // play garbage, or Tone, which would mis-interpret the frame).
        let mut conv = FullToHalfConverter::new();
        let input = fullrate_erasure_input();
        let out = conv.convert(&input).expect("convert");
        let frame = decode_half(&out);
        assert_eq!(classify_halfrate_frame(&frame.info), FrameKind::Erasure);
    }

    #[test]
    fn half_to_full_emits_erasure_on_erasure_input() {
        // Half-rate erasure input → full-rate erasure output that
        // triggers frame repeat at the receiver.
        let mut conv = HalfToFullConverter::new();
        let input = halfrate_erasure_input();
        let out = conv.convert(&input).expect("convert");
        let frame = decode_full(&out);
        let mut sink = DecoderState::new();
        match dequantize(&frame.info, &mut sink) {
            Err(DecodeError::BadPitch) => {}
            other => panic!("expected BadPitch on re-decode, got {other:?}"),
        }
    }

    #[test]
    fn half_to_full_encodes_tone_as_voice() {
        // Tone ID 1 → full-rate voice frame. The far-side receiver
        // will synthesize it as a 1-2-harmonic voiced sinusoid (not
        // exact to DVSI's tone generator, but audible and continuous
        // rather than a dropout).
        let mut conv = HalfToFullConverter::new();
        let input = halfrate_tone_input();
        let out = conv.convert(&input).expect("convert tone");
        let frame = decode_full(&out);
        let mut sink = DecoderState::new();
        let params = dequantize(&frame.info, &mut sink).expect("tone → voice");
        assert!(params.harmonic_count() >= 9, "tone yielded L={}", params.harmonic_count());
    }

    #[test]
    fn half_to_full_erasure_is_idempotent() {
        // Per §2.8 frame-repeat semantics the encoder state must not
        // advance on erasure. Observe externally: two consecutive
        // erasure inputs produce bit-identical output frames. A state
        // advance would produce different predictor residuals even
        // with identical inputs.
        let mut conv = HalfToFullConverter::new();
        let a = conv.convert(&halfrate_erasure_input()).unwrap();
        let b = conv.convert(&halfrate_erasure_input()).unwrap();
        assert_eq!(a, b, "erasure output should be deterministic");
    }

    #[test]
    fn full_to_half_erasure_is_idempotent() {
        let mut conv = FullToHalfConverter::new();
        let a = conv.convert(&fullrate_erasure_input()).unwrap();
        let b = conv.convert(&fullrate_erasure_input()).unwrap();
        assert_eq!(a, b, "erasure output should be deterministic");
    }
}
