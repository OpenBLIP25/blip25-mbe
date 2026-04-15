//! Same-rate repeater — parse, optionally clean up via FEC, repack at
//! the same rate.
//!
//! Equivalent to [`super::converter`] with matching input and output
//! rates. Its purpose is FEC error correction and retransmission
//! without a rate change. When the input is already clean the output
//! reproduces the same bits (modulo any quantizer/predictor drift,
//! which the encoder is designed to keep below quantizer precision).
//!
//! Two states are held — one for the upstream decoder, one for the
//! downstream encoder — for the same reason described in
//! [`super::converter`]: the upstream and downstream peers each have
//! their own state evolution that the repeater must mirror.
//!
//! Corresponds to the AMBE-3000 `PKT_RPT_MODE = 0x01` configuration
//! with identical decoder and encoder rate parameters.

use crate::p25_fullrate::dequantize::{DecoderState, dequantize, quantize};
use crate::p25_fullrate::frame::{decode_frame as decode_full, encode_frame as encode_full};
use crate::p25_fullrate::priority::prioritize as prioritize_full;
use crate::p25_halfrate::dequantize::{
    DecoderState as HalfDecoderState, dequantize as dequantize_half, quantize as quantize_half,
};
use crate::p25_halfrate::frame::{
    DIBITS_PER_FRAME, decode_frame as decode_half, encode_frame as encode_half,
};

use super::converter::ConvertError;

// ---------------------------------------------------------------------------
// P25 Phase 1 full-rate (144-bit IMBE) repeater
// ---------------------------------------------------------------------------

/// Full-rate IMBE repeater.
#[derive(Clone, Debug, Default)]
pub struct FullRateRepeater {
    decoder: DecoderState,
    encoder: DecoderState,
}

impl FullRateRepeater {
    /// Cold-start repeater.
    pub fn new() -> Self { Self::default() }

    /// Decode + re-encode one full-rate frame.
    pub fn repeat(&mut self, dibits: &[u8; 72]) -> Result<[u8; 72], ConvertError> {
        let frame = decode_full(dibits);
        let params = dequantize(&frame.info, &mut self.decoder)?;
        let l = params.harmonic_count();
        let b = quantize(&params, &mut self.encoder)?;
        let u = prioritize_full(&b, l);
        Ok(encode_full(&u))
    }
}

// ---------------------------------------------------------------------------
// P25 Phase 2 half-rate (72-bit AMBE+2) repeater
// ---------------------------------------------------------------------------

/// Half-rate AMBE+2 repeater.
#[derive(Clone, Debug, Default)]
pub struct HalfRateRepeater {
    decoder: HalfDecoderState,
    encoder: HalfDecoderState,
}

impl HalfRateRepeater {
    /// Cold-start repeater.
    pub fn new() -> Self { Self::default() }

    /// Decode + re-encode one half-rate frame.
    pub fn repeat(&mut self, dibits: &[u8; DIBITS_PER_FRAME]) -> Result<[u8; DIBITS_PER_FRAME], ConvertError> {
        let frame = decode_half(dibits);
        let params = dequantize_half(&frame.info, &mut self.decoder)?;
        let u = quantize_half(&params, &mut self.encoder).map_err(ConvertError::Decode)?;
        Ok(encode_half(&u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbe_params::MbeParams;

    fn fullrate_voice_params() -> MbeParams {
        let omega_0 = 0.20f32;
        let l = MbeParams::harmonic_count_for(omega_0);
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        MbeParams::new(omega_0, l, &voiced, &amps).unwrap()
    }

    #[test]
    fn full_rate_repeater_produces_valid_dibits() {
        let mut src = DecoderState::new();
        let params = fullrate_voice_params();
        let l = params.harmonic_count();
        let b = quantize(&params, &mut src).unwrap();
        let u = prioritize_full(&b, l);
        let input = encode_full(&u);

        let out = FullRateRepeater::new().repeat(&input).expect("repeat");
        for (i, d) in out.iter().enumerate() {
            assert!(*d < 4, "dibit {i} not in range: {d}");
        }
    }

    #[test]
    fn half_rate_repeater_produces_valid_dibits() {
        let mut src = HalfDecoderState::new();
        let omega_0 = 0.18f32;
        let l = MbeParams::harmonic_count_for(omega_0);
        let voiced: Vec<bool> = (1..=l).map(|h| h <= l / 2).collect();
        let amps: Vec<f32> = (1..=l)
            .map(|h| 100.0 * (-(h as f32) * 0.05).exp())
            .collect();
        let params = MbeParams::new(omega_0, l, &voiced, &amps).unwrap();
        let u = quantize_half(&params, &mut src).unwrap();
        let input = encode_half(&u);

        let out = HalfRateRepeater::new().repeat(&input).expect("repeat");
        for (i, d) in out.iter().enumerate() {
            assert!(*d < 4, "dibit {i} not in range: {d}");
        }
    }

    #[test]
    fn full_rate_repeater_eventually_idempotent() {
        let mut src = DecoderState::new();
        let params = fullrate_voice_params();
        let l = params.harmonic_count();

        let mut frames: Vec<[u8; 72]> = Vec::new();
        for _ in 0..30 {
            let b = quantize(&params, &mut src).unwrap();
            let u = prioritize_full(&b, l);
            frames.push(encode_full(&u));
        }

        let mut rep = FullRateRepeater::new();
        let mut last_pair: Option<([u8; 72], [u8; 72])> = None;
        for frame in &frames {
            let out = rep.repeat(frame).unwrap();
            for (i, d) in out.iter().enumerate() {
                assert!(*d < 4, "dibit {i} not in range: {d}");
            }
            last_pair = Some((*frame, out));
        }

        let (_in_last, out_last) = last_pair.unwrap();
        let echoed = decode_full(&out_last);
        let mut sink = DecoderState::new();
        for frame in &frames[..frames.len() - 1] {
            let f = decode_full(frame);
            let _ = dequantize(&f.info, &mut sink);
        }
        let echoed_params = dequantize(&echoed.info, &mut sink).expect("decode");
        assert_eq!(echoed_params.harmonic_count(), l);
    }
}
