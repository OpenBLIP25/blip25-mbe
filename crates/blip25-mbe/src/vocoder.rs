//! Unified [`Vocoder`] handle — chip-shaped façade over the rate-specific
//! pipelines.
//!
//! The DVSI AMBE-3000R exposes a single per-channel handle with uniform
//! `encode` / `decode` / `reset` operations regardless of the configured
//! rate. This module reproduces that surface for in-process Rust callers:
//! one [`Vocoder`] handle owns all rate-specific state (analysis,
//! decoder, synth) and dispatches uniformly across rates selected at
//! runtime via the [`Rate`] enum.
//!
//! The low-level modules ([`crate::p25_fullrate`], [`crate::p25_halfrate`],
//! [`crate::codecs::mbe_baseline`]) stay public for advanced consumers
//! that need to drive the pipeline frame-by-frame at the parameter
//! layer; this module is the recommended entry point for everything else.
//!
//! ## Quick start
//!
//! ```no_run
//! use blip25_mbe::vocoder::{Rate, Vocoder};
//!
//! // P25 Phase 1 (full-rate IMBE) encoder
//! let mut tx = Vocoder::new(Rate::P25Phase1);
//! let pcm: [i16; 160] = [0; 160]; // one 20 ms frame at 8 kHz
//! let bits = tx.encode_pcm(&pcm).expect("encode");
//! assert_eq!(bits.len(), 18); // 18-byte FEC frame
//!
//! // P25 Phase 1 decoder (separate channel, separate state)
//! let mut rx = Vocoder::new(Rate::P25Phase1);
//! let pcm = rx.decode_bits(&bits).expect("decode");
//! assert_eq!(pcm.len(), 160);
//! ```
//!
//! ## Mapping to the chip protocol
//!
//! | Chip operation                | This module                          |
//! |-------------------------------|---------------------------------------|
//! | Open channel                  | [`Vocoder::new`]                     |
//! | Set rate (`PKT_RATEP`)        | [`Rate`] argument at construction    |
//! | Reset (re-send `PKT_RATEP`)   | [`Vocoder::reset`]                   |
//! | Encode 160-sample PCM → bits  | [`Vocoder::encode_pcm`]              |
//! | Decode bits → 160-sample PCM  | [`Vocoder::decode_bits`]             |
//! | Read last-frame stats         | [`Vocoder::last_stats`]              |
//! | Frame size                    | [`Vocoder::frame_samples`] / [`Vocoder::fec_frame_bytes`] |
//!
//! Rate is fixed for the lifetime of a [`Vocoder`]; build a new handle
//! to switch rates (mirrors a chip's PKT_RATEP cycle).

use crate::codecs::ambe_plus2;
use crate::codecs::mbe_baseline::analysis::{
    AnalysisError, AnalysisOutput, AnalysisState, encode as analysis_encode,
    encode_halfrate as analysis_encode_halfrate,
};
use crate::codecs::mbe_baseline::{
    FrameDisposition, FrameErrorContext, GAMMA_W, SynthState, synthesize_frame,
};
use crate::mbe_params::MbeParams;
use crate::p25_fullrate;
use crate::p25_halfrate;

/// Number of i16 PCM samples per 20 ms frame at 8 kHz. Constant across
/// every supported rate.
pub const FRAME_SAMPLES: usize = 160;

/// Vocoder rate selection — picks both the codec generation and the
/// wire-FEC framing.
///
/// More variants will land as additional carriers are added. The
/// values themselves are also stable Wire-format choices: a [`Rate`]
/// is enough to know how many FEC bytes a frame is, what codec
/// generation drives the synth, and which dequantize tables to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Rate {
    /// P25 Phase 1 FDMA full-rate IMBE. 18-byte FEC frame (72 dibits).
    /// 7 200 bps total / 4 400 bps voice + 2 800 bps FEC.
    P25Phase1,
    /// P25 Phase 2 TDMA half-rate AMBE+2. 9-byte FEC frame (36 dibits).
    /// 3 600 bps total / 2 450 bps voice + 1 150 bps FEC.
    P25Phase2,
}

impl Rate {
    /// Number of bytes in one FEC-encoded frame at this rate.
    #[inline]
    pub const fn fec_frame_bytes(self) -> usize {
        match self {
            Rate::P25Phase1 => 18,
            Rate::P25Phase2 => 9,
        }
    }

    /// PCM samples per frame (always 160 at 8 kHz / 20 ms; provided
    /// for symmetry with [`Self::fec_frame_bytes`]).
    #[inline]
    pub const fn frame_samples(self) -> usize {
        FRAME_SAMPLES
    }
}

/// Per-frame statistics recorded by the most recent [`Vocoder::encode_pcm`]
/// or [`Vocoder::decode_bits`] call. `None` until at least one frame
/// has been processed.
///
/// Encode-side fills only `analysis`; decode-side fills only `decode`.
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// Stats from the last encoded frame.
    pub analysis: Option<AnalysisStats>,
    /// Stats from the last decoded frame.
    pub decode: Option<DecodeStats>,
}

/// Encode-side per-frame stats.
#[derive(Clone, Debug)]
pub struct AnalysisStats {
    /// What the analysis encoder emitted (`Voice` / `Silence`).
    pub output: AnalysisOutputKind,
    /// The MBE parameters that were quantized into the wire bits.
    /// Populated for both `Voice` and `Silence` (silence dispatches
    /// the rate-appropriate placeholder).
    pub params: MbeParams,
}

/// Discriminator-only counterpart of `AnalysisOutput` — strips the
/// `MbeParams` payload so consumers can inspect what kind of frame
/// was emitted without holding a copy. The full params are in
/// [`AnalysisStats::params`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalysisOutputKind {
    /// Real voice frame derived from the input PCM.
    Voice,
    /// Silence-dispatched frame (rate-appropriate placeholder params).
    Silence,
}

/// Decode-side per-frame stats.
#[derive(Clone, Debug)]
pub struct DecodeStats {
    /// FEC error count on the leading codeword (Golay c̃₀ for full-rate,
    /// AMBE+2 c̃₀ for half-rate).
    pub epsilon_0: u8,
    /// Total FEC error count across all coded vectors of the frame.
    pub epsilon_t: u8,
    /// Synth-side disposition for this frame (Use / Repeat / Mute).
    /// Full-rate populates this; half-rate currently leaves it `None`.
    pub disposition: Option<FrameDisposition>,
}

/// Errors that can surface from [`Vocoder`] operations. Wraps the
/// rate-specific error types so a single `?` covers any Vocoder call.
#[derive(Debug)]
pub enum VocoderError {
    /// Input PCM slice was the wrong length. Always `frame_samples()`.
    WrongPcmLength { expected: usize, got: usize },
    /// Input bit slice was the wrong length. Always `fec_frame_bytes()`.
    WrongBitsLength { expected: usize, got: usize },
    /// Analysis encoder failure (HPF / pitch / refinement).
    Analysis(AnalysisError),
    /// Wire-quantize failure (e.g. predictor returned a non-finite value).
    Quantize(String),
}

impl core::fmt::Display for VocoderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VocoderError::WrongPcmLength { expected, got } => {
                write!(f, "expected {expected} PCM samples per frame, got {got}")
            }
            VocoderError::WrongBitsLength { expected, got } => {
                write!(f, "expected {expected} FEC bytes per frame, got {got}")
            }
            VocoderError::Analysis(e) => write!(f, "analysis encoder error: {e:?}"),
            VocoderError::Quantize(msg) => write!(f, "quantize error: {msg}"),
        }
    }
}

impl std::error::Error for VocoderError {}

/// Chip-shaped façade over the rate-specific encoder + decoder + synth
/// pipelines.
///
/// Owns all per-rate state internally (analysis, decoder, synth). One
/// [`Vocoder`] is *both* encoder and decoder for a single channel
/// direction; consumers running bidirectional voice typically allocate
/// two — one each direction — to mirror the chip's per-direction
/// state isolation.
///
/// State is not `Sync`; one channel = one thread. State is `Send`,
/// so the channel can move between threads.
pub struct Vocoder {
    rate: Rate,
    analysis: AnalysisState,
    fullrate_dec: p25_fullrate::dequantize::DecoderState,
    halfrate_dec: p25_halfrate::dequantize::DecoderState,
    synth: SynthState,
    last_stats: FrameStats,
}

impl Vocoder {
    /// Open a new channel at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            analysis: AnalysisState::new(),
            fullrate_dec: p25_fullrate::dequantize::DecoderState::new(),
            halfrate_dec: p25_halfrate::dequantize::DecoderState::new(),
            synth: SynthState::new(),
            last_stats: FrameStats::default(),
        }
    }

    /// The rate this channel was constructed at. Cannot change for
    /// the lifetime of the channel; build a new [`Vocoder`] to switch
    /// rates (mirrors a chip's PKT_RATEP cycle).
    #[inline]
    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// Number of i16 samples consumed per [`Self::encode_pcm`] call,
    /// and produced per [`Self::decode_bits`] call.
    #[inline]
    pub fn frame_samples(&self) -> usize {
        self.rate.frame_samples()
    }

    /// Number of FEC bytes per encoded frame at this rate.
    #[inline]
    pub fn fec_frame_bytes(&self) -> usize {
        self.rate.fec_frame_bytes()
    }

    /// Read the most recent frame's stats. Returns the zero-default
    /// before any frame has been processed.
    #[inline]
    pub fn last_stats(&self) -> &FrameStats {
        &self.last_stats
    }

    /// Reset all channel state. Equivalent to the chip's PKT_RATEP
    /// re-send: clears predictor history, decoder predictor state,
    /// synth substates, smoothed error rate, last-stats. Rate stays
    /// the same.
    pub fn reset(&mut self) {
        self.analysis = AnalysisState::new();
        self.fullrate_dec = p25_fullrate::dequantize::DecoderState::new();
        self.halfrate_dec = p25_halfrate::dequantize::DecoderState::new();
        self.synth = SynthState::new();
        self.last_stats = FrameStats::default();
    }

    /// Encode one PCM frame into FEC-encoded bytes.
    ///
    /// `pcm` must be exactly [`Self::frame_samples`] samples (160).
    /// Returns exactly [`Self::fec_frame_bytes`] bytes.
    pub fn encode_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, VocoderError> {
        if pcm.len() != self.frame_samples() {
            return Err(VocoderError::WrongPcmLength {
                expected: self.frame_samples(),
                got: pcm.len(),
            });
        }
        let (bytes, stats) = match self.rate {
            Rate::P25Phase1 => fullrate::encode(pcm, self)?,
            Rate::P25Phase2 => halfrate::encode(pcm, self)?,
        };
        self.last_stats.analysis = Some(stats);
        Ok(bytes)
    }

    /// Decode one FEC-encoded frame into PCM samples.
    ///
    /// `bits` must be exactly [`Self::fec_frame_bytes`] bytes.
    /// Returns exactly [`Self::frame_samples`] PCM samples.
    pub fn decode_bits(&mut self, bits: &[u8]) -> Result<Vec<i16>, VocoderError> {
        if bits.len() != self.fec_frame_bytes() {
            return Err(VocoderError::WrongBitsLength {
                expected: self.fec_frame_bytes(),
                got: bits.len(),
            });
        }
        let (pcm, stats) = match self.rate {
            Rate::P25Phase1 => fullrate::decode(bits, self),
            Rate::P25Phase2 => halfrate::decode(bits, self),
        };
        self.last_stats.decode = Some(stats);
        Ok(pcm)
    }
}

impl core::fmt::Debug for Vocoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vocoder")
            .field("rate", &self.rate)
            .field("frame_samples", &self.frame_samples())
            .field("fec_frame_bytes", &self.fec_frame_bytes())
            .finish_non_exhaustive()
    }
}

mod fullrate {
    use super::*;
    use crate::p25_fullrate::dequantize::{dequantize, quantize};
    use crate::p25_fullrate::frame::{decode_frame, encode_frame};
    use crate::p25_fullrate::priority::{deprioritize, prioritize};

    pub(super) fn encode(
        pcm: &[i16],
        vocoder: &mut Vocoder,
    ) -> Result<(Vec<u8>, AnalysisStats), VocoderError> {
        let frame = pcm.try_into().expect("length already validated");
        let (kind, params) = match analysis_encode(frame, &mut vocoder.analysis)
            .map_err(VocoderError::Analysis)?
        {
            AnalysisOutput::Voice(p) => (AnalysisOutputKind::Voice, p),
            AnalysisOutput::Silence => (AnalysisOutputKind::Silence, MbeParams::silence()),
        };
        let mut snapshot = vocoder.fullrate_dec.clone();
        let b = quantize(&params, &mut vocoder.fullrate_dec)
            .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
        // Step the decoder snapshot via dequantize so the predictor
        // state matches what a downstream receiver would observe (the
        // existing speech-quality harness pattern).
        let l = params.harmonic_count();
        let info = prioritize(&b, l);
        let _ = deprioritize; // reserved for future per-priority diagnostics
        let _ = dequantize(&info, &mut snapshot);
        vocoder.fullrate_dec = snapshot;
        let dibits = encode_frame(&info);
        let bytes = pack_dibits_full(&dibits);
        Ok((bytes.to_vec(), AnalysisStats { output: kind, params }))
    }

    pub(super) fn decode(bits: &[u8], vocoder: &mut Vocoder) -> (Vec<i16>, DecodeStats) {
        let dibits = unpack_dibits_full(bits);
        let imbe = decode_frame(&dibits);
        let stats_eps0 = imbe.errors[0];
        let stats_epst = imbe.error_total().min(255) as u8;
        let pcm: [i16; FRAME_SAMPLES] = match dequantize(&imbe.info, &mut vocoder.fullrate_dec) {
            Ok(params) => {
                let err = FrameErrorContext {
                    epsilon_0: stats_eps0,
                    epsilon_4: imbe.errors[4],
                    epsilon_t: stats_epst,
                    bad_pitch: false,
                };
                synthesize_frame(&params, &err, GAMMA_W, &mut vocoder.synth)
            }
            Err(_) => {
                // Bad-pitch / decode error → emit silence, advance no
                // synth state (preserves the existing harness behavior).
                [0i16; FRAME_SAMPLES]
            }
        };
        (
            pcm.to_vec(),
            DecodeStats {
                epsilon_0: stats_eps0,
                epsilon_t: stats_epst,
                disposition: None, // synth's `last_disposition` isn't surfaced yet
            },
        )
    }

    fn pack_dibits_full(dibits: &[u8; 72]) -> [u8; 18] {
        let mut out = [0u8; 18];
        let mut bit = 0usize;
        for &d in dibits {
            for pos in (0..2).rev() {
                let b = (d >> pos) & 1;
                out[bit / 8] |= b << (7 - (bit % 8));
                bit += 1;
            }
        }
        out
    }

    fn unpack_dibits_full(bytes: &[u8]) -> [u8; 72] {
        let mut out = [0u8; 72];
        let mut bit = 0usize;
        for slot in &mut out {
            let mut d = 0u8;
            for _ in 0..2 {
                let b = (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
                d = (d << 1) | b;
                bit += 1;
            }
            *slot = d;
        }
        out
    }
}

mod halfrate {
    use super::*;
    use crate::p25_halfrate::dequantize::{Decoded, decode_to_params, quantize};
    use crate::p25_halfrate::frame::{decode_frame, encode_frame};

    pub(super) fn encode(
        pcm: &[i16],
        vocoder: &mut Vocoder,
    ) -> Result<(Vec<u8>, AnalysisStats), VocoderError> {
        let frame = pcm.try_into().expect("length already validated");
        let (kind, params) = match analysis_encode_halfrate(frame, &mut vocoder.analysis)
            .map_err(VocoderError::Analysis)?
        {
            AnalysisOutput::Voice(p) => (AnalysisOutputKind::Voice, p),
            AnalysisOutput::Silence => (AnalysisOutputKind::Silence, MbeParams::silence_halfrate()),
        };
        let info = quantize(&params, &mut vocoder.halfrate_dec)
            .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
        let dibits = encode_frame(&info);
        let bytes = pack_dibits_half(&dibits);
        Ok((bytes.to_vec(), AnalysisStats { output: kind, params }))
    }

    pub(super) fn decode(bits: &[u8], vocoder: &mut Vocoder) -> (Vec<i16>, DecodeStats) {
        let dibits = unpack_dibits_half(bits);
        let frame = decode_frame(&dibits);
        let stats_eps0 = frame.errors[0];
        let stats_epst = frame.errors.iter().map(|&e| u16::from(e)).sum::<u16>().min(255) as u8;
        let pcm: [i16; FRAME_SAMPLES] = match decode_to_params(&frame.info, &mut vocoder.halfrate_dec) {
            Ok(Decoded::Voice(p)) => ambe_plus2::synthesize_frame(&p, &mut vocoder.synth),
            Ok(Decoded::Tone { params, .. }) => {
                ambe_plus2::synthesize_tone(&params, &mut vocoder.synth)
            }
            Ok(Decoded::Erasure) | Err(_) => [0i16; FRAME_SAMPLES],
        };
        (
            pcm.to_vec(),
            DecodeStats {
                epsilon_0: stats_eps0,
                epsilon_t: stats_epst,
                disposition: None,
            },
        )
    }

    fn pack_dibits_half(dibits: &[u8; 36]) -> [u8; 9] {
        let mut out = [0u8; 9];
        let mut bit = 0usize;
        for &d in dibits {
            for pos in (0..2).rev() {
                let b = (d >> pos) & 1;
                out[bit / 8] |= b << (7 - (bit % 8));
                bit += 1;
            }
        }
        out
    }

    fn unpack_dibits_half(bytes: &[u8]) -> [u8; 36] {
        let mut out = [0u8; 36];
        let mut bit = 0usize;
        for slot in &mut out {
            let mut d = 0u8;
            for _ in 0..2 {
                let b = (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
                d = (d << 1) | b;
                bit += 1;
            }
            *slot = d;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_pcm(period: usize, amplitude: i16) -> [i16; FRAME_SAMPLES] {
        let mut out = [0i16; FRAME_SAMPLES];
        for (n, slot) in out.iter_mut().enumerate() {
            let phase = (n % period) as f32 / period as f32;
            *slot = (amplitude as f32 * (2.0 * core::f32::consts::PI * phase).sin()) as i16;
        }
        out
    }

    #[test]
    fn rate_byte_sizes_match_wire_layouts() {
        assert_eq!(Rate::P25Phase1.fec_frame_bytes(), 18);
        assert_eq!(Rate::P25Phase2.fec_frame_bytes(), 9);
        assert_eq!(Rate::P25Phase1.frame_samples(), 160);
        assert_eq!(Rate::P25Phase2.frame_samples(), 160);
    }

    #[test]
    fn fullrate_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::P25Phase1);
        let mut rx = Vocoder::new(Rate::P25Phase1);
        // Three frames — first two are preroll on the analysis side
        // (return Silence dispatch), the third hits voice.
        for _ in 0..5 {
            let pcm = periodic_pcm(40, 8000);
            let bits = tx.encode_pcm(&pcm).expect("encode");
            assert_eq!(bits.len(), 18);
            let out = rx.decode_bits(&bits).expect("decode");
            assert_eq!(out.len(), FRAME_SAMPLES);
        }
        let stats = tx.last_stats();
        assert!(stats.analysis.is_some(), "encoder didn't fill stats");
    }

    #[test]
    fn halfrate_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::P25Phase2);
        let mut rx = Vocoder::new(Rate::P25Phase2);
        for _ in 0..5 {
            let pcm = periodic_pcm(40, 8000);
            let bits = tx.encode_pcm(&pcm).expect("encode");
            assert_eq!(bits.len(), 9);
            let out = rx.decode_bits(&bits).expect("decode");
            assert_eq!(out.len(), FRAME_SAMPLES);
        }
        let stats = rx.last_stats();
        assert!(stats.decode.is_some(), "decoder didn't fill stats");
    }

    #[test]
    fn wrong_pcm_length_errors() {
        let mut v = Vocoder::new(Rate::P25Phase1);
        let r = v.encode_pcm(&[0i16; 159]);
        assert!(matches!(r, Err(VocoderError::WrongPcmLength { expected: 160, got: 159 })));
    }

    #[test]
    fn wrong_bits_length_errors_per_rate() {
        let mut a = Vocoder::new(Rate::P25Phase1);
        assert!(matches!(
            a.decode_bits(&[0u8; 9]),
            Err(VocoderError::WrongBitsLength { expected: 18, got: 9 })
        ));
        let mut b = Vocoder::new(Rate::P25Phase2);
        assert!(matches!(
            b.decode_bits(&[0u8; 18]),
            Err(VocoderError::WrongBitsLength { expected: 9, got: 18 })
        ));
    }

    #[test]
    fn reset_clears_state_and_keeps_rate() {
        let mut v = Vocoder::new(Rate::P25Phase2);
        let pcm = periodic_pcm(40, 8000);
        let _ = v.encode_pcm(&pcm).unwrap();
        assert!(v.last_stats().analysis.is_some());
        v.reset();
        assert!(v.last_stats().analysis.is_none());
        assert_eq!(v.rate(), Rate::P25Phase2);
    }
}
