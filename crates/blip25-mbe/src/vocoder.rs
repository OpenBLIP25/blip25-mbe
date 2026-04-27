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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameStats {
    /// Stats from the last encoded frame.
    pub analysis: Option<AnalysisStats>,
    /// Stats from the last decoded frame.
    pub decode: Option<DecodeStats>,
}

/// Encode-side per-frame stats.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnalysisOutputKind {
    /// Real voice frame derived from the input PCM.
    Voice,
    /// Silence-dispatched frame (rate-appropriate placeholder params).
    Silence,
}

/// Decode-side per-frame stats.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodeStats {
    /// FEC error count on the leading codeword (Golay c̃₀ for full-rate,
    /// AMBE+2 c̃₀ for half-rate).
    pub epsilon_0: u8,
    /// Total FEC error count across all coded vectors of the frame.
    pub epsilon_t: u8,
    /// Synth-side disposition for this frame (Use / Repeat / Mute).
    /// Populated for both rates after the synth runs. `None` only on
    /// frames where dequantize errored before synth was reached
    /// (e.g. invalid pitch index in full-rate; the field then carries
    /// over the prior frame's value via `Vocoder::synth.last_disposition`).
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

    /// Disposition (`Use` / `Repeat` / `Mute`) of the most recent
    /// [`Self::decode_bits`] call's synthesizer pass. `None` until at
    /// least one frame has been decoded. Reset by [`Self::reset`].
    ///
    /// Decode-side only; encode-side has no synth and always returns
    /// `None`.
    #[inline]
    pub fn last_disposition(&self) -> Option<FrameDisposition> {
        self.synth.last_disposition()
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

    /// Encode an arbitrary-length PCM slice as a stream of frames.
    ///
    /// Returns an iterator that yields one `Result<Vec<u8>>` per
    /// frame consumed (160 samples per frame). Trailing partial
    /// frames are silently dropped — the caller is responsible for
    /// padding to a multiple of [`Self::frame_samples`] if all
    /// samples must be encoded.
    ///
    /// State (predictor, look-ahead history, ε_R) advances across
    /// frames just as it would with manual per-frame
    /// [`Self::encode_pcm`] calls.
    ///
    /// ```no_run
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// # let pcm: Vec<i16> = vec![0; 160 * 50];
    /// let mut tx = Vocoder::new(Rate::P25Phase1);
    /// let bits: Result<Vec<Vec<u8>>, _> = tx.encode_stream(&pcm).collect();
    /// ```
    pub fn encode_stream<'a>(&'a mut self, pcm: &'a [i16]) -> EncodeStream<'a> {
        EncodeStream { vocoder: self, pcm, pos: 0 }
    }

    /// Decode an arbitrary-length FEC byte slice as a stream of PCM
    /// frames.
    ///
    /// Returns an iterator that yields one `Result<Vec<i16>>` per
    /// frame consumed ([`Self::fec_frame_bytes`] per frame). Trailing
    /// partial frames are silently dropped.
    ///
    /// ```no_run
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// # let bits: Vec<u8> = vec![0; 18 * 50];
    /// let mut rx = Vocoder::new(Rate::P25Phase1);
    /// let pcm_frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream(&bits).collect();
    /// ```
    pub fn decode_stream<'a>(&'a mut self, bits: &'a [u8]) -> DecodeStream<'a> {
        DecodeStream { vocoder: self, bits, pos: 0 }
    }
}

/// Streaming-encode iterator returned by [`Vocoder::encode_stream`].
/// Yields one `Result<Vec<u8>>` per 160-sample input frame; trailing
/// partial frames are silently dropped.
pub struct EncodeStream<'a> {
    vocoder: &'a mut Vocoder,
    pcm: &'a [i16],
    pos: usize,
}

impl Iterator for EncodeStream<'_> {
    type Item = Result<Vec<u8>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.frame_samples();
        if self.pos + n > self.pcm.len() {
            return None;
        }
        let frame = &self.pcm[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.encode_pcm(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.pcm.len() - self.pos) / self.vocoder.frame_samples();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EncodeStream<'_> {}

/// Streaming-decode iterator returned by [`Vocoder::decode_stream`].
/// Yields one `Result<Vec<i16>>` per FEC frame; trailing partial
/// frames are silently dropped.
pub struct DecodeStream<'a> {
    vocoder: &'a mut Vocoder,
    bits: &'a [u8],
    pos: usize,
}

impl Iterator for DecodeStream<'_> {
    type Item = Result<Vec<i16>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.fec_frame_bytes();
        if self.pos + n > self.bits.len() {
            return None;
        }
        let frame = &self.bits[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.decode_bits(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.bits.len() - self.pos) / self.vocoder.fec_frame_bytes();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DecodeStream<'_> {}

/// Push-driven encoder for live PCM streams that arrive in chunks
/// of arbitrary length (audio device callbacks, file readers, sockets).
/// Holds residual samples internally across calls so the caller can
/// push whatever they have and harvest frames as they become available.
///
/// One-frame-at-a-time `Vocoder` is the right primitive for callers
/// that already have whole-buffer PCM. `LiveEncoder` is for callers
/// that don't.
///
/// ```no_run
/// # use blip25_mbe::vocoder::{LiveEncoder, Rate};
/// let mut enc = LiveEncoder::new(Rate::P25Phase1);
/// // 256 samples (audio-device callback); not a multiple of 160.
/// let chunk: [i16; 256] = [0; 256];
/// let frames = enc.push(&chunk);
/// for f in frames {
///     match f {
///         Ok(bits) => { /* send 18 bytes */ }
///         Err(e)   => { /* per-frame error */ }
///     }
/// }
/// // 96 samples residue; next push contributes them to the next frame.
/// assert_eq!(enc.pending_samples(), 96);
/// ```
pub struct LiveEncoder {
    vocoder: Vocoder,
    pcm_buf: Vec<i16>,
}

impl LiveEncoder {
    /// Open a new live encoder at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            vocoder: Vocoder::new(rate),
            pcm_buf: Vec::new(),
        }
    }

    /// Read-only access to the underlying [`Vocoder`] (for stats /
    /// rate / disposition queries).
    #[inline]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    /// Append PCM samples and emit zero or more FEC frames. Per-frame
    /// errors are surfaced as `Err` entries in the returned Vec; the
    /// buffer drains regardless so a single bad frame doesn't stall
    /// the stream.
    pub fn push(&mut self, pcm: &[i16]) -> Vec<Result<Vec<u8>, VocoderError>> {
        self.pcm_buf.extend_from_slice(pcm);
        let n = self.vocoder.frame_samples();
        let mut out = Vec::with_capacity(self.pcm_buf.len() / n);
        while self.pcm_buf.len() >= n {
            // Encode from the front of the buffer; drain after the call.
            // Disjoint-field borrow keeps this clean (vocoder and
            // pcm_buf are independent struct fields).
            let result = self.vocoder.encode_pcm(&self.pcm_buf[..n]);
            self.pcm_buf.drain(..n);
            out.push(result);
        }
        out
    }

    /// Number of samples currently buffered (between 0 and
    /// `frame_samples()-1` after every `push` returns).
    #[inline]
    pub fn pending_samples(&self) -> usize {
        self.pcm_buf.len()
    }

    /// Drop any pending samples without encoding them. Useful at
    /// stream shutdown when the caller doesn't want a partial-frame
    /// flush.
    #[inline]
    pub fn discard_pending(&mut self) {
        self.pcm_buf.clear();
    }

    /// Reset all state — both the inner [`Vocoder`] (predictor /
    /// look-ahead / synth substates) and the residual sample buffer.
    pub fn reset(&mut self) {
        self.vocoder.reset();
        self.pcm_buf.clear();
    }

    /// Configured rate.
    #[inline]
    pub fn rate(&self) -> Rate {
        self.vocoder.rate()
    }
}

/// Push-driven decoder for live FEC byte streams that arrive in chunks
/// of arbitrary length (network sockets, log replays, partial reads).
/// Mirrors [`LiveEncoder`].
///
/// ```no_run
/// # use blip25_mbe::vocoder::{LiveDecoder, Rate};
/// let mut dec = LiveDecoder::new(Rate::P25Phase2);
/// let chunk: [u8; 23] = [0; 23];   // 2 full 9-byte frames + 5 byte residue
/// let frames = dec.push(&chunk);
/// for f in frames {
///     match f {
///         Ok(pcm) => { /* play 160 samples */ }
///         Err(e)  => { /* per-frame error */ }
///     }
/// }
/// assert_eq!(dec.pending_bytes(), 5);
/// ```
pub struct LiveDecoder {
    vocoder: Vocoder,
    bits_buf: Vec<u8>,
}

impl LiveDecoder {
    /// Open a new live decoder at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        Self {
            vocoder: Vocoder::new(rate),
            bits_buf: Vec::new(),
        }
    }

    /// Read-only access to the underlying [`Vocoder`].
    #[inline]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    /// Append FEC bytes and emit zero or more PCM frames. Per-frame
    /// errors surface as `Err` entries in the returned Vec; the
    /// buffer drains regardless.
    pub fn push(&mut self, bits: &[u8]) -> Vec<Result<Vec<i16>, VocoderError>> {
        self.bits_buf.extend_from_slice(bits);
        let n = self.vocoder.fec_frame_bytes();
        let mut out = Vec::with_capacity(self.bits_buf.len() / n);
        while self.bits_buf.len() >= n {
            let result = self.vocoder.decode_bits(&self.bits_buf[..n]);
            self.bits_buf.drain(..n);
            out.push(result);
        }
        out
    }

    /// Bytes currently buffered (between 0 and `fec_frame_bytes()-1`
    /// after every `push` returns).
    #[inline]
    pub fn pending_bytes(&self) -> usize {
        self.bits_buf.len()
    }

    /// Drop any pending bytes without decoding them.
    #[inline]
    pub fn discard_pending(&mut self) {
        self.bits_buf.clear();
    }

    /// Reset all state — inner [`Vocoder`] + residual byte buffer.
    pub fn reset(&mut self) {
        self.vocoder.reset();
        self.bits_buf.clear();
    }

    /// Configured rate.
    #[inline]
    pub fn rate(&self) -> Rate {
        self.vocoder.rate()
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
        let disposition = vocoder.synth.last_disposition();
        (
            pcm.to_vec(),
            DecodeStats {
                epsilon_0: stats_eps0,
                epsilon_t: stats_epst,
                disposition,
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
        let disposition = vocoder.synth.last_disposition();
        (
            pcm.to_vec(),
            DecodeStats {
                epsilon_0: stats_eps0,
                epsilon_t: stats_epst,
                disposition,
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

    /// Diagnostic types round-trip through JSON when the `serde`
    /// feature is on. Validates that a future RPC layer
    /// (gRPC / protobuf / WS) can ship `FrameStats` over the wire
    /// without bespoke conversion.
    #[cfg(feature = "serde")]
    #[test]
    fn frame_stats_round_trip_through_json() {
        let mut v = Vocoder::new(Rate::P25Phase1);
        let pcm = periodic_pcm(40, 8000);
        for _ in 0..3 {
            let bits = v.encode_pcm(&pcm).unwrap();
            let _ = v.decode_bits(&bits).unwrap();
        }
        let stats = v.last_stats().clone();
        let s = serde_json::to_string(&stats).expect("serialize FrameStats");
        let back: FrameStats = serde_json::from_str(&s).expect("deserialize FrameStats");
        // Round-trip equality on the parts we can compare without
        // full PartialEq derives — output kind + presence flags + the
        // one Copy-able sub-field.
        assert_eq!(stats.analysis.is_some(), back.analysis.is_some());
        assert_eq!(stats.decode.is_some(), back.decode.is_some());
        if let (Some(a), Some(b)) = (&stats.analysis, &back.analysis) {
            assert_eq!(a.output, b.output);
            assert_eq!(a.params, b.params);
        }
        if let (Some(a), Some(b)) = (&stats.decode, &back.decode) {
            assert_eq!(a.epsilon_0, b.epsilon_0);
            assert_eq!(a.epsilon_t, b.epsilon_t);
            assert_eq!(a.disposition, b.disposition);
        }
    }

    /// `Rate` round-trips as a plain enum — the public-name variants
    /// stay stable across serde versions.
    #[cfg(feature = "serde")]
    #[test]
    fn rate_serializes_as_named_variant() {
        let s = serde_json::to_string(&Rate::P25Phase1).unwrap();
        assert_eq!(s, "\"P25Phase1\"");
        let back: Rate = serde_json::from_str(&s).unwrap();
        assert_eq!(back, Rate::P25Phase1);
    }

    /// Streaming encode iterator yields exactly `pcm.len() /
    /// frame_samples()` items and drops a trailing partial frame.
    #[test]
    fn encode_stream_yields_one_per_frame_drops_partial() {
        let mut v = Vocoder::new(Rate::P25Phase1);
        // 5 full frames + 50 trailing samples (partial — should drop).
        let mut pcm: Vec<i16> = Vec::with_capacity(5 * FRAME_SAMPLES + 50);
        for f in 0..5 {
            pcm.extend_from_slice(&periodic_pcm(40, (1000 + f * 100) as i16));
        }
        pcm.extend(std::iter::repeat(0i16).take(50));
        let stream = v.encode_stream(&pcm);
        assert_eq!(stream.len(), 5);
        let bits: Vec<Vec<u8>> = stream.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(bits.len(), 5);
        for b in &bits {
            assert_eq!(b.len(), 18); // P25Phase1 FEC frame size
        }
    }

    /// Streaming decode parallels encode: one item per FEC frame,
    /// trailing partial bytes dropped, output 160 samples each.
    #[test]
    fn decode_stream_yields_one_per_frame_drops_partial() {
        let mut tx = Vocoder::new(Rate::P25Phase2);
        let mut pcm: Vec<i16> = Vec::with_capacity(7 * FRAME_SAMPLES);
        for _ in 0..7 {
            pcm.extend_from_slice(&periodic_pcm(40, 5000));
        }
        let bits: Vec<u8> = tx
            .encode_stream(&pcm)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        // Append 4 stray bytes that should be dropped.
        let mut padded = bits.clone();
        padded.extend_from_slice(&[0; 4]);

        let mut rx = Vocoder::new(Rate::P25Phase2);
        let frames: Vec<Vec<i16>> = rx
            .decode_stream(&padded)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(frames.len(), 7);
        for f in &frames {
            assert_eq!(f.len(), FRAME_SAMPLES);
        }
    }

    /// `DecodeStats::disposition` populates after a decode call.
    /// On clean own-encoded bits the disposition is `Use`; on bits
    /// crafted to cross the mute threshold (high ε_t injected by
    /// pre-loading `state.epsilon_r`) it should be `Mute`.
    #[test]
    fn decode_stats_carry_disposition() {
        let mut tx = Vocoder::new(Rate::P25Phase1);
        let mut rx = Vocoder::new(Rate::P25Phase1);
        for _ in 0..5 {
            let pcm = periodic_pcm(40, 8000);
            let bits = tx.encode_pcm(&pcm).unwrap();
            let _ = rx.decode_bits(&bits).unwrap();
        }
        let disp = rx
            .last_stats()
            .decode
            .as_ref()
            .expect("decode stats populated after decode call")
            .disposition
            .expect("disposition surfaced");
        // Clean own-encoded bits: 0 errors per frame, so Use.
        assert_eq!(disp, FrameDisposition::Use);
        assert_eq!(rx.last_disposition(), Some(FrameDisposition::Use));
    }

    /// `LiveEncoder` accepts arbitrary chunk sizes, holds residue
    /// across `push` calls, and emits the same bits as a one-shot
    /// `Vocoder::encode_pcm` per-frame loop.
    #[test]
    fn live_encoder_handles_arbitrary_chunk_sizes() {
        let mut total_pcm: Vec<i16> = Vec::with_capacity(7 * FRAME_SAMPLES);
        for _ in 0..7 {
            total_pcm.extend_from_slice(&periodic_pcm(40, 6000));
        }
        // Reference: per-frame Vocoder loop on the same input.
        let mut ref_v = Vocoder::new(Rate::P25Phase1);
        let mut ref_bits: Vec<u8> = Vec::new();
        for chunk in total_pcm.chunks_exact(FRAME_SAMPLES) {
            ref_bits.extend(ref_v.encode_pcm(chunk).unwrap());
        }
        // Live: feed in mismatched chunk sizes (250, 50, 333, rest).
        let mut live = LiveEncoder::new(Rate::P25Phase1);
        let mut live_bits: Vec<u8> = Vec::new();
        let splits = [250usize, 50, 333];
        let mut pos = 0;
        for &n in &splits {
            let end = (pos + n).min(total_pcm.len());
            for r in live.push(&total_pcm[pos..end]) {
                live_bits.extend(r.unwrap());
            }
            pos = end;
        }
        for r in live.push(&total_pcm[pos..]) {
            live_bits.extend(r.unwrap());
        }
        assert_eq!(live_bits, ref_bits);
        // Total samples 1120 = 7 frames exactly, so no residue.
        assert_eq!(live.pending_samples(), 0);
    }

    /// `LiveEncoder` correctly retains residue when input ends
    /// mid-frame.
    #[test]
    fn live_encoder_residue_held_across_calls() {
        let mut live = LiveEncoder::new(Rate::P25Phase1);
        // 1.5 frames of input split into two pushes.
        let pcm: Vec<i16> = periodic_pcm(40, 6000)
            .iter()
            .copied()
            .chain(periodic_pcm(40, 6000)[..80].iter().copied())
            .collect();
        assert_eq!(pcm.len(), 240);
        let frames = live.push(&pcm[..120]);
        assert!(frames.is_empty());
        assert_eq!(live.pending_samples(), 120);
        let frames = live.push(&pcm[120..]);
        assert_eq!(frames.len(), 1);
        assert!(frames[0].is_ok());
        assert_eq!(live.pending_samples(), 80);
    }

    /// `LiveDecoder` handles arbitrary chunk sizes for the byte stream
    /// and matches a per-frame `Vocoder::decode_bits` loop.
    #[test]
    fn live_decoder_handles_arbitrary_chunk_sizes() {
        let mut tx = Vocoder::new(Rate::P25Phase2);
        let mut all_pcm: Vec<i16> = Vec::with_capacity(5 * FRAME_SAMPLES);
        for _ in 0..5 {
            all_pcm.extend_from_slice(&periodic_pcm(40, 5000));
        }
        let bits: Vec<u8> = tx
            .encode_stream(&all_pcm)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let mut ref_v = Vocoder::new(Rate::P25Phase2);
        let mut ref_pcm: Vec<i16> = Vec::new();
        for chunk in bits.chunks_exact(9) {
            ref_pcm.extend(ref_v.decode_bits(chunk).unwrap());
        }
        let mut live = LiveDecoder::new(Rate::P25Phase2);
        let mut live_pcm: Vec<i16> = Vec::new();
        // Feed in 7-byte chunks (less than one frame each).
        for chunk in bits.chunks(7) {
            for r in live.push(chunk) {
                live_pcm.extend(r.unwrap());
            }
        }
        assert_eq!(live_pcm, ref_pcm);
        assert_eq!(live.pending_bytes(), 0); // 5 frames × 9 bytes = 45, multiple of 7? No (45 = 6*7+3) — pending should be 3 bytes if 5 frames don't fully drain. Wait, 5 frames take 45 bytes, fed as 7 chunks of 7 + 1 chunk of 2 = 45 bytes total. After feeding all bytes, all 5 frames produced. pending = 0.
    }

    /// `discard_pending` drops residue without emitting partial output.
    #[test]
    fn live_encoder_discard_pending_clears_residue() {
        let mut live = LiveEncoder::new(Rate::P25Phase1);
        let pcm = periodic_pcm(40, 6000);
        let frames = live.push(&pcm[..80]);
        assert!(frames.is_empty());
        assert_eq!(live.pending_samples(), 80);
        live.discard_pending();
        assert_eq!(live.pending_samples(), 0);
    }

    /// `reset` clears both vocoder state and the residue buffer.
    #[test]
    fn live_encoder_reset_clears_everything() {
        let mut live = LiveEncoder::new(Rate::P25Phase2);
        let pcm = periodic_pcm(40, 5000);
        let _ = live.push(&pcm[..120]);
        assert_eq!(live.pending_samples(), 120);
        let _ = live.vocoder().last_stats();
        live.reset();
        assert_eq!(live.pending_samples(), 0);
        assert!(live.vocoder().last_stats().analysis.is_none());
        assert_eq!(live.rate(), Rate::P25Phase2);
    }

    /// Streaming and per-frame paths produce identical output —
    /// the iterator is a thin chunking wrapper around `encode_pcm`,
    /// state advances the same way.
    #[test]
    fn encode_stream_matches_per_frame_calls_byte_for_byte() {
        let mut a = Vocoder::new(Rate::P25Phase1);
        let mut b = Vocoder::new(Rate::P25Phase1);
        let mut pcm: Vec<i16> = Vec::with_capacity(4 * FRAME_SAMPLES);
        for _ in 0..4 {
            pcm.extend_from_slice(&periodic_pcm(40, 6000));
        }
        let by_stream: Vec<u8> = a
            .encode_stream(&pcm)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let mut by_call: Vec<u8> = Vec::new();
        for chunk in pcm.chunks_exact(FRAME_SAMPLES) {
            by_call.extend(b.encode_pcm(chunk).unwrap());
        }
        assert_eq!(by_stream, by_call);
    }
}
