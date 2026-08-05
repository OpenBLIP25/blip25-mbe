//! Unified [`Vocoder`] handle — reference-shaped façade over the rate-specific
//! pipelines.
//!
//! The reference AMBE-3000R exposes a single per-channel handle with uniform
//! `encode` / `decode` / `reset` operations regardless of the configured
//! rate. This module reproduces that surface for in-process Rust callers:
//! one [`Vocoder`] handle owns the blip25_codec codec state and dispatches
//! uniformly across rates selected at runtime via the [`Rate`] enum.
//!
//! The low-level wire modules ([`crate::imbe7200`], [`crate::rate33`]) stay
//! public for advanced consumers that need frame-by-frame wire access; this
//! module is the recommended entry point for everything else.
//!
//! ## Quick start
//!
//! ```rust
//! use blip25_mbe::vocoder::{Rate, Vocoder};
//!
//! // Encode: PCM in → bits out, one call, best quality for the rate.
//! let tx = Vocoder::new(Rate::Imbe7200x4400);
//! let pcm = vec![0i16; 160 * 10]; // any length; one frame per 20 ms
//! let frames = tx.encode(&pcm);   // Vec of 18-byte FEC frames
//! assert!(frames.iter().all(|f| f.len() == 18));
//!
//! // Decode: bits → PCM (separate channel, separate state).
//! let mut rx = Vocoder::new(Rate::Imbe7200x4400);
//! for f in &frames {
//!     assert_eq!(rx.decode_bits(f).expect("decode").len(), 160);
//! }
//! ```
//!
//! ## Mapping to the reference protocol
//!
//! | Reference operation                | This module                          |
//! |-------------------------------|---------------------------------------|
//! | Open channel                  | [`Vocoder::new`]                     |
//! | Set rate (`PKT_RATEP`)        | [`Rate`] argument at construction    |
//! | Reset (re-send `PKT_RATEP`)   | [`Vocoder::reset`]                   |
//! | Encode PCM → bits (one call)  | [`Vocoder::encode`]                  |
//! | Encode one 160-sample frame   | [`Vocoder::encode_pcm`] (low-level)  |
//! | Decode bits → 160-sample PCM  | [`Vocoder::decode_bits`]             |
//! | Read last-frame stats         | [`Vocoder::last_stats`]              |
//! | Frame size                    | [`Vocoder::frame_samples`] / [`Vocoder::fec_frame_bytes`] |
//!
//! Rate is fixed for the lifetime of a [`Vocoder`]; build a new handle
//! to switch rates (mirrors a reference's PKT_RATEP cycle).

use crate::enhancement::{self, EnhancementMode, EnhancementState};
use crate::mbe_params::MbeParams;

/// 8 kHz mono — the only sample rate the vocoder produces.
const SAMPLE_RATE_HZ: f32 = 8_000.0;

/// Number of i16 PCM samples per 20 ms frame at 8 kHz. Constant across
/// every supported rate.
pub const FRAME_SAMPLES: usize = 160;

/// AMBE pitch cell whose decoded ω₀ (≈0.1955 rad/sample) re-quantizes to the
/// IMBE silence-default pitch index `b0=25` (L=14) — the fixed low-energy pitch
/// the reference IMBE encoder emits on silence frames. Injected as the forced `b0`
/// on IMBE frames whose source RMS falls below [`IMBE_SILENCE_RMS`]; the reference
/// spectral-DP pitch saturates to a spurious high `b0` there.
const IMBE_SILENCE_B0: u8 = 31;

/// Source-RMS threshold below which an IMBE frame is treated as silence and its
/// pitch pinned to [`IMBE_SILENCE_B0`]. Voiced frames sit far above this (agreeing
/// frames average RMS ~1400–1800; silence frames ~80–350), so the split is clean.
const IMBE_SILENCE_RMS: f64 = 400.0;

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
    Imbe7200x4400,
    /// IMBE info-only — same codec as [`Self::Imbe7200x4400`] with the
    /// Annex H Golay/Hamming/PN FEC layer stripped. 11-byte wire frame
    /// (88 prioritized info bits packed MSB-first). 4 400 bps total
    /// (= voice). Byte layout matches JMBE / OP25 / the reference `p25_nofec`
    /// convention bit-for-bit (validated against the reference tv-rc references
    /// at 100% bit-exact). Use this for storage when you specifically
    /// need an info-only IMBE archive; otherwise prefer the
    /// FEC-bearing [`Self::Imbe7200x4400`] format — see
    /// `docs/wire_formats_and_storage.md`.
    Imbe4400x4400,
    /// P25 Phase 2 TDMA half-rate AMBE+2. 9-byte FEC frame (36 dibits).
    /// 3 600 bps total / 2 450 bps voice + 1 150 bps FEC. Also the
    /// vocoder-layer format for DMR Tier II/III voice frames (modulo
    /// carrier-specific burst framing).
    AmbePlus2_3600x2450,
    /// AMBE+2 half-rate info-only — same 49 info bits as
    /// [`Self::AmbePlus2_3600x2450`] with the Golay/Hamming/PN FEC
    /// layer stripped. 7-byte wire frame: the 49 info bits in the reference
    /// **r34 order** plus 7 trailing pad bits. 2 450 bps total
    /// (= voice).
    ///
    /// **Byte layout — IMPORTANT, this is NOT naive sequential.** The
    /// bytes are packed by [`crate::rate33::frame::pack_no_fec`], which
    /// applies the reference r34 **3-way column interleave**
    /// ([`crate::rate33::frame::R34_BIT_ORDER`]) over the
    /// û₀‖û₁‖û₂‖û₃ bits — NOT a plain MSB-first sequential packing. This
    /// layout is byte-exact with the reference rate-index 34 no-FEC stream
    /// (the table was derived and validated against the reference RC r33↔r34
    /// reference vectors). Consumers that need natural / "AMBE_d" order —
    /// e.g. mbelib, or an IDAS/NXDN over-the-air wire, both of which use
    /// the *sequential* order — MUST de-interleave first via
    /// [`crate::rate33::frame::unpack_no_fec`]. For storage, the
    /// FEC-bearing [`Self::AmbePlus2_3600x2450`] is recommended — see
    /// `docs/wire_formats_and_storage.md`.
    AmbePlus2_2450x2450,
}

impl Rate {
    /// Number of bytes in one wire frame at this rate. Includes FEC
    /// for the FEC-bearing variants ([`Self::Imbe7200x4400`],
    /// [`Self::AmbePlus2_3600x2450`]) and is just the packed info bits
    /// for the no-FEC variants ([`Self::Imbe4400x4400`],
    /// [`Self::AmbePlus2_2450x2450`]).
    #[inline]
    pub const fn fec_frame_bytes(self) -> usize {
        match self {
            Rate::Imbe7200x4400 => 18,
            Rate::Imbe4400x4400 => 11,
            Rate::AmbePlus2_3600x2450 => 9,
            Rate::AmbePlus2_2450x2450 => 7,
        }
    }

    /// Soft-decision channel-bit count for one frame — the number of LLRs
    /// [`Vocoder::decode_soft`] expects (one per FEC channel bit). `None` for
    /// the no-FEC info-only rates, which carry no FEC layer to soft-decode.
    pub const fn soft_frame_bits(self) -> Option<usize> {
        match self {
            Rate::Imbe7200x4400 => Some(144),
            Rate::AmbePlus2_3600x2450 => Some(72),
            Rate::Imbe4400x4400 | Rate::AmbePlus2_2450x2450 => None,
        }
    }

    /// PCM samples per frame (always 160 at 8 kHz / 20 ms; provided
    /// for symmetry with [`Self::fec_frame_bytes`]).
    #[inline]
    pub const fn frame_samples(self) -> usize {
        FRAME_SAMPLES
    }

    /// One wire frame, [`Self::fec_frame_bytes`] long, marked as an **erasure**.
    ///
    /// Feed this to [`Vocoder::decode_bits`] in place of a frame the transport
    /// lost. The decoder repeats the previous good frame
    /// ([`FrameDisposition::Repeat`]) and, after four consecutive erasures,
    /// falls back to comfort noise ([`FrameDisposition::Mute`]) — the same
    /// concealment a radio applies when the channel drops frames.
    ///
    /// Both codecs mark an erasure in-band, by placing the pitch index outside
    /// its valid range: `b0 ∈ [120, 127]` for the half-rate AMBE+2 wire (the
    /// `û₀(11..6) ∈ {0x3C, 0x3D}` erasure escape), `b0 > 207` for IMBE. This
    /// returns the maximal marker in each range, so no single bit error can
    /// turn it back into a decodable pitch. Every other field is zero: nothing
    /// downstream of the erasure test reads them.
    pub fn erasure_frame(self) -> Vec<u8> {
        match self {
            Rate::AmbePlus2_3600x2450 => {
                let mut b = [0u16; 9];
                b[0] = AMBE_ERASURE_B0;
                blip25_codec::shared::encode_frame::encode_r33(&b).to_vec()
            }
            Rate::AmbePlus2_2450x2450 => {
                let mut b = [0u16; 9];
                b[0] = AMBE_ERASURE_B0;
                blip25_codec::shared::encode_frame::encode_r34(&b).to_vec()
            }
            Rate::Imbe7200x4400 => imbe_erasure_fec_frame().to_vec(),
            Rate::Imbe4400x4400 => {
                imbe_pipeline::fec_to_info_bytes(&imbe_erasure_fec_frame()).to_vec()
            }
        }
    }
}

/// Half-rate pitch index marking an erasure: the top of the `[120, 127]`
/// erasure escape range (`û₀(11..6) == 0x3C`).
const AMBE_ERASURE_B0: u16 = 127;

/// Prioritized `u₀` whose IMBE pitch index decodes to 252 — outside the valid
/// `[0, 207]` range, so the frame is an erasure marker.
const IMBE_ERASURE_U0: u16 = 0xFFF;

/// The 18-byte IMBE FEC frame carrying [`IMBE_ERASURE_U0`].
fn imbe_erasure_fec_frame() -> [u8; 18] {
    let mut u = [0u16; 8];
    u[0] = IMBE_ERASURE_U0;
    blip25_codec::imbe::enframe(&u)
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
    /// What the analysis encoder emitted (`Voice` / `Silence` / `Tone`).
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
    /// Annex T tone frame — encode-side detected a clean tone in the
    /// PCM and emitted the matching `(I_D, A_D)` payload instead of
    /// running the voice analysis pipeline. Half-rate (AMBE+2) only; gated on
    /// [`Vocoder::set_tone_detection`], off by default (opt-in).
    Tone {
        /// Annex T tone ID.
        id: u8,
        /// 7-bit log-amplitude (`A_D`).
        amplitude: u8,
    },
}

pub use blip25_codec::FrameDisposition;

/// Decode-side per-frame stats.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodeStats {
    /// FEC error count on the leading codeword (Golay c̃₀ for full-rate,
    /// AMBE+2 c̃₀ for half-rate).
    pub epsilon_0: u8,
    /// Total FEC error count across all coded vectors of the frame.
    pub epsilon_t: u8,
    /// Concealment disposition applied to this frame (BABA-1 §5.6/§5.7): normal
    /// decode, previous-frame repeat, or comfort-noise mute.
    pub disposition: blip25_codec::FrameDisposition,
}

/// Errors that can surface from [`Vocoder`] operations. Wraps the
/// rate-specific error types so a single `?` covers any Vocoder call.
#[derive(Debug)]
pub enum VocoderError {
    /// Input PCM slice was the wrong length. Always `frame_samples()`.
    WrongPcmLength {
        /// Number of samples the channel expected.
        expected: usize,
        /// Number of samples the caller passed.
        got: usize,
    },
    /// Input bit slice was the wrong length. Always `fec_frame_bytes()`.
    WrongBitsLength {
        /// Number of bytes the channel expected.
        expected: usize,
        /// Number of bytes the caller passed.
        got: usize,
    },
    /// Wire-quantize failure during transcode (a rate grid rejected the
    /// parameters, or the predictor returned a non-finite value).
    Quantize(String),
    /// Requested transcode direction is not supported. The pair of rates
    /// has no parameter-domain converter wired up.
    UnsupportedTranscode {
        /// Rate of the input FEC frame.
        from: Rate,
        /// Rate of the output FEC frame.
        to: Rate,
    },
    /// Soft-decision decode was requested on a rate with no FEC layer to
    /// soft-decode (the no-FEC info-only rates).
    SoftUnsupported {
        /// The rate that has no FEC to soft-decode.
        rate: Rate,
    },
    /// Soft-decision LLR slice was the wrong length. Always
    /// [`Rate::soft_frame_bits`] for the rate.
    WrongSoftLength {
        /// Number of LLRs the rate expected.
        expected: usize,
        /// Number of LLRs the caller passed.
        got: usize,
    },
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
            VocoderError::Quantize(msg) => write!(f, "quantize error: {msg}"),
            VocoderError::UnsupportedTranscode { from, to } => {
                write!(f, "unsupported transcode direction: {from:?} -> {to:?}")
            }
            VocoderError::SoftUnsupported { rate } => {
                write!(
                    f,
                    "soft-decision decode not supported for {rate:?} (no FEC layer to soft-decode)"
                )
            }
            VocoderError::WrongSoftLength { expected, got } => {
                write!(f, "expected {expected} soft LLRs per frame, got {got}")
            }
        }
    }
}

impl std::error::Error for VocoderError {}

/// Reference-shaped façade over the rate-specific encoder + decoder + synth
/// pipelines.
///
/// Owns all per-rate state internally (analysis, decoder, synth). One
/// [`Vocoder`] is *both* encoder and decoder for a single channel
/// direction; consumers running bidirectional voice typically allocate
/// two — one each direction — to mirror the reference's per-direction
/// state isolation.
///
/// State is not `Sync`; one channel = one thread. State is `Send`,
/// so the channel can move between threads.
pub struct Vocoder {
    rate: Rate,
    last_stats: FrameStats,
    /// Optional post-decode enhancement chain (biquad + compressor +
    /// boundary fade). [`EnhancementMode::None`] by default (a strict
    /// no-op), so the shipped path is byte/sample-identical to `blip25_codec`
    /// direct. This is the one standalone DSP hook kept for the future
    /// AGC / noise-removal audio chain; see [`crate::enhancement`].
    enhancement: EnhancementMode,
    enhancement_state: EnhancementState,
    /// Disposition of the PREVIOUS decoded frame, so the enhancement chain's
    /// boundary fade can arm on a `Repeat`/`Mute` -> `Use` transition. Tracked
    /// here rather than read from `last_stats` because `last_stats` has already
    /// been overwritten with the current frame by the time `apply` runs.
    prev_disposition: FrameDisposition,
    /// Stateful blip25_codec AMBE+2 decoder (r33/r34), carried across frames.
    /// Present only for the two AMBE+2 rates.
    w_ambe_dec: Option<blip25_codec::Decoder>,
    /// Stateful blip25_codec IMBE decoder, carried across frames. Present only for
    /// the two IMBE rates.
    w_imbe_dec: Option<blip25_codec::ImbeDecoder>,
    /// Stateful blip25_codec encoder (shared analysis front end; AMBE+2 via
    /// `encode_frame_r33/r34`, IMBE via `encode_imbe_frame`). Carried across
    /// frames for the one-frame-look-ahead streaming contract.
    #[cfg(feature = "encode")]
    w_enc: Option<blip25_codec::enc::Encoder>,
    /// One-frame-look-ahead FIFO for the blip25_codec encoder: the console codec
    /// emits frame *f−1*'s bits when fed frame *f* (`None` on the very first
    /// call). We buffer emissions here and hand back one frame per
    /// [`Self::encode_pcm`] call, draining the tail via
    /// [`Self::flush_encode`]. See that method's docs for the exact contract.
    #[cfg(feature = "encode")]
    w_enc_queue: std::collections::VecDeque<Vec<u8>>,
    /// Whether [`Self::encode`] detects Annex-T tones (Knox / DTMF / call-progress
    /// / alert) and emits tone frames instead of voice analysis. **OFF by
    /// default** — the reference console encoder does not tone-detect (DTMF is
    /// signaled out-of-band), so speaker-ready output matches it with detection
    /// off. Opt in via [`Self::set_tone_detection`] for the rare FNE case that
    /// needs vocoder tone frames. Half-rate AMBE+2 only.
    #[cfg(feature = "encode")]
    tone_detection: bool,
}

impl Vocoder {
    /// Open a new channel at the given rate, all state cold.
    ///
    /// Encode and decode route through the [`blip25_codec`] engine for all
    /// four rates.
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            last_stats: FrameStats::default(),
            // Enhancement: NONE by default. The shipped decode path IS the
            // reference console codec; layering a post-filter would make the
            // output diverge from `blip25_codec` direct. Opt in via set_enhancement.
            enhancement: EnhancementMode::None,
            enhancement_state: EnhancementState::default(),
            prev_disposition: FrameDisposition::Use,
            w_ambe_dec: match rate {
                Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                    Some(blip25_codec::Decoder::new())
                }
                _ => None,
            },
            w_imbe_dec: match rate {
                Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => Some(blip25_codec::ImbeDecoder::new()),
                _ => None,
            },
            #[cfg(feature = "encode")]
            w_enc: Some(blip25_codec::enc::Encoder::new()),
            #[cfg(feature = "encode")]
            w_enc_queue: std::collections::VecDeque::new(),
            #[cfg(feature = "encode")]
            tone_detection: false,
        }
    }

    /// Enable or disable encode-side Annex-T tone detection (**default OFF**).
    /// When on, [`Self::encode`] (AMBE+2 r33) detects clean single/dual tones —
    /// Knox, DTMF, call-progress, alert — in the input and emits the matching
    /// byte-exact tone frame instead of running voice analysis on it. OFF by
    /// default because the reference console encoder does not tone-detect (DTMF is
    /// carried out-of-band), so speaker-ready output matches it with detection
    /// off; enable it only for the rare FNE deployment that must carry vocoder
    /// tone frames. Persistent across [`Self::reset`].
    #[cfg(feature = "encode")]
    pub fn set_tone_detection(&mut self, enable: bool) {
        self.tone_detection = enable;
    }

    /// Whether encode-side tone detection is enabled (see
    /// [`Self::set_tone_detection`]).
    #[inline]
    #[cfg(feature = "encode")]
    pub fn tone_detection(&self) -> bool {
        self.tone_detection
    }

    /// Configure the post-decoder enhancement chain. Off by default
    /// ([`EnhancementMode::None`] — spec-faithful PCM). When set to
    /// [`EnhancementMode::Classical`], decoded PCM passes through a
    /// biquad cascade + soft-knee compressor + boundary-fade chain
    /// before [`Self::decode_bits`] returns. See
    /// [`crate::enhancement`] for stage details and AIC33 mapping.
    ///
    /// Resets the chain's runtime filter state (delay lines, envelope,
    /// pending fade) so the new mode starts clean. Persistent: stays
    /// configured across [`Self::reset`].
    pub fn set_enhancement(&mut self, mode: EnhancementMode) {
        self.enhancement = mode;
        self.enhancement_state = EnhancementState::default();
    }

    /// Currently configured enhancement mode.
    #[inline]
    pub fn enhancement(&self) -> &EnhancementMode {
        &self.enhancement
    }

    /// Start a fluent builder for this rate. Equivalent to
    /// `VocoderBuilder::new(rate)`.
    #[inline]
    pub fn builder(rate: Rate) -> VocoderBuilder {
        VocoderBuilder::new(rate)
    }

    /// The rate this channel was constructed at. Cannot change for
    /// the lifetime of the channel; build a new [`Vocoder`] to switch
    /// rates (mirrors a reference's PKT_RATEP cycle).
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

    /// Reset all channel state — the reference PKT_RATEP re-send equivalent.
    /// Rebuilds the blip25_codec decoders/encoder from cold, clears the look-ahead
    /// FIFO and last-frame stats. The configured enhancement mode is preserved
    /// (configuration, not channel state).
    pub fn reset(&mut self) {
        self.last_stats = FrameStats::default();
        self.enhancement_state = EnhancementState::default();
        self.prev_disposition = FrameDisposition::Use;
        self.w_ambe_dec = match self.rate {
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                Some(blip25_codec::Decoder::new())
            }
            _ => None,
        };
        self.w_imbe_dec = match self.rate {
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => Some(blip25_codec::ImbeDecoder::new()),
            _ => None,
        };
        #[cfg(feature = "encode")]
        {
            self.w_enc = Some(blip25_codec::enc::Encoder::new());
            self.w_enc_queue.clear();
        }
    }

    /// Encode one PCM frame into FEC-encoded bytes through the vendored
    /// [`blip25_codec`] reference codec.
    ///
    /// `pcm` must be exactly [`Self::frame_samples`] samples (160). The console
    /// codec has a one-frame analysis look-ahead, so the FIRST returned frame is
    /// an all-zero placeholder (real frame 0 lands on the next call) and the
    /// final frame stays buffered until [`Self::flush_encode`] drains it.
    ///
    /// This does **not** emit the same bits as [`Self::encode`] at any rate: the
    /// reference pitch (`b0`) is injected only on the whole-buffer path, and on both
    /// AMBE+2 rates that path additionally swaps in the tracked `b1` voicing,
    /// clamps the gain on silent frames, and overlays Annex-T tone frames —
    /// none of which happen here.
    #[cfg(feature = "encode")]
    pub fn encode_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, VocoderError> {
        if pcm.len() != self.frame_samples() {
            return Err(VocoderError::WrongPcmLength {
                expected: self.frame_samples(),
                got: pcm.len(),
            });
        }
        let frame: &[i16; FRAME_SAMPLES] = pcm.try_into().expect("length already validated");
        let enc = self.w_enc.as_mut().expect("blip25_codec encoder present");
        let emitted: Option<Vec<u8>> = match self.rate {
            Rate::AmbePlus2_3600x2450 => enc.encode_frame_r33(frame).map(|b| b.to_vec()),
            Rate::AmbePlus2_2450x2450 => enc.encode_frame_r34(frame).map(|b| b.to_vec()),
            Rate::Imbe7200x4400 => enc.encode_imbe_frame(frame).map(|b| b.to_vec()),
            Rate::Imbe4400x4400 => enc
                .encode_imbe_frame(frame)
                .map(|b| imbe_fec_to_info_bytes(&b).to_vec()),
        };
        if let Some(b) = emitted {
            self.w_enc_queue.push_back(b);
        }
        let bytes = self
            .w_enc_queue
            .pop_front()
            .unwrap_or_else(|| vec![0u8; self.rate.fec_frame_bytes()]);
        let params = match self.rate {
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => MbeParams::silence(),
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                MbeParams::silence_ambe_plus2()
            }
        };
        self.last_stats.analysis = Some(AnalysisStats {
            output: AnalysisOutputKind::Voice,
            params,
        });
        Ok(bytes)
    }

    /// Drain the blip25_codec encoder's remaining look-ahead frame(s) at end-of-stream.
    /// Byte-identical to `blip25_codec::enc::Encoder::flush_r33` / `flush_r34` /
    /// `flush_imbe` on the same frame history.
    #[cfg(feature = "encode")]
    pub fn flush_encode(&mut self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        if let Some(enc) = self.w_enc.as_mut() {
            match self.rate {
                Rate::AmbePlus2_3600x2450 => {
                    for b in enc.flush_r33() {
                        self.w_enc_queue.push_back(b.to_vec());
                    }
                }
                Rate::AmbePlus2_2450x2450 => {
                    for b in enc.flush_r34() {
                        self.w_enc_queue.push_back(b.to_vec());
                    }
                }
                Rate::Imbe7200x4400 => {
                    for b in enc.flush_imbe() {
                        self.w_enc_queue.push_back(b.to_vec());
                    }
                }
                Rate::Imbe4400x4400 => {
                    for b in enc.flush_imbe() {
                        self.w_enc_queue
                            .push_back(imbe_fec_to_info_bytes(&b).to_vec());
                    }
                }
            }
        }
        while let Some(b) = self.w_enc_queue.pop_front() {
            out.push(b);
        }
        out
    }

    /// Encode a whole PCM buffer to FEC frames — **the one encode path: PCM in,
    /// bits out, no options.** Applies the best available quality for the
    /// configured [`Rate`] automatically:
    ///
    /// Every rate runs the reverse-engineered reference analysis: reference
    /// pitch (`b0`), the tracked `b1` voicing word, the silence gate, and — on
    /// AMBE+2 — the Annex-T tone overlay. Fricatives do not buzz and inter-word
    /// silence stays quiet.
    ///
    /// `pcm` may be any length; returns one FEC frame per 20 ms of input,
    /// rounded up. Output is byte-identical to feeding the same audio through
    /// [`LiveEncoder`], so this is a convenience for callers that already hold
    /// the whole buffer — not a higher-quality path. Real-time callers should
    /// use [`LiveEncoder`]. [`Self::encode_pcm`] is the low-level per-frame
    /// primitive both are built on and emits different bits: it applies none of
    /// the reference analysis above.
    #[cfg(feature = "encode")]
    pub fn encode(&self, pcm: &[i16]) -> Vec<Vec<u8>> {
        match self.rate {
            // Both AMBE+2 rates carry the SAME 49 info bits and go through the
            // SAME reference-voicing analysis; r33 packs with FEC, r34 without.
            // (r34 is the primary path: encode → encrypt → FEC externally.)
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                let with_fec = matches!(self.rate, Rate::AmbePlus2_3600x2450);
                // One output frame per 20 ms of input. The reference analysis
                // spends one frame of look-ahead, so a partial trailing frame is
                // completed with silence and one further silent frame is appended
                // to buy that look-ahead back; the emitted count is then exactly
                // `ceil(len / 160)`. `AmbeStream::flush` pads identically, so
                // both paths analyse the same buffer.
                let want = pcm.len().div_ceil(FRAME_SAMPLES);
                if want == 0 {
                    return Vec::new();
                }
                let mut padded = Vec::with_capacity((want + 1) * FRAME_SAMPLES);
                padded.extend_from_slice(pcm);
                padded.resize((want + 1) * FRAME_SAMPLES, 0);
                let mut out = self.encode_ambe2_reference_voicing(&padded, with_fec);
                out.truncate(want);
                out
            }
            _ => {
                // One output frame per 20 ms of input, so a partial trailing
                // frame is completed with silence rather than discarded.
                let residue = pcm.len() % FRAME_SAMPLES;
                let padded: Vec<i16>;
                let pcm = if residue == 0 {
                    pcm
                } else {
                    padded = pcm
                        .iter()
                        .copied()
                        .chain(std::iter::repeat_n(0i16, FRAME_SAMPLES - residue))
                        .collect();
                    &padded
                };
                // Per-frame path for the other rates (look-ahead: drop the
                // placeholder frames, flush the tail).
                let mut enc = Vocoder::new(self.rate);
                // Whole-buffer IMBE sources its amplitudes from the reference's
                // fixed-point spectrum mechanism (bit-exact ported FFT feeding
                // `band_decompress`, slot-1 exponent) via the LIVE `gap2_mid`
                // window. The estimator's window leads pitch by one frame, so
                // the two-frame analysis look-ahead (`set_live_gap2_amps`)
                // delays the emit enough that `gap2_mid` is the frame-(f+1)
                // window the mechanism needs. This is set only on this
                // whole-buffer encoder; the streaming `encode_pcm`/`LiveEncoder`
                // path keeps the one-frame look-ahead and the synthesized-DFT
                // amplitude proxy, so its public look-ahead contract is
                // unchanged. IMBE never calls the AMBE `analyze_frame`, so the
                // flag's only effects here are the look-ahead depth and the
                // mechanism amplitude source in `analyze_imbe_frame`.
                let imbe = matches!(self.rate, Rate::Imbe7200x4400 | Rate::Imbe4400x4400);
                if imbe {
                    if let Some(e) = enc.w_enc.as_mut() {
                        e.set_live_gap2_amps(true);
                    }
                }
                // Reference pitch for IMBE — THE single pitch path (same RE'd
                // spectral-DP pitch that fixed AMBE+2; owner-confirmed better).
                // The packer tone branch is half-rate only, so this uses the
                // variant without it and matches `ImbeStream`.
                {
                    let reference_b0 =
                        blip25_codec::enc::pcm_encode::encode_pcm_b0_no_tone_branch(
                            pcm,
                            blip25_codec::enc::pcm_encode::EncodeOpts::default(),
                        );
                    if !reference_b0.is_empty() {
                        let nf = pcm.len() / FRAME_SAMPLES;
                        // Silence gate: on low-energy frames the reference spectral-DP
                        // pitch saturates to a spurious high b0, whereas reference
                        // pins IMBE silence to a fixed default (b0=25). Detect
                        // silence by per-frame source RMS and force the AMBE cell
                        // that re-quantizes to that default; keep the reference pitch
                        // (voiced-solved) everywhere else.
                        let forced: Vec<u8> = (0..nf)
                            .map(|f| {
                                let chunk = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
                                let ss: f64 =
                                    chunk.iter().map(|&x| (x as f64) * (x as f64)).sum();
                                let rms = (ss / FRAME_SAMPLES as f64).sqrt();
                                if rms < IMBE_SILENCE_RMS {
                                    IMBE_SILENCE_B0
                                } else {
                                    reference_b0[(f + 2).min(reference_b0.len() - 1)]
                                }
                            })
                            .collect();
                        if let Some(e) = enc.w_enc.as_mut() {
                            e.set_forced_b0(forced);
                        }
                    }
                }
                // Reference voicing for IMBE: the shared reverse-engineered `b1_track`
                // metric (byte-identical to the reference encoder's voicing word),
                // expanded per-harmonic inside `analyze_imbe_frame`. Replaces the
                // float `decide_voicing_cfg` path, which over-voices fricative high
                // bands. `b1_track` needs the r33 gap2/prefiltered analysis logs
                // (IMBE analysis doesn't write them), so run one r33 pass purely to
                // populate them, then align to the same (f+2) source-frame index as
                // the forced reference pitch above.
                if imbe {
                    use blip25_codec::enc::b1_audio::{b1_track_from_logs, RingRefineMode};
                    let nframes = (pcm.len() / FRAME_SAMPLES).saturating_sub(1);
                    if nframes > 0 {
                        let mut la = blip25_codec::enc::Encoder::new();
                        for f in pcm.chunks_exact(FRAME_SAMPLES) {
                            let a: &[i16; FRAME_SAMPLES] = f.try_into().expect("chunk is 160");
                            let _ = la.encode_frame_r33(a);
                        }
                        let _ = la.flush_r33();
                        let bt = b1_track_from_logs(
                            la.gap2_mid_log(),
                            la.gap2_slot1_log(),
                            la.gap2_slot2_log(),
                            la.prefiltered_log(),
                            pcm,
                            nframes,
                            RingRefineMode::Off,
                        );
                        let b1w: Vec<u16> = bt.iter().map(|f| f.b1).collect();
                        if !b1w.is_empty() {
                            let n = pcm.len() / FRAME_SAMPLES;
                            let forced_b1: Vec<u16> = (0..n)
                                .map(|f| b1w[(f + 2).min(b1w.len() - 1)])
                                .collect();
                            if let Some(e) = enc.w_enc.as_mut() {
                                e.set_forced_b1(forced_b1);
                            }
                        }
                    }
                }
                let mut out: Vec<Vec<u8>> = Vec::new();
                for f in pcm.chunks_exact(FRAME_SAMPLES) {
                    if let Ok(b) = enc.encode_pcm(f) {
                        out.push(b);
                    }
                }
                // The IMBE encoder above runs the two-frame look-ahead
                // (`set_live_gap2_amps`), so it fills with two leading all-zero
                // placeholders; the one-frame look-ahead fills with one. Drop as
                // many leading placeholders as the look-ahead depth so the output
                // frame count equals the input.
                let placeholders = if imbe { 2 } else { 1 };
                for _ in 0..placeholders {
                    if !out.is_empty() {
                        out.remove(0);
                    }
                }
                out.extend(enc.flush_encode());
                out
            }
        }
    }

    /// AMBE+2 (r33) reference-voicing + silence-gate encode — the [`Self::encode`]
    /// path for [`Rate::AmbePlus2_3600x2450`]. blip25's own amplitude/pitch with
    /// the `b1` (voicing) byte from the reverse-engineered [`blip25_codec`] `b1_track`
    /// (byte-identical to the reference encoder), plus a gain clamp on silent
    /// frames. A/B-validated against the reference vectors (fricative buzz removed,
    /// inter-word floor pulled down). Whole-clip (b1_track needs the stream).
    #[cfg(feature = "encode")]
    fn encode_ambe2_reference_voicing(&self, pcm: &[i16], with_fec: bool) -> Vec<Vec<u8>> {
        use blip25_codec::enc::b1_audio::{b1_track_from_logs_lb, RingRefineMode};
        debug_assert!(
            matches!(
                self.rate,
                Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450
            ),
            "reference-voicing encode is AMBE+2 only"
        );
        // Hostile input class: sub-frame PCM (empty or < 160 samples) — zero full
        // frames to analyze, and the b1_track log pipeline needs at least one; emit
        // no frames (the same "one frame per full 20 ms" contract as chunks_exact).
        if pcm.len() < FRAME_SAMPLES {
            return Vec::new();
        }
        const R33: usize = 9;

        // 1. blip25's own per-frame amplitude/pitch frames (drop the look-ahead
        //    placeholder, flush the tail) — same convention as `encode_pcm`.
        //
        // TWO-PASS (whole-buffer only). The reference's internal analysis window
        // leads its pitch by one frame, so band_decompress must be fed emit
        // frame f's amplitudes from the frame-(f+1) window to align the two.
        // PASS 1 runs the normal encoder purely to populate `gap2_mid_log` (and
        // the b1-track logs); PASS 2 replays it with a per-frame gap2 override
        // set to `gap2_mid_log[f+1]` and the mechanism/reference-exact amplitude
        // chain, and PASS 2's frames are the output. The streaming `AmbeStream`
        // path CANNOT do this — it has no frame-(f+1) look-ahead — so it stays
        // on the one-frame-skewed proxy until Route A (+20 ms look-ahead) is
        // taken; see `AmbeStream` and the module note.
        //
        // Reference spectral-DP pitch — THE single pitch path (no pyin fallback:
        // ear-validated to beat pyin decisively, killing the subharmonic
        // "axe"/"ancient" distortion). Injected per analyze-frame f as
        // reference_b0[f+2] (validated alignment). Uses the pitch-only chain
        // (`encode_pcm_b0`), which skips the amplitude/VQ work — byte-identical
        // b0 to a full reference encode at a fraction of the cost. The SAME forced_b0
        // drives both passes, so b0 (and b1) are byte-identical to the one-pass
        // encoder — only the amplitude/gain fields (b2..b8) move.
        let forced_b0: Option<Vec<u8>> = {
            let reference_b0 = blip25_codec::enc::pcm_encode::encode_pcm_b0(
                pcm,
                blip25_codec::enc::pcm_encode::EncodeOpts::default(),
            );
            if reference_b0.is_empty() {
                None
            } else {
                let nf = pcm.len() / FRAME_SAMPLES;
                Some(
                    (0..nf)
                        .map(|f| reference_b0[(f + 2).min(reference_b0.len() - 1)])
                        .collect(),
                )
            }
        };

        // PASS 1: populate the analysis logs (discard the bits).
        let mut e = blip25_codec::enc::Encoder::new();
        if let Some(fb) = forced_b0.as_ref() {
            e.set_forced_b0(fb.clone());
        }
        for f in pcm.chunks_exact(FRAME_SAMPLES) {
            let a: &[i16; FRAME_SAMPLES] = f.try_into().expect("chunk is 160");
            let _ = e.encode_frame_r33(a);
        }
        let _ = e.flush_r33();

        // PASS 2: replay with Route A — the two-frame look-ahead makes the LIVE
        // `gap2_mid` at each emit refer to the pitch-aligned (frame f+2) window,
        // so a single forward pass yields the aligned mechanism amplitudes +
        // reference-exact b2 with no per-frame override array. (Pass 1 above still
        // runs at the one-frame look-ahead purely to populate the b1-track logs,
        // which were fit to that timing.)
        // The b1 track is derived from pass 1's logs, so it is available before
        // pass 2 runs — which is what lets the low-band floor's gate read the
        // b1 that will actually be transmitted. Only `e2` may carry the floor:
        // giving it to `e` would move the very logs this track is built from.
        let nframes = (pcm.len() / FRAME_SAMPLES).saturating_sub(1);
        let bt = b1_track_from_logs_lb(
            e.gap2_mid_log(),
            e.gap2_slot1_log(),
            e.gap2_slot2_log(),
            e.prefiltered_log(),
            pcm,
            nframes,
            RingRefineMode::Off,
        );
        let orac_b1: Vec<u16> = bt.iter().map(|f| f.b1).collect();

        let mut e2 = blip25_codec::enc::Encoder::new();
        if let Some(fb) = forced_b0.as_ref() {
            e2.set_forced_b0(fb.clone());
        }
        e2.set_live_gap2_amps(true);
        e2.set_amp_next(amp_next_enabled());
        if !orac_b1.is_empty() {
            // The packer tone branch's `(L, step)` override for the same frame
            // whose `b̂₀` `forced_b0` injected: analyse frame `f` carries
            // reference frame `f + BLAG`'s packing. A no-op with the branch off.
            let last = orac_b1.len() - 1;
            let nf = pcm.len() / FRAME_SAMPLES;
            e2.set_forced_l56(
                (0..=nf)
                    .map(|f| {
                        blip25_codec::enc::pcm_encode::tone_branch_l56(
                            orac_b1[(f + BLAG).min(last)],
                        )
                    })
                    .collect(),
            );
        }
        if lowband_floor_enabled() && !orac_b1.is_empty() {
            // Analyse frame `f` is packed with `orac_b1[f + BLAG]` whenever the
            // lag search below lands on `AMBE_B1_OUTPUT_LAG`, which is the same
            // assumption `forced_b0` already makes.
            let last = orac_b1.len() - 1;
            let nf = pcm.len() / FRAME_SAMPLES;
            e2.set_lowband_floor(true);
            e2.set_lowband_gate_b1((0..=nf).map(|f| orac_b1[(f + BLAG).min(last)]).collect());
        }
        let mut frames: Vec<[u8; R33]> = Vec::new();
        for f in pcm.chunks_exact(FRAME_SAMPLES) {
            let a: &[i16; FRAME_SAMPLES] = f.try_into().expect("chunk is 160");
            if let Some(b) = e2.encode_frame_r33(a) {
                frames.push(b);
            }
        }
        if !frames.is_empty() {
            frames.remove(0);
        }
        frames.extend(e2.flush_r33());

        // 2. Reference voicing per frame from the RE'd b1_track pipeline. Reuse
        //    the analysis logs the real-bits pass above already produced
        //    (`e` ran the full r34-equivalent analysis; r33/r34 share the
        //    prefilter + gap-log writers and diverge only in bit-packing), so
        //    the b1 chain does not re-run the encoder two more times.
        //    (`orac_b1` is computed above, between the two passes.)

        // 3. Align blip25's stream to the b1_track index (the two use different
        //    look-ahead conventions), then swap `b1` and repack the r33 frame.
        //    The offset is [`AMBE_B1_OUTPUT_LAG`], the same convention
        //    `forced_b0` and the tone branch's `(L, step)` override above are
        //    built on.
        let lag = AMBE_B1_OUTPUT_LAG;

        // 4. Annex-T tone overlay. A streaming detector reproduces the reference's tone
        //    vs voice decision (and 1-frame onset lag) per source frame; where
        //    it fires, the frame is emitted as a byte-exact tone frame instead
        //    of the voice bits. Output frame `i` corresponds to source frame
        //    `i + 1` (the look-ahead drop), so the detection consumed for
        //    output `i` is source frame `i + 1`.
        let tones_enabled = self.tone_detection;
        let mut det = blip25_codec::tone::ToneDetector::new();
        let tone_of: Vec<Option<blip25_codec::tone::ToneFrameFields>> = pcm
            .chunks_exact(FRAME_SAMPLES)
            .map(|c| if tones_enabled { det.process(c) } else { None })
            .collect();

        frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if let Some(t) = tone_of.get(i + 1).copied().flatten() {
                    let r33 = Self::tone_frame_bytes(t.id, t.amplitude);
                    return if with_fec {
                        r33
                    } else {
                        let tb = blip25_codec::tables::deprioritize(
                            &blip25_codec::frame::decode_bytes(&r33).info,
                        );
                        blip25_codec::enc::encode_frame::encode_r34(&tb).to_vec()
                    };
                }
                let mut b =
                    blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(f).info);
                let j = i as i64 + lag;
                if j >= 0 && (j as usize) < orac_b1.len() {
                    b[1] = orac_b1[j as usize];
                }
                // r33 packs with FEC, r34 (2450, the encrypt-then-FEC path) without.
                if with_fec {
                    blip25_codec::enc::encode_frame::encode_r33(&b).to_vec()
                } else {
                    blip25_codec::enc::encode_frame::encode_r34(&b).to_vec()
                }
            })
            .collect()
    }

    /// Build the 9-byte r33 FEC frame for an Annex-T tone `(I_D, A_D)`. Produces
    /// bytes byte-identical to the reference tone frames (verified across the tone
    /// vectors): the prioritized info vectors carry the §2.10.1 signature and
    /// Table 20 layout, FEC-encoded and packed to dibits/bytes.
    fn tone_frame_bytes(id: u8, amplitude: u8) -> Vec<u8> {
        let info = crate::rate33::dequantize::encode_tone_frame_info(id, amplitude);
        let dibits = crate::rate33::frame::encode_frame(&info);
        let mut out = vec![0u8; 9];
        for (i, &d) in dibits.iter().enumerate() {
            let (hi, lo) = ((d >> 1) & 1, d & 1);
            out[(2 * i) / 8] |= hi << (7 - ((2 * i) % 8));
            out[(2 * i + 1) / 8] |= lo << (7 - ((2 * i + 1) % 8));
        }
        out
    }

    /// Decode one FEC-encoded frame into PCM through the vendored [`blip25_codec`]
    /// reference codec. `bits` must be exactly [`Self::fec_frame_bytes`] bytes;
    /// returns exactly [`Self::frame_samples`] samples. Any configured
    /// [`Self::set_enhancement`] post-filter is applied before returning.
    pub fn decode_bits(&mut self, bits: &[u8]) -> Result<Vec<i16>, VocoderError> {
        if bits.len() != self.fec_frame_bytes() {
            return Err(VocoderError::WrongBitsLength {
                expected: self.fec_frame_bytes(),
                got: bits.len(),
            });
        }
        let (mut pcm, stats) = self.decode_via_codec(bits);
        // The fade arms on the first `Use` frame AFTER a `Repeat`/`Mute`, so it
        // is keyed on the previous frame's disposition, not this one's.
        let prev_was_use = self.prev_disposition == FrameDisposition::Use;
        enhancement::apply(
            &self.enhancement,
            &mut self.enhancement_state,
            &mut pcm,
            SAMPLE_RATE_HZ,
            prev_was_use,
        );
        self.prev_disposition = stats.disposition;
        self.last_stats.decode = Some(stats);
        Ok(pcm)
    }

    /// Number of soft-decision LLRs [`Self::decode_soft`] expects per frame,
    /// or `None` for the no-FEC rates (no FEC layer to soft-decode).
    #[inline]
    pub fn soft_frame_bits(&self) -> Option<usize> {
        self.rate.soft_frame_bits()
    }

    /// Decode one frame from **soft-decision** channel bits (LLRs) instead of
    /// hard bytes.
    ///
    /// `llrs` holds one signed value per FEC channel bit — sign = the hard
    /// decision, magnitude = confidence — in raw frame-bit order (`SD0`
    /// first, the order [`crate::reference_soft_decision::unpack_nibble_stream`]
    /// yields from a reference `*_sd.bit` frame). Length must equal
    /// [`Self::soft_frame_bits`] (144 for [`Rate::Imbe7200x4400`], 72 for
    /// [`Rate::AmbePlus2_3600x2450`]).
    ///
    /// Runs the crate's soft Golay/Hamming FEC (Chase-II — the ~2 dB coding
    /// gain over hard slicing), then re-encodes the recovered info to a clean
    /// FEC frame and synthesizes through the shipped [`Self::decode_bits`]
    /// path — so decoder state, concealment, and enhancement behave exactly as
    /// for a clean hard decode. On error-free input the result is
    /// bit-identical to `decode_bits`; the soft path diverges only by
    /// *rescuing* frames a hard decode would corrupt.
    ///
    /// Soft decode is only meaningful for the FEC-bearing rates; the no-FEC
    /// info-only rates return [`VocoderError::SoftUnsupported`].
    ///
    /// ```rust
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// let mut rx = Vocoder::new(Rate::Imbe7200x4400);
    /// let llrs = [0i8; 144]; // one LLR per channel bit
    /// let pcm = rx.decode_soft(&llrs).unwrap();
    /// assert_eq!(pcm.len(), rx.frame_samples());
    /// ```
    pub fn decode_soft(&mut self, llrs: &[i8]) -> Result<Vec<i16>, VocoderError> {
        let expected = self
            .rate
            .soft_frame_bits()
            .ok_or(VocoderError::SoftUnsupported { rate: self.rate })?;
        if llrs.len() != expected {
            return Err(VocoderError::WrongSoftLength {
                expected,
                got: llrs.len(),
            });
        }
        // Soft-decode the FEC to recovered info vectors, then re-encode a clean
        // FEC frame and run the shipped hard path (blip25_codec synth + conceal +
        // enhancement + state). The re-encoded frame is error-free, so blip25_codec's
        // hard FEC decode is lossless.
        let fec: Vec<u8> = match self.rate {
            Rate::Imbe7200x4400 => {
                let soft: &[i8; 144] = llrs.try_into().expect("length checked above");
                let frame = crate::imbe7200::frame::decode_frame_soft(soft);
                imbe_pipeline::info_vec_to_fec_bytes(&frame.info).to_vec()
            }
            Rate::AmbePlus2_3600x2450 => {
                let soft: &[i8; 72] = llrs.try_into().expect("length checked above");
                let frame = crate::rate33::frame::decode_frame_soft(soft);
                ambe_plus2_pipeline::info_vec_to_fec_bytes(&frame.info).to_vec()
            }
            // soft_frame_bits() returned Some(_) only for the two rates above.
            _ => unreachable!("no-FEC rates gated by soft_frame_bits()"),
        };
        self.decode_bits(&fec)
    }

    /// Decode one wire frame to raw PCM through the vendored [`blip25_codec`] codec
    /// (no enhancement — [`Self::decode_bits`] applies it). Length pre-validated.
    fn decode_via_codec(&mut self, bits: &[u8]) -> (Vec<i16>, DecodeStats) {
        match self.rate {
            Rate::AmbePlus2_3600x2450 => {
                let dec = self.w_ambe_dec.as_mut().expect("AMBE decoder present");
                let (pcm, disposition, epsilon_0, epsilon_t) = dec.decode_pcm_fixed_concealed(bits);
                (
                    pcm.to_vec(),
                    DecodeStats {
                        epsilon_0,
                        epsilon_t,
                        disposition,
                    },
                )
            }
            Rate::AmbePlus2_2450x2450 => {
                let mut r34 = [0u8; 7];
                r34.copy_from_slice(&bits[..7]);
                let b = blip25_codec::shared::encode_frame::decode_r34(&r34);
                let r33 = blip25_codec::shared::encode_frame::encode_r33(&b);
                let dec = self.w_ambe_dec.as_mut().expect("AMBE decoder present");
                // r34 is FEC-less info; the re-encoded r33 carries no channel
                // errors, so the gate is inert (always Use) — routed through the
                // concealed path only for a uniform disposition surface.
                let (pcm, disposition, epsilon_0, epsilon_t) = dec.decode_pcm_fixed_concealed(&r33);
                (
                    pcm.to_vec(),
                    DecodeStats {
                        epsilon_0,
                        epsilon_t,
                        disposition,
                    },
                )
            }
            Rate::Imbe7200x4400 => {
                let mut fr = [0u8; 18];
                fr.copy_from_slice(&bits[..18]);
                let dec = self.w_imbe_dec.as_mut().expect("IMBE decoder present");
                let (pcm, disposition, epsilon_0, epsilon_t) = dec.decode_pcm_concealed(&fr);
                (
                    pcm.to_vec(),
                    DecodeStats {
                        epsilon_0,
                        epsilon_t,
                        disposition,
                    },
                )
            }
            Rate::Imbe4400x4400 => {
                let fr = imbe_pipeline::info_to_fec_bytes(bits);
                let dec = self.w_imbe_dec.as_mut().expect("IMBE decoder present");
                // The re-encoded FEC frame carries no channel errors, so the
                // ε-gate is inert here; an out-of-range pitch index still marks
                // the frame as an erasure and drives repeat/mute.
                let (pcm, disposition, epsilon_0, epsilon_t) = dec.decode_pcm_concealed(&fr);
                (
                    pcm.to_vec(),
                    DecodeStats {
                        epsilon_0,
                        epsilon_t,
                        disposition,
                    },
                )
            }
        }
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
    /// ```rust
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// # let pcm: Vec<i16> = vec![0; 160 * 5];
    /// let mut tx = Vocoder::new(Rate::Imbe7200x4400);
    /// let bits: Result<Vec<Vec<u8>>, _> = tx.encode_stream(&pcm).collect();
    /// assert_eq!(bits.unwrap().len(), 5);
    /// ```
    #[cfg(feature = "encode")]
    pub fn encode_stream<'a>(&'a mut self, pcm: &'a [i16]) -> EncodeStream<'a> {
        EncodeStream {
            vocoder: self,
            pcm,
            pos: 0,
        }
    }

    /// Decode an arbitrary-length FEC byte slice as a stream of PCM
    /// frames.
    ///
    /// Returns an iterator that yields one `Result<Vec<i16>>` per
    /// frame consumed ([`Self::fec_frame_bytes`] per frame). Trailing
    /// partial frames are silently dropped.
    ///
    /// ```rust
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// # let bits: Vec<u8> = vec![0; 18 * 5];
    /// let mut rx = Vocoder::new(Rate::Imbe7200x4400);
    /// let pcm_frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream(&bits).collect();
    /// assert_eq!(pcm_frames.unwrap().len(), 5);
    /// ```
    pub fn decode_stream<'a>(&'a mut self, bits: &'a [u8]) -> DecodeStream<'a> {
        DecodeStream {
            vocoder: self,
            bits,
            pos: 0,
        }
    }

    /// Decode an arbitrary-length soft-decision LLR slice as a stream of PCM
    /// frames — the soft-decision counterpart of [`Self::decode_stream`].
    ///
    /// Yields one `Result<Vec<i16>>` per [`Self::soft_frame_bits`] LLRs
    /// consumed ([`Self::decode_soft`] per frame); trailing partial frames are
    /// silently dropped. For the no-FEC rates (no soft frame size) the stream
    /// is empty — call [`Self::decode_soft`] directly to get the explicit
    /// [`VocoderError::SoftUnsupported`].
    ///
    /// ```rust
    /// # use blip25_mbe::vocoder::{Rate, Vocoder};
    /// # let llrs: Vec<i8> = vec![0; 144 * 5];
    /// let mut rx = Vocoder::new(Rate::Imbe7200x4400);
    /// let frames: Result<Vec<Vec<i16>>, _> = rx.decode_stream_soft(&llrs).collect();
    /// assert_eq!(frames.unwrap().len(), 5);
    /// ```
    pub fn decode_stream_soft<'a>(&'a mut self, llrs: &'a [i8]) -> DecodeSoftStream<'a> {
        DecodeSoftStream {
            vocoder: self,
            llrs,
            pos: 0,
        }
    }
}

/// Streaming-encode iterator returned by [`Vocoder::encode_stream`].
/// Yields one `Result<Vec<u8>>` per 160-sample input frame; trailing
/// partial frames are silently dropped.
#[cfg(feature = "encode")]
pub struct EncodeStream<'a> {
    vocoder: &'a mut Vocoder,
    pcm: &'a [i16],
    pos: usize,
}

#[cfg(feature = "encode")]
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

#[cfg(feature = "encode")]
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

/// Streaming soft-decision-decode iterator returned by
/// [`Vocoder::decode_stream_soft`]. Yields one `Result<Vec<i16>>` per
/// [`Vocoder::soft_frame_bits`] LLRs; trailing partial frames are dropped.
/// Empty for the no-FEC rates (which have no soft frame size).
pub struct DecodeSoftStream<'a> {
    vocoder: &'a mut Vocoder,
    llrs: &'a [i8],
    pos: usize,
}

impl Iterator for DecodeSoftStream<'_> {
    type Item = Result<Vec<i16>, VocoderError>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.vocoder.soft_frame_bits()?; // None (no-FEC rate) => empty stream
        if self.pos + n > self.llrs.len() {
            return None;
        }
        let frame = &self.llrs[self.pos..self.pos + n];
        self.pos += n;
        Some(self.vocoder.decode_soft(frame))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.vocoder.soft_frame_bits() {
            Some(n) => {
                let remaining = (self.llrs.len() - self.pos) / n;
                (remaining, Some(remaining))
            }
            None => (0, Some(0)),
        }
    }
}

impl ExactSizeIterator for DecodeSoftStream<'_> {}

/// Direction of a [`Transcoder`] — the input/output rate pair.
///
/// Scales O(N) for N rates instead of O(N²) enum variants. Not every
/// `(from, to)` pair has a wired parameter-domain converter; see
/// [`Transcoder::new`] for the supported set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscodeDirection {
    /// Rate of the input FEC frame.
    pub from: Rate,
    /// Rate of the output FEC frame.
    pub to: Rate,
}

impl TranscodeDirection {
    /// Construct a direction from a `(from, to)` pair.
    #[inline]
    pub const fn new(from: Rate, to: Rate) -> Self {
        Self { from, to }
    }

    /// Bytes per input FEC frame for this direction.
    #[inline]
    pub const fn input_frame_bytes(self) -> usize {
        self.from.fec_frame_bytes()
    }

    /// Bytes per output FEC frame for this direction.
    #[inline]
    pub const fn output_frame_bytes(self) -> usize {
        self.to.fec_frame_bytes()
    }
}

/// Bridge P25 Phase 1 ↔ P25 Phase 2 at the wire-bits layer.
///
/// Internally the transcoder runs the parameter-domain converter
/// (BABA-A §11 — extract params from one rate's bits, then re-quantize
/// at the other rate without any PCM round-trip). Avoids the
/// 8-kHz-PCM detour, so quality stays at the parameter-extraction
/// floor rather than going through analysis-encode → synthesis →
/// analysis-encode again.
///
/// Direction is fixed at construction. State (cross-rate predictor,
/// last-good frame for repeats) is internal and advances per call.
///
/// ```rust
/// use blip25_mbe::vocoder::{Rate, Transcoder};
///
/// let mut tx = Transcoder::new(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450).unwrap();
/// let phase1_bits: [u8; 18] = [0; 18];
/// let phase2_bits = tx.transcode(&phase1_bits).unwrap();
/// assert_eq!(phase2_bits.len(), 9);
/// ```
pub struct Transcoder {
    direction: TranscodeDirection,
    full_to_half: Option<crate::rate_conversion::FullToHalfConverter>,
    half_to_full: Option<crate::rate_conversion::HalfToFullConverter>,
}

impl Transcoder {
    /// Open a new transcoder for the `(from, to)` rate pair.
    ///
    /// Cross-codec pairs (run the parameter-domain converter):
    /// - `(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450)`
    /// - `(Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400)`
    ///
    /// Same-codec FEC ↔ no-FEC pairs (pure wire-layer bit shuffling,
    /// no codec or predictor state):
    /// - `(Rate::Imbe7200x4400, Rate::Imbe4400x4400)` — strip Annex H FEC
    /// - `(Rate::Imbe4400x4400, Rate::Imbe7200x4400)` — add Annex H FEC
    /// - `(Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450)` — strip half-rate FEC
    /// - `(Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450)` — add half-rate FEC
    ///
    /// Any other combination returns
    /// [`VocoderError::UnsupportedTranscode`].
    pub fn new(from: Rate, to: Rate) -> Result<Self, VocoderError> {
        let direction = TranscodeDirection { from, to };
        match (from, to) {
            (Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450) => Ok(Self {
                direction,
                full_to_half: Some(crate::rate_conversion::FullToHalfConverter::new()),
                half_to_full: None,
            }),
            (Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400) => Ok(Self {
                direction,
                full_to_half: None,
                half_to_full: Some(crate::rate_conversion::HalfToFullConverter::new()),
            }),
            // Same-codec FEC ↔ no-FEC pairs are stateless wire transforms;
            // both converters stay None.
            (Rate::Imbe7200x4400, Rate::Imbe4400x4400)
            | (Rate::Imbe4400x4400, Rate::Imbe7200x4400)
            | (Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450)
            | (Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450) => Ok(Self {
                direction,
                full_to_half: None,
                half_to_full: None,
            }),
            _ => Err(VocoderError::UnsupportedTranscode { from, to }),
        }
    }

    /// Direction this transcoder was opened in.
    #[inline]
    pub fn direction(&self) -> TranscodeDirection {
        self.direction
    }

    /// Transcode one input FEC frame to one output FEC frame.
    ///
    /// `bits` must be exactly [`TranscodeDirection::input_frame_bytes`]
    /// long; the returned `Vec` has exactly
    /// [`TranscodeDirection::output_frame_bytes`].
    pub fn transcode(&mut self, bits: &[u8]) -> Result<Vec<u8>, VocoderError> {
        let in_n = self.direction.input_frame_bytes();
        if bits.len() != in_n {
            return Err(VocoderError::WrongBitsLength {
                expected: in_n,
                got: bits.len(),
            });
        }
        match (self.direction.from, self.direction.to) {
            (Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450) => {
                let dibits_in = unpack_dibits_n::<72>(bits);
                let dibits_out = self
                    .full_to_half
                    .as_mut()
                    .expect("constructed with this direction")
                    .convert(&dibits_in)
                    .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
                Ok(pack_dibits_n::<36, 9>(&dibits_out).to_vec())
            }
            (Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400) => {
                let dibits_in = unpack_dibits_n::<36>(bits);
                let dibits_out = self
                    .half_to_full
                    .as_mut()
                    .expect("constructed with this direction")
                    .convert(&dibits_in)
                    .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
                Ok(pack_dibits_n::<72, 18>(&dibits_out).to_vec())
            }
            (Rate::Imbe7200x4400, Rate::Imbe4400x4400) => {
                Ok(imbe_pipeline::fec_to_info_bytes(bits).to_vec())
            }
            (Rate::Imbe4400x4400, Rate::Imbe7200x4400) => {
                Ok(imbe_pipeline::info_to_fec_bytes(bits).to_vec())
            }
            (Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450) => {
                Ok(ambe_plus2_pipeline::fec_to_info_bytes(bits).to_vec())
            }
            (Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450) => {
                Ok(ambe_plus2_pipeline::info_to_fec_bytes(bits).to_vec())
            }
            (from, to) => Err(VocoderError::UnsupportedTranscode { from, to }),
        }
    }

    /// Reset all transcoder state. Equivalent to opening a fresh
    /// channel.
    pub fn reset(&mut self) {
        *self = Self::new(self.direction.from, self.direction.to)
            .expect("direction was validated at construction");
    }
}

fn unpack_dibits_n<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
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

fn pack_dibits_n<const N: usize, const B: usize>(dibits: &[u8; N]) -> [u8; B] {
    let mut out = [0u8; B];
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

/// Look-ahead (source frames) reserved before a frame's reference voicing is
/// finalised. `b1_track` needs ~12 frames of forward context (FLUSH_LOOKAHEAD
/// ≈ 1876 samples); this is that plus a margin. It is the streaming latency.
#[cfg(feature = "encode")]
const B1_RESERVE: usize = 16;

/// Minimum newly-finalisable frames before a pump runs the windowed `b1_track`,
/// so its (bounded) context+look-ahead re-analysis is amortised over a batch
/// instead of paid per frame. Adds up to this many frames of latency.
#[cfg(feature = "encode")]
const B1_MIN_BATCH: usize = 16;

/// Voicing offset on the SOURCE-frame axis: `AmbeStream` packs source frame
/// `s` with `b1_track[s + BLAG]`.
#[cfg(feature = "encode")]
const BLAG: usize = 2;

/// Opt-in for the encoder's gated low-band spectral floor
/// (`BLIP25_LOWBAND_FLOOR=1`), read once.
///
/// The env read lives here rather than in `blip25-codec`: that crate reads no
/// environment at all, and the floor's Encoder-side controls are plain setters.
/// Off, the AMBE+2 amplitude path is byte-identical to a build without it.
#[cfg(feature = "encode")]
fn lowband_floor_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BLIP25_LOWBAND_FLOOR").as_deref() == Ok("1"))
}

/// Opt-in for the encoder's fixed-point amplitude shape quantizer
/// (`BLIP25_AMP_NEXT=1`), read once. Same rationale for the env read living
/// here as [`lowband_floor_enabled`]. Off, `b3..b8` come from the real-valued
/// forward transform and the emitted bits are byte-identical to a build
/// without it.
#[cfg(feature = "encode")]
fn amp_next_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BLIP25_AMP_NEXT").as_deref() == Ok("1"))
}

/// The same offset on the OUTPUT-frame axis. Output frame `i` is source frame
/// `i + 1` (the dropped look-ahead placeholder), so the two constants differ by
/// exactly that one frame.
///
/// This is the whole encoder's voicing convention, not an estimate: whole-buffer
/// `encode` already hard-wires it in `forced_b0` (`reference_b0[f + 2]`) and in
/// the tone branch's `(L, step)` override (`orac_b1[f + BLAG]`), and
/// [`AmbeStream`] applies the same offset per frame. Deriving the emitted `b1`'s
/// offset any other way lets one field disagree with the injections the rest of
/// the frame was built from, and cannot be reproduced by a streaming caller —
/// any data fit over the clip needs frames the stream has not received yet.
#[cfg(feature = "encode")]
const AMBE_B1_OUTPUT_LAG: i64 = BLAG as i64 + 1;

/// Single-pass stateful AMBE+2 streamer — byte-exact to whole-buffer
/// [`Vocoder::encode`] at ~`B1_RESERVE`-frame latency, the console's own
/// per-frame-streaming-with-look-ahead behaviour.
///
/// The whole-buffer reference is three cross-frame chains: reference-pitch (`b0`), the
/// blip25 amplitude Encoder (`b2..b8`, a *predictive* VQ), and `b1_track`
/// voicing (`b1`). Each is carried here as live state instead of re-derived per
/// window, so there is no predictor drift:
/// * `reference` — the persistent [`B0Audio`] tracker, advanced one frame at a
///   time with the same voicing masks as whole-buffer → byte-exact `b0`.
/// * `e` — one persistent [`Encoder`]; feeding it frames in order carries the
///   amplitude predictor → byte-exact `b2..b8` (given the same `b0`).
/// * `b1_track` — recomputed on the *growing* prefix each pump; causal + bounded
///   look-ahead makes any frame ≥ `B1_RESERVE` back byte-exact.
///
/// `pref`/`raw` grow with the transmission (from index 0 — the reference tracker
/// indexes `pref` by an internal frame counter). That is bounded for PTT; a
/// continuous stream would want [`LiveEncoder::reset`] between transmissions.
#[cfg(feature = "encode")]
struct AmbeStream {
    with_fec: bool,
    tone_on: bool,
    /// The low-band floor is on for this stream, so the Encoder needs the
    /// transmitted `b1` for every frame it analyses — which costs `BLAG` frames
    /// of emit latency it does not otherwise pay.
    floor: bool,
    e: blip25_codec::enc::Encoder,
    reference: blip25_codec::enc::b0_audio::B0Audio,
    pref_state: blip25_codec::enc::audio_prefilter::PrefilterState,
    pref: Vec<i16>,
    raw: Vec<i16>,
    reference_b0: Vec<u8>,
    /// The packer tone branch's gate per reference frame, carried alongside
    /// `reference_b0` so the amplitude Encoder can be told which frames' `b̂₀`
    /// is a tone index (see `Encoder::set_forced_l56`).
    reference_l56: Vec<bool>,
    forced_b0: Vec<u8>,
    forced_l56: Vec<bool>,
    b1: Vec<u16>,
    tone_of: Vec<Option<blip25_codec::tone::ToneFrameFields>>,
    det: blip25_codec::tone::ToneDetector,
    reference_frame: usize,
    horizon: usize,
    fed: usize,
    emitted_src: usize,
    /// Total output frames emitted so far (for `pending_samples`).
    out_count: usize,
    flushed: bool,
    pend: std::collections::VecDeque<(usize, [u8; 9])>,
    // The persistent hdr30 gate word (the one slow-converging b1_track state,
    // carried so a bounded analysis window stays byte-exact).
    hdr: Hdr30State,
    hdr30: Vec<i32>,
    /// The low-band noise rewrite's other persistent producer: the noise-tracker
    /// block window per source frame. Populated only when the stage is on.
    lb_hdr: Box<blip25_codec::enc::b1_audio::Hdr30Track>,
    lb_win: Vec<[i16; 168]>,
}

/// The persistent producer of the a4 gate word a streamer carries across pumps,
/// in whichever form `b1_track` derives it whole-buffer.
#[cfg(feature = "encode")]
enum Hdr30State {
    /// The energy VAD over the raw samples.
    Vad(blip25_codec::enc::b1_audio::Hdr30Vad),
    /// The noise tracker's own register, over the prefiltered stream. Boxed:
    /// the tracker's per-band state dwarfs the VAD's four scalars.
    Track(Box<blip25_codec::enc::b1_audio::Hdr30Track>),
}

#[cfg(feature = "encode")]
impl Hdr30State {
    fn new() -> Self {
        if blip25_codec::enc::b1_audio::noisy_next_enabled() {
            Self::Track(Box::new(blip25_codec::enc::b1_audio::Hdr30Track::new()))
        } else {
            Self::Vad(blip25_codec::enc::b1_audio::Hdr30Vad::new())
        }
    }
}

/// Extend `hdr30` with the gate word of every source frame up to `upto`,
/// advancing the persistent producer. Frames are 160 samples; a short trailing
/// frame reads as zero-padded, matching the whole-buffer derivations.
#[cfg(feature = "encode")]
fn advance_hdr30(
    hdr: &mut Hdr30State,
    hdr30: &mut Vec<i32>,
    raw: &[i16],
    pref: &[i16],
    upto: usize,
) {
    while hdr30.len() < upto {
        let v = match hdr {
            Hdr30State::Vad(v) => {
                let s = hdr30.len() * FRAME_SAMPLES;
                let end = (s + FRAME_SAMPLES).min(raw.len());
                let frame = if s < end { &raw[s..end] } else { &[][..] };
                v.push_frame(frame)
            }
            Hdr30State::Track(t) => t.push_frame(pref),
        };
        hdr30.push(v);
    }
}

#[cfg(feature = "encode")]
impl AmbeStream {
    fn new(with_fec: bool, tone_on: bool) -> Self {
        // Route A: the amplitude Encoder runs with the two-frame look-ahead so
        // its live `gap2_mid` is pitch-aligned, matching whole-buffer
        // `Vocoder::encode` on b2..b8 (not just b0/b1).
        let mut e = blip25_codec::enc::Encoder::new();
        e.set_live_gap2_amps(true);
        let floor = lowband_floor_enabled();
        e.set_lowband_floor(floor);
        e.set_amp_next(amp_next_enabled());
        Self {
            with_fec,
            tone_on,
            floor,
            e,
            reference: blip25_codec::enc::b0_audio::B0Audio::new(),
            pref_state: blip25_codec::enc::audio_prefilter::PrefilterState::default(),
            pref: Vec::new(),
            raw: Vec::new(),
            reference_b0: Vec::new(),
            reference_l56: Vec::new(),
            forced_b0: Vec::new(),
            forced_l56: Vec::new(),
            b1: Vec::new(),
            tone_of: Vec::new(),
            det: blip25_codec::tone::ToneDetector::new(),
            reference_frame: 0,
            horizon: 0,
            fed: 0,
            emitted_src: 0,
            out_count: 0,
            flushed: false,
            pend: std::collections::VecDeque::new(),
            hdr: Hdr30State::new(),
            hdr30: Vec::new(),
            lb_hdr: Box::new(blip25_codec::enc::b1_audio::Hdr30Track::new()),
            lb_win: Vec::new(),
        }
    }

    /// Advance the persistent hdr30 producer to cover source frames `[0, upto)`.
    fn advance_hdr30(&mut self, upto: usize) {
        advance_hdr30(&mut self.hdr, &mut self.hdr30, &self.raw, &self.pref, upto);
        if blip25_codec::enc::b1_audio::lb_noise_enabled() {
            while self.lb_win.len() < upto {
                let _ = self.lb_hdr.push_frame(&self.pref);
                self.lb_win.push(self.lb_hdr.window());
            }
        }
    }

    /// Prefilter + buffer new PCM, then finalise/emit whatever frames now have
    /// their look-ahead.
    fn push(&mut self, pcm: &[i16]) -> Vec<Vec<u8>> {
        if !pcm.is_empty() {
            self.flushed = false;
        }
        self.absorb(pcm);
        self.pump(false)
    }

    /// End of stream: complete a partial frame with silence and append one
    /// further silent frame to buy back the analysis look-ahead, so `n` frames
    /// of caller audio yield `n` output frames. Whole-buffer `encode` pads
    /// identically, which is what keeps the two analysing the same buffer.
    /// Repeating it without new input is a no-op.
    fn flush(&mut self) -> Vec<Vec<u8>> {
        if self.flushed {
            return Vec::new();
        }
        self.flushed = true;
        if self.raw.is_empty() {
            return Vec::new();
        }
        let want = self.raw.len().div_ceil(FRAME_SAMPLES);
        let pad = (want + 1) * FRAME_SAMPLES - self.raw.len();
        let zeros = vec![0i16; pad];
        self.absorb(&zeros);
        self.pump(true)
    }

    fn absorb(&mut self, pcm: &[i16]) {
        let (pref_new, state) =
            blip25_codec::enc::audio_prefilter::prefilter(&self.pref_state, pcm);
        self.pref_state = state;
        self.pref.extend_from_slice(&pref_new);
        self.raw.extend_from_slice(pcm);
    }

    /// Advance every chain as far as the current buffer allows and return the
    /// newly-final output frames in order. See the struct docs.
    fn pump(&mut self, flush: bool) -> Vec<Vec<u8>> {
        use blip25_codec::enc::b1_audio::{b1_track_hdr30, RingRefineMode};
        let n = FRAME_SAMPLES;
        let avail = self.raw.len() / n;
        let mut out: Vec<Vec<u8>> = Vec::new();
        if avail < 2 {
            return out;
        }
        // N input frames → N-1 emittable frames (the reference codec's one-frame
        // look-ahead), the same convention as whole-buffer `encode`.
        let nframes = avail - 1;
        let new_horizon = if flush {
            nframes
        } else {
            nframes.saturating_sub(B1_RESERVE)
        };
        if !flush && new_horizon.saturating_sub(self.horizon) < B1_MIN_BATCH {
            return out;
        }
        // (1) b1_track on a BOUNDED window (not the growing prefix — that recomputes
        //     FFT gaps in O(n²)). `b1_track` is causal + bounded look-ahead, so a
        //     window carrying `ctx` frames of warm-up before the finalise point is
        //     byte-exact for the finalised frames. `bt[k]` is global frame
        //     `win_start + k`.
        let ctx = 8usize;
        let win_start = self.horizon.saturating_sub(ctx);
        let win_nframes = (avail - win_start).saturating_sub(1);
        // Persistent hdr30 for every available frame (covers both the standalone
        // and from_logs windows) — mutate before the immutable slices below.
        self.advance_hdr30(avail);
        let win_raw = &self.raw[win_start * n..];
        let win_pref = &self.pref[win_start * n..];
        let hdr_win = &self.hdr30[win_start..win_start + win_nframes];
        // (1a) The low-band noise rewrite needs the pitch tracker's fit-error
        //      ring for the same frames, which is produced from the masks of a
        //      pass that has not run yet. The stage cannot move the masks
        //      (`ctx+0x91a` is written before it), so run the chain once for
        //      them, advance the persistent pitch tracker over the frames that
        //      are about to be finalised, then run it again with the stage's
        //      inputs. Frames past the tracker read the ring's `32767` seed;
        //      they are never finalised in this pump and are recomputed in the
        //      next one, when the tracker has reached them.
        let reference_target = if flush {
            nframes
        } else {
            (new_horizon + 2).min(nframes)
        };
        let lb_on = blip25_codec::enc::b1_audio::lb_noise_enabled();
        let mut raw_b0: Vec<(usize, u8)> = Vec::new();
        let bt = if lb_on {
            let btm = b1_track_hdr30(win_pref, win_raw, win_nframes, RingRefineMode::Off, hdr_win);
            while self.reference_frame < reference_target {
                let cf = self.reference_frame;
                let a11 = if cf == 0 {
                    0
                } else {
                    i32::from(btm[cf - 1 - win_start].mask)
                };
                let b0 = self
                    .reference
                    .push_pcm_frame_with_prev_mask(&self.pref, a11);
                raw_b0.push((cf, b0));
                self.reference_frame += 1;
            }
            let lb_win: Vec<blip25_codec::enc::b1_audio::LbNoiseIn> = (0..win_nframes)
                .map(|k| {
                    let f = win_start + k;
                    blip25_codec::enc::b1_audio::LbNoiseIn {
                        st: self.lb_win[f],
                        q0: self.reference.c62c_at(f),
                        q2: if f == 0 {
                            32767
                        } else {
                            self.reference.c62c_at(f - 1)
                        },
                    }
                })
                .collect();
            blip25_codec::enc::b1_audio::b1_track_hdr30_lb(
                win_pref,
                win_raw,
                win_nframes,
                RingRefineMode::Off,
                hdr_win,
                &lb_win,
            )
        } else {
            b1_track_hdr30(win_pref, win_raw, win_nframes, RingRefineMode::Off, hdr_win)
        };
        // (2) finalise b1 + tone (in order) for the newly-final source frames.
        for f in self.horizon..new_horizon {
            self.b1.push(bt[f - win_start].b1);
            let tf = if self.tone_on {
                self.det.process(&self.raw[f * n..f * n + n])
            } else {
                None
            };
            self.tone_of.push(tf);
        }
        // (3) advance the persistent reference-pitch tracker (uses only final masks).
        for &(cf, b0) in &raw_b0 {
            let fr = &bt[cf - win_start];
            self.reference_b0
                .push(blip25_codec::enc::pcm_encode::b0_with_tone_branch(b0, fr));
            self.reference_l56
                .push(blip25_codec::enc::pcm_encode::tone_branch_l56(fr.b1));
        }
        while self.reference_frame < reference_target {
            let cf = self.reference_frame;
            let a11 = if cf == 0 {
                0
            } else {
                i32::from(bt[cf - 1 - win_start].mask)
            };
            let b0 = self.reference.push_pcm_frame_with_prev_mask(&self.pref, a11);
            let fr = &bt[cf - win_start];
            self.reference_b0
                .push(blip25_codec::enc::pcm_encode::b0_with_tone_branch(b0, fr));
            self.reference_l56
                .push(blip25_codec::enc::pcm_encode::tone_branch_l56(fr.b1));
            self.reference_frame += 1;
        }
        // (4) forced-b0 for the persistent amplitude Encoder (whole-buffer maps
        //     emit source `src` to reference frame `src + 2`, clamped).
        let emit_src_max = if flush {
            nframes
        } else if self.floor {
            // The floor's gate input for source `src` is the TRANSMITTED
            // `b1[src + BLAG]`, final only through `new_horizon - 1`. Emitting
            // through `new_horizon - 1 - BLAG` costs nothing: step (6) already
            // holds every frame with `s + BLAG >= b1.len()`.
            new_horizon.saturating_sub(1 + BLAG)
        } else {
            new_horizon.saturating_sub(1)
        };
        while self.forced_b0.len() <= emit_src_max {
            let src = self.forced_b0.len();
            let idx = (src + 2).min(self.reference_b0.len().saturating_sub(1));
            self.forced_b0.push(self.reference_b0[idx]);
            self.forced_l56.push(self.reference_l56[idx]);
        }
        self.e.set_forced_b0(self.forced_b0.clone());
        self.e.set_forced_l56(self.forced_l56.clone());
        if self.floor && !self.b1.is_empty() {
            // Same lag `pack` applies, so the gate sees the `b1` that is
            // actually transmitted rather than the Encoder's internal estimate.
            let last = self.b1.len() - 1;
            let gate_b1: Vec<u16> = (0..=emit_src_max)
                .map(|src| self.b1[(src + BLAG).min(last)])
                .collect();
            self.e.set_lowband_gate_b1(gate_b1);
        }
        // (5) feed raw frames to the Encoder; each emission is the next source.
        //     Route A's two-frame look-ahead means emission `s` (analyze frame
        //     `s`) lands only after frame `s + 2` is fed, so to have sources up
        //     to `emit_src_max` in `pend` we must feed through `emit_src_max + 3`
        //     (one deeper than the old one-frame look-ahead's `+ 2`).
        let feed_to = if flush {
            avail
        } else {
            (emit_src_max + 3).min(avail)
        };
        while self.fed < feed_to {
            let fr: [i16; FRAME_SAMPLES] =
                self.raw[self.fed * n..self.fed * n + n].try_into().unwrap();
            if let Some(r33) = self.e.encode_frame_r33(&fr) {
                let s = self.emitted_src;
                self.emitted_src += 1;
                self.pend.push_back((s, r33));
            }
            self.fed += 1;
        }
        if flush {
            for r33 in self.e.flush_r33() {
                let s = self.emitted_src;
                self.emitted_src += 1;
                self.pend.push_back((s, r33));
            }
        }
        // (6) pack + emit finalised frames in order; drop source-0 (the dropped
        //     look-ahead placeholder — whole-buffer `frames.remove(0)`). Hold a
        //     frame back until its lagged voicing `b1[s + BLAG]` is final (the
        //     tail at flush keeps the Encoder's own b1, as whole-buffer does).
        while let Some(&(s, _)) = self.pend.front() {
            if !flush && s + BLAG >= self.b1.len() {
                break;
            }
            let (s, r33) = self.pend.pop_front().unwrap();
            if s == 0 {
                continue;
            }
            out.push(self.pack(s, &r33));
            self.out_count += 1;
        }
        self.horizon = new_horizon;
        out
    }

    /// Apply the reference overrides (voicing b1, tone overlay) to the Encoder's
    /// r33 bytes for source frame `s` and pack to r33/r34 — the same steps as
    /// whole-buffer `encode_ambe2_reference_voicing`.
    fn pack(&self, s: usize, r33: &[u8; 9]) -> Vec<u8> {
        // Annex-T tone overlay (default off): byte-exact tone frame where fired.
        if let Some(t) = self.tone_of.get(s).copied().flatten() {
            let tr = Vocoder::tone_frame_bytes(t.id, t.amplitude);
            return if self.with_fec {
                tr
            } else {
                let tb = blip25_codec::tables::deprioritize(
                    &blip25_codec::frame::decode_bytes(&tr).info,
                );
                blip25_codec::enc::encode_frame::encode_r34(&tb).to_vec()
            };
        }
        let mut b =
            blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(r33).info);
        // Output voicing from the standalone b1_track (byte-exact on speech; the
        // whole-buffer's forced `from_logs`+lag-search voicing diverges on pure
        // tones, which isn't reproducible per-frame without the global lag search).
        if let Some(&v) = self.b1.get(s + BLAG) {
            b[1] = v;
        }
        if self.with_fec {
            blip25_codec::enc::encode_frame::encode_r33(&b).to_vec()
        } else {
            blip25_codec::enc::encode_frame::encode_r34(&b).to_vec()
        }
    }
}

/// Frames of warm-up context a bounded `b1_track` window carries before its
/// first finalised frame. `b1_track` is causal with bounded look-ahead, so a
/// window that starts this far back reproduces the whole-buffer answer for
/// every frame it finalises. Measured: 4 diverges on one frame of the 1273-frame
/// DVSI speech vector, 8 and above are clean.
#[cfg(feature = "encode")]
const B1_CTX: usize = 8;

/// Single-pass stateful IMBE streamer — byte-exact to whole-buffer
/// [`Vocoder::encode`] at ~`B1_RESERVE`-frame latency.
///
/// IMBE consumes the whole-buffer analysis on the *input* side: `forced_b0`
/// (reference spectral-DP pitch behind a silence gate) and `forced_b1` (the
/// reference voicing word) are read inside `analyze_imbe_frame`, where they
/// drive the per-harmonic V/UV expansion and the amplitude quantizer. There is
/// therefore no post-hoc repack as on the AMBE+2 side — a frame cannot be
/// emitted until both forced entries covering it are final.
///
/// Each cross-frame chain is carried as live state rather than re-derived per
/// window, so nothing drifts:
/// * `reference` — the persistent [`B0Audio`] tracker, advanced one frame at a
///   time with the same masks `encode_pcm_b0` uses → byte-exact `b0`.
/// * `e` — one persistent [`Encoder`], fed in order → byte-exact amplitudes.
/// * `hdr` — the persistent hdr30 producer, the one slow-converging
///   `b1_track` state.
///
/// Unlike [`AmbeStream`], no source frame is dropped: `nf` input frames yield
/// `nf` output frames, matching whole-buffer `encode`.
#[cfg(feature = "encode")]
struct ImbeStream {
    /// `Imbe4400x4400` ships the 88 info bits without the FEC parity.
    info_only: bool,
    e: blip25_codec::enc::Encoder,
    reference: blip25_codec::enc::b0_audio::B0Audio,
    pref_state: blip25_codec::enc::audio_prefilter::PrefilterState,
    pref: Vec<i16>,
    raw: Vec<i16>,
    hdr: Hdr30State,
    hdr30: Vec<i32>,
    reference_b0: Vec<u8>,
    reference_frame: usize,
    b1: Vec<u16>,
    /// Both indexed by ANALYSIS frame, which for IMBE is also the output index.
    forced_b0: Vec<u8>,
    forced_b1: Vec<u16>,
    horizon: usize,
    fed: usize,
    out_count: usize,
    flushed: bool,
}

#[cfg(feature = "encode")]
impl ImbeStream {
    fn new(info_only: bool) -> Self {
        // The whole-buffer IMBE arm runs the two-frame look-ahead so the live
        // `gap2_mid` window feeding the mechanism amplitudes is pitch-aligned;
        // match it here or the amplitude fields diverge.
        let mut e = blip25_codec::enc::Encoder::new();
        e.set_live_gap2_amps(true);
        // The forced vectors grow as the analysis finalises, so a feed that runs
        // ahead of them must fail loudly rather than quietly encode a frame with
        // the estimator — that is exactly the fricative buzz coming back.
        e.set_forced_strict(true);
        Self {
            info_only,
            e,
            reference: blip25_codec::enc::b0_audio::B0Audio::new(),
            pref_state: blip25_codec::enc::audio_prefilter::PrefilterState::default(),
            pref: Vec::new(),
            raw: Vec::new(),
            hdr: Hdr30State::new(),
            hdr30: Vec::new(),
            reference_b0: Vec::new(),
            reference_frame: 0,
            b1: Vec::new(),
            forced_b0: Vec::new(),
            forced_b1: Vec::new(),
            horizon: 0,
            fed: 0,
            out_count: 0,
            flushed: false,
        }
    }

    /// Prefilter + buffer new PCM, then emit whatever frames are now final.
    fn push(&mut self, pcm: &[i16]) -> Vec<Vec<u8>> {
        if !pcm.is_empty() {
            self.flushed = false;
        }
        self.absorb(pcm);
        self.pump(false)
    }

    /// End of stream: complete a partial frame with silence so the trailing
    /// audio is carried, then finalise everything and drain the Encoder.
    /// Repeating it without new input is a no-op — the silence completion must
    /// happen once, or the second flush would append a whole extra frame.
    fn flush(&mut self) -> Vec<Vec<u8>> {
        if self.flushed {
            return Vec::new();
        }
        self.flushed = true;
        let residue = self.raw.len() % FRAME_SAMPLES;
        if residue != 0 {
            let pad = vec![0i16; FRAME_SAMPLES - residue];
            self.absorb(&pad);
        }
        self.pump(true)
    }

    fn absorb(&mut self, pcm: &[i16]) {
        let (pref_new, state) =
            blip25_codec::enc::audio_prefilter::prefilter(&self.pref_state, pcm);
        self.pref_state = state;
        self.pref.extend_from_slice(&pref_new);
        self.raw.extend_from_slice(pcm);
    }

    /// Feed source frames `[self.fed, feed_to)` to the Encoder, then drain its
    /// look-ahead when `flush`. Every emission is the next analysis frame.
    fn feed(&mut self, feed_to: usize, flush: bool, out: &mut Vec<Vec<u8>>) {
        let n = FRAME_SAMPLES;
        let before = out.len();
        while self.fed < feed_to {
            let fr: [i16; FRAME_SAMPLES] =
                self.raw[self.fed * n..self.fed * n + n].try_into().unwrap();
            if let Some(b) = self.e.encode_imbe_frame(&fr) {
                out.push(self.pack(&b));
            }
            self.fed += 1;
        }
        if flush {
            for b in self.e.flush_imbe() {
                out.push(self.pack(&b));
            }
        }
        self.out_count += out.len() - before;
    }

    fn pack(&self, b: &[u8; blip25_codec::imbe::FRAME_BYTES]) -> Vec<u8> {
        if self.info_only {
            imbe_fec_to_info_bytes(b).to_vec()
        } else {
            b.to_vec()
        }
    }

    /// Advance every chain as far as the buffer allows and return the newly
    /// final output frames in order. See the struct docs.
    fn pump(&mut self, flush: bool) -> Vec<Vec<u8>> {
        use blip25_codec::enc::b1_audio::{b1_track_hdr30, RingRefineMode};
        let n = FRAME_SAMPLES;
        let avail = self.raw.len() / n;
        let mut out: Vec<Vec<u8>> = Vec::new();
        if avail < 2 {
            // Below the reference analysis' minimum context. Whole-buffer
            // `encode` leaves both forced vectors unset here too, so the
            // Encoder's own estimators drive the single frame.
            if flush && avail == 1 {
                self.feed(1, true, &mut out);
            }
            return out;
        }
        // N input frames → N-1 analysable frames (the reference analysis'
        // one-frame look-ahead), the same convention as `encode_pcm_b0`.
        let nframes = avail - 1;
        let new_horizon = if flush {
            nframes
        } else {
            nframes.saturating_sub(B1_RESERVE)
        };
        if !flush && new_horizon.saturating_sub(self.horizon) < B1_MIN_BATCH {
            return out;
        }
        // (1) b1_track on a BOUNDED window carrying `B1_CTX` frames of warm-up.
        //     `bt[k]` is global frame `win_start + k`.
        let win_start = self.horizon.saturating_sub(B1_CTX);
        let win_nframes = (avail - win_start).saturating_sub(1);
        self.advance_hdr30(avail);
        let bt = b1_track_hdr30(
            &self.pref[win_start * n..],
            &self.raw[win_start * n..],
            win_nframes,
            RingRefineMode::Off,
            &self.hdr30[win_start..win_start + win_nframes],
        );
        // (2) finalise the voicing word for the newly-final analysis frames.
        for f in self.horizon..new_horizon {
            self.b1.push(bt[f - win_start].b1);
        }
        // (3) advance the persistent reference-pitch tracker (final masks only).
        let reference_target = if flush { nframes } else { new_horizon };
        while self.reference_frame < reference_target {
            let cf = self.reference_frame;
            let a11 = if cf == 0 {
                0
            } else {
                i32::from(bt[cf - 1 - win_start].mask)
            };
            let b0 = self
                .reference
                .push_pcm_frame_with_prev_mask(&self.pref, a11);
            self.reference_b0.push(b0);
            self.reference_frame += 1;
        }
        // (4) grow the forced vectors over the analysis frames whose inputs are
        //     final. Whole-buffer reads frame f's OWN audio for the silence gate
        //     but frame f+2 for the pitch and voicing; reproduce that asymmetry
        //     exactly. The `.min(len - 1)` clamps are whole-buffer's tail
        //     behaviour and must bite only at flush.
        let covered = if flush {
            avail
        } else {
            new_horizon.saturating_sub(2)
        };
        while self.forced_b0.len() < covered {
            let f = self.forced_b0.len();
            debug_assert!(
                flush || (f + 2 < self.b1.len() && f + 2 < self.reference_b0.len()),
                "ImbeStream finalised analysis frame {f} before its (f+2) inputs"
            );
            let chunk = &self.raw[f * n..f * n + n];
            let ss: f64 = chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let rms = (ss / FRAME_SAMPLES as f64).sqrt();
            let b0 = if rms < IMBE_SILENCE_RMS {
                IMBE_SILENCE_B0
            } else {
                self.reference_b0[(f + 2).min(self.reference_b0.len() - 1)]
            };
            self.forced_b0.push(b0);
            self.forced_b1.push(self.b1[(f + 2).min(self.b1.len() - 1)]);
        }
        self.e.set_forced_b0(self.forced_b0.clone());
        self.e.set_forced_b1(self.forced_b1.clone());
        // (5) feed. The two-frame look-ahead means feeding source `k` emits
        //     analysis frame `k - 2`, so covering analysis frames `< covered`
        //     needs sources through `covered + 2`.
        let feed_to = if flush {
            avail
        } else {
            (covered + 2).min(avail)
        };
        self.feed(feed_to, flush, &mut out);
        self.horizon = new_horizon;
        out
    }

    fn advance_hdr30(&mut self, upto: usize) {
        advance_hdr30(&mut self.hdr, &mut self.hdr30, &self.raw, &self.pref, upto);
    }
}

/// Push-driven encoder for live PCM streams that arrive in chunks
/// of arbitrary length (audio device callbacks, file readers, sockets).
/// Holds residual samples internally across calls so the caller can
/// push whatever they have and harvest frames as they become available.
///
/// One-frame-at-a-time `Vocoder` is the right primitive for callers
/// that already have whole-buffer PCM. `LiveEncoder` is for callers
/// that don't.
///
/// The streamer reproduces the whole-buffer reference analysis causally, so it
/// holds a bounded look-ahead before a frame's bits are final and emits in
/// bursts rather than one frame per 160 samples. [`Self::pending_samples`]
/// reports what is still held. A caller that needs a frame per 20 ms with no
/// look-ahead wants the [`Vocoder::encode_pcm`] primitive instead, and gets
/// lower quality for it.
///
/// ```rust
/// # use blip25_mbe::vocoder::{LiveEncoder, Rate};
/// let mut enc = LiveEncoder::new(Rate::Imbe7200x4400);
/// // 256 samples (audio-device callback); not a multiple of 160.
/// let chunk: [i16; 256] = [0; 256];
/// let frames = enc.push(&chunk);
/// assert!(frames.is_empty());                    // still inside the look-ahead
/// assert_eq!(enc.pending_samples(), 256);
/// // End of stream: flush completes the partial frame with silence and
/// // finalises everything held, so the trailing audio is never lost.
/// let tail = enc.flush().unwrap();
/// assert_eq!(tail.len(), 2);                     // ceil(256 / 160)
/// assert_eq!(enc.pending_samples(), 0);
/// ```
#[cfg(feature = "encode")]
pub struct LiveEncoder {
    vocoder: Vocoder,
    /// The single-pass stateful streamer for the configured rate — byte-exact
    /// to whole-buffer `encode` at ~`B1_RESERVE`-frame latency.
    stream: Stream,
}

/// The rate-appropriate single-pass streamer. The two differ in where the
/// reference analysis binds (AMBE+2 repacks the emitted frame, IMBE forces the
/// analysis inputs), how many source frames are dropped (one vs none) and how
/// the tail is clamped, so they are separate implementations rather than one
/// generic driven by a config.
#[cfg(feature = "encode")]
enum Stream {
    Ambe(AmbeStream),
    Imbe(ImbeStream),
}

#[cfg(feature = "encode")]
impl Stream {
    fn push(&mut self, pcm: &[i16]) -> Vec<Vec<u8>> {
        match self {
            Stream::Ambe(s) => s.push(pcm),
            Stream::Imbe(s) => s.push(pcm),
        }
    }

    fn flush(&mut self) -> Vec<Vec<u8>> {
        match self {
            Stream::Ambe(s) => s.flush(),
            Stream::Imbe(s) => s.flush(),
        }
    }

    fn received(&self) -> usize {
        match self {
            Stream::Ambe(s) => s.raw.len(),
            Stream::Imbe(s) => s.raw.len(),
        }
    }

    /// Samples already accounted for by an emitted frame. AMBE+2 additionally
    /// consumes the dropped source-0 look-ahead placeholder once anything has
    /// emitted; IMBE drops no source frame.
    fn accounted(&self) -> usize {
        match self {
            Stream::Ambe(s) => {
                let placeholder = if s.out_count > 0 { FRAME_SAMPLES } else { 0 };
                s.out_count * FRAME_SAMPLES + placeholder
            }
            Stream::Imbe(s) => s.out_count * FRAME_SAMPLES,
        }
    }

    fn cold(&self) -> Self {
        match self {
            Stream::Ambe(s) => Stream::Ambe(AmbeStream::new(s.with_fec, s.tone_on)),
            Stream::Imbe(s) => Stream::Imbe(ImbeStream::new(s.info_only)),
        }
    }
}

#[cfg(feature = "encode")]
impl LiveEncoder {
    /// Open a new live encoder at the given rate, all state cold.
    pub fn new(rate: Rate) -> Self {
        let stream = match rate {
            Rate::AmbePlus2_3600x2450 => Stream::Ambe(AmbeStream::new(true, false)),
            Rate::AmbePlus2_2450x2450 => Stream::Ambe(AmbeStream::new(false, false)),
            Rate::Imbe7200x4400 => Stream::Imbe(ImbeStream::new(false)),
            _ => Stream::Imbe(ImbeStream::new(true)),
        };
        Self {
            vocoder: Vocoder::new(rate),
            stream,
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
        self.stream.push(pcm).into_iter().map(Ok).collect()
    }

    /// Samples received but not yet represented by an emitted frame: the
    /// sub-frame residue plus the bounded look-ahead the streamer holds before
    /// a frame's reference analysis is final. Drains to 0 after [`Self::flush`].
    #[inline]
    pub fn pending_samples(&self) -> usize {
        self.stream
            .received()
            .saturating_sub(self.stream.accounted())
    }

    /// Drop any pending samples without encoding them. Useful at
    /// stream shutdown when the caller doesn't want a partial-frame
    /// flush. Resets the streamer to a cold state.
    #[inline]
    pub fn discard_pending(&mut self) {
        self.stream = self.stream.cold();
    }

    /// Finish the stream: encode any pending residue (zero-padded to a
    /// full frame) and drain the reference codec's one-frame look-ahead,
    /// returning every remaining FEC frame in order.
    ///
    /// Returns a (possibly empty) `Vec` of frames because the blip25_codec
    /// look-ahead makes it impossible to bound the tail to a single
    /// frame: encoding the padded residue emits the *previously* buffered
    /// frame, and [`Vocoder::flush_encode`] then yields the residue frame
    /// itself — so a residue flush legitimately produces two frames, and
    /// an exact-multiple flush produces the one look-ahead frame. An empty
    /// buffer with nothing buffered in the codec returns an empty `Vec`.
    ///
    /// Call this once at end-of-stream so the trailing 20 ms of audio
    /// isn't lost; the only cost is at most one frame of zero-padding
    /// tacked onto the last word of audio when residue is present.
    ///
    /// Concatenating all [`Self::push`] outputs with this `flush`, then
    /// dropping the leading frame-0 look-ahead placeholder, is
    /// byte-identical to encoding the same PCM straight through
    /// `blip25_codec::enc::Encoder` (`encode_frame_r33` per frame + `flush_r33`).
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>, VocoderError> {
        Ok(self.stream.flush())
    }

    /// Reset all state — both the inner [`Vocoder`] (predictor /
    /// look-ahead / synth substates) and the streamer.
    pub fn reset(&mut self) {
        self.vocoder.reset();
        self.stream = self.stream.cold();
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
/// ```rust
/// # use blip25_mbe::vocoder::{LiveDecoder, Rate};
/// let mut dec = LiveDecoder::new(Rate::AmbePlus2_3600x2450);
/// let chunk: [u8; 23] = [0; 23];   // 2 full 9-byte frames + 5 byte residue
/// let frames = dec.push(&chunk);
/// assert_eq!(frames.len(), 2);
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

/// Fluent builder for [`Vocoder`]. Configures the rate and the optional
/// post-decode enhancement chain in one expression instead of `new` + a
/// setter.
///
/// ```rust
/// use blip25_mbe::vocoder::{Rate, Vocoder};
///
/// let tx = Vocoder::builder(Rate::AmbePlus2_3600x2450).build();
/// assert_eq!(tx.rate(), Rate::AmbePlus2_3600x2450);
/// ```
#[derive(Clone, Debug)]
pub struct VocoderBuilder {
    rate: Rate,
    enhancement: EnhancementMode,
}

impl VocoderBuilder {
    /// New builder defaulting to the same configuration as [`Vocoder::new`]:
    /// post-decode enhancement is [`EnhancementMode::None`], so the default
    /// build is the reference console codec unaltered (see
    /// [`WHY_THE_REFERENCE_CODEC.md`](https://github.com/openBLIP25/blip25-mbe/blob/main/docs/WHY_THE_REFERENCE_CODEC.md)). Opt into the Classical research
    /// post-filter with `.enhancement(EnhancementMode::Classical(..))`.
    #[inline]
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            enhancement: EnhancementMode::None,
        }
    }

    /// Configure the post-decoder enhancement chain.
    /// See [`Vocoder::set_enhancement`].
    #[inline]
    pub fn enhancement(mut self, mode: EnhancementMode) -> Self {
        self.enhancement = mode;
        self
    }

    /// Materialize the [`Vocoder`].
    pub fn build(self) -> Vocoder {
        let mut v = Vocoder::new(self.rate);
        v.set_enhancement(self.enhancement);
        v
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

/// Strip the IMBE Golay/Hamming/PN FEC wire layer: an 18-byte full-rate frame
/// → the 11-byte info-only ([`Rate::Imbe4400x4400`]) frame. Pure bit-shuffling
/// (no codec state). Public so external parity / round-trip harnesses can build
/// the info-only wire bytes that the reference [`blip25_codec`] IMBE encoder's
/// 18-byte output maps to.
pub fn imbe_fec_to_info_bytes(fec_bytes: &[u8; 18]) -> [u8; 11] {
    imbe_pipeline::fec_to_info_bytes(fec_bytes)
}

mod imbe_pipeline {
    use super::{pack_info_bits, unpack_info_bits};
    use crate::imbe7200::frame::{decode_frame, encode_frame, INFO_WIDTHS};

    fn pack_info_full(info: &[u16; 8]) -> [u8; 11] {
        let mut out = [0u8; 11];
        pack_info_bits(info, &INFO_WIDTHS, &mut out);
        out
    }

    fn unpack_info_full(bytes: &[u8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        unpack_info_bits(bytes, &INFO_WIDTHS, &mut out);
        out
    }

    /// Wire-layer transcode: 18-byte IMBE FEC frame -> 11-byte info-only frame.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 11] {
        let dibits = unpack_dibits_full(fec_bytes);
        let frame = decode_frame(&dibits);
        pack_info_full(&frame.info)
    }

    /// Wire-layer transcode: 11-byte IMBE info-only frame -> 18-byte FEC frame.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 18] {
        let info = unpack_info_full(info_bytes);
        let dibits = encode_frame(&info);
        pack_dibits_full(&dibits)
    }

    /// Soft-decoded info vectors -> 18-byte FEC frame for the blip25_codec decoder.
    /// Same re-encode as [`info_to_fec_bytes`], straight off the `[u16; 8]`
    /// the soft FEC decoder returns (no info-byte round-trip).
    pub(super) fn info_vec_to_fec_bytes(info: &[u16; 8]) -> [u8; 18] {
        pack_dibits_full(&encode_frame(info))
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

mod ambe_plus2_pipeline {
    use crate::rate33::frame::{decode_frame, encode_frame, pack_no_fec, unpack_no_fec};

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

    fn pack_info_half(info: &[u16; 4]) -> [u8; 7] {
        pack_no_fec(info)
    }

    fn unpack_info_half(bytes: &[u8]) -> [u16; 4] {
        unpack_no_fec(bytes)
    }

    /// Wire-layer transcode: 9-byte AMBE+2 FEC frame -> 7-byte info-only frame.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 7] {
        let dibits = unpack_dibits_half(fec_bytes);
        let frame = decode_frame(&dibits);
        pack_info_half(&frame.info)
    }

    /// Wire-layer transcode: 7-byte AMBE+2 info-only frame -> 9-byte FEC frame.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 9] {
        let info = unpack_info_half(info_bytes);
        let dibits = encode_frame(&info);
        pack_dibits_half(&dibits)
    }

    /// Soft-decoded info vectors -> 9-byte FEC frame for the blip25_codec decoder.
    /// Same re-encode as [`info_to_fec_bytes`], straight off the `[u16; 4]`
    /// the soft FEC decoder returns.
    pub(super) fn info_vec_to_fec_bytes(info: &[u16; 4]) -> [u8; 9] {
        pack_dibits_half(&encode_frame(info))
    }
}

/// Pack an info-bit vector MSB-first into a byte buffer. Bit `k` of
/// `info[i]` is the high-order bit of that field; remaining bytes after
/// the last info bit are left at their initial value (callers pass a
/// zero-initialized slice when they want trailing pad bits to be zero).
fn pack_info_bits<const N: usize>(info: &[u16; N], widths: &[u8; N], out: &mut [u8]) {
    let mut bit_idx = 0usize;
    for i in 0..N {
        let w = widths[i] as usize;
        let v = info[i];
        for k in (0..w).rev() {
            let b = ((v >> k) & 1) as u8;
            out[bit_idx / 8] |= b << (7 - (bit_idx % 8));
            bit_idx += 1;
        }
    }
}

fn unpack_info_bits<const N: usize>(bytes: &[u8], widths: &[u8; N], out: &mut [u16; N]) {
    let mut bit_idx = 0usize;
    for i in 0..N {
        let w = widths[i] as usize;
        let mut v = 0u16;
        for _ in 0..w {
            let b = (bytes[bit_idx / 8] >> (7 - (bit_idx % 8))) & 1;
            v = (v << 1) | u16::from(b);
            bit_idx += 1;
        }
        out[i] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Vocoder::encode` (AMBE+2 r33) detects a clean synthesized tone and
    /// emits Annex-T tone frames (byte-exact for the given `id`/`A_D`), while
    /// the surrounding path stays voice. Proves the encode-side tone overlay is
    /// wired and produces frames the decoder classifies as tones.
    #[test]
    fn encode_emits_tone_frames_for_a_tone() {
        // Synthesize a Knox dual tone (id145: 603.2 / 1055.6 Hz) at a mid level.
        let (f1, f2) = (603.21_f64, 1055.62_f64);
        let peak = 3000.0_f64;
        let n = 20 * FRAME_SAMPLES;
        let pcm: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64;
                let s = peak * (2.0 * std::f64::consts::PI * f1 * t / 8000.0).sin()
                    + peak * (2.0 * std::f64::consts::PI * f2 * t / 8000.0 + 0.7).sin();
                s.round().clamp(-32768.0, 32767.0) as i16
            })
            .collect();
        let mut enc = Vocoder::new(Rate::AmbePlus2_3600x2450);
        enc.set_tone_detection(true); // opt-in: detection is off by default now
        let frames = enc.encode(&pcm);
        let tone_frames = frames
            .iter()
            .filter(|f| {
                let info = blip25_codec::frame::decode_bytes(f).info;
                blip25_codec::tone::classify(&info) == blip25_codec::tone::FrameKind::Tone
            })
            .count();
        // The steady body (all but a couple of onset frames) must be tone.
        assert!(
            tone_frames >= frames.len() - 4,
            "only {tone_frames}/{} tone",
            frames.len()
        );
        // And they carry id145.
        let ids: std::collections::BTreeSet<u8> = frames
            .iter()
            .filter_map(|f| {
                let info = blip25_codec::frame::decode_bytes(f).info;
                blip25_codec::tone::parse_tone_frame(&info).map(|t| t.id)
            })
            .collect();
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![145]);

        // With tone detection disabled, the same input encodes as voice
        // (no tone-signature frames).
        let mut enc_off = Vocoder::new(Rate::AmbePlus2_3600x2450);
        enc_off.set_tone_detection(false);
        assert!(!enc_off.tone_detection());
        let voice_only = enc_off.encode(&pcm);
        let tone_frames_off = voice_only
            .iter()
            .filter(|f| {
                blip25_codec::tone::classify(&blip25_codec::frame::decode_bytes(f).info)
                    == blip25_codec::tone::FrameKind::Tone
            })
            .count();
        assert_eq!(
            tone_frames_off, 0,
            "tone detection off must emit no tone frames"
        );
    }

    /// The `L = 56` gate must be a property of the codebook, not a threshold
    /// we asserted. Over the 32 rows the structural predicate ("no band
    /// carries code 1, some band carries code 2") must select exactly
    /// `b̂₁ ∈ {18..=31}`.
    #[test]
    fn l56_gate_selects_exactly_b1_18_through_31() {
        use crate::rate33::dequantize::l56_gate_from_b1;
        let fired: Vec<u8> = (0u8..32).filter(|&b| l56_gate_from_b1(b)).collect();
        let want: Vec<u8> = (18u8..32).collect();
        assert_eq!(fired, want, "gate does not match the codebook's structure");
    }

    fn periodic_pcm(period: usize, amplitude: i16) -> [i16; FRAME_SAMPLES] {
        let mut out = [0i16; FRAME_SAMPLES];
        for (n, slot) in out.iter_mut().enumerate() {
            let phase = (n % period) as f32 / period as f32;
            *slot = (amplitude as f32 * (2.0 * core::f32::consts::PI * phase).sin()) as i16;
        }
        out
    }

    #[test]
    fn encode_produces_valid_decodable_frames_all_rates() {
        let mut pcm: Vec<i16> = Vec::new();
        for _ in 0..12 {
            pcm.extend_from_slice(&periodic_pcm(40, 8000));
        }
        // The one encode path works for every rate: PCM in -> decodable bits out.
        for rate in [
            Rate::AmbePlus2_3600x2450,
            Rate::AmbePlus2_2450x2450,
            Rate::Imbe7200x4400,
            Rate::Imbe4400x4400,
        ] {
            let frames = Vocoder::new(rate).encode(&pcm);
            assert!(!frames.is_empty(), "{rate:?} should emit frames");
            let want = rate.fec_frame_bytes();
            assert!(
                frames.iter().all(|f| f.len() == want),
                "{rate:?} frames are {want} bytes"
            );
            let mut dec = Vocoder::new(rate);
            for f in &frames {
                assert_eq!(dec.decode_bits(f).unwrap().len(), FRAME_SAMPLES);
            }
        }
    }

    #[test]
    fn rate_byte_sizes_match_wire_layouts() {
        assert_eq!(Rate::Imbe7200x4400.fec_frame_bytes(), 18);
        assert_eq!(Rate::Imbe4400x4400.fec_frame_bytes(), 11);
        assert_eq!(Rate::AmbePlus2_3600x2450.fec_frame_bytes(), 9);
        assert_eq!(Rate::AmbePlus2_2450x2450.fec_frame_bytes(), 7);
        assert_eq!(Rate::Imbe7200x4400.frame_samples(), 160);
        assert_eq!(Rate::AmbePlus2_3600x2450.frame_samples(), 160);
    }

    #[test]
    fn no_fec_imbe_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::Imbe4400x4400);
        let mut rx = Vocoder::new(Rate::Imbe4400x4400);
        for _ in 0..5 {
            let pcm = periodic_pcm(40, 8000);
            let bits = tx.encode_pcm(&pcm).expect("encode");
            assert_eq!(
                bits.len(),
                11,
                "no-FEC IMBE wire frame is 11 bytes (88 info bits)"
            );
            let out = rx.decode_bits(&bits).expect("decode");
            assert_eq!(out.len(), FRAME_SAMPLES);
        }
    }

    #[test]
    fn no_fec_ambeplus2_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_2450x2450);
        let mut rx = Vocoder::new(Rate::AmbePlus2_2450x2450);
        for _ in 0..5 {
            let pcm = periodic_pcm(40, 6000);
            let bits = tx.encode_pcm(&pcm).expect("encode");
            assert_eq!(
                bits.len(),
                7,
                "no-FEC AMBE+2 wire frame is 7 bytes (49 info bits + 7 pad)"
            );
            let out = rx.decode_bits(&bits).expect("decode");
            assert_eq!(out.len(), FRAME_SAMPLES);
        }
    }

    /// FEC and no-FEC variants should reach the same MbeParams (same
    /// codec underneath) — verified by encoding both ways and decoding
    /// the no-FEC bits, which must give identical PCM to a clean FEC
    /// roundtrip when no channel errors are injected.
    #[test]
    fn no_fec_full_matches_fec_full_on_clean_channel() {
        let mut tx_fec = Vocoder::new(Rate::Imbe7200x4400);
        let mut rx_fec = Vocoder::new(Rate::Imbe7200x4400);
        let mut tx_raw = Vocoder::new(Rate::Imbe4400x4400);
        let mut rx_raw = Vocoder::new(Rate::Imbe4400x4400);
        // Prime the reference encoder's one-frame look-ahead: the first
        // encode_pcm call returns a look-ahead-fill placeholder, so warm both
        // encoders once (both then emit the previous frame's real bits, which
        // stay perfectly aligned since the input is identical).
        let warm = periodic_pcm(39, 8000);
        let _ = tx_fec.encode_pcm(&warm);
        let _ = tx_raw.encode_pcm(&warm);
        for k in 0..6 {
            let pcm = periodic_pcm(40 + k, 8000);
            let pcm_fec = rx_fec
                .decode_bits(&tx_fec.encode_pcm(&pcm).unwrap())
                .unwrap();
            let pcm_raw = rx_raw
                .decode_bits(&tx_raw.encode_pcm(&pcm).unwrap())
                .unwrap();
            assert_eq!(
                pcm_fec, pcm_raw,
                "no-FEC and FEC paths must match on a clean channel (frame {k})"
            );
        }
    }

    #[test]
    fn no_fec_half_matches_fec_half_on_clean_channel() {
        let mut tx_fec = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut rx_fec = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut tx_raw = Vocoder::new(Rate::AmbePlus2_2450x2450);
        let mut rx_raw = Vocoder::new(Rate::AmbePlus2_2450x2450);
        // Prime the reference encoder's one-frame look-ahead (see the full-rate
        // twin for why): the first encode_pcm returns a placeholder.
        let warm = periodic_pcm(39, 6000);
        let _ = tx_fec.encode_pcm(&warm);
        let _ = tx_raw.encode_pcm(&warm);
        for k in 0..6 {
            let pcm = periodic_pcm(40 + k, 6000);
            let pcm_fec = rx_fec
                .decode_bits(&tx_fec.encode_pcm(&pcm).unwrap())
                .unwrap();
            let pcm_raw = rx_raw
                .decode_bits(&tx_raw.encode_pcm(&pcm).unwrap())
                .unwrap();
            assert_eq!(
                pcm_fec, pcm_raw,
                "no-FEC and FEC paths must match on a clean channel (frame {k})"
            );
        }
    }

    #[test]
    fn imbe_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::Imbe7200x4400);
        let mut rx = Vocoder::new(Rate::Imbe7200x4400);
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
    fn ambe_plus2_roundtrip_smoke() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut rx = Vocoder::new(Rate::AmbePlus2_3600x2450);
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
        let mut v = Vocoder::new(Rate::Imbe7200x4400);
        let r = v.encode_pcm(&[0i16; 159]);
        assert!(matches!(
            r,
            Err(VocoderError::WrongPcmLength {
                expected: 160,
                got: 159
            })
        ));
    }

    #[test]
    fn wrong_bits_length_errors_per_rate() {
        let mut a = Vocoder::new(Rate::Imbe7200x4400);
        assert!(matches!(
            a.decode_bits(&[0u8; 9]),
            Err(VocoderError::WrongBitsLength {
                expected: 18,
                got: 9
            })
        ));
        let mut b = Vocoder::new(Rate::AmbePlus2_3600x2450);
        assert!(matches!(
            b.decode_bits(&[0u8; 18]),
            Err(VocoderError::WrongBitsLength {
                expected: 9,
                got: 18
            })
        ));
    }

    #[test]
    fn reset_clears_state_and_keeps_rate() {
        let mut v = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let pcm = periodic_pcm(40, 8000);
        let _ = v.encode_pcm(&pcm).unwrap();
        assert!(v.last_stats().analysis.is_some());
        v.reset();
        assert!(v.last_stats().analysis.is_none());
        assert_eq!(v.rate(), Rate::AmbePlus2_3600x2450);
    }

    /// `Rate` round-trips as a plain enum — the public-name variants
    /// stay stable across serde versions.
    #[cfg(feature = "serde")]
    #[test]
    fn rate_serializes_as_named_variant() {
        let s = serde_json::to_string(&Rate::Imbe7200x4400).unwrap();
        assert_eq!(s, "\"Imbe7200x4400\"");
        let back: Rate = serde_json::from_str(&s).unwrap();
        assert_eq!(back, Rate::Imbe7200x4400);
    }

    /// Streaming encode iterator yields exactly `pcm.len() /
    /// frame_samples()` items and drops a trailing partial frame.
    #[test]
    fn encode_stream_yields_one_per_frame_drops_partial() {
        let mut v = Vocoder::new(Rate::Imbe7200x4400);
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
            assert_eq!(b.len(), 18); // Imbe7200x4400 FEC frame size
        }
    }

    /// Streaming decode parallels encode: one item per FEC frame,
    /// trailing partial bytes dropped, output 160 samples each.
    #[test]
    fn decode_stream_yields_one_per_frame_drops_partial() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
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

        let mut rx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let frames: Vec<Vec<i16>> = rx
            .decode_stream(&padded)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(frames.len(), 7);
        for f in &frames {
            assert_eq!(f.len(), FRAME_SAMPLES);
        }
    }

    /// Audio a caller has not sent yet cannot change the frames already emitted.
    ///
    /// The AMBE+2 voicing offset is [`AMBE_B1_OUTPUT_LAG`] applied per frame,
    /// not a quantity fitted over the clip, so appending audio must leave the
    /// leading output frames byte-identical. A global fit would break exactly
    /// here — and it is also what makes `Vocoder::encode` reproducible by
    /// `LiveEncoder`, which never sees the whole clip.
    #[cfg(feature = "encode")]
    #[test]
    fn ambe2_trailing_audio_cannot_move_earlier_frames() {
        let n = FRAME_SAMPLES;
        let tone: Vec<i16> = (0..80 * n)
            .map(|i| {
                let t = i as f64 / 8000.0;
                (6000.0 * (2.0 * std::f64::consts::PI * 180.0 * t).sin()) as i16
            })
            .collect();
        for rate in [Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450] {
            let head = Vocoder::new(rate).encode(&tone[..48 * n]);
            // The head's own trailing frames still move with later audio (the
            // analysis look-ahead genuinely reaches into it); everything before
            // that must not.
            let stable = head.len().saturating_sub(B1_RESERVE);
            assert!(
                stable >= 32,
                "test is vacuous: only {stable} frames compared"
            );
            for extra in [1usize, 7, 20, 32] {
                let longer = Vocoder::new(rate).encode(&tone[..(48 + extra) * n]);
                assert_eq!(
                    head[..stable],
                    longer[..stable],
                    "{extra} extra frames of audio changed already-emitted frames"
                );
            }
        }
    }

    /// `LiveEncoder` accepts arbitrary chunk sizes, holds residue
    /// across `push` calls, and emits the same bits as whole-buffer
    /// `Vocoder::encode` on the same PCM — the single-encoder invariant.
    #[test]
    fn live_encoder_handles_arbitrary_chunk_sizes() {
        let mut total_pcm: Vec<i16> = Vec::with_capacity(7 * FRAME_SAMPLES);
        for _ in 0..7 {
            total_pcm.extend_from_slice(&periodic_pcm(40, 6000));
        }
        for rate in [
            Rate::Imbe7200x4400,
            Rate::Imbe4400x4400,
            Rate::AmbePlus2_3600x2450,
            Rate::AmbePlus2_2450x2450,
        ] {
            let ref_bits: Vec<u8> = Vocoder::new(rate).encode(&total_pcm).concat();
            // Live: feed in mismatched chunk sizes (250, 50, 333, rest).
            let mut live = LiveEncoder::new(rate);
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
            for f in live.flush().unwrap() {
                live_bits.extend(f);
            }
            assert_eq!(
                live_bits, ref_bits,
                "{rate:?}: LiveEncoder diverges from Vocoder::encode"
            );
            // Total samples 1120 = 7 frames exactly, so nothing is held back.
            assert_eq!(live.pending_samples(), 0);
        }
    }

    /// `LiveEncoder` accounts for every sample it is handed. The emission
    /// *schedule* is the streamer's business — it holds a bounded look-ahead
    /// and emits in bursts — but `pending_samples` must always equal received
    /// minus emitted, and `flush` must drain it to zero.
    #[test]
    fn live_encoder_residue_held_across_calls() {
        for rate in [
            Rate::Imbe7200x4400,
            Rate::Imbe4400x4400,
            Rate::AmbePlus2_3600x2450,
            Rate::AmbePlus2_2450x2450,
        ] {
            let mut live = LiveEncoder::new(rate);
            // 1.5 frames of input split into two pushes.
            let pcm: Vec<i16> = periodic_pcm(40, 6000)
                .iter()
                .copied()
                .chain(periodic_pcm(40, 6000)[..80].iter().copied())
                .collect();
            assert_eq!(pcm.len(), 240);

            let mut received = 0usize;
            let mut emitted = 0usize;
            for part in [&pcm[..120], &pcm[120..]] {
                let frames = live.push(part);
                received += part.len();
                emitted += frames.len();
                assert!(frames.iter().all(|r| r.is_ok()));
                assert!(
                    live.pending_samples() + emitted * FRAME_SAMPLES <= received,
                    "{rate:?}: pending_samples double-counts emitted audio"
                );
            }
            let tail = live.flush().unwrap();
            emitted += tail.len();
            assert_eq!(
                emitted,
                pcm.len().div_ceil(FRAME_SAMPLES),
                "{rate:?}: flush must complete the partial frame, not drop it"
            );
            assert_eq!(live.pending_samples(), 0);
        }
    }

    /// `LiveDecoder` handles arbitrary chunk sizes for the byte stream
    /// and matches a per-frame `Vocoder::decode_bits` loop.
    #[test]
    fn live_decoder_handles_arbitrary_chunk_sizes() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
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
        let mut ref_v = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut ref_pcm: Vec<i16> = Vec::new();
        for chunk in bits.chunks_exact(9) {
            ref_pcm.extend(ref_v.decode_bits(chunk).unwrap());
        }
        let mut live = LiveDecoder::new(Rate::AmbePlus2_3600x2450);
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
        let mut live = LiveEncoder::new(Rate::Imbe7200x4400);
        let pcm = periodic_pcm(40, 6000);
        let frames = live.push(&pcm[..80]);
        assert!(frames.is_empty());
        assert_eq!(live.pending_samples(), 80);
        live.discard_pending();
        assert_eq!(live.pending_samples(), 0);
    }

    /// Transcoder bridges P25 Phase 1 ↔ Phase 2 at the FEC-byte
    /// boundary. State advances per call; rates are validated.
    #[test]
    fn transcoder_phase1_to_phase2_changes_frame_size() {
        let mut tx = Transcoder::new(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450).unwrap();
        assert_eq!(
            tx.direction(),
            TranscodeDirection::new(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450)
        );
        // Encode some Phase 1 frames first to get realistic bits.
        let mut enc = Vocoder::new(Rate::Imbe7200x4400);
        let pcm = periodic_pcm(40, 6000);
        for _ in 0..3 {
            let phase1 = enc.encode_pcm(&pcm).unwrap();
            assert_eq!(phase1.len(), 18);
            let phase2 = tx.transcode(&phase1).unwrap();
            assert_eq!(phase2.len(), 9, "P1→P2 transcode produces 9-byte frames");
        }
    }

    #[test]
    fn transcoder_phase2_to_phase1_changes_frame_size() {
        let mut tx = Transcoder::new(Rate::AmbePlus2_3600x2450, Rate::Imbe7200x4400).unwrap();
        let mut enc = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let pcm = periodic_pcm(40, 6000);
        for _ in 0..3 {
            let phase2 = enc.encode_pcm(&pcm).unwrap();
            assert_eq!(phase2.len(), 9);
            let phase1 = tx.transcode(&phase2).unwrap();
            assert_eq!(phase1.len(), 18);
        }
    }

    #[test]
    fn transcoder_rejects_wrong_input_length() {
        let mut tx = Transcoder::new(Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450).unwrap();
        assert!(matches!(
            tx.transcode(&[0u8; 9]),
            Err(VocoderError::WrongBitsLength {
                expected: 18,
                got: 9
            })
        ));
    }

    /// Unsupported `(from, to)` pair surfaces as `UnsupportedTranscode`
    /// rather than a panic.
    #[test]
    fn transcoder_rejects_unsupported_pair() {
        let res = Transcoder::new(Rate::Imbe7200x4400, Rate::Imbe7200x4400);
        assert!(matches!(
            res.err(),
            Some(VocoderError::UnsupportedTranscode {
                from: Rate::Imbe7200x4400,
                to: Rate::Imbe7200x4400,
            })
        ));
    }

    /// `flush` zero-pads residue and drains the look-ahead, returning all
    /// remaining frames; on an empty/idle buffer it returns an empty `Vec`.
    #[test]
    fn live_encoder_flush_emits_padded_residue() {
        let mut live = LiveEncoder::new(Rate::Imbe7200x4400);
        // Empty buffer, nothing buffered in the codec → flush is empty.
        assert!(live.flush().unwrap().is_empty());

        // Half-frame residue → flush emits the padded residue frame plus
        // the look-ahead frame the reference codec was still holding.
        let pcm = periodic_pcm(40, 6000);
        let _ = live.push(&pcm[..80]);
        assert_eq!(live.pending_samples(), 80);
        let tail = live.flush().unwrap();
        assert!(
            !tail.is_empty(),
            "residue flush must emit at least one frame"
        );
        for frame in &tail {
            assert_eq!(frame.len(), 18);
        }
        assert_eq!(live.pending_samples(), 0);

        // Subsequent flush is a no-op (buffer + look-ahead both drained).
        assert!(live.flush().unwrap().is_empty());
    }

    /// `reset` clears both vocoder state and the residue buffer.
    #[test]
    fn live_encoder_reset_clears_everything() {
        let mut live = LiveEncoder::new(Rate::AmbePlus2_3600x2450);
        let pcm = periodic_pcm(40, 5000);
        let _ = live.push(&pcm[..120]);
        assert_eq!(live.pending_samples(), 120);
        let _ = live.vocoder().last_stats();
        live.reset();
        assert_eq!(live.pending_samples(), 0);
        assert!(live.vocoder().last_stats().analysis.is_none());
        assert_eq!(live.rate(), Rate::AmbePlus2_3600x2450);
    }

    /// Streaming and per-frame paths produce identical output —
    /// the iterator is a thin chunking wrapper around `encode_pcm`,
    /// state advances the same way.
    #[test]
    fn encode_stream_matches_per_frame_calls_byte_for_byte() {
        let mut a = Vocoder::new(Rate::Imbe7200x4400);
        let mut b = Vocoder::new(Rate::Imbe7200x4400);
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

    // --- soft-decision decode (decode_soft) ---

    /// `n` strong LLRs from a hard FEC frame's channel bits (MSB-first — the
    /// raw frame-bit order `decode_frame_soft` expects). Sign = the bit.
    fn frame_to_strong_llrs(fec: &[u8], n: usize) -> Vec<i8> {
        (0..n)
            .map(|i| {
                if (fec[i / 8] >> (7 - (i % 8))) & 1 == 1 {
                    100
                } else {
                    -100
                }
            })
            .collect()
    }

    /// On error-free input `decode_soft` is bit-identical to `decode_bits`:
    /// the soft FEC recovers the exact info, the re-encoded frame equals the
    /// original, and the shared blip25_codec synth produces the same PCM. This is
    /// the whole wiring contract — the coding gain over hard decode is proven
    /// on real reference `*_sd` vectors by the `roundtrip_sanity --soft` harness.
    #[test]
    fn decode_soft_matches_hard_on_clean() {
        for rate in [Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450] {
            let n = rate.soft_frame_bits().unwrap();
            let fec = match rate {
                Rate::Imbe7200x4400 => {
                    imbe_pipeline::info_vec_to_fec_bytes(&[1, 2, 3, 4, 5, 6, 7, 8]).to_vec()
                }
                Rate::AmbePlus2_3600x2450 => {
                    ambe_plus2_pipeline::info_vec_to_fec_bytes(&[3, 5, 7, 9]).to_vec()
                }
                _ => unreachable!(),
            };
            let hard = Vocoder::new(rate).decode_bits(&fec).unwrap();
            let llrs = frame_to_strong_llrs(&fec, n);
            let soft = Vocoder::new(rate).decode_soft(&llrs).unwrap();
            assert_eq!(hard, soft, "{rate:?}: decode_soft(clean) != decode_bits");
            assert_eq!(soft.len(), rate.frame_samples());
        }
    }

    #[test]
    fn decode_soft_rejects_bad_input() {
        let mut imbe = Vocoder::new(Rate::Imbe7200x4400);
        assert!(matches!(
            imbe.decode_soft(&[0i8; 100]),
            Err(VocoderError::WrongSoftLength {
                expected: 144,
                got: 100
            })
        ));
        let mut nofec = Vocoder::new(Rate::Imbe4400x4400);
        assert!(matches!(
            nofec.decode_soft(&[0i8; 144]),
            Err(VocoderError::SoftUnsupported {
                rate: Rate::Imbe4400x4400
            })
        ));
    }

    #[test]
    fn decode_stream_soft_matches_per_frame() {
        let rate = Rate::Imbe7200x4400;
        let n = rate.soft_frame_bits().unwrap();
        let fec = imbe_pipeline::info_vec_to_fec_bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let llrs: Vec<i8> = (0..4).flat_map(|_| frame_to_strong_llrs(&fec, n)).collect();

        let mut sv = Vocoder::new(rate);
        let streamed: Vec<Vec<i16>> = sv.decode_stream_soft(&llrs).map(|r| r.unwrap()).collect();

        // Per-frame reference shares state across frames, just like the stream.
        let mut pv = Vocoder::new(rate);
        let per_frame: Vec<Vec<i16>> = llrs
            .chunks_exact(n)
            .map(|c| pv.decode_soft(c).unwrap())
            .collect();

        assert_eq!(streamed, per_frame);
        assert_eq!(streamed.len(), 4);

        // No-FEC rate: empty stream (decode_soft would return SoftUnsupported).
        let mut nofec = Vocoder::new(Rate::Imbe4400x4400);
        assert_eq!(nofec.decode_stream_soft(&[0i8; 144]).count(), 0);
    }

    const ALL_RATES: [Rate; 4] = [
        Rate::Imbe7200x4400,
        Rate::Imbe4400x4400,
        Rate::AmbePlus2_3600x2450,
        Rate::AmbePlus2_2450x2450,
    ];

    /// The erasure marker is a well-formed wire frame at every rate.
    #[test]
    fn erasure_frame_is_frame_sized() {
        for rate in ALL_RATES {
            assert_eq!(
                rate.erasure_frame().len(),
                rate.fec_frame_bytes(),
                "{rate:?} erasure frame is not one wire frame"
            );
        }
    }

    /// Every rate — including the two no-FEC ones, whose erasure signal cannot
    /// come from channel-error counts — conceals an erasure frame by repeating,
    /// then mutes once four have arrived in a row.
    #[test]
    fn erasure_frames_drive_repeat_then_mute() {
        for rate in ALL_RATES {
            let mut v = Vocoder::new(rate);
            let pcm: Vec<i16> = (0..FRAME_SAMPLES)
                .map(|i| ((i as f32 * 0.3).sin() * 6000.0) as i16)
                .collect();
            let good = Vocoder::new(rate).encode_pcm(&pcm).expect("encode primes");
            v.decode_bits(&good).expect("prime decodes");

            let eras = rate.erasure_frame();
            let dispositions: Vec<FrameDisposition> = (0..5)
                .map(|_| {
                    v.decode_bits(&eras).expect("erasure frame decodes");
                    v.last_stats().decode.as_ref().unwrap().disposition
                })
                .collect();

            assert_eq!(
                &dispositions[..3],
                &[FrameDisposition::Repeat; 3],
                "{rate:?} did not repeat on erasure"
            );
            assert_eq!(
                &dispositions[3..],
                &[FrameDisposition::Mute; 2],
                "{rate:?} did not mute after 4 consecutive erasures"
            );
        }
    }

    /// A muted frame really is comfort noise, not repeated audio: §7.7 / §5.7
    /// specify uniform noise on [-5, 5].
    #[test]
    fn mute_output_is_comfort_noise() {
        for rate in ALL_RATES {
            let mut v = Vocoder::new(rate);
            let eras = rate.erasure_frame();
            let mut pcm = Vec::new();
            for _ in 0..6 {
                pcm = v.decode_bits(&eras).expect("erasure frame decodes");
            }
            assert_eq!(
                v.last_stats().decode.as_ref().unwrap().disposition,
                FrameDisposition::Mute
            );
            let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
            assert!(
                peak <= 5,
                "{rate:?} mute output peaked at {peak}, expected <= 5"
            );
        }
    }

    /// A clean voice frame reports `Use` at every rate — the concealment
    /// telemetry stays inert when nothing is wrong.
    #[test]
    fn clean_frames_report_use() {
        for rate in ALL_RATES {
            let mut v = Vocoder::new(rate);
            let pcm: Vec<i16> = (0..FRAME_SAMPLES * 4)
                .map(|i| ((i as f32 * 0.3).sin() * 6000.0) as i16)
                .collect();
            let mut enc = Vocoder::new(rate);
            // The encoder carries one frame of algorithmic delay, so its first
            // output is a priming frame rather than coded speech. Only the
            // frames after it represent a clean channel.
            for (i, frame) in pcm.chunks(FRAME_SAMPLES).enumerate() {
                let bits = enc.encode_pcm(frame).expect("encode");
                v.decode_bits(&bits).expect("decode");
                let d = v.last_stats().decode.as_ref().unwrap();
                assert_eq!(d.disposition, FrameDisposition::Use, "{rate:?} frame {i}");
                if i > 0 {
                    assert_eq!((d.epsilon_0, d.epsilon_t), (0, 0), "{rate:?} frame {i}");
                }
            }
        }
    }

    /// Concealment reporting on the IMBE rates is not hardcoded: an erasure is
    /// distinguishable from a clean frame through the public API alone. This is
    /// the property that was missing while both IMBE arms reported `Use`.
    #[test]
    fn imbe_disposition_is_not_hardcoded() {
        for rate in [Rate::Imbe7200x4400, Rate::Imbe4400x4400] {
            let mut v = Vocoder::new(rate);
            let pcm: Vec<i16> = (0..FRAME_SAMPLES)
                .map(|i| ((i as f32 * 0.3).sin() * 6000.0) as i16)
                .collect();
            let good = Vocoder::new(rate).encode_pcm(&pcm).expect("encode");
            v.decode_bits(&good).expect("decode");
            assert_eq!(
                v.last_stats().decode.as_ref().unwrap().disposition,
                FrameDisposition::Use
            );

            v.decode_bits(&rate.erasure_frame()).expect("decode");
            assert_eq!(
                v.last_stats().decode.as_ref().unwrap().disposition,
                FrameDisposition::Repeat,
                "{rate:?} still reports a fixed disposition"
            );
        }
    }
}
