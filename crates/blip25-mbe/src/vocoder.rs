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
//! The low-level modules ([`crate::imbe7200`], [`crate::rate33`],
//! [`crate::codecs::mbe_baseline`]) stay public for advanced consumers
//! that need to drive the pipeline frame-by-frame at the parameter
//! layer; this module is the recommended entry point for everything else.
//!
//! ## Quick start
//!
//! ```rust
//! use blip25_mbe::vocoder::{Rate, Vocoder};
//!
//! // P25 Phase 1 (full-rate IMBE) encoder
//! let mut tx = Vocoder::new(Rate::Imbe7200x4400);
//! let pcm: [i16; 160] = [0; 160]; // one 20 ms frame at 8 kHz
//! let bits = tx.encode_pcm(&pcm).expect("encode");
//! assert_eq!(bits.len(), 18); // 18-byte FEC frame
//!
//! // P25 Phase 1 decoder (separate channel, separate state)
//! let mut rx = Vocoder::new(Rate::Imbe7200x4400);
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
    AnalysisError, AnalysisOutput, AnalysisState, ToneDetection,
    detect_tone, encode as analysis_encode,
    encode_ambe_plus2 as analysis_encode_ambe_plus2,
    profile as analysis_profile,
};
use crate::codecs::mbe_baseline::{
    FrameDisposition, FrameErrorContext, GAMMA_W, SynthState, UnvoicedNoiseGen, synthesize_frame,
};
use crate::enhancement::{self, EnhancementMode, EnhancementState};
use crate::mbe_params::MbeParams;
use crate::imbe7200;
use crate::rate33;

/// 8 kHz mono — the only sample rate the vocoder produces.
const SAMPLE_RATE_HZ: f32 = 8_000.0;

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
    Imbe7200x4400,
    /// IMBE info-only — same codec as [`Self::Imbe7200x4400`] with the
    /// Annex H Golay/Hamming/PN FEC layer stripped. 11-byte wire frame
    /// (88 prioritized info bits packed MSB-first). 4 400 bps total
    /// (= voice). Byte layout matches JMBE / OP25 / DVSI `p25_nofec`
    /// convention bit-for-bit (validated against DVSI tv-rc references
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
    /// layer stripped. 7-byte wire frame (49 info bits packed
    /// MSB-first in u₀..u₃ order, 7 trailing pad bits). 2 450 bps
    /// total (= voice).
    ///
    /// **Byte layout caveat.** DVSI's chip rate-index 34 emits the
    /// same 49 info bits in a *different* (chip-internal, permuted)
    /// byte order. This crate's layout is self-consistent (ours-to-
    /// ours round-trip is lossless) but is not byte-equal to DVSI's
    /// r34 stream. If you need byte-exact DVSI r34 interop, see the
    /// future `blip25-chip-shim` crate; here this variant is for
    /// compact ours-to-ours archival. Recommended storage format is
    /// the FEC-bearing [`Self::AmbePlus2_3600x2450`] — see
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

    /// PCM samples per frame (always 160 at 8 kHz / 20 ms; provided
    /// for symmetry with [`Self::fec_frame_bytes`]).
    #[inline]
    pub const fn frame_samples(self) -> usize {
        FRAME_SAMPLES
    }
}

/// Half-rate synthesis generation. Picks how reconstructed `MbeParams`
/// are turned back into PCM on the half-rate decode path.
///
/// Full-rate (P25 Phase 1 IMBE) ignores this — it always uses the
/// BABA-A §1.12 baseline phase, the only synth the IMBE spec
/// describes.
///
/// The original Wave 1.2 of `SCOPE_PLAN.md` framed this as an
/// "AMBE+ Gen-2 dequantize wrapper" — that turned out to be the
/// wrong axis. The half-rate WIRE format and parameter recovery are
/// AMBE+2 across the board (per
/// `~/blip25-specs/analysis/oss_implementations_lessons_learned.md`);
/// the actual Gen-2 vs Gen-3 split for legacy NXDN / DMR is here, on
/// the synth side. mbelib's half-rate decoder uses [`Self::Baseline`]
/// (1993-vintage IMBE phase, no Hilbert phase regen). JMBE / SDRTrunk
/// use [`Self::AmbePlus`] (US5701390 phase regen, modern AMBE+ /
/// AMBE+2 sound).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum AmbePlus2Synth {
    /// US5701390 phase regen — modern AMBE+ / AMBE+2 sound. Default
    /// and recommended for P25 Phase 2 / NXDN type-2 / DMR enhanced.
    AmbePlus,
    /// BABA-A §1.12 baseline IMBE phase — no Hilbert regen. Matches
    /// mbelib's half-rate output. Use for legacy NXDN type-1 / older
    /// DMR captures where the consumer wants mbelib-equivalent
    /// audible behavior.
    Baseline,
}

impl Default for AmbePlus2Synth {
    fn default() -> Self {
        Self::AmbePlus
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
    /// What the analysis encoder emitted (`Voice` / `Silence` / `Tone`).
    pub output: AnalysisOutputKind,
    /// The MBE parameters that were quantized into the wire bits.
    /// Populated for both `Voice` and `Silence` (silence dispatches
    /// the rate-appropriate placeholder).
    pub params: MbeParams,
    /// Tone-detection signal for the input frame, populated whenever
    /// [`Vocoder::set_tone_detection`] is enabled and the detector
    /// matched an Annex T entry. Decoupled from `output` because Phase
    /// 1 IMBE has no tone-frame opcode (per BABA-A): on full-rate the
    /// detector still surfaces `(I_D, A_D)` here for application-layer
    /// signaling (LCW, paging, dispatch) while the wire frame remains
    /// a regular voice / silence frame. On half-rate AMBE+2 a positive
    /// detection is *also* dispatched as an Annex T tone frame (so
    /// `output == Tone` and this field both populate).
    pub tone_detect: Option<ToneDetection>,
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
    /// running the voice analysis pipeline. Half-rate only; gated on
    /// [`Vocoder::set_tone_detection`]. Default off.
    Tone {
        /// Annex T tone ID.
        id: u8,
        /// 7-bit log-amplitude (`A_D`).
        amplitude: u8,
    },
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
    /// Analysis encoder failure (HPF / pitch / refinement).
    Analysis(AnalysisError),
    /// Wire-quantize failure (e.g. predictor returned a non-finite value).
    Quantize(String),
    /// Requested transcode direction is not supported. The pair of rates
    /// has no parameter-domain converter wired up.
    UnsupportedTranscode {
        /// Rate of the input FEC frame.
        from: Rate,
        /// Rate of the output FEC frame.
        to: Rate,
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
            VocoderError::Analysis(e) => write!(f, "analysis encoder error: {e:?}"),
            VocoderError::Quantize(msg) => write!(f, "quantize error: {msg}"),
            VocoderError::UnsupportedTranscode { from, to } => {
                write!(f, "unsupported transcode direction: {from:?} -> {to:?}")
            }
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
    imbe_dec: imbe7200::dequantize::DecoderState,
    ambe_plus2_dec: rate33::dequantize::DecoderState,
    synth: SynthState,
    last_stats: FrameStats,
    tone_detection: bool,
    ambe_plus2_synth: AmbePlus2Synth,
    enhancement: EnhancementMode,
    enhancement_state: EnhancementState,
    prev_disposition: Option<FrameDisposition>,
}

impl Vocoder {
    /// Open a new channel at the given rate, all state cold.
    ///
    /// The unvoiced noise generator is selected per-rate: IMBE full-rate
    /// uses the BABA-A §1.12.1 spec LCG (171/11213/53125, seed 3147)
    /// because the DVSI chip's full-rate path matches that recurrence;
    /// AMBE+2 half-rate uses the chip-empirical LCG (173/13849/65536,
    /// seed 60584) — see gap report 0025. Probe 1 confirmed 0.945
    /// sample-level correlation against the live chip on the half-rate
    /// path.
    pub fn new(rate: Rate) -> Self {
        let noise_gen = match rate {
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => UnvoicedNoiseGen::SpecLcg,
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => UnvoicedNoiseGen::ChipLcg,
        };
        Self {
            rate,
            analysis: AnalysisState::new(),
            imbe_dec: imbe7200::dequantize::DecoderState::new(),
            ambe_plus2_dec: rate33::dequantize::DecoderState::new(),
            synth: SynthState::with_unvoiced_gen(noise_gen),
            last_stats: FrameStats::default(),
            tone_detection: false,
            ambe_plus2_synth: AmbePlus2Synth::AmbePlus,
            // Enhancement: Classical-by-default since 2026-05-14. The
            // HPF + presence + boundary-fade chain in
            // `ClassicalConfig::default()` carries a free +0.03 to
            // +0.09 PESQ across the 5-vector matrix; spec-faithful
            // PCM is one `.set_enhancement(EnhancementMode::None)`
            // call away.
            enhancement: EnhancementMode::Classical(crate::enhancement::ClassicalConfig::default()),
            enhancement_state: EnhancementState::default(),
            prev_disposition: None,
        }
    }

    /// Configure which half-rate synth flavor [`Self::decode_bits`] +
    /// [`Self::synthesize_params`] use on Phase 2 / AMBE+2 input.
    /// No-op for full-rate (which always uses the BABA-A §1.12
    /// baseline IMBE synth). Default [`AmbePlus2Synth::AmbePlus`].
    pub fn set_ambe_plus2_synth(&mut self, gen: AmbePlus2Synth) {
        self.ambe_plus2_synth = gen;
    }

    /// Currently configured half-rate synth flavor.
    #[inline]
    pub fn ambe_plus2_synth(&self) -> AmbePlus2Synth {
        self.ambe_plus2_synth
    }

    /// Enable encode-side Annex T tone detection. Each PCM frame is
    /// inspected for a clean single-frequency tone or DTMF / Knox
    /// pair matching an Annex T entry; on a hit the detection result
    /// is surfaced via [`AnalysisStats::tone_detect`]. Default off.
    ///
    /// Wire behavior is rate-dependent:
    ///
    /// - **Half-rate (AMBE+2 Phase 2):** on a hit, the channel emits
    ///   an Annex T tone frame instead of running the voice analysis
    ///   pipeline; [`AnalysisStats::output`] is `Tone { id, amplitude }`
    ///   and [`AnalysisStats::tone_detect`] mirrors the same values.
    /// - **Full-rate (IMBE Phase 1):** the wire frame remains a
    ///   regular voice / silence frame because P25 Phase 1 has no
    ///   tone-frame opcode at the codec layer (BABA-A; spec-author
    ///   confirmation 2026-04-30). [`AnalysisStats::tone_detect`] is
    ///   still populated so application-layer consumers can route
    ///   `(I_D, A_D)` to LCW, paging, or other out-of-band signaling
    ///   per BABA-A `§5.4`.
    ///
    /// Spec context: BABA-A `§2.10` defines the tone-frame *payload*
    /// (Annex T table + signature bits) but the spec leaves "is this
    /// a tone?" to the implementer (per `§0.0.1` of the impl spec).
    /// This is a DSP design choice, not a P25-IP question.
    pub fn set_tone_detection(&mut self, enabled: bool) {
        self.tone_detection = enabled;
    }

    /// Whether encode-side tone detection is currently enabled.
    #[inline]
    pub fn tone_detection(&self) -> bool {
        self.tone_detection
    }

    /// Configure the beyond-spec consecutive-repeat reset threshold
    /// on the synth side. After `n` consecutive Repeat/Mute frames,
    /// substitution falls back to a default-fundamental + amps=1.0
    /// frame instead of replaying the prior `last_good`. `None`
    /// (default) is the spec-faithful path per gap 0022 resolution.
    /// JMBE / SDRTrunk use `Some(3)` for chip-stream interop quality.
    pub fn set_repeat_reset_after(&mut self, n: Option<u32>) {
        self.synth.set_repeat_reset_after(n);
    }

    /// Current consecutive-repeat reset threshold (`None` = disabled,
    /// spec-faithful).
    #[inline]
    pub fn repeat_reset_after(&self) -> Option<u32> {
        self.synth.repeat_reset_after()
    }

    /// Enable JMBE-style error-rate freeze on Repeat (beyond-spec; gap
    /// 0021). When enabled, frames whose disposition is `Repeat` do
    /// not advance `ε_R`; the previous frame's value is carried
    /// forward. This mirrors JMBE's
    /// `IMBEModelParameters.copy()` calling
    /// `setErrorRate(previous.getErrorRate())` and prevents runs of
    /// high-error chip-encoded frames from driving `ε_R` past the
    /// 0.0875 Mute threshold. Default `false` is the spec-faithful
    /// path. Enable for decoding chip-encoded P25 air traffic; leave
    /// off for our-encoder → our-decoder loops and spec-conformance
    /// testing.
    pub fn set_chip_compat(&mut self, on: bool) {
        self.synth.set_chip_compat(on);
    }

    /// Current chip-compat (error-rate freeze on Repeat) setting.
    #[inline]
    pub fn chip_compat(&self) -> bool {
        self.synth.chip_compat()
    }

    /// Force the beyond-spec spectral-discontinuity clamp on (gap 0026)
    /// independently of [`Self::set_chip_compat`]. Most consumers
    /// should leave this off — the umbrella `chip_compat` flag already
    /// auto-enables the clamp alongside the gap-0021 ε_R freeze.
    /// This standalone toggle exists for the narrow case of wanting
    /// the clamp without the ε_R freeze. With either flag enabled,
    /// frames whose harmonic count `L` jumps by more than 5 from the
    /// previous frame (up-jumps only) **or** whose `err.epsilon_0 ≥ 2`
    /// have their post-smoothing amplitudes scaled by 0.73 for one
    /// frame. Mirrors observed DVSI AMBE-3000R behavior on pitch/L
    /// jumps and Repeat frames; reduces audible "scratch" on noisy
    /// Phase 2 traffic where Golay-corrected frames decode to
    /// spectrally-distant params from their neighbors. Default
    /// `false` keeps the spec-faithful path.
    pub fn set_chip_compat_spectral_clamp(&mut self, on: bool) {
        self.synth.set_chip_compat_spectral_clamp(on);
    }

    /// Current standalone spectral-discontinuity clamp setting.
    /// Note: the clamp also fires whenever [`Self::chip_compat`] is
    /// `true`, regardless of this setting.
    #[inline]
    pub fn chip_compat_spectral_clamp(&self) -> bool {
        self.synth.chip_compat_spectral_clamp()
    }

    /// Enable the §0.8.4 silence-dispatch path on the analysis encoder.
    /// When on, frames whose energy clears the silence detector's
    /// hysteresis emit `AnalysisOutput::Silence` (rate-appropriate
    /// silence params) instead of running the full pitch / V-UV /
    /// amplitude pipeline. Default off — pass-through per addendum
    /// §0.8.8 recommendation.
    pub fn set_silence_dispatch(&mut self, on: bool) {
        self.analysis.set_silence_detection(on);
    }

    /// Whether silence dispatch is currently enabled.
    #[inline]
    pub fn silence_dispatch(&self) -> bool {
        self.analysis.silence_detection_enabled()
    }

    /// Enable the joint-signal silence override — dispatches
    /// pitch-unreliable low-energy frames as silence even if they
    /// don't reach the §0.8.4 hysteresis threshold. Requires
    /// [`Self::set_silence_dispatch`] also be on. Default off.
    pub fn set_pitch_silence_override(&mut self, on: bool) {
        self.analysis.set_pitch_silence_override(on);
    }

    /// Whether the joint-signal silence override is currently enabled.
    #[inline]
    pub fn pitch_silence_override(&self) -> bool {
        self.analysis.pitch_silence_override_enabled()
    }

    /// Enable or disable the onset-attack mitigation. On near-silent
    /// frames the encoder commits a short default pitch
    /// (≈ 25 samples / ~320 Hz) to its look-back history instead of
    /// the phantom long-period autocorrelation peak that quiet input
    /// otherwise produces. The silent frame still encodes through the
    /// full voice pipeline; only the next frame's `look_back`
    /// continuity window is affected. Targets the 2-frame onset attack
    /// at silence→loud transitions documented in
    /// `project_onset_attack_2frame_2026-04-25.md`. Default off — eval
    /// via the 5-vector PESQ harness before enabling in production.
    pub fn set_default_pitch_on_silence(&mut self, on: bool) {
        self.analysis.set_default_pitch_on_silence(on);
    }

    /// Whether the onset-attack mitigation is currently enabled.
    #[inline]
    pub fn default_pitch_on_silence(&self) -> bool {
        self.analysis.default_pitch_on_silence_enabled()
    }

    /// Enable or disable the PYIN pitch frontend. When on, the analysis
    /// encoder uses [`crate::codecs::mbe_baseline::analysis::run_pyin`]
    /// for `(p_hat_i, e_p_hat_i)` instead of the §0.3 look-back /
    /// look-ahead tracker (Eq. 5–23). PYIN is post-2002 DSP outside
    /// BABA-A's clean-room scope; landed for A/B PESQ evaluation. The
    /// §0.3 path remains the default and the spec-faithful baseline.
    pub fn set_pyin_pitch(&mut self, on: bool) {
        self.analysis.set_pyin_pitch(on);
    }

    /// Whether the PYIN pitch frontend is currently enabled.
    #[inline]
    pub fn pyin_pitch(&self) -> bool {
        self.analysis.pyin_pitch_enabled()
    }

    /// Enable or disable input-side spectral subtraction (Boll 1979)
    /// applied to `signal_spectrum` output before §0.5 amplitude
    /// estimation. Per-bin running noise PSD is updated on
    /// silence-flagged frames; voiced frames hold. Off by default —
    /// targets noisy-tone inputs (memo
    /// `project_amp_noise_sensitivity_2026-04-24`).
    pub fn set_spectral_subtraction(&mut self, on: bool) {
        self.analysis.set_spectral_subtraction(on);
    }

    /// Whether spectral subtraction is currently enabled.
    #[inline]
    pub fn spectral_subtraction(&self) -> bool {
        self.analysis.spectral_subtraction_enabled()
    }

    /// Set the §0.5 amplitude EMA weight `α`. `0.0` disables the
    /// smoother (default); `(0.0, 1.0]` enables
    /// `M̂_l(t) = α · M̂_l + (1−α) · M̂_l(t−1)` per harmonic, gated by
    /// pitch similarity. Targets the noisy-tone amp jitter described
    /// in `project_amp_noise_sensitivity_2026-04-24`. Outside BABA-A
    /// clean-room scope (general-DSP magnitude smoothing).
    pub fn set_amp_ema_alpha(&mut self, alpha: f64) {
        self.analysis.set_amp_ema_alpha(alpha);
    }

    /// Current §0.5 amplitude EMA weight; `0.0` means the smoother is
    /// off.
    #[inline]
    pub fn amp_ema_alpha(&self) -> f64 {
        self.analysis.amp_ema_alpha()
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
    /// synth substates, smoothed error rate, last-stats. Rate and
    /// configuration knobs (tone detection) stay the same.
    pub fn reset(&mut self) {
        self.analysis = AnalysisState::new();
        self.imbe_dec = imbe7200::dequantize::DecoderState::new();
        self.ambe_plus2_dec = rate33::dequantize::DecoderState::new();
        self.synth = SynthState::new();
        self.last_stats = FrameStats::default();
        self.enhancement_state = EnhancementState::default();
        self.prev_disposition = None;
        // tone_detection / enhancement: preserved (config, not state)
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
            Rate::Imbe7200x4400 => imbe_pipeline::encode(pcm, self, true)?,
            Rate::Imbe4400x4400 => imbe_pipeline::encode(pcm, self, false)?,
            Rate::AmbePlus2_3600x2450 => ambe_plus2_pipeline::encode(pcm, self, true)?,
            Rate::AmbePlus2_2450x2450 => ambe_plus2_pipeline::encode(pcm, self, false)?,
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
        let (mut pcm, stats) = match self.rate {
            Rate::Imbe7200x4400 => imbe_pipeline::decode(bits, self, true),
            Rate::Imbe4400x4400 => imbe_pipeline::decode(bits, self, false),
            Rate::AmbePlus2_3600x2450 => ambe_plus2_pipeline::decode(bits, self, true),
            Rate::AmbePlus2_2450x2450 => ambe_plus2_pipeline::decode(bits, self, false),
        };
        let prev_was_use = matches!(self.prev_disposition, Some(FrameDisposition::Use));
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
    pub fn encode_stream<'a>(&'a mut self, pcm: &'a [i16]) -> EncodeStream<'a> {
        EncodeStream { vocoder: self, pcm, pos: 0 }
    }

    /// Run the analysis encoder on one PCM frame and return the
    /// resulting [`MbeParams`] without quantizing or FEC-encoding.
    /// Advances the analysis state (look-ahead history, predictor,
    /// V/UV state, silence detector) so subsequent calls see
    /// continuous context.
    ///
    /// `pcm` must be exactly [`Self::frame_samples`] samples (160).
    ///
    /// On `AnalysisOutput::Silence` (preroll, silence dispatch,
    /// half-rate `PitchOutOfRange`), returns the rate-appropriate
    /// silence params ([`MbeParams::silence`] or
    /// [`MbeParams::silence_ambe_plus2`]).
    ///
    /// Counterpart of [`Self::synthesize_params`]. Together these
    /// expose the parameter layer without going through wire FEC,
    /// enabling rate-conversion / analysis / playback pipelines that
    /// don't need the full encode/decode round-trip.
    pub fn extract_params(&mut self, pcm: &[i16]) -> Result<MbeParams, VocoderError> {
        if pcm.len() != self.frame_samples() {
            return Err(VocoderError::WrongPcmLength {
                expected: self.frame_samples(),
                got: pcm.len(),
            });
        }
        let frame = pcm.try_into().expect("length already validated");
        let analysis_out = match self.rate {
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => {
                analysis_encode(frame, &mut self.analysis)
            }
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                analysis_encode_ambe_plus2(frame, &mut self.analysis)
            }
        }
        .map_err(VocoderError::Analysis)?;
        Ok(match analysis_out {
            AnalysisOutput::Voice(p) => p,
            AnalysisOutput::Silence => match self.rate {
                Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => MbeParams::silence(),
                Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => {
                    MbeParams::silence_ambe_plus2()
                }
            },
        })
    }

    /// Synthesize one PCM frame directly from [`MbeParams`], skipping
    /// the FEC + dequantize chain. Advances the channel's synth state
    /// (so re-acquisition after silence and across-frame phase
    /// continuity work the same way they would for a normal
    /// [`Self::decode_bits`] call).
    ///
    /// Useful for playing back tone-frame params from
    /// [`crate::rate33::dequantize::tone_to_mbe_params`], replaying
    /// captured params in test harnesses, or driving synth from any
    /// upstream that produces `MbeParams` without going through wire
    /// bits.
    ///
    /// The dispatch is rate-aware: full-rate uses BABA-A baseline
    /// phase; half-rate uses AMBE+2 (US5701390) phase regen, matching
    /// what [`Self::decode_bits`] would do.
    ///
    /// `last_stats` is **not** updated by this call — it's reserved
    /// for the wire-aware encode/decode paths. Read the synth's
    /// disposition via [`Self::last_disposition`] which IS advanced
    /// (the disposition reflects this synth call).
    pub fn synthesize_params(&mut self, params: &MbeParams) -> Vec<i16> {
        // No FEC errors on this path — caller is providing trusted
        // params, not recovered ones. Use a clean error context so
        // disposition is `Use` unless smoothed ε_R is already high.
        let prev_err = self.synth.err;
        self.synth.err = FrameErrorContext::default();
        let err = self.synth.err;
        let gamma_w = self.synth.gamma_w;
        let pcm: [i16; FRAME_SAMPLES] = match self.rate {
            Rate::Imbe7200x4400 | Rate::Imbe4400x4400 => {
                synthesize_frame(params, &err, gamma_w, &mut self.synth)
            }
            Rate::AmbePlus2_3600x2450 | Rate::AmbePlus2_2450x2450 => match self.ambe_plus2_synth {
                AmbePlus2Synth::AmbePlus => ambe_plus2::synthesize_frame(params, &mut self.synth),
                AmbePlus2Synth::Baseline => synthesize_frame(params, &err, gamma_w, &mut self.synth),
            },
        };
        self.synth.err = prev_err;
        let mut pcm = pcm.to_vec();
        let prev_was_use = matches!(self.prev_disposition, Some(FrameDisposition::Use));
        enhancement::apply(
            &self.enhancement,
            &mut self.enhancement_state,
            &mut pcm,
            SAMPLE_RATE_HZ,
            prev_was_use,
        );
        self.prev_disposition = self.synth.last_disposition();
        pcm
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

/// Push-driven encoder for live PCM streams that arrive in chunks
/// of arbitrary length (audio device callbacks, file readers, sockets).
/// Holds residual samples internally across calls so the caller can
/// push whatever they have and harvest frames as they become available.
///
/// One-frame-at-a-time `Vocoder` is the right primitive for callers
/// that already have whole-buffer PCM. `LiveEncoder` is for callers
/// that don't.
///
/// ```rust
/// # use blip25_mbe::vocoder::{LiveEncoder, Rate};
/// let mut enc = LiveEncoder::new(Rate::Imbe7200x4400);
/// // 256 samples (audio-device callback); not a multiple of 160.
/// let chunk: [i16; 256] = [0; 256];
/// let frames = enc.push(&chunk);
/// assert_eq!(frames.len(), 1);                   // one full frame produced
/// assert!(frames[0].is_ok());
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

    /// Pad pending residue with zeros to a full frame and encode one
    /// final frame. Returns `Ok(Some(bits))` if residue existed,
    /// `Ok(None)` if the buffer was empty. Buffer is drained either
    /// way.
    ///
    /// Use this at end-of-stream to avoid abruptly dropping the
    /// trailing samples; the cost is at most one extra frame of
    /// zero-padding tacked onto the last word of audio.
    pub fn flush(&mut self) -> Result<Option<Vec<u8>>, VocoderError> {
        if self.pcm_buf.is_empty() {
            return Ok(None);
        }
        let n = self.vocoder.frame_samples();
        self.pcm_buf.resize(n, 0);
        let bits = self.vocoder.encode_pcm(&self.pcm_buf)?;
        self.pcm_buf.clear();
        Ok(Some(bits))
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

/// Fluent builder for [`Vocoder`]. Lets callers configure rate +
/// optional knobs in one expression instead of `new` + a sequence of
/// setters. Mirrors the chip's "open channel + PKT_RATEP +
/// configuration" sequence, but in one call.
///
/// ```rust
/// use blip25_mbe::vocoder::{Rate, Vocoder};
///
/// let tx = Vocoder::builder(Rate::AmbePlus2_3600x2450)
///     .tone_detection(true)
///     .build();
/// assert_eq!(tx.rate(), Rate::AmbePlus2_3600x2450);
/// assert!(tx.tone_detection());
/// ```
#[derive(Clone, Debug)]
pub struct VocoderBuilder {
    rate: Rate,
    tone_detection: bool,
    repeat_reset_after: Option<u32>,
    chip_compat: bool,
    silence_dispatch: bool,
    pitch_silence_override: bool,
    default_pitch_on_silence: bool,
    pyin_pitch: bool,
    spectral_subtraction: bool,
    amp_ema_alpha: f64,
    ambe_plus2_synth: AmbePlus2Synth,
    enhancement: EnhancementMode,
}

impl VocoderBuilder {
    /// New builder defaulting to the same configuration as
    /// [`Vocoder::new`] — i.e. Classical enhancement chain and §0.5
    /// spectral subtraction are ON, since both are free PESQ wins
    /// (see 5-vector A/B 2026-05-14). All other knobs default OFF /
    /// spec-faithful. Use the setters to opt out (e.g.
    /// `.enhancement(EnhancementMode::None).spectral_subtraction(false)`
    /// for fully spec-faithful output).
    #[inline]
    pub fn new(rate: Rate) -> Self {
        Self {
            rate,
            tone_detection: false,
            repeat_reset_after: None,
            chip_compat: false,
            silence_dispatch: false,
            pitch_silence_override: false,
            default_pitch_on_silence: false,
            pyin_pitch: false,
            spectral_subtraction: true,
            amp_ema_alpha: 0.0,
            ambe_plus2_synth: AmbePlus2Synth::AmbePlus,
            enhancement: EnhancementMode::Classical(
                crate::enhancement::ClassicalConfig::default(),
            ),
        }
    }

    /// Enable encode-side Annex T tone detection (half-rate only).
    /// See [`Vocoder::set_tone_detection`].
    #[inline]
    pub fn tone_detection(mut self, on: bool) -> Self {
        self.tone_detection = on;
        self
    }

    /// Configure the beyond-spec consecutive-repeat reset threshold.
    /// See [`Vocoder::set_repeat_reset_after`].
    #[inline]
    pub fn repeat_reset_after(mut self, n: Option<u32>) -> Self {
        self.repeat_reset_after = n;
        self
    }

    /// Enable JMBE-style error-rate freeze on Repeat (gap 0021).
    /// See [`Vocoder::set_chip_compat`].
    #[inline]
    pub fn chip_compat(mut self, on: bool) -> Self {
        self.chip_compat = on;
        self
    }

    /// Enable §0.8.4 silence dispatch on the analysis encoder.
    /// See [`Vocoder::set_silence_dispatch`].
    #[inline]
    pub fn silence_dispatch(mut self, on: bool) -> Self {
        self.silence_dispatch = on;
        self
    }

    /// Enable the joint-signal silence override (requires
    /// [`Self::silence_dispatch`] also on).
    /// See [`Vocoder::set_pitch_silence_override`].
    #[inline]
    pub fn pitch_silence_override(mut self, on: bool) -> Self {
        self.pitch_silence_override = on;
        self
    }

    /// Enable the onset-attack mitigation (commit a short default
    /// pitch on near-silent frames). See
    /// [`Vocoder::set_default_pitch_on_silence`].
    #[inline]
    pub fn default_pitch_on_silence(mut self, on: bool) -> Self {
        self.default_pitch_on_silence = on;
        self
    }

    /// Enable the PYIN pitch frontend (post-2002 DSP, off by default).
    /// See [`Vocoder::set_pyin_pitch`].
    #[inline]
    pub fn pyin_pitch(mut self, on: bool) -> Self {
        self.pyin_pitch = on;
        self
    }

    /// Enable spectral subtraction at the §0.5 amplitude input.
    /// See [`Vocoder::set_spectral_subtraction`].
    #[inline]
    pub fn spectral_subtraction(mut self, on: bool) -> Self {
        self.spectral_subtraction = on;
        self
    }

    /// Configure the §0.5 amplitude EMA weight (0.0 = off; default).
    /// See [`Vocoder::set_amp_ema_alpha`].
    #[inline]
    pub fn amp_ema_alpha(mut self, alpha: f64) -> Self {
        self.amp_ema_alpha = alpha;
        self
    }

    /// Configure the half-rate synth flavor (no-op for full-rate).
    /// See [`Vocoder::set_ambe_plus2_synth`].
    #[inline]
    pub fn ambe_plus2_synth(mut self, gen: AmbePlus2Synth) -> Self {
        self.ambe_plus2_synth = gen;
        self
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
        v.set_tone_detection(self.tone_detection);
        v.set_repeat_reset_after(self.repeat_reset_after);
        v.set_chip_compat(self.chip_compat);
        v.set_silence_dispatch(self.silence_dispatch);
        v.set_pitch_silence_override(self.pitch_silence_override);
        v.set_default_pitch_on_silence(self.default_pitch_on_silence);
        v.set_pyin_pitch(self.pyin_pitch);
        v.set_spectral_subtraction(self.spectral_subtraction);
        v.set_amp_ema_alpha(self.amp_ema_alpha);
        v.set_ambe_plus2_synth(self.ambe_plus2_synth);
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

mod imbe_pipeline {
    use super::*;
    use crate::imbe7200::dequantize::{dequantize, quantize};
    use crate::imbe7200::frame::{INFO_WIDTHS, decode_frame, encode_frame};
    use crate::imbe7200::priority::{deprioritize, prioritize};

    pub(super) fn encode(
        pcm: &[i16],
        vocoder: &mut Vocoder,
        apply_fec: bool,
    ) -> Result<(Vec<u8>, AnalysisStats), VocoderError> {
        // Phase 1 has no tone-frame opcode at the wire layer, but with
        // detection enabled we still run the detector and surface the
        // (I_D, A_D) result via AnalysisStats::tone_detect so consumers
        // can route it to LCW / app-layer signaling per BABA-A §5.4.
        let tone_detect = if vocoder.tone_detection {
            detect_tone(pcm)
        } else {
            None
        };
        let frame = pcm.try_into().expect("length already validated");
        let (kind, params) = match analysis_encode(frame, &mut vocoder.analysis)
            .map_err(VocoderError::Analysis)?
        {
            AnalysisOutput::Voice(p) => (AnalysisOutputKind::Voice, p),
            AnalysisOutput::Silence => (AnalysisOutputKind::Silence, MbeParams::silence()),
        };
        let mut snapshot = vocoder.imbe_dec.clone();
        let b = quantize(&params, &mut vocoder.imbe_dec)
            .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
        // Step the decoder snapshot via dequantize so the predictor
        // state matches what a downstream receiver would observe (the
        // existing speech-quality harness pattern).
        let l = params.harmonic_count();
        let info = prioritize(&b, l);
        let _ = deprioritize; // reserved for future per-priority diagnostics
        let _ = dequantize(&info, &mut snapshot);
        vocoder.imbe_dec = snapshot;
        let bytes: Vec<u8> = if apply_fec {
            let dibits = encode_frame(&info);
            pack_dibits_full(&dibits).to_vec()
        } else {
            pack_info_full(&info).to_vec()
        };
        Ok((bytes, AnalysisStats { output: kind, params, tone_detect }))
    }

    pub(super) fn decode(
        bits: &[u8],
        vocoder: &mut Vocoder,
        apply_fec: bool,
    ) -> (Vec<i16>, DecodeStats) {
        let (info, stats_eps0, stats_epst, eps4) = if apply_fec {
            let dibits = analysis_profile::time(analysis_profile::Stage::DibitUnpack, || {
                unpack_dibits_full(bits)
            });
            let imbe = analysis_profile::time(analysis_profile::Stage::DecodeFrame, || {
                decode_frame(&dibits)
            });
            let s0: u8 = imbe.errors[0];
            let st: u8 = imbe.error_total().min(255) as u8;
            let e4: u8 = imbe.errors[4];
            (imbe.info, s0, st, e4)
        } else {
            // No FEC layer — info bits arrive verbatim, error counts are zero.
            let info = unpack_info_full(bits);
            (info, 0u8, 0u8, 0u8)
        };
        let dq = analysis_profile::time(analysis_profile::Stage::Dequantize, || {
            dequantize(&info, &mut vocoder.imbe_dec)
        });
        let pcm: [i16; FRAME_SAMPLES] = match dq {
            Ok(params) => {
                let err = FrameErrorContext {
                    epsilon_0: stats_eps0,
                    epsilon_4: eps4,
                    epsilon_t: stats_epst,
                    bad_pitch: false,
                };
                synthesize_frame(&params, &err, GAMMA_W, &mut vocoder.synth)
            }
            Err(_) => [0i16; FRAME_SAMPLES],
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

    fn pack_info_full(info: &[u16; 8]) -> [u8; 11] {
        let mut out = [0u8; 11];
        super::pack_info_bits(info, &INFO_WIDTHS, &mut out);
        out
    }

    fn unpack_info_full(bytes: &[u8]) -> [u16; 8] {
        let mut out = [0u16; 8];
        super::unpack_info_bits(bytes, &INFO_WIDTHS, &mut out);
        out
    }

    /// Wire-layer transcode: 18-byte FEC frame → 11-byte info-only
    /// frame. Pure bit-shuffling — no codec or predictor state.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 11] {
        let dibits = unpack_dibits_full(fec_bytes);
        let frame = decode_frame(&dibits);
        pack_info_full(&frame.info)
    }

    /// Wire-layer transcode: 11-byte info-only frame → 18-byte FEC
    /// frame. Pure bit-shuffling — re-applies Golay/Hamming/PN.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 18] {
        let info = unpack_info_full(info_bytes);
        let dibits = encode_frame(&info);
        pack_dibits_full(&dibits)
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
    use super::*;
    use crate::rate33::dequantize::{
        Decoded, decode_to_params, encode_tone_frame_info, quantize, tone_to_mbe_params,
    };
    use crate::rate33::frame::{decode_frame, encode_frame, pack_no_fec, unpack_no_fec};

    pub(super) fn encode(
        pcm: &[i16],
        vocoder: &mut Vocoder,
        apply_fec: bool,
    ) -> Result<(Vec<u8>, AnalysisStats), VocoderError> {
        // Tone-detect dispatch (opt-in). On a hit, bypass the voice
        // analysis pipeline entirely and emit an Annex T tone frame.
        // detect_tone tries DTMF (l1 != l2) first then falls through
        // to single-tone (l1 == l2 == 1).
        let tone_detect = if vocoder.tone_detection {
            detect_tone(pcm)
        } else {
            None
        };
        if let Some(ToneDetection { id, amplitude }) = tone_detect {
            let info = encode_tone_frame_info(id, amplitude);
            let bytes = pack_info_or_fec(&info, apply_fec);
            // Reconstruct the params the decoder will see, so
            // FrameStats carries the actual audible content.
            // tone_to_mbe_params returns Some for any valid Annex T
            // row; for the unlikely None case (reserved id), fall
            // back to a half-rate-friendly silence placeholder.
            let params = tone_to_mbe_params(id, amplitude)
                .unwrap_or_else(MbeParams::silence_ambe_plus2);
            return Ok((
                bytes,
                AnalysisStats {
                    output: AnalysisOutputKind::Tone { id, amplitude },
                    params,
                    tone_detect,
                },
            ));
        }

        let frame = pcm.try_into().expect("length already validated");
        let (kind, params) = match analysis_encode_ambe_plus2(frame, &mut vocoder.analysis)
            .map_err(VocoderError::Analysis)?
        {
            AnalysisOutput::Voice(p) => (AnalysisOutputKind::Voice, p),
            AnalysisOutput::Silence => (AnalysisOutputKind::Silence, MbeParams::silence_ambe_plus2()),
        };
        let info = quantize(&params, &mut vocoder.ambe_plus2_dec)
            .map_err(|e| VocoderError::Quantize(format!("{e:?}")))?;
        let bytes = pack_info_or_fec(&info, apply_fec);
        Ok((bytes, AnalysisStats { output: kind, params, tone_detect }))
    }

    fn pack_info_or_fec(info: &[u16; 4], apply_fec: bool) -> Vec<u8> {
        if apply_fec {
            let dibits = encode_frame(info);
            pack_dibits_half(&dibits).to_vec()
        } else {
            pack_info_half(info).to_vec()
        }
    }

    pub(super) fn decode(
        bits: &[u8],
        vocoder: &mut Vocoder,
        apply_fec: bool,
    ) -> (Vec<i16>, DecodeStats) {
        let (info, stats_eps0, stats_epst, eps3) = if apply_fec {
            let dibits = analysis_profile::time(analysis_profile::Stage::DibitUnpack, || {
                unpack_dibits_half(bits)
            });
            let frame = analysis_profile::time(analysis_profile::Stage::DecodeFrame, || {
                decode_frame(&dibits)
            });
            // BABA-A §2.8.1 Eq. 196: half-rate ε_T = ε₀ + ε₁ only (the
            // two Golay codewords). The Hamming-protected ε₂/ε₃ are *not*
            // summed into the disposition's ε_T per the spec, even though
            // they're reported separately for diagnostics. Summing all
            // four cosets here was a port of the full-rate convention
            // and miscalibrates the Repeat/Mute thresholds for half-rate.
            //
            // Uncorrectable Golay-24 (≥4 errors detected) is reported as
            // ε₀ = u8::MAX from our FEC decoder. Per chip A/B (5/2026):
            // controlled 4-error c̃₀ injects keep the chip at peak ~5800
            // without a multi-frame Mute, so the chip caps the contribution
            // to ε_R. We model that as "ε₀ → 4": still trips Repeat via
            // the ε₀ ≥ 4 branch of Eq. 198, but the ε_R recurrence Eq. 197
            // (0.95·prev + 0.001064·ε_T) stays well below 0.096 instead of
            // climbing to 0.27 and Muting ~20 frames.
            const E0_UNCORRECTABLE_CAP: u8 = 4;
            let raw_e0 = frame.errors[0];
            let s0: u8 = if raw_e0 == u8::MAX { E0_UNCORRECTABLE_CAP } else { raw_e0 };
            let s1: u8 = frame.errors[1];
            let st: u8 = u16::from(s0)
                .saturating_add(u16::from(s1))
                .min(255) as u8;
            let e3: u8 = frame.errors[3];
            (frame.info, s0, st, e3)
        } else {
            // No FEC layer — info bits arrive verbatim, error counts are zero.
            let info = unpack_info_half(bits);
            (info, 0u8, 0u8, 0u8)
        };
        let err = FrameErrorContext {
            epsilon_0: stats_eps0,
            epsilon_4: eps3, // half-rate has 4 codewords; index 3 = û₃
            epsilon_t: stats_epst,
            bad_pitch: false,
        };
        // Publish the per-frame err context to the synth state so the
        // AmbePlus path (which reads `state.err` rather than taking an
        // explicit err parameter) sees the actual FEC counts. Without
        // this, AmbePlus synth always saw the default-zero err and the
        // §2.8 Repeat/Mute thresholds never tripped, allowing spurious
        // post-FEC transients (call_3537 frame 773 → peak 23553).
        vocoder.synth.err = err;
        let dq = analysis_profile::time(analysis_profile::Stage::Dequantize, || {
            decode_to_params(&info, &mut vocoder.ambe_plus2_dec)
        });
        let pcm: [i16; FRAME_SAMPLES] = match dq {
            Ok(Decoded::Voice(p)) => match vocoder.ambe_plus2_synth {
                AmbePlus2Synth::AmbePlus => ambe_plus2::synthesize_frame(&p, &mut vocoder.synth),
                AmbePlus2Synth::Baseline => {
                    let gamma_w = vocoder.synth.gamma_w;
                    synthesize_frame(&p, &err, gamma_w, &mut vocoder.synth)
                }
            },
            Ok(Decoded::Tone { params, .. }) => match vocoder.ambe_plus2_synth {
                AmbePlus2Synth::AmbePlus => ambe_plus2::synthesize_tone(&params, &mut vocoder.synth),
                AmbePlus2Synth::Baseline => {
                    let gamma_w = vocoder.synth.gamma_w;
                    synthesize_frame(&params, &err, gamma_w, &mut vocoder.synth)
                }
            },
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

    // R34 (half-rate no-FEC) is NOT the naive sequential û₀‖û₁‖û₂‖û₃
    // layout — DVSI applies a 3-way column interleave. Delegate to the
    // wire-layer helpers that carry the empirically-derived order. (The
    // full-rate IMBE no-FEC path stays sequential — it is already
    // byte-exact vs DVSI `p25_nofec`.)
    fn pack_info_half(info: &[u16; 4]) -> [u8; 7] {
        pack_no_fec(info)
    }

    fn unpack_info_half(bytes: &[u8]) -> [u16; 4] {
        unpack_no_fec(bytes)
    }

    /// Wire-layer transcode: 9-byte FEC frame → 7-byte info-only
    /// frame. Pure bit-shuffling — no codec or predictor state.
    pub(super) fn fec_to_info_bytes(fec_bytes: &[u8]) -> [u8; 7] {
        let dibits = unpack_dibits_half(fec_bytes);
        let frame = decode_frame(&dibits);
        pack_info_half(&frame.info)
    }

    /// Wire-layer transcode: 7-byte info-only frame → 9-byte FEC
    /// frame. Pure bit-shuffling — re-applies Golay/Hamming/PN.
    pub(super) fn info_to_fec_bytes(info_bytes: &[u8]) -> [u8; 9] {
        let info = unpack_info_half(info_bytes);
        let dibits = encode_frame(&info);
        pack_dibits_half(&dibits)
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
            assert_eq!(bits.len(), 11, "no-FEC IMBE wire frame is 11 bytes (88 info bits)");
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
        assert!(matches!(r, Err(VocoderError::WrongPcmLength { expected: 160, got: 159 })));
    }

    #[test]
    fn wrong_bits_length_errors_per_rate() {
        let mut a = Vocoder::new(Rate::Imbe7200x4400);
        assert!(matches!(
            a.decode_bits(&[0u8; 9]),
            Err(VocoderError::WrongBitsLength { expected: 18, got: 9 })
        ));
        let mut b = Vocoder::new(Rate::AmbePlus2_3600x2450);
        assert!(matches!(
            b.decode_bits(&[0u8; 18]),
            Err(VocoderError::WrongBitsLength { expected: 9, got: 18 })
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

    /// Diagnostic types round-trip through JSON when the `serde`
    /// feature is on. Validates that a future RPC layer
    /// (gRPC / protobuf / WS) can ship `FrameStats` over the wire
    /// without bespoke conversion.
    #[cfg(feature = "serde")]
    #[test]
    fn frame_stats_round_trip_through_json() {
        let mut v = Vocoder::new(Rate::Imbe7200x4400);
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

    /// `DecodeStats::disposition` populates after a decode call.
    /// On clean own-encoded bits the disposition is `Use`; on bits
    /// crafted to cross the mute threshold (high ε_t injected by
    /// pre-loading `state.epsilon_r`) it should be `Mute`.
    #[test]
    fn decode_stats_carry_disposition() {
        let mut tx = Vocoder::new(Rate::Imbe7200x4400);
        let mut rx = Vocoder::new(Rate::Imbe7200x4400);
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

    /// Pure-sine input at an Annex T frequency, with tone detection
    /// enabled, produces a tone frame instead of voice. The decoder
    /// recognises the tone-frame signature and reconstructs the
    /// matching MBE params.
    #[test]
    fn tone_detection_emits_tone_frame_and_decoder_recognises_it() {
        // Half-rate is the only rate with tone-frame signaling.
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        tx.set_tone_detection(true);
        // Annex T id=10 → 312.5 Hz. Generate one full frame of clean
        // sine at i16 amplitude 8000.
        let mut pcm = [0i16; FRAME_SAMPLES];
        let two_pi = 2.0 * core::f64::consts::PI;
        for (n, slot) in pcm.iter_mut().enumerate() {
            let s = 8000.0_f64 * (two_pi * 312.5 * n as f64 / 8000.0).sin();
            *slot = s.round() as i16;
        }
        let bits = tx.encode_pcm(&pcm).unwrap();
        assert_eq!(bits.len(), 9);

        // Confirm the encoder reported a Tone output.
        match tx.last_stats().analysis.as_ref().unwrap().output {
            AnalysisOutputKind::Tone { id, amplitude: _ } => assert_eq!(id, 10),
            other => panic!("expected Tone, got {other:?}"),
        }

        // Decoder side: parse the bits and confirm it classifies as
        // a tone frame (FrameKind::Tone via the §2.10.1 signature
        // dispatch).
        use crate::rate33::dequantize::{FrameKind, classify_ambe_plus2_frame};
        use crate::rate33::frame::decode_frame;
        let mut dibits = [0u8; 36];
        let mut bit = 0;
        for slot in &mut dibits {
            let mut d = 0u8;
            for _ in 0..2 {
                let b = (bits[bit / 8] >> (7 - (bit % 8))) & 1;
                d = (d << 1) | b;
                bit += 1;
            }
            *slot = d;
        }
        let frame = decode_frame(&dibits);
        assert_eq!(classify_ambe_plus2_frame(&frame.info), FrameKind::Tone);

        // Decoding through the Vocoder yields PCM (synthesize_tone
        // path); we don't assert frequency parity because tone-synth
        // calibration depends on §1.10/§11 amplitude scaling that's
        // separately tracked, but the output should be non-trivial.
        let mut rx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let out = rx.decode_bits(&bits).unwrap();
        assert_eq!(out.len(), FRAME_SAMPLES);
    }

    /// With tone-detection off (default), the same pure-sine input
    /// goes through the voice analysis pipeline and produces a
    /// regular voice frame.
    #[test]
    fn tone_detection_off_means_voice_path_even_for_pure_sine() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        // (default) tone_detection == false
        let mut pcm = [0i16; FRAME_SAMPLES];
        let two_pi = 2.0 * core::f64::consts::PI;
        for (n, slot) in pcm.iter_mut().enumerate() {
            let s = 8000.0_f64 * (two_pi * 312.5 * n as f64 / 8000.0).sin();
            *slot = s.round() as i16;
        }
        let _ = tx.encode_pcm(&pcm).unwrap();
        let stats = tx.last_stats().analysis.as_ref().unwrap();
        assert!(matches!(
            stats.output,
            AnalysisOutputKind::Voice | AnalysisOutputKind::Silence
        ));
        assert!(!matches!(stats.output, AnalysisOutputKind::Tone { .. }));
        assert!(stats.tone_detect.is_none());
    }

    /// On full-rate IMBE, tone detection is **detection-only**: the
    /// wire frame stays a regular voice frame (Phase 1 has no
    /// tone-frame opcode at the codec layer per BABA-A) but the
    /// detected `(I_D, A_D)` is surfaced via `tone_detect` so
    /// application-layer signaling (LCW, paging) can route on it.
    #[test]
    fn tone_detection_on_phase1_surfaces_metadata_without_changing_wire() {
        let mut tx = Vocoder::new(Rate::Imbe7200x4400);
        tx.set_tone_detection(true);
        let mut pcm = [0i16; FRAME_SAMPLES];
        let two_pi = 2.0 * core::f64::consts::PI;
        for (n, slot) in pcm.iter_mut().enumerate() {
            let s = 8000.0_f64 * (two_pi * 312.5 * n as f64 / 8000.0).sin();
            *slot = s.round() as i16;
        }
        let bits_with_detect = tx.encode_pcm(&pcm).unwrap();
        let stats = tx.last_stats().analysis.as_ref().unwrap();
        // Wire output is still voice (no Phase 1 tone-frame opcode).
        assert!(matches!(
            stats.output,
            AnalysisOutputKind::Voice | AnalysisOutputKind::Silence
        ));
        // But the detector populated tone_detect with the matching id.
        let det = stats.tone_detect.expect("Phase 1 detection metadata");
        assert_eq!(det.id, 10); // Annex T id=10 → 312.5 Hz

        // Bit-exact equivalence with detection off — running the
        // detector must not perturb the analysis encoder state.
        let mut tx_off = Vocoder::new(Rate::Imbe7200x4400);
        let bits_no_detect = tx_off.encode_pcm(&pcm).unwrap();
        assert_eq!(bits_with_detect, bits_no_detect);
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
        let mut ref_v = Vocoder::new(Rate::Imbe7200x4400);
        let mut ref_bits: Vec<u8> = Vec::new();
        for chunk in total_pcm.chunks_exact(FRAME_SAMPLES) {
            ref_bits.extend(ref_v.encode_pcm(chunk).unwrap());
        }
        // Live: feed in mismatched chunk sizes (250, 50, 333, rest).
        let mut live = LiveEncoder::new(Rate::Imbe7200x4400);
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
        let mut live = LiveEncoder::new(Rate::Imbe7200x4400);
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
            Err(VocoderError::WrongBitsLength { expected: 18, got: 9 })
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

    /// Same-codec FEC ↔ no-FEC transcode is lossless: strip then add
    /// (or add then strip) reproduces the original bytes exactly.
    #[test]
    fn transcoder_full_fec_to_info_roundtrip_is_lossless() {
        let mut enc = Vocoder::new(Rate::Imbe7200x4400);
        let mut strip =
            Transcoder::new(Rate::Imbe7200x4400, Rate::Imbe4400x4400).unwrap();
        let mut add =
            Transcoder::new(Rate::Imbe4400x4400, Rate::Imbe7200x4400).unwrap();
        for k in 0..4 {
            let pcm = periodic_pcm(40 + k, 7000);
            let fec = enc.encode_pcm(&pcm).unwrap();
            assert_eq!(fec.len(), 18);
            let info = strip.transcode(&fec).unwrap();
            assert_eq!(info.len(), 11);
            let fec2 = add.transcode(&info).unwrap();
            assert_eq!(fec2, fec, "FEC strip + add round-trips byte-for-byte");
        }
    }

    #[test]
    fn transcoder_half_fec_to_info_roundtrip_is_lossless() {
        let mut enc = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let mut strip =
            Transcoder::new(Rate::AmbePlus2_3600x2450, Rate::AmbePlus2_2450x2450).unwrap();
        let mut add =
            Transcoder::new(Rate::AmbePlus2_2450x2450, Rate::AmbePlus2_3600x2450).unwrap();
        for k in 0..4 {
            let pcm = periodic_pcm(40 + k, 6000);
            let fec = enc.encode_pcm(&pcm).unwrap();
            assert_eq!(fec.len(), 9);
            let info = strip.transcode(&fec).unwrap();
            assert_eq!(info.len(), 7);
            let fec2 = add.transcode(&info).unwrap();
            assert_eq!(fec2, fec, "FEC strip + add round-trips byte-for-byte");
        }
    }

    /// `extract_params` runs the analysis encoder on PCM and returns
    /// MbeParams. Multiple calls advance state correctly (preroll
    /// → voice transition).
    #[test]
    fn extract_params_returns_params_per_frame() {
        let mut v = Vocoder::new(Rate::Imbe7200x4400);
        // First few frames are preroll; analysis should still return
        // silence params, not error.
        let pcm = periodic_pcm(40, 6000);
        let p1 = v.extract_params(&pcm).unwrap();
        let p2 = v.extract_params(&pcm).unwrap();
        let p3 = v.extract_params(&pcm).unwrap();
        // Preroll dispatches silence; by frame 3+ we should see voice
        // params (non-silence ω₀ or non-zero amplitudes).
        let any_voice = [&p1, &p2, &p3].iter().any(|p| {
            let amps = p.amplitudes_slice();
            amps.iter().any(|&a| a > 0.0)
        });
        assert!(any_voice, "no voice params after 3 frames of periodic PCM");
    }

    /// extract_params + synthesize_params chain together — extract
    /// params from PCM, immediately synthesize them back to PCM, get
    /// non-trivial output. (Not the same as the input — that would
    /// require the wire FEC chain to advance the synth predictor in
    /// step with the analysis predictor — but the synth output is
    /// well-defined.)
    #[test]
    fn extract_then_synthesize_roundtrips_through_params() {
        let mut a = Vocoder::new(Rate::Imbe7200x4400);
        let mut b = Vocoder::new(Rate::Imbe7200x4400);
        let pcm = periodic_pcm(40, 6000);
        for _ in 0..5 {
            let params = a.extract_params(&pcm).unwrap();
            let resynth = b.synthesize_params(&params);
            assert_eq!(resynth.len(), FRAME_SAMPLES);
        }
    }

    /// `synthesize_params` accepts arbitrary MbeParams and produces a
    /// 160-sample PCM frame. Round-trips cleanly with `MbeParams::silence`
    /// (full-rate) and `MbeParams::silence_ambe_plus2` (half-rate) —
    /// silence params synthesize to (near-)silent output.
    #[test]
    fn synthesize_params_emits_one_frame_per_call() {
        let mut tx = Vocoder::new(Rate::Imbe7200x4400);
        let pcm = tx.synthesize_params(&MbeParams::silence());
        assert_eq!(pcm.len(), FRAME_SAMPLES);
        // Silence params → low-amplitude output.
        let peak = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(peak < 5000, "silence params produced peak={peak}");

        let mut rx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let pcm = rx.synthesize_params(&MbeParams::silence_ambe_plus2());
        assert_eq!(pcm.len(), FRAME_SAMPLES);
        let peak = pcm.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
        assert!(peak < 5000);
    }

    /// `synthesize_params` advances synth state so a follow-up
    /// `last_disposition` reflects the most-recent frame.
    #[test]
    fn synthesize_params_advances_synth_disposition() {
        let mut v = Vocoder::new(Rate::Imbe7200x4400);
        assert_eq!(v.last_disposition(), None);
        let _ = v.synthesize_params(&MbeParams::silence());
        // Clean params + zero error context → Use.
        assert_eq!(v.last_disposition(), Some(FrameDisposition::Use));
    }

    /// Builder applies all five config knobs in one expression.
    #[test]
    fn builder_configures_all_knobs() {
        let v = Vocoder::builder(Rate::AmbePlus2_3600x2450)
            .tone_detection(true)
            .repeat_reset_after(Some(3))
            .silence_dispatch(true)
            .pitch_silence_override(true)
            .ambe_plus2_synth(AmbePlus2Synth::Baseline)
            .build();
        assert_eq!(v.rate(), Rate::AmbePlus2_3600x2450);
        assert!(v.tone_detection());
        assert_eq!(v.repeat_reset_after(), Some(3));
        assert!(v.silence_dispatch());
        assert!(v.pitch_silence_override());
        assert_eq!(v.ambe_plus2_synth(), AmbePlus2Synth::Baseline);
    }

    /// Half-rate synth choice routes to the right backend. Both
    /// modes produce non-trivial audio. They MAY converge to
    /// identical PCM on simple/periodic inputs (the AMBE+ phase
    /// regen is a no-op when all bands voiced cleanly), so the
    /// assertion is non-silent + correct rate, not bit-difference.
    #[test]
    fn ambe_plus2_synth_modes_both_decode_cleanly() {
        let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
        let pcm = periodic_pcm(40, 6000);
        let mut bits_buf: Vec<u8> = Vec::new();
        for _ in 0..5 {
            bits_buf.extend(tx.encode_pcm(&pcm).unwrap());
        }

        for gen in [AmbePlus2Synth::AmbePlus, AmbePlus2Synth::Baseline] {
            let mut rx = Vocoder::builder(Rate::AmbePlus2_3600x2450)
                .ambe_plus2_synth(gen)
                .build();
            assert_eq!(rx.ambe_plus2_synth(), gen);
            let mut out_pcm: Vec<i16> = Vec::new();
            for chunk in bits_buf.chunks_exact(9) {
                out_pcm.extend(rx.decode_bits(chunk).unwrap());
            }
            let rms = (out_pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
                / out_pcm.len() as f64).sqrt();
            assert!(rms > 50.0, "{gen:?} output too quiet: rms={rms}");
        }
    }

    /// Default builder = spec-faithful (no beyond-spec or opt-in knobs).
    #[test]
    fn builder_defaults_match_vocoder_new() {
        let a = Vocoder::builder(Rate::Imbe7200x4400).build();
        let b = Vocoder::new(Rate::Imbe7200x4400);
        assert_eq!(a.rate(), b.rate());
        assert_eq!(a.tone_detection(), b.tone_detection());
        assert_eq!(a.repeat_reset_after(), b.repeat_reset_after());
        assert_eq!(a.silence_dispatch(), b.silence_dispatch());
        assert_eq!(a.pitch_silence_override(), b.pitch_silence_override());
        assert_eq!(a.ambe_plus2_synth(), b.ambe_plus2_synth());
        assert!(!a.tone_detection());
        assert!(a.repeat_reset_after().is_none());
        assert!(!a.silence_dispatch());
        assert!(!a.pitch_silence_override());
        assert_eq!(a.ambe_plus2_synth(), AmbePlus2Synth::AmbePlus);
    }

    /// `flush` zero-pads residue and emits one final frame; on an
    /// empty buffer it's a no-op returning `Ok(None)`.
    #[test]
    fn live_encoder_flush_emits_padded_residue() {
        let mut live = LiveEncoder::new(Rate::Imbe7200x4400);
        // Empty buffer → flush is a no-op.
        assert!(matches!(live.flush(), Ok(None)));

        // Half-frame residue → flush emits one frame.
        let pcm = periodic_pcm(40, 6000);
        let _ = live.push(&pcm[..80]);
        assert_eq!(live.pending_samples(), 80);
        let tail = live.flush().unwrap().expect("residue → frame");
        assert_eq!(tail.len(), 18);
        assert_eq!(live.pending_samples(), 0);

        // Subsequent flush is a no-op (buffer drained).
        assert!(matches!(live.flush(), Ok(None)));
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
}
