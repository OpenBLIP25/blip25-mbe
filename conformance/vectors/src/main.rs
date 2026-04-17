//! Vector-conformance harness — runs DVSI published test vectors through
//! blip25-mbe and reports per-vector pass/fail.
//!
//! Not published to crates.io. Expects `DVSI/Vectors/` at the workspace
//! root (symlink is fine) with the standard `tv-std/` and `tv-rc/` layouts.
//!
//! ## File-format inference
//!
//! The DVSI `.bit` files are not formally documented in the impl spec
//! (the format lives in DVSI's own client). We infer the most natural
//! conventions and validate them empirically by cross-checking the FEC
//! and no-FEC paths against each other:
//!
//! * **`p25/<name>.bit`** (full-FEC) — 18 bytes per 20 ms frame = 144
//!   bits packed MSB-first. The 144 bits are 72 dibits in transmission
//!   order, each dibit's high bit transmitted first. Feeds directly
//!   into [`blip25_mbe::p25_fullrate::frame::decode_frame`].
//!
//! * **`p25_nofec/<name>.bit`** (info-only) — 11 bytes per 20 ms frame
//!   = 88 bits packed MSB-first. The 88 bits are the eight info
//!   vectors `û₀..û₇` concatenated in order with widths 12, 12, 12, 12,
//!   11, 11, 11, 7 bits — same byte stream the encoder produces after
//!   bit prioritization, before any FEC.
//!
//! Cross-check: decoding `p25/<name>.bit` via the channel codec should
//! produce the exact `[u16; 8]` info vectors stored in
//! `p25_nofec/<name>.bit`. Bit-exact agreement validates both the
//! channel codec (deinterleave + Golay/Hamming + PN) and the no-FEC
//! packing convention against DVSI ground truth.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

use blip25_mbe::codecs::mbe_baseline::{
    FrameErrorContext, GAMMA_W, SynthState, synthesize_frame, synthesize_frame_ambe_plus,
};
use blip25_mbe::codecs::mbe_baseline::analysis::{
    AnalysisError, AnalysisOutput, AnalysisState, VuvResult,
    band_count_for, band_for_harmonic,
    encode_halfrate as analysis_encode_halfrate,
    encode_with_trace as analysis_encode_with_trace,
};
use blip25_mbe::fec::{golay_23_12_encode, hamming_15_11_encode};
use blip25_mbe::p25_fullrate::dequantize::{
    DecodeError, DecoderState, dequantize as dequantize_full, quantize as quantize_full,
};
use blip25_mbe::p25_fullrate::fec::{deinterleave as deinterleave_full, modulation_masks};
use blip25_mbe::p25_fullrate::frame::{
    Frame as FullFrame, INFO_WIDTHS,
    decode_frame as decode_full_frame, encode_frame as encode_full_frame,
};
use blip25_mbe::p25_fullrate::priority::{
    deprioritize as deprioritize_full, prioritize as prioritize_full,
};
use blip25_mbe::p25_halfrate::dequantize::{
    Decoded, DecoderState as HalfDecoderState, decode_to_params,
    dequantize as dequantize_half, quantize as quantize_half,
};
use blip25_mbe::p25_halfrate::frame::{
    decode_frame as decode_half_frame, deinterleave as deinterleave_half,
};
use blip25_mbe::p25_halfrate::priority::{
    deprioritize as deprioritize_half,
};
use blip25_mbe::rate_conversion::converter::{
    ConvertError, FullToHalfConverter, HalfToFullConverter,
};
use blip25_mbe::mbe_params::MbeParams;

const BYTES_PER_FEC_FRAME: usize = 18;
const BYTES_PER_NOFEC_FRAME: usize = 11;
const DIBITS_PER_FRAME: usize = 72;

/// Run DVSI test vectors through blip25-mbe and report conformance.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the DVSI vectors directory (containing `tv-std/`, `tv-rc/`).
    #[arg(long, default_value = "DVSI/Vectors")]
    vectors: PathBuf,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Cross-validate the channel codec: decode `p25/<name>.bit` and
    /// compare its 88 info bits per frame against `p25_nofec/<name>.bit`.
    /// Bit-exact agreement = channel codec + nofec packing both correct.
    Compare {
        /// Vector name (e.g. `alert`, `clean`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// Stop after the first divergence, printing diagnostics.
        #[arg(long)]
        stop_on_first: bool,
    },
    /// Decode `p25/<name>.bit` (FEC frames) all the way to PCM via
    /// the §1.10–§1.12 synthesizer, and compare the output to
    /// `p25/<name>.pcm` (the DVSI reference). Reports sample-count
    /// match, RMS error, peak error, and signal-to-noise ratio.
    /// Bit-exact agreement is **not** expected — γ_w is a placeholder
    /// and the synthesizer's float precision differs from DVSI's
    /// internal arithmetic.
    DecodePcm {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// Override γ_w (default: spec value 146.643269).
        #[arg(long)]
        gamma_w: Option<f64>,
        /// Global scale factor applied to every M̃_l before
        /// synthesis. Default 1.0. Candidate hypothesis for the §11
        /// γ_w investigation: our dequant produces M̃_l ≈ 150× too
        /// large, so a scale of 1/110 or similar should improve SNR.
        #[arg(long, default_value_t = 1.0)]
        m_scale: f64,
        /// If set, write the synthesized PCM to this path (raw
        /// 16-bit little-endian, 8 kHz) for auditory A/B against
        /// DVSI reference.
        #[arg(long)]
        write_pcm: Option<std::path::PathBuf>,
    },
    /// Scan a full-rate vector for V/UV transition frames that
    /// isolate §1.12.2 Eq. 131 vs Eq. 132 candidates, then run xcorr
    /// on each and report which candidate's branch shows the
    /// phase-inversion signature. The §11 investigation's candidate
    /// discriminator.
    ScanTransitions {
        /// Vector name (e.g. `clean`).
        name: String,
        /// Minimum voiced harmonic change required to call it a
        /// transition (out of L). Default 0.6.
        #[arg(long, default_value_t = 0.6)]
        min_transition_ratio: f32,
        /// Maximum number of each transition type to print.
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Single-frame cross-correlation diagnostic for the §11 voiced-
    /// synthesis investigation. Decodes one frame from the full-rate
    /// FEC vector, runs the §1.12 synthesizer (voiced-only unless
    /// `--both` is set), and compares to the DVSI reference PCM for
    /// the same 160-sample window. Reports peak correlation, zero-lag
    /// correlation, and suggests which equation subset is implicated.
    XcorrFrame {
        /// Vector name (e.g. `clean`).
        name: String,
        /// Frame index to inspect (0-based).
        #[arg(long, default_value_t = 0)]
        frame: usize,
        /// Include the unvoiced component (default: voiced-only).
        #[arg(long)]
        both: bool,
        /// Override γ_w (only meaningful with `--both`).
        #[arg(long)]
        gamma_w: Option<f64>,
        /// M̃_l scale factor. Default 1.0.
        #[arg(long, default_value_t = 1.0)]
        m_scale: f64,
    },
    /// Half-rate bit-level roundtrip: decode each `r33/<name>.bit`
    /// frame to `MbeParams`, re-encode via `quantize`, and
    /// compare the recovered `û` vectors to the original. Pitch
    /// (`b̂₀`) and V/UV row-0 endpoints should roundtrip bit-exact;
    /// gain and spectral params may drift (see `dequantize`
    /// tests for the known stability boundaries).
    HalfrateRoundtrip {
        /// Vector name (e.g. `alert`).
        name: String,
    },
    /// Half-rate equivalent of `decode-pcm`. Reads
    /// `r33/<name>.bit` (rate index 39 = 2450 voice + 1150 FEC = 3600 bps total per the
    /// USB-3000 manual §3.3.4), runs the full half-rate pipeline:
    /// dibits → Frame → MbeParams → PCM via the shared §1.12
    /// synthesizer, and compares to `r33/<name>.pcm`.
    DecodePcmHalfrate {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Override γ_w (default: spec value 146.643269).
        #[arg(long)]
        gamma_w: Option<f64>,
        /// Optional path to write the synthesized PCM (raw 16-bit
        /// little-endian, mono, 8 kHz). Useful for audible A/B
        /// comparison against the DVSI reference.
        #[arg(long)]
        write_pcm: Option<std::path::PathBuf>,
        /// Phase mode for voiced synthesis. `ambe-plus` (default,
        /// spec-conformant for half-rate per AMBE-3000_Decoder §5) uses
        /// US5701390 Hilbert phase regeneration; `baseline` uses
        /// BABA-A Eq. 141 noise-perturbed phase. Exposed for A/B
        /// listening tests of the phase regeneration's perceptual
        /// claim.
        #[arg(long, value_enum, default_value_t = HalfratePhaseMode::AmbePlus)]
        phase_mode: HalfratePhaseMode,
    },
    /// Round-trip test: decode each `p25_nofec/<name>.bit` frame to
    /// `MbeParams` then re-encode. Pitch (`b̂₀`) and V/UV (`b̂₁`) are
    /// expected to match bit-exact; gain DCT bits (`b̂₂..b̂₇`) and HOC
    /// bits (`b̂₈..b̂_{L+1}`) may drift because the encoder/decoder
    /// gain-DCT pair is intentionally lossy by design (Eq. 61 vs
    /// Eq. 69 — see analysis disambiguations §9).
    Roundtrip {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
    },
    /// Dump per-frame half-rate decode state for the first N voice
    /// frames of `r33/<name>.bit`. Logs ω₀, L, voicing pattern,
    /// amplitude summary, and PCM peak/RMS after synthesis — lets us
    /// see where in the dequantize → synth chain the output is going
    /// to silence.
    HalfrateInspect {
        /// Vector name (e.g. `clean`).
        name: String,
        /// How many voice frames to dump.
        #[arg(long, default_value_t = 3)]
        frames: usize,
    },
    /// Synthetic encoder diagnostic: build speech-scale MbeParams,
    /// run quantize, and log the b̂ vector it produces. If
    /// our quantizer + Γ̃ derivation are correct, b̂₂ should land in
    /// the high-20s to 31 for typical speech (§13.2).
    HalfrateEncoderSanity,
    /// Dump Annex S → Golay seam for one frame: 24-bit ĉ₀ after
    /// deinterleave, 12-bit û₀ after Golay decode, and the pairwise
    /// bit-identity check between ĉ₀[0..11] (systematic info positions
    /// per G's [I | P] form) and û₀[0..11].
    HalfrateSeamDump {
        /// Vector name (e.g. `clean`).
        name: String,
        /// Frame index to inspect (0-based).
        #[arg(long, default_value_t = 1)]
        frame: usize,
    },
    /// Golay [24,12] error-count histogram across every frame in a
    /// half-rate `.bit` file. Diagnostic for the
    /// `analysis/dvsi_test_vector_modes.md` fingerprint: rate-33
    /// (FEC) input should report 0 errors on every frame; rate-39
    /// (no-FEC) input fed through the same path produces the
    /// `C(24,k)/2325` histogram shape that signals a rate-mode mixup.
    HalfrateFecHistogram {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Rate-mode subdirectory under `tv-rc/` or `tv-std/tv/`
        /// (`r33`, `r34`, or `r39`).
        #[arg(long, default_value = "r33")]
        rate: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
    },
    /// Diagnose PN modulation by re-encoding the no-FEC reference and
    /// XORing against the deinterleaved FEC frame to recover DVSI's
    /// actual mask. Compare to our computed mask to see exactly how
    /// the conventions diverge.
    PnDiag {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// Frame index to inspect.
        #[arg(long, default_value_t = 0)]
        frame: usize,
    },
    /// Cross-rate conversion end-to-end: feed a full-rate DVSI FEC
    /// vector (`p25/<name>.bit`) through `FullToHalfConverter` and
    /// re-decode the 36-dibit half-rate output, or feed a half-rate
    /// vector (`r33/<name>.bit`) through `HalfToFullConverter` and
    /// re-decode the 72-dibit full-rate output. Reports per-frame
    /// parameter-level distortion between the source-rate `MbeParams`
    /// and the destination-rate `MbeParams` (after one quantization
    /// pass through the destination rate's grid).
    RateConvert {
        /// Direction of conversion.
        #[arg(long, value_enum)]
        direction: RateConvertDirection,
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
    },
    /// Cross-rate round-trip: full → half → full, or half → full → half.
    /// Measures cumulative distortion after two quantization passes
    /// through both rate grids plus two full decode/re-encode cycles.
    /// This is the strongest "does it actually work end-to-end" test
    /// of the parametric rate-conversion chain.
    RateConvertRoundtrip {
        /// Direction of the first hop (A→B). The round-trip comes back
        /// to the A rate.
        #[arg(long, value_enum)]
        direction: RateConvertDirection,
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
    },
    /// Measure target-stream magnitude smoothness: RMS of
    /// `|M̃_l(frame N+1) − M̃_l(frame N)|` averaged over target-rate
    /// frames. This is the metric the §4.5 cross-rate predictor
    /// actually targets (US7634399 col. 7 "frame-to-frame magnitude
    /// instability over ~50 frames"); per-frame source-vs-output
    /// comparison (as `rate-convert-roundtrip` does) doesn't detect
    /// the predictor's smoothing effect. Reports both baseline
    /// (predictor off) and with-predictor numbers in one run for
    /// direct A/B comparison.
    /// Half-rate PCM → MbeParams analysis encoder conformance. Drives
    /// `DVSI/Vectors/.../r33/<name>.pcm` through
    /// `codecs::mbe_baseline::analysis::encode_halfrate` in 20 ms frames,
    /// dequantizes the parallel `<name>.bit` chip bitstream via
    /// `p25_halfrate::dequantize` for the reference `MbeParams`, and
    /// reports parameter-level distortion (pitch cents, L mismatch %,
    /// voicing %, amplitude dB RMSE). Analogous to `analysis-encode`
    /// but for the half-rate wire format. Preroll frames (first two)
    /// are tallied non-comparable.
    HalfrateAnalysisEncode {
        /// Vector name (e.g. `clean`, `alert`).
        name: String,
        /// Compare encoder output at iteration `f` against chip bit-frame
        /// `f + chip_offset`. Default −2 matches the encoder's 2-frame
        /// look-ahead delay (same convention as `analysis-encode`).
        #[arg(long, default_value_t = -2)]
        chip_offset: i32,
        /// Enable the encoder's §0.8.4 silence detector. Silent frames
        /// emit `AnalysisOutput::Silence` and are tallied under
        /// "encoder silence" rather than compared.
        #[arg(long)]
        silence_dispatch: bool,
    },
    RateConvertSmoothness {
        /// Conversion direction — i.e. which target stream's
        /// smoothness we measure.
        #[arg(long, value_enum)]
        direction: RateConvertDirection,
        /// Vector name (e.g. `alert`, `clean`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
    },
    /// PCM → `MbeParams` analysis encoder conformance. Drives
    /// `DVSI/Vectors/.../p25/<name>.pcm` through
    /// [`blip25_mbe::codecs::mbe_baseline::analysis::encode`] in 20 ms
    /// frames, dequantizes the parallel `<name>.bit` chip bitstream
    /// via `p25_fullrate::dequantize` to get the reference per-frame
    /// `MbeParams`, and reports parameter-level distortion
    /// (pitch cents, L mismatch %, voicing %, amplitude dB RMSE).
    /// Preroll frames (first two) are tallied as non-comparable.
    /// Expected baseline per project memory: pitch/voicing close,
    /// amplitude RMSE large until §11 / γ_w / M̃_l calibration bugs
    /// are resolved on live chip.
    AnalysisEncode {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// If > 0, print per-frame `(chip ω̂_0, encoder ω̂_0, Δcents,
        /// chip L, encoder L, voicing)` for the first N post-preroll
        /// comparable frames. Localizes where in the pitch tracker
        /// the encoder diverges from the chip reference (sub-multiple,
        /// cold-start bias, look-back/look-ahead).
        #[arg(long, default_value_t = 0)]
        dump_frames: usize,
        /// Compare encoder output at iteration `f` against chip bit-
        /// frame `f + chip_offset`. Our encoder has a structural
        /// 2-frame look-ahead delay (`encode(pcm_f)` for `f ≥ 2`
        /// emits the analysis of `pcm_{f-2}`). DVSI's chip does not
        /// visibly delay its output stream (bit-frame N = analysis
        /// of pcm_N), so default `-2` aligns both to the same
        /// analyzed PCM frame. Empirically verified on `clean.pcm`:
        /// voicing 70.7% → 78.2%, amplitude RMS 20.2 → 14.2 dB going
        /// from offset 0 to −2. Iters where `f + offset` is out of
        /// range are tallied as non-comparable.
        #[arg(long, default_value_t = -2, allow_hyphen_values = true)]
        chip_offset: i32,
        /// Enable the encoder's §0.8.4 silence detector. Silent frames
        /// emit `AnalysisOutput::Silence` and are tallied under
        /// `encoder silence` rather than compared. Off by default per
        /// addendum §0.8.8 (pass-through); enable when the chip
        /// reference is known to dispatch silence via a default pitch
        /// index (e.g. b̂_0=25 on low-energy frames), which otherwise
        /// pollutes aggregate pitch/L stats.
        #[arg(long)]
        silence_dispatch: bool,
        /// Run each voice frame through quantize → prioritize →
        /// dequantize and report three stat blocks: (1) pre-quantizer
        /// encoder vs chip (the existing metric), (2) post-roundtrip
        /// encoder vs chip (apples-to-apples, both post-quantizer),
        /// (3) quantizer distortion (encoder pre-Q vs its own
        /// roundtrip). Isolates whether the amplitude residual is
        /// pre- or post-quantizer.
        #[arg(long)]
        quantizer_roundtrip: bool,
        /// After each frame, override the encoder's V/UV history
        /// (`vuv_prev`) with the chip's band-level voicing decisions.
        /// Breaks the self-reinforcing feedback loop: if disagreement
        /// drops significantly vs the unseeded run, the feedback
        /// divergence is the dominant cause of voicing mismatch.
        #[arg(long)]
        chip_seed_vuv: bool,
        /// Wire each voice frame through the full encode path
        /// (quantize → prioritize → encode_frame) and compare the
        /// resulting info vectors and dibits against the chip's
        /// `.bit` file. Reports per-info-vector and per-parameter
        /// bit match rates.
        #[arg(long)]
        bitstream: bool,
    },
    /// Run the normative P25 full-rate DVSI test vector sweep driven by
    /// `tv-std/tv/cmpp25.txt` (576 test lines × {alert, clean, cp*,
    /// dam*, dtmf*, dtone*, sine*, t*, tambf*, tia*, ucarf*, ucarm*,
    /// xfer, zero}). Each line is one of:
    ///
    ///   * `-dec <input.bit> -cmp <ref.pcm>` — bit → PCM (via §1.10/§1.11/§1.12)
    ///   * `-enc <input.pcm> -cmp <ref.bit>` — PCM → bit (via §0 analysis + §5/§8 wire encode)
    ///   * `-encdec <input.pcm> -cmp <ref.pcm>` — PCM → bit → PCM round-trip
    ///
    /// FEC mode is determined by the config words on each line: `0x1030`
    /// enables FEC (→ `p25/` target dir), `0x0000` disables it (→
    /// `p25_nofec/` target dir).
    ///
    /// Reports per-op and per-content-category summaries with bit-exact
    /// rates on decode paths and PCM RMS / SNR on encode/encdec paths.
    /// Soft-decision decode lines (`-sdbits 4`) are skipped — that
    /// feature is not implemented.
    P25Sweep {
        /// Path to the test-command file. Defaults to
        /// `<vectors>/tv-std/tv/cmpp25.txt`.
        #[arg(long)]
        cmp: Option<PathBuf>,
        /// Optional input-stem filter (substring match on the input
        /// filename, e.g. `clean`, `dtmf`, `dam`). When set, only
        /// matching lines run — useful for debugging one content
        /// category.
        #[arg(long)]
        filter: Option<String>,
        /// Optional op filter: run only `dec`, `enc`, or `encdec`
        /// lines. When unset, runs all three.
        #[arg(long)]
        op: Option<String>,
        /// Stop on first failure within each op, printing the failing
        /// line plus per-line diagnostics.
        #[arg(long)]
        stop_on_first: bool,
    },
}

/// Direction argument for [`Cmd::RateConvert`] and
/// [`Cmd::RateConvertRoundtrip`].
#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum RateConvertDirection {
    /// Full-rate IMBE (144-bit) → half-rate AMBE+2 (72-bit).
    FullToHalf,
    /// Half-rate AMBE+2 (72-bit) → full-rate IMBE (144-bit).
    HalfToFull,
}

/// Phase-mode selector for [`Cmd::DecodePcmHalfrate`]. Default is
/// `AmbePlus` (spec-conformant for half-rate per AMBE-3000_Decoder §5).
#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum HalfratePhaseMode {
    /// BABA-A Eq. 141 noise-perturbed phase (full-rate default).
    Baseline,
    /// US5701390 §5 Hilbert phase regeneration (half-rate default).
    AmbePlus,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Cmd::Compare { name, rc, stop_on_first } => {
            cmd_compare(&args.vectors, &name, rc, stop_on_first)
        }
        Cmd::PnDiag { name, rc, frame } => cmd_pn_diag(&args.vectors, &name, rc, frame),
        Cmd::Roundtrip { name, rc } => cmd_roundtrip(&args.vectors, &name, rc),
        Cmd::DecodePcm { name, rc, gamma_w, m_scale, write_pcm } => cmd_decode_pcm(
            &args.vectors,
            &name,
            rc,
            gamma_w.unwrap_or(GAMMA_W),
            m_scale,
            write_pcm.as_deref(),
        ),
        Cmd::DecodePcmHalfrate { name, gamma_w, write_pcm, phase_mode } => {
            cmd_decode_pcm_halfrate(
                &args.vectors,
                &name,
                gamma_w.unwrap_or(GAMMA_W),
                write_pcm.as_deref(),
                phase_mode,
            )
        }
        Cmd::HalfrateInspect { name, frames } => {
            cmd_halfrate_inspect(&args.vectors, &name, frames)
        }
        Cmd::HalfrateEncoderSanity => cmd_halfrate_encoder_sanity(),
        Cmd::HalfrateSeamDump { name, frame } => {
            cmd_halfrate_seam_dump(&args.vectors, &name, frame)
        }
        Cmd::HalfrateRoundtrip { name } => cmd_halfrate_roundtrip(&args.vectors, &name),
        Cmd::HalfrateFecHistogram { name, rate, rc } => {
            cmd_halfrate_fec_histogram(&args.vectors, &name, &rate, rc)
        }
        Cmd::ScanTransitions { name, min_transition_ratio, limit } => {
            cmd_scan_transitions(&args.vectors, &name, min_transition_ratio, limit)
        }
        Cmd::XcorrFrame { name, frame, both, gamma_w, m_scale } => cmd_xcorr_frame(
            &args.vectors,
            &name,
            frame,
            both,
            gamma_w.unwrap_or(GAMMA_W),
            m_scale,
        ),
        Cmd::RateConvert { direction, name, rc } => {
            cmd_rate_convert(&args.vectors, direction, &name, rc)
        }
        Cmd::RateConvertRoundtrip { direction, name, rc } => {
            cmd_rate_convert_roundtrip(&args.vectors, direction, &name, rc)
        }
        Cmd::RateConvertSmoothness { direction, name, rc } => {
            cmd_rate_convert_smoothness(&args.vectors, direction, &name, rc)
        }
        Cmd::HalfrateAnalysisEncode { name, chip_offset, silence_dispatch } => {
            cmd_halfrate_analysis_encode(&args.vectors, &name, chip_offset, silence_dispatch)
        }
        Cmd::AnalysisEncode { name, rc, dump_frames, chip_offset, silence_dispatch, quantizer_roundtrip, chip_seed_vuv, bitstream } => {
            cmd_analysis_encode(&args.vectors, &name, rc, dump_frames, chip_offset, silence_dispatch, quantizer_roundtrip, chip_seed_vuv, bitstream)
        }
        Cmd::P25Sweep { cmp, filter, op, stop_on_first } => {
            cmd_p25_sweep(&args.vectors, cmp.as_deref(), filter.as_deref(), op.as_deref(), stop_on_first)
        }
    }
}

const FRAME_SAMPLES: usize = 160;

fn scale_params(params: &MbeParams, scale: f64) -> MbeParams {
    let l = params.harmonic_count();
    let voiced: Vec<bool> = params.voiced_slice().to_vec();
    let amps: Vec<f32> = params
        .amplitudes_slice()
        .iter()
        .map(|&a| ((f64::from(a) * scale) as f32).max(0.0))
        .collect();
    MbeParams::new(params.omega_0(), l, &voiced, &amps)
        .expect("scaled params valid (non-negative, finite)")
}

fn cmd_decode_pcm(
    root: &Path,
    name: &str,
    rc: bool,
    gamma_w: f64,
    m_scale: f64,
    write_pcm: Option<&Path>,
) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let pcm_path = dir.join("p25").join(format!("{name}.pcm"));

    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    let pcm_ref_bytes = fs::read(&pcm_path)
        .with_context(|| format!("read reference PCM {}", pcm_path.display()))?;

    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "FEC file {} is {} bytes — not a multiple of {}",
            fec_path.display(),
            fec_bytes.len(),
            BYTES_PER_FEC_FRAME
        ));
    }
    if pcm_ref_bytes.len() % 2 != 0 {
        return Err(anyhow!("PCM file is not a whole number of i16 samples"));
    }
    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    let n_ref_samples = pcm_ref_bytes.len() / 2;

    println!("vector:        {name}{}", if rc { " (rc)" } else { "" });
    println!("frames:        {n_frames}");
    println!("ref samples:   {n_ref_samples} (= {} frames)", n_ref_samples / FRAME_SAMPLES);

    // Decode the reference PCM as i16 little-endian.
    let pcm_ref: Vec<i16> = pcm_ref_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Run the full pipeline.
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    let mut pcm_out: Vec<i16> = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    let mut bad_pitch_frames = 0usize;

    for f in 0..n_frames {
        let fec = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(fec);
        let imbe = decode_full_frame(&dibits);
        let params = match dequantize_full(&imbe.info, &mut decoder_state) {
            Ok(p) => p,
            Err(_) => {
                bad_pitch_frames += 1;
                pcm_out.extend(std::iter::repeat(0i16).take(FRAME_SAMPLES));
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let params_scaled = if (m_scale - 1.0).abs() > 1e-12 {
            scale_params(&params, m_scale)
        } else {
            params
        };
        let pcm = synthesize_frame(&params_scaled, &err, gamma_w, &mut synth_state);
        pcm_out.extend_from_slice(&pcm);
    }

    println!("decoded:       {} samples", pcm_out.len());
    if bad_pitch_frames > 0 {
        println!("bad-pitch frames (silenced): {bad_pitch_frames}");
    }
    println!();

    // Compare. Use the shorter of the two so we don't index OOB.
    let n = pcm_out.len().min(pcm_ref.len());
    if n == 0 {
        return Err(anyhow!("no samples to compare"));
    }
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    let mut peak_err = 0f64;
    let mut bit_exact = 0usize;
    for i in 0..n {
        let a = f64::from(pcm_out[i]);
        let b = f64::from(pcm_ref[i]);
        let e = a - b;
        sum_sq_err += e * e;
        sum_sq_ref += b * b;
        peak_err = peak_err.max(e.abs());
        if pcm_out[i] == pcm_ref[i] {
            bit_exact += 1;
        }
    }
    let rms_err = (sum_sq_err / n as f64).sqrt();
    let snr_db = if sum_sq_err > 0.0 {
        10.0 * (sum_sq_ref / sum_sq_err).log10()
    } else {
        f64::INFINITY
    };
    let pct_exact = 100.0 * bit_exact as f64 / n as f64;

    println!("Comparison vs DVSI reference PCM ({n} samples):");
    println!("  bit-exact samples:  {bit_exact} / {n}   ({pct_exact:.2}%)");
    println!("  RMS error:          {rms_err:.2}");
    println!("  peak error:         {peak_err:.0}");
    println!("  SNR:                {snr_db:.2} dB");

    if let Some(out_path) = write_pcm {
        let mut bytes = Vec::with_capacity(pcm_out.len() * 2);
        for s in &pcm_out {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(out_path, &bytes)
            .with_context(|| format!("write synthesized PCM to {}", out_path.display()))?;
        println!("wrote {} samples ({} bytes) to {}", pcm_out.len(), bytes.len(), out_path.display());
    }
    Ok(())
}

fn cmd_roundtrip(root: &Path, name: &str, rc: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let nofec_path = dir.join("p25_nofec").join(format!("{name}.bit"));
    let nofec_bytes = fs::read(&nofec_path)
        .with_context(|| format!("read no-FEC vector {}", nofec_path.display()))?;
    if nofec_bytes.len() % BYTES_PER_NOFEC_FRAME != 0 {
        return Err(anyhow!(
            "no-FEC file {} is {} bytes — not a multiple of {}",
            nofec_path.display(),
            nofec_bytes.len(),
            BYTES_PER_NOFEC_FRAME
        ));
    }
    let n_frames = nofec_bytes.len() / BYTES_PER_NOFEC_FRAME;

    println!("vector:        {name}{}", if rc { " (rc)" } else { "" });
    println!("frames:        {n_frames}");
    println!();

    let mut state_dec = DecoderState::new();
    let mut state_enc = DecoderState::new();
    let mut bad_pitch = 0usize;
    let mut pitch_match = 0usize;
    let mut vuv_match = 0usize;
    let mut vec_matches = [0usize; 8];

    for f in 0..n_frames {
        let frame_bytes =
            &nofec_bytes[f * BYTES_PER_NOFEC_FRAME..(f + 1) * BYTES_PER_NOFEC_FRAME];
        let u_orig = unpack_nofec_info(frame_bytes);

        // Decode → params.
        let params = match dequantize_full(&u_orig, &mut state_dec) {
            Ok(p) => p,
            Err(_) => {
                bad_pitch += 1;
                continue;
            }
        };

        // Re-encode using the encoder state (which evolves in lockstep).
        let b_back = match quantize_full(&params, &mut state_enc) {
            Ok(b) => b,
            Err(_) => {
                bad_pitch += 1;
                continue;
            }
        };
        let u_back = prioritize_full(&b_back, params.harmonic_count());

        // Per-vector match accounting.
        for i in 0..8 {
            if u_orig[i] == u_back[i] {
                vec_matches[i] += 1;
            }
        }

        // Pitch (b̂₀) and V/UV (b̂₁): pull from the original u via
        // deprioritize so we can compare at the parameter level too.
        let b_orig = deprioritize_full(&u_orig, params.harmonic_count());
        if b_orig[0] == b_back[0] {
            pitch_match += 1;
        }
        if b_orig[1] == b_back[1] {
            vuv_match += 1;
        }
    }

    let usable = n_frames - bad_pitch;
    println!("bad-pitch frames: {bad_pitch} (skipped)");
    println!("usable frames:    {usable}");
    println!();
    println!("Per-vector û match counts (out of {usable}):");
    for i in 0..8 {
        let pct = 100.0 * vec_matches[i] as f64 / usable as f64;
        println!(
            "  û_{i}  width {:>3}   {:>5} / {usable}   {:5.1}%",
            INFO_WIDTHS[i], vec_matches[i], pct
        );
    }
    println!();
    println!(
        "Parameter-level matches: pitch (b̂₀) {pitch_match}/{usable}, V/UV (b̂₁) {vuv_match}/{usable}"
    );

    if pitch_match == usable && vuv_match == usable {
        println!();
        println!("PASS — pitch and V/UV bit-exact across all usable frames.");
        Ok(())
    } else {
        Err(anyhow!(
            "expected bit-exact pitch + V/UV roundtrip; pitch={pitch_match}/{usable}, V/UV={vuv_match}/{usable}"
        ))
    }
}

fn cmd_pn_diag(root: &Path, name: &str, rc: bool, frame: usize) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_bytes = fs::read(dir.join("p25").join(format!("{name}.bit")))?;
    let nofec_bytes = fs::read(dir.join("p25_nofec").join(format!("{name}.bit")))?;

    let fec_frame = &fec_bytes[frame * BYTES_PER_FEC_FRAME..(frame + 1) * BYTES_PER_FEC_FRAME];
    let nofec_frame =
        &nofec_bytes[frame * BYTES_PER_NOFEC_FRAME..(frame + 1) * BYTES_PER_NOFEC_FRAME];

    let dibits = unpack_dibits(fec_frame);
    let c_tilde = deinterleave_full(&dibits);
    let u_expected = unpack_nofec_info(nofec_frame);

    // Re-encode the no-FEC û_i to get the v̂_i the encoder produced
    // before PN modulation (only for FEC-protected vectors).
    let v_expected: [u32; 8] = [
        golay_23_12_encode(u_expected[0]),
        golay_23_12_encode(u_expected[1]),
        golay_23_12_encode(u_expected[2]),
        golay_23_12_encode(u_expected[3]),
        u32::from(hamming_15_11_encode(u_expected[4])),
        u32::from(hamming_15_11_encode(u_expected[5])),
        u32::from(hamming_15_11_encode(u_expected[6])),
        u32::from(u_expected[7]),
    ];

    // Recovered DVSI mask = c̃ ⊕ v̂. Our computed mask uses û₀ as seed.
    let dvsi_mask: [u32; 8] = std::array::from_fn(|i| c_tilde[i] ^ v_expected[i]);
    let our_mask = modulation_masks(u_expected[0]);

    let widths = [23u8, 23, 23, 23, 15, 15, 15, 7];
    println!("frame {frame}, û₀ = 0x{:03x}", u_expected[0]);
    println!();
    println!("vec  width  c̃ (deinterleaved)  v̂ (re-encoded)   DVSI mask    our mask");
    for i in 0..8 {
        println!(
            "  {i}    {:>3}    0x{:06x}            0x{:06x}         0x{:06x}    0x{:06x}",
            widths[i], c_tilde[i], v_expected[i], dvsi_mask[i], our_mask[i]
        );
    }

    // Show the bit-reversed comparison for the PN-modulated vectors.
    println!();
    println!("PN-modulated vectors only (1..=6), bit-reversal check:");
    println!("vec  DVSI mask              our mask              our mask reversed");
    for i in 1..=6 {
        let n = widths[i] as u32;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let reversed = reverse_bits_within(our_mask[i], widths[i]);
        let mark_norm = if dvsi_mask[i] & mask == our_mask[i] & mask { '✓' } else { ' ' };
        let mark_rev = if dvsi_mask[i] & mask == reversed & mask { '✓' } else { ' ' };
        println!(
            "  {i}   0x{:06x}              0x{:06x} {}            0x{:06x} {}",
            dvsi_mask[i] & mask,
            our_mask[i] & mask,
            mark_norm,
            reversed & mask,
            mark_rev,
        );
    }
    Ok(())
}

const BYTES_PER_HALFRATE_FRAME: usize = 9;
const DIBITS_PER_HALFRATE_FRAME: usize = 36;

fn unpack_dibits_halfrate(bytes: &[u8]) -> [u8; DIBITS_PER_HALFRATE_FRAME] {
    debug_assert_eq!(bytes.len(), BYTES_PER_HALFRATE_FRAME);
    let mut r = BitReader::new(bytes);
    let mut out = [0u8; DIBITS_PER_HALFRATE_FRAME];
    for i in 0..DIBITS_PER_HALFRATE_FRAME {
        let hi = r.read(1) as u8;
        let lo = r.read(1) as u8;
        out[i] = (hi << 1) | lo;
    }
    out
}

fn cmd_decode_pcm_halfrate(
    root: &Path,
    name: &str,
    gamma_w: f64,
    write_pcm: Option<&Path>,
    phase_mode: HalfratePhaseMode,
) -> Result<()> {
    // Half-rate DVSI vectors live under tv-std/tv/r33/ (rate index 33 — P25 half-rate with FEC
    // per USB-3000 manual §3.3.4 = 3600 bps AMBE+2).
    let dir = root.join("tv-std").join("tv").join("r33");
    let bit_path = dir.join(format!("{name}.bit"));
    let pcm_path = dir.join(format!("{name}.pcm"));

    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    let pcm_ref_bytes = fs::read(&pcm_path)
        .with_context(|| format!("read reference PCM {}", pcm_path.display()))?;

    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!(
            "half-rate .bit file {} is {} bytes — not a multiple of {}",
            bit_path.display(),
            bit_bytes.len(),
            BYTES_PER_HALFRATE_FRAME
        ));
    }
    if pcm_ref_bytes.len() % 2 != 0 {
        return Err(anyhow!("PCM file is not a whole number of i16 samples"));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;
    let n_ref_samples = pcm_ref_bytes.len() / 2;

    println!("vector:        {name} (half-rate, r33 — P25 with FEC)");
    println!("frames:        {n_frames}");
    println!(
        "ref samples:   {n_ref_samples} (= {} frames)",
        n_ref_samples / FRAME_SAMPLES
    );
    println!("γ_w:           {gamma_w}");
    println!();

    let pcm_ref: Vec<i16> = pcm_ref_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut dec_state = HalfDecoderState::new();
    let mut synth_state = SynthState::new();
    let mut pcm_out: Vec<i16> = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    let mut voice_frames = 0usize;
    let mut tone_frames = 0usize;
    let mut erasure_frames = 0usize;

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_half_frame(&dibits);
        let err = FrameErrorContext {
            epsilon_0: ambe.errors[0],
            epsilon_4: 0, // half-rate doesn't have a c̃₄ vector
            epsilon_t: ambe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        // `phase_mode` dispatches between Baseline (BABA-A Eq. 141
        // noise-perturbed phase) and AmbePlus (US5701390 §5 Hilbert
        // phase regeneration). Default is AmbePlus per
        // AMBE-3000_Decoder_Implementation_Spec §5; Baseline is
        // available for A/B listening tests of the phase-regen
        // perceptual claim.
        let synth: fn(&MbeParams, &FrameErrorContext, f64, &mut SynthState) -> [i16; FRAME_SAMPLES] =
            match phase_mode {
                HalfratePhaseMode::Baseline => synthesize_frame,
                HalfratePhaseMode::AmbePlus => synthesize_frame_ambe_plus,
            };
        match decode_to_params(&ambe.info, &mut dec_state) {
            Ok(Decoded::Voice(params)) => {
                voice_frames += 1;
                let pcm = synth(&params, &err, gamma_w, &mut synth_state);
                pcm_out.extend_from_slice(&pcm);
            }
            Ok(Decoded::Tone { fields: _, params }) => {
                tone_frames += 1;
                // Tone frames feed the MBE synthesizer per §2.10.3.
                let pcm = synth(&params, &err, gamma_w, &mut synth_state);
                pcm_out.extend_from_slice(&pcm);
            }
            Ok(Decoded::Erasure) | Err(_) => {
                erasure_frames += 1;
                pcm_out.extend(std::iter::repeat(0i16).take(FRAME_SAMPLES));
            }
        }
    }

    println!("decoded:       {} samples", pcm_out.len());
    println!(
        "frame mix:     voice={voice_frames}, tone={tone_frames}, erasure={erasure_frames}"
    );
    println!();

    let n = pcm_out.len().min(pcm_ref.len());
    if n == 0 {
        return Err(anyhow!("no samples to compare"));
    }
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    let mut peak_err = 0f64;
    let mut bit_exact = 0usize;
    for i in 0..n {
        let a = f64::from(pcm_out[i]);
        let b = f64::from(pcm_ref[i]);
        let e = a - b;
        sum_sq_err += e * e;
        sum_sq_ref += b * b;
        peak_err = peak_err.max(e.abs());
        if pcm_out[i] == pcm_ref[i] {
            bit_exact += 1;
        }
    }
    let rms_err = (sum_sq_err / n as f64).sqrt();
    let snr_db = if sum_sq_err > 0.0 {
        10.0 * (sum_sq_ref / sum_sq_err).log10()
    } else {
        f64::INFINITY
    };
    let pct_exact = 100.0 * bit_exact as f64 / n as f64;

    println!("Comparison vs DVSI r33 reference PCM ({n} samples):");
    println!("  bit-exact samples:  {bit_exact} / {n}   ({pct_exact:.2}%)");
    println!("  RMS error:          {rms_err:.2}");
    println!("  peak error:         {peak_err:.0}");
    println!("  SNR:                {snr_db:.2} dB");

    if let Some(out_path) = write_pcm {
        let mut bytes = Vec::with_capacity(pcm_out.len() * 2);
        for s in &pcm_out {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(out_path, &bytes)
            .with_context(|| format!("write PCM to {}", out_path.display()))?;
        println!("\nwrote synthesized PCM: {} ({} samples)", out_path.display(), pcm_out.len());
    }

    Ok(())
}

fn cmd_halfrate_seam_dump(root: &Path, name: &str, frame_idx: usize) -> Result<()> {
    use blip25_mbe::fec::golay_24_12_decode;

    let dir = root.join("tv-std").join("tv").join("r33");
    let bit_bytes = fs::read(dir.join(format!("{name}.bit")))?;
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;
    if frame_idx >= n_frames {
        return Err(anyhow!("frame {frame_idx} out of range (have {n_frames})"));
    }

    let frame = &bit_bytes[frame_idx * BYTES_PER_HALFRATE_FRAME..(frame_idx + 1) * BYTES_PER_HALFRATE_FRAME];
    let dibits = unpack_dibits_halfrate(frame);

    println!("=== Frame {frame_idx} of r33/{name}.bit ===\n");

    // Step 1: ĉ vectors after Annex S deinterleave.
    let c = deinterleave_half(&dibits);
    println!("After Annex S deinterleave:");
    println!("  ĉ₀ (24 bits): 0x{:06x} = {:024b}", c[0] & 0xFF_FFFF, c[0] & 0xFF_FFFF);
    println!("  ĉ₁ (23 bits): 0x{:06x}", c[1] & 0x7F_FFFF);
    println!("  ĉ₂ (11 bits): 0x{:03x}", c[2] & 0x7FF);
    println!("  ĉ₃ (14 bits): 0x{:04x}", c[3] & 0x3FFF);
    println!();

    // Step 2: Golay-decode ĉ₀ → û₀ (both straight and with 24-bit
    // reversal, to test the [I|P] vs [P|I] hypothesis).
    let d0 = golay_24_12_decode(c[0]);
    let u0 = d0.info;
    println!("After Golay-24-12 decode of ĉ₀ (current convention):");
    println!("  û₀ (12 bits): 0x{:03x} = {:012b}  (errors={})", u0, u0, d0.errors);
    let c0_rev = {
        let mut r = 0u32;
        for i in 0..24 {
            if (c[0] >> i) & 1 == 1 { r |= 1 << (23 - i); }
        }
        r
    };
    let d0r = golay_24_12_decode(c0_rev);
    let u0r = d0r.info;
    println!("After Golay-24-12 decode of bit-reversed ĉ₀:");
    println!("  ĉ₀_rev     : 0x{c0_rev:06x}");
    println!("  û₀ (12 bits): 0x{:03x} = {:012b}  (errors={})", u0r, u0r, d0r.errors);
    println!();

    // Step 3: bit-identity check ĉ₀[0..11] vs û₀[0..11]. In a [I|P]
    // systematic code, these should be bit-identical.
    println!("Systematic bit correspondence check (should be identical if [I|P]-form Golay):");
    println!("  idx  ĉ₀[k]  û₀[k]  match");
    let mut match_low = 0usize;
    let mut match_rev = 0usize;
    for k in 0..12 {
        let cb = ((c[0] >> k) & 1) as u8;
        let ub = ((u0 >> k) & 1) as u8;
        let urev = ((u0 >> (11 - k)) & 1) as u8;
        let m_lo = if cb == ub { "✓" } else { " " };
        if cb == ub { match_low += 1; }
        if cb == urev { match_rev += 1; }
        println!("  {k:>3}    {cb}      {ub}      {m_lo}");
    }
    println!();
    println!("  ĉ₀[0..11] matches û₀[0..11]: {match_low}/12");
    println!("  ĉ₀[0..11] matches û₀[11..0] reversed: {match_rev}/12");
    if match_rev == 12 && match_low < 12 {
        println!("\n  *** û₀ is bit-reversed vs the systematic positions of ĉ₀. ***");
    }
    println!();

    // Step 4: Also check the high half of ĉ₀ (positions 12..23) — in
    // an [I|P] code these are parity bits, so they shouldn't be
    // identical to anything in û₀. Just report their pattern.
    print!("  ĉ₀[12..23] (parity in [I|P] form): ");
    for k in 12..24 {
        print!("{}", (c[0] >> k) & 1);
    }
    println!();

    // Step 5: Annex S bit-lane check — symbols 0 and 2 put ĉ₀[23]
    // and ĉ₀[22] in the bit1 (MSB-of-dibit) lane. Verify our lane
    // assignment matches.
    println!();
    println!("Annex S bit-lane sanity (per user-provided PDF description):");
    println!("  symbol 0 raw dibit: {}  (bit1={}, bit0={})",
        dibits[0], (dibits[0] >> 1) & 1, dibits[0] & 1);
    println!("  symbol 2 raw dibit: {}  (bit1={}, bit0={})",
        dibits[2], (dibits[2] >> 1) & 1, dibits[2] & 1);
    println!("  Per spec: bit1 of symbol 0 → ĉ₀[23], bit1 of symbol 2 → ĉ₀[22].");
    println!("  Our deinterleaver result: ĉ₀[23] = {}, ĉ₀[22] = {}",
        (c[0] >> 23) & 1, (c[0] >> 22) & 1);

    Ok(())
}

fn cmd_halfrate_encoder_sanity() -> Result<()> {
    // Speech-scale voiced frame: pitch ≈ 150 Hz, L = 27, all voiced,
    // M_l following a -6 dB/octave-ish roll-off typical for voiced
    // speech. Peak M_l = 2000 (ln₂ ≈ 11), mean ≈ 500 (ln₂ ≈ 9).
    use std::f32::consts::PI;
    let omega_0 = 2.0 * PI * 150.0 / 8000.0;
    let l = MbeParams::harmonic_count_for(omega_0);
    let voiced: Vec<bool> = (1..=l).map(|_| true).collect();
    let amps: Vec<f32> = (1..=l)
        .map(|h| 2000.0 / (h as f32).powf(0.6))
        .collect();
    let params = MbeParams::new(omega_0, l, &voiced, &amps).unwrap();

    println!("Synthetic speech-scale MbeParams:");
    println!("  ω₀ = {omega_0:.4} rad/s ({:.1} Hz)", params.fundamental_hz());
    println!("  L  = {l}");
    println!("  M_l range = {:.1} .. {:.1}", amps.iter().cloned().fold(f32::INFINITY, f32::min),
        amps.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("  log₂(M_l) range ≈ {:.2} .. {:.2}",
        amps.iter().cloned().fold(f32::INFINITY, f32::min).log2(),
        amps.iter().cloned().fold(f32::NEG_INFINITY, f32::max).log2());
    println!();

    let mut state = HalfDecoderState::new();

    // Run the encoder a few times so cross-frame state settles.
    for iter in 0..5 {
        let u = quantize_half(&params, &mut state).unwrap();
        let b = deprioritize_half(&u);
        println!("iter {iter}: b̂ = [b0={}, b1={}, b2={}, b3={}, b4={}, b5={}, b6={}, b7={}, b8={}]",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]);
        println!("         u = [0x{:03x}, 0x{:03x}, 0x{:03x}, 0x{:04x}]",
            u[0], u[1], u[2], u[3]);
        println!("         prev_γ = {:.4}", state.previous_gamma());
    }

    println!();
    println!("Spec prediction (§13.2 + Eq. 151): b̂₂ should settle in high 20s to 31.");
    Ok(())
}

fn cmd_halfrate_inspect(root: &Path, name: &str, frames_to_dump: usize) -> Result<()> {
    let dir = root.join("tv-std").join("tv").join("r33");
    let bit_path = dir.join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read {}", bit_path.display()))?;
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    let mut dec_state = HalfDecoderState::new();
    let mut synth_state = SynthState::new();
    let mut dumped = 0usize;

    println!("vector: {name} — {n_frames} frames total\n");

    for f in 0..n_frames {
        if dumped >= frames_to_dump {
            break;
        }
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_half_frame(&dibits);
        let err = FrameErrorContext {
            epsilon_0: ambe.errors[0],
            epsilon_4: 0,
            epsilon_t: ambe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let decoded = decode_to_params(&ambe.info, &mut dec_state);

        match decoded {
            Ok(Decoded::Voice(params)) => {
                dumped += 1;
                let l = params.harmonic_count();
                let amps = params.amplitudes_slice();
                let voiced = params.voiced_slice();
                let amp_min = amps.iter().cloned().fold(f32::INFINITY, f32::min);
                let amp_max = amps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let amp_mean = amps.iter().sum::<f32>() / amps.len() as f32;
                let vcount = voiced.iter().filter(|&&v| v).count();
                let nonfin = amps.iter().filter(|a| !a.is_finite()).count();

                // Raw bit layout: show u[] and the deprioritized b[].
                let b = deprioritize_half(&ambe.info);
                println!("  raw u[]  = [0x{:03x}, 0x{:03x}, 0x{:03x}, 0x{:04x}]",
                    ambe.info[0], ambe.info[1], ambe.info[2], ambe.info[3]);
                println!("  b̂ vals   = [b0={}, b1={}, b2={}, b3={}, b4={}, b5={}, b6={}, b7={}, b8={}]",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]);

                let pcm = synthesize_frame(&params, &err, GAMMA_W, &mut synth_state);
                let pcm_peak = pcm.iter().map(|s| s.abs()).max().unwrap_or(0);
                let pcm_rms = (pcm.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
                    / pcm.len() as f64).sqrt();

                println!("frame {f} (voice):");
                println!("  ω₀            = {:.6} rad/s  ({:.1} Hz)",
                    params.omega_0(), params.fundamental_hz());
                println!("  L             = {l}");
                println!("  voiced count  = {vcount}/{l}");
                println!("  amp min/mean/max = {amp_min:.4} / {amp_mean:.4} / {amp_max:.4}");
                println!("  amp non-finite  = {nonfin}");
                println!("  first 8 amps  = {:?}",
                    amps.iter().take(8).map(|a| format!("{a:.3}")).collect::<Vec<_>>());
                println!("  first 8 vuv   = {:?}", &voiced[..voiced.len().min(8)]);
                println!("  prev_gamma    = {:.6}", dec_state.previous_gamma());
                println!("  prev_L        = {}", dec_state.previous_l());
                println!("  PCM peak/rms  = {pcm_peak} / {pcm_rms:.2}");
                println!();
            }
            Ok(Decoded::Tone { .. }) => {
                let _ = synthesize_frame(&MbeParams::silence(), &err, GAMMA_W, &mut synth_state);
                println!("frame {f} (tone) — skipped");
            }
            Ok(Decoded::Erasure) | Err(_) => {
                let _ = synthesize_frame(&MbeParams::silence(), &err, GAMMA_W, &mut synth_state);
                println!("frame {f} (erasure) — skipped");
            }
        }
    }
    Ok(())
}

fn cmd_scan_transitions(
    root: &Path,
    name: &str,
    min_ratio: f32,
    limit: usize,
) -> Result<()> {
    let dir = root.join("tv-std").join("tv").join("p25");
    let fec_path = dir.join(format!("{name}.bit"));
    let pcm_path = dir.join(format!("{name}.pcm"));
    let fec_bytes = fs::read(&fec_path)?;
    let pcm_ref_bytes = fs::read(&pcm_path)?;
    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    let pcm_ref: Vec<i16> = pcm_ref_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Walk the full vector. For each frame, record (prev voiced count,
    // curr voiced count, L). Classify transitions when the curr frame
    // has ≥ min_ratio · L voiced harmonics and prev had ≤ (1 −
    // min_ratio) · prev_L, or vice versa.
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();

    #[derive(Clone)]
    struct FrameSnapshot {
        idx: usize,
        l: u8,
        voiced_count: u8,
        our_pcm: [i16; FRAME_SAMPLES],
        omega_0: f32,
    }
    let mut prev: Option<FrameSnapshot> = None;
    let mut uv_to_v: Vec<(FrameSnapshot, FrameSnapshot)> = Vec::new();
    let mut v_to_uv: Vec<(FrameSnapshot, FrameSnapshot)> = Vec::new();

    for f in 0..n_frames {
        let fec = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(fec);
        let imbe = decode_full_frame(&dibits);
        let params = match dequantize_full(&imbe.info, &mut decoder_state) {
            Ok(p) => p,
            Err(_) => {
                prev = None;
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        // Voiced-only synthesis for the xcorr — matches what
        // cmd_xcorr_frame does.
        let pcm = synthesize_frame(&params, &err, 0.0, &mut synth_state);
        let voiced_count = params
            .voiced_slice()
            .iter()
            .filter(|&&v| v)
            .count() as u8;
        let curr = FrameSnapshot {
            idx: f,
            l: params.harmonic_count(),
            voiced_count,
            our_pcm: pcm,
            omega_0: params.omega_0(),
        };

        if let Some(p) = prev.as_ref() {
            let prev_ratio = f32::from(p.voiced_count) / f32::from(p.l);
            let curr_ratio = f32::from(curr.voiced_count) / f32::from(curr.l);
            // UV → V: prev mostly unvoiced, curr mostly voiced. Isolates
            // Eq. 132 (candidate #2).
            if prev_ratio <= (1.0 - min_ratio) && curr_ratio >= min_ratio {
                uv_to_v.push((p.clone(), curr.clone()));
            }
            // V → UV: prev mostly voiced, curr mostly unvoiced. Isolates
            // Eq. 131 (candidate #1).
            else if prev_ratio >= min_ratio && curr_ratio <= (1.0 - min_ratio) {
                v_to_uv.push((p.clone(), curr.clone()));
            }
        }
        prev = Some(curr);
    }

    println!("vector:   {name}");
    println!("frames:   {n_frames}");
    println!(
        "criteria: prev/curr voiced ratio ≥ {:.0}% or ≤ {:.0}%",
        min_ratio * 100.0,
        (1.0 - min_ratio) * 100.0
    );
    println!();

    fn xcorr_and_print(
        label: &str,
        candidate: &str,
        fires_on: &str,
        frames: &[(FrameSnapshot, FrameSnapshot)],
        pcm_ref: &[i16],
        limit: usize,
    ) {
        println!("=== {label} — isolates {candidate} ({fires_on}) ===");
        if frames.is_empty() {
            println!("  (no matching frames found — try lowering --min-transition-ratio)");
            println!();
            return;
        }
        println!("  frames found: {}", frames.len());
        println!(
            "  {:>5}  {:>4}  {:>4}/{:>2}  {:>4}/{:>2}  {:>7}  {:>6}  {:>4}",
            "frame", "ω̃₀", "prev", "L₋", "curr", "L₀", "peak", "zero", "lag"
        );
        for (p, c) in frames.iter().take(limit) {
            let dvsi_start = c.idx * FRAME_SAMPLES;
            let dvsi_end = dvsi_start + FRAME_SAMPLES;
            if dvsi_end > pcm_ref.len() {
                continue;
            }
            let dvsi = &pcm_ref[dvsi_start..dvsi_end];
            // Find peak lag in ±20.
            let mut peak_corr = 0f64;
            let mut peak_lag = 0i32;
            for lag in -20..=20i32 {
                let corr = xcorr_normalized(&c.our_pcm, dvsi, lag);
                if corr.abs() > peak_corr.abs() {
                    peak_corr = corr;
                    peak_lag = lag;
                }
            }
            let zero_lag = xcorr_normalized(&c.our_pcm, dvsi, 0);
            println!(
                "  {:>5}  {:.3}  {:>4}/{:>2}  {:>4}/{:>2}  {:>+7.3}  {:>+6.3}  {:>+4}",
                c.idx,
                c.omega_0,
                p.voiced_count,
                p.l,
                c.voiced_count,
                c.l,
                peak_corr,
                zero_lag,
                peak_lag
            );
        }
        // Aggregate signature.
        let mut sum_peak = 0f64;
        let mut sum_peak_abs = 0f64;
        let mut lag_hist = [0usize; 41]; // lags −20..+20
        let mut neg_peak_count = 0usize;
        let mut pos_peak_count = 0usize;
        let mut scored = 0usize;
        for (_, c) in frames {
            let dvsi_start = c.idx * FRAME_SAMPLES;
            let dvsi_end = dvsi_start + FRAME_SAMPLES;
            if dvsi_end > pcm_ref.len() {
                continue;
            }
            let dvsi = &pcm_ref[dvsi_start..dvsi_end];
            let mut peak_corr = 0f64;
            let mut peak_lag = 0i32;
            for lag in -20..=20i32 {
                let corr = xcorr_normalized(&c.our_pcm, dvsi, lag);
                if corr.abs() > peak_corr.abs() {
                    peak_corr = corr;
                    peak_lag = lag;
                }
            }
            sum_peak += peak_corr;
            sum_peak_abs += peak_corr.abs();
            lag_hist[(peak_lag + 20) as usize] += 1;
            if peak_corr < 0.0 { neg_peak_count += 1; }
            if peak_corr > 0.0 { pos_peak_count += 1; }
            scored += 1;
        }
        if scored > 0 {
            let mean_peak = sum_peak / scored as f64;
            let mean_peak_abs = sum_peak_abs / scored as f64;
            println!();
            println!(
                "  aggregate over {scored}: mean peak corr = {mean_peak:+.3}, mean |peak| = {mean_peak_abs:.3}"
            );
            println!(
                "  sign split: {neg_peak_count} negative, {pos_peak_count} positive"
            );
            let mut lag_buckets: Vec<(i32, usize)> = (0..41usize)
                .filter(|&i| lag_hist[i] > 0)
                .map(|i| (i as i32 - 20, lag_hist[i]))
                .collect();
            lag_buckets.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = lag_buckets
                .iter()
                .take(6)
                .map(|(l, n)| format!("{l:+}:{n}"))
                .collect();
            println!("  lag top-6 (peak:count): {}", top.join("  "));
            let neg_pct = 100.0 * neg_peak_count as f64 / scored as f64;
            if neg_pct > 70.0 && mean_peak_abs > 0.3 {
                println!(
                    "  → SIGNATURE MATCH: {neg_pct:.0}% negative peaks w/ |peak| > 0.3 → 180° sign-flip on {candidate}"
                );
            } else if mean_peak_abs < 0.2 {
                println!(
                    "  → No clear structural match — error may be elsewhere (window, state init, or {candidate} is innocent)"
                );
            } else {
                println!(
                    "  → Mixed signal — {candidate} is partially implicated but not the sole cause"
                );
            }
        }
        println!();
    }

    // Also classify V → V frames (both fully voiced) to isolate
    // Eq. 133/134 (the V→V branches), which are different from #1/#2.
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    let mut prev: Option<FrameSnapshot> = None;
    let mut v_to_v: Vec<(FrameSnapshot, FrameSnapshot)> = Vec::new();
    for f in 0..n_frames {
        let fec = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(fec);
        let imbe = decode_full_frame(&dibits);
        let params = match dequantize_full(&imbe.info, &mut decoder_state) {
            Ok(p) => p,
            Err(_) => {
                prev = None;
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let pcm = synthesize_frame(&params, &err, 0.0, &mut synth_state);
        let voiced_count = params.voiced_slice().iter().filter(|&&v| v).count() as u8;
        let curr = FrameSnapshot {
            idx: f,
            l: params.harmonic_count(),
            voiced_count,
            our_pcm: pcm,
            omega_0: params.omega_0(),
        };
        if let Some(p) = prev.as_ref() {
            let prev_ratio = f32::from(p.voiced_count) / f32::from(p.l);
            let curr_ratio = f32::from(curr.voiced_count) / f32::from(curr.l);
            if prev_ratio >= min_ratio && curr_ratio >= min_ratio {
                v_to_v.push((p.clone(), curr.clone()));
            }
        }
        prev = Some(curr);
    }

    xcorr_and_print(
        "V → UV transitions",
        "Eq. 131 (candidate #1)",
        "cos(ω̃₀(−1)·n·l + φ_l(−1)) on prev-voiced harmonics",
        &v_to_uv,
        &pcm_ref,
        limit,
    );
    xcorr_and_print(
        "UV → V transitions",
        "Eq. 132 (candidate #2)",
        "cos(ω̃₀(0)·(n−N)·l + φ_l(0)) on curr-voiced harmonics",
        &uv_to_v,
        &pcm_ref,
        limit,
    );
    xcorr_and_print(
        "V → V stable voicing",
        "Eq. 133/134 (candidate #3/#4)",
        "sum-of-both (l≥8 or |Δω·l/ω|≥0.1) OR amplitude-ramp (l<8 and small Δω)",
        &v_to_v,
        &pcm_ref,
        limit,
    );
    Ok(())
}

fn cmd_xcorr_frame(
    root: &Path,
    name: &str,
    frame_idx: usize,
    both: bool,
    gamma_w: f64,
    m_scale: f64,
) -> Result<()> {
    let dir = root.join("tv-std").join("tv").join("p25");
    let fec_path = dir.join(format!("{name}.bit"));
    let pcm_path = dir.join(format!("{name}.pcm"));

    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    let pcm_ref_bytes = fs::read(&pcm_path)
        .with_context(|| format!("read reference PCM {}", pcm_path.display()))?;

    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    if frame_idx >= n_frames {
        return Err(anyhow!("frame {frame_idx} out of range (0..{n_frames})"));
    }
    let pcm_ref: Vec<i16> = pcm_ref_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Run the decoder up to and including `frame_idx`, accumulating
    // state but only emitting the final frame's PCM.
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    let mut our_frame = [0i16; FRAME_SAMPLES];
    let mut frame_info: Option<(f32, u8, u8, usize, Vec<bool>, Vec<f32>)> = None;

    for f in 0..=frame_idx {
        let fec = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(fec);
        let imbe = decode_full_frame(&dibits);
        let params = dequantize_full(&imbe.info, &mut decoder_state)
            .map_err(|e| anyhow!("dequant failed at frame {f}: {e:?}"))?;
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let effective_gamma_w = if both { gamma_w } else { 0.0 };
        let params_scaled = if (m_scale - 1.0).abs() > 1e-12 {
            scale_params(&params, m_scale)
        } else {
            params.clone()
        };
        let pcm = synthesize_frame(&params_scaled, &err, effective_gamma_w, &mut synth_state);
        if f == frame_idx {
            our_frame = pcm;
            let voiced_count = params.voiced_slice().iter().filter(|&&v| v).count();
            let peak_m = params
                .amplitudes_slice()
                .iter()
                .map(|m| m.abs())
                .fold(0f32, f32::max);
            frame_info = Some((
                params.omega_0(),
                params.harmonic_count(),
                voiced_count as u8,
                peak_m as usize,
                params.voiced_slice().to_vec(),
                params.amplitudes_slice().to_vec(),
            ));
        }
    }

    // DVSI reference window for this frame.
    let start = frame_idx * FRAME_SAMPLES;
    let end = start + FRAME_SAMPLES;
    if end > pcm_ref.len() {
        return Err(anyhow!(
            "DVSI PCM too short for frame {frame_idx}: need {end}, have {}",
            pcm_ref.len()
        ));
    }
    let dvsi = &pcm_ref[start..end];

    // Header
    let (omega_0, l, voiced_count, peak_m_int, voiced, amps) = frame_info.unwrap();
    let peak_m = f32::max(peak_m_int as f32, amps.iter().fold(0f32, |a, &b| a.max(b)));
    let voiced_idx: Vec<usize> = voiced
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i + 1) } else { None })
        .collect();
    println!("vector:         {name}");
    println!("frame:          {frame_idx} of {n_frames}");
    println!("decode path:    {}", if both { "voiced + unvoiced (γ_w applied)" } else { "voiced only (γ_w = 0)" });
    println!("m_scale:        {m_scale}");
    println!(
        "ω̃₀ = {:.4} (= {:.1} Hz),  L = {l},  voiced = {voiced_count} / {l}",
        omega_0,
        omega_0 * 8000.0 / (2.0 * core::f32::consts::PI)
    );
    println!(
        "voiced harmonics: {}",
        voiced_idx.iter().take(12).map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    );
    println!("peak M̃_l:       {peak_m:.0}");
    println!();

    // Stats per side.
    let our_rms = rms(&our_frame);
    let dvsi_rms = rms(dvsi);
    println!("our  RMS:        {our_rms:.0}");
    println!("DVSI RMS:        {dvsi_rms:.0}");
    println!();

    // Cross-correlation over a ±20 sample lag window. Normalize by
    // the product of RMS values × N so values live in [-1, +1].
    const LAG_HALF: i32 = 20;
    let mut peak_corr = 0f64;
    let mut peak_lag = 0i32;
    println!("Normalized cross-correlation (lag: our[n] · dvsi[n + lag]):");
    for lag in (-LAG_HALF..=LAG_HALF).step_by(1) {
        let c = xcorr_normalized(&our_frame, dvsi, lag);
        if c.abs() > peak_corr.abs() {
            peak_corr = c;
            peak_lag = lag;
        }
        if lag.rem_euclid(5) == 0 || lag == peak_lag {
            let mark = if lag == peak_lag { " ← peak" } else { "" };
            println!("  lag = {lag:>3}:  {c:>+.4}{mark}");
        }
    }
    let zero_lag = xcorr_normalized(&our_frame, dvsi, 0);
    println!();
    println!("peak corr:  {peak_corr:+.4} at lag = {peak_lag}");
    println!("zero-lag:   {zero_lag:+.4}");

    // Interpretation heuristic.
    println!();
    let peak_abs = peak_corr.abs();
    let zero_abs = zero_lag.abs();
    println!("Diagnostic interpretation:");
    if peak_abs > 0.5 && peak_lag.abs() > 0 {
        let phase_rad = f64::from(omega_0) * f64::from(peak_lag);
        let phase_deg = phase_rad * 180.0 / core::f64::consts::PI;
        println!(
            "  ✓ HIGH peak ({peak_abs:.2}) at NON-ZERO lag ({peak_lag}) — likely PHASE-TRACKING issue."
        );
        println!(
            "    At ω̃₀ = {omega_0:.4}, lag {peak_lag} ≈ {phase_rad:.3} rad = {phase_deg:.1}° shift"
        );
        println!("    at the fundamental. Inspect §1.12.2 Eq. 134–141 (voiced phase evolution,");
        println!("    φ_l / ψ_l / Δω_l); also §1.13 Annex A φ_l(−1), ψ_l(−1) init values.");
    } else if peak_abs > 0.5 && peak_lag.abs() == 0 {
        println!("  ✓ HIGH zero-lag correlation ({peak_abs:.2}) — phase tracking is aligned.");
        println!("    Residual error is likely amplitude or harmonic structure, not phase.");
        println!("    Inspect §1.10 enhancement or §1.11 smoothing for amplitude shaping.");
    } else {
        println!("  ✗ LOW peak correlation ({peak_abs:.2}) — waveform structurally different.");
        println!("    Candidates (in priority order): OLA branch selection across the 5");
        println!("    V/UV transition cases in §1.12.2 (Eq. 130–134); per-frame φ/ψ state");
        println!("    init (§1.13 Annex A); window application (wS vs wS(n−N) alignment in");
        println!("    Eq. 131/132); or §1.12.2 Eq. 127 sum bounds / factor-of-2.");
    }
    println!();
    println!(
        "If you want to see the waveforms directly, first 16 samples side by side:"
    );
    println!("  n   our     DVSI    diff");
    for n in 0..16 {
        let o = our_frame[n];
        let d = dvsi[n];
        let diff = i32::from(o) - i32::from(d);
        println!("  {n:>2}  {o:>6}  {d:>6}  {diff:>+6}");
    }
    Ok(())
}

fn rms(samples: &[i16]) -> f64 {
    let n = samples.len() as f64;
    let ss: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (ss / n).sqrt()
}

/// Normalized cross-correlation at a given integer lag. Returns
/// Σ a[n]·b[n+lag] / (N · RMS(a) · RMS(b)) so magnitudes live in
/// [-1, +1]. Out-of-range indices contribute zero.
fn xcorr_normalized(a: &[i16], b: &[i16], lag: i32) -> f64 {
    let n = a.len() as f64;
    let rms_a = rms(a);
    let rms_b = rms(b);
    if rms_a < 1e-6 || rms_b < 1e-6 {
        return 0.0;
    }
    let mut acc = 0f64;
    for i in 0..a.len() {
        let j = i as i32 + lag;
        if j >= 0 && (j as usize) < b.len() {
            acc += f64::from(a[i]) * f64::from(b[j as usize]);
        }
    }
    acc / (n * rms_a * rms_b)
}

fn cmd_halfrate_roundtrip(root: &Path, name: &str) -> Result<()> {
    let dir = root.join("tv-std").join("tv").join("r33");
    let bit_path = dir.join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!("half-rate .bit file is not a whole number of frames"));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    println!("vector:        {name} (half-rate, r33 — P25 with FEC)");
    println!("frames:        {n_frames}");
    println!();

    let mut state_dec = HalfDecoderState::new();
    let mut state_enc = HalfDecoderState::new();
    let mut voice_frames = 0usize;
    let mut non_voice_frames = 0usize;
    let mut pitch_match = 0usize;
    let mut vuv_match = 0usize;
    let mut gain_match = 0usize;

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_half_frame(&dibits);
        match decode_to_params(&ambe.info, &mut state_dec) {
            Ok(Decoded::Voice(params)) => {
                voice_frames += 1;
                let u_back = match quantize_half(&params, &mut state_enc) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                let b_orig = deprioritize_half(&ambe.info);
                let b_back = deprioritize_half(&u_back);
                if b_orig[0] == b_back[0] {
                    pitch_match += 1;
                }
                if b_orig[1] == b_back[1] {
                    vuv_match += 1;
                }
                if b_orig[2] == b_back[2] {
                    gain_match += 1;
                }
            }
            _ => non_voice_frames += 1,
        }
    }

    println!("voice frames:     {voice_frames}");
    println!("non-voice frames: {non_voice_frames} (tone/erasure, skipped)");
    println!();
    if voice_frames == 0 {
        println!("No voice frames to compare.");
        return Ok(());
    }
    let pct = |n: usize| 100.0 * n as f64 / voice_frames as f64;
    println!("Parameter-level bit-exact roundtrip (voice frames only):");
    println!("  pitch   b̂₀:  {pitch_match:>5} / {voice_frames}   {:5.1}%", pct(pitch_match));
    println!("  V/UV    b̂₁:  {vuv_match:>5} / {voice_frames}   {:5.1}%", pct(vuv_match));
    println!("  gain    b̂₂:  {gain_match:>5} / {voice_frames}   {:5.1}%", pct(gain_match));
    Ok(())
}

fn cmd_halfrate_fec_histogram(root: &Path, name: &str, rate: &str, rc: bool) -> Result<()> {
    let base = if rc { root.join("tv-rc") } else { root.join("tv-std").join("tv") };
    let bit_path = base.join(rate).join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!(
            "{} is {} bytes — not a multiple of the 9-byte (72-bit) half-rate frame. \
             This is the expected shape for rate 33 and rate 39. \
             Rate 34 is 49-bit payload-only (7 bytes/frame) and the Golay \
             [24,12] histogram does not apply to it.",
            bit_path.display(),
            bit_bytes.len()
        ));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    let mut hist = [0usize; 5]; // 0, 1, 2, 3, uncorrectable
    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_half_frame(&dibits);
        let e = ambe.errors[0];
        let bucket = if e == u8::MAX { 4 } else { usize::from(e).min(3) };
        hist[bucket] += 1;
    }

    let label = if rc { format!("tv-rc/{rate}") } else { format!("tv-std/tv/{rate}") };
    println!("vector:  {name} ({label})");
    println!("frames:  {n_frames}");
    println!();
    println!("Golay [24,12] error-count histogram (ĉ₀):");
    let pct = |n: usize| 100.0 * n as f64 / n_frames as f64;
    println!("  errors=0:            {:>5}   {:5.1}%", hist[0], pct(hist[0]));
    println!("  errors=1:            {:>5}   {:5.1}%", hist[1], pct(hist[1]));
    println!("  errors=2:            {:>5}   {:5.1}%", hist[2], pct(hist[2]));
    println!("  errors=3:            {:>5}   {:5.1}%", hist[3], pct(hist[3]));
    println!("  uncorrectable (≥4):  {:>5}   {:5.1}%", hist[4], pct(hist[4]));
    println!();
    let correctable = hist[0] + hist[1] + hist[2] + hist[3];
    println!("correctable frames:    {correctable} / {n_frames}   ({:.1}%)", pct(correctable));
    println!();
    // Expected rate-33 chance hit rate for random 24-bit vectors: 2^12/2^24 ≈ 2.4e-4.
    // One or two "errors=0" frames out of a few hundred is consistent with rate-39
    // noise matching a codeword by chance; a cluster that dominates is a rate-33 signal.
    let chance_zero_cap = (n_frames / 100).max(2);
    let correctable_pct = 100 * correctable / n_frames;
    if hist[0] == n_frames {
        println!("PASS — every frame is a valid Golay [24,12] codeword. Input is genuine rate 33 (FEC-encoded).");
    } else if hist[0] <= chance_zero_cap && (50..=65).contains(&correctable_pct) {
        println!("FINGERPRINT MATCH — errors=0 count is at chance level and correctable-frame ratio is ~57%.");
        println!("This is the `C(24,k)/2325` signature of no-FEC bits fed through a Golay decoder.");
        println!("Input is NOT rate 33. Likely rate 39 (72 raw voice bits, no FEC) — see");
        println!("~/blip25-specs/analysis/dvsi_test_vector_modes.md.");
    } else {
        println!("AMBIGUOUS — neither a clean rate-33 pass nor the rate-39 fingerprint.");
        println!("Investigate: real channel errors, a different rate mode, or a bit-layout bug.");
    }
    Ok(())
}

fn reverse_bits_within(v: u32, n: u8) -> u32 {
    let mut r = 0u32;
    for k in 0..n {
        if (v >> k) & 1 == 1 {
            r |= 1u32 << (n - 1 - k);
        }
    }
    r
}

/// Locate the directory holding `<name>.pcm` and the `p25/`, `p25_nofec/`
/// subdirectories. The standard layout has them under `tv-std/tv/`; the
/// reference-channel set lives under `tv-rc/`.
fn vector_dir(root: &Path, rc: bool) -> PathBuf {
    if rc {
        root.join("tv-rc")
    } else {
        root.join("tv-std").join("tv")
    }
}

fn cmd_compare(root: &Path, name: &str, rc: bool, stop_on_first: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let nofec_path = dir.join("p25_nofec").join(format!("{name}.bit"));

    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    let nofec_bytes = fs::read(&nofec_path)
        .with_context(|| format!("read no-FEC vector {}", nofec_path.display()))?;

    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "FEC file {} is {} bytes — not a multiple of {} (frame size)",
            fec_path.display(),
            fec_bytes.len(),
            BYTES_PER_FEC_FRAME
        ));
    }
    if nofec_bytes.len() % BYTES_PER_NOFEC_FRAME != 0 {
        return Err(anyhow!(
            "no-FEC file {} is {} bytes — not a multiple of {} (frame size)",
            nofec_path.display(),
            nofec_bytes.len(),
            BYTES_PER_NOFEC_FRAME
        ));
    }

    let n_fec_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    let n_nofec_frames = nofec_bytes.len() / BYTES_PER_NOFEC_FRAME;
    if n_fec_frames != n_nofec_frames {
        return Err(anyhow!(
            "frame count mismatch: FEC={n_fec_frames}, no-FEC={n_nofec_frames}"
        ));
    }
    let n_frames = n_fec_frames;

    println!("vector:    {name}{}", if rc { " (rc)" } else { "" });
    println!("frames:    {n_frames}");
    println!();

    let mut matches = 0usize;
    let mut mismatches = 0usize;
    let mut total_fec_errors = 0u32;

    for f in 0..n_frames {
        let fec_frame = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let nofec_frame =
            &nofec_bytes[f * BYTES_PER_NOFEC_FRAME..(f + 1) * BYTES_PER_NOFEC_FRAME];

        let dibits = unpack_dibits(fec_frame);
        let decoded: FullFrame = decode_full_frame(&dibits);
        let expected = unpack_nofec_info(nofec_frame);
        total_fec_errors += u32::from(decoded.error_total());

        if decoded.info == expected {
            matches += 1;
        } else {
            mismatches += 1;
            if stop_on_first {
                report_mismatch(f, &decoded, &expected);
                println!();
                println!(
                    "result:    {matches} / {n_frames} frames matched ({mismatches} mismatches before stop)"
                );
                std::process::exit(1);
            }
        }
    }

    println!("matched:   {matches} / {n_frames}");
    println!("mismatched: {mismatches}");
    println!("avg FEC errors per frame: {:.3}", f64::from(total_fec_errors) / n_frames as f64);

    if mismatches == 0 {
        println!();
        println!("PASS — channel codec output matches DVSI no-FEC reference for every frame.");
        Ok(())
    } else {
        Err(anyhow!("{mismatches} frame(s) mismatched"))
    }
}

fn report_mismatch(frame_idx: usize, decoded: &FullFrame, expected: &[u16; 8]) {
    println!("FIRST DIVERGENCE at frame {frame_idx}");
    println!("  vector  width  decoded   expected  errors");
    for i in 0..8 {
        let mark = if decoded.info[i] == expected[i] { ' ' } else { '*' };
        println!(
            "  û_{i}    {:>3}    0x{:04x}    0x{:04x}    {}     {mark}",
            INFO_WIDTHS[i], decoded.info[i], expected[i], decoded.errors[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Bit unpacking
// ---------------------------------------------------------------------------

/// Read bits MSB-first from a byte slice. Returns each multi-bit read as a
/// `u16` with the *first-read* bit at the highest used position (i.e. a
/// 12-bit read returns a value where bit 11 is the first bit on the wire).
struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read(&mut self, n_bits: u8) -> u16 {
        let mut v = 0u16;
        for _ in 0..n_bits {
            let byte = self.bytes[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | u16::from(bit);
            self.pos += 1;
        }
        v
    }
}

/// MSB-first bit packer — inverse of [`BitReader`]. Writes bits
/// into a caller-owned byte buffer; the buffer must be pre-zeroed
/// and sized to hold the full bit payload.
struct BitWriter<'a> {
    bytes: &'a mut [u8],
    pos: usize,
}

impl<'a> BitWriter<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn write(&mut self, value: u16, n_bits: u8) {
        for i in (0..n_bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.pos / 8;
            let bit_idx = 7 - (self.pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.pos += 1;
        }
    }
}

/// Pack 72 dibits into 18 FEC bytes (inverse of [`unpack_dibits`]).
fn pack_dibits(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; BYTES_PER_FEC_FRAME] {
    let mut out = [0u8; BYTES_PER_FEC_FRAME];
    let mut w = BitWriter::new(&mut out);
    for &d in dibits {
        let hi = (d >> 1) & 1;
        let lo = d & 1;
        w.write(u16::from(hi), 1);
        w.write(u16::from(lo), 1);
    }
    out
}

/// Pack 8 info vectors into 11 no-FEC bytes (inverse of [`unpack_nofec_info`]).
fn pack_nofec_info(info: &[u16; 8]) -> [u8; BYTES_PER_NOFEC_FRAME] {
    let mut out = [0u8; BYTES_PER_NOFEC_FRAME];
    let mut w = BitWriter::new(&mut out);
    for (i, &v) in info.iter().enumerate() {
        w.write(v, INFO_WIDTHS[i]);
    }
    out
}

/// Unpack 18 FEC bytes into 72 dibits in transmission order. Each dibit's
/// high bit is the bit transmitted first.
fn unpack_dibits(bytes: &[u8]) -> [u8; 72] {
    debug_assert_eq!(bytes.len(), BYTES_PER_FEC_FRAME);
    let mut r = BitReader::new(bytes);
    let mut out = [0u8; DIBITS_PER_FRAME];
    for i in 0..DIBITS_PER_FRAME {
        let hi = r.read(1) as u8;
        let lo = r.read(1) as u8;
        out[i] = (hi << 1) | lo;
    }
    out
}

/// Unpack 11 no-FEC bytes into the 8 info vectors `û₀..û₇`. The 88 bits
/// are concatenated MSB-first with the widths in [`INFO_WIDTHS`]. Each
/// returned `u16` has element-k-at-bit-k packing matching the rest of
/// the `blip25-mbe` API.
fn unpack_nofec_info(bytes: &[u8]) -> [u16; 8] {
    debug_assert_eq!(bytes.len(), BYTES_PER_NOFEC_FRAME);
    let mut r = BitReader::new(bytes);
    let mut info = [0u16; 8];
    for i in 0..8 {
        info[i] = r.read(INFO_WIDTHS[i]);
    }
    info
}

// ---------------------------------------------------------------------------
// Rate-conversion harness (cmd: rate-convert, rate-convert-roundtrip)
// ---------------------------------------------------------------------------

/// Accumulator for parameter-level distortion between a source
/// `MbeParams` and a destination `MbeParams` (either the far-side of a
/// single conversion, or the re-decoded result of an A→B→A round-trip).
#[derive(Default)]
struct ConvertStats {
    frames: usize,
    /// Sum of |cents error| per comparable frame. `cents = 1200 ·
    /// log2(ω₀_dst / ω₀_src)`.
    pitch_cents_abs_sum: f64,
    /// Sum of squared cents error for RMS.
    pitch_cents_sq_sum: f64,
    /// Frames where `L_dst != L_src`.
    l_mismatches: usize,
    /// Distribution of |L_dst − L_src|.
    l_delta_hist: std::collections::BTreeMap<i32, usize>,
    /// Harmonics where we could compare (min(L_src, L_dst) per frame).
    harmonics_compared: usize,
    /// Harmonics where `ṽ` matched.
    voicing_matches: usize,
    /// Sum of squared (20·log10) amplitude error per comparable
    /// harmonic. Uses `max(M, 1e-6)` to guard against log(0).
    amp_db_sq_sum: f64,
    /// Sum of |20·log10 ratio| per comparable harmonic.
    amp_db_abs_sum: f64,
}

impl ConvertStats {
    fn push(&mut self, src: &MbeParams, dst: &MbeParams) {
        self.frames += 1;

        let w_src = f64::from(src.omega_0());
        let w_dst = f64::from(dst.omega_0());
        let cents = 1200.0 * (w_dst / w_src).log2();
        self.pitch_cents_abs_sum += cents.abs();
        self.pitch_cents_sq_sum += cents * cents;

        let l_src = src.harmonic_count() as i32;
        let l_dst = dst.harmonic_count() as i32;
        if l_src != l_dst {
            self.l_mismatches += 1;
        }
        *self.l_delta_hist.entry(l_dst - l_src).or_insert(0) += 1;

        let l_min = l_src.min(l_dst) as usize;
        let v_src = src.voiced_slice();
        let v_dst = dst.voiced_slice();
        let a_src = src.amplitudes_slice();
        let a_dst = dst.amplitudes_slice();
        for i in 0..l_min {
            self.harmonics_compared += 1;
            if v_src[i] == v_dst[i] {
                self.voicing_matches += 1;
            }
            // Guard against log2(0) — MbeParams enforces non-negative
            // amplitudes but silence frames can carry exact zeros.
            let a_s = (a_src[i] as f64).max(1e-6);
            let a_d = (a_dst[i] as f64).max(1e-6);
            let db = 20.0 * (a_d / a_s).log10();
            self.amp_db_abs_sum += db.abs();
            self.amp_db_sq_sum += db * db;
        }
    }

    fn print(&self, label: &str) {
        if self.frames == 0 {
            println!("{label}: 0 comparable frames (nothing to report)");
            return;
        }
        let n = self.frames as f64;
        let hn = self.harmonics_compared.max(1) as f64;
        let pitch_rms = (self.pitch_cents_sq_sum / n).sqrt();
        let pitch_mean_abs = self.pitch_cents_abs_sum / n;
        let l_mismatch_pct = 100.0 * self.l_mismatches as f64 / n;
        let voicing_pct = 100.0 * self.voicing_matches as f64
            / (self.harmonics_compared.max(1) as f64);
        let amp_rms_db = (self.amp_db_sq_sum / hn).sqrt();
        let amp_mean_abs_db = self.amp_db_abs_sum / hn;

        println!("{label} ({} comparable frames):", self.frames);
        println!(
            "  pitch error:       mean |Δ| = {pitch_mean_abs:6.2} cents   RMS = {pitch_rms:6.2} cents"
        );
        println!(
            "  L mismatch:        {} / {} frames ({:.1}%)",
            self.l_mismatches, self.frames, l_mismatch_pct
        );
        if !self.l_delta_hist.is_empty() {
            let mut entries: Vec<(i32, usize)> = self
                .l_delta_hist
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            entries.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = entries
                .iter()
                .take(5)
                .map(|(d, c)| format!("Δ={d:+}: {c}"))
                .collect();
            println!("  L Δ top-5:         {}", top.join(", "));
        }
        println!(
            "  voicing agreement: {} / {} harmonics ({:.1}%)",
            self.voicing_matches, self.harmonics_compared, voicing_pct
        );
        println!(
            "  amplitude error:   mean |Δ| = {amp_mean_abs_db:5.2} dB   RMS = {amp_rms_db:5.2} dB"
        );
    }
}

/// Tally of frames that couldn't participate in parameter comparison.
#[derive(Default)]
struct SkipStats {
    src_decode_errors: usize,
    dst_decode_errors: usize,
    convert_errors: usize,
    /// Source was a tone frame; converter passed it through to the
    /// destination rate (as voice for full-rate dest, as tone-signal
    /// dibits for same-rate) successfully. Excluded from parameter
    /// comparison because tone params aren't voice params.
    tone_passthrough: usize,
    /// Source was an erasure/silence frame; converter emitted an
    /// erasure signal on the destination side.
    erasure_passthrough: usize,
    /// Converter was expected to emit an erasure signal on the
    /// destination side for a non-voice source, but produced
    /// something else (voice or a different kind). A bug.
    erasure_mismatch: usize,
}

impl SkipStats {
    fn non_comparable(&self) -> usize {
        self.src_decode_errors
            + self.dst_decode_errors
            + self.convert_errors
            + self.tone_passthrough
            + self.erasure_passthrough
            + self.erasure_mismatch
    }

    fn print(&self) {
        if self.non_comparable() == 0 {
            println!("non-comparable frames: 0");
            return;
        }
        println!("non-comparable frames: {}", self.non_comparable());
        if self.tone_passthrough > 0 {
            println!("  tone pass-through (src→dst):     {}", self.tone_passthrough);
        }
        if self.erasure_passthrough > 0 {
            println!("  erasure pass-through (src→dst):  {}", self.erasure_passthrough);
        }
        if self.erasure_mismatch > 0 {
            println!("  erasure MISMATCH (bug — inspect): {}", self.erasure_mismatch);
        }
        if self.src_decode_errors > 0 {
            println!("  source dequantize errors:        {}", self.src_decode_errors);
        }
        if self.convert_errors > 0 {
            println!("  converter errors (dst pitch):    {}", self.convert_errors);
        }
        if self.dst_decode_errors > 0 {
            println!("  dst dequantize errors:           {}", self.dst_decode_errors);
        }
    }
}

fn cmd_rate_convert(
    root: &Path,
    direction: RateConvertDirection,
    name: &str,
    rc: bool,
) -> Result<()> {
    match direction {
        RateConvertDirection::FullToHalf => cmd_rate_convert_full_to_half(root, name, rc),
        RateConvertDirection::HalfToFull => cmd_rate_convert_half_to_full(root, name, rc),
    }
}

fn cmd_rate_convert_full_to_half(root: &Path, name: &str, rc: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "FEC file {} is {} bytes — not a multiple of {}",
            fec_path.display(),
            fec_bytes.len(),
            BYTES_PER_FEC_FRAME
        ));
    }
    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;

    println!("vector:            {name}{} (full → half)", if rc { " (rc)" } else { "" });
    println!("frames:            {n_frames}");
    println!();

    let mut src_state = DecoderState::new();
    let mut dst_state = HalfDecoderState::new();
    let mut conv = FullToHalfConverter::new();
    let mut stats = ConvertStats::default();
    let mut skips = SkipStats::default();

    for f in 0..n_frames {
        let frame = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(frame);

        let src_frame = decode_full_frame(&dibits);
        let src_params_res = dequantize_full(&src_frame.info, &mut src_state);

        let dst_dibits = match conv.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                continue;
            }
        };

        match src_params_res {
            Ok(src_params) => {
                let dst_frame = decode_half_frame(&dst_dibits);
                let dst_params = match dequantize_half(&dst_frame.info, &mut dst_state) {
                    Ok(p) => p,
                    Err(_) => {
                        skips.dst_decode_errors += 1;
                        continue;
                    }
                };
                stats.push(&src_params, &dst_params);
            }
            Err(DecodeError::BadPitch) => {
                // Full-rate reserved pitch — converter should emit a
                // half-rate erasure signal.
                let dst_frame = decode_half_frame(&dst_dibits);
                use blip25_mbe::p25_halfrate::dequantize::{FrameKind, classify_halfrate_frame};
                match classify_halfrate_frame(&dst_frame.info) {
                    FrameKind::Erasure => skips.erasure_passthrough += 1,
                    _ => skips.erasure_mismatch += 1,
                }
            }
            Err(_) => skips.src_decode_errors += 1,
        }
    }

    skips.print();
    println!();
    stats.print("Source → half-rate (voice frames only)");
    Ok(())
}

fn cmd_rate_convert_half_to_full(root: &Path, name: &str, rc: bool) -> Result<()> {
    let base = if rc { root.join("tv-rc") } else { root.join("tv-std").join("tv") };
    let bit_path = base.join("r33").join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!(
            "half-rate .bit file {} is {} bytes — not a multiple of {}",
            bit_path.display(),
            bit_bytes.len(),
            BYTES_PER_HALFRATE_FRAME
        ));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    println!("vector:            {name}{} (half → full, r33)", if rc { " (rc)" } else { "" });
    println!("frames:            {n_frames}");
    println!();

    let mut src_state = HalfDecoderState::new();
    let mut dst_state = DecoderState::new();
    let mut conv = HalfToFullConverter::new();
    let mut stats = ConvertStats::default();
    let mut skips = SkipStats::default();

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);

        // Classify the source kind via the tone-aware dispatch, then
        // exercise the converter on ALL kinds — tone/erasure frames
        // are expected to pass through as voice/erasure on the
        // destination side, and we now verify that rather than
        // pre-filtering them out.
        let ambe = decode_half_frame(&dibits);
        let src_kind = decode_to_params(&ambe.info, &mut src_state);

        let dst_dibits = match conv.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                continue;
            }
        };

        match src_kind {
            Ok(Decoded::Voice(src_params)) => {
                let dst_frame = decode_full_frame(&dst_dibits);
                let dst_params = match dequantize_full(&dst_frame.info, &mut dst_state) {
                    Ok(p) => p,
                    Err(_) => {
                        skips.dst_decode_errors += 1;
                        continue;
                    }
                };
                stats.push(&src_params, &dst_params);
            }
            Ok(Decoded::Tone { .. }) => {
                // Tone → full-rate voice. Verify dst re-decodes to
                // something plausible (or is an erasure signal if
                // the tone's ω₀ lands outside full-rate grid). Either
                // outcome is a legitimate pass-through.
                let dst_frame = decode_full_frame(&dst_dibits);
                match dequantize_full(&dst_frame.info, &mut dst_state) {
                    Ok(_) | Err(DecodeError::BadPitch) => {
                        skips.tone_passthrough += 1;
                    }
                    Err(_) => skips.dst_decode_errors += 1,
                }
            }
            Ok(Decoded::Erasure) | Err(_) => {
                // Erasure → full-rate erasure signal. Destination
                // must re-decode as BadPitch (frame-repeat trigger
                // at §1.11).
                let dst_frame = decode_full_frame(&dst_dibits);
                match dequantize_full(&dst_frame.info, &mut dst_state) {
                    Err(DecodeError::BadPitch) => skips.erasure_passthrough += 1,
                    _ => skips.erasure_mismatch += 1,
                }
            }
        }
    }

    skips.print();
    println!();
    stats.print("Source → full-rate (voice frames only)");
    Ok(())
}

fn cmd_rate_convert_roundtrip(
    root: &Path,
    direction: RateConvertDirection,
    name: &str,
    rc: bool,
) -> Result<()> {
    match direction {
        RateConvertDirection::FullToHalf => cmd_rate_convert_roundtrip_full(root, name, rc),
        RateConvertDirection::HalfToFull => cmd_rate_convert_roundtrip_half(root, name, rc),
    }
}

fn cmd_rate_convert_roundtrip_full(root: &Path, name: &str, rc: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!("FEC file is not a whole number of frames"));
    }
    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;

    println!("vector:            {name}{} (full → half → full)", if rc { " (rc)" } else { "" });
    println!("frames:            {n_frames}");
    println!();

    let mut src_state = DecoderState::new();
    let mut final_state = DecoderState::new();
    let mut a_to_b = FullToHalfConverter::new();
    let mut b_to_a = HalfToFullConverter::new();
    let mut stats = ConvertStats::default();
    let mut skips = SkipStats::default();

    for f in 0..n_frames {
        let frame = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(frame);

        let src_frame = decode_full_frame(&dibits);
        let src_params = match dequantize_full(&src_frame.info, &mut src_state) {
            Ok(p) => p,
            Err(_) => {
                skips.src_decode_errors += 1;
                continue;
            }
        };

        let mid = match a_to_b.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                continue;
            }
        };
        let back = match b_to_a.convert(&mid) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                continue;
            }
        };
        let final_frame = decode_full_frame(&back);
        let final_params = match dequantize_full(&final_frame.info, &mut final_state) {
            Ok(p) => p,
            Err(_) => {
                skips.dst_decode_errors += 1;
                continue;
            }
        };

        stats.push(&src_params, &final_params);
    }

    skips.print();
    println!();
    stats.print("Source → half → full");
    Ok(())
}

fn cmd_rate_convert_roundtrip_half(root: &Path, name: &str, rc: bool) -> Result<()> {
    let base = if rc { root.join("tv-rc") } else { root.join("tv-std").join("tv") };
    let bit_path = base.join("r33").join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!("half-rate file is not a whole number of frames"));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    println!("vector:            {name}{} (half → full → half, r33)", if rc { " (rc)" } else { "" });
    println!("frames:            {n_frames}");
    println!();

    let mut src_state = HalfDecoderState::new();
    let mut final_state = HalfDecoderState::new();
    let mut a_to_b = HalfToFullConverter::new();
    let mut b_to_a = FullToHalfConverter::new();
    let mut stats = ConvertStats::default();
    let mut skips = SkipStats::default();
    let mut outbound_errs = 0usize;
    let mut inbound_errs = 0usize;
    let mut inbound_examples: Vec<(usize, f32, u8, f32)> = Vec::new();

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);

        let ambe = decode_half_frame(&dibits);
        let src_params = match decode_to_params(&ambe.info, &mut src_state) {
            Ok(Decoded::Voice(p)) => p,
            Ok(Decoded::Tone { .. }) => {
                skips.tone_passthrough += 1;
                continue;
            }
            Ok(Decoded::Erasure) => {
                skips.erasure_passthrough += 1;
                continue;
            }
            Err(_) => {
                skips.src_decode_errors += 1;
                continue;
            }
        };

        let mid = match a_to_b.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                outbound_errs += 1;
                continue;
            }
        };
        let back = match b_to_a.convert(&mid) {
            Ok(b) => b,
            Err(_) => {
                skips.convert_errors += 1;
                inbound_errs += 1;
                if inbound_examples.len() < 5 {
                    // Decode the `mid` full-rate frame to see what ω₀ the
                    // full-rate encoder actually emitted for this
                    // source frame.
                    let mf = decode_full_frame(&mid);
                    let mut probe = DecoderState::new();
                    let mid_omega = dequantize_full(&mf.info, &mut probe)
                        .map(|p| p.omega_0())
                        .unwrap_or(f32::NAN);
                    inbound_examples.push((
                        f,
                        src_params.omega_0(),
                        src_params.harmonic_count(),
                        mid_omega,
                    ));
                }
                continue;
            }
        };
        let final_frame = decode_half_frame(&back);
        let final_params = match dequantize_half(&final_frame.info, &mut final_state) {
            Ok(p) => p,
            Err(_) => {
                skips.dst_decode_errors += 1;
                continue;
            }
        };

        stats.push(&src_params, &final_params);
    }

    skips.print();
    if outbound_errs > 0 || inbound_errs > 0 {
        println!(
            "  (breakdown: outbound a→b {outbound_errs}, inbound b→a {inbound_errs})"
        );
    }
    if !inbound_examples.is_empty() {
        println!();
        println!("First {} inbound-hop failures (ω₀ src → full-rate mid):", inbound_examples.len());
        for (f, src_w, src_l, mid_w) in &inbound_examples {
            println!("  frame {f:>3}: src ω₀={src_w:.6} (L={src_l}) → mid ω₀={mid_w:.6}");
        }
    }
    println!();
    stats.print("Source → full → half");
    Ok(())
}

// ---------------------------------------------------------------------------
// Rate-convert smoothness metric (cmd: rate-convert-smoothness)
//
// Measures RMS of `|M̃_l(frame N+1) − M̃_l(frame N)|` across target-stream
// voice frames, averaged over the comparable harmonics of each pair.
// This is the metric the §4.5 cross-rate predictor actually targets
// (US7634399 col. 7 frame-to-frame magnitude smoothing); the existing
// rate-convert-roundtrip metric is insensitive because it compares
// unsmoothed source to output rather than output-vs-output.
// ---------------------------------------------------------------------------

/// Per-run smoothness statistics. Aggregated across frame pairs where
/// both the prior and current target-stream frames decoded as Voice
/// and share the same L̂ (cross-frame diff is well-defined only on a
/// common harmonic grid; L̂ changes are counted separately).
#[derive(Default)]
struct SmoothnessStats {
    /// Comparable frame pairs (prior + current both Voice, same L̂).
    comparable_pairs: usize,
    /// Frame pairs where L̂ changed — diff undefined, tallied.
    l_transition_pairs: usize,
    /// Pairs skipped because at least one side wasn't Voice (erasure,
    /// tone, silence, decode error).
    non_voice_pairs: usize,
    /// Σ|Δ M̃_l|² over all comparable harmonics (log2 domain), used to
    /// compute RMS. Log-domain so the metric is a scale-invariant
    /// smoothness measure rather than a linear-dB that conflates
    /// amplitude range with smoothness.
    sum_sq_log_delta: f64,
    /// Number of harmonic samples contributing to `sum_sq_log_delta`.
    n_harmonic_samples: usize,
    /// Linear-domain Σ|Δ M̃_l|² and count — for reporting alongside
    /// the log-domain metric (the chip's predictor operates in log
    /// domain per US7634399 col. 6, but linear is more intuitive).
    sum_sq_linear_delta: f64,
    n_linear_samples: usize,
}

impl SmoothnessStats {
    fn push_pair(&mut self, prev: &MbeParams, curr: &MbeParams) {
        if prev.harmonic_count() != curr.harmonic_count() {
            self.l_transition_pairs += 1;
            return;
        }
        self.comparable_pairs += 1;
        let l = prev.harmonic_count() as usize;
        let prev_a = prev.amplitudes_slice();
        let curr_a = curr.amplitudes_slice();
        // Linear-domain diff includes all harmonics — near-zero
        // amplitudes contribute ≈0 and don't distort the RMS.
        for i in 0..l {
            let d_lin = f64::from(curr_a[i]) - f64::from(prev_a[i]);
            self.sum_sq_linear_delta += d_lin * d_lin;
            self.n_linear_samples += 1;
        }
        // Log-domain diff is only meaningful when both sides have
        // non-trivial magnitudes. A harmonic that ping-pongs between
        // voiced and unvoiced-near-zero would dominate log-RMS via
        // the floor, which reflects voicing changes rather than the
        // smoothness the predictor targets. Gate at 1.0 (a generous
        // threshold — voiced M̃_l typically exceeds 10 on speech).
        const LOG_DOMAIN_THRESHOLD: f32 = 1.0;
        for i in 0..l {
            if prev_a[i] >= LOG_DOMAIN_THRESHOLD && curr_a[i] >= LOG_DOMAIN_THRESHOLD {
                let d_log = f64::from(curr_a[i]).log2() - f64::from(prev_a[i]).log2();
                self.sum_sq_log_delta += d_log * d_log;
                self.n_harmonic_samples += 1;
            }
        }
    }

    fn non_voice(&mut self) {
        self.non_voice_pairs += 1;
    }

    fn log_rms(&self) -> f64 {
        if self.n_harmonic_samples == 0 {
            return 0.0;
        }
        (self.sum_sq_log_delta / self.n_harmonic_samples as f64).sqrt()
    }

    fn linear_rms(&self) -> f64 {
        if self.n_linear_samples == 0 {
            return 0.0;
        }
        (self.sum_sq_linear_delta / self.n_linear_samples as f64).sqrt()
    }

    fn print(&self, label: &str) {
        println!("{label}:");
        println!(
            "  comparable pairs:      {} (L-transitions {}, non-voice {})",
            self.comparable_pairs, self.l_transition_pairs, self.non_voice_pairs,
        );
        println!(
            "  |Δ M̃_l| log2-RMS:      {:.4}    (linear-domain equivalent: {:.4}×)",
            self.log_rms(),
            (2.0_f64).powf(self.log_rms()),
        );
        println!("  |Δ M̃_l| linear-RMS:    {:.2}", self.linear_rms());
    }
}

fn cmd_rate_convert_smoothness(
    root: &Path,
    direction: RateConvertDirection,
    name: &str,
    rc: bool,
) -> Result<()> {
    match direction {
        RateConvertDirection::FullToHalf => smoothness_full_to_half(root, name, rc),
        RateConvertDirection::HalfToFull => smoothness_half_to_full(root, name, rc),
    }
}

/// Run one pass of full → half conversion with the predictor either on
/// or off, collecting smoothness statistics on the target (half-rate)
/// stream.
fn measure_smoothness_full_to_half(
    fec_bytes: &[u8],
    n_frames: usize,
    predictor_enabled: bool,
) -> SmoothnessStats {
    let mut conv = FullToHalfConverter::new();
    conv.set_predictor_enabled(predictor_enabled);
    let mut target_state = HalfDecoderState::new();
    let mut stats = SmoothnessStats::default();
    let mut prev: Option<MbeParams> = None;

    for f in 0..n_frames {
        let frame = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(frame);
        let dst_dibits = match conv.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                stats.non_voice();
                prev = None;
                continue;
            }
        };
        let dst_frame = decode_half_frame(&dst_dibits);
        match decode_to_params(&dst_frame.info, &mut target_state) {
            Ok(Decoded::Voice(params)) => {
                if let Some(prev_p) = &prev {
                    stats.push_pair(prev_p, &params);
                }
                prev = Some(params);
            }
            _ => {
                stats.non_voice();
                prev = None;
            }
        }
    }
    stats
}

fn smoothness_full_to_half(root: &Path, name: &str, rc: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!("FEC file is not a whole number of frames"));
    }
    let n_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;

    println!(
        "vector:            {name}{} (full → half smoothness)",
        if rc { " (rc)" } else { "" }
    );
    println!("frames:            {n_frames}");
    println!();

    let off = measure_smoothness_full_to_half(&fec_bytes, n_frames, false);
    off.print("Predictor OFF (§4.5 bypassed)");
    println!();
    let on = measure_smoothness_full_to_half(&fec_bytes, n_frames, true);
    on.print("Predictor ON  (§4.5 blend active)");
    println!();

    print_smoothness_delta(&off, &on);
    Ok(())
}

/// Run one pass of half → full conversion with the predictor either on
/// or off, collecting smoothness statistics on the target (full-rate)
/// stream.
fn measure_smoothness_half_to_full(
    bit_bytes: &[u8],
    n_frames: usize,
    predictor_enabled: bool,
) -> SmoothnessStats {
    let mut conv = HalfToFullConverter::new();
    conv.set_predictor_enabled(predictor_enabled);
    let mut target_state = DecoderState::new();
    let mut stats = SmoothnessStats::default();
    let mut prev: Option<MbeParams> = None;

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let dst_dibits = match conv.convert(&dibits) {
            Ok(b) => b,
            Err(_) => {
                stats.non_voice();
                prev = None;
                continue;
            }
        };
        let dst_frame = decode_full_frame(&dst_dibits);
        match dequantize_full(&dst_frame.info, &mut target_state) {
            Ok(params) => {
                if let Some(prev_p) = &prev {
                    stats.push_pair(prev_p, &params);
                }
                prev = Some(params);
            }
            Err(_) => {
                stats.non_voice();
                prev = None;
            }
        }
    }
    stats
}

fn smoothness_half_to_full(root: &Path, name: &str, rc: bool) -> Result<()> {
    let base = if rc { root.join("tv-rc") } else { root.join("tv-std").join("tv") };
    let bit_path = base.join("r33").join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!("half-rate file is not a whole number of frames"));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    println!(
        "vector:            {name}{} (half → full smoothness, r33)",
        if rc { " (rc)" } else { "" }
    );
    println!("frames:            {n_frames}");
    println!();

    let off = measure_smoothness_half_to_full(&bit_bytes, n_frames, false);
    off.print("Predictor OFF (§4.5 bypassed)");
    println!();
    let on = measure_smoothness_half_to_full(&bit_bytes, n_frames, true);
    on.print("Predictor ON  (§4.5 blend active)");
    println!();

    print_smoothness_delta(&off, &on);
    Ok(())
}

fn print_smoothness_delta(off: &SmoothnessStats, on: &SmoothnessStats) {
    let off_log = off.log_rms();
    let on_log = on.log_rms();
    let delta_log = on_log - off_log;
    let off_lin = off.linear_rms();
    let on_lin = on.linear_rms();
    let delta_lin = on_lin - off_lin;
    println!("A/B comparison:");
    println!("  log2-RMS   OFF = {off_log:.4}, ON = {on_log:.4}, Δ = {delta_log:+.4}");
    println!("  linear-RMS OFF = {off_lin:.2},   ON = {on_lin:.2},   Δ = {delta_lin:+.2}");
    let lin_better = delta_lin < 0.0;
    let log_better = delta_log < 0.0;
    match (lin_better, log_better) {
        (true, true) => {
            println!("  ✓ Predictor reduces frame-to-frame magnitude swings in both domains.");
        }
        (true, false) => {
            println!(
                "  ~ Mixed: linear-RMS improves (predictor IS smoothing in magnitude domain)\n    \
                   but log-RMS regresses (investigating — likely the §4.3.3 energy-preserving\n    \
                   offset `+0.5·log₂(R_prev)` introducing log-domain variation on pitch\n    \
                   transitions). Per-vector behavior varies."
            );
        }
        (false, true) => {
            println!("  ~ Mixed: log-RMS improves but linear-RMS regresses — unusual pattern.");
        }
        (false, false) => {
            println!("  ✗ Predictor does NOT reduce smoothness in either domain — investigate.");
        }
    }
}

// ---------------------------------------------------------------------------
// Half-rate analysis-encoder harness (cmd: halfrate-analysis-encode)
//
// Parallels `cmd_analysis_encode` but drives `encode_halfrate` against
// tv-rc/r33 PCM + bits. Compares per-frame MbeParams between the
// analysis encoder and the chip's dequantize of the r33 reference
// bitstream. Same chip_offset = −2 convention as full-rate.
// ---------------------------------------------------------------------------

fn cmd_halfrate_analysis_encode(
    root: &Path,
    name: &str,
    chip_offset: i32,
    silence_dispatch: bool,
) -> Result<()> {
    // Half-rate vectors live under tv-rc/r33/ per the existing
    // cmd_decode_pcm_halfrate convention.
    let dir = root.join("tv-rc").join("r33");
    let pcm_path = dir.join(format!("{name}.pcm"));
    let bit_path = dir.join(format!("{name}.bit"));

    let pcm_bytes = fs::read(&pcm_path)
        .with_context(|| format!("read PCM vector {}", pcm_path.display()))?;
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate bit vector {}", bit_path.display()))?;

    if pcm_bytes.len() % (FRAME_SAMPLES * 2) != 0 {
        return Err(anyhow!(
            "PCM file {} is {} bytes — not a whole number of 20 ms frames",
            pcm_path.display(),
            pcm_bytes.len()
        ));
    }
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!(
            "half-rate bit file {} is {} bytes — not a multiple of {}",
            bit_path.display(),
            bit_bytes.len(),
            BYTES_PER_HALFRATE_FRAME
        ));
    }
    let n_pcm_frames = pcm_bytes.len() / (FRAME_SAMPLES * 2);
    let n_bit_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;
    if n_pcm_frames != n_bit_frames {
        return Err(anyhow!(
            "frame-count mismatch: {} PCM frames vs {} bit frames",
            n_pcm_frames,
            n_bit_frames
        ));
    }
    let n_frames = n_pcm_frames;

    let pcm: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    println!(
        "vector:            {name} (half-rate r33, analysis-encode)"
    );
    println!("frames:            {n_frames}");
    if chip_offset != 0 {
        println!(
            "chip offset:       {chip_offset:+} (chip[f + offset] vs encoder[f])"
        );
    }
    if silence_dispatch {
        println!("silence dispatch:  on (silent frames tallied, not compared)");
    }
    println!();

    // Pre-decode every chip bit-frame. Only Voice frames feed into
    // stats; Tone / Erasure get tallied separately.
    let mut chip_state = HalfDecoderState::new();
    let chip_frames: Vec<Decoded> = (0..n_frames)
        .map(|f| {
            let fec = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
            let dibits = unpack_dibits_halfrate(fec);
            let ambe = decode_half_frame(&dibits);
            decode_to_params(&ambe.info, &mut chip_state).unwrap_or(Decoded::Erasure)
        })
        .collect();

    let mut encoder_state = AnalysisState::new();
    if silence_dispatch {
        encoder_state.set_silence_detection(true);
    }
    let mut stats = ConvertStats::default();
    let mut stats_l_match = ConvertStats::default();
    let mut stats_pitch_close = ConvertStats::default();
    let mut stats_l_and_pitch = ConvertStats::default();
    let mut preroll_skips = 0usize;
    let mut encoder_silence = 0usize;
    let mut tone_frames = 0usize;
    let mut erasure_frames = 0usize;

    for f in 0..n_frames {
        let pcm_frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let encoded = analysis_encode_halfrate(pcm_frame, &mut encoder_state);

        let chip_idx = f as i32 + chip_offset;
        let chip_ref: Option<&Decoded> = if chip_idx >= 0 && (chip_idx as usize) < n_frames {
            Some(&chip_frames[chip_idx as usize])
        } else {
            None
        };

        match (encoded, chip_ref) {
            (Ok(AnalysisOutput::Voice(enc_params)), Some(Decoded::Voice(ref_params))) => {
                stats.push(ref_params, &enc_params);
                let lc = ref_params.harmonic_count();
                let le = enc_params.harmonic_count();
                let l_match = lc == le;
                let wc = f64::from(ref_params.omega_0());
                let we = f64::from(enc_params.omega_0());
                let cents = 1200.0 * (we / wc).log2();
                let pitch_close = cents.abs() < 50.0;
                if l_match {
                    stats_l_match.push(ref_params, &enc_params);
                }
                if pitch_close {
                    stats_pitch_close.push(ref_params, &enc_params);
                }
                if l_match && pitch_close {
                    stats_l_and_pitch.push(ref_params, &enc_params);
                }
            }
            (Ok(AnalysisOutput::Voice(_)), Some(Decoded::Tone { .. })) => {
                tone_frames += 1;
            }
            (Ok(AnalysisOutput::Voice(_)), Some(Decoded::Erasure)) => {
                erasure_frames += 1;
            }
            (Ok(AnalysisOutput::Silence), _) => {
                if chip_ref.is_none() {
                    preroll_skips += 1;
                } else {
                    encoder_silence += 1;
                }
            }
            (Err(_), _) | (_, None) => {
                preroll_skips += 1;
            }
        }
    }

    println!(
        "non-comparable frames: preroll {preroll_skips}, encoder silence {encoder_silence}, \
         chip tone {tone_frames}, chip erasure {erasure_frames}"
    );
    println!();

    stats.print("All voice frames");
    println!();
    stats_l_match.print("L-matching frames only");
    println!();
    stats_pitch_close.print("Pitch-close (<50¢) frames only");
    println!();
    stats_l_and_pitch.print("L-matching AND pitch-close frames");

    Ok(())
}

// ---------------------------------------------------------------------------
// Analysis-encoder harness (cmd: analysis-encode)
// ---------------------------------------------------------------------------

/// Non-comparable frame tally for the PCM → `MbeParams` analysis
/// encoder. Parallels `SkipStats` but tracks categories specific to
/// the analysis path (preroll, encoder-emitted silence, chip-side
/// dequantize failures).
#[derive(Default)]
struct AnalysisSkipStats {
    /// Preroll frames — the encoder needs two frames of look-ahead
    /// before it can emit voice (`PREROLL_FRAMES = 2`). These always
    /// return `AnalysisOutput::Silence` and are structurally
    /// non-comparable against chip voice output.
    preroll: usize,
    /// Post-preroll frames the encoder dispatched as silence (only
    /// possible if §0.8 silence detection is enabled; currently off).
    encoder_silence: usize,
    /// `encode()` returned `Err` (pitch out of range, spec gap, etc.).
    encoder_errors: usize,
    /// Chip-side `dequantize_full` returned `Err` (reserved pitch,
    /// tone, erasure, etc.).
    chip_decode_errors: usize,
    /// Chip bit-frame index `f + chip_offset` fell outside
    /// `[0, n_frames)`. Only nonzero when `--chip-offset` is used.
    chip_out_of_range: usize,
}

impl AnalysisSkipStats {
    fn non_comparable(&self) -> usize {
        self.preroll
            + self.encoder_silence
            + self.encoder_errors
            + self.chip_decode_errors
            + self.chip_out_of_range
    }

    fn print(&self) {
        if self.non_comparable() == 0 {
            println!("non-comparable frames: 0");
            return;
        }
        println!("non-comparable frames: {}", self.non_comparable());
        if self.preroll > 0 {
            println!("  preroll (encoder look-ahead fill): {}", self.preroll);
        }
        if self.encoder_silence > 0 {
            println!("  encoder silence dispatch:          {}", self.encoder_silence);
        }
        if self.encoder_errors > 0 {
            println!("  encoder errors:                    {}", self.encoder_errors);
        }
        if self.chip_decode_errors > 0 {
            println!("  chip dequantize errors:            {}", self.chip_decode_errors);
        }
        if self.chip_out_of_range > 0 {
            println!("  chip bit-frame out of range:       {}", self.chip_out_of_range);
        }
    }
}

/// Format a compact voicing signature: `V`/`U`/`-` per harmonic for
/// `l = 1..=L`, truncated to at most `cap` entries.
fn voicing_sig(params: &MbeParams, cap: usize) -> String {
    let l = params.harmonic_count() as usize;
    let v = params.voiced_slice();
    let n = l.min(cap);
    let mut s = String::with_capacity(n + 2);
    for i in 0..n {
        s.push(if v[i] { 'V' } else { 'U' });
    }
    if l > n {
        s.push('…');
    }
    s
}

/// Per-band V/UV disagreement tracker for analysis-encode diagnostics.
/// Accumulates statistics on which bands disagree between encoder and chip,
/// and the D_k / Θ_ξ margins at disagreeing bands.
struct VuvBandStats {
    /// Max band index seen (for printing).
    max_k: u8,
    /// Total frames analyzed.
    frames: usize,
    /// Per-band: total frames where this band was compared.
    band_compared: [usize; 13],
    /// Per-band: frames where encoder and chip disagree.
    band_disagree: [usize; 13],
    /// Per-band: encoder voiced → chip unvoiced.
    band_v_to_u: [usize; 13],
    /// Per-band: encoder unvoiced → chip voiced.
    band_u_to_v: [usize; 13],
    /// Per-band: sum of D_k on disagree frames (for mean D_k at disagree).
    band_dk_sum_disagree: [f64; 13],
    /// Per-band: sum of Θ_ξ on disagree frames.
    band_theta_sum_disagree: [f64; 13],
    /// Per-band: sum of (D_k - Θ_ξ) on all frames (for mean margin).
    band_margin_sum: [f64; 13],
    /// Per-band: sum of |D_k - Θ_ξ| on all frames.
    band_margin_abs_sum: [f64; 13],
}

impl Default for VuvBandStats {
    fn default() -> Self {
        Self {
            max_k: 0,
            frames: 0,
            band_compared: [0; 13],
            band_disagree: [0; 13],
            band_v_to_u: [0; 13],
            band_u_to_v: [0; 13],
            band_dk_sum_disagree: [0.0; 13],
            band_theta_sum_disagree: [0.0; 13],
            band_margin_sum: [0.0; 13],
            band_margin_abs_sum: [0.0; 13],
        }
    }
}

impl VuvBandStats {
    fn push(&mut self, chip: &MbeParams, enc: &MbeParams, vuv: &VuvResult) {
        self.frames += 1;
        let l_chip = chip.harmonic_count();
        let l_enc = enc.harmonic_count();
        let k_enc = vuv.k_hat;
        let k_chip = band_count_for(l_chip);
        let k_min = k_enc.min(k_chip);
        if k_min > self.max_k {
            self.max_k = k_min;
        }

        for k in 1..=k_min {
            let ki = k as usize;
            self.band_compared[ki] += 1;

            // Encoder band-level decision from VuvResult.
            let enc_voiced = vuv.vuv[ki] == 1;

            // Chip band-level decision: sample the first harmonic in
            // this band from the chip's per-harmonic voicing.
            let first_l_in_band = 3 * u32::from(k) - 2;
            let chip_voiced = if (first_l_in_band as u8) <= l_chip {
                chip.voiced_slice()[(first_l_in_band - 1) as usize]
            } else {
                false
            };

            let d_k = vuv.d_k[ki];
            let theta = vuv.theta_k[ki];
            let margin = d_k - theta;
            self.band_margin_sum[ki] += margin;
            self.band_margin_abs_sum[ki] += margin.abs();

            if enc_voiced != chip_voiced {
                self.band_disagree[ki] += 1;
                self.band_dk_sum_disagree[ki] += d_k;
                self.band_theta_sum_disagree[ki] += theta;
                if enc_voiced {
                    self.band_v_to_u[ki] += 1;
                } else {
                    self.band_u_to_v[ki] += 1;
                }
            }
        }
    }

    fn print(&self) {
        if self.frames == 0 {
            println!("V/UV band analysis: 0 frames");
            return;
        }
        println!("V/UV per-band analysis ({} frames):", self.frames);
        println!(
            "  {:>4}  {:>6} {:>6} {:>6}  {:>5} {:>5}  {:>8} {:>8}  {:>8}",
            "band", "compar", "disag", "dis%",
            "V→U", "U→V",
            "D̄_k dis", "Θ̄_ξ dis", "|margin|"
        );
        for k in 1..=self.max_k as usize {
            let c = self.band_compared[k];
            if c == 0 { continue; }
            let d = self.band_disagree[k];
            let pct = 100.0 * d as f64 / c as f64;
            let dk_mean = if d > 0 { self.band_dk_sum_disagree[k] / d as f64 } else { 0.0 };
            let th_mean = if d > 0 { self.band_theta_sum_disagree[k] / d as f64 } else { 0.0 };
            let margin_abs = self.band_margin_abs_sum[k] / c as f64;
            println!(
                "  {:>4}  {:>6} {:>6} {:>5.1}%  {:>5} {:>5}  {:>8.4} {:>8.4}  {:>8.4}",
                k, c, d, pct,
                self.band_v_to_u[k], self.band_u_to_v[k],
                dk_mean, th_mean, margin_abs
            );
        }
    }
}

/// Bitstream-level conformance statistics: info vectors and dibits.
struct BitstreamStats {
    frames: usize,
    /// Per-info-vector match counts.
    vec_matches: [usize; 8],
    /// Parameter-level: pitch (b₀) matches.
    pitch_matches: usize,
    /// Parameter-level: V/UV (b₁) matches.
    vuv_matches: usize,
    /// Parameter-level: all gain DCT bits (b₂..b₇) match.
    gain_matches: usize,
    /// Dibit-level: total dibits compared.
    dibits_compared: usize,
    /// Dibit-level: dibits that matched.
    dibits_matched: usize,
}

impl Default for BitstreamStats {
    fn default() -> Self {
        Self {
            frames: 0,
            vec_matches: [0; 8],
            pitch_matches: 0,
            vuv_matches: 0,
            gain_matches: 0,
            dibits_compared: 0,
            dibits_matched: 0,
        }
    }
}

impl BitstreamStats {
    fn print(&self) {
        if self.frames == 0 {
            println!("Bitstream comparison: 0 frames");
            return;
        }
        let n = self.frames;
        println!("Bitstream comparison ({n} frames):");
        println!("  parameter bits:");
        println!(
            "    pitch (b̂₀):   {:>5} / {n} ({:.1}%)",
            self.pitch_matches,
            100.0 * self.pitch_matches as f64 / n as f64
        );
        println!(
            "    V/UV (b̂₁):    {:>5} / {n} ({:.1}%)",
            self.vuv_matches,
            100.0 * self.vuv_matches as f64 / n as f64
        );
        println!(
            "    gain (b̂₂₋₇):  {:>5} / {n} ({:.1}%)",
            self.gain_matches,
            100.0 * self.gain_matches as f64 / n as f64
        );
        println!("  info vectors:");
        for i in 0..8 {
            let pct = 100.0 * self.vec_matches[i] as f64 / n as f64;
            println!(
                "    û_{i}  width {:>3}   {:>5} / {n}   {:5.1}%",
                INFO_WIDTHS[i], self.vec_matches[i], pct
            );
        }
        let dpct = 100.0 * self.dibits_matched as f64
            / self.dibits_compared.max(1) as f64;
        println!(
            "  dibits:          {} / {} ({:.1}%)",
            self.dibits_matched, self.dibits_compared, dpct
        );
    }
}

fn cmd_analysis_encode(
    root: &Path,
    name: &str,
    rc: bool,
    dump_frames: usize,
    chip_offset: i32,
    silence_dispatch: bool,
    quantizer_roundtrip: bool,
    chip_seed_vuv: bool,
    bitstream: bool,
) -> Result<()> {
    let dir = vector_dir(root, rc);
    let pcm_path = dir.join("p25").join(format!("{name}.pcm"));
    let fec_path = dir.join("p25").join(format!("{name}.bit"));

    let pcm_bytes = fs::read(&pcm_path)
        .with_context(|| format!("read PCM vector {}", pcm_path.display()))?;
    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;

    if pcm_bytes.len() % (FRAME_SAMPLES * 2) != 0 {
        return Err(anyhow!(
            "PCM file {} is {} bytes — not a whole number of 20 ms frames",
            pcm_path.display(),
            pcm_bytes.len()
        ));
    }
    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "FEC file {} is {} bytes — not a multiple of {}",
            fec_path.display(),
            fec_bytes.len(),
            BYTES_PER_FEC_FRAME
        ));
    }
    let n_pcm_frames = pcm_bytes.len() / (FRAME_SAMPLES * 2);
    let n_fec_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    if n_pcm_frames != n_fec_frames {
        return Err(anyhow!(
            "frame-count mismatch: {} PCM frames vs {} FEC frames",
            n_pcm_frames,
            n_fec_frames
        ));
    }
    let n_frames = n_pcm_frames;

    let pcm: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    println!(
        "vector:            {name}{} (analysis encode)",
        if rc { " (rc)" } else { "" }
    );
    println!("frames:            {n_frames}");
    if chip_offset != 0 {
        println!("chip offset:       {chip_offset:+} (chip[f + offset] vs encoder[f])");
    }
    if silence_dispatch {
        println!("silence dispatch:  on (silent frames tallied, not compared)");
    }
    if quantizer_roundtrip {
        println!("quantizer roundtrip: on (encode → quantize → dequantize decomposition)");
    }
    if chip_seed_vuv {
        println!("chip-seed V/UV:    on (override encoder vuv_prev with chip decisions each frame)");
    }
    if bitstream {
        println!("bitstream:         on (full encode path → dibit/info-vector comparison)");
    }
    println!();

    // Pre-decode every chip bit-frame so we can index by offset.
    // Also store per-frame dibits and info vectors for bitstream comparison.
    let mut chip_state = DecoderState::new();
    let mut chip_dibits_all: Vec<[u8; 72]> = Vec::with_capacity(n_frames);
    let mut chip_info_all: Vec<[u16; 8]> = Vec::with_capacity(n_frames);
    let chip_params: Vec<std::result::Result<MbeParams, DecodeError>> = (0..n_frames)
        .map(|f| {
            let fec = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
            let dibits = unpack_dibits(fec);
            chip_dibits_all.push(dibits);
            let imbe = decode_full_frame(&dibits);
            chip_info_all.push(imbe.info);
            dequantize_full(&imbe.info, &mut chip_state)
        })
        .collect();

    let mut encoder_state = AnalysisState::new();
    if silence_dispatch {
        encoder_state.set_silence_detection(true);
    }
    let mut stats = ConvertStats::default();
    let mut stats_l_match = ConvertStats::default();
    let mut stats_pitch_close = ConvertStats::default();
    let mut stats_l_and_pitch = ConvertStats::default();
    let mut vuv_band_stats = VuvBandStats::default();
    let mut vuv_band_stats_l_match = VuvBandStats::default();
    let mut rt_vs_chip_stats = ConvertStats::default();
    let mut q_loss_stats = ConvertStats::default();
    let mut rt_state = DecoderState::new();
    let mut rt_errors = 0usize;
    let mut bs_state = DecoderState::new();
    let mut bs_stats = BitstreamStats::default();
    let mut skips = AnalysisSkipStats::default();
    let mut dumped = 0usize;

    if dump_frames > 0 {
        println!(
            "{:>5}  {:>9} {:>9} {:>8}   {:>3} {:>3} {:>4}   \
             {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}   {:<16}  {}",
            "frame", "ω̂_0 chip", "ω̂_0 enc", "Δ cents",
            "Lc", "Le", "ΔL",
            "P̂_I", "P_chip", "E(P̂_B)", "E(P̂_F)", "E(P̂_I)", "E(P_ch)",
            "voicing chip", "voicing enc"
        );
    }

    for f in 0..n_frames {
        let pcm_frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let encoded = analysis_encode_with_trace(pcm_frame, &mut encoder_state);

        let chip_idx = f as i32 + chip_offset;
        let chip_ref: Option<&std::result::Result<MbeParams, DecodeError>> =
            if chip_idx >= 0 && (chip_idx as usize) < n_frames {
                Some(&chip_params[chip_idx as usize])
            } else {
                None
            };

        match (encoded, chip_ref) {
            (Ok((AnalysisOutput::Voice(enc_params), trace)), Some(Ok(ref_params))) => {
                if dumped < dump_frames {
                    let wc = f64::from(ref_params.omega_0());
                    let we = f64::from(enc_params.omega_0());
                    let cents = 1200.0 * (we / wc).log2();
                    let lc = ref_params.harmonic_count() as i32;
                    let le = enc_params.harmonic_count() as i32;
                    let tr = trace.as_ref().expect("voice frame always carries trace");
                    let p_chip = 2.0 * std::f64::consts::PI / wc;
                    let e_pb = tr.pitch_search_current.e_of_p(tr.p_hat_b);
                    let e_pf = tr.pitch_search_current.e_of_p(tr.p_hat_f);
                    let e_pchip = tr.pitch_search_current.e_of_p(p_chip);
                    println!(
                        "{:>5}  {:>9.5} {:>9.5} {:>+8.1}   {:>3} {:>3} {:>+4}   \
                         {:>6.2} {:>6.2} {:>8.4} {:>8.4} {:>8.4} {:>8.4}   {:<16}  {}",
                        f,
                        wc,
                        we,
                        cents,
                        lc,
                        le,
                        le - lc,
                        tr.p_hat_i,
                        p_chip,
                        e_pb,
                        e_pf,
                        tr.e_p_hat_i,
                        e_pchip,
                        voicing_sig(ref_params, 16),
                        voicing_sig(&enc_params, 16),
                    );
                    dumped += 1;
                }
                stats.push(ref_params, &enc_params);

                let l_match = ref_params.harmonic_count() == enc_params.harmonic_count();
                let cents_abs = (1200.0
                    * (f64::from(enc_params.omega_0()) / f64::from(ref_params.omega_0())).log2())
                .abs();
                let pitch_close = cents_abs < 50.0;
                if l_match {
                    stats_l_match.push(ref_params, &enc_params);
                }
                if pitch_close {
                    stats_pitch_close.push(ref_params, &enc_params);
                }
                if l_match && pitch_close {
                    stats_l_and_pitch.push(ref_params, &enc_params);
                }

                if let Some(ref vuv) = trace.as_ref().and_then(|t| t.vuv_result.as_ref()) {
                    vuv_band_stats.push(ref_params, &enc_params, vuv);
                    if l_match {
                        vuv_band_stats_l_match.push(ref_params, &enc_params, vuv);
                    }
                }

                if quantizer_roundtrip {
                    let mut q_state = rt_state.clone();
                    match quantize_full(&enc_params, &mut q_state) {
                        Ok(bits) => {
                            let l = enc_params.harmonic_count();
                            let info = prioritize_full(&bits, l);
                            match dequantize_full(&info, &mut rt_state) {
                                Ok(rt_params) => {
                                    rt_vs_chip_stats.push(ref_params, &rt_params);
                                    q_loss_stats.push(&enc_params, &rt_params);
                                }
                                Err(_) => { rt_errors += 1; }
                            }
                        }
                        Err(_) => { rt_errors += 1; }
                    }
                }

                if bitstream {
                    let ci = chip_idx as usize;
                    let mut q_state = bs_state.clone();
                    if let Ok(bits) = quantize_full(&enc_params, &mut q_state) {
                        let l = enc_params.harmonic_count();
                        let enc_info = prioritize_full(&bits, l);
                        let enc_dibits = encode_full_frame(&enc_info);
                        let chip_info = &chip_info_all[ci];
                        let chip_dibits = &chip_dibits_all[ci];

                        bs_stats.frames += 1;
                        for i in 0..8 {
                            if enc_info[i] == chip_info[i] {
                                bs_stats.vec_matches[i] += 1;
                            }
                        }
                        // Parameter-level: deprioritize both to compare
                        // at the b̂ parameter level.
                        let b_enc = deprioritize_full(&enc_info, l);
                        // Chip's L may differ; use chip's L for its own
                        // deprioritize (pitch index determines L).
                        let chip_l = ref_params.harmonic_count();
                        let b_chip = deprioritize_full(chip_info, chip_l);
                        if b_enc[0] == b_chip[0] {
                            bs_stats.pitch_matches += 1;
                        }
                        if b_enc[1] == b_chip[1] {
                            bs_stats.vuv_matches += 1;
                        }
                        let gain_ok = (2..8).all(|i| b_enc[i] == b_chip[i]);
                        if gain_ok {
                            bs_stats.gain_matches += 1;
                        }
                        for i in 0..72 {
                            bs_stats.dibits_compared += 1;
                            if enc_dibits[i] == chip_dibits[i] {
                                bs_stats.dibits_matched += 1;
                            }
                        }

                        // Advance bs_state with reconstructed amplitudes
                        // (matching encoder's internal predictor).
                        let _ = dequantize_full(&enc_info, &mut bs_state);
                    }
                }
            }
            (Ok((AnalysisOutput::Silence, _)), _) => {
                if f < blip25_mbe::codecs::mbe_baseline::analysis::PREROLL_FRAMES as usize {
                    skips.preroll += 1;
                } else {
                    skips.encoder_silence += 1;
                }
            }
            (Ok(_), Some(Err(_))) => {
                skips.chip_decode_errors += 1;
            }
            (Ok(_), None) => {
                skips.chip_out_of_range += 1;
            }
            (Err(AnalysisError::WrongFrameLength), _) => {
                return Err(anyhow!(
                    "encoder rejected frame {f} as wrong length (bug in harness)"
                ));
            }
            (Err(_), _) => {
                skips.encoder_errors += 1;
            }
        }

        // Chip-seed V/UV: override the encoder's vuv_prev with the
        // chip's band-level voicing for the aligned frame, so the
        // next encoder frame uses the chip's V/UV history instead of
        // its own. This breaks the self-reinforcing feedback loop.
        if chip_seed_vuv {
            if let Some(Ok(ref_params)) = chip_ref {
                let l_chip = ref_params.harmonic_count();
                let k_chip = band_count_for(l_chip);
                let chip_v = ref_params.voiced_slice();
                // Extract per-band decisions from chip's per-harmonic
                // voicing: sample the first harmonic in each band.
                let mut band_decisions = [0u8; 13];
                for k in 1..=k_chip {
                    let first_l = 3 * u32::from(k) - 2;
                    if (first_l as u8) <= l_chip {
                        band_decisions[k as usize] =
                            if chip_v[(first_l - 1) as usize] { 1 } else { 0 };
                    }
                }
                encoder_state.vuv_state_mut()
                    .override_vuv_prev(&band_decisions, k_chip);
            }
        }
    }

    if dump_frames > 0 {
        println!();
    }
    skips.print();
    println!();
    stats.print("All voice frames (pre-quantizer)");
    println!();
    stats_l_match.print("L-matching frames only (enc L == chip L)");
    println!();
    stats_pitch_close.print("Pitch-close frames only (|Δ| < 50 cents)");
    println!();
    stats_l_and_pitch.print("L-matching AND pitch-close frames");
    println!();
    vuv_band_stats.print();
    if vuv_band_stats_l_match.frames > 0 {
        println!();
        println!("V/UV per-band (L-matching frames only, {} frames):", vuv_band_stats_l_match.frames);
        vuv_band_stats_l_match.print();
    }
    if quantizer_roundtrip {
        println!();
        rt_vs_chip_stats.print("Roundtrip vs chip (post-quantizer, apples-to-apples)");
        println!();
        q_loss_stats.print("Quantizer distortion (encoder pre-Q vs own roundtrip)");
        if rt_errors > 0 {
            println!("  roundtrip errors:  {rt_errors}");
        }
    }
    if bitstream {
        println!();
        bs_stats.print();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// P25 normative-vector sweep (cmd: p25-sweep)
// ---------------------------------------------------------------------------

/// FEC config word that enables on-the-wire FEC per the
/// AMBE-3000_Test_Vector_Reference. When this flag appears at position
/// `-r` index 2 (third rate word), lines target `p25/<name>.bit`;
/// otherwise (`0x0000`) they target `p25_nofec/<name>.bit`.
const FEC_ENABLE_WORD: &str = "0x1030";

/// One parsed line from `cmpp25.txt`.
#[derive(Clone, Debug)]
struct TestLine {
    /// 1-based line number in the source file (for error reporting).
    line_no: usize,
    /// Operation: `-dec`, `-enc`, or `-encdec`.
    op: SweepOp,
    /// Whether FEC is enabled (determines target subdir).
    fec: bool,
    /// Input file path relative to the cmp file's directory.
    input: String,
    /// Reference file path relative to the cmp file's directory.
    reference: String,
    /// Soft-decision bit width — only set for `-dec -sdbits N` lines,
    /// which we skip (feature not implemented).
    sdbits: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SweepOp {
    Dec,
    Enc,
    EncDec,
}

impl SweepOp {
    fn as_str(&self) -> &'static str {
        match self {
            SweepOp::Dec => "dec",
            SweepOp::Enc => "enc",
            SweepOp::EncDec => "encdec",
        }
    }
}

/// Content-category buckets used for per-category aggregate reporting.
/// The category is the alphabetic-prefix of the input stem, e.g.
/// `clean`, `dam`, `dtmf`, `dtone`, `sine`, `t`, `tambf`, `tia`,
/// `ucarf`, `ucarm`, `xfer`, `zero`. For paths like `p25/clean.bit`
/// (which appear as the input on `-dec` lines), the leading directory
/// is stripped before computing the prefix.
fn content_category(path: &str) -> String {
    // Strip directory prefix (`p25/`, `p25_nofec/`) so `-dec` inputs
    // bucket by speech name, not by output dir.
    let base = path.rsplit('/').next().unwrap_or(path);
    // Strip the `.pcm` / `.bit` suffix.
    let stem = base.rsplit_once('.').map(|(s, _)| s).unwrap_or(base);
    // Category = longest leading alphabetic run. `_` and digits end
    // the run: `dam_e1_hd` → `dam`, `dtone_1` → `dtone`, `sine0_1k` →
    // `sine`. Empty prefix (no leading alpha) → fall back to full stem.
    let prefix: String = stem
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if prefix.is_empty() { stem.to_string() } else { prefix }
}

fn parse_cmpp25_line(line: &str, line_no: usize) -> Result<Option<TestLine>> {
    // Strip trailing / leading whitespace. Skip blank lines and comments.
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.is_empty() {
        return Ok(None);
    }
    // Expected form:
    //   -c P25 -q [-sdbits N] -{enc,dec,encdec} -r W0 W1 W2 W3 W4 W5 <input> -cmp <ref>
    // Required anchors: `-c P25`, an op in {-enc, -dec, -encdec},
    // `-r` followed by six hex rate words, and `-cmp <ref>` at the end.
    let mut op: Option<SweepOp> = None;
    let mut sdbits: Option<u8> = None;
    let mut rate_words: Vec<&str> = Vec::new();
    let mut input: Option<String> = None;
    let mut reference: Option<String> = None;
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        match t {
            "-c" => {
                if toks.get(i + 1) != Some(&"P25") {
                    return Err(anyhow!("line {line_no}: non-P25 codec `{:?}`", toks.get(i + 1)));
                }
                i += 2;
            }
            "-q" => {
                i += 1;
            }
            "-enc" => {
                op = Some(SweepOp::Enc);
                i += 1;
            }
            "-dec" => {
                op = Some(SweepOp::Dec);
                i += 1;
            }
            "-encdec" => {
                op = Some(SweepOp::EncDec);
                i += 1;
            }
            "-sdbits" => {
                let n = toks.get(i + 1).ok_or_else(|| {
                    anyhow!("line {line_no}: -sdbits missing value")
                })?;
                sdbits = Some(
                    n.parse::<u8>()
                        .map_err(|e| anyhow!("line {line_no}: bad sdbits `{n}`: {e}"))?,
                );
                i += 2;
            }
            "-r" => {
                // Six rate words follow.
                for j in 1..=6 {
                    let w = toks.get(i + j).ok_or_else(|| {
                        anyhow!("line {line_no}: -r needs 6 words, only {} present", j - 1)
                    })?;
                    rate_words.push(w);
                }
                i += 7;
            }
            "-cmp" => {
                reference = Some(
                    toks.get(i + 1)
                        .ok_or_else(|| anyhow!("line {line_no}: -cmp missing value"))?
                        .to_string(),
                );
                i += 2;
            }
            other => {
                // Positional input path (first bare token after all flags).
                if input.is_none() {
                    input = Some(other.to_string());
                } else {
                    return Err(anyhow!(
                        "line {line_no}: unexpected extra positional token `{other}`"
                    ));
                }
                i += 1;
            }
        }
    }
    let op = op.ok_or_else(|| anyhow!("line {line_no}: missing operation flag"))?;
    let input = input.ok_or_else(|| anyhow!("line {line_no}: missing input filename"))?;
    let reference = reference.ok_or_else(|| anyhow!("line {line_no}: missing -cmp target"))?;
    if rate_words.len() != 6 {
        return Err(anyhow!(
            "line {line_no}: expected 6 -r words, got {}",
            rate_words.len()
        ));
    }
    // FEC enable word lives at rate_words[2] per the Test_Vector_Reference §4.3.
    let fec = rate_words[2] == FEC_ENABLE_WORD;
    Ok(Some(TestLine {
        line_no,
        op,
        fec,
        input,
        reference,
        sdbits,
    }))
}

/// Rolling accumulator for one (op, fec_mode) stratum of the sweep.
#[derive(Default)]
struct SweepStratum {
    lines: usize,
    passed: usize,
    skipped_sdbits: usize,
    errored: usize,
    // Decode/round-trip path metrics — PCM sample stats.
    pcm_lines: usize,
    pcm_total_samples: u64,
    pcm_bit_exact: u64,
    pcm_sum_sq_err: f64,
    pcm_sum_sq_ref: f64,
    pcm_peak_err: f64,
    // Encode path metrics — bit stats.
    bit_lines: usize,
    bit_total_frames: u64,
    bit_exact_frames: u64,
    bit_byte_exact: u64,
    bit_total_bytes: u64,
    // Decoder bad-pitch frames (silenced by our decoder).
    bad_pitch_frames: u64,
}

impl SweepStratum {
    fn record_pcm(
        &mut self,
        total: u64,
        exact: u64,
        sum_sq_err: f64,
        sum_sq_ref: f64,
        peak_err: f64,
        bad_pitch: u64,
    ) {
        self.pcm_lines += 1;
        self.pcm_total_samples += total;
        self.pcm_bit_exact += exact;
        self.pcm_sum_sq_err += sum_sq_err;
        self.pcm_sum_sq_ref += sum_sq_ref;
        self.pcm_peak_err = self.pcm_peak_err.max(peak_err);
        self.bad_pitch_frames += bad_pitch;
    }

    fn record_bits(
        &mut self,
        total_frames: u64,
        exact_frames: u64,
        total_bytes: u64,
        exact_bytes: u64,
    ) {
        self.bit_lines += 1;
        self.bit_total_frames += total_frames;
        self.bit_exact_frames += exact_frames;
        self.bit_total_bytes += total_bytes;
        self.bit_byte_exact += exact_bytes;
    }

    fn pcm_snr_db(&self) -> f64 {
        if self.pcm_sum_sq_err > 0.0 && self.pcm_sum_sq_ref > 0.0 {
            10.0 * (self.pcm_sum_sq_ref / self.pcm_sum_sq_err).log10()
        } else if self.pcm_sum_sq_err == 0.0 && self.pcm_total_samples > 0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }

    fn pcm_rms(&self) -> f64 {
        if self.pcm_total_samples > 0 {
            (self.pcm_sum_sq_err / self.pcm_total_samples as f64).sqrt()
        } else {
            f64::NAN
        }
    }

    fn pcm_bit_exact_pct(&self) -> f64 {
        if self.pcm_total_samples > 0 {
            100.0 * self.pcm_bit_exact as f64 / self.pcm_total_samples as f64
        } else {
            f64::NAN
        }
    }

    fn bit_frame_pct(&self) -> f64 {
        if self.bit_total_frames > 0 {
            100.0 * self.bit_exact_frames as f64 / self.bit_total_frames as f64
        } else {
            f64::NAN
        }
    }

    fn bit_byte_pct(&self) -> f64 {
        if self.bit_total_bytes > 0 {
            100.0 * self.bit_byte_exact as f64 / self.bit_total_bytes as f64
        } else {
            f64::NAN
        }
    }
}

/// Outcome of running a single cmpp25 line.
#[derive(Clone, Copy, Debug)]
enum LineOutcome {
    SkippedSdbits,
    Pcm {
        total: u64,
        exact: u64,
        sum_sq_err: f64,
        sum_sq_ref: f64,
        peak_err: f64,
        bad_pitch: u64,
    },
    Bits {
        total_frames: u64,
        exact_frames: u64,
        total_bytes: u64,
        exact_bytes: u64,
    },
}

fn run_line(cmp_dir: &Path, line: &TestLine) -> Result<LineOutcome> {
    if line.sdbits.is_some() {
        return Ok(LineOutcome::SkippedSdbits);
    }
    match line.op {
        SweepOp::Dec => run_dec(cmp_dir, line),
        SweepOp::Enc => run_enc(cmp_dir, line),
        SweepOp::EncDec => run_encdec(cmp_dir, line),
    }
}

fn run_dec(cmp_dir: &Path, line: &TestLine) -> Result<LineOutcome> {
    let input_path = cmp_dir.join(&line.input);
    let ref_path = cmp_dir.join(&line.reference);
    let bit_bytes = fs::read(&input_path)
        .with_context(|| format!("read input {}", input_path.display()))?;
    let ref_pcm_bytes = fs::read(&ref_path)
        .with_context(|| format!("read reference {}", ref_path.display()))?;

    let frame_bytes = if line.fec { BYTES_PER_FEC_FRAME } else { BYTES_PER_NOFEC_FRAME };
    if bit_bytes.len() % frame_bytes != 0 {
        return Err(anyhow!(
            "{}: {} bytes not a multiple of {}",
            input_path.display(),
            bit_bytes.len(),
            frame_bytes
        ));
    }
    if ref_pcm_bytes.len() % 2 != 0 {
        return Err(anyhow!("reference PCM not whole number of i16 samples"));
    }
    let n_frames = bit_bytes.len() / frame_bytes;
    let ref_pcm: Vec<i16> = ref_pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let (pcm_out, bad_pitch) = decode_to_pcm(&bit_bytes, line.fec, n_frames)?;
    Ok(summarize_pcm(&pcm_out, &ref_pcm, bad_pitch))
}

fn run_enc(cmp_dir: &Path, line: &TestLine) -> Result<LineOutcome> {
    let input_path = cmp_dir.join(&line.input);
    let ref_path = cmp_dir.join(&line.reference);
    let pcm_bytes = fs::read(&input_path)
        .with_context(|| format!("read input {}", input_path.display()))?;
    let ref_bit_bytes = fs::read(&ref_path)
        .with_context(|| format!("read reference {}", ref_path.display()))?;

    let frame_bytes = if line.fec { BYTES_PER_FEC_FRAME } else { BYTES_PER_NOFEC_FRAME };
    if ref_bit_bytes.len() % frame_bytes != 0 {
        return Err(anyhow!("reference .bit not a multiple of {}", frame_bytes));
    }
    let n_ref_frames = ref_bit_bytes.len() / frame_bytes;
    let pcm = load_pcm_padded_to_frames(&pcm_bytes, n_ref_frames);

    let our_bits = encode_pcm_to_bits(&pcm, n_ref_frames, line.fec)?;
    Ok(summarize_bits(&our_bits, &ref_bit_bytes, frame_bytes))
}

fn run_encdec(cmp_dir: &Path, line: &TestLine) -> Result<LineOutcome> {
    let input_path = cmp_dir.join(&line.input);
    let ref_path = cmp_dir.join(&line.reference);
    let pcm_bytes = fs::read(&input_path)
        .with_context(|| format!("read input {}", input_path.display()))?;
    let ref_pcm_bytes = fs::read(&ref_path)
        .with_context(|| format!("read reference {}", ref_path.display()))?;

    if ref_pcm_bytes.len() % (FRAME_SAMPLES * 2) != 0 {
        return Err(anyhow!("reference PCM not whole number of 20 ms frames"));
    }
    let n_frames = ref_pcm_bytes.len() / (FRAME_SAMPLES * 2);
    let pcm = load_pcm_padded_to_frames(&pcm_bytes, n_frames);
    let ref_pcm: Vec<i16> = ref_pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // PCM → bits → PCM.
    let our_bits = encode_pcm_to_bits(&pcm, n_frames, line.fec)?;
    let (pcm_out, bad_pitch) = decode_to_pcm(&our_bits, line.fec, n_frames)?;
    Ok(summarize_pcm(&pcm_out, &ref_pcm, bad_pitch))
}

/// Load an input `.pcm` file as `Vec<i16>`, zero-padding or truncating
/// to exactly `n_frames * FRAME_SAMPLES` samples. DVSI input PCMs are
/// often not whole-frame lengths (the final frame is zero-padded by
/// the encoder); reference outputs are always frame-aligned.
fn load_pcm_padded_to_frames(pcm_bytes: &[u8], n_frames: usize) -> Vec<i16> {
    let target = n_frames * FRAME_SAMPLES;
    let mut pcm: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    pcm.resize(target, 0);
    pcm
}

/// Decode `bit_bytes` (FEC or no-FEC) to `i16` PCM samples, returning
/// the output and the count of frames replaced with silence because
/// dequantize rejected the pitch index.
fn decode_to_pcm(
    bit_bytes: &[u8],
    fec: bool,
    n_frames: usize,
) -> Result<(Vec<i16>, u64)> {
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    let mut pcm_out: Vec<i16> = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    let mut bad_pitch = 0u64;

    for f in 0..n_frames {
        let (info, err) = if fec {
            let fec_frame = &bit_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
            let dibits = unpack_dibits(fec_frame);
            let imbe = decode_full_frame(&dibits);
            let err = FrameErrorContext {
                epsilon_0: imbe.errors[0],
                epsilon_4: imbe.errors[4],
                epsilon_t: imbe.error_total().min(255) as u8,
                bad_pitch: false,
            };
            (imbe.info, err)
        } else {
            let nofec_frame =
                &bit_bytes[f * BYTES_PER_NOFEC_FRAME..(f + 1) * BYTES_PER_NOFEC_FRAME];
            let info = unpack_nofec_info(nofec_frame);
            // No-FEC frames have no error context (nothing to correct).
            let err = FrameErrorContext {
                epsilon_0: 0,
                epsilon_4: 0,
                epsilon_t: 0,
                bad_pitch: false,
            };
            (info, err)
        };

        let params = match dequantize_full(&info, &mut decoder_state) {
            Ok(p) => p,
            Err(_) => {
                bad_pitch += 1;
                pcm_out.extend(std::iter::repeat(0i16).take(FRAME_SAMPLES));
                continue;
            }
        };
        let pcm = synthesize_frame(&params, &err, GAMMA_W, &mut synth_state);
        pcm_out.extend_from_slice(&pcm);
    }
    Ok((pcm_out, bad_pitch))
}

/// Encode PCM through `AnalysisState::encode` + quantize + (optional
/// FEC) into a contiguous byte buffer matching the DVSI `.bit` frame
/// format. Frames where analysis returns Silence (including the
/// preroll frames 0 and 1) are emitted as the `MbeParams::silence()`
/// frame so that byte counts stay aligned with the reference.
fn encode_pcm_to_bits(pcm: &[i16], n_frames: usize, fec: bool) -> Result<Vec<u8>> {
    let frame_bytes = if fec { BYTES_PER_FEC_FRAME } else { BYTES_PER_NOFEC_FRAME };
    let mut out = Vec::with_capacity(n_frames * frame_bytes);
    let mut enc_state = AnalysisState::new();
    let mut quant_state = DecoderState::new();

    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let params = match analysis_encode_with_trace(frame, &mut enc_state) {
            Ok((AnalysisOutput::Voice(p), _)) => p,
            Ok((AnalysisOutput::Silence, _)) => MbeParams::silence(),
            Err(_) => MbeParams::silence(),
        };
        let b = match quantize_full(&params, &mut quant_state) {
            Ok(b) => b,
            Err(_) => {
                // Quantizer rejected the frame — emit zero bits so the
                // buffer stays aligned. Chip-side reference will also
                // have encoded something for this frame, so we count
                // this as a mismatch downstream.
                out.extend(std::iter::repeat(0u8).take(frame_bytes));
                continue;
            }
        };
        let u = prioritize_full(&b, params.harmonic_count());
        if fec {
            let dibits = encode_full_frame(&u);
            // `encode_full_frame` returns 72 dibits as `[u8; 72]`;
            // re-pack into 18 bytes in transmission order.
            let packed = pack_dibits(&dibits);
            out.extend_from_slice(&packed);
        } else {
            let packed = pack_nofec_info(&u);
            out.extend_from_slice(&packed);
        }
    }
    Ok(out)
}

fn summarize_pcm(our: &[i16], reference: &[i16], bad_pitch: u64) -> LineOutcome {
    let n = our.len().min(reference.len());
    let mut exact = 0u64;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut peak_err = 0.0f64;
    for i in 0..n {
        let a = f64::from(our[i]);
        let b = f64::from(reference[i]);
        let e = a - b;
        sum_sq_err += e * e;
        sum_sq_ref += b * b;
        peak_err = peak_err.max(e.abs());
        if our[i] == reference[i] {
            exact += 1;
        }
    }
    LineOutcome::Pcm {
        total: n as u64,
        exact,
        sum_sq_err,
        sum_sq_ref,
        peak_err,
        bad_pitch,
    }
}

fn summarize_bits(our: &[u8], reference: &[u8], frame_bytes: usize) -> LineOutcome {
    let n = our.len().min(reference.len());
    let n_frames = n / frame_bytes;
    let mut exact_bytes = 0u64;
    for i in 0..n {
        if our[i] == reference[i] {
            exact_bytes += 1;
        }
    }
    let mut exact_frames = 0u64;
    for f in 0..n_frames {
        let start = f * frame_bytes;
        let end = start + frame_bytes;
        if our[start..end] == reference[start..end] {
            exact_frames += 1;
        }
    }
    LineOutcome::Bits {
        total_frames: n_frames as u64,
        exact_frames,
        total_bytes: n as u64,
        exact_bytes,
    }
}

fn cmd_p25_sweep(
    root: &Path,
    cmp_path: Option<&Path>,
    filter: Option<&str>,
    op_filter: Option<&str>,
    stop_on_first: bool,
) -> Result<()> {
    let cmp_path_owned;
    let cmp_path: &Path = match cmp_path {
        Some(p) => p,
        None => {
            cmp_path_owned = root.join("tv-std").join("tv").join("cmpp25.txt");
            &cmp_path_owned
        }
    };
    let cmp_dir = cmp_path
        .parent()
        .ok_or_else(|| anyhow!("cmp file has no parent directory"))?;

    let contents = fs::read_to_string(cmp_path)
        .with_context(|| format!("read {}", cmp_path.display()))?;

    // Parse every line.
    let mut lines: Vec<TestLine> = Vec::new();
    for (idx, raw) in contents.lines().enumerate() {
        let line_no = idx + 1;
        if let Some(tl) = parse_cmpp25_line(raw, line_no)? {
            lines.push(tl);
        }
    }

    // Apply filters.
    let op_filter = op_filter.map(|s| s.to_ascii_lowercase());
    let lines: Vec<TestLine> = lines
        .into_iter()
        .filter(|tl| {
            filter.is_none_or(|f| tl.input.contains(f))
                && op_filter
                    .as_deref()
                    .is_none_or(|o| tl.op.as_str() == o)
        })
        .collect();

    println!("cmp file:    {}", cmp_path.display());
    println!("lines:       {} (after filters)", lines.len());
    if let Some(f) = filter {
        println!("filter:      --filter {f}");
    }
    if let Some(o) = op_filter.as_deref() {
        println!("op filter:   --op {o}");
    }
    println!();

    // Stratified stats by (op, fec_mode) and by (op, fec_mode, category).
    use std::collections::BTreeMap;
    let mut strata: BTreeMap<(SweepOp, bool), SweepStratum> = BTreeMap::new();
    let mut cat_strata: BTreeMap<(SweepOp, bool, String), SweepStratum> = BTreeMap::new();

    for tl in &lines {
        let key = (tl.op, tl.fec);
        let cat = content_category(&tl.input);
        let cat_key = (tl.op, tl.fec, cat);
        let stratum = strata.entry(key).or_default();
        let cat_stratum = cat_strata.entry(cat_key).or_default();
        stratum.lines += 1;
        cat_stratum.lines += 1;

        match run_line(cmp_dir, tl) {
            Ok(LineOutcome::SkippedSdbits) => {
                stratum.skipped_sdbits += 1;
                cat_stratum.skipped_sdbits += 1;
            }
            Ok(LineOutcome::Pcm {
                total,
                exact,
                sum_sq_err,
                sum_sq_ref,
                peak_err,
                bad_pitch,
            }) => {
                stratum.passed += 1;
                cat_stratum.passed += 1;
                stratum.record_pcm(total, exact, sum_sq_err, sum_sq_ref, peak_err, bad_pitch);
                cat_stratum.record_pcm(total, exact, sum_sq_err, sum_sq_ref, peak_err, bad_pitch);
            }
            Ok(LineOutcome::Bits {
                total_frames,
                exact_frames,
                total_bytes,
                exact_bytes,
            }) => {
                stratum.passed += 1;
                cat_stratum.passed += 1;
                stratum.record_bits(total_frames, exact_frames, total_bytes, exact_bytes);
                cat_stratum.record_bits(total_frames, exact_frames, total_bytes, exact_bytes);
            }
            Err(e) => {
                stratum.errored += 1;
                cat_stratum.errored += 1;
                eprintln!(
                    "line {}: {} {} -cmp {} — ERROR: {e:#}",
                    tl.line_no,
                    tl.op.as_str(),
                    tl.input,
                    tl.reference
                );
                if stop_on_first {
                    return Err(e);
                }
            }
        }
    }

    // Report by (op, fec).
    println!("=== Per-op / per-fec aggregate ===");
    println!(
        "{:<8} {:<6} {:>6} {:>6} {:>6} {:>6}  {}",
        "op", "fec", "lines", "pass", "err", "skip", "metrics"
    );
    for ((op, fec), s) in &strata {
        let metrics = stratum_metrics_line(*op, s);
        println!(
            "{:<8} {:<6} {:>6} {:>6} {:>6} {:>6}  {metrics}",
            op.as_str(),
            if *fec { "FEC" } else { "nofec" },
            s.lines,
            s.passed,
            s.errored,
            s.skipped_sdbits,
        );
    }

    // Per-category breakdown (sorted by category name within each op/fec).
    println!();
    println!("=== Per-content-category breakdown ===");
    println!(
        "{:<8} {:<6} {:<10} {:>5}  {}",
        "op", "fec", "category", "lines", "metrics"
    );
    for ((op, fec, cat), s) in &cat_strata {
        let metrics = stratum_metrics_line(*op, s);
        println!(
            "{:<8} {:<6} {:<10} {:>5}  {metrics}",
            op.as_str(),
            if *fec { "FEC" } else { "nofec" },
            cat,
            s.lines,
        );
    }

    // Overall result — pass if every non-skipped line ran without errors.
    let total_err: usize = strata.values().map(|s| s.errored).sum();
    if total_err > 0 {
        return Err(anyhow!("{total_err} line(s) errored"));
    }
    Ok(())
}

fn stratum_metrics_line(op: SweepOp, s: &SweepStratum) -> String {
    match op {
        SweepOp::Dec | SweepOp::EncDec => {
            if s.pcm_lines == 0 {
                return String::from("(no PCM runs)");
            }
            format!(
                "bit-exact {:5.2}%  RMS {:7.2}  peak {:>5.0}  SNR {:>6.2} dB  bad-pitch frames {}",
                s.pcm_bit_exact_pct(),
                s.pcm_rms(),
                s.pcm_peak_err,
                s.pcm_snr_db(),
                s.bad_pitch_frames,
            )
        }
        SweepOp::Enc => {
            if s.bit_lines == 0 {
                return String::from("(no bit runs)");
            }
            format!(
                "frames-exact {:5.2}%  bytes-exact {:5.2}%  ({}/{} frames, {}/{} bytes)",
                s.bit_frame_pct(),
                s.bit_byte_pct(),
                s.bit_exact_frames,
                s.bit_total_frames,
                s.bit_byte_exact,
                s.bit_total_bytes,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_msb_first() {
        // 0b10110100 → reads 4 bits as 0b1011 = 11.
        let bytes = [0b1011_0100];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(4), 0b1011);
        assert_eq!(r.read(4), 0b0100);
    }

    #[test]
    fn bit_reader_crosses_byte_boundary() {
        // 0b1111_0000 0b1111_0000 → read 12 bits = 0b1111_0000_1111
        let bytes = [0b1111_0000, 0b1111_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(12), 0b1111_0000_1111);
    }

    #[test]
    fn unpack_dibits_first_byte() {
        // Byte 0xE4 = 0b11_10_01_00 → dibits 3, 2, 1, 0
        let bytes = [0u8; BYTES_PER_FEC_FRAME];
        let mut bytes = bytes;
        bytes[0] = 0b11_10_01_00;
        let d = unpack_dibits(&bytes);
        assert_eq!(d[0], 3);
        assert_eq!(d[1], 2);
        assert_eq!(d[2], 1);
        assert_eq!(d[3], 0);
    }

    #[test]
    fn unpack_nofec_widths_match_info_widths() {
        // Packing is sequential per INFO_WIDTHS = [12,12,12,12,11,11,11,7]
        // = 88 bits = 11 bytes.
        let total: u16 = INFO_WIDTHS.iter().map(|&w| u16::from(w)).sum();
        assert_eq!(total, 88);
        assert_eq!(BYTES_PER_NOFEC_FRAME, 11);
    }

    #[test]
    fn unpack_nofec_first_vector() {
        // First 12 bits = 0xABC means û₀ = 0xABC.
        // 0xABC = 1010_1011_1100 in 12 bits.
        // Packed MSB-first: byte 0 = 1010_1011, byte 1 high nibble = 1100.
        let mut bytes = [0u8; BYTES_PER_NOFEC_FRAME];
        bytes[0] = 0b1010_1011;
        bytes[1] = 0b1100_0000;
        let info = unpack_nofec_info(&bytes);
        assert_eq!(info[0], 0xABC);
    }

    #[test]
    fn bit_writer_roundtrips_with_bit_reader() {
        let mut bytes = [0u8; 4];
        let mut w = BitWriter::new(&mut bytes);
        w.write(0b1011, 4);
        w.write(0b0100, 4);
        w.write(0b1111_0000_1111, 12);
        drop(w);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(4), 0b1011);
        assert_eq!(r.read(4), 0b0100);
        assert_eq!(r.read(12), 0b1111_0000_1111);
    }

    #[test]
    fn pack_dibits_is_inverse_of_unpack() {
        // Patterned input: dibit i = i % 4 → 0,1,2,3,0,1,...
        let mut src = [0u8; DIBITS_PER_FRAME];
        for (i, s) in src.iter_mut().enumerate() {
            *s = (i as u8) % 4;
        }
        let packed = pack_dibits(&src);
        let back = unpack_dibits(&packed);
        assert_eq!(back, src);
    }

    #[test]
    fn pack_nofec_info_is_inverse_of_unpack() {
        // Values that fill their widths: 0xABC, 0x555, ..., 0x7F.
        let src: [u16; 8] = [0xABC, 0x555, 0xFFF, 0x123, 0x7AB, 0x456, 0x1A5, 0x7F];
        let packed = pack_nofec_info(&src);
        let back = unpack_nofec_info(&packed);
        assert_eq!(back, src);
    }

    #[test]
    fn content_category_strips_digits_and_dots() {
        assert_eq!(content_category("alert.pcm"), "alert");
        assert_eq!(content_category("clean.pcm"), "clean");
        assert_eq!(content_category("cp31.pcm"), "cp");
        assert_eq!(content_category("dam_10ov.pcm"), "dam");
        assert_eq!(content_category("dtmf15n.pcm"), "dtmf");
        assert_eq!(content_category("dtone_1.pcm"), "dtone");
        assert_eq!(content_category("sine0_1k.pcm"), "sine");
        assert_eq!(content_category("t01.pcm"), "t");
        assert_eq!(content_category("tambf22a.pcm"), "tambf");
        assert_eq!(content_category("ucarf15a.pcm"), "ucarf");
        assert_eq!(content_category("zero.pcm"), "zero");
    }

    #[test]
    fn content_category_strips_leading_directory() {
        assert_eq!(content_category("p25/clean.bit"), "clean");
        assert_eq!(content_category("p25_nofec/alert.bit"), "alert");
        assert_eq!(content_category("p25/dam_e1_hd.bit"), "dam");
    }

    #[test]
    fn parse_cmpp25_encdec_fec_line() {
        let line = "-c P25 -q -encdec -r 0x0558 0x086b 0x1030 0x0000 0x0000 0x0190 alert.pcm -cmp p25/alert.pcm";
        let tl = parse_cmpp25_line(line, 1).unwrap().unwrap();
        assert_eq!(tl.op, SweepOp::EncDec);
        assert!(tl.fec);
        assert_eq!(tl.input, "alert.pcm");
        assert_eq!(tl.reference, "p25/alert.pcm");
        assert!(tl.sdbits.is_none());
    }

    #[test]
    fn parse_cmpp25_enc_nofec_line() {
        let line = "-c P25 -q -enc -r 0x0558 0x086b 0x0000 0x0000 0x0000 0x0158 clean.pcm -cmp p25_nofec/clean.bit";
        let tl = parse_cmpp25_line(line, 1).unwrap().unwrap();
        assert_eq!(tl.op, SweepOp::Enc);
        assert!(!tl.fec);
        assert_eq!(tl.input, "clean.pcm");
        assert_eq!(tl.reference, "p25_nofec/clean.bit");
    }

    #[test]
    fn parse_cmpp25_dec_sdbits_line() {
        let line = "-c P25 -q -dec -sdbits 4 -r 0x0558 0x086b 0x1030 0x0000 0x0000 0x0190 p25/dam_e1_sd.bit -cmp p25/dam_e1_sd.pcm";
        let tl = parse_cmpp25_line(line, 1).unwrap().unwrap();
        assert_eq!(tl.op, SweepOp::Dec);
        assert_eq!(tl.sdbits, Some(4));
    }

    #[test]
    fn parse_cmpp25_skips_blank_and_comment_lines() {
        assert!(parse_cmpp25_line("", 1).unwrap().is_none());
        assert!(parse_cmpp25_line("   ", 2).unwrap().is_none());
        assert!(parse_cmpp25_line("# comment", 3).unwrap().is_none());
    }
}
