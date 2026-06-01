//! PCM-quality A/B benchmark harness.
//!
//! Direction memo: `project_direction_shift_pcm_quality_2026-04-24.md`.
//!
//! Runs a reference PCM through four codec-chain combinations and
//! writes the resulting WAVs for perceptual scoring. The cells are:
//!
//! 1. `our_enc_our_dec` — blip25-mbe encode + blip25-mbe decode.
//! 2. `our_enc_chip_dec` — blip25-mbe encode, AMBE-3000R chip decode.
//! 3. `chip_enc_our_dec` — chip encode, blip25-mbe decode.
//! 4. `chip_enc_chip_dec` — chip encode + chip decode.
//!
//! A fifth WAV (`ref.wav`) carries the reference PCM unchanged for
//! use as the PESQ/STOI reference channel. Per-cell timing and frame
//! counts are written to `meta.json`.
//!
//! The chip side is driven by `ssh <host> dvsi {encode,decode}` — the
//! driver lives at `conformance/chip/python/dvsi_driver.py` and is
//! installed as `/usr/local/bin/dvsi` on the chip host.
//! Per-frame chip throughput is ~33 ms/frame (realtime-capable) since
//! the `read_response` desync fix (2026-04-30) — it had been ~2.24 s/frame
//! when the driver waited on a phantom parity byte. Full-length vectors
//! run in seconds, so `--frames` is for scoping coverage, not wall time.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use blip25_mbe::codecs::ambe_plus2;
use blip25_mbe::codecs::mbe_baseline::analysis::{
    AnalysisOutput, AnalysisState, encode as analysis_encode, encode_ambe_plus2 as analysis_encode_ambe_plus2,
};
use blip25_mbe::codecs::mbe_baseline::{
    FrameDisposition, FrameErrorContext, GAMMA_W, SynthState, synthesize_frame,
};
use blip25_mbe::enhancement::{
    self as enh, ClassicalConfig, EnhancementMode, EnhancementState,
};
use blip25_mbe::mbe_params::MbeParams;
use blip25_mbe::imbe_wire::dequantize::{
    DecoderState, dequantize as dequantize_full, quantize as quantize_full,
};
use blip25_mbe::imbe_wire::frame::{
    decode_frame as decode_full_frame, encode_frame as encode_full_frame,
};
use blip25_mbe::imbe_wire::priority::{
    deprioritize as deprioritize_full, prioritize as prioritize_full,
};
use blip25_mbe::ambe_plus2_wire::dequantize::{
    DecoderState as HalfDecoderState, Decoded as HalfDecoded,
    decode_to_params as decode_to_params_half, quantize as quantize_half,
};
use blip25_mbe::ambe_plus2_wire::frame::{
    DIBITS_PER_FRAME as HALF_DIBITS_PER_FRAME, encode_frame as encode_half_frame,
};

const FRAME_SAMPLES: usize = 160;
const BYTES_PER_FEC_FRAME: usize = 18;
const DIBITS_PER_FRAME: usize = 72;
const SAMPLE_RATE: u32 = 8_000;
/// 36 dibits = 72 bits → 9 bytes per AMBE+2 half-rate FEC frame.
const HALF_BYTES_PER_FEC_FRAME: usize = 9;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Produce the 4-cell WAV matrix for a given reference PCM.
    AbMatrix {
        /// Path to reference PCM (8 kHz mono LE i16, raw, no header).
        #[arg(long)]
        pcm: PathBuf,
        /// Output directory — will be created if missing. Five WAVs
        /// and one `meta.json` are written here.
        #[arg(long)]
        out_dir: PathBuf,
        /// Cap frames processed. Each frame is 20 ms of audio. Chip
        /// throughput is ~33 ms/frame (realtime) since the 2026-04-30
        /// `read_response` desync fix, so this scopes coverage, not wall time.
        #[arg(long)]
        frames: Option<usize>,
        /// SSH user@host for the chip driver (e.g. `root@192.168.1.6`).
        /// Must have the `dvsi` CLI installed at `/usr/local/bin/dvsi`.
        #[arg(long, default_value = "root@192.168.1.6")]
        chip_host: String,
        /// Skip all chip-side cells (use when the chip is unavailable
        /// or for a local-only dry run). Only `our_enc_our_dec.wav`
        /// and `ref.wav` are produced.
        #[arg(long)]
        skip_chip: bool,
        /// Enable §0.8.4 silence detection in our analysis encoder.
        #[arg(long)]
        silence_dispatch: bool,
        /// Enable the joint-signal silence override (requires
        /// --silence-dispatch).
        #[arg(long)]
        pitch_silence_override: bool,
        /// Enable the onset-attack mitigation: on near-silent frames,
        /// commit a short default pitch (period 25 ≈ chip's b̂₀ ≈ 25)
        /// to pitch_history instead of the phantom long-period
        /// autocorrelation peak. Independent of --silence-dispatch.
        /// See `project_onset_attack_flag_landed_2026-04-27.md`.
        #[arg(long)]
        default_pitch_on_silence: bool,
        /// Multiply M̂_l amplitudes by this factor before wire quantize.
        /// Default 1.0 (no scaling). Probes whether a global amplitude
        /// calibration error is contributing to the PESQ gap.
        #[arg(long, default_value_t = 1.0)]
        amp_scale: f32,
        /// V/UV override for perceptual probing. `all-voiced` forces
        /// every band to V, `all-unvoiced` to U, `invert` flips each
        /// band's decision. `none` leaves §0.7's decision intact.
        #[arg(long, value_enum, default_value_t = VuvOverride::None)]
        vuv_override: VuvOverride,
        /// Beyond-spec: after N consecutive Repeat/Mute frames, fall
        /// back to a default-fundamental (b̂₀=134) + amps=1.0 frame
        /// instead of replaying the prior `last_good`. JMBE / SDRTrunk
        /// use N=3 for chip-stream interop quality. Default 0
        /// (disabled, spec-faithful per gap 0022).
        #[arg(long, default_value_t = 0)]
        repeat_reset_after: u32,
        /// Replace the §0.3 pitch tracker with the PYIN frontend
        /// (post-2002 DSP). A/B against the spec-faithful path. Off by
        /// default.
        #[arg(long)]
        pyin_pitch: bool,
        /// Disable Boll 1979 spectral subtraction on `signal_spectrum`
        /// before §0.5 amplitude estimation. ON by default (5-vector
        /// A/B 2026-05-14 shows 0 speech impact / +0.08 knox_1 lift).
        /// Pass this to A/B against the spec-faithful encoder input.
        #[arg(long)]
        no_spectral_subtraction: bool,
        /// EMA weight on §0.5 spectral amplitudes. `0.0` (default) =
        /// off; `(0.0, 1.0]` enables `M̂_l(t) = α·M̂_l + (1−α)·M̂_l(t−1)`,
        /// pitch-gated. Targets the noisy-tone amp-jitter described in
        /// `project_amp_noise_sensitivity_2026-04-24`. General-DSP
        /// magnitude smoothing, outside BABA-A clean-room scope.
        #[arg(long, default_value_t = 0.0)]
        amp_ema_alpha: f64,
        /// Post-decoder enhancement chain applied to PCM before
        /// `our_enc_our_dec.wav` / `chip_enc_our_dec.wav` are written.
        /// `classical` (default since 2026-05-14): AIC33-shaped HPF
        /// + presence + boundary-fade chain (+0.03 to +0.09 PESQ
        /// across the 5-vector matrix). `none` is spec-faithful
        /// synthesis only.
        #[arg(long, value_enum, default_value_t = EnhancementCli::Classical)]
        enhancement: EnhancementCli,
        /// Stage-isolation knobs for `--enhancement classical`. Each
        /// `--enh-no-*` flag disables the corresponding stage so the
        /// individual contribution can be A/B'd. `--enh-hpf-cutoff`
        /// overrides the default 250 Hz HPF cutoff (set lower to avoid
        /// eating low-male-voice fundamentals).
        #[arg(long)]
        enh_no_hpf: bool,
        #[arg(long)]
        enh_no_peaking: bool,
        #[arg(long)]
        enh_no_compressor: bool,
        #[arg(long)]
        enh_no_fade: bool,
        #[arg(long)]
        enh_hpf_cutoff: Option<f32>,
        /// Instrumentation: classify each decoded frame as "tone-like"
        /// (all-voiced + stable pitch + stable envelope) vs not, and
        /// print per-frame + summary counts on the our_enc_our_dec
        /// path. Doesn't change synthesis or enhancement output.
        #[arg(long)]
        log_tone_frames: bool,
    },
    /// Self-baseline our_enc → our_dec PESQ on the half-rate (AMBE+2)
    /// path. Slim version of `ab-matrix`: no chip side, just our
    /// pipeline through `encode_ambe_plus2` + `ambe_plus2_wire::quantize` +
    /// 9-byte FEC frames + `ambe_plus2_wire::dequantize` + AMBE+2 synth.
    HalfrateAbMatrix {
        /// Path to reference PCM (8 kHz mono LE i16, raw, no header).
        #[arg(long)]
        pcm: PathBuf,
        /// Output directory — will be created if missing.
        #[arg(long)]
        out_dir: PathBuf,
        /// Cap frames processed.
        #[arg(long)]
        frames: Option<usize>,
        /// SSH user@host for the chip driver (e.g. `root@192.168.1.6`).
        /// Must have the `dvsi` CLI installed at `/usr/local/bin/dvsi`
        /// with the `encode-halfrate` / `decode-halfrate` subcommands.
        /// The AMBE-3000R natively runs AMBE+2, so for half-rate the
        /// chip is a genuine bit-faithful oracle (unlike full-rate IMBE).
        #[arg(long, default_value = "root@192.168.1.6")]
        chip_host: String,
        /// Skip all chip-side cells (use when the chip is unavailable or
        /// for a local-only dry run). Only `our_enc_our_dec.wav` and
        /// `ref.wav` are produced.
        #[arg(long)]
        skip_chip: bool,
        /// Enable §0.8.4 silence detection in the analysis encoder.
        #[arg(long)]
        silence_dispatch: bool,
        /// Enable the joint-signal silence override.
        #[arg(long)]
        pitch_silence_override: bool,
        /// Enable the onset-attack mitigation. See AbMatrix doc.
        #[arg(long)]
        default_pitch_on_silence: bool,
        /// Enable Annex T tone-frame dispatch on the encoder. When the
        /// detector recognises a single-tone / DTMF / Knox / call-progress
        /// input, the encoder bypasses the voice pipeline and emits a
        /// tone-frame opcode instead. Decoder-side reconstruction is
        /// deterministic from the Annex T table — no random-phase
        /// harmonic synthesis. Off by default. Phase 1 IMBE has no
        /// equivalent path; this flag only affects the half-rate
        /// (AMBE+2 / Phase 2) pipeline.
        #[arg(long)]
        tone_detection: bool,
        /// EMA weight on §0.5 spectral amplitudes — see `ab-matrix`
        /// for semantics. `0.0` (default) = off.
        #[arg(long, default_value_t = 0.0)]
        amp_ema_alpha: f64,
        /// Disable §0.5 spectral subtraction (ON by default — see
        /// `ab-matrix` for semantics).
        #[arg(long)]
        no_spectral_subtraction: bool,
        /// Post-decoder enhancement chain applied to PCM before
        /// `our_enc_our_dec.wav` is written. Same chain used by the
        /// full-rate `ab-matrix` subcommand. `none` (default) is
        /// `classical` (default since 2026-05-14): AIC33-shaped HPF
        /// + presence + boundary-fade chain. `none` is spec-faithful
        /// synthesis only. Identical default across rates — the
        /// half-rate path reuses `ClassicalConfig::default`.
        #[arg(long, value_enum, default_value_t = EnhancementCli::Classical)]
        enhancement: EnhancementCli,
        /// Stage-isolation knobs for `--enhancement classical`. Same
        /// semantics as the full-rate `ab-matrix` flags.
        #[arg(long)]
        enh_no_hpf: bool,
        #[arg(long)]
        enh_no_peaking: bool,
        #[arg(long)]
        enh_no_compressor: bool,
        #[arg(long)]
        enh_no_fade: bool,
        #[arg(long)]
        enh_hpf_cutoff: Option<f32>,
    },
    /// Decode a stream of post-FEC AMBE+2 half-rate info vectors (one
    /// frame per line, 4 space-separated hex words = û₀ û₁ û₂ û₃,
    /// each 0..0xFFFF) directly into 8 kHz mono i16 PCM.
    ///
    /// Bypasses ALL P25-specific FEC (Annex H interleave, Golay/Hamming
    /// decode, PN demodulation). Useful for decoding voice frames from
    /// non-P25 AMBE+2 carriers — DMR, NXDN, D-STAR — once the consumer
    /// has stripped the wire-specific FEC into the same 4-vector
    /// post-FEC info shape that `ambe_plus2_wire::dequantize::dequantize`
    /// consumes. The 49-bit b̂₀..b̂₈ layout is shared across AMBE+2
    /// standards; the prioritization into 4 info vectors is also
    /// shared (per BABA-C / DVSI AMBE-3000 protocol). FEC and frame
    /// envelope differ between standards.
    ///
    /// Input format: text file, one frame per line, blank lines and
    /// `#`-prefixed comments ignored. Lines like:
    ///   `02af  0c1d  0a91  0014`
    ///
    /// Or pipe a `[u16; 4]` binary stream via `--binary`.
    DecodeRawHalfrate {
        /// Path to the info-vector input. Use `-` for stdin.
        #[arg(long)]
        input: PathBuf,
        /// Output WAV path.
        #[arg(long)]
        out_wav: PathBuf,
        /// Treat input as packed binary (8 bytes/frame, little-endian
        /// u16 quadruples) rather than text-hex.
        #[arg(long)]
        binary: bool,
    },
    /// Decode a stream of raw AMBE+2 half-rate parameter words
    /// (b̂₀..b̂₈, one frame per line, 9 space-separated decimal
    /// values) directly into 8 kHz mono i16 PCM.
    ///
    /// One step lower than `decode-raw-ambe-plus2`: the consumer
    /// supplies the 9 b̂ values directly, we run prioritize +
    /// dequantize + AMBE+2 synth. The b̂₀..b̂₈ shape is shared across
    /// AMBE+2 standards (P25 Phase 2 / DMR enhanced / NXDN type-2),
    /// so this is the most-portable entry point for non-P25 carriers
    /// where wire-FEC and prioritization both differ from P25.
    ///
    /// Per-field widths (b̂₀=7, b̂₁=5, b̂₂=5, b̂₃=11, b̂₄=4, b̂₅=4,
    /// b̂₆=4, b̂₇=4, b̂₈=4) are enforced; out-of-range values error.
    DecodeRawMbe {
        /// Path to the b̂-vector input. Use `-` for stdin.
        #[arg(long)]
        input: PathBuf,
        /// Output WAV path.
        #[arg(long)]
        out_wav: PathBuf,
    },
    /// Decode a stream of P25 Phase 2 / AMBE+2 half-rate FEC frames
    /// (9 bytes per frame = 36 dibits) into 8 kHz mono i16 PCM. Runs
    /// the full ambe_plus2_wire decode chain: unpack dibits →
    /// `decode_frame` (Annex H deinterleave + Golay/Hamming) →
    /// `dequantize` → AMBE+2 synth.
    ///
    /// Two input formats:
    /// - Text: one 18-hex-char line per frame (lines like
    ///   `E9C1F6445F718B761D`, `#`-comments and blank lines OK).
    /// - Binary: raw 9-byte frames concatenated (`--binary`).
    DecodeFecHalfrate {
        /// Path to FEC-frame input. Use `-` for stdin.
        #[arg(long)]
        input: PathBuf,
        /// Output WAV path.
        #[arg(long)]
        out_wav: PathBuf,
        /// Treat input as packed binary (9 bytes per frame) rather
        /// than text-hex.
        #[arg(long)]
        binary: bool,
    },
    /// Decode a stream of P25 Phase 1 / IMBE full-rate FEC frames
    /// (18 bytes per frame = 72 dibits) into 8 kHz mono i16 PCM.
    /// Runs the full imbe_wire decode chain through the chip-shaped
    /// `Vocoder::decode_bits`.
    ///
    /// Two input formats:
    /// - Text: one 36-hex-char line per frame, `#`-comments OK.
    /// - Binary: raw 18-byte frames concatenated (`--binary`).
    DecodeFecFullrate {
        /// Path to FEC-frame input. Use `-` for stdin.
        #[arg(long)]
        input: PathBuf,
        /// Output WAV path.
        #[arg(long)]
        out_wav: PathBuf,
        /// Treat input as packed binary (18 bytes per frame).
        #[arg(long)]
        binary: bool,
        /// JMBE-style error-rate freeze on Repeat (gap 0021). Default
        /// off — spec-faithful path. Enable for chip-encoded P25
        /// traffic to keep ε_R from climbing past the Mute threshold
        /// during runs of high-error frames.
        #[arg(long)]
        chip_compat: bool,
        /// Beyond-spec consecutive-repeat reset threshold. After N
        /// consecutive Repeat/Mute frames, fall back to a default
        /// fundamental + amps=1.0 frame instead of the prior snapshot.
        /// JMBE / SDRTrunk use 3.
        #[arg(long)]
        repeat_reset_after: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum VuvOverride {
    None,
    AllVoiced,
    AllUnvoiced,
    Invert,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum EnhancementCli {
    None,
    Classical,
}

#[derive(Clone, Copy, Debug, Default)]
struct EnhancementOverrides {
    no_hpf: bool,
    no_peaking: bool,
    no_compressor: bool,
    no_fade: bool,
    hpf_cutoff_hz: Option<f32>,
}

impl EnhancementCli {
    fn to_mode(self, ov: EnhancementOverrides) -> EnhancementMode {
        match self {
            EnhancementCli::None => EnhancementMode::None,
            EnhancementCli::Classical => {
                let mut cfg = ClassicalConfig::default();
                if ov.no_hpf {
                    cfg.biquads[0] = None;
                } else if let Some(hz) = ov.hpf_cutoff_hz {
                    cfg.biquads[0] =
                        Some(blip25_mbe::enhancement::Biquad::high_pass(8_000.0, hz, 0.707));
                }
                if ov.no_peaking {
                    cfg.biquads[1] = None;
                }
                if ov.no_compressor {
                    cfg.compressor = None;
                }
                if ov.no_fade {
                    cfg.boundary_fade_samples = 0;
                }
                EnhancementMode::Classical(cfg)
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Cmd::AbMatrix {
            pcm,
            out_dir,
            frames,
            chip_host,
            skip_chip,
            silence_dispatch,
            pitch_silence_override,
            default_pitch_on_silence,
            amp_scale,
            vuv_override,
            repeat_reset_after,
            pyin_pitch,
            no_spectral_subtraction,
            amp_ema_alpha,
            enhancement,
            enh_no_hpf,
            enh_no_peaking,
            enh_no_compressor,
            enh_no_fade,
            enh_hpf_cutoff,
            log_tone_frames,
        } => cmd_ab_matrix(
            &pcm,
            &out_dir,
            frames,
            &chip_host,
            skip_chip,
            silence_dispatch,
            pitch_silence_override,
            default_pitch_on_silence,
            amp_scale,
            vuv_override,
            repeat_reset_after,
            pyin_pitch,
            !no_spectral_subtraction,
            amp_ema_alpha,
            enhancement.to_mode(EnhancementOverrides {
                no_hpf: enh_no_hpf,
                no_peaking: enh_no_peaking,
                no_compressor: enh_no_compressor,
                no_fade: enh_no_fade,
                hpf_cutoff_hz: enh_hpf_cutoff,
            }),
            log_tone_frames,
        ),
        Cmd::HalfrateAbMatrix {
            pcm,
            out_dir,
            frames,
            chip_host,
            skip_chip,
            silence_dispatch,
            pitch_silence_override,
            default_pitch_on_silence,
            tone_detection,
            amp_ema_alpha,
            no_spectral_subtraction,
            enhancement,
            enh_no_hpf,
            enh_no_peaking,
            enh_no_compressor,
            enh_no_fade,
            enh_hpf_cutoff,
        } => cmd_ambe_plus2_ab_matrix(
            &pcm,
            &out_dir,
            frames,
            &chip_host,
            skip_chip,
            silence_dispatch,
            pitch_silence_override,
            default_pitch_on_silence,
            tone_detection,
            amp_ema_alpha,
            !no_spectral_subtraction,
            enhancement.to_mode(EnhancementOverrides {
                no_hpf: enh_no_hpf,
                no_peaking: enh_no_peaking,
                no_compressor: enh_no_compressor,
                no_fade: enh_no_fade,
                hpf_cutoff_hz: enh_hpf_cutoff,
            }),
        ),
        Cmd::DecodeRawHalfrate { input, out_wav, binary } => {
            cmd_decode_raw_ambe_plus2(&input, &out_wav, binary)
        }
        Cmd::DecodeRawMbe { input, out_wav } => cmd_decode_raw_mbe(&input, &out_wav),
        Cmd::DecodeFecHalfrate { input, out_wav, binary } => {
            cmd_decode_fec_ambe_plus2(&input, &out_wav, binary)
        }
        Cmd::DecodeFecFullrate { input, out_wav, binary, chip_compat, repeat_reset_after } => {
            cmd_decode_fec_imbe(&input, &out_wav, binary, chip_compat, repeat_reset_after)
        }
    }
}

fn cmd_ab_matrix(
    pcm_path: &Path,
    out_dir: &Path,
    frames: Option<usize>,
    chip_host: &str,
    skip_chip: bool,
    silence_dispatch: bool,
    pitch_silence_override: bool,
    default_pitch_on_silence: bool,
    amp_scale: f32,
    vuv_override: VuvOverride,
    repeat_reset_after: u32,
    pyin_pitch: bool,
    spectral_subtraction: bool,
    amp_ema_alpha: f64,
    enhancement: EnhancementMode,
    log_tone_frames: bool,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create out_dir {}", out_dir.display()))?;

    let pcm = read_pcm_i16(pcm_path)?;
    let n_frames_total = pcm.len() / FRAME_SAMPLES;
    let n_frames = frames
        .map(|n| n.min(n_frames_total))
        .unwrap_or(n_frames_total);
    let pcm = &pcm[..n_frames * FRAME_SAMPLES];
    eprintln!(
        "reference PCM: {} samples = {} frames ({:.1} s)",
        pcm.len(),
        n_frames,
        pcm.len() as f64 / f64::from(SAMPLE_RATE)
    );

    let ref_wav = out_dir.join("ref.wav");
    write_wav(&ref_wav, pcm)?;
    eprintln!("wrote {}", ref_wav.display());

    // ---- cell 1: our_enc + our_dec (fully local) ----
    let t0 = Instant::now();
    let our_bits = our_encode(
        pcm,
        silence_dispatch,
        pitch_silence_override,
        default_pitch_on_silence,
        amp_scale,
        vuv_override,
        pyin_pitch,
        spectral_subtraction,
        amp_ema_alpha,
    )?;
    let our_enc_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "our encode: {} frames in {:.2} s ({:.1} ms/frame)",
        our_bits.len() / BYTES_PER_FEC_FRAME,
        our_enc_secs,
        1000.0 * our_enc_secs / (our_bits.len() / BYTES_PER_FEC_FRAME).max(1) as f64
    );

    let t0 = Instant::now();
    let our_enc_our_dec = our_decode(&our_bits, repeat_reset_after, &enhancement, log_tone_frames);
    let our_dec_secs = t0.elapsed().as_secs_f64();
    let wav_path = out_dir.join("our_enc_our_dec.wav");
    write_wav(&wav_path, &our_enc_our_dec)?;
    eprintln!(
        "our_enc_our_dec: {} samples in {:.2} s → {}",
        our_enc_our_dec.len(),
        our_dec_secs,
        wav_path.display()
    );

    // Save our bits for downstream comparisons.
    let our_bit_path = out_dir.join("our.bit");
    fs::write(&our_bit_path, &our_bits)
        .with_context(|| format!("write our bits {}", our_bit_path.display()))?;

    let mut meta = Meta {
        pcm_path: pcm_path.display().to_string(),
        frames: n_frames,
        our_enc_secs,
        our_dec_secs,
        chip_enc_secs: None,
        chip_dec_secs: None,
        chip_host: chip_host.to_string(),
        silence_dispatch,
        pitch_silence_override,
        skip_chip,
    };

    if !skip_chip {
        // ---- cell 4 sub-step: chip encodes ref PCM ----
        let remote_pcm = format!("/tmp/blip25_ab_ref_{}.pcm", std::process::id());
        let remote_chip_bit = format!("/tmp/blip25_ab_chip_{}.bit", std::process::id());
        let remote_our_bit = format!("/tmp/blip25_ab_our_{}.bit", std::process::id());
        let remote_cedec_pcm = format!("/tmp/blip25_ab_cec_{}.pcm", std::process::id());
        let remote_oedec_pcm = format!("/tmp/blip25_ab_oec_{}.pcm", std::process::id());

        // Upload the reference PCM we actually analyzed (truncated to
        // --frames) so the chip sees the same input.
        let local_trimmed_pcm = out_dir.join("trimmed_ref.pcm");
        write_raw_pcm(&local_trimmed_pcm, pcm)?;
        scp_to(&local_trimmed_pcm, chip_host, &remote_pcm)?;

        // Chip encode: PCM → chip.bit
        let t0 = Instant::now();
        ssh_run(
            chip_host,
            &format!(
                "dvsi encode '{remote_pcm}' '{remote_chip_bit}' --frames {n_frames}"
            ),
        )?;
        let chip_enc_secs = t0.elapsed().as_secs_f64();
        meta.chip_enc_secs = Some(chip_enc_secs);
        eprintln!(
            "chip encode: {} frames in {:.1} s ({:.1} s/frame)",
            n_frames,
            chip_enc_secs,
            chip_enc_secs / n_frames as f64
        );

        // Fetch chip bits so we can run them through our decoder.
        let local_chip_bit = out_dir.join("chip.bit");
        scp_from(chip_host, &remote_chip_bit, &local_chip_bit)?;
        let chip_bits = fs::read(&local_chip_bit)?;

        // ---- cell 3: chip_enc + our_dec ----
        let chip_enc_our_dec = our_decode(&chip_bits, repeat_reset_after, &enhancement, false);
        let wav_path = out_dir.join("chip_enc_our_dec.wav");
        write_wav(&wav_path, &chip_enc_our_dec)?;
        eprintln!("chip_enc_our_dec → {}", wav_path.display());

        // Upload OUR bits so the chip can decode them.
        let local_our_bit = out_dir.join("our.bit");
        scp_to(&local_our_bit, chip_host, &remote_our_bit)?;

        // Two chip decodes in parallel? The chip is a single device —
        // must run sequentially. Also the chip-decode throughput is
        // roughly real-time, so these are fast.
        let t0 = Instant::now();
        ssh_run(
            chip_host,
            &format!(
                "dvsi decode '{remote_chip_bit}' '{remote_cedec_pcm}' --frames {n_frames}"
            ),
        )?;
        ssh_run(
            chip_host,
            &format!(
                "dvsi decode '{remote_our_bit}' '{remote_oedec_pcm}' --frames {n_frames}"
            ),
        )?;
        let chip_dec_secs = t0.elapsed().as_secs_f64();
        meta.chip_dec_secs = Some(chip_dec_secs);

        // Fetch chip's PCM outputs.
        let local_cec = out_dir.join("chip_enc_chip_dec.pcm");
        let local_oec = out_dir.join("our_enc_chip_dec.pcm");
        scp_from(chip_host, &remote_cedec_pcm, &local_cec)?;
        scp_from(chip_host, &remote_oedec_pcm, &local_oec)?;
        let cec_pcm = read_pcm_i16(&local_cec)?;
        let oec_pcm = read_pcm_i16(&local_oec)?;

        let wav_path = out_dir.join("chip_enc_chip_dec.wav");
        write_wav(&wav_path, &cec_pcm)?;
        eprintln!("chip_enc_chip_dec → {}", wav_path.display());

        let wav_path = out_dir.join("our_enc_chip_dec.wav");
        write_wav(&wav_path, &oec_pcm)?;
        eprintln!("our_enc_chip_dec → {}", wav_path.display());

        // Cleanup: best-effort remote and local tmp file removal.
        let _ = ssh_run(
            chip_host,
            &format!(
                "rm -f '{remote_pcm}' '{remote_chip_bit}' '{remote_our_bit}' \
                 '{remote_cedec_pcm}' '{remote_oedec_pcm}'"
            ),
        );
        let _ = fs::remove_file(&local_trimmed_pcm);
        let _ = fs::remove_file(&local_cec);
        let _ = fs::remove_file(&local_oec);
    }

    let meta_path = out_dir.join("meta.json");
    fs::write(&meta_path, meta.to_json())
        .with_context(|| format!("write meta {}", meta_path.display()))?;
    eprintln!("wrote {}", meta_path.display());

    Ok(())
}

/// Encode a PCM slice through our analysis + full-rate quantize +
/// channel encoder + dibit-pack pipeline. Returns the concatenated
/// 18-byte FEC frames. Silence/preroll frames are encoded by
/// quantizing `MbeParams::silence()` so every frame yields 18 bytes.
fn our_encode(
    pcm: &[i16],
    silence_dispatch: bool,
    pitch_silence_override: bool,
    default_pitch_on_silence: bool,
    amp_scale: f32,
    vuv_override: VuvOverride,
    pyin_pitch: bool,
    spectral_subtraction: bool,
    amp_ema_alpha: f64,
) -> Result<Vec<u8>> {
    let n_frames = pcm.len() / FRAME_SAMPLES;
    let mut state = AnalysisState::new();
    if silence_dispatch {
        state.set_silence_detection(true);
    }
    if pitch_silence_override {
        state.set_pitch_silence_override(true);
    }
    if default_pitch_on_silence {
        state.set_default_pitch_on_silence(true);
    }
    if pyin_pitch {
        state.set_pyin_pitch(true);
    }
    // Spectral subtraction is ON by default in `AnalysisState::new`;
    // the harness forwards the CLI bool unconditionally so
    // `--no-spectral-subtraction` can disable it. The §0.8.4 silence
    // detector runs every frame regardless of dispatch, so the noise
    // estimator gets primed either way.
    state.set_spectral_subtraction(spectral_subtraction);
    if amp_ema_alpha > 0.0 {
        state.set_amp_ema_alpha(amp_ema_alpha);
    }
    let mut decoder_state_for_quantize = DecoderState::new();
    let mut out = Vec::with_capacity(n_frames * BYTES_PER_FEC_FRAME);
    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let analysis = analysis_encode(frame, &mut state)
            .map_err(|e| anyhow!("analysis error at frame {f}: {e:?}"))?;
        let params = match analysis {
            AnalysisOutput::Voice(p) => p,
            AnalysisOutput::Silence => MbeParams::silence(),
        };
        // Probe-modification pass: amp_scale, vuv_override.
        let needs_rebuild = (amp_scale - 1.0).abs() > 1e-6
            || !matches!(vuv_override, VuvOverride::None);
        let params = if needs_rebuild {
            let l = params.harmonic_count();
            let omega = params.omega_0();
            let v: Vec<bool> = params.voiced_slice()[..l as usize]
                .iter()
                .map(|&v| match vuv_override {
                    VuvOverride::None => v,
                    VuvOverride::AllVoiced => true,
                    VuvOverride::AllUnvoiced => false,
                    VuvOverride::Invert => !v,
                })
                .collect();
            let raw_amps = params.amplitudes_slice();
            let amps: Vec<f32> = raw_amps[..l as usize]
                .iter()
                .map(|a| a * amp_scale)
                .collect();
            MbeParams::new(omega, l, &v, &amps)
                .map_err(|_| anyhow!("probe-modified MbeParams rejected at frame {f}"))?
        } else {
            params
        };
        // Quantize → param-level b[59]; prioritize → 8 info vectors;
        // channel-encode → 72 dibits; pack → 18 bytes. Snapshot the
        // decoder state and advance it via `dequantize_full` so the
        // NEXT frame's predictor matches what the receiver sees
        // (quantize_full mutates its state argument open-loop to the
        // encoder's own target M̃_l; dequantize advances to the
        // post-transmission reconstruction).
        let mut ds_snapshot = decoder_state_for_quantize.clone();
        let b = quantize_full(&params, &mut decoder_state_for_quantize)
            .map_err(|e| anyhow!("quantize error at frame {f}: {e:?}"))?;
        let l = params.harmonic_count();
        let info = prioritize_full(&b, l);
        let _ = dequantize_full(&info, &mut ds_snapshot);
        decoder_state_for_quantize = ds_snapshot;

        let dibits = encode_full_frame(&info);
        out.extend_from_slice(&pack_dibits(&dibits));
    }
    Ok(out)
}

/// Decode a concatenated FEC bit stream through our channel decoder +
/// full-rate dequantize + synthesizer. Bad-pitch frames emit a silent
/// 160-sample block (consistent with the `cmd_decode_pcm` behavior in
/// the vectors harness).
fn our_decode(
    fec: &[u8],
    repeat_reset_after: u32,
    enhancement: &EnhancementMode,
    log_tone_frames: bool,
) -> Vec<i16> {
    let n_frames = fec.len() / BYTES_PER_FEC_FRAME;
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    if repeat_reset_after > 0 {
        synth_state.set_repeat_reset_after(Some(repeat_reset_after));
    }
    let mut enh_state = EnhancementState::default();
    let mut prev_disposition: Option<FrameDisposition> = None;
    let mut tone_gate = ToneGate::default();
    let mut out = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    for f in 0..n_frames {
        let bytes = &fec[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_dibits(bytes);
        let imbe = decode_full_frame(&dibits);
        let _ = deprioritize_full; // kept in scope for future priority-level diag
        let params = match dequantize_full(&imbe.info, &mut decoder_state) {
            Ok(p) => p,
            Err(_) => {
                out.extend(std::iter::repeat(0i16).take(FRAME_SAMPLES));
                prev_disposition = None;
                if log_tone_frames {
                    tone_gate.observe_erasure(f);
                }
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let mut pcm = synthesize_frame(&params, &err, GAMMA_W, &mut synth_state).to_vec();
        if log_tone_frames {
            tone_gate.observe(f, &params, &pcm);
        }
        let prev_was_use = matches!(prev_disposition, Some(FrameDisposition::Use));
        enh::apply(enhancement, &mut enh_state, &mut pcm, SAMPLE_RATE as f32, prev_was_use);
        prev_disposition = synth_state.last_disposition();
        out.extend_from_slice(&pcm);
    }
    if log_tone_frames {
        tone_gate.summary();
    }
    out
}

/// Rolling tone-likeness classifier.
///
/// **Original design** required `all bands voiced` — but the prototype
/// run on `knox_1` (the canonical "noisy tone" vector) showed 0 of 125
/// frames had all bands voiced: the per-band V/UV decision marks noise
/// bands as unvoiced *because of the noise floor*, even on a sustained
/// tone. We drop that condition; the real noisy-tone signature is
/// stable pitch + stable envelope alone. Voiced fraction is logged as
/// instrumentation but doesn't gate.
///
/// A frame is "tone-like" iff:
///   1. |Δ pitch| from prior frame ≤ 20 cents, AND
///   2. last 4 frames all satisfied (1), AND
///   3. envelope (frame RMS dB) σ across that 4-frame window < 0.5 dB.
const TONE_GATE_WINDOW: usize = 4;
const TONE_GATE_CENTS_TOL: f32 = 20.0;
const TONE_GATE_RMS_SIGMA_DB: f32 = 0.5;

#[derive(Default)]
struct ToneGate {
    last_omega_0: Option<f32>,
    consec_stable: usize,
    rms_db_window: std::collections::VecDeque<f32>,
    total_frames: u32,
    tone_frames: u32,
    max_streak: u32,
    cur_streak: u32,
}

impl ToneGate {
    fn observe(&mut self, frame_idx: usize, params: &MbeParams, pcm: &[i16]) {
        self.total_frames += 1;

        let l = params.harmonic_count() as usize;
        let voiced = params.voiced_slice();
        let all_voiced = l > 0 && voiced.iter().take(l).all(|&v| v);

        let omega_0 = params.omega_0();
        let pitch_stable = match self.last_omega_0 {
            Some(prev) if prev > 0.0 && omega_0 > 0.0 => {
                let cents = 1200.0 * (omega_0 / prev).log2();
                cents.abs() <= TONE_GATE_CENTS_TOL
            }
            _ => false,
        };
        self.last_omega_0 = Some(omega_0);

        let rms_db = frame_rms_db(pcm);
        if self.rms_db_window.len() == TONE_GATE_WINDOW {
            self.rms_db_window.pop_front();
        }
        self.rms_db_window.push_back(rms_db);

        if pitch_stable {
            self.consec_stable += 1;
        } else {
            self.consec_stable = 0;
        }

        let env_stable = self.rms_db_window.len() == TONE_GATE_WINDOW
            && rms_sigma(&self.rms_db_window) < TONE_GATE_RMS_SIGMA_DB;

        let is_tone = self.consec_stable >= TONE_GATE_WINDOW && env_stable;
        if is_tone {
            self.tone_frames += 1;
            self.cur_streak += 1;
            if self.cur_streak > self.max_streak {
                self.max_streak = self.cur_streak;
            }
        } else {
            self.cur_streak = 0;
        }

        eprintln!(
            "tone-gate frame={frame_idx:4} L={l:2} all_voiced={all_voiced} \
             omega_0={omega_0:.4} pitch_stable={pitch_stable} rms_db={rms_db:6.2} \
             tone={is_tone}"
        );
    }

    fn observe_erasure(&mut self, frame_idx: usize) {
        self.total_frames += 1;
        self.consec_stable = 0;
        self.cur_streak = 0;
        self.last_omega_0 = None;
        self.rms_db_window.clear();
        eprintln!("tone-gate frame={frame_idx:4} ERASURE");
    }

    fn summary(&self) {
        let pct = if self.total_frames > 0 {
            100.0 * self.tone_frames as f32 / self.total_frames as f32
        } else {
            0.0
        };
        eprintln!(
            "tone-gate SUMMARY: tone_frames={}/{} ({:.1}%) max_streak={}",
            self.tone_frames, self.total_frames, pct, self.max_streak
        );
    }
}

fn frame_rms_db(pcm: &[i16]) -> f32 {
    if pcm.is_empty() {
        return -120.0;
    }
    let sumsq: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sumsq / pcm.len() as f64).sqrt() / 32_768.0;
    if rms > 1e-12 {
        20.0 * (rms as f32).log10()
    } else {
        -120.0
    }
}

fn rms_sigma(window: &std::collections::VecDeque<f32>) -> f32 {
    let n = window.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean = window.iter().sum::<f32>() / n;
    let var = window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    var.sqrt()
}

// ----------------------------------------------------------------------------
// Half-rate (AMBE+2) self-baseline.
// ----------------------------------------------------------------------------

fn cmd_ambe_plus2_ab_matrix(
    pcm_path: &Path,
    out_dir: &Path,
    frames: Option<usize>,
    chip_host: &str,
    skip_chip: bool,
    silence_dispatch: bool,
    pitch_silence_override: bool,
    default_pitch_on_silence: bool,
    tone_detection: bool,
    amp_ema_alpha: f64,
    spectral_subtraction: bool,
    enhancement: EnhancementMode,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create out_dir {}", out_dir.display()))?;
    let pcm = read_pcm_i16(pcm_path)?;
    let n_frames_total = pcm.len() / FRAME_SAMPLES;
    let n_frames = frames
        .map(|n| n.min(n_frames_total))
        .unwrap_or(n_frames_total);
    let pcm = &pcm[..n_frames * FRAME_SAMPLES];
    eprintln!(
        "reference PCM: {} samples = {} frames ({:.1} s)",
        pcm.len(),
        n_frames,
        pcm.len() as f64 / f64::from(SAMPLE_RATE)
    );

    let ref_wav = out_dir.join("ref.wav");
    write_wav(&ref_wav, pcm)?;

    let t0 = Instant::now();
    let our_bits = our_encode_ambe_plus2(
        pcm,
        silence_dispatch,
        pitch_silence_override,
        default_pitch_on_silence,
        tone_detection,
        amp_ema_alpha,
        spectral_subtraction,
    )?;
    let enc_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "ambe_plus2 encode: {} frames in {:.2} s ({:.1} ms/frame)",
        our_bits.len() / HALF_BYTES_PER_FEC_FRAME,
        enc_secs,
        1000.0 * enc_secs / (our_bits.len() / HALF_BYTES_PER_FEC_FRAME).max(1) as f64
    );

    let our_bit_path = out_dir.join("our.bit");
    fs::write(&our_bit_path, &our_bits)
        .with_context(|| format!("write our.bit {}", our_bit_path.display()))?;

    let t0 = Instant::now();
    let our_pcm = our_decode_ambe_plus2(&our_bits, &enhancement);
    let dec_secs = t0.elapsed().as_secs_f64();
    let wav_path = out_dir.join("our_enc_our_dec.wav");
    write_wav(&wav_path, &our_pcm)?;
    eprintln!(
        "ambe_plus2 decode: {} samples in {:.2} s → {}",
        our_pcm.len(),
        dec_secs,
        wav_path.display()
    );

    let mut meta = Meta {
        pcm_path: pcm_path.display().to_string(),
        frames: n_frames,
        our_enc_secs: enc_secs,
        our_dec_secs: dec_secs,
        chip_enc_secs: None,
        chip_dec_secs: None,
        chip_host: chip_host.to_string(),
        silence_dispatch,
        pitch_silence_override,
        skip_chip,
    };

    if !skip_chip {
        // Half-rate AMBE+2: the AMBE-3000R runs this codec natively, so
        // the chip cells here are a genuine bit-faithful oracle. Mirrors
        // the full-rate `cmd_ab_matrix` chip block, but drives the
        // `encode-halfrate` / `decode-halfrate` CLI paths and 9-byte frames.
        let pid = std::process::id();
        let remote_pcm = format!("/tmp/blip25_hr_ref_{pid}.pcm");
        let remote_chip_bit = format!("/tmp/blip25_hr_chip_{pid}.ambe9");
        let remote_our_bit = format!("/tmp/blip25_hr_our_{pid}.ambe9");
        let remote_cedec_pcm = format!("/tmp/blip25_hr_cec_{pid}.pcm");
        let remote_oedec_pcm = format!("/tmp/blip25_hr_oec_{pid}.pcm");

        // Upload the reference PCM we actually analyzed (truncated to
        // --frames) so the chip sees the same input.
        let local_trimmed_pcm = out_dir.join("trimmed_ref.pcm");
        write_raw_pcm(&local_trimmed_pcm, pcm)?;
        scp_to(&local_trimmed_pcm, chip_host, &remote_pcm)?;

        // ---- chip encode: PCM → chip.ambe9 (9-byte FEC frames) ----
        let t0 = Instant::now();
        ssh_run(
            chip_host,
            &format!("dvsi encode-halfrate '{remote_pcm}' '{remote_chip_bit}' --frames {n_frames}"),
        )?;
        let chip_enc_secs = t0.elapsed().as_secs_f64();
        meta.chip_enc_secs = Some(chip_enc_secs);
        eprintln!(
            "chip encode (half-rate): {} frames in {:.1} s ({:.1} ms/frame)",
            n_frames,
            chip_enc_secs,
            1000.0 * chip_enc_secs / n_frames.max(1) as f64
        );

        // Fetch chip bits so we can run them through our decoder.
        let local_chip_bit = out_dir.join("chip.ambe9");
        scp_from(chip_host, &remote_chip_bit, &local_chip_bit)?;
        let chip_bits = fs::read(&local_chip_bit)?;

        // ---- cell 3: chip_enc + our_dec ----
        let chip_enc_our_dec = our_decode_ambe_plus2(&chip_bits, &enhancement);
        let wav_path = out_dir.join("chip_enc_our_dec.wav");
        write_wav(&wav_path, &chip_enc_our_dec)?;
        eprintln!("chip_enc_our_dec → {}", wav_path.display());

        // Upload OUR bits so the chip can decode them.
        scp_to(&our_bit_path, chip_host, &remote_our_bit)?;

        // ---- cell 4 + cell 2: chip decodes both bitstreams ----
        let t0 = Instant::now();
        ssh_run(
            chip_host,
            &format!("dvsi decode-halfrate '{remote_chip_bit}' '{remote_cedec_pcm}' --frames {n_frames}"),
        )?;
        ssh_run(
            chip_host,
            &format!("dvsi decode-halfrate '{remote_our_bit}' '{remote_oedec_pcm}' --frames {n_frames}"),
        )?;
        let chip_dec_secs = t0.elapsed().as_secs_f64();
        meta.chip_dec_secs = Some(chip_dec_secs);

        // Fetch chip's PCM outputs.
        let local_cec = out_dir.join("chip_enc_chip_dec.pcm");
        let local_oec = out_dir.join("our_enc_chip_dec.pcm");
        scp_from(chip_host, &remote_cedec_pcm, &local_cec)?;
        scp_from(chip_host, &remote_oedec_pcm, &local_oec)?;
        let cec_pcm = read_pcm_i16(&local_cec)?;
        let oec_pcm = read_pcm_i16(&local_oec)?;

        // ---- cell 4: chip_enc + chip_dec ----
        let wav_path = out_dir.join("chip_enc_chip_dec.wav");
        write_wav(&wav_path, &cec_pcm)?;
        eprintln!("chip_enc_chip_dec → {}", wav_path.display());

        // ---- cell 2: our_enc + chip_dec ----
        let wav_path = out_dir.join("our_enc_chip_dec.wav");
        write_wav(&wav_path, &oec_pcm)?;
        eprintln!("our_enc_chip_dec → {}", wav_path.display());

        // Cleanup: best-effort remote and local tmp file removal.
        let _ = ssh_run(
            chip_host,
            &format!(
                "rm -f '{remote_pcm}' '{remote_chip_bit}' '{remote_our_bit}' \
                 '{remote_cedec_pcm}' '{remote_oedec_pcm}'"
            ),
        );
        let _ = fs::remove_file(&local_trimmed_pcm);
        let _ = fs::remove_file(&local_cec);
        let _ = fs::remove_file(&local_oec);
    }

    let meta_path = out_dir.join("meta.json");
    fs::write(&meta_path, meta.to_json())
        .with_context(|| format!("write meta {}", meta_path.display()))?;
    eprintln!("wrote {}", meta_path.display());

    Ok(())
}

fn our_encode_ambe_plus2(
    pcm: &[i16],
    silence_dispatch: bool,
    pitch_silence_override: bool,
    default_pitch_on_silence: bool,
    tone_detection: bool,
    amp_ema_alpha: f64,
    spectral_subtraction: bool,
) -> Result<Vec<u8>> {
    if tone_detection {
        // Use the Vocoder façade — the tone-frame dispatch lives in
        // `ambe_plus2_pipeline::encode` (vocoder.rs) and isn't exposed
        // by the lower-level `analysis_encode_ambe_plus2` path.
        use blip25_mbe::vocoder::{AnalysisOutputKind, Rate, Vocoder};
        let mut v = Vocoder::builder(Rate::AmbePlus2_3600x2450)
            .silence_dispatch(silence_dispatch)
            .pitch_silence_override(pitch_silence_override)
            .default_pitch_on_silence(default_pitch_on_silence)
            .tone_detection(true)
            .amp_ema_alpha(amp_ema_alpha)
            .spectral_subtraction(spectral_subtraction)
            .build();
        let n_frames = pcm.len() / FRAME_SAMPLES;
        let mut out = Vec::with_capacity(n_frames * HALF_BYTES_PER_FEC_FRAME);
        let mut tone_count: u32 = 0;
        let mut id_hist: std::collections::BTreeMap<u8, u32> = std::collections::BTreeMap::new();
        let mut amp_sum: u64 = 0;
        let mut amp_min: u8 = 127;
        let mut amp_max: u8 = 0;
        for f in 0..n_frames {
            let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
            let bits = v.encode_pcm(frame)
                .map_err(|e| anyhow!("vocoder encode error at frame {f}: {e:?}"))?;
            if let Some(stats) = v.last_stats().analysis.as_ref() {
                if let AnalysisOutputKind::Tone { id, amplitude } = stats.output {
                    tone_count += 1;
                    *id_hist.entry(id).or_insert(0) += 1;
                    amp_sum += u64::from(amplitude);
                    amp_min = amp_min.min(amplitude);
                    amp_max = amp_max.max(amplitude);
                }
            }
            out.extend_from_slice(&bits);
        }
        eprintln!(
            "tone-frame dispatch: {tone_count}/{n_frames} frames emitted as tone frames"
        );
        if tone_count > 0 {
            let summary: Vec<String> = id_hist
                .iter()
                .map(|(id, count)| format!("id={id}:{count}"))
                .collect();
            let amp_mean = amp_sum as f64 / tone_count as f64;
            eprintln!("  tone-id histogram: {}", summary.join(" "));
            eprintln!(
                "  tone A_D: mean={amp_mean:.1} min={amp_min} max={amp_max} (range 0..127)"
            );
        }
        return Ok(out);
    }

    let n_frames = pcm.len() / FRAME_SAMPLES;
    let mut state = AnalysisState::new();
    if silence_dispatch {
        state.set_silence_detection(true);
    }
    if pitch_silence_override {
        state.set_pitch_silence_override(true);
    }
    if default_pitch_on_silence {
        state.set_default_pitch_on_silence(true);
    }
    if amp_ema_alpha > 0.0 {
        state.set_amp_ema_alpha(amp_ema_alpha);
    }
    state.set_spectral_subtraction(spectral_subtraction);
    let mut decoder_state = HalfDecoderState::new();
    let mut out = Vec::with_capacity(n_frames * HALF_BYTES_PER_FEC_FRAME);
    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        // `encode_ambe_plus2` now dispatches PitchOutOfRange as Silence
        // internally (blip25-mbe Wave 1.1 fix). `MbeParams::silence_ambe_plus2`
        // gives a half-rate-table-compatible placeholder. Only treat
        // remaining analysis errors as fatal.
        let params = match analysis_encode_ambe_plus2(frame, &mut state) {
            Ok(AnalysisOutput::Voice(p)) => p,
            Ok(AnalysisOutput::Silence) => MbeParams::silence_ambe_plus2(),
            Err(e) => return Err(anyhow!("analysis-ambe-plus2 error at frame {f}: {e:?}")),
        };
        let info = quantize_half(&params, &mut decoder_state)
            .map_err(|e| anyhow!("ambe_plus2 quantize error at frame {f}: {e:?}"))?;
        let dibits = encode_half_frame(&info);
        out.extend_from_slice(&pack_dibits_half(&dibits));
    }
    Ok(out)
}

fn our_decode_ambe_plus2(fec: &[u8], enhancement: &EnhancementMode) -> Vec<i16> {
    // Route through the Vocoder façade so the post-decoder enhancement
    // chain (HPF + peaking + compressor + boundary-fade + output gain)
    // applies to half-rate output identically to the full-rate path.
    use blip25_mbe::vocoder::{Rate, Vocoder};
    let n_frames = fec.len() / HALF_BYTES_PER_FEC_FRAME;
    let mut v = Vocoder::builder(Rate::AmbePlus2_3600x2450)
        .enhancement(enhancement.clone())
        .build();
    let mut out = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    for f in 0..n_frames {
        let bytes = &fec[f * HALF_BYTES_PER_FEC_FRAME..(f + 1) * HALF_BYTES_PER_FEC_FRAME];
        // decode_bits handles dibit unpack + Annex H decode + dequantize
        // + Voice/Tone/Erasure dispatch internally.
        let pcm = v
            .decode_bits(bytes)
            .unwrap_or_else(|_| vec![0i16; FRAME_SAMPLES]);
        out.extend_from_slice(&pcm);
    }
    out
}

/// Decode a text/binary stream of post-FEC half-rate info vectors
/// `[û₀, û₁, û₂, û₃]` into PCM via `dequantize_half` + AMBE+2 synth.
///
/// Wire-FEC-agnostic entry point for AMBE+2 half-rate. Useful for any
/// carrier whose voice frames carry the same post-FEC info-vector
/// layout (P25 Phase 2, DMR enhanced full-rate, NXDN type-2) once the
/// consumer's wire-extractor has stripped that carrier's FEC.
fn cmd_decode_raw_ambe_plus2(input: &Path, out_wav: &Path, binary: bool) -> Result<()> {
    use std::io::Read;
    let raw: Vec<u8> = if input == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(input).with_context(|| format!("read {}", input.display()))?
    };

    let frames: Vec<[u16; 4]> = if binary {
        if raw.len() % 8 != 0 {
            return Err(anyhow!(
                "binary input must be a whole number of 8-byte frames; got {} bytes",
                raw.len()
            ));
        }
        raw.chunks_exact(8)
            .map(|c| {
                [
                    u16::from_le_bytes([c[0], c[1]]),
                    u16::from_le_bytes([c[2], c[3]]),
                    u16::from_le_bytes([c[4], c[5]]),
                    u16::from_le_bytes([c[6], c[7]]),
                ]
            })
            .collect()
    } else {
        let text = String::from_utf8(raw).context("input is not valid UTF-8 text")?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let stripped = line.split('#').next().unwrap_or("").trim();
            if stripped.is_empty() {
                continue;
            }
            let toks: Vec<&str> = stripped.split_whitespace().collect();
            if toks.len() != 4 {
                return Err(anyhow!(
                    "line {}: expected 4 hex words (û₀..û₃), got {} ({})",
                    lineno + 1,
                    toks.len(),
                    stripped
                ));
            }
            let mut info = [0u16; 4];
            for (i, t) in toks.iter().enumerate() {
                info[i] = u16::from_str_radix(t.trim_start_matches("0x"), 16)
                    .with_context(|| format!("line {}: parse {} as hex", lineno + 1, t))?;
            }
            out.push(info);
        }
        out
    };

    let mut decoder = HalfDecoderState::new();
    let mut synth = SynthState::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
    let mut last_voice: Option<MbeParams> = None;
    let mut decoded = 0usize;
    let mut tones = 0usize;
    let mut erasures = 0usize;
    for (f, info) in frames.iter().enumerate() {
        let block = match decode_to_params_half(info, &mut decoder) {
            Ok(HalfDecoded::Voice(p)) => {
                last_voice = Some(p.clone());
                decoded += 1;
                ambe_plus2::synthesize_frame(&p, &mut synth)
            }
            Ok(HalfDecoded::Tone { params, .. }) => {
                last_voice = Some(params.clone());
                tones += 1;
                ambe_plus2::synthesize_tone(&params, &mut synth)
            }
            Ok(HalfDecoded::Erasure) => {
                erasures += 1;
                match &last_voice {
                    Some(p) => ambe_plus2::synthesize_frame(p, &mut synth),
                    None => [0i16; FRAME_SAMPLES],
                }
            }
            Err(e) => {
                erasures += 1;
                eprintln!("frame {f}: dequantize error {e:?}, emitting silence");
                [0i16; FRAME_SAMPLES]
            }
        };
        pcm.extend_from_slice(&block);
    }
    write_wav(out_wav, &pcm)?;
    eprintln!(
        "frames={} voice={} tone={} erasure_or_err={}  → {}",
        frames.len(),
        decoded,
        tones,
        erasures,
        out_wav.display()
    );
    Ok(())
}

/// Decode a text stream of raw AMBE+2 half-rate parameter words
/// (b̂₀..b̂₈, one frame per line, 9 decimal values) directly into PCM.
///
/// One step lower than `cmd_decode_raw_ambe_plus2`: the caller supplies
/// the 9 b̂ values pre-prioritization. We run `ambe_plus2_wire::priority::
/// prioritize` to produce the 4 info vectors, then `dequantize` +
/// AMBE+2 synth.
///
/// b̂-field widths from BABA-C / AMBE-3000 protocol: b̂₀=7 (pitch),
/// b̂₁=5 (V/UV), b̂₂=5 (gain), b̂₃=11, b̂₄=4, b̂₅=4, b̂₆=4, b̂₇=4, b̂₈=4.
/// Total 48 bits + 1 don't-care padding = 49-bit voice payload.
fn cmd_decode_raw_mbe(input: &Path, out_wav: &Path) -> Result<()> {
    use blip25_mbe::ambe_plus2_wire::priority::prioritize as prioritize_half;
    use std::io::Read;
    let raw: Vec<u8> = if input == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(input).with_context(|| format!("read {}", input.display()))?
    };
    let text = String::from_utf8(raw).context("input is not valid UTF-8 text")?;

    // Per BABA-C / AMBE-3000 b̂ widths.
    const WIDTHS: [u8; 9] = [7, 5, 5, 11, 4, 4, 4, 4, 4];

    let mut frames: Vec<[u16; 9]> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let stripped = line.split('#').next().unwrap_or("").trim();
        if stripped.is_empty() {
            continue;
        }
        let toks: Vec<&str> = stripped.split_whitespace().collect();
        if toks.len() != 9 {
            return Err(anyhow!(
                "line {}: expected 9 b̂ values, got {} ({})",
                lineno + 1,
                toks.len(),
                stripped
            ));
        }
        let mut b = [0u16; 9];
        for (i, t) in toks.iter().enumerate() {
            // Accept hex (0x...) or decimal.
            let v = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                u16::from_str_radix(h, 16)
            } else {
                t.parse::<u16>()
            };
            let v = v.with_context(|| format!("line {}: parse {}", lineno + 1, t))?;
            let max = (1u32 << WIDTHS[i]) - 1;
            if u32::from(v) > max {
                return Err(anyhow!(
                    "line {}: b̂_{} = {} exceeds {}-bit max {}",
                    lineno + 1,
                    i,
                    v,
                    WIDTHS[i],
                    max
                ));
            }
            b[i] = v;
        }
        frames.push(b);
    }

    let mut decoder = HalfDecoderState::new();
    let mut synth = SynthState::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
    let mut last_voice: Option<MbeParams> = None;
    let mut decoded = 0usize;
    let mut tones = 0usize;
    let mut erasures = 0usize;
    for (f, b) in frames.iter().enumerate() {
        let info = prioritize_half(b);
        let block = match decode_to_params_half(&info, &mut decoder) {
            Ok(HalfDecoded::Voice(p)) => {
                last_voice = Some(p.clone());
                decoded += 1;
                ambe_plus2::synthesize_frame(&p, &mut synth)
            }
            Ok(HalfDecoded::Tone { params, .. }) => {
                last_voice = Some(params.clone());
                tones += 1;
                ambe_plus2::synthesize_tone(&params, &mut synth)
            }
            Ok(HalfDecoded::Erasure) => {
                erasures += 1;
                match &last_voice {
                    Some(p) => ambe_plus2::synthesize_frame(p, &mut synth),
                    None => [0i16; FRAME_SAMPLES],
                }
            }
            Err(e) => {
                erasures += 1;
                eprintln!("frame {f}: dequantize error {e:?}, emitting silence");
                [0i16; FRAME_SAMPLES]
            }
        };
        pcm.extend_from_slice(&block);
    }
    write_wav(out_wav, &pcm)?;
    eprintln!(
        "frames={} voice={} tone={} erasure_or_err={}  → {}",
        frames.len(),
        decoded,
        tones,
        erasures,
        out_wav.display()
    );
    Ok(())
}

/// Decode FEC-level half-rate frames (9 bytes / 36 dibits each) into
/// PCM via our P25 Phase 2 wire decode + AMBE+2 synth.
///
/// Wire-FEC-aware entry point. Useful for raw chip dumps, off-air
/// captures, and any consumer that has 9-byte AMBE+2 FEC frames in
/// hand. Mirrors `decode-raw-ambe-plus2` but one layer up — input is
/// pre-FEC-decode bytes, output is PCM after the full wire chain.
fn cmd_decode_fec_ambe_plus2(input: &Path, out_wav: &Path, binary: bool) -> Result<()> {
    use std::io::Read;
    let raw: Vec<u8> = if input == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(input).with_context(|| format!("read {}", input.display()))?
    };

    let frames: Vec<[u8; HALF_BYTES_PER_FEC_FRAME]> = if binary {
        if raw.len() % HALF_BYTES_PER_FEC_FRAME != 0 {
            return Err(anyhow!(
                "binary input must be a whole number of 9-byte frames; got {} bytes",
                raw.len()
            ));
        }
        raw.chunks_exact(HALF_BYTES_PER_FEC_FRAME)
            .map(|c| {
                let mut a = [0u8; HALF_BYTES_PER_FEC_FRAME];
                a.copy_from_slice(c);
                a
            })
            .collect()
    } else {
        let text = String::from_utf8(raw).context("input is not valid UTF-8 text")?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let stripped = line.split('#').next().unwrap_or("").trim();
            if stripped.is_empty() {
                continue;
            }
            // Strip whitespace/punctuation, accept e.g. "E9C1F6445F718B761D"
            // or "e9 c1 f6 44 5f 71 8b 76 1d".
            let hex: String = stripped
                .chars()
                .filter(|c| !c.is_whitespace() && *c != ':' && *c != '-')
                .collect();
            if hex.len() != 2 * HALF_BYTES_PER_FEC_FRAME {
                return Err(anyhow!(
                    "line {}: expected {} hex chars (= {} bytes), got {} ({})",
                    lineno + 1,
                    2 * HALF_BYTES_PER_FEC_FRAME,
                    HALF_BYTES_PER_FEC_FRAME,
                    hex.len(),
                    stripped
                ));
            }
            let mut bytes = [0u8; HALF_BYTES_PER_FEC_FRAME];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
                    .with_context(|| format!("line {}: parse hex byte at {}", lineno + 1, i))?;
            }
            out.push(bytes);
        }
        out
    };

    // Drive the chip-shaped Vocoder API rather than `our_decode_ambe_plus2`'s
    // probe-flag-aware path -- this is a clean carrier-agnostic decode and
    // wants the carrier-agnostic surface.
    use blip25_mbe::vocoder::{Rate, Vocoder};
    let mut vocoder = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let mut pcm: Vec<i16> = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
    for (f, bytes) in frames.iter().enumerate() {
        let frame = vocoder
            .decode_bits(bytes)
            .with_context(|| format!("decode frame {f}"))?;
        pcm.extend_from_slice(&frame);
    }
    write_wav(out_wav, &pcm)?;
    eprintln!(
        "frames={}  samples={}  rms={:.0}  → {}",
        frames.len(),
        pcm.len(),
        if pcm.is_empty() {
            0.0
        } else {
            (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt()
        },
        out_wav.display()
    );
    Ok(())
}

/// Decode FEC-level full-rate IMBE frames (18 bytes / 72 dibits each)
/// into PCM via the chip-shaped Vocoder API.
///
/// Mirrors `cmd_decode_fec_ambe_plus2` but for P25 Phase 1 input.
fn cmd_decode_fec_imbe(
    input: &Path,
    out_wav: &Path,
    binary: bool,
    chip_compat: bool,
    repeat_reset_after: Option<u32>,
) -> Result<()> {
    use blip25_mbe::vocoder::{Rate, Vocoder};
    use std::io::Read;
    let raw: Vec<u8> = if input == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        fs::read(input).with_context(|| format!("read {}", input.display()))?
    };

    let frames: Vec<[u8; BYTES_PER_FEC_FRAME]> = if binary {
        if raw.len() % BYTES_PER_FEC_FRAME != 0 {
            return Err(anyhow!(
                "binary input must be a whole number of {}-byte frames; got {} bytes",
                BYTES_PER_FEC_FRAME,
                raw.len()
            ));
        }
        raw.chunks_exact(BYTES_PER_FEC_FRAME)
            .map(|c| {
                let mut a = [0u8; BYTES_PER_FEC_FRAME];
                a.copy_from_slice(c);
                a
            })
            .collect()
    } else {
        let text = String::from_utf8(raw).context("input is not valid UTF-8 text")?;
        let mut out = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let stripped = line.split('#').next().unwrap_or("").trim();
            if stripped.is_empty() {
                continue;
            }
            let hex: String = stripped
                .chars()
                .filter(|c| !c.is_whitespace() && *c != ':' && *c != '-')
                .collect();
            if hex.len() != 2 * BYTES_PER_FEC_FRAME {
                return Err(anyhow!(
                    "line {}: expected {} hex chars (= {} bytes), got {} ({})",
                    lineno + 1,
                    2 * BYTES_PER_FEC_FRAME,
                    BYTES_PER_FEC_FRAME,
                    hex.len(),
                    stripped
                ));
            }
            let mut bytes = [0u8; BYTES_PER_FEC_FRAME];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
                    .with_context(|| format!("line {}: parse hex byte at {}", lineno + 1, i))?;
            }
            out.push(bytes);
        }
        out
    };

    let mut vocoder = Vocoder::builder(Rate::Imbe7200x4400)
        .chip_compat(chip_compat)
        .repeat_reset_after(repeat_reset_after)
        .build();
    let mut pcm: Vec<i16> = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
    for (f, bytes) in frames.iter().enumerate() {
        let frame = vocoder
            .decode_bits(bytes)
            .with_context(|| format!("decode frame {f}"))?;
        pcm.extend_from_slice(&frame);
    }
    write_wav(out_wav, &pcm)?;
    eprintln!(
        "frames={}  samples={}  rms={:.0}  → {}",
        frames.len(),
        pcm.len(),
        if pcm.is_empty() {
            0.0
        } else {
            (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt()
        },
        out_wav.display()
    );
    Ok(())
}

fn pack_dibits_half(dibits: &[u8; HALF_DIBITS_PER_FRAME]) -> [u8; HALF_BYTES_PER_FEC_FRAME] {
    let mut out = [0u8; HALF_BYTES_PER_FEC_FRAME];
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

// ----------------------------------------------------------------------------
// Dibit pack/unpack — inline so we don't depend on the vectors crate's
// private helpers. MSB-first bit order matching the P25 transmission spec.
// ----------------------------------------------------------------------------

fn pack_dibits(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; BYTES_PER_FEC_FRAME] {
    let mut out = [0u8; BYTES_PER_FEC_FRAME];
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

fn unpack_dibits(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
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

// ----------------------------------------------------------------------------
// WAV I/O and PCM helpers.
// ----------------------------------------------------------------------------

fn read_pcm_i16(path: &Path) -> Result<Vec<i16>> {
    let bytes = fs::read(path)
        .with_context(|| format!("read PCM {}", path.display()))?;
    if bytes.len() % 2 != 0 {
        return Err(anyhow!("PCM file {} has odd byte count", path.display()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok(out)
}

fn write_raw_pcm(path: &Path, pcm: &[i16]) -> Result<()> {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, bytes)
        .with_context(|| format!("write PCM {}", path.display()))
}

fn write_wav(path: &Path, pcm: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create WAV {}", path.display()))?;
    for &s in pcm {
        w.write_sample(s)
            .with_context(|| format!("write sample to {}", path.display()))?;
    }
    w.finalize()
        .with_context(|| format!("finalize WAV {}", path.display()))?;
    Ok(())
}

// ----------------------------------------------------------------------------
// SSH/SCP orchestration for chip-side encode/decode.
// ----------------------------------------------------------------------------

fn ssh_run(host: &str, cmd: &str) -> Result<()> {
    let status = Command::new("ssh")
        .arg("-o").arg("BatchMode=yes")
        .arg("-o").arg("StrictHostKeyChecking=accept-new")
        .arg(host)
        .arg(cmd)
        .status()
        .with_context(|| format!("ssh {host} '{cmd}'"))?;
    if !status.success() {
        return Err(anyhow!("ssh {host} exit status {:?}", status.code()));
    }
    Ok(())
}

fn scp_to(local: &Path, host: &str, remote: &str) -> Result<()> {
    let status = Command::new("scp")
        .arg("-q")
        .arg(local)
        .arg(format!("{host}:{remote}"))
        .status()
        .with_context(|| format!("scp {} → {host}:{remote}", local.display()))?;
    if !status.success() {
        return Err(anyhow!("scp upload exit status {:?}", status.code()));
    }
    Ok(())
}

fn scp_from(host: &str, remote: &str, local: &Path) -> Result<()> {
    let status = Command::new("scp")
        .arg("-q")
        .arg(format!("{host}:{remote}"))
        .arg(local)
        .status()
        .with_context(|| format!("scp {host}:{remote} → {}", local.display()))?;
    if !status.success() {
        return Err(anyhow!("scp download exit status {:?}", status.code()));
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Meta (run metadata) — written as hand-serialized JSON to avoid a
// serde dep for this small record.
// ----------------------------------------------------------------------------

struct Meta {
    pcm_path: String,
    frames: usize,
    our_enc_secs: f64,
    our_dec_secs: f64,
    chip_enc_secs: Option<f64>,
    chip_dec_secs: Option<f64>,
    chip_host: String,
    silence_dispatch: bool,
    pitch_silence_override: bool,
    skip_chip: bool,
}

impl Meta {
    fn to_json(&self) -> String {
        fn f(v: Option<f64>) -> String {
            match v {
                Some(x) => format!("{x:.4}"),
                None => "null".into(),
            }
        }
        format!(
            r#"{{
  "pcm_path": "{}",
  "frames": {},
  "our_enc_secs": {:.4},
  "our_dec_secs": {:.4},
  "chip_enc_secs": {},
  "chip_dec_secs": {},
  "chip_host": "{}",
  "silence_dispatch": {},
  "pitch_silence_override": {},
  "skip_chip": {}
}}
"#,
            self.pcm_path.replace('"', "\\\""),
            self.frames,
            self.our_enc_secs,
            self.our_dec_secs,
            f(self.chip_enc_secs),
            f(self.chip_dec_secs),
            self.chip_host.replace('"', "\\\""),
            self.silence_dispatch,
            self.pitch_silence_override,
            self.skip_chip
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_dibits_roundtrip() {
        let dibits: [u8; DIBITS_PER_FRAME] = std::array::from_fn(|i| (i as u8) & 3);
        let bytes = pack_dibits(&dibits);
        let back = unpack_dibits(&bytes);
        assert_eq!(dibits, back);
    }

    #[test]
    fn pack_dibits_msb_first() {
        let mut dibits = [0u8; DIBITS_PER_FRAME];
        dibits[0] = 0b11; // both high bits set
        dibits[1] = 0b10; // only the high bit set
        let bytes = pack_dibits(&dibits);
        // First byte top nibble: 1 1 | 1 0 | 0 0 | 0 0 = 0b1110_0000 = 0xE0
        assert_eq!(bytes[0], 0b1110_0000);
    }
}
