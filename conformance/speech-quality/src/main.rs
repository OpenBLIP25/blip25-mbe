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
//! Per-frame chip throughput is ~1.8–15 s (see
//! `project_chip_unblock_session_2026-04-23`), so budget `--frames`
//! to a few hundred when building baseline numbers.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use blip25_mbe::codecs::mbe_baseline::analysis::{
    AnalysisOutput, AnalysisState, encode as analysis_encode,
};
use blip25_mbe::codecs::mbe_baseline::{
    FrameErrorContext, GAMMA_W, SynthState, synthesize_frame,
};
use blip25_mbe::mbe_params::MbeParams;
use blip25_mbe::p25_fullrate::dequantize::{
    DecoderState, dequantize as dequantize_full, quantize as quantize_full,
};
use blip25_mbe::p25_fullrate::frame::{
    decode_frame as decode_full_frame, encode_frame as encode_full_frame,
};
use blip25_mbe::p25_fullrate::priority::{
    deprioritize as deprioritize_full, prioritize as prioritize_full,
};

const FRAME_SAMPLES: usize = 160;
const BYTES_PER_FEC_FRAME: usize = 18;
const DIBITS_PER_FRAME: usize = 72;
const SAMPLE_RATE: u32 = 8_000;

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
        /// encode throughput is ~1.8–15 s/frame, so 300 frames ≈
        /// 9–75 min wall time on the chip side.
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
        /// Enable the experimental Viterbi pitch tracker (replaces
        /// §0.3.4–§0.3.6 with an online Viterbi over the Eq. 11 grid).
        #[arg(long)]
        viterbi_pitch: bool,
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
        /// EMA smoothing coefficient on per-harmonic M̂_l. New = α·M̂_l
        /// + (1−α)·prev. Default 1.0 = no smoothing. Reset on L change
        /// or silence frame. 0.3-0.7 is typical for perceptual stability
        /// without destroying transients.
        #[arg(long, default_value_t = 1.0)]
        amp_ema: f32,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum VuvOverride {
    None,
    AllVoiced,
    AllUnvoiced,
    Invert,
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
            viterbi_pitch,
            amp_scale,
            vuv_override,
            amp_ema,
        } => cmd_ab_matrix(
            &pcm,
            &out_dir,
            frames,
            &chip_host,
            skip_chip,
            silence_dispatch,
            pitch_silence_override,
            viterbi_pitch,
            amp_scale,
            vuv_override,
            amp_ema,
        ),
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
    viterbi_pitch: bool,
    amp_scale: f32,
    vuv_override: VuvOverride,
    amp_ema: f32,
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
    let our_bits = our_encode(pcm, silence_dispatch, pitch_silence_override, viterbi_pitch, amp_scale, vuv_override, amp_ema)?;
    let our_enc_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "our encode: {} frames in {:.2} s ({:.1} ms/frame)",
        our_bits.len() / BYTES_PER_FEC_FRAME,
        our_enc_secs,
        1000.0 * our_enc_secs / (our_bits.len() / BYTES_PER_FEC_FRAME).max(1) as f64
    );

    let t0 = Instant::now();
    let our_enc_our_dec = our_decode(&our_bits);
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
        let chip_enc_our_dec = our_decode(&chip_bits);
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
    viterbi_pitch: bool,
    amp_scale: f32,
    vuv_override: VuvOverride,
    amp_ema: f32,
) -> Result<Vec<u8>> {
    let n_frames = pcm.len() / FRAME_SAMPLES;
    let mut state = AnalysisState::new();
    if silence_dispatch {
        state.set_silence_detection(true);
    }
    if pitch_silence_override {
        state.set_pitch_silence_override(true);
    }
    if viterbi_pitch {
        state.set_viterbi_pitch(true);
    }
    let mut decoder_state_for_quantize = DecoderState::new();
    let mut out = Vec::with_capacity(n_frames * BYTES_PER_FEC_FRAME);
    // Per-harmonic EMA state. Empty = reset (cold start, silence, or
    // L change). Stored as f32 to match MbeParams amplitudes.
    let ema_enabled = (amp_ema - 1.0).abs() > 1e-6 && amp_ema > 0.0;
    let mut ema_prev: Vec<f32> = Vec::new();
    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let analysis = analysis_encode(frame, &mut state)
            .map_err(|e| anyhow!("analysis error at frame {f}: {e:?}"))?;
        let params = match analysis {
            AnalysisOutput::Voice(p) => p,
            AnalysisOutput::Silence => {
                // Reset EMA so subsequent voice frame re-initializes.
                if ema_enabled {
                    ema_prev.clear();
                }
                MbeParams::silence()
            }
        };
        // Probe-modification pass: amp_scale, vuv_override, amp_ema.
        let needs_rebuild = (amp_scale - 1.0).abs() > 1e-6
            || !matches!(vuv_override, VuvOverride::None)
            || ema_enabled;
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
            let mut amps: Vec<f32> = raw_amps[..l as usize]
                .iter()
                .map(|a| a * amp_scale)
                .collect();
            // EMA smoothing in linear amplitude space. Reset when L
            // changes (per-harmonic state becomes meaningless).
            if ema_enabled {
                if ema_prev.len() != amps.len() {
                    ema_prev = amps.clone();
                } else {
                    for (out_a, prev_a) in amps.iter_mut().zip(ema_prev.iter()) {
                        *out_a = amp_ema * *out_a + (1.0 - amp_ema) * *prev_a;
                    }
                    ema_prev = amps.clone();
                }
            }
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
fn our_decode(fec: &[u8]) -> Vec<i16> {
    let n_frames = fec.len() / BYTES_PER_FEC_FRAME;
    let mut decoder_state = DecoderState::new();
    let mut synth_state = SynthState::new();
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
                continue;
            }
        };
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        let pcm = synthesize_frame(&params, &err, GAMMA_W, &mut synth_state);
        out.extend_from_slice(&pcm);
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
