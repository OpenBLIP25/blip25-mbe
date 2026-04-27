//! Live DVSI chip synthesis-bug bisection harness.
//!
//! Builds a set of minimal synthetic [`MbeParams`] scenarios, runs them
//! through the full-rate IMBE encoder to produce on-wire 18-byte FEC
//! frames, synthesizes each scenario locally via
//! [`synthesize_frame`], and writes both the `.bit` file (for chip
//! replay on `pve`) and the `.local.pcm` file (for later comparison
//! against the chip's output).
//!
//! Usage (conceptual flow; each subcommand only does one piece):
//!
//!   # 1. Generate scenarios locally.
//!   conformance-chip bisect-generate --out /tmp/bisect
//!
//!   # 2. Copy to the chip host (pve).
//!   scp /tmp/bisect/*.bit root@192.168.1.6:/tmp/bisect/
//!
//!   # 3. For each scenario on pve: `dvsi decode <name>.bit <name>.chip.pcm`.
//!   #    Each invocation re-inits the chip so state is clean per scenario.
//!
//!   # 4. Pull the chip outputs back and compare.
//!   scp "root@192.168.1.6:/tmp/bisect/*.chip.pcm" /tmp/bisect/
//!   conformance-chip bisect-compare --dir /tmp/bisect
//!
//! The encoder and the local synthesizer are driven from the same
//! 18-byte FEC bytes that the chip sees, so any PCM divergence isolates
//! the synthesis bug to a specific §1.12.2 V/UV branch.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use blip25_mbe::codecs::mbe_baseline::analysis::{
    encode as analysis_encode, AnalysisOutput, AnalysisState,
};
use blip25_mbe::codecs::mbe_baseline::{synthesize_frame, FrameErrorContext, SynthState, GAMMA_W};
use blip25_mbe::mbe_params::{MbeParams, L_MIN};
use blip25_mbe::imbe_wire::dequantize::{
    band_for_harmonic, dequantize as dequantize_full, quantize as quantize_full, vuv_band_count,
    DecoderState,
};
use blip25_mbe::imbe_wire::frame::{
    decode_frame as decode_full_frame, encode_frame as encode_full_frame,
};
use blip25_mbe::imbe_wire::priority::prioritize as prioritize_full;

const FRAME_SAMPLES: usize = 160;
const BYTES_PER_FEC_FRAME: usize = 18;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate bisection scenarios: `<name>.bit` + `<name>.local.pcm`
    /// + `<name>.meta.txt` into an output directory.
    BisectGenerate {
        /// Output directory (created if missing).
        #[arg(long)]
        out: PathBuf,
    },
    /// Compare a directory's `<name>.local.pcm` files against the
    /// corresponding `<name>.chip.pcm` files fetched from the chip host.
    BisectCompare {
        /// Directory containing both `*.local.pcm` and `*.chip.pcm` files.
        #[arg(long)]
        dir: PathBuf,
    },
    /// Dump per-frame quantized parameters and dequantized MbeParams
    /// for every scenario. Useful for sanity-checking what bits the
    /// chip is actually seeing.
    BisectDebug {
        /// Output directory produced by `bisect-generate`.
        #[arg(long)]
        dir: PathBuf,
    },
    /// Decode a real `.bit` file via our decoder, then *re-encode* each
    /// dequantized MbeParams through our encoder. Writes the resulting
    /// "round-tripped" bytes. Comparing chip output on the original
    /// bytes vs. the round-tripped bytes tells us whether our encoder
    /// produces bits the chip treats equivalently.
    ReencodeVector {
        /// Input .bit file.
        #[arg(long)]
        input: PathBuf,
        /// Output .bit file (round-tripped through our codec).
        #[arg(long)]
        out: PathBuf,
        /// Number of frames to process (default: all).
        #[arg(long)]
        frames: Option<usize>,
    },
    /// Generate amplitude/spectral-scale probe scenarios for
    /// `gap_reports/0001_chip_probe_menu.md`. Writes per-probe `.bit`
    /// files, local PCM, and a manifest; chip-side decode + compare
    /// follows the same flow as `bisect-generate` / `bisect-compare`.
    ProbeGenerate {
        /// Output directory (created if missing).
        #[arg(long)]
        out: PathBuf,
    },
    /// Post-decode chip PCM comparison for `probe-generate` scenarios.
    /// Reports per-probe steady-state RMS / peak statistics that answer
    /// the γ_w / §1.10 / M̃_l questions in gap report 0001.
    ProbeCompare {
        /// Directory containing both `*.local.pcm` and `*.chip.pcm` files
        /// produced by probe-generate + chip decode.
        #[arg(long)]
        dir: PathBuf,
    },
    /// Compare our analysis encoder against the chip's encoder on the
    /// same PCM. Decodes both bitstreams via our wire decoder so the
    /// diff is reported in MbeParams space (L̂, ω̂_0, V/UV mask,
    /// per-harmonic log₂|M̃_l|). Per gap report 0001 §4.
    ///
    /// Inputs:
    ///   `--pcm`         raw LE i16 8 kHz mono PCM (the source signal)
    ///   `--chip-bits`   chip-emitted .bit (produced by `dvsi encode <pcm>`)
    ChipVsOurs {
        /// Source PCM file (LE i16, 8 kHz mono).
        #[arg(long)]
        pcm: PathBuf,
        /// Chip-emitted bits for the same PCM (18 bytes per frame).
        #[arg(long = "chip-bits")]
        chip_bits: PathBuf,
        /// Number of frames to compare (default: full file).
        #[arg(long)]
        frames: Option<usize>,
        /// Skip this many leading frames before reporting (default: 0).
        /// Both sides still process every frame so predictor state stays
        /// consistent — this only suppresses output rows.
        #[arg(long, default_value_t = 0)]
        start: usize,
        /// Frame-alignment offset between chip bits and our encoder
        /// output. Comparison pair becomes `(ours[i], chip[i + offset])`.
        /// Use `-2` for tv-rc reference bits to compensate our encoder's
        /// 3-frame lookahead (it emits the analysis of frame `i-2` when
        /// fed PCM frame `i`). Default `0` for raw frame-by-frame.
        #[arg(long = "align-offset", default_value_t = 0, allow_hyphen_values = true)]
        align_offset: i32,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::BisectGenerate { out } => cmd_bisect_generate(&out),
        Cmd::BisectCompare { dir } => cmd_bisect_compare(&dir),
        Cmd::BisectDebug { dir } => cmd_bisect_debug(&dir),
        Cmd::ReencodeVector { input, out, frames } => cmd_reencode_vector(&input, &out, frames),
        Cmd::ProbeGenerate { out } => cmd_probe_generate(&out),
        Cmd::ProbeCompare { dir } => cmd_probe_compare(&dir),
        Cmd::ChipVsOurs {
            pcm,
            chip_bits,
            frames,
            start,
            align_offset,
        } => cmd_chip_vs_ours(&pcm, &chip_bits, frames, start, align_offset),
    }
}

// ---------------------------------------------------------------------------
// Scenario model
// ---------------------------------------------------------------------------

/// A synthesis-branch bisection scenario: a short sequence of
/// [`MbeParams`] frames designed to exercise a specific §1.12.2
/// per-harmonic V/UV transition.
struct Scenario {
    name: &'static str,
    description: &'static str,
    frames: Vec<MbeParams>,
}

fn cmd_bisect_generate(out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;

    let scenarios = all_scenarios();
    let mut manifest = String::new();
    manifest.push_str("# bisect scenarios (generated by conformance-chip)\n");
    manifest.push_str("# name\tframes\tdescription\n");

    for sc in &scenarios {
        let (fec, local_pcm) =
            encode_and_synth(&sc.frames).with_context(|| format!("scenario {}", sc.name))?;

        let bit_path = out_dir.join(format!("{}.bit", sc.name));
        fs::write(&bit_path, &fec).with_context(|| format!("write {}", bit_path.display()))?;

        let pcm_path = out_dir.join(format!("{}.local.pcm", sc.name));
        let mut pcm_bytes = Vec::with_capacity(local_pcm.len() * 2);
        for s in &local_pcm {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(&pcm_path, &pcm_bytes)
            .with_context(|| format!("write {}", pcm_path.display()))?;

        let meta_path = out_dir.join(format!("{}.meta.txt", sc.name));
        let mut meta = String::new();
        meta.push_str(&format!("scenario:    {}\n", sc.name));
        meta.push_str(&format!("frames:      {}\n", sc.frames.len()));
        meta.push_str(&format!("description: {}\n\n", sc.description));
        for (i, p) in sc.frames.iter().enumerate() {
            let n_voiced = p.voiced_slice().iter().filter(|&&v| v).count();
            let amp_rms = {
                let a = p.amplitudes_slice();
                (a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / (a.len().max(1) as f64))
                    .sqrt()
            };
            meta.push_str(&format!(
                "  frame {i}: ω₀={:.4} ({:.1} Hz), L={}, voiced={}/{}, M_rms={:.1}\n",
                p.omega_0(),
                p.fundamental_hz(),
                p.harmonic_count(),
                n_voiced,
                p.harmonic_count(),
                amp_rms,
            ));
        }
        fs::write(&meta_path, meta).with_context(|| format!("write {}", meta_path.display()))?;

        manifest.push_str(&format!(
            "{}\t{}\t{}\n",
            sc.name,
            sc.frames.len(),
            sc.description,
        ));

        println!(
            "wrote {}.bit ({} bytes) + {}.local.pcm ({} samples)",
            sc.name,
            fec.len(),
            sc.name,
            local_pcm.len(),
        );
    }

    let manifest_path = out_dir.join("manifest.tsv");
    fs::write(&manifest_path, manifest)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!("\nmanifest: {}", manifest_path.display());
    println!(
        "next: scp {}/*.bit root@192.168.1.6:/tmp/bisect/",
        out_dir.display()
    );
    Ok(())
}

/// Encode a scenario through quantize → prioritize → FEC, and in
/// parallel synthesize it locally by feeding the *same* FEC bytes back
/// through decode → dequantize → synthesize.
///
/// The local synthesis shares the chip's input (the 18-byte FEC
/// frames), so comparing local vs chip PCM isolates the synthesizer.
fn encode_and_synth(frames: &[MbeParams]) -> Result<(Vec<u8>, Vec<i16>)> {
    // Encoder predictor state (drives the log-magnitude prediction
    // feedback loop — must see frames in order).
    let mut enc_state = DecoderState::new();

    let mut fec_bytes: Vec<u8> = Vec::with_capacity(frames.len() * BYTES_PER_FEC_FRAME);
    for (i, p) in frames.iter().enumerate() {
        let b = quantize_full(p, &mut enc_state)
            .map_err(|e| anyhow!("frame {i}: quantize failed: {e:?}"))?;
        let u = prioritize_full(&b, p.harmonic_count());
        let dibits = encode_full_frame(&u);
        let packed = pack_dibits_to_bytes(&dibits);
        fec_bytes.extend_from_slice(&packed);
    }

    // Local synth path — decode + dequantize + synthesize, mirroring
    // what the chip does.
    let mut dec_state = DecoderState::new();
    let mut synth_state = SynthState::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(frames.len() * FRAME_SAMPLES);
    for i in 0..frames.len() {
        let fec = &fec_bytes[i * BYTES_PER_FEC_FRAME..(i + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_bytes_to_dibits(fec);
        let imbe = decode_full_frame(&dibits);
        let params = dequantize_full(&imbe.info, &mut dec_state)
            .map_err(|e| anyhow!("frame {i}: dequantize failed: {e:?}"))?;
        // Synthetic scenarios are clean — no FEC errors expected.
        let err = FrameErrorContext::default();
        let samples = synthesize_frame(&params, &err, GAMMA_W, &mut synth_state);
        pcm.extend_from_slice(&samples);
    }

    Ok((fec_bytes, pcm))
}

// ---------------------------------------------------------------------------
// Dibit ↔ byte packing (MSB-first, 2 bits per dibit, 9 dibits per 18-bit run)
// ---------------------------------------------------------------------------

/// Pack 72 dibits into 18 bytes MSB-first. Each dibit's high bit is the
/// first transmitted bit, matching the `unpack_dibits` convention used
/// by the vector-conformance harness and the DVSI `.bit` files.
fn pack_dibits_to_bytes(dibits: &[u8; 72]) -> [u8; 18] {
    let mut out = [0u8; 18];
    let mut bit = 0usize;
    for &d in dibits {
        let hi = (d >> 1) & 1;
        let lo = d & 1;
        out[bit / 8] |= hi << (7 - (bit % 8));
        bit += 1;
        out[bit / 8] |= lo << (7 - (bit % 8));
        bit += 1;
    }
    out
}

/// Inverse of [`pack_dibits_to_bytes`].
fn unpack_bytes_to_dibits(bytes: &[u8]) -> [u8; 72] {
    debug_assert_eq!(bytes.len(), 18);
    let mut out = [0u8; 72];
    for i in 0..72 {
        let hi_bit = 2 * i;
        let lo_bit = 2 * i + 1;
        let hi = (bytes[hi_bit / 8] >> (7 - (hi_bit % 8))) & 1;
        let lo = (bytes[lo_bit / 8] >> (7 - (lo_bit % 8))) & 1;
        out[i] = (hi << 1) | lo;
    }
    out
}

// ---------------------------------------------------------------------------
// Scenario catalog
// ---------------------------------------------------------------------------

fn all_scenarios() -> Vec<Scenario> {
    vec![
        scenario_silence_only(),
        scenario_uv_to_v_single_harm(),
        scenario_uv_to_v_flat(),
        scenario_v_to_uv_single_harm(),
        scenario_vv_ramp_stable(),
        scenario_vv_sum_large_pitch_jump(),
        scenario_long_stable_voiced(),
    ]
}

/// A long, stable, all-voiced sequence. Lets any chip-side AGC / attack
/// envelope settle before we look at the output. Any first-order chip
/// startup transient should decay long before frame 30.
fn scenario_long_stable_voiced() -> Scenario {
    let frames: Vec<MbeParams> = (0..40).map(|_| build_voiced_flat(0.1404, 500.0)).collect();
    Scenario {
        name: "07_long_stable_voiced",
        description: "40 identical voiced frames (ω₀≈0.14, M_l=500 flat). Frames \
             0-20 settle chip-side AGC / attack. Frames 20-39 are the \
             steady-state comparison window.",
        frames,
    }
}

/// Silent frames (baseline — both sides should produce ~silence).
fn scenario_silence_only() -> Scenario {
    Scenario {
        name: "01_silence_only",
        description: "5 silent frames. Baseline for noise-floor comparison.",
        frames: (0..5).map(|_| MbeParams::silence()).collect(),
    }
}

/// Unvoiced → voiced transition with a single-harmonic target (UvV
/// branch).
fn scenario_uv_to_v_single_harm() -> Scenario {
    let mut frames = Vec::new();
    frames.push(MbeParams::silence());
    frames.push(MbeParams::silence());
    // Voiced single-harmonic target at ~179 Hz (ω₀ ≈ 0.1404, L = 20).
    for _ in 0..3 {
        frames.push(build_voiced_single_harmonic(0.1404, 1000.0));
    }
    Scenario {
        name: "02_uv_to_v_single_harm",
        description: "Frames 0-1 silence, 2-4 voiced fundamental (ω₀≈0.1404, \
             M_1=1000). Exercises UvV branch on frame 2; VVRamp on 3-4.",
        frames,
    }
}

/// Unvoiced → voiced with flat amplitude across all harmonics.
fn scenario_uv_to_v_flat() -> Scenario {
    let mut frames = Vec::new();
    frames.push(MbeParams::silence());
    frames.push(MbeParams::silence());
    for _ in 0..3 {
        frames.push(build_voiced_flat(0.1404, 500.0));
    }
    Scenario {
        name: "03_uv_to_v_flat",
        description: "Frames 0-1 silence, 2-4 voiced with uniform M_l=500 across all \
             L=20 harmonics, all bands voiced. Exercises UvV across all \
             harmonics simultaneously.",
        frames,
    }
}

/// Voiced → unvoiced transition (VUv branch).
fn scenario_v_to_uv_single_harm() -> Scenario {
    let mut frames = Vec::new();
    for _ in 0..3 {
        frames.push(build_voiced_single_harmonic(0.1404, 1000.0));
    }
    frames.push(MbeParams::silence());
    frames.push(MbeParams::silence());
    Scenario {
        name: "04_v_to_uv_single_harm",
        description: "Frames 0-2 voiced fundamental (ω₀≈0.1404, M_1=1000), 3-4 \
             silence. Exercises VUv branch on frame 3.",
        frames,
    }
}

/// All-voiced with a tiny pitch drift frame-to-frame (VVRamp branch).
/// Pitch change ratio stays below 0.1, so the synthesizer takes the
/// ramp path at [`crate::codecs::mbe_baseline`]:716.
fn scenario_vv_ramp_stable() -> Scenario {
    // Start just above ω₀=0.14 and drift by ~0.5% per frame — well
    // within the Δω·l/ω < 0.1 threshold for all l ≤ 7 (below the
    // l >= 8 VVSum trigger).
    let mut frames = Vec::new();
    let base = 0.1404_f32;
    for i in 0..5 {
        let omega = base * (1.0 + 0.005 * i as f32);
        frames.push(build_voiced_flat(omega, 500.0));
    }
    Scenario {
        name: "05_vv_ramp_stable",
        description: "5 voiced frames with ~0.5%/frame pitch drift around ω₀≈0.14. \
             Low-order harmonics go through the VVRamp branch; l≥8 \
             harmonics always take VVSum.",
        frames,
    }
}

/// All-voiced with a large pitch jump frame-to-frame (VVSum branch).
fn scenario_vv_sum_large_pitch_jump() -> Scenario {
    let mut frames = Vec::new();
    frames.push(build_voiced_flat(0.08, 500.0));
    frames.push(build_voiced_flat(0.16, 500.0)); // 2x jump → |Δω·l/ω| = 1.0 for l=1
    frames.push(build_voiced_flat(0.08, 500.0));
    frames.push(build_voiced_flat(0.16, 500.0));
    frames.push(build_voiced_flat(0.08, 500.0));
    Scenario {
        name: "06_vv_sum_large_pitch_jump",
        description: "5 voiced frames alternating ω₀ ∈ {0.08, 0.16}. Forces VVSum \
             branch for all harmonics.",
        frames,
    }
}

// ---------------------------------------------------------------------------
// MbeParams builders
// ---------------------------------------------------------------------------

/// Build a voiced MbeParams with a single dominant fundamental and a
/// small positive floor on all other harmonics (to avoid log2(0)
/// pathologies in the predictor).
fn build_voiced_single_harmonic(omega_0: f32, m_fundamental: f32) -> MbeParams {
    let l = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    let k = vuv_band_count(l);
    let voiced = band_uniform_voicing(l, k, |_band| true);
    let mut amps = vec![1.0_f32; l as usize];
    amps[0] = m_fundamental;
    MbeParams::new(omega_0, l, &voiced, &amps).expect("single-harmonic params")
}

/// Build a voiced MbeParams with flat amplitude across all harmonics
/// and all bands voiced.
fn build_voiced_flat(omega_0: f32, m_flat: f32) -> MbeParams {
    let l = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    let k = vuv_band_count(l);
    let voiced = band_uniform_voicing(l, k, |_band| true);
    let amps = vec![m_flat; l as usize];
    MbeParams::new(omega_0, l, &voiced, &amps).expect("flat params")
}

/// Construct a band-uniform per-harmonic voicing vector. The collapsed
/// encoder only keeps the first harmonic's V/UV per band
/// (`collapse_vuv`), so scenarios must be band-uniform to round-trip.
fn band_uniform_voicing<F: Fn(u8) -> bool>(l: u8, k: u8, band_is_voiced: F) -> Vec<bool> {
    (1..=l)
        .map(|lh| band_is_voiced(band_for_harmonic(lh, k)))
        .collect()
}

// ---------------------------------------------------------------------------
// Compare subcommand
// ---------------------------------------------------------------------------

fn cmd_bisect_compare(dir: &Path) -> Result<()> {
    let manifest_path = dir.join("manifest.tsv");
    let manifest_txt = fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "read manifest {} — did bisect-generate run first?",
            manifest_path.display()
        )
    })?;

    println!(
        "{:<34}  {:>7}  {:>7}  {:>7}  {:>7}",
        "scenario (per-frame)", "SNR_dB", "rms_err", "peak", "xcorr",
    );
    println!("{}", "-".repeat(82));

    for line in manifest_txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        let name = cols.next().unwrap_or("");
        let n_frames: usize = cols.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if name.is_empty() || n_frames == 0 {
            continue;
        }

        let local_path = dir.join(format!("{name}.local.pcm"));
        let chip_path = dir.join(format!("{name}.chip.pcm"));
        let local = read_pcm_le_i16(&local_path)?;
        let chip = match read_pcm_le_i16(&chip_path) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:<34}  MISSING ({})",
                    name,
                    e.to_string().lines().next().unwrap_or("")
                );
                continue;
            }
        };

        for f in 0..n_frames {
            let a = f * FRAME_SAMPLES;
            let b = a + FRAME_SAMPLES;
            if b > local.len() || b > chip.len() {
                println!("{:<34}  frame {f}: short file", name);
                break;
            }
            let (snr, rms_err, peak, xcorr) = frame_metrics(&local[a..b], &chip[a..b]);
            println!(
                "{:<34}  {:>+7.2}  {:>7.1}  {:>7}  {:>+7.3}",
                format!("{name}[{f}]"),
                snr,
                rms_err,
                peak,
                xcorr,
            );
        }
    }

    Ok(())
}

fn cmd_bisect_debug(dir: &Path) -> Result<()> {
    for sc in all_scenarios() {
        println!("\n=== {} ===", sc.name);
        let mut enc_state = DecoderState::new();
        let mut dec_state = DecoderState::new();
        for (i, p) in sc.frames.iter().enumerate() {
            let input_amps = p.amplitudes_slice();
            let input_voiced = p.voiced_slice();
            let b = quantize_full(p, &mut enc_state).map_err(|e| anyhow!("quantize: {e:?}"))?;
            let u = prioritize_full(&b, p.harmonic_count());
            let dibits = encode_full_frame(&u);
            let packed = pack_dibits_to_bytes(&dibits);
            let dibits_rt = unpack_bytes_to_dibits(&packed);
            let imbe = decode_full_frame(&dibits_rt);
            let params_rt = dequantize_full(&imbe.info, &mut dec_state)
                .map_err(|e| anyhow!("dequantize: {e:?}"))?;
            let a_in_rms = rms_f32(input_amps);
            let a_rt_rms = rms_f32(params_rt.amplitudes_slice());
            let n_voiced_in = input_voiced.iter().filter(|&&v| v).count();
            let n_voiced_rt = params_rt.voiced_slice().iter().filter(|&&v| v).count();
            let k = vuv_band_count(p.harmonic_count()) as usize;
            println!(
                "frame {i}: b0={} b1=0b{:0k$b} L_in={} L_rt={}  M_in rms={:.1} → M_rt rms={:.1}  voiced {}/{} → {}/{}",
                b[0],
                b[1],
                p.harmonic_count(),
                params_rt.harmonic_count(),
                a_in_rms,
                a_rt_rms,
                n_voiced_in,
                p.harmonic_count(),
                n_voiced_rt,
                params_rt.harmonic_count(),
            );
            // Show first 8 dequantized amplitudes to see the profile.
            let amps: Vec<String> = params_rt
                .amplitudes_slice()
                .iter()
                .take(8)
                .map(|a| format!("{a:6.1}"))
                .collect();
            println!("  M_rt[1..8] = [{}]", amps.join(", "));
            let u_vals: Vec<String> = u.iter().map(|v| format!("{v:03x}")).collect();
            println!("  u₀..û₇ = [{}]", u_vals.join(", "));
            let b_raw: Vec<String> = (0..p.harmonic_count() as usize + 3)
                .map(|i| format!("{:x}", b[i]))
                .collect();
            println!("  b[0..L+2] = [{}]", b_raw.join(","));
        }
    }
    let _ = dir;
    Ok(())
}

fn cmd_reencode_vector(input: &Path, out: &Path, frames: Option<usize>) -> Result<()> {
    let bytes = fs::read(input).with_context(|| format!("read {}", input.display()))?;
    if bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!("input not a whole number of 18-byte frames"));
    }
    let total = bytes.len() / BYTES_PER_FEC_FRAME;
    let n = frames.unwrap_or(total).min(total);

    let mut dec_state = DecoderState::new();
    let mut enc_state = DecoderState::new();
    let mut out_bytes = Vec::with_capacity(n * BYTES_PER_FEC_FRAME);
    let mut exact_match = 0usize;
    let mut diff_bits_total = 0u32;

    for i in 0..n {
        let fec_in = &bytes[i * BYTES_PER_FEC_FRAME..(i + 1) * BYTES_PER_FEC_FRAME];
        let dibits = unpack_bytes_to_dibits(fec_in);
        let imbe = decode_full_frame(&dibits);
        let params = dequantize_full(&imbe.info, &mut dec_state)
            .map_err(|e| anyhow!("frame {i}: dequantize: {e:?}"))?;
        let b = quantize_full(&params, &mut enc_state)
            .map_err(|e| anyhow!("frame {i}: quantize: {e:?}"))?;
        let u = prioritize_full(&b, params.harmonic_count());
        let dibits_out = encode_full_frame(&u);
        let packed = pack_dibits_to_bytes(&dibits_out);

        if packed == fec_in {
            exact_match += 1;
        } else {
            for (a, b) in packed.iter().zip(fec_in.iter()) {
                diff_bits_total += (a ^ b).count_ones();
            }
        }
        out_bytes.extend_from_slice(&packed);
    }

    fs::write(out, &out_bytes).with_context(|| format!("write {}", out.display()))?;
    println!("frames processed: {n}");
    println!("bit-exact match:  {exact_match}/{n}");
    println!(
        "total diff bits:  {diff_bits_total} (across {n} frames = {} bits)",
        n * 144
    );
    println!("wrote: {}", out.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Probe scenarios (§3.1 γ_w, §3.B §1.10 bypass, §3.C predictor, §3.2 M̃_l)
//
// Per gap report 0001 + spec-author feedback 2026-04-17:
// - Each probe is a 20-frame block; measure on frames 10–19 (warm-up bypass).
// - Defeat chip stationary-signal suppressor by alternating between two
//   near-identical MbeParams per frame (jitter factor 0.99 / 1.00).
// - γ_w (§3.1) — all-unvoiced flat-amplitude sweeps at a fixed ω₀, L.
// - §3.B (§1.10 bypass) — two single-harmonic variants at same amplitude:
//   `l ≤ L/8` (§1.10 first branch = identity) vs `l > L/8` (γ-rescaled).
//   If PCM peaks match, §1.10 is correctly energy-preserving.
// - §3.C (predictor state) — scenario 03 from bisect; post-task-#6 the
//   PCM should now match chip's synthesis of the same bits.
// - §3.2 (M̃_l scale) — single-harmonic at l=2, L=20 so 8·l=16 ≤ L.
//   Expected PCM peak per Eq. 127: 2·A. Amplitude sweep confirms linearity.
// ---------------------------------------------------------------------------

/// A probe scenario — owned strings since names are parameterized.
struct ProbeScenario {
    name: String,
    description: String,
    /// Tag identifying which probe section this scenario belongs to.
    probe: &'static str,
    /// The 20 MbeParams frames to drive through encode→decode→synth.
    frames: Vec<MbeParams>,
    /// Expected observable (for the probe-compare stage).
    expect: ProbeExpect,
}

/// What `probe-compare` should compute for this scenario.
#[derive(Clone, Copy, Debug)]
enum ProbeExpect {
    /// Report steady-state PCM RMS (frames 10..20). Used for γ_w.
    Rms { amplitude: f32 },
    /// Report steady-state PCM peak-magnitude (frames 10..20). Used for
    /// M̃_l probe and §1.10 bypass comparisons.
    Peak { harmonic_l: u8, amplitude: f32, l_total: u8 },
}

/// Probe block length. Each block runs the chip on a sustained
/// signal long enough that the trailing measurement is past startup
/// transients.
const PROBE_BLOCK_FRAMES: usize = 20;

/// Build a voiced MbeParams with flat amplitude across harmonics but a
/// tiny jitter so the input is not bit-identical frame-to-frame
/// (defeats the chip's stationary-signal suppressor).
fn jitter_factor(frame_idx: usize) -> f32 {
    if frame_idx & 1 == 0 { 1.00 } else { 0.99 }
}

fn build_unvoiced_flat(omega_0: f32, m_flat: f32) -> MbeParams {
    let l = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    let voiced = vec![false; l as usize];
    let amps = vec![m_flat; l as usize];
    MbeParams::new(omega_0, l, &voiced, &amps).expect("unvoiced-flat params")
}

/// Build a voiced MbeParams with energy on exactly one harmonic `l`
/// (1-indexed), and a small positive floor on the rest. All bands
/// voiced so the target harmonic is voiced per the band-collapse rule.
fn build_voiced_only_at_l(omega_0: f32, target_l: u8, amplitude: f32) -> MbeParams {
    let l_total = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    assert!(
        target_l >= 1 && target_l <= l_total,
        "target_l {target_l} out of range 1..={l_total}"
    );
    let k = vuv_band_count(l_total);
    let voiced = band_uniform_voicing(l_total, k, |_| true);
    let mut amps = vec![1.0_f32; l_total as usize];
    amps[(target_l - 1) as usize] = amplitude;
    MbeParams::new(omega_0, l_total, &voiced, &amps)
        .expect("single-harmonic-at-l params")
}

/// §3.1 γ_w — all-unvoiced flat amplitude. Generate one 20-frame
/// scenario per amplitude in `{10, 30, 100, 300}` at a fixed pitch.
/// Amplitudes > 300 trigger chip-side clipping / mute suppression per
/// the first probe pass; below 10 the Annex E/F midtread rounds to
/// zero. The four points give a linear-fit window of chip vs. local
/// output for γ_w estimation.
fn probe_scenarios_3_1_gamma_w() -> Vec<ProbeScenario> {
    let omega_0 = 0.1404_f32; // ~179 Hz → L ≈ 44
    [10.0_f32, 30.0, 100.0, 300.0]
        .iter()
        .map(|&amp| {
            let mut frames = Vec::with_capacity(PROBE_BLOCK_FRAMES);
            for i in 0..PROBE_BLOCK_FRAMES {
                let m = amp * jitter_factor(i);
                frames.push(build_unvoiced_flat(omega_0, m));
            }
            let name = format!("probe_3_1_gamma_w_amp{:04}", amp as u32);
            let description = format!(
                "§3.1 γ_w: 20 frames all-unvoiced, ω₀={omega_0:.4}, M_l={amp} (jitter 0.99/1.00)"
            );
            ProbeScenario {
                name,
                description,
                probe: "3.1",
                frames,
                expect: ProbeExpect::Rms { amplitude: amp },
            }
        })
        .collect()
}

/// §3.B §1.10 bypass check — two single-harmonic scenarios at the same
/// amplitude but different `l` / `L` configurations:
///   (a) `l = 2`, `L = 44` (via `ω₀ = 0.1404`) → `8·l = 16 ≤ L` → §1.10
///       first branch is identity on this harmonic.
///   (b) `l = 15`, `L = 44` (same ω₀) → `8·l = 120 > L` → §1.10 fires.
/// Using the **same L** for both sides pins the confound to the
/// low-l-vs-high-l §1.10 branch. Both harmonics stay well below Nyquist
/// (l=15 → 15·0.1404 ≈ 2.11 rad/sample ≈ 2.7 kHz at 8 kHz fs).
///
/// Amplitudes kept conservative (≤ 400) to avoid the chip's clipping-
/// suppression behavior observed in the first probe pass.
fn probe_scenarios_3_b_bypass() -> Vec<ProbeScenario> {
    let omega_0 = 0.1404_f32; // L = 44
    let l_total = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    let target_l_low = 2_u8; // 8·l = 16 ≤ 44 → §1.10 identity
    let target_l_high = 15_u8; // 8·l = 120 > 44 → §1.10 fires
    assert!(target_l_high <= l_total, "l_high in range");
    let mut out = Vec::new();
    for &amp in &[100.0_f32, 400.0] {
        for (tag, target_l) in &[("lowL", target_l_low), ("highL", target_l_high)] {
            let mut frames = Vec::with_capacity(PROBE_BLOCK_FRAMES);
            for i in 0..PROBE_BLOCK_FRAMES {
                let m = amp * jitter_factor(i);
                frames.push(build_voiced_only_at_l(omega_0, *target_l, m));
            }
            out.push(ProbeScenario {
                name: format!("probe_3_b_{tag}_amp{:04}", amp as u32),
                description: format!(
                    "§3.B {tag}: l={target_l}/{l_total} ({} §1.10), amp={amp}",
                    if *tag == "lowL" { "identity" } else { "fires" },
                ),
                probe: "3.B",
                frames,
                expect: ProbeExpect::Peak {
                    harmonic_l: *target_l,
                    amplitude: amp,
                    l_total,
                },
            });
        }
    }
    out
}

/// §3.C predictor-state verification — the post-task-#6 version of
/// scenario 03 (silence→voiced flat). Before task #6, this scenario
/// had catastrophic M̃_l divergence frame-over-frame; after the fix the
/// encoder/decoder states stay in lockstep. The chip PCM comparison
/// shouldn't reveal any residual predictor divergence now; if it does,
/// something orthogonal to our just-fixed bug is active.
fn probe_scenario_3_c_predictor() -> ProbeScenario {
    // Reuse the 20-frame pattern: 2 silence preroll → 18 voiced flat.
    let omega_0 = 0.1404_f32;
    let mut frames = Vec::with_capacity(PROBE_BLOCK_FRAMES);
    frames.push(MbeParams::silence());
    frames.push(MbeParams::silence());
    for i in 2..PROBE_BLOCK_FRAMES {
        let m = 500.0 * jitter_factor(i);
        frames.push(build_voiced_flat(omega_0, m));
    }
    ProbeScenario {
        name: "probe_3_c_predictor_state".to_string(),
        description: "§3.C: silence→voiced flat (ω₀=0.1404, M_l=500). Post-task-#6, the \
             encoder's predictor tracks what the decoder reconstructs; \
             chip comparison should be clean."
            .to_string(),
        probe: "3.C",
        frames,
        expect: ProbeExpect::Rms { amplitude: 500.0 },
    }
}

/// §3.2 M̃_l voiced scale — target `l=2, L=20`, `8·l=16 ≤ L` so §1.10
/// is identity. Expected steady-state PCM peak at harmonic l·ω₀ ≈ 2·A
/// per Eq. 127 (the factor-of-2 in the synthesizer sum). Amplitude
/// sweep verifies linearity.
///
/// Amplitudes kept in the chip's responsive range (≤ 1600) based on
/// the first probe pass — above that the chip mutes the output for
/// synthetic single-harmonic inputs.
fn probe_scenarios_3_2_m_tilde() -> Vec<ProbeScenario> {
    let omega_0 = 2.0 * std::f32::consts::PI / 20.0; // L = 20
    let l_total = MbeParams::harmonic_count_for(omega_0).max(L_MIN);
    let target_l = 2_u8;
    [100.0_f32, 400.0, 1600.0]
        .iter()
        .map(|&amp| {
            let mut frames = Vec::with_capacity(PROBE_BLOCK_FRAMES);
            for i in 0..PROBE_BLOCK_FRAMES {
                let m = amp * jitter_factor(i);
                frames.push(build_voiced_only_at_l(omega_0, target_l, m));
            }
            ProbeScenario {
                name: format!("probe_3_2_mtilde_amp{:04}", amp as u32),
                description: format!(
                    "§3.2 M̃_l: l={target_l}/{l_total} (§1.10 identity), amp={amp}; \
                     expected PCM peak ≈ 2·amp per Eq. 127"
                ),
                probe: "3.2",
                frames,
                expect: ProbeExpect::Peak {
                    harmonic_l: target_l,
                    amplitude: amp,
                    l_total,
                },
            }
        })
        .collect()
}

fn all_probe_scenarios() -> Vec<ProbeScenario> {
    let mut scenarios = Vec::new();
    scenarios.extend(probe_scenarios_3_1_gamma_w());
    scenarios.extend(probe_scenarios_3_b_bypass());
    scenarios.push(probe_scenario_3_c_predictor());
    scenarios.extend(probe_scenarios_3_2_m_tilde());
    scenarios
}

fn cmd_probe_generate(out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;

    let scenarios = all_probe_scenarios();
    let mut manifest = String::new();
    manifest.push_str("# probe scenarios (generated by conformance-chip)\n");
    manifest.push_str("# name\tframes\tprobe\tdescription\n");

    for sc in &scenarios {
        let (fec, local_pcm) =
            encode_and_synth(&sc.frames).with_context(|| format!("scenario {}", sc.name))?;

        fs::write(out_dir.join(format!("{}.bit", sc.name)), &fec)
            .with_context(|| format!("write {}.bit", sc.name))?;

        let mut pcm_bytes = Vec::with_capacity(local_pcm.len() * 2);
        for s in &local_pcm {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(out_dir.join(format!("{}.local.pcm", sc.name)), &pcm_bytes)
            .with_context(|| format!("write {}.local.pcm", sc.name))?;

        // Per-probe metadata (expectations as a machine-readable line
        // that probe-compare re-parses).
        let expect_line = match sc.expect {
            ProbeExpect::Rms { amplitude } => format!("EXPECT rms amplitude={amplitude}"),
            ProbeExpect::Peak { harmonic_l, amplitude, l_total } => format!(
                "EXPECT peak harmonic_l={harmonic_l} amplitude={amplitude} l_total={l_total}"
            ),
        };
        let meta = format!(
            "scenario:    {name}\nframes:      {nframes}\nprobe:       {probe}\n\
             description: {desc}\n{expect}\n",
            name = sc.name,
            nframes = sc.frames.len(),
            probe = sc.probe,
            desc = sc.description,
            expect = expect_line,
        );
        fs::write(out_dir.join(format!("{}.meta.txt", sc.name)), meta)
            .with_context(|| format!("write {}.meta.txt", sc.name))?;

        manifest.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            sc.name,
            sc.frames.len(),
            sc.probe,
            sc.description,
        ));

        println!(
            "wrote {}.bit ({} bytes) + {}.local.pcm ({} samples) [probe {}]",
            sc.name,
            fec.len(),
            sc.name,
            local_pcm.len(),
            sc.probe,
        );
    }

    fs::write(out_dir.join("manifest.tsv"), manifest)
        .with_context(|| format!("write {}/manifest.tsv", out_dir.display()))?;
    println!(
        "\nnext: scp {}/*.bit root@192.168.1.6:/tmp/probe/\n\
         then: ssh root@192.168.1.6 'for f in /tmp/probe/*.bit; do dvsi decode $f ${{f%.bit}}.chip.pcm; done'\n\
         then: scp 'root@192.168.1.6:/tmp/probe/*.chip.pcm' {}/ && probe-compare --dir {}",
        out_dir.display(),
        out_dir.display(),
        out_dir.display(),
    );
    Ok(())
}

/// Per-frame RMS, 20 values.
fn per_frame_rms(pcm: &[i16]) -> Vec<f64> {
    (0..PROBE_BLOCK_FRAMES)
        .map(|i| {
            let a = i * FRAME_SAMPLES;
            let b = a + FRAME_SAMPLES;
            if b > pcm.len() {
                return 0.0;
            }
            let win = &pcm[a..b];
            let sum_sq: f64 = win.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (sum_sq / FRAME_SAMPLES as f64).sqrt()
        })
        .collect()
}

/// Per-frame peak, 20 values.
fn per_frame_peak(pcm: &[i16]) -> Vec<i32> {
    (0..PROBE_BLOCK_FRAMES)
        .map(|i| {
            let a = i * FRAME_SAMPLES;
            let b = a + FRAME_SAMPLES;
            if b > pcm.len() {
                return 0;
            }
            pcm[a..b].iter().map(|&x| (x as i32).abs()).max().unwrap_or(0)
        })
        .collect()
}

/// Max per-frame RMS over the full block. Robust to both chip-side
/// AGC ramp (peak reached late) and chip-side mute suppression
/// (peak reached early before muting).
fn max_frame_rms(pcm: &[i16]) -> f64 {
    per_frame_rms(pcm)
        .into_iter()
        .fold(0.0_f64, |acc, x| acc.max(x))
}

/// Max per-frame peak magnitude over the full block.
fn max_frame_peak(pcm: &[i16]) -> i32 {
    per_frame_peak(pcm).into_iter().max().unwrap_or(0)
}

fn cmd_probe_compare(dir: &Path) -> Result<()> {
    let manifest_path = dir.join("manifest.tsv");
    let manifest_txt = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;

    println!(
        "{:<36}  {:>5}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>8}",
        "scenario", "probe",
        "loc_rms_mx", "chp_rms_mx", "rms_r",
        "loc_pk_mx", "chp_pk_mx", "pk_r",
    );
    println!("{}", "-".repeat(108));

    for line in manifest_txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        let name = cols.next().unwrap_or("");
        let _n = cols.next();
        let probe = cols.next().unwrap_or("");
        if name.is_empty() {
            continue;
        }

        let local = read_pcm_le_i16(&dir.join(format!("{name}.local.pcm")))?;
        let chip = match read_pcm_le_i16(&dir.join(format!("{name}.chip.pcm"))) {
            Ok(v) => v,
            Err(_) => {
                println!("{:<36}  {:>5}  MISSING chip.pcm", name, probe);
                continue;
            }
        };
        // Report max-per-frame values; robust to chip AGC ramp and to
        // mute-after-N-frames clipping rejection.
        let local_rms = max_frame_rms(&local);
        let chip_rms = max_frame_rms(&chip);
        let local_pk = max_frame_peak(&local);
        let chip_pk = max_frame_peak(&chip);
        let rms_ratio = if local_rms > 0.1 { chip_rms / local_rms } else { 0.0 };
        let pk_ratio = if local_pk > 0 {
            chip_pk as f64 / local_pk as f64
        } else {
            0.0
        };

        println!(
            "{:<36}  {:>5}  {:>10.1}  {:>10.1}  {:>8.3}  {:>10}  {:>10}  {:>8.3}",
            name, probe, local_rms, chip_rms, rms_ratio, local_pk, chip_pk, pk_ratio,
        );
    }

    Ok(())
}

fn rms_f32(v: &[f32]) -> f64 {
    let n = v.len().max(1) as f64;
    (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n).sqrt()
}

fn read_pcm_le_i16(path: &Path) -> Result<Vec<i16>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 2 != 0 {
        return Err(anyhow!(
            "{}: not a whole number of i16 samples",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Per-frame SNR (chip as reference), RMS error, peak error, and
/// normalized cross-correlation. `xcorr` is in [-1, +1]; ~+1 means
/// our synth is shape-correct, near-0 means structural divergence.
fn frame_metrics(ours: &[i16], chip: &[i16]) -> (f64, f64, i32, f64) {
    let n = ours.len().min(chip.len()) as f64;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_chip = 0f64;
    let mut sum_sq_ours = 0f64;
    let mut sum_prod = 0f64;
    let mut peak = 0i32;
    for i in 0..ours.len().min(chip.len()) {
        let a = ours[i] as f64;
        let b = chip[i] as f64;
        let e = a - b;
        sum_sq_err += e * e;
        sum_sq_chip += b * b;
        sum_sq_ours += a * a;
        sum_prod += a * b;
        let abs_e = (ours[i] as i32 - chip[i] as i32).abs();
        if abs_e > peak {
            peak = abs_e;
        }
    }
    let snr = if sum_sq_chip > 0.0 && sum_sq_err > 0.0 {
        10.0 * (sum_sq_chip / sum_sq_err).log10()
    } else if sum_sq_err == 0.0 && sum_sq_chip > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let rms_err = (sum_sq_err / n.max(1.0)).sqrt();
    let denom = (sum_sq_ours * sum_sq_chip).sqrt();
    let xcorr = if denom > 0.0 { sum_prod / denom } else { 0.0 };
    (snr, rms_err, peak, xcorr)
}

// ---------------------------------------------------------------------------
// Chip-encode vs our-encode comparison (MbeParams-level diff).
//
// Both bit streams (chip's and ours) are decoded through our own
// imbe_wire wire decoder so any mismatch is attributable to the
// analysis-encoder side, not to a wire-decoder discrepancy. Each side
// gets its own DecoderState because the predictor evolves with the
// bits actually emitted on that side.
// ---------------------------------------------------------------------------

/// Per-frame slot: either a recovered MbeParams or the reason the slot
/// couldn't produce one.
enum FrameSlot {
    Voice(MbeParams),
    Silence,
    /// Wire-decoder rejected the bits (chip emitted control / reserved).
    /// Carries the raw u₀ for diagnostics.
    Ctrl(u16),
}

fn cmd_chip_vs_ours(
    pcm_path: &Path,
    chip_bits_path: &Path,
    frames_arg: Option<usize>,
    start: usize,
    align_offset: i32,
) -> Result<()> {
    let pcm = read_pcm_le_i16(pcm_path)?;
    let chip_bytes = fs::read(chip_bits_path)
        .with_context(|| format!("read {}", chip_bits_path.display()))?;
    if chip_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "{}: not a whole number of {}-byte frames",
            chip_bits_path.display(),
            BYTES_PER_FEC_FRAME
        ));
    }

    let n_pcm_frames = pcm.len() / FRAME_SAMPLES;
    let n_bit_frames = chip_bytes.len() / BYTES_PER_FEC_FRAME;
    let total = n_pcm_frames.min(n_bit_frames);
    let n = frames_arg.unwrap_or(total).min(total);
    if start >= n {
        return Err(anyhow!(
            "--start {start} >= comparable frames {n} (pcm={n_pcm_frames}, chip-bits={n_bit_frames})"
        ));
    }
    if n_pcm_frames != n_bit_frames {
        eprintln!(
            "warning: pcm has {n_pcm_frames} frames, chip-bits has {n_bit_frames} — comparing the first {total}"
        );
    }

    let mut analysis = AnalysisState::new();
    let mut chip_dec = DecoderState::new();
    let mut ours_dec = DecoderState::new();
    let mut ours_enc = DecoderState::new();

    // Process every frame in order so both predictor states evolve
    // correctly. Comparison happens in a second pass so `--align-offset`
    // can pair `ours[i]` with `chip[i + offset]`.
    let mut ours_slots: Vec<FrameSlot> = Vec::with_capacity(n);
    let mut chip_slots: Vec<FrameSlot> = Vec::with_capacity(n);

    for i in 0..n {
        let pcm_frame = &pcm[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES];
        let chip_frame = &chip_bytes[i * BYTES_PER_FEC_FRAME..(i + 1) * BYTES_PER_FEC_FRAME];

        // Chip side. Control frames (b̂_0 ∈ 120..127 erasure / silence /
        // tone, or reserved 208..255) fail our voice dequantizer; treat
        // those as a non-comparable slot, not a hard abort.
        let chip_dibits = unpack_bytes_to_dibits(chip_frame);
        let chip_imbe = decode_full_frame(&chip_dibits);
        chip_slots.push(match dequantize_full(&chip_imbe.info, &mut chip_dec) {
            Ok(p) => FrameSlot::Voice(p),
            Err(_) => FrameSlot::Ctrl(chip_imbe.info[0]),
        });

        // Our side. Silence / preroll → no MbeParams to compare.
        let our_out = analysis_encode(pcm_frame, &mut analysis)
            .map_err(|e| anyhow!("frame {i}: analysis encode: {e:?}"))?;
        let our_params = match our_out {
            AnalysisOutput::Voice(p) => p,
            AnalysisOutput::Silence => {
                ours_slots.push(FrameSlot::Silence);
                continue;
            }
        };

        // Round-trip our params through our wire encoder → decoder so
        // the comparison is apples-to-apples (chip params went through
        // dequantize too).
        let b = quantize_full(&our_params, &mut ours_enc)
            .map_err(|e| anyhow!("frame {i}: our quantize: {e:?}"))?;
        let u = prioritize_full(&b, our_params.harmonic_count());
        let our_dibits = encode_full_frame(&u);
        let our_packed = pack_dibits_to_bytes(&our_dibits);
        let our_dibits_rt = unpack_bytes_to_dibits(&our_packed);
        let ours_imbe = decode_full_frame(&our_dibits_rt);
        let ours_params = dequantize_full(&ours_imbe.info, &mut ours_dec)
            .map_err(|e| anyhow!("frame {i}: ours dequantize: {e:?}"))?;
        ours_slots.push(FrameSlot::Voice(ours_params));
    }

    println!(
        "{:>5}  {:<8}  {:>3} {:>3} {:>3}  {:>7} {:>7} {:>+7}  {:>4} {:>4} {:>3}  {:>9} {:>9}",
        "frame", "src",
        "Lc", "Lo", "dL",
        "ω0_c", "ω0_o", "dcent",
        "Vbc", "Vbo", "dV",
        "rms_dlog2", "pk_dlog2",
    );
    println!("{}", "-".repeat(98));

    let mut compared = 0usize;
    let mut l_match = 0usize;
    let mut sum_abs_dl = 0u64;
    let mut sum_abs_dcent = 0.0_f64;
    let mut omega_within_25c = 0usize;
    let mut vmask_match = 0usize;
    let mut sum_rms_dlog2 = 0.0_f64;
    let mut sum_pk_dlog2 = 0.0_f64;

    for i in start..n {
        let chip_idx = i as i32 + align_offset;
        if chip_idx < 0 || (chip_idx as usize) >= n {
            continue;
        }
        let chip_idx = chip_idx as usize;
        match (&ours_slots[i], &chip_slots[chip_idx]) {
            (FrameSlot::Silence, _) => println!("{:>5}  {:<8}", i, "silence"),
            (FrameSlot::Ctrl(_), _) => unreachable!("ours never produces Ctrl"),
            (_, FrameSlot::Ctrl(u0)) => {
                println!("{:>5}  {:<8}  chip ctrl  (u₀=0x{:03x})", i, "ctrl", u0)
            }
            (_, FrameSlot::Silence) => unreachable!("chip slot is Voice or Ctrl"),
            (FrameSlot::Voice(ours_p), FrameSlot::Voice(chip_p)) => {
                let m = mbe_params_diff(chip_p, ours_p);
                println!(
                    "{:>5}  {:<8}  {:>3} {:>3} {:>+3}  {:>7.4} {:>7.4} {:>+7.0}  {:>4} {:>4} {:>+3}  {:>9.3} {:>9.3}",
                    i, "voice",
                    m.l_chip, m.l_ours, (m.l_ours as i32 - m.l_chip as i32),
                    m.omega_chip, m.omega_ours, m.dcents,
                    m.v_bands_chip, m.v_bands_ours,
                    (m.v_bands_ours as i32 - m.v_bands_chip as i32),
                    m.rms_dlog2, m.peak_dlog2,
                );
                compared += 1;
                if m.l_chip == m.l_ours {
                    l_match += 1;
                }
                sum_abs_dl += (m.l_ours as i32 - m.l_chip as i32).unsigned_abs() as u64;
                sum_abs_dcent += m.dcents.abs();
                if m.dcents.abs() <= 25.0 {
                    omega_within_25c += 1;
                }
                if m.v_bands_chip == m.v_bands_ours && m.vmask_exact {
                    vmask_match += 1;
                }
                sum_rms_dlog2 += m.rms_dlog2;
                sum_pk_dlog2 += m.peak_dlog2;
            }
        }
    }

    println!("{}", "-".repeat(98));
    if compared == 0 {
        println!("no voice frames in window — nothing to aggregate");
        return Ok(());
    }
    let f = compared as f64;
    println!(
        "aggregate over {compared} voice frames (start={start}, n={n}, align_offset={align_offset}):\n  \
         L exact:        {l_match}/{compared} ({:.1}%)   mean |ΔL| = {:.2}\n  \
         |Δω₀| (cents):  mean = {:.1}     within ±25 ¢: {omega_within_25c}/{compared} ({:.1}%)\n  \
         V/UV exact:     {vmask_match}/{compared} ({:.1}%)\n  \
         log₂|M̃_l| diff: mean RMS = {:.3}     mean peak = {:.3}",
        100.0 * l_match as f64 / f,
        sum_abs_dl as f64 / f,
        sum_abs_dcent / f,
        100.0 * omega_within_25c as f64 / f,
        100.0 * vmask_match as f64 / f,
        sum_rms_dlog2 / f,
        sum_pk_dlog2 / f,
    );
    Ok(())
}

/// Per-frame MbeParams diff between the chip's and our decoded params.
struct ParamsDiff {
    l_chip: u8,
    l_ours: u8,
    omega_chip: f32,
    omega_ours: f32,
    dcents: f64,
    v_bands_chip: u8,
    v_bands_ours: u8,
    /// True iff the per-band V/UV vector is bit-exact across the
    /// shared band count (min of the two).
    vmask_exact: bool,
    rms_dlog2: f64,
    peak_dlog2: f64,
}

fn mbe_params_diff(chip: &MbeParams, ours: &MbeParams) -> ParamsDiff {
    let l_chip = chip.harmonic_count();
    let l_ours = ours.harmonic_count();
    let omega_chip = chip.omega_0();
    let omega_ours = ours.omega_0();
    let dcents = if omega_chip > 0.0 && omega_ours > 0.0 {
        1200.0 * ((omega_ours as f64) / (omega_chip as f64)).log2()
    } else {
        0.0
    };

    let k_chip = vuv_band_count(l_chip);
    let k_ours = vuv_band_count(l_ours);
    let v_bands_chip = collapse_voiced_bands(chip.voiced_slice(), k_chip);
    let v_bands_ours = collapse_voiced_bands(ours.voiced_slice(), k_ours);

    let mut vmask_exact = k_chip == k_ours;
    if vmask_exact {
        // Both have the same band count; check per-band agreement on
        // the first harmonic of each band (the wire encoder's
        // collapse rule).
        for band in 1..=k_chip {
            let lh = first_harmonic_of_band(band, k_chip).min(l_chip);
            if chip.voiced(lh) != ours.voiced(lh) {
                vmask_exact = false;
                break;
            }
        }
    }

    // Per-harmonic log₂ amplitude diff over the overlap.
    let l_overlap = l_chip.min(l_ours);
    let chip_amps = chip.amplitudes_slice();
    let ours_amps = ours.amplitudes_slice();
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut counted = 0u32;
    for l in 1..=l_overlap {
        let i = (l - 1) as usize;
        let mc = chip_amps[i] as f64;
        let mo = ours_amps[i] as f64;
        if mc <= 0.0 || mo <= 0.0 {
            continue;
        }
        let d = (mo / mc).log2();
        sum_sq += d * d;
        let ad = d.abs();
        if ad > peak {
            peak = ad;
        }
        counted += 1;
    }
    let rms_dlog2 = if counted > 0 {
        (sum_sq / counted as f64).sqrt()
    } else {
        0.0
    };
    ParamsDiff {
        l_chip,
        l_ours,
        omega_chip,
        omega_ours,
        dcents,
        v_bands_chip,
        v_bands_ours,
        vmask_exact,
        rms_dlog2,
        peak_dlog2: peak,
    }
}

/// Count voiced bands per the wire encoder's band-collapse rule:
/// band `k` is voiced iff its first harmonic is voiced.
fn collapse_voiced_bands(voiced: &[bool], k: u8) -> u8 {
    let mut n = 0u8;
    let l_total = voiced.len() as u8;
    for band in 1..=k {
        let lh = first_harmonic_of_band(band, k).min(l_total);
        if lh >= 1 && voiced[(lh - 1) as usize] {
            n += 1;
        }
    }
    n
}

/// First harmonic index belonging to the given 1-indexed band, given
/// the total band count `k`. Bands are 3-harmonic blocks per the
/// `band_for_harmonic` definition.
fn first_harmonic_of_band(band: u8, _k: u8) -> u8 {
    // band_for_harmonic(l, k) = ((l - 1) / 3) + 1, capped at k. So
    // band b's first harmonic is (b - 1)*3 + 1.
    (band.saturating_sub(1)) * 3 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dibit_byte_roundtrip() {
        let mut d = [0u8; 72];
        for (i, v) in d.iter_mut().enumerate() {
            *v = (i % 4) as u8;
        }
        let bytes = pack_dibits_to_bytes(&d);
        let d2 = unpack_bytes_to_dibits(&bytes);
        assert_eq!(d, d2);
    }

    #[test]
    fn pack_first_byte_is_first_4_dibits_msb_first() {
        // First 4 dibits → 8 bits in byte 0. A single-bit dibit at
        // position 0 (dibit value 2 = 10b) should set bit 7 of byte 0.
        let mut d = [0u8; 72];
        d[0] = 2; // '10'
        let bytes = pack_dibits_to_bytes(&d);
        assert_eq!(bytes[0], 0b1000_0000);
    }

    #[test]
    fn silence_scenario_encodes_cleanly() {
        let sc = scenario_silence_only();
        let (fec, pcm) = encode_and_synth(&sc.frames).unwrap();
        assert_eq!(fec.len(), sc.frames.len() * BYTES_PER_FEC_FRAME);
        assert_eq!(pcm.len(), sc.frames.len() * FRAME_SAMPLES);
    }

    #[test]
    fn uv_to_v_scenario_produces_nonzero_output() {
        let sc = scenario_uv_to_v_single_harm();
        let (_fec, pcm) = encode_and_synth(&sc.frames).unwrap();
        // Voiced frames (2-4) should carry real signal.
        let voiced_range = &pcm[2 * FRAME_SAMPLES..];
        let energy: f64 = voiced_range.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0, "voiced frames produced zero PCM");
    }
}
