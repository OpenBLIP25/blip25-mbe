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
//!   into [`blip25_mbe::imbe_frames::full_rate::decode_fullrate_frame`].
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
    FrameErrorContext, GAMMA_W, SynthState, synthesize_frame,
};
use blip25_mbe::fec::{golay_23_12_encode, hamming_15_11_encode};
use blip25_mbe::imbe_frames::dequantize::{
    DecoderState, dequantize_fullrate, quantize_fullrate,
};
use blip25_mbe::imbe_frames::fec::{deinterleave_fullrate, modulation_masks_fullrate};
use blip25_mbe::imbe_frames::dequantize_halfrate::{
    HalfrateDecoded, HalfrateDecoderState, decode_halfrate_to_params, dequantize_halfrate,
    quantize_halfrate,
};
use blip25_mbe::imbe_frames::full_rate::{
    INFO_WIDTHS, ImbeFrame, decode_fullrate_frame,
};
use blip25_mbe::imbe_frames::half_rate::decode_halfrate_frame;
use blip25_mbe::imbe_frames::priority::{
    deprioritize_fullrate, prioritize_fullrate,
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
    /// Half-rate bit-level roundtrip: decode each `r39/<name>.bit`
    /// frame to `MbeParams`, re-encode via `quantize_halfrate`, and
    /// compare the recovered `û` vectors to the original. Pitch
    /// (`b̂₀`) and V/UV row-0 endpoints should roundtrip bit-exact;
    /// gain and spectral params may drift (see `dequantize_halfrate`
    /// tests for the known stability boundaries).
    HalfrateRoundtrip {
        /// Vector name (e.g. `alert`).
        name: String,
    },
    /// Half-rate equivalent of `decode-pcm`. Reads
    /// `r39/<name>.bit` (rate index 39 = 3600 bps AMBE+2 per the
    /// USB-3000 manual §3.3.4), runs the full half-rate pipeline:
    /// dibits → AmbeFrame → MbeParams → PCM via the shared §1.12
    /// synthesizer, and compares to `r39/<name>.pcm`.
    DecodePcmHalfrate {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Override γ_w (default: spec value 146.643269).
        #[arg(long)]
        gamma_w: Option<f64>,
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
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Cmd::Compare { name, rc, stop_on_first } => {
            cmd_compare(&args.vectors, &name, rc, stop_on_first)
        }
        Cmd::PnDiag { name, rc, frame } => cmd_pn_diag(&args.vectors, &name, rc, frame),
        Cmd::Roundtrip { name, rc } => cmd_roundtrip(&args.vectors, &name, rc),
        Cmd::DecodePcm { name, rc, gamma_w, m_scale } => cmd_decode_pcm(
            &args.vectors,
            &name,
            rc,
            gamma_w.unwrap_or(GAMMA_W),
            m_scale,
        ),
        Cmd::DecodePcmHalfrate { name, gamma_w } => {
            cmd_decode_pcm_halfrate(&args.vectors, &name, gamma_w.unwrap_or(GAMMA_W))
        }
        Cmd::HalfrateRoundtrip { name } => cmd_halfrate_roundtrip(&args.vectors, &name),
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
        let imbe = decode_fullrate_frame(&dibits);
        let params = match dequantize_fullrate(&imbe.info, &mut decoder_state) {
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
        let params = match dequantize_fullrate(&u_orig, &mut state_dec) {
            Ok(p) => p,
            Err(_) => {
                bad_pitch += 1;
                continue;
            }
        };

        // Re-encode using the encoder state (which evolves in lockstep).
        let b_back = match quantize_fullrate(&params, &mut state_enc) {
            Ok(b) => b,
            Err(_) => {
                bad_pitch += 1;
                continue;
            }
        };
        let u_back = prioritize_fullrate(&b_back, params.harmonic_count());

        // Per-vector match accounting.
        for i in 0..8 {
            if u_orig[i] == u_back[i] {
                vec_matches[i] += 1;
            }
        }

        // Pitch (b̂₀) and V/UV (b̂₁): pull from the original u via
        // deprioritize so we can compare at the parameter level too.
        let b_orig = deprioritize_fullrate(&u_orig, params.harmonic_count());
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
    let c_tilde = deinterleave_fullrate(&dibits);
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
    let our_mask = modulation_masks_fullrate(u_expected[0]);

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

fn cmd_decode_pcm_halfrate(root: &Path, name: &str, gamma_w: f64) -> Result<()> {
    // Half-rate DVSI vectors live under tv-std/tv/r39/ (rate index 39
    // per USB-3000 manual §3.3.4 = 3600 bps AMBE+2).
    let dir = root.join("tv-std").join("tv").join("r39");
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

    println!("vector:        {name} (half-rate, r39)");
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

    let mut dec_state = HalfrateDecoderState::new();
    let mut synth_state = SynthState::new();
    let mut pcm_out: Vec<i16> = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    let mut voice_frames = 0usize;
    let mut tone_frames = 0usize;
    let mut erasure_frames = 0usize;

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_halfrate_frame(&dibits);
        let err = FrameErrorContext {
            epsilon_0: ambe.errors[0],
            epsilon_4: 0, // half-rate doesn't have a c̃₄ vector
            epsilon_t: ambe.error_total().min(255) as u8,
            bad_pitch: false,
        };
        match decode_halfrate_to_params(&ambe.info, &mut dec_state) {
            Ok(HalfrateDecoded::Voice(params)) => {
                voice_frames += 1;
                let pcm = synthesize_frame(&params, &err, gamma_w, &mut synth_state);
                pcm_out.extend_from_slice(&pcm);
            }
            Ok(HalfrateDecoded::Tone { fields: _, params }) => {
                tone_frames += 1;
                // Tone frames feed the MBE synthesizer per §2.10.3.
                let pcm = synthesize_frame(&params, &err, gamma_w, &mut synth_state);
                pcm_out.extend_from_slice(&pcm);
            }
            Ok(HalfrateDecoded::Erasure) | Err(_) => {
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

    println!("Comparison vs DVSI r39 reference PCM ({n} samples):");
    println!("  bit-exact samples:  {bit_exact} / {n}   ({pct_exact:.2}%)");
    println!("  RMS error:          {rms_err:.2}");
    println!("  peak error:         {peak_err:.0}");
    println!("  SNR:                {snr_db:.2} dB");
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
        let imbe = decode_fullrate_frame(&dibits);
        let params = match dequantize_fullrate(&imbe.info, &mut decoder_state) {
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
        let imbe = decode_fullrate_frame(&dibits);
        let params = match dequantize_fullrate(&imbe.info, &mut decoder_state) {
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
        let imbe = decode_fullrate_frame(&dibits);
        let params = dequantize_fullrate(&imbe.info, &mut decoder_state)
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
    let dir = root.join("tv-std").join("tv").join("r39");
    let bit_path = dir.join(format!("{name}.bit"));
    let bit_bytes = fs::read(&bit_path)
        .with_context(|| format!("read half-rate vector {}", bit_path.display()))?;
    if bit_bytes.len() % BYTES_PER_HALFRATE_FRAME != 0 {
        return Err(anyhow!("half-rate .bit file is not a whole number of frames"));
    }
    let n_frames = bit_bytes.len() / BYTES_PER_HALFRATE_FRAME;

    println!("vector:        {name} (half-rate, r39)");
    println!("frames:        {n_frames}");
    println!();

    let mut state_dec = HalfrateDecoderState::new();
    let mut state_enc = HalfrateDecoderState::new();
    let mut voice_frames = 0usize;
    let mut non_voice_frames = 0usize;
    let mut pitch_match = 0usize;
    let mut vuv_match = 0usize;
    let mut gain_match = 0usize;

    use blip25_mbe::imbe_frames::priority::deprioritize_halfrate;

    for f in 0..n_frames {
        let frame = &bit_bytes[f * BYTES_PER_HALFRATE_FRAME..(f + 1) * BYTES_PER_HALFRATE_FRAME];
        let dibits = unpack_dibits_halfrate(frame);
        let ambe = decode_halfrate_frame(&dibits);
        match decode_halfrate_to_params(&ambe.info, &mut state_dec) {
            Ok(HalfrateDecoded::Voice(params)) => {
                voice_frames += 1;
                let u_back = match quantize_halfrate(&params, &mut state_enc) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                let b_orig = deprioritize_halfrate(&ambe.info);
                let b_back = deprioritize_halfrate(&u_back);
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
        let decoded: ImbeFrame = decode_fullrate_frame(&dibits);
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

fn report_mismatch(frame_idx: usize, decoded: &ImbeFrame, expected: &[u16; 8]) {
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
}
