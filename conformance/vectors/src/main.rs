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
