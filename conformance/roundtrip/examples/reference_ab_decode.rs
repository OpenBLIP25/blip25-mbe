//! Reference decode-only A/B renderer.
//!
//! For each reference test vector `<name>` and each of the two P25 rates that
//! ship a reference decode, this emits a fixed five-file listening set.
//! Everything is DECODE-ONLY: both the reference output and our output are
//! decodes of the SAME reference-produced MBE bits (`tv-rc/<subdir>/<name>.bit`),
//! so the only variable is the decoder. (Encode is deliberately out of scope
//! here.)
//!
//! Per (vector, rate), five WAVs at 8 kHz:
//!   <rate>_1_source.wav              — original PCM (tv-rc/<name>.pcm), mono
//!   <rate>_2_ref_decode.wav          — reference decode of its own bits, mono
//!   <rate>_3_ours_decode.wav         — our decode of the same bits, mono
//!   <rate>_4_AB_L-ref_R-ours.wav     — stereo A/B: left=reference, right=ours
//!   <rate>_5_AB_L-ours_R-ref.wav     — stereo A/B, channels flipped
//!
//! rate label is `imbe` (P25 Phase 1) or `ambe2` (P25 Phase 2, r33).
//!
//! Usage:
//!   cargo run -p blip25-conformance-roundtrip --example reference_ab_decode -- \
//!     --vectors reference-material/Vectors --out roundtrip_check/ab [names...]
//! Default names: clean dam mark.

use anyhow::{Context, Result};
use blip25_mbe::vocoder::{Rate, Vocoder};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

const FRAME: usize = 160;
const SR: u32 = 8000;

#[derive(Parser, Debug)]
#[command(
    about = "Reference decode-only A/B renderer (source / reference decode / our decode / stereo A/B)"
)]
struct Args {
    /// Reference vectors root (contains tv-rc/).
    #[arg(long, default_value = "reference-material/Vectors")]
    vectors: PathBuf,
    /// Reference base under --vectors.
    #[arg(long, default_value = "tv-rc")]
    base: String,
    /// Output directory (a subdir per vector is created).
    #[arg(long, default_value = "roundtrip_check/ab")]
    out: PathBuf,
    /// Vector names to render (default: clean dam mark).
    names: Vec<String>,
}

/// (rate, subdir under tv-rc, short label, FEC frame bytes).
const REFS: [(Rate, &str, &str, usize); 2] = [
    (Rate::Imbe7200x4400, "p25", "imbe", 18),
    (Rate::AmbePlus2_3600x2450, "r33", "ambe2", 9),
];

fn read_pcm(path: &Path) -> Result<Vec<i16>> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn write_wav(path: &Path, channels: u16, interleaved: &[i16]) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let spec = hound::WavSpec {
        channels,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in interleaved {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

/// Interleave two mono channels into a stereo buffer, truncating to the shorter.
fn interleave(left: &[i16], right: &[i16]) -> Vec<i16> {
    let n = left.len().min(right.len());
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

/// Decode every reference `.bit` frame through our public decoder.
fn our_decode(rate: Rate, bits: &[u8], n: usize) -> Result<Vec<i16>> {
    let mut v = Vocoder::new(rate);
    let mut out = Vec::with_capacity(bits.len() / n * FRAME);
    for fr in bits.chunks_exact(n) {
        out.extend(
            v.decode_bits(fr)
                .map_err(|e| anyhow::anyhow!("decode: {e:?}"))?,
        );
    }
    Ok(out)
}

/// Sample-level Pearson r over the common span, and fraction of differing samples.
fn distinctness(a: &[i16], b: &[i16]) -> (f64, usize, usize, i32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0, 0, 0);
    }
    let (a, b) = (&a[..n], &b[..n]);
    let (ma, mb) = (
        a.iter().map(|&x| x as f64).sum::<f64>() / n as f64,
        b.iter().map(|&x| x as f64).sum::<f64>() / n as f64,
    );
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    let (mut diffs, mut maxd) = (0usize, 0i32);
    for i in 0..n {
        let (dx, dy) = (a[i] as f64 - ma, b[i] as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
        if a[i] != b[i] {
            diffs += 1;
        }
        maxd = maxd.max((a[i] as i32 - b[i] as i32).abs());
    }
    let r = if va > 0.0 && vb > 0.0 {
        cov / (va.sqrt() * vb.sqrt())
    } else {
        0.0
    };
    (r, diffs, n, maxd)
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if args.names.is_empty() {
        args.names = vec!["clean".into(), "dam".into(), "mark".into()];
    }
    let tvrc = args.vectors.join(&args.base);
    anyhow::ensure!(tvrc.is_dir(), "{} not found", tvrc.display());

    println!(
        "{:<10} {:<6} {:>6} | reference-decode vs our-decode (same bits): r, %samples differing, maxΔ",
        "vector", "rate", "frames"
    );

    for name in &args.names {
        let src = read_pcm(&tvrc.join(format!("{name}.pcm")))
            .with_context(|| format!("source pcm for {name}"))?;
        let dir = args.out.join(name);

        for (rate, subdir, label, n) in REFS {
            let bit_path = tvrc.join(subdir).join(format!("{name}.bit"));
            let ref_path = tvrc.join(subdir).join(format!("{name}.pcm"));
            if !bit_path.is_file() || !ref_path.is_file() {
                println!(
                    "{name:<10} {label:<6} {:>6} | (no reference — skipped)",
                    "-"
                );
                continue;
            }

            let ref_bits = fs::read(&bit_path)?;
            let ref_dec = read_pcm(&ref_path)?; // reference decode of its own bits
            let our_dec = our_decode(rate, &ref_bits, n)?; // our decode of the same bits
            let frames = ref_bits.len() / n;

            // 1..3 mono
            let p = |suf: &str| dir.join(format!("{label}_{suf}.wav"));
            write_wav(&p("1_source"), 1, &src)?;
            write_wav(&p("2_ref_decode"), 1, &ref_dec)?;
            write_wav(&p("3_ours_decode"), 1, &our_dec)?;
            // 4..5 stereo A/B (aligned from sample 0, truncated to the shorter)
            write_wav(&p("4_AB_L-ref_R-ours"), 2, &interleave(&ref_dec, &our_dec))?;
            write_wav(&p("5_AB_L-ours_R-ref"), 2, &interleave(&our_dec, &ref_dec))?;

            let (r, diffs, ncmp, maxd) = distinctness(&ref_dec, &our_dec);
            let pct = 100.0 * diffs as f64 / ncmp.max(1) as f64;
            println!(
                "{name:<10} {label:<6} {frames:>6} | r={r:+.3}  {pct:>5.1}% differ  maxΔ={maxd}"
            );
        }
    }

    println!("\nFiles under: {}/<vector>/", args.out.display());
    println!("A/B: pan hard-left vs hard-right on file 4 (or 5, flipped) to compare decoders.");
    Ok(())
}
