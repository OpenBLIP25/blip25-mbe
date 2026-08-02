//! Reference test-vector roundtrip sanity harness — the shipped `blip25_codec`
//! codec as a pure "bits in / bits out" function, measured against the reference
//! vocoder's ground truth.
//!
//! Everything runs through the PUBLIC [`Vocoder`] API (the default/shipped
//! path — no research levers touched), so what this reports is exactly what a
//! downstream integrator gets.
//!
//! For each vector `<name>` we use the two P25 rates that ship a reference-decoded
//! `.pcm`:
//!   * IMBE   (P25 Phase 1) — `tv-rc/p25/<name>.{bit,pcm}` (18-byte FEC frames)
//!   * AMBE+2 (P25 Phase 2) — `tv-rc/r33/<name>.{bit,pcm}` ( 9-byte FEC frames)
//! with the source input at `tv-rc/<name>.pcm`.
//!
//! Three independent comparisons against the reference's own encode/decode:
//!   1. DECODE conformance — our `decode_bits(reference .bit)` vs reference `.pcm`
//!      (isolates OUR DECODER: same input bits, compare output audio).
//!   2. ENCODE conformance — our `encode_pcm(source)` frames   vs reference `.bit`
//!      (isolates OUR ENCODER: byte-exact frame rate).
//!   3. ROUNDTRIP          — our decode(our encode(source))    vs reference `.pcm`
//!      (the WAV -> codec -> WAV the operator described).
//!
//! Emits a CSV over every eligible vector and renders WAVs for a listening set:
//!   <out>/<name>/<rate>__source.wav
//!   <out>/<name>/<rate>__ours_decode_of_ref_bits.wav
//!   <out>/<name>/<rate>__ours_roundtrip.wav
//!   <out>/<name>/<rate>__ref_roundtrip.wav
//!
//! Usage:
//!   cargo run -p blip25-conformance-roundtrip --example reference_roundtrip_sanity -- \
//!     --vectors reference-material/Vectors --out roundtrip_check [--all-wavs] [names...]

use anyhow::{Context, Result};
use blip25_mbe::vocoder::{Rate, Vocoder};
use clap::Parser;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const FRAME: usize = 160;
const SR: u32 = 8000;

#[derive(Parser, Debug)]
#[command(about = "Reference test-vector roundtrip sanity check (shipped blip25_codec codec)")]
struct Args {
    /// Reference vectors root (contains tv-rc/).
    #[arg(long, default_value = "reference-material/Vectors")]
    vectors: PathBuf,
    /// Output directory for the CSV and rendered WAVs.
    #[arg(long, default_value = "roundtrip_check")]
    out: PathBuf,
    /// Reference base under --vectors: "tv-rc" (broad RC corpus) or
    /// "tv-std/tv" (official normative set — the genuine IMBE reference).
    #[arg(long, default_value = "tv-rc")]
    base: String,
    /// Render WAVs for EVERY eligible vector (default: only the listening set).
    #[arg(long)]
    all_wavs: bool,
    /// Only process these vector names (default: all with a reference decoded ref).
    names: Vec<String>,
}

/// FEC-frame byte width for the two rates that carry a reference decoded output.
fn fec_bytes(rate: Rate) -> usize {
    match rate {
        Rate::Imbe7200x4400 => 18,
        Rate::AmbePlus2_3600x2450 => 9,
        _ => unreachable!("only the two FEC rates with a reference .pcm are used"),
    }
}

/// The two (rate, subdir, short-label) references that have a decoded `.pcm`.
const REFS: [(Rate, &str, &str); 2] = [
    (Rate::Imbe7200x4400, "p25", "imbe"),
    (Rate::AmbePlus2_3600x2450, "r33", "ambe2"),
];

/// Representative listening set (rendered to WAV unless --all-wavs).
const LISTEN: &[&str] = &[
    "clean", "mark", "dam", "fambf22c", "lvl", "tia11", // speech
    "alert", "knox_1", "grbge", "sine0_1k", // tone / noise / pure
];

fn read_pcm(path: &Path) -> Result<Vec<i16>> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn write_wav(path: &Path, pcm: &[i16]) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in pcm {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    n: usize,
    exact: usize,
    snr_db: f64,
    r: f64,
    /// Per-frame (20 ms) RMS-energy envelope correlation. This is the
    /// PHASE-INVARIANT content ruler: MBE-family decoders (esp. IMBE) use a
    /// different voiced-phase realization than the reference, so the sample waveform
    /// decorrelates while the energy/loudness structure still matches. env_r
    /// near 1.0 = "same speech"; the sample-level r/SNR only track the shared
    /// phase, which AMBE+2 has and IMBE largely does not.
    env_r: f64,
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (dx, dy) = (a[i] - ma, b[i] - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va > 0.0 && vb > 0.0 {
        cov / (va.sqrt() * vb.sqrt())
    } else {
        0.0
    }
}

/// Per-frame (160-sample) RMS energy sequence.
fn frame_energy(x: &[i16]) -> Vec<f64> {
    x.chunks(FRAME)
        .map(|c| (c.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

/// Sample-level comparison of `a` against reference `b` over their common span,
/// plus the phase-invariant per-frame energy-envelope correlation.
fn compare(a: &[i16], b: &[i16]) -> Metrics {
    let n = a.len().min(b.len());
    if n == 0 {
        return Metrics::default();
    }
    let (mut se, mut sr, mut exact) = (0f64, 0f64, 0usize);
    let (mut sa, mut sb) = (0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        se += (x - y) * (x - y);
        sr += y * y;
        sa += x;
        sb += y;
        if a[i] == b[i] {
            exact += 1;
        }
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (dx, dy) = (a[i] as f64 - ma, b[i] as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    let r = if va > 0.0 && vb > 0.0 {
        cov / (va.sqrt() * vb.sqrt())
    } else {
        0.0
    };
    let snr_db = if se > 0.0 {
        10.0 * (sr / se).log10()
    } else {
        f64::INFINITY
    };
    let env_r = pearson(&frame_energy(&a[..n]), &frame_energy(&b[..n]));
    Metrics {
        n,
        exact,
        snr_db,
        r,
        env_r,
    }
}

/// Decode every reference `.bit` frame through our public decoder.
fn our_decode(rate: Rate, bits: &[u8]) -> Result<Vec<i16>> {
    let n = fec_bytes(rate);
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

/// Encode source PCM through our public encoder, look-ahead aligned: the
/// frame-0 placeholder is dropped and the look-ahead FIFO is flushed at the end
/// (same convention as the blip25_codec acceptance gate). Returns real emitted frames.
fn our_encode(rate: Rate, pcm: &[i16]) -> Result<Vec<Vec<u8>>> {
    let mut v = Vocoder::new(rate);
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for f in pcm.chunks_exact(FRAME) {
        frames.push(
            v.encode_pcm(f)
                .map_err(|e| anyhow::anyhow!("encode: {e:?}"))?,
        );
    }
    if !frames.is_empty() {
        frames.remove(0); // drop the all-zero look-ahead placeholder
    }
    frames.extend(v.flush_encode());
    Ok(frames)
}

fn frame_byte_exact(ours: &[Vec<u8>], reference: &[u8], n: usize) -> (usize, usize) {
    let ref_frames: Vec<&[u8]> = reference.chunks_exact(n).collect();
    let m = ours.len().min(ref_frames.len());
    let mut exact = 0;
    for i in 0..m {
        if ours[i].as_slice() == ref_frames[i] {
            exact += 1;
        }
    }
    (exact, m)
}

struct Row {
    name: String,
    rate: &'static str,
    frames: usize,
    dec: Metrics,
    enc_exact: usize,
    enc_total: usize,
    rt: Metrics,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tvrc = args.vectors.join(&args.base);
    anyhow::ensure!(
        tvrc.is_dir(),
        "{} not found under {}",
        args.base,
        args.vectors.display()
    );
    fs::create_dir_all(&args.out).ok();

    // Discover vector names: every top-level tv-rc/*.pcm that has a decoded
    // reference under at least one of the two rate dirs.
    let mut names: BTreeSet<String> = BTreeSet::new();
    if args.names.is_empty() {
        for e in fs::read_dir(&tvrc)? {
            let p = e?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("pcm") {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    let has_ref = REFS.iter().any(|(_, d, _)| {
                        tvrc.join(d).join(format!("{stem}.bit")).is_file()
                            && tvrc.join(d).join(format!("{stem}.pcm")).is_file()
                    });
                    if has_ref {
                        names.insert(stem.to_string());
                    }
                }
            }
        }
    } else {
        names.extend(args.names.iter().cloned());
    }

    println!(
        "Scanning {} vectors x {} rates through the shipped Vocoder API\n",
        names.len(),
        REFS.len()
    );

    let mut rows: Vec<Row> = Vec::new();
    for name in &names {
        let src = read_pcm(&tvrc.join(format!("{name}.pcm"))).unwrap_or_default();
        for (rate, subdir, label) in REFS {
            let n = fec_bytes(rate);
            let bit_path = tvrc.join(subdir).join(format!("{name}.bit"));
            let ref_path = tvrc.join(subdir).join(format!("{name}.pcm"));
            if !bit_path.is_file() || !ref_path.is_file() {
                continue;
            }
            let ref_bits = fs::read(&bit_path)?;
            let ref_pcm = read_pcm(&ref_path)?;
            let frames = ref_bits.len() / n;

            // 1. DECODE conformance: our decode of the reference's own bits vs reference pcm.
            let our_dec = our_decode(rate, &ref_bits)?;
            let dec = compare(&our_dec, &ref_pcm);

            // 2/3. ENCODE + ROUNDTRIP from the source PCM.
            let (enc_exact, enc_total, rt_pcm, rt) = if src.len() >= FRAME {
                let ours = our_encode(rate, &src)?;
                let (ex, tot) = frame_byte_exact(&ours, &ref_bits, n);
                // Decode our own bits back for the end-to-end roundtrip.
                let mut dec2 = Vocoder::new(rate);
                let mut rt_pcm = Vec::with_capacity(ours.len() * FRAME);
                for fr in &ours {
                    rt_pcm.extend(
                        dec2.decode_bits(fr)
                            .map_err(|e| anyhow::anyhow!("rt decode: {e:?}"))?,
                    );
                }
                let rt = compare(&rt_pcm, &ref_pcm);
                (ex, tot, rt_pcm, rt)
            } else {
                (0, 0, Vec::new(), Metrics::default())
            };

            // Render WAVs for the listening set (or everything with --all-wavs).
            if args.all_wavs || LISTEN.contains(&name.as_str()) {
                let dir = args.out.join(name);
                let pre = |suf: &str| dir.join(format!("{label}__{suf}.wav"));
                write_wav(&pre("source"), &src)?;
                write_wav(&pre("ours_decode_of_ref_bits"), &our_dec)?;
                write_wav(&pre("ref_roundtrip"), &ref_pcm)?;
                if !rt_pcm.is_empty() {
                    write_wav(&pre("ours_roundtrip"), &rt_pcm)?;
                }
            }

            rows.push(Row {
                name: name.clone(),
                rate: label,
                frames,
                dec,
                enc_exact,
                enc_total,
                rt,
            });
        }
    }

    // Console summary. env_r (per-frame energy envelope) is the phase-invariant
    // "same content" ruler; wave_r/snr are the sample-level "same waveform".
    println!(
        "{:<12} {:<6} {:>6} | DECODE: our decode(ref bits) vs ref pcm          | ENCODE     | ROUNDTRIP",
        "vector", "rate", "frames"
    );
    println!(
        "{:<12} {:<6} {:>6} | {:>7} {:>7} {:>7} {:>8} | {:>10} | {:>7}",
        "", "", "", "env_r", "wave_r", "snr_dB", "samp-eq%", "byte-eq%", "env_r"
    );
    for r in &rows {
        let dex = 100.0 * r.dec.exact as f64 / r.dec.n.max(1) as f64;
        let enc_pct = if r.enc_total > 0 {
            format!("{:.0}%", 100.0 * r.enc_exact as f64 / r.enc_total as f64)
        } else {
            "n/a".to_string()
        };
        println!(
            "{:<12} {:<6} {:>6} | {:>7.3} {:>7.3} {:>7.2} {:>7.1}% | {:>10} | {:>7.3}",
            r.name, r.rate, r.frames, r.dec.env_r, r.dec.r, r.dec.snr_db, dex, enc_pct, r.rt.env_r
        );
    }

    // CSV.
    let csv_path = args.out.join("metrics.csv");
    let mut csv = String::from(
        "vector,rate,frames,dec_samples,dec_env_r,dec_r,dec_snr_db,dec_samp_exact_pct,enc_frame_exact,enc_frame_total,enc_byte_exact_pct,rt_env_r,rt_r,rt_snr_db\n",
    );
    for r in &rows {
        let dex = 100.0 * r.dec.exact as f64 / r.dec.n.max(1) as f64;
        let encp = if r.enc_total > 0 {
            100.0 * r.enc_exact as f64 / r.enc_total as f64
        } else {
            0.0
        };
        csv.push_str(&format!(
            "{},{},{},{},{:.5},{:.5},{:.3},{:.3},{},{},{:.2},{:.5},{:.5},{:.3}\n",
            r.name,
            r.rate,
            r.frames,
            r.dec.n,
            r.dec.env_r,
            r.dec.r,
            r.dec.snr_db,
            dex,
            r.enc_exact,
            r.enc_total,
            encp,
            r.rt.env_r,
            r.rt.r,
            r.rt.snr_db,
        ));
    }
    fs::write(&csv_path, &csv)?;
    println!("\nCSV: {}", csv_path.display());
    println!("WAVs under: {}/<vector>/", args.out.display());

    // Aggregate headline (means over rows, decode path).
    for label in ["imbe", "ambe2"] {
        let sub: Vec<&Row> = rows.iter().filter(|r| r.rate == label).collect();
        if sub.is_empty() {
            continue;
        }
        let mean =
            |f: &dyn Fn(&Row) -> f64| sub.iter().map(|r| f(r)).sum::<f64>() / sub.len() as f64;
        let enc_byte = mean(&|r| {
            if r.enc_total > 0 {
                100.0 * r.enc_exact as f64 / r.enc_total as f64
            } else {
                0.0
            }
        });
        println!(
            "\n{:<6} ({} vectors): DECODE content(env_r)={:.3}  waveform(wave_r)={:.3}  snr={:.1}dB   |   ENCODE byte-exact={:.1}%",
            label.to_uppercase(),
            sub.len(),
            mean(&|r| r.dec.env_r),
            mean(&|r| r.dec.r),
            mean(&|r| if r.dec.snr_db.is_finite() { r.dec.snr_db } else { 0.0 }),
            enc_byte,
        );
    }
    Ok(())
}
