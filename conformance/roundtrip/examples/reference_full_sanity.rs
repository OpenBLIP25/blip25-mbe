//! Full reference sanity set: for every vector with a reference decode, emit two
//! stereo A/B files per rate (IMBE + AMBE+2), Left = reference, Right = ours:
//!   <name>/<rate>__2_decode_L-ref_R-ours.wav      — our decode of reference bits
//!                                                    (both from the SAME bits →
//!                                                    isolates OUR DECODER)
//!   <name>/<rate>__3_roundtrip_L-ref_R-ours.wav   — our encode(source) → our
//!                                                    decode (the whole pipeline)
//! plus <rate>__1_source.wav. Roundtrip uses the shipped `Vocoder::encode`
//! (AMBE+2 reference voicing + silence gate; IMBE per-frame) and the shipped
//! decoder.
//!
//! Tones are included and matter especially on IMBE, which has no tone codeword
//! (Phase 1 encodes a tone as ordinary voice frames).
//!
//! Usage: cargo run -p blip25-conformance-roundtrip --example reference_full_sanity -- \
//!          [--out roundtrip_check/sanity] [names...]   (default: all vectors)

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const FRAME: usize = 160;
const TVRC: &str = "reference-material/Vectors/tv-rc";
// (rate, subdir, label, fec_bytes)
const REFS: [(Rate, &str, &str, usize); 2] = [
    (Rate::Imbe7200x4400, "p25", "imbe", 18),
    (Rate::AmbePlus2_3600x2450, "r33", "ambe2", 9),
];

fn read_pcm(p: &Path) -> Vec<i16> {
    fs::read(p)
        .map(|b| {
            b.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        })
        .unwrap_or_default()
}
fn write_wav(path: &Path, ch: u16, inter: &[i16]) {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).ok();
    }
    let spec = hound::WavSpec {
        channels: ch,
        sample_rate: 8000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in inter {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
}
fn interleave(a: &[i16], b: &[i16]) -> Vec<i16> {
    let n = a.len().min(b.len());
    let mut o = Vec::with_capacity(n * 2);
    for i in 0..n {
        o.push(a[i]);
        o.push(b[i]);
    }
    o
}
fn frame_energy(x: &[i16]) -> Vec<f64> {
    x.chunks(FRAME)
        .map(|c| {
            (c.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / c.len().max(1) as f64).sqrt()
        })
        .collect()
}
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 4 {
        return 0.0;
    }
    let (ma, mb) = (
        a[..n].iter().sum::<f64>() / n as f64,
        b[..n].iter().sum::<f64>() / n as f64,
    );
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (a[i] - ma, b[i] - mb);
        c += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va > 0.0 && vb > 0.0 {
        c / (va.sqrt() * vb.sqrt())
    } else {
        0.0
    }
}
/// Best whole-frame lag of `ours` vs `reference` by envelope correlation, plus env_r.
fn align_env_r(ours: &[i16], reference: &[i16]) -> (i64, f64) {
    let (eo, ed) = (frame_energy(ours), frame_energy(reference));
    let mut best = (0i64, -2.0);
    for lag in -6..=6i64 {
        let r = if lag >= 0 {
            pearson(&eo[lag as usize..], &ed)
        } else {
            pearson(&eo, &ed[(-lag) as usize..])
        };
        if r > best.1 {
            best = (lag, r);
        }
    }
    best
}
fn shift(ours: &[i16], reference: &[i16], lag: i64) -> (Vec<i16>, Vec<i16>) {
    if lag > 0 {
        (
            ours[(lag as usize * FRAME).min(ours.len())..].to_vec(),
            reference.to_vec(),
        )
    } else if lag < 0 {
        (
            ours.to_vec(),
            reference[((-lag) as usize * FRAME).min(reference.len())..].to_vec(),
        )
    } else {
        (ours.to_vec(), reference.to_vec())
    }
}
fn our_decode(rate: Rate, bits: &[u8], n: usize) -> Vec<i16> {
    let mut v = Vocoder::new(rate);
    let mut out = Vec::new();
    for fr in bits.chunks_exact(n) {
        if let Ok(p) = v.decode_bits(fr) {
            out.extend(p);
        }
    }
    out
}
fn our_roundtrip(rate: Rate, src: &[i16]) -> Vec<i16> {
    let enc = Vocoder::new(rate);
    let frames = enc.encode(src);
    let mut dec = Vocoder::new(rate);
    let mut out = Vec::new();
    for f in &frames {
        if let Ok(p) = dec.decode_bits(f) {
            out.extend(p);
        }
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = PathBuf::from("roundtrip_check/sanity");
    if let Some(i) = args.iter().position(|a| a == "--out") {
        out = PathBuf::from(args[i + 1].clone());
        args.drain(i..=i + 1);
    }
    let root = Path::new(TVRC);
    let mut names: BTreeSet<String> = BTreeSet::new();
    if args.is_empty() {
        for e in fs::read_dir(root).unwrap() {
            let p = e.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("pcm") {
                if let Some(s) = p.file_stem().and_then(|s| s.to_str()) {
                    if REFS
                        .iter()
                        .any(|(_, d, _, _)| root.join(d).join(format!("{s}.pcm")).is_file())
                    {
                        names.insert(s.to_string());
                    }
                }
            }
        }
    } else {
        names.extend(args);
    }

    println!(
        "{:<12} {:<6} {:>7} {:>10} {:>12}",
        "vector", "rate", "frames", "decode_r", "roundtrip_r"
    );
    let mut csv = String::from("vector,rate,frames,decode_env_r,roundtrip_env_r\n");
    for name in &names {
        let src = read_pcm(&root.join(format!("{name}.pcm")));
        if src.len() < FRAME * 2 {
            continue;
        }
        for (rate, sub, label, n) in REFS {
            let bit = root.join(sub).join(format!("{name}.bit"));
            let refp = root.join(sub).join(format!("{name}.pcm"));
            if !bit.is_file() || !refp.is_file() {
                continue;
            }
            let ref_bits = fs::read(&bit).unwrap();
            let ref_pcm = read_pcm(&refp);
            let frames = ref_bits.len() / n;

            // decode isolation: same bits, our decoder vs the reference — frame-aligned.
            let our_dec = our_decode(rate, &ref_bits, n);
            let dec_r = pearson(&frame_energy(&our_dec), &frame_energy(&ref_pcm));

            // full roundtrip: our encode(source) -> our decode, aligned to reference.
            let our_rt = our_roundtrip(rate, &src);
            let (lag, rt_r) = align_env_r(&our_rt, &ref_pcm);
            let (rt_a, ref_a) = shift(&our_rt, &ref_pcm, lag);

            let dir = out.join(name);
            let pre = |s: &str| dir.join(format!("{label}__{s}.wav"));
            write_wav(&pre("1_source"), 1, &src);
            write_wav(
                &pre("2_decode_L-ref_R-ours"),
                2,
                &interleave(&ref_pcm, &our_dec),
            );
            write_wav(
                &pre("3_roundtrip_L-ref_R-ours"),
                2,
                &interleave(&ref_a, &rt_a),
            );

            println!("{name:<12} {label:<6} {frames:>7} {dec_r:>10.3} {rt_r:>12.3}");
            csv.push_str(&format!("{name},{label},{frames},{dec_r:.4},{rt_r:.4}\n"));
        }
    }
    fs::create_dir_all(&out).ok();
    fs::write(out.join("metrics.csv"), csv).unwrap();
    println!("\nWAVs + metrics.csv under {}/<vector>/", out.display());
}
