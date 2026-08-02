//! FINAL PARITY SANITY CHECK — all four codec paths vs the reference vocoder.
//!
//! For every reference test vector, emit stereo WAVs (LEFT = ours, RIGHT =
//! reference) plus an env_r metrics table so any vector that is "off" pops out
//! numerically.
//!
//!   {label}/{vec}_1_decode.wav     L = our decode(reference bits)
//!                                  R = reference decode (rNN/{v}.pcm)
//!   {label}/{vec}_2_roundtrip.wav  L = our decode(our encode(source))
//!                                  R = reference roundtrip (rNN/{v}.pcm)
//!
//! Note: rNN/{v}.pcm is BOTH the reference decode-of-bits AND — because the
//! reference produced those bits by encoding the source — the reference full
//! roundtrip. There is no separate "roundtrip" file; the same reference serves
//! both comparisons. The DECODE pair isolates our decoder (identical bits both
//! sides); the ROUNDTRIP pair exercises our encoder + decoder against the
//! reference's whole pipeline.
//!
//! Roundtrip is emitted only where a source PCM is shipped ({root}/{v}.pcm).
//!
//! Usage:
//!   cargo run -p blip25-conformance-roundtrip --example roundtrip_sanity -- \
//!     [--root reference-material/Vectors/tv-std/tv] [--out roundtrip_check/tv-std] \
//!     [--no-decode] [--no-roundtrip] [names...]

use blip25_mbe::reference_soft_decision::unpack_nibble_stream;
use blip25_mbe::vocoder::{Rate, Vocoder};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const FRAME: usize = 160;
// (rate, subdir, label, frame_bytes)
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

/// Reference `*_sd.bit` soft-decision nibble stream -> hard packed bytes (MSB-first).
/// Each channel bit is stored as a 4-bit offset-binary SD value whose MSB is the
/// hard decision (`llr > 0`); we slice that and repack to the standard hard
/// frame so the shipped hard-bit decoder can run. NB: this is the *hard*-decision
/// decode of the error-injected channel — the reference `.pcm` used the reference
/// *soft*-decision FEC, so at high BER ours will legitimately show more residual
/// error. The soft-vs-hard FEC gain is quantified separately in `reference_soft_gain`.
fn sd_to_hard(sd_bytes: &[u8]) -> Vec<u8> {
    let llrs = unpack_nibble_stream(sd_bytes); // 2 per byte, SD0 first, sign = hard bit
    let mut out = vec![0u8; llrs.len() / 8];
    for (i, &l) in llrs.iter().enumerate() {
        if l > 0 {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
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

/// SOFT-decision decode of a raw `*_sd.bit` nibble stream through the shipped
/// `Vocoder::decode_soft` (Chase-II soft Golay/Hamming) — the apples-to-apples
/// path vs the reference's soft-decoded output. `n` is the hard frame size (18/9), so
/// each frame is `n*8` LLRs.
fn our_decode_soft(rate: Rate, sd_bytes: &[u8], n: usize) -> Vec<i16> {
    let llrs = unpack_nibble_stream(sd_bytes); // 2 per byte, SD0 first, sign = hard bit
    let per_frame = n * 8; // 144 (IMBE) / 72 (AMBE) LLRs
    let mut v = Vocoder::new(rate);
    let mut out = Vec::new();
    for chunk in llrs.chunks_exact(per_frame) {
        if let Ok(p) = v.decode_soft(chunk) {
            out.extend(p);
        }
    }
    out
}
fn our_roundtrip(rate: Rate, src: &[i16]) -> Vec<i16> {
    let frames = Vocoder::new(rate).encode(src); // shipped encode path
    let mut dec = Vocoder::new(rate);
    let mut out = Vec::new();
    for f in &frames {
        if let Ok(p) = dec.decode_bits(f) {
            out.extend(p);
        }
    }
    out
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

/// Coarse sample lag that best aligns `ours` to `refr` (frame-step search, then
/// ±FRAME fine refine). Positive lag = drop leading samples of `ours`. Handles
/// the encoder look-ahead / decoder group-delay offsets so the ear can center.
fn best_lag(ours: &[i16], refr: &[i16], max: i64) -> i64 {
    let n = ours.len().min(refr.len());
    if n < 4000 {
        return 0;
    }
    let (w0, w1) = (n / 6, 5 * n / 6);
    let score = |lag: i64| -> f64 {
        let (mut c, mut eo) = (0.0f64, 0.0f64);
        for i in w0..w1 {
            let j = i as i64 + lag;
            if j >= 0 && (j as usize) < ours.len() {
                let a = ours[j as usize] as f64;
                c += a * refr[i] as f64;
                eo += a * a;
            }
        }
        c / (eo.sqrt() + 1.0)
    };
    let mut best = (f64::MIN, 0i64);
    let mut lag = -max;
    while lag <= max {
        let s = score(lag);
        if s > best.0 {
            best = (s, lag);
        }
        lag += FRAME as i64;
    }
    for l in (best.1 - FRAME as i64)..=(best.1 + FRAME as i64) {
        let s = score(l);
        if s > best.0 {
            best = (s, l);
        }
    }
    best.1
}

fn apply_lag(ours: &[i16], lag: i64) -> Vec<i16> {
    if lag >= 0 {
        ours.iter().skip(lag as usize).copied().collect()
    } else {
        let mut v = vec![0i16; (-lag) as usize];
        v.extend_from_slice(ours);
        v
    }
}

fn write_stereo(path: &Path, left: &[i16], right: &[i16]) {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).ok();
    }
    let n = left.len().min(right.len());
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 8000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..n {
        w.write_sample(left[i]).unwrap(); // L = ours
        w.write_sample(right[i]).unwrap(); // R = reference
    }
    w.finalize().unwrap();
}

/// Align `ours` to `reference`, write the L=ours/R=reference WAV, return (lag, env_r).
fn emit(path: &Path, ours: &[i16], reference: &[i16]) -> (i64, f64) {
    if ours.is_empty() || reference.is_empty() {
        return (0, f64::NAN);
    }
    let lag = best_lag(ours, reference, 8 * FRAME as i64);
    let l = apply_lag(ours, lag);
    let r = pearson(&frame_energy(&l), &frame_energy(reference));
    write_stereo(path, &l, reference);
    (lag, r)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take_opt = |args: &mut Vec<String>, key: &str| -> Option<String> {
        args.iter().position(|a| a == key).map(|i| {
            let v = args[i + 1].clone();
            args.drain(i..=i + 1);
            v
        })
    };
    let take_flag = |args: &mut Vec<String>, key: &str| -> bool {
        args.iter()
            .position(|a| a == key)
            .map(|i| args.remove(i))
            .is_some()
    };

    let root = PathBuf::from(
        take_opt(&mut args, "--root").unwrap_or_else(|| "reference-material/Vectors/tv-std/tv".into()),
    );
    let out = PathBuf::from(
        take_opt(&mut args, "--out").unwrap_or_else(|| "roundtrip_check/tv-std".into()),
    );
    let no_decode = take_flag(&mut args, "--no-decode");
    let no_roundtrip = take_flag(&mut args, "--no-roundtrip");
    let soft = take_flag(&mut args, "--soft"); // `*_sd` vectors: soft FEC vs hard-slice
    let filter: BTreeSet<String> = args.into_iter().collect();

    println!("root = {}", root.display());
    println!("out  = {}", out.display());
    println!(
        "decode={}  roundtrip={}  sd_mode={}\n",
        if no_decode { "off" } else { "on" },
        if no_roundtrip { "off" } else { "on" },
        if soft { "soft-FEC" } else { "hard-slice" }
    );

    let mut csv = String::from(
        "rate,vector,frames,decode_env_r,decode_lag,roundtrip_env_r,roundtrip_lag,note\n",
    );
    // rows: (rate_label, vector, frames, dec_r, rt_r)
    let mut rows: Vec<(String, String, usize, f64, f64)> = Vec::new();

    for (rate, sub, label, n) in REFS {
        let subdir = root.join(sub);
        if !subdir.is_dir() {
            eprintln!("(no {} dir — skipping {label})", subdir.display());
            continue;
        }
        // Enumerate every vector that has a .bit under this rate.
        let mut stems: BTreeSet<String> = BTreeSet::new();
        for e in fs::read_dir(&subdir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("bit") {
                if let Some(s) = p.file_stem().and_then(|s| s.to_str()) {
                    if filter.is_empty() || filter.contains(s) {
                        stems.insert(s.to_string());
                    }
                }
            }
        }

        for v in &stems {
            let bitp = subdir.join(format!("{v}.bit"));
            let refp = subdir.join(format!("{v}.pcm"));
            let raw_bits = fs::read(&bitp).unwrap_or_default();
            let ref_pcm = read_pcm(&refp);
            if raw_bits.is_empty() || ref_pcm.is_empty() {
                continue;
            }
            // Soft-decision vectors (`*_sd.bit`) store each channel bit as a
            // 4-bit SD nibble (4x the hard size). `--soft` decodes them through
            // the shipped soft FEC (`decode_soft`); otherwise we hard-slice each
            // MSB so the hard decoder sees valid frames instead of mis-framed
            // garbage. Non-`_sd` vectors are ordinary hard frames either way.
            let is_sd = v.ends_with("_sd");
            let hard_bits = if is_sd {
                sd_to_hard(&raw_bits)
            } else {
                raw_bits.clone()
            };
            let frames = hard_bits.len() / n;
            let sd_soft = is_sd && soft;

            let (mut dec_r, mut dec_lag) = (f64::NAN, 0i64);
            if !no_decode {
                let our_dec = if sd_soft {
                    our_decode_soft(rate, &raw_bits, n)
                } else {
                    our_decode(rate, &hard_bits, n)
                };
                let (lag, r) = emit(
                    &out.join(label).join(format!("{v}_1_decode.wav")),
                    &our_dec,
                    &ref_pcm,
                );
                dec_r = r;
                dec_lag = lag;
            }

            // Roundtrip only where a source PCM is shipped.
            let src = read_pcm(&root.join(format!("{v}.pcm")));
            let (mut rt_r, mut rt_lag) = (f64::NAN, 0i64);
            let mut note = String::new();
            if src.len() >= FRAME * 2 {
                if !no_roundtrip {
                    let our_rt = our_roundtrip(rate, &src);
                    let (lag, r) = emit(
                        &out.join(label).join(format!("{v}_2_roundtrip.wav")),
                        &our_rt,
                        &ref_pcm,
                    );
                    rt_r = r;
                    rt_lag = lag;
                }
            } else {
                note = "decode-only (no source)".into();
            }
            if is_sd {
                let tag = if sd_soft {
                    "sd:soft-fec"
                } else {
                    "sd:hard-sliced"
                };
                note = if note.is_empty() {
                    tag.into()
                } else {
                    format!("{tag}; {note}")
                };
            }

            csv.push_str(&format!(
                "{label},{v},{frames},{},{dec_lag},{},{rt_lag},{note}\n",
                fmt(dec_r),
                fmt(rt_r)
            ));
            rows.push((label.to_string(), v.clone(), frames, dec_r, rt_r));
        }
    }

    fs::create_dir_all(&out).ok();
    fs::write(out.join("metrics.csv"), &csv).unwrap();

    // Print worst-first within each rate so anything "off" is at the top.
    for (_, sub, label, _) in REFS {
        let _ = sub;
        let mut r: Vec<_> = rows.iter().filter(|x| x.0 == label).collect();
        if r.is_empty() {
            continue;
        }
        r.sort_by(|a, b| nan_low(a.3).partial_cmp(&nan_low(b.3)).unwrap());
        println!("== {label}  (worst decode_env_r first) ==");
        println!(
            "{:<14} {:>6} {:>10} {:>13}",
            "vector", "frames", "decode_r", "roundtrip_r"
        );
        for (_, v, f, dr, rr) in r {
            println!("{v:<14} {f:>6} {:>10} {:>13}", fmt(*dr), fmt(*rr));
        }
        println!();
    }
    println!("metrics.csv + WAVs under {}/", out.display());
}

fn fmt(x: f64) -> String {
    if x.is_nan() {
        "-".into()
    } else {
        format!("{x:.3}")
    }
}
fn nan_low(x: f64) -> f64 {
    if x.is_nan() {
        2.0
    } else {
        x
    }
}
