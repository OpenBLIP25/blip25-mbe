//! How often the encode-side silence gate binds on real speech.
//!
//! The gate clamps `b2 <= SILENCE_GAIN_CAP (2)` on source frames whose RMS is
//! below `SILENCE_RMS (35.0)`. It only does anything on frames where BOTH hold
//! AND the gain law's natural `b2` exceeds 2. Counts those over the corpus.
//!
//! Usage: `cargo run --release --example roomtone_probe [vector-root]`
//! (default root `reference-material/Vectors/tv-rc`).
use blip25_mbe::vocoder::{Rate, Vocoder};
use std::fs;

const SILENCE_RMS: f64 = 35.0;
const CAP: u16 = 2;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reference-material/Vectors/tv-rc".into());
    let mut names: Vec<String> = fs::read_dir(&root)
        .expect("root")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "pcm").then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();

    let (mut quiet, mut binds, mut frames_tot) = (0usize, 0usize, 0usize);
    let mut binding_vectors: Vec<(String, usize, usize)> = Vec::new();

    for name in &names {
        let Ok(raw) = fs::read(format!("{root}/{name}.pcm")) else {
            continue;
        };
        let pcm: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        if pcm.len() < 160 * 48 {
            continue;
        }
        let rms: Vec<f64> = pcm
            .chunks_exact(160)
            .map(|c| (c.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / 160.0).sqrt())
            .collect();
        // Shipped path (gate applied).
        let frames = Vocoder::new(Rate::AmbePlus2_3600x2450).encode(&pcm);
        // Pre-gate reference: the same engine chain with the reference pitch forced,
        // but WITHOUT the vocoder's post-quantization b2 clamp.
        let mut e = blip25_codec::enc::Encoder::new();
        let reference = blip25_codec::enc::pcm_encode::encode_pcm_b0(
            &pcm,
            blip25_codec::enc::pcm_encode::EncodeOpts::default(),
        );
        if !reference.is_empty() {
            let nf = pcm.len() / 160;
            e.set_forced_b0((0..nf).map(|f| reference[(f + 2).min(reference.len() - 1)]).collect());
        }
        let mut raw_frames = Vec::new();
        for f in pcm.chunks_exact(160) {
            if let Some(x) = e.encode_frame_r33(f.try_into().unwrap()) {
                raw_frames.push(x);
            }
        }
        if !raw_frames.is_empty() {
            raw_frames.remove(0);
        }
        raw_frames.extend(e.flush_r33());

        let (mut q, mut b) = (0usize, 0usize);
        for (i, f) in frames.iter().enumerate() {
            let Some(rf) = raw_frames.get(i) else { break };
            let b2 =
                blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(f).info)[2];
            let b2_raw =
                blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(rf).info)[2];
            frames_tot += 1;
            if rms.get(i + 1).is_some_and(|&r| r < SILENCE_RMS) {
                q += 1;
                if b2_raw > CAP && b2 != b2_raw {
                    b += 1;
                }
            }
        }
        quiet += q;
        binds += b;
        if b > 0 {
            binding_vectors.push((name.clone(), q, b));
        }
    }

    println!("frames total          : {frames_tot}");
    println!(
        "quiet (rms < {SILENCE_RMS})   : {quiet}  ({:.2}%)",
        100.0 * quiet as f64 / frames_tot.max(1) as f64
    );
    println!(
        "gate ACTUALLY clamped  : {binds}  ({:.3}% of all frames)",
        100.0 * binds as f64 / frames_tot.max(1) as f64
    );
    println!("\nvectors with any such frame ({}):", binding_vectors.len());
    for (n, q, b) in binding_vectors.iter().take(15) {
        println!("  {n:<14} quiet={q:<6} clamped={b}");
    }
}
