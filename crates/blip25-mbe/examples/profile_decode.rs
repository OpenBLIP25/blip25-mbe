//! Per-stage timing breakdown of the decode (synth) pipeline.
//!
//! Requires the `profile` feature on `blip25-mbe`:
//!
//!   cargo run --release --features blip25-mbe/profile \
//!       --example profile_decode -p blip25-mbe
//!
//! Encodes 1000 frames of synthetic speech-like input first, then
//! decodes the resulting bit stream and prints cumulative time per
//! decode stage. Use to identify the biggest optimization targets on
//! the synth side now that the encode side has been chased down to
//! ~333 µs/frame.
//!
//! Stages are defined in
//! [`blip25_mbe::codecs::mbe_baseline::analysis::profile::Stage`].

use blip25_mbe::codecs::mbe_baseline::analysis::profile::{Stage, reset, snapshot};
use blip25_mbe::vocoder::{Rate, Vocoder};
use std::time::Instant;

const N_FRAMES: usize = 1000;

fn main() {
    if !cfg!(feature = "profile") {
        eprintln!(
            "warning: the `profile` feature is OFF — per-stage \
             timings will all be zero. Re-run with:\n\
             \n\
             cargo run --release --features blip25-mbe/profile \
             --example profile_decode -p blip25-mbe\n"
        );
    }
    let pcm = synthetic_speech(N_FRAMES);

    run(Rate::Imbe7200x4400, "Phase 1", &pcm);
    println!();
    run(Rate::AmbePlus2_3600x2450, "Phase 2", &pcm);

    let _ = Stage::ALL;
}

fn run(rate: Rate, label: &str, pcm: &[i16]) {
    // Encode once (un-instrumented — we only care about decode timings).
    let mut tx = Vocoder::new(rate);
    let mut bits: Vec<Vec<u8>> = Vec::with_capacity(N_FRAMES);
    for chunk in pcm.chunks_exact(160) {
        bits.push(tx.encode_pcm(chunk).expect("encode"));
    }

    let mut rx = Vocoder::new(rate);
    reset();
    let t0 = Instant::now();
    for frame_bits in &bits {
        let _ = rx.decode_bits(frame_bits).expect("decode");
    }
    let total_secs = t0.elapsed().as_secs_f64();
    let stats = snapshot();

    println!("=== {label} decode: {N_FRAMES} frames ===");
    println!(
        "wall clock total : {:.3} s  ({:.1} µs/frame, {:.1}× realtime)",
        total_secs,
        1e6 * total_secs / N_FRAMES as f64,
        (N_FRAMES as f64 * 0.020) / total_secs,
    );
    println!(
        "instrumented sum : {:.3} s",
        stats.total_ns_sum() as f64 / 1e9
    );
    println!("{:>17} {:>14} {:>10} {:>8}", "stage", "total (µs)", "µs/frame", "%");
    let total = stats.total_ns_sum().max(1) as f64;
    let frames = stats.decode_frames.max(1);
    for (stage, ns) in stats.entries() {
        let label = stage.label();
        let total_us = ns as f64 / 1_000.0;
        let per_frame = total_us / frames as f64;
        let pct = 100.0 * (ns as f64) / total;
        println!("{label:>17} {total_us:>14.1} {per_frame:>10.2} {pct:>7.1}%");
    }
    println!("---");
    println!("frames profiled: {}", stats.decode_frames);
}

fn synthetic_speech(n_frames: usize) -> Vec<i16> {
    let total = n_frames * 160;
    let mut pcm = vec![0i16; total];
    for (n, slot) in pcm.iter_mut().enumerate() {
        let s1 = 4000.0
            * (2.0 * core::f64::consts::PI * 220.0 * n as f64 / 8000.0).sin();
        let s2 = 2000.0
            * (2.0 * core::f64::consts::PI * 660.0 * n as f64 / 8000.0).sin();
        *slot = (s1 + s2).round() as i16;
    }
    pcm
}
