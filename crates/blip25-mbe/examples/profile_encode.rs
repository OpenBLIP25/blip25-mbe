//! Per-stage timing breakdown of the analysis encoder.
//!
//! Requires the `profile` feature on `blip25-mbe`:
//!
//!   cargo run --release --features blip25-mbe/profile \
//!       --example profile_encode -p blip25-mbe
//!
//! Encodes 1000 frames of synthetic speech-like input through the
//! full analysis encoder, then prints the cumulative time per stage
//! plus a percentage of total encode time. Use to identify the
//! biggest remaining optimization targets after the rustfft swap
//! (commit `ed52a36`).
//!
//! Stages are defined in
//! [`blip25_mbe::codecs::mbe_baseline::analysis::profile::Stage`].

use blip25_mbe::codecs::mbe_baseline::analysis::profile::{reset, snapshot, Stage};
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
             --example profile_encode -p blip25-mbe\n"
        );
    }
    let pcm = synthetic_speech(N_FRAMES);

    let mut tx = Vocoder::new(Rate::Imbe7200x4400);
    reset();
    let t0 = Instant::now();
    for chunk in pcm.chunks_exact(160) {
        let _ = tx.encode_pcm(chunk).expect("encode");
    }
    let total_secs = t0.elapsed().as_secs_f64();
    let stats = snapshot();

    println!("=== Phase 1 encode: {N_FRAMES} frames ===");
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
    println!(
        "{:>17} {:>14} {:>10} {:>8}",
        "stage", "total (µs)", "µs/frame", "%"
    );
    let total = stats.total_ns_sum().max(1) as f64;
    for (stage, ns) in stats.entries() {
        let label = stage.label();
        let total_us = ns as f64 / 1_000.0;
        let per_frame = total_us / stats.frames.max(1) as f64;
        let pct = 100.0 * (ns as f64) / total;
        println!("{label:>17} {total_us:>14.1} {per_frame:>10.2} {pct:>7.1}%");
    }
    println!("---");
    println!(
        "frames profiled: {}  (preroll frames are excluded — only voice frames hit Stage::Hpf)",
        stats.frames
    );

    // Same for Phase 2.
    let mut tx = Vocoder::new(Rate::AmbePlus2_3600x2450);
    reset();
    let t0 = Instant::now();
    for chunk in pcm.chunks_exact(160) {
        let _ = tx.encode_pcm(chunk).expect("encode");
    }
    let total_secs = t0.elapsed().as_secs_f64();
    let stats = snapshot();

    println!("\n=== Phase 2 encode: {N_FRAMES} frames ===");
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
    println!(
        "{:>17} {:>14} {:>10} {:>8}",
        "stage", "total (µs)", "µs/frame", "%"
    );
    let total = stats.total_ns_sum().max(1) as f64;
    for (stage, ns) in stats.entries() {
        let label = stage.label();
        let total_us = ns as f64 / 1_000.0;
        let per_frame = total_us / stats.frames.max(1) as f64;
        let pct = 100.0 * (ns as f64) / total;
        println!("{label:>17} {total_us:>14.1} {per_frame:>10.2} {pct:>7.1}%");
    }
    let _ = Stage::ALL;
}

fn synthetic_speech(n_frames: usize) -> Vec<i16> {
    let total = n_frames * 160;
    let mut pcm = vec![0i16; total];
    for (n, slot) in pcm.iter_mut().enumerate() {
        let s1 = 4000.0 * (2.0 * core::f64::consts::PI * 220.0 * n as f64 / 8000.0).sin();
        let s2 = 2000.0 * (2.0 * core::f64::consts::PI * 660.0 * n as f64 / 8000.0).sin();
        *slot = (s1 + s2).round() as i16;
    }
    pcm
}
