//! Throughput micro-benchmark for [`blip25_mbe::vocoder::Vocoder`].
//!
//! Not a Criterion-grade benchmark — just a quick wall-clock
//! measurement to know the order of magnitude. Encodes /
//! decodes 1000 frames per direction at each rate, reports
//! frames/sec and a per-frame microsecond figure.
//!
//! Run:
//!   `cargo run --release --example vocoder_bench`

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::time::Instant;

const N_FRAMES: usize = 1000;
const FRAME_SAMPLES: usize = 160;

fn main() {
    let pcm = synthetic_speech(N_FRAMES);

    for &rate in &[Rate::Imbe7200x4400, Rate::AmbePlus2_3600x2450] {
        bench_rate(rate, &pcm);
    }
}

fn bench_rate(rate: Rate, pcm: &[i16]) {
    println!("\n=== {rate:?} ===");

    // Encode
    let mut tx = Vocoder::new(rate);
    let mut encoded: Vec<u8> = Vec::with_capacity(N_FRAMES * tx.fec_frame_bytes());
    let t0 = Instant::now();
    for chunk in pcm.chunks_exact(FRAME_SAMPLES) {
        encoded.extend(tx.encode_pcm(chunk).expect("encode"));
    }
    let enc_secs = t0.elapsed().as_secs_f64();
    let enc_us_per_frame = 1e6 * enc_secs / N_FRAMES as f64;
    let enc_realtime_factor = (N_FRAMES as f64 * 0.020) / enc_secs;
    println!(
        "encode: {N_FRAMES} frames in {:.3} s  {:.1} µs/frame  ({:.0}× real-time)",
        enc_secs, enc_us_per_frame, enc_realtime_factor
    );

    // Decode
    let mut rx = Vocoder::new(rate);
    let mut decoded: Vec<i16> = Vec::with_capacity(N_FRAMES * FRAME_SAMPLES);
    let t0 = Instant::now();
    for chunk in encoded.chunks_exact(rx.fec_frame_bytes()) {
        decoded.extend(rx.decode_bits(chunk).expect("decode"));
    }
    let dec_secs = t0.elapsed().as_secs_f64();
    let dec_us_per_frame = 1e6 * dec_secs / N_FRAMES as f64;
    let dec_realtime_factor = (N_FRAMES as f64 * 0.020) / dec_secs;
    println!(
        "decode: {N_FRAMES} frames in {:.3} s  {:.1} µs/frame  ({:.0}× real-time)",
        dec_secs, dec_us_per_frame, dec_realtime_factor
    );

    println!(
        "round-trip: {:.1} µs/frame  ({:.0}× real-time)",
        enc_us_per_frame + dec_us_per_frame,
        20_000.0 / (enc_us_per_frame + dec_us_per_frame),
    );
}

fn synthetic_speech(n_frames: usize) -> Vec<i16> {
    let total = n_frames * FRAME_SAMPLES;
    let mut pcm = vec![0i16; total];
    for (n, slot) in pcm.iter_mut().enumerate() {
        // Mix of two slow-changing tones — voice-like spectral content
        // for the analysis encoder to chew on.
        let s1 = 4000.0
            * (2.0 * core::f64::consts::PI * 220.0 * n as f64 / 8000.0).sin();
        let s2 = 2000.0
            * (2.0 * core::f64::consts::PI * 660.0 * n as f64 / 8000.0).sin();
        *slot = (s1 + s2).round() as i16;
    }
    pcm
}
