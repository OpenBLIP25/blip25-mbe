//! Encode must never panic on hostile PCM.
//!
//! The encoder runs pitch estimation, spectral analysis, and vector
//! quantization over caller-supplied audio. Pathological input — full-scale
//! square waves, alternating rails, pure DC, all-zero silence — drives the
//! analysis chain into edge cases that ordinary speech never reaches.

#![no_main]

use blip25_mbe::vocoder::{Rate, Vocoder};
use libfuzzer_sys::fuzz_target;

const RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

fuzz_target!(|data: &[u8]| {
    let Some((&sel, payload)) = data.split_first() else {
        return;
    };
    let rate = RATES[sel as usize % RATES.len()];

    // Two bytes per sample, big-endian, so the fuzzer can reach i16::MIN and
    // i16::MAX rails directly.
    let pcm: Vec<i16> = payload
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();

    let mut v = Vocoder::new(rate);
    let n = v.frame_samples();

    for frame in pcm.chunks(n) {
        let _ = v.encode_pcm(frame);
    }

    // Wrong-length input must be rejected, not panic.
    let _ = v.encode_pcm(&pcm);
    let _ = v.flush_encode();
});
