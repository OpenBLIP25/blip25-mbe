//! Rate conversion must never panic on hostile wire bits.
//!
//! `Transcoder` walks bits through the parameter domain — dequantize on one
//! wire, requantize on the other — so a malformed frame propagates through two
//! quantizer index paths rather than one.

#![no_main]

use blip25_mbe::vocoder::{Rate, Transcoder};
use libfuzzer_sys::fuzz_target;

const RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::AmbePlus2_2450x2450,
];

fuzz_target!(|data: &[u8]| {
    // Two selector bytes: source rate and destination rate. Invalid pairs
    // (same-rate, unsupported direction) must be rejected by `new`, not panic.
    if data.len() < 2 {
        return;
    }
    let from = RATES[data[0] as usize % RATES.len()];
    let to = RATES[data[1] as usize % RATES.len()];
    let payload = &data[2..];

    let Ok(mut t) = Transcoder::new(from, to) else {
        return;
    };

    let n = from.fec_frame_bytes();
    for chunk in payload.chunks(n) {
        let _ = t.transcode(chunk);
    }

    t.reset();
    let _ = t.transcode(payload);
});
