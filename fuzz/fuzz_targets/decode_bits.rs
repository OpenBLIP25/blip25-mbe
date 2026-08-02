//! Hard-decision decode must never panic on attacker-controlled bytes.
//!
//! This is the highest-risk entry point in the crate: in a real deployment the
//! bytes come off the air, through a demodulator that has no idea whether it is
//! looking at a valid voice frame, noise, or a deliberately malformed one.

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
    // First byte selects the rate so one corpus exercises all four wire
    // formats; the rest is the frame payload.
    let Some((&sel, payload)) = data.split_first() else {
        return;
    };
    let rate = RATES[sel as usize % RATES.len()];

    let mut v = Vocoder::new(rate);

    // Feed the payload as a stream of frames, so cross-frame decoder state
    // (concealment counters, repeat/mute gates, synth history) is exercised
    // rather than just a single cold decode.
    let n = v.fec_frame_bytes();
    for chunk in payload.chunks(n) {
        let _ = v.decode_bits(chunk);
    }

    // A reset mid-stream must leave the handle usable.
    v.reset();
    let _ = v.decode_bits(payload);
});
