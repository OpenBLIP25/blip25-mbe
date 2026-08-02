//! Soft-decision decode must never panic on hostile LLRs.
//!
//! LLRs come from a demodulator's confidence estimate, so their magnitudes are
//! entirely untrusted — including all-saturated (`i8::MIN` / `i8::MAX`) inputs
//! that drive the Chase-II search into its worst case.

#![no_main]

use blip25_mbe::vocoder::{Rate, Vocoder};
use libfuzzer_sys::fuzz_target;

// Only the FEC-bearing rates accept soft input; the no-FEC rates must return
// SoftUnsupported rather than panic, so both are included on purpose.
const RATES: [Rate; 4] = [
    Rate::Imbe7200x4400,
    Rate::AmbePlus2_3600x2450,
    Rate::Imbe4400x4400,
    Rate::AmbePlus2_2450x2450,
];

fuzz_target!(|data: &[u8]| {
    let Some((&sel, payload)) = data.split_first() else {
        return;
    };
    let rate = RATES[sel as usize % RATES.len()];

    // Reinterpret the payload as signed LLRs.
    let llrs: Vec<i8> = payload.iter().map(|&b| b as i8).collect();

    let mut v = Vocoder::new(rate);

    match v.soft_frame_bits() {
        Some(n) => {
            for chunk in llrs.chunks(n) {
                let _ = v.decode_soft(chunk);
            }
        }
        // No-FEC rate: must reject cleanly at any length.
        None => {
            let _ = v.decode_soft(&llrs);
        }
    }
});
