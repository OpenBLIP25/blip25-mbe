//! Dump per-frame quantized AMBE+2 half-rate fields `b̂₀..b̂₈` from a
//! stream of half-rate frames. One frame per line, 9 space-separated
//! decimals.
//!
//! `b̂₀` = pitch, `b̂₁` = V/UV VQ index, `b̂₂` = differential-gain index,
//! `b̂₃..b̂₈` = PRBA + HOC spectral. Used to measure per-field encoder
//! agreement (mean |Δ|, lag-1 autocorr) the way Miranda's OTA report does.
//!
//! Usage: halfrate_field_dump <file> [fec9|natural7|nofec7]
//!   fec9     (default): 9-byte P25 Annex H FEC frames (`our.bit`).
//!   natural7: 7-byte NATURAL-order (AMBE_d) frames — 49 info bits
//!            MSB-first û₀(12)‖û₁(12)‖û₂(11)‖û₃(14) + 7 pad. This is the
//!            IDAS/mbelib over-the-air layout (e.g. VE-PG4
//!            `pg4_ambe_49bit_canonical_7b.bin`).
//!   nofec7:  7-byte R34 column-interleaved no-FEC frames — the layout
//!            blip25's LiveEncoder / NXDN / Fusion consoles emit.
//!
//! All three modes are deprioritized into the SAME b̂₀..b̂₈ fields for an
//! apples-to-apples per-field comparison. The natural7 / nofec7 paths use
//! the public `blip25_mbe::rate33::fields_from_natural` /
//! `fields_from_no_fec` one-liners; flat-slicing the prioritized 49-bit
//! vector instead would mix multiple parameters per slice.

use blip25_mbe::rate33::priority::AMBE_B_COUNT;
use blip25_mbe::rate33::{fields_from_fec, fields_from_natural, fields_from_no_fec};
use std::fs;

/// Frame bytes → the nine deprioritized `b̂₀..b̂₈` parameters.
type Extract = fn(&[u8]) -> [u16; AMBE_B_COUNT];

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: halfrate_field_dump <file> [fec9|natural7|nofec7]");
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "fec9".to_string());
    let bytes = fs::read(&path).unwrap();
    let (stride, fields): (usize, Extract) = match mode.as_str() {
        "natural7" => (7, fields_from_natural),
        "nofec7" => (7, fields_from_no_fec),
        _ => (9, fields_from_fec),
    };
    for f in 0..(bytes.len() / stride) {
        let b = fields(&bytes[f * stride..(f + 1) * stride]);
        let line: Vec<String> = b.iter().map(|x| x.to_string()).collect();
        println!("{}", line.join(" "));
    }
}
