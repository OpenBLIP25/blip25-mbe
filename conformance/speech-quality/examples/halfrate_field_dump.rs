//! Dump per-frame quantized AMBE+2 half-rate fields `b̂₀..b̂₈` from a
//! stream of 9-byte FEC frames (the `our.bit` format produced by
//! `halfrate-ab-matrix`). One frame per line, 9 space-separated decimals.
//!
//! `b̂₀` = pitch, `b̂₁` = V/UV VQ index, `b̂₂` = differential-gain index,
//! `b̂₃..b̂₈` = PRBA + HOC spectral. Used to measure parameter-trajectory
//! jumpiness (mean |Δ|, lag-1 autocorr) the way Miranda's OTA report does.
//!
//! Usage: halfrate_field_dump <file> [fec9|natural7]
//!   fec9     (default): 9-byte P25 Annex H FEC frames (`our.bit`).
//!   natural7: 7-byte NATURAL-order (AMBE_d) frames — 49 info bits
//!            MSB-first û₀(12)‖û₁(12)‖û₂(11)‖û₃(14) + 7 pad. This is
//!            the IDAS/mbelib over-the-air layout (e.g. VE-PG4
//!            `pg4_ambe_49bit_canonical_7b.bin`); it is deprioritized
//!            into the SAME b̂₀..b̂₈ fields for an apples-to-apples
//!            per-field comparison against our encoder.

use blip25_mbe::rate33::frame::{decode_frame, DIBITS_PER_FRAME, INFO_WIDTHS};
use blip25_mbe::rate33::priority::deprioritize;
use std::fs;

fn unpack(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
    let mut bit = 0usize;
    for slot in &mut out {
        let mut d = 0u8;
        for _ in 0..2 {
            let b = (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            d = (d << 1) | b;
            bit += 1;
        }
        *slot = d;
    }
    out
}

/// Pack 7 bytes of natural-order (AMBE_d) info bits into `û₀..û₃`.
fn natural7_to_info(bytes: &[u8]) -> [u16; 4] {
    let mut natural = [0u8; 49];
    for (i, slot) in natural.iter_mut().enumerate() {
        *slot = (bytes[i / 8] >> (7 - (i % 8))) & 1;
    }
    let mut info = [0u16; 4];
    let mut idx = 0usize;
    for (oi, &w) in INFO_WIDTHS.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | u16::from(natural[idx]);
            idx += 1;
        }
        info[oi] = v;
    }
    info
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: halfrate_field_dump <file> [fec9|natural7]");
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "fec9".to_string());
    let bytes = fs::read(&path).unwrap();
    let (stride, natural) = match mode.as_str() {
        "natural7" => (7usize, true),
        _ => (9usize, false),
    };
    let n = bytes.len() / stride;
    for f in 0..n {
        let chunk = &bytes[f * stride..(f + 1) * stride];
        let info = if natural {
            natural7_to_info(chunk)
        } else {
            decode_frame(&unpack(chunk)).info
        };
        let b = deprioritize(&info);
        let line: Vec<String> = b.iter().map(|x| x.to_string()).collect();
        println!("{}", line.join(" "));
    }
}
