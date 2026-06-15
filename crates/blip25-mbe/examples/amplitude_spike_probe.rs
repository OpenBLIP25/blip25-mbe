//! Build a clean (no-error) AMBE+2 stream where a single high-gain
//! "spike" frame is inserted into a low-gain steady state. This isolates
//! the chip's §1.11.3 amplitude smoothing (τ_M / γ_M) clamp behavior from
//! any FEC-error path.
//!
//! Frame schedule:
//!   0-49: low-gain steady (b̂₂ = LOW)
//!   50:   spike (b̂₂ = HIGH)
//!   51-99: low-gain steady
//!
//! Usage: amplitude_spike_probe <low_b2> <high_b2>
//! Output: 100 × 9 bytes = 900 bytes .ambe9 stream on stdout.

use blip25_mbe::rate33::dequantize::{decode_to_params, Decoded, DecoderState};
use blip25_mbe::rate33::frame::{encode_frame, DIBITS_PER_FRAME};
use blip25_mbe::rate33::priority::{prioritize, AMBE_B_COUNT};

fn pack_dibits(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; 9] {
    let mut out = [0u8; 9];
    let mut bit = 0;
    for &d in dibits {
        for pos in (0..2).rev() {
            let b = (d >> pos) & 1;
            out[bit / 8] |= b << (7 - (bit % 8));
            bit += 1;
        }
    }
    out
}

fn build_frame(b2: u16) -> [u8; 9] {
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = 60; // pitch
    b[1] = 0; // all voiced
    b[2] = b2; // gain (varied)
    let info = prioritize(&b);
    let mut _dec = DecoderState::new();
    if !matches!(decode_to_params(&info, &mut _dec), Ok(Decoded::Voice(_))) {
        eprintln!("frame with b2={} failed to decode as voice", b2);
        std::process::exit(1);
    }
    let dibits = encode_frame(&info);
    pack_dibits(&dibits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let low: u16 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(8);
    let high: u16 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(28);
    let low_bytes = build_frame(low);
    let high_bytes = build_frame(high);

    let mut out = Vec::with_capacity(100 * 9);
    for i in 0..100 {
        let frame = if i == 50 { high_bytes } else { low_bytes };
        out.extend_from_slice(&frame);
    }
    use std::io::Write;
    std::io::stdout().write_all(&out).unwrap();
    eprintln!(
        "Wrote 100 frames: low b̂₂={} except f=50 with b̂₂={}",
        low, high
    );
}
