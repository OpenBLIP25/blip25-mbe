//! Build a clean stream with a single-frame spectral discontinuity:
//! steady low-pitch frames with one frame inserted at high pitch
//! (different L). Isolates chip's response to a frame-to-frame
//! spectral-envelope mismatch (vs the simpler gain-only case probed
//! in amplitude_spike_probe).
//!
//! Usage: spectral_jump_probe <low_b0> <high_b0> [b2]
//!   low_b0 / high_b0: pitch index 0-119 (60 ≈ pitch 100 Hz)
//!   b2 (default 20): gain index 0-31 (held same on both)

use blip25_mbe::rate33::dequantize::{DecoderState, Decoded, decode_to_params};
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

fn build_frame(b0: u16, b2: u16) -> ([u8; 9], usize) {
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = b0;
    b[1] = 0;
    b[2] = b2;
    let info = prioritize(&b);
    let mut _dec = DecoderState::new();
    let l = match decode_to_params(&info, &mut _dec) {
        Ok(Decoded::Voice(p)) => p.harmonic_count() as usize,
        _ => {
            eprintln!("frame with b0={} b2={} failed to decode", b0, b2);
            std::process::exit(1);
        }
    };
    let dibits = encode_frame(&info);
    (pack_dibits(&dibits), l)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let low_b0: u16 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(60);
    let high_b0: u16 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let b2: u16 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(20);

    let (low_bytes, low_l) = build_frame(low_b0, b2);
    let (high_bytes, high_l) = build_frame(high_b0, b2);
    eprintln!("low: b0={} → L={}; high: b0={} → L={}; b2={} on both",
              low_b0, low_l, high_b0, high_l, b2);

    let mut out = Vec::with_capacity(100 * 9);
    for i in 0..100 {
        let frame = if i == 50 { high_bytes } else { low_bytes };
        out.extend_from_slice(&frame);
    }
    use std::io::Write;
    std::io::stdout().write_all(&out).unwrap();
}
