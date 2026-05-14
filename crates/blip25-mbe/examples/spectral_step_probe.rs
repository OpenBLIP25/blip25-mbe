//! Build a clean stream where pitch/L *steps* once and holds at the
//! new value (vs spectral_jump_probe which is a single-frame spike).
//! Tells us whether chip's amplitude clamp is jump-detection-based
//! (fires once) or steady-state-amplitude-based (fires until A_M
//! settles).
//!
//! Frame schedule:
//!   0-49: pitch=low_b0
//!   50-99: pitch=high_b0
//!
//! Usage: spectral_step_probe <low_b0> <high_b0> [b2]

use blip25_mbe::ambe_plus2_wire::dequantize::{DecoderState, Decoded, decode_to_params};
use blip25_mbe::ambe_plus2_wire::frame::{encode_frame, DIBITS_PER_FRAME};
use blip25_mbe::ambe_plus2_wire::priority::{prioritize, AMBE_B_COUNT};

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
    b[0] = b0; b[1] = 0; b[2] = b2;
    let info = prioritize(&b);
    let mut _dec = DecoderState::new();
    let l = match decode_to_params(&info, &mut _dec) {
        Ok(Decoded::Voice(p)) => p.harmonic_count() as usize,
        _ => std::process::exit(1),
    };
    (pack_dibits(&encode_frame(&info)), l)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let low_b0: u16 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(30);
    let high_b0: u16 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(110);
    let b2: u16 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(20);

    let (low_bytes, low_l) = build_frame(low_b0, b2);
    let (high_bytes, high_l) = build_frame(high_b0, b2);
    eprintln!("low: b0={} → L={}; high: b0={} → L={}; b2={} on both",
              low_b0, low_l, high_b0, high_l, b2);

    let mut out = Vec::with_capacity(100 * 9);
    for i in 0..100 {
        let frame = if i < 50 { low_bytes } else { high_bytes };
        out.extend_from_slice(&frame);
    }
    use std::io::Write;
    std::io::stdout().write_all(&out).unwrap();
}
