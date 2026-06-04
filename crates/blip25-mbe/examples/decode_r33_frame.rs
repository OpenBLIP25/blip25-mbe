//! Decode a single r33 (AMBE+2 half-rate FEC, 9-byte) frame given as hex
//! on the command line, print its classification and parameters, and show
//! the equivalent r34 (no-FEC, 7-byte) serialization via the R34_BIT_ORDER
//! interleave. Independent cross-check of the r34 bit order on a frame
//! that did NOT come from the DVSI vectors.
//!
//! Run: cargo run --release --example decode_r33_frame -- cea8 fe83 acc4 5820 0a

use blip25_mbe::rate33::dequantize::{Decoded, DecoderState, decode_to_params};
use blip25_mbe::rate33::frame::{
    DIBITS_PER_FRAME, INFO_WIDTHS, decode_frame, pack_no_fec, unpack_no_fec,
};

fn bytes_to_dibits(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[(2 * i) / 8] >> (7 - ((2 * i) % 8))) & 1;
        let lo = (bytes[(2 * i + 1) / 8] >> (7 - ((2 * i + 1) % 8))) & 1;
        *slot = (hi << 1) | lo;
    }
    out
}

fn main() {
    // Collect all hex on argv, strip spaces, parse to bytes.
    let hex: String = std::env::args().skip(1).collect::<Vec<_>>().join("");
    let hex: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    assert!(hex.len() == 18, "need 9 bytes (18 hex digits), got {} digits", hex.len());
    let bytes: Vec<u8> = (0..9)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    println!("r33 input (9 bytes): {bytes:02x?}");

    let dibits = bytes_to_dibits(&bytes);
    let frame = decode_frame(&dibits);
    println!("Golay/Hamming errors: {:?}", frame.errors);
    println!(
        "49 info bits  û₀..û₃ = [{:#05x}, {:#05x}, {:#05x}, {:#05x}]  (widths {:?})",
        frame.info[0], frame.info[1], frame.info[2], frame.info[3], INFO_WIDTHS
    );

    let mut st = DecoderState::new();
    match decode_to_params(&frame.info, &mut st) {
        Ok(Decoded::Voice(p)) => {
            let hz = p.omega_0() * 8000.0 / (2.0 * std::f32::consts::PI);
            println!("kind: VOICE   f0 = {hz:.1} Hz   L = {}   omega_0 = {:.5}", p.harmonic_count(), p.omega_0());
        }
        Ok(Decoded::Tone { fields, params }) => {
            let hz = params.omega_0() * 8000.0 / (2.0 * std::f32::consts::PI);
            println!(
                "kind: TONE    I_D = {} (0x{:02x})   A_D = {}   bridged f0 ≈ {hz:.1} Hz   L = {}",
                fields.id, fields.id, fields.amplitude, params.harmonic_count()
            );
        }
        Ok(Decoded::Erasure) => println!("kind: ERASURE"),
        Err(e) => println!("decode error: {e:?}"),
    }

    let r34 = pack_no_fec(&frame.info);
    println!("\nr34 (no-FEC, 7 bytes), DVSI interleave: {r34:02x?}");
    print!("r34 as hex: ");
    for b in &r34 {
        print!("{b:02x}");
    }
    println!();
    let back = unpack_no_fec(&r34);
    println!("r34 → unpack roundtrip matches info: {}", back == frame.info);
}
