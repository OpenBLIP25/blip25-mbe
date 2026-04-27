//! Take chip.bit frame 0, decode → info[], re-encode → dibits, XOR
//! against the original dibits. Print where bits differ by codeword
//! slot (c0..c7) to localize the FEC-count-mismatch bug.

use blip25_mbe::imbe_wire::fec::deinterleave;
use blip25_mbe::imbe_wire::frame::{decode_frame, encode_frame};
use std::fs;

fn unpack_dibits_msb(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
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

fn popcount_u32(x: u32) -> u32 {
    x.count_ones()
}

fn main() {
    let path = std::env::args().nth(1).expect("bit file arg");
    let frame_idx: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("frame idx"))
        .unwrap_or(0);
    let bytes = fs::read(&path).unwrap();
    let chunk = &bytes[frame_idx * 18..(frame_idx + 1) * 18];
    let dibits = unpack_dibits_msb(chunk);

    // Decode to info[], re-encode back to dibits.
    let frame = decode_frame(&dibits);
    println!(
        "frame {frame_idx}: info = {:?}",
        frame.info.map(|w| format!("{:04x}", w))
    );
    println!("  errors per slot: {:?}", frame.errors);
    let re_dibits = encode_frame(&frame.info);

    // Deinterleave both, compare per-codeword.
    let c_orig = deinterleave(&dibits);
    let c_roundtrip = deinterleave(&re_dibits);
    let widths = [23u32, 23, 23, 23, 15, 15, 15, 7];
    let mut total_diff = 0u32;
    for i in 0..8 {
        let mask = if widths[i] == 32 { u32::MAX } else { (1u32 << widths[i]) - 1 };
        let a = c_orig[i] & mask;
        let b = c_roundtrip[i] & mask;
        let xor = a ^ b;
        let diff = popcount_u32(xor);
        total_diff += diff;
        println!(
            "  c[{i}] orig=0x{a:07x} rt=0x{b:07x}  diff={diff} xor=0b{xor:023b}",
        );
    }
    println!("  TOTAL bit-diff (c̃-level): {total_diff}");

    // Also show the raw dibit diff in hex.
    let mut raw_diff = 0u32;
    for (o, r) in dibits.iter().zip(re_dibits.iter()) {
        raw_diff += (o ^ r).count_ones();
    }
    println!("  dibit-level raw diff: {raw_diff} bits across 144");
}
