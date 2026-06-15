//! For a target 23-bit codeword, find all 12-bit info values whose
//! our-Golay encoding has Hamming distance ≤ D from the target. Helps
//! identify whether chip's Golay is the same as ours (distance 0 for
//! some info) or a different encoding (distance > 3 for every info).

use blip25_mbe::fec::golay_23_12_encode;

fn main() {
    let arg = std::env::args().nth(1).expect("target hex (23-bit)");
    let target: u32 = u32::from_str_radix(&arg, 16).expect("hex parse");
    let max_dist: u32 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("max-dist"))
        .unwrap_or(3);
    println!(
        "target 0x{:07x} (binary {:023b})",
        target & 0x7f_ffff,
        target & 0x7f_ffff
    );
    let mut hist = [0u32; 24];
    for info in 0u16..=0xFFF {
        let cw = golay_23_12_encode(info);
        let d = (cw ^ (target & 0x7f_ffff)).count_ones();
        hist[d as usize] += 1;
        if d <= max_dist {
            println!("  info=0x{:03x}  encoded=0x{:07x}  dist={d}", info, cw);
        }
    }
    println!("dist histogram:");
    for (d, c) in hist.iter().enumerate() {
        if *c > 0 {
            println!("  dist={d:2}  count={c}");
        }
    }
}
