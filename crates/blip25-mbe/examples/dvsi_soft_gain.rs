//! Soft-decision FEC gain test against DVSI's own reference vectors.
//!
//! For each `clean_eN_sd.bit` (full-rate IMBE WITH FEC, soft-decision
//! format, N% injected channel BER), decode every frame two ways and
//! score both against the ground-truth info recovered from the clean,
//! error-free `clean.bit`:
//!
//!   * **soft** — `decode_frame_soft` over the 4-bit SD values
//!     (Chase-II soft Golay/Hamming, the chip's soft-decision path)
//!   * **hard** — hard-slice each SD value's MSB, then `decode_frame`
//!     (bounded-distance Golay/Hamming, the baseline)
//!
//! The soft decoder should recover strictly more frames as BER climbs —
//! the Golay/Hamming coding gain. Pure info-bit metric (no synthesis),
//! so the result is deterministic and unambiguous.
//!
//! Usage:
//!   cargo run --release -p blip25-mbe --example dvsi_soft_gain -- [VECTOR_DIR]
//!
//! VECTOR_DIR defaults to the DVSI `tv-std/tv/p25` directory.

use blip25_mbe::dvsi_soft_decision::unpack_nibble_stream;
use blip25_mbe::imbe_wire::frame::{decode_frame, decode_frame_soft, Frame};

const HARD_BYTES_PER_FRAME: usize = 18; // 144 channel bits packed 8/byte
const SOFT_BYTES_PER_FRAME: usize = 72; // 144 SD nibbles packed 2/byte
const DEFAULT_DIR: &str =
    "/mnt/share/P25-IQ-Samples/Research/DVSI Vectors/tv-std/tv/p25";

/// MSB-first 18 bytes -> 72 dibits, dibit = (hi<<1)|lo, matching the
/// crate's `unpack_dibits_n::<72>` convention.
fn hard_frame_to_dibits(frame: &[u8]) -> [u8; 72] {
    let mut bits = [0u8; 144];
    for (i, &b) in frame.iter().enumerate() {
        for k in 0..8 {
            bits[i * 8 + k] = (b >> (7 - k)) & 1;
        }
    }
    let mut dibits = [0u8; 72];
    for (d, slot) in dibits.iter_mut().enumerate() {
        *slot = (bits[2 * d] << 1) | bits[2 * d + 1];
    }
    dibits
}

/// 72 SD-nibble bytes -> 144 soft LLRs (SD0 first).
fn soft_frame_to_llrs(frame: &[u8]) -> [i8; 144] {
    let llrs = unpack_nibble_stream(frame);
    let mut out = [0i8; 144];
    out.copy_from_slice(&llrs[..144]);
    out
}

/// Hard decision of a soft frame: slice each LLR's sign, regroup dibits.
fn soft_llrs_to_dibits(soft: &[i8; 144]) -> [u8; 72] {
    let mut dibits = [0u8; 72];
    for (d, slot) in dibits.iter_mut().enumerate() {
        let hi = u8::from(soft[2 * d] > 0);
        let lo = u8::from(soft[2 * d + 1] > 0);
        *slot = (hi << 1) | lo;
    }
    dibits
}

fn frames_eq(a: &Frame, b: &Frame) -> bool {
    a.info == b.info
}

fn vector_mismatches(a: &Frame, b: &Frame) -> u32 {
    (0..8).filter(|&i| a.info[i] != b.info[i]).count() as u32
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_DIR.to_string());

    // Ground truth: decode the clean, error-free .bit.
    let clean = std::fs::read(format!("{dir}/clean.bit"))
        .unwrap_or_else(|e| panic!("read {dir}/clean.bit: {e}"));
    let truth: Vec<Frame> = clean
        .chunks_exact(HARD_BYTES_PER_FRAME)
        .map(|c| decode_frame(&hard_frame_to_dibits(c)))
        .collect();
    let truth_resid: u32 = truth.iter().map(|f| f.error_total() as u32).sum();
    println!(
        "ground truth: {} frames from clean.bit (residual FEC errors {} — \
         should be ~0 if bit ordering matches)\n",
        truth.len(),
        truth_resid
    );

    println!(
        "{:<8} {:>7} {:>14} {:>14} {:>10} {:>12} {:>12}",
        "vector", "frames", "hard ok", "soft ok", "rescued", "hard verr", "soft verr"
    );
    println!("{}", "-".repeat(82));

    for level in ["e1", "e2", "e5", "e10"] {
        let path = format!("{dir}/clean_{level}_sd.bit");
        let Ok(buf) = std::fs::read(&path) else {
            println!("{level:<8} (missing: {path})");
            continue;
        };

        let mut hard_ok = 0u32;
        let mut soft_ok = 0u32;
        let mut rescued = 0u32; // soft correct AND hard wrong
        let mut hard_verr = 0u32;
        let mut soft_verr = 0u32;

        for (i, chunk) in buf.chunks_exact(SOFT_BYTES_PER_FRAME).enumerate() {
            if i >= truth.len() {
                break;
            }
            let soft = soft_frame_to_llrs(chunk);
            let f_soft = decode_frame_soft(&soft);
            let f_hard = decode_frame(&soft_llrs_to_dibits(&soft));

            let h = frames_eq(&f_hard, &truth[i]);
            let s = frames_eq(&f_soft, &truth[i]);
            hard_ok += u32::from(h);
            soft_ok += u32::from(s);
            rescued += u32::from(s && !h);
            hard_verr += vector_mismatches(&f_hard, &truth[i]);
            soft_verr += vector_mismatches(&f_soft, &truth[i]);
        }

        let n = (buf.len() / SOFT_BYTES_PER_FRAME).min(truth.len());
        let pct = |x: u32| 100.0 * x as f64 / n as f64;
        println!(
            "{:<8} {:>7} {:>8} {:>5.1}% {:>8} {:>5.1}% {:>10} {:>12} {:>12}",
            level,
            n,
            hard_ok,
            pct(hard_ok),
            soft_ok,
            pct(soft_ok),
            rescued,
            hard_verr,
            soft_verr,
        );
    }

    println!(
        "\nframe ok = all 8 info vectors match clean truth; verr = total \
         mismatched info vectors (lower is better); rescued = frames soft \
         got right that hard got wrong."
    );
}
