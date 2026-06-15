//! Build a controlled error-injection probe:
//!   - 100 clean all-voiced frames
//!   - Inject N-error bursts at known frame indices to trigger Repeat
//!   - Surrounding frames remain clean
//!
//! Output: 100 × 9 bytes = 900 bytes .ambe9 stream.

use blip25_mbe::rate33::dequantize::{decode_to_params, Decoded, DecoderState};
use blip25_mbe::rate33::frame::{deinterleave, encode_frame, interleave, DIBITS_PER_FRAME};
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

fn unpack_dibits(bytes: &[u8; 9]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
    let mut bit = 0;
    for slot in out.iter_mut() {
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

/// Flip the bottom N bits of c̃₀ (extended Golay 24-bit codeword). The chip's
/// Golay decoder corrects up to 3 errors and reports the count as ε₀.
fn inject_errors_in_c0(bytes: [u8; 9], n_errors: u32) -> [u8; 9] {
    let dibits = unpack_dibits(&bytes);
    let mut codewords = deinterleave(&dibits);
    // Flip bottom n_errors bits of c0
    let flip_mask: u32 = (1u32 << n_errors) - 1;
    codewords[0] ^= flip_mask;
    let new_dibits = interleave(&codewords);
    pack_dibits(&new_dibits)
}

fn main() {
    // All-voiced clean reference frame
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = 60;
    b[1] = 0;
    b[2] = 16; // pitch, all-voiced, mid gain; rest=0
    let info = prioritize(&b);
    // verify it decodes
    let mut _dec = DecoderState::new();
    if !matches!(decode_to_params(&info, &mut _dec), Ok(Decoded::Voice(_))) {
        eprintln!("clean probe frame failed to decode as voice");
        std::process::exit(1);
    }
    let dibits = encode_frame(&info);
    let clean = pack_dibits(&dibits);

    // Build the probe stream
    // Frame schedule:
    //   0-49: clean (warmup)
    //   50: 4-error burst (ε₀=4 → Repeat trigger, "ε₀ ≥ 4" branch)
    //   51-69: clean
    //   70: 2-error burst (ε₀=2, ε_T ≥ 2)
    //   71-89: clean
    //   90: 6-error burst (ε₀=4+ ... actually Golay only corrects 3, may be
    //                       reported as detected uncorrectable. Use 3 instead.)
    //   91-99: clean
    let mut out = Vec::with_capacity(100 * 9);
    for i in 0..100 {
        let frame = if i == 50 {
            inject_errors_in_c0(clean, 4)
        } else if i == 70 {
            inject_errors_in_c0(clean, 2)
        } else if i == 90 {
            inject_errors_in_c0(clean, 3)
        } else {
            clean
        };
        out.extend_from_slice(&frame);
    }
    let binary = std::env::args().any(|a| a == "--binary");
    if binary {
        use std::io::Write;
        std::io::stdout().write_all(&out).unwrap();
    } else {
        for chunk in out.chunks_exact(9) {
            let hex: String = chunk.iter().map(|b| format!("{:02x}", b)).collect();
            println!("{}", hex);
        }
    }
    eprintln!("Wrote 100 frames; errors injected at indices 50 (4-err), 70 (2-err), 90 (3-err)");
}
