//! Probe chip's post-Repeat behaviour following a *spectrally loud*
//! previous frame. Mimics #3537 f=772-774 structure:
//!   f=49: switch params to a louder/different spectral envelope
//!   f=50: inject 4-error c̃₀ → forces Repeat → replays f=49's loud
//!         params (the spec-faithful Eq. 99-104 path)
//!   f=51+: clean steady frames (recovery)
//!
//! Output: 100 frames × 9 bytes on stdout (--binary) or hex per line.
//!
//! Schedule controls (defaults match the #3537 case):
//!   --steady_b0    pitch index of steady frames (default 30 → L=14)
//!   --spike_b0     pitch index of f=49 (default 110 → L=49)
//!   --steady_b2    gain index of steady frames (default 16)
//!   --spike_b2     gain index of f=49 (default 28 — louder)

use blip25_mbe::rate33::dequantize::{
    DecoderState, Decoded, decode_to_params,
};
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

fn inject_errors_in_c0(bytes: [u8; 9], n_errors: u32) -> [u8; 9] {
    let dibits = unpack_dibits(&bytes);
    let mut codewords = deinterleave(&dibits);
    codewords[0] ^= (1u32 << n_errors) - 1;
    pack_dibits(&interleave(&codewords))
}

fn build_frame(b0: u16, b2: u16) -> [u8; 9] {
    let mut b = [0u16; AMBE_B_COUNT];
    b[0] = b0; b[1] = 0; b[2] = b2;
    let info = prioritize(&b);
    let mut _dec = DecoderState::new();
    if !matches!(decode_to_params(&info, &mut _dec), Ok(Decoded::Voice(_))) {
        eprintln!("frame b0={} b2={} did not decode as voice", b0, b2);
        std::process::exit(1);
    }
    pack_dibits(&encode_frame(&info))
}

fn arg(args: &[String], flag: &str, dflt: u16) -> u16 {
    for w in args.windows(2) {
        if w[0] == flag { return w[1].parse().unwrap_or(dflt); }
    }
    dflt
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steady_b0 = arg(&args, "--steady_b0", 30);
    let spike_b0  = arg(&args, "--spike_b0",  110);
    let steady_b2 = arg(&args, "--steady_b2", 16);
    let spike_b2  = arg(&args, "--spike_b2",  28);
    let binary = args.iter().any(|a| a == "--binary");

    let steady = build_frame(steady_b0, steady_b2);
    let spike  = build_frame(spike_b0, spike_b2);

    let mut out = Vec::with_capacity(100 * 9);
    for i in 0..100 {
        let bytes = if i == 49 {
            spike
        } else if i == 50 {
            // 4-error c̃₀ → uncorrectable → ε₀=u8::MAX → after our cap → ε₀=4
            // → Eq. 198 Repeat branch → replays f=49 (spike) params.
            inject_errors_in_c0(spike, 4)
        } else {
            steady
        };
        out.extend_from_slice(&bytes);
    }

    if binary {
        use std::io::Write;
        std::io::stdout().write_all(&out).unwrap();
    } else {
        for chunk in out.chunks_exact(9) {
            let hex: String = chunk.iter().map(|b| format!("{:02x}", b)).collect();
            println!("{}", hex);
        }
    }
    eprintln!(
        "steady b0={} b2={}; spike at f=49 (b0={} b2={}); 4-err Repeat at f=50 (replays f=49)",
        steady_b0, steady_b2, spike_b0, spike_b2
    );
}
