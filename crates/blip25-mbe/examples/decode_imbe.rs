//! Decode an IMBE full-rate .bit file (18 bytes/frame) into PCM via our
//! codec. Mirrors the chip's `dvsi decode` subcommand.
//!
//! Usage:
//!   decode_imbe <in.bit> <out.wav> [--chip-compat]

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: decode_imbe <in.bit> <out.wav> [--chip-compat]");
        std::process::exit(2);
    }
    let chip_compat = args.iter().any(|a| a == "--chip-compat");
    let bits = std::fs::read(&args[1]).unwrap();
    let mut v = Vocoder::new(Rate::Imbe7200x4400);
    v.set_chip_compat(chip_compat);
    let fb = v.fec_frame_bytes();
    let mut pcm: Vec<i16> = Vec::new();
    for chunk in bits.chunks_exact(fb) {
        let s = v.decode_bits(chunk).unwrap();
        pcm.extend(s);
    }
    let n = pcm.len();
    let bytes_pcm = (n * 2) as u32;
    let mut f = File::create(&args[2]).unwrap();
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36 + bytes_pcm).to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.write_all(&8000u32.to_le_bytes()).unwrap();
    f.write_all(&16000u32.to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&bytes_pcm.to_le_bytes()).unwrap();
    for s in pcm {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
    eprintln!("Wrote {} ({} frames, chip_compat: {})", args[2], n / 160, chip_compat);
}
