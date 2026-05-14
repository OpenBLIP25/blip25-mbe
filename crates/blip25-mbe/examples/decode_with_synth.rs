//! Decode a half-rate .bit/.ambe9 stream with a chosen AMBE+2 synth mode.
//!
//! Usage: decode_with_synth <ambe_path> <out_wav> <AmbePlus|Baseline>

use blip25_mbe::vocoder::{AmbePlus2Synth, Rate, Vocoder};
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: decode_with_synth <in> <out.wav> <AmbePlus|Baseline> [--spectral-clamp]");
        std::process::exit(2);
    }
    let mode = match args[3].as_str() {
        "AmbePlus" | "ambeplus" => AmbePlus2Synth::AmbePlus,
        "Baseline" | "baseline" => AmbePlus2Synth::Baseline,
        other => { eprintln!("unknown synth mode: {}", other); std::process::exit(2); }
    };
    let spectral_clamp = args.iter().any(|a| a == "--spectral-clamp");
    let bits = std::fs::read(&args[1]).unwrap();
    let mut v = Vocoder::new(Rate::AmbePlus2_3600x2450);
    v.set_ambe_plus2_synth(mode);
    v.set_chip_compat_spectral_clamp(spectral_clamp);
    let fb = v.fec_frame_bytes();
    let mut pcm: Vec<i16> = Vec::new();
    for chunk in bits.chunks_exact(fb) {
        let s = v.decode_bits(chunk).unwrap();
        pcm.extend(s);
    }
    // Write WAV (8 kHz mono 16-bit)
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
    eprintln!("Wrote {} (synth mode: {:?}, spectral_clamp: {})", args[2], mode, spectral_clamp);
}
