//! Encode an i16-LE 8 kHz PCM file through `Vocoder::encode_pcm` at
//! Imbe7200x4400 and write the concatenated 18-byte FEC frames as a
//! `.bit` file. Used to feed identical bits into JMBE and our decoder
//! for downstream synth-divergence probes.
//!
//! Usage: `encode_pcm_to_bit <in.pcm> <out.bit>`

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::env;
use std::fs;

const FRAME_SAMPLES: usize = 160;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: encode_pcm_to_bit <in.pcm> <out.bit>");
        std::process::exit(2);
    }
    let raw = fs::read(&args[1])?;
    if raw.len() % 2 != 0 {
        anyhow::bail!("pcm length not 16-bit aligned");
    }
    let pcm: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n_frames = pcm.len() / FRAME_SAMPLES;

    let mut voc = Vocoder::new(Rate::Imbe7200x4400);
    let mut out: Vec<u8> = Vec::with_capacity(n_frames * 18);
    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let bits = voc.encode_pcm(frame)?;
        if bits.len() != 18 {
            anyhow::bail!("frame {f}: expected 18 bytes, got {}", bits.len());
        }
        out.extend_from_slice(&bits);
    }
    fs::write(&args[2], &out)?;
    eprintln!("encoded {} frames -> {}", n_frames, &args[2]);
    Ok(())
}
