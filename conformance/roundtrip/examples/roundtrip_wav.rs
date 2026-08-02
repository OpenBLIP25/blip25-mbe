//! Encode a PCM file through our AMBE+2 rate-33 encoder and decode it straight
//! back, writing the result as a WAV for listening.
//!
//! Usage:
//!   roundtrip_wav <in.pcm> <out.wav>
//!
//! This is the end-to-end ear check: whatever the field-match scoreboards say,
//! this is what the codec actually sounds like. Set `BLIP25_NO_ENHANCEMENT`
//! to bypass the outbound conditioning chain and hear the bare codec.
//!
//! Field-match percentages and perceptual quality are **not** the same axis:
//! driving a field's agreement up can cost audible quality. A listening pass
//! is a required check on any promotion, not a formality.

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::env;
use std::fs;

const FRAME_SAMPLES: usize = 160;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: roundtrip_wav <in.pcm> <out.wav>");
        std::process::exit(2);
    }
    let raw = fs::read(&args[1])?;
    anyhow::ensure!(raw.len() % 2 == 0, "input not 16-bit aligned");
    let pcm: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n_frames = pcm.len() / FRAME_SAMPLES;

    let mut enc = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let mut dec = Vocoder::new(Rate::AmbePlus2_3600x2450);
    // The outbound conditioning chain is default-ON and includes an output-gain
    // stage, so it confounds any measurement of the codec's own level
    // behaviour. This switch isolates the codec.
    if std::env::var("BLIP25_NO_ENHANCEMENT").is_ok() {
        enc.set_enhancement(blip25_mbe::enhancement::EnhancementMode::None);
        dec.set_enhancement(blip25_mbe::enhancement::EnhancementMode::None);
    }
    let mut out: Vec<i16> = Vec::with_capacity(n_frames * FRAME_SAMPLES);
    for f in 0..n_frames {
        let bits = enc.encode_pcm(&pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES])?;
        out.extend_from_slice(&dec.decode_bits(&bits)?);
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&args[2], spec)?;
    for s in &out {
        w.write_sample(*s)?;
    }
    w.finalize()?;

    let rms =
        |v: &[i16]| (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    eprintln!(
        "{} frames -> {} ({} samples, in_rms {:.0}, out_rms {:.0}, ratio {:.3})",
        n_frames,
        &args[2],
        out.len(),
        rms(&pcm),
        rms(&out),
        rms(&out) / rms(&pcm).max(1.0)
    );
    Ok(())
}
