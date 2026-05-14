//! Full-rate IMBE equivalent of `spectral_jump_probe`. Generates a
//! synthetic PCM stream with a single-frame pitch jump (steady 200 Hz
//! → 80 Hz at frame 50 → back to 200 Hz), encodes through our codec
//! at `Rate::Imbe7200x4400`, and writes the .bit stream to stdout.
//!
//! Designed for the spec-author's follow-up #1 from gap 0026
//! disposition: does DVSI's full-rate decoder apply the same
//! beyond-spec spectral-discontinuity attenuation as half-rate?
//!
//! Usage:
//!   imbe_jump_probe > /tmp/imbe_jump.bit
//! Then:
//!   ssh pve 'cd /tmp && dvsi decode-imbe imbe_jump.bit imbe_jump_chip.pcm'

use blip25_mbe::vocoder::{Rate, Vocoder};
use std::f32::consts::PI;
use std::io::Write;

fn main() {
    let sr = 8000.0f32;
    let samples_per_frame = 160;
    let n_frames = 100;
    let jump_frame = 50;
    let f_low = 200.0f32;  // steady-state pitch
    let f_high = 80.0f32;  // jump-frame pitch (lower freq → more harmonics, larger L)
    let amp = 4000.0f32;

    // Build PCM: continuous phase across the jump to avoid click that
    // isn't pitch-driven.
    let mut pcm = Vec::with_capacity(n_frames * samples_per_frame);
    let mut phase = 0.0f32;
    for f in 0..n_frames {
        let freq = if f == jump_frame { f_high } else { f_low };
        let dphase = 2.0 * PI * freq / sr;
        for _ in 0..samples_per_frame {
            pcm.push((amp * phase.sin()).round() as i16);
            phase += dphase;
            while phase >= 2.0 * PI { phase -= 2.0 * PI; }
        }
    }

    // Encode through our codec at full-rate IMBE, one frame at a time.
    let mut v = Vocoder::new(Rate::Imbe7200x4400);
    let frame_bytes = v.fec_frame_bytes();
    let mut all_bits = Vec::with_capacity(n_frames * frame_bytes);
    for chunk in pcm.chunks_exact(samples_per_frame) {
        let b = v.encode_pcm(chunk).expect("encode");
        all_bits.extend(b);
    }
    eprintln!("Wrote {} frames ({} bytes/frame, {} total bytes); jump at f={} (200 Hz → 80 Hz)",
              n_frames, frame_bytes, all_bits.len(), jump_frame);
    std::io::stdout().write_all(&all_bits).unwrap();
}
