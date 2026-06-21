//! Encode a raw 8 kHz mono i16 PCM file through the AMBE+2 half-rate
//! analysis encoder and dump per-frame info vectors (`û₀ û₁ û₂ û₃`, hex)
//! — same format as `halfrate_dump_info`, but encoding directly so we can
//! toggle the IDAS "pitch-pair" transport mode.
//!
//! Usage: idas_pairmode <pcm> <normal|pair|self>
//!
//! - `normal`: baseline (no forcing).
//! - `pair`:  odd-indexed frames (the IDAS shift-by-7 slots, whose b̂₀ the
//!            wire drops and the radio copies from the preceding even
//!            slot) are encoded with their pitch forced to the preceding
//!            even frame's *refined* ω, so the dual-permutation is
//!            lossless and the spectral bits are estimated on that ω.
//! - `self`:  control — force every frame to its own refined ω (one-pass
//!            cannot do this, so this re-uses the *previous* frame's ω as
//!            a sanity proxy; prefer `normal` vs `pair`).

use blip25_mbe::codecs::mbe_baseline::analysis::{
    encode_ambe_plus2, AnalysisOutput, AnalysisState,
};
use blip25_mbe::mbe_params::MbeParams;
use blip25_mbe::rate33::dequantize::{quantize, DecoderState};
use std::fs;

const FRAME_SAMPLES: usize = 160;

fn main() {
    let mut args = std::env::args().skip(1);
    let pcm_path = args
        .next()
        .expect("usage: idas_pairmode <pcm> <normal|pair>");
    let mode = args.next().unwrap_or_else(|| "normal".to_string());
    let pair = mode == "pair";

    let bytes = fs::read(&pcm_path).expect("read pcm");
    let pcm: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n_frames = pcm.len() / FRAME_SAMPLES;

    let mut state = AnalysisState::new();
    let mut dec = DecoderState::new();
    let mut prev_even_omega: Option<f64> = None;

    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let is_odd = f % 2 == 1;
        if pair && is_odd {
            state.set_forced_pitch_omega(prev_even_omega);
        } else {
            state.set_forced_pitch_omega(None);
        }
        let params = match encode_ambe_plus2(frame, &mut state) {
            Ok(AnalysisOutput::Voice(p)) => p,
            Ok(AnalysisOutput::Silence) => MbeParams::silence_ambe_plus2(),
            Err(e) => {
                eprintln!("frame {f}: encode error {e:?}");
                MbeParams::silence_ambe_plus2()
            }
        };
        // Capture each even frame's refined ω to carry to the next odd slot.
        if !is_odd {
            prev_even_omega = Some(f64::from(params.omega_0()));
        }
        let info = quantize(&params, &mut dec).expect("quantize");
        println!(
            "{:04x} {:04x} {:04x} {:04x}",
            info[0], info[1], info[2], info[3]
        );
    }
}
