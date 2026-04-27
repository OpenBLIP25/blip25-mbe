//! Per-frame disposition histogram on chip.bit. Calls our spec-derived
//! `frame_disposition` directly with the per-frame ε_t / ε_0 from
//! `decode_frame`, mirroring the smoothed ε_R update the synth applies.

use blip25_mbe::codecs::mbe_baseline::{FrameDisposition, FrameErrorContext, frame_disposition};
use blip25_mbe::imbe_wire::dequantize::{DecodeError, DecoderState, dequantize};
use blip25_mbe::imbe_wire::frame::decode_frame;
use std::fs;

const FRAME_BYTES: usize = 18;

fn unpack_dibits_msb(bytes: &[u8]) -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut bit = 0usize;
    for slot in &mut out {
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

fn main() {
    let path = std::env::args().nth(1).expect("bit file arg");
    let bytes = fs::read(&path).unwrap();
    let n = bytes.len() / FRAME_BYTES;
    let mut st = DecoderState::new();
    let mut eps_r: f64 = 0.0;
    let mut counts = [0u32; 3]; // Use, Repeat, Mute
    let mut bad_pitch_count = 0u32;
    let mut first_mute_frame: Option<usize> = None;
    for f in 0..n {
        let chunk = &bytes[f * FRAME_BYTES..(f + 1) * FRAME_BYTES];
        let dibits = unpack_dibits_msb(chunk);
        let imbe = decode_frame(&dibits);
        let bad_pitch = matches!(dequantize(&imbe.info, &mut st), Err(DecodeError::BadPitch));
        if bad_pitch {
            bad_pitch_count += 1;
        }
        let err = FrameErrorContext {
            epsilon_0: imbe.errors[0],
            epsilon_4: imbe.errors[4],
            epsilon_t: imbe.error_total().min(255) as u8,
            bad_pitch,
        };
        let (disp, new_r) = frame_disposition(&err, eps_r);
        eps_r = new_r;
        let idx = match disp {
            FrameDisposition::Use => 0,
            FrameDisposition::Repeat => 1,
            FrameDisposition::Mute => 2,
        };
        counts[idx] += 1;
        if disp == FrameDisposition::Mute && first_mute_frame.is_none() {
            first_mute_frame = Some(f);
        }
        if f < 5 || (first_mute_frame == Some(f)) {
            println!(
                "f={:4}  e0={} e4={} et={:3}  eps_r={:.5}  -> {:?}",
                f, err.epsilon_0, err.epsilon_4, err.epsilon_t, eps_r, disp
            );
        }
    }
    println!(
        "\ntotal frames: {n}  Use={}  Repeat={}  Mute={}  bad_pitch_dequant={}  first_mute_frame={:?}  final_eps_r={:.5}",
        counts[0], counts[1], counts[2], bad_pitch_count, first_mute_frame, eps_r
    );
}
