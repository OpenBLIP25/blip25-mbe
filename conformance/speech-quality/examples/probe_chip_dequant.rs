//! Probe: dequantize chip.bit through our full-rate pipeline and dump
//! per-frame (omega_0, L, voicing density, amplitude RMS, err0).
//!
//! Diagnoses whether the chip_enc_our_dec attenuation bug lives in
//! FEC/dequant (params would be garbage) vs synth (params look OK,
//! output still wrong).

use blip25_mbe::p25_fullrate::dequantize::{DecodeError, DecoderState, dequantize};
use blip25_mbe::p25_fullrate::frame::decode_frame;
use std::fs;

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
    let limit: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("limit"))
        .unwrap_or(30);
    let bytes = fs::read(&path).unwrap();
    let n = bytes.len() / 18;

    println!("frame  err0 err4 etot  b0     omega_0   L  voiced  amp_rms  amp_peak  note");
    let mut st = DecoderState::new();
    for f in 0..n.min(limit) {
        let chunk = &bytes[f * 18..(f + 1) * 18];
        let dibits = unpack_dibits_msb(chunk);
        let imbe = decode_frame(&dibits);
        let b0 = imbe.info[0];
        let etot = imbe.error_total();
        match dequantize(&imbe.info, &mut st) {
            Ok(p) => {
                let l = p.harmonic_count();
                let voiced = p.voiced_slice()[..l as usize]
                    .iter()
                    .filter(|&&v| v)
                    .count();
                let amps = &p.amplitudes_slice()[..l as usize];
                let rms = (amps.iter().map(|a| a * a).sum::<f32>() / l as f32).sqrt();
                let peak = amps.iter().fold(0f32, |a, &b| a.max(b.abs()));
                println!(
                    "{:4}  {:3} {:3} {:4}  {:04x}   {:7.5}  {:2}   {:2}/{:<2}  {:7.4}  {:8.4}  ok",
                    f,
                    imbe.errors[0],
                    imbe.errors[4],
                    etot,
                    b0,
                    p.omega_0(),
                    l,
                    voiced,
                    l,
                    rms,
                    peak
                );
            }
            Err(e) => {
                let note = match e {
                    DecodeError::BadPitch => "BAD_PITCH",
                    _ => "OTHER_ERR",
                };
                println!(
                    "{:4}  {:3} {:3} {:4}  {:04x}   --       --   -- --  --       --        {}",
                    f, imbe.errors[0], imbe.errors[4], etot, b0, note
                );
            }
        }
    }
}
