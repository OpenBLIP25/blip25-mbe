//! Per-frame diagnostic diff between two .bit files (18-byte full-rate
//! frames). Decodes each via our `dequantize_full` and prints
//! (ω_0, L, voicing-popcount, mean-amp-dB) side-by-side.
//!
//! Invoked as: `diff_per_frame <a.bit> <b.bit> [--from N] [--count N]`
//! Labels default to filenames. Useful for spotting where our encoder
//! diverges from a reference encoder at silence→speech boundaries.

use blip25_mbe::imbe_wire::dequantize::{DecoderState, dequantize};
use blip25_mbe::imbe_wire::frame::decode_frame;
use std::env;
use std::fs;

fn unpack_dibits(bytes: &[u8]) -> [u8; 72] {
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

#[derive(Default)]
struct FrameInfo {
    omega: f32,
    l: u8,
    voiced_count: u8,
    amp_rms_db: f64,
    /// True on a dequantize-side error. Set for diagnostic
    /// visibility under a debugger; not read in the current code
    /// path but preserved so the struct's role stays self-documenting.
    #[allow(dead_code)]
    err: bool,
}

fn decode_one(bytes: &[u8], st: &mut DecoderState) -> FrameInfo {
    let dibits = unpack_dibits(bytes);
    let imbe = decode_frame(&dibits);
    match dequantize(&imbe.info, st) {
        Ok(p) => {
            let l = p.harmonic_count();
            let voiced_count = p.voiced_slice()[..l as usize]
                .iter()
                .filter(|v| **v)
                .count() as u8;
            let amps = p.amplitudes_slice();
            let mut sum_sq_db = 0.0f64;
            let mut n = 0usize;
            for i in 0..l as usize {
                let a = amps[i].max(1e-6) as f64;
                let db = 20.0 * a.log10();
                sum_sq_db += db * db;
                n += 1;
            }
            let amp_rms_db = if n > 0 {
                (sum_sq_db / n as f64).sqrt()
            } else {
                0.0
            };
            FrameInfo {
                omega: p.omega_0(),
                l,
                voiced_count,
                amp_rms_db,
                err: false,
            }
        }
        Err(_) => FrameInfo {
            err: true,
            ..Default::default()
        },
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: diff_per_frame <a.bit> <b.bit> [--from N] [--count N]");
        std::process::exit(2);
    }
    let path_a = &args[1];
    let path_b = &args[2];
    let mut from = 0usize;
    let mut count: Option<usize> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--from" => {
                from = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--count" => {
                count = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
                std::process::exit(2);
            }
        }
    }
    let a = fs::read(path_a).unwrap();
    let b = fs::read(path_b).unwrap();
    let n_a = a.len() / 18;
    let n_b = b.len() / 18;
    let n = n_a.min(n_b);
    let end = count.map(|c| (from + c).min(n)).unwrap_or(n);

    let label_a = std::path::Path::new(path_a)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or("a".into());
    let label_b = std::path::Path::new(path_b)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or("b".into());

    let mut st_a = DecoderState::new();
    let mut st_b = DecoderState::new();
    // Advance state to `from` without printing.
    for f in 0..from {
        let _ = decode_one(&a[f * 18..(f + 1) * 18], &mut st_a);
        let _ = decode_one(&b[f * 18..(f + 1) * 18], &mut st_b);
    }

    println!(
        "{:>5}  {:>9} {:>9}  {:>3} {:>3}  {:>3} {:>3}  {:>7} {:>7}   same?",
        "frame",
        format!("{}.ω", label_a),
        format!("{}.ω", label_b),
        format!("{}.L", label_a),
        format!("{}.L", label_b),
        format!("{}.V", label_a),
        format!("{}.V", label_b),
        format!("{}.amp", label_a),
        format!("{}.amp", label_b),
    );
    for f in from..end {
        let fa = decode_one(&a[f * 18..(f + 1) * 18], &mut st_a);
        let fb = decode_one(&b[f * 18..(f + 1) * 18], &mut st_b);
        let bytes_same = a[f * 18..(f + 1) * 18] == b[f * 18..(f + 1) * 18];
        println!(
            "{:>5}  {:>9.5} {:>9.5}  {:>3} {:>3}  {:>3} {:>3}  {:>7.2} {:>7.2}   {}",
            f,
            fa.omega,
            fb.omega,
            fa.l,
            fb.l,
            fa.voiced_count,
            fb.voiced_count,
            fa.amp_rms_db,
            fb.amp_rms_db,
            if bytes_same { "=" } else { "X" },
        );
    }
}
