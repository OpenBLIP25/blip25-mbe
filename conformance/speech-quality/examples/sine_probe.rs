//! Per-frame MbeParams dump for a raw-i16 PCM file. Runs our analysis
//! encoder (no wire round-trip) and prints `ω̂_0`, L̂, voiced-count,
//! and per-harmonic M̂_l for each non-preroll frame. Used to
//! characterize amplitude jitter on synthetic stationary signals
//! (e.g. pure sines) where the reference encoder would hold params
//! rock-steady.

use blip25_mbe::codecs::mbe_baseline::analysis::{
    AnalysisOutput, AnalysisState, encode as analysis_encode,
};
use std::env;
use std::fs;

const FRAME_SAMPLES: usize = 160;

fn read_pcm(path: &str) -> Vec<i16> {
    let bytes = fs::read(path).unwrap();
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: sine_probe <in.pcm> [--dump-harmonics N]");
        std::process::exit(2);
    }
    let path = &args[1];
    let dump_h: Option<usize> = args
        .iter()
        .position(|a| a == "--dump-harmonics")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let pcm = read_pcm(path);
    let n_frames = pcm.len() / FRAME_SAMPLES;
    eprintln!("input: {} ({} frames × {} samples)", path, n_frames, FRAME_SAMPLES);

    let mut state = AnalysisState::new();

    // Header.
    let mut hdr = format!("{:>5}  {:>9}  {:>3} {:>3}  {:>8}  {:>8}  {:>8}  {:>8}",
        "frame", "ω̂_0", "L̂", "V", "M̂_1(dB)", "M̂_2(dB)", "M̂_3(dB)", "Σ M̂²");
    if let Some(h) = dump_h {
        for i in 1..=h {
            hdr.push_str(&format!("  {:>7}", format!("M̂_{}", i)));
        }
    }
    println!("{hdr}");

    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
        let out = analysis_encode(frame, &mut state).unwrap();
        let params = match out {
            AnalysisOutput::Voice(p) => p,
            AnalysisOutput::Silence => {
                println!("{:>5}  {:>9}  {:>3} {:>3}  <preroll or silence>", f, "-", "-", "-");
                continue;
            }
        };
        let l = params.harmonic_count();
        let voiced_count = params.voiced_slice()[..l as usize]
            .iter()
            .filter(|v| **v)
            .count();
        let amps = params.amplitudes_slice();
        let total_energy: f64 = amps[..l as usize]
            .iter()
            .map(|a| f64::from(*a) * f64::from(*a))
            .sum();
        let get_db = |l_idx: usize| -> f64 {
            if l_idx <= l as usize {
                20.0 * (f64::from(amps[l_idx - 1]).max(1e-6)).log10()
            } else {
                f64::NAN
            }
        };
        let mut row = format!("{:>5}  {:>9.5}  {:>3} {:>3}  {:>8.2}  {:>8.2}  {:>8.2}  {:>8.3e}",
            f,
            params.omega_0(),
            l, voiced_count,
            get_db(1), get_db(2), get_db(3),
            total_energy,
        );
        if let Some(h) = dump_h {
            for i in 1..=h {
                if i <= l as usize {
                    row.push_str(&format!("  {:>7.3}", amps[i - 1]));
                } else {
                    row.push_str(&format!("  {:>7}", "-"));
                }
            }
        }
        println!("{row}");
    }
}
