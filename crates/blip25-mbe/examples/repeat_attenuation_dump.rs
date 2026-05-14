//! Dump per-frame disposition, errors, peak, rms for an .ambe9 stream.
//! For each Repeat-triggering frame, also show the surrounding frames'
//! amplitudes so we can fit the chip's post-Repeat attenuation curve.

use blip25_mbe::vocoder::{Rate, Vocoder};
use blip25_mbe::codecs::mbe_baseline::FrameDisposition;

fn main() {
    let path = std::env::args().nth(1).expect("usage: repeat_attenuation_dump <in.ambe9>");
    let bytes = std::fs::read(&path).unwrap();
    let mut v = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let nf = bytes.len() / 9;
    let mut per_frame: Vec<(usize, u8, u8, FrameDisposition, i32, f64)> = Vec::with_capacity(nf);
    for i in 0..nf {
        let frame = &bytes[i * 9..i * 9 + 9];
        let pcm = v.decode_bits(frame).unwrap();
        let peak = pcm.iter().map(|s| s.unsigned_abs() as i32).max().unwrap_or(0);
        let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
        let disp = v.last_disposition().unwrap_or(FrameDisposition::Use);
        let stats = v.last_stats();
        let (s0, st) = stats.decode.as_ref().map(|d| (d.epsilon_0, d.epsilon_t)).unwrap_or((0, 0));
        per_frame.push((i, s0, st, disp, peak, rms));
    }
    // Print: each Repeat-or-Mute frame + 3 frames before and after
    let mut shown: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut events: Vec<usize> = Vec::new();
    for (i, _, _, disp, _, _) in &per_frame {
        if !matches!(disp, FrameDisposition::Use) {
            events.push(*i);
        }
    }
    if events.is_empty() {
        println!("No Repeat/Mute events in {} frames.", nf);
        return;
    }
    println!("Found {} Repeat/Mute events in {} frames.", events.len(), nf);
    println!("{:>5} {:>5} {:>5} {:>10} {:>7} {:>8}", "frame", "ε₀", "ε_T", "disp", "peak", "rms");
    for &ev in &events {
        let lo = ev.saturating_sub(3);
        let hi = (ev + 4).min(nf);
        for j in lo..hi {
            if shown.insert(j) {
                let (idx, s0, st, disp, peak, rms) = per_frame[j];
                let marker = if !matches!(disp, FrameDisposition::Use) { " <--" } else { "" };
                println!("{:>5} {:>5} {:>5} {:>10?} {:>7} {:>8.1}{}",
                         idx, s0, st, disp, peak, rms, marker);
            }
        }
        println!();
    }
}
