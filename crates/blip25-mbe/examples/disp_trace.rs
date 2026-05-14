//! Trace per-frame disposition + epsilon_r for half-rate decode.

use blip25_mbe::vocoder::{Rate, Vocoder};

fn main() {
    let path = std::env::args().nth(1).expect("usage: disp_trace <in.ambe9>");
    let bytes = std::fs::read(&path).unwrap();
    let mut v = Vocoder::new(Rate::AmbePlus2_3600x2450);
    let nf = bytes.len() / 9;
    let win: Vec<usize> = std::env::args()
        .nth(2)
        .map(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            (parts[0].parse::<usize>().unwrap()..=parts[1].parse::<usize>().unwrap()).collect()
        })
        .unwrap_or_else(|| (760..=785).collect());
    let mut peak_log = vec![0u32; nf];
    let mut disp_log: Vec<(u8, u8, &'static str)> = Vec::new();
    for i in 0..nf {
        let frame = &bytes[i * 9..i * 9 + 9];
        let pcm = v.decode_bits(frame).unwrap();
        let peak = pcm.iter().map(|s| s.unsigned_abs() as u32).max().unwrap_or(0);
        peak_log[i] = peak;
        let disp = v.last_disposition();
        let stats = v.last_stats();
        let (s0, st) = stats
            .decode
            .as_ref()
            .map(|d| (d.epsilon_0, d.epsilon_t))
            .unwrap_or((0, 0));
        let label: &'static str = match disp {
            Some(blip25_mbe::codecs::mbe_baseline::FrameDisposition::Use) => "Use",
            Some(blip25_mbe::codecs::mbe_baseline::FrameDisposition::Repeat) => "Repeat",
            Some(blip25_mbe::codecs::mbe_baseline::FrameDisposition::Mute) => "Mute",
            None => "None",
        };
        disp_log.push((s0, st, label));
        if win.contains(&i) {
            let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt();
            println!(
                "f={:4} disp={:7}  ε₀={} ε_T={}  peak={:5} rms={:6.1}",
                i, label, s0, st, peak, rms
            );
        }
    }
}
