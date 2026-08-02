//! Establish the frame alignment of every b1-relevant stream against the reference's
//! own r33 bits, using `b0` (pitch) as the ruler.
//!
//! Pitch is the field we reproduce best (~97–98% against the console), so it
//! gives an unambiguous read of how our emitted frame stream indexes relative
//! to the reference's. Voicing cannot serve that role: our clean-room b1 and the RE'd
//! tracker disagree ~48% of the time even when correctly aligned, which is why
//! the in-tree ±3 lag search has no signal and rails at its boundary.
//!
//! Reports, pooled over the corpus and by lag:
//!   ours_b0    vs reference_b0   -> where our EMITTED stream sits
//!   tracker_b1 vs reference_b1   -> where the REFERENCE VOICING stream sits
//! The difference between the two peaks is the index offset the injection needs.

use std::fs;

const LAGS: std::ops::RangeInclusive<i64> = -4..=4;

fn agree_by_lag(ours: &[u16], theirs: &[u16], acc: &mut [(usize, usize)]) {
    for (li, lag) in LAGS.enumerate() {
        let (mut a, mut n) = (0usize, 0usize);
        for (i, &t) in theirs.iter().enumerate() {
            let j = i as i64 + lag;
            if j >= 0 && (j as usize) < ours.len() {
                n += 1;
                if ours[j as usize] == t {
                    a += 1;
                }
            }
        }
        acc[li].0 += a;
        acc[li].1 += n;
    }
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reference-material/Vectors/tv-rc".to_string());
    let mut names: Vec<String> = fs::read_dir(&root)
        .expect("vector root")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "pcm").then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();

    let n = (LAGS.end() - LAGS.start() + 1) as usize;
    let (mut b0_acc, mut trk_acc, mut ours_b1_acc) =
        (vec![(0, 0); n], vec![(0, 0); n], vec![(0, 0); n]);
    let mut used = 0;

    for name in &names {
        let (Ok(raw), Ok(bits)) = (
            fs::read(format!("{root}/{name}.pcm")),
            fs::read(format!("{root}/r33/{name}.bit")),
        ) else {
            continue;
        };
        let pcm: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        if pcm.len() < 160 * 48 || bits.len() < 9 * 48 {
            continue;
        }
        let dv: Vec<[u16; 9]> = bits
            .chunks_exact(9)
            .map(|f| blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(f).info))
            .collect();
        let reference_b0: Vec<u16> = dv.iter().map(|b| b[0]).collect();
        let reference_b1: Vec<u16> = dv.iter().map(|b| b[1]).collect();

        // Our emitted stream, exactly as `Vocoder::encode` builds it (reference b0
        // injected, look-ahead placeholder dropped, tail flushed).
        let mut e = blip25_codec::enc::Encoder::new();
        let reference = blip25_codec::enc::pcm_encode::encode_pcm_b0(
            &pcm,
            blip25_codec::enc::pcm_encode::EncodeOpts::default(),
        );
        if !reference.is_empty() {
            let nf = pcm.len() / 160;
            e.set_forced_b0((0..nf).map(|f| reference[(f + 2).min(reference.len() - 1)]).collect());
        }
        let mut frames = Vec::new();
        for f in pcm.chunks_exact(160) {
            if let Some(b) = e.encode_frame_r33(f.try_into().unwrap()) {
                frames.push(b);
            }
        }
        if !frames.is_empty() {
            frames.remove(0);
        }
        frames.extend(e.flush_r33());
        if frames.is_empty() {
            continue;
        }
        let ours: Vec<[u16; 9]> = frames
            .iter()
            .map(|f| blip25_codec::tables::deprioritize(&blip25_codec::frame::decode_bytes(f).info))
            .collect();
        let ours_b0: Vec<u16> = ours.iter().map(|b| b[0]).collect();
        let ours_b1: Vec<u16> = ours.iter().map(|b| b[1]).collect();

        let nframes = (pcm.len() / 160).saturating_sub(1);
        let trk: Vec<u16> = blip25_codec::enc::b1_audio::b1_track_from_logs(
            e.gap2_mid_log(),
            e.gap2_slot1_log(),
            e.gap2_slot2_log(),
            e.prefiltered_log(),
            &pcm,
            nframes,
            blip25_codec::enc::b1_audio::RingRefineMode::Off,
        )
        .iter()
        .map(|f| f.b1)
        .collect();

        agree_by_lag(&ours_b0, &reference_b0, &mut b0_acc);
        agree_by_lag(&trk, &reference_b1, &mut trk_acc);
        agree_by_lag(&ours_b1, &reference_b1, &mut ours_b1_acc);
        used += 1;
    }

    let show = |label: &str, acc: &[(usize, usize)]| {
        println!("\n=== {label} (pooled, {used} vectors) ===");
        let mut best = (0i64, 0.0f64);
        for (li, lag) in LAGS.enumerate() {
            let p = 100.0 * acc[li].0 as f64 / acc[li].1.max(1) as f64;
            if p > best.1 {
                best = (lag, p);
            }
            println!(
                "  lag {lag:>2}: {p:>5.1}%  {}",
                "#".repeat((p / 2.0) as usize)
            );
        }
        println!("  -> peak at lag {} ({:.1}%)", best.0, best.1);
    };
    show("ours emitted b0 vs the reference b0", &b0_acc);
    show("tracker b1 vs the reference b1", &trk_acc);
    show("ours emitted b1 (clean-room) vs the reference b1", &ours_b1_acc);
}
