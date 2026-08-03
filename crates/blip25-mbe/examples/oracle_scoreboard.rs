//! Per-field agreement between our encoder and the Wave7k oracle.
//!
//! Reads oracle reference bits produced by the dependency-window harness
//! (7-byte natural-order AMBE+2 frames, one per input frame) and encodes the
//! same PCM through `LiveEncoder`, then reports per-parameter agreement.
//!
//! Usage: oracle_scoreboard <vector.pcm> <vector.ambe2.bits>

use blip25_mbe::rate33::{fields_from_natural, fields_from_no_fec};
use blip25_mbe::vocoder::{LiveEncoder, Rate};
use std::fs;

const FR: usize = 160;
const NB: usize = 9;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let pcm: Vec<i16> = fs::read(&a[1])
        .unwrap()
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let oracle: Vec<[u16; NB]> = fs::read(&a[2])
        .unwrap()
        .chunks_exact(7)
        .map(fields_from_natural)
        .collect();

    // Ours: the streaming path at the no-FEC half rate (same 49 info bits).
    let mut le = LiveEncoder::new(Rate::AmbePlus2_2450x2450);
    let mut ours: Vec<[u16; NB]> = Vec::new();
    for ch in pcm.chunks(FR) {
        for r in le.push(ch) {
            ours.push(fields_from_no_fec(&r.unwrap()));
        }
    }
    for f in le.flush().unwrap() {
        ours.push(fields_from_no_fec(&f));
    }

    println!("frames: ours {} oracle {}", ours.len(), oracle.len());
    // Alignment is not assumed: score every offset and report the best.
    let score = |off: i64| -> ([usize; NB], usize, usize) {
        let mut hit = [0usize; NB];
        let (mut all9, mut n) = (0usize, 0usize);
        for i in 0..ours.len() {
            let j = i as i64 + off;
            if j < 0 || j as usize >= oracle.len() {
                continue;
            }
            let (o, r) = (&ours[i], &oracle[j as usize]);
            let mut all = true;
            for b in 0..NB {
                if o[b] == r[b] {
                    hit[b] += 1;
                } else {
                    all = false;
                }
            }
            if all {
                all9 += 1;
            }
            n += 1;
        }
        (hit, all9, n)
    };
    let mut best = (i64::MIN, 0usize);
    for off in -4..=4i64 {
        let (h, _, n) = score(off);
        let s: usize = h.iter().sum();
        println!("  offset {off:+}: b0 {:.1}%  sum {:.1}%", 100.0 * h[0] as f64 / n.max(1) as f64,
                 100.0 * s as f64 / (NB * n.max(1)) as f64);
        if s as i64 > best.0 {
            best = (s as i64, (off + 4) as usize);
        }
    }
    let off = best.1 as i64 - 4;
    let (hit, all9, n) = score(off);
    let pct = |x: usize| 100.0 * x as f64 / n as f64;
    println!("\nBEST OFFSET {off:+} over {n} frames");
    for b in 0..NB {
        println!("  b{:<5} {:>7.2}%", b, pct(hit[b]));
    }
    println!("  {:<6} {:>7.2}%   <- whole-frame", "ALL9", pct(all9));
}
