//! Where and how our encoder's parameters diverge from the Wave7k oracle.
//!
//! The scoreboard reports how OFTEN each field agrees. This reports how FAR it
//! is wrong when it does, using the metric each field actually admits:
//!
//! * `b0` (pitch) and `b2` (gain) index monotonic scalar ladders, so an index
//!   delta is meaningful: +-1 is a neighbouring level (rounding or tie-break),
//!   a large delta is a wrong input to the search.
//! * `b1` is a voicing bitfield. Index arithmetic says nothing; the Hamming
//!   distance counts how many bands disagree.
//! * `b3..b8` index VECTOR quantisers (PRBA24 is 512x3). Neighbouring indices
//!   are not neighbouring vectors, so index deltas are noise. What is
//!   meaningful is the distance between the codebook ROWS the two indices
//!   select, normalised by the codebook's own RMS row spacing: ~1 means we
//!   picked a typical wrong neighbour, >>1 means the search input is wrong.
//!
//! Usage: oracle_divergence <vector.pcm> <vector.ambe2.bits> [alignment-offset]

use blip25_mbe::rate33::{fields_from_natural, fields_from_no_fec};
use blip25_mbe::vocoder::{LiveEncoder, Rate};
use std::collections::BTreeMap;
use std::fs;

const NB: usize = 9;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let off: i64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let pcm: Vec<i16> = fs::read(&a[1]).unwrap().chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let oracle: Vec<[u16; NB]> = fs::read(&a[2]).unwrap().chunks_exact(7)
        .map(fields_from_natural).collect();

    let mut le = LiveEncoder::new(Rate::AmbePlus2_2450x2450);
    let mut ours: Vec<[u16; NB]> = Vec::new();
    for ch in pcm.chunks(160) {
        for r in le.push(ch) { ours.push(fields_from_no_fec(&r.unwrap())); }
    }
    for f in le.flush().unwrap() { ours.push(fields_from_no_fec(&f)); }

    // Codebook row lookups for the VQ fields, and a scale for "typical" spacing.
    let cb = |b: usize| -> Option<blip25_codebooks::Codebook> {
        match b {
            3 => Some(blip25_codebooks::prba24()),
            4 => Some(blip25_codebooks::prba58()),
            5 => Some(blip25_codebooks::hoc_b5()),
            6 => Some(blip25_codebooks::hoc_b6()),
            7 => Some(blip25_codebooks::hoc_b7()),
            8 => Some(blip25_codebooks::hoc_b8()),
            _ => None,
        }
    };
    let rowdist = |c: &blip25_codebooks::Codebook, i: usize, j: usize| -> f64 {
        if i >= c.rows || j >= c.rows { return f64::NAN; }
        (0..c.cols)
            .map(|k| {
                let a = c.raw[i * c.cols + k] as f64;
                let b = c.raw[j * c.cols + k] as f64;
                (a - b) * (a - b)
            })
            .sum::<f64>()
            .sqrt()
    };
    // Typical spacing: RMS distance between consecutive rows.
    let mut typical = [1.0f64; NB];
    for b in 3..NB {
        if let Some(c) = cb(b) {
            let mut acc = 0.0;
            for i in 1..c.rows { let d = rowdist(&c, i - 1, i); acc += d * d; }
            typical[b] = (acc / (c.rows - 1) as f64).sqrt().max(1e-9);
        }
    }
    let mut vqdist: [Vec<f64>; NB] = Default::default();
    let mut hamming: Vec<u32> = Vec::new();

    // Per field: how big is the error when it is wrong?
    let mut delta: [BTreeMap<i64, usize>; NB] = Default::default();
    let mut n = 0usize;
    for i in 0..ours.len() {
        let j = i as i64 + off;
        if j < 0 || j as usize >= oracle.len() { continue; }
        n += 1;
        for b in 0..NB {
            let (o, r) = (ours[i][b], oracle[j as usize][b]);
            let d = o as i64 - r as i64;
            if d != 0 {
                *delta[b].entry(d).or_default() += 1;
                if b == 1 { hamming.push((o ^ r).count_ones()); }
                if let Some(c) = cb(b) {
                    let dist = rowdist(&c, o as usize, r as usize);
                    if dist.is_finite() { vqdist[b].push(dist / typical[b]); }
                }
            }
        }
    }
    println!("{n} frames, alignment offset {off:+}\n");
    println!("SCALAR ladders (index delta is meaningful)");
    println!("  {:<4} {:>7} {:>8} {:>9}   {}", "fld", "wrong%", "|d|=1%", "median|d|", "top deltas");
    for b in [0usize, 2] {
        let wrong: usize = delta[b].values().sum();
        if wrong == 0 { println!("  b{b}      0.00"); continue; }
        let one: usize = delta[b].iter().filter(|(d, _)| d.abs() == 1).map(|(_, c)| *c).sum();
        let mut mags: Vec<i64> = delta[b].iter().flat_map(|(d, c)| std::iter::repeat(d.abs()).take(*c)).collect();
        mags.sort_unstable();
        let mut top: Vec<(i64, usize)> = delta[b].iter().map(|(d, c)| (*d, *c)).collect();
        top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        println!("  b{:<3} {:>7.2} {:>8.1} {:>9}   {}", b,
            100.0 * wrong as f64 / n as f64, 100.0 * one as f64 / wrong as f64,
            mags[mags.len() / 2],
            top.iter().take(4).map(|(d, c)| format!("{d:+}:{c}")).collect::<Vec<_>>().join("  "));
    }
    println!("\nVOICING bitfield (Hamming over bands)");
    {
        let wrong: usize = delta[1].values().sum();
        let mut h = hamming.clone();
        h.sort_unstable();
        let one = h.iter().filter(|&&x| x == 1).count();
        println!("  b1   {:>7.2}% wrong   1 band differs: {:.1}%   median bands differing: {}",
            100.0 * wrong as f64 / n as f64,
            100.0 * one as f64 / wrong.max(1) as f64,
            if h.is_empty() { 0 } else { h[h.len() / 2] });
    }
    println!("\nVECTOR quantisers (row distance / typical row spacing)");
    println!("  {:<4} {:>7} {:>10} {:>10} {:>10}", "fld", "wrong%", "median", "p10", "p90");
    for b in 3..NB {
        let wrong: usize = delta[b].values().sum();
        let mut v = vqdist[b].clone();
        v.sort_by(|a, c| a.partial_cmp(c).unwrap());
        if v.is_empty() { println!("  b{b}      {:>6.2}   (no rows)", 100.0 * wrong as f64 / n as f64); continue; }
        let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!("  b{:<3} {:>7.2} {:>10.2} {:>10.2} {:>10.2}", b,
            100.0 * wrong as f64 / n as f64, q(0.5), q(0.1), q(0.9));
    }
    println!("\nVQ distance ~1 = we chose a typical adjacent codebook entry (search is\nclose, quantiser tie-break or fine input error). >>1 = the vector fed to\nthe search is wrong, not the search.");
}
