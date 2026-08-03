//! Where and how our encoder's parameters diverge from the Wave7k oracle.
//!
//! The scoreboard reports how OFTEN each field agrees. This reports how FAR it
//! is wrong when it does: an index off by one is a rounding or tie-break
//! difference inside a quantiser search, while a large or unbounded error means
//! the value fed to that search is wrong. The two need different fixes.
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

    // Per field: how big is the error when it is wrong?
    let mut delta: [BTreeMap<i64, usize>; NB] = Default::default();
    let mut n = 0usize;
    for i in 0..ours.len() {
        let j = i as i64 + off;
        if j < 0 || j as usize >= oracle.len() { continue; }
        n += 1;
        for b in 0..NB {
            let d = ours[i][b] as i64 - oracle[j as usize][b] as i64;
            if d != 0 { *delta[b].entry(d).or_default() += 1; }
        }
    }
    println!("{n} frames, alignment offset {off:+}\n");
    println!("{:<4} {:>7} {:>8} {:>8} {:>8}   {}", "fld", "wrong%", "|d|=1%", "|d|<=2%", "median|d|", "top deltas (delta:count)");
    for b in 0..NB {
        let wrong: usize = delta[b].values().sum();
        if wrong == 0 { println!("{:<4} {:>7}", format!("b{b}"), "0.00"); continue; }
        let one: usize = delta[b].iter().filter(|(d, _)| d.abs() == 1).map(|(_, c)| *c).sum();
        let two: usize = delta[b].iter().filter(|(d, _)| d.abs() <= 2).map(|(_, c)| *c).sum();
        let mut mags: Vec<i64> = delta[b].iter().flat_map(|(d, c)| std::iter::repeat(d.abs()).take(*c)).collect();
        mags.sort_unstable();
        let med = mags[mags.len() / 2];
        let mut top: Vec<(i64, usize)> = delta[b].iter().map(|(d, c)| (*d, *c)).collect();
        top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        let tops: Vec<String> = top.iter().take(4).map(|(d, c)| format!("{d:+}:{c}")).collect();
        println!("{:<4} {:>7.2} {:>8.1} {:>8.1} {:>8}   {}",
            format!("b{b}"), 100.0 * wrong as f64 / n as f64,
            100.0 * one as f64 / wrong as f64, 100.0 * two as f64 / wrong as f64,
            med, tops.join("  "));
    }
    println!("\n|d|=1% and |d|<=2% are shares OF THE WRONG FRAMES: high means a\nneighbouring quantiser cell (rounding/tie-break); low means a wrong input.");
}
