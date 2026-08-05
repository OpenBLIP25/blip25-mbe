//! Where and how our IMBE encoder's parameters diverge from the Wave7k oracle.
//!
//! `imbe_scoreboard` reports how OFTEN each field agrees. This reports how FAR
//! it is wrong when it does, using the metric each field admits:
//!
//! * `b̂₀` (pitch) indexes a monotone ladder, so an index delta is meaningful.
//!   It is also the field every other field's *position* depends on, so its
//!   errors are reported twice: raw, and split by whether the oracle's cell is
//!   even reachable by our chain.
//! * `b̂₁` is a `K`-bit per-band V/UV word — Hamming over bands.
//! * `b̂₂` (gain) indexes a monotone 6-bit ladder — index delta.
//! * `b̂₃..b̂_{L+1}` are scalar quantizer indices of the amplitude DCT, with
//!   widths that taper down the block. A raw index delta is not comparable
//!   across positions, so each is also normalised by its own field width: 1.0
//!   means "wrong by the whole dynamic range of that coefficient".
//!
//! The PITCH GRID section reports the structural constraint that no amount of
//! estimator work removes: which IMBE cells our chain can reach at all.
//!
//! Usage:
//!   imbe_divergence <vector.pcm> <vector.imbe.bits> [--offset N] [--live] [--unit 11|12]

#[path = "support/imbe_bits.rs"]
mod imbe_bits;

use imbe_bits::*;
use std::collections::BTreeMap;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.len() < 2 {
        eprintln!(
            "usage: imbe_divergence <src.pcm> <ref.imbe.bits> [--offset N] [--live] [--unit 11|12]"
        );
        std::process::exit(2);
    }
    let flag = |name: &str| argv.iter().any(|a| a == name);
    let val = |name: &str| -> Option<i64> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .and_then(|s| s.parse().ok())
    };
    let live = flag("--live");
    let unit = val("--unit").map(|v| v as usize);
    let off = val("--offset").unwrap_or(1);

    let pcm = read_pcm(&argv[0]);
    let (refr, _u) = load_reference(&argv[1], unit);
    let ours = if live {
        ours_live(&pcm)
    } else {
        ours_whole(&pcm)
    };

    let pairs: Vec<(&Dec, &Dec)> = ours
        .iter()
        .enumerate()
        .filter_map(|(i, o)| {
            let j = i as i64 + off;
            (j >= 0 && (j as usize) < refr.len()).then(|| (o, &refr[j as usize]))
        })
        .collect();
    let n = pairs.len();
    println!("{n} frame pairs, alignment offset {off:+}\n");

    // ---- pitch grid --------------------------------------------------------
    let ok = reachable_imbe_b0();
    let reach: Vec<i16> = (0..208i16).filter(|&b| ok[b as usize]).collect();
    println!("PITCH GRID — the structural constraint");
    println!(
        "  the AMBE ladder our chain carries has 120 cells; the IMBE grid has 208.\n  reachable IMBE cells: {} ({:.1}%)",
        reach.len(),
        100.0 * reach.len() as f64 / 208.0
    );
    // Longest run of consecutive unreachable cells, and where.
    let (mut run, mut best_run, mut best_at) = (0i16, 0i16, 0i16);
    for b in 0..208i16 {
        if ok[b as usize] {
            run = 0;
        } else {
            run += 1;
            if run > best_run {
                best_run = run;
                best_at = b - run + 1;
            }
        }
    }
    println!("  longest unreachable run: {best_run} cells starting at b̂₀={best_at}");
    let miss = pairs
        .iter()
        .filter(|(_, r)| !in_range(r.b0) || !ok[r.b0 as usize])
        .count();
    println!(
        "  oracle frames on an unreachable cell: {:.2}%  <- b̂₀ can never exceed {:.2}%",
        pc(miss, n),
        pc(n - miss, n)
    );
    // Of the frames that ARE reachable, how often do we get them?
    let reach_hit = pairs
        .iter()
        .filter(|(o, r)| in_range(r.b0) && ok[r.b0 as usize] && o.b0 == r.b0)
        .count();
    println!(
        "  b̂₀ on reachable-cell frames only: {:.2}%   (headroom the estimator owns)",
        pc(reach_hit, n - miss)
    );

    // ---- b0 ---------------------------------------------------------------
    let mut d0: BTreeMap<i64, usize> = BTreeMap::new();
    for (o, r) in &pairs {
        if o.b0 != r.b0 {
            *d0.entry(i64::from(o.b0) - i64::from(r.b0)).or_default() += 1;
        }
    }
    println!("\nb̂₀ PITCH (monotone ladder, 208 cells)");
    report_scalar(&d0, n);
    // Is the error concentrated where the oracle's own pitch jumps? That was
    // the AMBE finding: mistiming of a transition, not a wrong pitch.
    let (mut jump_err, mut jump_n, mut flat_err, mut flat_n) = (0usize, 0usize, 0usize, 0usize);
    for i in 1..pairs.len() {
        let (o, r) = pairs[i];
        let rp = pairs[i - 1].1;
        let jump = (i32::from(r.b0) - i32::from(rp.b0)).abs() > 10;
        if jump {
            jump_n += 1;
            if o.b0 != r.b0 {
                jump_err += 1;
            }
        } else {
            flat_n += 1;
            if o.b0 != r.b0 {
                flat_err += 1;
            }
        }
    }
    println!(
        "  frames at an oracle pitch JUMP (|Δ|>10): {:.1}% of frames, b̂₀ wrong on {:.1}% of them\n  frames on a flat contour:                b̂₀ wrong on {:.1}% of them",
        pc(jump_n, n),
        pc(jump_err, jump_n),
        pc(flat_err, flat_n)
    );

    // ---- L ----------------------------------------------------------------
    let mut dl: BTreeMap<i64, usize> = BTreeMap::new();
    for (o, r) in &pairs {
        if o.l != r.l {
            *dl.entry(i64::from(o.l) - i64::from(r.l)).or_default() += 1;
        }
    }
    println!("\nL HARMONIC COUNT (derived from b̂₀; sets the whole bit layout)");
    report_scalar(&dl, n);

    // Everything below is only nameable when the two sides agree on L.
    let lm: Vec<(&Dec, &Dec)> = pairs
        .iter()
        .copied()
        .filter(|(o, r)| o.valid && r.valid && o.l == r.l)
        .collect();
    let nl = lm.len();
    println!("\n--- fields below are scored on the {nl} L-matching frames ---");

    // ---- b1 ----------------------------------------------------------------
    let mut ham: Vec<u32> = Vec::new();
    for (o, r) in &lm {
        if o.b[1] != r.b[1] {
            ham.push((o.b[1] ^ r.b[1]).count_ones());
        }
    }
    let wrong1 = ham.len();
    ham.sort_unstable();
    println!("\nb̂₁ V/UV (per-band bitfield, K bands)");
    println!(
        "  wrong {:.2}%   exactly one band differs {:.1}%   median bands differing {}",
        pc(wrong1, nl),
        pc(ham.iter().filter(|&&h| h == 1).count(), wrong1),
        if ham.is_empty() {
            0
        } else {
            ham[ham.len() / 2]
        }
    );
    // Direction: do we over-voice or under-voice? b̂₁ bit set == voiced in this
    // packing, so popcount difference is signed evidence.
    let (mut over, mut under) = (0usize, 0usize);
    for (o, r) in &lm {
        let (a, b) = (o.b[1].count_ones(), r.b[1].count_ones());
        if a > b {
            over += 1;
        } else if a < b {
            under += 1;
        }
    }
    println!(
        "  more bands set than the oracle: {:.1}%   fewer: {:.1}%",
        pc(over, nl),
        pc(under, nl)
    );

    // ---- b2 ----------------------------------------------------------------
    let mut d2: BTreeMap<i64, usize> = BTreeMap::new();
    for (o, r) in &lm {
        if o.b[2] != r.b[2] {
            *d2.entry(i64::from(o.b[2]) - i64::from(r.b[2])).or_default() += 1;
        }
    }
    println!("\nb̂₂ GAIN (monotone 6-bit ladder)");
    report_scalar(&d2, nl);

    // ---- amplitudes --------------------------------------------------------
    println!("\nb̂₃..b̂_{{L+1}} AMPLITUDE COEFFICIENTS");
    println!(
        "  {:>6} {:>7} {:>8} {:>9} {:>10} {:>8}",
        "decile", "wrong%", "|d|=1%", "median|d|", "med|d|/2^w", "meanwid"
    );
    let mut dec: Vec<(usize, usize, usize, Vec<i64>, f64, usize)> =
        (0..10).map(|_| (0, 0, 0, Vec::new(), 0.0, 0)).collect();
    for (o, r) in &lm {
        let rng = o.amp_range();
        let len = rng.len().max(1);
        for (k, p) in rng.clone().enumerate() {
            let d = (10 * k / len).min(9);
            let w = field_width(o.l, p).max(1);
            dec[d].0 += 1;
            dec[d].4 += f64::from(w);
            dec[d].5 += 1;
            if o.b[p] != r.b[p] {
                dec[d].1 += 1;
                let delta = i64::from(o.b[p]) - i64::from(r.b[p]);
                if delta.abs() == 1 {
                    dec[d].2 += 1;
                }
                dec[d].3.push(delta.abs());
                dec[d].4 += 0.0;
            }
        }
    }
    for (i, (tot, wrong, one, mags, wsum, wn)) in dec.iter_mut().enumerate() {
        if *tot == 0 {
            continue;
        }
        let med = median(mags);
        let meanw = *wsum / (*wn).max(1) as f64;
        println!(
            "  {:>6} {:>7.2} {:>8.1} {:>9} {:>10.3} {:>8.2}",
            i,
            pc(*wrong, *tot),
            pc(*one, (*wrong).max(1)),
            med,
            med as f64 / 2f64.powf(meanw),
            meanw
        );
    }

    // Conditioning: is the amplitude block a victim of the fields above it, or
    // independently wrong? Teacher-forcing is not available without a live
    // capture, but conditioning on agreement is.
    let cond = |pred: &dyn Fn(&Dec, &Dec) -> bool| -> (usize, usize) {
        let (mut num, mut den) = (0usize, 0usize);
        for (o, r) in &lm {
            if !pred(o, r) {
                continue;
            }
            for p in o.amp_range() {
                den += 1;
                if o.b[p] == r.b[p] {
                    num += 1;
                }
            }
        }
        (num, den)
    };
    println!("\n  amplitude agreement conditioned on the fields above it");
    let (a, b) = cond(&|_, _| true);
    println!(
        "    unconditional            {:>6.2}%  ({b} coefficients)",
        pc(a, b)
    );
    let (a, b) = cond(&|o, r| o.b[1] == r.b[1]);
    println!("    given b̂₁ agrees          {:>6.2}%  ({b})", pc(a, b));
    let (a, b) = cond(&|o, r| o.b[2] == r.b[2]);
    println!("    given b̂₂ agrees          {:>6.2}%  ({b})", pc(a, b));
    let (a, b) = cond(&|o, r| o.b[1] == r.b[1] && o.b[2] == r.b[2]);
    println!("    given b̂₁ and b̂₂ agree    {:>6.2}%  ({b})", pc(a, b));

    println!(
        "\nReading: a decile whose median|d|/2^w is small chose a neighbouring level\n(precision); one near 1 chose an unrelated level (the input to the quantizer\nis wrong, not the quantizer). b̂₀'s reachable-cell figure separates the two\nkinds of pitch loss: grid bottleneck versus estimator."
    );
}

fn in_range(b0: i16) -> bool {
    (0..208).contains(&b0)
}

fn report_scalar(d: &BTreeMap<i64, usize>, n: usize) {
    let wrong: usize = d.values().sum();
    if wrong == 0 {
        println!("  0.00% wrong");
        return;
    }
    let one: usize = d
        .iter()
        .filter(|(k, _)| k.abs() == 1)
        .map(|(_, v)| *v)
        .sum();
    let mut mags: Vec<i64> = d
        .iter()
        .flat_map(|(k, v)| std::iter::repeat_n(k.abs(), *v))
        .collect();
    let med = median(&mut mags);
    let hi: usize = d.iter().filter(|(k, _)| **k > 0).map(|(_, v)| *v).sum();
    let mut top: Vec<(i64, usize)> = d.iter().map(|(k, v)| (*k, *v)).collect();
    top.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    println!(
        "  wrong {:.2}%   |Δ|=1 {:.1}% of errors   median|Δ| {med}   ours HIGH on {:.1}% of errors",
        pc(wrong, n),
        pc(one, wrong),
        pc(hi, wrong)
    );
    println!(
        "  top deltas: {}",
        top.iter()
            .take(5)
            .map(|(k, c)| format!("{k:+}:{c}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
}
