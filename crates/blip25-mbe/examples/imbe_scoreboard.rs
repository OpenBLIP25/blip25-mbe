//! Per-field agreement between our IMBE encoder and the Wave7k oracle.
//!
//! The IMBE counterpart of `oracle_scoreboard`. Two things make it a different
//! instrument rather than a retargeting of that one:
//!
//! * **The field list is not fixed.** AMBE+2 always has nine parameters; IMBE
//!   has `L+3`, and `L` is itself derived from the pitch index, so the number
//!   of fields — and the bit width of each — changes frame to frame and differs
//!   between the two sides whenever the pitch differs. Fields are therefore
//!   reported as `b̂₀`, `b̂₁` (V/UV, `K` bands), `b̂₂` (gain), the amplitude block
//!   `b̂₃..b̂_{L+1}` (aggregate, per-index, and per-position-decile) and the sync
//!   bit `b̂_{L+2}`, with the amplitude figures conditioned on the two sides
//!   agreeing on `L`.
//!
//! * **Transmitted fields hide the pitch chain.** `Vocoder::encode` feeds IMBE
//!   an AMBE 7-bit pitch word, requantized onto the 8-bit IMBE grid. Two
//!   quantizations sit between the estimator and `b̂₀`, so a transmitted-field
//!   audit cannot see sub-step error — the failure mode that once read 99.67% on
//!   AMBE's `b0` while the internal pitch word agreed 35.6% of the time. The
//!   INTERNAL STATE section reads the pitch word directly and reports cell
//!   reachability and sub-cell placement.
//!
//! Alignment is swept, never assumed; the sweep curve is printed so its
//! sharpness is visible rather than asserted.
//!
//! Usage:
//!   imbe_scoreboard <vector.pcm> <vector.imbe.bits> [--live] [--offset N]
//!                   [--unit 11|12] [--sweep N]

#[path = "support/imbe_bits.rs"]
mod imbe_bits;

use imbe_bits::*;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.len() < 2 {
        eprintln!("usage: imbe_scoreboard <src.pcm> <ref.imbe.bits> [--live] [--offset N] [--unit 11|12] [--sweep N]");
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
    let span = val("--sweep").unwrap_or(6);
    let unit = val("--unit").map(|v| v as usize);

    let pcm = read_pcm(&argv[0]);
    let (refr, unit_bytes) = load_reference(&argv[1], unit);
    let ours = if live {
        ours_live(&pcm)
    } else {
        ours_whole(&pcm)
    };

    println!(
        "source {}  ({} frames of PCM)\nreference {}  ({}-byte units, {} frames)\nours: {} path, {} frames\n",
        argv[0],
        pcm.len().div_ceil(FRAME),
        argv[1],
        unit_bytes,
        refr.len(),
        if live { "LiveEncoder streaming" } else { "Vocoder::encode whole-buffer" },
        ours.len(),
    );

    // ---- alignment sweep --------------------------------------------------
    // Scored on raw payload BIT agreement: it needs no field model, so it
    // cannot be biased by the field model being wrong.
    let sweep = |off: i64| -> (usize, usize, usize, usize, usize, usize) {
        let (mut bits, mut bytes, mut frames, mut b0, mut lm, mut n) = (0, 0, 0, 0, 0, 0);
        for (i, o) in ours.iter().enumerate() {
            let j = i as i64 + off;
            if j < 0 || j as usize >= refr.len() {
                continue;
            }
            let r = &refr[j as usize];
            n += 1;
            for p in 0..PAYLOAD {
                bits += 8 - (o.raw[p] ^ r.raw[p]).count_ones() as usize;
                if o.raw[p] == r.raw[p] {
                    bytes += 1;
                }
            }
            if o.raw == r.raw {
                frames += 1;
            }
            if o.b0 == r.b0 {
                b0 += 1;
            }
            if o.l == r.l {
                lm += 1;
            }
        }
        (bits, bytes, frames, b0, lm, n)
    };

    println!("ALIGNMENT SWEEP (scored on raw payload bits — no field model)");
    println!(
        "  {:>6} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "off", "bit%", "byte%", "frame%", "b0%", "L%"
    );
    let mut best = (f64::NEG_INFINITY, 0i64);
    let mut curve: Vec<(i64, f64)> = Vec::new();
    for off in -span..=span {
        let (bits, bytes, frames, b0, lm, n) = sweep(off);
        if n == 0 {
            continue;
        }
        let bp = 100.0 * bits as f64 / (8 * PAYLOAD * n) as f64;
        curve.push((off, bp));
        println!(
            "  {:>+6} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            off,
            bp,
            pc(bytes, PAYLOAD * n),
            pc(frames, n),
            pc(b0, n),
            pc(lm, n),
        );
        if bp > best.0 {
            best = (bp, off);
        }
    }
    let off = val("--offset").unwrap_or(best.1);
    // Sharpness: how far the winner stands above its immediate neighbours and
    // above the best non-adjacent offset. A plumbing offset is a spike; a
    // content-driven coincidence is a plateau.
    let at = |o: i64| curve.iter().find(|c| c.0 == o).map(|c| c.1);
    let nb = [at(best.1 - 1), at(best.1 + 1)]
        .into_iter()
        .flatten()
        .fold(f64::NEG_INFINITY, f64::max);
    let far = curve
        .iter()
        .filter(|c| (c.0 - best.1).abs() >= 2)
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\n  peak at {:+} = {:.2}%   adjacent best {:.2}% (-{:.2})   non-adjacent best {:.2}% (-{:.2})",
        best.1,
        best.0,
        nb,
        best.0 - nb,
        far,
        best.0 - far
    );
    if off != best.1 {
        println!("  (scoring at forced offset {off:+})");
    }

    // ---- scored population -------------------------------------------------
    let pairs: Vec<(&Dec, &Dec)> = ours
        .iter()
        .enumerate()
        .filter_map(|(i, o)| {
            let j = i as i64 + off;
            (j >= 0 && (j as usize) < refr.len()).then(|| (o, &refr[j as usize]))
        })
        .collect();
    let n = pairs.len();
    println!("\nSCORED at offset {off:+} over {n} frame pairs");

    // ---- per-byte ----------------------------------------------------------
    println!("\nPER-BYTE (priority order: byte 0 carries the strongest-protected bits)");
    print!("  exact%  ");
    for p in 0..PAYLOAD {
        let hit = pairs.iter().filter(|(o, r)| o.raw[p] == r.raw[p]).count();
        print!("{:>6.1}", pc(hit, n));
    }
    print!("\n  bit%    ");
    for p in 0..PAYLOAD {
        let hit: usize = pairs
            .iter()
            .map(|(o, r)| 8 - (o.raw[p] ^ r.raw[p]).count_ones() as usize)
            .sum();
        print!("{:>6.1}", pc(hit, 8 * n));
    }
    let allbits: usize = pairs
        .iter()
        .map(|(o, r)| {
            (0..PAYLOAD)
                .map(|p| 8 - (o.raw[p] ^ r.raw[p]).count_ones() as usize)
                .sum::<usize>()
        })
        .sum();
    println!(
        "\n  overall bit agreement {:.2}%   whole-frame byte-exact {:.2}%",
        pc(allbits, 8 * PAYLOAD * n),
        pc(pairs.iter().filter(|(o, r)| o.raw == r.raw).count(), n),
    );

    // ---- per-field ---------------------------------------------------------
    let both = |o: &Dec, r: &Dec| o.valid && r.valid;
    let nv = pairs.iter().filter(|(o, r)| both(o, r)).count();
    let lm: Vec<&(&Dec, &Dec)> = pairs
        .iter()
        .filter(|(o, r)| both(o, r) && o.l == r.l)
        .collect();

    println!(
        "\nPER-FIELD ({nv} both-valid frames, {} of them L-matching)",
        lm.len()
    );
    println!(
        "  b0 pitch  {:>7.2}%   (all {n} frames; L-independent bit positions)",
        pc(pairs.iter().filter(|(o, r)| o.b0 == r.b0).count(), n)
    );
    println!(
        "  L harms   {:>7.2}%   (both-valid)",
        pc(
            pairs
                .iter()
                .filter(|(o, r)| both(o, r) && o.l == r.l)
                .count(),
            nv
        )
    );
    println!(
        "  K bands   {:>7.2}%   (both-valid)",
        pc(
            pairs
                .iter()
                .filter(|(o, r)| both(o, r) && o.k == r.k)
                .count(),
            nv
        )
    );
    println!("  b1 v/uv   {:>7.2}%   (L-matching only — otherwise K differs and the field is not the same field)",
        pc(lm.iter().filter(|(o, r)| o.b[1] == r.b[1]).count(), lm.len()));
    println!(
        "  b2 gain   {:>7.2}%   (L-matching)",
        pc(
            lm.iter().filter(|(o, r)| o.b[2] == r.b[2]).count(),
            lm.len()
        )
    );
    println!(
        "  sync bit  {:>7.2}%   (L-matching; b̂_{{L+2}})",
        pc(
            lm.iter()
                .filter(|(o, r)| o.b[o.top()] == r.b[r.top()])
                .count(),
            lm.len()
        )
    );

    // Amplitudes: aggregate over L-matching frames, plus per-index and by
    // position decile (the widths taper down the block, so a flat per-index
    // list over 53 indices is unreadable).
    let (mut anum, mut aden) = (0usize, 0usize);
    let mut idx_hit = [0usize; IMBE_B_MAX];
    let mut idx_n = [0usize; IMBE_B_MAX];
    let mut dec_hit = [0usize; 10];
    let mut dec_n = [0usize; 10];
    let mut amp_all = 0usize;
    for (o, r) in lm.iter() {
        let mut all = true;
        let rng = o.amp_range();
        let len = rng.len().max(1);
        for (k, p) in rng.clone().enumerate() {
            aden += 1;
            idx_n[p] += 1;
            let d = (10 * k / len).min(9);
            dec_n[d] += 1;
            if o.b[p] == r.b[p] {
                anum += 1;
                idx_hit[p] += 1;
                dec_hit[d] += 1;
            } else {
                all = false;
            }
        }
        if all {
            amp_all += 1;
        }
    }
    // The same aggregate WITHOUT the L condition. Conditioning on L selects a
    // population, so the conditioned figure moves when `L` moves even if no
    // amplitude decision changed; the unconditional one scores every reference
    // coefficient, counting an L-mismatched frame's whole amplitude block wrong.
    let (mut unum, mut uden) = (0usize, 0usize);
    for (o, r) in pairs.iter() {
        if !o.valid || !r.valid {
            continue;
        }
        let same_l = o.l == r.l;
        for p in r.amp_range() {
            uden += 1;
            if same_l && o.b[p] == r.b[p] {
                unum += 1;
            }
        }
    }
    println!(
        "  amps agg  {:>7.2}%   ({aden} coefficients, L-matching frames)",
        pc(anum, aden)
    );
    // The same block WITHOUT the L condition. Conditioning on `L` scores a
    // different population every time `L` moves, so the conditioned figure can
    // fall while the block gets better (or the reverse); the gap between these
    // three numbers IS the b̂₀/L cap on the amplitude field.
    //   * common   — the prefix both sides have, so neither side's extra
    //                coefficients count against it.
    //   * refblock — every coefficient the reference transmits, so a short `L`
    //                pays for the tail it never sent.
    let (mut un, mut ud, mut un2, mut ud2) = (0usize, 0usize, 0usize, 0usize);
    for (o, r) in pairs.iter().filter(|(o, r)| both(o, r)) {
        let common = o.amp_range().end.min(r.amp_range().end);
        for p in 3..common {
            ud += 1;
            un += usize::from(o.b[p] == r.b[p]);
        }
        for p in r.amp_range() {
            ud2 += 1;
            un2 += usize::from(o.b[p] == r.b[p]);
        }
    }
    println!(
        "  amps unc  {:>7.2}%   ({ud} coefficients, ALL both-valid frames, common prefix)",
        pc(un, ud)
    );
    println!(
        "  amps ref  {:>7.2}%   ({ud2} coefficients, ALL both-valid frames, reference's own block)",
        pc(un2, ud2)
    );
    println!(
        "  amps unc  {:>7.2}%   ({uden} coefficients, ALL both-valid frames)",
        pc(unum, uden)
    );
    println!(
        "  amps all  {:>7.2}%   (every b̂₃..b̂_{{L+1}} correct in the frame)",
        pc(amp_all, lm.len())
    );

    println!("\n  amplitude block by position within b̂₃..b̂_{{L+1}} (decile)");
    print!("    ");
    for d in 0..10 {
        print!("{:>7.1}", pc(dec_hit[d], dec_n[d]));
    }
    println!("\n           ^ first coefficient                          last ^");

    println!("\n  amplitude per-index (b̂₃.. , frames sharing that index)");
    for row in (3..IMBE_B_MAX).collect::<Vec<_>>().chunks(8) {
        print!("    ");
        for &p in row {
            if idx_n[p] > 0 {
                print!("b{:<2}{:>6.1} ", p, pc(idx_hit[p], idx_n[p]));
            }
        }
        println!();
    }

    // Whole-frame parameter match: every b̂ from 0 to L+2.
    let all = lm
        .iter()
        .filter(|(o, r)| (0..=o.top()).all(|p| o.b[p] == r.b[p]))
        .count();
    println!(
        "\n  ALL b̂    {:>7.2}%   of L-matching frames   ({:.2}% of all {n} frames)",
        pc(all, lm.len()),
        pc(all, n)
    );

    // ---- prefix survival ---------------------------------------------------
    // Where does the frame die? P(b̂₀..b̂_k all correct) as k walks the vector.
    println!("\nPREFIX SURVIVAL — P(b̂₀..b̂_k all correct), L-matching frames");
    let stages: [(&str, usize); 4] = [("b0", 0), ("+b1", 1), ("+b2", 2), ("+b3", 3)];
    for (label, upto) in stages {
        let s = lm
            .iter()
            .filter(|(o, r)| (0..=upto.min(o.top())).all(|p| o.b[p] == r.b[p]))
            .count();
        print!("  {label} {:.2}%   ", pc(s, lm.len()));
    }
    println!();
    for cut in [6usize, 10, 16, 24, 40] {
        let s = lm
            .iter()
            .filter(|(o, r)| (0..=cut.min(o.top())).all(|p| o.b[p] == r.b[p]))
            .count();
        print!("  +b{cut} {:.2}%   ", pc(s, lm.len()));
    }
    println!();

    // ---- sync bit ----------------------------------------------------------
    // b̂_{L+2} is one bit at the very end of the payload. It is scored like any
    // other field above, but it is the one field whose reference value is not a
    // measurement of the audio at all, so it gets its own read-out: whether the
    // reference is a pure frame-parity toggle, and what a correct one would be
    // worth on whole-frame exactness.
    println!("\nSYNC BIT b̂_{{L+2}} (payload byte 10, bit 0)");
    let refseq: Vec<u8> = pairs.iter().map(|(_, r)| r.raw[PAYLOAD - 1] & 1).collect();
    let ourseq: Vec<u8> = pairs.iter().map(|(o, _)| o.raw[PAYLOAD - 1] & 1).collect();
    let toggle = |seq: &[u8], phase: u8| {
        seq.iter()
            .enumerate()
            .filter(|(i, &b)| b == (phase ^ (*i as u8 & 1)))
            .count()
    };
    let (rp0, rp1) = (toggle(&refseq, 0), toggle(&refseq, 1));
    let (op0, op1) = (toggle(&ourseq, 0), toggle(&ourseq, 1));
    println!(
        "  reference: {:.2}% ones;  matches a strict frame-parity toggle on {:.2}% (best phase) of frames",
        pc(refseq.iter().filter(|&&b| b == 1).count(), n),
        pc(rp0.max(rp1), n)
    );
    println!(
        "  ours:      {:.2}% ones;  matches a strict frame-parity toggle on {:.2}% (best phase) of frames",
        pc(ourseq.iter().filter(|&&b| b == 1).count(), n),
        pc(op0.max(op1), n)
    );
    let sync_hit = pairs
        .iter()
        .filter(|(o, r)| (o.raw[PAYLOAD - 1] & 1) == (r.raw[PAYLOAD - 1] & 1))
        .count();
    let rest_exact = pairs
        .iter()
        .filter(|(o, r)| {
            o.raw[..PAYLOAD - 1] == r.raw[..PAYLOAD - 1]
                && o.raw[PAYLOAD - 1] >> 1 == r.raw[PAYLOAD - 1] >> 1
        })
        .count();
    println!(
        "  agreement {:.2}%   whole-frame exact IGNORING this bit: {:.2}% (against {:.2}% with it)",
        pc(sync_hit, n),
        pc(rest_exact, n),
        pc(pairs.iter().filter(|(o, r)| o.raw == r.raw).count(), n),
    );

    // ---- internal state ----------------------------------------------------
    internal_state(&pcm, &pairs, live);
}

/// Read the pitch word the chain actually carries, not the one it transmits.
fn internal_state(pcm: &[i16], pairs: &[(&Dec, &Dec)], live: bool) {
    println!("\nINTERNAL STATE — the pitch word behind b̂₀");
    let track = internal_ambe_b0(pcm);
    if track.is_empty() {
        println!("  (no AMBE pitch track for this clip)");
        return;
    }

    // 1. Reachability. Our b̂₀ is the image of a COARSER ladder: the AMBE pitch
    //    table has 128 entries, the IMBE grid 208 cells. Cells with no AMBE
    //    preimage can never be emitted no matter how good the estimator is, so
    //    any oracle frame landing there is lost before the search starts.
    let ok = reachable_imbe_b0();
    let reach = ok.iter().take(208).filter(|&&v| v).count();
    let unreachable = pairs
        .iter()
        .filter(|(_, r)| {
            let b = r.b0;
            !(0..208).contains(&b) || !ok[b as usize]
        })
        .count();
    let distinct_ref: std::collections::BTreeSet<i16> = pairs.iter().map(|(_, r)| r.b0).collect();
    let unreach_cells = distinct_ref
        .iter()
        .filter(|&&b| !(0..208).contains(&b) || !ok[b as usize])
        .count();
    println!(
        "  IMBE cells reachable from the AMBE ladder: {reach}/208 ({:.1}%)",
        100.0 * reach as f64 / 208.0
    );
    println!(
        "  oracle frames whose b̂₀ cell is UNREACHABLE by our chain: {:.2}%  ({unreach_cells} of {} distinct oracle cells)",
        pc(unreachable, pairs.len()),
        distinct_ref.len()
    );
    println!("  (the reachability ceiling above binds only the AMBE-index round trip, which is\n   what a build WITHOUT BLIP25_IMBE_NEXT uses; the section below reads the word)");

    // 1b. The pitch word itself, which is what the IMBE grid is addressed from
    //     when `BLIP25_IMBE_NEXT` is on. Two things are worth separating:
    //     whether the right ANALYSIS frame feeds the right EMIT frame (a
    //     plumbing question with a spike for an answer), and whether the misses
    //     that remain are sub-cell rounding or gross.
    if !live {
        let pw = internal_pitch_word(pcm);
        if !pw.is_empty() {
            println!("\n  pitch-word injection lag (b̂₀ from the word alone — no packer override)");
            print!("   ");
            let mut at_two = 0.0f64;
            for lag in 0..=4i64 {
                let (mut hit, mut n) = (0usize, 0usize);
                for (i, (_, r)) in pairs.iter().enumerate() {
                    let Some(&w) = pw.get((i as i64 + lag).max(0) as usize) else {
                        continue;
                    };
                    n += 1;
                    if i16::from(blip25_codec::imbe::quantize::imbe_b0_from_pitch_word(w)) == r.b0 {
                        hit += 1;
                    }
                }
                let v = pc(hit, n);
                if lag == 2 {
                    at_two = v;
                }
                print!("  lag{lag:+} {v:.2}%");
            }
            println!("\n    (the shipped injection is lag +2; a plumbing offset is a spike, a fit is a plateau)");

            let mut d: Vec<i64> = Vec::new();
            for (i, (_, r)) in pairs.iter().enumerate() {
                let Some(&w) = pw.get(i + 2) else { continue };
                let b = i16::from(blip25_codec::imbe::quantize::imbe_b0_from_pitch_word(w));
                if b != r.b0 {
                    d.push(i64::from(b - r.b0).abs());
                }
            }
            let one = d.iter().filter(|&&x| x == 1).count();
            println!(
                "    at lag +2: {at_two:.2}% exact; of the {} misses {:.1}% are ±1 cell, median |Δ| {}",
                d.len(),
                pc(one, d.len()),
                median(&mut d)
            );
            println!("    (±1 is a resolution loss; a large median is the packer emitting something\n     that is not a pitch measurement at all)");
        }
    }

    // 1c. The packer's all-unvoiced override, scored as the event it is.
    let (mut tp, mut fp, mut fnn) = (0usize, 0usize, 0usize);
    for (o, r) in pairs {
        match (o.b0 == IMBE_UV_B0, r.b0 == IMBE_UV_B0) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fnn += 1,
            _ => {}
        }
    }
    println!(
        "\n  all-unvoiced override (b̂₀ == {IMBE_UV_B0}): reference fires on {:.2}% of frames;\n    ours precision {:.2}%  recall {:.2}%   (tp {tp} fp {fp} fn {fnn})",
        pc(tp + fnn, pairs.len()),
        pc(tp, tp + fp),
        pc(tp, tp + fnn)
    );

    // 2. Sub-cell placement. Where does our continuous ω₀ sit relative to the
    //    oracle's cell? |Δ| in cells: <0.5 means the IMBE quantizer would have
    //    rounded to the oracle's cell had nothing else intervened.
    let mut dcell: Vec<f64> = Vec::new();
    let mut within_half = 0usize;
    let mut counted = 0usize;
    for (i, (_, r)) in pairs.iter().enumerate() {
        // pairs[i] corresponds to our emitted frame index i (offset applied on
        // the reference side), and the streaming path's emit schedule differs,
        // so only the whole-buffer path indexes the track directly.
        if live {
            break;
        }
        let Some(&ab0) = track.get(i) else { continue };
        let Some(w) = ambe_omega0(ab0) else { continue };
        if !(0..208).contains(&r.b0) {
            continue;
        }
        let c = imbe_omega0(r.b0);
        let width = (imbe_omega0(r.b0 + 1) - c).abs().max(1e-12);
        let d = (w - c) / width;
        dcell.push(d);
        if d.abs() < 0.5 {
            within_half += 1;
        }
        counted += 1;
    }
    if counted == 0 {
        println!(
            "  sub-cell placement: not measured on the streaming path (emit schedule differs)"
        );
    } else {
        let mut abs: Vec<f64> = dcell.iter().map(|d| d.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| abs[((abs.len() - 1) as f64 * p) as usize];
        let mean: f64 = dcell.iter().sum::<f64>() / dcell.len() as f64;
        println!(
            "  our internal ω₀ vs the oracle's b̂₀ cell, in IMBE cell widths ({counted} frames):"
        );
        println!(
            "    within ±0.5 cell (would quantize to the oracle's cell): {:.2}%",
            pc(within_half, counted)
        );
        println!(
            "    |Δ| median {:.2}  p10 {:.2}  p90 {:.2}   signed mean {:+.2}",
            q(0.5),
            q(0.1),
            q(0.9),
            mean
        );
        println!(
            "    (a transmitted-b̂₀ audit sees only the sign of the rounding; this sees the error)"
        );
    }

    // 3. Split the b̂₀ loss into the two quantizations it passes through.
    //
    //    The oracle's cell implies an ω₀. Ask which AMBE cell that ω₀ would
    //    land in, and compare it to the AMBE cell our chain actually carries.
    //    That separates "our pitch estimate is wrong" from "our pitch estimate
    //    is right and the AMBE→IMBE requantization threw it away".
    let ambe_cells: Vec<f64> = (0..=119u8).filter_map(ambe_omega0).collect();
    let q_ambe = |w: f64| -> usize {
        let mut best = (f64::INFINITY, 0usize);
        for (i, &c) in ambe_cells.iter().enumerate() {
            let d = (c - w).abs();
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    };
    let (mut have, mut ambe_ok, mut both_ok, mut ceiling, mut direct) = (0usize, 0, 0, 0, 0);
    for (i, (o, r)) in pairs.iter().enumerate() {
        if live {
            break;
        }
        let Some(&ab0) = track.get(i) else { continue };
        let Some(w) = ambe_omega0(ab0) else { continue };
        if !(0..208).contains(&r.b0) {
            continue;
        }
        have += 1;
        let (q, _) = blip25_codec::imbe::quantize_pitch(w as f32);
        if i16::from(q) == r.b0 {
            direct += 1;
        }
        let want = imbe_omega0(r.b0);
        let a_ref = q_ambe(want);
        // Ceiling: a PERFECT AMBE-level pitch, requantized onto the IMBE grid.
        let (qc, _) = blip25_codec::imbe::quantize_pitch(ambe_cells[a_ref] as f32);
        if i16::from(qc) == r.b0 {
            ceiling += 1;
        }
        if a_ref == usize::from(ab0) {
            ambe_ok += 1;
            if o.b0 == r.b0 {
                both_ok += 1;
            }
        }
    }
    if have > 0 {
        println!("\n  the two quantizations b̂₀ passes through ({have} frames):");
        println!(
            "    our AMBE cell == the cell the oracle's ω₀ falls in:  {:.2}%   <- the estimate",
            pc(ambe_ok, have)
        );
        println!(
            "    ... and b̂₀ still survives the AMBE→IMBE requantize:  {:.2}% of those",
            pc(both_ok, ambe_ok)
        );
        println!(
            "    CEILING: a perfect AMBE-cell pitch requantized to IMBE: {:.2}%   <- the grid",
            pc(ceiling, have)
        );
        println!(
            "    actual b̂₀ from requantizing our internal word:        {:.2}%",
            pc(direct, have)
        );
        println!(
            "    (the gap between the ceiling and 100% is lost to carrying pitch as an\n     AMBE index rather than a fundamental; no estimator work recovers it)"
        );
    }
}
