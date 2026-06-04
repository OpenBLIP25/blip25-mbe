//! Empirically derive the DVSI R34 (half-rate no-FEC, 49-bit) bit
//! serialization order from the DVSI RC test vectors.
//!
//! Method: for each frame, decode the matching `r33/<name>.bit` (FEC,
//! validated path) to its 49 information bits in natural
//! `û₀‖û₁‖û₂‖û₃` MSB-first order, and read the raw `r34/<name>.bit`
//! 7-byte frame as 56 MSB-first bits. Across all frames of all
//! vectors, each bit position has a "signature" (its value sequence).
//! A natural bit `i` corresponds to the r34 bit `j` whose signature is
//! identical. A 1:1 signature match across every position proves r34
//! is a pure permutation and reveals the exact mapping.
//!
//! Run: `cargo run --release --example derive_r34_order -- <vec dir> <stem>...`
//! e.g. `... -- DVSI/Vectors/tv-rc alltone clean dam mark alert`

use std::path::Path;

use blip25_mbe::rate33::frame::{decode_frame, INFO_WIDTHS, DIBITS_PER_FRAME};

const FEC_BYTES: usize = 9; // r33
const NOFEC_BYTES: usize = 7; // r34
const INFO_BITS: usize = 49;
const NOFEC_BITS: usize = NOFEC_BYTES * 8; // 56

fn bytes_to_dibits(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    // MSB-first bit pairs, matching unpack_dibits_ambe_plus2 in the harness.
    let mut out = [0u8; DIBITS_PER_FRAME];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[(2 * i) / 8] >> (7 - ((2 * i) % 8))) & 1;
        let lo = (bytes[(2 * i + 1) / 8] >> (7 - ((2 * i + 1) % 8))) & 1;
        *slot = (hi << 1) | lo;
    }
    out
}

/// Natural 49-bit info vector: û₀(12) ‖ û₁(12) ‖ û₂(11) ‖ û₃(14), MSB-first.
fn natural_bits(info: &[u16; 4]) -> [u8; INFO_BITS] {
    let mut out = [0u8; INFO_BITS];
    let mut idx = 0;
    for (w, &v) in INFO_WIDTHS.iter().zip(info.iter()) {
        for k in (0..*w as usize).rev() {
            out[idx] = ((v >> k) & 1) as u8;
            idx += 1;
        }
    }
    out
}

fn r34_bits(bytes: &[u8]) -> [u8; NOFEC_BITS] {
    let mut out = [0u8; NOFEC_BITS];
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = (bytes[j / 8] >> (7 - (j % 8))) & 1;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: derive_r34_order <vec dir> <stem>...");
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let stems = &args[2..];

    // signatures[pos] = Vec<bit> over all frames, for natural (49) and r34 (56).
    let mut nat_sig: Vec<Vec<u8>> = vec![Vec::new(); INFO_BITS];
    let mut r34_sig: Vec<Vec<u8>> = vec![Vec::new(); NOFEC_BITS];
    let mut total_frames = 0usize;

    for stem in stems {
        let r33 = std::fs::read(dir.join("r33").join(format!("{stem}.bit")))
            .unwrap_or_else(|e| panic!("read r33/{stem}.bit: {e}"));
        let r34 = std::fs::read(dir.join("r34").join(format!("{stem}.bit")))
            .unwrap_or_else(|e| panic!("read r34/{stem}.bit: {e}"));
        let n33 = r33.len() / FEC_BYTES;
        let n34 = r34.len() / NOFEC_BYTES;
        let n = n33.min(n34);
        if n33 != n34 {
            eprintln!("WARN {stem}: r33 has {n33} frames, r34 has {n34}; using {n}");
        }
        for f in 0..n {
            let dibits = bytes_to_dibits(&r33[f * FEC_BYTES..(f + 1) * FEC_BYTES]);
            let frame = decode_frame(&dibits);
            let nb = natural_bits(&frame.info);
            let rb = r34_bits(&r34[f * NOFEC_BYTES..(f + 1) * NOFEC_BYTES]);
            for i in 0..INFO_BITS {
                nat_sig[i].push(nb[i]);
            }
            for j in 0..NOFEC_BITS {
                r34_sig[j].push(rb[j]);
            }
        }
        total_frames += n;
        println!("loaded {stem}: {n} frames");
    }
    println!("\ntotal frames: {total_frames}\n");

    // For each natural bit, find r34 positions with identical signature.
    let mut mapping = vec![Vec::<usize>::new(); INFO_BITS];
    for i in 0..INFO_BITS {
        for j in 0..NOFEC_BITS {
            if nat_sig[i] == r34_sig[j] {
                mapping[i].push(j);
            }
        }
    }

    // Report. Label each natural bit by its (word, bit-in-word).
    let labels: Vec<String> = {
        let mut v = Vec::new();
        for (wi, w) in INFO_WIDTHS.iter().enumerate() {
            for k in (0..*w).rev() {
                v.push(format!("u{wi}[{k}]"));
            }
        }
        v
    };

    let mut clean = true;
    println!("natural_bit  label     -> r34 position(s)   (#candidates)");
    for i in 0..INFO_BITS {
        let cands = &mapping[i];
        let varies = nat_sig[i].iter().any(|&b| b != nat_sig[i][0]);
        let mark = if cands.len() == 1 { " " } else { clean = false; "*" };
        println!(
            "{mark} nat[{i:2}]   {:8} -> {:?}{}",
            labels[i],
            cands,
            if varies { "" } else { "  (CONSTANT natural bit — ambiguous)" }
        );
    }

    // r34 positions never claimed by any varying natural bit = padding/const.
    println!("\nr34 bit variation (over all frames):");
    for j in 0..NOFEC_BITS {
        let varies = r34_sig[j].iter().any(|&b| b != r34_sig[j][0]);
        if !varies {
            println!("  r34[{j:2}] = CONSTANT {}", r34_sig[j][0]);
        }
    }

    if clean {
        println!("\n=> CLEAN 1:1 permutation. Mapping (r34 position -> natural label):");
        let mut inv = vec![String::from("?"); NOFEC_BITS];
        // r34_to_nat[j] = natural-bit index packed at r34 position j.
        let mut r34_to_nat = vec![usize::MAX; INFO_BITS];
        for i in 0..INFO_BITS {
            inv[mapping[i][0]] = labels[i].clone();
            r34_to_nat[mapping[i][0]] = i;
        }
        for j in 0..NOFEC_BITS {
            println!("  r34[{j:2}] = {}", inv[j]);
        }
        // Bijection check over the 49 used positions.
        let mut seen = vec![false; INFO_BITS];
        let mut bij = true;
        for &n in r34_to_nat.iter() {
            if n == usize::MAX || seen[n] {
                bij = false;
                break;
            }
            seen[n] = true;
        }
        println!("\nbijection over 49 positions: {bij}");
        print!("\n/// r34[j] (no-FEC bit j, MSB-first) = natural info bit R34_BIT_ORDER[j],\n/// where natural = û₀(12)‖û₁(12)‖û₂(11)‖û₃(14) MSB-first. Derived from DVSI RC vectors.\npub const R34_BIT_ORDER: [u8; 49] = [");
        for (j, &n) in r34_to_nat.iter().enumerate() {
            if j % 12 == 0 {
                print!("\n    ");
            }
            print!("{n}, ");
        }
        println!("\n];");
    } else {
        println!("\n=> Some positions ambiguous (constant bits). Add more/varied vectors to disambiguate.");
    }
}
