//! Which 49-bit r34 layout does the reference emit?
//!
//! Takes matched `r33/<v>.bit` (9 B/frame, FEC) and `r34/<v>.bit` (7 B/frame,
//! no FEC) vectors, decodes each r33 frame to `b0..b8`, and asks which candidate
//! packing reproduces the reference's own r34 bytes. The two files are the same source
//! audio at two rates, so this reads the wire format directly, with no
//! dependence on our encoder being right.
//!
//! Expected result over the full corpus (101 vectors, 110,840 frames): the reference
//! 3-way column interleave (`R34_BIT_ORDER`) matches 100.00% of clean-vector
//! frames (97,177/97,177); parameter-order and natural-order packings match
//! zero. The only non-matching vectors are `*_dtx` (comfort-noise frames are
//! not a repacking of the voice stream) and `dam_e{1,2,5,10}_hd`, whose hit
//! rate falls monotonically with injected BER — channel errors, not a layout
//! mismatch.
//!
//! Candidate A is the shipped packer, so A and C must agree exactly; a split
//! between them is a regression. The hermetic guard is
//! `tests/codec_acceptance.rs::r34_wire_layout_is_reference_interleave_in_both_layers`.
//!
//! Usage: `cargo run --release --example r34_layout_probe [vector-root]`
//! (default root `reference-material/Vectors/tv-rc`).

use std::fs;

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reference-material/Vectors/tv-rc".to_string());

    let mut totals = [0usize; 3];
    let mut frames_total = 0usize;
    let mut vectors = 0usize;

    let mut names: Vec<String> = fs::read_dir(format!("{root}/r34"))
        .expect("r34 dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()? == "bit" {
                Some(p.file_stem()?.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();
    names.sort();

    for name in &names {
        let (Ok(r33), Ok(r34)) = (
            fs::read(format!("{root}/r33/{name}.bit")),
            fs::read(format!("{root}/r34/{name}.bit")),
        ) else {
            continue;
        };
        if r33.len() / 9 == 0 || r33.len() / 9 != r34.len() / 7 {
            eprintln!(
                "skip {name}: frame-count mismatch {} vs {}",
                r33.len() / 9,
                r34.len() / 7
            );
            continue;
        }
        vectors += 1;

        let mut hits = [0usize; 3];
        let n = r33.len() / 9;
        for i in 0..n {
            let f33 = &r33[i * 9..i * 9 + 9];
            let want: [u8; 7] = r34[i * 7..i * 7 + 7].try_into().unwrap();

            // r33 -> u0..u3 -> b0..b8
            let frame = blip25_codec::frame::decode_bytes(f33);
            let b = blip25_codec::tables::deprioritize(&frame.info);

            // Candidate A: the engine's own packer (b0..b8 MSB-first, no permute)
            let a = blip25_codec::enc::encode_frame::encode_r34(&b);

            // Candidate C: R34_BIT_ORDER 3-way column interleave over u0..u3
            let c = blip25_mbe::rate33::frame::pack_no_fec(&frame.info);

            // Candidate B: natural u0..u3 MSB-first, no permute
            let mut bits = [0u8; 49];
            let widths = [12usize, 12, 11, 14];
            let mut k = 0;
            for (w, &word) in widths.iter().zip(frame.info.iter()) {
                for j in (0..*w).rev() {
                    if k < 49 {
                        bits[k] = ((word >> j) & 1) as u8;
                        k += 1;
                    }
                }
            }
            let mut bnat = [0u8; 7];
            for (k, &bit) in bits.iter().enumerate() {
                bnat[k / 8] |= bit << (7 - (k % 8));
            }

            for (slot, cand) in [a, bnat, c].iter().enumerate() {
                if cand == &want {
                    hits[slot] += 1;
                }
            }
        }

        frames_total += n;
        for s in 0..3 {
            totals[s] += hits[s];
        }
        println!(
            "{name:<14} frames={n:<5}  A(engine)={:<5} B(natural)={:<5} C(R34_BIT_ORDER)={:<5}",
            hits[0], hits[1], hits[2]
        );
    }

    println!("\n=== {vectors} vectors, {frames_total} frames ===");
    let pct = |h: usize| 100.0 * h as f64 / frames_total.max(1) as f64;
    println!(
        "A  engine encode_r34        {:>7} / {frames_total}  ({:.2}%)",
        totals[0],
        pct(totals[0])
    );
    println!(
        "B  natural u0..u3 MSB-first {:>7} / {frames_total}  ({:.2}%)",
        totals[1],
        pct(totals[1])
    );
    println!(
        "C  rate33 R34_BIT_ORDER     {:>7} / {frames_total}  ({:.2}%)",
        totals[2],
        pct(totals[2])
    );
}
