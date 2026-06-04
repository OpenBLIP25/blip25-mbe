//! Soft-decision FEC gain test against DVSI's own reference vectors.
//!
//! For each error-injected `*_sd.bit` vector (WITH FEC, 4-bit soft-decision
//! format, N% injected channel BER), decode every frame two ways and score
//! both against the ground-truth info recovered from the clean, error-free
//! `.bit`:
//!
//!   * **soft** — `decode_frame_soft` over the 4-bit SD values
//!     (Chase-II soft Golay/Hamming, the chip's soft-decision path)
//!   * **hard** — hard-slice each SD value's MSB, then `decode_frame`
//!     (bounded-distance Golay/Hamming, the baseline)
//!
//! The soft decoder should recover strictly more frames as BER climbs —
//! the Golay/Hamming coding gain. Pure info-bit metric (no synthesis),
//! so the result is deterministic and unambiguous.
//!
//! Two rates are supported, selected with `--rate`:
//!   * **full** (default) — full-rate IMBE (`imbe7200`, 144 channel bits,
//!     8 info vectors). Ground truth `clean.bit`, vectors `clean_eN_sd.bit`.
//!   * **33** — half-rate AMBE+2 / DVSI Rate 33 (`rate33`, 72
//!     channel bits, 4 info vectors). This is the P25 Phase 2 path, shared
//!     bit-for-bit with DMR and NXDN. Ground truth `dam.bit`, vectors
//!     `dam_eN_sd.bit`.
//!
//! Usage:
//!   cargo run --release -p blip25-mbe --example dvsi_soft_gain -- [--rate 33] [VECTOR_DIR]

use blip25_mbe::dvsi_soft_decision::unpack_nibble_stream;

const DEFAULT_DIR_FULL: &str =
    "/mnt/share/P25-IQ-Samples/Research/DVSI Vectors/tv-std/tv/p25";
const DEFAULT_DIR_R33: &str =
    "/mnt/share/P25-IQ-Samples/Research/DVSI Vectors/tv-rc/r33";

/// MSB-first hard `.bit` bytes -> dibits, dibit = (hi<<1)|lo, matching the
/// crate's `unpack_dibits_n` convention. Length is `8 * bytes / 2` dibits.
fn hard_bytes_to_dibits(frame: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(frame.len() * 8);
    for &b in frame {
        for k in 0..8 {
            bits.push((b >> (7 - k)) & 1);
        }
    }
    bits.chunks_exact(2).map(|c| (c[0] << 1) | c[1]).collect()
}

/// 4-bit SD nibble stream -> soft LLRs (SD0 first), one per channel bit.
fn soft_bytes_to_llrs(frame: &[u8]) -> Vec<i8> {
    unpack_nibble_stream(frame)
}

/// Hard decision of a soft frame: slice each LLR's sign, regroup dibits.
fn soft_llrs_to_dibits(soft: &[i8]) -> Vec<u8> {
    soft.chunks_exact(2)
        .map(|c| (u8::from(c[0] > 0) << 1) | u8::from(c[1] > 0))
        .collect()
}

/// Per-rate decode wiring: turns a clean hard `.bit` chunk and a noisy soft
/// `*_sd.bit` chunk into the decoded info vectors, hiding the `Frame` type.
struct RateOps {
    /// Bytes per frame in a hard `.bit` file.
    hard_bytes: usize,
    /// Bytes per frame in a soft `*_sd.bit` file (4-bit SD nibbles, 2/byte).
    soft_bytes: usize,
    /// Indices of the FEC-coded info vectors — the only ones soft decoding
    /// can rescue. Uncoded vectors (`û₂`/`û₃` for Rate 33, `û₇` for full
    /// rate) pass through verbatim and gate the all-vector "frame ok" metric
    /// regardless of soft vs hard, so they're broken out separately.
    coded_idx: &'static [usize],
    /// Decode a clean hard frame -> (info vectors, residual FEC errors).
    decode_hard: fn(&[u8]) -> (Vec<u16>, u16),
    /// Decode a noisy soft frame -> (soft info, hard-sliced info).
    decode_soft: fn(&[u8]) -> (Vec<u16>, Vec<u16>),
}

fn full_rate_ops() -> RateOps {
    use blip25_mbe::imbe7200::frame::{decode_frame, decode_frame_soft};
    fn hard(chunk: &[u8]) -> (Vec<u16>, u16) {
        let dibits: [u8; 72] = hard_bytes_to_dibits(chunk).try_into().unwrap();
        let f = decode_frame(&dibits);
        (f.info.to_vec(), f.error_total())
    }
    fn soft(chunk: &[u8]) -> (Vec<u16>, Vec<u16>) {
        let llrs: [i8; 144] = soft_bytes_to_llrs(chunk).try_into().unwrap();
        let f_soft = decode_frame_soft(&llrs);
        let dibits: [u8; 72] = soft_llrs_to_dibits(&llrs).try_into().unwrap();
        let f_hard = decode_frame(&dibits);
        (f_soft.info.to_vec(), f_hard.info.to_vec())
    }
    // û₀..û₃ Golay, û₄..û₆ Hamming; only û₇ (7 bits) is uncoded.
    RateOps {
        hard_bytes: 18,
        soft_bytes: 72,
        coded_idx: &[0, 1, 2, 3, 4, 5, 6],
        decode_hard: hard,
        decode_soft: soft,
    }
}

fn rate33_ops() -> RateOps {
    use blip25_mbe::rate33::frame::{decode_frame, decode_frame_soft};
    fn hard(chunk: &[u8]) -> (Vec<u16>, u16) {
        let dibits: [u8; 36] = hard_bytes_to_dibits(chunk).try_into().unwrap();
        let f = decode_frame(&dibits);
        (f.info.to_vec(), f.error_total())
    }
    fn soft(chunk: &[u8]) -> (Vec<u16>, Vec<u16>) {
        let llrs: [i8; 72] = soft_bytes_to_llrs(chunk).try_into().unwrap();
        let f_soft = decode_frame_soft(&llrs);
        let dibits: [u8; 36] = soft_llrs_to_dibits(&llrs).try_into().unwrap();
        let f_hard = decode_frame(&dibits);
        (f_soft.info.to_vec(), f_hard.info.to_vec())
    }
    // û₀ [24,12] Golay, û₁ [23,12] Golay; û₂ (11) and û₃ (14) are uncoded.
    RateOps {
        hard_bytes: 9,
        soft_bytes: 36,
        coded_idx: &[0, 1],
        decode_hard: hard,
        decode_soft: soft,
    }
}

fn vector_mismatches(a: &[u16], b: &[u16]) -> u32 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u32
}

fn main() {
    // Parse `--rate full|33` and an optional positional VECTOR_DIR.
    let mut rate = "full".to_string();
    let mut dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rate" => rate = args.next().unwrap_or_else(|| "full".to_string()),
            other => dir = Some(other.to_string()),
        }
    }

    let (ops, default_dir, truth_name, vec_prefix) = match rate.as_str() {
        "33" | "r33" | "rate33" => (rate33_ops(), DEFAULT_DIR_R33, "dam.bit", "dam"),
        "full" | "fullrate" | "imbe" => {
            (full_rate_ops(), DEFAULT_DIR_FULL, "clean.bit", "clean")
        }
        other => {
            eprintln!("unknown --rate {other:?} (expected `full` or `33`)");
            std::process::exit(2);
        }
    };
    let dir = dir.unwrap_or_else(|| default_dir.to_string());

    println!("rate {rate}: vectors in {dir}\n");

    // Ground truth: decode the clean, error-free .bit.
    let clean = std::fs::read(format!("{dir}/{truth_name}"))
        .unwrap_or_else(|e| panic!("read {dir}/{truth_name}: {e}"));
    let mut truth: Vec<Vec<u16>> = Vec::new();
    let mut truth_resid = 0u32;
    for chunk in clean.chunks_exact(ops.hard_bytes) {
        let (info, resid) = (ops.decode_hard)(chunk);
        truth_resid += u32::from(resid);
        truth.push(info);
    }
    println!(
        "ground truth: {} frames from {truth_name} (residual FEC errors {truth_resid} — \
         should be ~0 if bit ordering matches)\n",
        truth.len(),
    );

    // Whether every coded info vector matches truth (uncoded vectors ignored).
    let coded_match = |a: &[u16], b: &[u16]| ops.coded_idx.iter().all(|&k| a[k] == b[k]);

    println!(
        "all info vectors:                          coded vectors only ({} of {}):",
        ops.coded_idx.len(),
        truth.first().map_or(0, |t| t.len()),
    );
    println!(
        "{:<6} {:>6} {:>10} {:>10} {:>6}   {:>10} {:>10} {:>6}",
        "vector", "frames", "hard ok", "soft ok", "resc", "hard ok", "soft ok", "resc"
    );
    println!("{}", "-".repeat(74));

    for level in ["e1", "e2", "e5", "e10"] {
        let path = format!("{dir}/{vec_prefix}_{level}_sd.bit");
        let Ok(buf) = std::fs::read(&path) else {
            println!("{level:<6} (missing: {path})");
            continue;
        };

        let mut hard_ok = 0u32;
        let mut soft_ok = 0u32;
        let mut rescued = 0u32; // soft frame correct AND hard frame wrong
        let mut c_hard_ok = 0u32;
        let mut c_soft_ok = 0u32;
        let mut c_rescued = 0u32; // coded vectors: soft right, hard wrong

        for (i, chunk) in buf.chunks_exact(ops.soft_bytes).enumerate() {
            if i >= truth.len() {
                break;
            }
            let (soft_info, hard_info) = (ops.decode_soft)(chunk);
            let h = hard_info == truth[i];
            let s = soft_info == truth[i];
            hard_ok += u32::from(h);
            soft_ok += u32::from(s);
            rescued += u32::from(s && !h);

            let ch = coded_match(&hard_info, &truth[i]);
            let cs = coded_match(&soft_info, &truth[i]);
            c_hard_ok += u32::from(ch);
            c_soft_ok += u32::from(cs);
            c_rescued += u32::from(cs && !ch);
        }

        let n = (buf.len() / ops.soft_bytes).min(truth.len());
        let pct = |x: u32| 100.0 * x as f64 / n as f64;
        println!(
            "{:<6} {:>6} {:>4} {:>4.0}% {:>4} {:>4.0}% {:>6}   {:>4} {:>4.0}% {:>4} {:>4.0}% {:>6}",
            level,
            n,
            hard_ok,
            pct(hard_ok),
            soft_ok,
            pct(soft_ok),
            rescued,
            c_hard_ok,
            pct(c_hard_ok),
            c_soft_ok,
            pct(c_soft_ok),
            c_rescued,
        );
    }

    println!(
        "\nframe ok = all info vectors match clean truth; resc = frames soft got \
         right that hard got wrong.\n\"coded vectors only\" scores just the \
         FEC-protected vectors — the only ones soft decoding can rescue. The gap \
         between the two halves is the uncoded-bit floor (û₂/û₃ at Rate 33, û₇ at \
         full rate) that no FEC can recover."
    );
}
