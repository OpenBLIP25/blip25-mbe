//! Vector-conformance harness — runs DVSI published test vectors through
//! blip25-mbe and reports per-vector pass/fail.
//!
//! Not published to crates.io. Expects `DVSI/Vectors/` at the workspace
//! root (symlink is fine) with the standard `tv-std/` and `tv-rc/` layouts.
//!
//! ## File-format inference
//!
//! The DVSI `.bit` files are not formally documented in the impl spec
//! (the format lives in DVSI's own client). We infer the most natural
//! conventions and validate them empirically by cross-checking the FEC
//! and no-FEC paths against each other:
//!
//! * **`p25/<name>.bit`** (full-FEC) — 18 bytes per 20 ms frame = 144
//!   bits packed MSB-first. The 144 bits are 72 dibits in transmission
//!   order, each dibit's high bit transmitted first. Feeds directly
//!   into [`blip25_mbe::imbe_frames::full_rate::decode_fullrate_frame`].
//!
//! * **`p25_nofec/<name>.bit`** (info-only) — 11 bytes per 20 ms frame
//!   = 88 bits packed MSB-first. The 88 bits are the eight info
//!   vectors `û₀..û₇` concatenated in order with widths 12, 12, 12, 12,
//!   11, 11, 11, 7 bits — same byte stream the encoder produces after
//!   bit prioritization, before any FEC.
//!
//! Cross-check: decoding `p25/<name>.bit` via the channel codec should
//! produce the exact `[u16; 8]` info vectors stored in
//! `p25_nofec/<name>.bit`. Bit-exact agreement validates both the
//! channel codec (deinterleave + Golay/Hamming + PN) and the no-FEC
//! packing convention against DVSI ground truth.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};

use blip25_mbe::fec::{golay_23_12_encode, hamming_15_11_encode};
use blip25_mbe::imbe_frames::fec::{deinterleave_fullrate, modulation_masks_fullrate};
use blip25_mbe::imbe_frames::full_rate::{
    INFO_WIDTHS, ImbeFrame, decode_fullrate_frame,
};

const BYTES_PER_FEC_FRAME: usize = 18;
const BYTES_PER_NOFEC_FRAME: usize = 11;
const DIBITS_PER_FRAME: usize = 72;

/// Run DVSI test vectors through blip25-mbe and report conformance.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the DVSI vectors directory (containing `tv-std/`, `tv-rc/`).
    #[arg(long, default_value = "DVSI/Vectors")]
    vectors: PathBuf,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Cross-validate the channel codec: decode `p25/<name>.bit` and
    /// compare its 88 info bits per frame against `p25_nofec/<name>.bit`.
    /// Bit-exact agreement = channel codec + nofec packing both correct.
    Compare {
        /// Vector name (e.g. `alert`, `clean`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// Stop after the first divergence, printing diagnostics.
        #[arg(long)]
        stop_on_first: bool,
    },
    /// Diagnose PN modulation by re-encoding the no-FEC reference and
    /// XORing against the deinterleaved FEC frame to recover DVSI's
    /// actual mask. Compare to our computed mask to see exactly how
    /// the conventions diverge.
    PnDiag {
        /// Vector name (e.g. `alert`).
        name: String,
        /// Use `tv-rc/` instead of `tv-std/tv/`.
        #[arg(long)]
        rc: bool,
        /// Frame index to inspect.
        #[arg(long, default_value_t = 0)]
        frame: usize,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Cmd::Compare { name, rc, stop_on_first } => {
            cmd_compare(&args.vectors, &name, rc, stop_on_first)
        }
        Cmd::PnDiag { name, rc, frame } => cmd_pn_diag(&args.vectors, &name, rc, frame),
    }
}

fn cmd_pn_diag(root: &Path, name: &str, rc: bool, frame: usize) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_bytes = fs::read(dir.join("p25").join(format!("{name}.bit")))?;
    let nofec_bytes = fs::read(dir.join("p25_nofec").join(format!("{name}.bit")))?;

    let fec_frame = &fec_bytes[frame * BYTES_PER_FEC_FRAME..(frame + 1) * BYTES_PER_FEC_FRAME];
    let nofec_frame =
        &nofec_bytes[frame * BYTES_PER_NOFEC_FRAME..(frame + 1) * BYTES_PER_NOFEC_FRAME];

    let dibits = unpack_dibits(fec_frame);
    let c_tilde = deinterleave_fullrate(&dibits);
    let u_expected = unpack_nofec_info(nofec_frame);

    // Re-encode the no-FEC û_i to get the v̂_i the encoder produced
    // before PN modulation (only for FEC-protected vectors).
    let v_expected: [u32; 8] = [
        golay_23_12_encode(u_expected[0]),
        golay_23_12_encode(u_expected[1]),
        golay_23_12_encode(u_expected[2]),
        golay_23_12_encode(u_expected[3]),
        u32::from(hamming_15_11_encode(u_expected[4])),
        u32::from(hamming_15_11_encode(u_expected[5])),
        u32::from(hamming_15_11_encode(u_expected[6])),
        u32::from(u_expected[7]),
    ];

    // Recovered DVSI mask = c̃ ⊕ v̂. Our computed mask uses û₀ as seed.
    let dvsi_mask: [u32; 8] = std::array::from_fn(|i| c_tilde[i] ^ v_expected[i]);
    let our_mask = modulation_masks_fullrate(u_expected[0]);

    let widths = [23u8, 23, 23, 23, 15, 15, 15, 7];
    println!("frame {frame}, û₀ = 0x{:03x}", u_expected[0]);
    println!();
    println!("vec  width  c̃ (deinterleaved)  v̂ (re-encoded)   DVSI mask    our mask");
    for i in 0..8 {
        println!(
            "  {i}    {:>3}    0x{:06x}            0x{:06x}         0x{:06x}    0x{:06x}",
            widths[i], c_tilde[i], v_expected[i], dvsi_mask[i], our_mask[i]
        );
    }

    // Show the bit-reversed comparison for the PN-modulated vectors.
    println!();
    println!("PN-modulated vectors only (1..=6), bit-reversal check:");
    println!("vec  DVSI mask              our mask              our mask reversed");
    for i in 1..=6 {
        let n = widths[i] as u32;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let reversed = reverse_bits_within(our_mask[i], widths[i]);
        let mark_norm = if dvsi_mask[i] & mask == our_mask[i] & mask { '✓' } else { ' ' };
        let mark_rev = if dvsi_mask[i] & mask == reversed & mask { '✓' } else { ' ' };
        println!(
            "  {i}   0x{:06x}              0x{:06x} {}            0x{:06x} {}",
            dvsi_mask[i] & mask,
            our_mask[i] & mask,
            mark_norm,
            reversed & mask,
            mark_rev,
        );
    }
    Ok(())
}

fn reverse_bits_within(v: u32, n: u8) -> u32 {
    let mut r = 0u32;
    for k in 0..n {
        if (v >> k) & 1 == 1 {
            r |= 1u32 << (n - 1 - k);
        }
    }
    r
}

/// Locate the directory holding `<name>.pcm` and the `p25/`, `p25_nofec/`
/// subdirectories. The standard layout has them under `tv-std/tv/`; the
/// reference-channel set lives under `tv-rc/`.
fn vector_dir(root: &Path, rc: bool) -> PathBuf {
    if rc {
        root.join("tv-rc")
    } else {
        root.join("tv-std").join("tv")
    }
}

fn cmd_compare(root: &Path, name: &str, rc: bool, stop_on_first: bool) -> Result<()> {
    let dir = vector_dir(root, rc);
    let fec_path = dir.join("p25").join(format!("{name}.bit"));
    let nofec_path = dir.join("p25_nofec").join(format!("{name}.bit"));

    let fec_bytes = fs::read(&fec_path)
        .with_context(|| format!("read FEC vector {}", fec_path.display()))?;
    let nofec_bytes = fs::read(&nofec_path)
        .with_context(|| format!("read no-FEC vector {}", nofec_path.display()))?;

    if fec_bytes.len() % BYTES_PER_FEC_FRAME != 0 {
        return Err(anyhow!(
            "FEC file {} is {} bytes — not a multiple of {} (frame size)",
            fec_path.display(),
            fec_bytes.len(),
            BYTES_PER_FEC_FRAME
        ));
    }
    if nofec_bytes.len() % BYTES_PER_NOFEC_FRAME != 0 {
        return Err(anyhow!(
            "no-FEC file {} is {} bytes — not a multiple of {} (frame size)",
            nofec_path.display(),
            nofec_bytes.len(),
            BYTES_PER_NOFEC_FRAME
        ));
    }

    let n_fec_frames = fec_bytes.len() / BYTES_PER_FEC_FRAME;
    let n_nofec_frames = nofec_bytes.len() / BYTES_PER_NOFEC_FRAME;
    if n_fec_frames != n_nofec_frames {
        return Err(anyhow!(
            "frame count mismatch: FEC={n_fec_frames}, no-FEC={n_nofec_frames}"
        ));
    }
    let n_frames = n_fec_frames;

    println!("vector:    {name}{}", if rc { " (rc)" } else { "" });
    println!("frames:    {n_frames}");
    println!();

    let mut matches = 0usize;
    let mut mismatches = 0usize;
    let mut total_fec_errors = 0u32;

    for f in 0..n_frames {
        let fec_frame = &fec_bytes[f * BYTES_PER_FEC_FRAME..(f + 1) * BYTES_PER_FEC_FRAME];
        let nofec_frame =
            &nofec_bytes[f * BYTES_PER_NOFEC_FRAME..(f + 1) * BYTES_PER_NOFEC_FRAME];

        let dibits = unpack_dibits(fec_frame);
        let decoded: ImbeFrame = decode_fullrate_frame(&dibits);
        let expected = unpack_nofec_info(nofec_frame);
        total_fec_errors += u32::from(decoded.error_total());

        if decoded.info == expected {
            matches += 1;
        } else {
            mismatches += 1;
            if stop_on_first {
                report_mismatch(f, &decoded, &expected);
                println!();
                println!(
                    "result:    {matches} / {n_frames} frames matched ({mismatches} mismatches before stop)"
                );
                std::process::exit(1);
            }
        }
    }

    println!("matched:   {matches} / {n_frames}");
    println!("mismatched: {mismatches}");
    println!("avg FEC errors per frame: {:.3}", f64::from(total_fec_errors) / n_frames as f64);

    if mismatches == 0 {
        println!();
        println!("PASS — channel codec output matches DVSI no-FEC reference for every frame.");
        Ok(())
    } else {
        Err(anyhow!("{mismatches} frame(s) mismatched"))
    }
}

fn report_mismatch(frame_idx: usize, decoded: &ImbeFrame, expected: &[u16; 8]) {
    println!("FIRST DIVERGENCE at frame {frame_idx}");
    println!("  vector  width  decoded   expected  errors");
    for i in 0..8 {
        let mark = if decoded.info[i] == expected[i] { ' ' } else { '*' };
        println!(
            "  û_{i}    {:>3}    0x{:04x}    0x{:04x}    {}     {mark}",
            INFO_WIDTHS[i], decoded.info[i], expected[i], decoded.errors[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Bit unpacking
// ---------------------------------------------------------------------------

/// Read bits MSB-first from a byte slice. Returns each multi-bit read as a
/// `u16` with the *first-read* bit at the highest used position (i.e. a
/// 12-bit read returns a value where bit 11 is the first bit on the wire).
struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read(&mut self, n_bits: u8) -> u16 {
        let mut v = 0u16;
        for _ in 0..n_bits {
            let byte = self.bytes[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | u16::from(bit);
            self.pos += 1;
        }
        v
    }
}

/// Unpack 18 FEC bytes into 72 dibits in transmission order. Each dibit's
/// high bit is the bit transmitted first.
fn unpack_dibits(bytes: &[u8]) -> [u8; 72] {
    debug_assert_eq!(bytes.len(), BYTES_PER_FEC_FRAME);
    let mut r = BitReader::new(bytes);
    let mut out = [0u8; DIBITS_PER_FRAME];
    for i in 0..DIBITS_PER_FRAME {
        let hi = r.read(1) as u8;
        let lo = r.read(1) as u8;
        out[i] = (hi << 1) | lo;
    }
    out
}

/// Unpack 11 no-FEC bytes into the 8 info vectors `û₀..û₇`. The 88 bits
/// are concatenated MSB-first with the widths in [`INFO_WIDTHS`]. Each
/// returned `u16` has element-k-at-bit-k packing matching the rest of
/// the `blip25-mbe` API.
fn unpack_nofec_info(bytes: &[u8]) -> [u16; 8] {
    debug_assert_eq!(bytes.len(), BYTES_PER_NOFEC_FRAME);
    let mut r = BitReader::new(bytes);
    let mut info = [0u16; 8];
    for i in 0..8 {
        info[i] = r.read(INFO_WIDTHS[i]);
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_msb_first() {
        // 0b10110100 → reads 4 bits as 0b1011 = 11.
        let bytes = [0b1011_0100];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(4), 0b1011);
        assert_eq!(r.read(4), 0b0100);
    }

    #[test]
    fn bit_reader_crosses_byte_boundary() {
        // 0b1111_0000 0b1111_0000 → read 12 bits = 0b1111_0000_1111
        let bytes = [0b1111_0000, 0b1111_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(12), 0b1111_0000_1111);
    }

    #[test]
    fn unpack_dibits_first_byte() {
        // Byte 0xE4 = 0b11_10_01_00 → dibits 3, 2, 1, 0
        let bytes = [0u8; BYTES_PER_FEC_FRAME];
        let mut bytes = bytes;
        bytes[0] = 0b11_10_01_00;
        let d = unpack_dibits(&bytes);
        assert_eq!(d[0], 3);
        assert_eq!(d[1], 2);
        assert_eq!(d[2], 1);
        assert_eq!(d[3], 0);
    }

    #[test]
    fn unpack_nofec_widths_match_info_widths() {
        // Packing is sequential per INFO_WIDTHS = [12,12,12,12,11,11,11,7]
        // = 88 bits = 11 bytes.
        let total: u16 = INFO_WIDTHS.iter().map(|&w| u16::from(w)).sum();
        assert_eq!(total, 88);
        assert_eq!(BYTES_PER_NOFEC_FRAME, 11);
    }

    #[test]
    fn unpack_nofec_first_vector() {
        // First 12 bits = 0xABC means û₀ = 0xABC.
        // 0xABC = 1010_1011_1100 in 12 bits.
        // Packed MSB-first: byte 0 = 1010_1011, byte 1 high nibble = 1100.
        let mut bytes = [0u8; BYTES_PER_NOFEC_FRAME];
        bytes[0] = 0b1010_1011;
        bytes[1] = 0b1100_0000;
        let info = unpack_nofec_info(&bytes);
        assert_eq!(info[0], 0xABC);
    }
}
