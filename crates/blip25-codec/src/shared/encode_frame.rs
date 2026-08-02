//! Half-rate AMBE+2 FEC / framing ENCODER.
//!
//! This is the exact bit-level inverse of [`crate::frame`] decode. It takes
//! the 9 quantized parameters `b0..b8` (the same `[u16; 9]` array produced by
//! [`crate::tables::deprioritize`]) and produces:
//!
//! * `encode_r33` — the 9-byte (72-bit) on-air r33 frame, including FEC
//!   (extended Golay(24,12) on `u0`, Golay(23,12) + PN modulation on `u1`),
//!   Annex-S interleave, and MSB-first dibit packing.
//! * `encode_r34` — the 7-byte (49-bit) no-FEC r34 "info" frame: the same
//!   `û₀..û₃` prioritized bits as r33, serialized MSB-first and permuted by
//!   the reference [`R34_BIT_ORDER`] 3-way column interleave.
//!
//! ## Spec / repo references
//! * Parameter widths: `[7,5,5,9,7,5,4,4,3]` = 49 bits.
//! * Info-vector widths `u0..u3 = [12,12,11,14]`,
//!   TIA-102.BABA-A §2.4–§2.6.
//! * Bit prioritization (`b -> u`): inverse of `crate::tables::deprioritize`,
//!   driven by the same `AMBE_BIT_MAP`.
//! * FEC: §1.5.1 Golay; PN modulation: §2.6 (`crate::frame::modulation_mask_c1`).
//! * Interleave: Annex S (inverse of `crate::frame::deinterleave`).
//!
//! ## Acceptance
//! `frame::decode_bytes(encode_r33(b)).info` deprioritizes back to `b` for any
//! valid (in-range) `b`. Proven for random inputs in the test module below.

use crate::fec::{golay_23_12_encode, golay_24_12_encode};
use crate::frame::{modulation_mask_c1, DIBITS_PER_FRAME};
use crate::tables::{AMBE_BIT_MAP, AMBE_B_COUNT, ANNEX_S};

/// Bytes in an r33 (FEC) frame.
pub const R33_BYTES: usize = 9;
/// Bytes in an r34 (no-FEC, 49-bit) info frame.
pub(crate) const R34_BYTES: usize = 7;

/// Map the 9 quantized parameters `b0..b8` into the 4 info vectors
/// `u0..u3` (LSB-aligned). This is the exact inverse of
/// [`crate::tables::deprioritize`].
///
/// Bits of `b` beyond each parameter's nominal width are ignored, so any
/// caller passing in-range values gets a faithful round-trip.
pub(crate) fn prioritize(b: &[u16; AMBE_B_COUNT]) -> [u16; 4] {
    let mut u = [0u16; 4];
    for m in AMBE_BIT_MAP.iter() {
        let bit = (b[m.src_param as usize] >> m.src_bit) & 1;
        u[m.dst_vec as usize] |= bit << m.dst_bit;
    }
    u
}

/// Build the four FEC code vectors `c0..c3` from the info vectors `u0..u3`.
///
/// * `c0` = Golay(24,12) of `u0`            (24 bits)
/// * `c1` = Golay(23,12) of `u1` ^ PN-mask  (23 bits)
/// * `c2` = `u2`                            (11 bits, uncoded)
/// * `c3` = `u3`                            (14 bits, uncoded)
fn encode_codevectors(u: &[u16; 4]) -> [u32; 4] {
    let c0 = golay_24_12_encode(u[0]);
    let mask = modulation_mask_c1(u[0]);
    let c1 = golay_23_12_encode(u[1]) ^ mask;
    let c2 = u32::from(u[2]) & 0x7FF;
    let c3 = u32::from(u[3]) & 0x3FFF;
    [c0, c1, c2, c3]
}

/// Interleave the four code vectors into 36 dibits via Annex S. This is the
/// inverse of [`crate::frame::deinterleave`].
fn interleave(c: &[u32; 4]) -> [u8; DIBITS_PER_FRAME] {
    let mut dibits = [0u8; DIBITS_PER_FRAME];
    for (sym, slot) in dibits.iter_mut().enumerate() {
        let e = &ANNEX_S[sym];
        let hi = ((c[e.bit1_vec as usize] >> e.bit1_idx) & 1) as u8;
        let lo = ((c[e.bit0_vec as usize] >> e.bit0_idx) & 1) as u8;
        *slot = (hi << 1) | lo;
    }
    dibits
}

/// Pack 36 dibits into 9 bytes, MSB-first. Inverse of
/// [`crate::frame::bytes_to_dibits`]: `dibit[i]` high bit -> frame bit `2i`,
/// low bit -> frame bit `2i+1`.
fn dibits_to_bytes(dibits: &[u8; DIBITS_PER_FRAME]) -> [u8; R33_BYTES] {
    let mut bytes = [0u8; R33_BYTES];
    for (i, &d) in dibits.iter().enumerate() {
        let hi = (d >> 1) & 1;
        let lo = d & 1;
        let bit_hi = 2 * i;
        let bit_lo = 2 * i + 1;
        bytes[bit_hi / 8] |= hi << (7 - (bit_hi % 8));
        bytes[bit_lo / 8] |= lo << (7 - (bit_lo % 8));
    }
    bytes
}

/// Encode quantized parameters `b0..b8` into a 9-byte r33 (FEC) frame.
///
/// Exact inverse of [`crate::frame::decode_bytes`]: for any in-range `b`,
/// `crate::tables::deprioritize(&decode_bytes(&encode_r33(&b)).info) == b`.
pub fn encode_r33(b: &[u16; AMBE_B_COUNT]) -> [u8; R33_BYTES] {
    let u = prioritize(b);
    let c = encode_codevectors(&u);
    let dibits = interleave(&c);
    dibits_to_bytes(&dibits)
}

/// Widths of the four info vectors `û₀..û₃`, MSB-first, summing to 49.
const R34_INFO_WIDTHS: [usize; 4] = [12, 12, 11, 14];

/// R34 (half-rate no-FEC, 49-bit / 7-byte) bit serialization order.
///
/// Frame bit `j` (MSB-first) carries natural info bit `R34_BIT_ORDER[j]`, where
/// the natural order is `û₀(12)‖û₁(12)‖û₂(11)‖û₃(14)` MSB-first. the reference does
/// **not** use the naive sequential layout: it emits a 3-way column interleave
/// (rows of 18/18/13 bits) — `0,18,36, 1,19,37, …`. Bits 49..55 are zero pad.
///
/// This must match the wire, not merely decode self-consistently: no-FEC
/// half-rate is what an NXDN/Fusion sink and the AMBE-3000R serial host format
/// expect.
///
/// **Provenance.** Empirically derived from the reference `r33`↔`r34` vector pairs, not
/// from spec text, and the tooling that derived it no longer exists — this
/// table is irreplaceable. Do not "correct" it into a sequential layout.
///
/// Kept byte-identical to `blip25_mbe::rate33::frame::R34_BIT_ORDER`; a test
/// asserts the two agree.
pub const R34_BIT_ORDER: [u8; 49] = [
    0, 18, 36, 1, 19, 37, 2, 20, 38, 3, 21, 39, 4, 22, 40, 5, 23, 41, 6, 24, 42, 7, 25, 43, 8, 26,
    44, 9, 27, 45, 10, 28, 46, 11, 29, 47, 12, 30, 48, 13, 31, 14, 32, 15, 33, 16, 34, 17, 35,
];

/// Encode quantized parameters `b0..b8` into a 7-byte r34 (no-FEC) info frame.
///
/// The parameters are first prioritized into `û₀..û₃` (the same mapping the
/// FEC-bearing r33 frame uses), serialized MSB-first, then permuted by
/// [`R34_BIT_ORDER`]. The 7 trailing bits are zero padding.
pub fn encode_r34(b: &[u16; AMBE_B_COUNT]) -> [u8; R34_BYTES] {
    let u = prioritize(b);
    let mut natural = [0u8; 49];
    let mut pos = 0usize;
    for (i, &width) in R34_INFO_WIDTHS.iter().enumerate() {
        for k in 0..width {
            natural[pos] = ((u[i] >> (width - 1 - k)) & 1) as u8;
            pos += 1;
        }
    }
    debug_assert_eq!(pos, 49);

    let mut bytes = [0u8; R34_BYTES];
    for (j, &src) in R34_BIT_ORDER.iter().enumerate() {
        bytes[j / 8] |= natural[src as usize] << (7 - (j % 8));
    }
    bytes
}

/// Decode a 7-byte r34 info frame back into `b0..b8`. Exact inverse of
/// [`encode_r34`].
pub fn decode_r34(bytes: &[u8; R34_BYTES]) -> [u16; AMBE_B_COUNT] {
    let mut natural = [0u8; 49];
    for (j, &dst) in R34_BIT_ORDER.iter().enumerate() {
        natural[dst as usize] = (bytes[j / 8] >> (7 - (j % 8))) & 1;
    }

    let mut u = [0u16; 4];
    let mut pos = 0usize;
    for (i, &width) in R34_INFO_WIDTHS.iter().enumerate() {
        let mut val = 0u16;
        for _ in 0..width {
            val = (val << 1) | u16::from(natural[pos]);
            pos += 1;
        }
        u[i] = val;
    }
    crate::tables::deprioritize(&u)
}
