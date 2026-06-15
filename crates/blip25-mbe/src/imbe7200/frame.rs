//! Full-rate IMBE frame — 144 bits carrying 88 parameter bits at 7200 bps.
//!
//! Per TIA-102.BABA-A §§1.2–1.7. This module composes the deinterleave
//! / FEC / PN primitives from [`super::fec`] into a single channel
//! codec: air-interface dibits ↔ the 88-bit information layer that
//! sits between the physical channel and the parameter domain.
//!
//! Exit point is BABA-A §8.1: eight info vectors `û₀..û₇` with widths
//! 12, 12, 12, 12, 11, 11, 11, 7 bits respectively (sum = 88). Bit
//! deprioritization and dequantization — which turn those 88 bits into
//! `MbeParams` — live downstream.

use crate::fec::{
    golay_23_12_decode, golay_23_12_decode_soft, golay_23_12_encode, hamming_15_11_decode,
    hamming_15_11_decode_soft, hamming_15_11_encode,
};

use super::fec::{
    deinterleave, interleave, modulation_masks, soft_deinterleave, soft_demodulate_vector,
    SOFT_BITS,
};

/// Width in bits of each info vector `û₀..û₇`. BABA-A §1.2.
pub const INFO_WIDTHS: [u8; 8] = [12, 12, 12, 12, 11, 11, 11, 7];

/// Width in bits of each code vector `c̃₀..c̃₇`. BABA-A §1.2.
pub const CODE_WIDTHS: [u8; 8] = [23, 23, 23, 23, 15, 15, 15, 7];

/// Sum of info widths — must equal 88.
pub const INFO_BITS_TOTAL: u16 = 88;

/// Decoded 88-bit information layer of a full-rate IMBE frame.
///
/// Each entry in `info` is the decoded vector `û_i`, LSB-aligned in a
/// `u16` (element `k` at bit `k`), with width given by [`INFO_WIDTHS`].
/// `errors[i]` is the per-vector FEC error count: 0–3 for Golay-coded
/// vectors (û₀..û₃), 0–1 for Hamming-coded vectors (û₄..û₆), and always
/// 0 for the uncoded û₇.
///
/// This struct sits exactly at the BABA-A §8.1 handoff between the
/// wire/channel layer and the parameter-decoding layer. It carries
/// nothing about pitch, L, or gain yet — that interpretation happens
/// downstream in bit deprioritization + dequantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Decoded info vectors û₀..û₇.
    pub info: [u16; 8],
    /// Per-vector FEC error count from decode.
    pub errors: [u8; 8],
}

impl Frame {
    /// Total error count across all vectors, used by BABA-A §4.1 error
    /// assessment (frame repeat / mute triggers).
    pub fn error_total(&self) -> u16 {
        self.errors.iter().map(|&e| e as u16).sum()
    }
}

/// Decode a full-rate IMBE frame from 72 air-interface dibit symbols
/// into the 88-bit information layer.
///
/// Pipeline (BABA-A §4.1):
/// 1. **Deinterleave** (Annex H): 72 dibits → 8 code vectors `c̃₀..c̃₇`.
/// 2. **Golay-decode `c̃₀`** (not PN-modulated since `m̂₀ = 0`) to
///    recover `û₀`, the seed for the PN generator.
/// 3. **Generate PN masks** from the full 12-bit `û₀`.
/// 4. **Demodulate** `c̃₁..c̃₆` with masks `m̂₁..m̂₆` → `v̂₁..v̂₆`.
/// 5. **FEC-decode** `v̂₁..v̂₃` with Golay, `v̂₄..v̂₆` with Hamming.
/// 6. **Uncoded passthrough**: `û₇ = c̃₇` (`m̂₇ = 0` and no FEC).
///
/// The returned `Frame` can be fed into a bit-deprioritizer +
/// dequantizer to obtain `MbeParams`.
pub fn decode_frame(dibits: &[u8; 72]) -> Frame {
    let c = deinterleave(dibits);

    // Step 2: Golay-decode c̃₀ directly (m̂₀ = 0).
    let d0 = golay_23_12_decode(c[0]);
    let u0 = d0.info;

    // Step 3: generate masks from the full 12-bit û₀.
    let masks = modulation_masks(u0);

    // Step 4+5: demodulate c̃₁..c̃₆ and FEC-decode.
    let d1 = golay_23_12_decode(c[1] ^ masks[1]);
    let d2 = golay_23_12_decode(c[2] ^ masks[2]);
    let d3 = golay_23_12_decode(c[3] ^ masks[3]);
    let d4 = hamming_15_11_decode((c[4] ^ masks[4]) as u16);
    let d5 = hamming_15_11_decode((c[5] ^ masks[5]) as u16);
    let d6 = hamming_15_11_decode((c[6] ^ masks[6]) as u16);

    // Step 6: û₇ is uncoded (7 bits, also unmodulated).
    let u7 = (c[7] & 0x7F) as u16;

    Frame {
        info: [
            d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, u7,
        ],
        errors: [
            d0.errors, d1.errors, d2.errors, d3.errors, d4.errors, d5.errors, d6.errors, 0,
        ],
    }
}

/// Soft-decision decode of a full-rate IMBE frame from 144 soft bits.
///
/// Input layout matches the C4FM soft-bit convention in p25-decoder:
/// one `i8` per bit, sign = hard decision (`>0` → 1, `≤0` → 0),
/// magnitude = confidence / LLR. Bits are ordered
/// `[hi_dibit_0, lo_dibit_0, hi_dibit_1, lo_dibit_1, …]` — matching
/// the high/low bit pairing the hard path reads from the dibit stream.
///
/// Pipeline mirrors [`decode_frame`]:
/// 1. Soft-deinterleave 144 bits → 8 MSB-first soft code vectors.
/// 2. Soft-Golay-decode `c̃₀` (no PN) → `û₀`.
/// 3. Compute hard masks from `û₀`; soft-demodulate `c̃₁..c̃₆` (mask
///    bit 1 flips the soft bit's sign, preserving magnitude).
/// 4. Soft-Golay-decode `c̃₁..c̃₃`, soft-Hamming-decode `c̃₄..c̃₆`.
/// 5. `û₇` hard-projects from the uncoded 7-bit soft vector (no FEC).
///
/// The `errors` field on the returned [`Frame`] reports the Chase-II
/// winner's hard error count on each FEC-coded vector. Across the
/// soft path, valid soft inputs can recover frames with more channel
/// errors than the bounded-distance hard decoder admits — the typical
/// ~2 dB coding gain over hard Golay and the usual single-bit gain
/// over hard Hamming.
pub fn decode_frame_soft(soft: &[i8; SOFT_BITS]) -> Frame {
    let mut c = soft_deinterleave(soft);

    // Step 2: soft-Golay-decode c̃₀ (no PN modulation).
    let d0 = golay_23_12_decode_soft(&c.golay[0]);
    let u0 = d0.info;

    // Step 3: derive hard masks and soft-demodulate c̃₁..c̃₆.
    let masks = modulation_masks(u0);
    soft_demodulate_vector(&mut c.golay[1], masks[1]);
    soft_demodulate_vector(&mut c.golay[2], masks[2]);
    soft_demodulate_vector(&mut c.golay[3], masks[3]);
    soft_demodulate_vector(&mut c.hamming[0], masks[4]);
    soft_demodulate_vector(&mut c.hamming[1], masks[5]);
    soft_demodulate_vector(&mut c.hamming[2], masks[6]);

    // Step 4: soft FEC decode.
    let d1 = golay_23_12_decode_soft(&c.golay[1]);
    let d2 = golay_23_12_decode_soft(&c.golay[2]);
    let d3 = golay_23_12_decode_soft(&c.golay[3]);
    let d4 = hamming_15_11_decode_soft(&c.hamming[0]);
    let d5 = hamming_15_11_decode_soft(&c.hamming[1]);
    let d6 = hamming_15_11_decode_soft(&c.hamming[2]);

    // Step 5: û₇ — project the 7-bit uncoded soft vector to hard bits,
    // MSB-first in element `k` at bit `k` order.
    let mut u7 = 0u16;
    for (i, &s) in c.uncoded.iter().enumerate() {
        if s > 0 {
            u7 |= 1u16 << (7 - 1 - i);
        }
    }

    Frame {
        info: [
            d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, u7,
        ],
        errors: [
            d0.errors, d1.errors, d2.errors, d3.errors, d4.errors, d5.errors, d6.errors, 0,
        ],
    }
}

/// Encode an 88-bit information layer into a full-rate IMBE frame
/// (72 dibit symbols).
///
/// Inverse of [`decode_frame`]:
/// 1. **FEC-encode** û₀..û₃ with Golay, û₄..û₆ with Hamming.
/// 2. **Generate PN masks** from û₀.
/// 3. **Modulate** v̂₁..v̂₆ with m̂₁..m̂₆ (c̃₀ and c̃₇ pass through).
/// 4. **Interleave** the 8 code vectors into 72 dibits via Annex H.
///
/// `info` widths must match [`INFO_WIDTHS`]; bits above each vector's
/// declared width are masked off before encoding.
pub fn encode_frame(info: &[u16; 8]) -> [u8; 72] {
    // Mask each input to its declared width so a caller that stuffs
    // stray high bits can't silently corrupt the frame.
    let u: [u16; 8] = core::array::from_fn(|i| {
        let w = INFO_WIDTHS[i] as u32;
        let m = if w == 16 {
            u16::MAX
        } else {
            ((1u32 << w) - 1) as u16
        };
        info[i] & m
    });

    let v0 = golay_23_12_encode(u[0]);
    let v1 = golay_23_12_encode(u[1]);
    let v2 = golay_23_12_encode(u[2]);
    let v3 = golay_23_12_encode(u[3]);
    let v4 = u32::from(hamming_15_11_encode(u[4]));
    let v5 = u32::from(hamming_15_11_encode(u[5]));
    let v6 = u32::from(hamming_15_11_encode(u[6]));
    let v7 = u32::from(u[7]);

    let masks = modulation_masks(u[0]);
    let c = [
        v0, // m̂₀ = 0
        v1 ^ masks[1],
        v2 ^ masks[2],
        v3 ^ masks[3],
        v4 ^ masks[4],
        v5 ^ masks[5],
        v6 ^ masks[6],
        v7, // m̂₇ = 0
    ];

    interleave(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Produce an arbitrary-but-valid info bundle with each vector
    /// masked to its declared width.
    fn sample_info(seed: u32) -> [u16; 8] {
        let mut state = seed;
        let mut out = [0u16; 8];
        for i in 0..8 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = INFO_WIDTHS[i] as u32;
            let m = if w == 16 {
                u16::MAX
            } else {
                ((1u32 << w) - 1) as u16
            };
            out[i] = (state as u16) & m;
        }
        out
    }

    #[test]
    fn encode_decode_roundtrip_zero() {
        let u = [0u16; 8];
        let f = encode_frame(&u);
        for (s, d) in f.iter().enumerate() {
            assert!(*d < 4, "symbol {s} is not a dibit");
        }
        let back = decode_frame(&f);
        assert_eq!(back.info, u);
        assert_eq!(back.errors, [0u8; 8]);
    }

    #[test]
    fn encode_decode_roundtrip_sampled() {
        for seed in [1u32, 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 42] {
            let u = sample_info(seed);
            let f = encode_frame(&u);
            let back = decode_frame(&f);
            assert_eq!(back.info, u, "seed 0x{seed:08x}");
            assert_eq!(back.errors, [0u8; 8], "seed 0x{seed:08x}");
        }
    }

    #[test]
    fn high_bits_are_masked_off() {
        // Stuffing bits beyond the declared widths must not corrupt the
        // frame — they get masked before FEC encoding.
        let mut u = [0u16; 8];
        for i in 0..8 {
            u[i] = 0xFFFF; // all ones; high bits must be ignored
        }
        let f = encode_frame(&u);
        let back = decode_frame(&f);
        for i in 0..8 {
            let w = INFO_WIDTHS[i] as u32;
            let m = ((1u32 << w) - 1) as u16;
            assert_eq!(back.info[i], m, "vector {i} width mask failed");
        }
    }

    #[test]
    fn single_bit_flip_in_golay_vector_is_corrected() {
        // Flipping one bit inside any of c̃₀..c̃₃'s 23-bit range should
        // be corrected (Golay handles up to 3 errors).
        let u = sample_info(0xA5A5A5A5);
        let clean = encode_frame(&u);
        for bit in 0..23 {
            // We can't flip a single codeword bit directly on the dibit
            // stream without knowing Annex H — instead, deinterleave,
            // flip a bit, re-interleave, then decode and verify.
            let c = deinterleave(&clean);
            let mut c_flipped = c;
            c_flipped[0] ^= 1u32 << bit;
            let dibits = interleave(&c_flipped);
            let out = decode_frame(&dibits);
            assert_eq!(out.info, u, "bit {bit} flip in c̃₀");
            assert_eq!(out.errors[0], 1, "bit {bit} flip in c̃₀");
        }
    }

    #[test]
    fn single_bit_flip_in_hamming_vector_is_corrected() {
        let u = sample_info(0x5A5A5A5A);
        let clean = encode_frame(&u);
        // Flip any single bit inside c̃₄ (15 bits).
        for bit in 0..15 {
            let c = deinterleave(&clean);
            let mut c_flipped = c;
            c_flipped[4] ^= 1u32 << bit;
            let dibits = interleave(&c_flipped);
            let out = decode_frame(&dibits);
            assert_eq!(out.info, u, "bit {bit} flip in c̃₄");
            assert_eq!(out.errors[4], 1, "bit {bit} flip in c̃₄");
        }
    }

    #[test]
    fn uncoded_vector_has_no_error_correction() {
        // û₇ is uncoded: any bit flip in c̃₇ passes through to the
        // decoded info (and errors[7] stays zero because there is no
        // FEC to report errors against).
        let u = sample_info(0xFACEFEED);
        let clean = encode_frame(&u);
        let c = deinterleave(&clean);
        let mut c_flipped = c;
        c_flipped[7] ^= 1u32 << 3;
        let dibits = interleave(&c_flipped);
        let out = decode_frame(&dibits);
        assert_eq!(out.info[7], u[7] ^ 0b1000);
        assert_eq!(out.errors[7], 0);
    }

    #[test]
    fn triple_error_in_golay_vector_is_corrected() {
        // Golay can correct up to 3 bit errors; verify a specific
        // three-bit pattern in c̃₁ is handled correctly (and that PN
        // demodulation does not interfere).
        let u = sample_info(0x01020304);
        let clean = encode_frame(&u);
        let c = deinterleave(&clean);
        let mut c_flipped = c;
        c_flipped[1] ^= (1u32 << 0) | (1u32 << 7) | (1u32 << 19);
        let dibits = interleave(&c_flipped);
        let out = decode_frame(&dibits);
        assert_eq!(out.info, u);
        assert_eq!(out.errors[1], 3);
        assert_eq!(out.error_total(), 3);
    }

    #[test]
    fn widths_sum_to_88_and_144() {
        assert_eq!(INFO_WIDTHS.iter().map(|&w| w as u16).sum::<u16>(), 88);
        assert_eq!(CODE_WIDTHS.iter().map(|&w| w as u16).sum::<u16>(), 144);
        assert_eq!(INFO_BITS_TOTAL, 88);
    }

    // ---- Soft-decision path (Gap B) -----------------------------------

    /// Inflate a 72-dibit hard stream into 144 soft bits at uniform
    /// confidence. Dibit order: `hi = (d >> 1) & 1`, `lo = d & 1`.
    fn dibits_to_soft(dibits: &[u8; 72], confidence: i8) -> [i8; 144] {
        let mut out = [0i8; 144];
        for (sym, &d) in dibits.iter().enumerate() {
            let hi = (d >> 1) & 1;
            let lo = d & 1;
            out[2 * sym] = if hi == 1 { confidence } else { -confidence };
            out[2 * sym + 1] = if lo == 1 { confidence } else { -confidence };
        }
        out
    }

    #[test]
    fn soft_decode_matches_hard_on_clean_input() {
        for seed in [0u32, 1, 0xDEADBEEF, 0xCAFEBABE, 0x12345678] {
            let u = if seed == 0 {
                [0u16; 8]
            } else {
                sample_info(seed)
            };
            let dibits = encode_frame(&u);
            let soft = dibits_to_soft(&dibits, 120);
            let soft_frame = decode_frame_soft(&soft);
            let hard_frame = decode_frame(&dibits);
            assert_eq!(soft_frame.info, hard_frame.info, "seed 0x{seed:08x}");
            assert_eq!(soft_frame.info, u, "seed 0x{seed:08x}");
            assert_eq!(soft_frame.errors, [0u8; 8]);
        }
    }

    #[test]
    fn soft_decode_recovers_weak_bit_errors_on_golay_vector() {
        // Four weak-bit flips inside c̃₁ after demodulation — beyond
        // Golay-23's bounded-distance capacity (t = 3). Hard decode
        // miscorrects; soft should still recover.
        let u = sample_info(0x01020304);
        let dibits = encode_frame(&u);
        let mut soft = dibits_to_soft(&dibits, 120);
        // Locate the 4 bits that belong to c₁'s MSB-first positions
        // 0, 4, 10, 18 and lower their magnitudes to 4 (well below
        // confidence 120) while flipping their signs.
        //
        // We perturb the soft stream directly rather than the dibit
        // stream: find each target soft index via the Annex H reverse
        // lookup and weaken+flip it.
        use super::super::fec::ANNEX_H;
        let targets = [(1u8, 0u8), (1, 4), (1, 10), (1, 18)]; // (vec_idx, msb_pos)
        for (vi, msb_pos) in targets {
            let lsb_pos = CODE_WIDTHS[vi as usize] - 1 - msb_pos;
            // Find the dibit symbol + hi/lo role covering (vi, lsb_pos).
            let (sym, is_hi) = (0..ANNEX_H.len())
                .find_map(|s| {
                    let e = ANNEX_H[s];
                    if e.bit1_vec == vi && e.bit1_idx == lsb_pos {
                        Some((s, true))
                    } else if e.bit0_vec == vi && e.bit0_idx == lsb_pos {
                        Some((s, false))
                    } else {
                        None
                    }
                })
                .expect("annex H covers every (vec, idx)");
            let soft_idx = 2 * sym + if is_hi { 0 } else { 1 };
            soft[soft_idx] = -soft[soft_idx].signum() * 4;
        }
        let frame = decode_frame_soft(&soft);
        assert_eq!(frame.info, u, "soft recovered 4-error c̃₁");
    }
}
