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
    golay_23_12_decode, golay_23_12_encode, hamming_15_11_decode, hamming_15_11_encode,
};

use super::fec::{
    deinterleave_fullrate, interleave_fullrate, modulation_masks_fullrate,
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
pub struct ImbeFrame {
    /// Decoded info vectors û₀..û₇.
    pub info: [u16; 8],
    /// Per-vector FEC error count from decode.
    pub errors: [u8; 8],
}

impl ImbeFrame {
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
/// The returned `ImbeFrame` can be fed into a bit-deprioritizer +
/// dequantizer to obtain `MbeParams`.
pub fn decode_fullrate_frame(dibits: &[u8; 72]) -> ImbeFrame {
    let c = deinterleave_fullrate(dibits);

    // Step 2: Golay-decode c̃₀ directly (m̂₀ = 0).
    let d0 = golay_23_12_decode(c[0]);
    let u0 = d0.info;

    // Step 3: generate masks from the full 12-bit û₀.
    let masks = modulation_masks_fullrate(u0);

    // Step 4+5: demodulate c̃₁..c̃₆ and FEC-decode.
    let d1 = golay_23_12_decode(c[1] ^ masks[1]);
    let d2 = golay_23_12_decode(c[2] ^ masks[2]);
    let d3 = golay_23_12_decode(c[3] ^ masks[3]);
    let d4 = hamming_15_11_decode((c[4] ^ masks[4]) as u16);
    let d5 = hamming_15_11_decode((c[5] ^ masks[5]) as u16);
    let d6 = hamming_15_11_decode((c[6] ^ masks[6]) as u16);

    // Step 6: û₇ is uncoded (7 bits, also unmodulated).
    let u7 = (c[7] & 0x7F) as u16;

    ImbeFrame {
        info: [d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, u7],
        errors: [d0.errors, d1.errors, d2.errors, d3.errors, d4.errors, d5.errors, d6.errors, 0],
    }
}

/// Encode an 88-bit information layer into a full-rate IMBE frame
/// (72 dibit symbols).
///
/// Inverse of [`decode_fullrate_frame`]:
/// 1. **FEC-encode** û₀..û₃ with Golay, û₄..û₆ with Hamming.
/// 2. **Generate PN masks** from û₀.
/// 3. **Modulate** v̂₁..v̂₆ with m̂₁..m̂₆ (c̃₀ and c̃₇ pass through).
/// 4. **Interleave** the 8 code vectors into 72 dibits via Annex H.
///
/// `info` widths must match [`INFO_WIDTHS`]; bits above each vector's
/// declared width are masked off before encoding.
pub fn encode_fullrate_frame(info: &[u16; 8]) -> [u8; 72] {
    // Mask each input to its declared width so a caller that stuffs
    // stray high bits can't silently corrupt the frame.
    let u: [u16; 8] = core::array::from_fn(|i| {
        let w = INFO_WIDTHS[i] as u32;
        let m = if w == 16 { u16::MAX } else { ((1u32 << w) - 1) as u16 };
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

    let masks = modulation_masks_fullrate(u[0]);
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

    interleave_fullrate(&c)
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
            let m = if w == 16 { u16::MAX } else { ((1u32 << w) - 1) as u16 };
            out[i] = (state as u16) & m;
        }
        out
    }

    #[test]
    fn encode_decode_roundtrip_zero() {
        let u = [0u16; 8];
        let f = encode_fullrate_frame(&u);
        for (s, d) in f.iter().enumerate() {
            assert!(*d < 4, "symbol {s} is not a dibit");
        }
        let back = decode_fullrate_frame(&f);
        assert_eq!(back.info, u);
        assert_eq!(back.errors, [0u8; 8]);
    }

    #[test]
    fn encode_decode_roundtrip_sampled() {
        for seed in [1u32, 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 42] {
            let u = sample_info(seed);
            let f = encode_fullrate_frame(&u);
            let back = decode_fullrate_frame(&f);
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
        let f = encode_fullrate_frame(&u);
        let back = decode_fullrate_frame(&f);
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
        let clean = encode_fullrate_frame(&u);
        for bit in 0..23 {
            // We can't flip a single codeword bit directly on the dibit
            // stream without knowing Annex H — instead, deinterleave,
            // flip a bit, re-interleave, then decode and verify.
            let c = deinterleave_fullrate(&clean);
            let mut c_flipped = c;
            c_flipped[0] ^= 1u32 << bit;
            let dibits = interleave_fullrate(&c_flipped);
            let out = decode_fullrate_frame(&dibits);
            assert_eq!(out.info, u, "bit {bit} flip in c̃₀");
            assert_eq!(out.errors[0], 1, "bit {bit} flip in c̃₀");
        }
    }

    #[test]
    fn single_bit_flip_in_hamming_vector_is_corrected() {
        let u = sample_info(0x5A5A5A5A);
        let clean = encode_fullrate_frame(&u);
        // Flip any single bit inside c̃₄ (15 bits).
        for bit in 0..15 {
            let c = deinterleave_fullrate(&clean);
            let mut c_flipped = c;
            c_flipped[4] ^= 1u32 << bit;
            let dibits = interleave_fullrate(&c_flipped);
            let out = decode_fullrate_frame(&dibits);
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
        let clean = encode_fullrate_frame(&u);
        let c = deinterleave_fullrate(&clean);
        let mut c_flipped = c;
        c_flipped[7] ^= 1u32 << 3;
        let dibits = interleave_fullrate(&c_flipped);
        let out = decode_fullrate_frame(&dibits);
        assert_eq!(out.info[7], u[7] ^ 0b1000);
        assert_eq!(out.errors[7], 0);
    }

    #[test]
    fn triple_error_in_golay_vector_is_corrected() {
        // Golay can correct up to 3 bit errors; verify a specific
        // three-bit pattern in c̃₁ is handled correctly (and that PN
        // demodulation does not interfere).
        let u = sample_info(0x01020304);
        let clean = encode_fullrate_frame(&u);
        let c = deinterleave_fullrate(&clean);
        let mut c_flipped = c;
        c_flipped[1] ^= (1u32 << 0) | (1u32 << 7) | (1u32 << 19);
        let dibits = interleave_fullrate(&c_flipped);
        let out = decode_fullrate_frame(&dibits);
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
}
