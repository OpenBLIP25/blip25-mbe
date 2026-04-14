//! Forward-error-correction primitives shared across wire formats.
//!
//! The P25 IMBE wire uses Golay(23,12) and Hamming(15,11) per
//! TIA-102.BABA-A §§1.5.1–1.5.2. This module implements both codes from
//! the systematic generator matrices given in that spec and exposes
//! `encode`/`decode` functions callable by the wire-format modules.
//!
//! Wire-format-specific FEC layout (which bits are covered by which code,
//! how they are interleaved, how PN demodulation fits in) lives in the
//! wire module — for IMBE, see [`crate::imbe_frames::fec`].
//!
//! Both codes are perfect (every 11-bit syndrome for Golay, every 4-bit
//! syndrome for Hamming maps to a unique minimum-weight error pattern),
//! so both decoders always produce a result. The returned `errors` field
//! counts how many bits were flipped to reach a valid codeword — that
//! equals the real number of channel errors only up to the code's
//! correction capacity (3 for Golay, 1 for Hamming). A received word
//! with more errors than that will miscorrect to the nearest codeword,
//! which is the accepted behaviour of bounded-distance decoding on a
//! perfect code.

/// Decoded result of a systematic FEC codeword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FecDecoded {
    /// Information bits, LSB-aligned in a `u16`.
    pub info: u16,
    /// Number of bit errors corrected, or [`u8::MAX`] if uncorrectable.
    pub errors: u8,
}


// ---------------------------------------------------------------------------
// Golay (23, 12), d = 7
// ---------------------------------------------------------------------------

/// Generator matrix for the [23, 12] Golay code in systematic form `[I₁₂ | P]`.
///
/// Source: TIA-102.BABA-A §1.5.1 (Section 7.3, Equation 81).
/// Generator polynomial `g(x) = x¹¹ + x⁹ + x⁷ + x⁶ + x⁵ + x + 1` (octal 6265).
///
/// Bit layout within each 23-bit row:
/// * bits 22..11 — identity portion, row `i` has a 1 at bit `22 - i`
/// * bits 10..0  — parity portion, bit 10 = x⁰ coefficient, bit 0 = x¹⁰
const GOLAY_23_12_GEN: [u32; 12] = [
    0b100000000000_11000111010,
    0b010000000000_01100011101,
    0b001000000000_11110110100,
    0b000100000000_01111011010,
    0b000010000000_00111101101,
    0b000001000000_11011001100,
    0b000000100000_01101100110,
    0b000000010000_00110110011,
    0b000000001000_11011100011,
    0b000000000100_10101001011,
    0b000000000010_10010011111,
    0b000000000001_10001110101,
];

/// Encode 12 information bits into a 23-bit Golay codeword.
///
/// `info` is interpreted as a 12-bit value (bits 11..0 used, higher bits
/// ignored). The most significant information bit is `info[11]` and maps
/// to codeword bit 22.
pub fn golay_23_12_encode(info: u16) -> u32 {
    let mut cw = 0u32;
    let info = u32::from(info) & 0xFFF;
    for i in 0..12 {
        if (info >> (11 - i)) & 1 == 1 {
            cw ^= GOLAY_23_12_GEN[i];
        }
    }
    cw
}

/// Compute the 11-bit syndrome of a 23-bit Golay codeword.
///
/// The parity-check matrix `H = [Pᵀ | I₁₁]` is derived from the generator
/// above; columns 0..11 of `H` are the transpose of the parity columns of
/// `G`, columns 12..22 are the identity. We compute the syndrome by
/// accumulating the generator's parity portion for each set information
/// bit and XORing with the received parity bits.
fn golay_23_12_syndrome(codeword: u32) -> u16 {
    let cw = codeword & 0x7F_FFFF;
    let info = (cw >> 11) & 0xFFF;
    let received_parity = cw & 0x7FF;
    let mut reconstructed_parity = 0u32;
    for i in 0..12 {
        if (info >> (11 - i)) & 1 == 1 {
            reconstructed_parity ^= GOLAY_23_12_GEN[i] & 0x7FF;
        }
    }
    (received_parity ^ reconstructed_parity) as u16
}

/// Syndrome-to-error-pattern table for Golay [23, 12].
///
/// The code is perfect: the 2¹¹ = 2048 syndromes are in bijection with
/// the 2¹¹ error patterns of weight ≤ 3 (1 + 23 + 253 + 1771). We build
/// the table once by enumerating those patterns in order of increasing
/// weight and recording each one's syndrome. Syndrome 0 maps to the
/// zero-error pattern; all other entries hold a pattern of weight 1, 2,
/// or 3.
fn golay_23_12_syndrome_table() -> &'static [u32; 2048] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 2048]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 2048];
        // Weight 1
        for a in 0..23 {
            let pat = 1u32 << a;
            table[golay_23_12_syndrome(pat) as usize] = pat;
        }
        // Weight 2
        for a in 0..23 {
            for b in (a + 1)..23 {
                let pat = (1u32 << a) | (1u32 << b);
                table[golay_23_12_syndrome(pat) as usize] = pat;
            }
        }
        // Weight 3
        for a in 0..23 {
            for b in (a + 1)..23 {
                for c in (b + 1)..23 {
                    let pat = (1u32 << a) | (1u32 << b) | (1u32 << c);
                    table[golay_23_12_syndrome(pat) as usize] = pat;
                }
            }
        }
        table
    })
}

/// Decode a 23-bit Golay codeword, correcting up to 3 bit errors.
///
/// The [23, 12] Golay code is perfect, so every syndrome has a unique
/// weight-≤3 error pattern. We precompute that syndrome → pattern map
/// once and apply it to each decode. Against a received word with more
/// than 3 bit errors the decoder produces a wrong but deterministic
/// result — a property of any bounded-distance decoder on a perfect code.
pub fn golay_23_12_decode(codeword: u32) -> FecDecoded {
    let cw = codeword & 0x7F_FFFF;
    let syndrome = golay_23_12_syndrome(cw);
    let pattern = golay_23_12_syndrome_table()[syndrome as usize];
    let corrected = cw ^ pattern;
    FecDecoded {
        info: ((corrected >> 11) & 0xFFF) as u16,
        errors: pattern.count_ones() as u8,
    }
}

// ---------------------------------------------------------------------------
// Hamming (15, 11), d = 3
// ---------------------------------------------------------------------------

/// Parity rows for the [15, 11] Hamming code in systematic form `[I₁₁ | P]`.
///
/// Source: TIA-102.BABA-A §1.5.2. Each entry is a 4-bit parity word in
/// which the MSB is the first parity column (p₀) and the LSB is p₃.
const HAMMING_15_11_PARITY: [u8; 11] = [
    0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b0111, 0b0110,
    0b0101, 0b0011,
];

/// Encode 11 information bits into a 15-bit Hamming codeword.
///
/// `info` is interpreted as an 11-bit value. The most significant
/// information bit is `info[10]` and maps to codeword bit 14. The 4
/// parity bits occupy codeword bits 3..0 (MSB = p₀).
pub fn hamming_15_11_encode(info: u16) -> u16 {
    let info = info & 0x7FF;
    let mut parity = 0u16;
    for i in 0..11 {
        if (info >> (10 - i)) & 1 == 1 {
            parity ^= u16::from(HAMMING_15_11_PARITY[i]);
        }
    }
    (info << 4) | (parity & 0xF)
}

/// Decode a 15-bit Hamming codeword, correcting up to 1 bit error.
///
/// The 4-bit syndrome locates the error:
/// * zero syndrome → no error, `errors = 0`
/// * syndrome equals one of the 11 parity rows → that info bit is flipped
///   and `errors = 1`
/// * syndrome is a unit vector (`0001`, `0010`, `0100`, `1000`) → the
///   corresponding parity bit is in error; no info bit is changed and
///   `errors = 1`
///
/// These 16 cases exhaust the syndrome space — [15, 11] Hamming is a
/// perfect code — so no other branches are reachable.
pub fn hamming_15_11_decode(codeword: u16) -> FecDecoded {
    let cw = codeword & 0x7FFF;
    let info = (cw >> 4) & 0x7FF;
    let received_parity = cw & 0xF;

    let mut reconstructed_parity = 0u16;
    for i in 0..11 {
        if (info >> (10 - i)) & 1 == 1 {
            reconstructed_parity ^= u16::from(HAMMING_15_11_PARITY[i]);
        }
    }
    let syndrome = (received_parity ^ reconstructed_parity) & 0xF;

    if syndrome == 0 {
        return FecDecoded { info, errors: 0 };
    }

    for i in 0..11 {
        if u16::from(HAMMING_15_11_PARITY[i]) == syndrome {
            let corrected = info ^ (1 << (10 - i));
            return FecDecoded { info: corrected, errors: 1 };
        }
    }

    // Parity-bit error: syndrome is a unit vector.
    FecDecoded { info, errors: 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golay_roundtrip_all_info_words() {
        for info in 0u16..(1 << 12) {
            let cw = golay_23_12_encode(info);
            assert!(cw >> 23 == 0, "encoded word must fit in 23 bits");
            let d = golay_23_12_decode(cw);
            assert_eq!(d.info, info);
            assert_eq!(d.errors, 0);
        }
    }

    #[test]
    fn golay_minimum_distance_is_seven() {
        // Every nonzero codeword must have weight at least 7.
        for info in 1u16..(1 << 12) {
            let cw = golay_23_12_encode(info);
            assert!(
                cw.count_ones() >= 7,
                "codeword for info 0x{:03x} has weight {}",
                info,
                cw.count_ones()
            );
        }
    }

    #[test]
    fn golay_corrects_up_to_three_errors() {
        // Sample a handful of info words and systematically flip up to 3 bits.
        for info in [0x000u16, 0x001, 0x555, 0xAAA, 0xFFF] {
            let cw = golay_23_12_encode(info);
            for a in 0..23 {
                for b in a..23 {
                    for c in b..23 {
                        let mut err = 1u32 << a;
                        if b != a {
                            err |= 1u32 << b;
                        }
                        if c != b {
                            err |= 1u32 << c;
                        }
                        let received = cw ^ err;
                        let d = golay_23_12_decode(received);
                        assert_eq!(d.info, info, "info={info:03x} err={err:07x}");
                        assert_eq!(d.errors, err.count_ones() as u8);
                    }
                }
            }
        }
    }

    #[test]
    fn hamming_roundtrip_all_info_words() {
        for info in 0u16..(1 << 11) {
            let cw = hamming_15_11_encode(info);
            assert!(cw >> 15 == 0);
            let d = hamming_15_11_decode(cw);
            assert_eq!(d.info, info);
            assert_eq!(d.errors, 0);
        }
    }

    #[test]
    fn hamming_minimum_distance_is_three() {
        for info in 1u16..(1 << 11) {
            let cw = hamming_15_11_encode(info);
            assert!(cw.count_ones() >= 3);
        }
    }

    #[test]
    fn hamming_corrects_any_single_error() {
        for info in [0x000u16, 0x001, 0x555, 0x2AA, 0x7FF] {
            let cw = hamming_15_11_encode(info);
            for bit in 0..15 {
                let received = cw ^ (1u16 << bit);
                let d = hamming_15_11_decode(received);
                assert_eq!(d.info, info, "info={info:03x} bit={bit}");
                assert_eq!(d.errors, 1);
            }
        }
    }

    #[test]
    fn golay_decodes_every_syndrome() {
        // Perfect-code property: every 11-bit syndrome has a weight-≤3
        // preimage. Verify by decoding every possible received word with
        // zero info bits — each syndrome must resolve to some pattern.
        let zero_cw = golay_23_12_encode(0);
        for syndrome_bits in 0u32..(1 << 11) {
            let received = zero_cw ^ syndrome_bits;
            let d = golay_23_12_decode(received);
            assert!(d.errors <= 3);
        }
    }
}
