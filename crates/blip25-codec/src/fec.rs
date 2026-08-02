// index loops are deliberate: the index is the bin/harmonic/tap/band/bit number
#![allow(clippy::needless_range_loop)]

//! Golay(23,12) + extended Golay(24,12) hard-decision FEC.
//!
//! TIA-102.BABA-A §1.5.1. Hard decoders only — the reference ground-truth vectors
//! are hard bit streams.

/// Decoded result of a systematic FEC codeword.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FecDecoded {
    /// Information bits, LSB-aligned in a `u16`.
    pub info: u16,
    /// Number of bit errors corrected, or [`u8::MAX`] if uncorrectable.
    pub errors: u8,
}

/// Generator matrix for the [23, 12] Golay code in systematic form `[I12 | P]`.
/// Source: TIA-102.BABA-A §1.5.1 (Eq. 81), g(x) = octal 6265.
// digit grouping is the 12|11 systematic [I12 | P] split, not nibbles
#[allow(clippy::unusual_byte_groupings)]
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
pub(crate) fn golay_23_12_encode(info: u16) -> u32 {
    let mut cw = 0u32;
    let info = u32::from(info) & 0xFFF;
    for i in 0..12 {
        if (info >> (11 - i)) & 1 == 1 {
            cw ^= GOLAY_23_12_GEN[i];
        }
    }
    cw
}

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

fn golay_23_12_syndrome_table() -> &'static [u32; 2048] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 2048]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 2048];
        for a in 0..23 {
            let pat = 1u32 << a;
            table[golay_23_12_syndrome(pat) as usize] = pat;
        }
        for a in 0..23 {
            for b in (a + 1)..23 {
                let pat = (1u32 << a) | (1u32 << b);
                table[golay_23_12_syndrome(pat) as usize] = pat;
            }
        }
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
pub(crate) fn golay_23_12_decode(codeword: u32) -> FecDecoded {
    let cw = codeword & 0x7F_FFFF;
    let syndrome = golay_23_12_syndrome(cw);
    let pattern = golay_23_12_syndrome_table()[syndrome as usize];
    let corrected = cw ^ pattern;
    FecDecoded {
        info: ((corrected >> 11) & 0xFFF) as u16,
        errors: pattern.count_ones() as u8,
    }
}

/// Encode 12 information bits into a 24-bit extended Golay codeword.
pub(crate) fn golay_24_12_encode(info: u16) -> u32 {
    let cw23 = golay_23_12_encode(info);
    let parity = cw23.count_ones() & 1;
    (cw23 << 1) | parity
}

/// Decode a 24-bit extended Golay codeword. Corrects up to 3 errors and
/// detects 4 (errors = u8::MAX sentinel on a detected weight-4 pattern).
pub(crate) fn golay_24_12_decode(codeword: u32) -> FecDecoded {
    let cw = codeword & 0xFF_FFFF;
    let cw23 = (cw >> 1) & 0x7F_FFFF;
    let received_parity = cw & 1;
    let d23 = golay_23_12_decode(cw23);
    let corrected23 = golay_23_12_encode(d23.info);
    let reconstructed_parity = corrected23.count_ones() & 1;
    let parity_matches = reconstructed_parity == received_parity;
    let errors = if d23.errors == 3 && !parity_matches {
        u8::MAX
    } else if parity_matches {
        d23.errors
    } else {
        d23.errors + 1
    };
    FecDecoded {
        info: d23.info,
        errors,
    }
}

// ---- Hamming [15,11] (IMBE full-rate u4..u6) ------------------------------
//
// Single-error-correcting systematic [15,11,3] Hamming code. Codeword layout:
// bits 14..4 = 11 info bits (MSB = info bit 0), bits 3..0 = 4 parity bits.
// Each info bit i contributes parity column HAMMING_P[i] (the 11 non-unit
// nonzero 4-bit syndromes); each parity bit j owns unit syndrome (1<<j). That
// makes all 15 single-bit syndromes distinct => corrects 1 error.
//
// NOTE: this is a correct, self-consistent [15,11] code (roundtrip + 1-error
// tested). The exact column ordering must equal IMBE's (TIA-102.BABA-A) for
// bit-exact firmware parity; that has not been confirmed against a vector.

const HAMMING_P: [u16; 11] = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];

/// Encode 11 information bits (LSB-aligned) into a 15-bit Hamming codeword.
pub(crate) fn hamming_15_11_encode(info: u16) -> u16 {
    let info = info & 0x7FF;
    let mut parity = 0u16;
    for i in 0..11 {
        if (info >> (10 - i)) & 1 == 1 {
            parity ^= HAMMING_P[i];
        }
    }
    (info << 4) | (parity & 0xF)
}

fn hamming_15_11_syndrome(cw: u16) -> u16 {
    let info = (cw >> 4) & 0x7FF;
    let received_parity = cw & 0xF;
    let mut recomputed = 0u16;
    for i in 0..11 {
        if (info >> (10 - i)) & 1 == 1 {
            recomputed ^= HAMMING_P[i];
        }
    }
    received_parity ^ recomputed
}

fn hamming_15_11_table() -> &'static [u16; 16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u16; 16]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u16; 16];
        for pos in 0..15u16 {
            let pat = 1u16 << pos;
            table[hamming_15_11_syndrome(pat) as usize] = pat;
        }
        table
    })
}

/// Decode a 15-bit Hamming codeword, correcting up to 1 bit error.
pub(crate) fn hamming_15_11_decode(codeword: u16) -> FecDecoded {
    let cw = codeword & 0x7FFF;
    let syndrome = hamming_15_11_syndrome(cw);
    let pattern = hamming_15_11_table()[syndrome as usize];
    let corrected = cw ^ pattern;
    FecDecoded {
        info: (corrected >> 4) & 0x7FF,
        errors: pattern.count_ones() as u8,
    }
}

#[cfg(test)]
mod tests {
    //! Golay and Hamming pins, with the reported error COUNT treated as part
    //! of the contract rather than as diagnostics: the concealment gate keys
    //! on it, so a count that drifts changes which frames get muted.
    //!
    //! Three properties are proved exhaustively rather than sampled, because
    //! the search spaces are small enough that sampling would be a choice to
    //! test less: every weight-<=3 pattern over a Golay codeword, every
    //! weight-4 pattern over an extended codeword, and every single-bit error
    //! over every one of the 2048 Hamming messages.

    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// All bit patterns of exactly `w` set bits within `n` bits, ascending.
    fn patterns(n: u32, w: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for p in 0u32..(1 << n) {
            if p.count_ones() == w {
                out.push(p);
            }
        }
        out
    }

    // ── Golay [23,12] ─────────────────────────────────────────────────────

    #[test]
    fn golay_23_12_encode_is_systematic() {
        // The generator is in [I12 | P] form, so the top 12 bits of the
        // codeword are the message verbatim.
        for info in 0u16..4096 {
            let cw = golay_23_12_encode(info);
            assert_eq!(
                cw >> 11,
                u32::from(info),
                "info {info:#05x} is not systematic"
            );
            assert_eq!(cw & !0x7F_FFFF, 0, "codeword wider than 23 bits");
        }
        assert_eq!(golay_23_12_encode(0x000), 0x00_0000);
        assert_eq!(golay_23_12_encode(0x001), 0x00_0c75);
        assert_eq!(golay_23_12_encode(0x800), 0x40_063a);
        assert_eq!(golay_23_12_encode(0xabc), 0x55_e11e);
        // All-ones is a codeword of this code, so it encodes to all-ones.
        assert_eq!(golay_23_12_encode(0xfff), 0x7f_ffff);
        // Input above 12 bits is masked, not rejected.
        assert_eq!(golay_23_12_encode(0xf000 | 0xabc), 0x55_e11e);
    }

    #[test]
    fn golay_23_12_clean_codewords_decode_with_zero_errors() {
        for info in 0u16..4096 {
            let d = golay_23_12_decode(golay_23_12_encode(info));
            assert_eq!(d, FecDecoded { info, errors: 0 }, "info {info:#05x}");
        }
    }

    #[test]
    fn golay_23_12_corrects_every_pattern_up_to_weight_three() {
        // Golay(23,12,7) is perfect: the 2048 syndromes are in bijection with
        // the 1 + 23 + 253 + 1771 = 2048 patterns of weight <= 3, so every one
        // of them is corrected exactly and the reported count is the true
        // weight. This is the whole correction capability, checked, not
        // sampled.
        let mut total = 0usize;
        for w in 0..=3u32 {
            for pat in patterns(23, w) {
                for &info in &[0x000u16, 0xabc, 0xfff, 0x555] {
                    let d = golay_23_12_decode(golay_23_12_encode(info) ^ pat);
                    assert_eq!(
                        d,
                        FecDecoded {
                            info,
                            errors: w as u8
                        },
                        "info {info:#05x} pattern {pat:#08x} (weight {w})"
                    );
                }
                total += 1;
            }
        }
        assert_eq!(total, 2048, "the weight-<=3 space is not the full 2048");
    }

    #[test]
    fn golay_23_12_miscorrects_at_weight_four() {
        // Beyond the correction radius the decoder does NOT report failure --
        // there is no syndrome left to signal it with. It silently picks the
        // nearest codeword and reports 3. Any caller treating `errors < 4` as
        // "trustworthy" is relying on this being impossible, and it is not.
        let cw = golay_23_12_encode(0xabc);
        let pat: u32 = (1 << 1) | (1 << 5) | (1 << 13) | (1 << 19);
        assert_eq!(pat.count_ones(), 4);
        let d = golay_23_12_decode(cw ^ pat);
        assert_eq!(
            d,
            FecDecoded {
                info: 0xbb0,
                errors: 3
            }
        );
        assert_ne!(d.info, 0xabc);

        // No weight-4 pattern is ever reported as such by the plain code.
        for pat in patterns(23, 4) {
            let d = golay_23_12_decode(cw ^ pat);
            assert_eq!(d.errors, 3, "pattern {pat:#08x}");
        }
    }

    #[test]
    fn golay_23_12_masks_input_above_23_bits() {
        let cw = golay_23_12_encode(0xabc);
        assert_eq!(golay_23_12_decode(cw), golay_23_12_decode(cw | 0xff80_0000));
    }

    // ── extended Golay [24,12] ────────────────────────────────────────────

    #[test]
    fn golay_24_12_encode_appends_an_even_parity_bit() {
        for info in 0u16..4096 {
            let cw24 = golay_24_12_encode(info);
            assert_eq!(cw24 >> 1, golay_23_12_encode(info));
            assert_eq!(
                cw24.count_ones() % 2,
                0,
                "info {info:#05x}: extended codeword must have even weight"
            );
        }
        assert_eq!(golay_24_12_encode(0x000), 0x00_0000);
        assert_eq!(golay_24_12_encode(0x001), 0x00_18eb);
        assert_eq!(golay_24_12_encode(0xabc), 0xab_c23c);
        assert_eq!(golay_24_12_encode(0xfff), 0xff_ffff);
    }

    #[test]
    fn golay_24_12_reports_the_true_weight_up_to_three() {
        // Including patterns that touch the parity bit, where the count comes
        // from the parity-disagreement branch (`d23.errors + 1`) rather than
        // from the syndrome.
        for w in 0..=3u32 {
            for pat in patterns(24, w) {
                for &info in &[0x000u16, 0xabc, 0xfff] {
                    let d = golay_24_12_decode(golay_24_12_encode(info) ^ pat);
                    assert_eq!(
                        d,
                        FecDecoded {
                            info,
                            errors: w as u8
                        },
                        "info {info:#05x} pattern {pat:#08x} (weight {w})"
                    );
                }
            }
        }
    }

    #[test]
    fn golay_24_12_reported_three_does_not_mean_three_corrected_bits() {
        let cw = golay_24_12_encode(0xabc);
        // Three errors inside the 23-bit body: the syndrome finds all three.
        let body3 = (1 << 4) | (1 << 10) | (1 << 21);
        assert_eq!(
            golay_24_12_decode(cw ^ body3),
            FecDecoded {
                info: 0xabc,
                errors: 3
            }
        );
        // Two in the body plus the parity bit: the syndrome finds two and the
        // parity mismatch adds one. Same reported 3, different provenance.
        let body2_par = (1 << 1) | (1 << 5) | 1;
        assert_eq!(
            golay_24_12_decode(cw ^ body2_par),
            FecDecoded {
                info: 0xabc,
                errors: 3
            }
        );
        // A parity-only flip is a real, reported error even though the body
        // was untouched.
        assert_eq!(
            golay_24_12_decode(cw ^ 1),
            FecDecoded {
                info: 0xabc,
                errors: 1
            }
        );
    }

    #[test]
    fn golay_24_12_flags_every_weight_four_pattern_as_uncorrectable() {
        // The extra parity bit buys detection of exactly one more error, and
        // it is signalled by the `u8::MAX` sentinel rather than by an error
        // count. Both weight-4 shapes reach it, by different routes:
        //   * 4 in the body -> the syndrome miscorrects to a codeword whose
        //     parity now disagrees;
        //   * 3 in the body + the parity bit -> the syndrome corrects
        //     correctly, and the parity bit is the one that disagrees.
        let cw = golay_24_12_encode(0xabc);

        let four_body = (1 << 2) | (1 << 6) | (1 << 12) | (1 << 23);
        let d = golay_24_12_decode(cw ^ four_body);
        assert_eq!(d.errors, u8::MAX);
        assert_eq!(
            d.info, 0x3ad,
            "the info field is garbage but still returned"
        );

        let three_plus_parity = (1 << 4) | (1 << 10) | (1 << 21) | 1;
        let d = golay_24_12_decode(cw ^ three_plus_parity);
        assert_eq!(d.errors, u8::MAX);
        assert_eq!(d.info, 0xabc, "here the body was corrected correctly");

        // Exhaustive over all C(24,4) = 10626 patterns, for three messages.
        let all4 = patterns(24, 4);
        assert_eq!(all4.len(), 10626);
        for &info in &[0x000u16, 0xabc, 0xfff] {
            let cw = golay_24_12_encode(info);
            for &pat in &all4 {
                assert_eq!(
                    golay_24_12_decode(cw ^ pat).errors,
                    u8::MAX,
                    "info {info:#05x} pattern {pat:#08x} was not detected"
                );
            }
        }
    }

    // ── Hamming [15,11] ───────────────────────────────────────────────────

    #[test]
    fn hamming_15_11_encode_is_systematic() {
        for info in 0u16..2048 {
            let cw = hamming_15_11_encode(info);
            assert_eq!(cw >> 4, info, "info {info:#05x} is not systematic");
            assert_eq!(cw & !0x7FFF, 0, "codeword wider than 15 bits");
        }
        assert_eq!(hamming_15_11_encode(0x000), 0x0000);
        assert_eq!(hamming_15_11_encode(0x001), 0x001f);
        assert_eq!(hamming_15_11_encode(0x400), 0x4003);
        assert_eq!(hamming_15_11_encode(0x5a3), 0x5a39);
        assert_eq!(hamming_15_11_encode(0x7ff), 0x7fff);
        // Input above 11 bits is masked, not rejected.
        assert_eq!(hamming_15_11_encode(0xf800 | 0x5a3), 0x5a39);
    }

    #[test]
    fn hamming_15_11_corrects_one_error_anywhere() {
        // All 2048 messages, all 15 bit positions. The code is perfect, so
        // every nonzero syndrome names exactly one position and the reported
        // count is always 1.
        for info in 0u16..2048 {
            let cw = hamming_15_11_encode(info);
            assert_eq!(hamming_15_11_decode(cw), FecDecoded { info, errors: 0 });
            for pos in 0..15 {
                assert_eq!(
                    hamming_15_11_decode(cw ^ (1 << pos)),
                    FecDecoded { info, errors: 1 },
                    "info {info:#05x} bit {pos}"
                );
            }
        }
    }

    #[test]
    fn hamming_15_11_miscorrects_at_two_errors() {
        // Two errors always land on a third position's syndrome, so the
        // decoder reports 1 error and hands back a message that is wrong. It
        // never reports 2. As with Golay, a caller cannot read `errors` as an
        // upper bound on damage.
        let info = 0x5a3;
        let cw = hamming_15_11_encode(info);
        // Both parity bits: the syndrome (0b0011) is info bit 0's column.
        assert_eq!(
            hamming_15_11_decode(cw ^ 0b11),
            FecDecoded {
                info: 0x1a3,
                errors: 1
            }
        );
        assert_eq!(
            hamming_15_11_decode(cw ^ (1 << 5) ^ (1 << 9)),
            FecDecoded {
                info: 0x581,
                errors: 1
            }
        );

        let mut wrong = 0;
        for a in 0..15 {
            for b in (a + 1)..15 {
                let d = hamming_15_11_decode(cw ^ (1 << a) ^ (1 << b));
                assert_eq!(d.errors, 1, "bits {a},{b}");
                if d.info != info {
                    wrong += 1;
                }
            }
        }
        assert_eq!(
            wrong, 105,
            "every one of the C(15,2) double errors corrupts info"
        );
    }

    #[test]
    fn hamming_15_11_masks_input_above_15_bits() {
        let cw = hamming_15_11_encode(0x5a3);
        assert_eq!(hamming_15_11_decode(cw), hamming_15_11_decode(cw | 0x8000));
    }

    // ── sweeps ────────────────────────────────────────────────────────────

    /// Re-bless by running with `--nocapture` and pasting the printed block.
    const SWEEPS: [(&str, u64); 4] = [
        ("golay_23_12", 0xa173cd9865d58e24),
        ("golay_24_12", 0x11ee77cec8068d23),
        ("hamming_15_11", 0x5a58279737c1c25b),
        // Raw sentinel count behind this hash: 6101 of 14 040 corruptions.
        ("uncorrectable_sentinels", 0xdaf68e1bce42a6ed),
    ];

    #[test]
    fn fec_sweeps_are_pinned() {
        fn feed(out: &mut Vec<u8>, d: FecDecoded) {
            out.extend_from_slice(&d.info.to_le_bytes());
            out.push(d.errors);
        }

        // Deterministic corruption schedule: for each message, walk a fixed
        // integer sequence of error patterns. No RNG, no floats.
        fn corruptions(bits: u32, seed: u32) -> Vec<u32> {
            let mut state = seed | 1;
            (0..24)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state & ((1u32 << bits) - 1)
                })
                .collect()
        }

        let mut actual: Vec<(&'static str, u64)> = Vec::new();

        let mut out = Vec::new();
        for info in (0u16..4096).step_by(7) {
            let cw = golay_23_12_encode(info);
            out.extend_from_slice(&cw.to_le_bytes());
            for pat in corruptions(23, u32::from(info) + 0x1234) {
                feed(&mut out, golay_23_12_decode(cw ^ pat));
            }
        }
        actual.push(("golay_23_12", fnv1a(&out)));

        out.clear();
        for info in (0u16..4096).step_by(7) {
            let cw = golay_24_12_encode(info);
            out.extend_from_slice(&cw.to_le_bytes());
            for pat in corruptions(24, u32::from(info) + 0x5678) {
                feed(&mut out, golay_24_12_decode(cw ^ pat));
            }
        }
        actual.push(("golay_24_12", fnv1a(&out)));

        out.clear();
        for info in (0u16..2048).step_by(3) {
            let cw = hamming_15_11_encode(info);
            out.extend_from_slice(&cw.to_le_bytes());
            for pat in corruptions(15, u32::from(info) + 0x9abc) {
                feed(&mut out, hamming_15_11_decode(cw ^ (pat as u16)));
            }
        }
        actual.push(("hamming_15_11", fnv1a(&out)));

        // How often the extended decoder reaches the u8::MAX sentinel across
        // the same schedule: the concealment gate's trigger rate in one number.
        out.clear();
        let mut sentinels = 0u32;
        for info in (0u16..4096).step_by(7) {
            let cw = golay_24_12_encode(info);
            for pat in corruptions(24, u32::from(info) + 0x5678) {
                if golay_24_12_decode(cw ^ pat).errors == u8::MAX {
                    sentinels += 1;
                }
            }
        }
        out.extend_from_slice(&sentinels.to_le_bytes());
        actual.push(("uncorrectable_sentinels", fnv1a(&out)));

        println!("\n// Re-bless by copying this block into SWEEPS:");
        println!("    const SWEEPS: [(&str, u64); {}] = [", actual.len());
        for (name, h) in &actual {
            println!("        (\"{name}\", {h:#018x}),");
        }
        println!("    ];");
        println!("    // uncorrectable_sentinels raw count: {sentinels}\n");

        assert!(
            SWEEPS.iter().all(|(_, h)| *h != 0),
            "SWEEPS still holds placeholder zeros — an unpinned gate reads as \
             passing, which is worse than no gate."
        );
        // The sentinel branch must actually be reached by this schedule, or
        // the fourth entry pins nothing.
        assert!(
            sentinels > 0,
            "no corruption in the sweep reached the uncorrectable sentinel"
        );
        assert_eq!(SWEEPS.len(), actual.len(), "SWEEPS entry count drifted");

        let mut bad = Vec::new();
        for ((name, got), (want_name, want)) in actual.iter().zip(SWEEPS.iter()) {
            assert_eq!(name, want_name, "SWEEPS is out of order");
            if got != want {
                bad.push(format!("{name}: expected {want:#018x}, got {got:#018x}"));
            }
        }
        assert!(
            bad.is_empty(),
            "FEC decode output or error counts changed:\n  {}\n\nThe error \
             count is not diagnostics — concealment keys on it, so a drift \
             here changes which frames get repeated or muted.",
            bad.join("\n  ")
        );
    }
}
