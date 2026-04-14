//! Half-rate IMBE/AMBE frame — 72 bits carrying 49 parameter bits at 3,600 bps.
//!
//! Per TIA-102.BABA-A §2.4–§2.6. Composes the generic Golay/Hamming
//! primitives from [`crate::fec`] into the half-rate wire layout:
//!
//! ```text
//! c₀ : [24, 12] extended Golay, 24 bits   (from û₀ = 12 info bits)
//! c₁ : [23, 12] Golay,          23 bits   (from û₁ = 12 info bits)
//! c₂ : uncoded,                 11 bits   (= û₂)
//! c₃ : uncoded,                 14 bits   (= û₃)
//!                                -------
//! total                          72 bits
//! ```
//!
//! PN modulation (§2.5, 24-value LCG) masks `c₁` only — `c₀` must be
//! demodulation-free so the decoder can recover `û₀` first, and `c₂`,
//! `c₃` are uncoded so PN would have nothing meaningful to scramble.
//!
//! Annex S interleaves the 72 bits into 36 dibit symbols; the table is
//! generated at build time from `spec_tables/annex_s_interleave.csv`.
//!
//! Bit-ordering convention matches the full-rate module: element `k`
//! at u32 bit `k`, with the highest-indexed bit being the first-
//! transmitted bit of the codeword (consistent with Annex S's
//! `c0[23]` → first dibit MSB and with the full-rate PN alignment
//! verified against DVSI).

use crate::fec::{
    golay_23_12_decode, golay_23_12_encode, golay_24_12_decode, golay_24_12_encode,
};

// ---------------------------------------------------------------------------
// Half-rate codebook tables (Annex L / M / N / O / P / Q / R)
// ---------------------------------------------------------------------------

/// One row of the half-rate pitch quantization table (Annex L):
/// a `(L̃, ω̃₀)` pair selected by the 7-bit `b̂₀` index.
#[derive(Clone, Copy, Debug)]
pub struct PitchEntry {
    /// Number of harmonics for this pitch index.
    pub l: u8,
    /// Fundamental frequency in radians/sample.
    pub omega_0: f32,
}

include!(concat!(env!("OUT_DIR"), "/annex_l_pitch.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_m_vuv.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_n_blocks.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_o_gain.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_p_prba24.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_q_prba58.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_r_hoc.rs"));

/// One row of the Annex T tone-frame parameter table (§2.10.3).
/// Carries the fundamental frequency and the two harmonic indices
/// that make up a tone (`l1 == l2` for single-frequency tones).
#[derive(Clone, Copy, Debug)]
pub struct ToneParams {
    /// Fundamental frequency in Hz.
    pub f0: f32,
    /// Harmonic index of the first tone component.
    pub l1: u8,
    /// Harmonic index of the second tone component (equal to `l1`
    /// for single-frequency tones).
    pub l2: u8,
}

include!(concat!(env!("OUT_DIR"), "/annex_t_tones.rs"));

// ---------------------------------------------------------------------------
// Widths and constants
// ---------------------------------------------------------------------------

/// Width in bits of each half-rate info vector `û₀..û₃`. BABA-A §2.3.
pub const INFO_WIDTHS: [u8; 4] = [12, 12, 11, 14];

/// Width in bits of each half-rate code vector `c₀..c₃`. BABA-A §2.4.
/// `c₀` is `[24, 12]` extended Golay, `c₁` is `[23, 12]` Golay, and
/// `c₂`, `c₃` are uncoded pass-through of `û₂`, `û₃`.
pub const CODE_WIDTHS: [u8; 4] = [24, 23, 11, 14];

/// Sum of info widths — must equal 49.
pub const INFO_BITS_TOTAL: u16 = 49;

/// Length of the half-rate PN sequence: `p_r(0)` seed plus
/// `p_r(1..=23)` for the 23-bit `m̂₁` mask.
pub const PN_SEQ_LEN_HALFRATE: usize = 24;

/// Number of dibit symbols per half-rate frame (72 bits / 2 bits per
/// symbol).
pub const DIBITS_PER_FRAME: usize = 36;

// ---------------------------------------------------------------------------
// Annex S interleave
// ---------------------------------------------------------------------------

/// One row of the Annex S interleaving table (same shape as
/// [`super::fec::AnnexHEntry`] for full-rate).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnnexSEntry {
    pub bit1_vec: u8,
    pub bit1_idx: u8,
    pub bit0_vec: u8,
    pub bit0_idx: u8,
}

include!(concat!(env!("OUT_DIR"), "/annex_s.rs"));

/// Deinterleave a 72-bit half-rate frame (36 dibits) into the 4 code
/// vectors `c₀..c₃`.
pub fn deinterleave_halfrate(dibits: &[u8; DIBITS_PER_FRAME]) -> [u32; 4] {
    let mut c = [0u32; 4];
    for (sym, d) in dibits.iter().enumerate() {
        let entry = ANNEX_S[sym];
        let hi = u32::from((d >> 1) & 1);
        let lo = u32::from(d & 1);
        c[entry.bit1_vec as usize] |= hi << entry.bit1_idx;
        c[entry.bit0_vec as usize] |= lo << entry.bit0_idx;
    }
    c
}

/// Interleave 4 code vectors `c₀..c₃` into 72 bits (36 dibits) per
/// Annex S. Inverse of [`deinterleave_halfrate`].
pub fn interleave_halfrate(codewords: &[u32; 4]) -> [u8; DIBITS_PER_FRAME] {
    let mut dibits = [0u8; DIBITS_PER_FRAME];
    for (sym, entry) in ANNEX_S.iter().enumerate() {
        let hi = ((codewords[entry.bit1_vec as usize] >> entry.bit1_idx) & 1) as u8;
        let lo = ((codewords[entry.bit0_vec as usize] >> entry.bit0_idx) & 1) as u8;
        dibits[sym] = (hi << 1) | lo;
    }
    dibits
}

// ---------------------------------------------------------------------------
// PN modulation (§2.5)
// ---------------------------------------------------------------------------

/// Generate the half-rate PN sequence `p_r(0..=23)`.
///
/// Same LCG as full-rate (§1.6 Eq. 84–85), but with only 24 values.
/// Seed is `16 · û₀` using the 12-bit info word that emerges from the
/// `[24, 12]` decode of `c₀`.
pub fn pn_sequence_halfrate(u0: u16) -> [u16; PN_SEQ_LEN_HALFRATE] {
    debug_assert!(u0 < 4096, "û₀ is a 12-bit info word");
    let mut pr = [0u16; PN_SEQ_LEN_HALFRATE];
    pr[0] = u0.wrapping_mul(16);
    for n in 1..PN_SEQ_LEN_HALFRATE {
        pr[n] = (173u32
            .wrapping_mul(pr[n - 1] as u32)
            .wrapping_add(13849)
            & 0xFFFF) as u16;
    }
    pr
}

/// Compute the 4 PN modulation masks `m̂₀..m̂₃` for half-rate.
///
/// * `m̂₀ = 0` (24 bits) — `c₀` must be decodable before PN is seeded.
/// * `m̂₁` — 23 bits from `p_r(1..=23)`; first PN at the highest bit
///   (bit 22), last PN at bit 0. Aligns first PN with first-
///   transmitted bit per the Annex H convention, consistent with the
///   full-rate DVSI-verified layout.
/// * `m̂₂ = 0` (11 bits) — `c₂` is uncoded; no PN.
/// * `m̂₃ = 0` (14 bits) — `c₃` is uncoded; no PN.
pub fn modulation_masks_halfrate(u0: u16) -> [u32; 4] {
    let pr = pn_sequence_halfrate(u0);
    let mut m = 0u32;
    for k in 0..23 {
        let bit = u32::from(pr[1 + k] >> 15); // MSB of p_r(n)
        m |= bit << (22 - k);
    }
    [0, m, 0, 0]
}

/// Demodulate a half-rate frame's 4 code vectors to recover
/// `v̂₀..v̂₃`. Same shape as the full-rate version: caller Golay-
/// decodes `c̃₀ → û₀` first to seed the PN generator, then XORs the
/// masks.
pub fn demodulate_halfrate(codewords: [u32; 4], u0: u16) -> [u32; 4] {
    let masks = modulation_masks_halfrate(u0);
    let mut v = [0u32; 4];
    for i in 0..4 {
        v[i] = codewords[i] ^ masks[i];
    }
    v
}

// ---------------------------------------------------------------------------
// Frame-level encode / decode
// ---------------------------------------------------------------------------

/// Decoded 49-bit information layer of a half-rate AMBE frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmbeFrame {
    /// Decoded info vectors `û₀..û₃`, LSB-aligned. Widths per
    /// [`INFO_WIDTHS`].
    pub info: [u16; 4],
    /// Per-vector FEC error count. `errors[0]` is 0–3 (or `u8::MAX`
    /// for an uncorrectable extended-Golay frame); `errors[1]` is
    /// 0–3 (perfect [23,12] Golay never reports uncorrectable).
    /// `errors[2]` and `errors[3]` are always 0 (uncoded).
    pub errors: [u8; 4],
}

impl AmbeFrame {
    /// Total error count across all vectors.
    pub fn error_total(&self) -> u16 {
        self.errors.iter().map(|&e| u16::from(e)).sum()
    }
}

/// Decode a half-rate frame: 36 dibits → [`AmbeFrame`].
///
/// Pipeline (BABA-A §2.4–§2.6 compose with §2.5):
/// 1. Deinterleave 36 dibits → `c̃₀..c̃₃`.
/// 2. `[24, 12]` extended-Golay-decode `c̃₀` → `û₀`.
/// 3. Generate PN masks from `û₀`; XOR into `c̃₁`.
/// 4. `[23, 12]` Golay-decode the result → `û₁`.
/// 5. Uncoded passthrough: `û₂ = c̃₂`, `û₃ = c̃₃`.
pub fn decode_halfrate_frame(dibits: &[u8; DIBITS_PER_FRAME]) -> AmbeFrame {
    let c = deinterleave_halfrate(dibits);

    let d0 = golay_24_12_decode(c[0]);
    let u0 = d0.info;
    let masks = modulation_masks_halfrate(u0);
    let d1 = golay_23_12_decode(c[1] ^ masks[1]);
    let u2 = (c[2] & 0x7FF) as u16; // 11 bits
    let u3 = (c[3] & 0x3FFF) as u16; // 14 bits

    AmbeFrame {
        info: [d0.info, d1.info, u2, u3],
        errors: [d0.errors, d1.errors, 0, 0],
    }
}

/// Encode 4 info vectors into 72 half-rate air-interface bits
/// (36 dibits). Inverse of [`decode_halfrate_frame`].
pub fn encode_halfrate_frame(info: &[u16; 4]) -> [u8; DIBITS_PER_FRAME] {
    // Mask each input to its declared width so stray high bits can't
    // silently corrupt the frame.
    let u: [u16; 4] = core::array::from_fn(|i| {
        let w = INFO_WIDTHS[i] as u32;
        let m = ((1u32 << w) - 1) as u16;
        info[i] & m
    });

    let v0 = golay_24_12_encode(u[0]);
    let v1 = golay_23_12_encode(u[1]);
    let v2 = u32::from(u[2]);
    let v3 = u32::from(u[3]);

    let masks = modulation_masks_halfrate(u[0]);
    let c = [v0, v1 ^ masks[1], v2, v3];
    interleave_halfrate(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_mask(i: usize) -> u32 {
        let len = CODE_WIDTHS[i] as u32;
        if len == 32 { u32::MAX } else { (1u32 << len) - 1 }
    }

    #[test]
    fn widths_sum_to_49_and_72() {
        assert_eq!(INFO_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 49);
        assert_eq!(CODE_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 72);
        assert_eq!(INFO_BITS_TOTAL, 49);
    }

    #[test]
    fn interleave_covers_every_codeword_bit_exactly_once() {
        let mut seen = [[false; 24]; 4];
        for entry in ANNEX_S.iter() {
            for (v, i) in [(entry.bit1_vec, entry.bit1_idx), (entry.bit0_vec, entry.bit0_idx)] {
                assert!((i as u8) < CODE_WIDTHS[v as usize]);
                assert!(!seen[v as usize][i as usize]);
                seen[v as usize][i as usize] = true;
            }
        }
        for (v, row) in seen.iter().enumerate() {
            for (i, &b) in row.iter().enumerate().take(CODE_WIDTHS[v] as usize) {
                assert!(b, "({v}, {i}) never covered");
            }
        }
    }

    #[test]
    fn interleave_deinterleave_roundtrip() {
        let mut cw = [0u32; 4];
        let mut state = 0xFEEDFACEu32;
        for i in 0..4 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            cw[i] = state & valid_mask(i);
        }
        let dibits = interleave_halfrate(&cw);
        for d in dibits.iter() {
            assert!(*d < 4);
        }
        let back = deinterleave_halfrate(&dibits);
        assert_eq!(back, cw);
    }

    #[test]
    fn single_bit_propagates_to_one_dibit_position() {
        for v in 0..4 {
            for idx in 0..CODE_WIDTHS[v] {
                let mut cw = [0u32; 4];
                cw[v] = 1u32 << idx;
                let dibits = interleave_halfrate(&cw);
                let ones: u32 = dibits.iter().map(|d| u32::from(*d).count_ones()).sum();
                assert_eq!(ones, 1, "(v={v}, idx={idx})");
                assert_eq!(deinterleave_halfrate(&dibits), cw);
            }
        }
    }

    #[test]
    fn pn_seed_halfrate_matches_formula() {
        for u0 in [0u16, 1, 42, 4095] {
            assert_eq!(pn_sequence_halfrate(u0)[0], u0.wrapping_mul(16));
        }
        let pr = pn_sequence_halfrate(1);
        assert_eq!(pr[0], 16);
        // pr[1] = (173·16 + 13849) mod 65536 = 16617
        assert_eq!(pr[1], 16617);
    }

    #[test]
    fn modulation_masks_m0_m2_m3_always_zero() {
        for u0 in [0u16, 1, 123, 4095] {
            let masks = modulation_masks_halfrate(u0);
            assert_eq!(masks[0], 0);
            assert_eq!(masks[2], 0);
            assert_eq!(masks[3], 0);
        }
    }

    #[test]
    fn modulation_masks_m1_fits_within_23_bits() {
        for u0 in [0u16, 1, 123, 4095] {
            let masks = modulation_masks_halfrate(u0);
            assert!(masks[1] < (1 << 23));
        }
    }

    #[test]
    fn demodulate_is_self_inverse() {
        let u0 = 0xA5Cu16;
        let masks = modulation_masks_halfrate(u0);
        let v: [u32; 4] = [0x123456, 0x654321, 0x7A5, 0x2F2F];
        let c: [u32; 4] = core::array::from_fn(|i| v[i] ^ masks[i]);
        let v2 = demodulate_halfrate(c, u0);
        assert_eq!(v2, v);
    }

    // ---- Frame-level roundtrip ------------------------------------------

    fn sample_info(seed: u32) -> [u16; 4] {
        let mut state = seed;
        core::array::from_fn(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = INFO_WIDTHS[i] as u32;
            let m = ((1u32 << w) - 1) as u16;
            (state as u16) & m
        })
    }

    #[test]
    fn frame_roundtrip_zero_and_sampled() {
        for seed in [0u32, 1, 0xDEADBEEF, 0xCAFEBABE, 0x12345678] {
            let u = if seed == 0 { [0u16; 4] } else { sample_info(seed) };
            let dibits = encode_halfrate_frame(&u);
            for (s, d) in dibits.iter().enumerate() {
                assert!(*d < 4, "symbol {s} not a dibit (seed {seed:08x})");
            }
            let back = decode_halfrate_frame(&dibits);
            assert_eq!(back.info, u, "seed 0x{seed:08x}");
            assert_eq!(back.errors, [0u8; 4], "seed 0x{seed:08x}");
        }
    }

    #[test]
    fn single_bit_flip_in_c0_is_corrected() {
        // c₀ is [24,12] extended Golay → corrects any single bit error.
        let u = sample_info(0xA5A5A5A5);
        let clean = encode_halfrate_frame(&u);
        for bit in 0..24 {
            let c = deinterleave_halfrate(&clean);
            let mut c_flipped = c;
            c_flipped[0] ^= 1u32 << bit;
            let dibits = interleave_halfrate(&c_flipped);
            let out = decode_halfrate_frame(&dibits);
            assert_eq!(out.info, u, "c̃₀ bit {bit}");
            assert_eq!(out.errors[0], 1);
        }
    }

    #[test]
    fn single_bit_flip_in_c1_is_corrected() {
        let u = sample_info(0x5A5A5A5A);
        let clean = encode_halfrate_frame(&u);
        for bit in 0..23 {
            let c = deinterleave_halfrate(&clean);
            let mut c_flipped = c;
            c_flipped[1] ^= 1u32 << bit;
            let dibits = interleave_halfrate(&c_flipped);
            let out = decode_halfrate_frame(&dibits);
            assert_eq!(out.info, u, "c̃₁ bit {bit}");
            assert_eq!(out.errors[1], 1);
        }
    }

    #[test]
    fn uncoded_vectors_have_no_error_correction() {
        let u = sample_info(0xFACEFEED);
        let clean = encode_halfrate_frame(&u);
        let c = deinterleave_halfrate(&clean);
        // Flip bit in c₂ — passes through to û₂.
        let mut c_flip2 = c;
        c_flip2[2] ^= 1u32 << 5;
        let out2 = decode_halfrate_frame(&interleave_halfrate(&c_flip2));
        assert_eq!(out2.info[2], u[2] ^ 0b10_0000);
        assert_eq!(out2.errors[2], 0);
        // Flip bit in c₃ — passes through to û₃.
        let mut c_flip3 = c;
        c_flip3[3] ^= 1u32 << 11;
        let out3 = decode_halfrate_frame(&interleave_halfrate(&c_flip3));
        assert_eq!(out3.info[3], u[3] ^ (1 << 11));
        assert_eq!(out3.errors[3], 0);
    }

    // ---- Annex L / M / N / O / P / Q / R spot checks ---------------------

    #[test]
    fn annex_l_spot_values_match_spec() {
        // Spec-provided spot values from impl-spec §12.8 / full-text
        // Annex L: b₀=0 → (L=9, ω₀=0.049971); b₀=119 → (L=56, ω₀=0.008125).
        assert_eq!(AMBE_PITCH_TABLE.len(), 120);
        let p0 = AMBE_PITCH_TABLE[0];
        assert_eq!(p0.l, 9);
        assert!((p0.omega_0 - 0.049971).abs() < 1e-6);
        let p_last = AMBE_PITCH_TABLE[119];
        assert_eq!(p_last.l, 56);
        assert!((p_last.omega_0 - 0.008125).abs() < 1e-6);
    }

    #[test]
    fn annex_l_omega_strictly_decreasing() {
        for w in AMBE_PITCH_TABLE.windows(2) {
            assert!(w[0].omega_0 > w[1].omega_0);
        }
    }

    #[test]
    fn annex_m_all_voiced_and_all_unvoiced_endpoints() {
        assert_eq!(AMBE_VUV_CODEBOOK.len(), 32);
        // b₁ = 0 → all 8 voiced (impl-spec §12.9 spot value).
        for &v in &AMBE_VUV_CODEBOOK[0] { assert!(v); }
        // b₁ = 16 → all 8 unvoiced.
        for &v in &AMBE_VUV_CODEBOOK[16] { assert!(!v); }
    }

    #[test]
    fn annex_n_block_lengths_sum_to_l() {
        assert_eq!(AMBE_BLOCK_LENGTHS.len(), 48);
        for (l_idx, row) in AMBE_BLOCK_LENGTHS.iter().enumerate() {
            let l = (l_idx + 9) as u32;
            let sum: u32 = row.iter().map(|&x| x as u32).sum();
            assert_eq!(sum, l, "L={l}");
        }
    }

    #[test]
    fn annex_o_spot_values_and_monotone() {
        assert_eq!(AMBE_GAIN_LEVELS.len(), 32);
        assert!((AMBE_GAIN_LEVELS[0] - (-2.0)).abs() < 1e-6);
        assert!((AMBE_GAIN_LEVELS[31] - 6.874496).abs() < 1e-6);
        for w in AMBE_GAIN_LEVELS.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn annex_p_first_entry_matches_spec() {
        // Impl-spec §12.12: b₃=0 → G₂=0.526055, G₃=−0.328567, G₄=−0.304727.
        assert_eq!(AMBE_PRBA24.len(), 512);
        let e = AMBE_PRBA24[0];
        assert!((e[0] - 0.526055).abs() < 1e-6);
        assert!((e[1] - (-0.328567)).abs() < 1e-6);
        assert!((e[2] - (-0.304727)).abs() < 1e-6);
    }

    #[test]
    fn annex_q_first_entry_matches_spec() {
        // Impl-spec §12.13: b₄=0 → G₅=−0.103660, G₆=0.094597, G₇=−0.013149, G₈=0.081501.
        assert_eq!(AMBE_PRBA58.len(), 128);
        let e = AMBE_PRBA58[0];
        assert!((e[0] - (-0.103660)).abs() < 1e-6);
        assert!((e[1] - 0.094597).abs() < 1e-6);
        assert!((e[2] - (-0.013149)).abs() < 1e-6);
        assert!((e[3] - 0.081501).abs() < 1e-6);
    }

    #[test]
    fn annex_r_hoc_shapes_and_spot_value() {
        assert_eq!(AMBE_HOC_B5.len(), 32);
        assert_eq!(AMBE_HOC_B6.len(), 16);
        assert_eq!(AMBE_HOC_B7.len(), 16);
        assert_eq!(AMBE_HOC_B8.len(), 8);
        // Impl-spec §12.14: b₅=0 → H₁,₁=0.264108, H₁,₂=0.045976,
        // H₁,₃=−0.200999, H₁,₄=−0.122344.
        let e = AMBE_HOC_B5[0];
        assert!((e[0] - 0.264108).abs() < 1e-6);
        assert!((e[1] - 0.045976).abs() < 1e-6);
        assert!((e[2] - (-0.200999)).abs() < 1e-6);
        assert!((e[3] - (-0.122344)).abs() < 1e-6);
    }

    #[test]
    fn width_masking_strips_stray_high_bits() {
        let mut u = [0xFFFFu16; 4];
        // Reduce within the u16 limits but above the declared widths.
        u[0] = 0xFFFF;
        let dibits = encode_halfrate_frame(&u);
        let back = decode_halfrate_frame(&dibits);
        for i in 0..4 {
            let w = INFO_WIDTHS[i] as u32;
            let m = ((1u32 << w) - 1) as u16;
            assert_eq!(back.info[i], m, "vector {i} mask mismatch");
        }
    }
}
