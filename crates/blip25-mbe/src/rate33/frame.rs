//! Half-rate AMBE+2 frame — 72 bits carrying 49 parameter bits at 3,600 bps.
//!
//! Per TIA-102.BABA-A §2.4–§2.6 (originally the BABA-1 addendum,
//! consolidated into BABA-A). The half-rate vocoder is AMBE+2 in
//! everything but name; BABA-A renames it "Half-Rate Vocoder" to dodge
//! DVSI's trademark. Composes the generic Golay primitives from
//! [`crate::fec`] into the half-rate wire layout:
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
    golay_23_12_decode, golay_23_12_decode_soft, golay_23_12_encode, golay_24_12_decode,
    golay_24_12_decode_soft, golay_24_12_encode,
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

/// R34 (half-rate no-FEC, 49-bit / 7-byte) bit serialization order.
///
/// Bit `j` (MSB-first) of the 7-byte no-FEC frame carries the natural
/// info bit `R34_BIT_ORDER[j]`, where the natural order is
/// `û₀(12)‖û₁(12)‖û₂(11)‖û₃(14)` MSB-first. This is NOT the naive
/// sequential layout — DVSI emits a 3-way column interleave (rows of
/// 18/18/13 bits): `0,18,36, 1,19,37, …`. Bits 49..55 of the frame are
/// zero padding.
///
/// Real-world relevance: AMBE+2 no-FEC half-rate is what an NXDN/Fusion
/// console emits, so this ordering must match the chip, not just decode
/// cleanly. Derived empirically from the DVSI RC `r33`↔`r34` vectors and
/// verified a bijection across alltone/clean/dam/mark/alert (see
/// `examples/derive_r34_order.rs`).
pub const R34_BIT_ORDER: [u8; 49] = [
    0, 18, 36, 1, 19, 37, 2, 20, 38, 3, 21, 39, 4, 22, 40, 5, 23, 41, 6, 24, 42, 7, 25, 43, 8, 26,
    44, 9, 27, 45, 10, 28, 46, 11, 29, 47, 12, 30, 48, 13, 31, 14, 32, 15, 33, 16, 34, 17, 35,
];

/// Pack the four info vectors `û₀..û₃` into a 7-byte R34 no-FEC frame,
/// applying the [`R34_BIT_ORDER`] interleave. Bits 49..55 are zero pad.
pub fn pack_no_fec(info: &[u16; 4]) -> [u8; 7] {
    let natural = natural_bits(info);
    let mut out = [0u8; 7];
    for (j, &src) in R34_BIT_ORDER.iter().enumerate() {
        out[j / 8] |= natural[src as usize] << (7 - (j % 8));
    }
    out
}

/// Inverse of [`pack_no_fec`]: recover `û₀..û₃` from a 7-byte R34 frame.
pub fn unpack_no_fec(bytes: &[u8]) -> [u16; 4] {
    let mut natural = [0u8; INFO_BITS_TOTAL as usize];
    for (j, &dst) in R34_BIT_ORDER.iter().enumerate() {
        natural[dst as usize] = (bytes[j / 8] >> (7 - (j % 8))) & 1;
    }
    let mut out = [0u16; 4];
    let mut idx = 0;
    for (oi, &w) in INFO_WIDTHS.iter().enumerate() {
        let mut v = 0u16;
        for _ in 0..w {
            v = (v << 1) | u16::from(natural[idx]);
            idx += 1;
        }
        out[oi] = v;
    }
    out
}

/// Flatten `û₀..û₃` into the 49 natural info bits, MSB-first per field.
fn natural_bits(info: &[u16; 4]) -> [u8; INFO_BITS_TOTAL as usize] {
    let mut natural = [0u8; INFO_BITS_TOTAL as usize];
    let mut idx = 0;
    for (&w, &v) in INFO_WIDTHS.iter().zip(info.iter()) {
        for k in (0..w as usize).rev() {
            natural[idx] = ((v >> k) & 1) as u8;
            idx += 1;
        }
    }
    natural
}

/// Length of the half-rate PN sequence: `p_r(0)` seed plus
/// `p_r(1..=23)` for the 23-bit `m̂₁` mask.
pub const PN_SEQ_LEN: usize = 24;

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
pub fn deinterleave(dibits: &[u8; DIBITS_PER_FRAME]) -> [u32; 4] {
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
/// Annex S. Inverse of [`deinterleave`].
pub fn interleave(codewords: &[u32; 4]) -> [u8; DIBITS_PER_FRAME] {
    let mut dibits = [0u8; DIBITS_PER_FRAME];
    for (sym, entry) in ANNEX_S.iter().enumerate() {
        let hi = ((codewords[entry.bit1_vec as usize] >> entry.bit1_idx) & 1) as u8;
        let lo = ((codewords[entry.bit0_vec as usize] >> entry.bit0_idx) & 1) as u8;
        dibits[sym] = (hi << 1) | lo;
    }
    dibits
}

/// Total soft-bit count for a half-rate AMBE+2 frame (36 dibits × 2).
pub const SOFT_BITS: usize = 72;

/// Soft-deinterleaved code vectors for a half-rate AMBE+2 frame.
/// `c̃₀` is 24-bit extended Golay, `c̃₁` is 23-bit Golay, `c̃₂`/`c̃₃`
/// are uncoded (11 and 14 bits). All MSB-first.
#[derive(Clone, Copy, Debug)]
pub struct SoftCodeVectors {
    /// `c̃₀` — soft extended Golay(24,12) codeword, MSB at index 0.
    pub c0: [i8; 24],
    /// `c̃₁` — soft Golay(23,12) codeword, MSB at index 0.
    pub c1: [i8; 23],
    /// `c̃₂` — 11-bit uncoded, MSB at index 0.
    pub c2: [i8; 11],
    /// `c̃₃` — 14-bit uncoded, MSB at index 0.
    pub c3: [i8; 14],
}

impl Default for SoftCodeVectors {
    fn default() -> Self {
        Self {
            c0: [0i8; 24],
            c1: [0i8; 23],
            c2: [0i8; 11],
            c3: [0i8; 14],
        }
    }
}

/// Soft-deinterleave a 72-bit soft stream into the 4 MSB-first soft
/// code vectors per Annex S.
///
/// Input layout: `soft[2*sym]` = high bit of dibit `sym`,
/// `soft[2*sym + 1]` = low bit. Output vectors are MSB-first so they
/// can be fed directly into the soft FEC decoders.
pub fn soft_deinterleave(soft: &[i8; SOFT_BITS]) -> SoftCodeVectors {
    let mut out = SoftCodeVectors::default();
    for (sym, entry) in ANNEX_S.iter().enumerate() {
        let hi = soft[2 * sym];
        let lo = soft[2 * sym + 1];
        place_soft(
            &mut out,
            entry.bit1_vec as usize,
            entry.bit1_idx as usize,
            hi,
        );
        place_soft(
            &mut out,
            entry.bit0_vec as usize,
            entry.bit0_idx as usize,
            lo,
        );
    }
    out
}

/// Place a soft bit into MSB-first position inside the addressed
/// half-rate code vector. `idx` is LSB-first from Annex S; `W - 1 -
/// idx` is the MSB-first position.
fn place_soft(out: &mut SoftCodeVectors, vec_idx: usize, idx: usize, s: i8) {
    match vec_idx {
        0 => out.c0[24 - 1 - idx] = s,
        1 => out.c1[23 - 1 - idx] = s,
        2 => out.c2[11 - 1 - idx] = s,
        3 => out.c3[14 - 1 - idx] = s,
        _ => unreachable!(),
    }
}

/// XOR a hard PN mask into a soft codeword, preserving magnitudes.
/// Same semantics as the full-rate `soft_demodulate_vector`: mask bit
/// 1 flips the sign of the corresponding soft bit. `mask` is LSB-first
/// in a u32; `soft` is MSB-first of width `W`.
pub fn soft_demodulate_vector<const W: usize>(soft: &mut [i8; W], mask: u32) {
    for (i, s) in soft.iter_mut().enumerate() {
        let mask_bit = (mask >> (W - 1 - i)) & 1;
        if mask_bit == 1 {
            *s = s.saturating_neg();
        }
    }
}

// ---------------------------------------------------------------------------
// PN modulation (§2.5)
// ---------------------------------------------------------------------------

/// Generate the half-rate PN sequence `p_r(0..=23)`.
///
/// Same LCG as full-rate (§1.6 Eq. 84–85), but with only 24 values.
/// Seed is `16 · û₀` using the 12-bit info word that emerges from the
/// `[24, 12]` decode of `c₀`.
pub fn pn_sequence(u0: u16) -> [u16; PN_SEQ_LEN] {
    debug_assert!(u0 < 4096, "û₀ is a 12-bit info word");
    let mut pr = [0u16; PN_SEQ_LEN];
    pr[0] = u0.wrapping_mul(16);
    for n in 1..PN_SEQ_LEN {
        pr[n] = (173u32.wrapping_mul(pr[n - 1] as u32).wrapping_add(13849) & 0xFFFF) as u16;
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
pub fn modulation_masks(u0: u16) -> [u32; 4] {
    let pr = pn_sequence(u0);
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
pub fn demodulate(codewords: [u32; 4], u0: u16) -> [u32; 4] {
    let masks = modulation_masks(u0);
    let mut v = [0u32; 4];
    for i in 0..4 {
        v[i] = codewords[i] ^ masks[i];
    }
    v
}

// ---------------------------------------------------------------------------
// Frame-level encode / decode
// ---------------------------------------------------------------------------

/// Decoded 49-bit information layer of a half-rate (rate-33) frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Decoded info vectors `û₀..û₃`, LSB-aligned. Widths per
    /// [`INFO_WIDTHS`].
    pub info: [u16; 4],
    /// Per-vector FEC error count. `errors[0]` is 0–3 (or `u8::MAX`
    /// for an uncorrectable extended-Golay frame); `errors[1]` is
    /// 0–3 (perfect `(23, 12)` Golay never reports uncorrectable).
    /// `errors[2]` and `errors[3]` are always 0 (uncoded).
    pub errors: [u8; 4],
}

impl Frame {
    /// Total error count across all vectors.
    pub fn error_total(&self) -> u16 {
        self.errors.iter().map(|&e| u16::from(e)).sum()
    }
}

/// Decode the four code vectors `c̃₀..c̃₃` of a half-rate AMBE+2 frame
/// into the 49-bit info layer. This is the **protocol-agnostic codec
/// FEC core** — Golay(24,12)/Golay(23,12) + û₀-seeded PN + uncoded
/// passthrough — and carries no interleave. It is the natural reuse
/// boundary for any half-rate AMBE+2 protocol (P25 Phase 2, DMR, NXDN):
/// each protocol's own deinterleave lands the OTA bits in these four
/// vectors, then calls here. See [`decode_frame`] for the P25-Phase-2
/// adapter that prepends the Annex-S deinterleave.
///
/// 1. `[24, 12]` extended-Golay-decode `c̃₀` → `û₀`.
/// 2. Generate PN masks from `û₀`; XOR into `c̃₁`.
/// 3. `[23, 12]` Golay-decode the result → `û₁`.
/// 4. Uncoded passthrough: `û₂ = c̃₂`, `û₃ = c̃₃`.
pub fn decode_code_vectors(c: [u32; 4]) -> Frame {
    let d0 = golay_24_12_decode(c[0]);
    let u0 = d0.info;
    let masks = modulation_masks(u0);
    let d1 = golay_23_12_decode(c[1] ^ masks[1]);
    let u2 = (c[2] & 0x7FF) as u16; // 11 bits
    let u3 = (c[3] & 0x3FFF) as u16; // 14 bits

    Frame {
        info: [d0.info, d1.info, u2, u3],
        errors: [d0.errors, d1.errors, 0, 0],
    }
}

/// Decode a half-rate frame from 36 P25-Phase-2 dibits → [`Frame`].
///
/// Thin Annex-S adapter over [`decode_code_vectors`]:
/// 1. Deinterleave 36 dibits → `c̃₀..c̃₃` (Annex S — P25-Phase-2-specific).
/// 2. Run the shared codec FEC core ([`decode_code_vectors`]).
pub fn decode_frame(dibits: &[u8; DIBITS_PER_FRAME]) -> Frame {
    decode_code_vectors(deinterleave(dibits))
}

/// Soft-decision decode of the four code vectors of a half-rate AMBE+2
/// frame → 49-bit info layer. The **protocol-agnostic soft codec FEC
/// core**, mirroring [`decode_code_vectors`] in the soft domain and
/// carrying no interleave. Reuse boundary for DMR/NXDN soft decode:
/// land the OTA soft bits in a [`SoftCodeVectors`] via the protocol's
/// own soft-deinterleave, then call here.
///
/// 1. Soft-Golay-24 decode `c̃₀` (no PN) → `û₀`.
/// 2. Hard-demodulate `c̃₁` by flipping soft signs per the PN mask.
/// 3. Soft-Golay-23 decode `c̃₁` → `û₁`.
/// 4. Project `c̃₂` / `c̃₃` to hard bits for `û₂` / `û₃` (uncoded).
pub fn decode_code_vectors_soft(mut c: SoftCodeVectors) -> Frame {
    let d0 = golay_24_12_decode_soft(&c.c0);
    let u0 = d0.info;

    let masks = modulation_masks(u0);
    soft_demodulate_vector(&mut c.c1, masks[1]);
    let d1 = golay_23_12_decode_soft(&c.c1);

    let mut u2 = 0u16;
    for (i, &s) in c.c2.iter().enumerate() {
        if s > 0 {
            u2 |= 1u16 << (11 - 1 - i);
        }
    }
    let mut u3 = 0u16;
    for (i, &s) in c.c3.iter().enumerate() {
        if s > 0 {
            u3 |= 1u16 << (14 - 1 - i);
        }
    }

    Frame {
        info: [d0.info, d1.info, u2, u3],
        errors: [d0.errors, d1.errors, 0, 0],
    }
}

/// Soft-decision decode of a half-rate AMBE+2 frame from 72 P25-Phase-2
/// soft bits.
///
/// Input convention matches the full-rate soft entry point:
/// `[hi_dibit_0, lo_dibit_0, …]`, sign = hard decision, magnitude =
/// confidence. Thin Annex-S adapter over [`decode_code_vectors_soft`]:
/// 1. Soft-deinterleave 72 bits → `c̃₀..c̃₃` (Annex S — P25-Phase-2-specific).
/// 2. Run the shared soft codec FEC core ([`decode_code_vectors_soft`]).
pub fn decode_frame_soft(soft: &[i8; SOFT_BITS]) -> Frame {
    decode_code_vectors_soft(soft_deinterleave(soft))
}

/// Encode 4 info vectors into 72 half-rate air-interface bits
/// (36 dibits). Inverse of [`decode_frame`].
pub fn encode_frame(info: &[u16; 4]) -> [u8; DIBITS_PER_FRAME] {
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

    let masks = modulation_masks(u[0]);
    let c = [v0, v1 ^ masks[1], v2, v3];
    interleave(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_mask(i: usize) -> u32 {
        let len = CODE_WIDTHS[i] as u32;
        if len == 32 {
            u32::MAX
        } else {
            (1u32 << len) - 1
        }
    }

    #[test]
    fn widths_sum_to_49_and_72() {
        assert_eq!(INFO_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 49);
        assert_eq!(CODE_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 72);
        assert_eq!(INFO_BITS_TOTAL, 49);
    }

    #[test]
    fn r34_bit_order_is_a_bijection() {
        let mut seen = [false; INFO_BITS_TOTAL as usize];
        for &p in R34_BIT_ORDER.iter() {
            assert!(
                (p as usize) < INFO_BITS_TOTAL as usize,
                "index {p} out of range"
            );
            assert!(!seen[p as usize], "natural bit {p} mapped twice");
            seen[p as usize] = true;
        }
        assert!(
            seen.iter().all(|&b| b),
            "R34_BIT_ORDER does not cover all 49 bits"
        );
    }

    #[test]
    fn r34_pack_unpack_roundtrip() {
        // Exercise the full field range; the top field is 14 bits.
        for seed in [0u16, 1, 0x555, 0xAAA, 0x1FFF, 0x3FFF] {
            let info = [
                seed & 0x0FFF,
                seed.rotate_left(3) & 0x0FFF,
                seed.rotate_left(5) & 0x07FF,
                seed.rotate_left(7) & 0x3FFF,
            ];
            let bytes = pack_no_fec(&info);
            // Pad bits 49..55 must be zero.
            assert_eq!(
                bytes[6] & 0x7F,
                0,
                "trailing pad bits not zero for {info:?}"
            );
            assert_eq!(unpack_no_fec(&bytes), info, "roundtrip failed for {info:?}");
        }
    }

    #[test]
    fn r34_matches_dvsi_first_bit_is_u0_msb() {
        // R34 bit 0 (MSB of byte 0) is the MSB of û₀ — the start of the
        // DVSI 3-way interleave. Guards against accidental reversion to
        // the naive sequential layout (which also starts with û₀ MSB, so
        // also pin bit 1 = û₁ bit 5 and bit 2 = û₃ bit 12).
        assert_eq!(R34_BIT_ORDER[0], 0); // û₀[11]
        assert_eq!(R34_BIT_ORDER[1], 18); // û₁[5]
        assert_eq!(R34_BIT_ORDER[2], 36); // û₃[12]
        let info = [0x800u16, 0, 0, 0]; // only û₀ MSB set
        assert_eq!(pack_no_fec(&info)[0] & 0x80, 0x80);
    }

    #[test]
    fn interleave_covers_every_codeword_bit_exactly_once() {
        let mut seen = [[false; 24]; 4];
        for entry in ANNEX_S.iter() {
            for (v, i) in [
                (entry.bit1_vec, entry.bit1_idx),
                (entry.bit0_vec, entry.bit0_idx),
            ] {
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
        let dibits = interleave(&cw);
        for d in dibits.iter() {
            assert!(*d < 4);
        }
        let back = deinterleave(&dibits);
        assert_eq!(back, cw);
    }

    #[test]
    fn single_bit_propagates_to_one_dibit_position() {
        for v in 0..4 {
            for idx in 0..CODE_WIDTHS[v] {
                let mut cw = [0u32; 4];
                cw[v] = 1u32 << idx;
                let dibits = interleave(&cw);
                let ones: u32 = dibits.iter().map(|d| u32::from(*d).count_ones()).sum();
                assert_eq!(ones, 1, "(v={v}, idx={idx})");
                assert_eq!(deinterleave(&dibits), cw);
            }
        }
    }

    #[test]
    fn pn_seed_ambe_plus2_matches_formula() {
        for u0 in [0u16, 1, 42, 4095] {
            assert_eq!(pn_sequence(u0)[0], u0.wrapping_mul(16));
        }
        let pr = pn_sequence(1);
        assert_eq!(pr[0], 16);
        // pr[1] = (173·16 + 13849) mod 65536 = 16617
        assert_eq!(pr[1], 16617);
    }

    #[test]
    fn modulation_masks_m0_m2_m3_always_zero() {
        for u0 in [0u16, 1, 123, 4095] {
            let masks = modulation_masks(u0);
            assert_eq!(masks[0], 0);
            assert_eq!(masks[2], 0);
            assert_eq!(masks[3], 0);
        }
    }

    #[test]
    fn modulation_masks_m1_fits_within_23_bits() {
        for u0 in [0u16, 1, 123, 4095] {
            let masks = modulation_masks(u0);
            assert!(masks[1] < (1 << 23));
        }
    }

    #[test]
    fn demodulate_is_self_inverse() {
        let u0 = 0xA5Cu16;
        let masks = modulation_masks(u0);
        let v: [u32; 4] = [0x123456, 0x654321, 0x7A5, 0x2F2F];
        let c: [u32; 4] = core::array::from_fn(|i| v[i] ^ masks[i]);
        let v2 = demodulate(c, u0);
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
            let u = if seed == 0 {
                [0u16; 4]
            } else {
                sample_info(seed)
            };
            let dibits = encode_frame(&u);
            for (s, d) in dibits.iter().enumerate() {
                assert!(*d < 4, "symbol {s} not a dibit (seed {seed:08x})");
            }
            let back = decode_frame(&dibits);
            assert_eq!(back.info, u, "seed 0x{seed:08x}");
            assert_eq!(back.errors, [0u8; 4], "seed 0x{seed:08x}");
        }
    }

    #[test]
    fn single_bit_flip_in_c0_is_corrected() {
        // c₀ is [24,12] extended Golay → corrects any single bit error.
        let u = sample_info(0xA5A5A5A5);
        let clean = encode_frame(&u);
        for bit in 0..24 {
            let c = deinterleave(&clean);
            let mut c_flipped = c;
            c_flipped[0] ^= 1u32 << bit;
            let dibits = interleave(&c_flipped);
            let out = decode_frame(&dibits);
            assert_eq!(out.info, u, "c̃₀ bit {bit}");
            assert_eq!(out.errors[0], 1);
        }
    }

    #[test]
    fn single_bit_flip_in_c1_is_corrected() {
        let u = sample_info(0x5A5A5A5A);
        let clean = encode_frame(&u);
        for bit in 0..23 {
            let c = deinterleave(&clean);
            let mut c_flipped = c;
            c_flipped[1] ^= 1u32 << bit;
            let dibits = interleave(&c_flipped);
            let out = decode_frame(&dibits);
            assert_eq!(out.info, u, "c̃₁ bit {bit}");
            assert_eq!(out.errors[1], 1);
        }
    }

    #[test]
    fn uncoded_vectors_have_no_error_correction() {
        let u = sample_info(0xFACEFEED);
        let clean = encode_frame(&u);
        let c = deinterleave(&clean);
        // Flip bit in c₂ — passes through to û₂.
        let mut c_flip2 = c;
        c_flip2[2] ^= 1u32 << 5;
        let out2 = decode_frame(&interleave(&c_flip2));
        assert_eq!(out2.info[2], u[2] ^ 0b10_0000);
        assert_eq!(out2.errors[2], 0);
        // Flip bit in c₃ — passes through to û₃.
        let mut c_flip3 = c;
        c_flip3[3] ^= 1u32 << 11;
        let out3 = decode_frame(&interleave(&c_flip3));
        assert_eq!(out3.info[3], u[3] ^ (1 << 11));
        assert_eq!(out3.errors[3], 0);
    }

    // ---- Annex L / M / N / O / P / Q / R spot checks ---------------------

    #[test]
    fn annex_l_spot_values_match_spec() {
        // Annex L stores cycles/sample on disk; the build script
        // converts to rad/sample at load (× 2π) so values here are in
        // the same units as MbeParams. b₀=0 → (L=9, 0.049971 c/s →
        // 0.313977 rad/s); b₀=119 → (L=56, 0.008125 c/s → 0.051051
        // rad/s). See impl-spec §12.8 + analysis §13.
        use core::f32::consts::PI;
        assert_eq!(AMBE_PITCH_TABLE.len(), 120);
        let p0 = AMBE_PITCH_TABLE[0];
        assert_eq!(p0.l, 9);
        assert!((p0.omega_0 - 0.049971 * 2.0 * PI).abs() < 1e-5);
        let p_last = AMBE_PITCH_TABLE[119];
        assert_eq!(p_last.l, 56);
        assert!((p_last.omega_0 - 0.008125 * 2.0 * PI).abs() < 1e-5);
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
        for &v in &AMBE_VUV_CODEBOOK[0] {
            assert!(v);
        }
        // b₁ = 16 → all 8 unvoiced.
        for &v in &AMBE_VUV_CODEBOOK[16] {
            assert!(!v);
        }
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
        let dibits = encode_frame(&u);
        let back = decode_frame(&dibits);
        for i in 0..4 {
            let w = INFO_WIDTHS[i] as u32;
            let m = ((1u32 << w) - 1) as u16;
            assert_eq!(back.info[i], m, "vector {i} mask mismatch");
        }
    }

    // ---- Soft-decision path (Gap B) -------------------------------------

    fn dibits_to_soft(dibits: &[u8; DIBITS_PER_FRAME], confidence: i8) -> [i8; SOFT_BITS] {
        let mut out = [0i8; SOFT_BITS];
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
                [0u16; 4]
            } else {
                sample_info(seed)
            };
            let dibits = encode_frame(&u);
            let soft = dibits_to_soft(&dibits, 120);
            let soft_frame = decode_frame_soft(&soft);
            let hard_frame = decode_frame(&dibits);
            assert_eq!(soft_frame.info, hard_frame.info, "seed 0x{seed:08x}");
            assert_eq!(soft_frame.info, u, "seed 0x{seed:08x}");
            assert_eq!(soft_frame.errors, [0u8; 4]);
        }
    }

    #[test]
    fn soft_decode_recovers_weak_bit_error_on_c0() {
        // Extended Golay(24,12) only corrects 1 error. Flipping 2 bits
        // leaves hard decode to miscorrect or flag uncorrectable; soft
        // with low confidence on both should recover via Chase-II
        // flipping over the least-reliable positions.
        let u = sample_info(0xA5A5A5A5);
        let dibits = encode_frame(&u);
        let mut soft = dibits_to_soft(&dibits, 120);
        // Weaken and flip 2 positions inside c₀ (MSB-first 0 and 5).
        let targets = [(0u8, 0u8), (0, 5)]; // (vec_idx, msb_pos)
        for (vi, msb_pos) in targets {
            let lsb_pos = CODE_WIDTHS[vi as usize] - 1 - msb_pos;
            let (sym, is_hi) = (0..ANNEX_S.len())
                .find_map(|s| {
                    let e = ANNEX_S[s];
                    if e.bit1_vec == vi && e.bit1_idx == lsb_pos {
                        Some((s, true))
                    } else if e.bit0_vec == vi && e.bit0_idx == lsb_pos {
                        Some((s, false))
                    } else {
                        None
                    }
                })
                .expect("annex S covers every (vec, idx)");
            let soft_idx = 2 * sym + if is_hi { 0 } else { 1 };
            soft[soft_idx] = -soft[soft_idx].signum() * 4;
        }
        let frame = decode_frame_soft(&soft);
        assert_eq!(frame.info, u, "soft recovered 2-error c̃₀");
    }
}
