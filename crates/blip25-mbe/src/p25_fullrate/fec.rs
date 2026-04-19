//! IMBE-specific FEC layout — which bits are covered by which code,
//! how PN modulation is composed on top, and how vectors are assembled
//! into a 144-bit frame.
//!
//! Per TIA-102.BABA-A. This module composes the generic Golay/Hamming
//! primitives in [`crate::fec`] into the full-rate wire layout and
//! carries the Annex H interleaving table (generated at build time
//! from `spec_tables/annex_h_interleave.csv` by `build.rs`).
//!
//! ## Bit-ordering convention (frame-wide)
//!
//! Each code vector is carried in a `u32` with **element `k` at bit `k`**
//! (LSB-first element ordering). For the 23-bit Golay codewords this
//! means bit 22 is the first-sent bit (the codeword's MSB). For the
//! 15-bit Hamming codewords, bit 14 is the MSB; for the 7-bit uncoded
//! vector, bit 6 is the MSB. Annex H uses matching index numbering
//! (`c0[22]` refers to `u32` bit 22 of `c0`), as does the PN mask
//! generator below.
//!
//! ## PN modulation (§1.6)
//!
//! After FEC encoding, every code vector except `v̂₀` and `v̂₇` is XORed
//! with a pseudo-noise mask derived from the 12-bit Golay info word `û₀`.
//! The decoder therefore Golay-decodes `c̃₀` first (since `m̂₀ = 0`),
//! recovers `û₀`, generates masks, then demodulates `c̃₁..c̃₆`. `c̃₇` is
//! passed through untouched because `m̂₇ = 0` (and `v̂₇` is uncoded).
//!
//! The seed uses **all 12 bits of `û₀`** (pitch `b₀` plus the spectral
//! MSBs that share the Golay vector), not only the 8-bit `b₀` pitch
//! value — see §1.6 correction note.

/// Length of the full-rate PN sequence: `p_r(0)` seed plus `p_r(1..=114)`
/// for the six non-zero masks (23 + 23 + 23 + 15 + 15 + 15 = 114 bits).
pub const PN_SEQ_LEN: usize = 115;

/// Lengths of the 8 modulation vectors in bits.
/// Matches the full-rate FEC layout: c0..c3 = 23-bit Golay,
/// c4..c6 = 15-bit Hamming, c7 = 7-bit uncoded.
pub const VECTOR_LENGTHS: [u8; 8] = [23, 23, 23, 23, 15, 15, 15, 7];

/// Generate the full-rate PN sequence `p_r(0..=114)`.
///
/// Per BABA-A §1.6 Eqs. 84–85:
///
/// ```text
/// p_r(0) = 16 · û₀                              û₀ ∈ [0, 4095]
/// p_r(n) = (173 · p_r(n-1) + 13849) mod 65536   1 ≤ n ≤ 114
/// ```
///
/// Each element is a 16-bit unsigned integer. The mask bit for position
/// `n` is the MSB of `p_r(n)` (bit 15); see [`pn_mask_bit`].
///
/// Panics in debug builds if `u0 >= 4096`.
pub fn pn_sequence(u0: u16) -> [u16; PN_SEQ_LEN] {
    debug_assert!(u0 < 4096, "û₀ is a 12-bit Golay info word");
    let mut pr = [0u16; PN_SEQ_LEN];
    pr[0] = u0.wrapping_mul(16);
    for n in 1..PN_SEQ_LEN {
        pr[n] = (173u32
            .wrapping_mul(pr[n - 1] as u32)
            .wrapping_add(13849)
            & 0xFFFF) as u16;
    }
    pr
}

/// Extract one PN mask bit from `p_r(n)`: bit 15 (MSB) of the 16-bit value.
///
/// Per BABA-A §1.6 Eqs. 86–93: `m = ⌊p_r(n) / 32768⌋`.
#[inline]
pub fn pn_mask_bit(pr_n: u16) -> u32 {
    (pr_n >> 15) as u32
}

/// Compute the 8 PN modulation masks `m̂₀..m̂₇` for full-rate IMBE.
///
/// Returns one `u32` per vector with mask bits packed at positions
/// `0..VECTOR_LENGTHS[i]`. The first PN value `p_r(start)` lands at
/// the highest used bit (`len-1`); the last PN value `p_r(start+len-1)`
/// lands at bit 0.
///
/// **PN-to-codeword alignment.** In our codeword storage convention,
/// `u32` bit `(len-1)` is the first-transmitted (MSB) bit of the
/// codeword (per the Annex H deinterleaver: row 0 routes
/// `c0[22]` — the first dibit's high bit — to `u32` bit 22). Aligning
/// the first PN value with the first-transmitted bit means the PN
/// advances *with* transmission order, which is what DVSI's encoder
/// produces. Validated empirically against the DVSI `tv-std/tv/p25/`
/// reference vectors: every recovered DVSI mask matches this packing.
///
/// Note: the spec's §1.6 C code documents `MASK_RANGE` with element-k-
/// at-bit-k packing, which is self-consistent but differs from the
/// codeword bit ordering at the wire boundary. The disagreement is
/// invisible in encode→decode roundtrips that share both ends.
///
/// * `m̂₀` and `m̂₇` are always zero (Eqs. 86, 93).
/// * `m̂₁..m̂₃` each consume 23 PN indices at offsets 1, 24, 47.
/// * `m̂₄..m̂₆` each consume 15 PN indices at offsets 70, 85, 100.
///
/// Total PN indices consumed: 114, matching the sum of the six modulated
/// vector lengths.
pub fn modulation_masks(u0: u16) -> [u32; 8] {
    let pr = pn_sequence(u0);
    let mut masks = [0u32; 8];

    // (vector index, PN start offset, length)
    const LAYOUT: [(usize, usize, usize); 6] = [
        (1, 1, 23),
        (2, 24, 23),
        (3, 47, 23),
        (4, 70, 15),
        (5, 85, 15),
        (6, 100, 15),
    ];

    for (vec_idx, start, len) in LAYOUT {
        let mut m = 0u32;
        for k in 0..len {
            // First PN value at highest bit (= first-transmitted bit).
            m |= pn_mask_bit(pr[start + k]) << (len - 1 - k);
        }
        masks[vec_idx] = m;
    }
    masks
}

/// Demodulate a full-rate frame's 8 encoded vectors `c̃₀..c̃₇` to recover
/// `v̂₀..v̂₇`.
///
/// The caller must first obtain `û₀` by Golay-decoding `c̃₀` (since
/// `m̂₀ = 0`, `c̃₀` is identical to `v̂₀` and can be decoded directly by
/// [`crate::fec::golay_23_12_decode`]). This function then XORs `c̃₁..c̃₆`
/// with their masks and returns `v̂₀..v̂₇` (with `v̂₀ = c̃₀` and
/// `v̂₇ = c̃₇` passed through).
pub fn demodulate(codewords: [u32; 8], u0: u16) -> [u32; 8] {
    let masks = modulation_masks(u0);
    let mut v = [0u32; 8];
    for i in 0..8 {
        v[i] = codewords[i] ^ masks[i];
    }
    v
}

// ---------------------------------------------------------------------------
// Annex H — full-rate interleaving
// ---------------------------------------------------------------------------

/// One row of the Annex H interleaving table. Each of the 72 dibit
/// symbols in an IMBE frame carries two bits; this struct records which
/// codeword vector (0–7) and which bit index within that vector each
/// of the two bits belongs to. `bit1_*` is the dibit's high bit (sent
/// first), `bit0_*` is the low bit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnnexHEntry {
    pub bit1_vec: u8,
    pub bit1_idx: u8,
    pub bit0_vec: u8,
    pub bit0_idx: u8,
}

include!(concat!(env!("OUT_DIR"), "/annex_h.rs"));

/// Deinterleave a 144-bit IMBE frame (72 dibit symbols) into the 8
/// code vectors `c₀..c₇` per BABA-A Annex H.
///
/// Each input byte is a dibit in `0..=3`: bit 1 is the high (first-sent)
/// bit, bit 0 is the low bit. Output vector lengths are 23, 23, 23, 23,
/// 15, 15, 15, 7 bits respectively, packed LSB-first per the module-level
/// convention (element `k` at bit `k`).
pub fn deinterleave(dibits: &[u8; 72]) -> [u32; 8] {
    let mut c = [0u32; 8];
    for (sym, d) in dibits.iter().enumerate() {
        let entry = ANNEX_H[sym];
        let hi = u32::from((d >> 1) & 1);
        let lo = u32::from(d & 1);
        c[entry.bit1_vec as usize] |= hi << entry.bit1_idx;
        c[entry.bit0_vec as usize] |= lo << entry.bit0_idx;
    }
    c
}

/// Interleave 8 code vectors `c₀..c₇` into a 144-bit IMBE frame
/// (72 dibit symbols) per BABA-A Annex H.
///
/// Inverse of [`deinterleave`]. Useful for encode paths and
/// roundtrip tests.
pub fn interleave(codewords: &[u32; 8]) -> [u8; 72] {
    let mut dibits = [0u8; 72];
    for (sym, entry) in ANNEX_H.iter().enumerate() {
        let hi = ((codewords[entry.bit1_vec as usize] >> entry.bit1_idx) & 1) as u8;
        let lo = ((codewords[entry.bit0_vec as usize] >> entry.bit0_idx) & 1) as u8;
        dibits[sym] = (hi << 1) | lo;
    }
    dibits
}

/// Total soft-bit count for a full-rate IMBE frame (72 dibits × 2).
pub const SOFT_BITS: usize = 144;

/// Soft-deinterleave a 144-bit soft stream into 8 MSB-first soft code
/// vectors sized per [`VECTOR_LENGTHS`]: four 23-bit, three 15-bit,
/// one 7-bit.
///
/// Input layout: `soft[2*sym]` is the high bit of dibit `sym`,
/// `soft[2*sym + 1]` is the low bit. This matches the `hi,lo` pairing
/// produced by the hard [`deinterleave`] + the common C4FM dibit
/// convention.
///
/// Each output vector is MSB-first so it can be fed directly into the
/// soft FEC decoders ([`crate::fec::golay_23_12_decode_soft`] /
/// [`crate::fec::hamming_15_11_decode_soft`]) without further
/// re-ordering.
///
/// Returns three arrays sized for the three wire-format regimes
/// (Golay-23, Hamming-15, uncoded) rather than a ragged `[[i8; W]; 8]`
/// so the caller can pass each vector directly to the appropriately-
/// typed soft decoder without slicing.
pub fn soft_deinterleave(soft: &[i8; SOFT_BITS]) -> SoftCodeVectors {
    let mut out = SoftCodeVectors::default();
    for (sym, entry) in ANNEX_H.iter().enumerate() {
        let hi = soft[2 * sym];
        let lo = soft[2 * sym + 1];
        place_soft(&mut out, entry.bit1_vec as usize, entry.bit1_idx as usize, hi);
        place_soft(&mut out, entry.bit0_vec as usize, entry.bit0_idx as usize, lo);
    }
    out
}

/// Place a soft bit into its MSB-first position inside the correct
/// output vector. `vec_idx` is 0..=7 per [`VECTOR_LENGTHS`]; `idx` is
/// the LSB-first element index from Annex H.
fn place_soft(out: &mut SoftCodeVectors, vec_idx: usize, idx: usize, s: i8) {
    let width = VECTOR_LENGTHS[vec_idx] as usize;
    let msb_pos = width - 1 - idx;
    match vec_idx {
        0..=3 => out.golay[vec_idx][msb_pos] = s,
        4..=6 => out.hamming[vec_idx - 4][msb_pos] = s,
        7 => out.uncoded[msb_pos] = s,
        _ => unreachable!(),
    }
}

/// Soft-deinterleaved code vectors for a full-rate IMBE frame. Four
/// 23-bit Golay-coded vectors (`c̃₀..c̃₃`), three 15-bit Hamming-coded
/// vectors (`c̃₄..c̃₆`), and one 7-bit uncoded vector (`c̃₇`). All
/// vectors are MSB-first.
#[derive(Clone, Copy, Debug)]
pub struct SoftCodeVectors {
    /// `c̃₀..c̃₃` — soft Golay-23 codewords, MSB at index 0.
    pub golay: [[i8; 23]; 4],
    /// `c̃₄..c̃₆` — soft Hamming-15 codewords, MSB at index 0.
    pub hamming: [[i8; 15]; 3],
    /// `c̃₇` — 7-bit uncoded, MSB at index 0.
    pub uncoded: [i8; 7],
}

impl Default for SoftCodeVectors {
    fn default() -> Self {
        Self { golay: [[0i8; 23]; 4], hamming: [[0i8; 15]; 3], uncoded: [0i8; 7] }
    }
}

/// XOR a 32-bit hard PN mask into a soft codeword vector, preserving
/// magnitudes. Each mask bit that is 1 flips the sign of the
/// corresponding soft bit; 0 leaves it alone.
///
/// `mask` is LSB-first (element `k` at bit `k`) to match the hard-
/// modulation convention in [`modulation_masks`]. `soft` is MSB-first
/// of width `W`: position 0 of `soft` corresponds to bit `W - 1` of
/// `mask`.
pub fn soft_demodulate_vector<const W: usize>(soft: &mut [i8; W], mask: u32) {
    for (i, s) in soft.iter_mut().enumerate() {
        let mask_bit = (mask >> (W - 1 - i)) & 1;
        if mask_bit == 1 {
            *s = s.saturating_neg();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pn_seed_zero_is_all_zero_run_of_lcg() {
        // u0 = 0 → pr[0] = 0, pr[1] = (0 + 13849) & 0xFFFF = 13849
        let pr = pn_sequence(0);
        assert_eq!(pr[0], 0);
        assert_eq!(pr[1], 13849);
        assert_eq!(pr[2], ((173u32 * 13849 + 13849) & 0xFFFF) as u16);
    }

    #[test]
    fn pn_seed_is_16_times_u0() {
        for u0 in [0u16, 1, 7, 255, 4095] {
            assert_eq!(pn_sequence(u0)[0], u0 * 16);
        }
    }

    #[test]
    fn pn_sequence_matches_recurrence_by_hand() {
        // u0 = 1 → pr[0] = 16, pr[1] = (173*16 + 13849) & 0xFFFF = 16617
        let pr = pn_sequence(1);
        assert_eq!(pr[0], 16);
        assert_eq!(pr[1], 16617);
        // pr[2] = (173*16617 + 13849) mod 65536
        let expected_2 = ((173u32 * 16617 + 13849) & 0xFFFF) as u16;
        assert_eq!(pr[2], expected_2);
    }

    #[test]
    fn mask_bit_is_msb_of_pr_value() {
        assert_eq!(pn_mask_bit(0), 0);
        assert_eq!(pn_mask_bit(0x7FFF), 0);
        assert_eq!(pn_mask_bit(0x8000), 1);
        assert_eq!(pn_mask_bit(0xFFFF), 1);
    }

    #[test]
    fn masks_zero_and_seven_are_zero() {
        for u0 in [0u16, 1, 42, 4095] {
            let m = modulation_masks(u0);
            assert_eq!(m[0], 0, "m̂₀ must be zero (Eq. 86), u0={u0}");
            assert_eq!(m[7], 0, "m̂₇ must be zero (Eq. 93), u0={u0}");
        }
    }

    #[test]
    fn masks_fit_within_vector_lengths() {
        for u0 in [0u16, 1, 42, 4095] {
            let m = modulation_masks(u0);
            for i in 0..8 {
                let len = VECTOR_LENGTHS[i] as u32;
                let max_mask = if len == 32 { u32::MAX } else { (1u32 << len) - 1 };
                assert!(
                    m[i] <= max_mask,
                    "mask {i} has bit beyond vector length {len} (u0={u0}, m=0x{:x})",
                    m[i]
                );
            }
        }
    }

    #[test]
    fn modulation_is_self_inverse() {
        // v̂ ⊕ m̂ = c̃, c̃ ⊕ m̂ = v̂. The demod step should recover the
        // original vectors when applied twice (or equivalently, when
        // applied to `v ⊕ m` given the same u0).
        let u0 = 0xABC;
        let masks = modulation_masks(u0);
        let v: [u32; 8] = [0x123, 0x456, 0x789, 0xABC, 0x111, 0x222, 0x333, 0x44];
        let c: [u32; 8] = std::array::from_fn(|i| v[i] ^ masks[i]);
        let v2 = demodulate(c, u0);
        assert_eq!(v2, v);
    }

    #[test]
    fn different_u0_produce_different_masks() {
        // Sanity: changing u0 changes the non-zero masks.
        let m1 = modulation_masks(1);
        let m2 = modulation_masks(2);
        // At least one of m1..m6 differs.
        let any_diff = (1..=6).any(|i| m1[i] != m2[i]);
        assert!(any_diff, "masks should depend on u0");
    }

    #[test]
    fn pn_consumes_exactly_114_indices_for_masks() {
        // Sanity: total mask bits across m1..m6 = 114, matching the
        // number of non-seed PN indices.
        let total: u8 = [1u8, 2, 3, 4, 5, 6].iter().map(|&i| VECTOR_LENGTHS[i as usize]).sum();
        assert_eq!(total, 114);
        assert_eq!(PN_SEQ_LEN - 1, 114);
    }

    // ---- Annex H interleaving ---------------------------------------------

    /// Mask that keeps only the valid bits of codeword `i`.
    fn valid_mask(i: usize) -> u32 {
        let len = VECTOR_LENGTHS[i] as u32;
        if len == 32 { u32::MAX } else { (1u32 << len) - 1 }
    }

    #[test]
    fn interleave_table_covers_every_codeword_bit_exactly_once() {
        // 144 dibit bits should map 1:1 onto (vector, bit_index) across
        // the 8 vectors' 23+23+23+23+15+15+15+7 = 144 positions.
        let mut seen = [[false; 23]; 8];
        for entry in ANNEX_H.iter() {
            for (v, i) in [(entry.bit1_vec, entry.bit1_idx), (entry.bit0_vec, entry.bit0_idx)] {
                assert!((i as u8) < VECTOR_LENGTHS[v as usize]);
                assert!(!seen[v as usize][i as usize], "double coverage at ({v}, {i})");
                seen[v as usize][i as usize] = true;
            }
        }
        for (v, row) in seen.iter().enumerate() {
            for (i, &b) in row.iter().enumerate().take(VECTOR_LENGTHS[v] as usize) {
                assert!(b, "({v}, {i}) never covered");
            }
        }
    }

    #[test]
    fn interleave_deinterleave_roundtrip() {
        // Deterministic pseudo-random codewords within valid widths.
        let mut cw = [0u32; 8];
        let mut state = 0xCAFEBABEu32;
        for i in 0..8 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            cw[i] = state & valid_mask(i);
        }
        let dibits = interleave(&cw);
        // All dibits must be in 0..=3.
        for (s, d) in dibits.iter().enumerate() {
            assert!(*d < 4, "symbol {s} produced {d} (not a dibit)");
        }
        let recovered = deinterleave(&dibits);
        assert_eq!(recovered, cw);
    }

    #[test]
    fn deinterleave_interleave_roundtrip() {
        // Other direction: arbitrary 72-dibit stream → 8 codewords →
        // re-interleave → same stream.
        let mut dibits = [0u8; 72];
        let mut state = 0xDEADBEEFu32;
        for d in dibits.iter_mut() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *d = (state >> 30) as u8; // top 2 bits, avoids LSB bias
        }
        let cw = deinterleave(&dibits);
        // Sanity: every decoded codeword fits within its declared width.
        for (i, c) in cw.iter().enumerate() {
            assert_eq!(c & !valid_mask(i), 0, "c[{i}] has bits beyond its width");
        }
        let re = interleave(&cw);
        assert_eq!(re, dibits);
    }

    #[test]
    fn single_bit_propagates_to_one_dibit_position() {
        // Setting exactly one bit in one codeword should produce a
        // frame with exactly one '1' dibit-bit set.
        for v in 0..8 {
            for idx in 0..VECTOR_LENGTHS[v] {
                let mut cw = [0u32; 8];
                cw[v] = 1u32 << idx;
                let dibits = interleave(&cw);
                let ones: u32 = dibits.iter().map(|d| u32::from(*d).count_ones()).sum();
                assert_eq!(
                    ones, 1,
                    "single bit at (v={v}, idx={idx}) produced {ones} dibit-bits"
                );
                assert_eq!(deinterleave(&dibits), cw);
            }
        }
    }
}
