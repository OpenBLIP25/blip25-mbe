//! IMBE-specific FEC layout — which bits are covered by which code,
//! how PN modulation is composed on top, and how vectors are assembled
//! into a 144-bit frame.
//!
//! Per TIA-102.BABA-A. This module composes the generic Golay/Hamming
//! primitives in [`crate::fec`] into the full-rate wire layout. The
//! Annex H deinterleaving lives alongside this module, in
//! [`crate::imbe_frames::full_rate`] once added.
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
pub const PN_SEQ_LEN_FULLRATE: usize = 115;

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
pub fn pn_sequence_fullrate(u0: u16) -> [u16; PN_SEQ_LEN_FULLRATE] {
    debug_assert!(u0 < 4096, "û₀ is a 12-bit Golay info word");
    let mut pr = [0u16; PN_SEQ_LEN_FULLRATE];
    pr[0] = u0.wrapping_mul(16);
    for n in 1..PN_SEQ_LEN_FULLRATE {
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
/// `0..VECTOR_LENGTHS[i]` (element `k` at bit `k`, matching the codeword
/// convention above).
///
/// * `m̂₀` and `m̂₇` are always zero (Eqs. 86, 93).
/// * `m̂₁..m̂₃` each consume 23 PN indices at offsets 1, 24, 47.
/// * `m̂₄..m̂₆` each consume 15 PN indices at offsets 70, 85, 100.
///
/// Total PN indices consumed: 114, matching the sum of the six modulated
/// vector lengths.
pub fn modulation_masks_fullrate(u0: u16) -> [u32; 8] {
    let pr = pn_sequence_fullrate(u0);
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
            m |= pn_mask_bit(pr[start + k]) << k;
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
pub fn demodulate_fullrate(codewords: [u32; 8], u0: u16) -> [u32; 8] {
    let masks = modulation_masks_fullrate(u0);
    let mut v = [0u32; 8];
    for i in 0..8 {
        v[i] = codewords[i] ^ masks[i];
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pn_seed_zero_is_all_zero_run_of_lcg() {
        // u0 = 0 → pr[0] = 0, pr[1] = (0 + 13849) & 0xFFFF = 13849
        let pr = pn_sequence_fullrate(0);
        assert_eq!(pr[0], 0);
        assert_eq!(pr[1], 13849);
        assert_eq!(pr[2], ((173u32 * 13849 + 13849) & 0xFFFF) as u16);
    }

    #[test]
    fn pn_seed_is_16_times_u0() {
        for u0 in [0u16, 1, 7, 255, 4095] {
            assert_eq!(pn_sequence_fullrate(u0)[0], u0 * 16);
        }
    }

    #[test]
    fn pn_sequence_matches_recurrence_by_hand() {
        // u0 = 1 → pr[0] = 16, pr[1] = (173*16 + 13849) & 0xFFFF = 16617
        let pr = pn_sequence_fullrate(1);
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
            let m = modulation_masks_fullrate(u0);
            assert_eq!(m[0], 0, "m̂₀ must be zero (Eq. 86), u0={u0}");
            assert_eq!(m[7], 0, "m̂₇ must be zero (Eq. 93), u0={u0}");
        }
    }

    #[test]
    fn masks_fit_within_vector_lengths() {
        for u0 in [0u16, 1, 42, 4095] {
            let m = modulation_masks_fullrate(u0);
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
        let masks = modulation_masks_fullrate(u0);
        let v: [u32; 8] = [0x123, 0x456, 0x789, 0xABC, 0x111, 0x222, 0x333, 0x44];
        let c: [u32; 8] = std::array::from_fn(|i| v[i] ^ masks[i]);
        let v2 = demodulate_fullrate(c, u0);
        assert_eq!(v2, v);
    }

    #[test]
    fn different_u0_produce_different_masks() {
        // Sanity: changing u0 changes the non-zero masks.
        let m1 = modulation_masks_fullrate(1);
        let m2 = modulation_masks_fullrate(2);
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
        assert_eq!(PN_SEQ_LEN_FULLRATE - 1, 114);
    }
}
