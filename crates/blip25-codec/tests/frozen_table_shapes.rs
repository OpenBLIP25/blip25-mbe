//! Shape assertions for the frozen Annex tables in `src/tables_generated.rs`.
//!
//! Rust's array-length types already carry part of the guarantee: a
//! `[PitchEntry; 120]` literal with 119 entries does not compile. What the type
//! system does NOT check is that those lengths are the ones the codec's indexing
//! actually assumes, nor that the contents are structurally what their consumers
//! require — Annex S a true 72-slot permutation, the bit prioritization map a
//! bijection over 49 bits that round-trips through `deprioritize`, Annex N block
//! lengths summing to L for every L. This file pins those.
//!
//! These are structural invariants, not a hash of the contents against itself —
//! that would only restate what `git diff` shows.

use blip25_codec::tables::{
    deprioritize, AMBE_BIT_MAP, AMBE_BLOCK_LENGTHS, AMBE_PITCH_TABLE, AMBE_VUV_CODEBOOK, ANNEX_S,
};

#[test]
fn annex_l_pitch_table_is_the_full_voice_range() {
    assert_eq!(
        AMBE_PITCH_TABLE.len(),
        120,
        "Annex L is 120 voice pitch indices"
    );
    // L is monotonically non-decreasing as omega_0 falls, and spans 9..=56.
    assert_eq!(AMBE_PITCH_TABLE[0].l, 9, "first row is L=9");
    assert_eq!(AMBE_PITCH_TABLE[119].l, 56, "last row is L=56");
    for w in AMBE_PITCH_TABLE.windows(2) {
        assert!(w[1].l >= w[0].l, "L must not decrease across the table");
        assert!(
            w[1].omega_0 < w[0].omega_0,
            "omega_0 must strictly decrease across the table"
        );
    }
    for e in AMBE_PITCH_TABLE.iter() {
        assert!((9..=56).contains(&e.l), "L out of range: {}", e.l);
        assert!(e.omega_0 > 0.0 && e.omega_0 < 1.0, "omega_0 out of range");
    }
}

#[test]
fn annex_m_vuv_codebook_is_32_rows_of_8_bands() {
    assert_eq!(AMBE_VUV_CODEBOOK.len(), 32, "b1 is 5 bits");
    // Row 0 is the all-voiced anchor, row 16 the all-unvoiced one; the quantizer
    // relies on those being present for its tie-break behaviour.
    assert!(
        AMBE_VUV_CODEBOOK[0].iter().all(|&v| v),
        "row 0 must be all-voiced"
    );
    assert!(
        AMBE_VUV_CODEBOOK[16].iter().all(|&v| !v),
        "row 16 must be all-unvoiced"
    );
}

#[test]
fn annex_n_block_lengths_cover_every_l_and_sum_correctly() {
    assert_eq!(AMBE_BLOCK_LENGTHS.len(), 48, "L = 9..=56 is 48 rows");
    for (i, row) in AMBE_BLOCK_LENGTHS.iter().enumerate() {
        let l = i + 9;
        let sum: usize = row.iter().map(|&v| usize::from(v)).sum();
        assert_eq!(sum, l, "block lengths for L={l} must sum to L, got {sum}");
        assert!(row.iter().all(|&v| v > 0), "L={l}: zero-length block");
    }
}

#[test]
fn annex_s_interleave_is_a_36_dibit_permutation() {
    assert_eq!(ANNEX_S.len(), 36, "one entry per dibit");
    // Every (vector, bit) slot addressed exactly once across both dibit halves,
    // over the 12+12+11+14 = 49 information+parity bit positions of c0..c3.
    let mut seen = std::collections::HashSet::new();
    for e in ANNEX_S.iter() {
        assert!(
            seen.insert((e.bit1_vec, e.bit1_idx)),
            "duplicate slot ({}, {})",
            e.bit1_vec,
            e.bit1_idx
        );
        assert!(
            seen.insert((e.bit0_vec, e.bit0_idx)),
            "duplicate slot ({}, {})",
            e.bit0_vec,
            e.bit0_idx
        );
        assert!(
            e.bit1_vec < 4 && e.bit0_vec < 4,
            "vector index out of range"
        );
    }
    assert_eq!(seen.len(), 72, "36 dibits address 72 distinct bit slots");
}

#[test]
fn bit_prioritization_is_a_bijection_over_49_bits() {
    assert_eq!(AMBE_BIT_MAP.len(), 49, "49 information bits");
    let mut src = std::collections::HashSet::new();
    let mut dst = std::collections::HashSet::new();
    for m in AMBE_BIT_MAP.iter() {
        assert!(
            src.insert((m.src_param, m.src_bit)),
            "duplicate source bit b{}[{}]",
            m.src_param,
            m.src_bit
        );
        assert!(
            dst.insert((m.dst_vec, m.dst_bit)),
            "duplicate destination bit u{}[{}]",
            m.dst_vec,
            m.dst_bit
        );
        assert!(m.src_param < 9, "b index out of range");
        assert!(m.dst_vec < 4, "u index out of range");
    }
    assert_eq!(src.len(), 49);
    assert_eq!(dst.len(), 49);
}

/// `deprioritize` is the inverse of the prioritization map, so a round trip
/// through it must be the identity for every in-range parameter set.
#[test]
fn deprioritize_round_trips_the_bit_map() {
    // Widths of b0..b8: 7,5,5,9,7,5,4,4,3 = 49 bits.
    const W: [u32; 9] = [7, 5, 5, 9, 7, 5, 4, 4, 3];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..2000 {
        let mut b = [0u16; 9];
        for (i, &w) in W.iter().enumerate() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            b[i] = ((state >> 33) as u16) & ((1u16 << w) - 1);
        }
        // Forward: scatter b -> u using the same map deprioritize inverts.
        let mut u = [0u16; 4];
        for m in AMBE_BIT_MAP.iter() {
            let bit = (b[m.src_param as usize] >> m.src_bit) & 1;
            u[m.dst_vec as usize] |= bit << m.dst_bit;
        }
        assert_eq!(deprioritize(&u), b, "bit map is not a clean round trip");
    }
}
