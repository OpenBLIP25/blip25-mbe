//! Bit prioritization — the layer between the AMBE+2 info vectors
//! `û_i` and the quantized voice parameters `b̂_l`.
//!
//! Per BABA-A §16.7, Tables 15–18. The mapping is fixed (L-independent)
//! because the half-rate pitch index encodes the `(L, ω₀)` pair via
//! Annex L rather than occupying variable-width coefficient fields.
//!
//! The table is generated at build time from `spec_tables/ambe_bit_prioritization.csv`,
//! with coverage invariants checked during generation.

use crate::bits::BitMap;

include!(concat!(env!("OUT_DIR"), "/ambe_bit_priority.rs"));

/// Number of half-rate parameters: `b̂₀..b̂₈`.
pub const AMBE_B_COUNT: usize = 9;

/// Half-rate parameter widths in bits, indexed by `src_param`.
/// Source: BABA-A §16.7 / `ambe_bit_prioritization.csv` header:
/// `b̂₀=7, b̂₁=5, b̂₂=5, b̂₃=9, b̂₄=7, b̂₅=5, b̂₆=4, b̂₇=4, b̂₈=3`; sum = 49.
pub const AMBE_PARAM_WIDTHS: [u8; AMBE_B_COUNT] = [7, 5, 5, 9, 7, 5, 4, 4, 3];

/// Half-rate info vector widths. Sum = 49. Note the spec's §2.3 table
/// wrote these as 12/12/12/13; the CSV corrects to 12/12/11/14 with
/// coverage invariants enforced at generation time.
pub const AMBE_VECTOR_WIDTHS: [u8; 4] = [12, 12, 11, 14];

/// Prioritize a quantized half-rate parameter array `b̂₀..b̂₈`
/// into the 4 info vectors `û₀..û₃`.
pub fn prioritize(b: &[u16; AMBE_B_COUNT]) -> [u16; 4] {
    let mut u = [0u16; 4];
    for m in &AMBE_BIT_MAP {
        let bit = (b[m.src_param as usize] >> m.src_bit) & 1;
        u[m.dst_vec as usize] |= bit << m.dst_bit;
    }
    u
}

/// Deprioritize 4 info vectors `û₀..û₃` into a quantized parameter
/// array `b̂₀..b̂₈`.
pub fn deprioritize(u: &[u16; 4]) -> [u16; AMBE_B_COUNT] {
    let mut b = [0u16; AMBE_B_COUNT];
    for m in &AMBE_BIT_MAP {
        let bit = (u[m.dst_vec as usize] >> m.dst_bit) & 1;
        b[m.src_param as usize] |= bit << m.src_bit;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_b(seed: u32) -> [u16; AMBE_B_COUNT] {
        let mut b = [0u16; AMBE_B_COUNT];
        let mut state = seed;
        for i in 0..AMBE_B_COUNT {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = AMBE_PARAM_WIDTHS[i] as u32;
            let mask = ((1u32 << w) - 1) as u16;
            b[i] = (state as u16) & mask;
        }
        b
    }

    #[test]
    fn roundtrip_sampled() {
        for seed in [1u32, 0xDEADBEEF, 0xCAFEBABE, 0xA5A5A5A5, 42] {
            let b = sample_b(seed);
            let u = prioritize(&b);
            for (i, &w) in AMBE_VECTOR_WIDTHS.iter().enumerate() {
                let mask = ((1u16 << w) - 1) as u16;
                assert_eq!(u[i] & !mask, 0,
                    "seed 0x{seed:08x}: û{i} has bits beyond width {w}");
            }
            let b2 = deprioritize(&u);
            assert_eq!(b2, b, "seed 0x{seed:08x}");
        }
    }

    #[test]
    fn zero_prioritizes_to_zero() {
        let u = prioritize(&[0u16; AMBE_B_COUNT]);
        assert_eq!(u, [0u16; 4]);
    }

    #[test]
    fn single_bit_propagation() {
        for m in AMBE_BIT_MAP.iter() {
            let mut b = [0u16; AMBE_B_COUNT];
            b[m.src_param as usize] = 1 << m.src_bit;
            let u = prioritize(&b);
            let ones: u32 = u.iter().map(|x| x.count_ones()).sum();
            assert_eq!(ones, 1);
            let b2 = deprioritize(&u);
            assert_eq!(b2, b);
        }
    }

    #[test]
    fn widths_sum_to_49() {
        assert_eq!(AMBE_PARAM_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 49);
        assert_eq!(AMBE_VECTOR_WIDTHS.iter().map(|&w| u16::from(w)).sum::<u16>(), 49);
    }
}
