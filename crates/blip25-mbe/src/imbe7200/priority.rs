//! Bit prioritization — the layer between the IMBE info vectors
//! `û_i` and the quantized voice parameters `b̂_l`.
//!
//! Per BABA-A §10. The mapping is `L`-dependent: for every `L ∈ [9, 56]`
//! the 88 info bits carry a specific subset of parameters with specific
//! bit allocations, scanned into `û₀..û₇` such that the most significant
//! bits receive the strongest FEC.
//!
//! The table is generated at build time from `spec_tables/imbe_bit_prioritization.csv`,
//! with coverage invariants checked during generation.

use crate::bits::BitMap;

include!(concat!(env!("OUT_DIR"), "/imbe_bit_priority.rs"));

/// Maximum full-rate parameter index: `b̂_{L+2}` for `L = 56` gives
/// `b̂_{58}`, so the parameter array needs `59` slots (indices 0..=58).
pub const IMBE_B_MAX: usize = 59;

/// Minimum full-rate harmonic count. Matches [`crate::mbe_params::L_MIN`].
pub const L_MIN: u8 = 9;

/// Maximum full-rate harmonic count. Matches [`crate::mbe_params::L_MAX`].
pub const L_MAX: u8 = 56;

/// Prioritize a quantized full-rate parameter array `b̂₀..b̂_{L+2}`
/// into the 8 info vectors `û₀..û₇`.
///
/// Bits are read from `b[src_param]` at position `src_bit` and placed
/// into `u[dst_vec]` at `dst_bit`. All parameter bits are LSB-first
/// (bit 0 = LSB) per the module-level convention.
///
/// Panics in debug builds if `l` is outside `[L_MIN, L_MAX]`.
pub fn prioritize(b: &[u16; IMBE_B_MAX], l: u8) -> [u16; 8] {
    debug_assert!((L_MIN..=L_MAX).contains(&l), "L out of range");
    let l_idx = (l - L_MIN) as usize;
    let mut u = [0u16; 8];
    for m in &IMBE_BIT_MAP[l_idx] {
        let bit = (b[m.src_param as usize] >> m.src_bit) & 1;
        u[m.dst_vec as usize] |= bit << m.dst_bit;
    }
    u
}

/// Deprioritize 8 info vectors `û₀..û₇` into a quantized parameter
/// array `b̂₀..b̂_{L+2}`.
///
/// Returns an `[u16; IMBE_B_MAX]` with entries `[0..=L+2]` populated;
/// entries `[L+3..]` are zeroed. The caller is expected to know `L`
/// (derived from the pitch MSBs in `û₀` via `b̂₀` → Eqs. 46–47).
///
/// Panics in debug builds if `l` is outside `[L_MIN, L_MAX]`.
pub fn deprioritize(u: &[u16; 8], l: u8) -> [u16; IMBE_B_MAX] {
    debug_assert!((L_MIN..=L_MAX).contains(&l), "L out of range");
    let l_idx = (l - L_MIN) as usize;
    let mut b = [0u16; IMBE_B_MAX];
    for m in &IMBE_BIT_MAP[l_idx] {
        let bit = (u[m.dst_vec as usize] >> m.dst_bit) & 1;
        b[m.src_param as usize] |= bit << m.src_bit;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum valid src_param observed for a given L in the table
    /// (i.e. b̂_{L+2}). Useful for constructing valid `b` arrays.
    fn max_src_param_for(l: u8) -> u8 {
        let l_idx = (l - L_MIN) as usize;
        IMBE_BIT_MAP[l_idx]
            .iter()
            .map(|m| m.src_param)
            .max()
            .unwrap()
    }

    /// Width (in bits) of parameter b̂_{src_param} for a given L,
    /// derived from the CSV table (max src_bit + 1 for that param).
    fn param_width(l: u8, src_param: u8) -> u8 {
        let l_idx = (l - L_MIN) as usize;
        let max_bit = IMBE_BIT_MAP[l_idx]
            .iter()
            .filter(|m| m.src_param == src_param)
            .map(|m| m.src_bit)
            .max();
        max_bit.map_or(0, |b| b + 1)
    }

    fn sample_b(l: u8, seed: u32) -> [u16; IMBE_B_MAX] {
        let mut b = [0u16; IMBE_B_MAX];
        let mut state = seed;
        for src in 0..=max_src_param_for(l) {
            let w = param_width(l, src) as u32;
            if w == 0 {
                continue;
            }
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let mask = if w == 16 {
                u16::MAX
            } else {
                ((1u32 << w) - 1) as u16
            };
            b[src as usize] = (state as u16) & mask;
        }
        b
    }

    #[test]
    fn roundtrip_all_l_values() {
        for l in L_MIN..=L_MAX {
            for seed in [1u32, 0xDEADBEEF, 0xA5A5A5A5] {
                let b = sample_b(l, seed ^ u32::from(l));
                let u = prioritize(&b, l);
                // All dst bits must fit within their vector widths.
                let widths = [12u8, 12, 12, 12, 11, 11, 11, 7];
                for (i, &w) in widths.iter().enumerate() {
                    let mask = if w == 16 {
                        u16::MAX
                    } else {
                        ((1u16 << w) - 1) as u16
                    };
                    assert_eq!(u[i] & !mask, 0, "L={l}: û{i} has bits beyond width {w}");
                }
                let b2 = deprioritize(&u, l);
                assert_eq!(b2, b, "L={l}, seed=0x{seed:08x}");
            }
        }
    }

    #[test]
    fn zero_prioritizes_to_zero() {
        for l in L_MIN..=L_MAX {
            let b = [0u16; IMBE_B_MAX];
            let u = prioritize(&b, l);
            assert_eq!(u, [0u16; 8]);
        }
    }

    #[test]
    fn pitch_msb_goes_to_u0() {
        // BABA-A §1.4: "The MSB of b̂₀ always goes into û₀."
        for l in L_MIN..=L_MAX {
            let mut b = [0u16; IMBE_B_MAX];
            b[0] = 0b1000_0000;
            let u = prioritize(&b, l);
            assert_ne!(u[0], 0, "L={l}: MSB of b̂₀ did not affect û₀");
        }
    }

    #[test]
    fn six_msbs_of_pitch_are_in_u0_at_fixed_positions() {
        // §1.3.1: the six MSBs of b̂₀ are placed in û₀ across all L.
        let l_min_positions: Vec<(u8, u8)> = IMBE_BIT_MAP[0]
            .iter()
            .filter(|m| m.src_param == 0 && m.src_bit >= 2)
            .map(|m| (m.dst_vec, m.dst_bit))
            .collect();
        assert_eq!(l_min_positions.len(), 6);
        for (v, _) in &l_min_positions {
            assert_eq!(*v, 0, "pitch MSB not in û₀ for L=9");
        }
        for l in (L_MIN + 1)..=L_MAX {
            let l_idx = (l - L_MIN) as usize;
            let positions: Vec<(u8, u8)> = IMBE_BIT_MAP[l_idx]
                .iter()
                .filter(|m| m.src_param == 0 && m.src_bit >= 2)
                .map(|m| (m.dst_vec, m.dst_bit))
                .collect();
            assert_eq!(
                positions, l_min_positions,
                "L={l}: pitch MSB positions differ from L=9"
            );
        }
    }

    #[test]
    fn single_bit_propagation() {
        for l in [L_MIN, 20, 35, L_MAX] {
            let l_idx = (l - L_MIN) as usize;
            for m in &IMBE_BIT_MAP[l_idx] {
                let mut b = [0u16; IMBE_B_MAX];
                b[m.src_param as usize] = 1 << m.src_bit;
                let u = prioritize(&b, l);
                let ones: u32 = u.iter().map(|x| x.count_ones()).sum();
                assert_eq!(
                    ones, 1,
                    "L={l} (src={}, bit={}) produced {ones} ones",
                    m.src_param, m.src_bit
                );
                let b2 = deprioritize(&u, l);
                assert_eq!(b2, b);
            }
        }
    }
}
