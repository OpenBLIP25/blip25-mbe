//! Baked-in normative spec tables (Annex L/M/N/O/S + bit-prioritization).
//!
//! Frozen into source in `tables_generated.rs`.
//! These are TIA-102.BABA-A spec constants describing the
//! *bit interpretation* (index -> (L, omega0), V/UV rows, block
//! lengths, FEC interleave, deprioritize map). The VQ codebooks
//! themselves (PRBA24/PRBA58/HOC/gain) come bit-exact from the firmware
//! via the `blip25_codebooks` crate, not from here.

include!("tables_generated.rs");

/// Number of half-rate parameters b0..b8.
pub(crate) const AMBE_B_COUNT: usize = 9;

/// Deprioritize 4 info vectors u0..u3 into a quantized parameter array b0..b8.
pub fn deprioritize(u: &[u16; 4]) -> [u16; AMBE_B_COUNT] {
    let mut b = [0u16; AMBE_B_COUNT];
    for m in AMBE_BIT_MAP.iter() {
        let bit = (u[m.dst_vec as usize] >> m.dst_bit) & 1;
        b[m.src_param as usize] |= bit << m.src_bit;
    }
    b
}
