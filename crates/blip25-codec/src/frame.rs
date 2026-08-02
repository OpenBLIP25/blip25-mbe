//! Half-rate AMBE+2 frame deframer: 9-byte (72-bit) r33 frame ->
//! 36 dibits -> Annex-S deinterleave -> Golay(24,12)/Golay(23,12) +
//! PN demod -> 49 info bits u0[12]||u1[12]||u2[11]||u3[14].
//!
//! TIA-102.BABA-A §2.4-§2.6.

use crate::fec::{golay_23_12_decode, golay_24_12_decode};
use crate::tables::ANNEX_S;

/// Dibit symbols per half-rate frame (72 bits / 2).
pub(crate) const DIBITS_PER_FRAME: usize = 36;

/// PN sequence length: p_r(0) seed plus p_r(1..=23).
pub(crate) const PN_SEQ_LEN: usize = 24;

/// Convert a 9-byte frame to 36 dibits (MSB-first bit packing).
/// `dibit[i]` high bit = frame bit 2i, low bit = frame bit 2i+1.
pub(crate) fn bytes_to_dibits(bytes: &[u8]) -> [u8; DIBITS_PER_FRAME] {
    let mut out = [0u8; DIBITS_PER_FRAME];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[(2 * i) / 8] >> (7 - ((2 * i) % 8))) & 1;
        let lo = (bytes[(2 * i + 1) / 8] >> (7 - ((2 * i + 1) % 8))) & 1;
        *slot = (hi << 1) | lo;
    }
    out
}

/// Deinterleave a 72-bit half-rate frame into the 4 code vectors c0..c3.
pub(crate) fn deinterleave(dibits: &[u8; DIBITS_PER_FRAME]) -> [u32; 4] {
    let mut c = [0u32; 4];
    for (sym, d) in dibits.iter().enumerate() {
        let entry = &ANNEX_S[sym];
        let hi = u32::from((d >> 1) & 1);
        let lo = u32::from(d & 1);
        c[entry.bit1_vec as usize] |= hi << entry.bit1_idx;
        c[entry.bit0_vec as usize] |= lo << entry.bit0_idx;
    }
    c
}

/// Generate the half-rate PN sequence p_r(0..=23) seeded by 16*u0.
pub(crate) fn pn_sequence(u0: u16) -> [u16; PN_SEQ_LEN] {
    let mut pr = [0u16; PN_SEQ_LEN];
    pr[0] = u0.wrapping_mul(16);
    for n in 1..PN_SEQ_LEN {
        pr[n] = (173u32.wrapping_mul(pr[n - 1] as u32).wrapping_add(13849) & 0xFFFF) as u16;
    }
    pr
}

/// The 23-bit PN mask m1 for c1; first PN at bit 22, last at bit 0.
pub(crate) fn modulation_mask_c1(u0: u16) -> u32 {
    let pr = pn_sequence(u0);
    let mut m = 0u32;
    for k in 0..23 {
        let bit = u32::from(pr[1 + k] >> 15); // MSB of p_r(n)
        m |= bit << (22 - k);
    }
    m
}

/// Decoded 49-bit info layer of a half-rate AMBE frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Decoded info vectors u0..u3, LSB-aligned. Widths 12/12/11/14 (sum 49).
    pub info: [u16; 4],
    /// Per-vector FEC error count. errors[0] is 0-3 (or u8::MAX for an
    /// uncorrectable extended-Golay c0); errors[1] is 0-3; errors[2],[3] = 0.
    pub errors: [u8; 4],
}

impl Frame {
    /// epsilon_T per BABA-A §2.8.1 Eq. 196 = eps_0 + eps_1 (Golay coverage).
    pub fn epsilon_t(&self) -> u8 {
        let e0 = if self.errors[0] == u8::MAX {
            4
        } else {
            self.errors[0]
        };
        e0.saturating_add(self.errors[1])
    }
}

/// Decode a half-rate frame: 36 dibits -> [`Frame`].
pub(crate) fn decode_frame(dibits: &[u8; DIBITS_PER_FRAME]) -> Frame {
    let c = deinterleave(dibits);
    let d0 = golay_24_12_decode(c[0]);
    let u0 = d0.info;
    let mask = modulation_mask_c1(u0);
    let d1 = golay_23_12_decode(c[1] ^ mask);
    let u2 = (c[2] & 0x7FF) as u16;
    let u3 = (c[3] & 0x3FFF) as u16;
    Frame {
        info: [d0.info, d1.info, u2, u3],
        errors: [d0.errors, d1.errors, 0, 0],
    }
}

/// Decode a 9-byte r33 frame directly. Slices shorter than 9 bytes are
/// zero-padded; bytes past the 9th are ignored (never panics).
pub fn decode_bytes(bytes: &[u8]) -> Frame {
    // Hostile input class: truncated wire frames (len < 9) — zero-pad the
    // missing tail so garbage degrades to a normal (erasure-prone) decode.
    let mut b = [0u8; 9];
    let n = bytes.len().min(9);
    b[..n].copy_from_slice(&bytes[..n]);
    decode_frame(&bytes_to_dibits(&b))
}
