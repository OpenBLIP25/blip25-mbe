//! IMBE full-rate (P25 Phase 1, 7200 bps) frame deframer: 144-bit frame →
//! 88 information bits as 8 prioritized vectors u0..u7.
//!
//! Structure (TIA-102.BABA-A §6–§7):
//!   c0..c3 : [23,12] Golay   (12 info → 23 coded)
//!   c4..c6 : [15,11] Hamming (11 info → 15 coded)
//!   c7     : 7 bits uncoded
//!   144 = 4·23 + 3·15 + 7 ;  info = 4·12 + 3·11 + 7 = 88.
//!
//! De-modulation: a PN sequence seeded from the decoded u0 is XORed across the
//! coded words c1..c6 (c0 is unmodulated so u0 can be recovered first to seed
//! the PN). The PN uses the IMBE LCG (173, 13849, mod 65536) — the SAME LCG the
//! synthesizer uses for unvoiced noise (a cross-consistency signal).
//!
//! Deframe/enframe are bit-exact against the reference P25 conformance vectors
//! (`tv-rc`/`tv-std` `p25/*.bit`+`.pcm` pairs). The 144 wire bits are NOT
//! sequential c0..c7 — a fixed interleave (`WIRE_TO_LOGICAL`, TIA-102.BABA-A's
//! IMBE voice-frame interleave) spreads burst errors across the frame;
//! deframe/enframe apply it before/after the sequential c0..c7 FEC layout. The
//! PN seed is `u0 << 4` (u0 left-shifted 4 bits before the first LCG step), not
//! raw u0.

use crate::fec::{
    golay_23_12_decode, golay_23_12_encode, hamming_15_11_decode, hamming_15_11_encode,
};

/// Bits per IMBE full-rate frame.
pub(crate) const FRAME_BITS: usize = 144;
/// Bytes per IMBE full-rate frame.
pub const FRAME_BYTES: usize = 18;
/// Info-bit widths of u0..u7. Sum = 88.
pub(crate) const U_WIDTHS: [u8; 8] = [12, 12, 12, 12, 11, 11, 11, 7];
/// Coded widths of c0..c7. Sum = 144.
pub(crate) const C_WIDTHS: [u8; 8] = [23, 23, 23, 23, 15, 15, 15, 7];

/// IMBE PN LCG (matches the synthesizer's unvoiced-noise LCG).
const PN_A: u32 = 173;
const PN_C: u32 = 13849;
const PN_M: u32 = 65536;

/// Wire bit `i` (MSB-first position in the 18-byte frame) → logical bit
/// position in the sequential c0..c7 layout (TIA-102.BABA-A IMBE voice-frame
/// interleave; cross-checked against two independent open-source P25
/// implementations). `logical[WIRE_TO_LOGICAL[i]] = wire[i]`.
#[rustfmt::skip]
const WIRE_TO_LOGICAL: [u8; FRAME_BITS] = [
    0, 24, 48, 72, 96, 120, 25,
    1, 73, 49, 121, 97, 2, 26, 50, 74, 98, 122, 27, 3, 75, 51, 123, 99, 4, 28, 52, 76, 100, 124,
    29, 5, 77, 53, 125, 101, 6, 30, 54, 78, 102, 126, 31, 7, 79, 55, 127, 103, 8, 32, 56, 80, 104,
    128, 33, 9, 81, 57, 129, 105, 10, 34, 58, 82, 106, 130, 35, 11, 83, 59, 131, 107, 12, 36, 60,
    84, 108, 132, 37, 13, 85, 61, 133, 109, 14, 38, 62, 86, 110, 134, 39, 15, 87, 63, 135, 111,
    16, 40, 64, 88, 112, 136, 41, 17, 89, 65, 137, 113, 18, 42, 66, 90, 114, 138, 43, 19, 91,
    67, 139, 115, 20, 44, 68, 92, 116, 140, 45, 21, 93, 69, 141, 117, 22, 46, 70, 94, 118, 142,
    47, 23, 95, 71, 143, 119,
];

/// Deinterleave: wire-order 18 bytes → logical (sequential c0..c7) 18 bytes.
fn deinterleave_frame(frame: &[u8; FRAME_BYTES]) -> [u8; FRAME_BYTES] {
    let mut logical = [0u8; FRAME_BYTES];
    for (i, &dst) in WIRE_TO_LOGICAL.iter().enumerate() {
        set_bit(&mut logical, dst as usize, get_bit(frame, i));
    }
    logical
}

/// Interleave: logical (sequential c0..c7) 18 bytes → wire-order 18 bytes.
fn interleave_frame(logical: &[u8; FRAME_BYTES]) -> [u8; FRAME_BYTES] {
    let mut wire = [0u8; FRAME_BYTES];
    for (i, &src) in WIRE_TO_LOGICAL.iter().enumerate() {
        set_bit(&mut wire, i, get_bit(logical, src as usize));
    }
    wire
}

/// Decoded IMBE frame: the 8 prioritized info vectors and per-word FEC errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImbeFrame {
    /// u0..u7 info words, LSB-aligned (widths per [`U_WIDTHS`]).
    pub u: [u16; 8],
    /// FEC errors corrected per word (u7 uncoded ⇒ 0).
    pub errors: [u8; 8],
}

/// MSB-first bit access over a byte slice.
#[inline]
fn get_bit(bytes: &[u8], idx: usize) -> u32 {
    ((bytes[idx / 8] >> (7 - (idx % 8))) & 1) as u32
}

#[inline]
fn set_bit(bytes: &mut [u8], idx: usize, v: u32) {
    if v & 1 == 1 {
        bytes[idx / 8] |= 1 << (7 - (idx % 8));
    }
}

/// Read `width` MSB-first bits starting at `pos` into an integer.
fn read_field(bytes: &[u8], pos: usize, width: u8) -> u32 {
    let mut v = 0u32;
    for i in 0..width as usize {
        v = (v << 1) | get_bit(bytes, pos + i);
    }
    v
}

fn write_field(bytes: &mut [u8], pos: usize, width: u8, val: u32) {
    for i in 0..width as usize {
        set_bit(bytes, pos + i, (val >> (width as usize - 1 - i)) & 1);
    }
}

/// Bit offset of codeword `k` in the frame (sequential layout assumption).
fn c_offset(k: usize) -> usize {
    C_WIDTHS[..k].iter().map(|&w| w as usize).sum()
}

/// Generate `nbits` of PN starting from seed `u0`, applied across c1..c6.
/// Returns a Vec of 0/1 modulation bits.
fn pn_stream(seed: u16, nbits: usize) -> Vec<u8> {
    let mut s = (seed as u32) << 4;
    let mut out = Vec::with_capacity(nbits);
    for _ in 0..nbits {
        s = (PN_A.wrapping_mul(s).wrapping_add(PN_C)) % PN_M;
        out.push(((s >> 15) & 1) as u8); // extraction convention: MSB
    }
    out
}

/// Deframe a 144-bit (18-byte) IMBE frame into u0..u7.
pub fn deframe(wire_frame: &[u8; FRAME_BYTES]) -> ImbeFrame {
    let frame = &deinterleave_frame(wire_frame);
    // c0 is unmodulated: decode to recover u0 (the PN seed).
    let c0 = read_field(frame, c_offset(0), C_WIDTHS[0]);
    let d0 = golay_23_12_decode(c0);
    let u0 = d0.info;

    // PN demod across c1..c6 (their coded bits), seeded from u0.
    let mod_bits: usize = C_WIDTHS[1..7].iter().map(|&w| w as usize).sum();
    let pn = pn_stream(u0, mod_bits);

    let mut u = [0u16; 8];
    let mut errors = [0u8; 8];
    u[0] = u0;
    errors[0] = d0.errors;

    let mut pn_i = 0usize;
    for k in 1..7 {
        let off = c_offset(k);
        let w = C_WIDTHS[k];
        let mut cw = read_field(frame, off, w);
        // de-modulate
        for b in 0..w as usize {
            let m = pn[pn_i] as u32;
            pn_i += 1;
            cw ^= m << (w as usize - 1 - b);
        }
        if k <= 3 {
            let d = golay_23_12_decode(cw);
            u[k] = d.info;
            errors[k] = d.errors;
        } else {
            let d = hamming_15_11_decode(cw as u16);
            u[k] = d.info;
            errors[k] = d.errors;
        }
    }
    // c7: 7 bits uncoded.
    u[7] = read_field(frame, c_offset(7), C_WIDTHS[7]) as u16;
    errors[7] = 0;

    ImbeFrame { u, errors }
}

/// Encode u0..u7 into a 144-bit IMBE frame (inverse of [`deframe`]; for tests
/// and for an eventual IMBE encoder). Applies FEC then PN modulation.
pub fn enframe(u: &[u16; 8]) -> [u8; FRAME_BYTES] {
    let mut frame = [0u8; FRAME_BYTES];
    // c0 unmodulated.
    write_field(
        &mut frame,
        c_offset(0),
        C_WIDTHS[0],
        golay_23_12_encode(u[0]),
    );

    let mod_bits: usize = C_WIDTHS[1..7].iter().map(|&w| w as usize).sum();
    let pn = pn_stream(u[0], mod_bits);
    let mut pn_i = 0usize;
    for k in 1..7 {
        let w = C_WIDTHS[k];
        let mut cw = if k <= 3 {
            golay_23_12_encode(u[k])
        } else {
            hamming_15_11_encode(u[k]) as u32
        };
        for b in 0..w as usize {
            let m = pn[pn_i] as u32;
            pn_i += 1;
            cw ^= m << (w as usize - 1 - b);
        }
        write_field(&mut frame, c_offset(k), w, cw);
    }
    write_field(&mut frame, c_offset(7), C_WIDTHS[7], u[7] as u32);
    interleave_frame(&frame)
}
