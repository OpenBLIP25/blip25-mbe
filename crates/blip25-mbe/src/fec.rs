//! Forward-error-correction primitives shared across wire formats.
//!
//! The P25 IMBE wire uses `Golay(23, 12)` and `Hamming(15, 11)` per
//! TIA-102.BABA-A Sections 10 and 11. AMBE-family wires use rate-specific
//! FEC schemes per DVSI's AMBE-3000 protocol specification. This module
//! hosts the generator polynomials, syndrome decoders, and interleavers
//! that those wire layers build on.
//!
//! Wire-format-specific FEC layout (which bits are covered by which code,
//! how they are interleaved, and which bits are PN-demodulated) lives in
//! the wire module — for IMBE, see [`crate::imbe_frames::fec`].
