//! DVSI AMBE-3000 chip protocol — 64 rate configurations (`r0..r63`).
//!
//! Per the DVSI AMBE-3000 Protocol Specification. Each rate entry
//! carries a 6 × 16-bit rate control word and its own bit allocation,
//! FEC layout, and codec generation (AMBE / AMBE+ / AMBE+2). Used for
//! direct interop with DVSI's vocoder hardware over the chip serial
//! protocol — distinct from any P25-on-the-air wire format.
//!
//! P25-specific wires live in [`crate::imbe7200`] and
//! [`crate::rate33`]. Some rates here align with P25 (rate 33 =
//! P25 half-rate with FEC, rate 34 = P25 half-rate without FEC), but
//! the DVSI chip protocol carries additional framing that the
//! over-the-air P25 modulators strip off.

pub mod rate_tables;
pub mod frame;
