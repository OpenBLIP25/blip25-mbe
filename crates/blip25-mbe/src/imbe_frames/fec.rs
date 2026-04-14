//! IMBE-specific FEC layout — which bits are covered by which code,
//! how they are interleaved, and how PN demodulation is applied.
//!
//! Per TIA-102.BABA-A Sections 10–11. Composes the generic FEC primitives
//! in [`crate::fec`] into the full-rate and half-rate wire layouts.
