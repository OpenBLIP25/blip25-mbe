//! P25 Phase 2 TDMA half-rate vocoder wire format.
//!
//! Per TIA-102.BABA-A §13–§17 + Annexes L–T (originally the BABA-1
//! addendum, consolidated into BABA-A in 2014). The half-rate wire is
//! the AMBE+2 codec — BABA-A renames it "Half-Rate Vocoder" to avoid
//! DVSI's trademark, but the bit rate (3,600 bps total = 2,450 voice
//! + 1,150 FEC), 72-bit frame size, and parameter structure are
//! AMBE+2 in everything but name. DVSI rate index 33 is documented as
//! "interoperable with APCO Project 25 half-rate with FEC."
//!
//! Pipeline:
//!
//! ```text
//!   36 dibits → frame::decode_frame → Frame { info: [u16; 4], errors }
//!             → priority::deprioritize → b̂₀..b̂₈
//!             → dequantize → MbeParams
//! ```
//!
//! Bit-by-bit symmetric on the encode side via [`frame::encode_frame`],
//! [`priority::prioritize`], and [`dequantize::quantize`].

pub mod dequantize;
pub mod frame;
pub mod priority;

/// Per-field extractors for diagnostic / integration use. Both return the
/// nine deprioritized parameters `[b̂₀..b̂₈]` (pitch, V/UV, gain, PRBA, HOC)
/// from a 7-byte no-FEC half-rate frame — [`fields_from_no_fec`] for the
/// R34 column-interleaved wire layout (blip25 / NXDN / Fusion),
/// [`fields_from_natural`] for the natural-order AMBE_d layout (mbelib,
/// Icom VE-PG4). Use these instead of flat-slicing the 49-bit prioritized
/// vector, which mixes parameters per slice. [`fields_from_fec`] is the
/// same for a 9-byte FEC frame (runs the full Golay decode first). Mirrors
/// the PyPI `blip25_mbe.rate33` Python API.
pub use frame::{fields_from_fec, fields_from_natural, fields_from_no_fec};
