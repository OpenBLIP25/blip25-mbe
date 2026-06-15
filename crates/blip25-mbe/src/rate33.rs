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
