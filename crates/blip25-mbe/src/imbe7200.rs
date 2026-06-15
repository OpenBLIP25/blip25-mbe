//! P25 Phase 1 FDMA full-rate vocoder wire format.
//!
//! Per TIA-102.BABA-A §1–§12 (the consolidated 1998 BABA spec). The
//! full-rate wire is the original IMBE codec from the BABA section:
//! 144-bit air frames at 7,200 bps, carrying 88 information bits at
//! 4,400 bps after Annex H interleave + Golay/Hamming FEC + PN
//! demodulation.
//!
//! Pipeline:
//!
//! ```text
//!   72 dibits → frame::decode → ImbeFrame { info: [u16; 8], errors }
//!             → priority::deprioritize → b̂₀..b̂_{L+2}
//!             → dequantize → MbeParams
//! ```
//!
//! Bit-by-bit symmetric on the encode side via [`frame::encode_frame`],
//! [`priority::prioritize`], and [`dequantize::quantize`].
//!
//! ## Codec independence
//!
//! This module is the **wire**, not the codec. Any codec generation
//! (legacy IMBE, AMBE+, AMBE+2 — see [`crate::codecs`]) can produce or
//! consume the [`crate::mbe_params::MbeParams`] that flows through this
//! pipeline. P25 Phase 1 fire-channel deployments that pair this wire
//! with the AMBE+2 codec for SCBA-mask noise immunity are the
//! motivating example for keeping the two layers separate.

pub mod dequantize;
pub mod fec;
pub mod frame;
pub mod priority;
