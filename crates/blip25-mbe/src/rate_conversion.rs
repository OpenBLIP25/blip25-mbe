//! Parametric rate conversion — the bits-to-bits operation in parameter space.
//!
//! ```text
//!   bits @ rate_A  →  parse  →  MbeParams  →  requantize  →  pack  →  bits @ rate_B
//! ```
//!
//! No PCM synthesis. No re-analysis. No codec. A single journey through
//! the parameter domain.
//!
//! This is the operation that preserves intelligibility when bridging
//! between MBE-family standards (P25, DMR, D-STAR, NXDN) where tandeming
//! — decoding to PCM and re-encoding — would lose roughly 40% of
//! intelligibility in noisy conditions per DVSI's ABC-MRT testing
//! (ANSI/ASA S3.2). Parametric rate conversion keeps the MRT score above
//! the 75% intelligibility threshold established by NIST/ITS research.
//!
//! Primary reference: US7634399 (voice transcoder, expired 2025-11-07),
//! continued in US7957963. See [`repeater`] for the same-rate variant
//! used for FEC cleanup or retransmission without a rate change.

pub mod converter;
pub mod repeater;

pub use converter::{ConvertError, FullToHalfConverter, HalfToFullConverter};
pub use repeater::{FullRateRepeater, HalfRateRepeater};
