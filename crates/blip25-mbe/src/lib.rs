//! # blip25-mbe
//!
//! A clean-room implementation of the MBE vocoder family.
//!
//! See [`DESIGN.md`](https://github.com/) at the repository root for the
//! architectural model. The public API is organized around three orthogonal
//! axes joined at a common parameter type:
//!
//! - [`mbe_params`] — the parameter model. The interchange type and the
//!   center of gravity of the crate.
//! - [`codecs`] — analysis and synthesis algorithms, one submodule per
//!   codec generation (`mbe_baseline`, `ambe`, `ambe_plus`, `ambe_plus2`).
//! - **Wire formats**, one module per protocol-rate combination:
//!   [`p25_fullrate`] (P25 Phase 1 IMBE, 144-bit), [`p25_halfrate`]
//!   (P25 Phase 2 AMBE+2, 72-bit), and [`dvsi_3000`] (DVSI chip protocol,
//!   r0..r63). Future protocols (DMR, D-STAR, NXDN, …) become sibling
//!   modules.
//! - [`rate_conversion`] — parameter-domain bits-to-bits conversion,
//!   a peer of the codec and wire layers, not a sub-concern of either.
//!
//! Primitives shared across layers live in [`fec`] and [`bits`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod bits;
pub mod fec;

pub mod mbe_params;

pub mod codecs;

pub mod p25_fullrate;
pub mod p25_halfrate;
pub mod dvsi_3000;

pub mod rate_conversion;
