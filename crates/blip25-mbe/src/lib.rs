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
//! - [`imbe_frames`] and [`ambe_frames`] — on-the-air wire formats.
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

pub mod imbe_frames;
pub mod ambe_frames;

pub mod rate_conversion;
