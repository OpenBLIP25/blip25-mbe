//! # blip25-mbe
//!
//! A clean-room implementation of the MBE vocoder family.
//!
//! ## Quick start
//!
//! Most consumers should use the chip-shaped [`vocoder::Vocoder`]
//! façade:
//!
//! ```rust
//! use blip25_mbe::vocoder::{Rate, Vocoder};
//!
//! // Open a P25 Phase 1 (full-rate IMBE) channel.
//! let mut tx = Vocoder::new(Rate::Imbe7200x4400);
//! let pcm: [i16; 160] = [0; 160];
//! let bits = tx.encode_pcm(&pcm).unwrap();
//! assert_eq!(bits.len(), 18);
//!
//! let mut rx = Vocoder::new(Rate::Imbe7200x4400);
//! let out = rx.decode_bits(&bits).unwrap();
//! assert_eq!(out.len(), 160);
//! ```
//!
//! Three streaming variants on top of the per-frame primitive:
//!
//! - [`vocoder::Vocoder::encode_stream`] / [`vocoder::Vocoder::decode_stream`]
//!   — slice → `Iterator<Item = Result<…>>`, drops trailing partial frames.
//! - [`vocoder::LiveEncoder`] / [`vocoder::LiveDecoder`] — chunk-driven
//!   with internal residue buffer for audio-callback / socket use.
//! - [`vocoder::VocoderBuilder`] — fluent configuration of optional
//!   knobs (tone detection, beyond-spec repeat reset).
//!
//! See [`vocoder`] for the full API and `INTEGRATION.md` for the
//! AMBE-3000R protocol → Vocoder operation correspondence.
//!
//! ## Module organization
//!
//! See [`DESIGN.md`](https://github.com/) at the repository root for
//! the architectural model. The public API is organized around four
//! orthogonal axes joined at a common parameter type:
//!
//! - [`vocoder`] — chip-shaped façade. **Recommended entry point.**
//! - [`mbe_params`] — the parameter model. The interchange type and
//!   the center of gravity of the crate.
//! - [`codecs`] — analysis and synthesis algorithms, one submodule
//!   per codec generation (`mbe_baseline`, `ambe`, `ambe_plus`,
//!   `ambe_plus2`).
//! - **Wire formats**, one module per protocol-rate combination:
//!   [`imbe7200`] (P25 Phase 1 IMBE, 144-bit), [`rate33`]
//!   (P25 Phase 2 AMBE+2, 72-bit), and [`dvsi_3000`] (DVSI chip
//!   protocol, r0..r63). Future protocols (DMR, D-STAR, NXDN, …)
//!   become sibling modules.
//! - [`rate_conversion`] — parameter-domain bits-to-bits conversion,
//!   a peer of the codec and wire layers, not a sub-concern of either.
//!
//! Primitives shared across layers live in [`fec`] and [`bits`].
//!
//! ## Cargo features
//!
//! - `serde` (off by default) — derive `Serialize` / `Deserialize` on
//!   the diagnostic types in [`vocoder`] (`Rate`, `FrameStats`,
//!   `AnalysisStats`, `AnalysisOutputKind`, `DecodeStats`) plus
//!   [`mbe_params::MbeParams`] and
//!   [`codecs::mbe_baseline::FrameDisposition`]. Useful for shipping
//!   stats / params over a future RPC layer (gRPC / protobuf / WS)
//!   without hand-rolled converters.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod bits;
pub mod fec;

pub mod mbe_params;

pub mod codecs;

pub mod imbe7200;
pub mod rate33;
pub mod dvsi_3000;
pub mod dvsi_soft_decision;

pub mod rate_conversion;

pub mod enhancement;
pub mod vocoder;
