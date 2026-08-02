//! # blip25-mbe
//!
//! A P25 IMBE / AMBE+2 vocoder crate. Encode and decode route through the
//! in-repo codec engine (`blip25_codec` + `blip25_codebooks`) — a pure
//! "bits in / bits out" function reproducing the reference vocoder's
//! output exactly.
//!
//! The engine is **patent-encumbered**, and its provenance divides by layer
//! rather than by codec. Both wire layers — framing, FEC, bit interpretation
//! — are spec-derived from TIA-102.BABA / BABA-A. The shared analysis and
//! synthesis core was recovered by reverse engineering, and **both** codecs'
//! audio comes from it, IMBE included. See [`PATENT_NOTICE.md`] and
//! [`ATTRIBUTION.md`] before use, and [`WHY_THE_REFERENCE_CODEC.md`] for the
//! rationale.
//!
//! ## Quick start
//!
//! Most consumers should use the reference-shaped [`vocoder::Vocoder`]
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
//! - [`vocoder::VocoderBuilder`] — fluent configuration of the optional
//!   post-decode enhancement chain.
//!
//! See [`vocoder`] for the full API and [`INTEGRATION.md`] for the
//! AMBE-3000R protocol → Vocoder operation correspondence.
//!
//! ## Module organization
//!
//! See [`DESIGN.md`] at the repository root for the architectural
//! model. The public API is organized around four orthogonal axes
//! joined at a common parameter type:
//!
//! - [`vocoder`] — reference-shaped façade. **Recommended entry point.**
//! - [`mbe_params`] — the parameter model, the interchange type used by
//!   the wire and rate-conversion layers.
//! - **Wire formats**, one module per protocol-rate combination:
//!   [`imbe7200`] (P25 Phase 1 IMBE, 144-bit) and [`rate33`]
//!   (P25 Phase 2 AMBE+2, 72-bit). Future protocols (DMR, D-STAR, NXDN, …)
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
//!   [`mbe_params::MbeParams`]. Useful for shipping stats / params over a
//!   future RPC layer (gRPC / protobuf / WS) without hand-rolled converters.
//!
//! With `serde` off, this crate pulls in no runtime dependencies outside
//! its own workspace.
//!
//! [`DESIGN.md`]: https://github.com/openBLIP25/blip25-mbe/blob/main/DESIGN.md
//! [`INTEGRATION.md`]: https://github.com/openBLIP25/blip25-mbe/blob/main/INTEGRATION.md
//! [`PATENT_NOTICE.md`]: https://github.com/openBLIP25/blip25-mbe/blob/main/PATENT_NOTICE.md
//! [`ATTRIBUTION.md`]: https://github.com/openBLIP25/blip25-mbe/blob/main/ATTRIBUTION.md
//! [`WHY_THE_REFERENCE_CODEC.md`]: https://github.com/openBLIP25/blip25-mbe/blob/main/docs/WHY_THE_REFERENCE_CODEC.md

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
// Doc comments cross-reference crate-internal (`pub(crate)`) helpers by name;
// those links are intentional and resolve within the source. The valuable
// `broken_intra_doc_links` lint stays on to catch genuinely dead links.
#![allow(rustdoc::private_intra_doc_links)]
// A decode-only build (`--no-default-features --features decode`) gates out the
// encode public API; the encode-only helpers it leaves behind are then unused.
// Silence that dead-code noise only in the decode-only configuration.
#![cfg_attr(not(feature = "encode"), allow(dead_code))]

pub mod bits;
pub mod fec;

pub mod mbe_params;

pub mod reference_soft_decision;
pub mod imbe7200;
pub mod rate33;

pub mod rate_conversion;

pub mod enhancement;
pub mod synth;
pub mod vocoder;
