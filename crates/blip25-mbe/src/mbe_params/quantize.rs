//! Quantization primitives shared across codec generations and wire formats.
//!
//! Specific codebooks (scalar bit-prioritized, split VQ, DCT-domain) live
//! in the codec generation that uses them; this module hosts the generic
//! scalar / vector quantizer infrastructure they build on.
