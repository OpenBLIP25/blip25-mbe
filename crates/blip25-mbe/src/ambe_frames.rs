//! DVSI AMBE-family wire formats — 64 rate configurations, three generations.
//!
//! Per the AMBE-3000 Protocol Specification, rates are indexed `r0..r63`:
//!
//! | Indices   | Generation | Codec module              |
//! |-----------|------------|---------------------------|
//! | r0..r15   | AMBE       | [`crate::codecs::ambe`]        |
//! | r16..r31  | AMBE+      | [`crate::codecs::ambe_plus`]   |
//! | r32..r63  | AMBE+2     | [`crate::codecs::ambe_plus2`]  |
//!
//! Each rate has its own bit allocation, FEC scheme, and rate control
//! word (6 × 16-bit). This module hosts the rate table and the generic
//! frame parser; codec-generation-specific logic lives in the corresponding
//! `codecs/*` submodule.

pub mod rate_tables;
pub mod frame;
