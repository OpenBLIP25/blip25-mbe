//! Bit-packing and bit-slicing primitives.
//!
//! Wire formats in this crate pack fields at non-byte-aligned boundaries
//! and use specific bit orderings defined by their respective specs
//! (bit-prioritization for IMBE, per-rate bit allocations for AMBE
//! frames). This module hosts the spec-traceable primitives that those
//! wire layers compose.
