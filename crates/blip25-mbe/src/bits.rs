//! Bit-packing and bit-slicing primitives shared across wire formats.
//!
//! Wire formats in this crate pack fields at non-byte-aligned boundaries
//! and use specific bit orderings defined by their respective specs.
//! This module hosts the spec-traceable primitives that wire layers
//! compose.

/// Single entry in a bit-prioritization table: one quantized parameter
/// bit `(src_param, src_bit)` is transmitted at `(dst_vec, dst_bit)`
/// within the info-vector bundle.
///
/// Used by both [`crate::imbe_wire::priority`] (BABA-A §10) and
/// [`crate::ambe_plus2_wire::priority`] (BABA-A §16.7); the build script
/// emits `BitMap { ... }` literals inside each rate's bit-priority
/// table, so this type must be in scope at the include site.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BitMap {
    pub src_param: u8,
    pub src_bit: u8,
    pub dst_vec: u8,
    pub dst_bit: u8,
}
