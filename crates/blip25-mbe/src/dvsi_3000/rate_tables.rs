//! The 64 AMBE rate configurations (`r0..r63`) and their rate control words.
//!
//! Per the DVSI AMBE-3000 Protocol Specification. Each rate entry carries
//! the 6 × 16-bit rate control word, the generation it belongs to
//! (AMBE / AMBE+ / AMBE+2), total frame bits, parameter bits, and FEC layout
//! identifier.
