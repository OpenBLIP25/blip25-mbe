//! Same-rate repeater — parse, optionally clean up via FEC, repack at the
//! same rate.
//!
//! Equivalent to [`super::converter`] with matching input and output
//! rates. Its purpose is FEC error correction and retransmission without
//! rate change, and (when combined with a frame analyzer) erasure-frame
//! substitution for frames whose FEC could not be cleanly decoded — so
//! that downstream decoders perform a clean frame repeat rather than
//! synthesizing corrupted parameters.
//!
//! Corresponds to the AMBE-3000 `PKT_RPT_MODE = 0x01` configuration with
//! identical decoder and encoder rate parameters.
