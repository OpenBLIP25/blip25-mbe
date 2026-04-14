//! AMBE+ — AMBE-2000, Generation 2 of the DVSI MBE codec family.
//!
//! Covers rate indices `r16..r31` in the AMBE-3000 rate table. Adds to the
//! Generation 1 algorithms:
//!
//! - US5701390 — phase regeneration from spectral-envelope shape,
//!   replacing the random-phase restarts of the baseline voiced synthesis
//!   path. Eliminates the "buzzy" artifact characteristic of baseline MBE.
//! - US6199037 — joint quantization of voicing metrics and pitch across
//!   subframes.
