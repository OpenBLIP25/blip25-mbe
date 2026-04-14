//! AMBE+2 — AMBE-3000, Generation 3 of the DVSI MBE codec family.
//!
//! Covers rate indices `r32..r63` in the AMBE-3000 rate table. The current
//! DVSI generation; the primary target of this project's chip-conformance
//! harness. Builds on AMBE+ with:
//!
//! - US8595002 — half-rate AMBE+2 split vector quantization of spectral
//!   magnitudes with DCT-domain codebooks; per-harmonic (rather than
//!   per-band) voicing decisions; data-dependent scrambling for error
//!   resilience.
//! - US8315860 — enhanced full-rate encoding: three-state voicing model
//!   (voiced / unvoiced / pulsed), fundamental-frequency-field repurposing
//!   when no voiced bands exist, tone detection, spectral sidelobe
//!   suppression, and noise suppression via spectral subtraction.
