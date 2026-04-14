//! Full-rate IMBE frame — 144 bits carrying 88 parameter bits at 7200 bps.
//!
//! TIA-102.BABA-A Sections 10–12. The 88 parameter bits encode:
//! `b₀` (8-bit pitch), `b₁` (voicing decisions, variable length K = f(ω₀)),
//! `b₂` (6-bit gain), and `b₃..b_{L+2}` (spectral amplitude DCT residuals
//! with bit allocation varying by `L`).
