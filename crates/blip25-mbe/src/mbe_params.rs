//! The MBE parameter model — the crate's interchange type.
//!
//! Every wire format in this crate parses to this representation. Every
//! codec generation consumes or produces it. Rate conversion operates
//! entirely within it.
//!
//! The MBE speech model, originating in Griffin & Lim (1988) and formalized
//! for P25 in TIA-102.BABA-A, represents a 20 ms speech frame as:
//!
//! - `ω₀` — fundamental frequency (pitch)
//! - `L`  — number of harmonics, derived from `ω₀`
//! - `v_l` — voiced/unvoiced decision for each harmonic `l = 1..=L`
//! - `M_l` — spectral amplitude for each harmonic `l = 1..=L`
//!
//! Plus gain and voicing-metric metadata used by specific quantizers.
//!
//! This module is deliberately independent of any wire format or codec
//! generation. Adding a new wire format or a new codec generation must
//! not require any change here.

pub mod quantize;
