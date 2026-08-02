//! DSP/BFP primitives shared by the encoder analysis pipeline ([`crate::enc`])
//! and the decoder synthesis pipeline ([`crate::dec`] + `lib.rs`).
//!
//! These modules were factored out of `enc/` so a decode-only build
//! (`--no-default-features --features decode`) can compile without the encoder
//! analysis stages: the decoder reaches them here, and `enc/` re-exports the
//! ones it also uses so its own call sites are unchanged.
//!
//! Two kinds live here:
//!
//! * **Whole modules** the decoder needs and that carried no encode-only
//!   dependency: [`atan2_bfp_divide`], [`atan2_chain`], [`bfp_add`],
//!   [`array_a_stage2`], [`loudness_transform`], [`cepstral_normalize`].
//! * **Extracted primitives** pulled out of otherwise encode-heavy modules:
//!   [`gamma_poly`] (from `loudness_fixed`), [`q_energy`] (from `pq_builder` /
//!   `band_decompress`), [`step_count`] (from `step_recursive_fixed`), and
//!   [`voicing_map`] (from `voicing_vq`).

pub mod array_a_stage2;
pub mod atan2_bfp_divide;
pub mod atan2_chain;
pub mod bfp_add;
pub mod cepstral_normalize;
pub mod encode_frame;
pub mod gamma_poly;
pub mod loudness_transform;
pub mod q_energy;
pub mod step_count;
pub mod voicing_map;
