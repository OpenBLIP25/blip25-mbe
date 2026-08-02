//! Decoder: r33/IMBE info bits → dequantized [`crate::dequantize::MbeParams`] →
//! fixed-point OLA synthesis → 8 kHz i16 PCM. Grouped submodules mirroring the
//! synthesis pipeline (excitation, M_l generation, linear amplitude, voiced
//! OLA, unvoiced synthesis), matching the layout of [`crate::enc`] and
//! [`crate::imbe`].

pub mod excitation;
pub mod gen_ml;
pub mod gen_ml_tables;
pub mod gen_resid;
pub mod gen_vmask;
pub mod linamp;
pub mod ola_body;
pub mod ola_driver;
pub mod ola_gen;
pub mod ola_leaves;
pub mod ola_osc;
pub mod ola_shift;
pub mod sitea_ml;
pub mod unv_frame;
pub mod unvoiced;
