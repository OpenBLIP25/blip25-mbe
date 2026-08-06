//! IMBE full-rate (P25 Phase 1, 7200 bps) DECODE front-end.
//!
//! Produces [`crate::dequantize::MbeParams`] from a 144-bit IMBE frame, which
//! then drives the SHARED MBE synthesis back-end (`crate::synth`). Only the
//! front-end differs from AMBE+2; everything from `MbeParams` onward is reused.
//!
//! `frame` (deframe + FEC + PN demod) is roundtrip-tested; `dequantize` is a
//! **bit-exact fixed-point** implementation of the TIA-102.BABA / the reference
//! 7200x4400 standard. The IMBE front-end constants (pitch→ω₀/L,
//! bit-allocation, step-sizes) are NOT tabulated in the reference firmware image —
//! they are computed in code by the same standard formulas, so the standard IS
//! the firmware behaviour (P25 interop freezes the IMBE bitstream).
//! `fixp`/`math`/`dsp`/`tables` provide the C55x fixed-point primitives and the
//! standard's quantizer tables.
//!
//! **Provenance — this subtree only, and not the audio it leads to.** The
//! IMBE bitstream and its fixed-point reference are published in TIA-102.BABA,
//! so the front-end here is standards-derived: there was nothing to reverse
//! engineer. OP25's `imbe_vocoder` implements the same standard and was
//! consulted as a bit-exactness cross-check only.
//!
//! Two caveats, both important if you are reasoning about licensing:
//! 1. [`tables::GAIN_QNT_TBL`] / [`tables::MONO65_ENCODE`] are **recovered from
//!    firmware**, not from the standard — the reference's real gain ladder diverges from
//!    the published Annex-E at its bottom eight levels, and [`quantize`] uses
//!    the firmware ladder.
//! 2. **Synthesis is not in this subtree.** [`crate::ImbeDecoder`] defaults to
//!    the shared, reverse-engineered DLL-exact OLA back-end (`ola_synth: true`),
//!    so IMBE *audio output* is RE-derived exactly like AMBE+2's. Only the
//!    front-end above differs between the two codecs.

pub mod dequantize;
pub mod dsp;
pub mod fixp;
pub mod frame;
pub mod math;
pub mod quantize;
#[cfg(feature = "encode")]
pub(crate) mod ref_amp;
pub mod tables;

pub use dequantize::{decode_params, ImbeOlaParams, ImbeState};
pub use frame::{deframe, enframe, ImbeFrame, FRAME_BYTES};
pub use quantize::{
    num_bands, pitch_cell, pitch_omega0, quantize_frame, quantize_pitch, quantize_to_u,
    ImbeEncState,
};
