//! P25 IMBE wire formats — on-the-air bit containers.
//!
//! Per TIA-102.BABA-A. This module handles wire-layer concerns — bit
//! allocation, FEC layout, interleaving, PN demodulation, bit prioritization
//! — and parses to or from the codec-agnostic
//! [`MbeParams`](crate::mbe_params) representation.
//!
//! | Submodule          | Frame size | Parameter bits | Channel rate |
//! |--------------------|-----------:|---------------:|-------------:|
//! | [`full_rate`]      | 144 bits   | 88 bits        | 7200 bps     |
//! | [`half_rate`]      |  72 bits   | 49 bits        | 3600 bps     |
//!
//! The IMBE wire format is fixed by interoperability. Any P25 Phase 1
//! transmitter, regardless of which codec generation produced the
//! parameters, produces these exact bit layouts.

pub mod full_rate;
pub mod half_rate;
pub mod fec;
