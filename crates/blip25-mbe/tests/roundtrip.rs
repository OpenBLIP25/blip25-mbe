//! Property tests — invariants that must hold across the pipeline.
//!
//! Examples (added as the implementation fills in):
//!
//! - `MbeParams::pack().unpack() == params` for each wire format
//! - `rate_conversion(rate_A → rate_A)` is the identity in parameter space
//!   modulo requantization error
//! - FEC encoders and syndrome decoders are mutual inverses on valid inputs
//!
//! None of these tests require DVSI material.

#[test]
fn placeholder() {
    // Real property tests are added alongside the modules they exercise.
}
