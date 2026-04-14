//! Tests for spec-provided examples.
//!
//! Where TIA-102 or DVSI public specs state concrete input/output pairs
//! — "codeword X encodes to Y", "rate control words for P25 are
//! `0x0558 0x086b ...`", "Golay generator polynomial is `0xC75`" — those
//! examples belong here, as assertions that our implementation matches
//! the normative spec.
//!
//! This file never reads from DVSI/ and never requires proprietary
//! material. Every assertion here is traceable to a publicly available
//! spec document.

#[test]
fn placeholder() {
    // Real tests are added alongside the modules they exercise.
}
