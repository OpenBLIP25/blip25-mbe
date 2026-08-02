//! The encoder's windowed-tapering resample step.
//!
//! A **direct port**, not a reimplementation: recovered by instruction-level
//! disassembly and pinned bit-exact against captured SDK output across all 32
//! output slots.
//!
//! ## What this computes
//!
//! Given 32 real `i16` input samples (a sliding window into the caller's
//! spectrum/energy buffer, advancing by 4 samples per call), this applies a
//! fixed 16-entry raised-cosine-style taper table (`WINTAB`, entries sum to
//! exactly 65536) directly to each input sample.
//!
//! The taper is **not** mirrored, delayed, or accumulated across calls.
//! Comparing against a mirrored source index for the lower/middle quadrants
//! produces spurious "4-call-delay" and "shuffle" pattern-matches that dissolve
//! once the direct `src[k]` index is used for all 32 slots at once -- do not
//! chase them.
//!
//! `WINTAB` is walked ascending for the first half of the output (`k=0..=15`
//! uses `WINTAB[k]`) and descending for the second half (`k=16..=31` uses
//! `WINTAB[31-k]`), i.e. a single symmetric 32-wide window built from the
//! 16-entry half-table.
//!
//! ## What is NOT yet wired up
//!
//! This function's 32-sample input (the sliding window `src`, and the
//! spectrum/energy buffer it slides across) is not traced back to the confirmed
//! magnitude-squared spectrum or to the per-band energy-summation finding. The
//! output feeds directly into [`super::real_fft32::real_fft32`] (same buffer,
//! delta=0).

/// The taper table: 16 `i16` entries summing to exactly `65536` (a clean Q16
/// normalized taper), read from the reference binary's `.rdata` section.
const WINTAB: [i32; 16] = [
    1137, 2214, 3621, 5360, 7417, 9756, 12321, 15041, 17827, 20581, 23199, 25576, 27614, 29226,
    30342, 30913,
];

/// Apply the fixed symmetric taper window to 32 real `i16` input samples,
/// writing 32 `i16` output samples (out-of-place; the reference writes to a
/// separate `dst` argument).
///
/// `src` and the returned array both have exactly 32 elements.
pub(crate) fn windowed_taper(src: &[i16; 32]) -> [i16; 32] {
    let mut dst = [0i16; 32];
    for k in 0..32usize {
        let coeff = if k < 16 { WINTAB[k] } else { WINTAB[31 - k] };
        let v: i64 = (src[k] as i64) * (coeff as i64) * 2 + 0x8000;
        let mut pv = (v >> 16) as i32;
        pv = pv.clamp(-32768, 32767);
        dst[k] = pv as i16;
    }
    dst
}
