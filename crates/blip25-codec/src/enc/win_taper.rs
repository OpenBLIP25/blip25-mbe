//! `a1ptr`/`win` -- the array [`super::loudness_transform::fft_bfp_transform`]
//! transforms -- its write site and content formula.
//!
//! ## Structure
//!
//! `a1ptr` is FRESH SCRATCH, not a self-referential or persistent accumulator:
//! it is filled entirely fresh on each call, with a hard maximum of 246 `i16`
//! words from the owning "outer table" function's stack allocation.
//!
//! Its write site is a 6-argument taper-driver call inside the driver body: the
//! "count" register is REASSIGNED to the function's ARG7 -- `a1ptr` itself,
//! threaded through unchanged from the outer function's local -- and passed as
//! **dest**, alongside **`localbuf`** (the `encoder_buf` window, copied then
//! exponent-normalized in place earlier in the SAME driver body) as **src**. So
//! `a1ptr` is fed directly from `localbuf` through one more transform, not from
//! a third unknown source.
//!
//! ## One driver, two tables
//!
//! The taper driver is a GENERAL 6-argument symmetric-taper driver. It is the
//! same function body as the fixed-32-sample `windowed_taper` primitive in
//! [`super::windowed_taper`]: that caller passes `count=32` and the small
//! `WINTAB` table, this one passes `count=199` and the much longer `RAMP_TABLE`.
//! Both share one body.
//!
//! That sharing has a practical consequence for instrumentation: a mid-loop
//! probe on the shared body also fires for interleaved `windowed_taper` calls,
//! computing `dest_idx`/`src_idx` against a stale reference and landing far
//! outside the plausible `[0,260]` range (the reproducible `dest_idx=-1149,
//! src_idx=-751, coeff=30913` pattern). Those iterations are cross-contamination
//! from the other caller, not a property of the `count=199` call -- filter by
//! plausible range.
//!
//! ## The two arms
//!
//! The driver walks `RAMP_TABLE` backward from its peak, applying the same
//! per-sample Q15 multiply-round-shift as [`super::windowed_taper`], to build:
//! * one ascending arm, `dest[1..=halfcount]` from
//!   `localbuf[halfcount+1..=2*halfcount]`;
//! * one special CENTER sample, `dest[0]` from `localbuf[halfcount]`, using the
//!   table's peak coefficient;
//! * one descending arm of `halfcount` words, `dest_idx` in
//!   `[256-halfcount, 255]` mirroring `src_idx` in `[0, halfcount-1]` -- the same
//!   total shape as the ascending arm, addressed from base 255 rather than 0.
//!
//! The descending arm reads the SAME front half of `localbuf` (indices
//! `0..HALFCOUNT-1`) paired with the SAME `RAMP_TABLE` index as `src_idx` --
//! **not** mirrored or reversed the way the ascending arm's `coeff` index is:
//! ```text
//! second_half[j] = q15_taper(localbuf[j], RAMP_TABLE[j])   for j in 0..HALFCOUNT
//! ```
//!
//! ## Scope
//!
//! This module does not derive `localbuf` from raw PCM; it takes `localbuf`
//! (post exponent-normalize) as an input, keeping each recovered stage
//! independently testable. See [`Self::from_raw_gap2`] for the raw-PCM path,
//! whose source array (`HistoryRing::gap2_window()` at `GAP2_OFFSET=28`) is
//! closed.

use super::block_exponent::block_exponent;

/// Per-frame length of `a1ptr`'s populated first half -- `localbuf`'s length
/// for the outer-table caller, `count=199`.
pub(crate) const LOCALBUF_LEN: usize = 199;
/// Half of [`LOCALBUF_LEN`] (`199 >> 1`, matching the reference's
/// arithmetic-shift truncation) -- both the ascending arm's length and the table
/// index the special center sample reads.
pub(crate) const HALFCOUNT: usize = LOCALBUF_LEN / 2;

/// The "ramp" taper table, read from the reference vocoder's `.rdata`; first
/// [`HALFCOUNT`]+1 = 100 entries, covering every index this module's formula
/// reads for the `count=199` case. A monotonically-increasing
/// quarter-cosine-shaped ramp -- NOT the small 16-entry `WINTAB`
/// ([`super::windowed_taper::WINTAB`]), which the same shared driver function
/// uses for its other caller.
const RAMP_TABLE: [i16; HALFCOUNT + 1] = [
    2390, 2398, 2419, 2453, 2502, 2564, 2639, 2729, 2831, 2948, 3077, 3219, 3375, 3544, 3725, 3919,
    4125, 4343, 4573, 4814, 5067, 5331, 5607, 5892, 6188, 6494, 6810, 7135, 7469, 7812, 8163, 8523,
    8889, 9264, 9645, 10033, 10427, 10826, 11231, 11641, 12056, 12474, 12896, 13321, 13750, 14180,
    14614, 15048, 15483, 15919, 16355, 16791, 17226, 17661, 18093, 18524, 18952, 19378, 19800,
    20219, 20633, 21042, 21447, 21848, 22242, 22629, 23010, 23384, 23751, 24110, 24461, 24804,
    25139, 25464, 25779, 26086, 26381, 26667, 26942, 27206, 27459, 27701, 27930, 28149, 28355,
    28549, 28730, 28898, 29054, 29196, 29326, 29443, 29546, 29635, 29711, 29773, 29821, 29856,
    29876, 29883,
];

/// The per-sample Q15 multiply-round-shift, byte-identical to
/// [`super::windowed_taper::windowed_taper`]'s inner formula: the taper driver
/// uses the same rounding and saturation, just walking a longer table over a
/// wider span.
fn q15_taper(sample: i16, coeff: i16) -> i16 {
    let v: i64 = (sample as i64) * (coeff as i64) * 2 + 0x8000;
    let mut pv = (v >> 16) as i32;
    pv = pv.clamp(-32768, 32767);
    pv as i16
}

/// `a1ptr`'s per-frame content. `first_half[0]` is the special center sample;
/// `first_half[1..=HALFCOUNT]` is the ascending arm. `second_half[j]` (`j` in
/// `0..HALFCOUNT`) is the descending arm, occupying the array's absolute indices
/// `256-HALFCOUNT..256` (`157..256` for `count=199`/`HALFCOUNT=99`) -- **not**
/// `HALFCOUNT+1..=2*HALFCOUNT`.
///
/// That leaves a structural gap at indices `HALFCOUNT+1..256-HALFCOUNT`
/// (`100..157`, 57 words) which the driver never writes on this call. Callers
/// must leave those 57 words at zero; the gap is real, not a missing finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WinArray {
    pub first_half: [i16; HALFCOUNT + 1],
    pub second_half: [i16; HALFCOUNT],
}

impl WinArray {
    /// Build `a1ptr` from `localbuf` -- the `encoder_buf` window's content,
    /// ALREADY exponent-normalized by the block-floating-point rescale. See
    /// [`Self::from_raw_gap2`] for a version that does that step too.
    ///
    /// Formula:
    /// ```text
    /// first_half[0]        = q15_taper(localbuf[HALFCOUNT],   RAMP_TABLE[HALFCOUNT])
    /// first_half[1 + i]    = q15_taper(localbuf[HALFCOUNT+1+i], RAMP_TABLE[HALFCOUNT-1-i])   for i in 0..HALFCOUNT
    /// second_half[j]       = q15_taper(localbuf[j], RAMP_TABLE[j])                            for j in 0..HALFCOUNT
    /// ```
    /// `second_half[j]` lands at the array's absolute index
    /// `256-HALFCOUNT+j` -- see [`WinArray`]'s own doc and
    /// [`super::loudness_fixed::win_from_gap2`] for how a caller assembles
    /// the full 256-word array from these two pieces (leaving the real,
    /// structural 57-word gap at indices `100..157` as zero).
    pub fn from_normalized_localbuf(localbuf: &[i16; LOCALBUF_LEN]) -> Self {
        let mut first_half = [0i16; HALFCOUNT + 1];
        first_half[0] = q15_taper(localbuf[HALFCOUNT], RAMP_TABLE[HALFCOUNT]);
        for i in 0..HALFCOUNT {
            let dest_idx = 1 + i;
            let src_idx = HALFCOUNT + 1 + i;
            let coeff = RAMP_TABLE[HALFCOUNT - 1 - i];
            first_half[dest_idx] = q15_taper(localbuf[src_idx], coeff);
        }
        let mut second_half = [0i16; HALFCOUNT];
        for (j, slot) in second_half.iter_mut().enumerate() {
            *slot = q15_taper(localbuf[j], RAMP_TABLE[j]);
        }
        WinArray {
            first_half,
            second_half,
        }
    }

    /// Same as [`Self::from_normalized_localbuf`], but starting from the RAW
    /// (pre-exponent-normalize) `gap2` array: applies the block-floating-point
    /// rescale (`shiftarg = -block_exponent(raw)`) first. `shiftarg` is always
    /// `>= 0` in practice, given [`block_exponent`]'s range, so the
    /// `shiftarg < 0` branch is never taken on real content. Truncating
    /// left-shift, matching the x86 shift idiom bit-for-bit.
    pub fn from_raw_gap2(raw: &[i16; LOCALBUF_LEN]) -> Self {
        let exp = block_exponent(raw);
        let shiftarg = (-exp) as u32;
        let mut localbuf = [0i16; LOCALBUF_LEN];
        if shiftarg == 0 {
            localbuf.copy_from_slice(raw);
        } else {
            for (dst, &src) in localbuf.iter_mut().zip(raw.iter()) {
                *dst = ((src as i32) << shiftarg.min(31)) as i16;
            }
        }
        Self::from_normalized_localbuf(&localbuf)
    }
}
