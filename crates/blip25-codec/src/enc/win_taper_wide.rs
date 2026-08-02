//! Slot 2's taper (the THIRD of the 3 per-frame `fft_bfp_transform`-loudness
//! topcalls) -- structurally the same Q15 multiply-round-shift taper as
//! [`super::win_taper`] (`HALFCOUNT=99`), but with a WIDER half-count
//! (`HALFCOUNT_WIDE=127`, `count=255`) driven by a genuinely SEPARATE 128-entry
//! ramp table, called from a genuinely separate static call site.
//!
//! ## No exponent-normalize step
//!
//! Unlike slots 0/1's `WinArray::from_raw_gap2`, slot 2's call site does NOT
//! pass its input through the block-exponent-normalize step before the shared
//! taper driver runs. `localbuf` is a byte-for-byte VERBATIM copy of the call
//! site's incoming `arg4`, starting at that pointer's index 0 -- no shift, no
//! rescale.
//!
//! ## `arg4` is the existing HistoryRing
//!
//! `arg4` is a FIXED pointer into `encoder_buf` at [`super::HistoryRing`]'s base
//! on every call, and the shift-copy feeding it (dest = ring base, src = ring
//! base + 80 words, count = 303 words) is exactly
//! [`super::HistoryRing::push_half`]'s `copy_within(80..383, 0)`. Slot 2 needs
//! NO new buffer, NO new filter stage, and NO new mechanism.
//!
//! Timing: the FIRST (gated-off) call of a frame sees the ring at
//! [`super::HistoryRing::gap2_window`]'s `gap2_mid` instant, after the frame's
//! first `push_half`. The SECOND (gated-on) call -- the one that actually
//! produces slot 2's `fft_bfp_transform` output -- sees the ring AFTER BOTH of
//! the frame's `push_half` calls, so [`super::HistoryRing::gap2_window_slot2`]
//! must be read at that later instant.
//!
//! ## Do not re-derive the taper by brute force
//!
//! Brute-forcing every `HistoryRing` offset and timing against slot 2's captured
//! PRE array scores at chance (~9%/2% partial, zero full matches) if the
//! `localbuf` content is wrong -- which looks identical to the taper formula
//! being wrong. The formula here is verified directly against a captured
//! non-degenerate `count=255` call, index for index.

/// Per-frame length of slot 2's populated `localbuf` (`count=255`).
pub(crate) const LOCALBUF_LEN_WIDE: usize = 255;
/// Half of [`LOCALBUF_LEN_WIDE`] (`255 >> 1`) -- both the ascending arm's length
/// and the table index the special center sample reads.
pub(crate) const HALFCOUNT_WIDE: usize = LOCALBUF_LEN_WIDE / 2;

/// The "wide ramp" taper table, read from the reference vocoder's `.rdata`;
/// `HALFCOUNT_WIDE + 1` = 128 entries. A genuinely separate table from
/// [`super::win_taper::RAMP_TABLE`] (100 entries): these 128 entries are
/// immediately followed, byte for byte, by the START of that other table, so the
/// bound is real and not a misread continuation.
const RAMP_TABLE_WIDE: [i16; HALFCOUNT_WIDE + 1] = [
    381, 789, 1223, 1685, 2174, 2693, 2859, 3030, 3204, 3382, 3564, 3749, 3939, 4132, 4329, 4530,
    4734, 4942, 5154, 5369, 5588, 5810, 6035, 6264, 6495, 6730, 6968, 7210, 7454, 7701, 7950, 8203,
    8458, 8715, 8976, 9238, 9503, 9770, 10039, 10310, 10583, 10857, 11134, 11412, 11691, 11972,
    12254, 12537, 12821, 13106, 13392, 13679, 13966, 14253, 14541, 14829, 15117, 15405, 15693,
    15980, 16267, 16554, 16839, 17124, 17408, 17691, 17972, 18252, 18531, 18808, 19084, 19357,
    19628, 19898, 20165, 20429, 20691, 20951, 21208, 21461, 21712, 21960, 22204, 22445, 22682,
    22916, 23146, 23372, 23594, 23812, 24026, 24236, 24441, 24641, 24837, 25029, 25215, 25397,
    25573, 25745, 25911, 26072, 26228, 26378, 26523, 26662, 26795, 26923, 27045, 27161, 27271,
    27375, 27473, 27565, 27651, 27731, 27804, 27871, 27932, 27987, 28035, 28077, 28112, 28141,
    28164, 28180, 28190, 28193,
];

/// The per-sample Q15 multiply-round-shift -- byte-identical to
/// [`super::win_taper::q15_taper`]: the same shared driver function, walking the
/// wider table over a wider span for this call site.
fn q15_taper(sample: i16, coeff: i16) -> i16 {
    let v: i64 = (sample as i64) * (coeff as i64) * 2 + 0x8000;
    let mut pv = (v >> 16) as i32;
    pv = pv.clamp(-32768, 32767);
    pv as i16
}

/// Slot 2's per-frame taper output. `first_half[0]` is the special center
/// sample; `first_half[1..=HALFCOUNT_WIDE]` is the ascending arm.
/// `second_half[j]` (`j` in `0..HALFCOUNT_WIDE`) is the descending arm,
/// occupying absolute indices `256-HALFCOUNT_WIDE..256` (`129..256`).
///
/// That leaves a single structural zero word at index `128`, which the driver
/// never writes on this call -- the same gap [`super::win_taper::WinArray`] has
/// for slots 0/1, just 1 word wide instead of 57.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WinArrayWide {
    pub first_half: [i16; HALFCOUNT_WIDE + 1],
    pub second_half: [i16; HALFCOUNT_WIDE],
}

impl WinArrayWide {
    /// `localbuf` here is slot 2's UN-normalized ring window
    /// (`HistoryRing::gap2_window_slot2()`'s output). Unlike slots 0/1's
    /// `WinArray::from_raw_gap2`, slot 2's call site applies no
    /// block-exponent-normalize step before the shared taper driver runs -- the
    /// captured `localbuf` matches `HistoryRing`'s raw content directly.
    ///
    /// Formula (mirrors
    /// [`super::win_taper::WinArray::from_normalized_localbuf`], just
    /// `HALFCOUNT_WIDE` instead of `HALFCOUNT`):
    /// ```text
    /// first_half[0]     = q15_taper(localbuf[HALFCOUNT_WIDE], RAMP_TABLE_WIDE[HALFCOUNT_WIDE])
    /// first_half[1 + i] = q15_taper(localbuf[HALFCOUNT_WIDE+1+i], RAMP_TABLE_WIDE[HALFCOUNT_WIDE-1-i])   for i in 0..HALFCOUNT_WIDE
    /// second_half[j]    = q15_taper(localbuf[j], RAMP_TABLE_WIDE[j])                                     for j in 0..HALFCOUNT_WIDE
    /// ```
    pub fn from_localbuf(localbuf: &[i16; LOCALBUF_LEN_WIDE]) -> Self {
        let mut first_half = [0i16; HALFCOUNT_WIDE + 1];
        first_half[0] = q15_taper(localbuf[HALFCOUNT_WIDE], RAMP_TABLE_WIDE[HALFCOUNT_WIDE]);
        for i in 0..HALFCOUNT_WIDE {
            let dest_idx = 1 + i;
            let src_idx = HALFCOUNT_WIDE + 1 + i;
            let coeff = RAMP_TABLE_WIDE[HALFCOUNT_WIDE - 1 - i];
            first_half[dest_idx] = q15_taper(localbuf[src_idx], coeff);
        }
        let mut second_half = [0i16; HALFCOUNT_WIDE];
        for (j, slot) in second_half.iter_mut().enumerate() {
            *slot = q15_taper(localbuf[j], RAMP_TABLE_WIDE[j]);
        }
        WinArrayWide {
            first_half,
            second_half,
        }
    }

    /// Assemble the full 256-word `WinState`-shaped array, matching
    /// [`super::win_taper::WinArray`]'s two-arm-plus-gap layout with a 1-word gap
    /// at index 128 instead of a 57-word gap.
    // takes &self to match the sibling win-state builders; by-value would be a needless API change
    #[allow(clippy::wrong_self_convention)]
    pub fn to_win_state(&self) -> [i16; 256] {
        let mut out = [0i16; 256];
        out[0..=HALFCOUNT_WIDE].copy_from_slice(&self.first_half);
        let desc_start = 256 - HALFCOUNT_WIDE;
        out[desc_start..256].copy_from_slice(&self.second_half);
        out
    }
}

/// Convenience: `HistoryRing::gap2_window_slot2()` (raw, un-normalized) straight
/// through to a full 256-word `WinState`-shaped array, mirroring
/// [`super::loudness_fixed::win_from_gap2`]'s role for slots 0/1.
pub(crate) fn win_from_gap2_slot2(gap2: &[i16; super::history_ring::GAP2_LEN_WIDE]) -> [i16; 256] {
    WinArrayWide::from_localbuf(gap2).to_win_state()
}
