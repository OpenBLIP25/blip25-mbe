//! The encoder's persistent 383-sample "history ring", which feeds the
//! 108-sample analysis staging buffer ([`super::analysis_stage_window`]).
//!
//! ## What this computes
//!
//! A persistent `RING_LEN = 0x17f` (383) sample buffer, zero-initialized to
//! match the reference's `Reset()`. Each 80-sample pre-filter output
//! ([`super::audio_prefilter::prefilter`]'s `dest`) is pushed via
//! [`HistoryRing::push_half`]: the ring shifts left by 80 (dropping the oldest
//! 80 samples) and the 80 new samples land at the tail. [`HistoryRing::stage`]
//! returns the ring's last 108 samples (`ring[275:383)`), the address and length
//! [`super::analysis_stage_window`] confirms as the `windowed_taper` source.
//!
//! ## Read timing matters
//!
//! Each frame pushes two halves, and the three per-frame loudness top-calls read
//! the ring at different points in that cadence. The per-frame `windowed_taper`
//! scan reads after only the FIRST of a frame's two pre-filter halves
//! (`2*frame+1` total half-calls, not `2*frame+2`). Comparing against the wrong
//! half looks like being off by roughly half a frame's content -- see the
//! per-slot constants below for which timing each uses.

/// Length of the persistent history ring (`0x17f` = 383).
pub(crate) const RING_LEN: usize = 0x17f;
/// Length of one pre-filter call's output span: each pre-filter call processes
/// 80 samples.
pub(crate) const HALF_LEN: usize = 80;
/// Length of the analysis staging window
/// ([`super::analysis_stage_window::STAGE_LEN`]).
pub(crate) const STAGE_LEN: usize = super::analysis_stage_window::STAGE_LEN;
/// Offset of the staging window within the ring (`RING_LEN - STAGE_LEN` = 275).
pub(crate) const STAGE_OFFSET: usize = RING_LEN - STAGE_LEN;

/// Length of the `fft_bfp_transform`/`arg2`-source array. There are three
/// per-frame stack copies out of this same ring, of sizes 199/199/255; this
/// constant covers the two 199-length slots.
pub(crate) const GAP2_LEN: usize = 199;
/// Offset within the ring of the `fft_bfp_transform`/`arg2`-source array, for
/// slot 0: **28** i16 words, the address-arithmetic-derived value. At offset 28
/// the 2nd and 3rd of the loudness top-call's three per-frame calls match this
/// ring bit-exactly on every frame of three files (254,919 samples, zero
/// mismatches). Only the FIRST of the three reads stale, previous-frame data.
///
/// Two traps to avoid when re-checking this:
/// * Pooling all three per-frame reads into one score dilutes a perfect
///   offset-28 signal toward noise, because the first read genuinely does not
///   match.
/// * Offset `108` scores 91-99.5% per tap on low-content frames. That is the
///   near-zero-content artifact: at low content, many wrong offsets score high
///   because most values are near zero. It is not the right offset for slot 0.
pub(crate) const GAP2_OFFSET: usize = 28;

/// The ring offset for the SECOND of the 3 per-frame
/// `fft_bfp_transform`-loudness topcalls ("slot 1"): **108**. Slot 1 uses the
/// SAME `gap2_mid` read timing as slot 0 and the same unmodified
/// [`win_taper`]/`win_from_gap2` 199-word taper -- only the window moves,
/// `ring[108..307)` instead of `ring[28..227)`.
///
/// Algebraically this is `GAP2_OFFSET` (28) read one push later, i.e. after BOTH
/// of the frame's prefilter halves rather than just the first, which is why slot
/// 1 appears to see the ring at `delta=+1` relative to slot 0.
pub(crate) const GAP2_OFFSET_SLOT1: usize = 108;

/// The width of slot 2's window (the THIRD of the 3 per-frame
/// `fft_bfp_transform`-loudness topcalls): 255. Slot 2 has a genuinely separate
/// static caller from slots 0/1, with `HALFCOUNT_WIDE=127` and its own 128-entry
/// ramp table -- distinct from slots 0/1's 100-entry table.
pub(crate) const GAP2_LEN_WIDE: usize = 255;

/// Slot 2's ring offset: **0**, i.e. the ring's index 0, read verbatim with NO
/// shift -- unlike [`GAP2_OFFSET`]/[`GAP2_OFFSET_SLOT1`], which both anchor at
/// nonzero offsets. `localbuf` is a byte-for-byte unmodified copy of the ring
/// starting at index 0, and the slot 2 call's incoming `arg4` is exactly this
/// [`HistoryRing`]'s state, bit-exact over 149,700 captured samples across two
/// files. See [`HistoryRing::gap2_window_slot2`] for the per-call timing.
///
/// Slot 2 needs NO new buffer, filter, or mechanism: the existing
/// `audio_prefilter`/`HistoryRing` chain covers it.
pub(crate) const GAP2_OFFSET_SLOT2: usize = 0;

/// The persistent 383-sample history ring. Zero-initialized at construction to
/// match the reference's `Reset()`, which leaves the ring all-zero before any audio
/// is pushed.
#[derive(Clone, Copy)]
pub(crate) struct HistoryRing {
    buf: [i16; RING_LEN],
}

impl HistoryRing {
    /// A fresh ring, matching `Reset()` (all-zero).
    pub fn new() -> Self {
        Self { buf: [0; RING_LEN] }
    }

    /// Push one 80-sample pre-filter output: shift the ring left by 80
    /// (dropping the oldest 80 samples) and insert `filtered` at the tail
    /// (`ring[RING_LEN-80..RING_LEN)`). This is the reference's shift-call geometry
    /// -- `dest=ring_base`, `src=ring_base+80samples`, `count=RING_LEN-80` --
    /// composed with the pre-filter's fixed write target
    /// `ring_base + (RING_LEN-80)*2` bytes.
    pub fn push_half(&mut self, filtered: &[i16; HALF_LEN]) {
        self.buf.copy_within(HALF_LEN..RING_LEN, 0);
        self.buf[RING_LEN - HALF_LEN..RING_LEN].copy_from_slice(filtered);
    }

    /// The ring's last [`STAGE_LEN`] samples — the `windowed_taper` source
    /// ([`super::analysis_stage_window`]).
    pub fn stage(&self) -> [i16; STAGE_LEN] {
        let mut out = [0i16; STAGE_LEN];
        out.copy_from_slice(&self.buf[STAGE_OFFSET..RING_LEN]);
        out
    }

    /// Full raw ring content, for calibration/diagnostics only (e.g.
    /// finding the exact offset a new consumer reads at, against captured
    /// reference data). Not part of the stable analysis API.
    pub fn debug_full(&self) -> [i16; RING_LEN] {
        self.buf
    }

    /// The ring's `[`GAP2_OFFSET`, `GAP2_OFFSET`+`GAP2_LEN`)` window -- slot 0's
    /// source for `fft_bfp_transform`'s `arg2` array. See [`GAP2_OFFSET`] for
    /// the caveats.
    pub fn gap2_window(&self) -> [i16; GAP2_LEN] {
        let mut out = [0i16; GAP2_LEN];
        out.copy_from_slice(&self.buf[GAP2_OFFSET..GAP2_OFFSET + GAP2_LEN]);
        out
    }

    /// The ring's [`GAP2_OFFSET_SLOT1`]-anchored 199-word window -- the source
    /// of the SECOND of the 3 per-frame `fft_bfp_transform`-loudness topcalls
    /// ("slot 1"), read at the SAME timing as [`Self::gap2_window`]: call it
    /// right after the frame's FIRST `push_half`.
    pub fn gap2_window_slot1(&self) -> [i16; GAP2_LEN] {
        let mut out = [0i16; GAP2_LEN];
        out.copy_from_slice(&self.buf[GAP2_OFFSET_SLOT1..GAP2_OFFSET_SLOT1 + GAP2_LEN]);
        out
    }

    /// The ring's [`GAP2_OFFSET_SLOT2`]-anchored (index 0),
    /// [`GAP2_LEN_WIDE`]-wide (255-word) window -- the source of the THIRD of the
    /// 3 per-frame `fft_bfp_transform`-loudness topcalls ("slot 2").
    ///
    /// Read at a DIFFERENT timing from
    /// [`Self::gap2_window`]/[`Self::gap2_window_slot1`]: call it right after the
    /// frame's SECOND `push_half`, i.e. after BOTH pre-filter halves have run.
    pub fn gap2_window_slot2(&self) -> [i16; GAP2_LEN_WIDE] {
        let mut out = [0i16; GAP2_LEN_WIDE];
        out.copy_from_slice(&self.buf[GAP2_OFFSET_SLOT2..GAP2_OFFSET_SLOT2 + GAP2_LEN_WIDE]);
        out
    }
}

impl Default for HistoryRing {
    fn default() -> Self {
        Self::new()
    }
}
