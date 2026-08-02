//! `COUNT` (harmonic count `L`) -> `STEP` (phase-accumulator seed) mapping,
//! supplying the `step` argument of `band_decompress`.
//!
//! Each entry is the modal (most frequent) `STEP` observed for that `COUNT`
//! across ~17.7k live captures of the reference DLL, spanning the full
//! observed AMBE+2 harmonic-count range `9..=56`. Every in-range entry is
//! captured data; the `K/COUNT` fallback applies only outside `9..=56`.
//!
//! ## The mapping is approximate, and that is inherent
//!
//! `STEP` is not a function of the integer `COUNT` alone: the captures hold
//! 104 distinct `STEP` values for 48 distinct `COUNT` values, and `STEP*COUNT`
//! is only roughly constant (mean 237587, stdev ~1.6%). The driving quantity
//! is finer-grained than `L` — most plausibly the continuous pitch
//! fundamental `omega_0`, since each integer `L` spans a whole range of
//! `omega_0`. That remains a hypothesis: confirming it needs a probe that
//! captures `omega_0` and `STEP` in the same frame, and the `[108c0-probe]`
//! hook carries no pitch field.
//!
//! Accuracy on real speech is **~57-59% exact**. Over the whole pooled
//! capture set — synthetic stimuli and real speech together — 99.9% of
//! observations fall within 5% of the modal value and the exact-match rate
//! reads higher (~62%). That pooled exact figure is misleading: a
//! constant-frequency tone collapses onto a single canonical `STEP` per
//! `COUNT` and so trivially agrees with its own mode. Real speech's richer
//! spectral content lands on the split boundaries far more often — `COUNT=56`,
//! for instance, splits three ways (`4217`, `4260`, and rarely `4325`), and
//! only pure tones give `4217` every time.
//!
//! **Do not re-fit this table.** Held-out cross-validation puts it at the
//! ceiling of what a `COUNT -> modal STEP` mechanism reaches on real speech:
//! per-file-equal weighting, real-speech-only tables, and 5x/20x real-speech
//! weighting all land within a point of the shipped values. Beating ~58%
//! requires modelling the finer-grained quantity, not reweighting the same
//! observations — and the DLL that produced them is retired, so the capture
//! set cannot be extended.

/// `(COUNT, modal STEP)` for every `COUNT` in `9..=56`. All 48 entries are
/// captured reference data; no in-range row is extrapolated.
const MODAL_STEP_TABLE: [(u8, u16); 48] = [
    (9, 26199),
    (10, 22837), // captured; a few % under K/COUNT, as the low-COUNT rows generally are
    (11, 20525),
    (12, 19608),
    (13, 17902),
    (14, 16317),
    (15, 15347),
    (16, 14438),
    (17, 14220),
    (18, 13175),
    (19, 12584),
    (20, 11658),
    (21, 11482),
    (22, 10803),
    (23, 10164),
    (24, 9859),
    (25, 9419),
    (26, 9139),
    (27, 8866),
    (28, 8474),
    (29, 8224),
    (30, 7836),
    (31, 7718),
    (32, 7372),
    (33, 7260),
    (34, 6935),
    (35, 6829),
    (36, 6726),
    (37, 6524),
    (38, 6327),
    (39, 6136),
    (40, 5952),
    (41, 5863),
    (42, 5686),
    (43, 5600),
    (44, 5431),
    (45, 5349),
    (46, 5269),
    (47, 5110),
    (48, 4956),
    (49, 4882),
    (50, 4809),
    (51, 4736),
    (52, 4664),
    (53, 4526),
    (54, 4457),
    (55, 4391),
    (56, 4217),
];

/// Mean of `STEP*COUNT` over the 48 modal entries. Used only to extrapolate
/// `COUNT` values outside the observed `9..=56` range.
const K_FALLBACK: i64 = 237587;

/// `COUNT -> STEP`: the captured modal value in range, a `K/COUNT`
/// fixed-point-reciprocal extrapolation outside it. See the module doc for
/// the accuracy this mapping actually achieves.
pub(crate) fn step_for_count(count: u32) -> i16 {
    for &(c, s) in MODAL_STEP_TABLE.iter() {
        if c as u32 == count {
            return s as i16;
        }
    }
    if count == 0 {
        return i16::MAX;
    }
    let est = K_FALLBACK / count as i64;
    est.clamp(0, i16::MAX as i64) as i16
}

/// `omega_0`-refined `STEP`, exact closed form. `STEP` is the Q11 fixed-point of
/// the harmonic spacing `omega_0` expressed in 256-point-FFT bins — the units
/// `band_decompress` steps its 129-entry mag-squared table with. The integer `L`
/// no longer enters the mapping.
pub(crate) fn step_for_omega(_l: u32, omega_0: f64) -> i16 {
    // Exact closed form: STEP is the Q11 fixed-point of the harmonic spacing
    // omega_0 expressed in 256-point-FFT bins — the units band_decompress
    // steps its 129-entry mag-squared table with. STEP = round(omega_0 * 2^18/pi)
    // = round(omega_0 * (256/2pi) * 2048). Reproduces captured reference STEP on
    // 12/12 single-omega cells and 35/36 modal cells exactly; the shipped modal
    // approximation diverged on 103/120 pitch-table entries by up to 1749.
    const K: f64 = 262144.0 / core::f64::consts::PI; // 2^18/pi = 83443.0268
    (omega_0 * K + 0.5).floor().clamp(0.0, i16::MAX as f64) as i16
}
