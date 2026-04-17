//! §0.3 — Initial pitch estimation.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.3 (BABA-A §5.1.1
//! through §5.1.4, Equations 5–23). Three stages:
//!
//!   (a) E(P) — error function on the half-sample grid, "close to 0 is good".
//!       Uses a squared-envelope autocorrelation r(t) of the LPF'd signal.
//!   (b) Look-back tracker: argmin E(P) on [0.8·P̂_{−1}, 1.2·P̂_{−1}],
//!       then CE_B = E(P̂_B) + E_{−1} + E_{−2}.
//!   (c) Look-ahead tracker: argmin over P_0 of E(P_0) + E_1(P̂_1(P_0))
//!       + E_2(P̂_2(P_0)), with pitch-continuity clamps; then sub-multiple
//!       cascade (Eq. 18/19/20) to correct octave-doubling.
//!   (d) Decision: CE_B ≤ 0.48 ⇒ P̂_B, else CE_B ≤ CE_F ⇒ P̂_B, else P̂_F.

use super::{analysis_window, lpf_tap};

/// Half-support of Annex B window `w_I(n)`: `n ∈ [−W_I_HALF, W_I_HALF]`.
pub const W_I_HALF: i32 = 150;

/// Half-support of Annex D LPF `h_LPF(n)`: `n ∈ [−H_LPF_HALF, H_LPF_HALF]`.
pub const H_LPF_HALF: i32 = 10;

/// Half-support of the input buffer that `PitchSearch::new` consumes.
/// The LPF needs `±H_LPF_HALF` samples beyond the `w_I` support, so the
/// input must cover `n ∈ [−(W_I_HALF + H_LPF_HALF), W_I_HALF + H_LPF_HALF]`.
pub const PITCH_INPUT_HALF: i32 = W_I_HALF + H_LPF_HALF;

/// Length of the input buffer passed to [`PitchSearch::new`]: 321 samples
/// covering `n ∈ [−160, 160]`, zero-extended outside.
pub const PITCH_INPUT_LEN: usize = (2 * PITCH_INPUT_HALF + 1) as usize;

/// Minimum of the pitch candidate grid (Eq. 11).
pub const PITCH_GRID_MIN: f64 = 21.0;

/// Maximum of the pitch candidate grid (Eq. 11).
pub const PITCH_GRID_MAX: f64 = 122.0;

/// Spacing between adjacent grid candidates (half-sample resolution).
pub const PITCH_GRID_STEP: f64 = 0.5;

/// Number of candidates on the Eq. 11 grid: `{21, 21.5, …, 121.5, 122}`.
pub const PITCH_GRID_LEN: usize = 203;

/// Pitch-continuity window half-width (Eq. 10, Eq. 14, Eq. 16):
/// `[0.8·P_ref, 1.2·P_ref]`.
const PITCH_CONTINUITY_LO: f64 = 0.8;
const PITCH_CONTINUITY_HI: f64 = 1.2;

/// Energy floor guarding the Eq. 5 denominator on silent frames.
const ENERGY_FLOOR: f64 = 1e-12;

/// Cold-start pitch for `P̂_{−1}` and `P̂_{−2}` per §0.3.7.
pub const PITCH_COLD_START: f64 = 100.0;

/// Eq. 9: `s_LPF(n) = Σ_{j} s(n − j)·h_LPF(j)` for `j ∈ [−10, 10]`.
///
/// `s_input` covers `n ∈ [−PITCH_INPUT_HALF, PITCH_INPUT_HALF]`. The
/// returned buffer covers the `w_I` support `n ∈ [−W_I_HALF, W_I_HALF]`.
/// Samples outside `s_input` are treated as zero (addendum §0.3.1).
pub fn compute_s_lpf(s_input: &[f64; PITCH_INPUT_LEN]) -> [f64; (2 * W_I_HALF + 1) as usize] {
    let mut out = [0.0f64; (2 * W_I_HALF + 1) as usize];
    for n in -W_I_HALF..=W_I_HALF {
        let mut acc = 0.0;
        for j in -H_LPF_HALF..=H_LPF_HALF {
            let src = n - j + PITCH_INPUT_HALF;
            if !(0..PITCH_INPUT_LEN as i32).contains(&src) {
                continue;
            }
            acc += s_input[src as usize] * f64::from(lpf_tap(j));
        }
        out[(n + W_I_HALF) as usize] = acc;
    }
    out
}

/// Quantizer state for one frame of pitch analysis.
///
/// Caches the Eq. 5 energy term and the Eq. 7 autocorrelation `r(t)`
/// for integer `t ∈ [0, 122]`, so `E(P)` can be evaluated cheaply for
/// any grid candidate. The fourth moment of `w_I` (a window-dependent
/// constant) is also precomputed so the denominator is one multiply +
/// one subtract away.
#[derive(Clone, Debug)]
pub struct PitchSearch {
    /// `Σ_{j} s²_LPF(j) · w²_I(j)` — total windowed energy.
    pub(super) energy: f64,
    /// `Σ w_I⁴(j)` — window-dependent denominator constant (Eq. 5).
    pub(super) w_i_fourth_moment: f64,
    /// `r(t) = Σ_{j} s_LPF(j)·w²_I(j) · s_LPF(j+t)·w²_I(j+t)` for
    /// `t ∈ [0, R_MAX_LAG]` per addendum §0.3.2 Eq. 7.
    ///
    /// `R_MAX_LAG = W_I_HALF = 150`: Eq. 5's sum reaches lags
    /// `n·P` with `|n| ≤ ⌊150/P⌋`, so `|n·P| ≤ 150` by construction.
    pub(super) r: [f64; R_MAX_LAG + 1],
}

const R_MAX_LAG: usize = W_I_HALF as usize;

impl PitchSearch {
    /// Build the per-frame pitch search from the 321-sample analysis
    /// buffer (see [`PITCH_INPUT_LEN`]).
    pub fn new(s_input: &[f64; PITCH_INPUT_LEN]) -> Self {
        let s_lpf = compute_s_lpf(s_input);
        Self::from_lpf(&s_lpf)
    }

    /// Build from an already-LPF'd buffer. Useful for tests that want
    /// to inject a synthetic `s_LPF(n)` without round-tripping through
    /// [`compute_s_lpf`].
    pub fn from_lpf(s_lpf: &[f64; (2 * W_I_HALF + 1) as usize]) -> Self {
        // Energy Σ s²_LPF·w²_I, fourth moment Σ w_I⁴, and the
        // per-frame envelope sw2[j] = s_LPF(j)·w²_I(j) that feeds the
        // Eq. 7 autocorrelation. Addendum §0.3.8: note s_LPF is
        // linear and the window is squared — see
        // analysis/vocoder_analysis_eq7_correction.md.
        let mut sw2 = [0.0f64; (2 * W_I_HALF + 1) as usize];
        let mut energy = 0.0;
        let mut w4 = 0.0;
        for j in -W_I_HALF..=W_I_HALF {
            let idx = (j + W_I_HALF) as usize;
            let s = s_lpf[idx];
            let w = f64::from(analysis_window(j));
            let w2 = w * w;
            sw2[idx] = s * w2;
            energy += s * s * w2;
            w4 += w2 * w2;
        }
        // Eq. 7 via hoisted envelope — r(t) = Σ sw2[j]·sw2[j+t].
        let mut r = [0.0f64; R_MAX_LAG + 1];
        for t in 0..=R_MAX_LAG as i32 {
            let mut acc = 0.0;
            let j_lo = (-W_I_HALF).max(-W_I_HALF - t);
            let j_hi = W_I_HALF.min(W_I_HALF - t);
            for j in j_lo..=j_hi {
                let a = sw2[(j + W_I_HALF) as usize];
                let b = sw2[(j + t + W_I_HALF) as usize];
                acc += a * b;
            }
            r[t as usize] = acc;
        }
        Self {
            energy,
            w_i_fourth_moment: w4,
            r,
        }
    }

    /// Eq. 8: linear interpolation of `r(t)` between integer lags.
    fn r_at(&self, t: f64) -> f64 {
        let t_floor = t.floor();
        let frac = t - t_floor;
        let ti = t_floor as usize;
        if ti + 1 >= self.r.len() {
            return *self.r.last().unwrap_or(&0.0);
        }
        (1.0 - frac) * self.r[ti] + frac * self.r[ti + 1]
    }

    /// Eq. 5: error function `E(P)`. Returns `1.0` (the "no-confidence"
    /// value) on silent frames where the denominator would blow up.
    pub fn e_of_p(&self, p: f64) -> f64 {
        if self.energy < ENERGY_FLOOR {
            return 1.0;
        }
        let n_max = (f64::from(W_I_HALF) / p).floor() as i32;
        let mut inner = 0.0;
        for n in -n_max..=n_max {
            inner += self.r_at((f64::from(n) * p).abs());
        }
        let num = self.energy - p * inner;
        let denom = self.energy * (1.0 - p * self.w_i_fourth_moment);
        if denom <= 0.0 {
            return 1.0;
        }
        num / denom
    }

    /// Argmin of `E(P)` over the half-sample grid within `[p_lo, p_hi]`,
    /// both bounds clamped to `[PITCH_GRID_MIN, PITCH_GRID_MAX]`. The
    /// range is inclusive on both ends. Returns `(P̂, E(P̂))`.
    pub fn argmin_in_range(&self, p_lo: f64, p_hi: f64) -> (f64, f64) {
        let p_lo = p_lo.max(PITCH_GRID_MIN);
        let p_hi = p_hi.min(PITCH_GRID_MAX);
        // Round up to the next grid point at p_lo; round down at p_hi.
        let i_lo = ((p_lo - PITCH_GRID_MIN) / PITCH_GRID_STEP).ceil() as i32;
        let i_hi = ((p_hi - PITCH_GRID_MIN) / PITCH_GRID_STEP).floor() as i32;
        let mut best_p = PITCH_GRID_MIN + f64::from(i_lo) * PITCH_GRID_STEP;
        let mut best_e = self.e_of_p(best_p);
        for i in (i_lo + 1)..=i_hi {
            let p = PITCH_GRID_MIN + f64::from(i) * PITCH_GRID_STEP;
            let e = self.e_of_p(p);
            if e < best_e {
                best_e = e;
                best_p = p;
            }
        }
        (best_p, best_e)
    }
}

/// Snap `p` to the nearest half-sample grid value. Returns `None` if
/// the snapped value falls outside `[PITCH_GRID_MIN, PITCH_GRID_MAX]`.
#[inline]
pub fn snap_to_pitch_grid(p: f64) -> Option<f64> {
    let snapped = (p / PITCH_GRID_STEP).round() * PITCH_GRID_STEP;
    if (PITCH_GRID_MIN..=PITCH_GRID_MAX).contains(&snapped) {
        Some(snapped)
    } else {
        None
    }
}

/// Past-frame context consumed by [`look_back`] — the selected pitch
/// from the previous frame and the frozen `E_{−1}`, `E_{−2}` scalar
/// values at the past two frames' selected pitches. Cold-start values
/// are encoded in [`LookBackContext::cold_start`] per §0.3.7.
#[derive(Clone, Copy, Debug)]
pub struct LookBackContext {
    /// `P̂_{−1}` — previous frame's selected pitch.
    pub prev_pitch: f64,
    /// `E_{−1}(P̂_{−1})` — previous frame's error at its selected pitch.
    pub prev_err_1: f64,
    /// `E_{−2}(P̂_{−2})` — two-frames-back error at its selected pitch.
    pub prev_err_2: f64,
}

impl LookBackContext {
    /// Cold-start values per addendum §0.3.7.
    pub const fn cold_start() -> Self {
        Self {
            prev_pitch: PITCH_COLD_START,
            prev_err_1: 0.0,
            prev_err_2: 0.0,
        }
    }
}

/// Look-back pitch tracking per Eq. 10–12. Returns `(P̂_B, CE_B)`.
pub fn look_back(current: &PitchSearch, ctx: LookBackContext) -> (f64, f64) {
    let p_lo = PITCH_CONTINUITY_LO * ctx.prev_pitch;
    let p_hi = PITCH_CONTINUITY_HI * ctx.prev_pitch;
    let (p_b, e_b) = current.argmin_in_range(p_lo, p_hi);
    let ce_b = e_b + ctx.prev_err_1 + ctx.prev_err_2;
    (p_b, ce_b)
}

/// Look-ahead pitch tracking per Eq. 13–17, plus the sub-multiple
/// cascade (Eq. 18–20). Returns `(P̂_F, CE_F)`.
///
/// `next1` and `next2` are the pitch-search tables for frames `t+1`
/// and `t+2`; the caller is responsible for the two-frame lookahead.
/// On cold start (before enough lookahead has accumulated) the higher-
/// level pipeline should route through silence/erasure emission per
/// addendum §0.3.7 rather than call this function with placeholder
/// tables — the result would be a pitch derived from zeros.
pub fn look_ahead(
    current: &PitchSearch,
    next1: &PitchSearch,
    next2: &PitchSearch,
) -> (f64, f64) {
    // Scan P_0 over the full half-sample grid. For each P_0, find
    // P̂_1 ∈ [0.8·P_0, 1.2·P_0] minimizing E_1, then P̂_2 ∈
    // [0.8·P̂_1, 1.2·P̂_1] minimizing E_2. CE_F(P_0) is the sum.
    let mut best_p0 = PITCH_GRID_MIN;
    let mut best_ce = f64::INFINITY;
    for i in 0..PITCH_GRID_LEN as i32 {
        let p0 = PITCH_GRID_MIN + f64::from(i) * PITCH_GRID_STEP;
        let e0 = current.e_of_p(p0);
        let (p1, e1) = next1.argmin_in_range(
            PITCH_CONTINUITY_LO * p0,
            PITCH_CONTINUITY_HI * p0,
        );
        let (_p2, e2) = next2.argmin_in_range(
            PITCH_CONTINUITY_LO * p1,
            PITCH_CONTINUITY_HI * p1,
        );
        let ce = e0 + e1 + e2;
        if ce < best_ce {
            best_ce = ce;
            best_p0 = p0;
        }
    }

    // Sub-multiple cascade per BABA-A §5.1.4 (PDF p. 15): check the
    // **smallest** sub-multiple first (largest `n`), then the next-
    // larger, and so on. The first one to satisfy any of Eq. 18/19/20
    // wins. Ratios are computed against `CE_F(P̂_0)`.
    //
    // Iteration: `n` runs from `⌊P̂_0 / 21⌋` down to `2`. Sub-multiples
    // with `P̂_0 / n < 21` are disregarded (PDF prose).
    let ce_p0 = best_ce;
    let n_max = (best_p0 / PITCH_GRID_MIN).floor() as u32;
    if n_max >= 2 {
        for n in (2..=n_max).rev() {
            let raw = best_p0 / f64::from(n);
            if raw < PITCH_GRID_MIN {
                continue;
            }
            let Some(sub) = snap_to_pitch_grid(raw) else {
                continue;
            };
            let e0s = current.e_of_p(sub);
            let (p1s, e1s) = next1.argmin_in_range(
                PITCH_CONTINUITY_LO * sub,
                PITCH_CONTINUITY_HI * sub,
            );
            let (_p2s, e2s) = next2.argmin_in_range(
                PITCH_CONTINUITY_LO * p1s,
                PITCH_CONTINUITY_HI * p1s,
            );
            let ce_sub = e0s + e1s + e2s;
            let ratio = if ce_p0 > 0.0 {
                ce_sub / ce_p0
            } else {
                f64::INFINITY
            };
            let eq18 = ce_sub <= 0.85 && ratio <= 1.7;
            let eq19 = ce_sub <= 0.40 && ratio <= 3.5;
            let eq20 = ce_sub <= 0.05;
            if eq18 || eq19 || eq20 {
                return (sub, ce_sub);
            }
        }
    }
    (best_p0, ce_p0)
}

/// Eq. 21–23: choose `P̂_I ∈ {P̂_B, P̂_F}`.
///
/// Bias toward pitch continuity: when the backward hypothesis is strong
/// in absolute terms (`CE_B ≤ 0.48`) it wins unconditionally; otherwise
/// the two compete head-to-head.
#[inline]
pub fn decide_initial_pitch(p_b: f64, ce_b: f64, p_f: f64, ce_f: f64) -> f64 {
    if ce_b <= 0.48 {
        p_b
    } else if ce_b <= ce_f {
        p_b
    } else {
        p_f
    }
}
