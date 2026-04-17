//! MBE baseline analysis encoder — PCM → [`MbeParams`].
//!
//! The forward direction of the vocoder: 8 kHz 16-bit PCM in, one
//! [`MbeParams`] out per 20 ms (160-sample) frame. The result is
//! consumed by any of the wire-side `quantize` pipelines
//! ([`crate::p25_fullrate::quantize`],
//! [`crate::p25_halfrate::quantize`],
//! [`crate::dvsi_3000::quantize`]) and is independent of wire format.
//!
//! # Status
//!
//! Implementation is in progress against the derived addendum
//! `~/blip25-specs/analysis/vocoder_analysis_encoder_addendum.md`
//! (§0.1–§0.11). The top-level [`encode`] entry point still reports
//! [`AnalysisError::SpecGap`] because the end-to-end pipeline
//! (§0.10) depends on sections not yet landed. What is implemented
//! below is §0.2 — the `S_w(m)` signal DFT and the `S_w(m, ω₀)` /
//! `A_l(ω₀)` harmonic basis — which is the critical-path primitive
//! consumed by §0.4 (pitch refinement), §0.5 (spectral amplitude),
//! and §0.7 (V/UV discriminant).
//!
//! # Numerical precision
//!
//! Per addendum §0.11.7 the analysis pipeline uses `f64` throughout;
//! the Annex B/C/D tables are stored as `f32` (extraction-time
//! fidelity) and promoted at use.
//!
//! # Annex tables available here
//!
//! - **Annex B** — analysis window `wI(n)`, 301 values, used by initial
//!   pitch estimation. [`analysis_window`], [`IMBE_ANALYSIS_WINDOW`].
//! - **Annex C** — pitch refinement window `wR(n)`, 221 values, used by
//!   pitch refinement and spectral amplitude estimation.
//!   [`refinement_window`], [`IMBE_REFINEMENT_WINDOW`].
//! - **Annex D** — FIR LPF `h_LPF(n)`, 21 taps, used by the initial
//!   pitch autocorrelation. [`lpf_tap`], [`IMBE_ANNEX_D_LPF`].

use crate::mbe_params::{MbeParams, SAMPLES_PER_FRAME};

include!(concat!(env!("OUT_DIR"), "/annex_b_analysis_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_c_refinement_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_d_lpf.rs"));

/// Lookup `wI(n)` for `n ∈ [−150, 150]`. Returns `0.0` outside that
/// range (the spec treats `wI` as zero outside its support).
#[inline]
pub fn analysis_window(n: i32) -> f32 {
    if !(-150..=150).contains(&n) {
        return 0.0;
    }
    IMBE_ANALYSIS_WINDOW[(n + 150) as usize]
}

/// Lookup `wR(n)` for `n ∈ [−110, 110]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn refinement_window(n: i32) -> f32 {
    if !(-110..=110).contains(&n) {
        return 0.0;
    }
    IMBE_REFINEMENT_WINDOW[(n + 110) as usize]
}

/// Lookup `h_LPF(n)` for `n ∈ [−10, 10]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn lpf_tap(n: i32) -> f32 {
    if !(-10..=10).contains(&n) {
        return 0.0;
    }
    IMBE_ANNEX_D_LPF[(n + 10) as usize]
}

// ---------------------------------------------------------------------------
// §0.2 — 256-point signal DFT and the S_w(m, ω₀) harmonic basis.
//
// Reference: vocoder_analysis_encoder_addendum.md §0.2 (BABA-A §5.1.5,
// Equations 24–30). The synthetic spectrum S_w(m, ω₀) and its
// per-harmonic amplitude A_l(ω₀) are consumed by:
//
//   §0.4 pitch refinement — minimizes E_R(ω₀) = Σ |S_w(m) − S_w(m, ω₀)|²
//   §0.5 spectral amplitude estimation — Eq. 43/44 use |S_w(m)|² and
//                                         the bin endpoints ⌈a_l⌉, ⌈b_l⌉
//   §0.7 V/UV discriminant — the voicing measure D_k compares
//                             |S_w(m) − A_l·W_R(·)|² against |S_w(m)|²
// ---------------------------------------------------------------------------

/// Size of the per-frame signal DFT (Eq. 29).
pub const DFT_SIZE: usize = 256;

/// Size of the window-only DFT (Eq. 30).
pub const WINDOW_DFT_SIZE: usize = 16384;

/// Support of Annex C window `w_R(n)`: `n ∈ [−W_R_HALF, W_R_HALF]`.
pub const W_R_HALF: i32 = 110;

/// Minimal complex number used by the analysis DFT path.
///
/// Holds real and imaginary components as `f64`. Implemented inline to
/// avoid a numerics dependency; the three operations we actually need
/// (scaled accumulation against a real window coefficient, magnitude,
/// scalar divide) are spelled out at their call sites rather than
/// wrapped in an arithmetic-operator trait soup.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// The additive identity.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Construct from real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Squared magnitude `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Precomputed `W_R(k)` table — the 16384-point DFT of the Annex C
/// window (Eq. 30). Because `w_R(n)` is real and symmetric, `W_R(k)`
/// is real-valued and symmetric in `k`; we store only `k ≥ 0` and mirror
/// on access.
///
/// Construction cost: a single 16384 × 221 loop at startup
/// (~3.6 M multiplies). Memory: `8193 × 8 B = 64 KiB`.
#[derive(Clone, Debug)]
pub struct HarmonicBasis {
    /// `W_R(k)` for `k ∈ [0, WINDOW_DFT_SIZE/2]`. Access via [`Self::w_r`].
    w_r_pos: Box<[f64; WINDOW_DFT_SIZE / 2 + 1]>,
}

impl HarmonicBasis {
    /// Build the `W_R(k)` table by direct evaluation of Eq. 30. O(N²)
    /// but runs once at startup.
    pub fn new() -> Self {
        let mut table = Box::new([0.0f64; WINDOW_DFT_SIZE / 2 + 1]);
        let two_pi = 2.0 * core::f64::consts::PI;
        let dft_size = WINDOW_DFT_SIZE as f64;
        for (k, slot) in table.iter_mut().enumerate() {
            let k_f = k as f64;
            let mut acc = 0.0;
            // w_R is symmetric; fold the n = +i / n = −i pairs to halve
            // the cosine evaluations. The n = 0 term is w_R(0) = 1.0.
            acc += f64::from(refinement_window(0));
            for n in 1..=W_R_HALF {
                let w = f64::from(refinement_window(n));
                if w == 0.0 {
                    continue;
                }
                let angle = two_pi * k_f * f64::from(n) / dft_size;
                acc += 2.0 * w * angle.cos();
            }
            *slot = acc;
        }
        Self { w_r_pos: table }
    }

    /// `W_R(k)` for any integer `k`. Returns `0.0` outside the
    /// `[−WINDOW_DFT_SIZE/2, WINDOW_DFT_SIZE/2]` stored range (in
    /// practice only `|k| ≲ 1024` is ever indexed per addendum
    /// §0.2.8; the wide bound is the PDF definition).
    #[inline]
    pub fn w_r(&self, k: i32) -> f64 {
        let abs = k.unsigned_abs() as usize;
        if abs >= self.w_r_pos.len() {
            0.0
        } else {
            self.w_r_pos[abs]
        }
    }

    /// Bin endpoints `⌈a_l⌉`, `⌈b_l⌉` for harmonic `l` at fundamental
    /// `ω₀` (Eq. 26–27). The support is the half-open integer interval
    /// `⌈a_l⌉ ≤ m < ⌈b_l⌉`.
    #[inline]
    pub fn bin_endpoints(l: u32, omega0: f64) -> (i32, i32) {
        let scale = DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let a_l = scale * (f64::from(l) - 0.5) * omega0;
        let b_l = scale * (f64::from(l) + 0.5) * omega0;
        (ceil_i32(a_l), ceil_i32(b_l))
    }

    /// Harmonic amplitude `A_l(ω₀)` per Eq. 28. The numerator is
    /// complex (it inherits the phase of `S_w(m)`); the denominator is
    /// real (sum of squared window-spectrum magnitudes over the
    /// harmonic's support).
    ///
    /// Returns [`Complex64::ZERO`] when the denominator is zero
    /// (empty support — e.g. a degenerate `ω₀` that collapses
    /// `⌈a_l⌉ = ⌈b_l⌉`). This is a numerical guard, not an expected
    /// condition for admissible `ω₀ ∈ [2π/123.125, 2π/19.875]`.
    pub fn harmonic_amplitude(
        &self,
        sw: &[Complex64; DFT_SIZE],
        l: u32,
        omega0: f64,
    ) -> Complex64 {
        let (m_lo, m_hi) = Self::bin_endpoints(l, omega0);
        let bin_scale = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let l_offset = bin_scale * f64::from(l) * omega0;
        let mut num_re = 0.0;
        let mut num_im = 0.0;
        let mut den = 0.0;
        for m in m_lo..m_hi {
            let k = floor_i32(64.0 * f64::from(m) - l_offset + 0.5);
            let wr = self.w_r(k);
            let s = sw[packed_index(m)];
            // W_R is real, so W_R* = W_R; Eq. 28's conjugate is a no-op.
            num_re += s.re * wr;
            num_im += s.im * wr;
            den += wr * wr;
        }
        if den > 0.0 {
            Complex64::new(num_re / den, num_im / den)
        } else {
            Complex64::ZERO
        }
    }

    /// Evaluate the synthetic spectrum `S_w(m, ω₀)` at a single
    /// `(m, l)` point per Eq. 25. The caller supplies the precomputed
    /// `A_l(ω₀)` (from [`Self::harmonic_amplitude`]). `m` must be in
    /// the harmonic's support `⌈a_l⌉ ≤ m < ⌈b_l⌉`; outside that, the
    /// PDF's synthetic spectrum is not defined (other harmonics cover
    /// adjacent bins).
    #[inline]
    pub fn synthetic_bin(&self, m: i32, l: u32, omega0: f64, a_l: Complex64) -> Complex64 {
        let bin_scale = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let k = floor_i32(64.0 * f64::from(m) - bin_scale * f64::from(l) * omega0 + 0.5);
        let wr = self.w_r(k);
        Complex64::new(a_l.re * wr, a_l.im * wr)
    }
}

impl Default for HarmonicBasis {
    fn default() -> Self {
        Self::new()
    }
}

/// 256-point DFT of the windowed input signal (Eq. 29).
///
/// `signal_centered` holds `s(−110), s(−109), …, s(110)` (221 samples)
/// — the portion of the analysis stream aligned with the current
/// frame's refinement window per addendum §0.1. Returns `S_w(m)` for
/// the two-sided index range `m ∈ [−127, 128]`, packed into a 256-entry
/// array via DFT periodicity: index `0..=128` holds `m = 0..=128`,
/// index `129..=255` holds `m = −127..=−1` (i.e. index `i` maps to
/// `m = i` if `i ≤ 128`, else `m = i − 256`). Use [`packed_index`] to
/// translate a two-sided `m` into its array slot.
///
/// Direct-form DFT; O(N·W) = 256·221 ≈ 56 k multiplies per frame.
/// Fast enough for correctness-first development and leaves the door
/// open for a future FFT-based reimplementation without changing the
/// signature.
pub fn signal_spectrum(signal_centered: &[f64; (2 * W_R_HALF + 1) as usize]) -> [Complex64; DFT_SIZE] {
    let mut out = [Complex64::ZERO; DFT_SIZE];
    let two_pi = 2.0 * core::f64::consts::PI;
    let dft_size = DFT_SIZE as f64;
    for (idx, slot) in out.iter_mut().enumerate() {
        let m = unpacked_m(idx);
        let m_f = f64::from(m);
        let mut re = 0.0;
        let mut im = 0.0;
        for n in -W_R_HALF..=W_R_HALF {
            let w = f64::from(refinement_window(n));
            if w == 0.0 {
                continue;
            }
            let s = signal_centered[(n + W_R_HALF) as usize];
            let angle = -two_pi * m_f * f64::from(n) / dft_size;
            re += s * w * angle.cos();
            im += s * w * angle.sin();
        }
        *slot = Complex64::new(re, im);
    }
    out
}

/// Map a two-sided DFT index `m ∈ [−127, 128]` to its packed slot in
/// `[0, 256)`, per the convention in [`signal_spectrum`].
#[inline]
pub fn packed_index(m: i32) -> usize {
    if m < 0 {
        (m + DFT_SIZE as i32) as usize
    } else {
        (m as usize) & (DFT_SIZE - 1)
    }
}

/// Inverse of [`packed_index`] for `idx ∈ [0, 256)`: returns `m`
/// in the two-sided range. Slots `0..=128` map to positive `m`;
/// slots `129..=255` map to `m − 256`.
#[inline]
fn unpacked_m(idx: usize) -> i32 {
    if idx <= DFT_SIZE / 2 {
        idx as i32
    } else {
        idx as i32 - DFT_SIZE as i32
    }
}

#[inline]
fn ceil_i32(x: f64) -> i32 {
    x.ceil() as i32
}

#[inline]
fn floor_i32(x: f64) -> i32 {
    x.floor() as i32
}

// ---------------------------------------------------------------------------
// §0.3 — Initial pitch estimation.
//
// Reference: vocoder_analysis_encoder_addendum.md §0.3 (BABA-A §5.1.1
// through §5.1.4, Equations 5–23). Three stages:
//
//   (a) E(P) — error function on the half-sample grid, "close to 0 is good".
//       Uses a squared-envelope autocorrelation r(t) of the LPF'd signal.
//   (b) Look-back tracker: argmin E(P) on [0.8·P̂_{−1}, 1.2·P̂_{−1}],
//       then CE_B = E(P̂_B) + E_{−1} + E_{−2}.
//   (c) Look-ahead tracker: argmin over P_0 of E(P_0) + E_1(P̂_1(P_0))
//       + E_2(P̂_2(P_0)), with pitch-continuity clamps; then sub-multiple
//       cascade (Eq. 18/19/20) to correct octave-doubling.
//   (d) Decision: CE_B ≤ 0.48 ⇒ P̂_B, else CE_B ≤ CE_F ⇒ P̂_B, else P̂_F.
// ---------------------------------------------------------------------------

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
    energy: f64,
    /// `Σ w_I⁴(j)` — window-dependent denominator constant (Eq. 5).
    w_i_fourth_moment: f64,
    /// `r(t) = Σ_{j} s_LPF(j)·w²_I(j) · s_LPF(j+t)·w²_I(j+t)` for
    /// `t ∈ [0, R_MAX_LAG]` per addendum §0.3.2 Eq. 7.
    ///
    /// `R_MAX_LAG = W_I_HALF = 150`: Eq. 5's sum reaches lags
    /// `n·P` with `|n| ≤ ⌊150/P⌋`, so `|n·P| ≤ 150` by construction.
    r: [f64; R_MAX_LAG + 1],
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

// ---------------------------------------------------------------------------
// §0.4 — Pitch refinement lives in the `refine` submodule.

pub mod refine;
pub use refine::{
    B0_MAX, L_HAT_MAX, L_HAT_MIN, N_REFINE_CANDIDATES, PitchRefinement,
    REFINE_OFFSETS, harmonic_count_for, quantize_pitch_index, refine_pitch,
};
#[cfg(test)]
use refine::residual_e_r;

// ---------------------------------------------------------------------------
// §0.7 — Voiced/Unvoiced determination.
//
// Reference: vocoder_analysis_encoder_addendum.md §0.7 (BABA-A §5.2
// Eq. 34–42, Figure 11). Per-band voicing decisions v̂_k ∈ {0, 1} for
// 1 ≤ k ≤ K̂, where K̂ is derived from L̂ via Eq. 34. Consumes
// precomputed S_w(m) from §0.2, ω̂_0/L̂/â_l/b̂_l from §0.4, and
// E(P̂_I) from §0.3.
// ---------------------------------------------------------------------------

/// Maximum band count `K̂` per Eq. 34 (cap).
pub const K_HAT_MAX: u8 = 12;

/// Minimum `ξ_max` floor per Eq. 41 Case 3.
pub const XI_MAX_FLOOR: f64 = 20000.0;

/// Threshold for Eq. 37 Case 1 (poor pitch match).
const E_P_POOR_MATCH_THRESHOLD: f64 = 0.5;

/// `Θ_ξ` base when the band was voiced last frame (hysteresis up).
const THETA_BASE_VOICED_HISTORY: f64 = 0.5625;

/// `Θ_ξ` base when the band was unvoiced last frame (hysteresis down).
const THETA_BASE_UNVOICED_HISTORY: f64 = 0.45;

/// Coefficient of `(k−1)·ω̂_0` in the `Θ_ξ` pitch/band modulation.
const THETA_PITCH_BAND_COEF: f64 = 0.3096;

/// Cross-frame V/UV encoder state per §0.7.8. `v̂_k(−1)` is stored
/// 1-indexed in `vuv_prev[1..=K_HAT_MAX]`; slot 0 is unused.
#[derive(Clone, Debug)]
pub struct VuvState {
    /// `ξ_max(−1)` — previous frame's tracked peak envelope.
    xi_max: f64,
    /// `v̂_k(−1)` — previous frame's decisions, 1-indexed.
    vuv_prev: [u8; (K_HAT_MAX + 1) as usize],
    /// `K̂(−1)` — previous frame's band count (unused slots past this
    /// index are 0).
    k_prev: u8,
}

impl VuvState {
    /// Cold-start initialization per §0.7.8: `ξ_max = XI_MAX_FLOOR`,
    /// `v̂_k(−1) = 0` for all `k`, `K̂(−1) = 0`.
    pub const fn cold_start() -> Self {
        Self {
            xi_max: XI_MAX_FLOOR,
            vuv_prev: [0; (K_HAT_MAX + 1) as usize],
            k_prev: 0,
        }
    }

    /// Current tracked peak-envelope value (for inspection / tests).
    pub fn xi_max(&self) -> f64 {
        self.xi_max
    }

    /// Previous-frame decision for band `k` (1-indexed). Returns 0 for
    /// out-of-range `k` (consistent with new-index cold-start behavior
    /// described in §0.7.8).
    pub fn vuv_prev(&self, k: u8) -> u8 {
        if (1..=K_HAT_MAX).contains(&k) {
            self.vuv_prev[k as usize]
        } else {
            0
        }
    }

    /// Override the per-band V/UV history with external decisions.
    /// Used for conformance diagnostics that inject the chip's V/UV
    /// trajectory to measure feedback-divergence contribution.
    /// `decisions` is 1-indexed (`decisions[k]` for band `k`);
    /// `k_hat` is the band count to write.
    pub fn override_vuv_prev(&mut self, decisions: &[u8], k_hat: u8) {
        self.vuv_prev.fill(0);
        let n = (k_hat as usize).min(decisions.len()).min(K_HAT_MAX as usize);
        for k in 1..=n {
            self.vuv_prev[k] = decisions[k];
        }
        self.k_prev = k_hat;
    }

    /// Override ξ_max with an external value.
    pub fn override_xi_max(&mut self, xi_max: f64) {
        self.xi_max = xi_max;
    }
}

impl Default for VuvState {
    fn default() -> Self {
        Self::cold_start()
    }
}

/// Output of [`determine_vuv`].
#[derive(Clone, Debug)]
pub struct VuvResult {
    /// Band count `K̂` per Eq. 34.
    pub k_hat: u8,
    /// Per-band decisions `v̂_k ∈ {0, 1}`, 1-indexed in `vuv[1..=k_hat]`.
    /// Slots past `k_hat` are zero.
    pub vuv: [u8; (K_HAT_MAX + 1) as usize],
    /// Values of `D_k` per band (useful for debugging / DVSI diffing).
    /// 1-indexed; slots past `k_hat` are zero.
    pub d_k: [f64; (K_HAT_MAX + 1) as usize],
    /// Per-band Θ_ξ thresholds (Eq. 37). 1-indexed; slots past `k_hat`
    /// are zero. `D_k < Θ_ξ(k)` → voiced.
    pub theta_k: [f64; (K_HAT_MAX + 1) as usize],
    /// Frame energy ratio `M(ξ)` (Eq. 42).
    pub m_xi: f64,
    /// Total spectral energy `ξ_0 = ξ_LF + ξ_HF` (Eq. 38–39).
    pub xi_0: f64,
    /// `ξ_max(0)` committed to state (same as `state.xi_max` after
    /// this call returns).
    pub xi_max_after: f64,
}

/// Band count `K̂` per Eq. 34.
#[inline]
pub fn band_count_for(l_hat: u8) -> u8 {
    if l_hat <= 36 {
        (l_hat + 2) / 3
    } else {
        K_HAT_MAX
    }
}

/// Voiced/unvoiced determination for one frame per Eq. 34–42.
///
/// Mutates `state`: `ξ_max` and `v̂_k(−1)` are advanced to the current
/// frame's values on return.
pub fn determine_vuv(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    refinement: &PitchRefinement,
    e_p_hat_i: f64,
    state: &mut VuvState,
) -> VuvResult {
    let l_hat = refinement.l_hat;
    let omega_hat = refinement.omega_hat;
    let k_hat = band_count_for(l_hat);

    // Eq. 38–40: one-sided energy statistics normalized by |W_R(0)|².
    let wr_dc = basis.w_r(0);
    let wr_dc_sq = wr_dc * wr_dc;
    let inv_wr_dc_sq = if wr_dc_sq > 0.0 { 1.0 / wr_dc_sq } else { 0.0 };
    let mut xi_lf = 0.0;
    for m in 0..=63 {
        xi_lf += sw[m].norm_sqr();
    }
    xi_lf *= inv_wr_dc_sq;
    let mut xi_hf = 0.0;
    for m in 64..=128 {
        xi_hf += sw[m].norm_sqr();
    }
    xi_hf *= inv_wr_dc_sq;
    let xi_0 = xi_lf + xi_hf;

    // Eq. 41: ξ_max update (attack / slow decay / floor).
    let xi_max_new = if xi_0 > state.xi_max {
        0.5 * state.xi_max + 0.5 * xi_0
    } else {
        let decay = 0.99 * state.xi_max + 0.01 * xi_0;
        decay.max(XI_MAX_FLOOR)
    };

    // Eq. 42: M(ξ) — HF-dominant branch multiplies by √(ξ_LF / 5·ξ_HF).
    let ratio_den = 0.01 * xi_max_new + xi_0;
    let ratio = if ratio_den > 0.0 {
        (0.0025 * xi_max_new + xi_0) / ratio_den
    } else {
        0.0
    };
    let m_xi = if xi_lf >= 5.0 * xi_hf {
        ratio
    } else if xi_hf > 0.0 {
        ratio * (xi_lf / (5.0 * xi_hf)).sqrt()
    } else {
        // Degenerate: ξ_HF ≈ 0 AND ξ_LF < 5·ξ_HF implies ξ_LF ≈ 0 too;
        // entire frame is silent. M(ξ) → 0.25 via the first-branch
        // limit (ratio → 0.25 when ξ_0 = 0 and ξ_max = floor). Route
        // through branch 1 instead.
        ratio
    };

    // Per-band D_k (Eq. 35/36) and decision (Eq. 37 + strict-less-than).
    let mut vuv = [0u8; (K_HAT_MAX + 1) as usize];
    let mut d_k = [0f64; (K_HAT_MAX + 1) as usize];
    let mut theta_k = [0f64; (K_HAT_MAX + 1) as usize];
    for k in 1..=k_hat {
        let l_lo = 3 * u32::from(k) - 2;
        let l_hi = if k < k_hat {
            3 * u32::from(k)
        } else {
            u32::from(l_hat)
        };

        // Numerator Σ|S_w(m) − S_w(m, ω̂_0)|² and denominator Σ|S_w(m)|²
        // accumulated over the contiguous harmonics l_lo..=l_hi.
        let mut num = 0.0;
        let mut den = 0.0;
        for l in l_lo..=l_hi {
            let a_l = basis.harmonic_amplitude(sw, l, omega_hat);
            let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(l, omega_hat);
            for m in m_lo..m_hi {
                let synth = basis.synthetic_bin(m, l, omega_hat, a_l);
                let observed = sw[packed_index(m)];
                let dre = observed.re - synth.re;
                let dim = observed.im - synth.im;
                num += dre * dre + dim * dim;
                den += observed.norm_sqr();
            }
        }
        let d = if den > 0.0 { num / den } else { 1.0 };
        d_k[k as usize] = d;

        // Eq. 37: Θ_ξ(k, ω̂_0).
        let theta = if e_p_hat_i > E_P_POOR_MATCH_THRESHOLD && k >= 2 {
            0.0
        } else {
            let base = if state.vuv_prev[k as usize] == 1 {
                THETA_BASE_VOICED_HISTORY
            } else {
                THETA_BASE_UNVOICED_HISTORY
            };
            let modulation = 1.0 - THETA_PITCH_BAND_COEF * f64::from(k - 1) * omega_hat;
            (base * modulation * m_xi).max(0.0)
        };

        theta_k[k as usize] = theta;
        vuv[k as usize] = if d < theta { 1 } else { 0 };
    }

    // Commit state for next frame.
    state.xi_max = xi_max_new;
    state.vuv_prev.fill(0);
    for k in 1..=k_hat {
        state.vuv_prev[k as usize] = vuv[k as usize];
    }
    state.k_prev = k_hat;

    VuvResult {
        k_hat,
        vuv,
        d_k,
        theta_k,
        m_xi,
        xi_0,
        xi_max_after: xi_max_new,
    }
}

// ---------------------------------------------------------------------------
// §0.5 — Spectral Amplitude Estimation.
//
// Reference: vocoder_analysis_encoder_addendum.md §0.5 (BABA-A §5.3
// Eq. 43 voiced / Eq. 44 unvoiced, Figure 13). Produces M̂_l for
// 1 ≤ l ≤ L̂ — magnitude-only, no complex decomposition. Runs AFTER
// §0.7 V/UV determination (per Figure 5 on PDF p. 9): each harmonic
// inherits its band's v̂_k bit, which picks between Eq. 43 and Eq. 44.
// ---------------------------------------------------------------------------

/// Per-harmonic spectral amplitudes. Index 0 unused; amplitudes live
/// at indices `1..=L̂` (1-indexed per BABA-A notation). Slots past `L̂`
/// are zero.
pub type SpectralAmplitudes = [f64; L_HAT_MAX as usize + 1];

/// Map harmonic index `l` (1-indexed) to the containing V/UV band
/// `k = ⌈l / 3⌉` for `l ≤ 3(K̂−1)`; harmonics above that all fall into
/// the highest band `K̂`. Returns 0 for `l = 0` (reserved/unused).
#[inline]
pub fn band_for_harmonic(l: u32, k_hat: u8) -> u8 {
    if l == 0 || k_hat == 0 {
        return 0;
    }
    let k = ((l + 2) / 3) as u8;
    k.min(k_hat)
}

/// Estimate per-harmonic spectral amplitudes `M̂_l` per Eq. 43 / Eq. 44.
///
/// `sw` is the precomputed signal DFT (§0.2, once per frame). `basis`
/// supplies `W_R(k)` for the Eq. 43 denominator and `Σ w_R(n) = W_R(0)`
/// for the Eq. 44 external factor. `refinement` carries `ω̂_0` and `L̂`.
/// `vuv` holds the per-band bits from §0.7.
///
/// Returns a 1-indexed array; `amplitudes[l] = M̂_l` for `1 ≤ l ≤ L̂`,
/// zero elsewhere. `M̂_0` is intentionally not written (PDF §5.3:
/// "the D.C. spectral amplitude is ignored").
pub fn estimate_spectral_amplitudes(
    sw: &[Complex64; DFT_SIZE],
    basis: &HarmonicBasis,
    refinement: &PitchRefinement,
    vuv: &VuvResult,
) -> SpectralAmplitudes {
    let mut out = [0.0f64; L_HAT_MAX as usize + 1];
    let l_hat = refinement.l_hat;
    if l_hat == 0 {
        return out;
    }
    let omega_hat = refinement.omega_hat;
    let k_hat = vuv.k_hat;
    let bin_scale_16384 = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
    // Σ w_R(n) = W_R(0) per Eq. 30 at m = 0 — reuse the precomputed basis table.
    let wr_sum = basis.w_r(0);

    for l in 1..=u32::from(l_hat) {
        let k = band_for_harmonic(l, k_hat);
        let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(l, omega_hat);
        if m_hi <= m_lo {
            continue;
        }

        // Numerator Σ |S_w(m)|² — shared between Eq. 43 and Eq. 44.
        let mut s_energy = 0.0;
        for m in m_lo..m_hi {
            s_energy += sw[packed_index(m)].norm_sqr();
        }

        let is_voiced = k >= 1 && vuv.vuv[k as usize] == 1;
        let m_l = if is_voiced {
            // Eq. 43: M̂_l = sqrt(Σ|S_w(m)|² / Σ|W_R(k_m)|²).
            let l_offset = bin_scale_16384 * f64::from(l) * omega_hat;
            let mut window_energy = 0.0;
            for m in m_lo..m_hi {
                let k_bin = floor_i32(64.0 * f64::from(m) - l_offset + 0.5);
                let wr = basis.w_r(k_bin);
                window_energy += wr * wr;
            }
            if window_energy > 0.0 {
                (s_energy / window_energy).sqrt()
            } else {
                0.0
            }
        } else {
            // Eq. 44: M̂_l = (1/Σw_R) · sqrt(Σ|S_w(m)|² / bins).
            let bins = (m_hi - m_lo) as f64;
            if bins > 0.0 && wr_sum != 0.0 {
                (s_energy / bins).sqrt() / wr_sum
            } else {
                0.0
            }
        };
        out[l as usize] = m_l;
    }
    out
}

// ---------------------------------------------------------------------------
// §0.6 — Predictor lives in the `predictor` submodule.
// ---------------------------------------------------------------------------

pub mod predictor;
pub use predictor::{
    L_TILDE_COLD_START, PredictionResidual, PredictorState, compute_prediction_residual,
    fullrate_rho_f64,
};

// ---------------------------------------------------------------------------
// §0.1 — HPF lives in the `hpf` submodule.
// ---------------------------------------------------------------------------

pub mod hpf;
pub use hpf::{HPF_POLE, HpfState};

// ---------------------------------------------------------------------------
// §0.8 — Silence detection lives in the `silence` submodule.
// ---------------------------------------------------------------------------

pub mod silence;
pub use silence::SilenceDetector;
// Internal re-exports for the in-mod tests module.
#[cfg(test)]
use silence::{
    SILENCE_ENTER_FRAMES, SILENCE_ETA_INIT, SILENCE_EXIT_FRAMES,
};

/// Errors reported by the analysis encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisError {
    /// The input PCM slice was not exactly [`SAMPLES_PER_FRAME`] samples.
    WrongFrameLength,
    /// Refined `ω̂_0` is outside the quantizable range
    /// `[2π/123.125, 2π/19.875]`. Per §0.4.5 the encoder should
    /// dispatch to silence; this error is surfaced when the caller
    /// cannot otherwise recover.
    PitchOutOfRange,
    /// Reserved for sections of the addendum that are not yet
    /// implemented.
    SpecGap,
}

/// Output of [`encode`] — a single 20 ms analysis frame.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisOutput {
    /// Silence frame. During preroll (first two input frames after a
    /// cold start or state reset) the encoder cannot produce a valid
    /// `P̂_I` yet; caller should emit the codec's silence bitstream.
    /// Also emitted if §0.8.4 silence detection (not yet implemented)
    /// decides the current frame is silent.
    Silence,
    /// Voice frame with reconstructed MBE parameters. The wire-side
    /// encoder ([`crate::p25_fullrate::dequantize::quantize`] or
    /// [`crate::p25_halfrate::dequantize::quantize`]) consumes this
    /// to produce the transmitted bitstream.
    Voice(MbeParams),
}

/// Diagnostic trace from [`encode_with_trace`] capturing the §0.3
/// pitch-tracker scalars for the current frame. `None` when pitch
/// search did not run (preroll).
///
/// All fields are in the encoder's native units: pitch periods in
/// samples, `E(P)` as the unitless Eq. 5 error. Carries the full
/// [`PitchSearch`] for the current frame so callers can query `E(P)`
/// at arbitrary pitches (e.g., the chip's reference pitch) without
/// re-running the HPF + LPF + autocorrelation pipeline.
#[derive(Clone, Debug)]
pub struct EncodeTrace {
    /// `P̂_B` from [`look_back`].
    pub p_hat_b: f64,
    /// `CE_B` — 3-frame cumulative error at the look-back pick.
    pub ce_b: f64,
    /// `P̂_F` from [`look_ahead`] (post-cascade).
    pub p_hat_f: f64,
    /// `CE_F` — 3-frame cumulative error at the look-ahead pick.
    pub ce_f: f64,
    /// `P̂_I` — [`decide_initial_pitch`] pick.
    pub p_hat_i: f64,
    /// `E(P̂_I)` — current-frame Eq. 5 error at the tracker's pick.
    pub e_p_hat_i: f64,
    /// Current frame's [`PitchSearch`] — callable via
    /// [`PitchSearch::e_of_p`] to evaluate Eq. 5 at any pitch on the
    /// same analyzed PCM window.
    pub pitch_search_current: PitchSearch,
    /// V/UV determination result (per-band D_k, Θ_ξ, decisions).
    /// `None` during preroll or silence-dispatched frames.
    pub vuv_result: Option<VuvResult>,
}

/// Shared [`HarmonicBasis`] — the Eq. 30 `W_R(k)` table is signal-
/// independent, so it is built once at first use and reused across
/// all `encode` calls.
fn shared_basis() -> &'static HarmonicBasis {
    use std::sync::OnceLock;
    static BASIS: OnceLock<HarmonicBasis> = OnceLock::new();
    BASIS.get_or_init(HarmonicBasis::new)
}

/// Run the closed-loop matched-decoder roundtrip per §0.6.6.
///
/// Takes the analysis-side outputs (`M̂_l, ω̂_0, L̂, v̂_k`) and the
/// current `PredictorState` (carrying `M̃_l(−1)`), calls the full-
/// rate wire quantizer plus the matched decoder, and returns the
/// reconstructed `M̃_l(0)` as 1-indexed f64. The caller commits the
/// result to `PredictorState` via [`PredictorState::commit`].
fn matched_decoder_roundtrip(
    m_hat: &SpectralAmplitudes,
    refinement: &PitchRefinement,
    vuv: &VuvResult,
    predictor: &PredictorState,
) -> Result<[f64; L_HAT_MAX as usize + 1], AnalysisError> {
    use crate::p25_fullrate::dequantize::{
        quantize as fullrate_quantize, reconstruct_amplitudes_from_bits, DecoderState,
    };
    let l_hat = refinement.l_hat;
    let omega_0_f32 = refinement.omega_hat as f32;

    // Build MbeParams from the analysis output. Downcast to f32 at
    // the wire boundary; precision at this stage is set by the
    // quantizer, not by f64 accumulation upstream.
    let mut amplitudes = Vec::with_capacity(l_hat as usize);
    let mut voiced = Vec::with_capacity(l_hat as usize);
    for l in 1..=u32::from(l_hat) {
        amplitudes.push(m_hat[l as usize] as f32);
        let k = band_for_harmonic(l, vuv.k_hat);
        let v = k >= 1 && vuv.vuv[k as usize] == 1;
        voiced.push(v);
    }
    let params = MbeParams::new(omega_0_f32, l_hat, &voiced, &amplitudes)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Seed a DecoderState from the analysis predictor. Both the
    // forward prediction (inside `fullrate_quantize`) and the
    // inverse reconstruction must see the same `M̃_l(−1)` / `L̃(−1)`,
    // so we snapshot before calling quantize (which mutates its
    // state argument open-loop).
    let prev_slice = predictor.m_tilde_prev_slice();
    let ds_snapshot = DecoderState::from_amplitudes(&prev_slice, predictor.l_tilde_prev());

    let mut ds_for_quantize = ds_snapshot.clone();
    let bits = fullrate_quantize(&params, &mut ds_for_quantize)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Reconstruct `M̃_l(0)` using the unmutated snapshot — matches
    // what a compliant receiver computes from the same bits.
    let reconstructed = reconstruct_amplitudes_from_bits(&bits, l_hat, &ds_snapshot);

    let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
    m_tilde[0] = 1.0;
    for i in 0..l_hat as usize {
        m_tilde[i + 1] = f64::from(reconstructed[i]);
    }
    Ok(m_tilde)
}

/// Lookahead buffer length — three frames of post-HPF PCM, enough
/// to center-window the current analyze frame plus `future_1` and
/// `future_2` per §0.3.5. Samples outside the buffer (past-frame
/// edge, beyond-future edge) are zero-extended per §0.1.3.
pub const LOOKAHEAD_LEN: usize = 3 * SAMPLES_PER_FRAME as usize;

/// Consolidated cross-frame state for the analysis encoder per
/// addendum §0.9.1 / §0.9.2.
#[derive(Clone, Debug)]
pub struct AnalysisState {
    /// HPF filter state (§0.1.2, Eq. 3).
    hpf: HpfState,
    /// Rolling post-HPF PCM buffer. Ordered oldest-first: samples
    /// `[0, 159]` are the current analyzable frame; `[160, 319]`
    /// `future_1`; `[320, 479]` `future_2`. During preroll only the
    /// leftmost `lookahead_fill × 160` samples are meaningful; the
    /// remainder is zero.
    lookahead: [f64; LOOKAHEAD_LEN],
    /// Number of 160-sample frames already placed in `lookahead`
    /// during the preroll fill phase (`0..=3`). At steady state it
    /// stays at `3`.
    lookahead_fill: u8,
    /// Pitch tracker history (§0.3.4: `P̂_{−1}, P̂_{−2}, E_{−1}, E_{−2}`).
    pitch_history: LookBackContext,
    /// V/UV threshold + per-band history (§0.7.8).
    vuv: VuvState,
    /// Matched-decoder predictor state (§0.6.7).
    predictor: PredictorState,
    /// Preroll counter per §0.9.4: saturates at `PREROLL_FRAMES`.
    /// Below that, the encoder is still filling its two-frame
    /// look-ahead and emits silence instead of voice.
    preroll: u8,
    /// §0.8.4 silence detector. Always updated per frame; only
    /// gates `encode`'s output when `silence_detection_enabled`
    /// is `true`.
    silence: SilenceDetector,
    /// Opt-in gate for §0.8.4 silence dispatch. Defaults to `false`
    /// (addendum §0.8.8: pass-through recommended for initial
    /// implementation; explicit silence dispatch only when
    /// DVSI-compatibility testing demands bit-exact silence frames).
    silence_detection_enabled: bool,
}

/// Preroll frame count at which the analysis encoder has enough
/// look-ahead state to produce a valid `P̂_I` (§0.9.4).
pub const PREROLL_FRAMES: u8 = 2;

impl AnalysisState {
    /// Cold-start encoder state per §0.9.2 `analysis_encoder_state_init`.
    pub const fn new() -> Self {
        Self {
            hpf: HpfState::new(),
            lookahead: [0.0; LOOKAHEAD_LEN],
            lookahead_fill: 0,
            pitch_history: LookBackContext::cold_start(),
            vuv: VuvState::cold_start(),
            predictor: PredictorState::cold_start(),
            preroll: 0,
            silence: SilenceDetector::cold_start(),
            silence_detection_enabled: false,
        }
    }

    /// Enable or disable §0.8.4 silence dispatch. When enabled,
    /// `encode` returns `AnalysisOutput::Silence` for frames the
    /// detector flags, and skips the predictor-state commit per
    /// §13.3 prose. Default: disabled (pass-through per §0.8.8
    /// recommendation).
    pub fn set_silence_detection(&mut self, enabled: bool) {
        self.silence_detection_enabled = enabled;
    }

    /// Whether §0.8.4 silence dispatch is currently enabled.
    #[inline]
    pub fn silence_detection_enabled(&self) -> bool {
        self.silence_detection_enabled
    }

    /// Read-only access to the silence detector state (for inspection
    /// and DVSI-calibration tooling).
    #[inline]
    pub fn silence_detector(&self) -> &SilenceDetector {
        &self.silence
    }

    /// Mutable access to the V/UV state for conformance diagnostics
    /// (e.g., injecting chip-side V/UV history to measure feedback
    /// divergence).
    #[inline]
    pub fn vuv_state_mut(&mut self) -> &mut VuvState {
        &mut self.vuv
    }

    /// Shift the lookahead buffer left by `SAMPLES_PER_FRAME` and
    /// append `new_frame` (post-HPF) at the right. During preroll,
    /// places the frame at slot `lookahead_fill` instead of shifting.
    fn ingest_frame(&mut self, new_frame: &[f64; SAMPLES_PER_FRAME as usize]) {
        let frame = SAMPLES_PER_FRAME as usize;
        if (self.lookahead_fill as usize) < LOOKAHEAD_LEN / frame {
            let start = (self.lookahead_fill as usize) * frame;
            self.lookahead[start..start + frame].copy_from_slice(new_frame);
            self.lookahead_fill += 1;
        } else {
            self.lookahead.copy_within(frame.., 0);
            let start = LOOKAHEAD_LEN - frame;
            self.lookahead[start..].copy_from_slice(new_frame);
        }
    }

    /// Extract a 321-sample window centered on the specified frame
    /// slot (`0 = current`, `1 = future_1`, `2 = future_2`) for
    /// §0.3 `PitchSearch` construction. Positions outside
    /// `[0, LOOKAHEAD_LEN)` are zero-extended per §0.1.3.
    fn extract_pitch_window(&self, frame_slot: usize) -> [f64; PITCH_INPUT_LEN] {
        let center = frame_slot * SAMPLES_PER_FRAME as usize
            + (SAMPLES_PER_FRAME as usize) / 2;
        let mut out = [0.0f64; PITCH_INPUT_LEN];
        for i in 0..PITCH_INPUT_LEN {
            let offset = i as i32 - PITCH_INPUT_HALF;
            let buf_idx = center as i32 + offset;
            if (0..LOOKAHEAD_LEN as i32).contains(&buf_idx) {
                out[i] = self.lookahead[buf_idx as usize];
            }
        }
        out
    }

    /// Extract a 221-sample window centered on the current analyzable
    /// frame for §0.2 `signal_spectrum`.
    fn extract_refinement_window(&self) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let center = (SAMPLES_PER_FRAME as usize) / 2;
        let mut out = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let len = (2 * W_R_HALF + 1) as usize;
        for i in 0..len {
            let offset = i as i32 - W_R_HALF;
            let buf_idx = center as i32 + offset;
            if (0..LOOKAHEAD_LEN as i32).contains(&buf_idx) {
                out[i] = self.lookahead[buf_idx as usize];
            }
        }
        out
    }

    /// `true` while the encoder is still filling its two-frame
    /// look-ahead and cannot produce a valid voice frame. During
    /// preroll the caller should emit silence (§0.9.4, §0.8).
    #[inline]
    pub fn in_preroll(&self) -> bool {
        self.preroll < PREROLL_FRAMES
    }

    /// Current preroll counter (`0..=PREROLL_FRAMES`).
    #[inline]
    pub fn preroll_counter(&self) -> u8 {
        self.preroll
    }

    /// Advance the preroll counter, saturating at `PREROLL_FRAMES`.
    pub fn advance_preroll(&mut self) {
        if self.preroll < PREROLL_FRAMES {
            self.preroll += 1;
        }
    }

    /// Mutable access to the HPF state (used by the §0.10 pipeline
    /// to filter each incoming 160-sample PCM frame before framing).
    #[inline]
    pub fn hpf_mut(&mut self) -> &mut HpfState {
        &mut self.hpf
    }

    /// Read-only access to the pitch tracker history.
    #[inline]
    pub fn pitch_history(&self) -> LookBackContext {
        self.pitch_history
    }

    /// Shift in a newly-computed `P̂_I` and its `E(P̂_I)` value:
    /// `P̂_{−2} ← P̂_{−1}`, `P̂_{−1} ← P̂_I`, same for the energy
    /// scalars. Per §0.9.3 step 1.
    pub fn commit_pitch(&mut self, p_hat_i: f64, e_p_hat_i: f64) {
        self.pitch_history.prev_err_2 = self.pitch_history.prev_err_1;
        self.pitch_history.prev_err_1 = e_p_hat_i;
        self.pitch_history.prev_pitch = p_hat_i;
    }

    /// Mutable V/UV state — `determine_vuv` commits it internally.
    #[inline]
    pub fn vuv_mut(&mut self) -> &mut VuvState {
        &mut self.vuv
    }

    /// Read-only V/UV state.
    #[inline]
    pub fn vuv(&self) -> &VuvState {
        &self.vuv
    }

    /// Mutable predictor state — caller is the §0.6.6 matched-decoder
    /// feedback loop in §0.10, which calls `commit` with reconstructed
    /// `M̃_l(0)` after the full-rate quantize/dequantize round-trip.
    #[inline]
    pub fn predictor_mut(&mut self) -> &mut PredictorState {
        &mut self.predictor
    }

    /// Read-only predictor state.
    #[inline]
    pub fn predictor(&self) -> &PredictorState {
        &self.predictor
    }

    /// Freeze cross-frame state — used when the frame is dispatched
    /// as silence / tone / erasure per §0.8: state is preserved so
    /// that subsequent voice frames resume from the last good state.
    /// This matches the decoder-side §1.9 "preserve through frame
    /// mute" behavior for the predictor, and the analysis-side
    /// symmetric case described in §0.10.
    pub fn freeze(&self) {
        // No-op marker: state is preserved by simply not calling
        // the commit_* / advance_preroll methods. This method
        // exists to document the contract in §0.10's dispatch path.
    }
}

impl Default for AnalysisState {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze a 20 ms (160-sample) frame of 8 kHz 16-bit PCM and produce
/// one analysis output per addendum §0.10.
///
/// Per-frame execution order (§0.10.1):
///
///  1. HPF input frame (§0.1, Eq. 3).
///  2. Ingest into the 480-sample rolling lookahead buffer.
///  3. Preroll check (§0.9.4): first two frames emit silence.
///  4. Compute `s_LPF`, `r(t)`, `E(P)` for the current + 2 future
///     frames via [`PitchSearch::new`] (§0.3.1–§0.3.3).
///  5. Pitch tracking → `P̂_I`, `E(P̂_I)` (§0.3.4–§0.3.6).
///  6. 256-pt DFT → `S_w(m)` (§0.2.1).
///  7. Pitch refinement → `ω̂_0, L̂, â_l, b̂_l, b̂_0` (§0.4).
///  8. V/UV determination → `K̂, v̂_k` (§0.7).
///  9. Spectral amplitude estimation → `M̂_l` (§0.5).
/// 10. Closed-loop matched-decoder roundtrip → `M̃_l(0)` (§0.6.6).
/// 11. Commit pitch history and predictor state (§0.9.3).
/// 12. Emit [`AnalysisOutput::Voice`].
///
/// §0.8 frame-type dispatch (silence detection, tone pass-through)
/// is not yet wired — all non-preroll frames emit voice.
pub fn encode(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<AnalysisOutput, AnalysisError> {
    encode_with_trace(pcm, state).map(|(out, _)| out)
}

/// Identical to [`encode`] but additionally returns an [`EncodeTrace`]
/// capturing the §0.3 pitch-tracker scalars for the analyzed frame.
/// The trace is `None` during preroll (pitch search does not run).
///
/// Intended for conformance diagnostics that need to evaluate `E(P)`
/// at arbitrary pitches (e.g., a reference chip's pitch) on the same
/// PCM window the encoder analyzed — see
/// [`PitchSearch::e_of_p`].
pub fn encode_with_trace(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<(AnalysisOutput, Option<EncodeTrace>), AnalysisError> {
    if pcm.len() != SAMPLES_PER_FRAME as usize {
        return Err(AnalysisError::WrongFrameLength);
    }

    // Step 1: HPF the incoming frame.
    let mut hpf_out = [0.0f64; SAMPLES_PER_FRAME as usize];
    state.hpf.run_pcm(pcm, &mut hpf_out);

    // §0.8.4: update the silence detector every frame with the
    // post-HPF frame energy (Σ s²(n)). The update runs whether or
    // not silence dispatch is enabled — the η tracker needs a
    // continuous signal to be useful.
    let frame_energy: f64 = hpf_out.iter().map(|s| s * s).sum();
    let silent = state.silence.update(frame_energy);

    // Step 2: ingest into the lookahead buffer.
    state.ingest_frame(&hpf_out);

    // Step 3: preroll check.
    if state.in_preroll() {
        state.advance_preroll();
        return Ok((AnalysisOutput::Silence, None));
    }

    // Steps 4-5: pitch tracking over the three-frame PitchSearch window.
    let cur_win = state.extract_pitch_window(0);
    let f1_win = state.extract_pitch_window(1);
    let f2_win = state.extract_pitch_window(2);
    let search_cur = PitchSearch::new(&cur_win);
    let search_f1 = PitchSearch::new(&f1_win);
    let search_f2 = PitchSearch::new(&f2_win);
    let (p_b, ce_b) = look_back(&search_cur, state.pitch_history);
    let (p_f, ce_f) = look_ahead(&search_cur, &search_f1, &search_f2);
    let p_hat_i = decide_initial_pitch(p_b, ce_b, p_f, ce_f);
    let e_p_hat_i = search_cur.e_of_p(p_hat_i);
    let mut trace = EncodeTrace {
        p_hat_b: p_b,
        ce_b,
        p_hat_f: p_f,
        ce_f,
        p_hat_i,
        e_p_hat_i,
        pitch_search_current: search_cur.clone(),
        vuv_result: None,
    };

    // Step 6: signal DFT.
    let basis = shared_basis();
    let sig_win = state.extract_refinement_window();
    let sw = signal_spectrum(&sig_win);

    // Step 7: pitch refinement.
    let refinement = refine_pitch(&sw, basis, p_hat_i);

    // Step 8: V/UV (must precede §0.5 per Figure 5).
    let vuv_result = determine_vuv(&sw, basis, &refinement, e_p_hat_i, &mut state.vuv);
    trace.vuv_result = Some(vuv_result.clone());

    // Step 9: spectral amplitude estimation.
    let m_hat = estimate_spectral_amplitudes(&sw, basis, &refinement, &vuv_result);

    // §0.8 silence gate. Per §13.3 prose: silence / tone / erasure
    // frames do NOT update the matched-decoder predictor state, but
    // pitch and V/UV history ARE always tracked (V/UV was already
    // committed inside `determine_vuv` above; pitch history is
    // committed unconditionally below).
    if state.silence_detection_enabled && silent {
        state.commit_pitch(p_hat_i, e_p_hat_i);
        state.advance_preroll();
        return Ok((AnalysisOutput::Silence, Some(trace)));
    }

    // Step 10: matched-decoder roundtrip (closed-loop §0.6.6).
    let m_tilde =
        matched_decoder_roundtrip(&m_hat, &refinement, &vuv_result, &state.predictor)?;

    // Step 11: commit state for the next frame.
    state.predictor.commit(&m_tilde, refinement.l_hat);
    state.commit_pitch(p_hat_i, e_p_hat_i);

    // Step 12: emit voice.
    let mut amplitudes = Vec::with_capacity(refinement.l_hat as usize);
    let mut voiced = Vec::with_capacity(refinement.l_hat as usize);
    for l in 1..=u32::from(refinement.l_hat) {
        amplitudes.push(m_hat[l as usize] as f32);
        let k = band_for_harmonic(l, vuv_result.k_hat);
        voiced.push(k >= 1 && vuv_result.vuv[k as usize] == 1);
    }
    let params = MbeParams::new(
        refinement.omega_hat as f32,
        refinement.l_hat,
        &voiced,
        &amplitudes,
    )
    .map_err(|_| AnalysisError::PitchOutOfRange)?;
    state.advance_preroll();
    Ok((AnalysisOutput::Voice(params), Some(trace)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_window_symmetry_and_endpoints() {
        assert_eq!(ANALYSIS_WINDOW_LEN, 301);
        // Outside support is zero.
        assert_eq!(analysis_window(-151), 0.0);
        assert_eq!(analysis_window(151), 0.0);
        // Even symmetry.
        for n in 1..=150 {
            assert!(
                (analysis_window(-n) - analysis_window(n)).abs() < 1e-5,
                "wI(-{n}) != wI({n})"
            );
        }
    }

    #[test]
    fn refinement_window_peak_and_symmetry() {
        assert_eq!(REFINEMENT_WINDOW_LEN, 221);
        assert!((refinement_window(0) - 1.0).abs() < 1e-6);
        assert_eq!(refinement_window(-111), 0.0);
        assert_eq!(refinement_window(111), 0.0);
        for n in 1..=110 {
            assert!(
                (refinement_window(-n) - refinement_window(n)).abs() < 1e-5,
                "wR(-{n}) != wR({n})"
            );
        }
    }

    #[test]
    fn lpf_symmetry_and_support() {
        assert_eq!(ANNEX_D_LPF_LEN, 21);
        for n in 1..=10 {
            assert!((lpf_tap(-n) - lpf_tap(n)).abs() < 1e-6, "h_LPF(-{n}) != h_LPF({n})");
        }
        assert_eq!(lpf_tap(-11), 0.0);
        assert_eq!(lpf_tap(11), 0.0);
    }

    #[test]
    fn encode_rejects_wrong_length() {
        let mut s = AnalysisState::new();
        assert_eq!(
            encode(&[0i16; 159], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
        assert_eq!(
            encode(&[0i16; 161], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
    }

    #[test]
    fn encode_emits_silence_during_preroll() {
        let mut s = AnalysisState::new();
        let pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // First two frames always return silence regardless of input.
        assert_eq!(encode(&pcm, &mut s), Ok(AnalysisOutput::Silence));
        assert_eq!(encode(&pcm, &mut s), Ok(AnalysisOutput::Silence));
    }

    /// Generate a broadband harmonic PCM frame at a given period
    /// (similar to `broadband_periodic` above but producing 160-sample
    /// `i16` slices for the end-to-end encode path).
    fn periodic_pcm_frame(period: f64, max_h: u32, phase_offset: f64) -> [i16; SAMPLES_PER_FRAME as usize] {
        let mut out = [0i16; SAMPLES_PER_FRAME as usize];
        let omega = 2.0 * core::f64::consts::PI / period;
        for (idx, slot) in out.iter_mut().enumerate() {
            let nf = idx as f64 + phase_offset;
            let mut s = 0.0;
            for h in 1..=max_h {
                s += (f64::from(h) * omega * nf).cos() / f64::from(h);
            }
            *slot = (s * 8000.0).clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
        out
    }

    /// Third frame and onward produce `Voice` output on a periodic
    /// input. Validates the preroll boundary and that the entire
    /// §0.3→§0.4→§0.5→§0.6→§0.7 + matched-decoder roundtrip
    /// pipeline returns well-formed output. Not a claim about which
    /// pitch is picked — the Eq. 5 large-P bias combined with the
    /// cold-start look-back window `[0.8·100, 1.2·100]` can land on
    /// any P in `[80, 120]` for a period-50 input, and refinement
    /// offsets ±1.125 on top of that. Full-accuracy pitch recovery
    /// validation belongs with DVSI PCM vectors.
    #[test]
    fn encode_third_frame_emits_voice_on_periodic_input() {
        let mut s = AnalysisState::new();
        let p0 = periodic_pcm_frame(50.0, 10, 0.0);
        let p1 = periodic_pcm_frame(50.0, 10, 160.0);
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        assert_eq!(encode(&p0, &mut s), Ok(AnalysisOutput::Silence));
        assert_eq!(encode(&p1, &mut s), Ok(AnalysisOutput::Silence));
        match encode(&p2, &mut s).unwrap() {
            AnalysisOutput::Voice(params) => {
                // Well-formed: ω_0 in admissible range.
                let omega = params.omega_0();
                assert!(omega.is_finite() && omega > 0.0);
                let p = 2.0 * core::f64::consts::PI as f32 / omega;
                assert!(
                    (19.875..=123.125).contains(&p),
                    "P = {p} out of admissible range"
                );
                // L̂ in [9, 56].
                let l = params.harmonic_count();
                assert!((L_HAT_MIN..=L_HAT_MAX).contains(&l));
                // At least one harmonic has nonzero amplitude (signal
                // is not silence).
                let any_nonzero = (0..l as usize)
                    .any(|i| params.amplitudes_slice()[i].abs() > 1e-6);
                assert!(any_nonzero, "all amplitudes zero on periodic input");
            }
            other => panic!("expected Voice, got {other:?}"),
        }
    }

    /// The matched-decoder roundtrip actually evolves predictor state
    /// frame-to-frame. After one real-frame emission, `L̃(−1)` no
    /// longer equals the cold-start value (30) — unless the
    /// refinement happened to produce L̂=30 exactly.
    #[test]
    fn encode_updates_predictor_state_after_first_real_frame() {
        let mut s = AnalysisState::new();
        let p0 = periodic_pcm_frame(50.0, 10, 0.0);
        let p1 = periodic_pcm_frame(50.0, 10, 160.0);
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        let _ = encode(&p0, &mut s);
        let _ = encode(&p1, &mut s);
        let cold_m_tilde_0: f64 = 1.0; // Eq. 56.
        let cold_m_tilde_1: f64 = s.predictor().read(1);
        assert_eq!(cold_m_tilde_1, 1.0, "pre-voice state should be cold");
        let _ = encode(&p2, &mut s).unwrap();
        // After a real-frame emission, at least one M̃_l should have
        // moved off 1.0 (unless the signal happens to reconstruct to
        // unity amplitude, which is extremely unlikely for broadband
        // periodic content).
        let any_moved = (1..=s.predictor().l_tilde_prev())
            .any(|l| (s.predictor().read(u32::from(l)) - 1.0).abs() > 1e-3);
        assert!(any_moved, "predictor state did not evolve after voice emission");
        assert_eq!(s.predictor().read(0), cold_m_tilde_0); // Eq. 56 invariant.
    }

    /// Silent input across multiple frames keeps the preroll gate
    /// working correctly: exactly 2 silence frames before the
    /// algorithm starts running. After preroll, silent PCM produces
    /// some output — possibly `Voice` with degenerate amplitudes or
    /// `Err` (pitch out of range); both are acceptable.
    #[test]
    fn encode_preroll_is_exactly_two_frames() {
        let mut s = AnalysisState::new();
        let pcm = [0i16; SAMPLES_PER_FRAME as usize];
        for i in 0..PREROLL_FRAMES {
            assert_eq!(
                encode(&pcm, &mut s),
                Ok(AnalysisOutput::Silence),
                "frame {i} should be silence during preroll"
            );
        }
        // Frame PREROLL_FRAMES is the first non-preroll frame.
        // Zero-PCM input may legitimately produce voice with
        // degenerate amplitudes, or fail with PitchOutOfRange; we
        // only assert that the preroll silence stream stopped.
        let result = encode(&pcm, &mut s);
        // Acceptable outcomes: any Voice, any Err, or silence-after-
        // preroll (if we later add §0.8.4 silence detection).
        match result {
            Ok(AnalysisOutput::Voice(_)) | Ok(AnalysisOutput::Silence) | Err(_) => {}
        }
    }

    // -----------------------------------------------------------------
    // §0.8 — silence detection
    // -----------------------------------------------------------------

    #[test]
    fn silence_detector_cold_start_is_not_silent() {
        let d = SilenceDetector::cold_start();
        assert!(!d.is_silent());
        assert_eq!(d.noise_floor(), SILENCE_ETA_INIT);
    }

    /// Sustained high energy exits silence after 3 consecutive voice
    /// votes (cold start begins not-in-silence, so first we need to
    /// enter silence to test the exit).
    #[test]
    fn silence_detector_exits_silence_after_voice_hysteresis() {
        let mut d = SilenceDetector::cold_start();
        // Force into silence by feeding many low-energy frames.
        for _ in 0..20 {
            d.update(0.0);
        }
        assert!(d.is_silent(), "should have entered silence");
        // Feed loud frames; should exit after SILENCE_EXIT_FRAMES.
        d.update(1e12);
        d.update(1e12);
        assert!(d.is_silent(), "should still be silent after 2 voice frames");
        d.update(1e12);
        assert!(
            !d.is_silent(),
            "should have exited silence after {SILENCE_EXIT_FRAMES} voice frames"
        );
    }

    /// Sustained low energy enters silence after 5 consecutive silent
    /// votes. First need to establish a nonzero noise floor.
    #[test]
    fn silence_detector_enters_silence_after_hysteresis() {
        let mut d = SilenceDetector::cold_start();
        // Bring η up to a high value first by feeding loud frames,
        // so subsequent 0-energy frames cleanly vote silent.
        for _ in 0..30 {
            d.update(1e8);
        }
        assert!(!d.is_silent(), "loud frames should not be silent");
        // Now feed silent frames.
        for i in 0..SILENCE_ENTER_FRAMES {
            assert!(!d.is_silent(), "should not be silent before {i} silent frames");
            d.update(0.0);
        }
        assert!(d.is_silent(), "should have entered silence after enter-hysteresis");
    }

    /// When silence detection is disabled (default), the detector's
    /// decision does not gate `encode`'s output.
    #[test]
    fn encode_with_silence_detection_disabled_never_emits_non_preroll_silence() {
        let mut s = AnalysisState::new();
        assert!(!s.silence_detection_enabled());
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Burn preroll.
        let _ = encode(&zero_pcm, &mut s);
        let _ = encode(&zero_pcm, &mut s);
        // Many subsequent silent frames — with detection disabled,
        // emit voice (possibly with degenerate amplitudes) or fail,
        // but NOT Silence.
        let mut silence_count = 0u32;
        let mut other_count = 0u32;
        for _ in 0..20 {
            match encode(&zero_pcm, &mut s) {
                Ok(AnalysisOutput::Silence) => silence_count += 1,
                _ => other_count += 1,
            }
        }
        assert_eq!(silence_count, 0);
        assert_eq!(other_count, 20);
    }

    /// With silence detection enabled and sustained silent input,
    /// the encoder eventually emits `AnalysisOutput::Silence`.
    #[test]
    fn encode_with_silence_detection_enabled_gates_silent_input() {
        let mut s = AnalysisState::new();
        s.set_silence_detection(true);
        assert!(s.silence_detection_enabled());
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Feed many silent frames; detector eventually enters silence.
        let mut saw_silence_after_preroll = false;
        for i in 0..30 {
            let out = encode(&zero_pcm, &mut s);
            if i >= PREROLL_FRAMES as usize && matches!(out, Ok(AnalysisOutput::Silence)) {
                saw_silence_after_preroll = true;
                break;
            }
        }
        assert!(
            saw_silence_after_preroll,
            "silence detector should trigger on sustained zero input"
        );
    }

    /// §13.3 invariant: when `encode` emits `Silence` post-preroll,
    /// the predictor state `M̃_l(−1)` / `L̃(−1)` does NOT advance.
    #[test]
    fn encode_silence_does_not_update_predictor_state() {
        let mut s = AnalysisState::new();
        s.set_silence_detection(true);
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Drive into silence.
        for _ in 0..30 {
            let _ = encode(&zero_pcm, &mut s);
        }
        // Snapshot predictor state.
        let l_before = s.predictor().l_tilde_prev();
        let m1_before = s.predictor().read(1);
        // More silent frames.
        for _ in 0..5 {
            match encode(&zero_pcm, &mut s) {
                Ok(AnalysisOutput::Silence) => {}
                Ok(AnalysisOutput::Voice(_)) => {
                    // Detector may have exited; stop the test early.
                    return;
                }
                Err(_) => return,
            }
        }
        let l_after = s.predictor().l_tilde_prev();
        let m1_after = s.predictor().read(1);
        assert_eq!(l_before, l_after, "L̃(−1) changed across silent frames");
        assert_eq!(m1_before, m1_after, "M̃_1(−1) changed across silent frames");
    }

    #[test]
    fn encode_rejects_wrong_length_in_preroll() {
        let mut s = AnalysisState::new();
        assert_eq!(
            encode(&[0i16; 159], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
        assert_eq!(
            encode(&[0i16; 161], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
    }

    // -----------------------------------------------------------------
    // §0.2 — signal DFT and harmonic basis
    // -----------------------------------------------------------------

    use core::f64::consts::PI;

    /// For `ω₀ = 2π/50` the bin endpoints in addendum §0.2.4 should
    /// match the numerical example exactly.
    #[test]
    fn bin_endpoints_match_addendum_numerical_example() {
        let omega0 = 2.0 * PI / 50.0;
        assert_eq!(HarmonicBasis::bin_endpoints(0, omega0), (-2, 3));
        assert_eq!(HarmonicBasis::bin_endpoints(1, omega0), (3, 8));
        assert_eq!(HarmonicBasis::bin_endpoints(2, omega0), (8, 13));
    }

    /// Per Eq. 27 / Eq. 26, adjacent harmonics share a bin endpoint
    /// (`b_l = a_{l+1}` exactly), so the supports are contiguous and
    /// non-overlapping in the integer grid.
    #[test]
    fn bin_endpoints_are_contiguous_across_harmonics() {
        for omega_denom in [20i32, 50, 100, 123] {
            let omega0 = 2.0 * PI / f64::from(omega_denom);
            for l in 0..10 {
                let (_, b_this) = HarmonicBasis::bin_endpoints(l, omega0);
                let (a_next, _) = HarmonicBasis::bin_endpoints(l + 1, omega0);
                assert_eq!(
                    b_this, a_next,
                    "ω₀ = 2π/{omega_denom}, harmonic {l}: b_l = {b_this}, a_(l+1) = {a_next}"
                );
            }
        }
    }

    /// `W_R(0) = Σ w_R(n)` falls out of Eq. 30 directly (cosine with
    /// `k = 0` is identically 1). Sanity check the precomputed table.
    #[test]
    fn w_r_zero_equals_window_sum() {
        let basis = HarmonicBasis::new();
        let expected: f64 = (-W_R_HALF..=W_R_HALF)
            .map(|n| f64::from(refinement_window(n)))
            .sum();
        assert!(
            (basis.w_r(0) - expected).abs() < 1e-9,
            "W_R(0) = {}, expected {}",
            basis.w_r(0),
            expected
        );
    }

    /// The window is real and symmetric so `W_R(k) = W_R(−k)`.
    #[test]
    fn w_r_is_symmetric_in_k() {
        let basis = HarmonicBasis::new();
        for k in [1, 7, 64, 100, 500, 1023, 8000] {
            assert_eq!(basis.w_r(k), basis.w_r(-k), "W_R({k}) != W_R(−{k})");
        }
    }

    /// Outside the stored range the accessor clamps to zero.
    #[test]
    fn w_r_returns_zero_out_of_range() {
        let basis = HarmonicBasis::new();
        assert_eq!(basis.w_r(WINDOW_DFT_SIZE as i32), 0.0);
        assert_eq!(basis.w_r(-(WINDOW_DFT_SIZE as i32)), 0.0);
    }

    #[test]
    fn packed_index_roundtrips_two_sided_range() {
        for m in -127..=128 {
            let idx = packed_index(m);
            assert!(idx < DFT_SIZE, "packed_index({m}) out of bounds");
            assert_eq!(unpacked_m(idx), m, "roundtrip failed for m = {m}");
        }
    }

    /// DFT of the all-zero signal is all-zero.
    #[test]
    fn signal_spectrum_of_zero_is_zero() {
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        for (i, s) in sw.iter().enumerate() {
            assert!(s.norm_sqr() < 1e-18, "S_w[{i}] = ({}, {})", s.re, s.im);
        }
    }

    /// For the constant signal `s(n) = 1`, `S_w(0) = Σ w_R(n) = W_R(0)`.
    /// A cheap invariant that catches window-indexing mistakes.
    #[test]
    fn signal_spectrum_of_constant_matches_w_r_at_dc() {
        let signal = [1.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        let basis = HarmonicBasis::new();
        let s0 = sw[packed_index(0)];
        assert!(
            (s0.re - basis.w_r(0)).abs() < 1e-9 && s0.im.abs() < 1e-9,
            "S_w(0) = ({}, {}), expected ({}, 0)",
            s0.re,
            s0.im,
            basis.w_r(0)
        );
    }

    fn cosine_tone(omega0: f64) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega0 * f64::from(n)).cos();
        }
        signal
    }

    fn sine_tone(omega0: f64) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega0 * f64::from(n)).sin();
        }
        signal
    }

    /// For `s(n) = cos(ω₀·n)`, the Eq. 28 projection at the fundamental
    /// should recover `A_1(ω₀) ≈ 1/2 + 0j` — the `1/2` comes from
    /// Euler's identity (`cos = (e^{jω₀n} + e^{−jω₀n})/2`), and Eq. 28
    /// is a least-squares projection onto the window-shifted basis so
    /// the window-energy denominator normalizes the response to the
    /// cosine coefficient. Tolerance is loose enough to absorb the
    /// small image spillover from the `e^{−jω₀n}` term at `−ω₀`.
    #[test]
    fn harmonic_projection_recovers_cosine_amplitude() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = cosine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        assert!(
            (a1.re - 0.5).abs() < 0.01 && a1.im.abs() < 0.01,
            "A_1 = ({}, {}), expected ≈ (0.5, 0)",
            a1.re,
            a1.im
        );
    }

    /// For `s(n) = sin(ω₀·n)`, `A_1(ω₀) ≈ 0 − j/2`: the `1/(2j)` from
    /// Euler contributes a pure-imaginary projection.
    #[test]
    fn harmonic_projection_recovers_sine_amplitude() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = sine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        assert!(
            a1.re.abs() < 0.01 && (a1.im + 0.5).abs() < 0.01,
            "A_1 = ({}, {}), expected ≈ (0, −0.5)",
            a1.re,
            a1.im
        );
    }

    /// A 2nd-harmonic cosine projects onto `A_2(ω₀) ≈ 1/2` and onto
    /// `A_1(ω₀) ≈ 0` (no fundamental present). Exercises the `l·ω₀`
    /// offset in the `W_R` index.
    #[test]
    fn harmonic_projection_isolates_second_harmonic() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (2.0 * omega0 * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        let a2 = basis.harmonic_amplitude(&sw, 2, omega0);
        assert!(
            a1.re.abs() < 0.05 && a1.im.abs() < 0.05,
            "A_1 leak = ({}, {})",
            a1.re,
            a1.im
        );
        assert!(
            (a2.re - 0.5).abs() < 0.01 && a2.im.abs() < 0.01,
            "A_2 = ({}, {}), expected ≈ (0.5, 0)",
            a2.re,
            a2.im
        );
    }

    /// `synthetic_bin` reconstructs `A_l · W_R(k)` on each bin. Given
    /// `A_l(ω₀)` from Eq. 28, the reconstructed `S_w(m, ω₀)` must match
    /// the signal DFT `S_w(m)` closely on a clean sinusoid (the whole
    /// point of the projection).
    #[test]
    fn synthetic_bin_reconstructs_clean_sinusoid() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = cosine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(1, omega0);
        let mut residual_energy = 0.0;
        let mut signal_energy = 0.0;
        for m in m_lo..m_hi {
            let observed = sw[packed_index(m)];
            let synthetic = basis.synthetic_bin(m, 1, omega0, a1);
            let dre = observed.re - synthetic.re;
            let dim = observed.im - synthetic.im;
            residual_energy += dre * dre + dim * dim;
            signal_energy += observed.norm_sqr();
        }
        // Well under 1% relative residual on a pure cosine.
        assert!(
            residual_energy / signal_energy.max(1e-12) < 0.01,
            "residual/signal = {}, expected < 0.01",
            residual_energy / signal_energy.max(1e-12)
        );
    }

    /// Zero-denominator guard: when the harmonic's support is empty
    /// (degenerate `ω₀ = 0` collapses every `⌈a_l⌉ = ⌈b_l⌉ = 0`),
    /// `harmonic_amplitude` returns `Complex64::ZERO` instead of NaN.
    #[test]
    fn harmonic_amplitude_empty_support_returns_zero() {
        let basis = HarmonicBasis::new();
        let sw = [Complex64::new(1.0, 1.0); DFT_SIZE];
        let a0 = basis.harmonic_amplitude(&sw, 0, 0.0);
        assert_eq!(a0, Complex64::ZERO);
    }

    // -----------------------------------------------------------------
    // §0.3 — initial pitch estimation
    // -----------------------------------------------------------------

    #[test]
    fn snap_to_pitch_grid_rounds_to_half_sample() {
        assert_eq!(snap_to_pitch_grid(50.0), Some(50.0));
        assert_eq!(snap_to_pitch_grid(50.1), Some(50.0));
        assert_eq!(snap_to_pitch_grid(50.25), Some(50.5)); // round-half-to-even on doubles
        assert_eq!(snap_to_pitch_grid(50.3), Some(50.5));
        assert_eq!(snap_to_pitch_grid(50.7), Some(50.5));
        assert_eq!(snap_to_pitch_grid(50.8), Some(51.0));
    }

    #[test]
    fn snap_to_pitch_grid_rejects_out_of_range() {
        // 20.9 rounds up to 21.0 (borderline in-range).
        assert_eq!(snap_to_pitch_grid(20.9), Some(21.0));
        // 20.6 rounds to 20.5, out of range — rejected.
        assert_eq!(snap_to_pitch_grid(20.6), None);
        // Values well below the grid are rejected.
        assert_eq!(snap_to_pitch_grid(10.0), None);
        assert_eq!(snap_to_pitch_grid(200.0), None);
        // Exactly on the boundaries: in-range.
        assert_eq!(snap_to_pitch_grid(21.0), Some(21.0));
        assert_eq!(snap_to_pitch_grid(122.0), Some(122.0));
    }

    /// On a silent frame `E(P) = 1.0` per the Eq. 5 silence guard.
    #[test]
    fn e_of_p_is_one_on_silence() {
        let s = [0.0f64; PITCH_INPUT_LEN];
        let search = PitchSearch::new(&s);
        for p in [21.0, 50.0, 100.0, 122.0] {
            assert_eq!(search.e_of_p(p), 1.0, "P = {p}");
        }
    }

    /// `argmin_in_range` returns a grid value and restricts to the
    /// requested window (clamped to the full grid bounds).
    #[test]
    fn argmin_in_range_clamps_and_returns_grid_value() {
        let s = [0.1f64; PITCH_INPUT_LEN]; // non-silent, unstructured
        let search = PitchSearch::new(&s);
        let (p, _) = search.argmin_in_range(40.0, 60.0);
        assert!((40.0..=60.0).contains(&p), "P = {p} out of [40, 60]");
        // P is on the half-sample grid.
        let doubled = (p * 2.0).round();
        assert!((doubled / 2.0 - p).abs() < 1e-9, "P = {p} not on grid");
    }

    fn periodic_input(period: f64) -> [f64; PITCH_INPUT_LEN] {
        // Sum of a few harmonics so the *squared* envelope has
        // fundamental period equal to `period`, not period/2. A pure
        // cosine would have period/2 in its squared envelope.
        let mut s = [0.0f64; PITCH_INPUT_LEN];
        let omega = 2.0 * core::f64::consts::PI / period;
        for (idx, slot) in s.iter_mut().enumerate() {
            let n = idx as i32 - PITCH_INPUT_HALF;
            let nf = f64::from(n);
            *slot = (omega * nf).cos()
                + 0.5 * (2.0 * omega * nf).cos()
                + 0.25 * (3.0 * omega * nf).cos();
        }
        s
    }

    /// `argmin_in_range` actually finds the minimum `E(P)` in the
    /// requested window — confirmed by brute-force sweep. (This is a
    /// pure correctness check on the search loop, not a claim about
    /// which pitch "should" win for a given signal; that end-to-end
    /// property depends on real PCM against DVSI vectors, since
    /// perfectly periodic synthetic signals can't exercise the
    /// tracking + cascade semantics — every `E(nP)` is too negative
    /// for the Eq. 18/19/20 thresholds to prune spurious
    /// sub-multiples.)
    #[test]
    fn argmin_in_range_matches_brute_force_sweep() {
        let s = periodic_input(50.0);
        let search = PitchSearch::new(&s);
        for (p_lo, p_hi) in [(21.0, 122.0), (40.0, 60.0), (80.0, 100.0), (45.0, 55.0)] {
            let (p_hat, e_hat) = search.argmin_in_range(p_lo, p_hi);
            // Sweep the grid within [p_lo, p_hi] and collect the true minimum.
            let mut brute_min = f64::INFINITY;
            let mut brute_p = p_hat;
            let mut p = PITCH_GRID_MIN;
            while p <= PITCH_GRID_MAX {
                if p >= p_lo && p <= p_hi {
                    let e = search.e_of_p(p);
                    if e < brute_min {
                        brute_min = e;
                        brute_p = p;
                    }
                }
                p += PITCH_GRID_STEP;
            }
            assert!(
                (e_hat - brute_min).abs() < 1e-12 && (p_hat - brute_p).abs() < 1e-9,
                "[{p_lo}, {p_hi}]: argmin_in_range returned ({p_hat}, {e_hat}), \
                 brute = ({brute_p}, {brute_min})"
            );
        }
    }

    /// Diagnostic probe: dump E(P) across the grid for a synthetic
    /// periodic signal. Not a true test — just prints the curve so
    /// failures can be reasoned about. Gated behind nocapture.
    #[test]
    #[ignore]
    fn probe_dump_e_of_p() {
        for true_p in [50.0, 65.0, 80.0, 100.0] {
            let s = periodic_input(true_p);
            let search = PitchSearch::new(&s);
            let (p_hat, e_hat) = search.argmin_in_range(PITCH_GRID_MIN, PITCH_GRID_MAX);
            println!("\n=== true_p = {true_p} ===");
            println!("argmin: P̂ = {p_hat}, E = {e_hat:.4}");
            println!("energy = {:.4e}", search.energy);
            println!("w_i_fourth_moment = {:.4e}", search.w_i_fourth_moment);
            let mut p = PITCH_GRID_MIN;
            while p <= PITCH_GRID_MAX {
                let e = search.e_of_p(p);
                if p == true_p || p == p_hat || (p - PITCH_GRID_MIN) % 10.0 == 0.0 {
                    println!("  E({p:6.1}) = {e:+.4}");
                }
                p += PITCH_GRID_STEP;
            }
        }
    }

    /// `look_back` with cold-start context returns `CE_B = E(P̂_B)`
    /// because `E_{−1} = E_{−2} = 0` per §0.3.7.
    #[test]
    fn look_back_cold_start_has_zero_history_contribution() {
        let s = periodic_input(50.0);
        let search = PitchSearch::new(&s);
        let ctx = LookBackContext::cold_start();
        let (p_b, ce_b) = look_back(&search, ctx);
        // P̂_B falls in [0.8·100, 1.2·100] = [80, 120].
        assert!(
            (80.0..=120.0).contains(&p_b),
            "P̂_B = {p_b} out of [80, 120]"
        );
        // CE_B = E(P̂_B) + 0 + 0.
        let e_b = search.e_of_p(p_b);
        assert!((ce_b - e_b).abs() < 1e-12, "CE_B = {ce_b}, E(P̂_B) = {e_b}");
    }

    /// `look_back` with a small `CE_B` (Eq. 21 path) drives the
    /// decision toward `P̂_B` regardless of `P̂_F`.
    #[test]
    fn decide_initial_pitch_eq21_wins_when_ce_b_is_small() {
        assert_eq!(decide_initial_pitch(50.0, 0.3, 70.0, 0.2), 50.0);
        assert_eq!(decide_initial_pitch(50.0, 0.48, 70.0, 0.1), 50.0);
    }

    /// Eq. 22: when `CE_B > 0.48` but still ≤ `CE_F`, backward wins.
    #[test]
    fn decide_initial_pitch_eq22_wins_when_ce_b_leq_ce_f() {
        assert_eq!(decide_initial_pitch(50.0, 0.6, 70.0, 0.65), 50.0);
        assert_eq!(decide_initial_pitch(50.0, 0.6, 70.0, 0.6), 50.0);
    }

    /// Eq. 23: when `CE_B > CE_F` and `CE_B > 0.48`, forward wins.
    #[test]
    fn decide_initial_pitch_eq23_picks_forward_when_ce_b_is_larger() {
        assert_eq!(decide_initial_pitch(50.0, 0.7, 70.0, 0.5), 70.0);
    }

    /// `look_ahead` returns a pitch on the half-sample grid and
    /// some finite cumulative-error value. End-to-end correctness
    /// (does it pick the "right" pitch on real speech?) requires
    /// DVSI PCM vectors — see the comment on
    /// [`e_of_p_local_minimum_within_continuity_window`].
    #[test]
    fn look_ahead_returns_grid_value_and_finite_ce() {
        let s = periodic_input(50.0);
        let cur = PitchSearch::new(&s);
        let n1 = PitchSearch::new(&s);
        let n2 = PitchSearch::new(&s);
        let (p_f, ce_f) = look_ahead(&cur, &n1, &n2);
        assert!(
            (PITCH_GRID_MIN..=PITCH_GRID_MAX).contains(&p_f),
            "P̂_F = {p_f} out of grid"
        );
        assert!(ce_f.is_finite(), "CE_F = {ce_f} not finite");
        let doubled = (p_f / PITCH_GRID_STEP).round();
        assert!(
            (doubled * PITCH_GRID_STEP - p_f).abs() < 1e-9,
            "P̂_F = {p_f} not on grid"
        );
    }

    /// Boundary conditions: `compute_s_lpf` zero-extends the input.
    /// For an all-zero input it produces an all-zero output.
    #[test]
    fn compute_s_lpf_of_zero_is_zero() {
        let s = [0.0f64; PITCH_INPUT_LEN];
        let s_lpf = compute_s_lpf(&s);
        for (i, &v) in s_lpf.iter().enumerate() {
            assert!(v.abs() < 1e-12, "s_LPF[{i}] = {v}");
        }
    }

    /// `compute_s_lpf` applied to a DC input recovers the filter's DC
    /// gain on the interior (away from zero-padded edges). Annex D
    /// `h_LPF` is a symmetric lowpass, so its DC gain is `Σ h_LPF(j)`.
    #[test]
    fn compute_s_lpf_interior_matches_filter_dc_gain() {
        let s = [1.0f64; PITCH_INPUT_LEN];
        let s_lpf = compute_s_lpf(&s);
        let dc_gain: f64 = (-H_LPF_HALF..=H_LPF_HALF)
            .map(|j| f64::from(lpf_tap(j)))
            .sum();
        // Interior samples (far from the ±150 edges) see the full
        // filter. Sample at n=0.
        let center = s_lpf[W_I_HALF as usize];
        assert!(
            (center - dc_gain).abs() < 1e-9,
            "s_LPF(0) = {center}, expected DC gain {dc_gain}"
        );
    }

    // -----------------------------------------------------------------
    // §0.4 — pitch refinement
    // -----------------------------------------------------------------

    #[test]
    fn refine_offsets_are_symmetric_and_quarter_sample_spaced() {
        // Spacing 0.25, symmetric around 0, excludes 0.
        assert!(!REFINE_OFFSETS.contains(&0.0));
        for i in 1..N_REFINE_CANDIDATES {
            let step = REFINE_OFFSETS[i] - REFINE_OFFSETS[i - 1];
            assert!((step - 0.25).abs() < 1e-12, "step at i={i}: {step}");
        }
        let sum: f64 = REFINE_OFFSETS.iter().sum();
        assert!(sum.abs() < 1e-12, "offsets not symmetric, sum = {sum}");
        // Range is ±9/8.
        assert_eq!(REFINE_OFFSETS[0], -9.0 / 8.0);
        assert_eq!(REFINE_OFFSETS[9], 9.0 / 8.0);
    }

    #[test]
    fn harmonic_count_respects_admissible_range() {
        // At ω̂_0 = 2π/123.125 (smallest pitch period): L̂ = ⌊0.9254·(π·123.125/(2π) + 0.25)⌋
        //                                                  = ⌊0.9254·62.0625⌋ = ⌊57.4⌋ = 57.
        // Clamped to 56.
        let omega_lo = 2.0 * core::f64::consts::PI / 123.125;
        assert_eq!(harmonic_count_for(omega_lo), 56);
        // At ω̂_0 = 2π/19.875 (largest pitch period): L̂ = ⌊0.9254·(9.9375 + 0.25)⌋
        //                                                 = ⌊0.9254·10.1875⌋ = ⌊9.43⌋ = 9.
        let omega_hi = 2.0 * core::f64::consts::PI / 19.875;
        assert_eq!(harmonic_count_for(omega_hi), 9);
        // Mid-range spot check: ω̂_0 = 2π/50 → L̂ = ⌊0.9254·(25 + 0.25)⌋ = ⌊23.4⌋ = 23.
        let omega_mid = 2.0 * core::f64::consts::PI / 50.0;
        assert_eq!(harmonic_count_for(omega_mid), 23);
    }

    #[test]
    fn quantize_pitch_index_covers_full_range() {
        // Lowest ω̂_0 → b̂_0 = 207.
        let omega_lo = 2.0 * core::f64::consts::PI / 123.125;
        assert_eq!(quantize_pitch_index(omega_lo), Some(207));
        // Highest ω̂_0 → b̂_0 = 0.
        let omega_hi = 2.0 * core::f64::consts::PI / 19.875;
        assert_eq!(quantize_pitch_index(omega_hi), Some(0));
        // Out of range returns None.
        let omega_too_hi = 2.0 * core::f64::consts::PI / 10.0; // P=10, very high ω
        assert_eq!(quantize_pitch_index(omega_too_hi), None);
        let omega_too_lo = 2.0 * core::f64::consts::PI / 250.0; // P=250, very low
        assert_eq!(quantize_pitch_index(omega_too_lo), None);
    }

    /// `quantize_pitch_index` and the fullrate decoder's pitch_decode
    /// must round-trip within one quantization step — the encoder's
    /// `−39` floor pairs with the decoder's `+39.5` offset to
    /// implement midtread quantization per addendum §0.4.4.
    #[test]
    fn quantize_pitch_index_roundtrips_through_decoder() {
        use core::f64::consts::PI;
        // Sweep P and check that quantize(2π/P) reconstructs to within
        // one quantization step of the input ω.
        for p_times_2 in 42u32..=244 {
            // P in half-sample units: 21.0 to 122.0.
            let p = f64::from(p_times_2) * 0.5;
            let omega_hat = 2.0 * PI / p;
            let b0 = quantize_pitch_index(omega_hat).expect("in range");
            // Decoder: ω̃_0 = 4π / (b̂_0 + 39.5).
            let omega_tilde = 4.0 * PI / (f64::from(b0) + 39.5);
            // One quantization step ≈ |dω̃/db̂| = 4π / (b̂+39.5)².
            let step = 4.0 * PI / (f64::from(b0) + 39.5).powi(2);
            assert!(
                (omega_hat - omega_tilde).abs() <= step,
                "P={p}: ω̂={omega_hat:.6}, ω̃={omega_tilde:.6}, step={step:.6}"
            );
        }
    }

    /// Build a broadband periodic signal: harmonics 1..=max_h at
    /// amplitudes `1/h` (natural glottal-like rolloff). Populates
    /// spectral bins across the full range `m ∈ [0, 128]`, so the
    /// Eq. 24 residual (summed from m=50 upward) has non-trivial
    /// signal content to fit.
    fn broadband_periodic(period: f64, max_h: u32) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let omega = 2.0 * core::f64::consts::PI / period;
        for (idx, slot) in signal.iter_mut().enumerate() {
            let n = idx as i32 - W_R_HALF;
            let nf = f64::from(n);
            let mut s = 0.0;
            for h in 1..=max_h {
                s += (f64::from(h) * omega * nf).cos() / f64::from(h);
            }
            *slot = s;
        }
        signal
    }

    /// `refine_pitch` returns a refinement within ±1.125 samples of
    /// `P̂_I` (quarter-sample resolution). Broadband signal so E_R
    /// above m=50 has non-trivial content.
    #[test]
    fn refine_pitch_returns_candidate_within_offset_range() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        let p_refined = 2.0 * PI / refined.omega_hat;
        assert!(
            (p_refined - 50.0).abs() <= 9.0 / 8.0 + 1e-6,
            "P refined = {p_refined}, out of ±1.125 around 50"
        );
        let matches_a_candidate = REFINE_OFFSETS
            .iter()
            .any(|&offset| (p_refined - (50.0 + offset)).abs() < 1e-9);
        assert!(matches_a_candidate, "P refined = {p_refined} not a candidate");
    }

    /// For a broadband harmonic signal at period 50 with `P̂_I = 50`,
    /// the refinement should pick a candidate close to 50. Allow ±3/8
    /// tolerance — the residual landscape around the truth is shallow
    /// and finite-precision DFT evaluation can push the minimum one
    /// or two steps off the nearest-to-truth candidate.
    #[test]
    fn refine_pitch_picks_close_candidate_on_broadband_signal() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        let p_refined = 2.0 * PI / refined.omega_hat;
        assert!(
            (p_refined - 50.0).abs() <= 3.0 / 8.0 + 1e-6,
            "P refined = {p_refined}, expected within ±3/8 of 50"
        );
    }

    /// L̂, â_l, b̂_l derived parameters are consistent with the refined
    /// ω̂_0 (matches `harmonic_count_for` and `HarmonicBasis::bin_endpoints`).
    #[test]
    fn refine_pitch_derived_parameters_are_consistent() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 50.0);
        assert_eq!(refined.l_hat, harmonic_count_for(refined.omega_hat));
        assert_eq!(refined.b0, quantize_pitch_index(refined.omega_hat).unwrap());
        assert!((L_HAT_MIN..=L_HAT_MAX).contains(&refined.l_hat));
        assert!(refined.b0 <= B0_MAX);
    }

    /// Diagnostic: dump E_R for each candidate.
    #[test]
    #[ignore]
    fn probe_dump_residual_e_r() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let omega_truth = 2.0 * PI / 50.0;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega_truth * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        for &offset in &REFINE_OFFSETS {
            let p = 50.0 + offset;
            let omega0 = 2.0 * PI / p;
            let e_r = residual_e_r(&sw, &basis, omega0);
            println!("  P = {p:6.3} (offset {offset:+.3})  ω₀ = {omega0:.6}  E_R = {e_r:.6e}");
        }
    }

    /// Residual is nonnegative (sum of squared magnitudes).
    #[test]
    fn refine_pitch_residual_is_nonnegative() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(40.0, 15);
        let sw = signal_spectrum(&signal);
        let refined = refine_pitch(&sw, &basis, 40.0);
        assert!(refined.e_r >= 0.0, "E_R = {}", refined.e_r);
        assert!(refined.e_r.is_finite());
    }

    // -----------------------------------------------------------------
    // §0.7 — V/UV determination
    // -----------------------------------------------------------------

    #[test]
    fn band_count_matches_eq34_table() {
        // Table values from addendum §0.7.1.
        assert_eq!(band_count_for(9), 3);
        assert_eq!(band_count_for(10), 4);
        assert_eq!(band_count_for(11), 4);
        assert_eq!(band_count_for(12), 4);
        assert_eq!(band_count_for(36), 12);
        assert_eq!(band_count_for(37), 12);
        assert_eq!(band_count_for(56), 12);
        // Spot-check monotonicity across the admissible range.
        let mut prev = 0u8;
        for l in L_HAT_MIN..=L_HAT_MAX {
            let k = band_count_for(l);
            assert!(k >= prev, "K̂({l}) = {k}, prev = {prev}");
            assert!(k <= K_HAT_MAX);
            prev = k;
        }
    }

    #[test]
    fn vuv_state_cold_start_matches_spec() {
        let s = VuvState::cold_start();
        assert_eq!(s.xi_max(), XI_MAX_FLOOR);
        for k in 0..=K_HAT_MAX {
            assert_eq!(s.vuv_prev(k), 0);
        }
        // Out-of-range access is zero, not a panic.
        assert_eq!(s.vuv_prev(K_HAT_MAX + 1), 0);
    }

    /// `ξ_max` attack case (Eq. 41 Case 1): when `ξ_0 > ξ_max(−1)`,
    /// the tracker moves halfway toward the current frame's `ξ_0`.
    /// Drive this directly by constructing a signal with `ξ_0` larger
    /// than the cold-start floor.
    #[test]
    fn vuv_xi_max_attack_blends_50_50() {
        let basis = HarmonicBasis::new();
        // Broadband signal → non-trivial ξ_0.
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let prev_xi_max = state.xi_max;
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);

        // Compute the ξ_0 we expect internally.
        let wr_dc = basis.w_r(0);
        let inv = 1.0 / (wr_dc * wr_dc);
        let mut xi_lf = 0.0;
        for m in 0..=63 {
            xi_lf += sw[m].norm_sqr();
        }
        xi_lf *= inv;
        let mut xi_hf = 0.0;
        for m in 64..=128 {
            xi_hf += sw[m].norm_sqr();
        }
        xi_hf *= inv;
        let xi_0 = xi_lf + xi_hf;
        if xi_0 > prev_xi_max {
            let expected = 0.5 * prev_xi_max + 0.5 * xi_0;
            assert!(
                (res.xi_max_after - expected).abs() < 1e-6 * expected.abs().max(1.0),
                "xi_max_after = {}, expected = {}",
                res.xi_max_after,
                expected
            );
        }
    }

    /// Silent frame: `ξ_0 ≈ 0`, Eq. 41 Case 2 would give decay ≈ 0.99·20000 = 19800
    /// (below floor), so Case 3 clamps to the floor.
    #[test]
    fn vuv_xi_max_stays_at_floor_on_silence() {
        let basis = HarmonicBasis::new();
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        // Fabricate a refinement for silence — exact values don't
        // matter here since all D_k denominators are zero.
        let refinement = PitchRefinement {
            omega_hat: 2.0 * core::f64::consts::PI / 50.0,
            l_hat: 23,
            b0: 41,
            e_r: 0.0,
        };
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        assert_eq!(res.xi_max_after, XI_MAX_FLOOR);
    }

    /// `K̂` matches the band count derived from the refinement's `L̂`.
    #[test]
    fn vuv_result_k_hat_matches_band_count() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        assert_eq!(res.k_hat, band_count_for(refinement.l_hat));
        assert!(res.k_hat <= K_HAT_MAX);
    }

    /// On silent input, every `D_k` falls back to 1.0 (denominator
    /// zero → "no confidence" convention), forcing unvoiced.
    #[test]
    fn vuv_silent_input_yields_all_unvoiced() {
        let basis = HarmonicBasis::new();
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        let refinement = PitchRefinement {
            omega_hat: 2.0 * core::f64::consts::PI / 50.0,
            l_hat: 23,
            b0: 41,
            e_r: 0.0,
        };
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        for k in 1..=res.k_hat {
            assert_eq!(res.vuv[k as usize], 0, "band {k} voiced on silence");
            assert_eq!(res.d_k[k as usize], 1.0);
        }
    }

    /// Poor-pitch-match case (Eq. 37 Case 1): when `E(P̂_I) > 0.5`,
    /// bands `k ≥ 2` get `Θ_ξ = 0`, forcing unvoiced for those bands.
    /// Band 1 is exempt from this forcing rule.
    #[test]
    fn vuv_poor_pitch_match_forces_bands_2_plus_unvoiced() {
        let basis = HarmonicBasis::new();
        // Broadband signal makes some bands naturally voiced with a
        // good pitch. We'll then tell determine_vuv that E(P̂_I) is
        // high to trigger the poor-match rule on bands 2+.
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.75, &mut state);
        for k in 2..=res.k_hat {
            assert_eq!(
                res.vuv[k as usize], 0,
                "band {k} voiced under poor pitch match"
            );
        }
    }

    /// Hysteresis: a band that was voiced last frame uses the higher
    /// threshold `0.5625` vs the lower `0.45` from cold start. For a
    /// borderline `D_k` between the two thresholds, the band flips
    /// from voiced (with history) to unvoiced (without).
    #[test]
    fn vuv_hysteresis_threshold_depends_on_previous_decision() {
        // This test checks the structural property: with identical
        // inputs, a state carrying v̂_k(−1) = 1 produces at least as
        // many voiced bands as a cold-start state (higher threshold
        // = easier to stay voiced). It does not claim a specific band
        // flips — that depends on the exact D_k values.
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut cold = VuvState::cold_start();
        let res_cold = determine_vuv(&sw, &basis, &refinement, 0.1, &mut cold);

        let mut warm = VuvState::cold_start();
        // Pre-seed warm state with v̂_k(−1) = 1 for all bands.
        for k in 1..=K_HAT_MAX {
            warm.vuv_prev[k as usize] = 1;
        }
        warm.k_prev = K_HAT_MAX;
        let res_warm = determine_vuv(&sw, &basis, &refinement, 0.1, &mut warm);

        let count_cold: u32 = (1..=res_cold.k_hat)
            .map(|k| u32::from(res_cold.vuv[k as usize]))
            .sum();
        let count_warm: u32 = (1..=res_warm.k_hat)
            .map(|k| u32::from(res_warm.vuv[k as usize]))
            .sum();
        assert!(
            count_warm >= count_cold,
            "warm voiced count {count_warm} < cold voiced count {count_cold}"
        );
    }

    /// State commit: after `determine_vuv`, the `state`'s `vuv_prev`
    /// and `xi_max` reflect the current frame's decisions and tracker.
    #[test]
    fn vuv_state_is_committed_after_call() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        assert_eq!(state.xi_max, res.xi_max_after);
        for k in 1..=res.k_hat {
            assert_eq!(state.vuv_prev[k as usize], res.vuv[k as usize]);
        }
        // Slots past k_hat are zeroed.
        for k in (res.k_hat + 1)..=K_HAT_MAX {
            assert_eq!(state.vuv_prev[k as usize], 0);
        }
    }

    /// `D_k` is in `[0, ~2]` on realistic inputs — it's a ratio of
    /// L2 residual to L2 signal, so exactly 0 means perfect fit and
    /// exactly 1 means synth is orthogonal to the observation.
    /// Degenerate denominator → 1.0 fallback.
    #[test]
    fn vuv_d_k_values_are_bounded_and_finite() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 20);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let res = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        for k in 1..=res.k_hat {
            let d = res.d_k[k as usize];
            assert!(d.is_finite() && d >= 0.0, "D_{k} = {d}");
        }
    }

    // -----------------------------------------------------------------
    // §0.5 — spectral amplitude estimation
    // -----------------------------------------------------------------

    #[test]
    fn band_for_harmonic_follows_ceil_l_div_3() {
        // K̂ = 12 (high-harmonic case): standard bands are l∈{1,2,3}→k=1, {4,5,6}→k=2, etc.
        assert_eq!(band_for_harmonic(1, 12), 1);
        assert_eq!(band_for_harmonic(2, 12), 1);
        assert_eq!(band_for_harmonic(3, 12), 1);
        assert_eq!(band_for_harmonic(4, 12), 2);
        assert_eq!(band_for_harmonic(33, 12), 11);
        assert_eq!(band_for_harmonic(34, 12), 12);
        assert_eq!(band_for_harmonic(56, 12), 12); // capped at K̂
        // Smaller K̂: everything above 3(K̂−1) goes to the highest band.
        assert_eq!(band_for_harmonic(9, 3), 3);
        assert_eq!(band_for_harmonic(10, 3), 3);
        assert_eq!(band_for_harmonic(5, 3), 2);
        // Edge: l=0 or K̂=0 returns 0.
        assert_eq!(band_for_harmonic(0, 12), 0);
        assert_eq!(band_for_harmonic(1, 0), 0);
    }

    /// A helper returning a builder-assembled `VuvResult` with all
    /// bands voiced (for §0.5 tests that bypass §0.7).
    fn vuv_all_voiced(k_hat: u8) -> VuvResult {
        let mut vuv = [0u8; (K_HAT_MAX + 1) as usize];
        for k in 1..=k_hat {
            vuv[k as usize] = 1;
        }
        VuvResult {
            k_hat,
            vuv,
            d_k: [0.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    fn vuv_all_unvoiced(k_hat: u8) -> VuvResult {
        VuvResult {
            k_hat,
            vuv: [0; (K_HAT_MAX + 1) as usize],
            d_k: [1.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    /// Spectral amplitudes on silent input are zero (or numerically
    /// negligible) for every harmonic.
    #[test]
    fn estimate_amplitudes_zero_on_silence() {
        let basis = HarmonicBasis::new();
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        let refinement = PitchRefinement {
            omega_hat: 2.0 * core::f64::consts::PI / 50.0,
            l_hat: 23,
            b0: 41,
            e_r: 0.0,
        };
        let vuv = vuv_all_voiced(band_count_for(refinement.l_hat));
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv);
        for l in 1..=refinement.l_hat {
            assert!(
                amps[l as usize].abs() < 1e-12,
                "M̂_{l} = {} on silence",
                amps[l as usize]
            );
        }
    }

    /// Eq. 43 recovers the amplitude of a pure cosine within its
    /// fundamental harmonic. For `s(n) = A·cos(ω·n)` with `ω = 2π/P`,
    /// `|S_w(m)|²` on the l=1 support has energy `(A/2)² · |W_R(k)|²`
    /// for each m (the main-lobe term), and Eq. 43's ratio gives
    /// `sqrt(Σ (A/2)² · |W_R|²  / Σ |W_R|²) = A/2`.
    #[test]
    fn estimate_amplitudes_voiced_recovers_half_cosine_amplitude() {
        use core::f64::consts::PI;
        let basis = HarmonicBasis::new();
        let period = 50.0;
        let amplitude = 1.0;
        let omega = 2.0 * PI / period;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = amplitude * (omega * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, period);
        let vuv = vuv_all_voiced(band_count_for(refinement.l_hat));
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv);
        // Expected: M̂_1 ≈ A/2 = 0.5. Allow 10% slack for quarter-sample
        // refinement offset and window sidelobe spillover.
        let m1 = amps[1];
        assert!(
            (m1 - 0.5).abs() < 0.05,
            "M̂_1 = {m1}, expected ≈ 0.5 (A/2 for pure cosine)"
        );
        // Higher harmonics should be near zero (no energy there).
        for l in 3..=refinement.l_hat {
            assert!(
                amps[l as usize].abs() < 0.05,
                "M̂_{l} = {} (spurious on pure-tone input)",
                amps[l as usize]
            );
        }
    }

    /// Eq. 43 and Eq. 44 produce different values for the same input.
    /// This validates the code path differentiation — if V/UV were
    /// ignored, all amplitudes would be identical between the two.
    #[test]
    fn estimate_amplitudes_voiced_and_unvoiced_differ() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let k_hat = band_count_for(refinement.l_hat);
        let v = vuv_all_voiced(k_hat);
        let u = vuv_all_unvoiced(k_hat);
        let amps_v = estimate_spectral_amplitudes(&sw, &basis, &refinement, &v);
        let amps_u = estimate_spectral_amplitudes(&sw, &basis, &refinement, &u);
        let mut any_diff = false;
        for l in 1..=refinement.l_hat {
            if (amps_v[l as usize] - amps_u[l as usize]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Eq. 43 and Eq. 44 produced identical amplitudes");
    }

    /// Amplitudes are non-negative (both Eq. 43 and Eq. 44 are `sqrt` outputs).
    #[test]
    fn estimate_amplitudes_are_nonnegative_and_finite() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let mut state = VuvState::cold_start();
        let vuv = determine_vuv(&sw, &basis, &refinement, 0.1, &mut state);
        let amps = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv);
        for l in 1..=refinement.l_hat {
            let a = amps[l as usize];
            assert!(a.is_finite() && a >= 0.0, "M̂_{l} = {a}");
        }
        // Slots past L̂ are zero.
        for l in (refinement.l_hat + 1)..=L_HAT_MAX {
            assert_eq!(amps[l as usize], 0.0);
        }
        // DC is always zero (not written).
        assert_eq!(amps[0], 0.0);
    }

    // -----------------------------------------------------------------
    // §0.9 — consolidated AnalysisState
    // -----------------------------------------------------------------

    #[test]
    fn analysis_state_cold_start_is_in_preroll() {
        let s = AnalysisState::new();
        assert!(s.in_preroll());
        assert_eq!(s.preroll_counter(), 0);
        assert_eq!(s.predictor().l_tilde_prev(), L_TILDE_COLD_START);
        assert_eq!(s.vuv().xi_max(), XI_MAX_FLOOR);
        assert_eq!(s.pitch_history().prev_pitch, PITCH_COLD_START);
    }

    #[test]
    fn analysis_state_preroll_saturates_at_two() {
        let mut s = AnalysisState::new();
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), 1);
        assert!(s.in_preroll());
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
        assert!(!s.in_preroll());
        // Saturates.
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
    }

    #[test]
    fn analysis_state_commit_pitch_shifts_history() {
        let mut s = AnalysisState::new();
        // Cold start: prev_pitch=100, prev_err_1=0, prev_err_2=0.
        s.commit_pitch(55.0, 0.25);
        let h1 = s.pitch_history();
        assert_eq!(h1.prev_pitch, 55.0);
        assert_eq!(h1.prev_err_1, 0.25);
        assert_eq!(h1.prev_err_2, 0.0); // was prev_err_1 = 0.0 before
        // Second commit: the old err_1 (0.25) shifts into err_2.
        s.commit_pitch(60.0, 0.30);
        let h2 = s.pitch_history();
        assert_eq!(h2.prev_pitch, 60.0);
        assert_eq!(h2.prev_err_1, 0.30);
        assert_eq!(h2.prev_err_2, 0.25);
    }

    #[test]
    fn analysis_state_hpf_mut_filters_samples() {
        let mut s = AnalysisState::new();
        // Impulse through HPF: y[0] = 1, y[1] = 0.99·1 − 1 = −0.01.
        let y0 = s.hpf_mut().step(1.0);
        let y1 = s.hpf_mut().step(0.0);
        assert!((y0 - 1.0).abs() < 1e-12);
        assert!((y1 - (-0.01)).abs() < 1e-12);
    }

    #[test]
    fn analysis_state_predictor_mut_allows_matched_decoder_commit() {
        let mut s = AnalysisState::new();
        let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=16u32 {
            m_tilde[l as usize] = f64::from(l) * 0.5;
        }
        s.predictor_mut().commit(&m_tilde, 16);
        assert_eq!(s.predictor().l_tilde_prev(), 16);
    }

    // -----------------------------------------------------------------
    // §0.1 — HPF
    // -----------------------------------------------------------------

    /// Zero input → zero output (HPF is LTI and has zero state at cold start).
    #[test]
    fn hpf_zero_input_is_zero_output() {
        let mut hpf = HpfState::new();
        for _ in 0..1000 {
            assert_eq!(hpf.step(0.0), 0.0);
        }
    }

    /// DC input → output decays to zero. With input `x = 1.0` forever,
    /// `y[n] = y[n−1] · 0.99 + (1 − 1) = 0.99·y[n−1]` after the first
    /// step. The first step is `y[0] = 1 − 0 + 0 = 1`, then `y[1] =
    /// 1 − 1 + 0.99·1 = 0.99`, `y[2] = 0.99² · 1 = 0.9801`, etc.
    /// After many samples the output should asymptote to zero.
    #[test]
    fn hpf_blocks_dc() {
        let mut hpf = HpfState::new();
        let mut last = 0.0;
        for i in 0..2000 {
            last = hpf.step(1.0);
            if i == 0 {
                assert!((last - 1.0).abs() < 1e-12, "y[0] = {last}");
            }
        }
        // After 2000 samples, DC is thoroughly suppressed.
        assert!(last.abs() < 1e-6, "DC not blocked: y[1999] = {last}");
    }

    /// Run_pcm promotes i16 samples to f64 correctly.
    #[test]
    fn hpf_run_pcm_matches_step_loop() {
        let pcm: Vec<i16> = (0..32).map(|i| (i * 100) as i16).collect();
        let mut expected = vec![0.0; pcm.len()];
        let mut hpf_a = HpfState::new();
        for (i, &x) in pcm.iter().enumerate() {
            expected[i] = hpf_a.step(f64::from(x));
        }
        let mut actual = vec![0.0; pcm.len()];
        let mut hpf_b = HpfState::new();
        hpf_b.run_pcm(&pcm, &mut actual);
        assert_eq!(actual, expected);
    }

    /// HPF is LTI: `filter(a·x + b·y) = a·filter(x) + b·filter(y)`,
    /// subject to matched initial state. Run two signals plus a linear
    /// combination and verify the relation.
    #[test]
    fn hpf_is_linear_and_time_invariant() {
        let x: Vec<f64> = (0..200).map(|n| (n as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..200).map(|n| (n as f64 * 0.05).cos()).collect();
        let mut a_buf = x.clone();
        let mut b_buf = y.clone();
        HpfState::new().run_in_place(&mut a_buf);
        HpfState::new().run_in_place(&mut b_buf);
        let mut combined: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| 2.0 * xi + 3.0 * yi)
            .collect();
        HpfState::new().run_in_place(&mut combined);
        for (i, (&c, (&a, &b))) in combined.iter().zip(a_buf.iter().zip(b_buf.iter())).enumerate() {
            let expected = 2.0 * a + 3.0 * b;
            assert!(
                (c - expected).abs() < 1e-9,
                "LTI violated at i={i}: {c} vs {expected}"
            );
        }
    }

    // -----------------------------------------------------------------
    // §0.6 — log-magnitude prediction residual
    // -----------------------------------------------------------------

    #[test]
    fn predictor_state_cold_start_matches_spec() {
        let st = PredictorState::cold_start();
        assert_eq!(st.l_tilde_prev(), L_TILDE_COLD_START);
        // M̃_l(−1) = 1.0 for all l per §6.3 prose; log₂ 1 = 0 makes the
        // first-frame predictor contribution vanish.
        for l in 0..=L_HAT_MAX as u32 {
            assert_eq!(st.read(l), 1.0, "M̃_{l}(−1) = {}", st.read(l));
        }
    }

    #[test]
    fn predictor_state_read_applies_eq56_eq57_boundaries() {
        let mut st = PredictorState::cold_start();
        // Commit a short vector with distinct values to see the boundaries.
        let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
        m_tilde[1] = 2.0;
        m_tilde[2] = 4.0;
        m_tilde[3] = 8.0;
        st.commit(&m_tilde, 3);
        // Eq. 56: M̃_0 = 1.0.
        assert_eq!(st.read(0), 1.0);
        // Interior values.
        assert_eq!(st.read(1), 2.0);
        assert_eq!(st.read(3), 8.0);
        // Eq. 57: beyond L̃(−1), clamp to last.
        assert_eq!(st.read(4), 8.0);
        assert_eq!(st.read(30), 8.0);
    }

    /// On cold start, `M̃_l(−1) = 1.0` for all l, so `log₂` of every
    /// past value is 0 and the predictor contributes zero. The
    /// residual `T̂_l` equals `log₂ M̂_l(0)` exactly.
    #[test]
    fn compute_prediction_residual_cold_start_equals_log2_amplitude() {
        let st = PredictorState::cold_start();
        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        // Fill with distinct positive values.
        for l in 1..=12u32 {
            m_hat[l as usize] = f64::from(l) * 0.5 + 1.0;
        }
        let t_hat = compute_prediction_residual(&m_hat, 12, &st);
        for l in 1..=12u32 {
            let expected = m_hat[l as usize].log2();
            assert!(
                (t_hat[l as usize] - expected).abs() < 1e-12,
                "T̂_{l} = {}, expected log₂({}) = {}",
                t_hat[l as usize],
                m_hat[l as usize],
                expected
            );
        }
    }

    /// Mean-preservation invariant: `mean_l T̂_l = mean_l log₂ M̂_l(0)`
    /// by construction (the mean-removed predictor term has mean 0).
    #[test]
    fn compute_prediction_residual_preserves_mean_of_log2_amplitude() {
        let mut st = PredictorState::cold_start();
        // Seed past state with non-trivial values so ρ·P has teeth.
        let mut past = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=20u32 {
            past[l as usize] = 1.0 + f64::from(l) * 0.1; // M̃ ∈ [1.1, 3.0]
        }
        st.commit(&past, 20);

        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=18u32 {
            m_hat[l as usize] = 0.5 + f64::from(l) * 0.2;
        }
        let l_hat = 18u8;
        let t_hat = compute_prediction_residual(&m_hat, l_hat, &st);

        let t_mean: f64 = (1..=u32::from(l_hat))
            .map(|l| t_hat[l as usize])
            .sum::<f64>()
            / f64::from(l_hat);
        let log_m_mean: f64 = (1..=u32::from(l_hat))
            .map(|l| m_hat[l as usize].log2())
            .sum::<f64>()
            / f64::from(l_hat);
        assert!(
            (t_mean - log_m_mean).abs() < 1e-12,
            "mean(T̂) = {t_mean}, mean(log₂ M̂) = {log_m_mean}"
        );
    }

    /// When `M̂_l(0) = M̃_l(−1)` exactly (perfect prediction),
    /// `T̂_l = (1−ρ)·log₂ M̂_l + ρ·mean(log₂ M̂)` — a partial blend
    /// toward the mean weighted by the predictor coefficient. In the
    /// limit `ρ → 1` the residual collapses to just the mean; at
    /// typical `ρ ∈ [0.4, 0.7]` it retains significant per-harmonic
    /// information.
    #[test]
    fn compute_prediction_residual_partial_blend_when_prediction_perfect() {
        let mut st = PredictorState::cold_start();
        let mut m_same = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=16u32 {
            m_same[l as usize] = 1.0 + f64::from(l) * 0.3;
        }
        st.commit(&m_same, 16);
        let t_hat = compute_prediction_residual(&m_same, 16, &st);

        let log_mean: f64 = (1..=16u32)
            .map(|l| m_same[l as usize].log2())
            .sum::<f64>()
            / 16.0;
        let rho = fullrate_rho_f64(16);
        for l in 1..=16u32 {
            let log_m = m_same[l as usize].log2();
            let expected = (1.0 - rho) * log_m + rho * log_mean;
            assert!(
                (t_hat[l as usize] - expected).abs() < 1e-9,
                "T̂_{l} = {}, expected (1-ρ)·log + ρ·mean = {}",
                t_hat[l as usize],
                expected
            );
        }
    }

    /// `L̂ ≠ L̃(−1)` mismatch: Eq. 52/53 linear interpolation handles
    /// the harmonic-index mapping. Sanity-check that the residual is
    /// well-defined (finite) across a rising/falling L̂ transition.
    #[test]
    fn compute_prediction_residual_handles_l_hat_mismatch() {
        let mut st = PredictorState::cold_start();
        let mut past = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=10u32 {
            past[l as usize] = 2.0_f64.powf(f64::from(l) * 0.1);
        }
        st.commit(&past, 10); // L̃(−1) = 10

        let mut m_hat = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=25u32 {
            m_hat[l as usize] = 2.0_f64.powf(f64::from(l) * 0.05);
        }
        // Current frame has more harmonics than the past frame.
        let t_hat = compute_prediction_residual(&m_hat, 25, &st);
        for l in 1..=25u32 {
            assert!(t_hat[l as usize].is_finite(), "T̂_{l} = {}", t_hat[l as usize]);
        }
        // And the other direction: current frame has fewer harmonics.
        let t_hat_shrink = compute_prediction_residual(&m_hat, 5, &st);
        for l in 1..=5u32 {
            assert!(
                t_hat_shrink[l as usize].is_finite(),
                "T̂_{l} (shrink) = {}",
                t_hat_shrink[l as usize]
            );
        }
    }

    /// `fullrate_rho_f64` matches the Eq. 55 schedule (within the
    /// f32→f64 cast precision of the underlying helper).
    #[test]
    fn fullrate_rho_matches_eq55_schedule_points() {
        let tol = 1e-5;
        // L̂ ≤ 15 → 0.40.
        assert!((fullrate_rho_f64(9) - 0.40).abs() < tol);
        assert!((fullrate_rho_f64(15) - 0.40).abs() < tol);
        // 15 < L̂ ≤ 24 → 0.03·L − 0.05.
        assert!((fullrate_rho_f64(16) - 0.43).abs() < tol);
        assert!((fullrate_rho_f64(20) - 0.55).abs() < tol);
        assert!((fullrate_rho_f64(24) - 0.67).abs() < tol);
        // L̂ > 24 → 0.70.
        assert!((fullrate_rho_f64(25) - 0.70).abs() < tol);
        assert!((fullrate_rho_f64(56) - 0.70).abs() < tol);
    }

    /// Highest-band span is `L̂ − 3K̂ + 3` harmonics, all sharing the
    /// same band index `K̂`. For `L̂=23, K̂=8`, the top band covers
    /// `l ∈ {22, 23}` (2 harmonics). Verify band_for_harmonic agrees
    /// and that estimate_spectral_amplitudes uses the same V/UV bit
    /// across that span.
    #[test]
    fn estimate_amplitudes_highest_band_spans_multiple_harmonics_consistently() {
        let basis = HarmonicBasis::new();
        let signal = broadband_periodic(50.0, 15);
        let sw = signal_spectrum(&signal);
        let refinement = refine_pitch(&sw, &basis, 50.0);
        let k_hat = band_count_for(refinement.l_hat);
        // Build a VuvResult where every band is voiced EXCEPT the
        // highest — which is unvoiced.
        let mut vuv_bits = [1u8; (K_HAT_MAX + 1) as usize];
        vuv_bits[0] = 0;
        vuv_bits[k_hat as usize] = 0;
        // Zero out above k_hat so state is clean.
        for k in (k_hat + 1)..=K_HAT_MAX {
            vuv_bits[k as usize] = 0;
        }
        let vuv = VuvResult {
            k_hat,
            vuv: vuv_bits,
            d_k: [0.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        };
        let amps_split = estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv);
        // Redo with all voiced — highest-band amplitudes should differ.
        let amps_all_v =
            estimate_spectral_amplitudes(&sw, &basis, &refinement, &vuv_all_voiced(k_hat));
        // Harmonics in the highest band: l ∈ [3·(K̂−1)+1, L̂].
        let l_hi_lo = 3 * u32::from(k_hat - 1) + 1;
        for l in l_hi_lo..=u32::from(refinement.l_hat) {
            assert_eq!(
                band_for_harmonic(l, k_hat),
                k_hat,
                "harmonic {l} should map to highest band"
            );
            // Eq. 43 (voiced) vs Eq. 44 (unvoiced) should diverge.
            let diff = (amps_split[l as usize] - amps_all_v[l as usize]).abs();
            assert!(
                diff > 1e-9,
                "harmonic {l}: voiced/unvoiced produced same M̂_l (diff = {diff})"
            );
        }
    }
}
