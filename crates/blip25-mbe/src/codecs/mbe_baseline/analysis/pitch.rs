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
use crate::mbe_params::SAMPLES_PER_FRAME;

/// Half-support of Annex B window `w_I(n)`: `n ∈ [−W_I_HALF, W_I_HALF]`.
pub const W_I_HALF: i32 = 150;

/// Half-support of Annex D LPF `h_LPF(n)`: `n ∈ [−H_LPF_HALF, H_LPF_HALF]`.
pub const H_LPF_HALF: i32 = 10;

/// Lookahead buffer length the encoder maintains. Mirrors
/// `super::LOOKAHEAD_LEN` but kept here as a `pub const` so the shared
/// LPF helper doesn't need a circular import.
pub const LOOKAHEAD_LEN: usize = 3 * SAMPLES_PER_FRAME as usize;

/// Per-frame "tail extension" each pitch slot's LPF reaches past the
/// frame center: from frame center, the slot's LPF output covers
/// `n ∈ [−W_I_HALF, W_I_HALF]`. Frame centers sit at `slot·160 + 80`,
/// so the union of all slot LPF outputs covers buffer indices
/// `[slot0_center − W_I_HALF, slot2_center + W_I_HALF]` =
/// `[80 − 150, 2·160 + 80 + 150]` = `[−70, 550]` (inclusive both
/// ends → 621 entries).
const SHARED_LPF_TAIL: usize = (W_I_HALF as usize) - (SAMPLES_PER_FRAME as usize / 2);

/// Length of the shared LPF output covering all 3 pitch-slot windows
/// at once. Buf range is `[−70, 550]` inclusive both ends, so 621
/// entries: `(slot2_right − slot0_left) + 1` =
/// `(2·SAMPLES_PER_FRAME + 2·W_I_HALF) + 1`.
pub const SHARED_LPF_LEN: usize =
    2 * SAMPLES_PER_FRAME as usize + 2 * W_I_HALF as usize + 1;

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
/// Samples outside `s_input` are treated as zero (addendum §0.3.1) —
/// but `PITCH_INPUT_HALF = W_I_HALF + H_LPF_HALF`, so for every output
/// `n` and every tap `j` the source index `n - j + PITCH_INPUT_HALF`
/// stays inside `[0, PITCH_INPUT_LEN)`. The bounds check is provable
/// dead and the inner loop runs without one.
pub fn compute_s_lpf(s_input: &[f64; PITCH_INPUT_LEN]) -> [f64; (2 * W_I_HALF + 1) as usize] {
    // Pre-load LPF taps as f64 in a stack array so the inner loop is
    // a tight FIR convolution (21 mul-adds per output sample).
    const TAPS: usize = (2 * H_LPF_HALF + 1) as usize;
    let mut h = [0.0f64; TAPS];
    for j in -H_LPF_HALF..=H_LPF_HALF {
        h[(j + H_LPF_HALF) as usize] = f64::from(lpf_tap(j));
    }

    let mut out = [0.0f64; (2 * W_I_HALF + 1) as usize];
    // Output index `i = n + W_I_HALF`; source index for tap `k` (=
    // `j + H_LPF_HALF`) is `n - j + PITCH_INPUT_HALF`
    //   = (i - W_I_HALF) - (k - H_LPF_HALF) + PITCH_INPUT_HALF
    //   = i + (PITCH_INPUT_HALF - W_I_HALF + H_LPF_HALF) - k
    //   = i + 2·H_LPF_HALF - k          (since PITCH_INPUT_HALF = W_I_HALF + H_LPF_HALF)
    // → starting `src` for tap k=0 at output i is `i + 2·H_LPF_HALF`,
    // and decrements by 1 for each tap. Bounds: i ∈ [0, 2·W_I_HALF],
    // k ∈ [0, 2·H_LPF_HALF], so src ∈ [0, 2·(W_I_HALF + H_LPF_HALF)]
    // = [0, PITCH_INPUT_LEN − 1]. Safe without per-iteration check.
    let two_h = 2 * H_LPF_HALF as usize;
    for (i, slot) in out.iter_mut().enumerate() {
        let src_base = i + two_h;
        let mut acc = 0.0;
        for k in 0..TAPS {
            acc += s_input[src_base - k] * h[k];
        }
        *slot = acc;
    }
    out
}

/// LPF the entire 480-sample lookahead buffer in one pass, producing
/// 621 outputs covering buffer indices `[−70, 550]`. Each pitch slot's
/// 301-entry local `s_LPF` array (centered at `slot·160 + 80`) is then
/// just a 301-entry slice of the shared output:
///   - slot 0 → `[0..301]`     (buf `[−70, 230]`)
///   - slot 1 → `[160..461]`   (buf `[90, 390]`)
///   - slot 2 → `[320..621]`   (buf `[250, 550]`)
///
/// Replaces 3 separate `compute_s_lpf` passes (each 301 outputs × 21
/// taps) with one wider pass (621 × 21) — 30% fewer FIR mults per
/// frame. The arithmetic identity that makes this work: the slot's
/// per-window zero-extension boundaries (slot 0 at slot-local i=80,
/// slot 2 at slot-local i=240+1) coincide with the lookahead buffer's
/// natural `[0, LOOKAHEAD_LEN)` boundary, so a single shared
/// zero-extension produces the same FIR outputs as per-slot windowing.
pub fn compute_s_lpf_shared(lookahead: &[f64; LOOKAHEAD_LEN]) -> [f64; SHARED_LPF_LEN] {
    const TAPS: usize = (2 * H_LPF_HALF + 1) as usize;
    let mut h = [0.0f64; TAPS];
    for j in -H_LPF_HALF..=H_LPF_HALF {
        h[(j + H_LPF_HALF) as usize] = f64::from(lpf_tap(j));
    }

    // Padded input covering buf logical positions `[−80, 560]`
    // (length 641 = SHARED_LPF_LEN + 2·H_LPF_HALF): zero outside
    // `[0, LOOKAHEAD_LEN)`, lookahead inside.  The padding width on
    // each side is `SHARED_LPF_TAIL + H_LPF_HALF = 80`.
    const PAD_LEFT: usize = SHARED_LPF_TAIL + H_LPF_HALF as usize;
    const PAD_LEN: usize = SHARED_LPF_LEN + 2 * H_LPF_HALF as usize;
    let mut padded = [0.0f64; PAD_LEN];
    padded[PAD_LEFT..PAD_LEFT + LOOKAHEAD_LEN].copy_from_slice(lookahead);

    // Output index `i` ∈ [0, SHARED_LPF_LEN) is at buffer logical
    // position `B = i − SHARED_LPF_TAIL`. The FIR reads padded at buf
    // `[B − H_LPF_HALF, B + H_LPF_HALF]`, which in padded indices is
    // `[B + PAD_LEFT − H_LPF_HALF, B + PAD_LEFT + H_LPF_HALF]` =
    // `[i, i + 2·H_LPF_HALF]` since `PAD_LEFT − SHARED_LPF_TAIL =
    // H_LPF_HALF`. So `src_base = i + 2·H_LPF_HALF`, decrement by k.
    let two_h = 2 * H_LPF_HALF as usize;
    let mut out = [0.0f64; SHARED_LPF_LEN];
    for (i, slot) in out.iter_mut().enumerate() {
        let src_base = i + two_h;
        let mut acc = 0.0;
        for k in 0..TAPS {
            acc += padded[src_base - k] * h[k];
        }
        *slot = acc;
    }
    out
}

/// Slice the shared LPF output for the given pitch slot. Returns a
/// 301-entry slice covering that slot's local `s_LPF` array indexed
/// by `n + W_I_HALF` for `n ∈ [−W_I_HALF, W_I_HALF]`.
#[inline]
pub fn slot_s_lpf(shared: &[f64; SHARED_LPF_LEN], slot: usize) -> &[f64] {
    let start = slot * SAMPLES_PER_FRAME as usize;
    &shared[start..start + (2 * W_I_HALF + 1) as usize]
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
    /// `E(P)` precomputed at every half-sample grid point for fast
    /// lookup. Indexed by `i ∈ [0, PITCH_GRID_LEN)` where the grid
    /// pitch is `PITCH_GRID_MIN + i · PITCH_GRID_STEP`. Built once in
    /// `from_lpf`; consumed by [`Self::argmin_in_range`] /
    /// [`Self::e_of_p_grid`] / [`look_ahead`]. Off-grid queries still
    /// recompute via [`Self::e_of_p`].
    e_grid: [f64; PITCH_GRID_LEN],
    /// Sparse-table range-minimum index over `e_grid`. Optional —
    /// only built for `PitchSearch` instances that will see many
    /// `argmin_in_range` calls (typically `next1`/`next2` inside
    /// [`look_ahead`]). When `None`, `argmin_in_range` falls back to
    /// the linear-scan path. Boxed to keep the sparse-less
    /// `PitchSearch` size small for `EncodeTrace::clone()`.
    ///
    /// Layout: `sparse[k][i]` = `(min_e, argmin_idx)` over
    /// `e_grid[i..i+2^k]` with lex-ordering on `(min_e, argmin_idx)`
    /// so that ties resolve to the smaller index — preserving the
    /// linear-scan path's tie-break (first occurrence wins).
    sparse: Option<Box<SparseRmq>>,
}

/// Number of sparse-table levels: `⌈log2(PITCH_GRID_LEN)⌉`. With
/// `PITCH_GRID_LEN = 203`, k ranges 0..=7 (2^7 = 128 ≤ 203, 2^8 = 256 > 203).
const SPARSE_LEVELS: usize = 8;

/// One row of the sparse-table RMQ — `(min_e, argmin_idx)` per grid
/// index `i`, covering `e_grid[i..min(i+2^k, PITCH_GRID_LEN)]`.
type SparseRow = [(f64, u16); PITCH_GRID_LEN];

/// Sparse-table range-min over `e_grid`. `sparse[k][i]` = lex-min over
/// `e_grid[i..min(i+2^k, PITCH_GRID_LEN)]`.
type SparseRmq = [SparseRow; SPARSE_LEVELS];

/// Lex-min on `(value, idx)` pairs — ties on `value` resolve to the
/// smaller `idx`, which mirrors the linear-scan path's "first
/// occurrence wins" tie-break in [`PitchSearch::argmin_in_range`].
#[inline]
fn lex_min(a: (f64, u16), b: (f64, u16)) -> (f64, u16) {
    match a.0.partial_cmp(&b.0) {
        Some(core::cmp::Ordering::Less) => a,
        Some(core::cmp::Ordering::Greater) => b,
        _ => {
            if a.1 <= b.1 {
                a
            } else {
                b
            }
        }
    }
}

const R_MAX_LAG: usize = W_I_HALF as usize;

/// FFT length used for the §0.3.2 Eq. 7 autocorrelation. Must be ≥
/// `sw2.len() + R_MAX_LAG` so circular wraparound doesn't pollute
/// the lags we care about (`t ∈ [0, R_MAX_LAG]`). 512 is the next
/// power of two above 301 + 150 = 451 — keeps rustfft on its
/// fastest radix-2 path.
const AUTOCORR_FFT_LEN: usize = 512;

/// Compute r(t) = Σ_j sw2[j] · sw2[j+t] for t ∈ [0, R_MAX_LAG] via
/// FFT-based autocorrelation. Faster than the direct O(N²) form
/// for the (N=301, R=150) sizes used here — rustfft's radix-2 path
/// at length 512 is heavily SIMD-optimized.
fn autocorr_via_fft(sw2: &[f64; (2 * W_I_HALF + 1) as usize]) -> [f64; R_MAX_LAG + 1] {
    use num_complex::Complex;
    use rustfft::{Fft, FftPlanner};
    use std::sync::OnceLock;

    static FFT_FWD: OnceLock<std::sync::Arc<dyn Fft<f64>>> = OnceLock::new();
    static FFT_INV: OnceLock<std::sync::Arc<dyn Fft<f64>>> = OnceLock::new();
    let fft_fwd = FFT_FWD.get_or_init(|| {
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(AUTOCORR_FFT_LEN)
    });
    let fft_inv = FFT_INV.get_or_init(|| {
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_inverse(AUTOCORR_FFT_LEN)
    });

    let mut buf = [Complex::<f64>::new(0.0, 0.0); AUTOCORR_FFT_LEN];
    for (i, &x) in sw2.iter().enumerate() {
        buf[i].re = x;
    }
    fft_fwd.process(&mut buf);
    // |X(f)|² in-place; for real input X has hermitian symmetry so
    // the result is real, but we keep the complex layout for the IFFT.
    for c in buf.iter_mut() {
        c.re = c.re * c.re + c.im * c.im;
        c.im = 0.0;
    }
    fft_inv.process(&mut buf);

    // rustfft's inverse is unnormalized — divide by N. Lag t is at
    // index t in the IFFT output (positive lags are at the start of
    // the buffer; circular-aliased negative lags fold to the tail and
    // we don't read them).
    let scale = 1.0 / AUTOCORR_FFT_LEN as f64;
    let mut r = [0.0f64; R_MAX_LAG + 1];
    for (t, slot) in r.iter_mut().enumerate() {
        *slot = buf[t].re * scale;
    }
    r
}

impl PitchSearch {
    /// Build the per-frame pitch search from the 321-sample analysis
    /// buffer (see [`PITCH_INPUT_LEN`]).
    pub fn new(s_input: &[f64; PITCH_INPUT_LEN]) -> Self {
        let s_lpf = compute_s_lpf(s_input);
        Self::from_lpf(&s_lpf)
    }

    /// Build from a slice of an already-LPF'd buffer. Used by the
    /// shared-LPF fast path: `compute_s_lpf_shared` produces 621
    /// outputs once per frame, then `slot_s_lpf` returns 301-entry
    /// slices that feed three calls to `from_lpf_slice`. Panics if
    /// the slice isn't exactly `2·W_I_HALF + 1` entries long.
    pub fn from_lpf_slice(s_lpf: &[f64]) -> Self {
        let arr: &[f64; (2 * W_I_HALF + 1) as usize] = s_lpf
            .try_into()
            .expect("from_lpf_slice expects exactly 2·W_I_HALF + 1 entries");
        Self::from_lpf(arr)
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
        // Eq. 7 via FFT — r(t) = Σ sw2[j]·sw2[j+t] is the
        // autocorrelation of sw2, so r = IFFT(|FFT(zero-padded sw2)|²).
        // We zero-pad to AUTOCORR_FFT_LEN (= 512, the next pow-2 ≥
        // sw2.len() + R_MAX_LAG) so the circular autocorrelation
        // matches the linear one over t ∈ [0, R_MAX_LAG]. FFT is
        // general DSP, not P25 IP — same clean-room justification as
        // the §0.2 signal_spectrum and §1.12.1 unvoiced-synth FFT.
        let r = autocorr_via_fft(&sw2);
        // Precompute E(P) at every half-sample grid point so the
        // hot loops in look_back / look_ahead become slice scans
        // instead of recomputing the Eq. 5 sum each visit.
        let mut e_grid = [0.0f64; PITCH_GRID_LEN];
        let mut tmp = Self {
            energy,
            w_i_fourth_moment: w4,
            r,
            e_grid: [0.0; PITCH_GRID_LEN],
            sparse: None,
        };
        for (i, slot) in e_grid.iter_mut().enumerate() {
            let p = PITCH_GRID_MIN + (i as f64) * PITCH_GRID_STEP;
            *slot = tmp.e_of_p(p);
        }
        tmp.e_grid = e_grid;
        tmp
    }

    /// Build the sparse-table RMQ over `e_grid` so that
    /// [`Self::argmin_in_range`] runs in `O(1)` per call instead of the
    /// default linear scan. Worth ~25 µs/frame on the [`look_ahead`]
    /// 203-iteration outer loop where each step does two
    /// `argmin_in_range` calls on the lookahead frames.
    ///
    /// Idempotent — calling more than once is a no-op (the table is
    /// determined by `e_grid`, which doesn't change after construction).
    pub fn enable_argmin_fast(&mut self) {
        if self.sparse.is_some() {
            return;
        }
        // Box the SparseRmq directly on the heap. The optimizer typically
        // elides the stack copy via NRVO for Box::new of a fixed-size
        // array literal, but core::array::from_fn keeps codegen sane on
        // any Rust version.
        let mut sparse: Box<SparseRmq> = Box::new(core::array::from_fn(|_| {
            [(f64::INFINITY, 0u16); PITCH_GRID_LEN]
        }));

        // Level 0: each cell is the value at its own index.
        for i in 0..PITCH_GRID_LEN {
            sparse[0][i] = (self.e_grid[i], i as u16);
        }
        // Level k > 0: cell i covers e_grid[i..i+2^k] = union of
        // (k−1, i) and (k−1, i + 2^(k−1)). When the right half spills
        // off the grid, just reuse the left (which is already edge-
        // clamped at the previous level).
        for k in 1..SPARSE_LEVELS {
            let half = 1usize << (k - 1);
            for i in 0..PITCH_GRID_LEN {
                let left = sparse[k - 1][i];
                let right = if i + half < PITCH_GRID_LEN {
                    sparse[k - 1][i + half]
                } else {
                    left
                };
                sparse[k][i] = lex_min(left, right);
            }
        }
        self.sparse = Some(sparse);
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

    /// `E(P)` at the `i`-th half-sample grid point — i.e.
    /// `E(PITCH_GRID_MIN + i · PITCH_GRID_STEP)`, looked up from the
    /// cache built in `from_lpf`. Cheap: single array access.
    /// Out-of-range `i` returns `1.0` (matching the off-grid `e_of_p`
    /// silent-frame fallback).
    #[inline]
    pub fn e_of_p_grid(&self, i: usize) -> f64 {
        self.e_grid.get(i).copied().unwrap_or(1.0)
    }

    /// Argmin of `E(P)` over the half-sample grid within `[p_lo, p_hi]`,
    /// both bounds clamped to `[PITCH_GRID_MIN, PITCH_GRID_MAX]`. The
    /// range is inclusive on both ends. Returns `(P̂, E(P̂))`.
    ///
    /// `O(N)` linear scan by default. When [`Self::enable_argmin_fast`]
    /// has been called, falls through to the sparse-table RMQ for
    /// `O(1)` per query — the hot path inside [`look_ahead`].
    pub fn argmin_in_range(&self, p_lo: f64, p_hi: f64) -> (f64, f64) {
        let p_lo = p_lo.max(PITCH_GRID_MIN);
        let p_hi = p_hi.min(PITCH_GRID_MAX);
        // Round up to the next grid point at p_lo; round down at p_hi.
        let i_lo = ((p_lo - PITCH_GRID_MIN) / PITCH_GRID_STEP).ceil() as i32;
        let i_hi = ((p_hi - PITCH_GRID_MIN) / PITCH_GRID_STEP).floor() as i32;
        if i_lo > i_hi {
            // Empty range after clamp — degenerate but possible if
            // p_lo > p_hi or both fall in the same sub-grid bin.
            let p = PITCH_GRID_MIN + f64::from(i_lo.max(0)) * PITCH_GRID_STEP;
            return (p, 1.0);
        }
        let i_lo_u = i_lo as usize;
        let i_hi_u = (i_hi as usize).min(PITCH_GRID_LEN - 1);

        // Fast path: sparse-table RMQ. Cover [i_lo, i_hi] with two
        // pre-aggregated windows of length 2^k where
        // k = ⌊log2(i_hi − i_lo + 1)⌋; lex-min the two cells.
        if let Some(sparse) = self.sparse.as_deref() {
            let len = i_hi_u - i_lo_u + 1;
            // ⌊log2(len)⌋, valid because len ≥ 1 here.
            let k = (len as u32).ilog2() as usize;
            let window = 1usize << k;
            let left = sparse[k][i_lo_u];
            let right = sparse[k][i_hi_u + 1 - window];
            let (best_e, best_i_u16) = lex_min(left, right);
            let best_p = PITCH_GRID_MIN + f64::from(best_i_u16) * PITCH_GRID_STEP;
            return (best_p, best_e);
        }

        // Slow path: linear scan.
        let slice = &self.e_grid[i_lo_u..=i_hi_u];
        let (rel_idx, &best_e) = slice
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .expect("non-empty slice");
        let best_i = i_lo_u + rel_idx;
        let best_p = PITCH_GRID_MIN + (best_i as f64) * PITCH_GRID_STEP;
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
    for i in 0..PITCH_GRID_LEN {
        let p0 = PITCH_GRID_MIN + (i as f64) * PITCH_GRID_STEP;
        let e0 = current.e_of_p_grid(i);
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

/// Ratio tolerance for treating `P̂_B` as an integer harmonic of `P̂_F`
/// in [`decide_initial_pitch_escape`].
pub const OCTAVE_ESCAPE_TOL: f64 = 0.12;

/// §0.3 decision with a sub-harmonic octave escape (`BLIP25_PITCH_DECIDE`;
/// a classic octave-error guard, general DSP). Overrides the look-back
/// pick **only** when the look-back period is an integer multiple `N ≥ 2`
/// of the look-ahead period (`P̂_B / P̂_F` within `±OCTAVE_ESCAPE_TOL` of
/// `N`) **and** look-ahead is genuinely better (`CE_F < CE_B`): the
/// tracker is trapped on a sub-harmonic that look-ahead escaped (high-F0
/// female/child voices rendered 1–2 octaves low). On normal voiced
/// speech `P̂_B ≈ P̂_F` (ratio ≈ 1, never ≥ 2) so this never fires; the
/// worst case picks `P̂_F`, already an admissible §0.3 output.
#[inline]
pub fn decide_initial_pitch_escape(p_b: f64, ce_b: f64, p_f: f64, ce_f: f64) -> f64 {
    if p_f > 0.0 && ce_f < ce_b {
        let ratio = p_b / p_f;
        if ratio >= 2.0 - OCTAVE_ESCAPE_TOL {
            let n = ratio.round();
            if n >= 2.0 && (ratio - n).abs() <= OCTAVE_ESCAPE_TOL {
                return p_f;
            }
        }
    }
    decide_initial_pitch(p_b, ce_b, p_f, ce_f)
}

/// Parabolic (3-point) sub-sample refinement of the `E(P)` minimum around
/// grid pitch `p` (`BLIP25_PITCH_SUBSAMPLE`; classic sub-sample
/// autocorrelation-peak refinement, general DSP). Fits a parabola through
/// `E(p−step)`, `E(p)`, `E(p+step)` (same `E(P)` the tracker minimized)
/// and returns the vertex, clamped to ±½ a grid step of `p`. Falls back
/// to `p` when the three points are not convex (`p` is not a local min)
/// or the vertex lands outside `[−½, ½]` step. A *dense* sub-grid argmin
/// is a known negative (chases spurious `E(P)` dips) — the 3-point
/// curvature fit is the robust form. Safe on silence: `e_of_p` returns
/// `1.0` there, so all three neighbours tie (`denom = 0`) → fallback.
#[inline]
pub fn parabolic_refine_pitch(search: &PitchSearch, p: f64) -> f64 {
    let step = PITCH_GRID_STEP;
    let pm = p - step;
    let pp = p + step;
    if pm < PITCH_GRID_MIN || pp > PITCH_GRID_MAX {
        return p;
    }
    let e0 = search.e_of_p(pm);
    let e1 = search.e_of_p(p);
    let e2 = search.e_of_p(pp);
    let denom = e0 - 2.0 * e1 + e2;
    if denom <= 0.0 {
        // Non-convex (p not a local min) → keep the grid pick.
        return p;
    }
    let delta = 0.5 * (e0 - e2) / denom; // vertex offset in grid steps
    if delta.abs() > 0.5 {
        return p;
    }
    (p + delta * step).clamp(PITCH_GRID_MIN, PITCH_GRID_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
