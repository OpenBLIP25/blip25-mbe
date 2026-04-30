//! PYIN pitch tracker — alternative frontend to §0.3 (Eq. 5–23).
//!
//! Reference: Mauch & Dixon, *PYIN: A Fundamental Frequency Estimator
//! Using Probabilistic Threshold Distributions*, ICASSP 2014.
//!
//! Per-frame core (HMM smoothing layer is intentionally deferred):
//!
//! 1. **Difference function** `d_t(τ) = Σ (x[j] − x[j+τ])²` over a
//!    `W`-sample comparison window.
//! 2. **Cumulative mean normalized difference**
//!    `d'_t(τ) = d_t(τ) / [(1/τ)·Σ_{j=1}^{τ} d_t(j)]` (the YIN
//!    normalization that turns `d` into a `[0, ~1]`-valued reliability
//!    score, lower = more confident).
//! 3. **Threshold array.** Instead of YIN's single threshold, evaluate
//!    a set `{s_1, ..., s_K}` weighted by a Beta(α, β) prior. For each
//!    `s_k` the candidate `τ̂_k` is the smallest local minimum of `d'`
//!    with `d'(τ̂_k) < s_k` (or `argmin d'` as fallback).
//! 4. **Aggregate** the threshold-weighted candidates into a single
//!    distribution over `τ` (sum of prior-weighted impulses) and pick
//!    `argmax`. Parabolic interpolation around the winner gives
//!    sub-sample resolution.
//!
//! Output is mapped to the §0.3 contract: `(p_hat_i, e_p_hat_i)` with
//! `p_hat_i` in samples and `e_p_hat_i` in `[0, 1]` (lower = more
//! confident, matching `PitchSearch::e_of_p`'s convention).
//!
//! The audio input is the raw HPF'd lookahead (no LPF). PYIN's
//! threshold-array aggregation supplies the noise robustness that the
//! §0.3 path gets from its LPF + sub-multiple cascade.

use super::pitch::{PITCH_GRID_MAX, PITCH_GRID_MIN};

/// Comparison-window length `W` (samples). One full frame at 8 kHz; the
/// difference function is `d(τ) = Σ_{j=0}^{W-1} (x[j] - x[j+τ])²`. The
/// caller must supply at least `W + τ_max` samples.
pub const PYIN_WINDOW: usize = 160;

/// Integer-τ grid length (`τ ∈ [τ_min, τ_max]` inclusive, integer step).
/// Sub-sample resolution comes from parabolic interpolation around the
/// argmax — same approach as YIN. Output period is in samples directly,
/// not in `PITCH_GRID_STEP` units.
pub const PYIN_GRID_LEN: usize =
    (PITCH_GRID_MAX as usize) - (PITCH_GRID_MIN as usize) + 1;

/// Threshold array `s_k` (PYIN paper §2.3). 100 evenly spaced values
/// over `(0, 1]`. Mauch & Dixon use `(0, 1]` with `K=100`.
pub const PYIN_NUM_THRESHOLDS: usize = 100;

/// Beta-distribution shape parameters for the threshold prior. Paper's
/// "Beta distribution" recommendation puts most mass near s ≈ α / (α+β)
/// = 0.10. (`α=2, β=18` ⇒ mean 0.10, mode 0.056, peak around the
/// classic YIN threshold of 0.1.)
const PYIN_BETA_ALPHA: f64 = 2.0;
const PYIN_BETA_BETA: f64 = 18.0;

/// `d_t(τ)` floor used to guard the cumulative-mean denominator on
/// pure-silence frames where `d_t(τ)` collapses to zero.
const D_FLOOR: f64 = 1e-12;

/// Confidence floor — `e_p_hat_i = 1.0` indicates the worst (silence).
/// `e_p_hat_i = 0.0` indicates a perfect periodic match.
const E_OF_P_FLOOR: f64 = 0.0;
const E_OF_P_CEIL: f64 = 1.0;

/// Log-pitch transition standard deviation for the HMM smoother
/// (octaves, expressed in `ln(τ)` units). Mauch & Dixon §3 default
/// 0.5 octaves (σ = ln(2)·0.5 ≈ 0.347). 2026-04-28 A/B sweep on
/// {clean, dam, alert, mark, knox_1} found 0.5 best on average across
/// the speech vectors; ¼ octave traded mark gains for alert losses
/// (tonal sweeps need a wider transition window). Retain the paper
/// default; tighter values are a per-vector tuning knob, not a
/// general win.
const HMM_LOG_PITCH_SIGMA: f64 = 0.346_573_590_279_973;

/// Floor for emission probability so a silent / unconfident frame can't
/// drive the running log-likelihood to -∞ and lock state. Each grid
/// point gets at least `EMISSION_FLOOR / PYIN_GRID_LEN` mass.
const EMISSION_FLOOR: f64 = 1.0e-6;

/// Per-frame leak / forgetting factor for the running Viterbi
/// log-likelihood. `α=0.0` is pure additive Viterbi (no forgetting);
/// `α=1.0` resets every frame. Small leak prevents very-confident
/// runs from sticky-locking past genuine onsets.
const HMM_LEAK_ALPHA: f64 = 0.05;

/// Pre-computed Beta(α, β) PDF samples for the threshold array.
/// Computed once at first use; same shape every frame.
fn beta_weights() -> [f64; PYIN_NUM_THRESHOLDS] {
    let mut w = [0.0f64; PYIN_NUM_THRESHOLDS];
    let mut total = 0.0f64;
    for (i, slot) in w.iter_mut().enumerate() {
        // s_k centered in its bin: (k + 0.5) / K ∈ (0, 1).
        let s = (i as f64 + 0.5) / PYIN_NUM_THRESHOLDS as f64;
        // Beta PDF up to a scalar normalization (we normalize below).
        // f(s) ∝ s^(α-1) · (1-s)^(β-1)
        let pdf = s.powf(PYIN_BETA_ALPHA - 1.0) * (1.0 - s).powf(PYIN_BETA_BETA - 1.0);
        *slot = pdf;
        total += pdf;
    }
    if total > 0.0 {
        for v in &mut w {
            *v /= total;
        }
    }
    w
}

/// Compute the YIN difference function `d(τ)` for `τ ∈ [τ_min, τ_max]`
/// over a `W`-sample comparison window starting at `x[0]`. Caller
/// guarantees `x.len() >= W + τ_max`.
fn difference_function(x: &[f64], tau_min: usize, tau_max: usize, w: usize) -> Vec<f64> {
    let n = tau_max - tau_min + 1;
    let mut d = vec![0.0f64; n];
    for (idx, tau) in (tau_min..=tau_max).enumerate() {
        let mut acc = 0.0f64;
        for j in 0..w {
            let diff = x[j] - x[j + tau];
            acc += diff * diff;
        }
        d[idx] = acc;
    }
    d
}

/// Cumulative-mean-normalized difference `d'(τ)` over the `τ ∈
/// [τ_min, τ_max]` slice. The denominator runs over `τ ∈ [1, τ_max]`,
/// so the caller supplies `d_full` covering `τ ∈ [0, τ_max]` (including
/// the `τ=0` slot which is `d(0) = 0`).
///
/// Returns `d_prime[idx]` for `idx ∈ [0, tau_max - tau_min]`,
/// corresponding to `τ ∈ [tau_min, tau_max]`. On near-zero input
/// (silent frames) `d` collapses to ≈0 everywhere and the YIN
/// normalization is undefined; force `d'(τ) = 1.0` in that case to
/// keep callers' confidence floor (`e_p ≈ 1`) sensible.
fn cmndf(d_full: &[f64], tau_min: usize, tau_max: usize) -> Vec<f64> {
    let n = tau_max - tau_min + 1;
    let mut out = vec![1.0f64; n];
    let mut running_sum = 0.0f64;
    // Walk τ = 1..=τ_max, building the running sum; emit d'(τ) when τ
    // is in the output range.
    for tau in 1..=tau_max {
        running_sum += d_full[tau];
        if tau >= tau_min {
            let mean = running_sum / tau as f64;
            // Silence guard: if the running mean is below the d() floor
            // there is no signal energy in the difference function;
            // pin d' at 1.0 so downstream argmax/threshold logic
            // doesn't latch onto numerical noise.
            out[tau - tau_min] = if mean <= D_FLOOR {
                1.0
            } else {
                d_full[tau] / mean
            };
        }
    }
    out
}

/// Smallest τ index in `d_prime` whose value is a local minimum AND
/// strictly below `threshold`. Returns `None` if no such index exists.
fn first_local_min_below(d_prime: &[f64], threshold: f64) -> Option<usize> {
    if d_prime.is_empty() {
        return None;
    }
    // Boundary rule (PYIN paper): allow the first sample as a minimum
    // if d'[0] < d'[1] and d'[0] < threshold. Symmetric for the last.
    if d_prime[0] < threshold && (d_prime.len() == 1 || d_prime[0] < d_prime[1]) {
        return Some(0);
    }
    for i in 1..d_prime.len() - 1 {
        if d_prime[i] < threshold
            && d_prime[i] < d_prime[i - 1]
            && d_prime[i] <= d_prime[i + 1]
        {
            return Some(i);
        }
    }
    if d_prime.len() >= 2
        && d_prime[d_prime.len() - 1] < threshold
        && d_prime[d_prime.len() - 1] < d_prime[d_prime.len() - 2]
    {
        return Some(d_prime.len() - 1);
    }
    None
}

/// Parabolic interpolation around `idx`. Returns the sub-sample offset
/// in *samples* (since the input grid is integer-τ). Boundary indices
/// return `0`.
fn parabolic_offset(grid: &[f64], idx: usize) -> f64 {
    if idx == 0 || idx + 1 >= grid.len() {
        return 0.0;
    }
    let y_m1 = grid[idx - 1];
    let y_0 = grid[idx];
    let y_p1 = grid[idx + 1];
    let denom = y_m1 - 2.0 * y_0 + y_p1;
    if denom.abs() < 1e-18 {
        0.0
    } else {
        let off = 0.5 * (y_m1 - y_p1) / denom;
        off.clamp(-0.5, 0.5)
    }
}

/// Map an integer grid index `i` (offset from `τ_min`) to a pitch
/// period in samples. With integer-τ resolution the offset is added
/// directly — caller may pass `idx + parabolic_off` for sub-sample.
#[inline]
fn grid_index_to_period(i: f64) -> f64 {
    PITCH_GRID_MIN + i
}

/// Per-frame PYIN intermediates: emission distribution `prob[idx]`
/// over `τ ∈ [τ_min, τ_max]` and the underlying `d'(τ)` curve (used
/// for parabolic interp + e_p mapping). The silent-frame argmin
/// fallback is consumed inside the construction loop and not stored.
struct PyinFrame {
    prob: [f64; PYIN_GRID_LEN],
    d_prime: Vec<f64>,
}

/// Compute the per-frame PYIN intermediates for an audio buffer of at
/// least `PYIN_WINDOW + PITCH_GRID_MAX` samples. No smoothing — caller
/// either uses `argmax(prob)` directly (`run_pyin`) or pipes the
/// distribution through HMM Viterbi (`run_pyin_smoothed`).
fn pyin_frame(x: &[f64]) -> PyinFrame {
    let tau_min = PITCH_GRID_MIN as usize; // 21
    let tau_max = PITCH_GRID_MAX as usize; // 122
    let w = PYIN_WINDOW;
    debug_assert!(x.len() >= w + tau_max,
        "pyin input must cover W + τ_max samples ({})", w + tau_max);

    // Step 1 — difference function `d(τ)` for τ ∈ [0, τ_max] (we need
    // the whole prefix for the cumulative mean denominator). Slot τ=0
    // is by definition zero; reuse zero allocations for it.
    let mut d_full = vec![0.0f64; tau_max + 1];
    for tau in 1..=tau_max {
        let mut acc = 0.0f64;
        for j in 0..w {
            let diff = x[j] - x[j + tau];
            acc += diff * diff;
        }
        d_full[tau] = acc;
    }
    let _ = difference_function; // referenced for docs; kept for tests.

    // Step 2 — cumulative-mean-normalized difference on [τ_min, τ_max].
    let d_prime = cmndf(&d_full, tau_min, tau_max);
    debug_assert_eq!(d_prime.len(), PYIN_GRID_LEN);

    // Step 3-4 — threshold-aggregated candidate distribution.
    let weights = beta_weights();
    let mut prob = [0.0f64; PYIN_GRID_LEN];
    let argmin_idx = {
        let mut best = 0usize;
        let mut best_v = d_prime[0];
        for (i, &v) in d_prime.iter().enumerate().skip(1) {
            if v < best_v {
                best_v = v;
                best = i;
            }
        }
        best
    };
    for k in 0..PYIN_NUM_THRESHOLDS {
        let s = (k as f64 + 0.5) / PYIN_NUM_THRESHOLDS as f64;
        let idx = first_local_min_below(&d_prime, s).unwrap_or(argmin_idx);
        prob[idx] += weights[k];
    }

    PyinFrame { prob, d_prime }
}

/// Convert a chosen grid index `best` (with parabolic refinement) into
/// the `(p_hat_i, e_p_hat_i)` shape the §0.3 contract expects.
fn frame_to_pitch(frame: &PyinFrame, best: usize) -> (f64, f64) {
    let off = parabolic_offset(&frame.d_prime, best);
    let p_hat_i = grid_index_to_period(best as f64 + off)
        .clamp(PITCH_GRID_MIN, PITCH_GRID_MAX);
    let e_p_hat_i = frame.d_prime[best].clamp(E_OF_P_FLOOR, E_OF_P_CEIL);
    (p_hat_i, e_p_hat_i)
}

/// Run PYIN per-frame (no temporal smoothing) on a contiguous audio
/// buffer of at least `PYIN_WINDOW + PITCH_GRID_MAX` samples. Returns
/// `(p_hat_i, e_p_hat_i)` matching the §0.3 contract: `p_hat_i` in
/// samples on the half-sample grid, `e_p_hat_i` in `[0, 1]` (lower =
/// more confident).
///
/// On silent / aperiodic input every threshold falls back to `argmin`,
/// which still picks a τ but with `e_p_hat_i` near `1.0` — the §0.8.4
/// silence detector and downstream V/UV gate are responsible for
/// suppressing those.
pub fn run_pyin(x: &[f64]) -> (f64, f64) {
    let frame = pyin_frame(x);

    // Argmax over the aggregated probability mass.
    let best = {
        let mut best = 0usize;
        let mut best_p = frame.prob[0];
        for (i, &p) in frame.prob.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best = i;
            }
        }
        best
    };

    frame_to_pitch(&frame, best)
}

/// Forward-only Viterbi state for the HMM-smoothed PYIN frontend.
/// Holds the previous-frame log-likelihood vector over the
/// `PYIN_GRID_LEN` integer-τ grid; cold-start is uniform.
#[derive(Clone, Debug)]
pub struct PyinHmmState {
    /// `log_alpha[i]` is the running joint log-probability at grid
    /// index `i` (= τ = `PITCH_GRID_MIN + i`).
    log_alpha: [f64; PYIN_GRID_LEN],
    /// `true` once at least one frame has been observed; before that
    /// the smoother degenerates to argmax of the current frame's
    /// emission (matching `run_pyin` for the very first call).
    primed: bool,
}

impl PyinHmmState {
    /// Cold-start state — uniform log-probability over the integer-τ
    /// grid, primed flag clear.
    pub const fn new() -> Self {
        Self {
            log_alpha: [0.0; PYIN_GRID_LEN],
            primed: false,
        }
    }

    /// Reset to cold-start (used by `Vocoder::reset`, etc).
    pub fn reset(&mut self) {
        self.log_alpha = [0.0; PYIN_GRID_LEN];
        self.primed = false;
    }
}

impl Default for PyinHmmState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-computed log-transition cost table. `LOG_TRANS[i][j]` =
/// `-(ln(τ_i) - ln(τ_j))² / (2σ²)` (un-normalized; the normalization
/// is constant across `i,j` and cancels under argmax). Built lazily
/// once per process via `OnceLock`.
fn log_trans_table() -> &'static [[f64; PYIN_GRID_LEN]; PYIN_GRID_LEN] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Box<[[f64; PYIN_GRID_LEN]; PYIN_GRID_LEN]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([[0.0f64; PYIN_GRID_LEN]; PYIN_GRID_LEN]);
        let two_sigma_sq = 2.0 * HMM_LOG_PITCH_SIGMA * HMM_LOG_PITCH_SIGMA;
        for i in 0..PYIN_GRID_LEN {
            let tau_i = PITCH_GRID_MIN + i as f64;
            let log_i = tau_i.ln();
            for j in 0..PYIN_GRID_LEN {
                let tau_j = PITCH_GRID_MIN + j as f64;
                let dlog = log_i - tau_j.ln();
                t[i][j] = -(dlog * dlog) / two_sigma_sq;
            }
        }
        t
    })
}

/// HMM-smoothed PYIN. Runs `pyin_frame` on `x`, builds the
/// log-emission vector, runs one forward-Viterbi step against the
/// prior `state.log_alpha` with log-pitch Gaussian transitions, and
/// returns the smoothed `(p_hat_i, e_p_hat_i)`.
///
/// Contract identical to `run_pyin` — the smoothing is purely
/// internal to `state`. Caller must use the same `PyinHmmState`
/// across frames in stream order; resetting it on stream
/// boundaries is the caller's responsibility.
pub fn run_pyin_smoothed(x: &[f64], state: &mut PyinHmmState) -> (f64, f64) {
    let frame = pyin_frame(x);

    // Log-emission. Floor each grid point so a near-zero prob doesn't
    // wipe the running α to -∞.
    let mut log_emit = [0.0f64; PYIN_GRID_LEN];
    let floor = EMISSION_FLOOR / PYIN_GRID_LEN as f64;
    for i in 0..PYIN_GRID_LEN {
        log_emit[i] = (frame.prob[i] + floor).ln();
    }

    if !state.primed {
        // First frame: smoothed = raw emission argmax (matches
        // `run_pyin` for the cold start). Seed α from the current
        // emission.
        state.log_alpha = log_emit;
        state.primed = true;
        let best = argmax_idx(&log_emit);
        return frame_to_pitch(&frame, best);
    }

    // Forward Viterbi step:
    //   log α_t(i) = max_j [ log α_{t-1}(j) + log p(τ_i | τ_j) ]
    //              + log p(obs_t | τ_i)
    // We want the smoothed argmax at frame t and we want to advance
    // α_t to feed the next frame.
    let trans = log_trans_table();
    let mut new_alpha = [0.0f64; PYIN_GRID_LEN];
    let prior = &state.log_alpha;
    let leaked: Vec<f64> = prior.iter().map(|v| v * (1.0 - HMM_LEAK_ALPHA)).collect();
    for i in 0..PYIN_GRID_LEN {
        let row = &trans[i];
        let mut best = f64::NEG_INFINITY;
        for j in 0..PYIN_GRID_LEN {
            let cand = leaked[j] + row[j];
            if cand > best {
                best = cand;
            }
        }
        new_alpha[i] = best + log_emit[i];
    }

    // Re-normalize so log α stays bounded across long streams. Subtract
    // the max (no effect on argmax / future transitions).
    let max_val = new_alpha
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_finite() {
        for v in &mut new_alpha {
            *v -= max_val;
        }
    }

    state.log_alpha = new_alpha;
    let best = argmax_idx(&new_alpha);
    frame_to_pitch(&frame, best)
}

#[inline]
fn argmax_idx(v: &[f64; PYIN_GRID_LEN]) -> usize {
    let mut best = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Synthetic 200 Hz cosine at 8 kHz — period 40 samples. PYIN should
    /// land within ±0.5 samples (one grid unit) and report low e_p.
    #[test]
    fn pyin_locks_pure_cosine_200hz() {
        let n = PYIN_WINDOW + PITCH_GRID_MAX as usize + 10;
        let mut x = vec![0.0f64; n];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (2.0 * PI * 200.0 * (i as f64) / 8000.0).cos();
        }
        let (p, e) = run_pyin(&x);
        assert!((p - 40.0).abs() <= 0.5, "expected period ≈ 40, got {p}");
        assert!(e < 0.05, "expected confident e_p, got {e}");
    }

    #[test]
    fn pyin_locks_pure_cosine_300hz() {
        // 300 Hz → period 26.67 samples. Our half-sample grid → 26.5.
        let n = PYIN_WINDOW + PITCH_GRID_MAX as usize + 10;
        let mut x = vec![0.0f64; n];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (2.0 * PI * 300.0 * (i as f64) / 8000.0).cos();
        }
        let (p, e) = run_pyin(&x);
        assert!((p - 26.67).abs() <= 0.6, "expected period ≈ 26.67, got {p}");
        assert!(e < 0.05, "expected confident e_p, got {e}");
    }

    #[test]
    fn pyin_silence_returns_high_e_p() {
        let x = vec![0.0f64; PYIN_WINDOW + PITCH_GRID_MAX as usize + 10];
        let (_p, e) = run_pyin(&x);
        // Silent input: d(τ) = 0 everywhere, d'(τ) = 0/0 → guarded to
        // d/D_FLOOR which is huge until clamped. After clamp it becomes
        // E_OF_P_CEIL (1.0). The exact value is implementation-defined
        // but should saturate near the ceiling, not the floor.
        assert!(e >= 0.5, "silence should report high e_p; got {e}");
    }

    #[test]
    fn pyin_grid_covers_pitch_range() {
        // PYIN runs on an integer-τ grid (parabolic for sub-sample);
        // the §0.3 path runs on a half-sample grid. The output of
        // both must land in `[PITCH_GRID_MIN, PITCH_GRID_MAX]`, but
        // the internal grid lengths differ.
        assert_eq!(PYIN_GRID_LEN, 102); // 21..=122 inclusive
    }

    /// HMM smoother on a stationary 200 Hz cosine: every frame should
    /// land at τ ≈ 40, e_p ≈ 0. Tests the streaming state machine
    /// runs without diverging.
    #[test]
    fn pyin_smoothed_locks_stationary_cosine() {
        let total = 8 * (PYIN_WINDOW + PITCH_GRID_MAX as usize);
        let mut x = vec![0.0f64; total];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (2.0 * PI * 200.0 * (i as f64) / 8000.0).cos();
        }
        let mut state = PyinHmmState::new();
        let win = PYIN_WINDOW + PITCH_GRID_MAX as usize;
        let hop = 160; // one frame
        let mut last_p = 0.0;
        let mut frames = 0usize;
        let mut start = 0usize;
        while start + win <= x.len() {
            let (p, e) = run_pyin_smoothed(&x[start..start + win], &mut state);
            assert!((p - 40.0).abs() <= 0.5, "frame {frames}: p={p}");
            assert!(e < 0.05, "frame {frames}: e={e}");
            last_p = p;
            start += hop;
            frames += 1;
        }
        assert!(frames >= 2, "need at least 2 frames to exercise the HMM step");
        assert!((last_p - 40.0).abs() <= 0.5);
    }

    /// HMM smoother handles silence → cosine onset without phantom
    /// long-period lock. After a few silence frames, the first
    /// voiced frame should still lock to ≈ 40.
    #[test]
    fn pyin_smoothed_silence_to_voice_onset() {
        let win = PYIN_WINDOW + PITCH_GRID_MAX as usize;
        let n_silent = 3;
        let mut state = PyinHmmState::new();
        let silent = vec![0.0f64; win];
        for _ in 0..n_silent {
            let _ = run_pyin_smoothed(&silent, &mut state);
        }
        // Now the voiced onset.
        let mut x = vec![0.0f64; win];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (2.0 * PI * 200.0 * (i as f64) / 8000.0).cos();
        }
        let (p, e) = run_pyin_smoothed(&x, &mut state);
        // After silent prior, the smoother is leaning on the leak.
        // We tolerate up to ±2 samples for the post-silence frame —
        // the next iteration should snap exactly.
        assert!((p - 40.0).abs() <= 2.0, "post-silence p={p}");
        assert!(e < 0.1, "post-silence e={e}");
    }
}
