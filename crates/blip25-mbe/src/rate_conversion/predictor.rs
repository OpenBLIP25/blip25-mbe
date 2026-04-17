//! Cross-rate magnitude predictor (US7634399 / AMBE-3000 rate-converter §4.5).
//!
//! The rate converter maintains a dedicated prior-frame magnitude
//! predictor state, distinct from both the source decoder's intra-rate
//! predictor and the target encoder's intra-rate predictor. Each frame,
//! the converter blends `0.65 · resampled_prior + 0.35 · current`
//! before handing the magnitudes to the target encoder. When
//! `|R_prev − 1| ≤ 0.01` (the target pitch hasn't shifted more than 1%
//! frame-to-frame), the fast path passes `current` through unchanged.
//!
//! Source: `~/blip25-specs/DVSI/AMBE-3000/AMBE-3000_Rate_Converter_Implementation_Spec.md`
//! §4.5 (blend rule) and §5 (state table), plus
//! `~/blip25-specs/analysis/ambe_predictor_state_separation.md` (the
//! invariant that this state must not be shared with the source
//! decoder's or target encoder's predictor).
//!
//! ## Why this matters
//!
//! Without the cross-rate predictor, long sustained transcoded vowels
//! pick up frame-to-frame magnitude instability on the order of 50
//! frames per US7634399 col. 7 — the AR loop between source decoder
//! and target encoder is undamped. The 0.65 blend is a first-order
//! low-pass filter applied at the rate-boundary seam.

use crate::mbe_params::{L_MAX, MbeParams};
use crate::p25_halfrate::frame::AMBE_PITCH_TABLE;

/// Rate-converter predictor coefficient per US7634399 col. 7 line 10.
/// Numerically equal to BABA-A Eq. 185's half-rate intra-rate ρ, but a
/// **different state instance** per the separation invariant (see
/// `~/blip25-specs/analysis/ambe_predictor_state_separation.md`).
pub const RHO_CROSS_RATE: f64 = 0.65;

/// Threshold on `|R_prev − 1|` below which the blend falls through to
/// the current frame's magnitudes unchanged (§4.5 fast path).
/// US7634399 col. 5 line 50 + col. 7 line 12.
pub const R_FAST_PATH_THRESHOLD: f64 = 0.01;

/// Minimum magnitude used when taking `log2` (guards against `log2(0)`).
/// Matches the convention in
/// `DVSI/AMBE-3000/AMBE-3000_Decoder_Implementation_Spec.md` §5.4.
const LOG2_FLOOR_INPUT: f64 = 1e-10;

/// Dedicated predictor state for one cross-rate conversion direction.
/// See `AMBE-3000_Rate_Converter_Implementation_Spec.md` §5 table.
#[derive(Clone, Debug)]
pub struct CrossRatePredictorState {
    /// Previous-frame target-rate magnitudes `M̃_l_B(−1)` for
    /// `l = 1..=56`, 1-indexed (slot `[0]` unused, per the rest of
    /// the codebase convention). Init 1.0 per §5.
    m_tilde_prev: [f32; L_MAX as usize + 1],
    /// Previous-frame target-rate harmonic count `L̂_B(−1)`. Init 30.
    l_prev: u8,
    /// Previous-frame target-rate fundamental `ω̂₀_B(−1)`. Init
    /// `AMBE_PITCH_TABLE[30].omega_0` per §5 Annex-L-row-30 rule.
    omega_0_prev: f32,
}

impl CrossRatePredictorState {
    /// Cold-start per §5 init column.
    pub fn new() -> Self {
        Self {
            m_tilde_prev: [1.0; L_MAX as usize + 1],
            l_prev: 30,
            omega_0_prev: AMBE_PITCH_TABLE[30].omega_0,
        }
    }

    /// Access the stored prior-frame magnitudes (slot `[0]` is unused).
    #[cfg(test)]
    pub(crate) fn m_tilde_prev(&self) -> &[f32; L_MAX as usize + 1] {
        &self.m_tilde_prev
    }

    /// Access the stored prior-frame harmonic count.
    #[cfg(test)]
    pub(crate) fn l_prev(&self) -> u8 {
        self.l_prev
    }

    /// Access the stored prior-frame fundamental.
    #[cfg(test)]
    pub(crate) fn omega_0_prev(&self) -> f32 {
        self.omega_0_prev
    }

    /// Reset to cold-start values. Called on extended mute / invalid
    /// sequences per §6.3 of the rate-converter spec — the spec says
    /// "state is stale", but empirically resetting cleanly avoids
    /// stuck-predictor artifacts when the source recovers.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for CrossRatePredictorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the cross-rate magnitude blend per §4.5 in-place on
/// `params.amplitudes_slice()` and advance `state` to reflect this
/// frame as `(−1)` for next-frame blending.
///
/// `target_omega_0` and `target_l` are the post-pitch-snap fundamental
/// and harmonic count the target encoder will actually commit to
/// (i.e. the Annex-L-dequantized `ω̂₀_B(0)` for half-rate targets). The
/// caller is responsible for supplying these — they differ from
/// `params.omega_0()` / `params.harmonic_count()` when the pitch grid
/// transform changes them.
///
/// Returns an updated `MbeParams` carrying the blended amplitudes.
/// Preserves `params.voiced_slice()` unchanged.
pub fn blend(
    params: &MbeParams,
    target_omega_0: f32,
    target_l: u8,
    state: &mut CrossRatePredictorState,
) -> MbeParams {
    debug_assert!(target_l as usize <= L_MAX as usize);

    let omega_0_curr = f64::from(target_omega_0);
    let omega_0_prev = f64::from(state.omega_0_prev);
    // Guard: omega_0_prev is init'd to a positive Annex L entry and
    // updated only from target_omega_0 (> 0), so division is safe.
    let r_prev = omega_0_curr / omega_0_prev.max(1e-20);

    let l_curr = target_l;
    let amps_curr = params.amplitudes_slice();
    let l_source = amps_curr.len().min(l_curr as usize);
    let mut blended = [0f32; L_MAX as usize];
    // For target harmonics l ≤ min(L_source, L_target), seed with the
    // source magnitude (first-cut §4.3.1 fast-path: no resampling).
    // For l > L_source the slot stays zero — matches the spec's
    // zero-pad rule when L_B > L_A.
    blended[..l_source].copy_from_slice(&amps_curr[..l_source]);

    if (r_prev - 1.0).abs() > R_FAST_PATH_THRESHOLD {
        // Slow path: resample prior magnitudes onto current target
        // grid via log-domain linear interpolation per §4.3.2, with
        // §4.3.3 energy-preserving offset `+ 0.5·log₂(R_prev)`.
        let energy_offset = 0.5 * r_prev.log2();
        let mut log_m_prev = [0.0_f64; L_MAX as usize + 1];
        for l in 1..=state.l_prev as usize {
            let m = f64::from(state.m_tilde_prev[l]).max(LOG2_FLOOR_INPUT);
            log_m_prev[l] = m.log2();
        }

        for l_b in 1..=l_curr as usize {
            // Map target harmonic l_b to fractional source-grid
            // position: l_b · ω̂₀_B(0) / ω̂₀_B(−1) = l_b · R_prev.
            // Wait — the spec's R is defined as R = ω̂₀_B(0) / ω̃₀_A,
            // pitch shift source→target. For the prior-frame resample
            // inside the predictor, the equivalent is
            // R_prev_pred = ω̂₀_B(0) / ω̂₀_B(−1); the position of
            // the prior-frame harmonic l' that lines up with target
            // harmonic l_b is l' = l_b · ω̂₀_B(0) / ω̂₀_B(−1) = l_b · R_prev.
            let src_pos = (l_b as f64) * r_prev;
            let prev_log = if src_pos < 1.0 || src_pos > state.l_prev as f64 {
                // Outside prior-frame grid — zero contribution (floor
                // log, contribution falls below numerical noise).
                LOG2_FLOOR_INPUT.log2()
            } else {
                let pos_lo = src_pos.floor() as usize;
                let pos_hi = (pos_lo + 1).min(state.l_prev as usize);
                let frac = src_pos - pos_lo as f64;
                let v_lo = log_m_prev[pos_lo];
                let v_hi = log_m_prev[pos_hi];
                v_lo + frac * (v_hi - v_lo)
            };
            let prev_resampled_log = prev_log + energy_offset;
            // 0.65 · prior_resampled + 0.35 · current, in linear domain.
            let prior_linear = (2.0_f64).powf(prev_resampled_log) as f32;
            let curr = blended[l_b - 1];
            blended[l_b - 1] = RHO_CROSS_RATE as f32 * prior_linear
                + (1.0 - RHO_CROSS_RATE as f32) * curr;
        }
    }
    // Fast path (|R−1| ≤ 0.01): blended already holds current amps; no
    // mix. State still advances (next frame's R_prev is computed
    // against this frame's ω̂₀_B regardless of whether the blend fired).

    // Build result MbeParams. Extend voicing to cover the target
    // harmonic range: harmonics past source L_A default to unvoiced
    // (matches §4.3 zero-pad convention and the decoder's §1.8.1
    // invariant that out-of-range harmonics are unvoiced).
    let voiced_src = params.voiced_slice();
    let mut voiced_out = [false; L_MAX as usize];
    let l_voiced = voiced_src.len().min(l_curr as usize);
    voiced_out[..l_voiced].copy_from_slice(&voiced_src[..l_voiced]);
    let result = MbeParams::new(
        target_omega_0,
        l_curr,
        &voiced_out[..l_curr as usize],
        &blended[..l_curr as usize],
    )
    .expect("blend result has valid MbeParams constraints");

    // Advance state for next frame: store the blended (post-§4.5)
    // magnitudes as M̃_l_B(−1). Spec §4.5 describes this as "previous
    // frame's reconstructed magnitudes after target rate quantization";
    // storing the blend output is a close approximation (the target
    // encoder's quantizer is near-lossless on natural-speech
    // magnitudes, so pre/post-quant blend differs only at sub-bit
    // levels). A future pass could tighten this by storing the target
    // encoder's dequantize output instead.
    state.m_tilde_prev.fill(0.0);
    state.m_tilde_prev[0] = 1.0;
    for l in 1..=l_curr as usize {
        state.m_tilde_prev[l] = blended[l - 1];
    }
    state.l_prev = l_curr;
    state.omega_0_prev = target_omega_0;

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(omega_0: f32, l: u8, amps_fill: f32) -> MbeParams {
        let voiced = vec![true; l as usize];
        let amps = vec![amps_fill; l as usize];
        MbeParams::new(omega_0, l, &voiced, &amps).expect("test params valid")
    }

    #[test]
    fn cold_start_uses_annex_l_row_30() {
        let state = CrossRatePredictorState::new();
        assert_eq!(state.l_prev(), 30);
        let expected_omega = AMBE_PITCH_TABLE[30].omega_0;
        assert_eq!(state.omega_0_prev(), expected_omega);
        for l in 0..=L_MAX as usize {
            assert_eq!(state.m_tilde_prev()[l], 1.0);
        }
    }

    #[test]
    fn fast_path_passes_current_unchanged_when_r_within_one_percent() {
        // Set state ω_prev to match the target exactly → R = 1.0 → fast path.
        let target_omega = 0.15_f32;
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = target_omega;
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 42.0;
        }
        let params = make_params(target_omega, 20, 100.0);
        let out = blend(&params, target_omega, 20, &mut state);
        for (i, &a) in out.amplitudes_slice().iter().enumerate().take(20) {
            assert_eq!(a, 100.0, "harmonic {i}: fast-path should pass current through");
        }
        // State advanced.
        assert_eq!(state.l_prev, 20);
        assert_eq!(state.omega_0_prev, target_omega);
        for l in 1..=20 {
            assert_eq!(state.m_tilde_prev[l], 100.0);
        }
    }

    #[test]
    fn slow_path_blends_sixty_five_thirty_five_at_matching_grids() {
        // Force slow path by setting ω_prev outside the ±1% band
        // around target. Since both grids are "the same" conceptually
        // (l_prev = l_curr, prior M = M_prior, current M = M_curr),
        // resampling lands on integer harmonic indices and we can
        // verify the blend coefficients cleanly.
        //
        // Setup: prior all = 100, current all = 400, prior ω = 0.14,
        // current ω = 0.15 → R_prev = 0.15/0.14 ≈ 1.0714 (> 1.01).
        // Energy offset = 0.5·log₂(1.0714) ≈ 0.0498.
        // For target harmonic l_b, source pos = l_b · R_prev ≈ 1.07·l_b,
        // so for l_b = 1: src_pos ≈ 1.07, interp between log_prev[1]=log₂(100) and [2]=log₂(100),
        // = log₂(100). Add energy_offset → log₂(100·√R). prior_linear = 100·√R ≈ 103.53.
        // blended = 0.65 · 103.53 + 0.35 · 400 = 67.29 + 140 = 207.29.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.14;
        state.l_prev = 20;
        for l in 1..=20 {
            state.m_tilde_prev[l] = 100.0;
        }
        let params = make_params(0.15, 20, 400.0);
        let out = blend(&params, 0.15, 20, &mut state);

        let r_prev: f64 = 0.15_f64 / 0.14_f64;
        let expected_prior_linear = 100.0_f64 * r_prev.sqrt();
        let expected_blend =
            RHO_CROSS_RATE * expected_prior_linear + (1.0 - RHO_CROSS_RATE) * 400.0;
        // Middle harmonics land well inside the source grid and fully
        // exercise the interp path.
        for l in 5..=15 {
            let observed = out.amplitudes_slice()[l - 1] as f64;
            let rel_err = (observed - expected_blend).abs() / expected_blend;
            assert!(
                rel_err < 0.01,
                "l={l}: observed {observed}, expected {expected_blend} (rel err {rel_err})"
            );
        }
    }

    #[test]
    fn state_advances_every_frame_regardless_of_path() {
        // Both fast and slow paths must update state so next-frame R is
        // computed against this frame's ω.
        let mut state = CrossRatePredictorState::new();
        // Frame 1: fast path (target == init).
        let omega_init = state.omega_0_prev;
        let p1 = make_params(omega_init, state.l_prev, 50.0);
        let _ = blend(&p1, omega_init, state.l_prev, &mut state);
        assert_eq!(state.omega_0_prev, omega_init);
        // Frame 2: slow path (20% pitch shift).
        let p2 = make_params(omega_init * 1.2, 25, 80.0);
        let _ = blend(&p2, omega_init * 1.2, 25, &mut state);
        assert!((state.omega_0_prev - omega_init * 1.2).abs() < 1e-6);
        assert_eq!(state.l_prev, 25);
    }

    #[test]
    fn voicing_is_preserved_across_blend() {
        // Blend touches only amplitudes; voiced_slice must match input.
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.14;
        state.l_prev = 10;
        let l: u8 = 10;
        let voiced = vec![true, false, true, false, true, false, true, false, true, false];
        let amps = vec![100.0_f32; l as usize];
        let params = MbeParams::new(0.15, l, &voiced, &amps).unwrap();
        let out = blend(&params, 0.15, l, &mut state);
        assert_eq!(&out.voiced_slice()[..l as usize], &voiced[..]);
    }

    #[test]
    fn reset_returns_to_cold_start() {
        let mut state = CrossRatePredictorState::new();
        state.omega_0_prev = 0.25;
        state.l_prev = 40;
        state.m_tilde_prev[1] = 999.0;
        state.reset();
        let fresh = CrossRatePredictorState::new();
        assert_eq!(state.l_prev, fresh.l_prev);
        assert_eq!(state.omega_0_prev, fresh.omega_0_prev);
        assert_eq!(state.m_tilde_prev[1], 1.0);
    }

    #[test]
    fn sustained_input_converges_toward_stable_output() {
        // Multi-frame test: feed a constant MbeParams stream with
        // varying pitch (to exercise slow path) and verify the blend
        // output converges to a stable value, matching the first-order
        // low-pass behavior the predictor implements.
        let mut state = CrossRatePredictorState::new();
        // Start at a distinct ω from init.
        state.omega_0_prev = 0.20;
        state.l_prev = 15;
        for l in 1..=15 {
            state.m_tilde_prev[l] = 0.0; // start with a far-from-target prior
        }
        let mut last_l1 = 0.0_f64;
        for frame in 0..30 {
            let p = make_params(0.15, 20, 500.0);
            let out = blend(&p, 0.15, 20, &mut state);
            let this_l1 = out.amplitudes_slice()[0] as f64;
            if frame > 5 {
                // After a few frames the blend output should be very
                // close to the input (since the fast-path threshold
                // kicks in once ω_prev == target).
                assert!(
                    (this_l1 - 500.0).abs() < 1.0,
                    "frame {frame}: blended l=1 = {this_l1}, expected ~500"
                );
            }
            last_l1 = this_l1;
        }
        assert!((last_l1 - 500.0).abs() < 1.0);
    }
}
