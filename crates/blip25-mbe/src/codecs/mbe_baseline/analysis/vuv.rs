//! §0.7 — Voiced/Unvoiced determination.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.7 (BABA-A §5.2
//! Eq. 34–42, Figure 11). Per-band voicing decisions v̂_k ∈ {0, 1} for
//! 1 ≤ k ≤ K̂, where K̂ is derived from L̂ via Eq. 34. Consumes
//! precomputed S_w(m) from §0.2, ω̂_0/L̂/â_l/b̂_l from §0.4, and
//! E(P̂_I) from §0.3.

use super::{packed_index, Complex64, HarmonicBasis, PitchRefinement, DFT_SIZE};

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

/// Maximum extra Θ gain for the `BLIP25_VUV_MXI_GRADE` lever (loudness-
/// graded relaxation on confident-pitch loud frames). HARD BOUND: Θ can
/// grow by at most `1 + VUV_MXI_GRADE_MAX_GAIN` (= 2.25×). The boost is
/// multiply-only and never zeroes Θ, so it can only ADD voiced bands —
/// it can never mute (the fork's unbounded variant full-file muted
/// flat-envelope voicing; this bound is the mitigation). See
/// `QUALITY_FINDINGS.md` §3.3.
const VUV_MXI_GRADE_MAX_GAIN: f64 = 1.25;

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
    /// Eq. 37 pitch/band Θ rolloff coefficient (`BLIP25_VUV_PITCH_COEF`).
    /// Spec value [`THETA_PITCH_BAND_COEF`] (0.3096); the chip behaves as
    /// if this is 0.0 (high bands voiced as tone, not noise — the spec
    /// rolloff systematically under-voices high bands). Opt-in override.
    pitch_band_coef: f64,
    /// Enable the `BLIP25_VUV_MXI_GRADE` loudness-graded Θ relaxation on
    /// confident-pitch (`E(P̂_I) < 0.5`) loud frames. Hard-bounded, never
    /// mutes (see [`VUV_MXI_GRADE_MAX_GAIN`]). Opt-in, default false.
    mxi_grade_enabled: bool,
    /// Sticky-voicing strength `s ∈ [0, 1)`. `0.0` (default) = spec.
    /// When `> 0`, the per-band threshold Θ is biased toward the
    /// previous frame's decision: `Θ·(1+s)` if `v̂_k(−1)=1` (easier to
    /// stay voiced), `Θ·(1−s)` if `v̂_k(−1)=0` (easier to stay
    /// unvoiced) — widening the existing voiced/unvoiced base-threshold
    /// hysteresis. Targets the OTA-measured voicing chatter (V lag-1
    /// autocorr 0.087 vs DVSI 0.205; UV transitions 3.5× too frequent,
    /// Miranda 2026-06-21, `QUALITY_FINDINGS.md` §3.3). The Eq. 37
    /// `E(P̂_I)>0.5` force-unvoice case (Θ=0) is unaffected.
    stickiness: f64,
}

impl VuvState {
    /// Cold-start initialization per §0.7.8: `ξ_max = XI_MAX_FLOOR`,
    /// `v̂_k(−1) = 0` for all `k`, `K̂(−1) = 0`.
    pub const fn cold_start() -> Self {
        Self {
            xi_max: XI_MAX_FLOOR,
            vuv_prev: [0; (K_HAT_MAX + 1) as usize],
            k_prev: 0,
            pitch_band_coef: THETA_PITCH_BAND_COEF,
            mxi_grade_enabled: false,
            stickiness: 0.0,
        }
    }

    /// Set the sticky-voicing strength `s` (clamped to `[0, 0.99]`).
    /// `0.0` (default) restores spec behavior. See [`VuvState::stickiness`].
    pub fn set_stickiness(&mut self, s: f64) {
        self.stickiness = if s.is_finite() {
            s.clamp(0.0, 0.99)
        } else {
            0.0
        };
    }

    /// Current sticky-voicing strength.
    #[inline]
    pub fn stickiness(&self) -> f64 {
        self.stickiness
    }

    /// Override the Eq. 37 pitch/band Θ rolloff coefficient
    /// (`BLIP25_VUV_PITCH_COEF`). Default is the spec value 0.3096; 0.0
    /// disables the high-band rolloff (chip-observed). Non-finite or
    /// negative inputs clamp to 0.0. The `E(P̂_I) > 0.5` force-unvoice
    /// rule is independent and unaffected.
    pub fn set_pitch_band_coef(&mut self, c: f64) {
        self.pitch_band_coef = if c.is_finite() && c >= 0.0 { c } else { 0.0 };
    }

    /// Current Eq. 37 pitch/band Θ rolloff coefficient.
    #[inline]
    pub fn pitch_band_coef(&self) -> f64 {
        self.pitch_band_coef
    }

    /// Enable/disable the `BLIP25_VUV_MXI_GRADE` loudness-graded Θ
    /// relaxation (opt-in, default off). Hard-bounded — can only add
    /// voiced bands on confident loud frames, never mutes.
    pub fn set_mxi_grade(&mut self, on: bool) {
        self.mxi_grade_enabled = on;
    }

    /// Whether the M(ξ) graded Θ relaxation is enabled.
    #[inline]
    pub fn mxi_grade_enabled(&self) -> bool {
        self.mxi_grade_enabled
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
        let n = (k_hat as usize)
            .min(decisions.len())
            .min(K_HAT_MAX as usize);
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

        // Eq. 37: Θ_ξ(k, ω̂_0). The `E(P̂_I) > 0.5` force-unvoice rule
        // (Case 1) is spec-faithful and kept verbatim — disabling it is
        // a measured negative (QUALITY_FINDINGS §3.3).
        let theta = if e_p_hat_i > E_P_POOR_MATCH_THRESHOLD && k >= 2 {
            0.0
        } else {
            let base = if state.vuv_prev[k as usize] == 1 {
                THETA_BASE_VOICED_HISTORY
            } else {
                THETA_BASE_UNVOICED_HISTORY
            };
            // BLIP25_VUV_PITCH_COEF: spec rolloff coefficient 0.3096;
            // 0.0 = chip-observed (no high-band under-voicing).
            let modulation = 1.0 - state.pitch_band_coef * f64::from(k - 1) * omega_hat;
            let theta_base = (base * modulation * m_xi).max(0.0);
            // BLIP25_VUV_MXI_GRADE: on confident-pitch (E(P̂_I) < 0.5)
            // loud (high M(ξ)) frames, relax Θ by up to 2.25× so marginal
            // high bands voice as the chip does. Multiply-only and
            // hard-bounded — never mutes.
            if state.mxi_grade_enabled && e_p_hat_i < E_P_POOR_MATCH_THRESHOLD {
                let grade = (2.0 * (m_xi - 0.5)).clamp(0.0, 1.0);
                theta_base * (1.0 + VUV_MXI_GRADE_MAX_GAIN * grade)
            } else {
                theta_base
            }
        };

        // Sticky V/UV: widen the hysteresis band around the existing
        // voiced/unvoiced base thresholds toward the previous frame's
        // decision, to cut the OTA-measured voicing chatter. `0.0`
        // (default) = spec. The force-unvoice case (Θ=0) is unaffected
        // because `0 · (1±s) = 0`.
        let theta = if state.stickiness > 0.0 {
            if state.vuv_prev[k as usize] == 1 {
                theta * (1.0 + state.stickiness)
            } else {
                theta * (1.0 - state.stickiness)
            }
        } else {
            theta
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

#[cfg(test)]
mod tests {
    use super::super::{refine_pitch, signal_spectrum, L_HAT_MAX, L_HAT_MIN, W_R_HALF};
    use super::*;

    /// Helper: broadband periodic signal for VUV tests (same as §0.4/§0.5).
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
}
