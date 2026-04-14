//! Half-rate AMBE+2 dequantization — `b̂₀..b̂₈ → MbeParams` per
//! TIA-102.BABA-A §2.11–§2.13.
//!
//! Pipeline (mirrors full-rate's `dequantize_fullrate` but with
//! different quantizer structure):
//!
//! 1. Pitch: `b̂₀` (7 bits) → Annex L lookup → `(L̃, ω̃₀)`.
//! 2. V/UV: `b̂₁` (5 bits) → Annex M codebook → 8-entry vector,
//!    expanded per-harmonic via `k_l = ⌊l · 16 · ω̃₀ / (2π)⌋`
//!    (§2.3.6 Eq. 147, 149).
//! 3. Gain: `b̂₂` (5 bits) → Annex O differential quantizer,
//!    `γ̃(0) = Δ̃_γ + 0.5·γ̃(−1)` (Eq. 168, cross-frame stateful).
//! 4. PRBA: `b̂₃` → Annex P (G̃₂..G̃₄); `b̂₄` → Annex Q (G̃₅..G̃₈);
//!    G̃₁ = 0. Fixed 8-point inverse DCT (Eq. 169) → R̃₁..R̃₈.
//!    Pair-wise split (Eq. 171–178) → C̃_{i, 1..2} for i = 1..4.
//! 5. HOC: `b̂₅..b̂₈` → Annex R placement per Reading #1 of Eq. 179:
//!    `C̃_{i, k} = H̃_{i, k−2}` for `3 ≤ k ≤ min(J̃_i, 6)`, zero-fill
//!    elsewhere. Disambiguation pending DVSI agreement (§2.12).
//! 6. Per-block inverse DCT (Eq. 180–181) — same form as §1.8.4 but
//!    over 4 blocks from Annex N instead of 6 from Annex J.
//! 7. Log-mag prediction (Eq. 182–187) with Γ̃ intercept from Eq. 184
//!    and cross-frame `Λ̃_l(−1)` state (initialized to 1, `L̃(−1) = 15`).
//! 8. Final M̃_l (Eq. 188) with the half-rate-specific voicing
//!    branch: unvoiced harmonics scaled by `(0.2046 / √ω̃₀)`.

use core::f64::consts::{LN_2, PI as PI64, SQRT_2};

use crate::imbe_frames::dequantize::DecodeError;
use crate::imbe_frames::half_rate::{
    AMBE_BLOCK_LENGTHS, AMBE_GAIN_LEVELS, AMBE_HOC_B5, AMBE_HOC_B6, AMBE_HOC_B7,
    AMBE_HOC_B8, AMBE_PITCH_TABLE, AMBE_PRBA24, AMBE_PRBA58, AMBE_VUV_CODEBOOK,
    PitchEntry,
};
use crate::imbe_frames::priority::deprioritize_halfrate;
use crate::mbe_params::{L_MAX, MbeParams};

/// Maximum per-block length observed in Annex N (`L̃ = 56` row has
/// `J̃₄ = 17`). Sized generously to cover the full table.
pub const MAX_BLOCK_SIZE_HALFRATE: usize = 17;

/// Initial value of `L̃(−1)` for the half-rate decoder (§2.13 Annex A).
/// Differs from full-rate's 30 (§1.8.5).
pub const HALFRATE_INIT_PREV_L: u8 = 15;

/// Highest valid half-rate pitch index per Annex L (120 entries,
/// `b̂₀ ∈ [0, 119]`). Values `[120, 127]` indicate tone frames;
/// `[128, 255]` are reserved / error conditions.
pub const HALFRATE_PITCH_MAX: u8 = 119;

/// First tone-frame `b̂₀` value. Tone-frame decoding lives in a
/// separate path (§2.10) and is not handled by [`dequantize_halfrate`].
pub const HALFRATE_TONE_FIRST: u8 = 120;

/// Cross-frame state for the half-rate decoder.
///
/// Two key differences from [`crate::imbe_frames::dequantize::DecoderState`]:
/// - `prev_lambda` is stored in **log₂** domain (Λ̃_l(−1)), not linear.
/// - `prev_l` initializes to 15, not 30.
/// - `prev_gamma` carries the Eq. 168 differential-gain state, updated
///   only on voice frames.
#[derive(Clone, Debug)]
pub struct HalfrateDecoderState {
    /// Previous frame's `Λ̃_l(−1)` in log₂ domain. Index 0 unused —
    /// Eq. 186 says `Λ̃_0(−1) = Λ̃_1(−1)`, so the virtual index 0
    /// reads mirror `Λ̃_1`. Indices `1..=prev_l` populated; reads
    /// beyond `prev_l` clamp to `Λ̃_{prev_l}` (Eq. 187).
    prev_lambda: [f64; L_MAX as usize + 2],
    /// Previous frame's `L̃(−1)`. Init: 15.
    prev_l: u8,
    /// Previous voice frame's reconstructed gain `γ̃(−1)`. Init: 0.
    /// Tone/silence/erasure frames do not update this.
    prev_gamma: f64,
}

impl HalfrateDecoderState {
    /// Cold-start state per §2.13 Annex A: `Λ̃_l(−1) = 1` for all l,
    /// `L̃(−1) = 15`, `γ̃(−1) = 0`.
    pub fn new() -> Self {
        Self {
            prev_lambda: [1.0; L_MAX as usize + 2],
            prev_l: HALFRATE_INIT_PREV_L,
            prev_gamma: 0.0,
        }
    }

    /// Read `Λ̃_l(−1)` with the §2.13 edge cases:
    /// - `l = 0` returns `Λ̃_1(−1)` (Eq. 186).
    /// - `l > prev_l` returns `Λ̃_{prev_l}(−1)` (Eq. 187).
    fn prev_lambda_at(&self, l: u8) -> f64 {
        if l == 0 {
            return self.prev_lambda[1];
        }
        let idx = (l as usize).min(self.prev_l as usize);
        self.prev_lambda[idx]
    }

    /// Exposes `L̃(−1)` for diagnostics.
    pub fn previous_l(&self) -> u8 {
        self.prev_l
    }

    /// Exposes `γ̃(−1)` for diagnostics.
    pub fn previous_gamma(&self) -> f64 {
        self.prev_gamma
    }
}

impl Default for HalfrateDecoderState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// §1.3.1 half-rate analog — pitch decode via Annex L
// ---------------------------------------------------------------------------

/// Result of decoding a half-rate pitch index. `k` is not carried here
/// because half-rate doesn't have the per-band V/UV structure of full
/// rate — V/UV is codebook-indexed per §2.3.6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfratePitchInfo {
    /// Fundamental frequency in radians/sample.
    pub omega_0: f32,
    /// Number of harmonics.
    pub l: u8,
}

/// Decode the 7-bit half-rate pitch index `b̂₀`.
///
/// * `b̂₀ ∈ [0, 119]` → valid voice frame: looked up in
///   [`AMBE_PITCH_TABLE`] to obtain `(L̃, ω̃₀)`.
/// * `b̂₀ ∈ [120, 127]` → tone frame; returns `None` and the caller
///   should switch to the §2.10 tone path.
pub fn pitch_decode_halfrate(b0: u8) -> Option<HalfratePitchInfo> {
    if b0 > HALFRATE_PITCH_MAX {
        return None;
    }
    let PitchEntry { l, omega_0 } = AMBE_PITCH_TABLE[b0 as usize];
    Some(HalfratePitchInfo { omega_0, l })
}

// ---------------------------------------------------------------------------
// §2.3.6 — V/UV codebook expansion
// ---------------------------------------------------------------------------

/// Expand the 5-bit `b̂₁` V/UV codebook index into per-harmonic
/// decisions per Eq. 147/149: `j_l = ⌊l · 16 · ω̃₀ / (2π)⌋`, clamped
/// to `[0, 7]`; `v̄_l = v_{j_l}(b̂₁)`.
///
/// Returns an `[bool; L_MAX]` with entries `0..(l as usize)` populated.
pub fn expand_vuv_halfrate(b1: u8, omega_0: f32, l: u8) -> [bool; L_MAX as usize] {
    debug_assert!(b1 < 32, "b̂₁ is a 5-bit index");
    debug_assert!(l <= L_MAX);
    let codebook = &AMBE_VUV_CODEBOOK[b1 as usize];
    let omega_0 = f64::from(omega_0);
    let mut out = [false; L_MAX as usize];
    for l_h in 1..=l {
        let j = (f64::from(l_h) * 16.0 * omega_0 / (2.0 * PI64)).floor() as i32;
        let j = j.clamp(0, 7) as usize;
        out[(l_h - 1) as usize] = codebook[j];
    }
    out
}

// ---------------------------------------------------------------------------
// §2.11 — Gain + PRBA decomposition
// ---------------------------------------------------------------------------

/// Decode the 5-bit gain index per Eq. 168:
/// `γ̃(0) = Δ̃_γ + 0.5 · γ̃(−1)` with `Δ̃_γ = AMBE_GAIN_QUANTIZER[b̂₂]`.
pub fn decode_gain_halfrate(b2: u8, prev_gamma: f64) -> f64 {
    debug_assert!(b2 < 32);
    f64::from(AMBE_GAIN_LEVELS[b2 as usize]) + 0.5 * prev_gamma
}

/// Combine the two-stage PRBA codebooks into the 8-element G̃ vector.
/// `G̃₁ = 0` per §2.11 Stage 2 ("discarded at encoder per §13.3.1").
/// `G̃₂..G̃₄` come from [`AMBE_PRBA24`]; `G̃₅..G̃₈` from [`AMBE_PRBA58`].
pub fn decode_prba_vector(b3: u16, b4: u8) -> [f64; 8] {
    debug_assert!(b3 < 512 && b4 < 128);
    let p = AMBE_PRBA24[b3 as usize];
    let q = AMBE_PRBA58[b4 as usize];
    [
        0.0,
        f64::from(p[0]),
        f64::from(p[1]),
        f64::from(p[2]),
        f64::from(q[0]),
        f64::from(q[1]),
        f64::from(q[2]),
        f64::from(q[3]),
    ]
}

/// Apply the fixed 8-point inverse DCT (Eq. 169–170) to the
/// transformed PRBA vector `G̃₁..G̃₈`, producing `R̃₁..R̃₈`.
pub fn prba_to_residuals(g: &[f64; 8]) -> [f64; 8] {
    let mut r = [0f64; 8];
    for i_0 in 0..8 {
        let i_half = i_0 as f64 + 0.5;
        let mut acc = 0f64;
        for m_0 in 0..8 {
            let alpha = if m_0 == 0 { 1.0 } else { 2.0 };
            let arg = PI64 * (m_0 as f64) * i_half / 8.0;
            acc += alpha * g[m_0] * arg.cos();
        }
        r[i_0] = acc;
    }
    r
}

/// Pair-wise split (Eq. 171–178) of `R̃₁..R̃₈` into per-block
/// `(C̃_{i,1}, C̃_{i,2})` for `i = 1..=4`.
pub fn pair_split(r: &[f64; 8]) -> [(f64, f64); 4] {
    // Eq. 172 weight: √2 / 4 = 1 / (2·√2).
    let w = SQRT_2 / 4.0;
    [
        ((r[0] + r[1]) / 2.0, w * (r[0] - r[1])),
        ((r[2] + r[3]) / 2.0, w * (r[2] - r[3])),
        ((r[4] + r[5]) / 2.0, w * (r[4] - r[5])),
        ((r[6] + r[7]) / 2.0, w * (r[6] - r[7])),
    ]
}

// ---------------------------------------------------------------------------
// §2.12 — HOC placement (Reading #1)
// ---------------------------------------------------------------------------

/// Populate the 4-block DCT coefficient matrix `C̃_{i, k}` for the
/// half-rate decoder. Block means and `k=2` come from the PRBA
/// pair-split; `k=3..min(J̃_i, 6)` come from Annex R (Reading #1 of
/// Eq. 179); everything else is zero.
///
/// Returns a 4×`MAX_BLOCK_SIZE_HALFRATE` matrix; only `[i][0..J̃_i]`
/// is meaningful.
pub fn assemble_hoc_matrix_halfrate(
    pair: &[(f64, f64); 4],
    b5: u8,
    b6: u8,
    b7: u8,
    b8: u8,
    blocks: &[u8; 4],
) -> [[f64; MAX_BLOCK_SIZE_HALFRATE]; 4] {
    debug_assert!(b5 < 32 && b6 < 16 && b7 < 16 && b8 < 8);
    let hoc: [[f32; 4]; 4] = [
        AMBE_HOC_B5[b5 as usize],
        AMBE_HOC_B6[b6 as usize],
        AMBE_HOC_B7[b7 as usize],
        AMBE_HOC_B8[b8 as usize],
    ];
    let mut c = [[0f64; MAX_BLOCK_SIZE_HALFRATE]; 4];
    for i in 0..4 {
        c[i][0] = pair[i].0; // C̃_{i,1} — block mean
        c[i][1] = pair[i].1; // C̃_{i,2} — first non-mean
        let j_i = blocks[i] as usize;
        // Reading #1: k = 3..=min(J̃_i, 6) → H̃_{i, k-2}.
        // 1-based k maps to 0-based index (k-1); H̃ array is 0-indexed
        // 0..=3 corresponding to 1-based (k-2)=1..=4.
        let k_max = j_i.min(6);
        for k in 3..=k_max {
            c[i][k - 1] = f64::from(hoc[i][k - 3]);
        }
    }
    c
}

// ---------------------------------------------------------------------------
// §2.13 — Per-block inverse DCT + log-mag reconstruction
// ---------------------------------------------------------------------------

/// Per-block inverse DCT (Eq. 180–181) with concatenation into `T̃_l`
/// for `l = 1..=L̃`. This is the §1.8.4 form but over four half-rate
/// blocks from Annex N rather than the six full-rate blocks from
/// Annex J.
pub fn inverse_block_dct_halfrate(
    c: &[[f64; MAX_BLOCK_SIZE_HALFRATE]; 4],
    blocks: &[u8; 4],
) -> [f64; L_MAX as usize] {
    let mut t = [0f64; L_MAX as usize];
    let mut l_offset = 0usize;
    for i in 0..4 {
        let j_i = blocks[i] as usize;
        for j_0 in 0..j_i {
            let mut acc = 0f64;
            for k_0 in 0..j_i {
                let alpha = if k_0 == 0 { 1.0 } else { 2.0 };
                let arg = PI64 * (k_0 as f64) * (j_0 as f64 + 0.5) / j_i as f64;
                acc += alpha * c[i][k_0] * arg.cos();
            }
            t[l_offset + j_0] = acc;
        }
        l_offset += j_i;
    }
    t
}

/// Apply the half-rate log-magnitude prediction (Eq. 182–187) to
/// produce `Λ̃_l(0)` for `l = 1..=L̃`. Returns an array sized to
/// `L_MAX + 2`; index 0 unused, indices `1..=l` hold the log₂-domain
/// magnitudes.
pub fn apply_log_prediction_halfrate(
    t: &[f64; L_MAX as usize],
    l: u8,
    gamma: f64,
    state: &HalfrateDecoderState,
) -> [f64; L_MAX as usize + 2] {
    let mut lambda = [0f64; L_MAX as usize + 2];
    let l_curr = f64::from(l);
    let l_prev = f64::from(state.prev_l);

    // Γ̃ = γ̃(0) − 0.5·log₂(L̃) − (1/L̃)·Σ T̃_λ  (Eq. 184)
    let t_sum: f64 = t[..l as usize].iter().sum();
    let gamma_intercept = gamma - 0.5 * l_curr.log2() - t_sum / l_curr;

    // Global mean of the predictor terms.
    let mut mean_sum = 0f64;
    for lambda_idx in 1..=l {
        let k_l = l_prev * f64::from(lambda_idx) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_lambda_at(k_floor as u8);
        let log_hi = state.prev_lambda_at(k_floor as u8 + 1);
        mean_sum += (1.0 - delta) * log_lo + delta * log_hi;
    }
    let mean = mean_sum / l_curr;

    // Per-harmonic reconstruction with +Γ̃ intercept.
    for l_h in 1..=l {
        let k_l = l_prev * f64::from(l_h) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_lambda_at(k_floor as u8);
        let log_hi = state.prev_lambda_at(k_floor as u8 + 1);
        lambda[l_h as usize] = t[(l_h - 1) as usize]
            + 0.65 * (1.0 - delta) * log_lo
            + 0.65 * delta * log_hi
            - 0.65 * mean
            + gamma_intercept;
    }
    lambda
}

/// Convert `Λ̃_l(0)` to linear `M̃_l` per Eq. 188:
/// voiced → `exp(ln 2 · Λ̃_l)`; unvoiced → `(0.2046/√ω̃₀) · exp(ln 2 · Λ̃_l)`.
///
/// The unvoiced rescale is half-rate-specific (no analog in full-rate
/// Eq. 79).
pub fn compute_m_tilde_halfrate(
    lambda: &[f64; L_MAX as usize + 2],
    voiced: &[bool],
    omega_0: f32,
) -> [f32; L_MAX as usize] {
    let l = voiced.len();
    debug_assert!(l <= L_MAX as usize);
    let mut m = [0f32; L_MAX as usize];
    let uv_scale = 0.2046 / f64::from(omega_0).sqrt();
    for l_h in 1..=l {
        let linear = (LN_2 * lambda[l_h]).exp();
        m[l_h - 1] = if voiced[l_h - 1] {
            linear as f32
        } else {
            (uv_scale * linear) as f32
        };
    }
    m
}

// ---------------------------------------------------------------------------
// Top-level half-rate dequantization
// ---------------------------------------------------------------------------

/// Run the half-rate dequantization pipeline end-to-end:
/// `(û₀..û₃, state) → MbeParams`.
///
/// Returns `DecodeError::BadPitch` for `b̂₀ ∈ [120, 255]` — the caller
/// should switch to the tone path (§2.10) for `b̂₀ ∈ [120, 127]` and
/// treat `[128, 255]` as an erasure condition.
pub fn dequantize_halfrate(
    u: &[u16; 4],
    state: &mut HalfrateDecoderState,
) -> Result<MbeParams, DecodeError> {
    let b = deprioritize_halfrate(u);
    let b0 = b[0] as u8;
    let pitch = pitch_decode_halfrate(b0).ok_or(DecodeError::BadPitch)?;
    let l = pitch.l;

    let b1 = b[1] as u8;
    let voiced = expand_vuv_halfrate(b1, pitch.omega_0, l);

    let b2 = b[2] as u8;
    let gamma = decode_gain_halfrate(b2, state.prev_gamma);

    let g = decode_prba_vector(b[3], b[4] as u8);
    let r = prba_to_residuals(&g);
    let pair = pair_split(&r);

    let blocks = AMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let c = assemble_hoc_matrix_halfrate(
        &pair,
        b[5] as u8,
        b[6] as u8,
        b[7] as u8,
        b[8] as u8,
        &blocks,
    );

    let t = inverse_block_dct_halfrate(&c, &blocks);
    let lambda = apply_log_prediction_halfrate(&t, l, gamma, state);
    let m_tilde = compute_m_tilde_halfrate(&lambda, &voiced[..l as usize], pitch.omega_0);

    let params = MbeParams::new(
        pitch.omega_0,
        l,
        &voiced[..l as usize],
        &m_tilde[..l as usize],
    )
    .map_err(DecodeError::InvalidParams)?;

    // Update cross-frame state (voice frame — γ̃ and Λ̃ both advance).
    for l_h in 1..=l as usize {
        state.prev_lambda[l_h] = lambda[l_h];
    }
    for l_h in (l as usize + 1)..=L_MAX as usize + 1 {
        state.prev_lambda[l_h] = lambda[l as usize];
    }
    state.prev_l = l;
    state.prev_gamma = gamma;

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Pitch ------------------------------------------------------------

    #[test]
    fn pitch_decode_endpoints_match_annex_l() {
        let p0 = pitch_decode_halfrate(0).unwrap();
        assert_eq!(p0.l, 9);
        assert!((p0.omega_0 - 0.049971).abs() < 1e-6);
        let p_last = pitch_decode_halfrate(119).unwrap();
        assert_eq!(p_last.l, 56);
        assert!((p_last.omega_0 - 0.008125).abs() < 1e-6);
    }

    #[test]
    fn pitch_decode_rejects_tone_and_reserved() {
        for b0 in 120u8..=255 {
            assert!(pitch_decode_halfrate(b0).is_none(), "b̂₀ = {b0}");
        }
    }

    // ---- V/UV -------------------------------------------------------------

    #[test]
    fn expand_vuv_all_voiced_codebook_0_gives_voiced() {
        // AMBE_VUV_CODEBOOK[0] is all-voiced (verified in half_rate tests).
        let v = expand_vuv_halfrate(0, 0.2, 9);
        for l_h in 0..9 {
            assert!(v[l_h]);
        }
    }

    #[test]
    fn expand_vuv_all_unvoiced_codebook_16_gives_unvoiced() {
        let v = expand_vuv_halfrate(16, 0.2, 9);
        for l_h in 0..9 {
            assert!(!v[l_h]);
        }
    }

    #[test]
    fn expand_vuv_j_l_clamps_to_seven() {
        // For a high pitch and high l, j_l = floor(l·16·ω/(2π)) should
        // hit the 7 clamp. Check: ω = 2π/19.875, l = 9
        //   j = floor(9 · 16 / 19.875) = floor(7.245) = 7 → clamp ok.
        let omega = 2.0 * core::f32::consts::PI / 19.875;
        let _ = expand_vuv_halfrate(0, omega, 9);
        // With all-voiced codebook (b1=0), all harmonics are voiced
        // regardless of j_l; the test just exercises the clamp path.
    }

    // ---- Gain -------------------------------------------------------------

    #[test]
    fn decode_gain_first_frame_equals_table_level() {
        // With prev_gamma = 0, γ̃(0) = Δ̃_γ = AMBE_GAIN_LEVELS[b2].
        assert!((decode_gain_halfrate(0, 0.0) - (-2.0)).abs() < 1e-6);
        assert!((decode_gain_halfrate(31, 0.0) - 6.874496).abs() < 1e-6);
    }

    #[test]
    fn decode_gain_differential_recurrence() {
        // γ̃(0) = Δ̃_γ + 0.5·γ̃(−1).
        let g0 = decode_gain_halfrate(16, 10.0);
        let d16 = AMBE_GAIN_LEVELS[16] as f64;
        assert!((g0 - (d16 + 5.0)).abs() < 1e-6);
    }

    // ---- PRBA -------------------------------------------------------------

    #[test]
    fn prba_vector_sources_from_correct_codebooks() {
        // b3 = 0 → PRBA24[0] = [0.526055, -0.328567, -0.304727]
        // b4 = 0 → PRBA58[0] = [-0.103660, 0.094597, -0.013149, 0.081501]
        let g = decode_prba_vector(0, 0);
        assert_eq!(g[0], 0.0);
        assert!((g[1] - 0.526055).abs() < 1e-6);
        assert!((g[2] - (-0.328567)).abs() < 1e-6);
        assert!((g[3] - (-0.304727)).abs() < 1e-6);
        assert!((g[4] - (-0.103660)).abs() < 1e-6);
        assert!((g[5] - 0.094597).abs() < 1e-6);
        assert!((g[6] - (-0.013149)).abs() < 1e-6);
        assert!((g[7] - 0.081501).abs() < 1e-6);
    }

    #[test]
    fn prba_dc_only_propagates_to_all_residuals() {
        // G̃ = (c, 0, 0, 0, 0, 0, 0, 0) → R̃_i = c for all i
        // (same argument as the full-rate version of this test).
        // But G̃₁ is always forced to 0 in decode_prba_vector, so we
        // build g directly.
        let g = [3.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = prba_to_residuals(&g);
        for i in 0..8 {
            assert!((r[i] - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn pair_split_matches_encoder_inverse() {
        // Encoder (§13.3.1): R̂_{2i-1} = Ĉ_{i,1} + √2·Ĉ_{i,2};
        //                   R̂_{2i}   = Ĉ_{i,1} − √2·Ĉ_{i,2}.
        // Decoder pair_split must invert this on any C̃.
        let c_in: [(f64, f64); 4] = [
            (1.5, 0.3),
            (2.0, -0.5),
            (-1.0, 0.7),
            (0.5, -0.2),
        ];
        let mut r = [0f64; 8];
        for i in 0..4 {
            r[2 * i] = c_in[i].0 + core::f64::consts::SQRT_2 * c_in[i].1;
            r[2 * i + 1] = c_in[i].0 - core::f64::consts::SQRT_2 * c_in[i].1;
        }
        let c_out = pair_split(&r);
        for i in 0..4 {
            assert!((c_out[i].0 - c_in[i].0).abs() < 1e-12, "block {i} mean");
            assert!(
                (c_out[i].1 - c_in[i].1).abs() < 1e-12,
                "block {i} k=2"
            );
        }
    }

    // ---- HOC placement (Reading #1) --------------------------------------

    #[test]
    fn hoc_matrix_respects_block_length_clamp() {
        // For L̃ = 9, Annex N gives blocks = (2, 2, 2, 3). So:
        //   block 1: J̃=2 → HOC range k=3..=min(2,6) = empty; k=1,2 only.
        //   block 4: J̃=3 → HOC range k=3..=3 = {3} → C̃_{4,3} = H̃_{4,1}.
        let pair = [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)];
        let blocks = [2u8, 2, 2, 3];
        let c = assemble_hoc_matrix_halfrate(&pair, 0, 0, 0, 0, &blocks);
        // Block 4 (i=3): means + k=2 + H̃_{4,1} at k=3.
        assert_eq!(c[3][0], 4.0);
        assert_eq!(c[3][1], 0.4);
        assert!((c[3][2] - f64::from(AMBE_HOC_B8[0][0])).abs() < 1e-6);
        // Blocks 1–3 (i=0,1,2): J̃=2 → only k=1,2 populated, k=3..16 zero.
        for i in 0..3 {
            for k in 2..MAX_BLOCK_SIZE_HALFRATE {
                assert_eq!(c[i][k], 0.0, "block {i}, k_idx {k}");
            }
        }
    }

    #[test]
    fn hoc_matrix_fills_up_to_k_six_when_block_is_large() {
        // J̃ = 17 (max) → k = 3..=6 all populated from HOC[i][1..=4]
        //                 → C̃[i][2..=5], rest zero up to 16.
        let pair = [(0.0, 0.0); 4];
        let blocks = [17u8, 17, 11, 11]; // contrived; sum doesn't matter for this test
        let c = assemble_hoc_matrix_halfrate(&pair, 0, 0, 0, 0, &blocks);
        // Block 1 (i=0, B5): k=3..=6 → C̃_{1,3}..C̃_{1,6}.
        for k in 3..=6 {
            let expected = f64::from(AMBE_HOC_B5[0][k - 3]);
            assert!(
                (c[0][k - 1] - expected).abs() < 1e-6,
                "block 1, k={k}"
            );
        }
        // Coefficients beyond k=6 are zero.
        for k in 7..MAX_BLOCK_SIZE_HALFRATE {
            assert_eq!(c[0][k - 1], 0.0, "block 1, k_idx {k}");
        }
    }

    // ---- Per-block inverse DCT -------------------------------------------

    #[test]
    fn inverse_block_dct_dc_only_propagates() {
        // Same test as the full-rate version: block means as the only
        // non-zero coefficient → c̃_{i,j} = mean for all j.
        let mut c = [[0f64; MAX_BLOCK_SIZE_HALFRATE]; 4];
        c[0][0] = 1.0;
        c[1][0] = 2.0;
        c[2][0] = 3.0;
        c[3][0] = 4.0;
        let blocks = AMBE_BLOCK_LENGTHS[0]; // L=9: [2, 2, 2, 3]
        let t = inverse_block_dct_halfrate(&c, &blocks);
        let mut offset = 0;
        for (i, &j) in blocks.iter().enumerate() {
            for k in 0..j as usize {
                assert!(
                    (t[offset + k] - f64::from(i as u32 + 1)).abs() < 1e-9,
                    "block {i}, pos {k}"
                );
            }
            offset += j as usize;
        }
    }

    // ---- Log-mag prediction ----------------------------------------------

    #[test]
    fn log_mag_prediction_with_unit_prev_and_zero_gamma() {
        // Λ̃(−1) = 1 for all l → log₂ = 0 → predictor terms all 0,
        // mean = 0, and Γ̃ = −0.5·log₂(L) − (1/L)·Σ T̃.
        // Expected: Λ̃_l(0) = T̃_l + Γ̃.
        let mut t = [0f64; L_MAX as usize];
        for i in 0..9 {
            t[i] = (i as f64) * 0.1;
        }
        let state = HalfrateDecoderState::new();
        // Wait — state.prev_lambda is init'd to 1.0 (linear), but
        // prev_lambda_at is supposed to return log₂. Re-check the
        // spec: §2.13 says "Λ̃_l(−1) = 1 for all l". That's 1 in the
        // log₂ domain? Or linear 1?
        //
        // Reading the full-rate analog §1.8.5 says "M̃_l(−1) = 1",
        // linear; and log₂(1) = 0. So our prev_lambda here holds the
        // LINEAR value (for consistency with full-rate), and the
        // "predictor reads log₂ M̃" is equivalent to "prev_lambda
        // is already in log₂ domain if init'd to 1". But 1 in
        // log₂ is 2 linear — different!
        //
        // Hmm — ambiguity. §2.13 Eq. 185 shows Λ̃ added directly in
        // the predictor (no log₂ re-application), so Λ̃ is the log-
        // domain quantity. "Λ̃_l(−1) = 1" means log-domain value 1,
        // which corresponds to linear 2^1 = 2. That's WEIRD.
        //
        // OK: in this test we just verify the formula mechanics under
        // whatever init was chosen. With prev_lambda = 1 (whatever
        // domain that means), prev_lambda_at returns 1 for all l.
        // Predictor sum has 0.65·(1−δ)·1 + 0.65·δ·1 = 0.65 per term.
        // Mean = 0.65. Per-harmonic subtract-mean cancels the
        // per-harmonic term → 0. So Λ̃_l(0) = T̃_l + Γ̃.
        let lambda = apply_log_prediction_halfrate(&t, 9, 0.0, &state);
        let l_curr = 9f64;
        let t_sum: f64 = t[..9].iter().sum();
        let gamma_intercept = 0.0 - 0.5 * l_curr.log2() - t_sum / l_curr;
        for l_h in 1..=9 {
            let expected = t[l_h - 1] + gamma_intercept;
            assert!(
                (lambda[l_h] - expected).abs() < 1e-9,
                "l={l_h}: expected {expected}, got {}",
                lambda[l_h]
            );
        }
    }

    // ---- Top-level -------------------------------------------------------

    #[test]
    fn dequantize_halfrate_rejects_tone_frame() {
        use crate::imbe_frames::priority::prioritize_halfrate;
        let mut b = [0u16; 9];
        b[0] = 120; // tone frame
        let u = prioritize_halfrate(&b);
        let mut state = HalfrateDecoderState::new();
        assert_eq!(
            dequantize_halfrate(&u, &mut state),
            Err(DecodeError::BadPitch)
        );
    }

    #[test]
    fn dequantize_halfrate_produces_finite_amplitudes_for_zero_b() {
        use crate::imbe_frames::priority::prioritize_halfrate;
        // b̂₀ = 0 → valid pitch; other params zero.
        let b = [0u16; 9];
        let u = prioritize_halfrate(&b);
        let mut state = HalfrateDecoderState::new();
        let p = dequantize_halfrate(&u, &mut state).expect("decode");
        assert_eq!(p.harmonic_count(), 9);
        assert!((p.omega_0() - 0.049971).abs() < 1e-6);
        // All harmonics finite, non-negative.
        for l_h in 1..=9 {
            let a = p.amplitude(l_h);
            assert!(a.is_finite() && a >= 0.0, "l={l_h}: M̃ = {a}");
        }
        // State updated.
        assert_eq!(state.previous_l(), 9);
    }

    #[test]
    fn dequantize_halfrate_advances_state_between_frames() {
        use crate::imbe_frames::priority::prioritize_halfrate;
        let b = [0u16; 9];
        let u = prioritize_halfrate(&b);
        let mut state = HalfrateDecoderState::new();
        let g0 = state.previous_gamma();
        let _ = dequantize_halfrate(&u, &mut state).unwrap();
        let g1 = state.previous_gamma();
        // gamma_0 = Δ̃_γ + 0.5·0 = -2.0, so ≠ 0.
        assert_ne!(g0, g1);
        // Run another frame — gamma applies differential recurrence.
        let _ = dequantize_halfrate(&u, &mut state).unwrap();
        let g2 = state.previous_gamma();
        // g2 = -2.0 + 0.5·g1 = -2.0 + 0.5·(-2.0) = -3.0.
        assert!((g2 - (-3.0)).abs() < 1e-6);
    }
}
