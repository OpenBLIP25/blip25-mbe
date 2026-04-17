//! Full-rate IMBE dequantization — turning the quantized parameter
//! array `b̂₀..b̂_{L+2}` into the codec-agnostic [`MbeParams`].
//!
//! Per TIA-102.BABA-A. This module is the bridge from the wire-side bit
//! layout (already deprioritized via [`super::priority`]) to the
//! synthesis-side parameter model.
//!
//! The dequantization pipeline as defined by BABA-A:
//! 1. Pitch decode (§1.3.1) — `b̂₀ → (ω̃₀, L̃, K̃)` via Eqs. 46–48.
//! 2. V/UV expansion (§1.3.2) — `b̂₁` (K bits) → per-harmonic `ṽ_l`.
//! 3. Gain dequantization (§6.3) — `b̂₂` via Annex E (64-level table).
//! 4. Gain DCT inverse (§6) — `b̂₃..b̂₇` via Annex F step sizes.
//! 5. HOC inverse (§6) — `b̂₈..b̂_{L+1}` via Annex G + final residual bit
//!    `b̂_{L+2}`.
//! 6. Inverse log-magnitude prediction (BABA-A §7) → `M̃_l`.
//!
//! This module currently implements steps 1 and 2. Steps 3–6 land as
//! the Annex E/F/G/J CSVs are wired through `build.rs`.
//!
//! [`MbeParams`]: crate::mbe_params::MbeParams

use core::f32::consts::PI;

use crate::mbe_params::{L_MAX, MbeParams, MbeParamsError};

use super::priority::deprioritize;

include!(concat!(env!("OUT_DIR"), "/annex_e_gain.rs"));

/// One row of Annex F: bit count and uniform-quantizer step size for
/// gain DCT coefficient `b̂_m` (`m ∈ {3..7}`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GainAlloc {
    pub b_m: u8,
    pub delta_m: f32,
}

/// One row of Annex G: bit allocation for one HOC coefficient.
/// `c_i` is the 1-based block index (1..6); `c_k` is the 1-based DCT
/// position inside that block (k ≥ 2 — k=1 is reserved for the block
/// mean, which comes from the gain side via `R̃_i`); `b_m` is the
/// voice-parameter index (8 + offset); `b_m_bits` is `B_m` (0..10).
#[derive(Clone, Copy, Debug)]
pub(crate) struct HocAlloc {
    pub c_i: u8,
    pub c_k: u8,
    pub b_m: u8,
    pub b_m_bits: u8,
}

include!(concat!(env!("OUT_DIR"), "/annex_f_gain_alloc.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_g_hoc_alloc.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_j_blocks.rs"));

/// Highest valid full-rate pitch index per BABA-A §1.3.1: values
/// `b̂₀ ∈ [0, 207]` are transmittable; `[208, 255]` are reserved and
/// indicate either uncorrectable FEC errors or a non-conformant encoder.
pub const PITCH_INDEX_MAX: u8 = 207;

/// Result of decoding a full-rate pitch index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchInfo {
    /// Fundamental frequency in radians/sample (Eq. 46).
    pub omega_0: f32,
    /// Number of harmonics (Eq. 47), clamped to `[L_MIN, L_MAX]`.
    pub l: u8,
    /// Number of V/UV bands (Eq. 48): `floor((L+2)/3)` if `L ≤ 36` else 12.
    pub k: u8,
}

/// Decode the 8-bit full-rate pitch index `b̂₀` into `(ω̃₀, L̃, K̃)`.
///
/// Per BABA-A §6.1 Eqs. 46–48 (forwarded through impl-spec §1.3.1):
///
/// ```text
/// ω̃₀ = 4π / (b̂₀ + 39.5)                                    (Eq. 46)
/// L̃  = floor(0.9254 · floor(π/ω̃₀ + 0.25)), clamped [9, 56]  (Eq. 47)
/// K̃  = floor((L̃+2)/3)  if L̃ ≤ 36, else 12                  (Eq. 48)
/// ```
///
/// Returns `None` for reserved `b̂₀ ∈ [208, 255]`. Callers should treat
/// `None` as a bad-frame condition and trigger frame repeat / mute per
/// BABA-A §4.1.
pub fn pitch_decode(b0: u8) -> Option<PitchInfo> {
    if b0 > PITCH_INDEX_MAX {
        return None;
    }
    let omega_0 = 4.0 * PI / (f32::from(b0) + 39.5);
    let l = MbeParams::harmonic_count_for(omega_0);
    let k = vuv_band_count(l);
    Some(PitchInfo { omega_0, l, k })
}

/// Number of V/UV bands `K̃` for a given harmonic count `L̃` (Eq. 48).
#[inline]
pub const fn vuv_band_count(l: u8) -> u8 {
    if l <= 36 { (l + 2) / 3 } else { 12 }
}

/// Band index `k_l ∈ [1, K]` for harmonic `l ∈ [1, L]` per BABA-A §1.3.2.
///
/// ```text
/// k_l = floor((l+2)/3)  if l ≤ 36, else 12
/// ```
///
/// The result is clamped to `K` so that for `L ≤ 36` (where `K < 12`)
/// the highest band absorbs any harmonic whose computed band exceeds
/// the actual band count.
#[inline]
pub const fn band_for_harmonic(l: u8, k: u8) -> u8 {
    let raw = if l <= 36 { (l + 2) / 3 } else { 12 };
    if raw > k { k } else { raw }
}

// ---------------------------------------------------------------------------
// Gain dequantization (Annex E)
// ---------------------------------------------------------------------------

/// Number of entries in the Annex E gain quantizer (6-bit index → 64 levels).
pub const GAIN_LEVELS: usize = 64;

/// Decode the 6-bit gain index `b̂₂` to the overall log-gain `G̃₁`
/// per BABA-A §6 / impl-spec §12.1.
///
/// `G̃₁` is the first element of the 6-element gain DCT vector and
/// represents the overall block-average log-gain. Subsequent gain DCT
/// coefficients `G̃₂..G̃₆` are reconstructed from `b̂₃..b̂₇` via Annex F
/// step sizes (next pipeline step).
///
/// Panics in debug builds if `b2 >= 64`.
#[inline]
pub fn decode_gain(b2: u8) -> f32 {
    debug_assert!(b2 < 64, "b̂₂ is a 6-bit value");
    IMBE_GAIN_LEVELS[b2 as usize]
}

/// Encode an overall log-gain value `G_1` to the 6-bit Annex E index
/// `b̂₂` (argmin over the table).
///
/// The Annex E table is strictly monotone increasing, so this is a
/// binary search that reports whichever of the two bracketing entries
/// is closest. Ties go to the lower index.
pub fn encode_gain(g1: f32) -> u8 {
    // Saturate at the boundaries.
    if g1 <= IMBE_GAIN_LEVELS[0] {
        return 0;
    }
    if g1 >= IMBE_GAIN_LEVELS[GAIN_LEVELS - 1] {
        return (GAIN_LEVELS - 1) as u8;
    }
    // Find the largest index with level ≤ g1, then compare with the next.
    let mut lo = 0usize;
    let mut hi = GAIN_LEVELS - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if IMBE_GAIN_LEVELS[mid] <= g1 { lo = mid; } else { hi = mid; }
    }
    let lo_dist = (g1 - IMBE_GAIN_LEVELS[lo]).abs();
    let hi_dist = (IMBE_GAIN_LEVELS[hi] - g1).abs();
    if hi_dist < lo_dist { hi as u8 } else { lo as u8 }
}

// ---------------------------------------------------------------------------
// Inverse uniform quantizer (§1.8.2, Eq. 68 / 71)
// ---------------------------------------------------------------------------

/// Midtread inverse uniform quantizer shared by Annex F (gain DCT) and
/// Annex G (HOC) per BABA-A §1.8.2 Eq. 68 / 71:
///
/// ```text
/// value = 0                                       if B̃_m = 0
/// value = Δ̃_m · (b̃_m − 2^{B̃_m−1} + 0.5)         otherwise
/// ```
#[inline]
pub fn decode_uniform(b_m: u16, b_m_bits: u8, delta_m: f32) -> f32 {
    if b_m_bits == 0 {
        return 0.0;
    }
    let half = 1u32 << (b_m_bits - 1);
    delta_m * (f32::from(b_m) - half as f32 + 0.5)
}

/// HOC quantizer step-size table (`Table 3` from §1.8.2). Indexed by
/// `B̃_m ∈ [0, 10]`; entry 0 is unused (returns 0 step size for absent
/// coefficients).
const HOC_STEP: [f32; 11] = [
    0.0, 1.20, 0.85, 0.65, 0.40, 0.28, 0.15, 0.08, 0.04, 0.02, 0.01,
];

/// HOC variance scaling (`Table 4` from §1.8.2). Indexed by DCT
/// position `k`. Entries 0 and 1 are unused (k starts at 2; k=1 is the
/// block mean, sourced from the gain side via `R̃_i`). For `k ≥ 10` the
/// scaling clamps to 0.170 per the Table 4 last row.
const HOC_SIGMA: [f32; 11] = [
    0.0, 0.0, 0.307, 0.241, 0.207, 0.190, 0.179, 0.173, 0.165, 0.170, 0.170,
];

/// HOC step size `Δ̃_m = HOC_STEP[B̃_m] · HOC_SIGMA[k]` per §1.8.2.
/// `B̃_m` clamps to 10; `k` clamps to 10 (Table 4's last row).
#[inline]
pub fn hoc_step_size(b_m_bits: u8, k: u8) -> f32 {
    let step_idx = (b_m_bits as usize).min(10);
    let k_idx = (k as usize).min(10);
    HOC_STEP[step_idx] * HOC_SIGMA[k_idx]
}

// ---------------------------------------------------------------------------
// Gain vector recovery (§1.8.3, Eq. 68/69/70)
// ---------------------------------------------------------------------------

/// Decode the 6-element gain DCT vector `G̃_1..G̃_6` from the quantized
/// parameter array `b̂` and the harmonic count `L̃`.
///
/// `b[2]` (`b̂₂`) is dequantized via Annex E; `b[3]..b[7]` (`b̂₃..b̂₇`)
/// via Eq. 68 with step sizes from Annex F (`Δ̃_m`).
///
/// Output indexed 0..6 corresponds to spec `G̃_1..G̃_6` (1-based).
pub fn decode_gain_dct(b: &[u16; 59], l: u8) -> [f32; 6] {
    debug_assert!((9..=56).contains(&l));
    let l_idx = (l - 9) as usize;
    let mut g = [0f32; 6];
    g[0] = decode_gain((b[2] & 0x3F) as u8); // Annex E
    for m_idx in 0..5 {
        // Spec m = 3..=7 → IMBE_GAIN_ALLOC column 0..=4; b̂_m at b[m].
        let alloc = IMBE_GAIN_ALLOC[l_idx][m_idx];
        g[m_idx + 1] = decode_uniform(b[m_idx + 3], alloc.b_m, alloc.delta_m);
    }
    g
}

/// Reconstruct the 6 per-block prediction-residual means `R̃_1..R̃_6`
/// from the dequantized gain DCT vector via §1.8.3 Eq. 69 (corrected):
///
/// ```text
/// R̃_i = Σ_{m=1}^{6} α(m) · G̃_m · cos[π·(m−1)·(i − 0.5) / 6]
///         for 1 ≤ i ≤ 6
/// α(1) = 1,  α(m) = 2 for m > 1
/// ```
///
/// Eq. 69 as printed in BABA-A page 30 shows `J̃_i` in the denominator,
/// but that is an editorial carry-over from Eq. 73 (HOC inverse DCT,
/// which is genuinely per-block). Eq. 61 (the forward gain DCT) uses
/// `/6`, and round-trip identity at `L̃ = 36` (where `J̃_i = 6` for all
/// `i`) requires the inverse to match. See
/// `analysis/vocoder_decode_disambiguations.md §2` (2026-04-16
/// correction) for the full derivation.
pub fn gain_to_residuals(g: &[f32; 6]) -> [f32; 6] {
    let mut r = [0f32; 6];
    for i_0 in 0..6 {
        let i_half = i_0 as f32 + 0.5;
        let mut acc = 0.0f32;
        for m_0 in 0..6 {
            let alpha = if m_0 == 0 { 1.0 } else { 2.0 };
            let arg = PI * (m_0 as f32) * i_half / 6.0;
            acc += alpha * g[m_0] * arg.cos();
        }
        r[i_0] = acc;
    }
    r
}

// ---------------------------------------------------------------------------
// HOC reconstruction (§1.8.4, Eq. 72/73/74)
// ---------------------------------------------------------------------------

/// Maximum block size: ⌈56 / 6⌉ = 10. Used to size HOC matrix rows.
pub const MAX_BLOCK_SIZE: usize = 10;

/// Assemble the per-block DCT coefficient matrix `C̃_{i,k}` for HOC
/// reconstruction, per BABA-A §1.8 Eq. 67 + 72.
///
/// * `r_i` provides the block means: `C̃_{i,1} = R̃_i` (Eq. 67).
/// * Entries `C̃_{i,k}` for `k ≥ 2` come from `b̂₈..b̂_{L+1}` via the
///   inverse uniform quantizer with HOC step sizes.
/// * The Annex G rows are pre-sorted by `(L, b_m)`, with `b_m` walking
///   8↑; the `(C_i, C_k)` columns route each one to the correct block.
///
/// Output is a 6×`MAX_BLOCK_SIZE` matrix; only `[i][0..J̃_i]` is
/// populated for each block.
pub fn assemble_hoc_matrix(
    b: &[u16; 59],
    l: u8,
    r_i: &[f32; 6],
) -> [[f32; MAX_BLOCK_SIZE]; 6] {
    debug_assert!((9..=56).contains(&l));
    let l_idx = (l - 9) as usize;
    let (off, len) = IMBE_HOC_OFFSETS[l_idx];

    let mut c = [[0f32; MAX_BLOCK_SIZE]; 6];
    // Block means from the gain side (k=1 → 0-based [0]).
    for i in 0..6 {
        c[i][0] = r_i[i];
    }
    // Higher-order coefficients from Annex G.
    for j in 0..len as usize {
        let entry = IMBE_HOC_ENTRIES[off as usize + j];
        let val = decode_uniform(
            b[entry.b_m as usize],
            entry.b_m_bits,
            hoc_step_size(entry.b_m_bits, entry.c_k),
        );
        c[(entry.c_i - 1) as usize][(entry.c_k - 1) as usize] = val;
    }
    c
}

/// Run the per-block inverse DCT over `C̃_{i,k}` and concatenate to
/// produce `T̃_l` for `l = 1..=L̃` per BABA-A §1.8.4 Eq. 73 + 74:
///
/// ```text
/// c̃_{i,j} = Σ_{k=1}^{J̃_i} α(k) · C̃_{i,k} · cos[π·(k−1)·(j − 0.5) / J̃_i]
/// α(1) = 1,  α(k) = 2 for k > 1
/// T̃_l = c̃_{i,j}   where l = j + Σ_{n=1}^{i-1} J̃_n
/// ```
///
/// `T̃_l` is the log₂-domain spectral-amplitude prediction residual for
/// the current frame; it feeds the inverse log-magnitude prediction
/// step (§1.8.5) which is added in the next commit.
///
/// Returns an `[f32; L_MAX]` array; only entries `0..L̃` are populated.
pub fn inverse_block_dct(
    c: &[[f32; MAX_BLOCK_SIZE]; 6],
    blocks: &[u8; 6],
) -> [f32; L_MAX as usize] {
    let mut t = [0f32; L_MAX as usize];
    let mut l_offset = 0usize;
    for i in 0..6 {
        let j_i = blocks[i] as usize;
        for j_0 in 0..j_i {
            let mut acc = 0.0f32;
            for k_0 in 0..j_i {
                let alpha = if k_0 == 0 { 1.0 } else { 2.0 };
                let arg = PI * (k_0 as f32) * (j_0 as f32 + 0.5) / j_i as f32;
                acc += alpha * c[i][k_0] * arg.cos();
            }
            t[l_offset + j_0] = acc;
        }
        l_offset += j_i;
    }
    t
}

// ---------------------------------------------------------------------------
// Inverse log-magnitude prediction (§1.8.5, Eq. 75–79)
// ---------------------------------------------------------------------------

/// Prediction gain `ρ` per BABA-A Eq. 55 / §1.8.5.
///
/// Full-rate ρ is an L̃(0)-dependent schedule (piecewise-linear in the
/// current frame's harmonic count), **not** the constant 0.65 that
/// applies only to half-rate (BABA-A Eq. 200). See spec §1.8.5.
#[inline]
pub fn fullrate_rho(l: u8) -> f32 {
    if l <= 15 {
        0.40
    } else if l <= 24 {
        0.03 * f32::from(l) - 0.05
    } else {
        0.70
    }
}

/// Initial harmonic count for the very first frame after reset
/// (`L̃(−1) = 30`), per §1.8.5 / §10 Annex A.
pub const INIT_PREV_L: u8 = 30;

/// Cross-frame state required by the inverse log-magnitude predictor.
///
/// Per §1.9: this state must be **preserved through frame mute** and
/// **updated on every successful or repeated frame**. After a cold
/// start, all `prev_m_linear` entries are 1.0, so `log₂ M̃_l(−1) = 0`
/// and the predictor's contribution vanishes on frame 0 regardless of
/// the ρ schedule.
#[derive(Clone, Debug)]
pub struct DecoderState {
    /// Previous frame's spectral amplitudes `M̃_l(−1)` in **linear**
    /// units. Index 0 holds the virtual `M̃_0(−1) = 1.0` from Eq. 78;
    /// indices `1..=prev_l` hold the reconstructed amplitudes.
    /// Indices beyond `prev_l` are unused and clamp on read per Eq. 79.
    prev_m_linear: [f32; L_MAX as usize + 2],
    /// Previous frame's harmonic count `L̃(−1)`.
    prev_l: u8,
    /// Previous frame's sync bit (Eq. 80).
    prev_sync: bool,
}

impl DecoderState {
    /// New decoder state with the §1.8.5 / §10 Annex A initialization:
    /// `M̃_l(−1) = 1` for all l, `L̃(−1) = 30`, sync bit = 0.
    pub fn new() -> Self {
        Self {
            prev_m_linear: [1.0; L_MAX as usize + 2],
            prev_l: INIT_PREV_L,
            prev_sync: false,
        }
    }

    /// Construct a DecoderState seeded with explicit `M̃_l(−1)`
    /// amplitudes and harmonic count. `amplitudes` is 0-indexed
    /// (entry `i` → harmonic `l = i + 1`), matching the
    /// [`MbeParams`] / `dequantize` convention. Sync bit starts at 0.
    ///
    /// Used by the analysis encoder's closed-loop matched-decoder
    /// roundtrip (addendum §0.6.6): the encoder syncs a fresh
    /// `DecoderState` from its own `PredictorState`, calls
    /// [`quantize`] + [`reconstruct_amplitudes_from_bits`] with a
    /// snapshot so both sides see identical predictor input.
    pub fn from_amplitudes(amplitudes: &[f32], l_prev: u8) -> Self {
        let mut s = Self::new();
        let n = amplitudes.len().min(l_prev as usize);
        for i in 0..n {
            s.prev_m_linear[i + 1] = amplitudes[i];
        }
        s.prev_l = l_prev;
        s
    }

    /// Read `M̃_l(−1)` per Eqs. 78–79: index 0 returns 1.0 (virtual),
    /// indices > `prev_l` clamp to `M̃_{prev_l}(−1)`.
    fn prev_m_at(&self, l: u8) -> f32 {
        if l == 0 {
            return 1.0; // Eq. 78
        }
        let idx = (l as usize).min(self.prev_l as usize);
        self.prev_m_linear[idx]
    }

    /// Previous harmonic count `L̃(−1)`.
    pub fn previous_l(&self) -> u8 {
        self.prev_l
    }

    /// Previous frame's sync bit.
    pub fn previous_sync(&self) -> bool {
        self.prev_sync
    }
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the inverse log-magnitude predictor (Eq. 75–79) to obtain the
/// log₂-domain spectral amplitudes for the current frame.
///
/// Returns an array sized to `L_MAX + 2`; index 0 is unused, indices
/// `1..=l` hold `log₂ M̃_l(0)`. The exponentiation step is left to the
/// caller so this stage is testable in isolation.
pub fn apply_log_prediction(
    t: &[f32; L_MAX as usize],
    l: u8,
    state: &DecoderState,
) -> [f32; L_MAX as usize + 2] {
    let mut log_m = [0f32; L_MAX as usize + 2];
    let l_curr = f32::from(l);
    let l_prev = f32::from(state.prev_l);

    // Pre-compute the global mean (the "− (ρ/L̃(0)) Σ" term in Eq. 77).
    // Doing it once lets us reuse it across all l_h iterations.
    let mut mean_sum = 0f32;
    for lambda in 1..=l {
        let k_lambda = l_prev * f32::from(lambda) / l_curr;
        let k_floor = k_lambda.floor();
        let delta = k_lambda - k_floor;
        let log_lo = state.prev_m_at(k_floor as u8).log2();
        let log_hi = state.prev_m_at(k_floor as u8 + 1).log2();
        mean_sum += (1.0 - delta) * log_lo + delta * log_hi;
    }
    let mean = mean_sum / l_curr;

    // Per-harmonic reconstruction.
    for l_h in 1..=l {
        let k_l = l_prev * f32::from(l_h) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_m_at(k_floor as u8).log2();
        let log_hi = state.prev_m_at(k_floor as u8 + 1).log2();
        let rho = fullrate_rho(l);
        log_m[l_h as usize] = t[(l_h - 1) as usize]
            + rho * (1.0 - delta) * log_lo
            + rho * delta * log_hi
            - rho * mean;
    }

    log_m
}

// ---------------------------------------------------------------------------
// Frame sync bit (§1.8.6, Eq. 80)
// ---------------------------------------------------------------------------

/// Compute the expected current-frame sync bit per Eq. 80: it toggles
/// each frame. The decoder can compare this to the received bit at
/// `b̂_{L+2}[0]` to detect frame-boundary anomalies.
#[inline]
pub fn expected_sync_bit(prev_sync: bool) -> bool {
    !prev_sync
}

// ---------------------------------------------------------------------------
// Pitch index extraction (§1.4.2 fixed slots)
// ---------------------------------------------------------------------------

/// Extract the 8-bit pitch index `b̂₀` directly from the info vectors,
/// before knowing `L`. Per §1.3.1 robustness note + §1.4.2 fixed
/// placement: `û₀[11..6] = b̂₀[7..2]`, `û₇[2..1] = b̂₀[1..0]`.
///
/// The 6 MSBs of `b̂₀` are protected by `û₀`'s [23,12] Golay; the 2
/// LSBs are uncoded (in `û₇`). This split lets the decoder derive `L`
/// even when `û₇` errors are present.
#[inline]
pub fn extract_pitch_index(u: &[u16; 8]) -> u8 {
    let msbs = ((u[0] >> 6) & 0x3F) as u8; // û₀[11..6] → b̂₀[7..2]
    let lsbs = ((u[7] >> 1) & 0x03) as u8; // û₇[2..1]  → b̂₀[1..0]
    (msbs << 2) | lsbs
}

// ---------------------------------------------------------------------------
// Top-level full-rate dequantization
// ---------------------------------------------------------------------------

/// Errors returned by [`dequantize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// `b̂₀` lies in the reserved range `[208, 255]` — frame should be
    /// muted or repeated per BABA-A §4.1 error logic.
    BadPitch,
    /// Reconstruction produced a non-finite or otherwise invalid
    /// spectral amplitude (typically a sign of FEC corruption).
    InvalidParams(MbeParamsError),
}

/// Run the full-rate dequantization pipeline end-to-end:
/// `(û₀..û₇, state) → MbeParams`.
///
/// Pipeline (BABA-A §1.3, §1.8):
/// 1. Extract `b̂₀` directly from `û₀`/`û₇` fixed slots.
/// 2. Pitch decode → `(ω̃₀, L̃, K̃)`.
/// 3. Deprioritize `û₀..û₇` → `b̂₀..b̂_{L+2}` using the L-indexed map.
/// 4. V/UV expand `b̂₁` → per-harmonic voicing.
/// 5. Gain DCT recover `b̂₂..b̂₇` → `G̃_1..G̃_6` → `R̃_1..R̃_6`.
/// 6. HOC + per-block inverse DCT → `T̃_l`.
/// 7. Inverse log-magnitude prediction → `log₂ M̃_l(0)` → `M̃_l`.
/// 8. Update `state.prev_m_linear`, `state.prev_l`, `state.prev_sync`
///    for the next frame.
/// Matched-decoder reconstruction of `M̃_l(0)` from the raw quantized
/// `b` array produced by [`quantize`]. Runs the inverse pipeline
/// (`decode_gain_dct` → residuals → HOC matrix → inverse block DCT →
/// `apply_log_prediction` → exp2) **without** updating `state`. The
/// caller is responsible for committing the returned amplitudes at
/// the right moment in the closed-loop encoder flow (analysis-side
/// §0.6.6: store as next frame's `M̃_l(−1)` only if the frame was
/// dispatched as voice).
///
/// Unlike [`dequantize`], this does not go through priority-
/// prioritization — the input `b` is the 59-word bitstream as
/// produced directly by [`quantize`].
pub fn reconstruct_amplitudes_from_bits(
    b: &[u16; 59],
    l: u8,
    state: &DecoderState,
) -> [f32; L_MAX as usize] {
    let blocks = IMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let g = decode_gain_dct(b, l);
    let r_i = gain_to_residuals(&g);
    let c = assemble_hoc_matrix(b, l, &r_i);
    let t = inverse_block_dct(&c, &blocks);
    let log_m = apply_log_prediction(&t, l, state);
    let mut out = [0f32; L_MAX as usize];
    for i in 0..l as usize {
        out[i] = log_m[i + 1].exp2();
    }
    out
}

pub fn dequantize(
    u: &[u16; 8],
    state: &mut DecoderState,
) -> Result<MbeParams, DecodeError> {
    let b0 = extract_pitch_index(u);
    let pitch = pitch_decode(b0).ok_or(DecodeError::BadPitch)?;
    let l = pitch.l;
    let k = pitch.k;

    let b = deprioritize(u, l);
    let voiced = expand_vuv(b[1], l, k);
    let g = decode_gain_dct(&b, l);
    let blocks = IMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let r_i = gain_to_residuals(&g);
    let c = assemble_hoc_matrix(&b, l, &r_i);
    let t = inverse_block_dct(&c, &blocks);
    let log_m = apply_log_prediction(&t, l, state);

    let mut amplitudes = [0f32; L_MAX as usize];
    for i in 0..l as usize {
        amplitudes[i] = log_m[i + 1].exp2();
    }

    let params = MbeParams::new(
        pitch.omega_0,
        l,
        &voiced[..l as usize],
        &amplitudes[..l as usize],
    )
    .map_err(DecodeError::InvalidParams)?;

    // Update decoder state for the next frame.
    state.prev_m_linear[0] = 1.0;
    for i in 1..=l as usize {
        state.prev_m_linear[i] = amplitudes[i - 1];
    }
    state.prev_l = l;
    // Eq. 80 sync bit lives at b̂_{L+2}[0].
    state.prev_sync = (b[(l + 2) as usize] & 1) != 0;

    Ok(params)
}

// ===========================================================================
// ENCODE (forward direction) — paired with the decode functions above.
// All references to spec §1.8 use the same equations rearranged.
// ===========================================================================

/// Encode a fundamental frequency `ω̃₀` to the 8-bit pitch index `b̂₀`
/// per BABA-A §6.1 Eq. 45: `b̂₀ = floor(4π/ω̃₀ − 39)`.
///
/// Returns `None` if the result is outside `[0, 207]` — i.e. `ω̃₀` is
/// outside the pitch estimator's valid range
/// `(2π/123.125, 2π/19.875)`.
pub fn encode_pitch(omega_0: f32) -> Option<u8> {
    if !(omega_0 > 0.0 && omega_0 < PI) {
        return None;
    }
    let raw = (4.0 * PI / omega_0 - 39.0).floor() as i32;
    if (0..=PITCH_INDEX_MAX as i32).contains(&raw) {
        Some(raw as u8)
    } else {
        None
    }
}

/// Inverse of [`expand_vuv`]: collapse per-harmonic voicing into the
/// K-bit `b̂₁` band-level word.
///
/// Per §1.3.2, all harmonics in a band share one V/UV decision. The
/// per-harmonic input is expected to be already band-uniform — every
/// harmonic in the same band must agree. We use the first harmonic
/// of each band as the band's V/UV (any harmonic in the band would
/// give the same answer for round-trippable inputs).
pub fn collapse_vuv(voiced: &[bool], k: u8) -> u16 {
    debug_assert!(k > 0 && k <= 12);
    let l = voiced.len() as u8;
    let mut b1 = 0u16;
    for band in 1..=k {
        // First harmonic in the band: smallest `l` with band_for_harmonic(l, k) == band.
        let first = (1..=l).find(|&l_h| band_for_harmonic(l_h, k) == band);
        let v = first.map_or(false, |l_h| voiced[(l_h - 1) as usize]);
        // Band 1 → MSB at bit (k-1); band K → LSB at bit 0.
        if v {
            b1 |= 1u16 << (k - band);
        }
    }
    b1
}

/// Inverse of [`decode_uniform`]: midtread uniform encoder per Eq. 68/71.
///
/// `value = Δ̃_m · (b̃_m − 2^{B̃_m−1} + 0.5)` solved for `b̃_m`:
///
/// ```text
/// b̃_m = round(value / Δ̃_m + 2^{B̃_m−1} − 0.5),  clamped to [0, 2^B̃_m − 1]
/// ```
///
/// For `B̃_m = 0` the coefficient is not transmitted; this returns 0.
#[inline]
pub fn encode_uniform(value: f32, b_m_bits: u8, delta_m: f32) -> u16 {
    if b_m_bits == 0 {
        return 0;
    }
    let half = 1u32 << (b_m_bits - 1);
    let max = (1u32 << b_m_bits) - 1;
    let raw = (value / delta_m + half as f32 - 0.5).round() as i32;
    raw.clamp(0, max as i32) as u16
}

/// Encode a 6-element gain DCT vector `G̃_1..G̃_6` to quantizer indices
/// `b̂₂..b̂₇`. `b̂₂` uses Annex E (argmin via [`encode_gain`]); `b̂₃..b̂₇`
/// use [`encode_uniform`] with Annex F step sizes.
pub fn encode_gain_dct(g: &[f32; 6], l: u8) -> [u16; 6] {
    debug_assert!((9..=56).contains(&l));
    let l_idx = (l - 9) as usize;
    let mut b = [0u16; 6];
    b[0] = u16::from(encode_gain(g[0]));
    for m_idx in 0..5 {
        let alloc = IMBE_GAIN_ALLOC[l_idx][m_idx];
        b[m_idx + 1] = encode_uniform(g[m_idx + 1], alloc.b_m, alloc.delta_m);
    }
    b
}

/// Forward gain DCT — given the 6 per-block prediction-residual means
/// `R̂_1..R̂_6`, compute the gain DCT vector `Ĝ_1..Ĝ_6` per BABA-A §6
/// Eq. 61:
///
/// ```text
/// Ĝ_m = (1/6) · Σ_{i=1}^{6} R̂_i · cos[π·(m−1)·(i−0.5) / 6]
///         for 1 ≤ m ≤ 6
/// ```
///
/// Exact inverse of [`gain_to_residuals`] post-Eq. 69 correction. Both
/// directions use denominator `6`; the forward DCT's `1/6` normalization
/// is what makes the pair self-inverse for 6-element vectors. See
/// `analysis/vocoder_decode_disambiguations.md` §2 (2026-04-16
/// correction) for the derivation.
pub fn residuals_to_gain(r: &[f32; 6]) -> [f32; 6] {
    let mut g = [0f32; 6];
    for m_0 in 0..6 {
        let mut acc = 0f32;
        for i_0 in 0..6 {
            let i_half = i_0 as f32 + 0.5;
            let arg = PI * (m_0 as f32) * i_half / 6.0;
            acc += r[i_0] * arg.cos();
        }
        g[m_0] = acc / 6.0;
    }
    g
}

/// Forward block DCT — pairs with [`inverse_block_dct`] per BABA-A §6
/// Eq. 73 inverse. Uses uniform `1/N` factor across all coefficients
/// (Eq. 61 convention); the `α(k)` weighting that distinguishes DC
/// from the rest lives entirely on the inverse side.
///
/// Run on each block independently to obtain `C̃_{i,k}`.
pub fn forward_block_dct(samples: &[f32], n: usize) -> [f32; MAX_BLOCK_SIZE] {
    let mut c = [0f32; MAX_BLOCK_SIZE];
    for k_0 in 0..n {
        let mut acc = 0f32;
        for j_0 in 0..n {
            let arg = PI * (k_0 as f32) * (j_0 as f32 + 0.5) / n as f32;
            acc += samples[j_0] * arg.cos();
        }
        c[k_0] = acc / n as f32;
    }
    c
}

/// Disassemble the 6-block HOC matrix into the parameter array
/// `b̂₈..b̂_{L+1}` by re-quantizing `C̃_{i,k}` for `k ≥ 2` per Annex G's
/// `(C_i, C_k, b_m, B_m)` schema. Mirror of [`assemble_hoc_matrix`].
///
/// Writes into `b[8..=L+1]`. The block means `C̃_{i,1}` are produced by
/// the gain DCT path and are not encoded here.
pub fn disassemble_hoc_matrix(
    c: &[[f32; MAX_BLOCK_SIZE]; 6],
    l: u8,
    b: &mut [u16; 59],
) {
    debug_assert!((9..=56).contains(&l));
    let l_idx = (l - 9) as usize;
    let (off, len) = IMBE_HOC_OFFSETS[l_idx];
    for j in 0..len as usize {
        let entry = IMBE_HOC_ENTRIES[off as usize + j];
        let val = c[(entry.c_i - 1) as usize][(entry.c_k - 1) as usize];
        b[entry.b_m as usize] = encode_uniform(
            val,
            entry.b_m_bits,
            hoc_step_size(entry.b_m_bits, entry.c_k),
        );
    }
}

/// Forward of [`apply_log_prediction`]: given `log₂ M̃_l(0)` and the
/// previous frame's state, recover the prediction residual `T̃_l` per
/// BABA-A §1.8.5 Eq. 77 rearranged:
///
/// ```text
/// T̃_l = log₂ M̃_l(0)
///       − ρ·(1−δ̃_l)·log₂ M̃_{⌊k̃_l⌋}(−1)
///       − ρ·δ̃_l    ·log₂ M̃_{⌊k̃_l⌋+1}(−1)
///       + (ρ / L̃(0)) · Σ_{λ} […]
/// ```
pub fn forward_log_prediction(
    log_m: &[f32; L_MAX as usize + 2],
    l: u8,
    state: &DecoderState,
) -> [f32; L_MAX as usize] {
    let mut t = [0f32; L_MAX as usize];
    let l_curr = f32::from(l);
    let l_prev = f32::from(state.prev_l);

    let mut mean_sum = 0f32;
    for lambda in 1..=l {
        let k_lambda = l_prev * f32::from(lambda) / l_curr;
        let k_floor = k_lambda.floor();
        let delta = k_lambda - k_floor;
        let log_lo = state.prev_m_at(k_floor as u8).log2();
        let log_hi = state.prev_m_at(k_floor as u8 + 1).log2();
        mean_sum += (1.0 - delta) * log_lo + delta * log_hi;
    }
    let mean = mean_sum / l_curr;

    for l_h in 1..=l {
        let k_l = l_prev * f32::from(l_h) / l_curr;
        let k_floor = k_l.floor();
        let delta = k_l - k_floor;
        let log_lo = state.prev_m_at(k_floor as u8).log2();
        let log_hi = state.prev_m_at(k_floor as u8 + 1).log2();
        let rho = fullrate_rho(l);
        t[(l_h - 1) as usize] = log_m[l_h as usize]
            - rho * (1.0 - delta) * log_lo
            - rho * delta * log_hi
            + rho * mean;
    }

    t
}

/// Errors returned by [`quantize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// `ω₀` is outside the encodable range (`b̂₀` would fall outside
    /// `[0, 207]`).
    PitchOutOfRange,
}

/// Run the full-rate quantization pipeline end-to-end:
/// `(MbeParams, &mut state) → b̂₀..b̂_{L+2}`.
///
/// Pipeline (mirror of [`dequantize`]):
/// 1. Encode pitch → `b̂₀`; derive `K`.
/// 2. Collapse per-harmonic voicing → `b̂₁`.
/// 3. Forward log-magnitude prediction → `T̃_l` (uses `state.prev_l`,
///    `state.prev_m_linear` — must be the previous frame's, not the
///    current).
/// 4. Per-block forward DCT over `T̃` slices → `C̃_{i,k}`.
/// 5. Pull out block means → `R̃_i`; forward DCT to `G̃_1..G̃_6`.
/// 6. Quantize `G̃` → `b̂₂..b̂₇` (Annex E + F).
/// 7. Quantize HOC `C̃_{i,k≥2}` → `b̂₈..b̂_{L+1}` (Annex G).
/// 8. `b̂_{L+2}` is a single sync bit — toggled from `state.prev_sync`.
/// 9. Update `state.prev_m_linear`, `state.prev_l`, `state.prev_sync`.
pub fn quantize(
    params: &MbeParams,
    state: &mut DecoderState,
) -> Result<[u16; 59], EncodeError> {
    let l = params.harmonic_count();
    let omega_0 = params.omega_0();
    let b0 = encode_pitch(omega_0).ok_or(EncodeError::PitchOutOfRange)?;
    let k = vuv_band_count(l);

    let mut b = [0u16; 59];
    b[0] = u16::from(b0);
    b[1] = collapse_vuv(params.voiced_slice(), k);

    // T̃_l = forward log-magnitude prediction of log₂ M̃_l(0).
    let mut log_m = [0f32; L_MAX as usize + 2];
    log_m[0] = 0.0; // log₂(1) virtual M̃_0
    for i in 0..l as usize {
        let m = params.amplitudes_slice()[i];
        // Guard against log2(0): pin to a small value matching MbeParams' invariant.
        log_m[i + 1] = if m > 0.0 { m.log2() } else { f32::NEG_INFINITY };
    }
    let t = forward_log_prediction(&log_m, l, state);

    // Forward block DCT per block → C̃_{i,k}.
    let blocks = IMBE_BLOCK_LENGTHS[(l - 9) as usize];
    let mut c_matrix = [[0f32; MAX_BLOCK_SIZE]; 6];
    let mut l_offset = 0usize;
    for i in 0..6 {
        let j_i = blocks[i] as usize;
        let mut block = [0f32; MAX_BLOCK_SIZE];
        for j in 0..j_i {
            block[j] = t[l_offset + j];
        }
        c_matrix[i] = forward_block_dct(&block, j_i);
        l_offset += j_i;
    }

    // Block means R̃_i = C̃_{i,1} → gain DCT → b̂₂..b̂₇.
    let r_i: [f32; 6] = std::array::from_fn(|i| c_matrix[i][0]);
    let g = residuals_to_gain(&r_i);
    let g_indices = encode_gain_dct(&g, l);
    for m_idx in 0..6 {
        b[m_idx + 2] = g_indices[m_idx];
    }

    // HOC C̃_{i,k≥2} → b̂₈..b̂_{L+1}.
    disassemble_hoc_matrix(&c_matrix, l, &mut b);

    // Sync bit b̂_{L+2}: toggle.
    let sync = !state.prev_sync;
    b[(l + 2) as usize] = u16::from(sync);

    // Update state for the next frame: prev_m = the *quantized*
    // amplitudes of this frame as the decoder would reconstruct them
    // (so encoder and decoder evolve in lockstep). For an exact
    // round-trip-able pipeline we use the input M̃ directly.
    state.prev_m_linear[0] = 1.0;
    for i in 1..=l as usize {
        state.prev_m_linear[i] = params.amplitudes_slice()[i - 1];
    }
    state.prev_l = l;
    state.prev_sync = sync;

    Ok(b)
}

// ---------------------------------------------------------------------------
// V/UV expansion (§1.3.2)
// ---------------------------------------------------------------------------

/// Expand the K-bit voicing word `b̂₁` into per-harmonic V/UV decisions.
///
/// Per BABA-A §1.3.2: each harmonic `l ∈ [1, L]` inherits the V/UV
/// decision of band `k_l` (see [`band_for_harmonic`]). The K bits of
/// `b̂₁` are stored MSB-first (band 1 = bit `K-1`, band K = bit 0)
/// per the §1.4.2 placement order.
///
/// Returns an array sized to `L_MAX`; only entries `0..(l as usize)`
/// are populated, the rest are `false`.
///
/// Panics in debug builds if `l` exceeds `L_MAX` or `k > 12`.
pub fn expand_vuv(b1: u16, l: u8, k: u8) -> [bool; L_MAX as usize] {
    debug_assert!(l <= L_MAX, "L exceeds L_MAX");
    debug_assert!(k > 0 && k <= 12, "K out of range");
    let mut out = [false; L_MAX as usize];
    for harm in 1..=l {
        let band = band_for_harmonic(harm, k); // 1..=K
        let bit_pos = k - band; // band 1 → MSB (bit K-1); band K → LSB (bit 0)
        out[(harm - 1) as usize] = ((b1 >> bit_pos) & 1) == 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Pitch decode ----------------------------------------------------

    #[test]
    fn pitch_decode_reserved_returns_none() {
        for b0 in 208u8..=255 {
            assert_eq!(pitch_decode(b0), None, "b0={b0}");
        }
    }

    #[test]
    fn pitch_decode_b0_zero_gives_l9_k3() {
        let p = pitch_decode(0).unwrap();
        assert_eq!(p.l, 9);
        assert_eq!(p.k, 3);
        // ω̃₀ = 4π / 39.5 ≈ 0.31819
        assert!((p.omega_0 - 4.0 * PI / 39.5).abs() < 1e-6);
    }

    #[test]
    fn pitch_decode_b0_max_gives_l56_k12() {
        let p = pitch_decode(PITCH_INDEX_MAX).unwrap();
        assert_eq!(p.l, 56);
        assert_eq!(p.k, 12);
        // ω̃₀ = 4π / 246.5
        assert!((p.omega_0 - 4.0 * PI / 246.5).abs() < 1e-6);
    }

    #[test]
    fn pitch_decode_omega_strictly_decreasing_in_b0() {
        let mut prev = f32::INFINITY;
        for b0 in 0..=PITCH_INDEX_MAX {
            let p = pitch_decode(b0).unwrap();
            assert!(p.omega_0 < prev, "ω̃₀ not decreasing at b0={b0}");
            prev = p.omega_0;
        }
    }

    #[test]
    fn pitch_decode_l_in_valid_range() {
        for b0 in 0..=PITCH_INDEX_MAX {
            let p = pitch_decode(b0).unwrap();
            assert!((9..=56).contains(&p.l), "L={} out of range at b0={b0}", p.l);
        }
    }

    #[test]
    fn vuv_band_count_at_boundary() {
        // L=36 → K = floor(38/3) = 12
        assert_eq!(vuv_band_count(36), 12);
        // L=37..56 → K = 12 (capped)
        for l in 37..=56 {
            assert_eq!(vuv_band_count(l), 12, "L={l}");
        }
        // L=9 → K=3
        assert_eq!(vuv_band_count(9), 3);
    }

    // ---- Annex F / G / J generated tables --------------------------------

    #[test]
    fn annex_f_table_shape_and_spot_values() {
        // 48 L values × 5 m values (m=3..=7).
        assert_eq!(IMBE_GAIN_ALLOC.len(), 48);
        for row in IMBE_GAIN_ALLOC.iter() {
            assert_eq!(row.len(), 5);
            for entry in row {
                assert!(entry.b_m >= 1 && entry.b_m <= 10);
                assert!(entry.delta_m > 0.0);
            }
        }
        // L=9, m=3 first row from CSV: B_m=10, Delta_m=0.003100.
        let r = IMBE_GAIN_ALLOC[0][0];
        assert_eq!(r.b_m, 10);
        assert!((r.delta_m - 0.003100).abs() < 1e-6);
        // L=9, m=4: B_m=9, Delta_m=0.004020.
        let r = IMBE_GAIN_ALLOC[0][1];
        assert_eq!(r.b_m, 9);
        assert!((r.delta_m - 0.004020).abs() < 1e-6);
    }

    #[test]
    fn annex_g_table_shape_and_offsets() {
        // 1272 entries total; 48 (offset, len) pairs.
        assert_eq!(IMBE_HOC_ENTRIES.len(), 1272);
        assert_eq!(IMBE_HOC_OFFSETS.len(), 48);

        // Per-L̂ length = L̂ − 6.
        for (i, &(_off, len)) in IMBE_HOC_OFFSETS.iter().enumerate() {
            let l = (i + 9) as u32;
            assert_eq!(len as u32, l - 6, "L={l}");
        }
        // Offsets must be contiguous and cover all entries.
        let (last_off, last_len) = IMBE_HOC_OFFSETS[47];
        assert_eq!(last_off + last_len, IMBE_HOC_ENTRIES.len() as u32);

        // Spot: L=9 first row → C_i=4, C_k=2, b_m=8, B_m=9.
        let r = IMBE_HOC_ENTRIES[0];
        assert_eq!((r.c_i, r.c_k, r.b_m, r.b_m_bits), (4, 2, 8, 9));
    }

    #[test]
    fn annex_g_b_m_walks_8_upward_per_l() {
        // Per spec §1.8.2 Eq. 72: the b_m index walks contiguously
        // from 8 upward within each L̂.
        for (l_idx, &(off, len)) in IMBE_HOC_OFFSETS.iter().enumerate() {
            let entries = &IMBE_HOC_ENTRIES[off as usize..(off + len) as usize];
            for (j, e) in entries.iter().enumerate() {
                assert_eq!(
                    e.b_m,
                    8 + j as u8,
                    "L={}: row {j} expected b_m={}, got {}",
                    l_idx + 9,
                    8 + j as u8,
                    e.b_m
                );
            }
        }
    }

    #[test]
    fn annex_j_block_lengths_sum_to_l() {
        for (l_idx, row) in IMBE_BLOCK_LENGTHS.iter().enumerate() {
            let l = (l_idx + 9) as u32;
            let sum: u32 = row.iter().map(|&x| x as u32).sum();
            assert_eq!(sum, l, "L={l}");
            // Eq. 66: non-decreasing.
            for i in 0..5 {
                assert!(row[i] <= row[i + 1], "L={l}: blocks not non-decreasing");
            }
        }
    }

    // ---- Gain dequantization (Annex E) -----------------------------------

    #[test]
    fn gain_table_endpoints_match_spec() {
        // Spec spot values from impl-spec §7.1 / §12.1.
        assert!((decode_gain(0) - (-2.842205)).abs() < 1e-6);
        assert!((decode_gain(63) - 8.695827).abs() < 1e-6);
    }

    #[test]
    fn gain_table_is_strictly_monotone() {
        for i in 0..63 {
            assert!(
                decode_gain(i) < decode_gain(i + 1),
                "gain table not strictly increasing at b̂₂={i}"
            );
        }
    }

    #[test]
    fn gain_encode_decode_roundtrip_on_table_values() {
        // Encoding a table value must return its index (within ties).
        for i in 0..64u8 {
            let g = decode_gain(i);
            assert_eq!(encode_gain(g), i, "roundtrip failed at b̂₂={i}");
        }
    }

    #[test]
    fn gain_encode_saturates_outside_range() {
        assert_eq!(encode_gain(-100.0), 0);
        assert_eq!(encode_gain(100.0), 63);
        assert_eq!(encode_gain(f32::NEG_INFINITY), 0);
        assert_eq!(encode_gain(f32::INFINITY), 63);
    }

    #[test]
    fn gain_encode_picks_nearest_neighbour() {
        // For every adjacent pair, the midpoint should round to one
        // side and a value just past it to the other.
        for i in 0..63u8 {
            let lo = decode_gain(i);
            let hi = decode_gain(i + 1);
            let mid = (lo + hi) / 2.0;
            // Just below the midpoint → the lower bin.
            let just_below = mid - (hi - lo) * 0.01;
            assert_eq!(encode_gain(just_below), i, "below midpoint at i={i}");
            // Just above the midpoint → the upper bin.
            let just_above = mid + (hi - lo) * 0.01;
            assert_eq!(encode_gain(just_above), i + 1, "above midpoint at i={i}");
        }
    }

    // ---- Inverse uniform quantizer (§1.8.2) ------------------------------

    #[test]
    fn decode_uniform_zero_bits_is_zero() {
        for delta in [0.001f32, 1.0, 100.0] {
            assert_eq!(decode_uniform(0, 0, delta), 0.0);
            assert_eq!(decode_uniform(7, 0, delta), 0.0);
        }
    }

    #[test]
    fn decode_uniform_midtread_centre_offset() {
        // For B=4: half = 8. Codes 7 and 8 straddle zero by ±0.5·Δ.
        let delta = 0.0964f32; // matches the spec example
        assert!((decode_uniform(7, 4, delta) - (-0.5 * delta)).abs() < 1e-7);
        assert!((decode_uniform(8, 4, delta) - 0.5 * delta).abs() < 1e-7);
        // Ends of the range:
        assert!((decode_uniform(0, 4, delta) - (-7.5 * delta)).abs() < 1e-7);
        assert!((decode_uniform(15, 4, delta) - 7.5 * delta).abs() < 1e-7);
    }

    #[test]
    fn hoc_step_size_matches_spec_example() {
        // §1.8.2: 4 bits at k=3 → Δ̃ = 0.40 · 0.241 = 0.0964.
        let s = hoc_step_size(4, 3);
        assert!((s - 0.0964).abs() < 1e-6);
    }

    #[test]
    fn hoc_step_size_clamps_high_k() {
        // k ≥ 10 must use the same value as k=10 (Table 4 last row).
        let s10 = hoc_step_size(5, 10);
        for k in 10..=20u8 {
            assert_eq!(hoc_step_size(5, k), s10);
        }
    }

    // ---- Gain vector recovery (§1.8.3) -----------------------------------

    #[test]
    fn decode_gain_dct_only_g1_with_zero_quantizer_indices() {
        // With b̂_3..b̂_7 all set to the midtread centre (2^{B-1}), the
        // decoded G̃_2..G̃_6 should each be +0.5·Δ_m, NOT zero — that's
        // the midtread bias. To get exactly zero across G̃_2..G̃_6 we
        // can't with this quantizer, but we can verify the formula
        // against direct computation.
        let mut b = [0u16; 59];
        b[2] = 31; // middle-ish gain index
        for m_idx in 0..5 {
            // Set b_m to the midtread centre.
            let alloc = IMBE_GAIN_ALLOC[0][m_idx]; // L=9
            b[m_idx + 3] = 1u16 << (alloc.b_m - 1);
        }
        let g = decode_gain_dct(&b, 9);
        assert!((g[0] - decode_gain(31)).abs() < 1e-6);
        for m_idx in 0..5 {
            let alloc = IMBE_GAIN_ALLOC[0][m_idx];
            let expected = alloc.delta_m * 0.5;
            assert!(
                (g[m_idx + 1] - expected).abs() < 1e-6,
                "G̃_{}: expected {expected}, got {}",
                m_idx + 2,
                g[m_idx + 1]
            );
        }
    }

    #[test]
    fn gain_to_residuals_constant_g1_propagates_uniformly() {
        // If only G̃_1 = c (DC term) and the rest are zero, the inverse
        // 6-point DCT must produce R̃_i = c for every i.
        let g = [3.5f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = gain_to_residuals(&g);
        for i in 0..6 {
            assert!((r[i] - 3.5).abs() < 1e-5, "R̃_{}: {}", i + 1, r[i]);
        }
    }

    #[test]
    fn gain_to_residuals_uses_fixed_six_denominator() {
        // Post-2026-04-16 correction: Eq. 69 uses `/6` uniformly, so the
        // output does NOT depend on the per-block lengths (Annex J). With
        // only G̃_3 = 1 and α(3) = 2, the six outputs form a cosine
        // sequence at frequency m=3 over the six i-indices.
        //
        //   arg(i_0) = π · 2 · (i_0 + 0.5) / 6
        //   R̃_{i_0+1} = 2 · cos(arg)
        //
        // Check the full sequence.
        let mut g = [0f32; 6];
        g[2] = 1.0; // G̃_3 only (m_0 = 2)
        let r = gain_to_residuals(&g);
        for i_0 in 0..6usize {
            let expected =
                2.0 * (std::f32::consts::PI * 2.0 * (i_0 as f32 + 0.5) / 6.0).cos();
            assert!(
                (r[i_0] - expected).abs() < 1e-5,
                "R̃_{}: got {}, expected {}",
                i_0 + 1,
                r[i_0],
                expected
            );
        }
    }

    // ---- HOC reconstruction (§1.8.4) -------------------------------------

    #[test]
    fn block_dct_forward_then_inverse_is_identity() {
        for &n in &[1usize, 2, 3, 5, 7, 10] {
            let samples: [f32; MAX_BLOCK_SIZE] = std::array::from_fn(|i| {
                if i < n { (i as f32 + 1.0).sin() * 2.0 } else { 0.0 }
            });
            let c = forward_block_dct(&samples, n);
            // Place c into block 0 of a 6-block matrix; rest empty.
            let mut matrix = [[0f32; MAX_BLOCK_SIZE]; 6];
            matrix[0] = c;
            // Block layout: block 0 takes n samples, blocks 1..5 are 0.
            let mut blocks = [0u8; 6];
            blocks[0] = n as u8;
            let t = inverse_block_dct(&matrix, &blocks);
            for i in 0..n {
                assert!(
                    (t[i] - samples[i]).abs() < 1e-4,
                    "n={n}, i={i}: expected {}, got {}",
                    samples[i],
                    t[i]
                );
            }
        }
    }

    #[test]
    fn assemble_hoc_matrix_block_means_match_r_i() {
        // C̃_{i,1} (k=1, 0-based [0]) must equal R̃_i for every block,
        // regardless of the b array (Annex G entries only fill k ≥ 2).
        let b = [0u16; 59];
        let r = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = assemble_hoc_matrix(&b, 9, &r);
        for i in 0..6 {
            assert_eq!(c[i][0], r[i]);
        }
    }

    #[test]
    fn assemble_hoc_matrix_routes_via_eq_72() {
        // For L=9 the first three Annex G entries are
        // (C_i=4, C_k=2, b_m=8, B_m=9), (5,2,9,8), (6,2,10,7).
        // Setting b[8]=midtread+1, b[9]=midtread+1, b[10]=midtread+1
        // should produce non-zero C̃ at exactly (3,1), (4,1), (5,1)
        // (0-based) and zero elsewhere.
        let mut b = [0u16; 59];
        b[8] = (1u16 << 8) + 1; // B_m=9 → midtread offset, then +1
        b[9] = (1u16 << 7) + 1; // B_m=8
        b[10] = (1u16 << 6) + 1; // B_m=7
        let r = [0f32; 6];
        let c = assemble_hoc_matrix(&b, 9, &r);

        // All block means are zero (r = 0).
        for i in 0..6 { assert_eq!(c[i][0], 0.0); }
        // Non-zero entries at the routed positions only.
        assert_ne!(c[3][1], 0.0, "(C_i=4, C_k=2)");
        assert_ne!(c[4][1], 0.0, "(C_i=5, C_k=2)");
        assert_ne!(c[5][1], 0.0, "(C_i=6, C_k=2)");
        // Untouched cells stay zero.
        assert_eq!(c[0][1], 0.0);
        assert_eq!(c[1][1], 0.0);
        assert_eq!(c[2][1], 0.0);
    }

    // ---- Inverse log-magnitude prediction (§1.8.5) -----------------------

    #[test]
    fn decoder_state_init_matches_spec() {
        let s = DecoderState::new();
        assert_eq!(s.previous_l(), 30);
        assert!(!s.previous_sync());
        // All entries 1.0 (including index 0 which is the virtual M̃_0).
        for i in 0..=L_MAX as usize + 1 {
            assert_eq!(s.prev_m_linear[i], 1.0);
        }
    }

    #[test]
    fn prev_m_at_clamps_for_indices_above_prev_l() {
        // Configure prev_l = 9 with distinct amplitudes; reads beyond
        // index 9 must return prev_m_linear[9].
        let mut s = DecoderState::new();
        s.prev_l = 9;
        for i in 0..=L_MAX as usize + 1 {
            s.prev_m_linear[i] = 0.5; // placeholder
        }
        s.prev_m_linear[0] = 1.0;
        for i in 1..=9 {
            s.prev_m_linear[i] = i as f32; // 1.0, 2.0, …, 9.0
        }
        assert_eq!(s.prev_m_at(0), 1.0);
        assert_eq!(s.prev_m_at(1), 1.0);
        assert_eq!(s.prev_m_at(9), 9.0);
        // Eq. 79: clamp.
        assert_eq!(s.prev_m_at(10), 9.0);
        assert_eq!(s.prev_m_at(56), 9.0);
        assert_eq!(s.prev_m_at(57), 9.0);
    }

    #[test]
    fn log_prediction_with_unit_prev_m_subtracts_to_zero() {
        // prev_m all 1.0 → log₂ = 0 everywhere → predictor terms all 0
        // → log_m[l] = T̃_l directly. Tests that the global mean term
        // and the per-harmonic terms cancel correctly.
        let mut t = [0f32; L_MAX as usize];
        for i in 0..30 {
            t[i] = (i as f32) * 0.1 - 1.5; // varied non-trivial T values
        }
        let state = DecoderState::new();
        let log_m = apply_log_prediction(&t, 30, &state);
        for l in 1..=30usize {
            assert!(
                (log_m[l] - t[l - 1]).abs() < 1e-5,
                "l={l}: expected {}, got {}",
                t[l - 1],
                log_m[l]
            );
        }
    }

    #[test]
    fn log_prediction_constant_prev_amplitude_subtracts_constant() {
        // If all prev M̃ are the same constant C ≠ 1 (so log₂C ≠ 0):
        //   per-harmonic prediction term = ρ · log₂C
        //   global mean term = ρ · log₂C
        //   net contribution = 0
        // Result: log_m[l] = T̃_l (same as unit case).
        let mut state = DecoderState::new();
        state.prev_l = 20;
        for i in 0..=L_MAX as usize + 1 {
            state.prev_m_linear[i] = 4.0;
        }
        state.prev_m_linear[0] = 1.0; // M̃_0 is always 1
        let mut t = [0f32; L_MAX as usize];
        for i in 0..15 { t[i] = 0.5; }
        let log_m = apply_log_prediction(&t, 15, &state);
        // Per-harmonic prediction term uses M̃_l(−1) for l>=1 (= 4.0).
        // Mean term averages the same 4.0 values.
        // The virtual M̃_0 = 1 introduces a small skew at l with
        // ⌊k_l⌋ = 0, but for l_curr=15 and l_prev=20 the smallest
        // k_l = 20·1/15 ≈ 1.33, so floor is 1 (not 0). No skew.
        for l in 1..=15 {
            assert!(
                (log_m[l] - t[l - 1]).abs() < 1e-5,
                "l={l}: expected {}, got {}",
                t[l - 1],
                log_m[l]
            );
        }
    }

    // ---- Frame sync (§1.8.6) ---------------------------------------------

    #[test]
    fn expected_sync_bit_toggles() {
        assert!(expected_sync_bit(false));
        assert!(!expected_sync_bit(true));
    }

    // ---- Pitch index extraction (§1.4.2 fixed slots) ---------------------

    #[test]
    fn extract_pitch_index_round_trip_via_priority_table() {
        use crate::p25_fullrate::priority::prioritize;
        // Build a known b array, prioritize at any L (any L works for
        // pitch since positions are L-invariant per §1.3.1 robustness),
        // and verify extraction recovers the exact b̂₀.
        for b0 in 0..=255u8 {
            let mut b = [0u16; 59];
            b[0] = u16::from(b0);
            // Other params kept zero.
            let u = crate::p25_fullrate::priority::prioritize(&b, 9);
            assert_eq!(extract_pitch_index(&u), b0, "b̂₀ = {b0}");
        }
    }

    // ---- Top-level dequantize -----------------------------------

    #[test]
    fn dequantize_rejects_reserved_pitch() {
        use crate::p25_fullrate::priority::prioritize;
        let mut b = [0u16; 59];
        b[0] = 220; // reserved
        let u = crate::p25_fullrate::priority::prioritize(&b, 9);
        let mut state = DecoderState::new();
        assert_eq!(
            dequantize(&u, &mut state),
            Err(DecodeError::BadPitch)
        );
    }

    #[test]
    fn dequantize_produces_sane_silence_for_zero_b() {
        use crate::p25_fullrate::priority::prioritize;
        // b̂₀ = 0 → ω₀ = 4π/39.5, L = 9, K = 3
        // All other bits zero → quantizer indices at midtread base
        //   (which represents −2^{B−1} + 0.5 step → small negative
        //    log-amplitudes, NOT silence per se). But this is a valid
        //    decode path — the caller should not crash.
        let b = [0u16; 59];
        let u = crate::p25_fullrate::priority::prioritize(&b, 9);
        let mut state = DecoderState::new();
        let p = dequantize(&u, &mut state).expect("decode");
        assert_eq!(p.harmonic_count(), 9);
        assert!((p.omega_0() - 4.0 * PI / 39.5).abs() < 1e-6);
        // All voicing bits in b̂₁ are zero → all unvoiced.
        for l in 1..=9 {
            assert!(!p.voiced(l));
        }
        // All amplitudes finite and non-negative.
        for l in 1..=9 {
            let a = p.amplitude(l);
            assert!(a.is_finite() && a >= 0.0, "l={l}: M̃ = {a}");
        }
        // State updated.
        assert_eq!(state.previous_l(), 9);
    }

    #[test]
    fn dequantize_advances_decoder_state() {
        use crate::p25_fullrate::priority::prioritize;
        let b = [0u16; 59];
        let u = crate::p25_fullrate::priority::prioritize(&b, 9);
        let mut state = DecoderState::new();
        let p1 = dequantize(&u, &mut state).unwrap();
        let prev_l_after_first = state.previous_l();
        assert_eq!(prev_l_after_first, p1.harmonic_count());

        // Run a second frame; result will differ because prev_m is no
        // longer the uniform init state.
        let p2 = dequantize(&u, &mut state).unwrap();
        assert_eq!(p2.harmonic_count(), p1.harmonic_count());
        assert_eq!(state.previous_l(), p2.harmonic_count());
    }

    // ---- Encoder side ----------------------------------------------------

    #[test]
    fn encode_pitch_roundtrip_across_full_range() {
        for b0 in 0..=PITCH_INDEX_MAX {
            let p = pitch_decode(b0).unwrap();
            assert_eq!(encode_pitch(p.omega_0), Some(b0), "b̂₀ = {b0}");
        }
    }

    #[test]
    fn encode_pitch_rejects_out_of_range_omega() {
        assert_eq!(encode_pitch(0.0), None);
        assert_eq!(encode_pitch(-0.1), None);
        assert_eq!(encode_pitch(PI), None);
        assert_eq!(encode_pitch(PI + 0.1), None);
        // Just above the highest pitch (b̂₀ = 0 → ω₀ = 4π/39.5 ≈ 0.318).
        assert_eq!(encode_pitch(0.5), None);
    }

    #[test]
    fn collapse_vuv_roundtrips_against_expand() {
        // For every L, K and every possible b̂₁ value, expand then
        // collapse should recover the original b̂₁.
        for l in [9u8, 16, 36, 37, 56] {
            let k = vuv_band_count(l);
            let max = 1u32 << k;
            for b1 in 0..max as u16 {
                let v = expand_vuv(b1, l, k);
                let b1_back = collapse_vuv(&v[..l as usize], k);
                assert_eq!(b1_back, b1, "L={l}, K={k}, b̂₁={b1:#x}");
            }
        }
    }

    #[test]
    fn encode_uniform_roundtrips_at_quantizer_grid() {
        // For every B in 1..=10, every code 0..2^B-1 must roundtrip.
        for b_bits in 1..=10u8 {
            let max = 1u32 << b_bits;
            let delta = 0.5f32; // arbitrary positive
            for code in 0..max as u16 {
                let v = decode_uniform(code, b_bits, delta);
                let back = encode_uniform(v, b_bits, delta);
                assert_eq!(back, code, "B={b_bits}, code={code}");
            }
        }
    }

    #[test]
    fn encode_uniform_zero_bits_returns_zero() {
        assert_eq!(encode_uniform(0.0, 0, 1.0), 0);
        assert_eq!(encode_uniform(123.0, 0, 1.0), 0);
    }

    #[test]
    fn encode_uniform_clamps_outside_range() {
        // B=4, half=8, range = (-7.5..=7.5)*Δ.
        assert_eq!(encode_uniform(-1000.0, 4, 1.0), 0);
        assert_eq!(encode_uniform(1000.0, 4, 1.0), 15);
    }

    #[test]
    fn forward_block_dct_inverts_inverse_block_dct() {
        // Check that the public forward_block_dct is the proper inverse
        // of inverse_block_dct (paired exact inversion).
        for &n in &[1usize, 2, 3, 5, 7, 10] {
            let samples: [f32; MAX_BLOCK_SIZE] =
                std::array::from_fn(|i| if i < n { (i as f32 + 1.0).sin() * 2.0 } else { 0.0 });
            let c = forward_block_dct(&samples, n);
            let mut matrix = [[0f32; MAX_BLOCK_SIZE]; 6];
            matrix[0] = c;
            let mut blocks = [0u8; 6];
            blocks[0] = n as u8;
            let t = inverse_block_dct(&matrix, &blocks);
            for i in 0..n {
                assert!((t[i] - samples[i]).abs() < 1e-4, "n={n}, i={i}");
            }
        }
    }

    #[test]
    fn gain_dct_is_self_inverse_post_eq_69_correction() {
        // Post-2026-04-16 correction: both Eq. 61 (forward) and Eq. 69
        // (inverse) use denominator 6, so the pair is exact-inverse for
        // all 6-element vectors regardless of the Annex J block layout
        // the decoder later applies downstream.
        let r: [f32; 6] = [1.0, -2.5, 0.3, 0.0, -0.7, 1.8];
        let g = residuals_to_gain(&r);
        let r2 = gain_to_residuals(&g);
        for i in 0..6 {
            assert!(
                (r2[i] - r[i]).abs() < 1e-4,
                "round-trip: i={i}: {} vs {}",
                r2[i],
                r[i]
            );
        }
    }

    #[test]
    fn gain_dct_dc_only_input_propagates_uniformly() {
        // R̂ = [c, c, c, c, c, c] → Ĝ_1 = c, Ĝ_2..6 ≈ 0 (DCT of constant).
        let c = 3.5;
        let r = [c; 6];
        let g = residuals_to_gain(&r);
        assert!((g[0] - c).abs() < 1e-5);
        for m in 1..6 {
            assert!(g[m].abs() < 1e-5, "Ĝ_{m} = {}", g[m]);
        }
    }

    #[test]
    fn forward_log_prediction_inverts_apply_log_prediction() {
        // Pick a state and a current frame, feed log_m through forward
        // then inverse — should recover log_m exactly (modulo float).
        let state = DecoderState::new();
        let l = 30u8;
        let mut log_m = [0f32; L_MAX as usize + 2];
        log_m[0] = 0.0; // virtual M̃_0 = 1 → log = 0
        for i in 1..=l as usize {
            log_m[i] = (i as f32 * 0.1) - 1.5; // varied
        }
        let t = forward_log_prediction(&log_m, l, &state);
        let log_m_back = apply_log_prediction(&t, l, &state);
        for i in 1..=l as usize {
            assert!(
                (log_m_back[i] - log_m[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                log_m_back[i],
                log_m[i]
            );
        }
    }

    #[test]
    fn quantize_pitch_and_vuv_roundtrip_against_dequantize() {
        // Pitch (Eq. 45 ↔ Eq. 46) and V/UV (collapse ↔ expand) are
        // both exact inverses; the gain DCT step is intentionally
        // lossy (see `analysis/vocoder_decode_disambiguations.md` §9)
        // so we don't expect b̂₂..b̂_{L+1} to roundtrip exactly. This
        // test pins down the bit-exact pieces.
        use crate::p25_fullrate::priority::prioritize;
        // Pick a pitch index whose decoded L lets us reuse the same L
        // in prioritize. b̂₀ = 0 → L = 9.
        let l = 9u8;
        let k = vuv_band_count(l);
        let mut b = [0u16; 59];
        b[0] = 0;
        b[1] = 0b101 & ((1u16 << k) - 1);
        let u = crate::p25_fullrate::priority::prioritize(&b, l);

        let mut state_dec = DecoderState::new();
        let params = dequantize(&u, &mut state_dec).unwrap();

        let mut state_enc = DecoderState::new();
        let b_back = quantize(&params, &mut state_enc).unwrap();

        assert_eq!(b_back[0], b[0], "b̂₀ pitch");
        assert_eq!(b_back[1], b[1], "b̂₁ V/UV");
    }

    // ---- V/UV expansion --------------------------------------------------

    #[test]
    fn expand_vuv_all_voiced() {
        // L=9, K=3. b̂₁ = 0b111 → all 9 harmonics voiced.
        let v = expand_vuv(0b111, 9, 3);
        for l in 0..9 {
            assert!(v[l], "harmonic {} not voiced", l + 1);
        }
        for l in 9..L_MAX as usize {
            assert!(!v[l], "padding bit set at l={}", l + 1);
        }
    }

    #[test]
    fn expand_vuv_all_unvoiced() {
        let v = expand_vuv(0, 9, 3);
        for l in 0..9 {
            assert!(!v[l], "harmonic {} unexpectedly voiced", l + 1);
        }
    }

    #[test]
    fn expand_vuv_band_groups_three_harmonics_each() {
        // L=9, K=3, b̂₁ = 0b101 (band 1 voiced, band 2 unvoiced, band 3 voiced).
        // Bands cover harmonics: 1..=3, 4..=6, 7..=9.
        let v = expand_vuv(0b101, 9, 3);
        for l in 1..=3 { assert!(v[l - 1], "band 1: l={l}"); }
        for l in 4..=6 { assert!(!v[l - 1], "band 2: l={l}"); }
        for l in 7..=9 { assert!(v[l - 1], "band 3: l={l}"); }
    }

    #[test]
    fn expand_vuv_band_msb_first() {
        // L=9, K=3, b̂₁ bit positions: bit 2 = band 1, bit 1 = band 2,
        // bit 0 = band 3. Set only band 1 (bit 2 = 1).
        let v = expand_vuv(0b100, 9, 3);
        for l in 1..=3 { assert!(v[l - 1], "band 1 MSB→bit K-1, l={l}"); }
        for l in 4..=9 { assert!(!v[l - 1], "expected unvoiced, l={l}"); }
    }

    #[test]
    fn expand_vuv_l_above_36_absorbs_into_band_12() {
        // L=56, K=12. Harmonics 34..=56 all share band 12.
        // Set only band 12 (LSB of b̂₁).
        let mut b1 = 0u16;
        b1 |= 1 << 0; // band 12 voiced
        let v = expand_vuv(b1, 56, 12);
        for l in 1..=33 { assert!(!v[l - 1], "low band, l={l}"); }
        for l in 34..=56 { assert!(v[l - 1], "band 12 absorbing, l={l}"); }
    }

    #[test]
    fn expand_vuv_l37_boundary() {
        // L=37, K=12. Harmonic 37 should map to band 12 even though
        // floor((37+2)/3) = 13 (out of K range) — the rule says
        // l > 36 → k_l = 12 directly.
        let mut b1 = 0u16;
        b1 |= 1 << 0; // band 12 voiced
        let v = expand_vuv(b1, 37, 12);
        assert!(v[36], "harmonic 37 should be band 12 (voiced)");
        for l in 1..=33 { assert!(!v[l - 1], "low band, l={l}"); }
        for l in 34..=37 { assert!(v[l - 1], "band 12, l={l}"); }
    }

    #[test]
    fn expand_vuv_l_under_36_no_clamping_needed() {
        // For L=36, K=12 — all band indices floor((l+2)/3) for l=1..36
        // produce values 1..12, no clamping happens.
        let v_all_v = expand_vuv(0xFFFu16, 36, 12);
        for l in 1..=36 { assert!(v_all_v[l - 1], "l={l}"); }
        let v_all_uv = expand_vuv(0, 36, 12);
        for l in 1..=36 { assert!(!v_all_uv[l - 1], "l={l}"); }
    }

    #[test]
    fn band_for_harmonic_matches_spec_examples() {
        // Cases from §1.3.2: L=9, K=3
        assert_eq!(band_for_harmonic(1, 3), 1);
        assert_eq!(band_for_harmonic(3, 3), 1);
        assert_eq!(band_for_harmonic(4, 3), 2);
        assert_eq!(band_for_harmonic(6, 3), 2);
        assert_eq!(band_for_harmonic(7, 3), 3);
        assert_eq!(band_for_harmonic(9, 3), 3);

        // L=37, K=12: harmonics 34..=36 → band 12 by formula;
        // harmonic 37 → band 12 by rule.
        assert_eq!(band_for_harmonic(33, 12), 11);
        assert_eq!(band_for_harmonic(34, 12), 12);
        assert_eq!(band_for_harmonic(36, 12), 12);
        assert_eq!(band_for_harmonic(37, 12), 12);
        assert_eq!(band_for_harmonic(56, 12), 12);
    }
}
