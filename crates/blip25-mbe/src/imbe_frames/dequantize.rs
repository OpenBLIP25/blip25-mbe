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

use crate::mbe_params::{L_MAX, MbeParams};

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
/// from the dequantized gain DCT vector via §1.8.3 Eq. 69:
///
/// ```text
/// R̃_i = Σ_{m=1}^{6} α(m) · G̃_m · cos[π·(m−1)·(i − 0.5) / J̃_i]
///         for 1 ≤ i ≤ 6
/// α(1) = 1,  α(m) = 2 for m > 1
/// ```
///
/// Per the spec note, this is a *per-block* inverse DCT — the cosine
/// denominator is `J̃_i` (the i-th block's length), not 6. The basis
/// frequency therefore changes block-by-block.
pub fn gain_to_residuals(g: &[f32; 6], blocks: &[u8; 6]) -> [f32; 6] {
    let mut r = [0f32; 6];
    for i_0 in 0..6 {
        let j_i = f32::from(blocks[i_0]);
        debug_assert!(j_i > 0.0, "Annex J: J̃_i must be positive");
        let i_half = i_0 as f32 + 0.5;
        let mut acc = 0.0f32;
        for m_0 in 0..6 {
            let alpha = if m_0 == 0 { 1.0 } else { 2.0 };
            let arg = PI * (m_0 as f32) * i_half / j_i;
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
    fn gain_to_residuals_constant_g1_propagates_to_all_blocks() {
        // If only G̃_1 = c (DC term) and the rest are zero, the per-
        // block inverse DCT must produce R̃_i = c for every block,
        // independent of J̃_i.
        let g = [3.5f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Try a non-trivial block layout.
        let blocks: [u8; 6] = IMBE_BLOCK_LENGTHS[(56 - 9) as usize]; // L=56
        let r = gain_to_residuals(&g, &blocks);
        for i in 0..6 {
            assert!((r[i] - 3.5).abs() < 1e-5, "R̃_{}: {}", i + 1, r[i]);
        }
    }

    #[test]
    fn gain_to_residuals_uses_per_block_denominator() {
        // Two blocks with different J̃_i → DCT phase differs, so R̃_i
        // should differ when only a higher-order G̃_m is non-zero.
        let mut g = [0f32; 6];
        g[2] = 1.0; // G̃_3 only (m_0 = 2)
        // Contrived blocks: block 0 length 1, block 1 length 5.
        // Eq. 69 with α(3)=2 and arg = π·(m-1)·(i-0.5)/J̃_i:
        //   block 0 (i_0=0, J̃=1): arg = π·2·0.5/1 = π   → cos=-1   → 2·1·(-1)   = -2.0
        //   block 1 (i_0=1, J̃=5): arg = π·2·1.5/5 = 3π/5 → cos≈-0.309 → 2·1·(-0.309) ≈ -0.618
        let blocks = [1u8, 5, 1, 1, 1, 1];
        let r = gain_to_residuals(&g, &blocks);
        assert!((r[0] - (-2.0)).abs() < 1e-5, "block 0: {}", r[0]);
        assert!((r[1] - (-0.618_034)).abs() < 1e-3, "block 1: {}", r[1]);
    }

    // ---- HOC reconstruction (§1.8.4) -------------------------------------

    /// Forward DCT helper used only for tests. The spec's encode form
    /// (Eq. 61 for the gain side) uses a *uniform* `1/N` factor across
    /// all coefficients; the `α(k)` weighting that distinguishes DC
    /// from the rest lives entirely on the inverse side (Eq. 73). So
    /// this forward must apply the same `1/N` factor for `k=0` and
    /// `k>0` to be a true inverse of `inverse_block_dct`.
    fn forward_block_dct(samples: &[f32], n: usize) -> [f32; MAX_BLOCK_SIZE] {
        let mut c = [0f32; MAX_BLOCK_SIZE];
        for k_0 in 0..n {
            let mut acc = 0.0f32;
            for j_0 in 0..n {
                let arg = PI * (k_0 as f32) * (j_0 as f32 + 0.5) / n as f32;
                acc += samples[j_0] * arg.cos();
            }
            c[k_0] = acc / n as f32;
        }
        c
    }

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
