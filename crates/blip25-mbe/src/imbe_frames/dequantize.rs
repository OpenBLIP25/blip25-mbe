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
