//! The MBE parameter model — the crate's interchange type.
//!
//! Every wire format in this crate parses to [`MbeParams`]. Every codec
//! generation consumes or produces it. Rate conversion operates entirely
//! within it.
//!
//! The MBE speech model, originating in Griffin & Lim (1988) and formalized
//! for P25 in TIA-102.BABA-A, represents a 20 ms speech frame as four
//! quantities — the synthesis boundary described in BABA-A §8.3:
//!
//! - `ω₀` — fundamental frequency (pitch) in radians/sample
//! - `L`  — number of harmonics, derived from `ω₀`
//! - `v_l` — voiced/unvoiced decision for each harmonic `l = 1..=L`
//! - `M_l` — spectral amplitude for each harmonic `l = 1..=L`
//!
//! After dequantization the gain is baked into `M_l`; it does not appear
//! separately at the synthesis boundary. Wire-side intermediate forms
//! (gain + residuals, band-level V/UV) belong in the wire modules, not
//! here.
//!
//! This module is deliberately independent of any wire format or codec
//! generation. Adding a new wire format or a new codec generation must
//! not require any change here.

pub mod quantize;

use core::f32::consts::PI;

/// Minimum harmonic count. BABA-A §1.1.
pub const L_MIN: u8 = 9;

/// Maximum harmonic count. BABA-A §1.1.
pub const L_MAX: u8 = 56;

/// Speech sample rate in Hz. BABA-A §1.1.
pub const SAMPLE_RATE_HZ: u32 = 8_000;

/// Samples per 20 ms frame. BABA-A §1.1.
pub const SAMPLES_PER_FRAME: u16 = 160;

/// Frame duration in milliseconds. BABA-A §1.1.
pub const FRAME_DURATION_MS: u16 = 20;

/// Storage width for per-harmonic arrays, sized to `L_MAX`.
const L_CAP: usize = L_MAX as usize;

/// MBE model parameters for a single 20 ms speech frame.
///
/// This is the interchange type that decouples wire format from codec.
/// It carries exactly what BABA-A §8.3 hands to the synthesis engine:
/// the fundamental frequency, the per-harmonic voicing vector, and the
/// per-harmonic spectral amplitudes.
///
/// Storage is fixed-size (stack-allocated, no heap). Entries for `l > L`
/// are implementation detail and must not be read.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MbeParams {
    /// Fundamental frequency in radians/sample. Range `(0, π)`.
    omega_0: f32,
    /// Harmonic count. Invariant: `L_MIN <= l <= L_MAX`.
    l: u8,
    /// Per-harmonic voicing. Index `l-1` holds `v_l`; entries `>= l` unused.
    #[cfg_attr(feature = "serde", serde(with = "serde_l_cap_bool"))]
    voiced: [bool; L_CAP],
    /// Per-harmonic spectral amplitudes. Index `l-1` holds `M_l`;
    /// entries `>= l` unused.
    #[cfg_attr(feature = "serde", serde(with = "serde_l_cap_f32"))]
    amplitudes: [f32; L_CAP],
}

#[cfg(feature = "serde")]
mod serde_l_cap_bool {
    use super::L_CAP;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(v: &[bool; L_CAP], s: S) -> Result<S::Ok, S::Error> {
        v.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[bool; L_CAP], D::Error> {
        let v = Vec::<bool>::deserialize(d)?;
        if v.len() != L_CAP {
            return Err(serde::de::Error::invalid_length(v.len(), &format!("{L_CAP}").as_str()));
        }
        let mut out = [false; L_CAP];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

#[cfg(feature = "serde")]
mod serde_l_cap_f32 {
    use super::L_CAP;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(v: &[f32; L_CAP], s: S) -> Result<S::Ok, S::Error> {
        v.as_slice().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; L_CAP], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        if v.len() != L_CAP {
            return Err(serde::de::Error::invalid_length(v.len(), &format!("{L_CAP}").as_str()));
        }
        let mut out = [0.0f32; L_CAP];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

/// Error returned when [`MbeParams`] inputs violate model invariants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbeParamsError {
    /// `ω₀` outside `(0, π)`.
    OmegaOutOfRange,
    /// `L` outside `[L_MIN, L_MAX]`.
    HarmonicCountOutOfRange,
    /// A spectral amplitude was negative or non-finite.
    InvalidAmplitude,
}

impl MbeParams {
    /// Construct from `ω₀`, `L`, and per-harmonic voicing and amplitude
    /// slices of length `L`.
    ///
    /// Validates the invariants stated on the field docs. The caller is
    /// responsible for ensuring `L` is consistent with `ω₀` (see
    /// [`Self::harmonic_count_for`] for the decoder rule); this
    /// constructor does not re-derive `L`. Full-rate IMBE derives the
    /// pair analytically from `b₀` (BABA-A §6.1 Eqs. 46–47); half-rate
    /// AMBE+2 derives them by lookup (Annex L). Either way the wire side
    /// owns the pairing, and the two are kept as supplied.
    pub fn new(
        omega_0: f32,
        l: u8,
        voiced: &[bool],
        amplitudes: &[f32],
    ) -> Result<Self, MbeParamsError> {
        if !(omega_0 > 0.0 && omega_0 < PI) {
            return Err(MbeParamsError::OmegaOutOfRange);
        }
        if l < L_MIN || l > L_MAX {
            return Err(MbeParamsError::HarmonicCountOutOfRange);
        }
        let n = l as usize;
        if voiced.len() != n || amplitudes.len() != n {
            return Err(MbeParamsError::HarmonicCountOutOfRange);
        }
        for &m in amplitudes {
            if !m.is_finite() || m < 0.0 {
                return Err(MbeParamsError::InvalidAmplitude);
            }
        }

        let mut v = [false; L_CAP];
        let mut a = [0.0f32; L_CAP];
        v[..n].copy_from_slice(voiced);
        a[..n].copy_from_slice(amplitudes);
        Ok(Self { omega_0, l, voiced: v, amplitudes: a })
    }

    /// A silent frame: all unvoiced, all amplitudes zero.
    ///
    /// BABA-A §6 treats silence as an MBE frame with near-zero amplitudes
    /// and all bands unvoiced. `ω₀` is set to the highest full-rate
    /// decoder value (`b̃₀ = 0`, i.e. `4π/39.5`, per §6.1 Eq. 46), which
    /// pairs with `L = L_MIN` under Eq. 47. The exact value is
    /// unobservable in the synthesized output when all amplitudes are
    /// zero.
    ///
    /// **Note on rate:** this constructor's ω₀ is full-rate-specific.
    /// 4π/39.5 ≈ 0.318 sits just above the half-rate Annex L pitch
    /// table's max (b̂₀=0 → ω₀ ≈ 0.314), so half-rate consumers must
    /// use [`Self::silence_ambe_plus2`] instead. Mixing them silently fails
    /// at `rate33::dequantize::quantize` with
    /// `DecodeError::BadPitch`.
    pub fn silence() -> Self {
        Self {
            omega_0: 4.0 * PI / 39.5,
            l: L_MIN,
            voiced: [false; L_CAP],
            amplitudes: [0.0; L_CAP],
        }
    }

    /// Half-rate-compatible silent frame.
    ///
    /// Same shape as [`Self::silence`] (all unvoiced, amplitudes zero) but
    /// with `ω₀` taken from the largest entry in the AMBE+2 Annex L
    /// pitch table (b̂₀=0 → ω₀ ≈ 0.314, L = 9). This satisfies
    /// `rate33::dequantize::encode_pitch`'s range check, which
    /// rejects [`Self::silence`]'s 0.318 as out of table.
    ///
    /// Use this when the caller is feeding into the half-rate
    /// quantizer and wants a "no-content" placeholder frame
    /// (analysis-encoder silence dispatch, erasure substitution,
    /// preroll output, etc).
    pub fn silence_ambe_plus2() -> Self {
        // The b̂₀=0 row of the Annex L table gives the largest
        // half-rate ω₀ in canonical form. Hardcoded here to avoid a
        // dependency on `rate33` from this module; the value
        // matches `AMBE_PITCH_TABLE[0]` exactly. If Annex L is ever
        // re-extracted with a corrected entry, update this constant
        // (and the test below) to match.
        const HALFRATE_B0_ZERO_OMEGA: f32 = 0.313_977;
        Self {
            omega_0: HALFRATE_B0_ZERO_OMEGA,
            l: L_MIN,
            voiced: [false; L_CAP],
            amplitudes: [0.0; L_CAP],
        }
    }

    /// Fundamental frequency in radians/sample.
    #[inline]
    pub fn omega_0(&self) -> f32 { self.omega_0 }

    /// Fundamental frequency in Hz, for diagnostics.
    #[inline]
    pub fn fundamental_hz(&self) -> f32 {
        self.omega_0 * (SAMPLE_RATE_HZ as f32) / (2.0 * PI)
    }

    /// Number of harmonics `L`.
    #[inline]
    pub fn harmonic_count(&self) -> u8 { self.l }

    /// Voicing decision for harmonic `l`. Indexed `1..=L`.
    ///
    /// Panics in debug builds if `l` is out of range; in release builds
    /// returns `false` for out-of-range indices.
    #[inline]
    pub fn voiced(&self, l: u8) -> bool {
        debug_assert!(l >= 1 && l <= self.l, "harmonic index out of range");
        *self.voiced.get(l as usize - 1).unwrap_or(&false)
    }

    /// Spectral amplitude `M_l` for harmonic `l`. Indexed `1..=L`.
    #[inline]
    pub fn amplitude(&self, l: u8) -> f32 {
        debug_assert!(l >= 1 && l <= self.l, "harmonic index out of range");
        *self.amplitudes.get(l as usize - 1).unwrap_or(&0.0)
    }

    /// Slice of voicing decisions `v_1..v_L`.
    #[inline]
    pub fn voiced_slice(&self) -> &[bool] {
        &self.voiced[..self.l as usize]
    }

    /// Slice of spectral amplitudes `M_1..M_L`.
    #[inline]
    pub fn amplitudes_slice(&self) -> &[f32] {
        &self.amplitudes[..self.l as usize]
    }

    /// Compute the harmonic count `L` implied by `ω₀` per BABA-A §6.1
    /// Eq. 47 (= §5.1.5 Eq. 31 for the estimator):
    ///
    /// ```text
    /// L = floor(0.9254 · floor(π / ω₀ + 0.25))
    /// ```
    ///
    /// Clamped to `[L_MIN, L_MAX]`. The inner floor rounds `π/ω₀` to
    /// the nearest integer with a 0.25 offset before the 0.9254 scale,
    /// so this is **not** the same as `floor(π/ω₀)` — do not simplify.
    pub fn harmonic_count_for(omega_0: f32) -> u8 {
        if !(omega_0 > 0.0) {
            return L_MIN;
        }
        let inner = (PI / omega_0 + 0.25).floor();
        let raw = (0.9254 * inner).floor() as i32;
        raw.clamp(L_MIN as i32, L_MAX as i32) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_all_zero_and_unvoiced() {
        let p = MbeParams::silence();
        assert_eq!(p.harmonic_count(), L_MIN);
        for l in 1..=p.harmonic_count() {
            assert_eq!(p.amplitude(l), 0.0);
            assert!(!p.voiced(l));
        }
    }

    #[test]
    fn silence_ambe_plus2_uses_half_rate_table_max_omega() {
        let p = MbeParams::silence_ambe_plus2();
        assert_eq!(p.harmonic_count(), L_MIN);
        for l in 1..=p.harmonic_count() {
            assert_eq!(p.amplitude(l), 0.0);
            assert!(!p.voiced(l));
        }
        // ω₀ must round-trip through `rate33::dequantize::encode_pitch`
        // back to b̂₀ = 0 (the largest-ω₀ entry); full-rate `silence`
        // doesn't.
        let b0 = crate::rate33::dequantize::encode_pitch(p.omega_0());
        assert_eq!(b0, Some(0));
        // Full-rate silence does NOT round-trip through half-rate
        // encode_pitch — that's the whole reason `silence_ambe_plus2`
        // exists.
        let p_full = MbeParams::silence();
        assert_eq!(crate::rate33::dequantize::encode_pitch(p_full.omega_0()), None);
    }

    #[test]
    fn rejects_omega_out_of_range() {
        let v = vec![false; L_MIN as usize];
        let a = vec![1.0; L_MIN as usize];
        assert_eq!(
            MbeParams::new(0.0, L_MIN, &v, &a),
            Err(MbeParamsError::OmegaOutOfRange)
        );
        assert_eq!(
            MbeParams::new(PI, L_MIN, &v, &a),
            Err(MbeParamsError::OmegaOutOfRange)
        );
    }

    #[test]
    fn rejects_harmonic_count_out_of_range() {
        let v = vec![false; 8];
        let a = vec![1.0; 8];
        assert_eq!(
            MbeParams::new(0.1, 8, &v, &a),
            Err(MbeParamsError::HarmonicCountOutOfRange)
        );
    }

    #[test]
    fn rejects_negative_amplitude() {
        let v = vec![false; L_MIN as usize];
        let mut a = vec![1.0f32; L_MIN as usize];
        a[3] = -0.1;
        assert_eq!(
            MbeParams::new(0.1, L_MIN, &v, &a),
            Err(MbeParamsError::InvalidAmplitude)
        );
    }

    #[test]
    fn harmonic_count_for_matches_spec_rule() {
        // ω₀ = π/20: π/ω₀=20, +0.25→20.25, floor=20, ·0.9254=18.508, floor=18
        assert_eq!(MbeParams::harmonic_count_for(PI / 20.0), 18);
        // ω₀ tiny → clamps to L_MAX
        assert_eq!(MbeParams::harmonic_count_for(0.001), L_MAX);
        // ω₀ large → clamps to L_MIN
        assert_eq!(MbeParams::harmonic_count_for(PI / 2.0), L_MIN);
    }

    #[test]
    fn harmonic_count_matches_full_rate_b0_endpoints() {
        // §6.1 Eq. 46: ω̃₀ = 4π/(b̃₀ + 39.5).
        // b̃₀ = 0 → ω̃₀ = 4π/39.5, and Eq. 47 pairs this with L = 9.
        let omega_at_b0_zero = 4.0 * PI / 39.5;
        assert_eq!(MbeParams::harmonic_count_for(omega_at_b0_zero), 9);
        // b̃₀ = 207 (highest valid) → ω̃₀ = 4π/246.5, pairs with L = 56.
        let omega_at_b0_max = 4.0 * PI / 246.5;
        assert_eq!(MbeParams::harmonic_count_for(omega_at_b0_max), 56);
    }

    #[test]
    fn accessors_round_trip() {
        let l = 12u8;
        let voiced: Vec<bool> = (0..l).map(|i| i % 2 == 0).collect();
        let amps: Vec<f32> = (0..l).map(|i| (i as f32) + 1.0).collect();
        let p = MbeParams::new(0.2, l, &voiced, &amps).unwrap();

        assert_eq!(p.harmonic_count(), l);
        assert_eq!(p.voiced_slice(), voiced.as_slice());
        assert_eq!(p.amplitudes_slice(), amps.as_slice());
        for i in 1..=l {
            assert_eq!(p.voiced(i), voiced[i as usize - 1]);
            assert_eq!(p.amplitude(i), amps[i as usize - 1]);
        }
    }
}
