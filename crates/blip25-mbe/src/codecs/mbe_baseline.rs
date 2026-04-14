//! MBE baseline codec — TIA-102.BABA-A §1.10–§1.13 (synthesis pipeline)
//! and §1.8 (analysis is shared with the upstream pipeline in
//! [`crate::imbe_frames::dequantize`]).
//!
//! This module hosts the synthesizer that turns [`MbeParams`] into
//! 8 kHz 16-bit PCM. The synthesis algorithm follows the 1993 P25
//! vocoder specification:
//!
//! - **§1.10 enhancement** — `M̃_l → M̄_l` via the W_l psycho-acoustic
//!   weighting.
//! - **§1.11 smoothing** — frame repeat / mute / V/UV smoothing.
//! - **§1.12.1 unvoiced** — LCG noise → DFT → per-band spectral shaping
//!   → IDFT → weighted overlap-add.
//! - **§1.12.2 voiced** — sinusoidal overlap-add with phase tracking
//!   across the four V/UV transition cases.
//!
//! Per TIA-102.BABG this baseline generation explicitly cannot pass the
//! enhanced-vocoder performance tests. It is implemented here for spec
//! completeness and as the reference point against which AMBE/+/+2
//! generations' improvements are measured.
//!
//! [`MbeParams`]: crate::mbe_params::MbeParams

use core::f64::consts::PI as PI64;

use crate::mbe_params::L_MAX;

include!(concat!(env!("OUT_DIR"), "/annex_i_synth_window.rs"));

/// Lookup `wS(n)` for `n ∈ [-105, 105]`. Returns 0.0 outside that range
/// (matching the spec — `wS` is treated as zero outside its support).
#[inline]
pub fn synth_window(n: i32) -> f32 {
    if !(-105..=105).contains(&n) {
        return 0.0;
    }
    IMBE_SYNTH_WINDOW[(n + 105) as usize]
}

// ---------------------------------------------------------------------------
// §1.10 — Spectral amplitude enhancement (Eq. 105–111)
// ---------------------------------------------------------------------------

/// Initial value for the local-energy state `S_E` per BABA-A §10
/// Annex A. Used when the synthesizer is cold-started.
pub const INIT_S_E: f64 = 75000.0;

/// Lower bound on `S_E` per Eq. 111. Floors out the recurrence so very
/// quiet frames don't drive the predictor toward zero.
pub const S_E_FLOOR: f64 = 10000.0;

/// Apply BABA-A §1.10 spectral amplitude enhancement to the
/// reconstructed magnitudes `M̃_l`, producing the synthesizer-side
/// magnitudes `M̄_l` and the updated local-energy state `S_E(0)`.
///
/// Per §1.10:
/// 1. Compute energy moments `R_M0`, `R_M1` (Eq. 105–106).
/// 2. Per-harmonic weight `W_l` (Eq. 107) with `8·l ≤ L̃` bypass and
///    `[0.5, 1.2]` clamp on `W_l` (Eq. 108).
/// 3. Energy-preserving rescale by `γ = sqrt(R_M0 / Σ M̄_l²)` (Eq. 109–110).
/// 4. Update `S_E(0) = max(0.95·S_E(−1) + 0.05·R_M0, 10000)` (Eq. 111).
///
/// All internal arithmetic is `f64` per the spec's reference C
/// implementation. Inputs and outputs at the `MbeParams` boundary
/// stay in `f32`.
///
/// `m_tilde` must have length `L̃ ≥ 1`; the returned array's entries
/// `[0..L̃]` are populated, the rest are zero.
///
/// **Edge cases.** If `R_M0` is zero (silence frame) the enhancement
/// reduces to a passthrough — `W_l` is undefined (denominator = 0)
/// and `γ` is also undefined; we return `M̄_l = M̃_l` and only update
/// `S_E`. Similarly for `R_M0 == R_M1` (denominator zero by another
/// path).
pub fn enhance_spectral_amplitudes(
    m_tilde: &[f32],
    omega_0: f32,
    s_e_prev: f64,
) -> ([f32; L_MAX as usize], f64) {
    let l = m_tilde.len();
    debug_assert!(l > 0 && l <= L_MAX as usize);
    let omega_0 = f64::from(omega_0);

    // Eq. 105–106: energy moments.
    let mut r_m0 = 0.0f64;
    let mut r_m1 = 0.0f64;
    for (i, &m) in m_tilde.iter().enumerate() {
        let l_one = (i + 1) as f64;
        let m2 = f64::from(m) * f64::from(m);
        r_m0 += m2;
        r_m1 += m2 * (omega_0 * l_one).cos();
    }

    let mut m_bar_f64 = [0f64; L_MAX as usize];
    let denom = omega_0 * r_m0 * (r_m0 - r_m1);

    if r_m0 <= 0.0 || denom.abs() < 1e-30 {
        // Silence-frame degenerate path — bypass enhancement.
        for (i, &m) in m_tilde.iter().enumerate() {
            m_bar_f64[i] = f64::from(m);
        }
    } else {
        // Eq. 107–108: per-harmonic weight with bypass + clamp.
        for (i, &m) in m_tilde.iter().enumerate() {
            let l_one = (i + 1) as f64;
            let bar = if 8 * (i + 1) <= l {
                f64::from(m) // first branch — bypass for the lowest ⌊L̃/8⌋
            } else {
                let num = r_m0 * r_m0 + r_m1 * r_m1
                    - 2.0 * r_m0 * r_m1 * (omega_0 * l_one).cos();
                if num <= 0.0 {
                    f64::from(m) // numerator pathology → passthrough
                } else {
                    let w_l = f64::from(m).sqrt() * (0.96 * num / denom).powf(0.25);
                    if w_l > 1.2 {
                        1.2 * f64::from(m)
                    } else if w_l < 0.5 {
                        0.5 * f64::from(m)
                    } else {
                        w_l * f64::from(m)
                    }
                }
            };
            m_bar_f64[i] = bar;
        }

        // Eq. 109–110: energy-preserving rescale.
        let mut sum_sq = 0.0f64;
        for &v in m_bar_f64.iter().take(l) {
            sum_sq += v * v;
        }
        if sum_sq > 1e-30 {
            let gamma = (r_m0 / sum_sq).sqrt();
            for v in m_bar_f64.iter_mut().take(l) {
                *v *= gamma;
            }
        }
    }

    // Eq. 111: S_E recurrence with floor.
    let s_e = (0.95 * s_e_prev + 0.05 * r_m0).max(S_E_FLOOR);

    let mut m_bar = [0f32; L_MAX as usize];
    for (i, &v) in m_bar_f64.iter().take(l).enumerate() {
        m_bar[i] = v as f32;
    }
    (m_bar, s_e)
}

// ---------------------------------------------------------------------------
// §1.11 — Adaptive smoothing, frame repeat / mute, V/UV smoothing
// ---------------------------------------------------------------------------

/// Initial value for the amplitude-smoothing state `τ_M` per BABA-A
/// §10 Annex A.
pub const INIT_TAU_M: f64 = 20480.0;

/// Smoothed-error-rate threshold above which the frame is muted
/// (Eq. mute condition in §1.11.2).
pub const MUTE_EPSILON_R_THRESHOLD: f64 = 0.0875;

/// Per-frame error context derived from FEC decoding (§1.5) plus the
/// pitch-validity check (§1.3.1). Drives the §1.11 smoothing
/// decisions.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameErrorContext {
    /// Bit errors corrected in û₀ (Golay).
    pub epsilon_0: u8,
    /// Bit errors corrected in û₄ specifically (used in V_M branch 2).
    pub epsilon_4: u8,
    /// Total bit errors across all 7 FEC-protected vectors `û₀..û₆`.
    pub epsilon_t: u8,
    /// Set when the decoded pitch index `b̂₀` is in the reserved range
    /// `[208, 255]` (or otherwise invalid) — forces a frame repeat
    /// regardless of the error counts.
    pub bad_pitch: bool,
}

/// Action the synthesizer should take for the current frame, per
/// BABA-A §1.11.1 / §1.11.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameDisposition {
    /// Synthesize from the current frame's parameters.
    Use,
    /// Substitute the previous frame's parameters (Eq. 99–104) and
    /// then synthesize.
    Repeat,
    /// Bypass the synthesizer and emit silence / low-amplitude noise.
    Mute,
}

/// Update the smoothed error rate `ε_R(0) = 0.95·ε_R(−1) + 0.05·(ε_T/144)`
/// and decide the frame disposition.
///
/// Per §1.11.1 the repeat trigger is:
/// `b̂₀ invalid` **OR** (`ε₀ ≥ 2` AND `ε_T ≥ 10 + 40·ε_R(0)`).
///
/// Per §1.11.2 mute supersedes repeat when `ε_R(0) > 0.0875`.
pub fn frame_disposition(
    err: &FrameErrorContext,
    epsilon_r_prev: f64,
) -> (FrameDisposition, f64) {
    let epsilon_r =
        0.95 * epsilon_r_prev + 0.05 * (f64::from(err.epsilon_t) / 144.0);
    let disp = if epsilon_r > MUTE_EPSILON_R_THRESHOLD {
        FrameDisposition::Mute
    } else if err.bad_pitch
        || (err.epsilon_0 >= 2
            && f64::from(err.epsilon_t) >= 10.0 + 40.0 * epsilon_r)
    {
        FrameDisposition::Repeat
    } else {
        FrameDisposition::Use
    };
    (disp, epsilon_r)
}

/// Output of the §1.11.3 V/UV-and-amplitude smoothing step.
#[derive(Clone, Debug)]
pub struct SmoothedFrame {
    /// Smoothed per-harmonic spectral amplitudes (`M̄_l` after γ_M scaling).
    /// Entries `[0..L]` are populated.
    pub m_bar: [f32; L_MAX as usize],
    /// Smoothed per-harmonic V/UV decisions (`v̄_l`). Entries `[0..L]`
    /// are populated; higher indices are `false`.
    pub v_bar: [bool; L_MAX as usize],
    /// Updated amplitude-smoothing state `τ_M(0)` for the next frame.
    pub tau_m: f64,
}

/// Apply the §1.11.3 V/UV-and-amplitude smoothing pipeline.
///
/// Inputs:
/// - `m_bar`: enhanced spectral amplitudes from §1.10 (`M̄_l`).
/// - `v_tilde`: per-harmonic V/UV from the §1.3.2 expansion (`ṽ_l`).
/// - `s_e`: current-frame `S_E(0)` from §1.10.
/// - `epsilon_r`: smoothed error rate `ε_R(0)` from
///   [`frame_disposition`].
/// - `epsilon_t`, `epsilon_4`: per-frame error counts (raw, not smoothed).
/// - `tau_m_prev`: previous frame's `τ_M(−1)`.
///
/// Returns the smoothed `M̄_l`, the smoothed per-harmonic V/UV `v̄_l`,
/// and the updated `τ_M(0)`.
pub fn apply_smoothing(
    m_bar: &[f32],
    v_tilde: &[bool],
    s_e: f64,
    epsilon_r: f64,
    epsilon_t: u8,
    epsilon_4: u8,
    tau_m_prev: f64,
) -> SmoothedFrame {
    let l = m_bar.len();
    debug_assert_eq!(v_tilde.len(), l);

    // Eq. 112: V_M threshold (3-branch).
    let v_m: f64 = if epsilon_r <= 0.005 && epsilon_t <= 4 {
        f64::INFINITY
    } else if epsilon_r <= 0.0125 && epsilon_4 == 0 {
        45.255 * s_e.powf(0.375) * (-277.26 * epsilon_r).exp()
    } else {
        1.414 * s_e.powf(0.375)
    };

    // Eq. 113: per-harmonic V/UV override (force voiced if M̄_l > V_M).
    let mut v_bar = [false; L_MAX as usize];
    for (i, &m) in m_bar.iter().enumerate() {
        v_bar[i] = if f64::from(m) > v_m { true } else { v_tilde[i] };
    }

    // Eq. 114: total amplitude.
    let a_m: f64 = m_bar.iter().take(l).map(|&v| f64::from(v)).sum();

    // Eq. 115: τ_M recurrence.
    let tau_m = if epsilon_r <= 0.005 && epsilon_t <= 6 {
        20480.0
    } else {
        6000.0 - 300.0 * f64::from(epsilon_t) + tau_m_prev
    };

    // Eq. 116: γ_M scaling.
    let gamma_m: f64 = if tau_m > a_m { 1.0 } else if a_m > 0.0 { tau_m / a_m } else { 1.0 };

    let mut m_bar_out = [0f32; L_MAX as usize];
    for (i, &m) in m_bar.iter().enumerate() {
        m_bar_out[i] = ((f64::from(m)) * gamma_m) as f32;
    }

    SmoothedFrame { m_bar: m_bar_out, v_bar, tau_m }
}

// ---------------------------------------------------------------------------
// §1.12.1 — Unvoiced synthesis (Eq. 117–126)
// ---------------------------------------------------------------------------

/// Samples per 20 ms frame at 8 kHz (`N` in §1.12).
pub const FRAME_SAMPLES: usize = 160;

/// Initial state of the white-noise LCG: `u(−105) = 3147` per
/// BABA-A §10 Annex A.
pub const NOISE_INIT: u32 = 3147;

/// Placeholder for the unvoiced-synthesis spectral scale `γ_w`
/// (Eq. 121). The true value is a constant computed from the
/// synthesis window `wS` (Annex I) and the pitch refinement window
/// `wR` (Annex C). Annex C has not yet been extracted, so this
/// placeholder is used until either Annex C lands or a DVSI
/// fixture value is captured.
pub const PLACEHOLDER_GAMMA_W: f64 = 1.0;

/// Cross-frame state for the unvoiced synthesizer (§1.12.1).
#[derive(Clone, Debug)]
pub struct UnvoicedSynthState {
    /// LCG state. Initial value: 3147 (= `u(−105)`).
    lcg: u32,
    /// Most recent 209 noise samples covering the synthesis window
    /// `wS` over `n = −104..104`.
    noise_window: [f64; 209],
    /// Previous frame's IDFT output `ũ_w(n, −1)` for `n = −128..127`.
    /// Indexed `[n + 128]`.
    prev_idft: [f64; 256],
    /// `false` until the first frame has populated `noise_window`.
    initialized: bool,
}

impl UnvoicedSynthState {
    /// Cold-start state per §1.13 / Annex A: empty noise window,
    /// LCG seeded with `u(−105) = 3147`, prev IDFT all zero.
    pub fn new() -> Self {
        Self {
            lcg: NOISE_INIT,
            noise_window: [0.0; 209],
            prev_idft: [0.0; 256],
            initialized: false,
        }
    }

    /// Advance the LCG one step: `u(n+1) = (171·u(n) + 11213) mod 53125`.
    #[inline]
    fn next_noise(&mut self) -> f64 {
        self.lcg = (171u32
            .wrapping_mul(self.lcg)
            .wrapping_add(11213))
            % 53125;
        f64::from(self.lcg)
    }

    /// Per-frame: shift the noise window forward by N=160 samples.
    /// First call generates 209 fresh samples; subsequent calls reuse
    /// the trailing 49 samples and generate 160 new ones.
    fn advance_window(&mut self) {
        if !self.initialized {
            for i in 0..209 {
                self.noise_window[i] = self.next_noise();
            }
            self.initialized = true;
        } else {
            self.noise_window.copy_within(160..209, 0);
            for i in 49..209 {
                self.noise_window[i] = self.next_noise();
            }
        }
    }
}

impl Default for UnvoicedSynthState {
    fn default() -> Self {
        Self::new()
    }
}

/// 256-point complex DFT of a windowed real input over `n = −104..=104`.
/// Returns `(re, im)` arrays of length 256 indexed `[m + 128]` for
/// `m = −128..127`.
fn dft_256_windowed(input: &[f64; 209]) -> ([f64; 256], [f64; 256]) {
    let mut re = [0f64; 256];
    let mut im = [0f64; 256];
    for m_idx in 0..256 {
        let m = m_idx as i32 - 128;
        let mut acc_re = 0f64;
        let mut acc_im = 0f64;
        for n_idx in 0..209 {
            let n = n_idx as i32 - 104;
            let arg = -2.0 * PI64 * f64::from(m) * f64::from(n) / 256.0;
            acc_re += input[n_idx] * arg.cos();
            acc_im += input[n_idx] * arg.sin();
        }
        re[m_idx] = acc_re;
        im[m_idx] = acc_im;
    }
    (re, im)
}

/// 256-point inverse DFT producing real-valued output (Eq. 125).
/// The 1/256 normalization is included.
fn idft_256(re: &[f64; 256], im: &[f64; 256]) -> [f64; 256] {
    let mut out = [0f64; 256];
    for n_idx in 0..256 {
        let n = n_idx as i32 - 128;
        let mut acc = 0f64;
        for m_idx in 0..256 {
            let m = m_idx as i32 - 128;
            let arg = 2.0 * PI64 * f64::from(m) * f64::from(n) / 256.0;
            acc += re[m_idx] * arg.cos() - im[m_idx] * arg.sin();
        }
        out[n_idx] = acc / 256.0;
    }
    out
}

/// Apply the per-band spectral shaping (Eq. 119–124) in-place on the
/// DFT outputs.
fn shape_spectrum(
    re: &mut [f64; 256],
    im: &mut [f64; 256],
    omega_0: f64,
    m_bar: &[f32],
    v_bar: &[bool],
    gamma_w: f64,
) {
    let l_count = m_bar.len();
    let scale = 256.0 / (2.0 * PI64);

    // First, zero everything outside band 1..L̃ (Eq. 124). Band edges:
    //   ⌈ã_1⌉ at the low end, ⌈b̃_L̃⌉ at the high end.
    let a1 = (scale * 0.5 * omega_0).ceil() as i32;
    let b_last = (scale * (l_count as f64 + 0.5) * omega_0).ceil() as i32;
    for m_idx in 0..256 {
        let m = m_idx as i32 - 128;
        if m.unsigned_abs() < a1 as u32 || (m.unsigned_abs() as i32) >= b_last {
            re[m_idx] = 0.0;
            im[m_idx] = 0.0;
        }
    }

    // Pre-compute the unmodified spectrum power for each band's norm
    // sum, since we'll be overwriting `re`/`im` during the sweep.
    let mut band_norm: Vec<f64> = Vec::with_capacity(l_count);
    let mut band_edges: Vec<(i32, i32)> = Vec::with_capacity(l_count);
    for l in 1..=l_count as i32 {
        let l_f = f64::from(l);
        let a_l = (scale * (l_f - 0.5) * omega_0).ceil() as i32;
        let b_l = (scale * (l_f + 0.5) * omega_0).ceil() as i32;
        band_edges.push((a_l, b_l));
        // Norm sum: η in [⌈ã_l⌉, ⌈b̃_l⌉) (half-open, count = b - a).
        let mut norm_sum = 0f64;
        let count = (b_l - a_l).max(0) as usize;
        for eta in a_l..b_l {
            // Both +η and −η contribute (the spec's "|m|" means both signs).
            for &sign in &[1i32, -1] {
                let m_idx = (sign * eta + 128) as usize;
                if m_idx < 256 {
                    norm_sum += re[m_idx] * re[m_idx] + im[m_idx] * im[m_idx];
                }
            }
        }
        // Average over (count) magnitudes counted twice (positive + negative).
        let norm = if count > 0 && norm_sum > 0.0 {
            (norm_sum / (2.0 * count as f64)).sqrt()
        } else {
            1.0
        };
        band_norm.push(norm);
    }

    // Per-band assignment.
    for (l_idx, &(a_l, b_l)) in band_edges.iter().enumerate() {
        let voiced = v_bar[l_idx];
        if voiced {
            // Eq. 119: zero voiced bands.
            for m_abs in a_l..=b_l {
                for &sign in &[1i32, -1] {
                    let m = sign * m_abs;
                    let m_idx = (m + 128) as usize;
                    if m_idx < 256 {
                        re[m_idx] = 0.0;
                        im[m_idx] = 0.0;
                    }
                }
            }
        } else {
            // Eq. 120: scale by γ_w · M̄_l / norm.
            let m_bar_l = f64::from(m_bar[l_idx]);
            let norm = band_norm[l_idx];
            let factor = if norm > 0.0 {
                gamma_w * m_bar_l / norm
            } else {
                0.0
            };
            for m_abs in a_l..=b_l {
                for &sign in &[1i32, -1] {
                    let m = sign * m_abs;
                    let m_idx = (m + 128) as usize;
                    if m_idx < 256 {
                        re[m_idx] *= factor;
                        im[m_idx] *= factor;
                    }
                }
            }
        }
    }
}

/// Synthesize the unvoiced component for one 20 ms frame per BABA-A
/// §1.12.1 Eq. 117–126.
///
/// Returns 160 PCM-domain samples (in `f64`; the final cast to `i16`
/// happens after voiced + unvoiced sum at the synthesis top level).
///
/// `gamma_w` is the spectral scale from Eq. 121. Pass
/// [`PLACEHOLDER_GAMMA_W`] until Annex C lands or a DVSI fixture
/// value is available.
pub fn synthesize_unvoiced(
    omega_0: f32,
    m_bar: &[f32],
    v_bar: &[bool],
    gamma_w: f64,
    state: &mut UnvoicedSynthState,
) -> [f64; FRAME_SAMPLES] {
    state.advance_window();

    // Window the noise: u(n) · wS(n) for n = −104..104.
    let mut windowed = [0f64; 209];
    for i in 0..209 {
        let n = i as i32 - 104;
        windowed[i] = state.noise_window[i] * f64::from(synth_window(n));
    }

    let (mut re, mut im) = dft_256_windowed(&windowed);
    shape_spectrum(&mut re, &mut im, f64::from(omega_0), m_bar, v_bar, gamma_w);
    let u_w = idft_256(&re, &im);

    // Eq. 126: weighted overlap-add with previous frame.
    let mut out = [0f64; FRAME_SAMPLES];
    for n in 0..FRAME_SAMPLES as i32 {
        let ws_n = f64::from(synth_window(n));
        let ws_n_minus_n = f64::from(synth_window(n - FRAME_SAMPLES as i32));
        let prev_term = if (0..=127).contains(&n) {
            ws_n * state.prev_idft[(n + 128) as usize]
        } else {
            0.0
        };
        let curr_term = if (32..=159).contains(&n) {
            ws_n_minus_n * u_w[(n - 32) as usize]
        } else {
            0.0
        };
        let denom = ws_n * ws_n + ws_n_minus_n * ws_n_minus_n;
        out[n as usize] = if denom > 1e-30 {
            (prev_term + curr_term) / denom
        } else {
            0.0
        };
    }

    state.prev_idft = u_w;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_window_endpoints_are_zero() {
        // Per Annex I header, wS(±105) = 0.
        assert_eq!(synth_window(-105), 0.0);
        assert_eq!(synth_window(105), 0.0);
    }

    #[test]
    fn synth_window_symmetric() {
        for n in 1..=105 {
            assert_eq!(
                synth_window(-n),
                synth_window(n),
                "wS(-{n}) != wS({n})"
            );
        }
    }

    #[test]
    fn synth_window_peaks_near_zero() {
        // The window is a tapered shape that peaks near n=0. Verify
        // the centre is the maximum (or among the max set).
        let centre = synth_window(0);
        for n in -105..=105 {
            assert!(
                synth_window(n) <= centre + 1e-6,
                "wS({n}) = {} exceeds wS(0) = {centre}",
                synth_window(n)
            );
        }
    }

    #[test]
    fn synth_window_out_of_range_is_zero() {
        // Spec treats wS outside [-105, 105] as zero (used in the OLA
        // formula in §1.12).
        assert_eq!(synth_window(-106), 0.0);
        assert_eq!(synth_window(106), 0.0);
        assert_eq!(synth_window(1000), 0.0);
        assert_eq!(synth_window(-1000), 0.0);
    }

    #[test]
    fn synth_window_table_length_matches_constant() {
        assert_eq!(IMBE_SYNTH_WINDOW.len(), SYNTH_WINDOW_LEN);
        assert_eq!(SYNTH_WINDOW_LEN, 211);
    }

    // ---- §1.10 enhancement -----------------------------------------------

    fn sum_sq(arr: &[f32]) -> f64 {
        arr.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    }

    #[test]
    fn enhance_silence_passes_through_and_floors_s_e() {
        // All-zero M̃_l → R_M0 = 0 → silence-frame degenerate path.
        let m_tilde = vec![0.0f32; 30];
        let (m_bar, s_e) = enhance_spectral_amplitudes(&m_tilde, 0.1, INIT_S_E);
        for i in 0..30 {
            assert_eq!(m_bar[i], 0.0, "M̄_{i} = {}", m_bar[i]);
        }
        // S_E recurrence with R_M0 = 0: 0.95·75000 = 71250 (above floor).
        assert!((s_e - 71250.0).abs() < 1e-6, "S_E = {s_e}");
    }

    #[test]
    fn enhance_preserves_energy_within_a_few_percent() {
        // The Eq. 109–110 rescale is *exact* energy-preserving in
        // double precision before the f64 → f32 cast; we just check
        // the round-tripped energy matches within a small tolerance.
        let l = 24usize;
        let m_tilde: Vec<f32> = (1..=l).map(|i| (i as f32 * 0.3).sin().abs() + 0.1).collect();
        let omega_0 = std::f32::consts::PI / 12.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        let energy_in = sum_sq(&m_tilde);
        let energy_out = sum_sq(&m_bar[..l]);
        let rel_err = (energy_out - energy_in).abs() / energy_in;
        assert!(rel_err < 1e-4, "energy drift {rel_err} (in={energy_in}, out={energy_out})");
    }

    #[test]
    fn enhance_s_e_floor_engages() {
        // After many quiet frames, S_E should converge to the floor.
        let m_tilde = vec![0.0f32; 30];
        let mut s_e = INIT_S_E;
        for _ in 0..200 {
            let (_, new_s_e) = enhance_spectral_amplitudes(&m_tilde, 0.1, s_e);
            s_e = new_s_e;
        }
        assert!((s_e - S_E_FLOOR).abs() < 1e-6, "S_E = {s_e}");
    }

    #[test]
    fn enhance_s_e_recurrence_responds_to_input_energy() {
        // Pumping in higher-energy frames should drive S_E up from
        // its initial value (with the 0.05 mixing factor).
        let m_tilde: Vec<f32> = (1..=20).map(|_| 100.0).collect(); // R_M0 ≈ 200000
        let (_, s_e1) = enhance_spectral_amplitudes(&m_tilde, 0.1, INIT_S_E);
        // S_E = 0.95·75000 + 0.05·200000 = 71250 + 10000 = 81250
        assert!((s_e1 - 81250.0).abs() < 1e-2, "S_E = {s_e1}");
    }

    #[test]
    fn enhance_low_harmonic_bypass_holds_per_section_1_10() {
        // For l with 8·l ≤ L̃, the W_l weighting is bypassed (Eq. 108
        // first branch). The bypass gives M̄_l = M̃_l *before* the
        // global γ rescale. Verify the per-harmonic ratio M̄_l / M̃_l
        // is identical across all bypassed l (i.e., they all get the
        // same γ scaling and no per-l W weighting).
        let l = 56usize; // many harmonics so 8·l ≤ L̃ for several values
        let m_tilde: Vec<f32> = (1..=l).map(|i| 1.0 + (i as f32 * 0.05).sin()).collect();
        let omega_0 = std::f32::consts::PI / 28.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        // Bypassed harmonics: l = 1..=⌊L/8⌋ = 1..=7. They all share
        // the same final γ multiplier.
        let r0 = m_bar[0] / m_tilde[0];
        for i in 1..7 {
            let r = m_bar[i] / m_tilde[i];
            assert!(
                (r - r0).abs() < 1e-4,
                "bypassed harmonic l={}: ratio {r} vs {r0}",
                i + 1
            );
        }
    }

    // ---- §1.11 smoothing / disposition -----------------------------------

    fn err(epsilon_0: u8, epsilon_t: u8, epsilon_4: u8) -> FrameErrorContext {
        FrameErrorContext { epsilon_0, epsilon_4, epsilon_t, bad_pitch: false }
    }

    #[test]
    fn frame_disposition_clean_frame_uses() {
        let (d, er) = frame_disposition(&err(0, 0, 0), 0.0);
        assert_eq!(d, FrameDisposition::Use);
        assert_eq!(er, 0.0);
    }

    #[test]
    fn frame_disposition_bad_pitch_repeats() {
        let mut e = err(0, 0, 0);
        e.bad_pitch = true;
        let (d, _) = frame_disposition(&e, 0.0);
        assert_eq!(d, FrameDisposition::Repeat);
    }

    #[test]
    fn frame_disposition_joint_error_threshold_repeats() {
        // ε₀ ≥ 2 AND ε_T ≥ 10 + 40·ε_R(0). With ε_R(prev) = 0 and ε_T = 12,
        // ε_R(0) = 0.05·12/144 ≈ 0.00417. Threshold = 10 + 40·0.00417 ≈ 10.17.
        // ε_T = 12 ≥ 10.17 → repeat.
        let (d, _) = frame_disposition(&err(2, 12, 0), 0.0);
        assert_eq!(d, FrameDisposition::Repeat);
        // ε₀ = 1 below threshold → use.
        let (d, _) = frame_disposition(&err(1, 12, 0), 0.0);
        assert_eq!(d, FrameDisposition::Use);
    }

    #[test]
    fn frame_disposition_high_error_rate_mutes() {
        // ε_R prev high enough that the smoothed value crosses 0.0875.
        let (d, _) = frame_disposition(&err(0, 50, 0), 0.5);
        assert_eq!(d, FrameDisposition::Mute);
    }

    #[test]
    fn smoothing_low_error_uses_infinite_v_m() {
        // Branch 1: ε_R ≤ 0.005 AND ε_T ≤ 4 → V_M = ∞. Even very tall
        // M̄_l should not flip an unvoiced harmonic to voiced.
        let m_bar = vec![1e6f32; 9];
        let v_tilde = vec![false; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, 75000.0, 0.001, 2, 0, INIT_TAU_M);
        for i in 0..9 {
            assert!(!s.v_bar[i], "v̄_{i} flipped to voiced under V_M = ∞");
        }
    }

    #[test]
    fn smoothing_v_uv_override_when_amplitude_exceeds_threshold() {
        // Force branch 3 (V_M = 1.414 · S_E^0.375). With S_E small, V_M
        // is small; large M̄ should flip to voiced.
        let s_e = 100.0; // V_M = 1.414 · 100^0.375 ≈ 1.414 · 5.84 ≈ 8.27
        let m_bar = vec![1.0f32, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0];
        let v_tilde = vec![false; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, s_e, 0.05, 10, 1, INIT_TAU_M);
        for i in 0..9 {
            let expected = m_bar[i] > 8.27;
            assert_eq!(s.v_bar[i], expected, "v̄_{i}");
        }
    }

    #[test]
    fn smoothing_amplitude_low_error_resets_tau_m() {
        // Branch 1 of Eq. 115: ε_R ≤ 0.005 AND ε_T ≤ 6 → τ_M = 20480.
        let m_bar = vec![10.0f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.001, 3, 0, 5000.0);
        assert_eq!(s.tau_m, 20480.0);
    }

    #[test]
    fn smoothing_amplitude_recurrence_under_errors() {
        // Branch 2: τ_M(0) = 6000 − 300·ε_T + τ_M(−1)
        let m_bar = vec![10.0f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.05, 8, 1, 10000.0);
        // expected τ_M = 6000 − 300·8 + 10000 = 13600
        assert!((s.tau_m - 13600.0).abs() < 1e-6);
    }

    #[test]
    fn smoothing_gamma_m_is_one_when_tau_m_exceeds_amplitude_total() {
        // A_M = Σ M̄_l. Tiny amplitudes → A_M < τ_M → γ_M = 1 (no scaling).
        let m_bar = vec![0.01f32; 9];
        let s = apply_smoothing(&m_bar, &vec![true; 9], 1000.0, 0.001, 0, 0, INIT_TAU_M);
        for i in 0..9 {
            assert!((s.m_bar[i] - 0.01).abs() < 1e-6);
        }
    }

    // ---- §1.12.1 unvoiced synthesis --------------------------------------

    #[test]
    fn lcg_first_few_values_match_recurrence() {
        // u(n+1) = (171·u(n) + 11213) mod 53125, u(−105) = 3147.
        let mut state = UnvoicedSynthState::new();
        // First call: u(−104) = (171·3147 + 11213) mod 53125
        //           = (538137 + 11213) mod 53125
        //           = 549350 mod 53125
        //           = 549350 - 10·53125 = 549350 - 531250 = 18100
        let v1 = state.next_noise();
        assert_eq!(v1 as u32, 18100);
        // Second: u(−103) = (171·18100 + 11213) mod 53125
        //                 = (3095100 + 11213) mod 53125
        //                 = 3106313 mod 53125
        //                 = 3106313 - 58·53125 = 3106313 - 3081250 = 25063
        let v2 = state.next_noise();
        assert_eq!(v2 as u32, 25063);
    }

    #[test]
    fn dft_idft_roundtrip_recovers_input() {
        // A real input that fits in n=−104..=104, surrounded by zero.
        let mut input = [0f64; 209];
        for i in 0..209 {
            let n = i as i32 - 104;
            input[i] = (f64::from(n) * 0.05).sin() * 100.0;
        }
        let (re, im) = dft_256_windowed(&input);
        let recovered = idft_256(&re, &im);
        // The IDFT spans n=−128..127 (256 values), our input spans n=-104..=104.
        // Recovered values for n in [-104, 104] should match the input
        // at the same n-position. recovered[(n+128)] = input[(n+104)].
        for n in -104..=104i32 {
            let in_val = input[(n + 104) as usize];
            let out_val = recovered[(n + 128) as usize];
            assert!(
                (out_val - in_val).abs() < 1e-6,
                "n={n}: in={in_val}, out={out_val}"
            );
        }
    }

    #[test]
    fn unvoiced_silence_input_produces_silence_output() {
        let m_bar = vec![0.0f32; 12];
        let v_bar = vec![false; 12];
        let mut state = UnvoicedSynthState::new();
        let pcm = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        for (i, &s) in pcm.iter().enumerate() {
            assert!(s.abs() < 1e-6, "n={i}: {s}");
        }
    }

    #[test]
    fn unvoiced_all_voiced_first_frame_produces_silence_output() {
        // Voiced bands → Ũ_w = 0 in those bands; with no unvoiced bands
        // there's no signal. Previous IDFT is also zero (init), so OLA
        // output is silence.
        let m_bar = vec![10.0f32; 12];
        let v_bar = vec![true; 12];
        let mut state = UnvoicedSynthState::new();
        let pcm = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        for (i, &s) in pcm.iter().enumerate() {
            assert!(s.abs() < 1e-6, "n={i}: {s}");
        }
    }

    #[test]
    fn unvoiced_nonzero_input_produces_nonzero_output() {
        let m_bar = vec![10.0f32; 12];
        let v_bar = vec![false; 12];
        let mut state = UnvoicedSynthState::new();
        // First frame's OLA blends prev IDFT (zero) with current IDFT.
        // So we run two frames and check the second has non-trivial energy.
        let _ = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        let pcm = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        let energy: f64 = pcm.iter().map(|&s| s * s).sum();
        assert!(energy > 1e-3, "energy too low: {energy}");
    }

    #[test]
    fn unvoiced_state_advances_between_frames() {
        let m_bar = vec![1.0f32; 9];
        let v_bar = vec![false; 9];
        let mut state = UnvoicedSynthState::new();
        let lcg_init = state.lcg;
        let _ = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        let lcg_after_1 = state.lcg;
        assert_ne!(lcg_after_1, lcg_init);
        let _ = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        let lcg_after_2 = state.lcg;
        assert_ne!(lcg_after_2, lcg_after_1);
    }

    #[test]
    fn unvoiced_output_length_is_one_frame() {
        let m_bar = vec![1.0f32; 9];
        let v_bar = vec![false; 9];
        let mut state = UnvoicedSynthState::new();
        let pcm = synthesize_unvoiced(0.2, &m_bar, &v_bar, PLACEHOLDER_GAMMA_W, &mut state);
        assert_eq!(pcm.len(), FRAME_SAMPLES);
        assert_eq!(FRAME_SAMPLES, 160);
    }

    #[test]
    fn smoothing_gamma_m_clamps_loud_frames() {
        // Loud amplitudes → A_M > τ_M → γ_M = τ_M / A_M < 1, all M̄_l scaled
        // by the same γ_M.
        let m_bar = vec![10000.0f32; 9];
        let v_tilde = vec![true; 9];
        let s = apply_smoothing(&m_bar, &v_tilde, 1000.0, 0.001, 0, 0, INIT_TAU_M);
        // A_M = 90000, τ_M = 20480, γ_M ≈ 0.2276.
        let expected_gamma = 20480.0 / 90000.0;
        for i in 0..9 {
            let ratio = s.m_bar[i] / 10000.0;
            assert!(
                (ratio - expected_gamma as f32).abs() < 1e-3,
                "γ_M ratio at l={}: {ratio} vs {expected_gamma}",
                i + 1
            );
        }
    }

    #[test]
    fn enhance_clamps_w_l_within_brackets() {
        // For an extreme spectral shape we expect some W_l values to
        // hit the clamp (1.2 high, 0.5 low). Non-bypassed M̄_l / M̃_l
        // ratios must lie in [0.5·γ, 1.2·γ] for some shared γ.
        let l = 12usize; // small enough that 8·l > L̃ for all l ≥ 2
        let m_tilde: Vec<f32> = vec![1e-3, 10.0, 0.01, 5.0, 0.05, 3.0, 0.1, 2.0, 0.2, 1.5, 0.3, 1.0];
        let omega_0 = std::f32::consts::PI / 6.0;
        let (m_bar, _) = enhance_spectral_amplitudes(&m_tilde, omega_0, INIT_S_E);
        // All non-bypassed (l ≥ 2 since L=12 means only l=1 is bypassed)
        // ratios M̄/M̃ before γ would be in [0.5, 1.2]. After γ scaling
        // they all share the same γ. So the ratio of any two should be
        // bounded by (1.2/0.5) = 2.4.
        let mut ratios: Vec<f32> = (1..l).map(|i| m_bar[i] / m_tilde[i]).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let span = ratios[ratios.len() - 1] / ratios[0];
        assert!(span <= 2.4 + 1e-4, "ratio span {span} exceeds 2.4");
    }
}
