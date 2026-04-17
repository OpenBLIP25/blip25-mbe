//! §0.2 — 256-point signal DFT and the S_w(m, ω₀) harmonic basis.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.2 (BABA-A §5.1.5,
//! Equations 24–30). The synthetic spectrum S_w(m, ω₀) and its
//! per-harmonic amplitude A_l(ω₀) are consumed by:
//!
//!   §0.4 pitch refinement — minimizes E_R(ω₀) = Σ |S_w(m) − S_w(m, ω₀)|²
//!   §0.5 spectral amplitude estimation — Eq. 43/44 use |S_w(m)|² and
//!                                         the bin endpoints ⌈a_l⌉, ⌈b_l⌉
//!   §0.7 V/UV discriminant — the voicing measure D_k compares
//!                             |S_w(m) − A_l·W_R(·)|² against |S_w(m)|²

use super::refinement_window;

/// Size of the per-frame signal DFT (Eq. 29).
pub const DFT_SIZE: usize = 256;

/// Size of the window-only DFT (Eq. 30).
pub const WINDOW_DFT_SIZE: usize = 16384;

/// Support of Annex C window `w_R(n)`: `n ∈ [−W_R_HALF, W_R_HALF]`.
pub const W_R_HALF: i32 = 110;

/// Minimal complex number used by the analysis DFT path.
///
/// Holds real and imaginary components as `f64`. Implemented inline to
/// avoid a numerics dependency; the three operations we actually need
/// (scaled accumulation against a real window coefficient, magnitude,
/// scalar divide) are spelled out at their call sites rather than
/// wrapped in an arithmetic-operator trait soup.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// The additive identity.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Construct from real and imaginary parts.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Squared magnitude `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Precomputed `W_R(k)` table — the 16384-point DFT of the Annex C
/// window (Eq. 30). Because `w_R(n)` is real and symmetric, `W_R(k)`
/// is real-valued and symmetric in `k`; we store only `k ≥ 0` and mirror
/// on access.
///
/// Construction cost: a single 16384 × 221 loop at startup
/// (~3.6 M multiplies). Memory: `8193 × 8 B = 64 KiB`.
#[derive(Clone, Debug)]
pub struct HarmonicBasis {
    /// `W_R(k)` for `k ∈ [0, WINDOW_DFT_SIZE/2]`. Access via [`Self::w_r`].
    w_r_pos: Box<[f64; WINDOW_DFT_SIZE / 2 + 1]>,
}

impl HarmonicBasis {
    /// Build the `W_R(k)` table by direct evaluation of Eq. 30. O(N²)
    /// but runs once at startup.
    pub fn new() -> Self {
        let mut table = Box::new([0.0f64; WINDOW_DFT_SIZE / 2 + 1]);
        let two_pi = 2.0 * core::f64::consts::PI;
        let dft_size = WINDOW_DFT_SIZE as f64;
        for (k, slot) in table.iter_mut().enumerate() {
            let k_f = k as f64;
            let mut acc = 0.0;
            // w_R is symmetric; fold the n = +i / n = −i pairs to halve
            // the cosine evaluations. The n = 0 term is w_R(0) = 1.0.
            acc += f64::from(refinement_window(0));
            for n in 1..=W_R_HALF {
                let w = f64::from(refinement_window(n));
                if w == 0.0 {
                    continue;
                }
                let angle = two_pi * k_f * f64::from(n) / dft_size;
                acc += 2.0 * w * angle.cos();
            }
            *slot = acc;
        }
        Self { w_r_pos: table }
    }

    /// `W_R(k)` for any integer `k`. Returns `0.0` outside the
    /// `[−WINDOW_DFT_SIZE/2, WINDOW_DFT_SIZE/2]` stored range (in
    /// practice only `|k| ≲ 1024` is ever indexed per addendum
    /// §0.2.8; the wide bound is the PDF definition).
    #[inline]
    pub fn w_r(&self, k: i32) -> f64 {
        let abs = k.unsigned_abs() as usize;
        if abs >= self.w_r_pos.len() {
            0.0
        } else {
            self.w_r_pos[abs]
        }
    }

    /// Bin endpoints `⌈a_l⌉`, `⌈b_l⌉` for harmonic `l` at fundamental
    /// `ω₀` (Eq. 26–27). The support is the half-open integer interval
    /// `⌈a_l⌉ ≤ m < ⌈b_l⌉`.
    #[inline]
    pub fn bin_endpoints(l: u32, omega0: f64) -> (i32, i32) {
        let scale = DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let a_l = scale * (f64::from(l) - 0.5) * omega0;
        let b_l = scale * (f64::from(l) + 0.5) * omega0;
        (ceil_i32(a_l), ceil_i32(b_l))
    }

    /// Harmonic amplitude `A_l(ω₀)` per Eq. 28. The numerator is
    /// complex (it inherits the phase of `S_w(m)`); the denominator is
    /// real (sum of squared window-spectrum magnitudes over the
    /// harmonic's support).
    ///
    /// Returns [`Complex64::ZERO`] when the denominator is zero
    /// (empty support — e.g. a degenerate `ω₀` that collapses
    /// `⌈a_l⌉ = ⌈b_l⌉`). This is a numerical guard, not an expected
    /// condition for admissible `ω₀ ∈ [2π/123.125, 2π/19.875]`.
    pub fn harmonic_amplitude(
        &self,
        sw: &[Complex64; DFT_SIZE],
        l: u32,
        omega0: f64,
    ) -> Complex64 {
        let (m_lo, m_hi) = Self::bin_endpoints(l, omega0);
        let bin_scale = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let l_offset = bin_scale * f64::from(l) * omega0;
        let mut num_re = 0.0;
        let mut num_im = 0.0;
        let mut den = 0.0;
        for m in m_lo..m_hi {
            let k = floor_i32(64.0 * f64::from(m) - l_offset + 0.5);
            let wr = self.w_r(k);
            let s = sw[packed_index(m)];
            // W_R is real, so W_R* = W_R; Eq. 28's conjugate is a no-op.
            num_re += s.re * wr;
            num_im += s.im * wr;
            den += wr * wr;
        }
        if den > 0.0 {
            Complex64::new(num_re / den, num_im / den)
        } else {
            Complex64::ZERO
        }
    }

    /// Evaluate the synthetic spectrum `S_w(m, ω₀)` at a single
    /// `(m, l)` point per Eq. 25. The caller supplies the precomputed
    /// `A_l(ω₀)` (from [`Self::harmonic_amplitude`]). `m` must be in
    /// the harmonic's support `⌈a_l⌉ ≤ m < ⌈b_l⌉`; outside that, the
    /// PDF's synthetic spectrum is not defined (other harmonics cover
    /// adjacent bins).
    #[inline]
    pub fn synthetic_bin(&self, m: i32, l: u32, omega0: f64, a_l: Complex64) -> Complex64 {
        let bin_scale = WINDOW_DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        let k = floor_i32(64.0 * f64::from(m) - bin_scale * f64::from(l) * omega0 + 0.5);
        let wr = self.w_r(k);
        Complex64::new(a_l.re * wr, a_l.im * wr)
    }
}

impl Default for HarmonicBasis {
    fn default() -> Self {
        Self::new()
    }
}

/// 256-point DFT of the windowed input signal (Eq. 29).
///
/// `signal_centered` holds `s(−110), s(−109), …, s(110)` (221 samples)
/// — the portion of the analysis stream aligned with the current
/// frame's refinement window per addendum §0.1. Returns `S_w(m)` for
/// the two-sided index range `m ∈ [−127, 128]`, packed into a 256-entry
/// array via DFT periodicity: index `0..=128` holds `m = 0..=128`,
/// index `129..=255` holds `m = −127..=−1` (i.e. index `i` maps to
/// `m = i` if `i ≤ 128`, else `m = i − 256`). Use [`packed_index`] to
/// translate a two-sided `m` into its array slot.
///
/// Direct-form DFT; O(N·W) = 256·221 ≈ 56 k multiplies per frame.
/// Fast enough for correctness-first development and leaves the door
/// open for a future FFT-based reimplementation without changing the
/// signature.
pub fn signal_spectrum(signal_centered: &[f64; (2 * W_R_HALF + 1) as usize]) -> [Complex64; DFT_SIZE] {
    let mut out = [Complex64::ZERO; DFT_SIZE];
    let two_pi = 2.0 * core::f64::consts::PI;
    let dft_size = DFT_SIZE as f64;
    for (idx, slot) in out.iter_mut().enumerate() {
        let m = unpacked_m(idx);
        let m_f = f64::from(m);
        let mut re = 0.0;
        let mut im = 0.0;
        for n in -W_R_HALF..=W_R_HALF {
            let w = f64::from(refinement_window(n));
            if w == 0.0 {
                continue;
            }
            let s = signal_centered[(n + W_R_HALF) as usize];
            let angle = -two_pi * m_f * f64::from(n) / dft_size;
            re += s * w * angle.cos();
            im += s * w * angle.sin();
        }
        *slot = Complex64::new(re, im);
    }
    out
}

/// Map a two-sided DFT index `m ∈ [−127, 128]` to its packed slot in
/// `[0, 256)`, per the convention in [`signal_spectrum`].
#[inline]
pub fn packed_index(m: i32) -> usize {
    if m < 0 {
        (m + DFT_SIZE as i32) as usize
    } else {
        (m as usize) & (DFT_SIZE - 1)
    }
}

/// Inverse of [`packed_index`] for `idx ∈ [0, 256)`: returns `m`
/// in the two-sided range. Slots `0..=128` map to positive `m`;
/// slots `129..=255` map to `m − 256`.
#[inline]
pub(super) fn unpacked_m(idx: usize) -> i32 {
    if idx <= DFT_SIZE / 2 {
        idx as i32
    } else {
        idx as i32 - DFT_SIZE as i32
    }
}

#[inline]
pub(super) fn ceil_i32(x: f64) -> i32 {
    x.ceil() as i32
}

#[inline]
pub(super) fn floor_i32(x: f64) -> i32 {
    x.floor() as i32
}
