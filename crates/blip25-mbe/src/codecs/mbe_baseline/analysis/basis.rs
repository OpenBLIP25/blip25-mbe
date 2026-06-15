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

    /// Fractional (continuous) band edges `a_l`, `b_l` in DFT-bin units
    /// for harmonic `l` at fundamental `ω₀` — the un-rounded inputs to
    /// [`Self::bin_endpoints`]. Used by the `BLIP25_BIN_EDGE=frac` amplitude
    /// integration to weight boundary bins by their coverage of the
    /// band `[a_l, b_l)` instead of the ceil-to-ceil integer rule (which
    /// drops the partial low-edge bin and loses band energy unevenly
    /// across `l` — the L-dependent gain bias).
    #[inline]
    pub fn bin_edges_frac(l: u32, omega0: f64) -> (f64, f64) {
        let scale = DFT_SIZE as f64 / (2.0 * core::f64::consts::PI);
        (
            scale * (f64::from(l) - 0.5) * omega0,
            scale * (f64::from(l) + 0.5) * omega0,
        )
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
/// Backed by [`rustfft`] — `O(N log N)` per frame. The previous
/// direct-form `O(N · W)` version (~56 k multiplies / frame) is
/// preserved as the spec-correctness reference in the test below.
/// FFT is general DSP, not P25 IP, so the swap is allowed under the
/// clean-room rule.
pub fn signal_spectrum(signal_centered: &[f64; (2 * W_R_HALF + 1) as usize]) -> [Complex64; DFT_SIZE] {
    use std::sync::OnceLock;
    use num_complex::Complex;
    use rustfft::{Fft, FftPlanner};

    static FFT: OnceLock<std::sync::Arc<dyn Fft<f64>>> = OnceLock::new();
    let fft = FFT.get_or_init(|| {
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(DFT_SIZE)
    });

    // Build the windowed signal in DFT-natural ordering: x[n] for
    // n ∈ [0, N) where x[(n + N) % N] = signal(n) · w_R(n) and the
    // logical n ∈ [-W_R_HALF, W_R_HALF] of the input. Outside that
    // range the window is zero, so the corresponding x[k] are zero.
    let mut buf = [Complex::<f64>::new(0.0, 0.0); DFT_SIZE];
    for n in -W_R_HALF..=W_R_HALF {
        let w = f64::from(refinement_window(n));
        if w == 0.0 {
            continue;
        }
        let s = signal_centered[(n + W_R_HALF) as usize];
        let idx = ((n.rem_euclid(DFT_SIZE as i32)) as usize) % DFT_SIZE;
        buf[idx] = Complex::new(s * w, 0.0);
    }

    fft.process(&mut buf);

    let mut out = [Complex64::ZERO; DFT_SIZE];
    for (slot, c) in out.iter_mut().zip(buf.iter()) {
        *slot = Complex64::new(c.re, c.im);
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
///
/// Public for diagnostic / probe use; the FFT-backed
/// [`signal_spectrum`] doesn't need it but the direct-form reference
/// in the tests does.
#[inline]
pub fn unpacked_m(idx: usize) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn bin_endpoints_match_addendum_numerical_example() {
        let omega0 = 2.0 * PI / 50.0;
        assert_eq!(HarmonicBasis::bin_endpoints(0, omega0), (-2, 3));
        assert_eq!(HarmonicBasis::bin_endpoints(1, omega0), (3, 8));
        assert_eq!(HarmonicBasis::bin_endpoints(2, omega0), (8, 13));
    }

    /// Per Eq. 27 / Eq. 26, adjacent harmonics share a bin endpoint
    /// (`b_l = a_{l+1}` exactly), so the supports are contiguous and
    /// non-overlapping in the integer grid.
    #[test]
    fn bin_endpoints_are_contiguous_across_harmonics() {
        for omega_denom in [20i32, 50, 100, 123] {
            let omega0 = 2.0 * PI / f64::from(omega_denom);
            for l in 0..10 {
                let (_, b_this) = HarmonicBasis::bin_endpoints(l, omega0);
                let (a_next, _) = HarmonicBasis::bin_endpoints(l + 1, omega0);
                assert_eq!(
                    b_this, a_next,
                    "ω₀ = 2π/{omega_denom}, harmonic {l}: b_l = {b_this}, a_(l+1) = {a_next}"
                );
            }
        }
    }

    /// `W_R(0) = Σ w_R(n)` falls out of Eq. 30 directly (cosine with
    /// `k = 0` is identically 1). Sanity check the precomputed table.
    #[test]
    fn w_r_zero_equals_window_sum() {
        let basis = HarmonicBasis::new();
        let expected: f64 = (-W_R_HALF..=W_R_HALF)
            .map(|n| f64::from(refinement_window(n)))
            .sum();
        assert!(
            (basis.w_r(0) - expected).abs() < 1e-9,
            "W_R(0) = {}, expected {}",
            basis.w_r(0),
            expected
        );
    }

    /// The window is real and symmetric so `W_R(k) = W_R(−k)`.
    #[test]
    fn w_r_is_symmetric_in_k() {
        let basis = HarmonicBasis::new();
        for k in [1, 7, 64, 100, 500, 1023, 8000] {
            assert_eq!(basis.w_r(k), basis.w_r(-k), "W_R({k}) != W_R(−{k})");
        }
    }

    /// Outside the stored range the accessor clamps to zero.
    #[test]
    fn w_r_returns_zero_out_of_range() {
        let basis = HarmonicBasis::new();
        assert_eq!(basis.w_r(WINDOW_DFT_SIZE as i32), 0.0);
        assert_eq!(basis.w_r(-(WINDOW_DFT_SIZE as i32)), 0.0);
    }

    #[test]
    fn packed_index_roundtrips_two_sided_range() {
        for m in -127..=128 {
            let idx = packed_index(m);
            assert!(idx < DFT_SIZE, "packed_index({m}) out of bounds");
            assert_eq!(unpacked_m(idx), m, "roundtrip failed for m = {m}");
        }
    }

    /// Direct-form reference DFT for the FFT-backed
    /// [`signal_spectrum`] to be asserted against. O(N·W) per Eq. 29
    /// — the formulation the spec describes literally. Kept in the
    /// test module to verify the FFT path produces the same output
    /// to floating-point tolerance.
    fn signal_spectrum_direct_reference(
        signal_centered: &[f64; (2 * W_R_HALF + 1) as usize],
    ) -> [Complex64; DFT_SIZE] {
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

    /// FFT-backed `signal_spectrum` matches the direct-form reference
    /// for a non-trivial input to within float-summation tolerance.
    #[test]
    fn signal_spectrum_fft_matches_direct_reference() {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        // A mix of two slow-changing tones and a small DC offset —
        // exercises both real and imaginary spectrum components.
        let two_pi = 2.0 * core::f64::consts::PI;
        for (i, slot) in signal.iter_mut().enumerate() {
            let n = i as f64 - W_R_HALF as f64;
            *slot = 100.0 + 1000.0 * (two_pi * n / 50.0).sin()
                + 500.0 * (two_pi * n / 23.5).cos();
        }
        let fft = signal_spectrum(&signal);
        let dir = signal_spectrum_direct_reference(&signal);
        // Per-bin tolerance: 1e-6 absolute / relative (FFT and the
        // direct sum differ only by float rounding order).
        for (k, (a, b)) in fft.iter().zip(dir.iter()).enumerate() {
            let de = (a.re - b.re).abs();
            let di = (a.im - b.im).abs();
            let scale = a.re.abs().max(b.re.abs()).max(1.0);
            assert!(
                de / scale < 1e-9 && di / scale < 1e-9,
                "bin {k} mismatch: FFT={a:?}, direct={b:?}, |Δre|={de}, |Δim|={di}"
            );
        }
    }

    /// DFT of the all-zero signal is all-zero.
    #[test]
    fn signal_spectrum_of_zero_is_zero() {
        let signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        for (i, s) in sw.iter().enumerate() {
            assert!(s.norm_sqr() < 1e-18, "S_w[{i}] = ({}, {})", s.re, s.im);
        }
    }

    /// For the constant signal `s(n) = 1`, `S_w(0) = Σ w_R(n) = W_R(0)`.
    /// A cheap invariant that catches window-indexing mistakes.
    #[test]
    fn signal_spectrum_of_constant_matches_w_r_at_dc() {
        let signal = [1.0f64; (2 * W_R_HALF + 1) as usize];
        let sw = signal_spectrum(&signal);
        let basis = HarmonicBasis::new();
        let s0 = sw[packed_index(0)];
        assert!(
            (s0.re - basis.w_r(0)).abs() < 1e-9 && s0.im.abs() < 1e-9,
            "S_w(0) = ({}, {}), expected ({}, 0)",
            s0.re,
            s0.im,
            basis.w_r(0)
        );
    }

    fn cosine_tone(omega0: f64) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega0 * f64::from(n)).cos();
        }
        signal
    }

    fn sine_tone(omega0: f64) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (omega0 * f64::from(n)).sin();
        }
        signal
    }

    /// For `s(n) = cos(ω₀·n)`, the Eq. 28 projection at the fundamental
    /// should recover `A_1(ω₀) ≈ 1/2 + 0j` — the `1/2` comes from
    /// Euler's identity (`cos = (e^{jω₀n} + e^{−jω₀n})/2`), and Eq. 28
    /// is a least-squares projection onto the window-shifted basis so
    /// the window-energy denominator normalizes the response to the
    /// cosine coefficient. Tolerance is loose enough to absorb the
    /// small image spillover from the `e^{−jω₀n}` term at `−ω₀`.
    #[test]
    fn harmonic_projection_recovers_cosine_amplitude() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = cosine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        assert!(
            (a1.re - 0.5).abs() < 0.01 && a1.im.abs() < 0.01,
            "A_1 = ({}, {}), expected ≈ (0.5, 0)",
            a1.re,
            a1.im
        );
    }

    /// For `s(n) = sin(ω₀·n)`, `A_1(ω₀) ≈ 0 − j/2`: the `1/(2j)` from
    /// Euler contributes a pure-imaginary projection.
    #[test]
    fn harmonic_projection_recovers_sine_amplitude() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = sine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        assert!(
            a1.re.abs() < 0.01 && (a1.im + 0.5).abs() < 0.01,
            "A_1 = ({}, {}), expected ≈ (0, −0.5)",
            a1.re,
            a1.im
        );
    }

    /// A 2nd-harmonic cosine projects onto `A_2(ω₀) ≈ 1/2` and onto
    /// `A_1(ω₀) ≈ 0` (no fundamental present). Exercises the `l·ω₀`
    /// offset in the `W_R` index.
    #[test]
    fn harmonic_projection_isolates_second_harmonic() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let mut signal = [0.0f64; (2 * W_R_HALF + 1) as usize];
        for n in -W_R_HALF..=W_R_HALF {
            signal[(n + W_R_HALF) as usize] = (2.0 * omega0 * f64::from(n)).cos();
        }
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        let a2 = basis.harmonic_amplitude(&sw, 2, omega0);
        assert!(
            a1.re.abs() < 0.05 && a1.im.abs() < 0.05,
            "A_1 leak = ({}, {})",
            a1.re,
            a1.im
        );
        assert!(
            (a2.re - 0.5).abs() < 0.01 && a2.im.abs() < 0.01,
            "A_2 = ({}, {}), expected ≈ (0.5, 0)",
            a2.re,
            a2.im
        );
    }

    /// `synthetic_bin` reconstructs `A_l · W_R(k)` on each bin. Given
    /// `A_l(ω₀)` from Eq. 28, the reconstructed `S_w(m, ω₀)` must match
    /// the signal DFT `S_w(m)` closely on a clean sinusoid (the whole
    /// point of the projection).
    #[test]
    fn synthetic_bin_reconstructs_clean_sinusoid() {
        let basis = HarmonicBasis::new();
        let omega0 = 2.0 * PI / 50.0;
        let signal = cosine_tone(omega0);
        let sw = signal_spectrum(&signal);
        let a1 = basis.harmonic_amplitude(&sw, 1, omega0);
        let (m_lo, m_hi) = HarmonicBasis::bin_endpoints(1, omega0);
        let mut residual_energy = 0.0;
        let mut signal_energy = 0.0;
        for m in m_lo..m_hi {
            let observed = sw[packed_index(m)];
            let synthetic = basis.synthetic_bin(m, 1, omega0, a1);
            let dre = observed.re - synthetic.re;
            let dim = observed.im - synthetic.im;
            residual_energy += dre * dre + dim * dim;
            signal_energy += observed.norm_sqr();
        }
        // Well under 1% relative residual on a pure cosine.
        assert!(
            residual_energy / signal_energy.max(1e-12) < 0.01,
            "residual/signal = {}, expected < 0.01",
            residual_energy / signal_energy.max(1e-12)
        );
    }

    /// Zero-denominator guard: when the harmonic's support is empty
    /// (degenerate `ω₀ = 0` collapses every `⌈a_l⌉ = ⌈b_l⌉ = 0`),
    /// `harmonic_amplitude` returns `Complex64::ZERO` instead of NaN.
    #[test]
    fn harmonic_amplitude_empty_support_returns_zero() {
        let basis = HarmonicBasis::new();
        let sw = [Complex64::new(1.0, 1.0); DFT_SIZE];
        let a0 = basis.harmonic_amplitude(&sw, 0, 0.0);
        assert_eq!(a0, Complex64::ZERO);
    }
}
