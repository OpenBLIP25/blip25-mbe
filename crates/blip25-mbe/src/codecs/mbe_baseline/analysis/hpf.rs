//! §0.1 — DC-blocking high-pass filter.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.1 (BABA-A §5, Eq. 3).
//! Single-pole IIR run on the raw 16-bit PCM stream before any analysis:
//!
//! ```text
//!   H(z) = (1 − z⁻¹) / (1 − 0.99·z⁻¹)
//!   y[n] = x[n] − x[n−1] + 0.99·y[n−1]
//! ```
//!
//! Pole at z = 0.99: slow-decay transient ~200 samples (25 ms) from cold
//! start; absorbed by the pitch tracker's own preroll (§0.3.7 / §0.9).

/// Pole of the Eq. 3 DC-blocking HPF.
pub const HPF_POLE: f64 = 0.99;

/// Persistent state for the Eq. 3 HPF: one input/output delay line.
#[derive(Clone, Copy, Debug, Default)]
pub struct HpfState {
    /// Previous input sample `x[n−1]`.
    prev_x: f64,
    /// Previous output sample `y[n−1]`.
    prev_y: f64,
}

impl HpfState {
    /// Cold-start: zero state per addendum §0.1.6 convention.
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_y: 0.0,
        }
    }

    /// Apply Eq. 3 to one sample: `y[n] = x[n] − x[n−1] + 0.99·y[n−1]`.
    #[inline]
    pub fn step(&mut self, x: f64) -> f64 {
        let y = x - self.prev_x + HPF_POLE * self.prev_y;
        self.prev_x = x;
        self.prev_y = y;
        y
    }

    /// Filter a block of 16-bit PCM samples into an f64 output buffer.
    /// Input and output lengths must match.
    pub fn run_pcm(&mut self, pcm: &[i16], out: &mut [f64]) {
        assert_eq!(pcm.len(), out.len());
        for (x, y) in pcm.iter().copied().zip(out.iter_mut()) {
            *y = self.step(f64::from(x));
        }
    }

    /// Filter a block of f64 samples in place.
    pub fn run_in_place(&mut self, samples: &mut [f64]) {
        for s in samples.iter_mut() {
            *s = self.step(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero input → zero output (HPF is LTI and has zero state at cold start).
    #[test]
    fn hpf_zero_input_is_zero_output() {
        let mut hpf = HpfState::new();
        for _ in 0..1000 {
            assert_eq!(hpf.step(0.0), 0.0);
        }
    }

    /// DC input → output decays to zero. With input `x = 1.0` forever,
    /// `y[n] = y[n−1] · 0.99 + (1 − 1) = 0.99·y[n−1]` after the first
    /// step. The first step is `y[0] = 1 − 0 + 0 = 1`, then `y[1] =
    /// 1 − 1 + 0.99·1 = 0.99`, `y[2] = 0.99² · 1 = 0.9801`, etc.
    /// After many samples the output should asymptote to zero.
    #[test]
    fn hpf_blocks_dc() {
        let mut hpf = HpfState::new();
        let mut last = 0.0;
        for i in 0..2000 {
            last = hpf.step(1.0);
            if i == 0 {
                assert!((last - 1.0).abs() < 1e-12, "y[0] = {last}");
            }
        }
        assert!(last.abs() < 1e-6, "DC not blocked: y[1999] = {last}");
    }

    /// Run_pcm promotes i16 samples to f64 correctly.
    #[test]
    fn hpf_run_pcm_matches_step_loop() {
        let pcm: Vec<i16> = (0..32).map(|i| (i * 100) as i16).collect();
        let mut expected = vec![0.0; pcm.len()];
        let mut hpf_a = HpfState::new();
        for (i, &x) in pcm.iter().enumerate() {
            expected[i] = hpf_a.step(f64::from(x));
        }
        let mut actual = vec![0.0; pcm.len()];
        let mut hpf_b = HpfState::new();
        hpf_b.run_pcm(&pcm, &mut actual);
        assert_eq!(actual, expected);
    }

    /// HPF is LTI: `filter(a·x + b·y) = a·filter(x) + b·filter(y)`,
    /// subject to matched initial state.
    #[test]
    fn hpf_is_linear_and_time_invariant() {
        let x: Vec<f64> = (0..200).map(|n| (n as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..200).map(|n| (n as f64 * 0.05).cos()).collect();
        let mut a_buf = x.clone();
        let mut b_buf = y.clone();
        HpfState::new().run_in_place(&mut a_buf);
        HpfState::new().run_in_place(&mut b_buf);
        let mut combined: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| 2.0 * xi + 3.0 * yi)
            .collect();
        HpfState::new().run_in_place(&mut combined);
        for (i, (&c, (&a, &b))) in combined
            .iter()
            .zip(a_buf.iter().zip(b_buf.iter()))
            .enumerate()
        {
            let expected = 2.0 * a + 3.0 * b;
            assert!(
                (c - expected).abs() < 1e-9,
                "LTI violated at i={i}: {c} vs {expected}"
            );
        }
    }
}
