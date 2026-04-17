//! §0.1 — DC-blocking high-pass filter.
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.1 (BABA-A §5, Eq. 3).
//! Single-pole IIR run on the raw 16-bit PCM stream before any analysis:
//!
//!   H(z) = (1 − z⁻¹) / (1 − 0.99·z⁻¹)
//!   y[n] = x[n] − x[n−1] + 0.99·y[n−1]
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
