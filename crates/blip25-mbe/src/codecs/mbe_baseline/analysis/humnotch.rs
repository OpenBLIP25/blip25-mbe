//! Pre-analysis mains-hum notch front-end (§3.4 / §3.6 follow-up).
//!
//! A **separable, general-DSP** front-end that removes narrowband mains hum
//! (60/120 Hz in the US; 50/100 Hz in the EU) from the input PCM *before*
//! the codec sees it. Diagnosed root cause (chip probes, 2026-06-13): on
//! hum-laden speech the §0.3 pitch tracker locks onto the mains line — the
//! 60 Hz fundamental (and the 120 Hz line, which sits *inside* the pitch
//! grid as a valid male f0) pull the estimated f0 down to ~65 Hz on ~30 %
//! of voiced frames, which then floods the low-band amplitude and voicing.
//! The §0.1 single-pole DC blocker (corner ~13 Hz) is transparent to hum,
//! and a broadband high-pass cannot help (the pitch re-locks to 120 Hz,
//! which is un-removable without killing low male voices). Two **narrow**
//! biquad notches (Q ≈ 10, ~12 Hz −3 dB bandwidth) surgically null the two
//! lines while leaving the 200–3400 Hz voice band (and the 180/240 Hz
//! harmonics) untouched.
//!
//! All general DSP (RBJ band-reject biquad) — outside the TIA-102
//! clean-room scope. It does NOT modify any spec §0.x algorithm; it is a
//! new front-end stage. Opt-in, default OFF. Zero added latency (IIR).
//! When composed with [`super::predenoise::PreDenoise`], run the notch
//! FIRST (null the line, then suppress residual broadband).
//!
//! Chip-validated (encoder cell `our_enc_chip_dec`, PESQ-nb vs clean ref):
//! hum-15 dB 2.576 → 2.999 (+0.423, exceeds the chip bar 2.958);
//! hum-10 dB 2.429 → 3.021 (+0.592). Clean speech damage ≈ 0 (|Δf0| 0.11 Hz);
//! male-voice safe (`mark.pcm` median f0 89 Hz: 4.79 Hz perturbation, 0
//! octave locks).

const SAMPLE_RATE: f64 = 8000.0;

/// Default mains fundamental (US 60 Hz). The notch nulls this and its
/// second harmonic (2·f0).
pub const MAINS_HZ_US: f64 = 60.0;
/// EU mains fundamental (50 Hz).
pub const MAINS_HZ_EU: f64 = 50.0;
/// Notch quality factor. Q≈10 gives a ~12 Hz −3 dB bandwidth — narrow
/// enough to spare a male fundamental grazing 120 Hz, wide enough to catch
/// the line. (Q>15 rings; Q<6 widens the damage.)
pub const NOTCH_Q: f64 = 10.0;

/// One RBJ band-reject (notch) biquad, Direct-Form-II transposed.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// RBJ notch at `f0` Hz, quality `q` (Audio-EQ-Cookbook).
    fn notch(f0: f64, q: f64) -> Self {
        let w0 = 2.0 * core::f64::consts::PI * f0 / SAMPLE_RATE;
        let cs = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: 1.0 / a0,
            b1: -2.0 * cs / a0,
            b2: 1.0 / a0,
            a1: -2.0 * cs / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Cascade of mains-line notches (the fundamental + its second harmonic).
/// Streaming, persistent state, zero latency.
#[derive(Clone, Debug)]
pub struct HumNotch {
    stages: [Biquad; 2],
}

impl HumNotch {
    /// Notch at `mains_hz` and `2·mains_hz` with the default [`NOTCH_Q`].
    pub fn new(mains_hz: f64) -> Self {
        Self {
            stages: [
                Biquad::notch(mains_hz, NOTCH_Q),
                Biquad::notch(2.0 * mains_hz, NOTCH_Q),
            ],
        }
    }

    /// US 60/120 Hz notch (the default).
    pub fn new_60_120() -> Self {
        Self::new(MAINS_HZ_US)
    }

    /// EU 50/100 Hz notch.
    pub fn new_50_100() -> Self {
        Self::new(MAINS_HZ_EU)
    }

    /// Reset filter memory (matches `Vocoder::reset`).
    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
    }

    /// Notch one frame of i16 PCM. Returns the same number of samples
    /// (zero latency). Rounds + clamps back to i16.
    pub fn process_frame(&mut self, frame: &[i16]) -> Vec<i16> {
        frame
            .iter()
            .map(|&x| {
                let mut s = f64::from(x);
                for stage in &mut self.stages {
                    s = stage.step(s);
                }
                s.round().clamp(-32768.0, 32767.0) as i16
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, n: usize, amp: f64) -> Vec<i16> {
        (0..n)
            .map(|i| (amp * (2.0 * core::f64::consts::PI * freq * i as f64 / SAMPLE_RATE).sin()) as i16)
            .collect()
    }

    fn rms(x: &[i16]) -> f64 {
        let s: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        (s / x.len().max(1) as f64).sqrt()
    }

    /// A 60 Hz tone is deeply attenuated; a 120 Hz tone too.
    #[test]
    fn nulls_the_mains_lines() {
        for f in [60.0, 120.0] {
            let mut h = HumNotch::new_60_120();
            let x = tone(f, 4000, 8000.0);
            let y = h.process_frame(&x);
            // skip the filter ring-in.
            let att_db = 20.0 * (rms(&y[1000..]) / rms(&x[1000..])).log10();
            assert!(att_db < -25.0, "{f} Hz only attenuated {att_db:.1} dB");
        }
    }

    /// The voice band passes through essentially unchanged.
    #[test]
    fn passes_the_voice_band() {
        for f in [300.0, 1000.0, 2500.0] {
            let mut h = HumNotch::new_60_120();
            let x = tone(f, 4000, 8000.0);
            let y = h.process_frame(&x);
            let r = 20.0 * (rms(&y[1000..]) / rms(&x[1000..])).log10();
            assert!(r.abs() < 0.5, "{f} Hz voice band changed {r:.2} dB");
        }
    }

    /// 180/240 Hz harmonics (adjacent to the 120 Hz notch) survive.
    #[test]
    fn spares_low_voice_harmonics() {
        for f in [180.0, 240.0] {
            let mut h = HumNotch::new_60_120();
            let x = tone(f, 4000, 8000.0);
            let y = h.process_frame(&x);
            let r = 20.0 * (rms(&y[1000..]) / rms(&x[1000..])).log10();
            assert!(r.abs() < 1.0, "{f} Hz harmonic changed {r:.2} dB");
        }
    }

    #[test]
    fn emits_same_length_and_resets() {
        let mut h = HumNotch::new_60_120();
        let out = h.process_frame(&vec![123i16; 160]);
        assert_eq!(out.len(), 160);
        h.reset();
        assert_eq!(h.stages[0].z1, 0.0);
        assert_eq!(h.stages[1].z2, 0.0);
    }
}
