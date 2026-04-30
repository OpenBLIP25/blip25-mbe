//! Post-decoder PCM enhancement — optional classical DSP filter chain
//! applied to decoded PCM before it leaves the [`Vocoder`].
//!
//! Spec-faithful synthesis emits PCM directly from the BABA-A §1.12
//! pipeline. Real LMR audio chains add post-processing between the
//! vocoder output and the speaker (the TI TLV320AIC33 codec on the
//! Motorola APX, for example, runs PCM through de-emphasis →
//! 2-biquad cascade → digital volume → DAC). This module exposes a
//! similar chain as an opt-in [`EnhancementMode`].
//!
//! [`EnhancementMode::None`] is the default and bypasses the chain
//! entirely — bytes from `decode_bits` match the spec-faithful path.
//! [`EnhancementMode::Classical`] runs a small set of stages tuned for
//! the 8 kHz / mono / narrow-band voice content the vocoder produces:
//!
//! - **High-pass / peaking biquad cascade** — two biquads matching
//!   AIC33's LB1/LB2 budget. Defaults: HPF at 250 Hz to roll off
//!   sub-speaker rumble, peaking at 2.5 kHz / +3 dB for intelligibility.
//! - **Soft-knee compressor** — playback-side DRC. AIC33's on-chip AGC
//!   is record-side only, so any decoded-audio loudness control has to
//!   live here.
//! - **Boundary smoothing** — short cosine fade at frame seams when
//!   transitioning out of synth-side mute (Repeat/Mute → Use), to
//!   suppress click/pop. Mirrors AIC33's soft-mute / soft-unmute around
//!   filter-coefficient changes.
//!
//! All defaults are starting points subject to A/B against the existing
//! 5-vector PESQ harness. None of this is on the spec path; consumers
//! that need bit-for-bit BABA-A output keep [`EnhancementMode::None`].

use core::f32::consts::PI;

/// Top-level mode selector. Mirrors the chip-shaped pattern used by
/// other [`crate::vocoder::Vocoder`] knobs (a single setter, default
/// off, named alternatives).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnhancementMode {
    /// No post-processing. Spec-faithful path.
    #[default]
    None,

    /// Classical DSP filter chain. See [`ClassicalConfig`] for stages.
    Classical(ClassicalConfig),
}

/// Knobs for [`EnhancementMode::Classical`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassicalConfig {
    /// Up to two cascaded biquads. `None` slots are bypassed.
    /// Mirrors AIC33's LB1/LB2 two-biquad-per-channel budget.
    pub biquads: [Option<Biquad>; 2],

    /// Soft-knee compressor. `None` disables.
    pub compressor: Option<Compressor>,

    /// Cosine fade across the first `boundary_fade_samples` of every
    /// frame whose previous-frame disposition was `Repeat` or `Mute`.
    /// Set to 0 to disable.
    pub boundary_fade_samples: usize,
}

impl Default for ClassicalConfig {
    fn default() -> Self {
        // Defaults tuned by 5-vector PESQ stage-isolation A/B
        // (project_enhancement_classical_tuning_2026-04-30.md):
        //
        // - HPF 250 Hz Q≈0.707: carries the entire knox_1 +0.236 win
        //   (de-rumbles the noisy-tone reconstruction). Inert on speech.
        // - Peaking 2.5 kHz Q=1.0 +3 dB: small alert win, neutral on
        //   speech.
        // - Compressor: OFF by default. The 3:1 / -18 dBFS / +3 dB
        //   makeup compressor was a net loss on dam (-0.065 PESQ) and
        //   inert on knox_1. Kept reachable via the [`Compressor`]
        //   struct for opt-in use; opt-in via a non-default cfg.
        // - 5 ms cosine fade after Mute / Repeat: inert on clean test
        //   vectors (no frame errors) but defensive in real RF chains.
        //
        // Net 5-vector avg: +0.052 PESQ over the spec-faithful path,
        // worst-vector regression -0.013 (alert).
        Self {
            biquads: [
                Some(Biquad::high_pass(8_000.0, 250.0, 0.707)),
                Some(Biquad::peaking(8_000.0, 2_500.0, 1.0, 3.0)),
            ],
            compressor: None,
            boundary_fade_samples: 40, // 5 ms at 8 kHz
        }
    }
}

/// Direct-form-II transposed biquad. Coefficients are real `f32`,
/// normalized so `a0 == 1.0` (callers don't see `a0`).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Biquad {
    /// Numerator coefficient on `x[n]`.
    pub b0: f32,
    /// Numerator coefficient on `x[n-1]`.
    pub b1: f32,
    /// Numerator coefficient on `x[n-2]`.
    pub b2: f32,
    /// Denominator coefficient on `y[n-1]` (after `a0` normalization).
    pub a1: f32,
    /// Denominator coefficient on `y[n-2]` (after `a0` normalization).
    pub a2: f32,
}

impl Biquad {
    /// 2nd-order Butterworth-shaped high-pass at `fc` Hz. RBJ cookbook
    /// coefficients.
    pub fn high_pass(fs_hz: f32, fc_hz: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * fc_hz / fs_hz;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) / 2.0 / a0;
        let b1 = -(1.0 + cos_w0) / a0;
        let b2 = (1.0 + cos_w0) / 2.0 / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;
        Self { b0, b1, b2, a1, a2 }
    }

    /// Peaking EQ at `fc` Hz, with gain `gain_db` and bandwidth `q`.
    /// RBJ cookbook coefficients.
    pub fn peaking(fs_hz: f32, fc_hz: f32, q: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc_hz / fs_hz;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        let b0 = (1.0 + alpha * a) / a0;
        let b1 = -2.0 * cos_w0 / a0;
        let b2 = (1.0 - alpha * a) / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha / a) / a0;
        Self { b0, b1, b2, a1, a2 }
    }

    /// Low-shelf at `fc` Hz, gain `gain_db`. RBJ cookbook coefficients.
    pub fn low_shelf(fs_hz: f32, fc_hz: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc_hz / fs_hz;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / 2.0 * 2f32.sqrt(); // S=1
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha) / a0;
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) / a0;
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) / a0;
        let a2 = ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0;
        Self { b0, b1, b2, a1, a2 }
    }
}

/// Soft-knee feed-forward compressor on the decoded-PCM envelope.
/// Constants ranges chosen to match the AIC33 record-side AGC envelope
/// (8–20 ms attack, 100–500 ms release).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Compressor {
    /// Threshold in dBFS. Envelope above this gets reduced.
    pub threshold_db: f32,
    /// Compression ratio. 1.0 = no compression; 3.0 = 3:1.
    pub ratio: f32,
    /// Attack time constant in milliseconds.
    pub attack_ms: f32,
    /// Release time constant in milliseconds.
    pub release_ms: f32,
    /// Post-compressor makeup gain in dB.
    pub makeup_db: f32,
}

/// Per-channel runtime state for the enhancement chain. Lives inside
/// [`crate::vocoder::Vocoder`]; cleared on `reset()`.
#[derive(Clone, Debug, Default)]
pub struct EnhancementState {
    biquad_state: [BiquadState; 2],
    /// Smoothed envelope in linear amplitude units (full-scale = 1.0).
    env: f32,
    /// Number of frames since the last non-Use disposition. Drives the
    /// boundary fade.
    pending_fade: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BiquadState {
    /// Direct-form-I delay line: x[n-1], x[n-2], y[n-1], y[n-2].
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadState {
    #[inline]
    fn process(&mut self, b: &Biquad, x: f32) -> f32 {
        let y = b.b0 * x + b.b1 * self.x1 + b.b2 * self.x2 - b.a1 * self.y1 - b.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Apply the configured enhancement to a 160-sample i16 PCM frame in
/// place. `prev_was_use` should be `true` when the prior frame's
/// disposition was `Use` (no fade needed); `false` on the first frame
/// after `Repeat` / `Mute`.
pub fn apply(
    mode: &EnhancementMode,
    state: &mut EnhancementState,
    pcm: &mut [i16],
    sample_rate_hz: f32,
    prev_was_use: bool,
) {
    let cfg = match mode {
        EnhancementMode::None => return,
        EnhancementMode::Classical(c) => c,
    };

    if !prev_was_use {
        state.pending_fade = cfg.boundary_fade_samples;
    }

    // Compressor envelope coefficients.
    let comp = cfg.compressor.as_ref();
    let (alpha_a, alpha_r, threshold_lin, makeup_lin, ratio) = if let Some(c) = comp {
        let a = (-1.0 / (c.attack_ms.max(0.1) * 0.001 * sample_rate_hz)).exp();
        let r = (-1.0 / (c.release_ms.max(0.1) * 0.001 * sample_rate_hz)).exp();
        let t = 10f32.powf(c.threshold_db / 20.0);
        let m = 10f32.powf(c.makeup_db / 20.0);
        (a, r, t, m, c.ratio.max(1.0))
    } else {
        (0.0, 0.0, 0.0, 1.0, 1.0)
    };

    for (i, sample) in pcm.iter_mut().enumerate() {
        let mut x = (*sample as f32) / 32_768.0;

        // Biquad cascade.
        for (slot_cfg, slot_state) in cfg.biquads.iter().zip(state.biquad_state.iter_mut()) {
            if let Some(b) = slot_cfg {
                x = slot_state.process(b, x);
            }
        }

        // Compressor (feed-forward, soft over-threshold via abs envelope).
        if comp.is_some() {
            let abs_x = x.abs();
            let alpha = if abs_x > state.env { alpha_a } else { alpha_r };
            state.env = alpha * state.env + (1.0 - alpha) * abs_x;
            let gain = if state.env > threshold_lin && state.env > 0.0 {
                let over_db = 20.0 * (state.env / threshold_lin).log10();
                let reduce_db = over_db * (1.0 - 1.0 / ratio);
                10f32.powf(-reduce_db / 20.0)
            } else {
                1.0
            };
            x *= gain * makeup_lin;
        }

        // Boundary fade (cosine half-window) on the first
        // `pending_fade` samples of the frame.
        if state.pending_fade > 0 && i < cfg.boundary_fade_samples {
            let n = cfg.boundary_fade_samples as f32;
            let pos = (cfg.boundary_fade_samples - state.pending_fade) as f32;
            let w = 0.5 - 0.5 * (PI * pos / n).cos();
            x *= w;
            state.pending_fade -= 1;
        }

        // Saturating cast back to i16.
        let y = (x * 32_768.0).clamp(-32_768.0, 32_767.0);
        *sample = y as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_pass_through() {
        let mut pcm = vec![1234i16; 160];
        let snapshot = pcm.clone();
        let mut state = EnhancementState::default();
        apply(
            &EnhancementMode::None,
            &mut state,
            &mut pcm,
            8_000.0,
            true,
        );
        assert_eq!(pcm, snapshot);
    }

    #[test]
    fn classical_default_runs_without_panic() {
        let mut pcm: Vec<i16> = (0..160).map(|i| (i as i16) * 100).collect();
        let mut state = EnhancementState::default();
        apply(
            &EnhancementMode::Classical(ClassicalConfig::default()),
            &mut state,
            &mut pcm,
            8_000.0,
            true,
        );
        // Output bounded; no NaN.
        for &s in &pcm {
            assert!(s.abs() < 32_767);
        }
    }

    #[test]
    fn high_pass_attenuates_dc() {
        // A pure DC signal should be heavily attenuated by an HPF.
        let mut pcm = vec![10_000i16; 8_000]; // 1 second of DC
        let mut state = EnhancementState::default();
        let cfg = ClassicalConfig {
            biquads: [Some(Biquad::high_pass(8_000.0, 250.0, 0.707)), None],
            compressor: None,
            boundary_fade_samples: 0,
        };
        apply(
            &EnhancementMode::Classical(cfg),
            &mut state,
            &mut pcm,
            8_000.0,
            true,
        );
        // After settling, output should be near zero.
        let tail_max = pcm[4_000..].iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(tail_max < 100, "DC not attenuated: tail max {tail_max}");
    }

    #[test]
    fn fade_starts_after_mute() {
        let mut pcm = vec![20_000i16; 160];
        let mut state = EnhancementState::default();
        let cfg = ClassicalConfig {
            biquads: [None, None],
            compressor: None,
            boundary_fade_samples: 40,
        };
        apply(
            &EnhancementMode::Classical(cfg),
            &mut state,
            &mut pcm,
            8_000.0,
            false, // prev was Mute
        );
        // First sample fully muted by cos(0)=1 → window=0; later samples reach 1.
        assert!(pcm[0].abs() < 100);
        assert!(pcm[40] > 19_000);
    }
}
