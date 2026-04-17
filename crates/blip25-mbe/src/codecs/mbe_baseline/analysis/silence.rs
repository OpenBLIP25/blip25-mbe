//! §0.8 — Frame-type dispatch (silence detection).
//!
//! Reference: vocoder_analysis_encoder_addendum.md §0.8. The PDF does
//! not prescribe encoder-side entry criteria for silence or tone.
//! §0.8.4 gives a baseline energy-threshold detector; §0.8.5 explicitly
//! recommends routing tone detection OUTSIDE the analysis encoder
//! (SAP-driven), so we implement silence here but leave tone as
//! upstream policy.

/// Fast-attack coefficient for the noise-floor tracker (new sample
/// LOWER than tracked floor — attack case).
pub(super) const SILENCE_ATTACK_NEW: f64 = 0.05;
/// Fast-attack retain coefficient for the previous floor estimate.
pub(super) const SILENCE_ATTACK_PREV: f64 = 0.95;
/// Slow-release coefficient for the noise-floor tracker (new sample
/// HIGHER than tracked floor — release case).
pub(super) const SILENCE_RELEASE_NEW: f64 = 0.01;
/// Slow-release retain coefficient.
pub(super) const SILENCE_RELEASE_PREV: f64 = 0.99;
/// Low multiplier: `E_f < α · η` → silent vote (α = 2).
pub(super) const SILENCE_ALPHA: f64 = 2.0;
/// High multiplier: `E_f > β · η` → voice vote (β = 4).
pub(super) const SILENCE_BETA: f64 = 4.0;
/// Frame-count hysteresis for entering silence.
pub(super) const SILENCE_ENTER_FRAMES: u8 = 5;
/// Frame-count hysteresis for exiting silence.
pub(super) const SILENCE_EXIT_FRAMES: u8 = 3;
/// Cold-start noise floor. Set high so the ratio test `E_f < α·η`
/// fires immediately on quiet input: silent audio enters silence
/// within the 5-frame hysteresis, while voiced audio (`E_f > β·η`)
/// immediately casts voice votes and never false-triggers. The value
/// matches `XI_MAX_FLOOR` (§0.7 Eq. 41) in order of magnitude.
pub(super) const SILENCE_ETA_INIT: f64 = 20000.0;

/// Running silence-detector state per addendum §0.8.4.
///
/// Opt-in: the detector is cheap to maintain per frame but the
/// silence DECISION only gates `encode`'s output when
/// [`super::AnalysisState::set_silence_detection`] is enabled. The
/// addendum's default recommendation (§0.8.8 final bullet) is
/// **pass-through** — let the pipeline run on silent input and emit
/// low-energy voice frames, matching DVSI chip behavior.
#[derive(Clone, Copy, Debug)]
pub struct SilenceDetector {
    /// Running noise-floor energy estimate.
    eta: f64,
    /// Count of consecutive silent-vote frames.
    silent_count: u8,
    /// Count of consecutive voice-vote frames.
    voice_count: u8,
    /// Current silence state (post-hysteresis).
    in_silence: bool,
}

impl SilenceDetector {
    /// Cold-start: noise floor = 1.0, no counts, not in silence.
    pub const fn cold_start() -> Self {
        Self {
            eta: SILENCE_ETA_INIT,
            silent_count: 0,
            voice_count: 0,
            in_silence: false,
        }
    }

    /// Current post-hysteresis silence decision.
    #[inline]
    pub fn is_silent(&self) -> bool {
        self.in_silence
    }

    /// Current noise-floor estimate (for inspection / DVSI-calibration).
    #[inline]
    pub fn noise_floor(&self) -> f64 {
        self.eta
    }

    /// Update the detector with the frame's energy `E_f = Σ s²(n)`.
    /// Updates `η`, the per-frame vote counters, and the hysteresis
    /// transition. Returns the post-update silence decision.
    pub fn update(&mut self, frame_energy: f64) -> bool {
        // η update: fast attack when energy drops, slow release when it rises.
        self.eta = if frame_energy < self.eta {
            SILENCE_ATTACK_PREV * self.eta + SILENCE_ATTACK_NEW * frame_energy
        } else {
            SILENCE_RELEASE_PREV * self.eta + SILENCE_RELEASE_NEW * frame_energy
        };
        // Per-frame vote.
        if frame_energy < SILENCE_ALPHA * self.eta {
            self.silent_count = self.silent_count.saturating_add(1);
            self.voice_count = 0;
        } else if frame_energy > SILENCE_BETA * self.eta {
            self.voice_count = self.voice_count.saturating_add(1);
            self.silent_count = 0;
        }
        // Hysteresis transition.
        if !self.in_silence && self.silent_count >= SILENCE_ENTER_FRAMES {
            self.in_silence = true;
        } else if self.in_silence && self.voice_count >= SILENCE_EXIT_FRAMES {
            self.in_silence = false;
        }
        self.in_silence
    }
}

impl Default for SilenceDetector {
    fn default() -> Self {
        Self::cold_start()
    }
}
