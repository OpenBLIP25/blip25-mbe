//! Per-stage timing instrumentation for the analysis encoder.
//!
//! Active only when the `profile` Cargo feature is enabled; every
//! function in this module is a no-op stub otherwise. Wraps each
//! step of [`super::encode_with_trace`] with a thread-local
//! `Instant`-based accumulator so consumers can read out a
//! per-stage breakdown via [`snapshot`].
//!
//! Zero overhead on the default build — the timing calls
//! compile-out under `#[cfg(feature = "profile")]`. The
//! `profile_encode` example enables the feature and demonstrates
//! end-to-end use.

#![allow(dead_code)]

/// Named stages of the analysis-encode and decode pipelines. Encode
/// stages match the flow in [`super::encode_with_trace`]; decode
/// stages match `synthesize_frame_with_mode` plus the wire layer
/// in [`crate::vocoder`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    // ---- analysis-encode pipeline ----
    /// `HpfState::run_pcm` — high-pass filter applied to the input frame.
    Hpf,
    /// Frame energy + silence detector update.
    SilenceDetect,
    /// `state.ingest_frame` — append post-HPF frame to lookahead buffer.
    Ingest,
    /// `state.extract_pitch_window` × 3 + `PitchSearch::new` × 3.
    PitchSearch,
    /// `look_back` + `look_ahead` + `decide_initial_pitch` + `e_of_p`
    /// for the chosen `P̂_I`.
    PitchTrack,
    /// `extract_refinement_window` + `signal_spectrum` (rustfft DFT).
    SignalSpectrum,
    /// `refine_pitch` (§0.4 quarter-sample refinement).
    RefinePitch,
    /// `determine_vuv` (§0.7 V/UV decision).
    Vuv,
    /// `estimate_spectral_amplitudes` (§0.5 Eq. 43/44 amplitudes).
    Amplitude,
    /// `matched_decoder_roundtrip` (closed-loop §0.6.6).
    MatchedDecoder,
    /// `predictor.commit` + `commit_pitch` + `MbeParams::new` + return.
    Tail,
    // ---- decode pipeline ----
    /// `unpack_dibits` — FEC byte stream → dibit array.
    DibitUnpack,
    /// `decode_frame` — Annex H deinterleave + Golay/Hamming decode.
    DecodeFrame,
    /// `dequantize` (rate-specific) — info vectors → MbeParams.
    Dequantize,
    /// `enhance_spectral_amplitudes` (§1.10 enhancement filter).
    Enhance,
    /// `apply_smoothing` (§1.11.3 V/UV + amplitude smoothing).
    Smoothing,
    /// `synthesize_unvoiced` (§1.12.1 noise + DFT + IDFT).
    SynthUnvoiced,
    /// `synthesize_voiced` (§1.12.2 sum of sinusoids per harmonic).
    SynthVoiced,
    /// Final mix + last_good snapshot + i16 quantization.
    SynthMix,
}

impl Stage {
    /// All stages, in pipeline order. Useful for `for s in Stage::ALL`
    /// loops in the consumer. Encode stages first, decode stages
    /// second; only one half is non-zero in any given timings snapshot
    /// (encoders and decoders are separate Vocoders typically).
    pub const ALL: &'static [Stage] = &[
        Stage::Hpf,
        Stage::SilenceDetect,
        Stage::Ingest,
        Stage::PitchSearch,
        Stage::PitchTrack,
        Stage::SignalSpectrum,
        Stage::RefinePitch,
        Stage::Vuv,
        Stage::Amplitude,
        Stage::MatchedDecoder,
        Stage::Tail,
        Stage::DibitUnpack,
        Stage::DecodeFrame,
        Stage::Dequantize,
        Stage::Enhance,
        Stage::Smoothing,
        Stage::SynthUnvoiced,
        Stage::SynthVoiced,
        Stage::SynthMix,
    ];

    /// Short snake_case label.
    pub const fn label(self) -> &'static str {
        match self {
            Stage::Hpf => "hpf",
            Stage::SilenceDetect => "silence_detect",
            Stage::Ingest => "ingest",
            Stage::PitchSearch => "pitch_search",
            Stage::PitchTrack => "pitch_track",
            Stage::SignalSpectrum => "signal_spectrum",
            Stage::RefinePitch => "refine_pitch",
            Stage::Vuv => "vuv",
            Stage::Amplitude => "amplitude",
            Stage::MatchedDecoder => "matched_decoder",
            Stage::Tail => "tail",
            Stage::DibitUnpack => "dibit_unpack",
            Stage::DecodeFrame => "decode_frame",
            Stage::Dequantize => "dequantize",
            Stage::Enhance => "enhance",
            Stage::Smoothing => "smoothing",
            Stage::SynthUnvoiced => "synth_unvoiced",
            Stage::SynthVoiced => "synth_voiced",
            Stage::SynthMix => "synth_mix",
        }
    }
}

/// Snapshot of the cumulative per-stage timings on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Cumulative wall-clock per stage, in nanoseconds. Indexed by
    /// the [`Stage`] enum's discriminant — keep in sync with
    /// [`Stage::ALL`].
    pub total_ns: [u64; 19],
    /// Number of encode frames observed (incremented at `Stage::Hpf`).
    pub frames: u64,
    /// Number of decode frames observed (incremented at `Stage::DibitUnpack`).
    pub decode_frames: u64,
}

impl Timings {
    /// Per-stage sample (cumulative ns, frame count) with the matching
    /// [`Stage`] label.
    pub fn entries(&self) -> impl Iterator<Item = (Stage, u64)> + '_ {
        Stage::ALL.iter().enumerate().map(|(i, &s)| (s, self.total_ns[i]))
    }

    /// Sum across all stages.
    pub fn total_ns_sum(&self) -> u64 {
        self.total_ns.iter().copied().sum()
    }
}

#[cfg(feature = "profile")]
mod inner {
    use super::Stage;
    use std::cell::RefCell;
    use std::time::Instant;

    thread_local! {
        static T: RefCell<super::Timings> = RefCell::new(super::Timings::default());
    }

    /// Time a closure, accumulating into the named stage. Increments
    /// the encode frame counter when called with `Stage::Hpf` and
    /// the decode frame counter when called with `Stage::DibitUnpack`.
    pub fn time<R>(stage: Stage, f: impl FnOnce() -> R) -> R {
        let t0 = Instant::now();
        let r = f();
        let dt = t0.elapsed().as_nanos() as u64;
        T.with(|t| {
            let mut t = t.borrow_mut();
            t.total_ns[stage as usize] = t.total_ns[stage as usize].saturating_add(dt);
            if matches!(stage, Stage::Hpf) {
                t.frames = t.frames.saturating_add(1);
            } else if matches!(stage, Stage::DibitUnpack) {
                t.decode_frames = t.decode_frames.saturating_add(1);
            }
        });
        r
    }

    pub fn snapshot() -> super::Timings {
        T.with(|t| *t.borrow())
    }

    pub fn reset() {
        T.with(|t| *t.borrow_mut() = super::Timings::default());
    }
}

#[cfg(not(feature = "profile"))]
mod inner {
    use super::Stage;

    /// No-op when the `profile` feature is off — the closure is
    /// inlined and the `stage` argument is dead.
    #[inline(always)]
    pub fn time<R>(_stage: Stage, f: impl FnOnce() -> R) -> R {
        f()
    }

    pub fn snapshot() -> super::Timings {
        super::Timings::default()
    }

    pub fn reset() {}
}

/// Time `f` and accumulate into the named stage (no-op when the
/// `profile` feature is off).
#[inline(always)]
pub fn time<R>(stage: Stage, f: impl FnOnce() -> R) -> R {
    inner::time(stage, f)
}

/// Snapshot of cumulative per-stage timings on the calling thread.
/// All zeros when the `profile` feature is off.
pub fn snapshot() -> Timings {
    inner::snapshot()
}

/// Reset cumulative timings on the calling thread.
pub fn reset() {
    inner::reset()
}
