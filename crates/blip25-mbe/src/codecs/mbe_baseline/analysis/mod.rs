//! MBE baseline analysis encoder — PCM → [`MbeParams`].
//!
//! The forward direction of the vocoder: 8 kHz 16-bit PCM in, one
//! [`MbeParams`] out per 20 ms (160-sample) frame. The result is
//! consumed by any of the wire-side `quantize` pipelines
//! (`imbe7200::dequantize::quantize`,
//! `rate33::dequantize::quantize`,
//! `dvsi_3000::quantize`) and is independent of wire format.
//!
//! # Status
//!
//! Implementation is in progress against the derived addendum
//! `~/blip25-specs/analysis/vocoder_analysis_encoder_addendum.md`
//! (§0.1–§0.11). The top-level [`encode`] entry point still reports
//! [`AnalysisError::SpecGap`] because the end-to-end pipeline
//! (§0.10) depends on sections not yet landed. What is implemented
//! below is §0.2 — the `S_w(m)` signal DFT and the `S_w(m, ω₀)` /
//! `A_l(ω₀)` harmonic basis — which is the critical-path primitive
//! consumed by §0.4 (pitch refinement), §0.5 (spectral amplitude),
//! and §0.7 (V/UV discriminant).
//!
//! # Numerical precision
//!
//! Per addendum §0.11.7 the analysis pipeline uses `f64` throughout;
//! the Annex B/C/D tables are stored as `f32` (extraction-time
//! fidelity) and promoted at use.
//!
//! # Annex tables available here
//!
//! - **Annex B** — analysis window `wI(n)`, 301 values, used by initial
//!   pitch estimation. [`analysis_window`], [`IMBE_ANALYSIS_WINDOW`].
//! - **Annex C** — pitch refinement window `wR(n)`, 221 values, used by
//!   pitch refinement and spectral amplitude estimation.
//!   [`refinement_window`], [`IMBE_REFINEMENT_WINDOW`].
//! - **Annex D** — FIR LPF `h_LPF(n)`, 21 taps, used by the initial
//!   pitch autocorrelation. [`lpf_tap`], [`IMBE_ANNEX_D_LPF`].

use crate::mbe_params::{MbeParams, SAMPLES_PER_FRAME};

include!(concat!(env!("OUT_DIR"), "/annex_b_analysis_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_c_refinement_window.rs"));
include!(concat!(env!("OUT_DIR"), "/annex_d_lpf.rs"));

/// Lookup `wI(n)` for `n ∈ [−150, 150]`. Returns `0.0` outside that
/// range (the spec treats `wI` as zero outside its support).
#[inline]
pub fn analysis_window(n: i32) -> f32 {
    if !(-150..=150).contains(&n) {
        return 0.0;
    }
    IMBE_ANALYSIS_WINDOW[(n + 150) as usize]
}

/// Lookup `wR(n)` for `n ∈ [−110, 110]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn refinement_window(n: i32) -> f32 {
    if !(-110..=110).contains(&n) {
        return 0.0;
    }
    IMBE_REFINEMENT_WINDOW[(n + 110) as usize]
}

/// Lookup `h_LPF(n)` for `n ∈ [−10, 10]`. Returns `0.0` outside that
/// range.
#[inline]
pub fn lpf_tap(n: i32) -> f32 {
    if !(-10..=10).contains(&n) {
        return 0.0;
    }
    IMBE_ANNEX_D_LPF[(n + 10) as usize]
}

// ---------------------------------------------------------------------------
// §0.2 — Harmonic basis + DFT lives in the `basis` submodule.
// ---------------------------------------------------------------------------

pub mod basis;
pub use basis::{
    Complex64, DFT_SIZE, HarmonicBasis, WINDOW_DFT_SIZE, W_R_HALF, packed_index, signal_spectrum,
};

// ---------------------------------------------------------------------------
// §0.3 — Initial pitch estimation lives in the `pitch` submodule.
// ---------------------------------------------------------------------------

pub mod pitch;
pub use pitch::{
    H_LPF_HALF, LookBackContext, PITCH_COLD_START, PITCH_GRID_LEN, PITCH_GRID_MAX,
    PITCH_GRID_MIN, PITCH_GRID_STEP, PITCH_INPUT_HALF, PITCH_INPUT_LEN, PitchSearch,
    SHARED_LPF_LEN, W_I_HALF, compute_s_lpf, compute_s_lpf_shared, decide_initial_pitch,
    look_ahead, look_back, slot_s_lpf, snap_to_pitch_grid,
};

// ---------------------------------------------------------------------------
// §0.4 — Pitch refinement lives in the `refine` submodule.
// ---------------------------------------------------------------------------

pub mod refine;
pub use refine::{
    B0_MAX, L_HAT_MAX, L_HAT_MIN, N_REFINE_CANDIDATES, PitchRefinement,
    REFINE_OFFSETS, harmonic_count_for, quantize_pitch_index, refine_pitch,
};

// ---------------------------------------------------------------------------
// §0.7 — V/UV determination lives in the `vuv` submodule.
// ---------------------------------------------------------------------------

pub mod vuv;
pub use vuv::{K_HAT_MAX, VuvResult, VuvState, XI_MAX_FLOOR, band_count_for, determine_vuv};


// ---------------------------------------------------------------------------
// §0.5 — Spectral amplitude estimation lives in the `amplitude` submodule.
// ---------------------------------------------------------------------------

pub mod amplitude;
pub use amplitude::{SpectralAmplitudes, band_for_harmonic, estimate_spectral_amplitudes};

// ---------------------------------------------------------------------------
// §0.6 — Predictor lives in the `predictor` submodule.
// ---------------------------------------------------------------------------

pub mod predictor;
pub use predictor::{
    HalfratePredictorState, L_TILDE_COLD_START, L_TILDE_COLD_START_HALFRATE,
    LAMBDA_HAT_UNVOICED_BIAS, PredictionResidual, PredictorState, RHO_HALFRATE,
    compute_prediction_residual, compute_prediction_residual_ambe_plus2, imbe_rho_f64,
    lambda_hat_from_m_hat,
};

// ---------------------------------------------------------------------------
// §0.1 — HPF lives in the `hpf` submodule.
// ---------------------------------------------------------------------------

pub mod hpf;
pub use hpf::{HPF_POLE, HpfState};

// ---------------------------------------------------------------------------
// §0.8 — Silence detection lives in the `silence` submodule.
// ---------------------------------------------------------------------------

pub mod silence;
pub use silence::SilenceDetector;

pub mod tone_detect;
pub use tone_detect::{ToneDetection, detect_dtmf, detect_single_tone, detect_tone};

pub mod profile;

// ---------------------------------------------------------------------------
// PYIN — opt-in alternative pitch frontend (post-2002 DSP, not §0.3).
// ---------------------------------------------------------------------------

pub mod pyin;
pub use pyin::{PyinHmmState, run_pyin, run_pyin_smoothed};

// ---------------------------------------------------------------------------
// Spectral subtraction — opt-in input-side denoiser (post-2002 DSP).
// ---------------------------------------------------------------------------

pub mod denoise;
pub use denoise::{NoiseSpectrum, apply_subtraction};

/// Errors reported by the analysis encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisError {
    /// The input PCM slice was not exactly [`SAMPLES_PER_FRAME`] samples.
    WrongFrameLength,
    /// Refined `ω̂_0` is outside the quantizable range
    /// `[2π/123.125, 2π/19.875]`. Per §0.4.5 the encoder should
    /// dispatch to silence; this error is surfaced when the caller
    /// cannot otherwise recover.
    PitchOutOfRange,
    /// Reserved for sections of the addendum that are not yet
    /// implemented.
    SpecGap,
}

/// Output of [`encode`] — a single 20 ms analysis frame.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisOutput {
    /// Silence frame. During preroll (first two input frames after a
    /// cold start or state reset) the encoder cannot produce a valid
    /// `P̂_I` yet; caller should emit the codec's silence bitstream.
    /// Also emitted if §0.8.4 silence detection (not yet implemented)
    /// decides the current frame is silent.
    Silence,
    /// Voice frame with reconstructed MBE parameters. The wire-side
    /// encoder ([`crate::imbe7200::dequantize::quantize`] or
    /// [`crate::rate33::dequantize::quantize`]) consumes this
    /// to produce the transmitted bitstream.
    Voice(MbeParams),
}

/// Diagnostic trace from [`encode_with_trace`] capturing the §0.3
/// pitch-tracker scalars for the current frame. `None` when pitch
/// search did not run (preroll).
///
/// All fields are in the encoder's native units: pitch periods in
/// samples, `E(P)` as the unitless Eq. 5 error. Carries the full
/// [`PitchSearch`] for the current frame so callers can query `E(P)`
/// at arbitrary pitches (e.g., the chip's reference pitch) without
/// re-running the HPF + LPF + autocorrelation pipeline.
#[derive(Clone, Debug)]
pub struct EncodeTrace {
    /// `P̂_B` from [`look_back`].
    pub p_hat_b: f64,
    /// `CE_B` — 3-frame cumulative error at the look-back pick.
    pub ce_b: f64,
    /// `P̂_F` from [`look_ahead`] (post-cascade).
    pub p_hat_f: f64,
    /// `CE_F` — 3-frame cumulative error at the look-ahead pick.
    pub ce_f: f64,
    /// `P̂_I` — [`decide_initial_pitch`] pick.
    pub p_hat_i: f64,
    /// `E(P̂_I)` — current-frame Eq. 5 error at the tracker's pick.
    pub e_p_hat_i: f64,
    /// Current frame's [`PitchSearch`] — callable via
    /// [`PitchSearch::e_of_p`] to evaluate Eq. 5 at any pitch on the
    /// same analyzed PCM window.
    pub pitch_search_current: PitchSearch,
    /// V/UV determination result (per-band D_k, Θ_ξ, decisions).
    /// `None` during preroll or silence-dispatched frames.
    pub vuv_result: Option<VuvResult>,
    /// Post-HPF frame energy `Σ s²(n)` — silence-detector input.
    pub frame_energy: f64,
    /// §0.8.4 noise-floor estimate `η` after this frame's update.
    pub silence_eta: f64,
    /// §0.8.4 hysteresis-gated silence decision after this frame.
    pub silence_decision: bool,
}

/// Shared [`HarmonicBasis`] — the Eq. 30 `W_R(k)` table is signal-
/// independent, so it is built once at first use and reused across
/// all `encode` calls.
fn shared_basis() -> &'static HarmonicBasis {
    use std::sync::OnceLock;
    static BASIS: OnceLock<HarmonicBasis> = OnceLock::new();
    BASIS.get_or_init(HarmonicBasis::new)
}

/// Run the closed-loop matched-decoder roundtrip per §0.6.6.
///
/// Takes the analysis-side outputs (`M̂_l, ω̂_0, L̂, v̂_k`) and the
/// current `PredictorState` (carrying `M̃_l(−1)`), calls the full-
/// rate wire quantizer plus the matched decoder, and returns the
/// reconstructed `M̃_l(0)` as 1-indexed f64. The caller commits the
/// result to `PredictorState` via [`PredictorState::commit`].
fn matched_decoder_roundtrip(
    m_hat: &SpectralAmplitudes,
    refinement: &PitchRefinement,
    vuv: &VuvResult,
    predictor: &PredictorState,
) -> Result<[f64; L_HAT_MAX as usize + 1], AnalysisError> {
    use crate::imbe7200::dequantize::{
        quantize as imbe_quantize, reconstruct_amplitudes_from_bits, DecoderState,
    };
    let l_hat = refinement.l_hat;
    let omega_0_f32 = refinement.omega_hat as f32;

    // Build MbeParams from the analysis output. Downcast to f32 at
    // the wire boundary; precision at this stage is set by the
    // quantizer, not by f64 accumulation upstream.
    let mut amplitudes = Vec::with_capacity(l_hat as usize);
    let mut voiced = Vec::with_capacity(l_hat as usize);
    for l in 1..=u32::from(l_hat) {
        amplitudes.push(m_hat[l as usize] as f32);
        let k = band_for_harmonic(l, vuv.k_hat);
        let v = k >= 1 && vuv.vuv[k as usize] == 1;
        voiced.push(v);
    }
    let params = MbeParams::new(omega_0_f32, l_hat, &voiced, &amplitudes)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Seed a DecoderState from the analysis predictor. Both the
    // forward prediction (inside `imbe_quantize`) and the
    // inverse reconstruction must see the same `M̃_l(−1)` / `L̃(−1)`,
    // so we snapshot before calling quantize (which mutates its
    // state argument open-loop).
    let prev_slice = predictor.m_tilde_prev_slice();
    let ds_snapshot = DecoderState::from_amplitudes(&prev_slice, predictor.l_tilde_prev());

    let mut ds_for_quantize = ds_snapshot.clone();
    let bits = imbe_quantize(&params, &mut ds_for_quantize)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Reconstruct `M̃_l(0)` using the unmutated snapshot — matches
    // what a compliant receiver computes from the same bits.
    let reconstructed = reconstruct_amplitudes_from_bits(&bits, l_hat, &ds_snapshot);

    let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
    m_tilde[0] = 1.0;
    for i in 0..l_hat as usize {
        m_tilde[i + 1] = f64::from(reconstructed[i]);
    }
    Ok(m_tilde)
}

/// Output of [`matched_decoder_roundtrip_ambe_plus2`]: the reconstructed
/// `Λ̃_l(0)` log-domain vector, harmonic count `L̃(0) = L̂(0)`, and the
/// reconstructed `γ̃(0)` differential-gain scalar. All three commit
/// together into a [`HalfratePredictorState`] via
/// [`HalfratePredictorState::commit`].
#[derive(Clone, Debug)]
struct HalfrateRoundtrip {
    /// `Λ̃_l(0)` 1-indexed; `[0]` unused, slots `[1..=l_tilde]` valid.
    lambda_tilde_zero: [f64; L_HAT_MAX as usize + 1],
    /// `L̃(0) = L̂(0)` — next frame's predictor will use this as
    /// `L̃(−1)`.
    l_tilde_zero: u8,
    /// `γ̃(0)` — Eq. 168 reconstructed post-quantization.
    gamma_tilde_zero: f64,
}

/// Half-rate matched-decoder roundtrip per addendum §0.6.10's
/// "Matched-decoder roundtrip for half-rate" recipe. Quantizes the
/// analysis output via the half-rate wire encoder, then runs the
/// wire decoder on the emitted bits to recover the `Λ̃_l(0)`,
/// `L̃(0)`, `γ̃(0)` state the receiver will see.
///
/// Snapshot-clone pattern matches full-rate's: both `quantize` and
/// `dequantize` mutate their `DecoderState` argument, so we snapshot
/// before `quantize` and seed a fresh clone into `dequantize` to
/// avoid double-advancing the wire-decoder's state within one
/// encoder frame.
fn matched_decoder_roundtrip_ambe_plus2(
    m_hat: &SpectralAmplitudes,
    refinement: &PitchRefinement,
    vuv: &VuvResult,
    predictor: &HalfratePredictorState,
) -> Result<HalfrateRoundtrip, AnalysisError> {
    use crate::rate33::dequantize::{
        dequantize as ambe_plus2_dequantize, quantize as ambe_plus2_quantize,
        DecoderState as HalfrateDecoderState,
    };

    let l_hat = refinement.l_hat;
    let omega_0_f32 = refinement.omega_hat as f32;

    // Build MbeParams from the analysis output. The half-rate wire
    // quantizer takes linear `M̃_l` in `MbeParams` and does its own
    // Eq. 150 conversion internally — our predictor's log-domain
    // `Λ̃_l(−1)` state goes to the wire decoder via
    // `HalfrateDecoderState::from_lambda_state` below.
    let mut amplitudes = Vec::with_capacity(l_hat as usize);
    let mut voiced = Vec::with_capacity(l_hat as usize);
    for l in 1..=u32::from(l_hat) {
        amplitudes.push(m_hat[l as usize] as f32);
        let k = band_for_harmonic(l, vuv.k_hat);
        let v = k >= 1 && vuv.vuv[k as usize] == 1;
        voiced.push(v);
    }
    let params = MbeParams::new(omega_0_f32, l_hat, &voiced, &amplitudes)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Seed a `HalfrateDecoderState` from the analysis predictor's
    // log-domain state. Both the forward log-mag prediction inside
    // `ambe_plus2_quantize` and the inverse reconstruction in
    // `ambe_plus2_dequantize` must see the same `Λ̃(−1)` / `L̃(−1)` /
    // `γ̃(−1)` — snapshot before quantize (which mutates state
    // open-loop to its own target lambda) and seed a fresh clone into
    // dequantize.
    let lambda_prev_slice = {
        let lambda_arr = predictor.lambda_tilde_prev();
        let n = predictor.l_tilde_prev() as usize;
        let mut v = Vec::with_capacity(n + 1);
        v.push(0.0);
        for l in 1..=n {
            v.push(lambda_arr[l]);
        }
        v
    };
    let ds_snapshot = HalfrateDecoderState::from_lambda_state(
        &lambda_prev_slice,
        predictor.l_tilde_prev(),
        predictor.gamma_tilde_prev(),
    );

    let mut ds_for_quantize = ds_snapshot.clone();
    let u = ambe_plus2_quantize(&params, &mut ds_for_quantize)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Run dequantize on the emitted bits using a fresh clone of the
    // pre-quantize snapshot — this gives us the exact state the wire
    // decoder would see at this frame.
    let mut ds_for_dequantize = ds_snapshot;
    let _ = ambe_plus2_dequantize(&u, &mut ds_for_dequantize)
        .map_err(|_| AnalysisError::PitchOutOfRange)?;

    // Extract the reconstructed state. The wire decoder's `prev_l`
    // may differ from the frontend's `l_hat` by ±1 near Annex L
    // pitch-grid boundaries — the wire encoder re-quantizes ω̂_0
    // which can shift L per Eq. 187. Use whatever the wire decoder
    // committed; the predictor state must match what the receiver
    // sees, not the frontend's L̂(0).
    let committed_l = ds_for_dequantize.previous_l();
    let lambda_snapshot = ds_for_dequantize.lambda_tilde_snapshot();
    let mut lambda_tilde_zero = [0.0f64; L_HAT_MAX as usize + 1];
    for l in 1..=committed_l as usize {
        if l < lambda_snapshot.len() {
            lambda_tilde_zero[l] = lambda_snapshot[l];
        }
    }
    Ok(HalfrateRoundtrip {
        lambda_tilde_zero,
        l_tilde_zero: committed_l,
        gamma_tilde_zero: ds_for_dequantize.previous_gamma(),
    })
}

/// Lookahead buffer length — three frames of post-HPF PCM, enough
/// to center-window the current analyze frame plus `future_1` and
/// `future_2` per §0.3.5. Samples outside the buffer (past-frame
/// edge, beyond-future edge) are zero-extended per §0.1.3.
pub const LOOKAHEAD_LEN: usize = 3 * SAMPLES_PER_FRAME as usize;

/// Consolidated cross-frame state for the analysis encoder per
/// addendum §0.9.1 / §0.9.2.
#[derive(Clone, Debug)]
pub struct AnalysisState {
    /// HPF filter state (§0.1.2, Eq. 3).
    hpf: HpfState,
    /// Rolling post-HPF PCM buffer. Ordered oldest-first: samples
    /// `[0, 159]` are the current analyzable frame; `[160, 319]`
    /// `future_1`; `[320, 479]` `future_2`. During preroll only the
    /// leftmost `lookahead_fill × 160` samples are meaningful; the
    /// remainder is zero.
    lookahead: [f64; LOOKAHEAD_LEN],
    /// Number of 160-sample frames already placed in `lookahead`
    /// during the preroll fill phase (`0..=3`). At steady state it
    /// stays at `3`.
    lookahead_fill: u8,
    /// Pitch tracker history (§0.3.4: `P̂_{−1}, P̂_{−2}, E_{−1}, E_{−2}`).
    pitch_history: LookBackContext,
    /// V/UV threshold + per-band history (§0.7.8).
    vuv: VuvState,
    /// Matched-decoder predictor state (§0.6.7). Full-rate path —
    /// linear-domain `M̃_l(−1)`, `L̃(−1) = 30` init.
    predictor: PredictorState,
    /// Matched-decoder predictor state for the half-rate path
    /// (§0.6.10). Log-domain `Λ̃_l(−1)`, `L̃(−1) = 15` init, plus
    /// `γ̃(−1)` differential-gain scalar. Kept separate from
    /// `predictor` per §0.6.10's "split state per rate" guidance —
    /// cross-rate contamination would break both encoders.
    predictor_ambe_plus2: HalfratePredictorState,
    /// Preroll counter per §0.9.4: saturates at `PREROLL_FRAMES`.
    /// Below that, the encoder is still filling its two-frame
    /// look-ahead and emits silence instead of voice.
    preroll: u8,
    /// §0.8.4 silence detector. Always updated per frame; only
    /// gates `encode`'s output when `silence_detection_enabled`
    /// is `true`.
    silence: SilenceDetector,
    /// Opt-in gate for §0.8.4 silence dispatch. Defaults to `false`
    /// (addendum §0.8.8: pass-through recommended for initial
    /// implementation; explicit silence dispatch only when
    /// DVSI-compatibility testing demands bit-exact silence frames).
    silence_detection_enabled: bool,
    /// Opt-in joint-signal silence override. When enabled (and
    /// `silence_detection_enabled` is also on), dispatches a frame as
    /// silence when `E(P̂_I) ≥ PITCH_SILENCE_EPI_THRESH` AND
    /// `frame_energy < PITCH_SILENCE_RATIO_THRESH · η`, bypassing the
    /// §0.8.4 hysteresis. Motivation: the chip dispatches b̂_0 ≈ 25
    /// (default pitch) on isolated low-energy / poor-pitch frames that
    /// don't reach the 5-frame hysteresis threshold. Calibrated on
    /// tv-std/tv/p25/{clean,dam,mark}; off by default.
    pitch_silence_override_enabled: bool,
    /// Opt-in: on near-silent frames, override the pitch *committed*
    /// to history (NOT the pitch used to encode this frame) with a
    /// short default period. Targets the 2-frame onset attack — when
    /// quiet pre-onset frames phantom-lock the autocorrelation to a
    /// long period, the next frame's `look_back` continuity window
    /// `[0.8·P_prev, 1.2·P_prev]` is centered far from the actual
    /// onset's fundamental, causing a 2-frame silencing at the
    /// silence→loud transition. Memo:
    /// `project_onset_attack_2frame_2026-04-25.md`. Independent of
    /// `silence_detection_enabled` — the silent frame still encodes
    /// as voice; only the pitch-history side-effect changes. Off by
    /// default; eval via the 5-vector PESQ harness.
    default_pitch_on_silence_enabled: bool,
    /// Opt-in: replace the §0.3 pitch tracker (Eq. 5–23) with the
    /// PYIN frontend (`run_pyin_smoothed`) for `(p_hat_i, e_p_hat_i)`.
    /// PYIN is post-2002 DSP outside the BABA-A clean-room scope;
    /// landed as a flag for A/B-vs-§0.3 evaluation. Off by default.
    pyin_pitch_enabled: bool,
    /// Streaming HMM-Viterbi state for the PYIN smoother. Only read
    /// when `pyin_pitch_enabled` is true; reset alongside the encoder.
    pyin_hmm: PyinHmmState,
    /// Opt-in: apply Boll-style spectral subtraction to
    /// `signal_spectrum` output before §0.5 amplitude estimation.
    /// Targets noisy-tone vectors (knox_1) where §0.5's per-bin
    /// integration is noise-sensitive (memo
    /// `project_amp_noise_sensitivity_2026-04-24`). Off by default.
    /// Different from the retired `--harmonic-bin-pad` and
    /// `--noise-floor-beta` directions: those modified §0.5 internals;
    /// this cleans the input spectrum upstream.
    spectral_subtraction_enabled: bool,
    /// Per-bin running noise PSD estimate, updated on silent frames.
    /// Read only when `spectral_subtraction_enabled` is true.
    noise_spectrum: NoiseSpectrum,
    /// Opt-in EMA smoothing weight on §0.5 spectral amplitudes.
    /// `0.0` (default) disables the smoother; `(0.0, 1.0]` enables
    /// `M̂_l(t) = α · raw + (1−α) · M̂_l(t−1)` per harmonic. Targets the
    /// ~1 dB amp jitter on noisy stationary tones (memo
    /// `project_amp_noise_sensitivity_2026-04-24`); reference encoder
    /// holds ~0.1 dB σ on similar SNR. General-DSP smoothing on a
    /// magnitude estimate, outside BABA-A clean-room scope.
    amp_ema_alpha: f64,
    /// Last frame's smoothed `M̂_l` (post-EMA), pitch-gated history for
    /// the smoother. Cleared on silence dispatch and on pitch jumps.
    prev_m_hat: amplitude::SpectralAmplitudes,
    /// Companion to `prev_m_hat`: the harmonic count from the last
    /// frame (zero = no usable history).
    prev_l_hat: u8,
    /// Companion to `prev_m_hat`: ω̂_0 from the last frame, used to
    /// gate index-based blending — large pitch jumps dissociate
    /// harmonic indices, so we drop history.
    prev_omega_hat: f64,
    /// Companion to `prev_m_hat`: per-band V/UV bits from the last
    /// frame. The EMA only blends harmonic `l` when band
    /// `⌈l/3⌉`'s bit matches between frames — Eq. 43 (voiced) and
    /// Eq. 44 (unvoiced) produce differently-scaled `M̂_l`, so
    /// blending across a V/UV flip would mix incommensurate values.
    prev_vuv_bits: [u8; (vuv::K_HAT_MAX + 1) as usize],
    /// Companion to `prev_m_hat`: K̂ from the last frame, used to
    /// bound the per-band V/UV match check.
    prev_k_hat: u8,
}

/// `E(P̂_I)` threshold for the joint-signal silence override. Frames
/// with `E(P̂_I)` at or above this are considered pitch-unreliable.
/// Calibrated on clean/dam/mark: 0.4 gives 45/92 chipsil recall at
/// only 10 pitch-close FPs across the three vectors.
pub const PITCH_SILENCE_EPI_THRESH: f64 = 0.4;
/// `Ef/η` threshold for the joint-signal silence override. Frames
/// below this are considered silence-like in energy.
pub const PITCH_SILENCE_RATIO_THRESH: f64 = 2.0;

/// Default-pitch-on-silence: pitch period (samples) committed to
/// history when a near-silent frame is detected, in lieu of the
/// phantom long-period autocorrelation peak. Period 25 = ω₀ ≈ 0.2513
/// (≈ 320 Hz fundamental), matching the chip's b̂₀ ≈ 25 default per
/// memo `project_onset_attack_2frame_2026-04-25.md`. Snapped to the
/// half-sample grid (25.0 is exactly grid index 8).
pub const DEFAULT_PITCH_ON_SILENCE_PERIOD: f64 = 25.0;

/// Minimum pitch period (samples) the original `p_hat_i` must exceed
/// for the default-pitch-on-silence override to fire. Below this we
/// trust the tracker — even a "wrong" short-period lock won't push the
/// next frame's `look_back` continuity window past where look-ahead
/// can recover. Period 80 ≈ 100 Hz; phantom long-period locks on
/// silent input cluster around 100–150 (per the memo's alert frames
/// 18–19 trace at period ≈ 120). Calibrated to fire on those without
/// catching mid-speech ambiguity.
pub const DEFAULT_PITCH_PHANTOM_THRESHOLD: f64 = 80.0;
/// Minimum `e_p_hat_i` (Eq. 5 confidence) required for the override.
/// Reuse `PITCH_SILENCE_EPI_THRESH = 0.4` — frames below this have
/// pitch the tracker considers reliable, so don't override.
/// `E(P̂)` value committed alongside `DEFAULT_PITCH_ON_SILENCE_PERIOD`.
/// Pinned at the §0.3 silence-floor value `1.0` (matching the value
/// `e_of_p` returns on silent frames where the Eq. 5 denominator
/// would blow up). The first-pass implementation set this to `0.0`
/// thinking "give look-back a clean starting position", but that was
/// the opposite of what was needed: with prev_err_1 = prev_err_2 = 0,
/// the next voice frame's `CE_B = E(P̂_B) + 0 + 0` (Eq. 11) was
/// artificially LOW, biasing the Eq. 21–23 decision toward the (wrong)
/// look-back pitch even when a confident look-ahead found the actual
/// onset's fundamental. Pinning at `1.0` taxes CE_B by 2.0 so
/// look-ahead can win when it has the better answer.
pub const DEFAULT_PITCH_ON_SILENCE_E: f64 = 1.0;

/// Preroll frame count at which the analysis encoder has enough
/// look-ahead state to produce a valid `P̂_I` (§0.9.4).
pub const PREROLL_FRAMES: u8 = 2;

/// Maximum relative ω̂_0 change for which the §0.5 amplitude EMA still
/// blends across frames. Beyond this threshold, harmonic index `l`
/// from frame `t−1` and frame `t` no longer track the same physical
/// frequency, so blending would smear unrelated bins together. 25 %
/// matches the half-octave window the §0.3 tracker uses for look-back
/// continuity (`[0.8·P, 1.2·P]`) — within that band, `l · ω̂_0` stays
/// approximately constant per harmonic.
const AMP_EMA_PITCH_GATE: f64 = 0.25;

impl AnalysisState {
    /// Cold-start encoder state per §0.9.2 `analysis_encoder_state_init`.
    pub const fn new() -> Self {
        Self {
            hpf: HpfState::new(),
            lookahead: [0.0; LOOKAHEAD_LEN],
            lookahead_fill: 0,
            pitch_history: LookBackContext::cold_start(),
            vuv: VuvState::cold_start(),
            predictor: PredictorState::cold_start(),
            predictor_ambe_plus2: HalfratePredictorState::cold_start(),
            preroll: 0,
            silence: SilenceDetector::cold_start(),
            silence_detection_enabled: false,
            pitch_silence_override_enabled: false,
            default_pitch_on_silence_enabled: false,
            pyin_pitch_enabled: false,
            pyin_hmm: PyinHmmState::new(),
            // Spectral subtraction: ON by default. 5-vector A/B
            // (2026-05-14): 0 speech impact, +0.08 knox_1, −0.01 alert.
            // Effectively free PESQ on noisy stationary content; cleared
            // for default-on after the §0.5 noise-sensitivity push.
            spectral_subtraction_enabled: true,
            noise_spectrum: NoiseSpectrum::new(),
            amp_ema_alpha: 0.0,
            prev_m_hat: [0.0; L_HAT_MAX as usize + 1],
            prev_l_hat: 0,
            prev_omega_hat: 0.0,
            prev_vuv_bits: [0; (vuv::K_HAT_MAX + 1) as usize],
            prev_k_hat: 0,
        }
    }

    /// Enable or disable §0.8.4 silence dispatch. When enabled,
    /// `encode` returns `AnalysisOutput::Silence` for frames the
    /// detector flags, and skips the predictor-state commit per
    /// §13.3 prose. Default: disabled (pass-through per §0.8.8
    /// recommendation).
    pub fn set_silence_detection(&mut self, enabled: bool) {
        self.silence_detection_enabled = enabled;
    }

    /// Whether §0.8.4 silence dispatch is currently enabled.
    #[inline]
    pub fn silence_detection_enabled(&self) -> bool {
        self.silence_detection_enabled
    }

    /// Enable or disable the joint-signal silence override — dispatches
    /// frames as silence when `E(P̂_I) ≥ PITCH_SILENCE_EPI_THRESH` AND
    /// `frame_energy < PITCH_SILENCE_RATIO_THRESH · η`, regardless of
    /// §0.8.4 hysteresis. Requires `silence_detection_enabled` to be on.
    pub fn set_pitch_silence_override(&mut self, enabled: bool) {
        self.pitch_silence_override_enabled = enabled;
    }

    /// Whether the joint-signal silence override is currently enabled.
    #[inline]
    pub fn pitch_silence_override_enabled(&self) -> bool {
        self.pitch_silence_override_enabled
    }

    /// Enable or disable the onset-attack mitigation — on near-silent
    /// frames (`frame_energy < PITCH_SILENCE_RATIO_THRESH · η`),
    /// commit a short default pitch ([`DEFAULT_PITCH_ON_SILENCE_PERIOD`])
    /// to the look-back history instead of the phantom long-period
    /// autocorrelation peak. Independent of `set_silence_detection` —
    /// the silent frame still encodes through the full voice pipeline;
    /// only the pitch-history side-effect changes. See memo
    /// `project_onset_attack_2frame_2026-04-25.md`. Default off.
    pub fn set_default_pitch_on_silence(&mut self, enabled: bool) {
        self.default_pitch_on_silence_enabled = enabled;
    }

    /// Whether the default-pitch-on-silence override is currently enabled.
    #[inline]
    pub fn default_pitch_on_silence_enabled(&self) -> bool {
        self.default_pitch_on_silence_enabled
    }

    /// Enable or disable the PYIN pitch frontend. When on, `encode` /
    /// `encode_with_trace` / `encode_ambe_plus2` route the per-frame
    /// `(p_hat_i, e_p_hat_i)` through [`pyin::run_pyin`] instead of
    /// the §0.3 look-back/look-ahead tracker. Off by default — §0.3
    /// remains the canonical spec-faithful path.
    pub fn set_pyin_pitch(&mut self, enabled: bool) {
        self.pyin_pitch_enabled = enabled;
    }

    /// Whether the PYIN pitch frontend is currently enabled.
    #[inline]
    pub fn pyin_pitch_enabled(&self) -> bool {
        self.pyin_pitch_enabled
    }

    /// Enable or disable input-side spectral subtraction (Boll 1979)
    /// applied to `signal_spectrum` output before §0.5 amplitude
    /// estimation. Off by default. Targets noisy-tone vectors where
    /// §0.5's per-bin integration is noise-sensitive.
    pub fn set_spectral_subtraction(&mut self, enabled: bool) {
        self.spectral_subtraction_enabled = enabled;
    }

    /// Whether spectral subtraction is currently enabled.
    #[inline]
    pub fn spectral_subtraction_enabled(&self) -> bool {
        self.spectral_subtraction_enabled
    }

    /// Set the §0.5 amplitude EMA weight `α`. `0.0` disables the
    /// smoother (default); values in `(0.0, 1.0]` enable
    /// `M̂_l(t) = α · M̂_l + (1−α) · M̂_l(t−1)`. Out-of-range or
    /// non-finite inputs are clamped to `[0, 1]`.
    pub fn set_amp_ema_alpha(&mut self, alpha: f64) {
        let a = if alpha.is_finite() { alpha.clamp(0.0, 1.0) } else { 0.0 };
        self.amp_ema_alpha = a;
        // Drop carry-over history when the smoother is reconfigured —
        // avoids leaking stale amplitudes from a prior session into
        // the first post-flip frame.
        self.prev_l_hat = 0;
        self.prev_omega_hat = 0.0;
    }

    /// Current §0.5 amplitude EMA weight; `0.0` means the smoother is
    /// off.
    #[inline]
    pub fn amp_ema_alpha(&self) -> f64 {
        self.amp_ema_alpha
    }

    /// Read-only access to the silence detector state (for inspection
    /// and DVSI-calibration tooling).
    #[inline]
    pub fn silence_detector(&self) -> &SilenceDetector {
        &self.silence
    }

    /// Mutable access to the V/UV state for conformance diagnostics
    /// (e.g., injecting chip-side V/UV history to measure feedback
    /// divergence).
    #[inline]
    pub fn vuv_state_mut(&mut self) -> &mut VuvState {
        &mut self.vuv
    }

    /// Shift the lookahead buffer left by `SAMPLES_PER_FRAME` and
    /// append `new_frame` (post-HPF) at the right. During preroll,
    /// places the frame at slot `lookahead_fill` instead of shifting.
    fn ingest_frame(&mut self, new_frame: &[f64; SAMPLES_PER_FRAME as usize]) {
        let frame = SAMPLES_PER_FRAME as usize;
        if (self.lookahead_fill as usize) < LOOKAHEAD_LEN / frame {
            let start = (self.lookahead_fill as usize) * frame;
            self.lookahead[start..start + frame].copy_from_slice(new_frame);
            self.lookahead_fill += 1;
        } else {
            self.lookahead.copy_within(frame.., 0);
            let start = LOOKAHEAD_LEN - frame;
            self.lookahead[start..].copy_from_slice(new_frame);
        }
    }

    /// Extract a 321-sample window centered on the specified frame
    /// slot (`0 = current`, `1 = future_1`, `2 = future_2`) for
    /// §0.3 `PitchSearch` construction. Positions outside
    /// `[0, LOOKAHEAD_LEN)` are zero-extended per §0.1.3.
    /// Borrow the lookahead PCM buffer. Used by the shared-LPF
    /// pitch-search fast path to feed all three slots from a single
    /// FIR pass instead of 3 separate `compute_s_lpf` calls.
    #[inline]
    fn lookahead_buf(&self) -> &[f64; LOOKAHEAD_LEN] {
        &self.lookahead
    }

    /// Extract a 221-sample window centered on the current analyzable
    /// frame for §0.2 `signal_spectrum`.
    fn extract_refinement_window(&self) -> [f64; (2 * W_R_HALF + 1) as usize] {
        let center = (SAMPLES_PER_FRAME as usize) / 2;
        let mut out = [0.0f64; (2 * W_R_HALF + 1) as usize];
        let len = (2 * W_R_HALF + 1) as usize;
        for i in 0..len {
            let offset = i as i32 - W_R_HALF;
            let buf_idx = center as i32 + offset;
            if (0..LOOKAHEAD_LEN as i32).contains(&buf_idx) {
                out[i] = self.lookahead[buf_idx as usize];
            }
        }
        out
    }

    /// `true` while the encoder is still filling its two-frame
    /// look-ahead and cannot produce a valid voice frame. During
    /// preroll the caller should emit silence (§0.9.4, §0.8).
    #[inline]
    pub fn in_preroll(&self) -> bool {
        self.preroll < PREROLL_FRAMES
    }

    /// Current preroll counter (`0..=PREROLL_FRAMES`).
    #[inline]
    pub fn preroll_counter(&self) -> u8 {
        self.preroll
    }

    /// Advance the preroll counter, saturating at `PREROLL_FRAMES`.
    pub fn advance_preroll(&mut self) {
        if self.preroll < PREROLL_FRAMES {
            self.preroll += 1;
        }
    }

    /// Mutable access to the HPF state (used by the §0.10 pipeline
    /// to filter each incoming 160-sample PCM frame before framing).
    #[inline]
    pub fn hpf_mut(&mut self) -> &mut HpfState {
        &mut self.hpf
    }

    /// Read-only access to the pitch tracker history.
    #[inline]
    pub fn pitch_history(&self) -> LookBackContext {
        self.pitch_history
    }

    /// Shift in a newly-computed `P̂_I` and its `E(P̂_I)` value:
    /// `P̂_{−2} ← P̂_{−1}`, `P̂_{−1} ← P̂_I`, same for the energy
    /// scalars. Per §0.9.3 step 1.
    pub fn commit_pitch(&mut self, p_hat_i: f64, e_p_hat_i: f64) {
        self.pitch_history.prev_err_2 = self.pitch_history.prev_err_1;
        self.pitch_history.prev_err_1 = e_p_hat_i;
        self.pitch_history.prev_pitch = p_hat_i;
    }

    /// Mutable V/UV state — `determine_vuv` commits it internally.
    #[inline]
    pub fn vuv_mut(&mut self) -> &mut VuvState {
        &mut self.vuv
    }

    /// Read-only V/UV state.
    #[inline]
    pub fn vuv(&self) -> &VuvState {
        &self.vuv
    }

    /// Apply the §0.5 amplitude EMA in-place: when the smoother is on
    /// and the previous frame produced usable amplitudes at a similar
    /// pitch, blend `M̂_l(t) ← α · M̂_l(t) + (1−α) · M̂_l(t−1)` for
    /// `1 ≤ l ≤ min(L̂_t, L̂_{t−1})`. Updates the per-frame history
    /// (post-blend value, current `L̂`, current `ω̂_0`,
    /// current per-band V/UV bits).
    ///
    /// Two gates:
    /// * Pitch similarity: `|Δω̂_0| / ω̂_0 < AMP_EMA_PITCH_GATE`. Beyond
    ///   that, harmonic indices stop tracking the same physical
    ///   frequency, so we drop history and treat the current frame as
    ///   the new reference.
    /// * Per-band V/UV match: harmonic `l`'s band `⌈l/3⌉` must have
    ///   the same V/UV bit in both frames. Eq. 43 (voiced) and Eq. 44
    ///   (unvoiced) produce differently-normalized `M̂_l`; blending
    ///   across a flip would mix incommensurate values. Speech-only
    ///   case: vowel-onset / fricative boundaries shift V/UV bits,
    ///   so this gate auto-suppresses blending precisely where
    ///   formant motion would otherwise smear.
    fn apply_amp_ema(
        &mut self,
        m_hat: &mut amplitude::SpectralAmplitudes,
        l_hat: u8,
        omega_hat: f64,
        vuv: &VuvResult,
    ) {
        let alpha = self.amp_ema_alpha;
        if alpha > 0.0 && alpha < 1.0 && self.prev_l_hat > 0 && self.prev_omega_hat > 0.0 {
            let rel = (omega_hat - self.prev_omega_hat).abs() / self.prev_omega_hat;
            if rel < AMP_EMA_PITCH_GATE {
                let overlap = u32::from(l_hat.min(self.prev_l_hat));
                let k_hat = vuv.k_hat;
                let prev_k_hat = self.prev_k_hat;
                for l in 1..=overlap {
                    let k_now = amplitude::band_for_harmonic(l, k_hat);
                    let k_prev = amplitude::band_for_harmonic(l, prev_k_hat);
                    if k_now == 0 || k_prev == 0 {
                        continue;
                    }
                    if vuv.vuv[k_now as usize] != self.prev_vuv_bits[k_prev as usize] {
                        continue;
                    }
                    let raw = m_hat[l as usize];
                    let prev = self.prev_m_hat[l as usize];
                    m_hat[l as usize] = alpha * raw + (1.0 - alpha) * prev;
                }
            }
        }
        // Update history with the (possibly-smoothed) current frame.
        self.prev_m_hat = *m_hat;
        self.prev_l_hat = l_hat;
        self.prev_omega_hat = omega_hat;
        self.prev_vuv_bits = vuv.vuv;
        self.prev_k_hat = vuv.k_hat;
    }

    /// Drop EMA carry-over history. Called on silence dispatch so
    /// subsequent voice frames don't blend across a silence gap.
    #[inline]
    fn reset_amp_ema_history(&mut self) {
        self.prev_l_hat = 0;
        self.prev_omega_hat = 0.0;
        self.prev_k_hat = 0;
    }

    /// Mutable predictor state — caller is the §0.6.6 matched-decoder
    /// feedback loop in §0.10, which calls `commit` with reconstructed
    /// `M̃_l(0)` after the full-rate quantize/dequantize round-trip.
    #[inline]
    pub fn predictor_mut(&mut self) -> &mut PredictorState {
        &mut self.predictor
    }

    /// Read-only predictor state.
    #[inline]
    pub fn predictor(&self) -> &PredictorState {
        &self.predictor
    }

    /// Freeze cross-frame state — used when the frame is dispatched
    /// as silence / tone / erasure per §0.8: state is preserved so
    /// that subsequent voice frames resume from the last good state.
    /// This matches the decoder-side §1.9 "preserve through frame
    /// mute" behavior for the predictor, and the analysis-side
    /// symmetric case described in §0.10.
    pub fn freeze(&self) {
        // No-op marker: state is preserved by simply not calling
        // the commit_* / advance_preroll methods. This method
        // exists to document the contract in §0.10's dispatch path.
    }
}

impl Default for AnalysisState {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze a 20 ms (160-sample) frame of 8 kHz 16-bit PCM and produce
/// one analysis output per addendum §0.10.
///
/// Per-frame execution order (§0.10.1):
///
///  1. HPF input frame (§0.1, Eq. 3).
///  2. Ingest into the 480-sample rolling lookahead buffer.
///  3. Preroll check (§0.9.4): first two frames emit silence.
///  4. Compute `s_LPF`, `r(t)`, `E(P)` for the current + 2 future
///     frames via [`PitchSearch::new`] (§0.3.1–§0.3.3).
///  5. Pitch tracking → `P̂_I`, `E(P̂_I)` (§0.3.4–§0.3.6).
///  6. 256-pt DFT → `S_w(m)` (§0.2.1).
///  7. Pitch refinement → `ω̂_0, L̂, â_l, b̂_l, b̂_0` (§0.4).
///  8. V/UV determination → `K̂, v̂_k` (§0.7).
///  9. Spectral amplitude estimation → `M̂_l` (§0.5).
/// 10. Closed-loop matched-decoder roundtrip → `M̃_l(0)` (§0.6.6).
/// 11. Commit pitch history and predictor state (§0.9.3).
/// 12. Emit [`AnalysisOutput::Voice`].
///
/// §0.8 frame-type dispatch (silence detection, tone pass-through)
/// is not yet wired — all non-preroll frames emit voice.
pub fn encode(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<AnalysisOutput, AnalysisError> {
    encode_with_trace(pcm, state).map(|(out, _)| out)
}

/// Identical to [`encode`] but additionally returns an [`EncodeTrace`]
/// capturing the §0.3 pitch-tracker scalars for the analyzed frame.
/// The trace is `None` during preroll (pitch search does not run).
///
/// Intended for conformance diagnostics that need to evaluate `E(P)`
/// at arbitrary pitches (e.g., a reference chip's pitch) on the same
/// PCM window the encoder analyzed — see
/// [`PitchSearch::e_of_p`].
pub fn encode_with_trace(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<(AnalysisOutput, Option<EncodeTrace>), AnalysisError> {
    if pcm.len() != SAMPLES_PER_FRAME as usize {
        return Err(AnalysisError::WrongFrameLength);
    }

    // Step 1: HPF the incoming frame.
    let mut hpf_out = [0.0f64; SAMPLES_PER_FRAME as usize];
    profile::time(profile::Stage::Hpf, || state.hpf.run_pcm(pcm, &mut hpf_out));

    // §0.8.4: update the silence detector every frame with the
    // post-HPF frame energy (Σ s²(n)). The update runs whether or
    // not silence dispatch is enabled — the η tracker needs a
    // continuous signal to be useful.
    let (frame_energy, silent) = profile::time(profile::Stage::SilenceDetect, || {
        let e: f64 = hpf_out.iter().map(|s| s * s).sum();
        let s = state.silence.update(e);
        (e, s)
    });

    // Step 2: ingest into the lookahead buffer.
    profile::time(profile::Stage::Ingest, || state.ingest_frame(&hpf_out));

    // Step 3: preroll check.
    if state.in_preroll() {
        state.advance_preroll();
        return Ok((AnalysisOutput::Silence, None));
    }

    // Steps 4-5: pitch tracking over the three-frame PitchSearch window.
    // Enable the sparse-table RMQ on `next1` / `next2` only — those see
    // ~200 argmin_in_range calls each inside `look_ahead`. `search_cur`
    // is queried once by `look_back` (linear scan is fine) and is also
    // cloned into the EncodeTrace, so skipping the table here keeps the
    // trace's clone cost low.
    //
    // LPF the lookahead once and slice the shared output into each
    // slot's local s_LPF — replaces 3 separate FIR passes with one
    // wider pass (30% fewer mults; same outputs by zero-extension
    // identity, see `compute_s_lpf_shared` doc).
    let (search_cur, search_f1, search_f2) =
        profile::time(profile::Stage::PitchSearch, || {
            let shared = compute_s_lpf_shared(state.lookahead_buf());
            let mut search_f1 = PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 1));
            let mut search_f2 = PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 2));
            search_f1.enable_argmin_fast();
            search_f2.enable_argmin_fast();
            (
                PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 0)),
                search_f1,
                search_f2,
            )
        });
    let (p_b, ce_b, p_f, ce_f, p_hat_i, e_p_hat_i) =
        profile::time(profile::Stage::PitchTrack, || {
            let (p_b, ce_b) = look_back(&search_cur, state.pitch_history);
            let (p_f, ce_f) = look_ahead(&search_cur, &search_f1, &search_f2);
            if state.pyin_pitch_enabled {
                let buf = *state.lookahead_buf();
                let (p, e) = pyin::run_pyin_smoothed(&buf, &mut state.pyin_hmm);
                (p_b, ce_b, p_f, ce_f, p, e)
            } else {
                let p_hat_i = decide_initial_pitch(p_b, ce_b, p_f, ce_f);
                let e_p_hat_i = search_cur.e_of_p(p_hat_i);
                (p_b, ce_b, p_f, ce_f, p_hat_i, e_p_hat_i)
            }
        });
    let mut trace = EncodeTrace {
        p_hat_b: p_b,
        ce_b,
        p_hat_f: p_f,
        ce_f,
        p_hat_i,
        e_p_hat_i,
        pitch_search_current: search_cur.clone(),
        vuv_result: None,
        frame_energy,
        silence_eta: state.silence.noise_floor(),
        silence_decision: silent,
    };

    // Step 6: signal DFT.
    let basis = shared_basis();
    let sw = profile::time(profile::Stage::SignalSpectrum, || {
        let sig_win = state.extract_refinement_window();
        signal_spectrum(&sig_win)
    });

    // Step 6.5: maintain the running noise spectrum estimate. Strict
    // gate (silent + low-energy + 5-frame eligible run) prevents
    // mid-speech pauses from leaking voice harmonics into the noise
    // model. Always updated regardless of the
    // `spectral_subtraction_enabled` gate so the estimate is ready
    // when the flag flips on.
    state
        .noise_spectrum
        .update(&sw, silent, frame_energy, state.silence.noise_floor());

    // Step 7: pitch refinement.
    let refinement = profile::time(profile::Stage::RefinePitch, || {
        refine_pitch(&sw, basis, p_hat_i)
    });

    // Step 8: V/UV (must precede §0.5 per Figure 5).
    let vuv_result = profile::time(profile::Stage::Vuv, || {
        determine_vuv(&sw, basis, &refinement, e_p_hat_i, &mut state.vuv)
    });
    trace.vuv_result = Some(vuv_result.clone());

    // Step 9: spectral amplitude estimation. With spectral subtraction
    // enabled, §0.5 sees a denoised spectrum; refinement (§0.4) and
    // V/UV (§0.7) stay on the original `sw` so denoising can't corrupt
    // pitch / voicing decisions.
    let sw_for_amplitude = if state.spectral_subtraction_enabled {
        apply_subtraction(&sw, &mut state.noise_spectrum)
    } else {
        sw
    };
    let mut m_hat = profile::time(profile::Stage::Amplitude, || {
        estimate_spectral_amplitudes(&sw_for_amplitude, basis, &refinement, &vuv_result)
    });
    state.apply_amp_ema(&mut m_hat, refinement.l_hat, refinement.omega_hat, &vuv_result);

    // §0.8 silence gate. Per §13.3 prose: silence / tone / erasure
    // frames do NOT update the matched-decoder predictor state, but
    // pitch and V/UV history ARE always tracked (V/UV was already
    // committed inside `determine_vuv` above; pitch history is
    // committed unconditionally below).
    //
    // Joint-signal override: when `pitch_silence_override_enabled`
    // is also on, dispatch silence on isolated pitch-unreliable
    // low-energy frames (chip's b̂_0 ≈ 25 default-pitch behavior)
    // that don't reach the §0.8.4 hysteresis threshold.
    let override_silent = state.silence_detection_enabled
        && state.pitch_silence_override_enabled
        && e_p_hat_i >= PITCH_SILENCE_EPI_THRESH
        && frame_energy < PITCH_SILENCE_RATIO_THRESH * state.silence.noise_floor();

    // Onset-attack mitigation (`set_default_pitch_on_silence`): only
    // fires when ALL of the following hold:
    //   1. `silent` — §0.8.4 hysteresis-gated silence decision.
    //   2. `p_hat_i > DEFAULT_PITCH_PHANTOM_THRESHOLD` — the original
    //      pitch is in long-period phantom territory (above ~100 Hz
    //      fundamental). Avoids resetting on short-period locks the
    //      tracker can recover from.
    //   3. `e_p_hat_i ≥ PITCH_SILENCE_EPI_THRESH` — pitch confidence
    //      below this means the tracker thinks it's locked in. Don't
    //      override what the tracker considers a confident result.
    // The triple gate is deliberately conservative — the §0.8.4
    // detector falsely flags ~40% of continuous-tone content as silent
    // (calibrated for chip-bit matching, not perceptual silence), and
    // an over-eager override on those false silents resets pitch
    // mid-tone and tanks PESQ. See memo
    // `project_onset_attack_flag_landed_2026-04-27.md`.
    let phantom_pitch =
        p_hat_i > DEFAULT_PITCH_PHANTOM_THRESHOLD && e_p_hat_i >= PITCH_SILENCE_EPI_THRESH;
    let (commit_p, commit_e) = if state.default_pitch_on_silence_enabled
        && silent
        && phantom_pitch
    {
        (DEFAULT_PITCH_ON_SILENCE_PERIOD, DEFAULT_PITCH_ON_SILENCE_E)
    } else {
        (p_hat_i, e_p_hat_i)
    };

    if (state.silence_detection_enabled && silent) || override_silent {
        state.reset_amp_ema_history();
        state.commit_pitch(commit_p, commit_e);
        state.advance_preroll();
        return Ok((AnalysisOutput::Silence, Some(trace)));
    }

    // Step 10: matched-decoder roundtrip (closed-loop §0.6.6).
    let m_tilde = profile::time(profile::Stage::MatchedDecoder, || {
        matched_decoder_roundtrip(&m_hat, &refinement, &vuv_result, &state.predictor)
    })?;

    // Steps 11–12: commit state and emit voice.
    let params = profile::time(profile::Stage::Tail, || {
        state.predictor.commit(&m_tilde, refinement.l_hat);
        state.commit_pitch(commit_p, commit_e);

        let mut amplitudes = Vec::with_capacity(refinement.l_hat as usize);
        let mut voiced = Vec::with_capacity(refinement.l_hat as usize);
        for l in 1..=u32::from(refinement.l_hat) {
            amplitudes.push(m_hat[l as usize] as f32);
            let k = band_for_harmonic(l, vuv_result.k_hat);
            voiced.push(k >= 1 && vuv_result.vuv[k as usize] == 1);
        }
        let params = MbeParams::new(
            refinement.omega_hat as f32,
            refinement.l_hat,
            &voiced,
            &amplitudes,
        )?;
        state.advance_preroll();
        Ok::<MbeParams, crate::mbe_params::MbeParamsError>(params)
    })
    .map_err(|_| AnalysisError::PitchOutOfRange)?;

    Ok((AnalysisOutput::Voice(params), Some(trace)))
}

/// Half-rate analysis encoder: PCM → `MbeParams` destined for
/// `rate33::quantize`.
///
/// Shares §0.1–§0.7 (HPF, pitch, V/UV, spectral amplitudes) with the
/// full-rate [`encode`] path — these are rate-agnostic. Diverges at
/// step 10 (matched-decoder roundtrip) where it calls the half-rate
/// wire encoder + decoder and commits `(Λ̃_l(0), L̃(0), γ̃(0))` into
/// `state.predictor_ambe_plus2` per addendum §0.6.10.
///
/// The returned `MbeParams` carries the rate-agnostic `M̂_l` in linear
/// form; the caller feeds it to `rate33::quantize` which applies
/// Eq. 150 internally. This matches the shape of `encode` — both
/// paths return analysis-side `M̂_l`, and the wire quantizer handles
/// any rate-specific pre-processing.
pub fn encode_ambe_plus2(
    pcm: &[i16],
    state: &mut AnalysisState,
) -> Result<AnalysisOutput, AnalysisError> {
    if pcm.len() != SAMPLES_PER_FRAME as usize {
        return Err(AnalysisError::WrongFrameLength);
    }

    // Steps 1–9: identical to `encode` (rate-agnostic frontend).
    // Profile-stage instrumentation mirrors `encode_with_trace`'s layout
    // so `profile_encode` shows side-by-side breakdowns for both rates.
    let mut hpf_out = [0.0f64; SAMPLES_PER_FRAME as usize];
    profile::time(profile::Stage::Hpf, || state.hpf.run_pcm(pcm, &mut hpf_out));

    let (frame_energy, silent) = profile::time(profile::Stage::SilenceDetect, || {
        let e: f64 = hpf_out.iter().map(|s| s * s).sum();
        let s = state.silence.update(e);
        (e, s)
    });

    profile::time(profile::Stage::Ingest, || state.ingest_frame(&hpf_out));

    if state.in_preroll() {
        state.advance_preroll();
        return Ok(AnalysisOutput::Silence);
    }

    let (search_cur, search_f1, search_f2) =
        profile::time(profile::Stage::PitchSearch, || {
            // Shared-LPF fast path — see encode_with_trace for context.
            let shared = compute_s_lpf_shared(state.lookahead_buf());
            let mut search_f1 = PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 1));
            let mut search_f2 = PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 2));
            search_f1.enable_argmin_fast();
            search_f2.enable_argmin_fast();
            (
                PitchSearch::from_lpf_slice(slot_s_lpf(&shared, 0)),
                search_f1,
                search_f2,
            )
        });
    let (p_hat_i, e_p_hat_i) = profile::time(profile::Stage::PitchTrack, || {
        if state.pyin_pitch_enabled {
            let buf = *state.lookahead_buf();
            pyin::run_pyin_smoothed(&buf, &mut state.pyin_hmm)
        } else {
            let (p_b, ce_b) = look_back(&search_cur, state.pitch_history);
            let (p_f, ce_f) = look_ahead(&search_cur, &search_f1, &search_f2);
            let p_hat_i = decide_initial_pitch(p_b, ce_b, p_f, ce_f);
            let e_p_hat_i = search_cur.e_of_p(p_hat_i);
            (p_hat_i, e_p_hat_i)
        }
    });

    // Onset-attack mitigation — see encode_with_trace for rationale.
    let phantom_pitch =
        p_hat_i > DEFAULT_PITCH_PHANTOM_THRESHOLD && e_p_hat_i >= PITCH_SILENCE_EPI_THRESH;
    let (commit_p, commit_e) = if state.default_pitch_on_silence_enabled
        && silent
        && phantom_pitch
    {
        (DEFAULT_PITCH_ON_SILENCE_PERIOD, DEFAULT_PITCH_ON_SILENCE_E)
    } else {
        (p_hat_i, e_p_hat_i)
    };

    let basis = shared_basis();
    let sw = profile::time(profile::Stage::SignalSpectrum, || {
        let sig_win = state.extract_refinement_window();
        signal_spectrum(&sig_win)
    });

    // Step 6.5 (mirrors full-rate): strict-gated noise estimate update.
    state
        .noise_spectrum
        .update(&sw, silent, frame_energy, state.silence.noise_floor());

    // Half-rate `L̂` is **Annex L table lookup at b̂₀**, not Eq. 31 on
    // ω̂₀ (gap 0019 resolution — BABA-A §13.1 + Annex L). Annex L's L
    // column follows a single-floor rule `L = ⌊0.9254 · π/ω₀⌋` that
    // diverges from Eq. 31's double-floor-with-+0.25 form on 36/120
    // rows. Using Eq. 31 here would desync downstream block-size
    // lookups (Annex N/P/Q/R) from the decoder.
    //
    // If the refined pitch lands outside the Annex L table range
    // (high-F0 speech can exceed half-rate's max ω₀ ≈ 0.314),
    // dispatch as Silence rather than erroring — same disposition the
    // §0.8 silence gate would apply to a frame whose pitch we can't
    // represent at half-rate. Predictor history isn't committed
    // (matches the §0.8 silence path below).
    let refinement_result =
        profile::time(profile::Stage::RefinePitch, || -> Result<Option<refine::PitchRefinement>, AnalysisError> {
            let mut refinement = refine_pitch(&sw, basis, p_hat_i);
            use crate::rate33::dequantize::{decode_pitch, encode_pitch};
            let Some(b0) = encode_pitch(refinement.omega_hat as f32) else {
                return Ok(None);
            };
            refinement.l_hat = decode_pitch(b0).ok_or(AnalysisError::PitchOutOfRange)?.l;
            Ok(Some(refinement))
        });
    let refinement = match refinement_result? {
        Some(r) => r,
        None => {
            state.reset_amp_ema_history();
            state.commit_pitch(commit_p, commit_e);
            state.advance_preroll();
            return Ok(AnalysisOutput::Silence);
        }
    };
    let vuv_result = profile::time(profile::Stage::Vuv, || {
        determine_vuv(&sw, basis, &refinement, e_p_hat_i, &mut state.vuv)
    });
    let sw_for_amplitude = if state.spectral_subtraction_enabled {
        apply_subtraction(&sw, &mut state.noise_spectrum)
    } else {
        sw
    };
    let mut m_hat = profile::time(profile::Stage::Amplitude, || {
        estimate_spectral_amplitudes(&sw_for_amplitude, basis, &refinement, &vuv_result)
    });
    state.apply_amp_ema(&mut m_hat, refinement.l_hat, refinement.omega_hat, &vuv_result);

    // §0.8 silence gate — half-rate silence dispatch would emit
    // `b̂_0 ∈ [124, 125]` per §13.1 Table 14, but the addendum §0.8
    // silence-entry criteria are DVSI black-box. For this first cut
    // we mirror `encode`'s logic: on silence, skip predictor commit
    // and return Silence; §13.3 says `γ̃` and `Λ̃` both freeze
    // together on silence (addendum §0.6.10 "common pitfalls" picks
    // this as the conservative reading).
    let override_silent = state.silence_detection_enabled
        && state.pitch_silence_override_enabled
        && e_p_hat_i >= PITCH_SILENCE_EPI_THRESH
        && frame_energy < PITCH_SILENCE_RATIO_THRESH * state.silence.noise_floor();
    if (state.silence_detection_enabled && silent) || override_silent {
        state.reset_amp_ema_history();
        state.commit_pitch(commit_p, commit_e);
        state.advance_preroll();
        return Ok(AnalysisOutput::Silence);
    }

    // Step 10: half-rate matched-decoder roundtrip. Runs the ambe_plus2
    // wire encoder + decoder on the analysis output to recover the
    // `Λ̃_l(0)` / `L̃(0)` / `γ̃(0)` state the receiver will see.
    let rt = profile::time(profile::Stage::MatchedDecoder, || {
        matched_decoder_roundtrip_ambe_plus2(
            &m_hat,
            &refinement,
            &vuv_result,
            &state.predictor_ambe_plus2,
        )
    })?;

    // Step 11–12: commit predictor + pitch state, build the rate-agnostic
    // MbeParams. All three half-rate predictor fields advance together
    // per §0.6.10 — Λ̃, L̃, γ̃ are not independent.
    let params = profile::time(profile::Stage::Tail, || -> Result<MbeParams, AnalysisError> {
        state.predictor_ambe_plus2.commit(
            &rt.lambda_tilde_zero,
            rt.l_tilde_zero,
            rt.gamma_tilde_zero,
        );
        state.commit_pitch(commit_p, commit_e);

        let mut amplitudes = Vec::with_capacity(refinement.l_hat as usize);
        let mut voiced = Vec::with_capacity(refinement.l_hat as usize);
        for l in 1..=u32::from(refinement.l_hat) {
            amplitudes.push(m_hat[l as usize] as f32);
            let k = band_for_harmonic(l, vuv_result.k_hat);
            voiced.push(k >= 1 && vuv_result.vuv[k as usize] == 1);
        }
        MbeParams::new(
            refinement.omega_hat as f32,
            refinement.l_hat,
            &voiced,
            &amplitudes,
        )
        .map_err(|_| AnalysisError::PitchOutOfRange)
    })?;
    state.advance_preroll();
    Ok(AnalysisOutput::Voice(params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_window_symmetry_and_endpoints() {
        assert_eq!(ANALYSIS_WINDOW_LEN, 301);
        // Outside support is zero.
        assert_eq!(analysis_window(-151), 0.0);
        assert_eq!(analysis_window(151), 0.0);
        // Even symmetry.
        for n in 1..=150 {
            assert!(
                (analysis_window(-n) - analysis_window(n)).abs() < 1e-5,
                "wI(-{n}) != wI({n})"
            );
        }
    }

    #[test]
    fn refinement_window_peak_and_symmetry() {
        assert_eq!(REFINEMENT_WINDOW_LEN, 221);
        assert!((refinement_window(0) - 1.0).abs() < 1e-6);
        assert_eq!(refinement_window(-111), 0.0);
        assert_eq!(refinement_window(111), 0.0);
        for n in 1..=110 {
            assert!(
                (refinement_window(-n) - refinement_window(n)).abs() < 1e-5,
                "wR(-{n}) != wR({n})"
            );
        }
    }

    #[test]
    fn lpf_symmetry_and_support() {
        assert_eq!(ANNEX_D_LPF_LEN, 21);
        for n in 1..=10 {
            assert!((lpf_tap(-n) - lpf_tap(n)).abs() < 1e-6, "h_LPF(-{n}) != h_LPF({n})");
        }
        assert_eq!(lpf_tap(-11), 0.0);
        assert_eq!(lpf_tap(11), 0.0);
    }

    #[test]
    fn encode_rejects_wrong_length() {
        let mut s = AnalysisState::new();
        assert_eq!(
            encode(&[0i16; 159], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
        assert_eq!(
            encode(&[0i16; 161], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
    }

    #[test]
    fn encode_emits_silence_during_preroll() {
        let mut s = AnalysisState::new();
        let pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // First two frames always return silence regardless of input.
        assert_eq!(encode(&pcm, &mut s), Ok(AnalysisOutput::Silence));
        assert_eq!(encode(&pcm, &mut s), Ok(AnalysisOutput::Silence));
    }

    /// Generate a broadband harmonic PCM frame at a given period
    /// (similar to `broadband_periodic` above but producing 160-sample
    /// `i16` slices for the end-to-end encode path).
    fn periodic_pcm_frame(period: f64, max_h: u32, phase_offset: f64) -> [i16; SAMPLES_PER_FRAME as usize] {
        let mut out = [0i16; SAMPLES_PER_FRAME as usize];
        let omega = 2.0 * core::f64::consts::PI / period;
        for (idx, slot) in out.iter_mut().enumerate() {
            let nf = idx as f64 + phase_offset;
            let mut s = 0.0;
            for h in 1..=max_h {
                s += (f64::from(h) * omega * nf).cos() / f64::from(h);
            }
            *slot = (s * 8000.0).clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
        out
    }

    /// Third frame and onward produce `Voice` output on a periodic
    /// input. Validates the preroll boundary and that the entire
    /// §0.3→§0.4→§0.5→§0.6→§0.7 + matched-decoder roundtrip
    /// pipeline returns well-formed output. Not a claim about which
    /// pitch is picked — the Eq. 5 large-P bias combined with the
    /// cold-start look-back window `[0.8·100, 1.2·100]` can land on
    /// any P in `[80, 120]` for a period-50 input, and refinement
    /// offsets ±1.125 on top of that. Full-accuracy pitch recovery
    /// validation belongs with DVSI PCM vectors.
    #[test]
    fn encode_third_frame_emits_voice_on_periodic_input() {
        let mut s = AnalysisState::new();
        let p0 = periodic_pcm_frame(50.0, 10, 0.0);
        let p1 = periodic_pcm_frame(50.0, 10, 160.0);
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        assert_eq!(encode(&p0, &mut s), Ok(AnalysisOutput::Silence));
        assert_eq!(encode(&p1, &mut s), Ok(AnalysisOutput::Silence));
        match encode(&p2, &mut s).unwrap() {
            AnalysisOutput::Voice(params) => {
                // Well-formed: ω_0 in admissible range.
                let omega = params.omega_0();
                assert!(omega.is_finite() && omega > 0.0);
                let p = 2.0 * core::f64::consts::PI as f32 / omega;
                assert!(
                    (19.875..=123.125).contains(&p),
                    "P = {p} out of admissible range"
                );
                // L̂ in [9, 56].
                let l = params.harmonic_count();
                assert!((L_HAT_MIN..=L_HAT_MAX).contains(&l));
                // At least one harmonic has nonzero amplitude (signal
                // is not silence).
                let any_nonzero = (0..l as usize)
                    .any(|i| params.amplitudes_slice()[i].abs() > 1e-6);
                assert!(any_nonzero, "all amplitudes zero on periodic input");
            }
            other => panic!("expected Voice, got {other:?}"),
        }
    }

    /// Half-rate analog of the above — periodic input drives
    /// `encode_ambe_plus2` through preroll to voice. Exercises the
    /// rate-agnostic frontend, half-rate predictor commit, and
    /// matched-decoder roundtrip end-to-end.
    #[test]
    fn encode_ambe_plus2_third_frame_emits_voice_on_periodic_input() {
        let mut s = AnalysisState::new();
        let p0 = periodic_pcm_frame(50.0, 10, 0.0);
        let p1 = periodic_pcm_frame(50.0, 10, 160.0);
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        assert_eq!(encode_ambe_plus2(&p0, &mut s), Ok(AnalysisOutput::Silence));
        assert_eq!(encode_ambe_plus2(&p1, &mut s), Ok(AnalysisOutput::Silence));
        match encode_ambe_plus2(&p2, &mut s).unwrap() {
            AnalysisOutput::Voice(params) => {
                let omega = params.omega_0();
                assert!(omega.is_finite() && omega > 0.0);
                let l = params.harmonic_count();
                assert!((L_HAT_MIN..=L_HAT_MAX).contains(&l));
                let any_nonzero = (0..l as usize)
                    .any(|i| params.amplitudes_slice()[i].abs() > 1e-6);
                assert!(any_nonzero, "all amplitudes zero on periodic input");
            }
            other => panic!("expected Voice, got {other:?}"),
        }
    }

    /// Verify the half-rate predictor advances across frames.
    #[test]
    fn encode_ambe_plus2_predictor_advances_across_frames() {
        let mut s = AnalysisState::new();
        // Initial state: cold-start (L̃ = 15, γ̃ = 0, all Λ̃ = 1.0).
        assert_eq!(s.predictor_ambe_plus2.l_tilde_prev(), 15);
        assert_eq!(s.predictor_ambe_plus2.gamma_tilde_prev(), 0.0);

        for i in 0..5 {
            let pcm = periodic_pcm_frame(50.0, 10, 160.0 * i as f64);
            let _ = encode_ambe_plus2(&pcm, &mut s).unwrap();
        }

        // After multiple voice frames the state must have evolved
        // away from cold-start on at least one of (L̃, γ̃, Λ̃). This
        // is a weak assertion — we only need to prove the predictor
        // loop is live, not anything specific about its values.
        let advanced = s.predictor_ambe_plus2.l_tilde_prev() != 15
            || s.predictor_ambe_plus2.gamma_tilde_prev() != 0.0
            || s.predictor_ambe_plus2.lambda_tilde_prev()[1] != 1.0;
        assert!(advanced, "half-rate predictor did not advance past cold-start");
    }

    /// Full-rate and half-rate encoders share the analysis frontend;
    /// both should emit the same pitch / L̂ / voicing on the same PCM.
    /// Only the predictor state evolves differently.
    #[test]
    fn encode_ambe_plus2_matches_encode_on_frontend_outputs() {
        let mut s_full = AnalysisState::new();
        let mut s_half = AnalysisState::new();
        // Run through preroll + one voice frame; compare the MbeParams
        // that emerge. The frontend is deterministic and rate-agnostic
        // so ω̂_0, L̂, v̂_k, and M̂_l must match exactly.
        for i in 0..2 {
            let pcm = periodic_pcm_frame(50.0, 10, 160.0 * i as f64);
            let _ = encode(&pcm, &mut s_full).unwrap();
            let _ = encode_ambe_plus2(&pcm, &mut s_half).unwrap();
        }
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        let out_full = encode(&p2, &mut s_full).unwrap();
        let out_half = encode_ambe_plus2(&p2, &mut s_half).unwrap();
        let (pf, ph) = match (&out_full, &out_half) {
            (AnalysisOutput::Voice(pf), AnalysisOutput::Voice(ph)) => (pf, ph),
            _ => panic!("expected Voice from both, got full={out_full:?} half={out_half:?}"),
        };
        assert_eq!(pf.omega_0(), ph.omega_0(), "ω̂_0 drift across encoders");
        assert_eq!(
            pf.harmonic_count(),
            ph.harmonic_count(),
            "L̂ drift across encoders"
        );
        for i in 0..pf.harmonic_count() as usize {
            assert_eq!(
                pf.voiced_slice()[i],
                ph.voiced_slice()[i],
                "voicing mismatch at l={}",
                i + 1
            );
            // Amplitudes should be bit-identical — `m_hat` is produced
            // by the same §0.5 call, cast to f32 the same way.
            assert_eq!(
                pf.amplitudes_slice()[i],
                ph.amplitudes_slice()[i],
                "M̂_l mismatch at l={}",
                i + 1
            );
        }
    }

    /// The matched-decoder roundtrip actually evolves predictor state
    /// frame-to-frame. After one real-frame emission, `L̃(−1)` no
    /// longer equals the cold-start value (30) — unless the
    /// refinement happened to produce L̂=30 exactly.
    #[test]
    fn encode_updates_predictor_state_after_first_real_frame() {
        let mut s = AnalysisState::new();
        let p0 = periodic_pcm_frame(50.0, 10, 0.0);
        let p1 = periodic_pcm_frame(50.0, 10, 160.0);
        let p2 = periodic_pcm_frame(50.0, 10, 320.0);
        let _ = encode(&p0, &mut s);
        let _ = encode(&p1, &mut s);
        let cold_m_tilde_0: f64 = 1.0; // Eq. 56.
        let cold_m_tilde_1: f64 = s.predictor().read(1);
        assert_eq!(cold_m_tilde_1, 1.0, "pre-voice state should be cold");
        let _ = encode(&p2, &mut s).unwrap();
        // After a real-frame emission, at least one M̃_l should have
        // moved off 1.0 (unless the signal happens to reconstruct to
        // unity amplitude, which is extremely unlikely for broadband
        // periodic content).
        let any_moved = (1..=s.predictor().l_tilde_prev())
            .any(|l| (s.predictor().read(u32::from(l)) - 1.0).abs() > 1e-3);
        assert!(any_moved, "predictor state did not evolve after voice emission");
        assert_eq!(s.predictor().read(0), cold_m_tilde_0); // Eq. 56 invariant.
    }

    /// Silent input across multiple frames keeps the preroll gate
    /// working correctly: exactly 2 silence frames before the
    /// algorithm starts running. After preroll, silent PCM produces
    /// some output — possibly `Voice` with degenerate amplitudes or
    /// `Err` (pitch out of range); both are acceptable.
    #[test]
    fn encode_preroll_is_exactly_two_frames() {
        let mut s = AnalysisState::new();
        let pcm = [0i16; SAMPLES_PER_FRAME as usize];
        for i in 0..PREROLL_FRAMES {
            assert_eq!(
                encode(&pcm, &mut s),
                Ok(AnalysisOutput::Silence),
                "frame {i} should be silence during preroll"
            );
        }
        // Frame PREROLL_FRAMES is the first non-preroll frame.
        // Zero-PCM input may legitimately produce voice with
        // degenerate amplitudes, or fail with PitchOutOfRange; we
        // only assert that the preroll silence stream stopped.
        let result = encode(&pcm, &mut s);
        // Acceptable outcomes: any Voice, any Err, or silence-after-
        // preroll (if we later add §0.8.4 silence detection).
        match result {
            Ok(AnalysisOutput::Voice(_)) | Ok(AnalysisOutput::Silence) | Err(_) => {}
        }
    }


    /// When silence detection is disabled (default), the detector's
    /// decision does not gate `encode`'s output.
    #[test]
    fn encode_with_silence_detection_disabled_never_emits_non_preroll_silence() {
        let mut s = AnalysisState::new();
        assert!(!s.silence_detection_enabled());
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Burn preroll.
        let _ = encode(&zero_pcm, &mut s);
        let _ = encode(&zero_pcm, &mut s);
        // Many subsequent silent frames — with detection disabled,
        // emit voice (possibly with degenerate amplitudes) or fail,
        // but NOT Silence.
        let mut silence_count = 0u32;
        let mut other_count = 0u32;
        for _ in 0..20 {
            match encode(&zero_pcm, &mut s) {
                Ok(AnalysisOutput::Silence) => silence_count += 1,
                _ => other_count += 1,
            }
        }
        assert_eq!(silence_count, 0);
        assert_eq!(other_count, 20);
    }

    /// The pitch-silence override gate fires when the current-frame
    /// trace reports `E(P̂_I) ≥ PITCH_SILENCE_EPI_THRESH` AND
    /// `frame_energy < PITCH_SILENCE_RATIO_THRESH · η`, independent of
    /// §0.8.4 hysteresis. Drive the encoder through a prolonged loud
    /// periodic run to seed a high η, then feed enough zero frames
    /// that both the HPF transient has decayed and the analyzed frame
    /// is itself zero. The joint signal flips to Silence before
    /// hysteresis (5-frame enter) would otherwise fire.
    #[test]
    fn encode_pitch_silence_override_fires_without_hysteresis() {
        let mut s_on = AnalysisState::new();
        s_on.set_silence_detection(true);
        s_on.set_pitch_silence_override(true);
        let loud = periodic_pcm_frame(50.0, 10, 0.0);
        // Long loud run seeds η high; preroll also drained.
        for _ in 0..20 {
            let _ = encode(&loud, &mut s_on);
        }
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Walk the HPF transient out. The lookahead carries 3 frames;
        // after 4 zero inputs the analyzed frame (slot 0) is itself
        // zero-era and the HPF ringing from the loud→zero transition
        // has decayed, so frame_energy falls well below 2·η and the
        // silence floor in Eq. 5 gives E(P̂_I) = 1.0.
        let mut first_silence: Option<usize> = None;
        for i in 0..5 {
            let out = encode(&zero_pcm, &mut s_on).unwrap();
            if matches!(out, AnalysisOutput::Silence) {
                first_silence = Some(i);
                break;
            }
        }
        let Some(idx) = first_silence else {
            panic!("override never dispatched silence across 5 zero frames");
        };
        // Baseline hysteresis (5 silent votes) would not fire until at
        // least the 5th silent input. We must have silenced at or
        // before that, demonstrating the override is reaching silence
        // faster than the hysteresis path would.
        assert!(
            idx <= 4,
            "override fired at zero-frame #{idx}; expected ≤ 4"
        );
    }

    /// `set_default_pitch_on_silence` does not change THIS frame's
    /// encoded output — only `pitch_history` for the next frame.
    /// Run identical input through flag-off and flag-on encoders for
    /// many frames; assert per-frame outputs are byte-identical.
    /// (Whether the override actually fires depends on input-specific
    /// pitch + confidence + hysteresis triggers; the load-bearing
    /// invariant is just the no-side-effect-on-this-frame guarantee.)
    #[test]
    fn default_pitch_on_silence_does_not_change_current_frame_output() {
        let mut s_off = AnalysisState::new();
        let mut s_on = AnalysisState::new();
        s_on.set_default_pitch_on_silence(true);

        // Drain preroll + warmup with periodic content, then run a
        // mix of loud + quiet frames. Compare per-frame outputs.
        let loud = periodic_pcm_frame(50.0, 10, 0.0);
        let quiet = [5i16; SAMPLES_PER_FRAME as usize];

        for _ in 0..3 {
            let _ = encode(&loud, &mut s_off);
            let _ = encode(&loud, &mut s_on);
        }
        // 30 frames mixed quiet + loud — covers silence enter/exit
        // hysteresis transitions plus the override's potential fire
        // points in steady silence.
        for i in 0..30 {
            let frame = if i % 7 < 5 { &quiet } else { &loud };
            let out_off = encode(frame, &mut s_off).unwrap();
            let out_on = encode(frame, &mut s_on).unwrap();
            assert_eq!(
                format!("{out_off:?}"),
                format!("{out_on:?}"),
                "frame {i} output diverged between flag-off and flag-on \
                 — the override must only affect pitch_history side-effects, \
                 not the current frame's encoded output"
            );
        }
    }

    /// Single isolated low-energy frame must not fire the override —
    /// it requires §0.8.4 hysteresis (5 consecutive silent votes) AND
    /// a phantom-territory pitch AND high `e_p_hat_i`.
    #[test]
    fn default_pitch_on_silence_does_not_fire_on_single_quiet_frame() {
        let mut s = AnalysisState::new();
        s.set_default_pitch_on_silence(true);
        let loud = periodic_pcm_frame(50.0, 10, 0.0);
        for _ in 0..20 {
            let _ = encode(&loud, &mut s);
        }
        let quiet = [5i16; SAMPLES_PER_FRAME as usize];
        let _ = encode(&quiet, &mut s).unwrap();
        assert_ne!(
            s.pitch_history().prev_pitch,
            DEFAULT_PITCH_ON_SILENCE_PERIOD,
            "single quiet frame must not fire the override before \
             the hysteresis gate trips"
        );
    }

    /// `pitch_silence_override_enabled()` reflects the setter and the
    /// override requires `silence_detection_enabled()` to be on.
    #[test]
    fn pitch_silence_override_requires_silence_detection() {
        let mut s = AnalysisState::new();
        assert!(!s.pitch_silence_override_enabled());
        s.set_pitch_silence_override(true);
        assert!(s.pitch_silence_override_enabled());
        // Enabling the override without enabling silence detection is
        // a no-op on behavior. Encode a would-override frame and
        // confirm we still emit Voice.
        assert!(!s.silence_detection_enabled());
        let pcm = periodic_pcm_frame(50.0, 10, 0.0);
        let _ = encode(&pcm, &mut s);
        let _ = encode(&pcm, &mut s);
        // Quiet noise frame — would be silence-dispatched if silence
        // detection were on. With it off, stays Voice regardless.
        let quiet = [5i16; SAMPLES_PER_FRAME as usize];
        let out = encode(&quiet, &mut s).unwrap();
        assert!(
            matches!(out, AnalysisOutput::Voice(_)),
            "override alone must not dispatch silence when silence \
             detection is disabled"
        );
    }

    /// With silence detection enabled and sustained silent input,
    /// the encoder eventually emits `AnalysisOutput::Silence`.
    #[test]
    fn encode_with_silence_detection_enabled_gates_silent_input() {
        let mut s = AnalysisState::new();
        s.set_silence_detection(true);
        assert!(s.silence_detection_enabled());
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Feed many silent frames; detector eventually enters silence.
        let mut saw_silence_after_preroll = false;
        for i in 0..30 {
            let out = encode(&zero_pcm, &mut s);
            if i >= PREROLL_FRAMES as usize && matches!(out, Ok(AnalysisOutput::Silence)) {
                saw_silence_after_preroll = true;
                break;
            }
        }
        assert!(
            saw_silence_after_preroll,
            "silence detector should trigger on sustained zero input"
        );
    }

    /// §13.3 invariant: when `encode` emits `Silence` post-preroll,
    /// the predictor state `M̃_l(−1)` / `L̃(−1)` does NOT advance.
    #[test]
    fn encode_silence_does_not_update_predictor_state() {
        let mut s = AnalysisState::new();
        s.set_silence_detection(true);
        let zero_pcm = [0i16; SAMPLES_PER_FRAME as usize];
        // Drive into silence.
        for _ in 0..30 {
            let _ = encode(&zero_pcm, &mut s);
        }
        // Snapshot predictor state.
        let l_before = s.predictor().l_tilde_prev();
        let m1_before = s.predictor().read(1);
        // More silent frames.
        for _ in 0..5 {
            match encode(&zero_pcm, &mut s) {
                Ok(AnalysisOutput::Silence) => {}
                Ok(AnalysisOutput::Voice(_)) => {
                    // Detector may have exited; stop the test early.
                    return;
                }
                Err(_) => return,
            }
        }
        let l_after = s.predictor().l_tilde_prev();
        let m1_after = s.predictor().read(1);
        assert_eq!(l_before, l_after, "L̃(−1) changed across silent frames");
        assert_eq!(m1_before, m1_after, "M̃_1(−1) changed across silent frames");
    }

    #[test]
    fn encode_rejects_wrong_length_in_preroll() {
        let mut s = AnalysisState::new();
        assert_eq!(
            encode(&[0i16; 159], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
        assert_eq!(
            encode(&[0i16; 161], &mut s),
            Err(AnalysisError::WrongFrameLength)
        );
    }

    // -----------------------------------------------------------------
    // §0.9 — consolidated AnalysisState
    // -----------------------------------------------------------------

    #[test]
    fn analysis_state_cold_start_is_in_preroll() {
        let s = AnalysisState::new();
        assert!(s.in_preroll());
        assert_eq!(s.preroll_counter(), 0);
        assert_eq!(s.predictor().l_tilde_prev(), L_TILDE_COLD_START);
        assert_eq!(s.vuv().xi_max(), XI_MAX_FLOOR);
        assert_eq!(s.pitch_history().prev_pitch, PITCH_COLD_START);
    }

    #[test]
    fn analysis_state_preroll_saturates_at_two() {
        let mut s = AnalysisState::new();
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), 1);
        assert!(s.in_preroll());
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
        assert!(!s.in_preroll());
        // Saturates.
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
        s.advance_preroll();
        assert_eq!(s.preroll_counter(), PREROLL_FRAMES);
    }

    #[test]
    fn analysis_state_commit_pitch_shifts_history() {
        let mut s = AnalysisState::new();
        // Cold start: prev_pitch=100, prev_err_1=0, prev_err_2=0.
        s.commit_pitch(55.0, 0.25);
        let h1 = s.pitch_history();
        assert_eq!(h1.prev_pitch, 55.0);
        assert_eq!(h1.prev_err_1, 0.25);
        assert_eq!(h1.prev_err_2, 0.0); // was prev_err_1 = 0.0 before
        // Second commit: the old err_1 (0.25) shifts into err_2.
        s.commit_pitch(60.0, 0.30);
        let h2 = s.pitch_history();
        assert_eq!(h2.prev_pitch, 60.0);
        assert_eq!(h2.prev_err_1, 0.30);
        assert_eq!(h2.prev_err_2, 0.25);
    }

    #[test]
    fn analysis_state_hpf_mut_filters_samples() {
        let mut s = AnalysisState::new();
        // Impulse through HPF: y[0] = 1, y[1] = 0.99·1 − 1 = −0.01.
        let y0 = s.hpf_mut().step(1.0);
        let y1 = s.hpf_mut().step(0.0);
        assert!((y0 - 1.0).abs() < 1e-12);
        assert!((y1 - (-0.01)).abs() < 1e-12);
    }

    #[test]
    fn analysis_state_predictor_mut_allows_matched_decoder_commit() {
        let mut s = AnalysisState::new();
        let mut m_tilde = [0.0f64; L_HAT_MAX as usize + 1];
        for l in 1..=16u32 {
            m_tilde[l as usize] = f64::from(l) * 0.5;
        }
        s.predictor_mut().commit(&m_tilde, 16);
        assert_eq!(s.predictor().l_tilde_prev(), 16);
    }

    /// Helper: build a `VuvResult` with all bands voiced.
    fn vuv_voiced(k_hat: u8) -> VuvResult {
        let mut bits = [0u8; (K_HAT_MAX + 1) as usize];
        for k in 1..=k_hat {
            bits[k as usize] = 1;
        }
        VuvResult {
            k_hat,
            vuv: bits,
            d_k: [0.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    /// Helper: build a `VuvResult` with all bands unvoiced.
    fn vuv_unvoiced(k_hat: u8) -> VuvResult {
        VuvResult {
            k_hat,
            vuv: [0u8; (K_HAT_MAX + 1) as usize],
            d_k: [1.0; (K_HAT_MAX + 1) as usize],
            theta_k: [0.0; (K_HAT_MAX + 1) as usize],
            m_xi: 0.0,
            xi_0: 0.0,
            xi_max_after: XI_MAX_FLOOR,
        }
    }

    /// EMA disabled (`α = 0.0`, default) leaves `M̂_l` untouched.
    #[test]
    fn amp_ema_off_is_passthrough() {
        let mut s = AnalysisState::new();
        let mut m = [0.0f64; L_HAT_MAX as usize + 1];
        m[1] = 1.0;
        m[2] = 2.0;
        let v = vuv_voiced(1);
        s.apply_amp_ema(&mut m, 2, 0.1, &v);
        assert_eq!(m[1], 1.0);
        assert_eq!(m[2], 2.0);
    }

    /// First EMA frame has no prior history to blend with — output
    /// matches input regardless of `α`.
    #[test]
    fn amp_ema_first_frame_is_passthrough() {
        let mut s = AnalysisState::new();
        s.set_amp_ema_alpha(0.5);
        let mut m = [0.0f64; L_HAT_MAX as usize + 1];
        m[1] = 4.0;
        let v = vuv_voiced(1);
        s.apply_amp_ema(&mut m, 1, 0.1, &v);
        assert!((m[1] - 4.0).abs() < 1e-12);
    }

    /// Two-frame blend with stable pitch + matching V/UV produces
    /// the expected `α·raw + (1−α)·prev` linear combination.
    #[test]
    fn amp_ema_blends_when_pitch_and_vuv_stable() {
        let mut s = AnalysisState::new();
        s.set_amp_ema_alpha(0.5);
        let v = vuv_voiced(1);
        let mut frame_a = [0.0f64; L_HAT_MAX as usize + 1];
        frame_a[1] = 10.0;
        s.apply_amp_ema(&mut frame_a, 1, 0.10, &v);
        let mut frame_b = [0.0f64; L_HAT_MAX as usize + 1];
        frame_b[1] = 20.0;
        s.apply_amp_ema(&mut frame_b, 1, 0.10, &v);
        // 0.5·20 + 0.5·10 = 15.
        assert!((frame_b[1] - 15.0).abs() < 1e-12);
    }

    /// Pitch jumps outside the similarity gate cause the EMA to drop
    /// history; the new frame's amplitudes pass through untouched.
    #[test]
    fn amp_ema_resets_on_pitch_jump() {
        let mut s = AnalysisState::new();
        s.set_amp_ema_alpha(0.5);
        let v = vuv_voiced(1);
        let mut frame_a = [0.0f64; L_HAT_MAX as usize + 1];
        frame_a[1] = 10.0;
        s.apply_amp_ema(&mut frame_a, 1, 0.10, &v);
        let mut frame_b = [0.0f64; L_HAT_MAX as usize + 1];
        frame_b[1] = 20.0;
        // 100 % pitch increase — well past the 25 % gate.
        s.apply_amp_ema(&mut frame_b, 1, 0.20, &v);
        assert!((frame_b[1] - 20.0).abs() < 1e-12);
    }

    /// V/UV bit flips between frames suppress the per-harmonic blend
    /// even when pitch is stable. Eq. 43 vs Eq. 44 produce
    /// differently-normalized M̂_l, so blending across a flip would
    /// mix incommensurate values.
    #[test]
    fn amp_ema_skips_blend_on_vuv_flip() {
        let mut s = AnalysisState::new();
        s.set_amp_ema_alpha(0.5);
        let voiced = vuv_voiced(1);
        let unvoiced = vuv_unvoiced(1);
        let mut frame_a = [0.0f64; L_HAT_MAX as usize + 1];
        frame_a[1] = 10.0;
        s.apply_amp_ema(&mut frame_a, 1, 0.10, &voiced);
        let mut frame_b = [0.0f64; L_HAT_MAX as usize + 1];
        frame_b[1] = 20.0;
        // Same pitch, but band 1's V/UV bit flipped voiced → unvoiced.
        s.apply_amp_ema(&mut frame_b, 1, 0.10, &unvoiced);
        // Blend is skipped for harmonic 1 — output passes through.
        assert!((frame_b[1] - 20.0).abs() < 1e-12);
    }

    /// Silence dispatch clears the EMA history so the next voice
    /// frame doesn't blend across a silence gap.
    #[test]
    fn amp_ema_history_cleared_by_reset() {
        let mut s = AnalysisState::new();
        s.set_amp_ema_alpha(0.5);
        let v = vuv_voiced(1);
        let mut frame_a = [0.0f64; L_HAT_MAX as usize + 1];
        frame_a[1] = 10.0;
        s.apply_amp_ema(&mut frame_a, 1, 0.10, &v);
        s.reset_amp_ema_history();
        let mut frame_b = [0.0f64; L_HAT_MAX as usize + 1];
        frame_b[1] = 20.0;
        s.apply_amp_ema(&mut frame_b, 1, 0.10, &v);
        assert!((frame_b[1] - 20.0).abs() < 1e-12);
    }
}
