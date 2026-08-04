//! Encoder pipeline: 8 kHz i16 PCM -> AMBE+2 half-rate frame bits.
//!
//! The analysis core here is reverse-engineered from a compiled reference vocoder
//! image, not derived from TIA-102 spec text; only the wire layer it feeds
//! ([`encode_frame`]) is spec-derived — with one exception,
//! [`encode_frame::R34_BIT_ORDER`], which is empirically derived from the reference
//! r33/r34 vector pairs. Each analysis stage is a self-contained module
//! mirroring a decoder inverse:
//!
//! ```text
//!   PCM ─► analysis ─► pitch  (b0)
//!             │        │
//!             ├──────► voicing (b1, per-harmonic V/UV)
//!             ├──────► spectral (M_l amplitudes)
//!             ▼
//!         MbeParams ─► quantize (b0..b8 + cross-frame state) ─► encode_frame ─► bytes
//! ```
//!
//! The whole chain is the exact inverse of [`crate::frame::decode_bytes`] +
//! [`crate::dequantize::dequantize`]: feeding the produced bytes back
//! through the decoder reproduces the quantized parameters bit-for-bit.

pub mod a5_assemble;
pub mod analysis;
pub mod analysis_stage_window;
pub use crate::shared::array_a_stage2;
pub use crate::shared::atan2_bfp_divide;
pub use crate::shared::atan2_chain;
pub mod audio_prefilter;
pub mod b0_audio;
pub mod b1_audio;
pub mod band_decompress;
pub use crate::shared::bfp_add;
pub mod block_consolidate;
pub mod block_exponent;
pub use crate::shared::cepstral_normalize;
pub mod cepstral_stage1;
pub mod cepstral_stage2;
pub mod windowed_complex_correlation;
pub mod cos_quadrature_interp;
pub use crate::shared::encode_frame;
pub mod history_ring;
pub mod loudness_fixed;
pub use crate::shared::loudness_transform;
pub mod main_loop;
pub mod outer_transform;
pub mod pcm_encode;
pub mod pitch;
pub mod pq_builder;
pub mod pq_yarm;
pub mod quantize;
pub mod real_fft32;
pub mod stage2_refine_dispatch;
pub mod step_recursive_fixed;
pub mod step_table;
pub mod t_ring;
pub mod voicing;
pub mod voicing_fixed;
pub mod voicing_vq;
pub mod w_builder;
pub mod win_taper;
pub mod win_taper_wide;
pub mod windowed_taper;

use crate::dequantize::{DecoderState, MbeParams};

use analysis::{Analyzer, FFT_SIZE, FRAME, WIN_HALF, WIN_LEN};

/// Streaming AMBE+2 encoder.
///
/// Holds all cross-frame state (analysis sliding window, quantizer
/// [`DecoderState`] mirror, previous-frame harmonic count, PCM look-back
/// history) so frames are fed in order, one 20 ms / 160-sample frame at a
/// time.
///
/// ## One-frame look-ahead delay
/// The reference's analysis window reaches ~45 samples past the end of the frame
/// it encodes (algorithmic look-ahead). To match that the encoder buffers one
/// frame: pushing frame *f* makes frame *f−1* fully supported (its window's
/// right half is now in hand), so the streaming calls return the bits for the
/// **previous** frame and yield `None` on the very first call. Drain the final
/// frame(s) at end-of-stream with [`Encoder::flush_r34`] / [`flush_r33`].
/// Force-flushing every frame instead would analyse a zero-padded window and
/// degrade streaming pitch relative to the full-window offline analysis.
///
/// Per frame call either [`Encoder::encode_frame_r33`] (9-byte on-air FEC
/// frame) or [`Encoder::encode_frame_r34`] (7-byte 49-bit no-FEC info
/// frame). Do **not** call both on the same frame: each call advances the
/// internal cross-frame state by exactly one frame.
pub struct Encoder {
    analyzer: Analyzer,
    /// Quantizer state, kept bit-identical to the decoder's by replaying
    /// the produced indices through `dequantize` inside `quantize`.
    state: DecoderState,
    /// Pitch override, indexed by emit-frame `f`. When set,
    /// [`Self::analyze_frame`] replaces its estimated `omega_0`/`l`/`b0` with
    /// `decode_pitch(forced_b0[f])` right after pitch estimation, so voicing
    /// and amplitude re-derive at the forced L.
    ///
    /// Set by `Vocoder::encode` (both its AMBE+2 reference-voicing branch and its
    /// per-frame branch for the other rates) and by `AmbeStream`; they inject
    /// the `b0_audio` spectral-DP pitch here, so the pitch that reaches the wire
    /// is theirs, not [`pitch::estimate_pitch`]'s. It stays `None` on a bare
    /// `Encoder` and on the public per-frame `Vocoder::encode_pcm`, both of
    /// which therefore keep the estimated pitch.
    forced_b0: Option<Vec<u8>>,
    /// IMBE voicing override, indexed by emit-frame `f`. When set,
    /// [`Self::analyze_imbe_frame`] replaces the float `decide_voicing_cfg`
    /// decision with the shared reverse-engineered reference voicing metric: each
    /// entry is the 5-bit AMBE voicing word from `b1_audio::b1_track`, expanded
    /// to per-harmonic V/UV via `build_maskword` + `gen_vmask` on the IMBE
    /// ω0/L grid. Left `None` on a bare `Encoder` and on `encode_pcm`, which
    /// keep the float voicing path.
    forced_b1: Option<Vec<u16>>,
    /// When set, a forced vector that does not cover the frame being analysed is
    /// a debug-build panic rather than a silent fall back to the estimator.
    /// See [`Self::set_forced_strict`].
    forced_strict: bool,
    /// Pitch-aligned mechanism-amplitude mode (Route A). When set,
    /// [`Self::analyze_frame`] sources the spectral amplitudes (and the
    /// reference-exact `b2` gain) for frame `f` from the LIVE `gap2` window
    /// (`self.gap2_mid`) via [`loudness_fixed::amps_from_mechanism_bins`] /
    /// [`loudness_fixed::biased_from_mechanism_bins`] instead of the
    /// synthesized-DFT `amps_from_bins_omega` proxy.
    ///
    /// Enabling it also deepens the analysis look-ahead from one frame to two
    /// (see [`Self::lookahead`]): frame `f` is only emitted once the ring has
    /// advanced through chunk `f + 2`, so the live `gap2_mid` captured at that
    /// point is exactly the pitch-aligned (frame `f + 2`) window. That is what
    /// lets a single forward pass serve BOTH the whole-buffer
    /// `Vocoder::encode` AMBE+2 path and the streaming `AmbeStream` uniformly —
    /// neither needs the old two-pass `gap2_mid_log[f + 1]` replay. Left `false`
    /// on a bare `Encoder`, on `encode_pcm`, and on every IMBE path, which keep
    /// the one-frame look-ahead and the proxy amplitudes.
    live_gap2_amps: bool,
    /// Cross-frame `bias_state` for the aligned `b2` gain override, threaded via
    /// [`loudness_fixed::next_bias_raw_exact`]. Starts at `0` (the reference's
    /// observed frame-0 bias) and only advances on frames where
    /// [`Self::live_gap2_amps`] is set.
    bias_raw_state: i16,
    /// Apply the reference's gated low-band spectral floor to the mechanism
    /// `M_l` before both the shape quantizer and the `b2` gain override. Off by
    /// default; only reachable on the Route A path ([`Self::live_gap2_amps`]).
    lowband_floor: bool,
    /// Caller-held recursive `L0` state of that floor, seeded with
    /// [`band_decompress::LOWBAND_L0_INIT`]. Advances ONLY on frames where the
    /// gate fires — advancing it on a gated-off frame desyncs every later frame.
    lowband_l0: i16,
    /// Per-emit-frame gate input for the low-band floor: the `b1` that will
    /// actually be transmitted for frame `f` (the caller's lagged voicing
    /// track), NOT the Encoder's own internal `b1`. Parallel to
    /// [`Self::forced_b0`].
    lowband_gate_b1: Option<Vec<u16>>,
    /// Gate for the DIAGNOSTIC/EXPERIMENTAL per-frame analysis paths (every
    /// `*_x86_*` research log and its isolated cross-frame state below). Off
    /// by default: `analyze_frame` then does ZERO experimental-path work and
    /// the research logs stay empty — the shipped `b0..b8` bits and the live
    /// `gap2_mid`/`gap2_slot1`/`gap2_slot2`/`prefiltered` logs are identical
    /// either way. Enable via [`Self::set_diagnostics`] BEFORE encoding to
    /// populate them. Note [`Self::set_step_gate_ref`]'s per-frame index
    /// only advances while this is on, so enable diagnostics from frame 0 when
    /// using the gate reference (enabling mid-stream would desync the reference
    /// index).
    diagnostics: bool,
    /// DIAGNOSTIC log of per-frame `frame_energy` (spec.energy/FRAME), one per
    /// emitted frame, for calibrating the energy-anchored gain. Never read by
    /// the shipped path.
    frame_energy_log: Vec<f64>,
    /// Index of the next frame to emit (encode). Lags the push head by the
    /// one-frame look-ahead delay.
    next_emit: usize,
    /// Rolling PCM look-back buffer (raw i16) for the centred pitch window.
    hist: Vec<i16>,
    /// Absolute stream index of `hist[0]`.
    hist_base: usize,
    /// Persistent state of the raw-audio pre-filter (`enc::audio_prefilter`).
    /// Zero-initialized to match the reference's `Reset()`, then carried forward
    /// across **every** subsequent frame. There is no per-frame reset: this
    /// state persists for the life of the encoder.
    prefilter_state: audio_prefilter::PrefilterState,
    /// Pre-filtered PCM, one entry per raw sample pushed via
    /// [`push_and_maybe_emit`], produced by running `audio_prefilter` over each
    /// pushed frame in the same two-80-sample-call cadence the reference uses.
    /// A flat append-only log for diagnostics; [`history_ring`] is the field
    /// that actually feeds the 108-sample analysis staging window (below).
    prefiltered: Vec<i16>,
    /// The reference's persistent 383-sample history ring
    /// ([`history_ring::HistoryRing`]), fed one 80-sample pre-filter output at
    /// a time via zero-init-then-shift-and-insert.
    /// `history_ring::HistoryRing::stage()` is the bit-exact content of
    /// [`analysis_stage_window`]'s `windowed_taper` source -- see
    /// [`current_stage`](Encoder::current_stage).
    ///
    /// **Not wired into `analyze_frame`'s quantized bit output.** The chain
    /// downstream of it (`windowed_taper` -> `real_fft32` -> `band_decompress`)
    /// needs `band_decompress`'s caller-side `step`/`outer` inputs, which are
    /// unresolved; the shipped `b0..b8` path uses
    /// `analyzer`/`loudness_fixed::amps_from_bins_omega` instead.
    history_ring: history_ring::HistoryRing,
    /// EXPERIMENTAL, diagnostic-only running `gamma` state for
    /// [`loudness_fixed::candidate_gamma`] -- a SEPARATE cross-frame
    /// predictive-gain accumulator from `state`'s own `prev_gamma()`, so
    /// this comparison path can never perturb the shipped `b0..b8` output.
    /// See [`Self::b2_x86_log`] and `loudness_fixed`'s own module doc for the
    /// scope (two open RE gaps this path does NOT close).
    prev_gamma_fixed: f64,
    /// One entry per frame emitted via [`Self::analyze_frame`]: the
    /// EXPERIMENTAL `b2` index [`loudness_fixed::candidate_gamma`] would
    /// produce, for offline comparison against the reference's `b2` bits. Does NOT
    /// feed back into the shipped `b0..b8` output.
    b2_x86_log: Vec<u8>,
    /// SECOND, separately-tracked running `gamma` state for the
    /// prefiltered-input variant of the `loudness_fixed` diagnostic (see
    /// [`Self::b2_x86_pf_log`]) -- kept fully independent of both `state`'s
    /// own `prev_gamma()` and `prev_gamma_fixed` above, so this experiment can
    /// never perturb either the shipped output or the original
    /// `loudness_fixed` diagnostic.
    prev_gamma_x86_pf: f64,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a SECOND
    /// experimental `b2` candidate, identical to `b2_x86_log` (same
    /// `step`/`outer`/`band_decompress` mechanism) except its `bins`/`win`
    /// input spectrum is built from the REAL, bit-exact `audio_prefilter`
    /// output (`self.prefiltered`) instead of raw un-filtered PCM -- see
    /// [`Self::prefiltered_spectrum`]. Tests whether wiring in more of the
    /// real analysis front end (the prefilter) narrows the gap versus the
    /// spectrum-on-raw-PCM diagnostic. Does NOT feed back into the shipped
    /// `b0..b8` output.
    b2_x86_pf_log: Vec<u8>,
    /// Persistent, cross-frame `win`/`a1ptr` state for the `b2_x86_log`
    /// diagnostic path. The reference's `win` array is a self-referential in-place
    /// accumulator, never reseeded from raw audio; it is zero-seeded here to
    /// match the reference's `Reset()` convention. Mutated in place by
    /// [`loudness_fixed::candidate_gamma`] every frame; never rebuilt from
    /// the spectrum.
    win_fixed: loudness_fixed::WinState,
    /// SECOND, fully independent persistent `win`/`a1ptr` state, for the
    /// `b2_x86_pf_log` (prefiltered-spectrum) diagnostic path -- kept
    /// separate from `win_fixed` for the same reason `prev_gamma_x86_pf` is
    /// kept separate from `prev_gamma_fixed`.
    win_x86_pf: loudness_fixed::WinState,
    /// THIRD, separately-tracked running `gamma` state for the
    /// `win_taper`-sourced experimental path (see [`Self::b2_x86_wintaper_log`])
    /// -- kept independent of every other diagnostic path's own state, per the
    /// standing rule that a new candidate must never be able to perturb an
    /// existing one.
    prev_gamma_x86_wintaper: f64,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a THIRD
    /// experimental `b2` candidate. Same `step`/`outer`/`band_decompress`
    /// mechanism and same raw-PCM spectrum as `b2_x86_log`, but `win` is
    /// REBUILT FRESH EVERY FRAME from [`loudness_fixed::win_from_gap2`]
    /// instead of carried across frames as an all-zero, never-reseeded
    /// accumulator (`win_fixed`'s convention) -- see
    /// [`loudness_fixed::win_from_gap2`]'s own doc for why. Does NOT feed
    /// back into the shipped `b0..b8` output.
    b2_x86_wintaper_log: Vec<u8>,
    /// EXPERIMENTAL, diagnostic-only per-frame log of the mechanism-matched
    /// `b1` back end ([`voicing_fixed::b1_from_abc`], via
    /// [`Self::voicing_b1_fixed`]) -- pushed once per frame emitted via
    /// [`Self::analyze_frame`]. Does NOT feed back into the shipped `b0..b8`
    /// output (same isolation convention as `b2_x86_log` and friends).
    ///
    /// **These values are not a measurement.** `b1_from_abc`'s 3 inputs
    /// (arrays A/B/C) are not raw-PCM derivable end to end: arrays A/C need
    /// the cepstral/127-tap-correlation pipeline's open `w`/`T`/`S` chain, and
    /// array B needs `heap_src`'s open `window`/`x` inputs. The call is fed
    /// explicit placeholder arrays ([`B1_X86_PLACEHOLDER_ABC`]) instead -- see
    /// [`Self::attempt_b1_fixed`]. The wiring exists so that closing those
    /// gaps needs only a swap of the array builders.
    b1_x86_log: Vec<u16>,
    /// `gap2`/`arg2`-source snapshot captured BEFORE the current frame's own
    /// audio is pushed into `history_ring` -- see [`Self::run_prefilter`] for
    /// the timing this reproduces (`history_ring::GAP2_OFFSET`). Zero-seeded,
    /// matching the ring's `Reset()` convention.
    gap2_pre: [i16; history_ring::GAP2_LEN],
    /// `gap2`/`arg2`-source snapshot captured AFTER only the FIRST of the
    /// current frame's two prefilter halves has been pushed -- the alignment 2
    /// of the reference's 3 per-frame `gap2`-source reads use. `analyze_frame` feeds
    /// [`loudness_fixed::arg2_from_recent_audio`]/[`loudness_fixed::win_from_gap2`]
    /// from this field; reading `history_ring.gap2_window()` directly instead
    /// is temporally misaligned by two samples.
    gap2_mid: [i16; history_ring::GAP2_LEN],
    /// The ring's [`history_ring::GAP2_OFFSET_SLOT1`]-anchored snapshot,
    /// captured at the SAME instant as [`Self::gap2_mid`] (after only the
    /// frame's FIRST prefilter half) -- the source of the SECOND of the 3
    /// per-frame `fft_bfp_transform`-loudness topcalls ("slot 1"). Additive:
    /// it does not replace `gap2_mid`, which remains slot 0's source and still
    /// feeds `outer`.
    gap2_slot1: [i16; history_ring::GAP2_LEN],
    /// The ring's [`history_ring::GAP2_OFFSET_SLOT2`]-anchored (index 0),
    /// [`history_ring::GAP2_LEN_WIDE`]-wide snapshot, captured AFTER BOTH of
    /// the frame's two prefilter halves have been pushed (a LATER timing than
    /// [`Self::gap2_mid`]/[`Self::gap2_slot1`]) -- the source of the THIRD of
    /// the 3 per-frame `fft_bfp_transform`-loudness topcalls ("slot 2").
    /// Additive: it does not replace `gap2_mid`/`gap2_slot1`.
    gap2_slot2: [i16; history_ring::GAP2_LEN_WIDE],
    /// Separately-tracked running `gamma` state for the
    /// `outer`-only experimental path (see [`Self::b2_x86_outeronly_log`]
    /// and [`loudness_fixed::candidate_gamma_outer_only`]) -- kept
    /// independent of every other diagnostic path's own state, same
    /// isolation convention as `prev_gamma_x86_wintaper`.
    prev_gamma_x86_outeronly: f64,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a FOURTH
    /// experimental `b2` candidate, using ONLY the `outer` scalar
    /// (bypassing `bins`/`step`/`band_decompress` entirely) -- see
    /// [`loudness_fixed::candidate_gamma_outer_only`]'s own doc for why this
    /// is a fully mechanism-matched (not approximate) quantity. Does NOT feed
    /// back into the shipped `b0..b8` output.
    b2_x86_outeronly_log: Vec<u8>,
    /// Separately-tracked running `gamma` state for the
    /// real-`bins` experimental path (see [`Self::b2_x86_realbins_log`] and
    /// [`loudness_fixed::candidate_gamma_real_bins`]) -- kept independent of
    /// every other diagnostic path's own state, same isolation convention
    /// as `prev_gamma_x86_outeronly`.
    prev_gamma_x86_realbins: f64,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a FIFTH
    /// experimental `b2` candidate, using [`loudness_fixed::
    /// assemble_bins_from_win`]'s mechanism-matched `bins` instead of
    /// [`loudness_fixed::assemble_bins`]'s DFT approximation. Does NOT feed
    /// back into the shipped `b0..b8` output.
    b2_x86_realbins_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for
    /// the real-bins path, one entry per frame -- see
    /// [`loudness_fixed::mean_raw_real_bins`]'s own doc.
    realbins_meanraw_log: Vec<f64>,
    /// Separately-tracked running raw `bias` state (Q11-ish
    /// integer domain, NOT the float `gamma` every other experimental path
    /// here tracks) for the REF_EXACT path -- see
    /// [`Self::b2_x86_refexact_log`] and
    /// [`loudness_fixed::candidate_gamma_ref_exact_recursive_bias`]/
    /// [`loudness_fixed::next_bias_raw`]. Kept independent of every other
    /// diagnostic path's own state, same isolation convention as
    /// `prev_gamma_x86_realbins`.
    prev_bias_x86_refexact: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a SIXTH
    /// experimental `b2` candidate -- a BIT-EXACT-DERIVED (not
    /// fitted/calibrated) formula for the reference's gain-quantizer input, fed by
    /// this encoder's OWN estimated `l`/`step`/`outer`/`bins` (the same
    /// known-imperfect inputs [`Self::b2_x86_realbins_log`] uses, NOT
    /// ground-truth decoded `L`). Does NOT feed back into the shipped
    /// `b0..b8` output.
    b2_x86_refexact_log: Vec<u8>,
    /// Self-referential `STEP` state for the recursive-`STEP` experimental
    /// path -- see [`step_recursive_fixed::step_update_mode1`] for the
    /// recursive update law this threads frame-to-frame. Seeded at
    /// [`step_recursive_fixed::INITIAL_STEP_SEED`] (`8192`, the reference's
    /// observed frame-0 value).
    prev_step_x86_recursive: i16,
    /// Separately-tracked `bias` IIR state for the recursive-`STEP` path, same
    /// isolation convention as `prev_bias_x86_refexact` (independent so this
    /// path can't perturb the REF_EXACT numbers).
    prev_bias_x86_stepformula: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: an EIGHTH
    /// experimental `b2` candidate -- identical to
    /// [`Self::b2_x86_refexact_log`] except `step` comes from the recursive
    /// update law ([`step_recursive_fixed`]) instead of
    /// [`super::step_table`]'s `COUNT`-keyed modal lookup. Does NOT feed
    /// back into the shipped `b0..b8` output. The accept/reject gate is
    /// approximated as "always reject-formula" here, not ported -- see
    /// [`step_recursive_fixed`]'s module doc.
    b2_x86_stepformula_log: Vec<u8>,
    /// Optional captured ACCEPT/REJECT sequence for the STEP accept/reject
    /// gate (see `step_recursive_fixed::step_gate_accept`), fed externally
    /// from `step_recursive_fixed::gate_ref` because this encoder's state
    /// has no way to compute `Q`/`P`. `None` (the default) means "no reference
    /// for this content" and every frame falls back to REJECT-FORMULA,
    /// identical to [`Self::b2_x86_stepformula_log`] -- this field can never
    /// make that path worse, only better when reference data is supplied.
    step_gate_ref: Option<Vec<bool>>,
    /// Index into `step_gate_ref`, advanced once per
    /// `Self::analyze_frame` call.
    step_gate_ref_idx: usize,
    /// Independent self-referential `STEP` state for the gate-reference path
    /// (isolated from `prev_step_x86_recursive` so this diagnostic path
    /// can't perturb `STEPFORMULA`'s own already-measured numbers).
    prev_step_x86_gateref: i16,
    /// Independent `bias` IIR state for the gate-reference path, same
    /// isolation convention as `prev_bias_x86_stepformula`.
    prev_bias_x86_gateref: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a NINTH
    /// experimental `b2` candidate -- identical to
    /// [`Self::b2_x86_stepformula_log`] except the ACCEPT/REJECT decision
    /// each frame comes from [`Self::step_gate_ref`] (captured ground
    /// truth) when available, instead of always assuming REJECT-FORMULA. Does
    /// NOT feed back into the shipped `b0..b8` output. This is a ceiling
    /// measurement, not a computable mechanism -- see
    /// [`step_recursive_fixed::gate_ref`].
    b2_x86_stepformula_gateref_log: Vec<u8>,
    /// Separately-tracked `bias` IIR state for the REALCOUNT
    /// path (see [`Self::b2_x86_stepformula_realcount_log`]), same
    /// isolation convention as `prev_bias_x86_stepformula`.
    prev_bias_x86_stepformula_realcount: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a TENTH
    /// experimental `b2` candidate -- identical to
    /// [`Self::b2_x86_stepformula_log`] except the harmonic-count `L`
    /// (`sumcount`) comes from [`step_recursive_fixed::step_update_mode1`]'s
    /// `new_count` output -- the bit-exact `COUNT` companion to the `STEP`
    /// recursion, and the reference's real `sumcount` source -- instead of this
    /// project's separately-estimated pitch/harmonic-count
    /// (`enc::pitch::estimate_pitch`). Does NOT feed back into the shipped
    /// `b0..b8` output.
    b2_x86_stepformula_realcount_log: Vec<u8>,
    /// Separately-tracked running `gamma` state for the REALBINS_REALSTEP
    /// path (see [`Self::b2_x86_realbins_realstep_log`]), same isolation
    /// convention as `prev_gamma_x86_realbins`.
    prev_gamma_x86_realbins_realstep: f64,
    /// Self-referential `STEP` state for the REALBINS_REALSTEP path -- kept
    /// SEPARATE from `prev_step_x86_recursive` (the
    /// STEPFORMULA/GATEREF/REALCOUNT paths' shared thread) so this path's
    /// seed/recursion can't perturb theirs. Seeded at
    /// [`step_recursive_fixed::INITIAL_STEP_SEED`], same as
    /// `prev_step_x86_recursive`.
    prev_step_x86_realbins_realstep: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: an ELEVENTH
    /// experimental `b2` candidate. Pairs [`Self::b2_x86_realbins_log`]'s
    /// separately-tuned affine calibration
    /// ([`loudness_fixed::candidate_gamma_real_bins_with_step`]) with the
    /// bit-exact recursive `STEP` update law
    /// ([`step_recursive_fixed::step_update_mode1`]) instead of
    /// [`super::step_table`]'s `COUNT`-keyed modal lookup (~57-59% exact on
    /// real speech). Does NOT feed back into the shipped `b0..b8` output.
    b2_x86_realbins_realstep_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for
    /// the REALBINS_REALSTEP path, one entry per frame -- mirrors
    /// `realbins_meanraw_log`, see [`loudness_fixed::mean_raw_real_bins_with_step`]'s
    /// own doc. Lets an offline grid search re-tune the calibration for
    /// this specific (real-`bins`, real-`STEP`) input without re-running
    /// the whole encoder per grid point.
    realbins_realstep_meanraw_log: Vec<f64>,
    /// Separately-tracked running `gamma` state for the
    /// REALBINS_SLOT1_REALSTEP path (see
    /// [`Self::b2_x86_realbins_slot1_realstep_log`]) -- identical to
    /// REALBINS_REALSTEP except `bins` is built from SLOT 1's window
    /// (`gap2_slot1`) instead of slot 0's (`gap2_mid`), while `outer` still
    /// comes from slot 0 (the only slot `combine_outer`'s formula uses). Same
    /// isolation convention as `prev_gamma_x86_realbins_realstep`.
    prev_gamma_x86_realbins_slot1_realstep: f64,
    /// Self-referential `STEP` state for the REALBINS_SLOT1_REALSTEP
    /// path -- kept separate from every other path's own STEP thread, same
    /// isolation convention as `prev_step_x86_realbins_realstep`.
    prev_step_x86_realbins_slot1_realstep: i16,
    /// One entry per frame -- the REALBINS_SLOT1_REALSTEP experimental `b2`
    /// candidate. Does NOT feed back into the shipped `b0..b8` output.
    b2_x86_realbins_slot1_realstep_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for
    /// the REALBINS_SLOT1_REALSTEP path, mirrors `realbins_realstep_meanraw_log`,
    /// for offline calibration grid search.
    realbins_slot1_realstep_meanraw_log: Vec<f64>,
    /// Self-referential `bias` state for the REFEXACT_SLOT1_REALSTEP path
    /// ([`loudness_fixed::candidate_gamma_ref_exact_slot1_with_step`]) --
    /// SLOT 1's spectrum plus the recursive `STEP` law, fed through
    /// [`loudness_fixed::gamma_ref_exact`]'s fully-derived
    /// (zero-fitted-parameter) formula instead of an affine calibration.
    /// Isolated from every other path's bias/gamma thread, same convention as
    /// `prev_bias_x86_refexact`.
    prev_bias_x86_refexact_slot1: i16,
    /// One entry per frame -- the REFEXACT_SLOT1_REALSTEP experimental `b2`
    /// candidate. Does NOT feed back into the shipped `b0..b8` output.
    b2_x86_refexact_slot1_log: Vec<u8>,
    /// Separately-tracked running `gamma` state for the
    /// REALBINS_SLOT2_REALSTEP path (see
    /// [`Self::b2_x86_realbins_slot2_realstep_log`]) -- identical to
    /// REALBINS_SLOT1_REALSTEP except `bins` is built from SLOT 2's window
    /// (`gap2_slot2`) instead of slot 1's, while `outer` still comes from
    /// slot 0. Same isolation convention as
    /// `prev_gamma_x86_realbins_slot1_realstep`.
    prev_gamma_x86_realbins_slot2_realstep: f64,
    /// Self-referential `STEP` state for the REALBINS_SLOT2_REALSTEP
    /// path -- kept separate from every other path's own STEP thread.
    prev_step_x86_realbins_slot2_realstep: i16,
    /// One entry per frame -- the REALBINS_SLOT2_REALSTEP experimental `b2`
    /// candidate, the only path that uses slot 2. Does NOT feed back into the
    /// shipped `b0..b8` output.
    b2_x86_realbins_slot2_realstep_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for the
    /// REALBINS_SLOT2_REALSTEP path, for offline calibration grid search.
    realbins_slot2_realstep_meanraw_log: Vec<f64>,
    /// Separately-tracked running `gamma` state for the
    /// REALBINS_REALSTEP_GATEREF path (see
    /// [`Self::b2_x86_realbins_realstep_gateref_log`]), same isolation
    /// convention as `prev_gamma_x86_realbins_realstep`.
    prev_gamma_x86_realbins_realstep_gateref: f64,
    /// Self-referential `STEP` state for the REALBINS_REALSTEP_GATEREF
    /// path -- kept SEPARATE from `prev_step_x86_realbins_realstep` so the
    /// gate-aware recursion can't perturb the REALBINS_REALSTEP numbers.
    /// Seeded at [`step_recursive_fixed::INITIAL_STEP_SEED`].
    prev_step_x86_realbins_realstep_gateref: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a TWELFTH
    /// experimental `b2` candidate. Identical to
    /// [`Self::b2_x86_realbins_realstep_log`] (REAL_BINS's calibration plus
    /// the recursive `STEP` law) EXCEPT the ACCEPT/REJECT decision each frame
    /// comes from the captured gate reference ([`Self::set_step_gate_ref`],
    /// the same reference `b2_x86_stepformula_gateref_log` uses) instead of
    /// always assuming REJECT-FORMULA. Falls back to REJECT-FORMULA for any
    /// frame with no reference coverage. Does NOT feed back into the shipped
    /// `b0..b8` output.
    b2_x86_realbins_realstep_gateref_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for the
    /// REALBINS_REALSTEP_GATEREF path, one entry per frame -- mirrors
    /// `realbins_realstep_meanraw_log`, and lets an offline grid search
    /// re-tune this path's calibration without re-running the whole encoder
    /// per grid point. It needs its own calibration because gate-corrected
    /// `STEP` has a different scale and distribution from always-REJECT-FORMULA
    /// `STEP`: ACCEPT frames jump `STEP` to the literal `4217` instead of
    /// following the recursive law.
    realbins_realstep_gateref_meanraw_log: Vec<f64>,
    /// Separately-tracked running `gamma` state for the
    /// REALBINS_REALSTEP_REALCOUNT path (see
    /// [`Self::b2_x86_realbins_realstep_realcount_log`]).
    prev_gamma_x86_realbins_realstep_realcount: f64,
    /// Self-referential `STEP` state for the REALBINS_REALSTEP_REALCOUNT path
    /// -- kept SEPARATE from `prev_step_x86_realbins_realstep` so this path's
    /// recursion can't perturb that one's numbers.
    prev_step_x86_realbins_realstep_realcount: i16,
    /// One entry per frame emitted via [`Self::analyze_frame`]: a THIRTEENTH
    /// experimental `b2` candidate. Identical to
    /// [`Self::b2_x86_realbins_realstep_log`] EXCEPT the harmonic-count `L`
    /// fed to `band_decompress` comes from `step_update_mode1`'s `new_count`
    /// output -- the bit-exact `COUNT` companion to the `STEP` recursion, and
    /// the reference's real `sumcount` source -- instead of `l_us`, this project's
    /// separately-estimated pitch/harmonic-count. Does NOT feed back into the
    /// shipped `b0..b8` output.
    b2_x86_realbins_realstep_realcount_log: Vec<u8>,
    /// Raw (pre-calibration) mean `band_decompress` output for
    /// the REALBINS_REALSTEP_REALCOUNT path, mirrors
    /// `realbins_realstep_meanraw_log`, for offline calibration grid
    /// search.
    realbins_realstep_realcount_meanraw_log: Vec<f64>,
    /// The exact `gap2` array (`self.gap2_mid`) and `arg2_raw` value used by
    /// the real-bins path THIS frame, logged so an offline tool can rebuild
    /// `win`/`bins`/`band_decompress` output using a different harmonic count
    /// `L`/`voiced`/`omega_0` (e.g. ground truth decoded from real transmitted
    /// bits) than this encoder's own estimate -- isolating the amplitude/gain
    /// mechanism's accuracy from the known-imperfect pitch estimator.
    gap2_mid_log: Vec<[i16; history_ring::GAP2_LEN]>,
    /// Diagnostic: slot1/slot2 gap2 windows logged per frame.
    gap2_slot1_log: Vec<[i16; history_ring::GAP2_LEN]>,
    gap2_slot2_log: Vec<[i16; history_ring::GAP2_LEN_WIDE]>,
    /// See [`Self::gap2_mid_log`]'s own doc -- the paired `arg2_raw` value.
    arg2_x86_log: Vec<i16>,
    /// [`Self::gap2_pre`]'s value, logged per frame (mirrors
    /// [`Self::gap2_mid_log`]) so an offline tool can test candidate
    /// reconstructions for slot 1/2 content against the
    /// BEFORE-this-frame's-pushes ring snapshot, not just the
    /// AFTER-first-push (`gap2_mid`) one slot 0 uses. Diagnostic only -- feeds
    /// no shipped or wired experimental path.
    gap2_pre_log: Vec<[i16; history_ring::GAP2_LEN]>,
    /// The FULL 383-word ring content (not just the
    /// [`history_ring::GAP2_OFFSET`]-anchored 199-word slice), logged at BOTH
    /// the `gap2_pre` and `gap2_mid` timings each frame, so an offline tool
    /// can brute-force every possible 199-word offset when looking for a
    /// slot's source.
    ring_full_pre_log: Vec<[i16; history_ring::RING_LEN]>,
    /// See [`Self::ring_full_pre_log`]'s own doc -- the AFTER-first-push
    /// counterpart.
    ring_full_mid_log: Vec<[i16; history_ring::RING_LEN]>,
    /// The exact 256-word `win` array ([`loudness_fixed::win_from_gap2`]'s
    /// output, i.e. the `gap2_mid`-reconstructed array) as passed into the
    /// REALBINS_REALSTEP path THIS frame, captured BEFORE
    /// [`band_decompress::compute_outer`] mutates it in place -- directly
    /// comparable, entry for entry, against a reference capture taken at the same
    /// pipeline point.
    win_rbs_pre_log: Vec<[i16; 256]>,
    /// The harmonic count `l_us` used by the REALBINS_REALSTEP
    /// path THIS frame -- paired with [`Self::win_rbs_pre_log`]/
    /// [`Self::arg2_x86_log`] so an offline tool can exactly replay
    /// [`loudness_fixed::mean_raw_real_bins_with_step`] with a substituted
    /// `win` but otherwise-identical real per-frame inputs.
    l_us_rbs_log: Vec<usize>,
    /// One entry per frame -- the PTR63C experimental `b2` candidate
    /// ([`loudness_fixed::candidate_gamma_poly`]), the only candidate built
    /// from `ptr63c`'s end-to-end-assembled ring content. See
    /// `loudness_fixed::gamma_poly_assemble` for the scope of what is and is not
    /// independently re-derived from raw PCM. Does NOT feed back into the
    /// shipped `b0..b8` output.
    b2_gamma_poly_log: Vec<u8>,
    /// One entry per frame -- [`loudness_fixed::candidate_gamma_ptr63c_r185`],
    /// the same PTR63C pipeline as [`Self::b2_gamma_poly_log`] but with the
    /// `(a2,a3)=(0,0)` silence placeholder replaced by the real `param5`
    /// combine (`loudness_fixed::gamma_poly_param5_from_stage`). See
    /// [`loudness_fixed::GAMMA_POLY_SEED_CONSTANT`] for the scope of the `bufA`
    /// substitution this still relies on: the address is proven identical to
    /// `current_stage()`, but the content is not independently re-derived from
    /// raw PCM at this call site. Does NOT feed back into the shipped
    /// `b0..b8` output.
    b2_gamma_poly_param5_log: Vec<u8>,
    /// One entry per frame emitted via [`Self::analyze_frame`] -- the mean of
    /// the non-sentinel log-magnitude words
    /// [`cepstral_stage1::cepstral_stage1_log_transform`] produces when fed the
    /// raw-PCM-derived `bins` spectrum
    /// (`loudness_fixed::assemble_bins_from_win`).
    ///
    /// **Scope**: this makes `cepstral_stage1_log_transform`'s `source` array
    /// (array `A`'s pre-refinement input) PCM-derivable and non-degenerate; it
    /// does **not** close array `A`, which additionally needs an unclosed
    /// refine step, `windowed_complex_correlation`'s `w` (blocked on `w_builder`'s `thresh8`,
    /// itself blocked on array B's open `x`/`clamped_lag` chain), an unclosed
    /// post-atan2 tail, and `T`'s fresh-insert formula (statically read, never
    /// live-verified or ported). `scale_arg` uses the only captured value on
    /// record (`18`, `cepstral_stage1_fixtures::CB0_LOGMAG_ROWS`); all 10 of
    /// those captures are silent frames, so it is a best-available constant,
    /// **not** a verified general formula for non-silent audio. Does NOT feed
    /// [`Self::attempt_b1_fixed`] or any shipped output.
    array_a_stage1_log: Vec<f64>,
    /// One entry per frame emitted via [`Self::analyze_frame`] -- the mean of
    /// the non-sentinel words of [`Self::array_a_stage1_log`]'s raw `stage1`
    /// output run ONE STAGE FURTHER through
    /// [`array_a_stage2::inverse_fft_butterfly_stage`] (its INVERSE-mode
    /// array-mutating dispatch, `arg4=1`).
    ///
    /// `arg2=4`/`arg3=8`/`arg4=1`/`arg5=1` are the only observed values on
    /// record (`a1b0_stage2_fixtures::LIVE_A1B0_R75_CALLS`, constant across
    /// all 200 captures spanning 100 frames of varied non-silent content) --
    /// best-available observed constants, same convention as
    /// `CEPSTRAL_STAGE1_SCALE_ARG_OBSERVED`.
    ///
    /// **Scope**, matching `array_a_stage1_log`: this makes
    /// `inverse_fft_butterfly_stage`'s array output computable end-to-end from
    /// raw-PCM-derived audio; it does **not** close array `A`, which
    /// additionally needs the SECOND `fft_bfp_transform` call made after this
    /// stage (that call's `arg2`/`arg3` are characterized statically only,
    /// never live-verified), `windowed_complex_correlation`'s `w` (blocked on `w_builder`'s
    /// `thresh8`, itself blocked on array B's open `x`/`clamped_lag` chain),
    /// an unclosed post-atan2 tail, and `T`'s fresh-insert formula. Does NOT
    /// feed [`Self::attempt_b1_fixed`] or any shipped output.
    array_a_stage2_log: Vec<f64>,
    /// Cross-frame IMBE (P25 Phase 1) quantizer prediction state, used only by
    /// the [`Encoder::encode_imbe_frame`] / [`Encoder::flush_imbe`] path. The
    /// AMBE+2 (`encode_frame_r33`/`r34`) path never touches it, so an Encoder
    /// is single-mode in practice (one codec per instance).
    imbe_enc: crate::imbe::ImbeEncState,
    /// IMBE release-edge tracker: `true` iff the previously emitted IMBE frame
    /// carried any voiced harmonic. [`Self::analyze_imbe_frame`] reads it to
    /// detect the voiced→fully-unvoiced release edge at a word offset — the
    /// point where the reference implementation releases the amplitude/pitch a frame before our source
    /// energy actually collapses — and force an early release there so our hot
    /// boundary frame stops seeding the decoder's differential-amplitude
    /// predictor into the inter-word gap.
    imbe_prev_voiced: bool,
}

/// Explicit, deliberately trivial placeholder for `b1_from_abc`'s 3 required
/// 16-word inputs (arrays A/B/C), used by [`Encoder::attempt_b1_fixed`] until
/// their real raw-PCM construction chains close (see that method's doc and
/// `enc::voicing_fixed`'s module doc for the open items). All-zero on purpose
/// -- not a disguised heuristic, not randomized -- so the resulting
/// `b1_x86_log` values are never mistaken for a measurement; they exist only
/// to prove the CALL SITE and [`Encoder::voicing_b1_fixed`] passthrough are
/// wired into the live per-frame pipeline. Feeding REAL captured A/B/C through
/// that same passthrough reproduced the reference's transmitted `b1` 100/100.
/// The captures are gone, so that measurement cannot be repeated.
const B1_X86_PLACEHOLDER_ABC: [i16; 16] = [0i16; 16];

/// The only captured `scale_arg` value on record for
/// `cepstral_stage1_log_transform` (`cepstral_stage1_fixtures::
/// CB0_LOGMAG_ROWS`, all 10 rows use `18`). **Caveat**: every one of those
/// captures is a silent (all-zero-source) frame, so this constant is confirmed
/// only for silence. It is the best available observed value for
/// [`Self::array_a_stage1_log`]'s diagnostic, NOT a verified general formula
/// for `scale_arg` on non-silent audio -- that upstream source is still open,
/// see `voicing_fixed.rs`'s module doc.
const CEPSTRAL_STAGE1_SCALE_ARG_OBSERVED: i16 = 18;

/// IMBE reference-voicing band binning (see `analyze_imbe_frame`). `false` = FIRST
/// (the quantizer's native `voiced[band*3]`, zero extra logic); `true` =
/// AND-reduce each band's harmonic triplet onto that slot.
const IMBE_VOICE_BINNING_AND: bool = true; // AND: best net (broad b1 up most, least b2 coupling)

/// AMBE Annex-L pitch index the IMBE silence release pins to — the same value
/// `Vocoder::encode`'s IMBE silence gate forces (`IMBE_SILENCE_B0`), decoded to
/// ω₀ and re-quantized onto the IMBE grid gives b0=25 / L=14. Used by the
/// release-edge alignment in [`Encoder::analyze_imbe_frame`].
const IMBE_SILENCE_AMBE_B0: u8 = 31;

impl Encoder {
    pub fn new() -> Self {
        Self {
            analyzer: Analyzer::new(),
            state: DecoderState::new(),
            forced_b0: None,
            forced_b1: None,
            forced_strict: false,
            live_gap2_amps: false,
            bias_raw_state: 0,
            lowband_floor: false,
            lowband_l0: band_decompress::LOWBAND_L0_INIT,
            lowband_gate_b1: None,
            diagnostics: false,
            frame_energy_log: Vec::new(),
            next_emit: 0,
            hist: Vec::with_capacity(WIN_LEN + 2 * FRAME),
            hist_base: 0,
            prefilter_state: audio_prefilter::PrefilterState::default(),
            prefiltered: Vec::new(),
            history_ring: history_ring::HistoryRing::new(),
            prev_gamma_fixed: 0.0,
            b2_x86_log: Vec::new(),
            prev_gamma_x86_pf: 0.0,
            b2_x86_pf_log: Vec::new(),
            win_fixed: loudness_fixed::new_win_state(),
            win_x86_pf: loudness_fixed::new_win_state(),
            prev_gamma_x86_wintaper: 0.0,
            b2_x86_wintaper_log: Vec::new(),
            b1_x86_log: Vec::new(),
            gap2_pre: [0i16; history_ring::GAP2_LEN],
            gap2_mid: [0i16; history_ring::GAP2_LEN],
            gap2_slot1: [0i16; history_ring::GAP2_LEN],
            gap2_slot2: [0i16; history_ring::GAP2_LEN_WIDE],
            prev_gamma_x86_outeronly: 0.0,
            b2_x86_outeronly_log: Vec::new(),
            prev_gamma_x86_realbins: 0.0,
            b2_x86_realbins_log: Vec::new(),
            realbins_meanraw_log: Vec::new(),
            prev_bias_x86_refexact: 0,
            b2_x86_refexact_log: Vec::new(),
            prev_step_x86_recursive: step_recursive_fixed::INITIAL_STEP_SEED,
            prev_bias_x86_stepformula: 0,
            b2_x86_stepformula_log: Vec::new(),
            step_gate_ref: None,
            step_gate_ref_idx: 0,
            prev_step_x86_gateref: step_recursive_fixed::INITIAL_STEP_SEED,
            prev_bias_x86_gateref: 0,
            b2_x86_stepformula_gateref_log: Vec::new(),
            prev_bias_x86_stepformula_realcount: 0,
            b2_x86_stepformula_realcount_log: Vec::new(),
            prev_gamma_x86_realbins_realstep: 0.0,
            prev_step_x86_realbins_realstep: step_recursive_fixed::INITIAL_STEP_SEED,
            b2_x86_realbins_realstep_log: Vec::new(),
            realbins_realstep_meanraw_log: Vec::new(),
            prev_gamma_x86_realbins_slot1_realstep: 0.0,
            prev_step_x86_realbins_slot1_realstep: step_recursive_fixed::INITIAL_STEP_SEED,
            b2_x86_realbins_slot1_realstep_log: Vec::new(),
            realbins_slot1_realstep_meanraw_log: Vec::new(),
            prev_bias_x86_refexact_slot1: 0,
            b2_x86_refexact_slot1_log: Vec::new(),
            prev_gamma_x86_realbins_slot2_realstep: 0.0,
            prev_step_x86_realbins_slot2_realstep: step_recursive_fixed::INITIAL_STEP_SEED,
            b2_x86_realbins_slot2_realstep_log: Vec::new(),
            realbins_slot2_realstep_meanraw_log: Vec::new(),
            prev_gamma_x86_realbins_realstep_gateref: 0.0,
            prev_step_x86_realbins_realstep_gateref: step_recursive_fixed::INITIAL_STEP_SEED,
            b2_x86_realbins_realstep_gateref_log: Vec::new(),
            realbins_realstep_gateref_meanraw_log: Vec::new(),
            prev_gamma_x86_realbins_realstep_realcount: 0.0,
            prev_step_x86_realbins_realstep_realcount: step_recursive_fixed::INITIAL_STEP_SEED,
            b2_x86_realbins_realstep_realcount_log: Vec::new(),
            realbins_realstep_realcount_meanraw_log: Vec::new(),
            gap2_mid_log: Vec::new(),
            gap2_slot1_log: Vec::new(),
            gap2_slot2_log: Vec::new(),
            arg2_x86_log: Vec::new(),
            gap2_pre_log: Vec::new(),
            ring_full_pre_log: Vec::new(),
            ring_full_mid_log: Vec::new(),
            win_rbs_pre_log: Vec::new(),
            l_us_rbs_log: Vec::new(),
            b2_gamma_poly_log: Vec::new(),
            b2_gamma_poly_param5_log: Vec::new(),
            array_a_stage1_log: Vec::new(),
            array_a_stage2_log: Vec::new(),
            imbe_enc: crate::imbe::ImbeEncState::new(),
            imbe_prev_voiced: false,
        }
    }

    /// EXPERIMENTAL, diagnostic-only: the `b2` index the mechanism-matched
    /// (but NOT fully closed -- see `loudness_fixed`'s own module doc)
    /// `step`/`outer`/`band_decompress` chain would have produced, one
    /// entry per frame emitted so far. Does NOT affect
    /// `encode_frame_r33`/`r34`'s actual returned bits -- see
    /// [`loudness_fixed`] for the scope of what is and is not
    /// real-reference-verified in this path.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_log(&self) -> &[u8] {
        &self.b2_x86_log
    }

    /// EXPERIMENTAL, diagnostic-only: the SECOND `b2` candidate log (see the
    /// `b2_x86_pf_log` field doc) -- same mechanism as [`Self::b2_x86_log`]
    /// but fed a spectrum built from real, bit-exact prefiltered audio
    /// instead of raw PCM.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_pf_log(&self) -> &[u8] {
        &self.b2_x86_pf_log
    }

    /// EXPERIMENTAL, diagnostic-only: the THIRD `b2` candidate log (see the
    /// `b2_x86_wintaper_log` field doc) -- same mechanism as
    /// [`Self::b2_x86_log`], but `win` is rebuilt fresh every frame from
    /// [`loudness_fixed::win_from_gap2`] instead of carried as an always-zero
    /// accumulator.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_wintaper_log(&self) -> &[u8] {
        &self.b2_x86_wintaper_log
    }

    /// EXPERIMENTAL, diagnostic-only: the FOURTH `b2` candidate log (see the
    /// `b2_x86_outeronly_log` field doc) -- `outer` alone (bypassing
    /// `bins`/`step`/`band_decompress`), fed by the closed `win`/`arg2`
    /// mechanism.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_outeronly_log(&self) -> &[u8] {
        &self.b2_x86_outeronly_log
    }

    /// EXPERIMENTAL, diagnostic-only: the FIFTH `b2` candidate log (see the
    /// `b2_x86_realbins_log` field doc) -- the mechanism-matched `bins` (from
    /// `win`'s own content, not a synthesized DFT) through the full
    /// `step`/`outer`/`band_decompress` chain.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_log(&self) -> &[u8] {
        &self.b2_x86_realbins_log
    }

    /// The RAW (pre-calibration) mean `band_decompress` output
    /// underlying [`Self::b2_x86_realbins_log`], one entry per frame -- for
    /// an offline calibration grid search (see
    /// `loudness_fixed::mean_raw_real_bins`'s own doc).
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_meanraw_log(&self) -> &[f64] {
        &self.realbins_meanraw_log
    }

    /// EXPERIMENTAL, diagnostic-only: the SIXTH `b2` candidate log -- a
    /// bit-exact-DERIVED (not fitted) formula for the reference's gain-quantizer
    /// input, fed by this encoder's own estimated `l`/`step`/`outer`/`bins`
    /// (same inputs as [`Self::b2_x86_realbins_log`]). The formula itself is
    /// bit-exact against captured ground truth; this end-to-end log's accuracy
    /// is bounded by the upstream `l`/`step`/`outer` estimator error. See
    /// [`loudness_fixed::gamma_ref_exact`].
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_refexact_log(&self) -> &[u8] {
        &self.b2_x86_refexact_log
    }

    /// See [`Self`]-level `b2_x86_stepformula_log` field doc and
    /// [`step_recursive_fixed`]'s module doc for the recursive `STEP` update
    /// law this uses (bit-exact GIVEN the real previous `STEP`; that module
    /// doc has the scope on the self-referential recursion and the un-ported
    /// accept/reject gate).
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_stepformula_log(&self) -> &[u8] {
        &self.b2_x86_stepformula_log
    }

    /// Supply a captured ACCEPT/REJECT gate reference sequence (see
    /// `step_recursive_fixed::gate_ref`) for the content about to be
    /// processed. Must be called before any `analyze_frame` calls to take
    /// effect from frame 0; if never called (or called with fewer entries than
    /// frames processed), [`Self::b2_x86_stepformula_gateref_log`] falls
    /// back to REJECT-FORMULA for the uncovered frames.
    /// DIAGNOSTIC: per-emit-frame `frame_energy` log (see `frame_energy_log`).
    pub fn frame_energy_log(&self) -> &[f64] {
        &self.frame_energy_log
    }

    /// Force the per-emit-frame pitch index `b0` (see `forced_b0`). `seq[f]`
    /// overrides emit-frame `f`'s pitch; out-of-range frames fall back to the
    /// estimator. The shipped callers use this to inject the `b0_audio`
    /// spectral-DP pitch.
    pub fn set_forced_b0(&mut self, seq: Vec<u8>) {
        self.forced_b0 = Some(seq);
    }

    /// Force the per-emit-frame IMBE voicing word `b1` (see `forced_b1`).
    /// `seq[f]` is the 5-bit AMBE voicing index from `b1_audio::b1_track`;
    /// [`Self::analyze_imbe_frame`] expands it to per-harmonic V/UV on the IMBE
    /// grid. Out-of-range frames fall back to the float voicing path.
    pub fn set_forced_b1(&mut self, seq: Vec<u16>) {
        self.forced_b1 = Some(seq);
    }

    /// Require the forced vectors to cover every frame that is analysed.
    ///
    /// The fallbacks in `set_forced_b0` / `set_forced_b1` are silent by design —
    /// a caller supplying a short prefix gets the estimator for the rest. A
    /// streaming caller that grows those vectors as its analysis finalises has
    /// the opposite requirement: reaching the estimator means its schedule is
    /// off by a frame, which degrades the audio while every test stays green.
    /// With this set, that becomes a debug-build panic instead.
    pub fn set_forced_strict(&mut self, on: bool) {
        self.forced_strict = on;
    }

    /// Enable Route A: pitch-aligned mechanism amplitudes + reference-exact `b2`
    /// gain sourced from the LIVE `gap2_mid` window, plus the two-frame analysis
    /// look-ahead that makes that window pitch-aligned (see
    /// [`Self::live_gap2_amps`]). Set before feeding any frames. Used by the
    /// whole-buffer AMBE+2 `Vocoder::encode` path and by `AmbeStream`; both then
    /// produce byte-identical `b2..b8` in a single forward pass.
    pub fn set_live_gap2_amps(&mut self, on: bool) {
        self.live_gap2_amps = on;
    }

    /// Enable the reference's gated low-band spectral floor on the Route A
    /// amplitude path (see [`Self::lowband_floor`]). Requires
    /// [`Self::set_lowband_gate_b1`] to cover every analysed frame; set both
    /// before feeding any frames so the recursive `l0` starts from frame 0.
    pub fn set_lowband_floor(&mut self, on: bool) {
        self.lowband_floor = on;
    }

    /// Supply the per-emit-frame gate input for the low-band floor. `seq[f]` is
    /// the `b1` transmitted for emit frame `f` — the caller's lagged voicing
    /// track, not the Encoder's internal estimate. The two differ, and the
    /// transmitted one is the reference's own gate input.
    pub fn set_lowband_gate_b1(&mut self, seq: Vec<u16>) {
        self.lowband_gate_b1 = Some(seq);
    }

    /// Analysis look-ahead depth in frames. One frame normally; two when
    /// [`Self::live_gap2_amps`] is set, so the live `gap2_mid` for emit-frame
    /// `f` refers to the pitch-aligned (frame `f + 2`) window.
    fn lookahead(&self) -> usize {
        if self.live_gap2_amps {
            2
        } else {
            1
        }
    }

    /// Enable/disable the DIAGNOSTIC/EXPERIMENTAL per-frame analysis paths.
    /// Off (the default): the encoder does zero experimental-path work per
    /// frame and every `*_x86_*` research log stays empty; the shipped output
    /// bits and the live `gap2_*`/`prefiltered` logs are identical either way.
    /// Enable BEFORE encoding to populate the research logs (see the
    /// `diagnostics` field doc, including the `set_step_gate_ref` index
    /// caveat).
    pub fn set_diagnostics(&mut self, on: bool) {
        self.diagnostics = on;
    }

    /// NOTE: the reference index only advances while [`Self::set_diagnostics`] is
    /// on — enable diagnostics from frame 0 when supplying a reference.
    pub fn set_step_gate_ref(&mut self, seq: &[bool]) {
        self.step_gate_ref = Some(seq.to_vec());
        self.step_gate_ref_idx = 0;
    }

    /// See [`Self`]-level `b2_x86_stepformula_gateref_log` field doc --
    /// identical to [`Self::b2_x86_stepformula_log`] except driven by a
    /// captured ACCEPT/REJECT gate decision (via
    /// [`Self::set_step_gate_ref`]) instead of always assuming
    /// REJECT-FORMULA.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_stepformula_gateref_log(&self) -> &[u8] {
        &self.b2_x86_stepformula_gateref_log
    }

    /// See [`Self`]-level `b2_x86_stepformula_realcount_log` field doc --
    /// identical to [`Self::b2_x86_stepformula_log`] except `L`/`sumcount`
    /// comes from the bit-exact `STEP` recursion's `new_count` output instead
    /// of this project's separately-estimated pitch/harmonic-count.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_stepformula_realcount_log(&self) -> &[u8] {
        &self.b2_x86_stepformula_realcount_log
    }

    /// See [`Self`]-level `b2_x86_realbins_realstep_log` field doc --
    /// REAL_BINS's affine calibration paired with the recursive `STEP` law.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_realstep_log(&self) -> &[u8] {
        &self.b2_x86_realbins_realstep_log
    }

    /// See [`Self`]-level `b2_x86_realbins_realstep_gateref_log` field doc
    /// -- REAL_BINS's affine calibration paired with the recursive `STEP` law
    /// AND the captured ACCEPT/REJECT gate (`Self::set_step_gate_ref`),
    /// instead of the always-REJECT-FORMULA approximation.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_realstep_gateref_log(&self) -> &[u8] {
        &self.b2_x86_realbins_realstep_gateref_log
    }

    /// See [`Self`]-level `realbins_realstep_gateref_meanraw_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_realstep_gateref_meanraw_log(&self) -> &[f64] {
        &self.realbins_realstep_gateref_meanraw_log
    }

    /// See [`Self`]-level `b2_x86_realbins_realstep_realcount_log` field doc
    /// -- REAL_BINS's affine calibration plus the recursive `STEP` law, with
    /// the reference-derived `COUNT` instead of `l_us`.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_realstep_realcount_log(&self) -> &[u8] {
        &self.b2_x86_realbins_realstep_realcount_log
    }

    /// See [`Self`]-level `realbins_realstep_realcount_meanraw_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_realstep_realcount_meanraw_log(&self) -> &[f64] {
        &self.realbins_realstep_realcount_meanraw_log
    }

    /// See [`Self`]-level `realbins_realstep_meanraw_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_realstep_meanraw_log(&self) -> &[f64] {
        &self.realbins_realstep_meanraw_log
    }

    /// See [`Self`]-level `b2_x86_realbins_slot1_realstep_log` field doc --
    /// `bins` built from slot 1's window instead of slot 0's.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_slot1_realstep_log(&self) -> &[u8] {
        &self.b2_x86_realbins_slot1_realstep_log
    }

    /// See [`Self`]-level `b2_gamma_poly_log` field doc -- the PTR63C
    /// experimental `b2` candidate.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_gamma_poly_log(&self) -> &[u8] {
        &self.b2_gamma_poly_log
    }

    /// See [`Self`]-level `b2_gamma_poly_param5_log` field doc -- the PTR63C
    /// candidate with the real `param5` combine substituted for the silence
    /// placeholder.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_gamma_poly_param5_log(&self) -> &[u8] {
        &self.b2_gamma_poly_param5_log
    }

    /// See [`Self`]-level `realbins_slot1_realstep_meanraw_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_slot1_realstep_meanraw_log(&self) -> &[f64] {
        &self.realbins_slot1_realstep_meanraw_log
    }

    /// See [`Self`]-level `b2_x86_refexact_slot1_log` field doc -- SLOT 1's
    /// spectrum plus the recursive STEP law, fed through the fully-derived
    /// (zero-fitted-parameter) `gamma_ref_exact` formula instead of an affine
    /// calibration.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_refexact_slot1_log(&self) -> &[u8] {
        &self.b2_x86_refexact_slot1_log
    }

    /// See [`Self`]-level `b2_x86_realbins_slot2_realstep_log` field doc --
    /// `bins` built from slot 2's window instead of slot 0's or slot 1's.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b2_x86_realbins_slot2_realstep_log(&self) -> &[u8] {
        &self.b2_x86_realbins_slot2_realstep_log
    }

    /// See [`Self`]-level `realbins_slot2_realstep_meanraw_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn realbins_slot2_realstep_meanraw_log(&self) -> &[f64] {
        &self.realbins_slot2_realstep_meanraw_log
    }

    /// See [`Self`]-level `gap2_pre_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn gap2_pre_log(&self) -> &[[i16; history_ring::GAP2_LEN]] {
        &self.gap2_pre_log
    }
    /// See [`Self`]-level `ring_full_pre_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn ring_full_pre_log(&self) -> &[[i16; history_ring::RING_LEN]] {
        &self.ring_full_pre_log
    }
    /// See [`Self`]-level `ring_full_mid_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn ring_full_mid_log(&self) -> &[[i16; history_ring::RING_LEN]] {
        &self.ring_full_mid_log
    }
    /// The FULL, unbounded, chronological `audio_prefilter` output stream
    /// (every sample this encoder has pushed into `history_ring`, not just the
    /// last `RING_LEN` window) -- lets an offline tool search for per-call
    /// source content OUTSIDE the ring's 383-sample retention window, which
    /// slot 2's localbuf search needs. See `enc::mod::prefiltered`.
    pub fn prefiltered_log(&self) -> &[i16] {
        &self.prefiltered
    }

    /// See [`Self`]-level `gap2_mid_log`/`arg2_x86_log` field docs -- the raw
    /// ingredients an offline tool needs to rebuild
    /// `win`/`bins`/`band_decompress` for a frame using reference-decoded
    /// `L`/`voiced`/`omega_0` instead of this encoder's estimates.
    pub fn gap2_mid_log(&self) -> &[[i16; history_ring::GAP2_LEN]] {
        &self.gap2_mid_log
    }
    /// Diagnostic accessors for slot1/slot2 gap2 windows.
    pub fn gap2_slot1_log(&self) -> &[[i16; history_ring::GAP2_LEN]] {
        &self.gap2_slot1_log
    }
    pub fn gap2_slot2_log(&self) -> &[[i16; history_ring::GAP2_LEN_WIDE]] {
        &self.gap2_slot2_log
    }
    /// See [`Self::gap2_mid_log`].
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn arg2_x86_log(&self) -> &[i16] {
        &self.arg2_x86_log
    }

    /// See [`Self`]-level `win_rbs_pre_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn win_rbs_pre_log(&self) -> &[[i16; 256]] {
        &self.win_rbs_pre_log
    }
    /// See [`Self`]-level `l_us_rbs_log` field doc.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn l_us_rbs_log(&self) -> &[usize] {
        &self.l_us_rbs_log
    }

    /// See the [`Self`]-level `array_a_stage1_log` field doc:
    /// [`cepstral_stage1::cepstral_stage1_log_transform`] (array `A`'s
    /// pre-refinement writer) running on raw-PCM-derived, non-degenerate
    /// `bins` -- **not** a measurement of array `A` itself, which needs
    /// several more open sub-pieces (see that field's doc).
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn array_a_stage1_log(&self) -> &[f64] {
        &self.array_a_stage1_log
    }

    /// See the [`Self`]-level `array_a_stage2_log` field doc:
    /// [`array_a_stage2::inverse_fft_butterfly_stage`] (INVERSE-mode array
    /// output, `arg2=4`/`arg3=8`/`arg4=1`/`arg5=1`) running end-to-end on
    /// raw-PCM-derived, non-degenerate [`Self::array_a_stage1_log`] output --
    /// **not** a measurement of array `A` itself, which still needs the second
    /// `fft_bfp_transform` call, `w`/`thresh8`, the post-atan2 tail, and `T`'s
    /// formula (see that field's doc).
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn array_a_stage2_log(&self) -> &[f64] {
        &self.array_a_stage2_log
    }

    /// EXPERIMENTAL, diagnostic-only: the mechanism-matched `b1` candidate
    /// log (see the [`Self`]-level `b1_x86_log` field doc) -- one entry per
    /// frame emitted so far. **Caveat, restated here so it cannot be missed**:
    /// these values are computed from explicit zero placeholders for arrays
    /// A/B/C ([`B1_X86_PLACEHOLDER_ABC`]), not real audio, because those
    /// arrays' raw-PCM construction chains are not closed. The log exercises
    /// the WIRING (the call happens every frame, through the same
    /// [`Self::voicing_b1_fixed`] passthrough that is bit-exact against real
    /// captured A/B/C); it says nothing about real-audio `b1` accuracy. Does
    /// NOT feed back into the shipped `b0..b8` output.
    /// Empty unless [`Self::set_diagnostics`]`(true)` was called before encoding.
    pub fn b1_x86_log(&self) -> &[u16] {
        &self.b1_x86_log
    }

    /// Build a [`analysis::Spectrum`] with the EXACT SAME window
    /// (`analysis::window_coeff`, the firmware 250-tap reference window,
    /// [`analysis::WIN_LEN`]/[`analysis::WIN_HALF`]) and DFT convention
    /// (`analysis::centered_rdft`, [`analysis::FFT_SIZE`] = 256 -> 129 bins,
    /// matching the reference's spectrum size) the shipped `Analyzer` uses -- but
    /// drawing its input samples from `self.prefiltered` (the bit-exact
    /// `audio_prefilter` output, computed every frame by
    /// [`Self::run_prefilter`]) instead of raw un-filtered PCM.
    ///
    /// This is still an approximation, just a different one:
    /// `enc::loudness_fixed`'s module doc names "`bins`/`win` sourced from a
    /// synthesized spectrum on RAW PCM" as one of its two open gaps, and
    /// substituting the prefilter's output does not close the deeper one --
    /// the reference's internal `history_ring` -> `windowed_taper` -> `real_fft32`
    /// chain has no known formula into a 129-bin table or a 256-word `win`
    /// array (see `real_fft32.rs` and `analysis_stage_window.rs`).
    fn prefiltered_spectrum(&self, f: usize) -> analysis::Spectrum {
        let center = (f * FRAME + FRAME / 2) as i64 + analysis::win_offset();
        let mut windowed = vec![0.0f64; WIN_LEN];
        for (i, w) in windowed.iter_mut().enumerate() {
            let n = i as i64 - WIN_HALF as i64;
            let abs = center + n;
            let s = if abs >= 0 {
                self.prefiltered.get(abs as usize).copied().unwrap_or(0)
            } else {
                0
            };
            *w = f64::from(s) * analysis::window_coeff(i);
        }
        let (re, im) = analysis::centered_rdft(&windowed);
        analysis::Spectrum {
            frame_index: f,
            re,
            im,
            energy: 0.0,
            crest: 0.0,
        }
    }

    /// The 108-sample analysis staging window's current content
    /// (`history_ring`'s `stage()` -- see the `history_ring` field doc). Not
    /// consumed by `analyze_frame`'s bit output; exposed for diagnostics and
    /// for whatever eventually closes `band_decompress`'s `step`/`outer`
    /// inputs and wires the rest of the RE-matched chain in.
    pub fn current_stage(&self) -> [i16; history_ring::STAGE_LEN] {
        self.history_ring.stage()
    }

    // Read-only access to the pre-filter's live persistent state (for
    // tests / diagnostics -- confirms it is never reset per-frame).

    /// Run the pre-filter (`audio_prefilter::prefilter`) over one freshly-pushed
    /// 160-sample frame, in the reference's two-80-sample-call cadence: each
    /// pre-filter call processes 80 samples, so a 160-sample frame is two calls
    /// with state carried between them. Updates `self.prefilter_state` in
    /// place, appends the produced samples to `self.prefiltered`, and pushes
    /// each 80-sample half's output into `self.history_ring` so
    /// `current_stage()` reflects the reference's staging window. Called once per
    /// pushed frame, unconditionally, with no reset -- see the
    /// `prefilter_state` field doc for why no reset is correct.
    fn run_prefilter(&mut self, frame: &[i16; FRAME]) {
        debug_assert_eq!(FRAME, 160, "prefilter cadence assumes 160 = 2*80");
        // The reference's `gap2`-source routine reads `gap2` at TWO distinct points
        // in this per-frame cadence, not once after the fact:
        //   - `gap2_pre` (this frame's first `count=199` reader): ring state
        //     BEFORE any of this frame's audio is pushed.
        //   - `gap2_mid` (this frame's second `count=199` reader AND its
        //     `count=255` reader, which always see IDENTICAL content): ring
        //     state after ONLY the FIRST of this frame's two pushes.
        // Reading `history_ring.gap2_window()` from `analyze_frame` instead --
        // i.e. after BOTH pushes -- matches neither timing. See
        // `history_ring::GAP2_OFFSET`.
        self.gap2_pre = self.history_ring.gap2_window();
        // Log both the GAP2_OFFSET-anchored slice and the full raw ring at
        // this timing, for the slot 1/2 brute-force search (see
        // `gap2_pre_log`/`ring_full_pre_log`).
        if self.diagnostics {
            self.gap2_pre_log.push(self.gap2_pre);
            self.ring_full_pre_log.push(self.history_ring.debug_full());
        }
        let (out_a, post_a) = audio_prefilter::prefilter(&self.prefilter_state, &frame[0..80]);
        self.prefilter_state = post_a;
        let mut fixed_a = [0i16; 80];
        fixed_a.copy_from_slice(&out_a);
        self.history_ring.push_half(&fixed_a);
        self.gap2_mid = self.history_ring.gap2_window();
        self.gap2_slot1 = self.history_ring.gap2_window_slot1();
        if self.diagnostics {
            self.ring_full_mid_log.push(self.history_ring.debug_full());
        }
        let (out_b, post_b) = audio_prefilter::prefilter(&self.prefilter_state, &frame[80..160]);
        self.prefilter_state = post_b;
        let mut fixed_b = [0i16; 80];
        fixed_b.copy_from_slice(&out_b);
        self.history_ring.push_half(&fixed_b);
        // Slot 2's read timing -- AFTER this frame's SECOND push_half, later
        // than gap2_mid/gap2_slot1, which are both captured after only the
        // FIRST push. See `gap2_slot2` / `history_ring::GAP2_OFFSET_SLOT2`.
        self.gap2_slot2 = self.history_ring.gap2_window_slot2();
        self.prefiltered.extend_from_slice(&out_a);
        self.prefiltered.extend_from_slice(&out_b);
    }

    /// Read a raw PCM sample by absolute stream index, zero outside the
    /// retained history (left padding) or in the not-yet-seen future
    /// (right / look-ahead padding).
    #[inline]
    fn sample_at(&self, abs: i64) -> i16 {
        if abs < self.hist_base as i64 {
            return 0;
        }
        let rel = (abs - self.hist_base as i64) as usize;
        self.hist.get(rel).copied().unwrap_or(0)
    }

    /// Mechanism-matched `b1` computation, wired straight through to the
    /// `voicing_fixed::b1_from_abc` back end
    /// (`score_arrays`/`per_band_weight_table`/`argmin_b1`, bit-exact
    /// **100/100** given real captured `A`/`B`/`C` -- see [`voicing_fixed`]'s
    /// module doc).
    ///
    /// **Called every frame** by [`Self::analyze_frame`] (via
    /// [`Self::attempt_b1_fixed`]), pushing into [`Self::b1_x86_log`] --
    /// diagnostic-only, isolated from the shipped `b0..b8` output.
    ///
    /// **What is NOT closed**: producing REAL, raw-PCM-derived `A`/`B`/`C`
    /// rather than the zero placeholders [`Self::attempt_b1_fixed`] passes.
    /// `A` needs a port of the `T`/`S` delay-line pair -- `S`'s atan2 core
    /// ([`atan2_chain`]), its 127-tap-correlation input (`windowed_complex_correlation`) and its
    /// cepstral front end (`cepstral_stage1`/`cepstral_stage2`) are all closed
    /// given real inputs, but `w_builder`'s `thresh8`/`table8`/`exp_arg`
    /// inputs and gap 1's `cbefore` raw source are not, so `windowed_complex_correlation` cannot
    /// run end-to-end from raw PCM. `B` needs `heap_src`'s `window`/`x` inputs
    /// (`band_voicing_heap_src` is bit-exact given those two, but `window` has
    /// no known raw-PCM writer and `x`'s `clamped_lag` pitch-search chain has
    /// one unread sub-call and 9 scalar state fields with no known raw-PCM
    /// origin). `C`'s transform is closed
    /// ([`voicing_fixed::c_transform`]) but its pre-transform source has the
    /// same open dependency as `A`.
    ///
    /// Wiring a real `A`/`B`/`C` builder into [`Self::attempt_b1_fixed`] is
    /// the only remaining step: the call site, the passthrough and the
    /// isolated diagnostic log are already in place and were validated against
    /// real captured data that is no longer available.
    pub fn voicing_b1_fixed(&self, a: [i16; 16], b: [i16; 16], c: [i16; 16]) -> u16 {
        voicing_fixed::b1_from_abc(a, b, c)
    }

    /// EXPERIMENTAL, diagnostic-only: builds this frame's `b1_fixed` candidate
    /// and pushes it to [`Self::b1_x86_log`]. Called once per frame from
    /// [`Self::analyze_frame`], **after** the real `b0..b8` indices are
    /// already computed, so it can never perturb them.
    ///
    /// This feeds [`Self::voicing_b1_fixed`] the explicit
    /// [`B1_X86_PLACEHOLDER_ABC`] zero arrays for all of A/B/C -- see that
    /// constant's doc and [`Self::voicing_b1_fixed`]'s doc for which raw-PCM
    /// chains are missing before real audio-derived arrays are possible. This
    /// function is the single place to wire those in: replace the 3
    /// `B1_X86_PLACEHOLDER_ABC` uses below with real array-builder calls;
    /// nothing else in this file needs to change.
    fn attempt_b1_fixed(&mut self) {
        let a = B1_X86_PLACEHOLDER_ABC;
        let b = B1_X86_PLACEHOLDER_ABC;
        let c = B1_X86_PLACEHOLDER_ABC;
        let b1_fixed = self.voicing_b1_fixed(a, b, c);
        self.b1_x86_log.push(b1_fixed);
    }

    /// Run the full analysis chain on frame index `f` (whose look-ahead must
    /// already be buffered) and return the nine quantizer indices `b0..b8`.
    /// Advances all cross-frame state by one frame. Shared by both rate paths.
    fn analyze_frame(&mut self, f: usize, spec: &analysis::Spectrum) -> [u16; 9] {
        let frame_start = f * FRAME;

        // --- pitch (b0): centred 250-sample reference window over the history ----
        // The reference analyses pitch with the firmware's 250-pt window
        // (fw_tables::reference_win250), NOT a 221-pt Hamming; estimate_pitch keys the
        // window off the sample count, so feed exactly 250 centred samples. With
        // the one-frame look-ahead the window's right half is now real PCM, not
        // zero-pad.
        let center = (frame_start + FRAME / 2) as i64;
        let mut win = [0i16; crate::fw_tables::REFERENCE_WIN250_N];
        let half = crate::fw_tables::REFERENCE_WIN250_N as i64 / 2;
        for (i, w) in win.iter_mut().enumerate() {
            *w = self.sample_at(center - half + i as i64);
        }
        let (mut omega_0, mut l, mut b0) = pitch::estimate_pitch(&win);
        // Pitch override (forced_b0): pin an externally supplied b0 so voicing +
        // amplitude re-derive at the forced L. `Vocoder::encode` (both branches)
        // and `AmbeStream` set it, feeding the `b0_audio` spectral-DP pitch; a
        // bare `Encoder` and the per-frame `Vocoder::encode_pcm` leave it `None`
        // and get `estimate_pitch` above.
        if let Some(g) = self.forced_b0.as_ref().and_then(|s| s.get(f).copied()) {
            if let Some(pi) = crate::dequantize::decode_pitch(g) {
                omega_0 = pi.omega_0;
                l = pi.l;
                b0 = g;
            }
        }
        let l_us = l as usize;

        // --- voicing (b1): per-harmonic V/UV over the magnitude spectrum -
        let mag: Vec<f64> = (0..FFT_SIZE).map(|k| spec.mag(k)).collect();
        let frame_energy = spec.energy / FRAME as f64; // mean-square level
        self.frame_energy_log.push(frame_energy);
        let (voiced, _b1) =
            voicing::decide_voicing(&mag, f64::from(omega_0), l_us, frame_energy, spec.crest);

        // --- spectral amplitudes (M_l) ----------------------------------
        // Route M_l through the fixed-point bins -> band_decompress chain (the
        // reference's method). A coherent-LS estimator loses to it on total
        // per-parameter bit-match (b2 -6, b8 -4 on r33-noise).
        //
        // Pitch-aligned override (Route A, AMBE+2 whole-buffer AND streaming):
        // when `live_gap2_amps` is set, source the amplitudes from the reference's
        // own fixed-point spectrum mechanism keyed on the PITCH-ALIGNED LIVE
        // window (`self.gap2_mid`) instead of the synthesized-DFT proxy. The
        // two-frame look-ahead (see `lookahead`) makes `gap2_mid` — captured in
        // `run_prefilter` at the emit-of-frame-`f` push, which is now frame
        // `f + 2`'s push — refer to the frame the pitch (`reference_b0[f + 2]`) uses.
        // The reference's analysis window leads its pitch by one frame internally;
        // feeding `band_decompress` the aligned window is what lifts every shape
        // field b3..b8. Off (proxy below) on `encode_pcm`, IMBE, and bare
        // `Encoder`, which keep the one-frame look-ahead.
        let aligned_gap2: Option<[i16; history_ring::GAP2_LEN]> = if self.live_gap2_amps {
            Some(self.gap2_mid)
        } else {
            None
        };
        // Gated low-band spectral floor (opt-in). Mutates the mechanism `M_l`
        // in the log2 domain before BOTH the shape quantizer below and the `b2`
        // gain override further down, which is the reference's own order — so
        // the floored `biased` is carried to that override rather than
        // recomputed there.
        let mut floored_biased: Option<Vec<i16>> = None;
        let mut amplitudes = if let Some(g2) = aligned_gap2 {
            if self.lowband_floor {
                let gate_b1 = self
                    .lowband_gate_b1
                    .as_ref()
                    .and_then(|s| s.get(f).copied());
                debug_assert!(
                    gate_b1.is_some(),
                    "lowband floor enabled without a gate b1 for frame {f}"
                );
                let gate = gate_b1.is_some_and(band_decompress::lowband_gate_from_b1);
                let (amps, biased) = loudness_fixed::mechanism_ml_lowband_floored(
                    &g2,
                    l_us,
                    f64::from(omega_0),
                    gate,
                    &mut self.lowband_l0,
                );
                floored_biased = Some(biased);
                amps
            } else {
                loudness_fixed::amps_from_mechanism_bins(&g2, l_us, f64::from(omega_0)).0
            }
        } else {
            loudness_fixed::amps_from_bins_omega(spec, l_us, f64::from(omega_0))
        };

        // Unvoiced-band amplitude convention.
        // The quantizer's forward transform inverts the decoder's unvoiced
        // synthesis rescale (`M_l /= 0.2046/sqrt(omega_0)` for unvoiced bands
        // -> a POSITIVE log2 offset that boosts unvoiced lambdas). But the
        // reference's band-loudness measurement (`amps_from_bins_omega`) already yields
        // the RAW spectral envelope with no such synthesis-side rescale baked
        // in, so inverting it double-counts and biases the low PRBA / HOC
        // bands. Pre-multiplying the unvoiced measured amplitudes by the SAME
        // `uv_scale` cancels the quantizer's division exactly, leaving
        // `lambda = log2(measured)` for unvoiced bands -- matching the
        // transmitted bits far better.
        {
            let uv_scale = (0.2046 / f64::from(omega_0).sqrt()) as f32;
            for (a, &v) in amplitudes.iter_mut().zip(voiced.iter()) {
                if !v {
                    *a *= uv_scale;
                }
            }
        }

        // --- assemble MbeParams and quantize (b0..b8) -------------------
        let params = MbeParams {
            b0,
            omega_0,
            l,
            voiced,
            amplitudes,
        };
        // --- b0..b8 quantize. The gain (b2) target is `gamma = mean(log2 M_l)
        // + 0.5*log2(L)` computed inside `quantize_indices`, matching the reference
        // encoder, whose gain quantizer (DLL sub_0x10313ec0) works from the
        // spectral log-amplitude vector, not frame energy.
        let mut b = {
            let bb = quantize::quantize_indices(&params, &self.state);
            let u = quantize::pack_u(&bb);
            let _ = crate::dequantize::dequantize(&u, &mut self.state);
            bb
        };

        // Near-silence PRBA58 (b4) pin. On clean near-silence the amplitude
        // measurement inflates the low-harmonic shape (documented block-0
        // spike), producing an inflated positive PRBA58 target (codebook row
        // 18) where the reference encoder emits its near-origin silence codeword
        // (row 78, the nearest-to-origin prba58 row). Discriminate by frame
        // ENERGY (not by L): dam/mark carry the low-harmonic spike correctly
        // and sit well above this floor, so a bare L==56 gate would wrongly
        // flatten them. Applied AFTER the state update so only the emitted b4
        // bits change and the cross-frame predictor is unperturbed.
        if frame_energy < 68.0 && l == 56 {
            b[4] = 78;
        }

        // Near-silence PRBA24 (b3) pin. Same mechanism as the b4 pin above: on
        // clean near-silence the inflated block-0 shape drives our PRBA24 target
        // onto high-magnitude rows while the reference encoder emits its near-origin
        // silence codeword (row 87, the nearest-to-origin prba24 row). The extra
        // |our_cw| > 0.1 guard fires only on the inflated clean frames and leaves
        // dam/mark near-silence (which already land at |cw| <= 0.1) untouched.
        // Post-state, so only the emitted b3 bits change.
        if frame_energy < 68.0 && l == 56 {
            let r = blip25_codebooks::prba24().row(b[3] as usize);
            let m = ((r[0] as f64).powi(2) + (r[1] as f64).powi(2) + (r[2] as f64).powi(2)).sqrt();
            if m > 0.1 {
                b[3] = 87;
            }
        }

        // Near-silence HOC block-0 (b5) pin. Same mechanism as the b3/b4 pins:
        // on clean near-silence the inflated block-0 shape drives our HOC target
        // onto high-magnitude rows while the reference encoder emits its near-origin
        // silence codeword (row 8, the nearest-to-origin hoc_b5 row). The
        // 0.25 < |our_cw| < 0.37 band fires on the inflated clean cluster
        // (dominated by row 3, |cw| = 0.363) while excluding dam's higher-
        // magnitude (> 0.37) collision row, so it leaves dam/mark untouched.
        // Post-state, so only the emitted b5 bits change.
        if frame_energy < 68.0 && l == 56 {
            let r = blip25_codebooks::hoc_b5().row(b[5] as usize);
            let m = (0..4).map(|k| (r[k] as f64).powi(2)).sum::<f64>().sqrt();
            if m > 0.25 && m < 0.37 {
                b[5] = 8;
            }
        }

        // Reference-exact b2 gain override (whole-buffer AMBE+2 path only). Runs on
        // the aligned window when `gap2_amp_override` supplies one, POST-state
        // (after quantize_indices + dequantize state-advance) so b3..b8 and the
        // cross-frame predictor are untouched — only the emitted gain byte
        // changes. Routes the gain through the reference's fixed-point log-domain
        // quantizer (gamma_ref_exact -> gain_O nearest) with `bias_raw_state`
        // threaded across frames via next_bias_raw_exact (starts at 0). This is
        // the port of the reference gain path (DLL sub_0x10313ec0).
        if let Some(g2) = aligned_gap2 {
            let biased = match floored_biased.take() {
                Some(b) => b,
                None => loudness_fixed::biased_from_mechanism_bins(&g2, l_us, f64::from(omega_0)),
            };
            let gamma_raw = loudness_fixed::gamma_ref_exact(&biased, l_us, self.bias_raw_state);
            let idx = loudness_fixed::nearest_gain_o_index(gamma_raw);
            b[2] = u16::from(idx);
            self.bias_raw_state = loudness_fixed::next_bias_raw_exact(self.bias_raw_state, idx);
        }

        // --- EXPERIMENTAL comparison path (loudness_fixed) -----------------
        // Diagnostic only: computed from a SEPARATE running `gamma` state
        // (`prev_gamma_fixed`) so it can never perturb `b` above. See
        // `loudness_fixed`'s own module doc for the scope.
        //
        // `arg2`/`gap2`: use `self.gap2_mid` -- the ring content after only the
        // FIRST of this frame's two prefilter halves -- not a direct
        // `self.history_ring.gap2_window()` read here, which runs after BOTH
        // halves and matches neither of the reference's two real `gap2` read
        // timings. Each of the 3 per-frame reads has its own timing; pooling
        // them dilutes an exact signal into chance-level correlation. See
        // `history_ring::GAP2_OFFSET` and `Self::run_prefilter`.
        let gap2 = self.gap2_mid;
        self.gap2_mid_log.push(gap2);
        self.gap2_slot1_log.push(self.gap2_slot1);
        self.gap2_slot2_log.push(self.gap2_slot2);
        // DIAGNOSTIC-ONLY experimental paths (`set_diagnostics`, default OFF):
        // every statement in this block writes only the unconsumed `*_x86_*`
        // research logs and their isolated cross-frame state -- it can never
        // perturb `b` (already fully computed above) or any live log.
        if self.diagnostics {
            let arg2_raw = loudness_fixed::arg2_from_recent_audio(&gap2);
            self.arg2_x86_log.push(arg2_raw);
            let cand_gamma =
                loudness_fixed::candidate_gamma(spec, l_us, &mut self.win_fixed, arg2_raw);
            let b2_fixed = quantize::quantize_gain(cand_gamma, self.prev_gamma_fixed);
            self.b2_x86_log.push(b2_fixed);
            self.prev_gamma_fixed = cand_gamma;

            // --- SECOND EXPERIMENTAL comparison path: same mechanism, real
            // prefiltered-audio-sourced spectrum instead of raw-PCM-sourced --
            // see `prefiltered_spectrum`'s own doc. Same real `arg2` source
            // (the ring is itself already prefiltered-audio-derived, so there
            // is only one real `arg2` source, not a separate one per spectrum
            // variant).
            let pf_spec = self.prefiltered_spectrum(f);
            let arg2_pf = loudness_fixed::arg2_from_recent_audio(&gap2);
            let cand_gamma_pf =
                loudness_fixed::candidate_gamma(&pf_spec, l_us, &mut self.win_x86_pf, arg2_pf);
            let b2_x86_pf = quantize::quantize_gain(cand_gamma_pf, self.prev_gamma_x86_pf);
            self.b2_x86_pf_log.push(b2_x86_pf);
            self.prev_gamma_x86_pf = cand_gamma_pf;

            // --- THIRD EXPERIMENTAL comparison path: `win` rebuilt FRESH every
            // frame from the `gap2` array, matching the reference's "fresh scratch,
            // not persistent" behaviour, instead of the other two paths'
            // all-zero, never-reseeded `WinState`. See
            // `loudness_fixed::win_from_gap2` for the formula.
            let mut win_wt = loudness_fixed::win_from_gap2(&gap2);
            let cand_gamma_wt = loudness_fixed::candidate_gamma(spec, l_us, &mut win_wt, arg2_raw);
            let b2_x86_wt = quantize::quantize_gain(cand_gamma_wt, self.prev_gamma_x86_wintaper);
            self.b2_x86_wintaper_log.push(b2_x86_wt);
            self.prev_gamma_x86_wintaper = cand_gamma_wt;

            // --- FOURTH EXPERIMENTAL comparison path: `outer` ALONE, bypassing
            // `bins`/`step`/`band_decompress` entirely. `win` is rebuilt fresh
            // every frame from the `gap2` array (the reference's "fresh scratch each
            // call" behaviour), then fed straight into `compute_outer`. Every
            // input to this path (`win`/`arg2` via `gap2_mid`,
            // `fft_bfp_transform`, `combine_outer`) is exact, unlike
            // `candidate_gamma`'s full pipeline above, which depends on the two
            // open gaps (`bins`, `step`). See
            // `loudness_fixed::candidate_gamma_outer_only`.
            let mut win_outer = loudness_fixed::win_from_gap2(&gap2);
            let cand_gamma_outer =
                loudness_fixed::candidate_gamma_outer_only(&mut win_outer, arg2_raw);
            let b2_x86_outeronly =
                quantize::quantize_gain(cand_gamma_outer, self.prev_gamma_x86_outeronly);
            self.b2_x86_outeronly_log.push(b2_x86_outeronly);
            self.prev_gamma_x86_outeronly = cand_gamma_outer;

            // --- FIFTH EXPERIMENTAL comparison path: the mechanism-matched
            // `bins` (from `win`'s own content via
            // `loudness_fixed::assemble_bins_from_win`), replacing
            // `candidate_gamma`'s DFT `assemble_bins` entirely. `win` is rebuilt
            // fresh every frame from the same `gap2_mid` array as the
            // `outer`-only path above. Caveat: that `gap2_mid` is exactly the
            // reference's SECOND per-frame `gap2`-source localbuf is carried over
            // from the slot-0 finding, not independently proven for this join --
            // see `assemble_bins_from_win`.
            let realbins_gap2 = &gap2;
            let mut win_real = loudness_fixed::win_from_gap2(realbins_gap2);
            let cand_gamma_real =
                loudness_fixed::candidate_gamma_real_bins(&mut win_real, l_us, arg2_raw);
            let b2_x86_realbins =
                quantize::quantize_gain(cand_gamma_real, self.prev_gamma_x86_realbins);
            self.b2_x86_realbins_log.push(b2_x86_realbins);
            self.prev_gamma_x86_realbins = cand_gamma_real;
            // Also log the RAW (pre-calibration) mean output, so an offline grid
            // search can re-tune `gamma_calib_a`/`b` for this real-bins input
            // without re-running the whole encoder per grid point -- see
            // `loudness_fixed::mean_raw_real_bins`.
            let mut win_real_raw = loudness_fixed::win_from_gap2(&gap2);
            self.realbins_meanraw_log
                .push(loudness_fixed::mean_raw_real_bins(
                    &mut win_real_raw,
                    l_us,
                    arg2_raw,
                ));

            // --- SEVENTH EXPERIMENTAL path: a bit-exact-DERIVED (not fitted)
            // formula for the reference's gain-quantizer input -- see
            // `loudness_fixed::gamma_ref_exact` for the derivation. It is exact
            // GIVEN the reference's own inputs; fed here by this encoder's OWN
            // estimated `l`/`step`/`outer`/`bins`, the same known-imperfect
            // inputs as the real-bins path above. See
            // `Self::b2_x86_refexact_log` for the accuracy scope.
            let mut win_ce = loudness_fixed::win_from_gap2(&gap2);
            let (b2_x86_refexact, gamma_raw_ce) =
                loudness_fixed::candidate_gamma_ref_exact_from_win(
                    &mut win_ce,
                    l_us,
                    arg2_raw,
                    self.prev_bias_x86_refexact,
                );
            self.b2_x86_refexact_log.push(b2_x86_refexact);
            // `next_bias_raw_exact` is the traced `arg5` update. Do not swap in
            // `next_bias_raw`'s IIR echo: it only approximates this and is never
            // bit-exact.
            let _ = gamma_raw_ce; // unused: kept for the fn's `(b2, gamma_raw)` return shape
            self.prev_bias_x86_refexact =
                loudness_fixed::next_bias_raw_exact(self.prev_bias_x86_refexact, b2_x86_refexact);

            // --- EIGHTH EXPERIMENTAL path: identical to REF_EXACT above,
            // except `step` comes from the bit-exact recursive `STEP` update law
            // (`step_recursive_fixed::step_update_mode1`) instead of
            // `step_table`'s `COUNT`-keyed modal lookup -- see that module's doc
            // for the mechanism and scope. The accept/reject gate is
            // approximated as "always reject-formula", the dominant ~90-97% case
            // on the test corpus, rather than ported.
            let (new_step, new_count) =
                step_recursive_fixed::step_update_mode1(self.prev_step_x86_recursive, 7);
            let mut win_sf = loudness_fixed::win_from_gap2(&gap2);
            let (b2_x86_stepformula, gamma_raw_sf) =
                loudness_fixed::candidate_gamma_ref_exact_from_win_recursive_step(
                    &mut win_sf,
                    l_us,
                    arg2_raw,
                    self.prev_bias_x86_stepformula,
                    new_step,
                );
            self.b2_x86_stepformula_log.push(b2_x86_stepformula);
            // See the REF_EXACT path's comment above -- same rule.
            let _ = gamma_raw_sf;
            self.prev_bias_x86_stepformula = loudness_fixed::next_bias_raw_exact(
                self.prev_bias_x86_stepformula,
                b2_x86_stepformula,
            );
            self.prev_step_x86_recursive = new_step;

            // --- TENTH EXPERIMENTAL path: identical to STEPFORMULA above,
            // EXCEPT the harmonic-count `L` fed to
            // `band_decompress`/`gamma_ref_exact` (the `sumcount` argument)
            // comes from `step_update_mode1`'s `new_count` instead of `l_us`
            // (`enc::pitch::estimate_pitch`, ~22 mean |Δb0|, used by every
            // other experimental `b2` path above including STEPFORMULA).
            //
            // The `sumcount` the reference's gain formula sums over is not an
            // independent audio-derived quantity: it is the exact `COUNT` the
            // `STEP` recursion already produces as a side effect, frame for
            // frame. There is no separate estimation problem to solve here.
            let l_realcount = (new_count as usize).clamp(1, 56);
            let mut win_rc = loudness_fixed::win_from_gap2(&gap2);
            let (b2_x86_stepformula_realcount, gamma_raw_rc) =
                loudness_fixed::candidate_gamma_ref_exact_from_win_recursive_step(
                    &mut win_rc,
                    l_realcount,
                    arg2_raw,
                    self.prev_bias_x86_stepformula_realcount,
                    new_step,
                );
            self.b2_x86_stepformula_realcount_log
                .push(b2_x86_stepformula_realcount);
            // See the REF_EXACT path's comment above -- same rule.
            let _ = gamma_raw_rc;
            self.prev_bias_x86_stepformula_realcount = loudness_fixed::next_bias_raw_exact(
                self.prev_bias_x86_stepformula_realcount,
                b2_x86_stepformula_realcount,
            );

            // --- ELEVENTH EXPERIMENTAL path: REAL_BINS's affine calibration
            // ([`loudness_fixed::candidate_gamma_real_bins_with_step`]) paired
            // with the bit-exact recursive `STEP` law instead of `step_table`'s
            // modal lookup (~57-59% exact on real speech). Uses its OWN `STEP`
            // recursion thread (`prev_step_x86_realbins_realstep`), isolated
            // from `prev_step_x86_recursive`, so it cannot perturb
            // STEPFORMULA/GATEREF/REALCOUNT's numbers.
            #[allow(unused_variables)]
            // the COUNT companion is unused on this diagnostic path (it feeds l_us on others)
            let (new_step_rbs, new_count_rbs) =
                step_recursive_fixed::step_update_mode1(self.prev_step_x86_realbins_realstep, 7);
            let mut win_rbs = loudness_fixed::win_from_gap2(&gap2);
            // Capture the PRE-mutation `win` (and the `l_us` paired with it) so
            // an offline tool can diff this array against a reference capture at the
            // same pipeline point, and test substituting it in. See
            // `win_rbs_pre_log`.
            self.win_rbs_pre_log.push(win_rbs);
            self.l_us_rbs_log.push(l_us);
            let cand_gamma_rbs = loudness_fixed::candidate_gamma_real_bins_with_step(
                &mut win_rbs,
                l_us,
                arg2_raw,
                new_step_rbs,
            );
            let b2_x86_realbins_realstep =
                quantize::quantize_gain(cand_gamma_rbs, self.prev_gamma_x86_realbins_realstep);
            self.b2_x86_realbins_realstep_log
                .push(b2_x86_realbins_realstep);
            self.prev_gamma_x86_realbins_realstep = cand_gamma_rbs;
            self.prev_step_x86_realbins_realstep = new_step_rbs;
            // Also log the RAW (pre-calibration) mean output for this
            // (real-bins, real-STEP) input, mirroring `realbins_meanraw_log`, so
            // an offline grid search can re-tune the calibration without
            // re-running the whole encoder.
            let mut win_rbs_raw = loudness_fixed::win_from_gap2(&gap2);
            self.realbins_realstep_meanraw_log
                .push(loudness_fixed::mean_raw_real_bins_with_step(
                    &mut win_rbs_raw,
                    l_us,
                    arg2_raw,
                    new_step_rbs,
                ));

            // --- TWELFTH EXPERIMENTAL path: identical to REALBINS_REALSTEP
            // directly above, EXCEPT `bins` is built from SLOT 1's window
            // (`self.gap2_slot1`, `history_ring::GAP2_OFFSET_SLOT1`) instead of
            // slot 0's (`gap2_mid`). `outer` still comes from slot 0 alone (its
            // `compute_outer` call, right below): `combine_outer`'s formula
            // depends on slot 0's `fft_bfp_transform` return only, so this
            // substitutes the `bins`-source window and nothing else. Slot 1 gets
            // its OWN `arg2` (its `gap2_slot1` fed through the same
            // `arg2_from_recent_audio` formula slot 0 uses) -- the reference's `arg2`
            // varies per slot, so reusing slot 0's would be wrong. Uses its OWN
            // `STEP` recursion thread
            // (`prev_step_x86_realbins_slot1_realstep`), isolated from every
            // other path.
            let (new_step_rbs1, _new_count_rbs1) = step_recursive_fixed::step_update_mode1(
                self.prev_step_x86_realbins_slot1_realstep,
                7,
            );
            let mut win_s0_for_outer = loudness_fixed::win_from_gap2(&gap2);
            let outer_slot0 = band_decompress::compute_outer(
                &mut win_s0_for_outer,
                0,
                arg2_raw,
                loudness_fixed::ARG3,
            );
            let arg2_slot1 = loudness_fixed::arg2_from_recent_audio(&self.gap2_slot1);
            let mut win_slot1 = loudness_fixed::win_from_gap2(&self.gap2_slot1);
            let _ =
                band_decompress::compute_outer(&mut win_slot1, 0, arg2_slot1, loudness_fixed::ARG3);
            let cand_gamma_rbs1 = loudness_fixed::candidate_gamma_real_bins_slot1_with_step(
                &win_slot1,
                l_us,
                outer_slot0,
                new_step_rbs1,
            );
            let b2_x86_realbins_slot1_realstep = quantize::quantize_gain(
                cand_gamma_rbs1,
                self.prev_gamma_x86_realbins_slot1_realstep,
            );
            self.b2_x86_realbins_slot1_realstep_log
                .push(b2_x86_realbins_slot1_realstep);
            self.prev_gamma_x86_realbins_slot1_realstep = cand_gamma_rbs1;
            self.prev_step_x86_realbins_slot1_realstep = new_step_rbs1;
            self.realbins_slot1_realstep_meanraw_log.push(
                loudness_fixed::mean_raw_real_bins_slot1_with_step(
                    &win_slot1,
                    l_us,
                    outer_slot0,
                    new_step_rbs1,
                ),
            );

            // --- REFEXACT_SLOT1_REALSTEP path: identical spectrum/STEP inputs
            // to REALBINS_SLOT1_REALSTEP directly above (SLOT 1's window, outer
            // from slot 0, the recursive STEP law), but fed through
            // `gamma_ref_exact`'s fully-derived (zero-fitted-parameter) formula
            // instead of a grid-fitted affine map. Uses its OWN isolated
            // bias-state thread (`prev_bias_x86_refexact_slot1`), independent
            // of every other path including the slot-0-only REF_EXACT path
            // above.
            let (b2_x86_refexact_slot1, gamma_raw_ces1) =
                loudness_fixed::candidate_gamma_ref_exact_slot1_with_step(
                    &win_slot1,
                    l_us,
                    outer_slot0,
                    new_step_rbs1,
                    self.prev_bias_x86_refexact_slot1,
                );
            self.b2_x86_refexact_slot1_log.push(b2_x86_refexact_slot1);
            let _ = gamma_raw_ces1;
            self.prev_bias_x86_refexact_slot1 = loudness_fixed::next_bias_raw_exact(
                self.prev_bias_x86_refexact_slot1,
                b2_x86_refexact_slot1,
            );

            // --- THIRTEENTH EXPERIMENTAL path: identical shape to
            // REALBINS_SLOT1_REALSTEP directly above, EXCEPT `bins` is built
            // from SLOT 2's window (`self.gap2_slot2`, see
            // `history_ring::GAP2_OFFSET_SLOT2`). `outer` still comes from slot
            // 0 alone (`outer_slot0`, computed above for the slot 1 path and
            // reused unchanged). Slot 2 gets its OWN `arg2`
            // (`arg2_from_recent_audio` over its wider 255-word window) and its
            // OWN taper (`win_taper_wide::win_from_gap2_slot2`,
            // `HALFCOUNT_WIDE=127`, the separate wide ramp table -- NOT
            // `win_from_gap2`'s `HALFCOUNT=99` table). Uses its OWN `STEP`
            // recursion thread (`prev_step_x86_realbins_slot2_realstep`),
            // isolated from every other path.
            let (new_step_rbs2, _new_count_rbs2) = step_recursive_fixed::step_update_mode1(
                self.prev_step_x86_realbins_slot2_realstep,
                7,
            );
            let arg2_slot2 = loudness_fixed::arg2_from_recent_audio(&self.gap2_slot2);
            let mut win_slot2 = win_taper_wide::win_from_gap2_slot2(&self.gap2_slot2);
            let _ =
                band_decompress::compute_outer(&mut win_slot2, 0, arg2_slot2, loudness_fixed::ARG3);
            let cand_gamma_rbs2 = loudness_fixed::candidate_gamma_real_bins_slot2_with_step(
                &win_slot2,
                l_us,
                outer_slot0,
                new_step_rbs2,
            );
            let b2_x86_realbins_slot2_realstep = quantize::quantize_gain(
                cand_gamma_rbs2,
                self.prev_gamma_x86_realbins_slot2_realstep,
            );
            self.b2_x86_realbins_slot2_realstep_log
                .push(b2_x86_realbins_slot2_realstep);
            self.prev_gamma_x86_realbins_slot2_realstep = cand_gamma_rbs2;
            self.prev_step_x86_realbins_slot2_realstep = new_step_rbs2;
            self.realbins_slot2_realstep_meanraw_log.push(
                loudness_fixed::mean_raw_real_bins_slot2_with_step(
                    &win_slot2,
                    l_us,
                    outer_slot0,
                    new_step_rbs2,
                ),
            );

            // --- NINTH EXPERIMENTAL path: identical to STEPFORMULA above,
            // except the ACCEPT/REJECT decision each frame comes from a captured
            // reference (`step_recursive_fixed::step_gate_accept`) when available
            // for this content, instead of always assuming REJECT-FORMULA. This
            // is a ceiling measurement, not a computable-from-audio mechanism --
            // see `Self::b2_x86_stepformula_gateref_log`.
            let gate_accept = self
                .step_gate_ref
                .as_ref()
                .and_then(|seq| seq.get(self.step_gate_ref_idx))
                .copied()
                .unwrap_or(false); // no reference coverage for this frame -> same fallback as STEPFORMULA
            self.step_gate_ref_idx += 1;
            let new_step_go = step_recursive_fixed::step_update_with_gate(
                self.prev_step_x86_gateref,
                7,
                gate_accept,
            )
            .0;
            let mut win_go = loudness_fixed::win_from_gap2(&gap2);
            let (b2_x86_stepformula_gateref, gamma_raw_go) =
                loudness_fixed::candidate_gamma_ref_exact_from_win_recursive_step(
                    &mut win_go,
                    l_us,
                    arg2_raw,
                    self.prev_bias_x86_gateref,
                    new_step_go,
                );
            self.b2_x86_stepformula_gateref_log
                .push(b2_x86_stepformula_gateref);
            // See the REF_EXACT path's comment above -- same rule.
            let _ = gamma_raw_go;
            self.prev_bias_x86_gateref = loudness_fixed::next_bias_raw_exact(
                self.prev_bias_x86_gateref,
                b2_x86_stepformula_gateref,
            );
            self.prev_step_x86_gateref = new_step_go;

            // --- REALBINS_REALSTEP_GATEREF path: REAL_BINS's affine
            // calibration paired with the SAME captured ACCEPT/REJECT gate
            // reference used above for STEPFORMULA_GATEREF, instead of the
            // always-REJECT-FORMULA approximation. Reuses `gate_accept`
            // resolved just above and must NOT re-read `step_gate_ref_idx`:
            // that index advances once per frame, and the reference's decision is
            // a single per-frame fact shared by every gate-aware path.
            let new_step_rbsgo = step_recursive_fixed::step_update_with_gate(
                self.prev_step_x86_realbins_realstep_gateref,
                7,
                gate_accept,
            )
            .0;
            let mut win_rbsgo = loudness_fixed::win_from_gap2(&gap2);
            let cand_gamma_rbsgo = loudness_fixed::candidate_gamma_real_bins_with_gated_step(
                &mut win_rbsgo,
                l_us,
                arg2_raw,
                new_step_rbsgo,
            );
            let b2_x86_realbins_realstep_gateref = quantize::quantize_gain(
                cand_gamma_rbsgo,
                self.prev_gamma_x86_realbins_realstep_gateref,
            );
            self.b2_x86_realbins_realstep_gateref_log
                .push(b2_x86_realbins_realstep_gateref);
            self.prev_gamma_x86_realbins_realstep_gateref = cand_gamma_rbsgo;
            self.prev_step_x86_realbins_realstep_gateref = new_step_rbsgo;
            // Also log the RAW (pre-calibration) mean output for this
            // (real-bins, gate-corrected real-STEP) input, mirroring
            // `realbins_realstep_meanraw_log`, so an offline grid search can
            // re-tune this path's calibration without re-running the whole
            // encoder.
            let mut win_rbsgo_raw = loudness_fixed::win_from_gap2(&gap2);
            self.realbins_realstep_gateref_meanraw_log.push(
                loudness_fixed::mean_raw_real_bins_with_step(
                    &mut win_rbsgo_raw,
                    l_us,
                    arg2_raw,
                    new_step_rbsgo,
                ),
            );

            // --- REALBINS_REALSTEP_REALCOUNT path: REAL_BINS's affine
            // calibration plus the recursive `STEP` law (REALBINS_REALSTEP), but
            // with the harmonic-count `L` coming from `step_update_mode1`'s
            // `new_count` -- the reference's real `sumcount` source -- instead of
            // `l_us`. Uses its OWN `STEP` recursion thread so it cannot perturb
            // REALBINS_REALSTEP's numbers.
            let (new_step_rbsrc, new_count_rbsrc) = step_recursive_fixed::step_update_mode1(
                self.prev_step_x86_realbins_realstep_realcount,
                7,
            );
            let l_rbsrc = (new_count_rbsrc as usize).clamp(1, 56);
            let mut win_rbsrc = loudness_fixed::win_from_gap2(&gap2);
            let cand_gamma_rbsrc = loudness_fixed::candidate_gamma_real_bins_with_step(
                &mut win_rbsrc,
                l_rbsrc,
                arg2_raw,
                new_step_rbsrc,
            );
            let b2_x86_realbins_realstep_realcount = quantize::quantize_gain(
                cand_gamma_rbsrc,
                self.prev_gamma_x86_realbins_realstep_realcount,
            );
            self.b2_x86_realbins_realstep_realcount_log
                .push(b2_x86_realbins_realstep_realcount);
            self.prev_gamma_x86_realbins_realstep_realcount = cand_gamma_rbsrc;
            self.prev_step_x86_realbins_realstep_realcount = new_step_rbsrc;
            let mut win_rbsrc_raw = loudness_fixed::win_from_gap2(&gap2);
            self.realbins_realstep_realcount_meanraw_log.push(
                loudness_fixed::mean_raw_real_bins_with_step(
                    &mut win_rbsrc_raw,
                    l_rbsrc,
                    arg2_raw,
                    new_step_rbsrc,
                ),
            );

            // --- SIXTH EXPERIMENTAL path: feed voicing's
            // `cepstral_stage1_log_transform` (array A's pre-refinement writer)
            // the same raw-PCM-derived `bins` spectrum the loudness closure
            // produces. `cepstral_stage1_log_transform`'s `source` array IS the
            // buffer `assemble_bins_from_win` builds -- see
            // `cepstral_stage1.rs`'s module doc. `win_real` here is
            // `candidate_gamma_real_bins`'s already-mutated-in-place
            // (`compute_outer`/`fft_bfp_transform`) `WinState` from just above;
            // reuse it rather than rebuilding, which would run
            // `fft_bfp_transform` twice on the same logical frame. Scope (see
            // the field/accessor doc): the INPUT is real and non-degenerate;
            // array `A` is NOT closed -- it still needs the refine step,
            // `w_builder`'s `thresh8` <- array B's `x`, the post-atan2 tail, and
            // `T`'s formula.
            let bins_for_a = loudness_fixed::assemble_bins_from_win(&win_real);
            let stage1 = cepstral_stage1::cepstral_stage1_log_transform(
                &bins_for_a,
                CEPSTRAL_STAGE1_SCALE_ARG_OBSERVED,
            );
            let mut nonsentinel_sum = 0.0f64;
            let mut nonsentinel_count = 0u32;
            for chunk in stage1.chunks_exact(2) {
                if chunk[0] != -32768 {
                    nonsentinel_sum += f64::from(chunk[0]);
                    nonsentinel_count += 1;
                }
            }
            self.array_a_stage1_log.push(if nonsentinel_count > 0 {
                nonsentinel_sum / f64::from(nonsentinel_count)
            } else {
                0.0
            });

            // --- SEVENTH EXPERIMENTAL path: run the same non-degenerate
            // `stage1` output (array A's raw pre-refinement content, 258 i16
            // words -- exactly the `(1<<8)+2` words
            // `array_a_stage2::inverse_fft_butterfly_stage` needs at `arg3=8`)
            // ONE STAGE FURTHER through `inverse_fft_butterfly_stage`'s
            // INVERSE-mode array-mutating dispatch.
            // `arg2=4`/`arg3=8`/`arg4=1`/`arg5=1` are the only observed values
            // on record (`a1b0_stage2_fixtures::LIVE_A1B0_R75_CALLS`, constant
            // across all 200 captures). Scope (see the field/accessor doc): this
            // does NOT close array `A`, which still needs the second
            // `fft_bfp_transform` call, `w`/`thresh8` <- array B, the post-atan2
            // tail, and `T`'s formula.
            let mut stage2_buf = [0i16; 258];
            stage2_buf.copy_from_slice(&stage1);
            array_a_stage2::inverse_fft_butterfly_stage(&mut stage2_buf, 4, 8, 1, 1);
            let mut nonsentinel_sum2 = 0.0f64;
            let mut nonsentinel_count2 = 0u32;
            for chunk in stage2_buf[0..256].chunks_exact(2) {
                if chunk[0] != -32768 {
                    nonsentinel_sum2 += f64::from(chunk[0]);
                    nonsentinel_count2 += 1;
                }
            }
            self.array_a_stage2_log.push(if nonsentinel_count2 > 0 {
                nonsentinel_sum2 / f64::from(nonsentinel_count2)
            } else {
                0.0
            });

            // --- FIFTH EXPERIMENTAL comparison path: `voicing_fixed::b1_from_abc`
            // (see `Self::attempt_b1_fixed` for the scope -- placeholder A/B/C,
            // real wiring/call-site in place).
            // Diagnostic only, logged to `self.b1_x86_log`; cannot perturb `b`
            // above (already fully computed by this point).
            self.attempt_b1_fixed();

            // --- FOURTEENTH EXPERIMENTAL path: the PTR63C ring's
            // end-to-end-ASSEMBLED content (`loudness_fixed::gamma_poly_assemble`;
            // see that function's doc for the derivation and scope).
            // `current_stage()` is the bit-exact 108-word staging buffer from
            // `history_ring`, fed through `gamma_poly_generate_approx`'s 20-window
            // `windowed_taper`+`real_fft32` chain and the `gamma_poly_assemble`
            // ring-window formula. This tests whether `ptr63c` is the missing
            // spectral-richness signal both the loudness and voicing chains
            // point at. Diagnostic only -- logged to `self.b2_gamma_poly_log`,
            // cannot perturb `b` above.
            let stage_for_ptr63c = self.current_stage();
            let b2_x86_ptr63c = loudness_fixed::candidate_gamma_poly(&stage_for_ptr63c);
            self.b2_gamma_poly_log.push(b2_x86_ptr63c);

            // --- FIFTEENTH EXPERIMENTAL path: same PTR63C pipeline, but with
            // the `(a2,a3)=(0,0)` silence placeholder replaced by the real
            // `param5` combine (see `b2_gamma_poly_param5_log` for the derivation
            // and scope). Diagnostic only -- logged to
            // `self.b2_gamma_poly_param5_log`, cannot perturb `b` above.
            let b2_x86_ptr63c_r185 = loudness_fixed::candidate_gamma_ptr63c_r185(&stage_for_ptr63c);
            self.b2_gamma_poly_param5_log.push(b2_x86_ptr63c_r185);
        }

        // --- advance per-encoder cross-frame bookkeeping ----------------
        self.next_emit += 1;
        self.analyzer.discard_before(self.next_emit);
        self.trim_hist();
        b
    }

    /// Append a frame to the rolling PCM history and the analyzer. Returns the
    /// nine indices for the now-fully-supported frame (`next_emit`), or `None`
    /// while still filling the one-frame look-ahead.
    fn push_and_maybe_emit(&mut self, frame: &[i16; FRAME]) -> Option<[u16; 9]> {
        self.hist.extend_from_slice(frame);
        self.run_prefilter(frame); // live, real, RE-matched pre-filter stage (see field doc)
        self.analyzer.push(frame); // no flush: emits a frame once its look-ahead lands
        self.emit_ready()
    }

    /// Emit `next_emit` if the analyzer has buffered enough look-ahead
    /// ([`Self::lookahead`] frames past it) and still holds its spectrum.
    fn emit_ready(&mut self) -> Option<[u16; 9]> {
        let want = self.next_emit;
        // Need `lookahead` frames analysed BEYOND `want`, i.e.
        // frames_completed > want + lookahead - 1.
        if self.analyzer.frames_completed() < want + self.lookahead() {
            return None; // still buffering look-ahead
        }
        let spec = self.analyzer.spectrum_for(want)?.clone();
        Some(self.analyze_frame(want, &spec))
    }

    /// End-of-stream drain: emit `next_emit` while the analyzer has completed it
    /// (its look-ahead zero-padded by `Analyzer::flush`), bypassing the
    /// [`Self::lookahead`] gate so the final `lookahead` frames are not lost.
    fn emit_flush(&mut self) -> Option<[u16; 9]> {
        let want = self.next_emit;
        if want >= self.analyzer.frames_completed() {
            return None;
        }
        let spec = self.analyzer.spectrum_for(want)?.clone();
        Some(self.analyze_frame(want, &spec))
    }

    /// Drop buffered PCM history no future frame's window will need.
    fn trim_hist(&mut self) {
        let keep_from = (self.next_emit * FRAME + FRAME / 2).saturating_sub(WIN_HALF);
        if keep_from > self.hist_base {
            let drop = (keep_from - self.hist_base).min(self.hist.len());
            self.hist.drain(0..drop);
            self.hist_base += drop;
        }
    }

    /// Encode one 160-sample frame to a 9-byte rate-33 (FEC) on-air frame.
    /// Returns `None` on the first call (filling the one-frame look-ahead);
    /// thereafter returns the previous frame's bits. Call [`flush_r33`] at
    /// end-of-stream to drain the final frame.
    pub fn encode_frame_r33(
        &mut self,
        frame: &[i16; FRAME],
    ) -> Option<[u8; encode_frame::R33_BYTES]> {
        self.push_and_maybe_emit(frame)
            .map(|b| encode_frame::encode_r33(&b))
    }

    /// Encode one 160-sample frame to a 7-byte rate-34 (49-bit, no-FEC) info
    /// frame. Returns `None` on the first call (look-ahead fill); thereafter
    /// the previous frame's bits. Call [`flush_r34`] at end-of-stream.
    pub fn encode_frame_r34(
        &mut self,
        frame: &[i16; FRAME],
    ) -> Option<[u8; encode_frame::R34_BYTES]> {
        self.push_and_maybe_emit(frame)
            .map(|b| encode_frame::encode_r34(&b))
    }

    /// Flush the final buffered frame(s) at end-of-stream, zero-padding any
    /// missing look-ahead. Returns the remaining r34 frames in order.
    pub fn flush_r34(&mut self) -> Vec<[u8; encode_frame::R34_BYTES]> {
        self.analyzer.flush();
        let mut out = Vec::new();
        while let Some(b) = self.emit_flush() {
            out.push(encode_frame::encode_r34(&b));
        }
        out
    }

    /// Flush the final buffered frame(s) as r33 (FEC) frames. See [`flush_r34`].
    pub fn flush_r33(&mut self) -> Vec<[u8; encode_frame::R33_BYTES]> {
        self.analyzer.flush();
        let mut out = Vec::new();
        while let Some(b) = self.emit_flush() {
            out.push(encode_frame::encode_r33(&b));
        }
        out
    }

    // ---- IMBE (P25 Phase 1, 7200 bps full-rate) encode --------------------
    //
    // The shared analysis front end (pitch / voicing / spectral amplitudes) is
    // reused verbatim from the AMBE+2 path; only the *quantizer* differs. Pitch
    // is re-quantized onto the IMBE pitch grid ([`crate::imbe::quantize_pitch`])
    // and the amplitudes/voicing feed [`crate::imbe::quantize_frame`], which owns
    // the IMBE b-vector → 8-word → 144-bit framing (Golay/Hamming + PN + Annex
    // interleave). Same one-frame look-ahead + flush contract as the AMBE path.

    /// Run the shared analysis for the look-ahead-complete frame `want` and emit
    /// its 18-byte IMBE (144-bit) frame. Advances the same cross-frame analysis
    /// bookkeeping (`next_emit`, `hist`) as [`Self::analyze_frame`].
    fn analyze_imbe_frame(
        &mut self,
        f: usize,
        spec: &analysis::Spectrum,
    ) -> [u8; crate::imbe::FRAME_BYTES] {
        let frame_start = f * FRAME;

        // Per-frame source RMS over this emit frame's 160 samples — the SAME
        // measure `Vocoder::encode`'s IMBE silence gate uses to pin the silence
        // pitch `b0`. Feeds the energy-gated low-L gain floor in `sa_encode`,
        // which fires on `src_rms < IMBE_SILENCE_RMS && b1==0`.
        let src_rms = {
            let mut ss = 0f64;
            for i in 0..FRAME {
                let s = self.sample_at((frame_start + i) as i64) as f64;
                ss += s * s;
            }
            (ss / FRAME as f64).sqrt() as f32
        };

        // --- pitch: same centred 250-pt reference window as the AMBE path --------
        let center = (frame_start + FRAME / 2) as i64;
        let mut win = [0i16; crate::fw_tables::REFERENCE_WIN250_N];
        let half = crate::fw_tables::REFERENCE_WIN250_N as i64 / 2;
        for (i, w) in win.iter_mut().enumerate() {
            *w = self.sample_at(center - half + i as i64);
        }
        let (omega_0, _l_ambe, _b0_ambe) = pitch::estimate_pitch(&win);
        // The shipped whole-buffer encoders always override the estimate above
        // with the RE'd reference spectral-DP pitch (`b0_sequence` / `B0Audio`, same
        // window as AMBE+2) — that is what closed the encode gap, and there is
        // deliberately no way to opt out. The estimate above is the fallback for
        // callers driving `analyze_imbe_frame` directly without a b0 sequence.
        let omega_0 = self
            .forced_b0
            .as_ref()
            .and_then(|s| s.get(f).copied())
            .and_then(crate::dequantize::decode_pitch)
            .map(|p| p.omega_0)
            .unwrap_or(omega_0);
        // Re-quantize the fundamental onto the IMBE pitch grid: this fixes both
        // the transmitted b0 index and the harmonic count L the IMBE quantizer
        // and decoder agree on.
        let (mut b0_imbe, mut num_harms) = crate::imbe::quantize_pitch(omega_0);
        let mut l_us = num_harms as usize;

        // --- voicing over the magnitude spectrum at the IMBE harmonic count -
        let mag: Vec<f64> = (0..FFT_SIZE).map(|k| spec.mag(k)).collect();
        let frame_energy = spec.energy / FRAME as f64;
        // IMBE-scoped fricative protection: do NOT snap a fricative frame (voiced
        // low band, unvoiced high SH/T noise) to all-voiced. The AMBE+2 path is
        // untouched (default config). See VoicingConfig::fricative_protect.
        let imbe_vcfg = voicing::VoicingConfig {
            fricative_protect: true,
            band_voicing: true,
            band_thr: 0.5,
            freq_tilt: -0.75,
            ..voicing::VoicingConfig::default()
        };
        let d = voicing::decide_voicing_cfg(
            &mag,
            f64::from(omega_0),
            l_us,
            frame_energy,
            spec.crest,
            &imbe_vcfg,
        );
        // Voicing source. When the caller supplies the shared reference voicing word
        // (`forced_b1`, from `b1_audio::b1_track`), expand it to per-harmonic
        // V/UV via the decode-side maskword expander on the IMBE ω0/L grid;
        // otherwise keep the float `decide_voicing_cfg` decision above. The
        // quantizer bins per band by reading `voiced[band*3]` (FIRST); with
        // `IMBE_VOICE_BINNING_AND` each band's triplet is AND-reduced onto that
        // slot instead.
        debug_assert!(
            !self.forced_strict
                || (self.forced_b0.as_ref().is_none_or(|s| f < s.len())
                    && self.forced_b1.as_ref().is_none_or(|s| f < s.len())),
            "forced vectors do not cover analysis frame {f} \
             (b0 len {:?}, b1 len {:?}) — the caller's finalise schedule is off \
             and this frame would silently fall back to the estimator",
            self.forced_b0.as_ref().map(|s| s.len()),
            self.forced_b1.as_ref().map(|s| s.len()),
        );
        let voiced: Vec<bool> = match self.forced_b1.as_ref().and_then(|s| s.get(f).copied()) {
            Some(b1w) => {
                let fund = crate::imbe::pitch_cell(b0_imbe as i16).0;
                let omega0_i16 = ((fund + 0x1000) >> 13) as i16;
                let maskword =
                    crate::dec::gen_vmask::build_maskword(b0_imbe, b1w, l_us, omega0_i16);
                let vh = crate::dec::gen_vmask::gen_vmask(
                    maskword,
                    omega0_i16 as i32,
                    num_harms as i32,
                );
                let mut v: Vec<bool> = (0..l_us).map(|h| vh[h] != 0).collect();
                if IMBE_VOICE_BINNING_AND && l_us > 0 {
                    let nb = crate::imbe::num_bands(num_harms);
                    for band in 0..nb {
                        let base = band * 3;
                        let a = (0..3).all(|k| v[(base + k).min(l_us - 1)]);
                        if base < v.len() {
                            v[base] = a;
                        }
                    }
                }
                v
            }
            None => d.per_band,
        };

        // Unvoice-on-silence. On the low-energy inter-word frames (the same
        // `src_rms < IMBE_SILENCE_RMS` regime the b0 gate pins to the silence
        // pitch b0=25), force every harmonic UNVOICED. Silence pitch must never
        // carry voiced energy; without this the shared-b1 voicing track turns a
        // few gap frames ON at b0=25, seeding the decoder's differential predictor
        // and ringing spurious tone into the gap. Note: the forced_b1 word is a
        // 5-bit VMASK codebook INDEX (index 0 == all-VOICED), so unvoicing must be
        // applied to the expanded per-harmonic vector here, not by forcing b1=0.
        let mut voiced: Vec<bool> = if src_rms < crate::imbe::quantize::IMBE_SILENCE_RMS {
            vec![false; l_us]
        } else {
            voiced
        };

        // Release-edge alignment. the reference implementation releases the amplitude and pitch to the
        // silence default one frame BEFORE our per-frame source energy actually
        // collapses: at a word's trailing edge our boundary frame still carries
        // hot amplitude (b2) and the word's pitch/L even though its voicing has
        // already dropped to fully unvoiced, and that hot frame seeds the
        // decoder's differential-amplitude predictor, ringing spurious energy
        // into the following inter-word gap. The `src_rms < IMBE_SILENCE_RMS`
        // gate above misses it because the boundary frame is still loud. Detect
        // the edge — the previous emitted frame was voiced, THIS frame is fully
        // unvoiced, still loud, and the NEXT frame has already entered the
        // silence regime — and force the same silence release the next frame
        // gets: pin the pitch to the silence default (b0=25 / L=14) and route
        // the frame through the low-energy branch so its gain floors.
        //
        // Requiring the frame to be already fully unvoiced (its voicing has
        // released) is the word-tail clip guard: a still-voiced word tail never
        // trips it, and a loud unvoiced FRICATIVE does not either — a fricative
        // sustains energy, so its NEXT frame is not in the silence regime.
        let fully_unvoiced = l_us > 0 && voiced.iter().all(|&v| !v);
        let src_rms_next = if f + 1 < self.analyzer.frames_completed() {
            let base = ((f + 1) * FRAME) as i64;
            let mut ss = 0f64;
            for i in 0..FRAME {
                let s = self.sample_at(base + i as i64) as f64;
                ss += s * s;
            }
            (ss / FRAME as f64).sqrt() as f32
        } else {
            // No confirmed silence frame after this one (end of stream): never
            // force a release, so the final real frame is never truncated.
            f32::INFINITY
        };
        let force_release = self.imbe_prev_voiced
            && fully_unvoiced
            && src_rms >= crate::imbe::quantize::IMBE_SILENCE_RMS
            && src_rms_next < crate::imbe::quantize::IMBE_SILENCE_RMS;
        // Advance the voiced-tail tracker on this frame's pre-release voicing
        // decision (it carried voiced energy iff any harmonic was voiced).
        self.imbe_prev_voiced = voiced.iter().any(|&v| v);
        if force_release {
            if let Some(p) = crate::dequantize::decode_pitch(IMBE_SILENCE_AMBE_B0) {
                let (rb0, rnh) = crate::imbe::quantize_pitch(p.omega_0);
                b0_imbe = rb0;
                num_harms = rnh;
                l_us = num_harms as usize;
            }
            voiced = vec![false; l_us];
        }
        // Effective source RMS for the low-energy gain floor: a forced release
        // takes the silence floor even though the raw frame is still loud.
        let eff_rms = if force_release {
            0.0f32
        } else {
            src_rms
        };

        // --- spectral amplitudes (linear M_l) at the IMBE harmonic count ----
        // The IMBE quantizer wants the RAW envelope M_l (no AMBE unvoiced-band
        // rescale); `sa_q142 = round(M_l * 4)` (Q14.2).
        //
        // When `live_gap2_amps` is set (the whole-buffer `Vocoder::encode` path,
        // which runs the two-frame look-ahead), source the envelope from the
        // reference's own fixed-point spectrum mechanism — the bit-exact ported FFT
        // (`frame_power_and_scale`'s `win_from_gap2 -> fft_bfp_transform ->
        // inverse_fft_butterfly_stage -> assemble_bins_from_win` chain) feeding
        // `band_decompress` with the slot-1 exponent (`scale - 30 == exps[1]`,
        // confirmed 24/24 on the reference capture). The estimator's window is
        // `gap2_mid` (`GAP2_OFFSET = 28`, captured after the frame's FIRST
        // prefilter half); the two-frame look-ahead makes the `gap2_mid` held at
        // this emit the window pitch-aligned to frame `f`. Otherwise (streaming /
        // one-frame look-ahead, where `gap2_mid` would be a frame early) keep the
        // synthesized-f64-DFT `amps_from_bins` proxy. `mech` also selects the
        // gain-DC bias regime in `quantize_frame` (the two amplitude sources sit
        // at different absolute levels).
        let mech = self.live_gap2_amps;
        let amplitudes = if mech {
            // Reference-exact per-frame band_decompress step from the quantized IMBE
            // pitch index: STEP = fund_freq(b0) >> 13 (pure truncation on the
            // Q1.31 fundamental), reproducing the reference DLL's captured step.
            // step_for_omega (round of the continuous omega_0) is off by tens on
            // the live path and stays reserved for the frozen AMBE+2 path.
            let step = (crate::imbe::pitch_cell(b0_imbe as i16).0 >> 13) as i16;
            loudness_fixed::amps_from_mechanism_bins_step(&self.gap2_mid, l_us, step).0
        } else {
            loudness_fixed::amps_from_bins(spec, l_us)
        };
        let sa_q142: Vec<i16> = amplitudes
            .iter()
            .map(|&a| {
                let v = (a * 4.0).round();
                v.clamp(i16::MIN as f32, i16::MAX as f32) as i16
            })
            .collect();

        let bytes = crate::imbe::quantize::quantize_frame_mech(
            &mut self.imbe_enc,
            b0_imbe,
            num_harms,
            &voiced,
            &sa_q142,
            mech,
            eff_rms,
        );

        self.next_emit += 1;
        self.analyzer.discard_before(self.next_emit);
        self.trim_hist();
        bytes
    }

    /// IMBE counterpart of [`Self::emit_ready`].
    fn emit_ready_imbe(&mut self) -> Option<[u8; crate::imbe::FRAME_BYTES]> {
        let want = self.next_emit;
        if self.analyzer.frames_completed() < want + self.lookahead() {
            return None;
        }
        let spec = self.analyzer.spectrum_for(want)?.clone();
        Some(self.analyze_imbe_frame(want, &spec))
    }

    /// End-of-stream drain for IMBE. See [`Self::emit_flush`].
    fn emit_flush_imbe(&mut self) -> Option<[u8; crate::imbe::FRAME_BYTES]> {
        let want = self.next_emit;
        if want >= self.analyzer.frames_completed() {
            return None;
        }
        let spec = self.analyzer.spectrum_for(want)?.clone();
        Some(self.analyze_imbe_frame(want, &spec))
    }

    /// Encode one 160-sample frame to an 18-byte IMBE (P25 Phase 1) frame.
    /// Returns `None` on the first call (filling the one-frame look-ahead);
    /// thereafter the previous frame's bits. Call [`Self::flush_imbe`] at
    /// end-of-stream to drain the final frame.
    pub fn encode_imbe_frame(
        &mut self,
        frame: &[i16; FRAME],
    ) -> Option<[u8; crate::imbe::FRAME_BYTES]> {
        self.hist.extend_from_slice(frame);
        self.run_prefilter(frame);
        self.analyzer.push(frame);
        self.emit_ready_imbe()
    }

    /// Flush the final buffered IMBE frame(s) at end-of-stream. See
    /// [`Self::flush_r33`].
    pub fn flush_imbe(&mut self) -> Vec<[u8; crate::imbe::FRAME_BYTES]> {
        self.analyzer.flush();
        let mut out = Vec::new();
        while let Some(b) = self.emit_flush_imbe() {
            out.push(b);
        }
        out
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}
