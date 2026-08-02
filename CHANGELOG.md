# Changelog

All notable changes to **blip25-mbe** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-25

> **Read this before upgrading from 0.2.x.** This release replaces the codec
> engine itself, not just the code around it. Under Cargo's 0.x rules a `^0.2`
> requirement will not resolve to 0.3, so nothing upgrades silently — but the
> upgrade is not a drop-in.

### Changed — **BREAKING**

- **The codec engine was replaced.** Releases 0.1.0–0.2.2 shipped a
  clean-room implementation derived from the TIA-102 specifications
  (`codecs::mbe_baseline`) with ~31 tunable quality levers. That path is
  **deleted**. All four rates now encode and decode through the
  `blip25-codec` engine, which reproduces the reference vocoder's output
  bit-for-bit.

  **Audio output from both codecs now comes from a reverse-engineered
  engine.** Provenance divides by layer, not by codec: both wire layers
  (framing, FEC, bit interpretation) remain spec-derived, while the shared
  analysis and synthesis core was **recovered by reverse engineering the reference vocoder's
  software vocoder**. IMBE is not exempt — `ImbeDecoder` defaults to that
  overlap-add back-end and IMBE encode runs that analysis chain. The one
  asymmetry is quantizer data: AMBE+2's VQ codebooks are entirely
  firmware-recovered, IMBE's tables are spec apart from the gain ladder.
  [`ATTRIBUTION.md`](./ATTRIBUTION.md) is authoritative.

  This changes three things a consumer should weigh:
  - **Output.** Encode and decode results differ from 0.2.x on the same
    input. The new output matches the reference vocoder that commercial
    P25 equipment runs; the old output was an independent approximation
    of the same specification.
  - **Tunability.** There is nothing left to tune. The codec is now a pure
    "bits in / bits out" function. Deviation from the reference is treated
    as a defect, not a setting.
  - **Provenance and legal posture.** The clean-room derivation chain no
    longer applies to the shipped codec — including full-rate IMBE, whose
    front-end is spec-derived but whose synthesis is not. Both codecs'
    audio comes from a patent-encumbered, reverse-engineered core. See
    [`ATTRIBUTION.md`](./ATTRIBUTION.md) for what is original versus
    reverse-engineered and [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) for the
    patent position — **US8359197 is active until 2028-05-20** and the
    half-rate (AMBE+2) path unavoidably reads on its claims.

  Rationale is in
  [`docs/WHY_THE_REFERENCE_CODEC.md`](./docs/WHY_THE_REFERENCE_CODEC.md).

- **The half-rate (AMBE+2) pitch estimator now defaults to the reference-family
  estimator**, promoted on a human A/B against held-out real off-air audio:
  counted digits are easier to pick out, most audibly at the onset of
  "3 4 5". Clarity over smoothness is the criterion — the target listener is
  copying radio traffic, not enjoying it. Independent conformance evidence:
  `b0` agreement with the console goes 22% → 97% on tuned material and
  5% → 74% on the held-out corpus. Note that every *fidelity* instrument
  scores this as approximately neutral (PESQ/STOI/segSNR flat); those metrics
  reward smoothness rather than intelligibility, which is why the listening
  test decided it. Full-rate IMBE is deliberately unchanged — the ported
  chain is AMBE+2-derived and there is no evidence it is right for Phase 1.
  That is an abstention, not an oversight.

- **`crates/dvsi_codebooks` is now `crates/blip25-codebooks`** (library name
  `blip25_codebooks`). The crate name carried a third-party trademark into a
  public registry; the descriptive attribution it stood for is stated in
  `ATTRIBUTION.md` instead. Only affects callers who depended on the internal
  crate directly, which was never supported.

### Removed — **BREAKING**

- **The spec-derived codec path and all its research API.** `codecs::mbe_baseline`
  and everything it exported, including the `analysis::profile` instrumentation
  and the ~31 `set_*` quality levers. (`FrameDisposition` survives the move — it
  is now re-exported from the engine crate at the same public path,
  `vocoder::FrameDisposition`.)
  Also removed: `Vocoder::extract_params` / `Vocoder::synthesize_params` —
  parameter-layer access now lives in the `synth`, `mbe_params`, `imbe7200`,
  and `rate33` modules.
- **The `profile` cargo feature**, which gated instrumentation in the deleted
  analysis encoder and had no remaining effect.
- **The `rustfft` and `num-complex` dependencies**, used only by the deleted
  spec-derived path. With `serde` off, the crate now pulls in no runtime
  dependencies outside its own workspace — this drops eight transitive crates
  from consumer dependency trees.
- **All runtime environment-variable levers (~90 variables, 81 read
  sites).** Every knob was a research/testing instrument; each site is folded
  to its previous *unset* default, so default behavior is bit-identical
  (hermetic BITHASH + reference corpus verified). Codec behavior no longer
  depends on ambient environment in any way. Removed families:
  - Pitch-tracker calibration: `DVSI_PITCH_*` (22 knobs)
  - Gain/amplitude calibration: `DVSI_GAMMA_*`, `DVSI_REALBINS_*`,
    `DVSI_OUTER_*`, `DVSI_PTR63C_*`, `DVSI_RBAMP_*`, `DVSI_SPEC_SCALE`,
    `DVSI_WIN_OFFSET`, `DVSI_KERNEL_D`
  - Voicing: `DVSI_V_BASE/TILT/PGATE/SNAP_HI/SNAP_LO`, `DVSI_DUMP_VOICING`,
    `A5_XARM`, `WAVE_A4GATE`, `WAVE_A5`, `WAVE_A4GATE_MARGIN`
  - Tone detector tuning: `TONE_*` (14 knobs, incl. `TONE_ENCODE_OFF` — tone
    encoding now follows the `tone_detection` API flag only)
  - Synthesis A/B knobs: `DVSI_FRESH_PHASE_STEP`, `DVSI_SOFT_LIMIT`,
    `DVSI_REGEN_SCALE/THRESH`, `IMBE_OLA_SYNTH`, `IMBE_OLA_OFFSET` — the
    soft peak-limiter and DLL-exact IMBE OLA synthesis are now unconditional
  - Research alternates: `DVSI_CLEANROOM_AMPS` (the clean-room spectral
    amplitude estimator module it gated is deleted), `DVSI_KEEP_UV_INVERSE`,
    `AMP_NORM`, `MBE_SMOOTH_MODE`, `AMBE_MATCH_PAD`, `DVSI_SPECTRAL_TILT`,
    `BLIP25_VUV_*`, `BLIP25_PITCH_QUANTIZER`, `BLIP25_DECODE_POSTFILTER*`,
    `B1_RESERVE/MIN_BATCH/CTX`
  Configuration that survives is typed API only (`set_enhancement`,
  `set_tone_detection`, `set_diagnostics`, …).

### Fixed

- **`AmbePlus2_2450x2450` (r34) now emits the real reference wire layout.** The
  49 information bits of the no-FEC half-rate frame were being packed in
  `b0..b8` parameter order with no permutation, where reference applies a 3-way
  column interleave (rows of 18/18/13). Two incompatible layouts were live at
  once: `Vocoder` encode/decode used the parameter order, while `Transcoder`
  and the documentation used the interleave. Each was internally
  self-consistent — a bijective mis-permutation still round-trips — so no test
  caught it, but **2450 output was not interoperable** with other AMBE+2
  no-FEC equipment (NXDN/Fusion sinks, the AMBE-3000R serial host format), and
  mixing `Vocoder::encode(2450)` with `Transcoder(2450→3600)` produced garbage
  parameters.

  Settled against the reference RC vectors, which ship the same source audio at
  both rates: over 101 vectors / 110,840 frames the interleaved layout matches
  **97,177/97,177 (100.00%)** of clean-vector frames, and the parameter-order
  layout matched **zero**. (The excluded vectors are `*_dtx`, where
  comfort-noise frames are not a repacking of the voice stream, and the
  `dam_e{1,2,5,10}_hd` error-injection set, whose match rate falls
  monotonically with injected BER.)

  **Impact:** r34 bytes emitted by 0.3.0 differ from every prior release, and
  from earlier 0.3.0 development builds. Decoded audio is unaffected — the
  ordering is self-inverse across our own encode→decode, so only the wire
  changed. Guarded hermetically by
  `codec_acceptance::r34_wire_layout_is_dvsi_interleave_in_both_layers`;
  reproduction in `examples/r34_layout_probe.rs`.
- **No panics on hostile input.** Nine panic paths reachable from the public
  API with attacker-controlled bytes or PCM were fixed at the entry layer, with
  permanent fuzz-style and soak regression guards added. Output bits are
  unchanged.
- **IMBE voiced phase continuity and loud-frame clipping.** Restored Eq.139
  phase accumulation (a prior change caused ~50 Hz beating) and replaced a hard
  i16 clip with a soft peak limiter, matching the reference's headroom.
- **AMBE+2 sustained tones (DTMF, single tones, sweeps) no longer collapse to
  silence.** An Annex-T tone frame carries `b̂₀` in the erasure range
  `[120,123]`, so the decoder's FEC-error/erasure concealment gate mis-read tone
  frames as erasures and **repeated the previous voice frame** — whose voiced OLA
  rings up then cancels to silence — instead of dispatching to tone synthesis.
  Tone frames never reached the synthesizer. Fix: the concealment gate now checks
  the unambiguous tone signature (`tone::classify`) first and routes tone frames
  straight to synthesis (the FEC repeat/mute machinery is for corrupted voice
  frames only). DTMF/sine/sweep tones now decode at the correct level and pitch
  (whole-signal rms ~0.9–0.95× the reference, up from ~0.006–0.2×). Root
  cause was the gate, not the tone synthesis; knox/dtone and the voice path are
  unchanged (tones use the isolated `self.syn` state, never the voice `fixed`
  state). Verified vs the reference AMBE-3000 chip trace of `dtone_10` (steady rms
  3486; ours 3412).

### Added

- **Soft-decision decode: `Vocoder::decode_soft(&[i8])`.** Takes one signed LLR
  per FEC channel bit (sign = hard decision, magnitude = confidence; `SD0`
  first — the order `reference_soft_decision::unpack_nibble_stream` yields) and runs
  the crate's soft Golay/Hamming FEC (Chase-II) before synthesizing through the
  shipped `decode_bits` path. On error-free input it is bit-identical to
  `decode_bits`; under channel errors it recovers frames a hard decode would
  corrupt (the ~2 dB Golay coding gain), verified as audio on reference `*_sd`
  vectors (e.g. AMBE `dam_e10_sd` env_r 0.54 → 0.90, IMBE `clean_e10_sd`
  0.25 → 0.59 via `roundtrip_sanity --soft`). Only the FEC-bearing rates
  support it; the no-FEC rates return `VocoderError::SoftUnsupported`. New
  helpers: `Rate::soft_frame_bits()` / `Vocoder::soft_frame_bits()` (144 for
  IMBE, 72 for AMBE+2) and the streaming `Vocoder::decode_stream_soft(&[i8])`
  (soft counterpart of `decode_stream`). Softbits must come from an upstream
  soft demod/slicer,
  so this serves standalone/other-protocol/chip-interop callers — an encrypted
  P25 pipeline does FEC (and soft-decision) in its air-interface layer, outside
  crypto, and hands the codec no-FEC info bits.

- **AMBE+2 Annex-T tone frames** now synthesize on decode and are detected on
  encode (`set_tone_detection` / `VocoderBuilder`), instead of being treated as
  voice.
- **`synth`** — a params → PCM module for consumers that need granular decode
  access without going through the wire layer.

### Performance

- **Encoder is ~11.3× faster** (23.8 → 2.1 ms/frame), with byte-identical
  output — guarded by a BITHASH test so the optimization cannot silently change
  bits. Wins came from pass deduplication and twiddle-table DFT caches. The
  encoder is now real-time by default rather than an offline tool.

- **Synthesis is ~19.9× faster** (3.126 → 0.157 ms per 20 ms frame), also
  byte-identical. `dft_256_windowed` and `idft_256` evaluated 262,144
  transcendental pairs per synthesized frame from angles that depend only on
  `(k, n)` and never change; those are now a lazily-built table. Decode cost
  drops from 15.6% to 0.8% of one core per active voice stream, which matters
  when synthesis runs inline on a thread that is also draining a radio.

### Internal

- Workspace is warning-clean under `-D warnings`, enforced in CI.
- ~8.4k lines of dead research scaffolding removed; 398 crate-internal `pub`
  items demoted to `pub(crate)`.
- **No build scripts.** Every generated table is now frozen into tracked source
  and the two `build.rs` files and their CSV inputs are gone, so what is
  compiled is what is reviewed and the published tarballs are pure source.
- The non-shipped WAVE-target amplitude chain and its DLL tables were deleted
  along with the `tools/reference-diff` stub, which never progressed past printing
  "not yet implemented".

## [0.2.2] - 2026-06-21

### Added

- **Public per-field extractors** `rate33::fields_from_no_fec`,
  `rate33::fields_from_natural`, and `rate33::fields_from_fec`:
  one-liners returning the nine deprioritized half-rate parameters
  `[b̂₀..b̂₈]` (pitch, V/UV, gain, PRBA, HOC) from a half-rate frame —
  `fields_from_no_fec` for a 7-byte R34 column-interleaved wire (blip25
  `LiveEncoder` / NXDN / Fusion), `fields_from_natural` for a 7-byte
  natural-order AMBE_d frame (mbelib, Icom VE-PG4 canonical), and
  `fields_from_fec` for a 9-byte Annex-S FEC frame (full Golay decode
  first). This brings the crates.io crate to parity with the PyPI
  `blip25_mbe.rate33` Python API (0.2.2). They exist so integrators
  compare encoder output per-parameter without flat-slicing the
  prioritized 49-bit vector, which mixes parameters per slice and
  produces misleading per-field statistics (the artifact behind the
  retracted v0.2.0 "gain jumpier / L4–L5 memoryless" OTA diagnosis).
  The `halfrate_field_dump` example gains a matching `nofec7` mode.

## [0.2.1] - 2026-06-21

### Added

- **Opt-in encode-quality smoothing levers** (all default-off and
  byte-identical to prior output): `AnalysisState::set_hf_amp_ema`
  (band-selective upper-harmonic / L4–L5 amplitude EMA),
  `AnalysisState::set_vuv_stickiness` (sticky per-band voicing), and
  `DecoderState::set_gain_smooth_beta` (quantize-time gain hysteresis).
  Also exposed as `halfrate-ab-matrix` flags. Measured net-neutral-to-
  negative on offline PESQ, so they remain opt-in/diagnostic, not defaults.
- **IDAS/NXDN transport pitch-pair mode**:
  `AnalysisState::set_forced_pitch_omega` forces a frame's pitch onto a
  supplied fundamental so the IDAS odd-slot dual-permutation is lossless;
  plus the `idas_pairmode` example.
- **`halfrate_field_dump` example**: dumps the deprioritized `b̂₀..b̂₈`
  fields from either 9-byte FEC frames or 7-byte natural-order (AMBE_d)
  frames, for apples-to-apples per-field encoder comparison.

### Fixed

- `Rate::AmbePlus2_2450x2450` doc comment: the 7-byte no-FEC frame is the
  reference **r34 column interleave** (`R34_BIT_ORDER`), not "naive sequential".
  Consumers needing natural / AMBE_d order (mbelib, IDAS/NXDN over-the-air)
  must `unpack_no_fec` first.

### Notes

- OTA/IDAS quality investigation (`QUALITY_FINDINGS.md` §3.1/§3.3):
  confirmed the half-rate bit order is correct; the encoder gap vs reference is
  parameter-value estimation accuracy + the decoder, not frame-to-frame
  trajectory smoothness (an earlier per-field "jumpiness" reading was a
  bit-field-mapping artifact).

## [0.2.0] - 2026-06-14

### Changed

- **Wire modules renamed by reference rate, not protocol** (breaking):
  `imbe_wire` → `imbe7200` (full-rate IMBE, 7200 bps) and
  `ambe_plus2_wire` → `rate33` (half-rate AMBE+2, reference rate index 33).
  The rate-named modules model the codec *channel frame* — the natural
  reuse point for other half-rate AMBE+2 protocols (DMR/NXDN/P25 Phase
  2). Submodules (`::frame`, `::priority`, `::dequantize`) and all
  public items are unchanged apart from the parent path; e.g.
  `ambe_plus2_wire::frame::R34_BIT_ORDER` is now
  `rate33::frame::R34_BIT_ORDER`. Pure rename, no behavior change
  (full + half-rate reference gain vectors and all tests bit-identical).
- **Production AMBE+2 encode stack is now the half-rate default** (encode
  behavior change). `Vocoder::new(AmbePlus2_*)` enables a chip-validated
  encode-quality stack — octave-escape pitch guard, parabolic sub-sample
  pitch, §0.4 refine-off, and a hard-bounded M(ξ) voicing relaxation
  (cannot mute) — for a measured **+0.060 PESQ-nb** in the ours→chip cell.
  Gated to AMBE+2 only (the sole chip-orackable codec); full-rate IMBE
  stays bit-for-bit spec-faithful. Opt out per-lever via the `set_*`
  methods. `AnalysisState::new()` itself remains the spec-faithful
  clean-room baseline.
- **`spectral_subtraction` now defaults OFF** (encode behavior change).
  The §0.5 Boll input-side noise subtraction is now opt-in
  (`set_spectral_subtraction(true)` / `VocoderBuilder::spectral_subtraction`),
  matching the §3.4 denoiser and hum-notch. It is a no-op on clean speech
  and its noise estimator self-primes on sustained tonal content (a pinned
  −0.46 dB motion-dependent encode-gain bias the reference chip never applies);
  defaulting it off keeps the encoder closer to the emulation target and
  makes `AnalysisState::new()` fully spec-faithful. The post-decode
  Classical enhancement chain remains ON by default (a measured PESQ win;
  opt out with `set_enhancement(EnhancementMode::None)`).

### Added

- **`reference_soft_decision` module** — reference soft-decision chip handoff: the
  4-bit soft-decision (LLR) packet format, `pack_channel_bits` /
  `unpack_packet`, nibble-stream helpers, and the `SdPacketHeader` /
  `SoftDecisionError` types, for soft-FEC interchange with reference hardware.
- **Opt-in pre-analysis denoiser front-ends** (§3.4, all default OFF):
  - log-MMSE STFT denoiser with an IMCRA babble-noise tracker
    (`PreDenoise`, `DenoiseKind`; `Vocoder::set_denoise` /
    `set_denoise_kind`) — a broadband-noise *exceed* lever (up to
    +0.41 PESQ vs chip on white noise) for noisy field audio.
  - 60/120 Hz mains-hum notch (`HumNotch`; `Vocoder::set_hum_notch` /
    `set_hum_notch_mains`) — chip-beating on hum-contaminated input.
- **Encode-quality setters on `Vocoder`** — `set_pitch_decide_escape`,
  `set_pitch_subsample`, `set_pitch_refine`, `set_vuv_mxi_grade`,
  `set_vuv_pitch_coef`, `set_amp_frac_band_edges`, `set_level_scale`,
  `set_silence_shape_zero`, plus matching getters, for per-lever control
  of the encode stack above.
- **Protocol-agnostic codec FEC core** — the `rate33` codec/FEC layer is
  now exposed as a reuse boundary for other half-rate AMBE+2 carriers
  (DMR/NXDN/P25 Phase 2); see `docs/wire_formats_and_storage.md`.

### Fixed

- **`predictor::read(0)` index arithmetic** — simplified a `min`/`max`
  combination that always evaluated to a constant (clippy `min_max`
  correctness lint) to the equivalent direct index; no behavior change.

## [0.1.1] - 2026-05-31

### Fixed

- **R34 (AMBE+2 half-rate no-FEC) bit order** — `Rate::AmbePlus2_2450x2450`
  now serializes the 49 information bits in the reference vocoder's fixed 3-way column
  interleave (`ambe_plus2_wire::frame::R34_BIT_ORDER`) instead of the naive
  sequential `û₀‖û₁‖û₂‖û₃` layout. Output is now byte-exact with the reference
  chip and with an NXDN/Fusion console's no-FEC stream. `r33↔r34` cross-rate
  transcode is bit-exact against the reference RC vectors in both directions.
  (IMBE no-FEC `p25_nofec` was already byte-exact and is unchanged.)

### Added

- `ambe_plus2_wire::frame::{R34_BIT_ORDER, pack_no_fec, unpack_no_fec}` — the
  R34 interleave table and its pack/unpack helpers, with regression tests.
- `examples/derive_r34_order.rs` — empirically derives the R34 order from the
  reference RC vectors (used to produce the table above).
- `examples/decode_r33_frame.rs` — decodes a single r33 frame to parameters and
  shows the equivalent r34 serialization; useful for inspecting real captures.

## [0.1.0] - 2026-05-15

Initial public release. Research-grade Rust implementation of the MBE
codec family (IMBE / AMBE / AMBE+ / AMBE+2) targeting P25 Phase 1 and
Phase 2 pipelines, derived from TIA-102 specifications and expired reference
patents. See [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) for the AMBE+2
patent caveat (US8359197 active until 2028-05-20).

### Added

- **Chip-shaped façade** — `Vocoder::new(Rate)` per-channel handle with
  `encode_pcm` / `decode_bits` / `encode_stream` / `decode_stream`,
  modeled on the reference AMBE-3000R API.
- **`LiveEncoder` / `LiveDecoder`** — chunk-driven streaming with
  internal residue buffers for audio-callback and socket use.
- **`Transcoder`** — parametric bridge between Phase 1 IMBE and Phase 2
  AMBE+2 wire formats (and same-codec FEC ↔ no-FEC pairs).
- **`VocoderBuilder`** — explicit configuration of optional enhancement,
  spectral subtraction, tone detection, and LCG generator selection.
- **`Rate` variants** — `Imbe7200x4400`, `Imbe4400x4400`,
  `AmbePlus2_3600x2450`, `AmbePlus2_2450x2450`, plus chip-internal r34.
- **Classical post-decoder enhancement** — HPF + peaking EQ + output gain
  filter chain, **enabled by default**. Opt out via
  `vocoder.set_enhancement(EnhancementMode::None)`.
- **Spectral-subtraction noise suppression** — Boll-style with
  spectral-stationarity gated noise estimate, **enabled by default**.
  Opt out via `vocoder.set_spectral_subtraction(false)`.
- **Tone-frame detection** — Annex T DTMF / Knox single-tone matching
  with end-to-end Phase 2 tone-frame routing.
- **`extract_params` / `synthesize_params`** — parameter-layer entry
  points for users who want to interpose between analysis and synthesis.
- **Cross-rate conformance harness** — `cargo run -p conformance-vectors
  -- cross-rate-compare` validates IMBE FEC ↔ no-FEC bit-equivalence
  against reference test vectors.
- **`serde` feature** (optional, off by default) — derives
  `Serialize` / `Deserialize` on the public diagnostic types
  (`Rate`, `FrameStats`, `AnalysisStats`, `MbeParams`, etc.) for RPC use.
- **`profile` feature** (optional, off by default) — per-stage timing
  instrumentation in the analysis encoder; zero overhead when disabled.

### Documentation

- `INTEGRATION.md` — chip-shaped façade walkthrough and migration notes
  for callers coming from a reference hardware path.
- `docs/codec_family_explainer.md` — wire-format-vs-implementation
  primer for the IMBE / AMBE / AMBE+ / AMBE+2 family.
- `docs/wire_formats_and_storage.md` — guide to the four `Rate`
  variants and their on-wire byte layouts.
- `examples/vocoder_demo` — exercises one-shot, streaming, live, params,
  builder, and transcoder paths in a single binary.

### Conformance

- Validated bit-by-bit against the reference `tv-rc` and `tv-std` test
  vector suites and via PESQ/STOI black-box comparison against the
  reference AMBE-3000R hardware. No code or algorithms imported from
  existing open-source MBE projects.

[Unreleased]: https://github.com/openBLIP25/blip25-mbe/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/openBLIP25/blip25-mbe/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/openBLIP25/blip25-mbe/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/openBLIP25/blip25-mbe/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/openBLIP25/blip25-mbe/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/openBLIP25/blip25-mbe/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/openBLIP25/blip25-mbe/releases/tag/v0.1.0
