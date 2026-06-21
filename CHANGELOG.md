# Changelog

All notable changes to **blip25-mbe** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  DVSI **r34 column interleave** (`R34_BIT_ORDER`), not "naive sequential".
  Consumers needing natural / AMBE_d order (mbelib, IDAS/NXDN over-the-air)
  must `unpack_no_fec` first.

### Notes

- OTA/IDAS quality investigation (`QUALITY_FINDINGS.md` §3.1/§3.3):
  confirmed the half-rate bit order is correct; the encoder gap vs DVSI is
  parameter-value estimation accuracy + the decoder, not frame-to-frame
  trajectory smoothness (an earlier per-field "jumpiness" reading was a
  bit-field-mapping artifact).

## [0.2.0] - 2026-06-14

### Changed

- **Wire modules renamed by DVSI rate, not protocol** (breaking):
  `imbe_wire` → `imbe7200` (full-rate IMBE, 7200 bps) and
  `ambe_plus2_wire` → `rate33` (half-rate AMBE+2, DVSI rate index 33).
  The rate-named modules model the codec *channel frame* — the natural
  reuse point for other half-rate AMBE+2 protocols (DMR/NXDN/P25 Phase
  2). Submodules (`::frame`, `::priority`, `::dequantize`) and all
  public items are unchanged apart from the parent path; e.g.
  `ambe_plus2_wire::frame::R34_BIT_ORDER` is now
  `rate33::frame::R34_BIT_ORDER`. Pure rename, no behavior change
  (full + half-rate DVSI gain vectors and all tests bit-identical).
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
  −0.46 dB motion-dependent encode-gain bias the DVSI chip never applies);
  defaulting it off keeps the encoder closer to the emulation target and
  makes `AnalysisState::new()` fully spec-faithful. The post-decode
  Classical enhancement chain remains ON by default (a measured PESQ win;
  opt out with `set_enhancement(EnhancementMode::None)`).

### Added

- **`dvsi_soft_decision` module** — DVSI soft-decision chip handoff: the
  4-bit soft-decision (LLR) packet format, `pack_channel_bits` /
  `unpack_packet`, nibble-stream helpers, and the `SdPacketHeader` /
  `SoftDecisionError` types, for soft-FEC interchange with DVSI hardware.
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
  now serializes the 49 information bits in DVSI's fixed 3-way column
  interleave (`ambe_plus2_wire::frame::R34_BIT_ORDER`) instead of the naive
  sequential `û₀‖û₁‖û₂‖û₃` layout. Output is now byte-exact with the DVSI
  chip and with an NXDN/Fusion console's no-FEC stream. `r33↔r34` cross-rate
  transcode is bit-exact against the DVSI RC vectors in both directions.
  (IMBE no-FEC `p25_nofec` was already byte-exact and is unchanged.)

### Added

- `ambe_plus2_wire::frame::{R34_BIT_ORDER, pack_no_fec, unpack_no_fec}` — the
  R34 interleave table and its pack/unpack helpers, with regression tests.
- `examples/derive_r34_order.rs` — empirically derives the R34 order from the
  DVSI RC vectors (used to produce the table above).
- `examples/decode_r33_frame.rs` — decodes a single r33 frame to parameters and
  shows the equivalent r34 serialization; useful for inspecting real captures.

## [0.1.0] - 2026-05-15

Initial public release. Research-grade Rust implementation of the MBE
codec family (IMBE / AMBE / AMBE+ / AMBE+2) targeting P25 Phase 1 and
Phase 2 pipelines, derived from TIA-102 specifications and expired DVSI
patents. See [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) for the AMBE+2
patent caveat (US8359197 active until 2028-05-20).

### Added

- **Chip-shaped façade** — `Vocoder::new(Rate)` per-channel handle with
  `encode_pcm` / `decode_bits` / `encode_stream` / `decode_stream`,
  modeled on the DVSI AMBE-3000R API.
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
  against DVSI test vectors.
- **`serde` feature** (optional, off by default) — derives
  `Serialize` / `Deserialize` on the public diagnostic types
  (`Rate`, `FrameStats`, `AnalysisStats`, `MbeParams`, etc.) for RPC use.
- **`profile` feature** (optional, off by default) — per-stage timing
  instrumentation in the analysis encoder; zero overhead when disabled.

### Documentation

- `INTEGRATION.md` — chip-shaped façade walkthrough and migration notes
  for callers coming from a DVSI hardware path.
- `docs/codec_family_explainer.md` — wire-format-vs-implementation
  primer for the IMBE / AMBE / AMBE+ / AMBE+2 family.
- `docs/wire_formats_and_storage.md` — guide to the four `Rate`
  variants and their on-wire byte layouts.
- `examples/vocoder_demo` — exercises one-shot, streaming, live, params,
  builder, and transcoder paths in a single binary.

### Conformance

- Validated bit-by-bit against the DVSI `tv-rc` and `tv-std` test
  vector suites and via PESQ/STOI black-box comparison against the
  DVSI AMBE-3000R hardware. No code or algorithms imported from
  existing open-source MBE projects.

[Unreleased]: https://github.com/openBLIP25/blip25-mbe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/openBLIP25/blip25-mbe/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/openBLIP25/blip25-mbe/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/openBLIP25/blip25-mbe/releases/tag/v0.1.0
