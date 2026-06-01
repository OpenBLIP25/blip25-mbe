# Changelog

All notable changes to **blip25-mbe** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/openBLIP25/blip25-mbe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/openBLIP25/blip25-mbe/releases/tag/v0.1.0
