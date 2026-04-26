# blip25-mbe scope plan

**Date:** 2026-04-25 (revised 2026-04-26).
**Status:** draft. This is a working plan — replace with a real RFC if formalized.

## Scope rule: codec only, chip-shaped surface, modern Rust API

**This repo is just the vocoder.** No air interface, no FEC outside what
the codec spec itself defines, no trunking, no DSP. Everything off the
codec proper lives in `~/p25-decoder/` (already split into per-area
crates by the user) or future sibling crates for NXDN / DMR / D-STAR.

Two design constraints orient the work in this repo:

1. **Chip-shaped surface.** The DVSI AMBE-3000R is the closest existing
   reference point for "what a vocoder library should expose." A
   consumer should be able to use blip25-mbe the way they'd use the
   chip — stateful per-stream channel handle, runtime rate selection,
   uniform encode/decode/reset/configure operations across all
   generations.

2. **Modern Rust API, transport-aware.** The library API is the
   primary interface (in-process Rust call). Anything that would
   eventually become a gRPC/protobuf/WS endpoint needs serde-friendly
   types now so the future shim is mechanical. Keep the core
   transport-agnostic.

## Where things stand at the codec layer

Solid:

- IMBE Phase 1 full-rate: encoder + decoder + 5-vector PESQ baseline.
- AMBE+2 half-rate: encoder + decoder + 5-vector PESQ baseline.
- All four AMBE generations (`mbe_baseline`, `ambe`, `ambe_plus`,
  `ambe_plus2`) expose `synthesize_frame`.
- Decode mute → comfort noise per BABA-A §1.11.2 (gap 0021).
- Spec-faithful repeat handling + optional JMBE-style reset
  (gap 0022).
- Carrier-agnostic CLI entry points: `decode-raw-halfrate` (4 info
  vectors), `decode-raw-mbe` (9 b̂ values).

Open:

- Two half-rate gotchas — `MbeParams::silence()` not in half-rate
  pitch range, `encode_halfrate` errors on out-of-range pitch instead
  of dispatching silence.
- AMBE+ (Gen 2) has synth but no dequantize wrapper at the 4-vector
  interface.
- 2-frame onset attack on speech onsets (spec-locked, multi-session).
- Tonal-content deficit on alert.pcm (spec-locked).

These are the pieces that need addressing in *this* repo.

## What "chip-shaped surface" looks like concretely

The chip exposes a single channel handle and a small set of operations:

| Chip operation         | Today's blip25-mbe equivalent      |
|------------------------|-------------------------------------|
| Open channel           | `AnalysisState::new()` + `SynthState::new()` (rate-specific imports) |
| Set rate (`PKT_RATEP`) | Pick the right `encode` vs `encode_halfrate` import |
| Encode PCM → bits      | Multi-step pipeline: `analysis_encode` → `quantize` → `encode_frame` → pack |
| Decode bits → PCM      | Multi-step: unpack → `decode_frame` → `dequantize` → `synthesize_frame` |
| Reset (`PKT_RATEP` re-send) | `AnalysisState::new()` etc — start over |
| Tone frame             | Half-rate decode-side has `Decoded::Tone`; encoder doesn't emit |
| Get error stats        | Read `imbe.errors[]` from `decode_frame` output |
| Set audio mode         | Not modeled                         |
| Stream multiple channels | Caller manages multiple state objects |

The shape is mostly right but **fragmented across 4-5 imports per
operation**. A consumer plumbing a P25-Phase-1-or-Phase-2 decoder has
to know full-rate vs half-rate at the import level, not as a runtime
parameter.

A `Codec`/`Vocoder` handle would consolidate this:

```rust
let mut vocoder = Vocoder::builder()
    .rate(Rate::P25Phase1)        // or P25Phase2, NxdnType2, etc
    .build();

// Encoder side
let bits = vocoder.encode(pcm_frame)?;

// Decoder side — different vocoder instance for the receive direction
let mut vocoder = Vocoder::builder().rate(Rate::P25Phase1).build();
let pcm = vocoder.decode(&bits)?;

// Stats / diagnostics
let stats = vocoder.last_frame_stats();    // err counts, disposition, MBE params
```

That's the shape. Rate selection at runtime; state internal; uniform
operations. Maps cleanly onto a future RPC surface (one streaming
RPC per direction, one config RPC for rate selection).

## Wave 1 — codec-API hardening (small, sequenced)

### 1.1 Half-rate API gotchas (1 session)

- `MbeParams::silence_halfrate()` constructor (or make
  `MbeParams::silence()` carry a half-rate-friendly ω₀ — easiest is
  to widen its range so half-rate's `encode_pitch` accepts it).
- `encode_halfrate` returns `Ok(AnalysisOutput::Silence)` instead of
  `Err(PitchOutOfRange)` when the analysis lands an ω₀ outside the
  half-rate Annex L pitch table.

These are mechanical fixes surfaced by this session's
`halfrate-ab-matrix` baseline.

### 1.2 AMBE+ (Gen 2) decode wrapper (1 session)

`codecs::ambe_plus::synthesize_frame` exists but consumers of legacy
NXDN/DMR voice (Gen 2) currently have no path from raw 4-vector
post-FEC info bits to PCM through the AMBE+ encoder layout. Add a
`p25_halfrate::dequantize` sibling targeting AMBE+'s parameter
encoding so `decode-raw-halfrate --gen ambe-plus` works. The synth
side stays untouched.

### 1.3 Unified `Vocoder` trait + builder (2-3 sessions)

The user-facing consolidation:

```rust
pub trait Vocoder {
    fn encode_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>>;
    fn decode_bits(&mut self, bits: &[u8]) -> Result<Vec<i16>>;
    fn reset(&mut self);
    fn last_stats(&self) -> FrameStats;
}

pub enum Rate {
    P25Phase1,                    // IMBE full-rate, 18-byte FEC frame
    P25Phase2,                    // AMBE+2 half-rate, 9-byte FEC frame
    // Future:
    // RawAmbePlus2 { width },    // bare 49-bit payload, no carrier FEC
    // RawAmbePlus { width },     // Gen 2 equivalent
    // RawImbe,                   // bare 88-bit info, no FEC
}

pub struct VocoderBuilder { /* rate, gen, mute_mode, repeat_reset, etc */ }
```

State internal. Rate selection at runtime. Maps onto every existing
multi-step pipeline as a thin wrapper. Doesn't replace the low-level
modules — they stay public for advanced consumers.

### 1.4 `serde`-friendly diagnostic types (1 session)

`FrameStats`, `MbeParams`, `FrameDisposition` should derive
`Serialize`/`Deserialize` (or have a thin shadow type that does)
so a future RPC layer can send them over the wire without a
hand-rolled converter. Keep `serde` an optional feature so the core
crate stays minimal-dep.

### 1.5 Streaming iterator API (1 session, optional)

```rust
let mut session = vocoder.encode_stream(pcm_iter);   // Iterator<Item = Result<Vec<u8>>>
let mut session = vocoder.decode_stream(bits_iter);  // Iterator<Item = Result<Vec<i16>>>
```

Convenient for consumers that already have an iterator/stream of PCM
or bit chunks. Not strictly necessary; the per-frame API is fine.

## Wave 2 — boundary cleanup

### 2.1 Document the chip-vs-blip25-mbe correspondence

A short reference table mapping AMBE-3000R protocol commands to
blip25-mbe operations, and listing the things our library deliberately
doesn't model (audio mode, channel framing, RTS/CTS flow control —
all chip-protocol concerns, not codec concerns). Goes in
`INTEGRATION.md` next to the existing "Spectral enhancement" section.

### 2.2 Pull non-codec content out of conformance tooling

`conformance-speech-quality` has accumulated diagnostic surface
across the gap-0021/0022 work. The codec-side stays
(`probe_chip_dequant`, `chip_bit_disposition`, `decode-raw-*`); the
JMBE-validation pieces are the gray zone — they're cross-codec
oracles, not codec primitives. Decide whether to keep
`JMBEDumpInfo.java` and `score_ab_matrix.py` in this repo or move
them to a sibling `blip25-mbe-tools` repo when their list grows.

## Out of scope, listed explicitly so it stays out

- P25 air interface (FDMA C4FM, TDMA H-CPM): `~/p25-decoder/blip25-edge`
  + `blip25-dsp`.
- TSBK / MBT / ISP / OSP framing + procedures:
  `~/p25-decoder/blip25-trunking`.
- PDU / SNDCP / LRRP / CAD: `~/p25-decoder/blip25-data`.
- NXDN / DMR / D-STAR wire formats: future sibling crates that
  consume blip25-mbe as a path/version dep.
- Encryption (block + ESS), link-layer authentication: belongs in
  the carrier project, not the codec.

## First concrete cut

If a session opens with "do something on this plan," the smallest
useful sequence is:

1. **Wave 1.1 — half-rate gotchas** (today's session). Concrete,
   self-contained, surfaced by this week's halfrate-ab-matrix run.
2. **Run the 24 cached AMBE dumps in `~/p25-decoder/ambe_dump_*.txt`
   through `decode-raw-mbe`** — validates the carrier-agnostic
   entry point against real captures, surfaces any prioritization
   mismatches before NXDN frames arrive.
3. **Wave 1.3 — unified `Vocoder` trait** if quality/PESQ is
   feeling tapped out. Pure ergonomics work; doesn't move PESQ
   numbers but makes the integration boundary cleaner.

Wave 1.1 + the dump-replay yields a one-session deliverable that
also exercises the new NXDN-readiness work end-to-end.
