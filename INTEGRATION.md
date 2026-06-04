# Integration

How a consumer project integrates `blip25-mbe`. The running example is
`p25-decoder` (P25 Phase 1 + Phase 2), because that is the first real
consumer; the same boundaries apply to any future DMR, D-STAR, or NXDN
consumer.

## The Boundary Principle

`blip25-mbe` is a **software chip**. The mental model is the DVSI AMBE-3000
on a ThumbDV: the consumer hands it framed wire bits and gets PCM back
(or the reverse). The chip has no knowledge of:

- RF, symbol timing, demodulation
- burst structure (LDU, TDMA superframe, ISCH, FACCH, DUID)
- system context (WACN, SYSID, NAC)
- scrambling keyed by system context (P25 Phase 2 LFSR)
- call lifecycle, grants, talkgroups, link-control words
- transport (SSTP, WAV files, audio sinks)

The chip knows three things:

1. **Wire format** — how to parse a 144-bit IMBE frame, a 72-bit AMBE+2
   frame, or a DVSI r0..r63 rate-configured frame into parameters.
2. **Parameter model** — the common `MbeParams` interchange type
   (ω₀, V/UV, M_l).
3. **Codec** — analysis/synthesis between `MbeParams` and 8 kHz PCM,
   one submodule per generation.

Rate conversion is parameter-domain bits-to-bits and is a peer of the
codec and wire layers, not a sub-concern of either.

## Public API Surface

A consumer imports only these:

```rust
// Wire layer — bits ↔ MbeParams
blip25_mbe::imbe7200::Frame::decode(bits: &[u8])       -> Result<MbeParams>
blip25_mbe::imbe7200::Frame::decode_soft(soft: &[i8])  -> Result<MbeParams>
blip25_mbe::rate33::Frame::decode(bits: &[u8])       -> Result<(MbeParams, FrameKind)>
blip25_mbe::rate33::Frame::decode_soft(soft: &[i8])  -> Result<(MbeParams, FrameKind)>
blip25_mbe::dvsi_3000::Frame::decode(bits: &[u8], rate: RateConfig) -> Result<MbeParams>

// Codec layer — MbeParams ↔ PCM
blip25_mbe::codecs::<gen>::Synthesizer::synthesize(&params, &prev) -> Vec<f32>
blip25_mbe::codecs::<gen>::Analyzer::analyze(&pcm)                 -> MbeParams

// Parameter layer — the common type
blip25_mbe::mbe_params::MbeParams

// Rate conversion — no codec, no PCM
blip25_mbe::rate_conversion::transcode(&params, target: RateConfig) -> Vec<u8>

// Shared FEC (re-exported, consumers may use directly)
blip25_mbe::fec::{golay, hamming, soft_golay, soft_hamming}
```

Wire `Frame::decode` is authoritative for its wire. Consumers should not
re-implement interleave patterns, FEC polynomial application, or priority
bit ordering — those are the chip's job.

## Chip-shaped façade: `Vocoder`

The recommended entry point for in-process Rust callers is the
`Vocoder` handle in `blip25_mbe::vocoder`. It owns all per-rate state
(analysis, decoder, synth) internally and presents a uniform
encode/decode/reset surface across rates selected at runtime via
`Rate`. Builds on top of the wire + codec + parameter layers above —
no functionality the low-level modules don't have, just consolidation
behind a single handle the way the chip presents a single channel.

```rust
use blip25_mbe::vocoder::{Rate, Vocoder};

// One channel direction = one Vocoder. Two channels for full-duplex.
let mut tx = Vocoder::new(Rate::Imbe7200x4400);    // or Rate::AmbePlus2_3600x2450
let bits = tx.encode_pcm(&pcm_frame)?;          // 18-byte FEC frame (or 9 for Phase 2)

let mut rx = Vocoder::new(Rate::Imbe7200x4400);
let pcm = rx.decode_bits(&bits)?;               // 160 samples i16

// Streaming variants for whole-buffer encode/decode:
let bits: Vec<Vec<u8>> = tx.encode_stream(&pcm).collect::<Result<_, _>>()?;
let frames: Vec<Vec<i16>> = rx.decode_stream(&all_bits).collect::<Result<_, _>>()?;

// Diagnostics (chip status-register equivalent):
let stats = tx.last_stats();   // FrameStats { analysis: Some(...), decode: None }
println!("emitted {:?}, params = {:?}", stats.analysis.as_ref().unwrap().output, stats.analysis.as_ref().unwrap().params);

// Reset (chip PKT_RATEP re-send equivalent):
tx.reset();
```

### AMBE-3000R protocol → `Vocoder` correspondence

| Chip operation                    | `Vocoder` equivalent              | Notes                                                          |
|-----------------------------------|------------------------------------|----------------------------------------------------------------|
| Open channel                      | `Vocoder::new(rate)`               |                                                                |
| Set rate (`PKT_RATEP`)            | `Rate` argument at construction    | Rate fixed for handle lifetime; build a new handle to switch.  |
| Reset (re-send `PKT_RATEP`)       | `Vocoder::reset()`                 | Clears all state, keeps rate.                                  |
| Encode 160-sample PCM → bits      | `encode_pcm(&[i16; 160])`          | Returns `Vec<u8>` of `fec_frame_bytes()`.                      |
| Decode bits → 160-sample PCM      | `decode_bits(&[u8])`               | Returns `Vec<i16>` of length `frame_samples()`.                |
| Bulk encode a buffer              | `encode_stream(&[i16])`            | `ExactSizeIterator<Item = Result<Vec<u8>>>`. Drops trailing partial frame. |
| Bulk decode a buffer              | `decode_stream(&[u8])`             | Mirrors `encode_stream`.                                       |
| Read last-frame status registers  | `last_stats()` → `&FrameStats`     | `analysis` (encode-side) and `decode` (decode-side) sub-structs. |
| Frame size info                   | `frame_samples()`, `fec_frame_bytes()` | Const per-rate.                                            |

`Vocoder` is enum-dispatched (no `dyn` overhead) and exhaustively
covers each `Rate` variant. Adding a new rate means adding a variant
to `Rate` plus a `match` arm in the enum dispatch — mechanical.

### What `Vocoder` deliberately does NOT model

These are chip-protocol concerns that don't exist at the codec layer:

- **Audio mode** (sample rate, format, gain). The chip's `PKT_AMODE`
  configures these; `Vocoder` only consumes/produces 8 kHz mono i16.
  Resampling and gain control are radio-side.
- **Channel framing / RTS-CTS / flow control.** The chip's serial
  protocol has framing for transport reliability; in-process Rust
  doesn't need it.
- **Multi-channel multiplex.** The chip can multiplex multiple
  channels over one serial port; `Vocoder` is one channel. Caller
  allocates as many handles as they need.
- **Tone-frame emission** (encode side). Half-rate decode handles
  `Decoded::Tone` already; encode-side tone dispatch (analyzing PCM
  and emitting an Annex T tone frame instead of voice) is not
  implemented. A real chip would.
- **Beyond-spec heuristics** (max-headroom-reset, error-rate-freeze
  on repeat). Available as opt-in flags on the lower-level paths
  (`SynthState::set_repeat_reset_after`); not on `Vocoder` because
  the chip-shaped API stays spec-faithful by default. See gap 0021 /
  0022 for the spec-vs-chip divergence story.

### Wire-format support matrix

| Rate                  | Wire bytes | Codec        | End-to-end | Carriers                                      |
|-----------------------|-----------:|--------------|:---------:|------------------------------------------------|
| `Imbe7200x4400`       |         18 | IMBE Gen 1   | ✅        | P25 Phase 1 FDMA voice                         |
| `Imbe4400x4400`       |         11 | IMBE Gen 1   | ✅        | IMBE codec, FEC layer stripped (88 prioritized info bits packed MSB-first). For lossless transports or diagnostic / cross-decoder tooling |
| `AmbePlus2_3600x2450` |          9 | AMBE+2 Gen 3 | ✅        | P25 Phase 2 TDMA voice; carrier-agnostic for any AMBE+2 sink that lays its post-FEC info as 4 vectors of widths {12,12,12,12} (NXDN type-2 typical, DMR enhanced) |
| `AmbePlus2_2450x2450` |          7 | AMBE+2 Gen 3 | ✅        | AMBE+2 half-rate codec, FEC layer stripped (49 info bits + 7 pad bits, MSB-first). DVSI PKT_RATEP rate 34 |

For carriers whose post-FEC info layout differs (older DMR, AMBE+
Gen 2 NXDN, D-STAR's specific bit layout), drop down a layer to
`decode-raw-mbe` (raw 9-value `b̂₀..b̂₈`) or `decode-raw-ambe-plus2`
(post-FEC 4-vector info) — both are available as CLI subcommands of
`conformance-speech-quality` and as direct library calls into
`rate33::dequantize` + `codecs::ambe_plus2`.

## What Stays in the Consumer (p25-decoder example)

These responsibilities are radio-side, not chip-side. They never migrate.

| Responsibility                        | File(s) in p25-decoder                                       | Why it stays                                                  |
|---------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| LDU → nine 144-bit IMBE frames        | `crates/blip25-core/src/ldu.rs`                              | Burst geometry; fixed offsets inside a 1728-bit LDU           |
| Status-symbol strip, link-control     | `crates/blip25-core/src/ldu.rs`, `link_control.rs`           | P25 signaling layered with voice, not codec                   |
| TDMA burst classification             | `crates/blip25-core/src/tdma/{burst,duid,isch,facch,ess}.rs` | Burst protocol parsing                                        |
| **LFSR descrambling (Phase 2)**       | `crates/blip25-core/src/tdma/lfsr.rs`                        | Keyed by WACN/SYSID/NAC; runs *before* 72-bit frames exist    |
| Call lifecycle, per-call WAV, SSTP    | `crates/blip25-vocoder/src/{lib,tdma}.rs`                    | Project-specific orchestration around the chip                |
| DVSI ThumbDV TCP client               | `crates/ambe-client/`                                        | Real-hardware peer of the software chip (useful for A/B)      |

The LFSR case is the clearest test of the boundary: a real DVSI chip
does not know about WACN/NAC either. The host feeds it already-descrambled
72-bit frames. Same contract for `blip25-mbe`.

## Consumer Shim

Each consumer writes a small adapter between its legacy types and
`MbeParams`. For p25-decoder, this is two files inside `blip25-vocoder`:

- `imbe_adapter.rs` — wraps `imbe7200::Frame` + `MbeParams` into the
  legacy `ImbeFrame`/`ImbeParams` shape, or migrates call sites to use
  `MbeParams` directly.
- `ambe_adapter.rs` — same for `rate33::Frame`, plus `FrameKind`
  routing (voice vs. tone).

Downstream orchestration (`Phase1Vocoder`, `TdmaVocoder`, per-call WAV
writers, SSTP emitters) is unchanged.

## Design Decisions at the Boundary

These came up concretely during the p25-decoder boundary mapping. Documented
so future protocol consumers do not re-derive them.

### Tone frames

Detection is wire-layer (a specific bit pattern in the 72-bit AMBE+2
frame). Synthesis is codec-layer (dual-sinusoid DTMF / ringback).

- `rate33::Frame::decode` returns `FrameKind::{Voice(MbeParams), Tone(ToneParams)}`.
- `codecs::<gen>::synthesize_tone(&ToneParams)` renders audio.

### Spectral enhancement

The enhancement and phase-regeneration algorithms (US5701390, US8595002,
US8315860) are codec-internal quality logic, not wire format. They live
in `codecs/ambe_plus` and `codecs/ambe_plus2`. They do not live in
`mbe_params/` and they do not live in any wire submodule.

A wire submodule's only contract with the codec is `bits ↔ MbeParams`.
Pairing the `imbe7200` wire with the `codecs::ambe_plus2` codec is a
valid combination (the SCBA-mask deployment pattern); the wire layer
should make that combination expressible.

### Soft-decision FEC

`blip25-mbe::fec` re-exports soft-decision Golay and Hamming so that
consumers which need them for non-vocoder purposes (e.g., P25 TSBK decode)
can pull them from a single place. If a consumer emerges that needs these
without the rest of `blip25-mbe`, factor them into a small `blip25-fec`
crate both depend on. Do not factor preemptively.

### Loudness calibration reference

`blip25-mbe` targets **DVSI chip PCM parity per BABA-A**. The only
valid calibration references are:

1. **Canonical DVSI test-vector PCM** — `DVSI/Vectors/tv-rc/r33/*.pcm`
   (half-rate) and `DVSI/Vectors/tv-std/tv/*.pcm` (full-rate).
2. **Live DVSI chip PCM** via `conformance/chip/` (subject to the
   chip-oracle caveats in `~/blip25-specs/analysis/ambe3000_chip_oracle_caveats.md`).

**Other P25 open-source decoders (SDRTrunk / JMBE / OP25) are NOT
valid references.** Their PCM output contains post-synthesis gain
(typically ~8× for SDRTrunk) that is layered on top of the BABA-A
synthesis pipeline. Calibrating `γ_w` or any other codec constant
against those outputs would require values far outside the spec's
plausible range and would break DVSI-chip conformance.

Measured 2026-04-19 on canonical DVSI tv-rc/r33 half-rate reference
PCM across 5 vectors (`alert`, `clean`, `cp0`, `cp1`, `cp31`): our
output RMS is **0.999× to 1.067× of DVSI reference** — within the
§1.10+§1.12 processing envelope documented in the chip-oracle
caveats analysis. This is the correct target; loudness parity with
SDRTrunk is not.

**Consumers requiring SDRTrunk-parity loudness for operational UX**
have two options:

1. Apply post-vocoder gain at the application layer (multiply the
   i16 PCM by the desired factor before handing to the WAV writer).
2. Set `ClassicalConfig::output_gain_db` on the [`enhancement`]
   chain — `+9.0` matches SDRTrunk's empirical post-decode gain. The
   gain saturates at i16 clip and runs after all other enhancement
   stages, so it composes cleanly with the HPF + peaking defaults.

Either is an application-layer choice — the codec itself stays
calibrated against DVSI parity per BABA-A.

#### Tones vs speech: random-phase reconstruction artifact

A pure-sine round-trip through the spec-faithful IMBE (full-rate)
path does **not** reproduce a pure sine. Per BABA-A Eq. 141, voiced
harmonics are synthesized with a fresh random phase per frame; the
inter-frame phase interpolation creates a frequency-modulated
approximation of the input tone. Empirical measurement (`amp_i16`
sweep at 349 Hz, 100-frame round-trip):

| Input amp | Δ peak | Δ RMS |
|-----------|--------|-------|
| 16384     | -0.86 dB | -4.91 dB |
| 8192      | -0.52 dB | -5.19 dB |
| 4096      | -0.26 dB | -5.00 dB |
| 2048      | -0.65 dB | -5.14 dB |

Peak amplitude is preserved; RMS is consistently ~5 dB lower
because the reconstructed waveform has higher crest factor than a
pure sine. **This is structural to MBE-class codecs** — fixing it
would require beyond-spec phase tracking. On real speech, RMS
round-trip is within ~0.3 dB (verified on the 5-vector PESQ
harness's `clean.pcm`); the random-phase RMS drop only manifests
on monotone / near-monotone inputs.

For Phase 2 (AMBE+2) the **Annex T tone-frame path bypasses MBE
synthesis entirely** and reconstructs deterministic sines from a
lookup table — round-trip on a sine is unity (±0.3 dB) post-2026-04-30
amplitude calibration fix. Use `Vocoder::set_tone_detection(true)`
for half-rate consumers carrying DTMF/Knox/single-tone content.
Phase 1 IMBE has no equivalent fast path; tones over Phase 1 will
always exhibit the random-phase RMS drop.

Reproducing the DVSI-parity measurement:

```
cargo run -p blip25-conformance-vectors --release -- \
    decode-pcm-ambe-plus2 clean --write-pcm /tmp/ours.pcm
```

and compare RMS to `DVSI/Vectors/tv-rc/r33/clean.pcm` — expect
ratio ≈ 1.0×.

### Parameter type unification

Consumers that predate `blip25-mbe` (such as p25-decoder) may have
separate `ImbeParams` and `AmbeParams` structs with overlapping fields.
Migrate to the unified `MbeParams` trait / type. Type aliases in the
consumer are acceptable during transition, but the long-term target is
direct use of the `blip25-mbe` types.

## Test Fixture Allocation

Fixtures move with the code that tests them.

**To `blip25-mbe/conformance/`:**
- Hex-encoded wire frames (`ambe_dump_*.txt`, IMBE frame dumps)
- Soft-decision test vectors (`*.soft8`)
- Reference PCM (`*.pcm`, `*.wav`) for end-to-end synthesis checks
- DVSI TIA-102 test vectors, when present

**Stays in the consumer:**
- Raw IQ captures (RF-layer)
- Symbol dumps, burst sync fixtures
- LFSR-descramble verification (system-context dependent)
- Anything that exercises code which did not migrate

## Scope of Migration from p25-decoder

Rough shape of the diff when `blip25-mbe` replaces the in-tree vocoder:

- **~2,700 lines move out**, cleanly: params + frames + tables + synth +
  enhance + FEC.
- **~500 lines stay**, unchanged in behavior: LDU extraction, LFSR,
  call lifecycle, SSTP, WAV.
- **Two new adapter files** in `blip25-vocoder` translate between legacy
  `ImbeParams`/`AmbeParams` and `MbeParams`.
- **Fixture shuffle**: vocoder-level test vectors into
  `blip25-mbe/conformance/vectors/`, IQ/symbol/burst fixtures stay.
- **No P25 protocol code is rewritten.** Only re-imports and adapter calls.

The change is surgical, not invasive. The mental model holds:
`blip25-mbe` is the chip, the consumer is the radio around it.
