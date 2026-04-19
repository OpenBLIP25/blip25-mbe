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
blip25_mbe::p25_fullrate::Frame::decode(bits: &[u8])       -> Result<MbeParams>
blip25_mbe::p25_fullrate::Frame::decode_soft(soft: &[i8])  -> Result<MbeParams>
blip25_mbe::p25_halfrate::Frame::decode(bits: &[u8])       -> Result<(MbeParams, FrameKind)>
blip25_mbe::p25_halfrate::Frame::decode_soft(soft: &[i8])  -> Result<(MbeParams, FrameKind)>
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

- `imbe_adapter.rs` — wraps `p25_fullrate::Frame` + `MbeParams` into the
  legacy `ImbeFrame`/`ImbeParams` shape, or migrates call sites to use
  `MbeParams` directly.
- `ambe_adapter.rs` — same for `p25_halfrate::Frame`, plus `FrameKind`
  routing (voice vs. tone).

Downstream orchestration (`Phase1Vocoder`, `TdmaVocoder`, per-call WAV
writers, SSTP emitters) is unchanged.

## Design Decisions at the Boundary

These came up concretely during the p25-decoder boundary mapping. Documented
so future protocol consumers do not re-derive them.

### Tone frames

Detection is wire-layer (a specific bit pattern in the 72-bit AMBE+2
frame). Synthesis is codec-layer (dual-sinusoid DTMF / ringback).

- `p25_halfrate::Frame::decode` returns `FrameKind::{Voice(MbeParams), Tone(ToneParams)}`.
- `codecs::<gen>::synthesize_tone(&ToneParams)` renders audio.

### Spectral enhancement

The enhancement and phase-regeneration algorithms (US5701390, US8595002,
US8315860) are codec-internal quality logic, not wire format. They live
in `codecs/ambe_plus` and `codecs/ambe_plus2`. They do not live in
`mbe_params/` and they do not live in any wire submodule.

A wire submodule's only contract with the codec is `bits ↔ MbeParams`.
Pairing the `p25_fullrate` wire with the `codecs::ambe_plus2` codec is a
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
should apply post-vocoder gain in the consumer (e.g. multiply the
i16 PCM by the desired factor before handing to the WAV writer).
That is an application-layer concern, not a codec concern.

Reproducing the DVSI-parity measurement:

```
cargo run -p blip25-conformance-vectors --release -- \
    decode-pcm-halfrate clean --write-pcm /tmp/ours.pcm
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
