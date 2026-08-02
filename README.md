# blip25-mbe

A Rust implementation of the reference MBE codec family for P25, functionally
parallel to the AMBE-3000R chip's codec layer.

The `Vocoder` API exposes four wire rates across two codecs — **IMBE**
(P25 Phase 1, full-rate 7200×4400 and info-only 4400×4400) and **AMBE+2**
(P25 Phase 2, half-rate 3600×2450 and info-only 2450×2450) — plus parametric
rate conversion between them. The AMBE+2 vocoder layer is also the voice
codec for DMR Tier II/III, modulo carrier-specific burst framing; NXDN and
D-STAR use different MBE variants that this crate does not encode.

> **Patent notice.** This source code is provided for research and
> interoperability study only. The half-rate (AMBE+2) implementation
> unavoidably reads on the claims of **US8359197**, active until
> **2028-05-20**. See [`PATENT_NOTICE.md`](./PATENT_NOTICE.md) for the full
> list, comparable-project survey, and project policy.

## Install

```toml
[dependencies]
blip25-mbe = "0.3"
```

Requires Rust 1.85 or later. With default features the crate pulls in no
runtime dependencies outside its own workspace; enable `serde` to derive
`Serialize` / `Deserialize` on the diagnostic and parameter types.

> **Upgrading from 0.2.x?** 0.3.0 replaces the codec engine wholesale and
> removes every runtime environment-variable lever. Read the
> [0.3.0 changelog entry](./CHANGELOG.md) before bumping — the migration is
> short but it is not a drop-in.

## Quick start

```rust
use blip25_mbe::vocoder::{Rate, Vocoder};

// Open a P25 Phase 1 (full-rate IMBE) channel.
let mut tx = Vocoder::new(Rate::Imbe7200x4400);
let pcm: [i16; 160] = [0; 160];
let bits = tx.encode_pcm(&pcm).unwrap();    // 18-byte FEC frame

let mut rx = Vocoder::new(Rate::Imbe7200x4400);
let pcm = rx.decode_bits(&bits).unwrap();   // 160 samples
```

Two runnable examples:

```bash
cargo run --release --example vocoder_demo  -p blip25-mbe   # full API walkthrough
cargo run --release --example vocoder_bench -p blip25-mbe   # throughput micro-benchmark
```

## Architecture

### What you depend on

```
blip25-mbe  = "0.3"        ← the only one you name
   └── blip25-codec        ← engine, implementation detail
         └── blip25-codebooks   ← VQ tables, leaf
```

Three crates, published to crates.io in lockstep at one version. The bottom
two exist there only because cargo cannot resolve path dependencies; neither
carries an API stability promise. A fourth workspace member,
`conformance/roundtrip`, is a `publish = false` harness whose examples read
the non-redistributable reference corpus, and `fuzz/` sits outside the workspace
entirely because it requires nightly.

Relative size is a fair map of where the difficulty lives:

| Crate | Lines | Role |
|---|---:|---|
| `blip25-codec` | 30,405 | the DSP |
| `blip25-mbe` | 12,776 | wire formats, public API, FEC |
| `blip25-codebooks` | 87 + blobs | data |

### The front door

`Vocoder` is a chip-shaped façade — one handle per channel direction,
modeled on the AMBE-3000R's per-channel API, enum-dispatched with no `dyn`.
Around the per-frame primitive:

- `encode_stream` / `decode_stream` — slice → `Iterator<Item = Result<…>>`
- `LiveEncoder` / `LiveDecoder` — chunk-driven with an internal residue
  buffer, for audio-callback and socket use
- `decode_soft` / `decode_stream_soft` — soft-decision decode from per-bit
  LLRs (`&[i8]`), for receivers that can surface demodulator confidence;
  `soft_frame_bits` gives the expected count
- `Transcoder` — P25 Phase 1 ↔ Phase 2 wire-bit bridge
- `set_tone_detection` / `set_enhancement`, and `VocoderBuilder` to
  configure the post-decode chain up front

See [`INTEGRATION.md`](./INTEGRATION.md) for the AMBE-3000R protocol →
`Vocoder` operation correspondence.

### Inside `blip25-mbe`

Three orthogonal axes meeting at one interchange type:

```
                    ┌──────────────┐
   wire  ──────────▶│  MbeParams   │◀────────── codec
   imbe7200 (3120)  │              │            synth (181)
   rate33   (3213)  │  ω₀          │            enhancement (397)
                    │  L           │
                    │  voiced[]    │
                    │  amplitudes[]│
                    └──────┬───────┘
                           │
                    rate_conversion (1370)
                    bits → params → bits, no PCM
```

- **`vocoder`** (2657) — the façade; owns all per-rate state
- **`imbe7200` / `rate33`** — one module per protocol-rate combination:
  deframe, FEC, deinterleave, dequantize
- **`mbe_params`** — the interchange type: fundamental frequency, harmonic
  count, per-harmonic voicing and spectral amplitudes
- **`rate_conversion`** — a *peer* of the wire and codec layers, not a
  decoder afterthought; converts in the parameter domain, never touching PCM
- **`fec`** (749) — Golay / Hamming, hard and soft decision
- **`synth`** (181) — thin adapter, params → PCM over the engine
- **`enhancement`** (397) — optional post-decode filter chain, **off by
  default**, because enabling it is a deviation from the reference

Two entry altitudes are supported on purpose: `Vocoder` for most callers,
and the layered free functions (`rate33::dequantize` → `synth::synthesize_frame`)
for anyone whose carrier lays out post-FEC bits differently. That pair is
the carrier-agnostic seam.

### Inside `blip25-codec`

```
enc/   44 files  18,962 lines   ← 62% of the engine
dec/   16 files   4,820
imbe/   8 files   1,809
root   10 files   4,814         (synth, tone, phase_regen, fec, frame, fixops)
```

The asymmetry is the shape of the problem: encode carries the difficulty —
`b1_audio.rs` (2346), `loudness_fixed.rs` (1951), `pitch.rs` (1231),
`b0_audio.rs` (975), `voicing_fixed.rs` (669) — while the decode side is
largely overlap-add machinery (six `ola_*.rs` files).

`imbe/` is deliberately self-contained, with its own
`frame`/`dequantize`/`quantize`/`tables`/`dsp`/`math`/`fixp`. The AMBE+2
front-end is instead *fused* with the shared core rather than living in its
own folder — a consequence of the engine being built AMBE+2-first. That is
why `dequantize` hosts `MbeParams` itself, the type IMBE also produces.

### The path through

```
encode:  &[i16; 160]
           → Vocoder::encode_pcm
           → match rate → enc.encode_frame_r33 / _r34 / encode_imbe_frame
           → [analysis: pitch → voicing → amplitude → VQ]
           → Vec<u8>

decode:  &[u8]
           → Vocoder::decode_bits  (length-checked first)
           → Decoder / ImbeDecoder
           → [FEC → deprioritize → tone classify → dequantize → OLA synth]
           → enhancement (no-op by default)
           → Vec<i16>
```

[`DESIGN.md`](./DESIGN.md) has the architectural rationale;
[`docs/WHY_THE_REFERENCE_CODEC.md`](./docs/WHY_THE_REFERENCE_CODEC.md)
explains why the codec matches the reference vocoder rather than chasing a
quality metric. New to the MBE family, or wondering why open-source P25
sounds worse than commercial radios?
[`docs/codec_family_explainer.md`](./docs/codec_family_explainer.md) is the
wire-format-vs-implementation story.

Before changing anything in the codec, read
[`docs/PROJECT_KNOWLEDGE.md`](./docs/PROJECT_KNOWLEDGE.md) — the constants that
must not be re-tuned, the metrics that mislead, and the approaches that are
ruled out.

## Testing

| Tier                | Requirements              | Who runs it                       |
|---------------------|---------------------------|-----------------------------------|
| Unit + integration  | None                      | Anyone, project CI                |
| Pinned output       | None                      | Project CI (`golden_output`)      |
| Routing parity      | None                      | Project CI (`codec_acceptance`)   |
| Vector conformance  | reference test vectors on disk | Developers with access       |

`cargo test` is meaningful on its own and is what CI gates on. The two
hermetic tiers do different jobs, and the distinction matters:

- **`golden_output`** compares codec output against hashes pinned as
  literals. This is the only test that can detect a codec regression, and
  it is what makes the cross-platform matrix meaningful — it runs on
  aarch64, macOS, and Windows, so a float divergence fails loudly instead
  of shipping.
- **`codec_acceptance`** compares the `Vocoder` façade against the engine
  called directly. That verifies *routing*, not output: if the engine
  changes, both sides change together and it stays green. Useful, but not
  a regression gate — do not mistake it for one.

Vector conformance against the reference corpus is additional, skips
automatically when the corpus is absent, and is never required for a
contributor to validate their work.

Every public entry point is fuzzed (`fuzz/`, requires nightly). See
[`RELEASING.md`](./RELEASING.md) for the release process.

## Provenance

Provenance splits by **layer**, not by codec — the two codecs are close to
symmetric:

| Layer | Provenance |
|---|---|
| Wire framing, FEC, bit interpretation (both codecs) | **Spec** — TIA-102.BABA / BABA-A, clean-room derived |
| Quantizer data | **AMBE+2: firmware** (all VQ codebooks). **IMBE: spec**, except the gain ladder |
| Analysis / encode chain (shared) | **Reverse-engineered** |
| Synthesis / decode audio (shared) | **Reverse-engineered** |

**Both codecs' wire layers are spec-derived.** Deframing, FEC, and bit
interpretation come from TIA-102.BABA / BABA-A; the AMBE+2 half was carried
over from this project's original clean-room implementation, and the IMBE half
was built from the published fixed-point reference plus ITU-T G.191 basic
operators.

**Both codecs' audio comes from a reverse-engineered core.** The shared
analysis and synthesis engine was recovered from a compiled reference vocoder image
— x86 disassembly transliterated to fixed-point Rust, constants recovered from
the binary's data section, correctness pinned bit-exact against the vocoder's
own output. It is not a copy of the reference vendor's source, which this project never
obtained or read; it *is* a derivation of the reference vocoder's behavior, and it reads
on the active patents above. IMBE is not exempt: `ImbeDecoder` defaults to that
same overlap-add back-end, and IMBE encode runs the same analysis chain.

**The one real asymmetry is quantizer data.** AMBE+2's vector-quantizer
codebooks (PRBA, HOC, gain) are entirely firmware-recovered. IMBE's quantizer
tables are published spec data, with a single exception — the gain ladder,
where the reference vendor's real table diverges from Annex-E at its bottom eight levels.

[`ATTRIBUTION.md`](./ATTRIBUTION.md) is the authoritative statement of what
is original work and what is derived, and states it in both directions —
neither overstating originality nor understating the derivation.

## License

MIT. See [`LICENSE`](./LICENSE).
