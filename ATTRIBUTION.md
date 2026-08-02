# Attribution & provenance

This document states plainly what in this repository is original work, what is
reverse-engineered for interoperability, and how the two were built. It exists
so the project's provenance is represented **accurately** — neither overstating
originality nor understating the substantial original engineering here.

It complements, and does not replace, [`PATENT_NOTICE.md`](./PATENT_NOTICE.md)
and [`docs/WHY_THE_REFERENCE_CODEC.md`](./docs/WHY_THE_REFERENCE_CODEC.md).

## The short version

The P25 MBE vocoder family (IMBE, AMBE+2) is proprietary and patent-encumbered.
This project did **not** obtain, read, or copy the reference vocoder's source
code. What it did was **reverse-engineer the vocoder's observable behavior**
from a compiled binary image and reproduce it bit-for-bit, then build an
extensive original codebase — protocol, DSP, tooling, and test infrastructure —
around that matched engine.

That distinction matters and cuts both ways:

- It is **not** a copy of the reference vocoder's source. The numeric engine was
  recovered by disassembly, observation, and measurement — not lifted from source.
- It **is** derived from the reference vocoder. Reproducing a patented algorithm's
  behavior by reverse engineering is still a derivation of that behavior, and
  the result reads on the same active patents. This is stated, not hidden.

## What is reverse-engineered (derived from the reference vocoder)

| Component | What it is | How it was obtained |
|---|---|---|
| `crates/blip25-codec` shared core (`enc/`, `dec/`, `synth`, `phase_regen`) | The analysis and synthesis engine — the numeric core that makes output match the reference codec. Used by **both** codecs: `ImbeDecoder` defaults to this overlap-add back-end, and `encode_imbe_frame` runs this analysis chain. | Reverse-engineered from a compiled reference vocoder image (x86 disassembly transliterated to fixed-point Rust; constants recovered from the binary's data section; correctness pinned bit-exact against the codec's own output) |
| `crates/blip25-codebooks` | AMBE+2 vector-quantizer codebooks (PRBA24 / PRBA58 / HOC b5–b8 / gain) | **Firmware-recovered.** These are the quantizer data the AMBE+2 front-end indexes; they are not published spec tables. |
| `crates/blip25-codec` AMBE+2 wire layer (`frame.rs`, `dequantize.rs`, `tables.rs`) | Deframing, Golay/PN FEC, Annex-S deinterleave, bit interpretation | **Spec-derived, clean-room.** Carried over from this project's original clean-room implementation (TIA-102.BABA-A §2.4–§2.6, §2.11–§2.13); the Annex L/M/N/O/S and bit-prioritization tables are extracted from the standard and frozen into source. |
| `crates/blip25-codec/src/imbe` (IMBE front-end) | Deframing, FEC, dequantization for full-rate IMBE — **not** its synthesis | **Spec-derived.** TIA-102.BABA publishes the IMBE bitstream and its bit-exact fixed-point reference, so this was built from that standard plus ITU-T G.191 basic operators, and a firmware scan confirmed the firmware image computes the same constants in code rather than tabulating them. **One exception:** `imbe::tables::GAIN_QNT_TBL` / `MONO65_ENCODE` are **firmware-recovered** — the reference vocoder's real gain ladder diverges from the published Annex-E at its bottom eight levels, and the encoder uses the firmware ladder. Other implementations of the same standard (OP25's `imbe_vocoder`) were consulted as a bit-exactness cross-check only. |
| `crates/blip25-mbe/src/generated` (Annex E/O/P/Q/R) | The **published** IMBE / AMBE+2 quantizer tables (gain levels, PRBA, HOC) that the `rate33` / `imbe7200` wire-layer dequantizers index | **Spec-derived, clean-room.** Normative TIA-102.BABA-A Annex data, verified against the published standard. These are separate data from the firmware-recovered `blip25-codebooks` tables above, which are what the shipped engine indexes. |

These crates are published to crates.io alongside `blip25-mbe`, which cannot
be published without them. They are implementation detail with no API stability
promise — depend on `blip25-mbe`. Publication does not change what they are:
patent-encumbered, derived from the reference vocoder, and provided for research and
interoperability study. See `PATENT_NOTICE.md` for the specific patents they
read on (notably **US8359197**, active to **2028-05-20**).

## What is original work

A large majority of the repository is first-party engineering that is **not**
the reference vocoder's and was not reverse-engineered from it:

- **P25 protocol layer** — the wire formats, NID/framing, FEC (Golay, Hamming,
  interleavers), bit orderings, and rate conversion, derived clean-room from the
  public TIA-102 standards, not from any vendor binary.
- **The library and its API** — the `Vocoder` surface, the single-pass streaming
  encoders/decoders (`EncodeStream`/`DecodeStream`/`LiveEncoder`), soft-decision
  decode, the enhancement/AGC layer, and the parametric rate-conversion path.
- **The clean-room specification effort** — the derived TIA-102 implementation
  specs that predated the matched engine (`~/blip25-specs`), a genuine
  independent reimplementation track in its own right.
- **The tone path** — Annex-T tone handling and the original detection
  heuristics.
- **The methodology and tooling** — this is where the "many hours" live, and
  they are real engineering: a large A/B ear-testing discipline against the
  reference, the diagnostic harnesses (spectral correlation, field agreement,
  byte-exact scoring, PESQ/STOI as locators), the brute-force parameter
  searches, and the quality investigation recorded in `QUALITY_FINDINGS.md`.
  Matching a black-box target bit-for-bit by measurement and iteration is
  itself substantial original work — it is the labor of interoperability
  engineering, not of copying.

## How correctness was reached

By ear and by measurement against the reference, not by access to it. The
target was a black box: a codec running on a dispatch console. Reaching
bit-exactness meant disassembling behavior, hypothesizing the fixed-point
math, and confirming it frame-by-frame against captured output — thousands of
iterations of hypothesis, A/B comparison, and refinement. That process
produced the matched engine and, along the way, the original protocol, DSP,
and tooling that make up most of this tree.

## Honest bottom line

Represent it as what it is: **original interoperability engineering built around
a reverse-engineered, patent-encumbered vocoder core.** That framing credits the
real and considerable work done here without misstating the codec's origin — and
it is the framing that holds up if the project is ever examined.
