# blip25-mbe

A clean-room implementation of the MBE vocoder family: IMBE (P25 Phase 1),
AMBE / AMBE+ / AMBE+2 (DVSI AMBE-1000/2000/3000 generations), and parametric
rate conversion between them.

Built from TIA-102 specifications and expired DVSI patents. Validated by
black-box comparison against DVSI hardware and published test vectors.
No code or algorithms borrowed from existing open-source vocoder projects.

See [`DESIGN.md`](./DESIGN.md) for the architectural model.

## Layout

```
crates/blip25-mbe/        The published crate. Self-contained; no proprietary material.
conformance/
  vectors/                DVSI test-vector conformance harness.       (publish = false)
  chip/                   Live DVSI USB-3000 chip conformance.        (publish = false)
  speech-quality/         BABG-style PESQ/POLQA scoring.              (publish = false)
tools/
  dvsi-diff/              Iteration tool for diagnosing vector failures.
```

## Testing tiers

| Tier                 | Requirements                    | Who runs it                           |
|----------------------|---------------------------------|---------------------------------------|
| Unit tests           | None                            | Anyone, crates.io CI                  |
| Vector conformance   | DVSI test vectors on disk       | Project CI / developers with access   |
| Chip conformance     | DVSI USB-3000 hardware          | Project CI with chip attached         |
| Speech quality       | Reference speech + PESQ scorer  | Later                                 |

The published crate's `cargo test` is meaningful on its own. Conformance
against DVSI material is additional, optional, and never required for a
contributor to validate their work.

## On this being AI-designed

This project was designed and scaffolded by Claude Opus 4.6, via Claude
Code, in collaboration with a human domain expert. The fact is called out
here because "AI-written" is sometimes used as a reason to dismiss a
project without looking at it, and the process behind this one is worth
being honest about.

A single prompt of "build me an MBE vocoder in Rust" would have produced
a port of OP25, SDRTrunk, dsdcc, or JMBE. Those codebases are what
modern language models have been trained on, so they reproduce them. The
result would have inherited the twenty-year-old conflation of wire format
with codec that every one of those open implementations shares.

That is not what happened. What happened was iterative correction:

- Before any code, the model was directed to read specific TIA-102
  implementation specs and given a clean-room directive: no reading of
  existing open-source vocoders. Ambiguous specs were to be reported as
  gaps, not guessed around.
- When the model proposed a module called `transcode/`, the author
  corrected it to "rate conversion" — the term used in the DVSI AMBE-3000
  protocol spec — and pointed it at the relevant document.
- When the model treated the codec as a single thing, the author pointed
  out that AMBE-1000, AMBE-2000, and AMBE-3000 are three generations
  sharing one parameter model, and that DVSI's own P25 / P25A / P25X
  modes run different codec algorithms behind identical wire bits.
- When the model drifted toward IMBE-centric naming, the author
  redirected it to `p25_fullrate` and `p25_halfrate` as peers (with
  `dvsi_3000` alongside for the chip-protocol wire), reflecting that
  each is a wire over the same shared parameter model.
- The three-tier test architecture — self-contained unit tests, vector
  blackbox, live-chip blackbox — came from the author requiring that
  contributors never be blocked by lack of proprietary material.

The design document's central claim — three orthogonal axes (wire,
parameter, codec) joined at a common interchange type, with rate
conversion as a first-class peer rather than a decoder afterthought —
is what the specs say. The model would not have arrived there alone.
It arrived there because a domain expert re-read specs with it, rejected
plausible-sounding first answers, and insisted on the clean decomposition
that the specs actually describe.

That is the case for AI as a force multiplier rather than a shortcut.
Most practitioners never find time to rebuild a familiar problem from
first principles. Directed collaboration with an AI assistant makes it
possible in a weekend that would otherwise have taken a sabbatical. The
quality of the output is a direct function of the expert's willingness
to correct and direct; the AI amplifies what the expert already knows,
it does not substitute for it.

If you are an expert in something you care about but have never had the
bandwidth to rebuild from the ground up, you have a collaborator now who
reads as fast as you can think. Used that way, what comes out is not an
AI project. It is the project you finally had the time to build.

## License

MIT. See [`LICENSE`](./LICENSE).
