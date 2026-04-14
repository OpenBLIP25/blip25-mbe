# blip25-mbe — P25 MBE Coder (Rust)

## Goal
Emulate a DVSI vocoder chip (IMBE for P25 Phase 1; eventually AMBE+2 for Phase 2
where possible). The reference for correctness is the DVSI chip's output on known
test vectors, not any open-source implementation.

## Implementation Directives

### Spec-first, not source-first
- **Always work from the TIA-102 implementation specs in `~/blip25-specs/`**, not
  from open-source P25 / MBE implementations (OP25, SDRTrunk, dsdcc, JMBE,
  imbe_vocoder, or any other P25/IMBE/AMBE project).
- If a spec is ambiguous, incomplete, or missing required detail, **stop and
  report the gap** — the user will have it added to the spec rather than have
  you guess or reference P25 third-party code. Strengthening the spec as we go
  is an explicit project goal: every gap you flag makes the spec more ready
  for the next implementer.
- Do not copy algorithms, constants, or structure from P25 open-source
  implementations. Cross-checking a single numeric constant against a reference
  is acceptable as a last-resort sanity check, but the design and code must
  come from the spec.

### What IS allowed
- **General SDR open-source libraries** (DSP primitives, modulation, filter
  design, generic coding libraries) that are not P25-specific.
- **LLM knowledge of widely-established, textbook methods** — Golay codes,
  Hamming codes, Reed-Solomon, CRC polynomials, standard DSP transforms,
  linear congruential generators, etc. These are mathematical constructs,
  not project IP. Implementing them from first principles (generator
  polynomials, parity-check matrices, standard decoding algorithms) is
  expected and is not considered "guessing."

The restriction is on the **P25 domain specifically**. General signal
processing and coding theory are fair game. The stop-on-gap rule applies to
**any** spec ambiguity — don't scope it narrowly.

### DVSI emulation target
- The goal is bit-exact (or as close as the spec allows) equivalence with the
  DVSI chip's behavior on the same inputs.
- DVSI materials are available under `./DVSI/` (symlinked; gitignored):
  - `DVSI/Software/` — DVSI reference software, docs, USB3K tooling
  - `DVSI/Vectors/` — official test vectors (`tv-rc`, `tv-std`)
- Use the DVSI test vectors as the primary conformance test suite. If our
  implementation diverges from the vectors, our implementation is wrong
  (not the vectors).
- DVSI material is freely downloadable but **not redistributable** — keep it
  out of the repo (already gitignored).

## Primary Specs for This Project
See `~/.claude/CLAUDE.md` for the full spec index. The directly relevant ones
for MBE/IMBE work:

- **IMBE vocoder (frame format, bit ordering, FEC, tone generation):**
  `~/blip25-specs/standards/TIA-102.BABA-A/P25_Vocoder_Implementation_Spec.md`
- **IMBE wire format vs MBE codec (read this first):**
  `~/blip25-specs/analysis/vocoder_wire_vs_codec.md`
- **Missing vocoder specs (AMBE+2 / BABB / BABG context):**
  `~/blip25-specs/analysis/vocoder_missing_specs.md`
- **FDMA air interface (IMBE placement in LDUs, frame context):**
  `~/blip25-specs/standards/TIA-102.BAAA-B/P25_FDMA_Common_Air_Interface_Implementation_Spec.md`

## Reporting Spec Gaps
When a spec is insufficient, write a clear gap report with:
1. Which spec file and section
2. What you need to know (specific question, not vague)
3. What options exist and why they can't be disambiguated from the spec alone
4. Why third-party source inspection is not a substitute (e.g., correctness,
   licensing, or emulation fidelity to DVSI)

The user will then get the spec updated.
