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
  write a gap report** to the spec-author agent rather than guessing or
  referencing P25 third-party code. Strengthening the spec as we go is an
  explicit project goal: every gap you flag makes the spec more ready for the
  next implementer. See "Reporting Spec Gaps" below for the handoff.
- Do not copy algorithms, constants, or structure from P25 open-source
  implementations. Cross-checking a single numeric constant against a reference
  is acceptable as a last-resort sanity check, but the design and code must
  come from the spec.

### Clean-room boundary — you do NOT read copyrighted source material
- You are the **implementer** role in a two-agent clean-room setup. A separate
  **spec-author** agent reads the copyrighted TIA-102 PDFs and full-text
  extracts; you never do.
- Allowed reading under `~/blip25-specs/`:
  - `standards/*/P25_*_Implementation_Spec.md` — the derived implementation specs
  - `analysis/*.md` — the derived cross-document analysis
  - `standards/*/annex_tables/*.csv` — derived extraction tables
  - `specs.toml`, `README.md`, top-level docs
- **Forbidden** (these are the spec-author's inputs, not yours):
  - Any `*.pdf` under `~/blip25-specs/` or `~/blip25-mbe/DVSI/`
  - Any `*_Full_Text.md` or `*_Summary.txt` in `~/blip25-specs/standards/`
  - Any TIA PDF anywhere on disk
- Rationale: preserves the clean-room derivation chain so the code and derived
  specs in this repo remain free of copyrighted content. If you accidentally
  load one of the forbidden files, stop, discard what you read, and ask the
  spec-author for a derived-work answer instead.

### Committed code must derive only from derived works
- All committed code, constants, and comments must trace to the derived
  implementation specs — never to PDFs, full-text, or summary extracts (you
  can't read those anyway, but this rule also bars copy-paste via the
  spec-author agent).
- Quoting non-trivial passages from any TIA source into code, comments, or
  docs is a copyright issue. Keep such quotes out of the repo.

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
4. Why a chip/test-vector probe is or isn't a substitute (some gaps can be
   resolved by observing DVSI chip behavior; others need spec language)
5. Any diagnostic evidence from the current implementation (diverging test
   vector frames, magnitude of error, conditional stats, etc.)

### Handoff to the spec-author agent
1. Write the gap report as a new file under
   `~/blip25-specs/gap_reports/NNNN_<short_slug>.md` (NNNN = next sequential
   number; check the directory first).
2. SendMessage to the `spec-author` teammate pointing them at the file.
3. Mark the relevant task blocked in the team task list if it can't proceed
   without a spec answer, and pick up a different task.
4. The spec-author will draft a spec update. The **user reviews and merges**
   spec updates before you read them — do not act on an unreviewed draft.

Never answer your own gap by reading copyrighted source material. If the
spec-author is idle and you're stuck, escalate to the user instead.
