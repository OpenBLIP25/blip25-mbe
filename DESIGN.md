# Design

## The Model

A vocoder is commonly described as the thing that turns bits into speech.
This project rejects that framing.

What actually exists in a digital voice system is three independent
translations, each with its own domain, its own literature, and its own
evolutionary pressures:

1. A **wire format** — a bit layout, protected by FEC, that survives the
   radio channel. A standardized container. The only thing that cannot
   change without breaking interoperability.

2. A **parameter model** — a compact mathematical description of speech
   that rides inside that container. For the MBE family this is
   (ω₀, V/UV, M_l): pitch, voicing decisions per harmonic, spectral
   amplitudes. Stable since Griffin & Lim, 1988.

3. A **codec** — the analysis and synthesis algorithms that move between
   PCM audio and the parameter model. Free to evolve. Four generations
   have shipped behind the same parameter model and, in the P25 case,
   the same wire format.

Most open-source P25 implementations collapse these three layers into one
and name the result "IMBE." The conflation is not harmless. It is the
reason open-source speech quality has not meaningfully improved in twenty
years, and it is the reason parametric rate conversion — the one operation
that lives entirely inside layer 2 — is largely absent from open code.
Software is shaped by the mental model of the people who write it, and
you cannot build what you cannot see.

## The Axes

The three layers imply three independent axes. Every vocoder operation is
a choice on each:

```
      ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
      │  Wire format   │       │  Parameter     │       │  PCM audio     │
      │                │       │  model         │       │                │
      │  IMBE full     │←wire→ │                │←codec→│                │
      │  IMBE half     │ pack  │   MbeParams    │  ana/ │  8 kHz         │
      │  AMBE r0..r63  │       │                │  syn  │  16-bit        │
      └────────────────┘       └────────────────┘       └────────────────┘
                                        ▲
                                        │
                            rate conversion lives here.
                              no codec. no PCM. no synthesis.
```

Wire format is fixed by interoperability.
Codec generation is a quality choice.
Parameter model is the common ground that makes the other two independent.

## The Center

The center of gravity is the parameter model, not the wire and not the PCM.
This is the architectural commitment that makes rate conversion obvious:

```
   bits @ rate_A  →  parse  →  MbeParams  →  requantize  →  pack  →  bits @ rate_B
```

No synthesis. No re-analysis. No PCM. A single journey through parameter
space. DVSI calls this "parametric rate conversion" in their own
documentation. It is the project's headline capability, not an afterthought
bolted onto a decoder.

## The Implications for Code Layout

```
mbe_params/       — the interchange type. the center.
codecs/           — analysis/synthesis, one submodule per generation.
  mbe_baseline/   — TIA-102.BABA-A, 1993. Cannot pass BABG; kept for completeness.
  ambe/           — AMBE-1000, Generation 1.
  ambe_plus/      — AMBE-2000, Generation 2. US5701390 phase regeneration.
  ambe_plus2/     — AMBE-3000, Generation 3. US8595002, US8315860.
imbe7200/     — P25 Phase 1 FDMA: 144-bit IMBE wire (BABA-A §1–§12).
rate33/     — P25 Phase 2 TDMA: 72-bit AMBE+2 wire (BABA-A §13–§17 +
                    Annexes L–T; renamed "Half-Rate Vocoder" in the spec
                    to dodge DVSI's trademark, but the wire is AMBE+2).
dvsi_3000/        — DVSI AMBE-3000 chip protocol (r0..r63 rate configurations).
                    Distinct from any over-the-air protocol; rates 33/34
                    happen to align with P25 half-rate but the chip
                    framing differs from what the P25 modulator emits.
rate_conversion/  — parameter-domain bits-to-bits. peer of codecs/ and frames/.
fec/              — shared FEC primitives (Golay, Hamming, interleavers).
bits/             — shared bit-packing primitives.
```

Three concerns, three top-level axes, one interchange type.

**Wire layer naming policy.** These modules are named by DVSI *rate*
(`imbe7200` = full-rate IMBE at 7200 bps; `rate33` = half-rate AMBE+2,
DVSI rate index 33) because they model the **codec channel frame** —
the rate-defined Golay/Hamming/PN/interleave layer sitting directly on
the parameter bits, the same boundary DVSI's own chip packet draws.
Protocol-specific over-the-air framing (burst layout, scrambling,
sync) lives *above* mbe in each protocol's CAI/air-interface layer, not
here.

**DMR/NXDN reuse — resolved 2026-06-04 (was hedged).** The reuse claim
splits cleanly into two layers, and the seam is now drawn in code:

- **Codec FEC core — SHARED bit-for-bit.** The 49→72-bit AMBE+2 FEC —
  `[24,12]` extended Golay on `c₀`, `[23,12]` Golay on `c₁`, the
  `û₀`-seeded 24-value LCG PN scramble of `c₁`, uncoded `c₂`/`c₃`, and
  the `û₀..û₃` = `[12,12,11,14]` bit prioritization — is defined by the
  **DVSI AMBE+2 vocoder at the 3600/2450 operating point**, not by any
  protocol. P25 Phase 2, DMR, and NXDN (4800/"EHR") all carry exactly
  this codec channel frame. Corroboration: open-source mbelib/DSD
  (general-codec reference, not P25 IP) decode all three through one
  shared `mbe_processAmbe3600x2450Frame` entry point. `rate33` exposes
  this core as [`decode_code_vectors`] / [`decode_code_vectors_soft`]
  (4 code vectors → `Frame`), reusable as-is by a future DMR/NXDN module.
- **Interleave — P25-Phase-2-SPECIFIC, NOT shared.** The map from OTA
  dibits to the four code vectors (`rate33`'s **Annex S**, 72 bits ↔ 36
  dibits) is part of P25's air interface. DMR and NXDN each define their
  own distinct voice-bit interleave (and their own burst layout/embedded
  signalling), so `decode_frame` / `decode_frame_soft` — which prepend
  Annex S — are P25-only adapters. A `dmr_voice/` / `nxdn_voice/` sibling
  supplies its protocol's (soft-)deinterleave to land bits in
  `[u32;4]` / `SoftCodeVectors`, then calls the shared core above.

So the reuse point is the **post-deinterleave code-vector frame**, not
the OTA-dibit entry. The codec layer (`codecs/`) is shared across
protocols regardless. (This supersedes the earlier "wire formats are
strictly P25-protocol-specific" framing, which predated moving protocol
burst layout up into the CAI layer.)

**The deeper boundary is rate 34, not rate 33 — and encryption proves
it.** There are two boundaries here, at two layers, and the *codec*
boundary is the deeper one:

- **rate-34 frame** = the bare **49-bit AMBE+2 parameter frame**
  (`mbe_params` / a `Frame`), FEC stripped. This is the true codec
  interface — FEC-agnostic, protocol-agnostic, and the same for P25
  Phase 2, DMR, and NXDN.
- **rate-33 / code vectors** (`SoftCodeVectors`) = the rate-34 frame
  wrapped in *one specific* deterministic channel-coding profile
  (Golay/PN). It is the input to a particular FEC decoder, not the codec
  itself.

The FEC layer is just the deterministic map between them. Three things
follow, and they shape what belongs in mbe vs. above it:

1. **Encryption sits at the rate-34 boundary**, not below it. P25/DMR/NXDN
   encrypt the *parameter bits*, with FEC wrapping the ciphertext — so a
   keystream XOR has a home only on the 49-bit frame. A real radio (e.g.
   APX) therefore runs its vocoder at rate 34 and does FEC + encryption in
   the host; the codec/chip never holds a key. mbe must expose the
   49-bit `Frame` as a first-class seam so a consumer can inject the XOR
   between FEC-decode and synthesis.
2. **rate-33 FEC is a *profile*, not the codec.** Its Golay/PN is one
   deterministic channel code chosen for the half-rate use case; rate 34
   is the open substrate beneath it. A future protocol may wrap the same
   49-bit frame in a different code (LDPC, none, custom UEP) — which is
   why bit **prioritization** (`rate33::priority`) is exported as an
   FEC-independent fact.
3. **mbe keeps its own Golay/PN by design.** As a standalone,
   cross-language library it cannot depend on a consumer's FEC crate, and
   it needs the math to model DVSI chip rate-33 interop. The P25/DMR/NXDN
   CAI layer holds its *own* shared copy (one FEC — the voice-frame FEC is
   identical across the three — plus per-protocol interleave). The two
   copies are generic coding theory (not P25 IP) and stay bit-exact via a
   shared test vector. See the consumer's SPEC.md §3.4 for the full
   two-axis (protocol × backend) model and the recorder backend.

**Why `imbe7200` is not equivalent to "IMBE wire."** BABA-A is a
consolidation of two independent vocoder specs: the original 1998
BABA (full-rate IMBE) and the BABA-1 addendum (half-rate AMBE+2).
The two wires are governed by the same document but they are not the
same vocoder. P25 Phase 1 fire-channel deployments in particular
sometimes pair the `imbe7200` wire with the `codecs::ambe_plus2`
codec for SCBA-mask noise immunity — a valid combination because the
wire layer's only contract with the codec is `bits ↔ MbeParams`.

## What We Do Not Do

**We do not read open-source vocoder implementations.** The misconceptions
we are correcting are encoded in those codebases. Reading them to learn
algorithms risks reproducing the same collapsed model we are working to
escape. Our constants and algorithms come from:

- TIA-102 published specifications — public, normative.
- DVSI public documentation — USB-3000 manual, AMBE-3000 protocol spec.
- Expired DVSI patents — public-domain technical documentation.
- Black-box validation against DVSI test vectors and hardware.

When a spec is ambiguous or incomplete, we report the gap and get the
spec updated. We do not guess by looking at someone else's guess. An
incorrect idea repeated a hundred times is not less incorrect for being
familiar.

**We do not conflate correctness with quality.** Correctness is spec
conformance — every bit in the wire, every coefficient in the FEC,
every entry in the quantizer. Measured by unit tests against spec examples.
Quality is perceptual audio fidelity — measured by black-box comparison
against DVSI hardware and, later, BABG PESQ scoring. These are different
concerns and live in different places.

**We do not require proprietary material to build.** The published crate
contains no DVSI material, no reference vectors, no recorded hardware
output. Anyone can clone, `cargo test`, and see green. Conformance against
DVSI vectors and hardware lives in separate workspace members that are
never published to crates.io and that gracefully handle the absence of
the oracle.

## What We Seek

An implementation where the shape of the code reflects the shape of the
domain. Where wire, parameter, and codec are three separate concerns
because they are three separate concerns. Where rate conversion is a
short, clear function because it actually is a short, clear function.
Where a new MBE generation is a new directory, not a rewrite.

The specs have been published for thirty-five years. The patents have
expired. The test vectors are free to download. The only thing that has
been missing is a clean decomposition.
