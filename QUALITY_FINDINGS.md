# blip25-mbe — audio-quality findings & improvement plan

**Date:** 2026-06-01
**Source:** synthesized from the `ambe3000-clone` chip-parity session (live
AMBE-3000R A/B on `pve`) plus this repo's inherited notes. Written for a
*fresh session* that wants to raise blip25-mbe's perceptual quality.

## Framing — what this document is and is NOT about

The sister project `ambe3000-clone` chases **bit-exact PCM** parity with the
AMBE-3000R. **That is explicitly out of scope here.** For blip25-mbe, ignore:

- floating-point vs fixed-point arithmetic differences,
- exact per-sample PCM sequences,
- raw harmonic **phase** offsets in steady voiced segments — the ear is
  largely insensitive to absolute harmonic phase in sustained voicing, so
  the chip-vs-ours phase scatter we measured (−167°…+150°) does **not** by
  itself cost quality. (Phase still matters in three narrow places — onsets,
  the V/UV boundary, and inter-frame continuity — flagged below.)

What this document **is** about: the places where blip25-mbe's perceived
audio quality (intelligibility, naturalness, timbre, transient clarity) falls
short of — and could ultimately exceed — the AMBE-3000R, on both **encode**
(PCM → bits) and **decode** (bits → PCM).

**Goal.** First reach parity with the AMBE-3000R; then exceed it. The chip is
1990s-era fixed-point silicon with known design compromises (catalogued in
`~/blip25-specs/analysis/ambe3000_chip_oracle_caveats.md`). Modern, float,
offline-capable code can beat several of them — but today blip25-mbe is **not
yet at parity**, so parity comes first.

> **Which chip-vs-spec deviations are actually quality levers** (and which are
> fixed-point artifacts to ignore) is distilled in
> `DVSI_DEVIATIONS_QUALITY_LEVERS.md` — read it before acting on any
> "the chip does X differently" note, so precision quirks don't get mistaken
> for quality work.

---

## 0. Measure before optimizing — the attribution method

> **Why "great on vectors, not as great in the wild" (the origin symptom).**
> Verified on the chip (ambe3000-clone, 2026-06-04, `conformance/baselines/
> encode_robustness_2026-06-04/` + `encode_ns_test_2026-06-04/`): the AMBE-3000
> is a **faithful "dumb" codec — NO AGC, NO low-level mute on real speech, NO
> noise suppression, NO activity gating** (output RMS tracks input 1.03× over a
> 200× level range; added noise passes straight through at 0.94×). So the wild
> gap is **not** a missing front-end DSP stage we forgot to build. It is the
> **same steady-state encoder/decoder quality gap** measured here (amplitude
> estimator, pitch corr ~0.55, voicing, a persistent ~1.5–2 dB level deficit,
> decode phase/synth) — just **more exposed on diverse real speech than on the
> few favorable tuning vectors. The encoder is somewhat OVERFIT to the test
> vectors.** Practical consequence: validate every change on a *diverse real
> corpus*, not on `clean`/`dam`; the tuning vectors flatter us and will keep
> hiding the gap. (Conditioning like AGC lives in the radio audio path, not the
> vocoder — so it's out of scope for matching the chip, though see §3.4 for the
> NS *exceed* opportunity.)

Quality work fails when you can't tell whether a deficit is in the **encoder**
or the **decoder**. We already have the instrument: the 4-cell matrix in
`conformance/speech-quality` (`ab-matrix`, `halfrate-ab-matrix`). Use it with a
**clean-speech reference** and a real perceptual scorer (PESQ wideband/POLQA;
confirm the scorer in `conformance/speech-quality` actually runs — the README
lists it as "later").

Drive clean speech `S` through every combination and PESQ each against `S`:

| cell | encoder | decoder | isolates |
|------|---------|---------|----------|
| chip→chip | chip | chip | the bar to beat (AMBE-3000R end-to-end) |
| chip→ours | chip | **ours** | **decoder** quality (encoder held = chip) |
| ours→chip | **ours** | chip | **encoder** quality (decoder held = chip) |
| ours→ours | ours | ours | our end-to-end |

- `chip→ours` vs `chip→chip` ⇒ is our **decoder** ≥ the chip's?
- `ours→chip` vs `chip→chip` ⇒ is our **encoder** ≥ the chip's?
- A deficit in only one cell pinpoints the layer; a deficit in both is
  cumulative.

Run this on a diverse clean-speech corpus (male/female, several languages,
plus the DVSI `tv-rc/r33` `clean`, `dam*`, `fambf*`, `tambf*` vectors) **before**
touching code. Every change below should be validated as a PESQ delta in the
relevant cell, not by ear alone.

---

> **2026-06-11 re-measurement at the fork's round-9 promoted defaults**
> (`ambe3000-clone/conformance/baselines/pesq_hr_matrix_2026-06-11/`, rate 33,
> live chip, 6 vectors incl. t03/ucarm15a, P.862 NB, RMS-sanity-gated cells):
> chip_enc+chip_dec **2.573**, our encode isolated **−0.204**, our decode
> isolated **−0.263**, end-to-end **−0.189**. The chip still wins every
> vector; the remaining gap is ~0.2 MOS per side (edge of audibility —
> clean speech worst at −0.27..−0.37, degraded material nil). Notable:
> DECODE now contributes at least as much as encode — the voiced τ/phase
> + unvoiced fine-structure walls carry real PESQ mass (with the caveat
> that P.862 penalizes time-varying delay, which may over-weight τ vs
> human ears). The "exceed" headroom for blip25-mbe is therefore on BOTH
> sides, and §2.2 (don't replicate chip artifacts) remains the cheapest
> route past the chip.

## 0a. Measured attribution — 2026-06-01 (clean-speech corpus, AMBE+2 half-rate)

Scorer stood up and confirmed running: `conformance/scripts/score_ab_matrix.py`
in the `~/.blip25-eval` venv (`pesq` + `pystoi`). **Band note:** the codec is
8 kHz / 4-kHz-band, so **narrowband PESQ (P.862) is the apples-to-apples metric**
and is the attribution basis. PESQ *wideband* (P.862.2) is only defined at
16 kHz — `pesq(8000,…,'wb')` hard-errors — so a supplementary `pesq_wb` column
was added that resamples 8→16 kHz; it is optimistic (the empty 4-8 kHz band
matches the reference) and is **not** used for attribution.

Corpus: `tv-rc` `clean, dam, fambf22c, fambm22a, tambf22a, tambf32b`, 1500
frames (30 s) each. Chip on `pve` (live AMBE-3000R). The chip natively runs
AMBE+2, so the **half-rate** matrix is a genuine oracle — that is where the
encoder/decoder attribution is valid. (Full-rate IMBE is *not* chip-orackable;
see `reference_chip_is_ambe_plus_2_only`. Full-rate is reported as self-quality
only: our_enc→our_dec mean PESQ-nb **3.06** vs clean ref.)

Baselines: `conformance/baselines/halfrate_ab_clean_corpus_2026-06-01.csv`,
`fullrate_self_corpus_2026-06-01.csv`.

**4-cell PESQ-nb deficits vs chip→chip (the bar):**

| vector | chip→chip | decoder (chip→ours) | encoder (ours→chip) | e2e (ours→ours) |
|--------|-----------|---------------------|---------------------|-----------------|
| clean    | 3.047 | −0.374 | −0.187 | −0.261 |
| dam      | 3.369 | −0.298 | −0.182 | −0.265 |
| fambf22c | 3.145 | −0.478 | −0.462 | −0.542 |
| fambm22a | 2.971 | −0.350 | −0.216 | −0.347 |
| tambf22a | 2.951 | −0.425 | −0.191 | −0.354 |
| tambf32b | 3.015 | −0.355 | −0.116 | −0.212 |
| **mean** | **3.083** | **−0.380** | **−0.226** | **−0.330** |

**Attribution: the DECODER is the dominant speech-quality deficit
(mean −0.38 PESQ), the encoder secondary (−0.23).** The decoder is the larger
deficit on every vector. This confirms **§2.1 (decode spectral-envelope
accuracy)** as the highest-scoring first lever, consistent with the §1
per-harmonic envelope evidence and the inherited "speech gap is decoder-side"
note. Encoder §3.1 (amplitude estimation) is the clear second lever.

Method caveat logged: on the first chip half-rate *encode* call of the session
(the `clean` vector) the chip emitted a low-energy/garbled bitstream
(chip→chip 1.16, RMS 237 vs ref 1307); a re-run after the chip was warm gave
the healthy 3.047 above. Treat the **first** chip-encode call per session as a
warm-up and re-run it. All other vectors were clean on the first pass.

---

## 1. Evidence collected this session (rate 33/34, AMBE+2 half-rate)

Probe: `ambe3000-clone/conformance/speech-quality/examples/phase_probe.rs`
feeds an identical **no-FEC (rate 34)** info vector to both the chip and our
decoder, so any divergence is pure decode-path (dequantize → §1.10 enhancement
→ synthesis), with the wire/FEC layer removed.

- **Decode spectral envelope diverges per-harmonic.** On a sustained voiced
  frame (leakage-free pitch ω₀ = π/20), the chip-vs-ours harmonic magnitude
  ratio ranged **0.59×–1.78×** — not a constant gain, a *shape* difference.
  Per-harmonic magnitude = timbre, and **timbre dominates perceived quality.**
  This is the single strongest decode-side lead.
- **Overall loudness** differed (~0.84× on that probe) but loudness is a
  consumer-layer concern (`CHIP_GAIN_CALIBRATION.md`); the *envelope shape* is
  the quality issue, not the scalar.
- **Inherited open quality items** (`SCOPE_PLAN.md`): 2-frame onset attack
  smearing on speech onsets; tonal-content deficit on `alert.pcm`. Both are
  audible-quality, not bit-exactness.
- **Encoder, per inherited `analysis-encode` conformance:** pitch and voicing
  track the chip reasonably, but **spectral-amplitude RMSE is large**. That is
  a direct encoder timbre deficit.

> **Voicing: TWO chip-anchored encoder levers pinned (ambe3000-clone round 6,
> 2026-06-11, `encode_b1_gain_2026-06-10/b1{speech,mech,lad}_r6/`).**
> (1) **The spec Θ is too strict on loud confident-pitch frames** — the chip
> voices marginal bands (D_k up to ~2Θ) when frame loudness M(ξ) (Eq. 42) is
> high and pitch is confident (E_P < 0.5); M(ξ) separates chip-voiced-we-not
> misses at AUC 0.94. The fork promoted Θ ×= 1 + 1.25·clip(2(M(ξ)−0.5),0,1)
> on E_P<0.5 frames: pooled b1 idx +8.3pp, chip-voiced recall +23.9pp, costs
> only cpvbad/dam80 ≈−2.3pp (noisy speech). QUALITY lever for blip25-mbe:
> under-voicing loud voiced high bands renders them as noise — buzzy/breathy
> high end on exactly the frames that matter most perceptually; the same
> graded relaxation (gate `BLIP25_VUV_MXI_GRADE`) is a direct PESQ candidate.
> (2) **The chip ALSO devoices temporally-diffuse bands our spec rule cannot
> see** (per-band within-period envelope concentration "tloc"; phase-blindness
> proven — |S| identical, D_k flat, chip flips): noise-filled or
> phase-incoherent bands get devoiced high-band-first, graded, pitch-dependent
> (P54/L24 immune). Wired as `BLIP25_VUV_TCONC` (default OFF; speech-negative
> standalone — speech errors run the OTHER way — but correct on degraded
> content). For blip25-mbe: a tloc-style FA-side gate is the principled
> "don't voice noise" complement when tuning voicing for quality.
> ⚠ Composition caveat (fork round 8, 2026-06-11): composing tloc WITH the
> graded relaxation (1) was measured chip-side and is strictly negative on
> speech at every threshold (chip-voiced speech high bands' tloc overlaps
> the noise regime — no separating threshold exists), while tone-side
> devoicing composes cleanly. If used as a quality gate, expect the same
> trade: it suppresses noise-band voicing at the cost of devoicing real
> loud high-band voicing; gate it on content class, not tloc alone.
>
> (3) **The chip ramps voicing EXTENT at voiced-run onsets/offsets**
> (fork round 9, 2026-06-11, `encode_b1_gain_2026-06-10/b1feat_r9/`):
> row 15 (`11100000`) at 82% of 400 run onsets, voiced-band count ramping
> 3.05→5.02→6.49 over the first frames, mirrored at offsets; loud bursts
> get a full 16→15→…→16 envelope. A two-sided onset rule keyed on M(ξ)
> (sparse onset rows → 15/16; quiet onsets suppressed) is now the fork's
> promoted default (+4.91pp pooled b1 parity; decode roundtrip LSD to the
> chip's own audio IMPROVES at onset frames). QUALITY lever for
> blip25-mbe: instant full-band voicing at onsets is a spec-faithful but
> chip-divergent behavior — a 1–2-frame voicing-extent ramp at onsets is
> a plausible attack-smoothness/PESQ candidate (the chip, presumably
> tuned by listening, chose it). Also from round 9: tloc-style speech
> devoicing essentially does not exist in the chip (the tone devoice law
> is content-gated, like the gain L-law) — don't port tloc to speech;
> and the strongest NEW speech voicing feature is lag-P band-envelope
> correlation ("epcorr", C-FA AUC 0.84, D_k-orthogonal) — a candidate
> "is this band really periodic" quality gate, untested in blip25-mbe.

Caveat: the probe input is synthetic flat-amplitude voicing; **reproduce the
envelope-divergence measurement on real chip-encoded speech** (decode the same
`tv-rc/r33/clean.bit` through chip vs ours, compare per-frame harmonic
envelopes) before drawing strong conclusions.

> **Resolved 2026-06-01b** (see §2.1 "Progress …b" and
> `ambe3000-clone/docs/lsd_attribution_1a.md`): done on `clean/dam/fambf22c/tambf22a`.
> The synthetic probe's "0.59×–1.78× per-harmonic ratio" **overstates** the
> real-speech *envelope-shape* divergence — phase-/gain-invariant shape-LSD shows
> only ~3.6 dB true divergence, of which ≤0.10 dB is a correctable smooth envelope.
> The rest is per-harmonic scatter (phase/placement signature).

---

## 2. Decode-side quality levers (bits → PCM), prioritized

### 2.1 (HIGH) Spectral-envelope accuracy: dequantize + log-mag prediction + §1.10
The probe shows our decoded harmonic envelope differs in *shape* from the chip
on identical bits. Suspects, in order:

1. **Log-magnitude prediction recurrence.** On a steady signal the predictor
   should converge to a stable envelope; if our recurrence (coefficients,
   the `M̄_l` feedback) differs from the chip's, the whole envelope drifts
   frame-over-frame even on identical bits. Audit `ambe_plus2_wire::dequantize`
   `apply_log_prediction` against the derived spec; verify steady-state
   convergence by feeding a constant info vector and checking `M̄_l` is flat.
2. **§1.10 spectral enhancement** (`enhance_spectral_amplitudes`). This is the
   formant-sharpening stage; getting its weighting wrong tilts timbre. Compare
   pre- and post-enhancement envelopes against the chip's output envelope.
3. **PRBA/HOC dequantization** (`decode_prba_vector`, `inverse_block_dct`,
   `assemble_hoc_matrix`). A bug here mis-shapes the residual spectrum.

**Measure:** per-frame log-spectral distortion (LSD, dB) of our envelope vs the
chip's output envelope on real speech bits; target a steady decrease. **Steps:**
isolate each stage with a unit probe (constant info vector in → inspect
intermediate `M̄_l`), fix divergence, re-run `chip→ours` PESQ.

**Progress 2026-06-01 (this session):**
- **Where it diverges (localization).** Per-band output LSD of `chip→ours` vs
  `chip→chip` on identical chip bits (`/tmp/lsd_probe.py`, 6-vector corpus):
  total ~8.7–9.4 dB, **rising with frequency** — 0–1 kHz ≈ 7.0–8.1 dB,
  1–2 kHz ≈ 7.8–8.6, **2–3 kHz ≈ 9.0–9.8, 3–4 kHz ≈ 8.8–9.9**. The decode
  envelope diverges most in the **upper bands** (mid/high harmonics = brightness
  / formant detail). (Output-domain LSD is inflated by random-phase unvoiced
  synthesis, so use it for *localization*, not absolute envelope error.)
- **Suspect #1 (log-prediction recurrence) — CLEARED.** New probe
  `conformance/.../examples/halfrate_envelope_converge.rs` feeds a fixed steady
  voiced info vector through `apply_log_prediction` with one advancing
  `DecoderState`. The envelope converges monotonically; the frame-to-frame
  delta decays by exactly the **0.65 predictor coefficient** per frame
  (15.2 → 7.6 → 3.8 → 1.9 → … dB; <0.01 dB by frame 16 ≈ 320 ms). That settling
  is the predictor's spec-defined memory, shared by the chip; on continuous
  (always-warm) speech it is **not** the divergence source. Init transient
  (`prev_lambda=1.0`, `prev_l=15`) only matters for the first ~16 frames of a
  cold stream — negligible for the 1500-frame PESQ runs.
- **Next (SUPERSEDED by Progress 2026-06-01b below).** This bullet originally
  proposed a post-§1.10 `M̄_l` envelope probe to split §1.10 vs HOC. That probe
  was built and run (phase-/gain-invariant shape-LSD on the bit-exact oracle) —
  **§1.10 is cleared and the whole magnitude-envelope-shape lever is near-
  exhausted (≤0.10 dB ceiling).** Do not pursue §1.10/HOC envelope *tilt*
  tuning; the residual is per-harmonic micro-scatter, not a smooth envelope.
  See 2026-06-01b.

**Progress 2026-06-01b (real-speech shape-LSD attribution + adversarial verify):**
Built that probe in `ambe3000-clone` (`examples/lsd_attribute.rs`) — but with a
**phase-invariant, gain-invariant shape-LSD** (per-frame mean removed over voiced
harmonics, joint-LS harmonic extraction) and a measurement-floor control, on the
bit-exact chip oracle (`clean/dam/fambf22c/tambf22a`, ~5k voiced frames). A
10-agent adversarial workflow independently re-derived every number. Full writeup:
`ambe3000-clone/docs/lsd_attribution_1a.md`. Verdicts that bear on this lever:
- **Suspect #2 (§1.10) — CLEARED.** Enhancement is shape-neutral: component
  shape-LSD `E−D` ≈ 0.53 dB, R²≈0.11 with the chip mismatch. Not the source.
- **Suspect #3 + the prediction (the dequant stage, `D−chip`) is the dominant
  TRUE carrier** (shape-LSD ≈5.2–5.6 dB, R²≈0.50 — ~3× any other stage). The
  log-prediction is *cleared for steady-state convergence* (prior note) but on
  real speech it **undercorrects the inter-frame tilt by ~37%** (residual tilt
  positively coupled to the pre-prediction tilt, k≈0.37).
- **DECISIVE: the magnitude-envelope *shape* lever is near-exhausted.** An
  *ideal* per-decile EQ — the ceiling of any envelope/tilt/§1.10/predictor-
  coefficient fix — recovers only **0.04–0.10 dB** of shape-LSD; the smooth
  decile envelope explains just **1.5–3.5%** of the variance. ~97% of the true
  ~3.6 dB per-harmonic divergence is **per-harmonic scatter**, not a smooth
  envelope. The prior per-band ~8.7–9.4 dB output-LSD was (as flagged)
  random-phase-unvoiced + units inflated; the true voiced-envelope-shape error
  is ~3.6 dB and almost none of it is a correctable timbre tilt.
- **Implication for the −0.38 decoder PESQ deficit (quality framing).** It is
  **not** a smooth-envelope/timbre-*tilt* error (≤0.10 dB recoverable). The
  ~3.6 dB residual is **per-harmonic magnitude micro-scatter** — for *perceptual
  quality* that is the candidate lever (harmonic fine-structure / placement),
  **plus the three narrow phase cases this doc flags (onsets, V/UV boundary,
  inter-frame continuity)**; steady-voicing absolute phase stays inaudible per
  the §Framing note, so it is *not* the quality lever here. (For the sister
  fork's **bit-exact parity** goal the same residual *is* phase — its
  `phase_divergence` tool measures −2.4 dB voiced SNR, ~71° per-harmonic phase
  RMS — but that is a parity concern, not a blip25-mbe quality one. *Update
  2026-06-01:* a 40-pitch sweep there found the voiced phase error splits into a
  deterministic linear-in-l term **ΔΘ(ω₀)=66.87·ω₀−1.38** (a ~66.9-sample
  time-origin offset + a per-harmonic ψ ramp; affine in ω₀ alone) whose exact
  per-pitch removal lifts voiced SNR −2.25→+0.95 dB, leaving a ~55° per-harmonic
  residual. Still inaudible in steady voicing, so still not a quality lever — but
  the time-origin / fundamental-phase finding could matter for the *onset* and
  *inter-frame-continuity* phase cases this doc flags as quality levers; worth a
  listen if those are revisited. *Update 2026-06-02:* the parity fork found the
  AMBE-3000R **re-initializes its voiced-phase synthesizer to cold-start at every
  voiced frame following an unvoiced frame** (chip probe; post-unvoiced onset ==
  cold start to 2.3°, independent of the unvoiced segment's pitch/amplitude). This
  is directly the **onset / V-UV-boundary phase case** this doc flags: a
  deterministic phase RESET at voicing onsets, not a carried accumulator. For
  blip25-mbe quality it means onset phase coherence is a *defined* target (re-seed
  phase at voicing onsets) — plausibly audible as onset crispness — worth A/B'ing
  if the onset case is revisited; gated as `BLIP25_PHASE_UV_RESET` in the fork's
  `mbe_baseline.rs`. The dominant *steady*-run residual is a within-run time-shift
  τ that resists a closed form (likely fixed-point) — still inaudible, still
  parity-only. *Update 2026-06-02b (envelope→phase + V/UV boundary):* a dedicated
  chip envelope-sweep (`ambe3000-clone/conformance/baselines/envsweep_2026-06-02/`)
  found the AMBE-3000R's envelope→phase response **is minimum-phase** (Hilbert of
  log-envelope; cepstral corr 0.95, scale ~0.85) **applied to ALL harmonics** —
  whereas our regen exempts `l≤L̃/4` and uses γ=0.44 (a low scale). Matching the
  chip = drop the exemption + raise the scale (`BLIP25_REGEN_ALL`+`REGEN_SCALE≈1.2`).
  Inaudible in steady voicing (parity-only), BUT the chip also injects a **large
  phase perturbation at voiced harmonics adjacent to a VOICING BOUNDARY when it
  coincides with a formant** (synthetic formant all-voiced vs mixed-voiced: l near
  the boundary swings 90–170°). That is squarely the **V/UV-boundary phase case**
  this doc flags as a *quality* lever — worth A/B'ing for boundary crispness if the
  V/UV case is revisited. The bulk real-frame "content surface" is this static,
  envelope×voicing-dependent boundary phase, not a simple min-phase operator.) The only
  structural envelope handle left is the predictor
  coefficient — and it was **swept and confirmed near-dead**: the chip-matching
  optimum of the §2.13 `0.65` coefficient is ≈**0.68** (the ~37% undercorrection),
  improving apples-to-apples `pcm_lsd` uniformly across all four vectors by only
  **~0.07–0.09 dB** with no regression (`ambe3000-clone/conformance/baselines/r33_predcoef_sweep_2026-06-01.csv`).
  ~0.08 dB is the whole win — at the ≤0.10 dB ceiling — so it was not applied
  (also shared with the analysis-encoder roundtrip). Decode quality is **phase-
  limited, not envelope-limited.**

**Progress 2026-06-02 (ambe3000-clone magdyn chip campaign — CORRECTS the two
notes above).** A constant-pitch magnitude-step chip campaign
(`ambe3000-clone/conformance/baselines/magdyn_phase_2026-06-02/`) resolves the
real-frame "content surface":
- The chip's voiced phase is **MEMORYLESS in magnitude** — after a spectral step
  it converges to the cold-destination phase in ~6–8 frames, no persistent offset.
  So the content surface is **not** a voicing-boundary/dynamics term.
- The content surface **IS the min-phase Hilbert operator** on the instantaneous
  envelope (chip-validated corr 0.96 vs our `ambe_phase_regen` on the chip's
  extracted M̄_l), applied to **ALL** harmonics — the chip does NOT use the
  `l≤⌊L̃/4⌋` regen exemption (REGEN_ALL). This **overturns** the "not a simple
  min-phase operator" note above.
- **The per-harmonic envelope micro-scatter (the ≤0.10 dB residual) is the SYNTH
  TRANSFER, not dequant/§1.10.** On cold probes (chip & ours decode the SAME
  codeword) the realized envelope ripples 46% (chip) vs 29% (ours); dequant is
  faithful (±0.17 log2) and §1.10 is a near no-op (ripple 0.29 enh vs 0.31 no-enh).
  The divergence is the chip's fixed-point synthesis reconstruction (window/DFT-256/
  OLA), partly a fixed H(ν). For perceptual quality this is the **envelope-shape +
  phase-coherence** lever (via the min-phase operator it couples magnitude and
  phase): matching the synth reconstruction would sharpen formant detail AND
  improve the onset/boundary phase coherence this doc flags. For bit-exact parity
  it is the unified decode target (the varying-pitch ψ/τ is separate).

**Progress 2026-06-03 (ambe3000-clone — §1.10 is WRONG-SIGNED vs the chip on real
speech; dequant wordlength is chip-observable ~Q5).** Sharpening the earlier
"§1.10 near no-op" note with the phase-invariant shape-LSD-vs-cached-chip-PCM
instrument on the 4 real-speech vectors
(`ambe3000-clone/conformance/baselines/enh_chip_wordlength_2026-06-03/`):
- **§1.10 makes the chip-envelope match WORSE, consistently.** `lsd_E−lsd_D = +0.09 dB`
  on all 4 files; per-frame the spec §1.10 hurts 72–75% of voiced frames (median
  +0.10 dB), strengthening to 77–82% on high-energy well-voiced frames. Relaxing
  the Eq.108 clamps to sharpen MORE degrades monotonically; the floor is at
  enhancement OFF. A pure linear-in-l tilt null-test matches/beats the structured
  clamp ⇒ the residual past "off" is a mild high-frequency ROLLOFF (the voiced
  synth-H(f)), not the §1.10 law. End-to-end `BLIP25_NO_ENHANCE` improves
  `pcm_lsd` (chip-PCM vs our-PCM) on all 4 files (~−0.05 dB).
- **Quality implication (the "exceed" framing).** The chip applies essentially NO
  spectral-amplitude sharpening on the half-rate path — its envelope is not
  formant-sharpened the way spec §1.10 prescribes. For *chip parity* that means
  turn §1.10 off. For *blip25-mbe quality* the inverse: the chip's lack of §1.10
  sharpening is plausibly a chip **limitation**, so spec §1.10 (or a better-tuned
  enhancement) stays a candidate "exceed" lever — do NOT remove it here on the
  strength of the parity result; A/B §1.10 sharpening for formant clarity instead.
- **Dequant log-mag wordlength is chip-observable at ~Q5** (≈0.19 dB log2 steps),
  via the same instrument over a fixed-point dequant gate — composes additively
  with §1.10-off (parity-only; envelope resolution, not a quality lever, but it
  bounds how fine the chip's magnitude representation actually is).

**Progress 2026-06-03b (ambe3000-clone — voiced phase-regen RE-TUNED + the
"min-phase operator explains ~half the voiced scatter" claim CORRECTED as
~⅔ overfitting).** Re-tuned the US5701390 regen against the 4 cached r33 vectors
(`phase_operator.rs` + `phase_divergence`, no live chip):
- **The chip's *effective* regen scale is BELOW the patent's γ=0.44, not above
  it.** Operator-fit on the chip's own phase gives s≈0.33–0.48 (nominal s=1.0 ⇔
  γ=0.44) ⇒ effective γ≈0.15–0.21, robust on all 4 vectors. End-to-end
  `phase_divergence` agrees: scale 0.4 **Pareto-dominates** the patent γ on all 4
  vectors on both the τ-removed per-harmonic phase residual (avg 71.65°→70.80°)
  and aligned voiced SNR (0.54→0.57 dB). **This OVERTURNS the 2026-06-01/-06-02
  note above that recommended `REGEN_SCALE≈1.2` + `REGEN_ALL`.** Wired
  `REGEN_SCALE_DEFAULT=0.4` as the AMBE+2 voiced default (regression test added);
  `REGEN_ALL` is a measured **NEGATIVE** (raises the residual) and stays OFF.
- **The "operator explains ~half the voiced scatter" (corr 0.96 / 53%) was ~⅔
  per-frame OVERFITTING.** A scrambled operator (same per-frame value
  distribution, harmonic alignment destroyed by reversal/roll) STILL scores
  31–36% under the free per-frame (s,τ,c) fit. The genuine reachable operator
  headroom is only ~8° (47°→39°), and most of *that* lives in the per-frame
  DC/φ₀ subspace the synth can't exploit without a free per-frame fundamental
  phase — hence turning the operator from OFF to full patent strength moves the
  end-to-end synth residual only ~0.3°. **§1.10-enhanced M̄_l as the operator
  input makes zero difference** (39.1°→39.2°), consistent with §1.10 being
  wrong-signed for this chip.
- **Net:** decode voiced quality is **NOT** regen-limited. The dominant reachable
  voiced lever is the **per-frame fundamental phase φ₀ + the within-run time-shift
  τ** (the content-dependent onset surface; `voiced-tau-is-onset-perrun-scalar`,
  `onset-formant-tau-content-not-minphase`), which the τ-only ceiling caps at
  ~+0.5 dB aligned SNR / ~70° residual. The regen retune is a small parity
  refinement, not a quality unlock.

**Progress 2026-06-03c (ambe3000-clone — voiced fundamental phase φ₀ pinned to a
chip-calibrated pitch law; first positive realized voiced-phase lever since
regen).** New tool `phase_phi0.rs` (per voiced frame, splits the chip-vs-our
phase error into a time-shift τ slope and an l-independent DC = φ₀; difference
and absolute fits; 4 cached r33 vectors, no live chip):
- **φ₀ IS content-determined** (scramble controls ≈0): a clean monotone
  **pitch-linear** component, `φ₀(ω₀) ≈ +22.6·ω₀ − 2.19 rad` (circular-linear
  r≈0.46 to ω₀), PLUS a genuine **per-run memory** (prev-φ₀ circ-circ r≈0.55 raw,
  0.46 after removing the ω₀ law — not a pitch-smoothness confound). It is **not**
  a free-running fundamental-phase accumulator (Δφ₀ vs ω₀·160 r≈0).
- **Realized** via a new `BLIP25_PHASE_CW` handle (φ₀=C+CW·ω₀ on the voiced
  harmonics; constants `PHI0_CW_VALIDATED=−22.6`, `PHI0_C_VALIDATED=2.19`,
  applied with the synth's internal sign, regression-guarded). Cross-validated
  (fit clean+dam / test famb+tamb — same optimum ridge). End-to-end
  `phase_divergence`: **beats the τ-only ceiling** — mean oracle-τ-aligned voiced
  SNR **+0.57→+1.09 dB**, per-harmonic resid **70.8°→68.8°**, raw SNR
  −2.41→−2.21 (never regresses). Kept env-gated/off by default (like N0/B).
- **Caveat / lever status:** the full φ₀ prize (≈12° per-harmonic given oracle τ)
  is **entangled with τ** — the DC helps raw bit-exact parity only modestly
  (+0.2 dB) because most of the gain is conditional on a correct per-frame τ, and
  the φ₀ memory is the same firmware-bound onset surface as τ. So φ₀ is a real
  *partial* quality lever (use the `BLIP25_PHASE_CW`/`_C` law if you want the
  voiced phase closer to the chip), but the voiced wall is still the τ + φ₀-memory
  onset content surface. Campaign:
  `conformance/baselines/voiced_phi0_2026-06-03/`.

### 2.2 (HIGH, the "exceed" win) Do NOT replicate the chip's quality-degrading artifacts
The chip applies beyond-spec behaviors that *hurt* quality
(`ambe3000_chip_oracle_caveats.md`): an amplitude-dependent **mute**, a hard
**±10000 internal cap**, a **stationary-signal smoother**, and a
**spectral-discontinuity attenuator** (transient ~27% RMS drop on large ΔL).
blip25-mbe's spec-faithful default should **not** copy these — by *not*
clamping/attenuating, our decoder can sound cleaner than the chip on dynamic
speech. (These belong only behind the opt-in `chip_compat` flag for emulation,
never on by default.) **Action:** confirm none of these are on by default;
A/B `chip→ours` vs `chip→chip` on dynamic/transient speech to claim the win.

### 2.3 (MED) Error-robustness: adaptive smoothing / repeat / mute (§1.11)
Under FEC errors (real RF traffic), repeat/mute decisions drive intelligibility.
The chip's recovery is a 1990s heuristic. **Exceed opportunity:** use the
soft-decision FEC metrics we already expose to do *graceful* spectral
interpolation across bad frames instead of hard repeat/mute. **Measure:** PESQ
vs clean ref on the `dam_e*` (error-injected) DVSI vectors, `chip→ours` vs
`chip→chip`.

### 2.4 (MED) Onset / transient clarity
The noted 2-frame onset attack smears consonants and word starts — a real
intelligibility cost. This is partly the synthesis window/overlap-add and the
voiced-onset phase/energy ramp. **Exceed opportunity:** shorter effective
attack via asymmetric/adaptive windowing at detected onsets. **Measure:** PESQ
+ STOI on plosive-rich speech; informal A/B on word onsets.

### 2.5 (LOW–MED) Voiced/unvoiced mix and tonal content
Unvoiced-band noise synthesis quality and the voiced/unvoiced crossover affect
"naturalness" and breathiness. Tonal inputs (`alert`, DTMF, Knox) have a noted
deficit; the half-rate Annex-T tone path bypasses MBE and should be exercised.
Phase only matters here at the boundary and for inter-frame continuity (avoid
clicks) — not as absolute offset.

**Unvoiced inter-band leak (2026-06-02, ambe3000-clone `dequant_probe_2026-06-02`).**
The spec's §1.12.1 per-frame DFT-256→band-shape→IDFT→OLA unvoiced synthesis has a
fixed **−17.9 dB inter-band leak**: each unvoiced peak band smears energy into its
neighbours (the per-frame ~160-sample effective window has ~−15 dB sinc sidelobes
at one-band-spacing). Measured against the AMBE-3000R with direct-index codewords
(chip and ours fed identical bits): the **chip leaks ~0** — its unvoiced spectrum
has near-perfect band isolation (shoulder ≈ M̃), so the chip renders sharper /
less-smeared unvoiced bands than our synthesis. Our flat→flat response is fine
(±0.3 dB); the leak only shows on peaky spectra. This is both a parity gap and a
**quality lever**: a sharper unvoiced spectrum (less band-to-band smear) is more
faithful to the input envelope. Reaching it needs **continuous per-band
bandpass-filtered noise** (a long, cross-frame-coherent effective window), not the
per-frame DFT-OLA — no window/taper/OLA knob reduces the −17.9 dB (all
−17.4…−18.3). NB: the half-rate DEQUANTIZER (gain/PRBA/HOC/DCT/§2.13 predictor) is
bit-faithful to the chip — the unvoiced envelope error is synthesis, not dequant.

**CORRECTION (2026-06-02, ambe3000-clone `wola_validate_2026-06-02`).** The
"continuous per-band bandpass-filtered noise / long cross-frame-coherent window"
fix above is **overturned at the waveform level.** It was implemented two ways
(long-window overlap-add WOLA + overlap-save OLS, both gated `BLIP25_UV_SYNTH`)
and measured by sample-xcorr vs the chip: the original per-frame 256-DFT-OLA
**best matches the chip's unvoiced waveform** — real-speech interior-unvoiced
xcorr old 0.842 > WOLA 0.689 > OLS 0.352; stationary flat probe old 0.967 > WOLA
0.907. WOLA *does* cut the −17.9 dB per-band POWER leak (notch −14.7→−27.6 dB)
but its long window blends overlapping noise blocks → smears the time-varying
envelope + noise phase → better power spectrum, WORSE waveform. So the inter-band
leak is a stationary-power-spectrum artifact, **NOT the perceptual/waveform
bottleneck** — don't chase it with a longer window. The real unvoiced lever is
the 256-path fine structure: the exact band-edge bin assignment (round-to-nearest
beats ceil on real speech, +0.4 dB SNR, pitch-dependent) and fixed-point
DFT/shaping precision. (Mechanism B: on extreme envelopes our §1.10 enhancement
hits the W_l upper clamp and boosts shoulders ×1.2 — a separate envelope lever
shared with voiced magnitude.)

**Voiced/unvoiced RMS ratio diverges by pitch (2026-06-03, ambe3000-clone
`vuv_codebook_recover_2026-06-03`).** Sweeping the V/UV pattern over a fixed base
frame (chip vs ours, same bits): as bands flip voiced, the **chip's output RMS
rises to a STABLE ~1.08×** the all-unvoiced level — pitch-independent (1.077 /
1.138 / 1.077 at f0≈249/207/120 Hz). **Our decoder's voiced RMS swings with
pitch**: 0.56–0.68× on sparse-harmonic frames (low/mid f0), ~1.01× on dense
(high-harmonic) frames. The base-dependence is the fingerprint of **phase
interference** in our voiced sinusoidal synthesis (few harmonics → coherent-phase
constructive/destructive swings dominate RMS; many harmonics → averages out),
whereas the chip's stable ratio implies its voiced harmonic phases are arranged
(or magnitudes normalized) to keep voiced level pitch-independent. ⇒ on low-pitch
voiced speech our output is audibly **too quiet** (down to ~0.6×), a perceptual
level error that is a direct manifestation of the voiced-phase/synthesis wall
(ambe3000-clone `voiced-phase-is-timeshift-tau` / onset-operator). Concrete chip
target: voiced/unvoiced RMS ≈ 1.08, flat across pitch. Not a separable scaling
constant — fixing it needs the chip's voiced phase/synthesis.

### 2.6 Handoff (APX firmware) deviations — quality relevance (2026-06-06)

A reverse-engineering hand-off surfaced bit-exact codebooks extracted from the
**Motorola APX `R39.15.00` DSP image** (DVSI *software* IMBE/AMBE+2 on a
TMS320C55x — a different target from the AMBE-3000R hardware chip). Most of its
deviations are **bucket B** (sub-LSB fixed-point artifacts → do NOT copy into the
float path, no quality value). Two are genuinely quality/interop relevant and are
**not currently reflected here or, likely, in open MBE decoders (mbelib/JMBE)**:

- **(FULL-RATE IMBE) DVSI extends the overall-gain ladder ("MONO65"), not plain
  Annex-E.** Decode-reachable part: the **low 8 levels are shifted DOWN** by
  `−0.25·(8−k)` (bottom level −4.842 dB vs Annex-E's −2.842). A decoder using the
  standard 64-level Annex-E renders DVSI/APX-encoded *quiet* frames (gain index
  0..7) **too loud by up to ~2 dB** — an audible level/interop error on low-energy
  speech. (The advertised "65th level" at +9.9199 is NOT decode-reachable — `b̂₂`
  is a 6-bit field, 0..63 — so it is an *encoder gain-search clamp ceiling*, not a
  decode entry; relevant only to encoder gain clamping, see §3.1.) **Action for
  blip25-mbe full-rate:** A/B Annex-E vs the APX low-8 downshift on quiet
  full-rate frames; the downshift is the likely-correct DVSI behaviour. Half-rate
  (Annex-O) gain is **un-extended** — no change there.

- **(ENCODE) DVSI's analysis window is a ~250-pt Hann-class window** (α≈0.501,
  reaches a true 0 at the edge) that differs from **all three** TIA windows
  (Annex B/C/I). This is an independent encode spectral-estimation lever — see
  §3.1/§3.2. Pairs with the ambe3000-clone finding that our refinement window is
  **mis-centred ~45 samples** (clips the left tail) — a likely real blip25-mbe
  analysis bug worth fixing regardless of the DVSI window shape.

- **Explicitly NOT quality levers (bucket B, confirmed):** the APX HOC books'
  truncate-toward-zero / ~2045 build scale (≤1 LSB Q11). An ambe3000-clone chip
  A/B (2026-06-06, b5 sweep) measured the entire HOC table effect at 0.07 dB
  band-power LSD — **~3× below the chip's 0.19 dB observability floor** and a
  15/17 coin-flip on which table the chip prefers. The same applies to the
  fixed-point dequant wordlength (~Q5) and the unvoiced fixed-point FFT residual:
  parity-only, no perceptual headroom.

---

## 3. Encode-side quality levers (PCM → bits), prioritized

The encoder ceiling caps everything — no decoder recovers detail the analyzer
threw away. Validate every change as a PESQ delta in the `ours→chip` cell.

### 3.1 (HIGH) Spectral-amplitude estimation
The known large amplitude RMSE vs chip is a direct timbre deficit at the
source. Audit `codecs::mbe_baseline::analysis::amplitude` and the
`M̃_l → Λ̃ → quantize` path. **Exceed opportunity:** the chip estimates
amplitudes with limited windows/precision; a modern multi-resolution or
reassigned-spectrum estimate can place harmonic energy more accurately. The
*encoder is where modern DSP most cleanly beats 1990s silicon.*

> **OTA forensics — Miranda / VE-PG4 IDAS field report (2026-06-21).** An
> external integrator running blip25 v0.2.0 over a real Icom IDAS repeater
> back-traced the on-air "robotic" defect to the encoder, independent of our
> chip work, and it converges with the §0a −0.23 encoder attribution. Method:
> same source PCM through blip25 v0.2.0 vs the VE-PG4 (real DVSI silicon),
> compared at the AMBE_d field level on 422 matched frame pairs, plus an
> mbelib loopback A/B (both bitstreams decoded by the *same* mbelib, so the
> delta is encoder-only). **First she decisively ruled out the bridge:** our
> `R34_BIT_ORDER` is byte-identical to her de-interleave table, `vu_flip`
> bit-10 is correct (0.00% even-slot disagreement), and a full single-bit +
> C(49,2) 2-bit-pair flip search finds no transform improving OTA fit by
> ≥0.5%. **The gap is parameter-trajectory jumpiness at quantize time:**
>
> | field | blip25 lag-1 autocorr | PG4 lag-1 autocorr | blip25 mean \|Δ\| | PG4 mean \|Δ\| |
> |---|---|---|---|---|
> | B0 gain  | 0.31 | 0.40 | 4.76 | 3.58 |
> | L4 (HF mag) | **0.05** | 0.24 | 20.7 | 18.4 |
> | L5 (HF mag) | **0.04** | 0.14 | 10.9 | 9.5 |
> | V (voicing) | **0.087** | 0.205 | — | — |
> | UV transitions | — | — | **3.5× too frequent** | baseline |
>
> mbelib-decoded frame-RMS deviates **13.7 dB** from the source envelope
> (PG4: 6.9 dB). Upper-band L4/L5 are essentially a random walk in our
> output where PG4 keeps real frame-to-frame coherence. This is the
> encode-side twin of §2.1/§1.10 upper-band decode work — both point at the
> HF spectral tail. **Caveat before "just add smoothing":** naive temporal
> smoothing has repeatedly tested net-negative on PESQ here
> (trusted-A_M EMA reverted 2026-05-14; §0.5 amp-EMA flag is default-off
> because it regressed speech), and her own transport-side pitch-pair fix
> (forcing odd-slot pitch = preceding even slot) was **null/slightly worse on
> the on-air A/B** (+1.8–3.4% pitch error, not better) despite a clean +0.15
> –0.21 PESQ offline — the real radio's pitch concealment doesn't match the
> "copy prev even" sim. So treat hysteresis on B0/L4/L5/V/UV as candidate
> levers to **PESQ/STOI-validate on a diverse corpus**, not as known wins;
> the highest-value, lowest-risk target is restoring L4/L5 frame coherence
> (autocorr 0.05→~0.2) without flattening genuine HF transitions. Corpus +
> her scripts/CSVs: `Miranda/` and the share folder
> `P25-IQ-Samples/blip25-pitch-pair-AB-2026-06-20/`.
>
> **CORRECTION (2026-06-21, same session — supersedes the per-field table
> above).** The per-field numbers above are an ARTIFACT of her field map.
> Her `pg4_vs_blip25_comprehensive.py` defines fields by *flat-slicing* the
> 49-bit prioritized info vector (`B0 gain = natural_bits[9:14]`, `L4 =
> [38:44]`, …). But that vector is the PRIORITIZED `û₀‖û₁‖û₂‖û₃` order; the
> real `b̂₀..b̂₈` fields are recovered by `deprioritize()` — a non-contiguous
> bit *gather*, NOT a flat slice. So her "gain"/"L4"/"L5" are scrambled
> priority-rank bit groups, not the actual parameters. Reproduced her exact
> flat-slice numbers (gain ac1 0.38/|Δ| 4.24 ≈ her 0.31/4.76; L4 ac1 0.008 ≈
> her 0.05), then recomputed on TRUE deprioritized fields — deprioritizing
> PG4's natural-order canonical too (validated: pitch shows high ac1, HOC
> near-zero — physically correct field separation; tool
> `examples/halfrate_field_dump.rs {fec9|natural7}`). Corrected per-field
> comparison, PG4 vs blip25 **promoted production stack**, 422 matched pairs:
>
> | true field | PG4 ac1 | blip25 ac1 | reading |
> |---|---|---|---|
> | b̂₀ pitch | 0.633 | 0.546 | blip25 modestly LESS coherent — the one real gap |
> | b̂₁ voicing | 0.666 | 0.703 | blip25 MORE coherent (vuv_mxi promoted helps) |
> | b̂₂ gain | 0.441 | 0.554 | blip25 MORE coherent — NOT jumpy |
> | b̂₃/b̂₄ prba | 0.387/0.258 | 0.402/0.290 | ≈ equal |
> | b̂₅–b̂₈ hoc | ~0.0 | ~0.0 | near-memoryless for BOTH — codec-inherent, not a defect |
>
> So: gain isn't jumpy, the HF tail is memoryless for the silicon too, and
> voicing is fine on the production stack. Frame-to-frame VALUE mismatch is
> high across all fields (gain 96%), so the real encoder gap is
> **parameter-estimation accuracy** (§3.1 amplitude / pitch) + the §2.1
> decoder deficit — **not trajectory smoothness**. This explains the
> campaign result below. The only genuine trajectory item is a small pitch
> coherence gap (b̂₀ 0.546 vs 0.633).
>
> **Smoothing campaign result (2026-06-21).** Three quantize-time hysteresis
> levers were implemented behind default-off `AnalysisState`/`DecoderState`
> setters + `halfrate-ab-matrix` flags and swept on Miranda's clip + 4 tv-rc
> vectors (our_enc→our_dec PESQ-nb):
> - `--hf-amp-ema-alpha` (band-selective L4/L5 amp EMA): NET-NEGATIVE on all
>   speech (mean −0.06…−0.34); rejected.
> - `--vuv-stickiness`: slightly negative (mean −0.01…−0.03); rejected.
> - `--gain-smooth-beta`: β=0.1 is offline-NEUTRAL (mean −0.001 PESQ, +0.031
>   on clean) and provably reduces gain |Δ|, but gain wasn't the problem, so
>   it's not a quality win — it just over-smooths an already-coherent field.
>
> Verdict: NO default change. Smoothing toward "PG4's trajectory" does not
> improve offline quality and the per-field premise was a measurement
> artifact. Levers retained as opt-in/diagnostic (and for an on-air A/B of
> the trajectory-coherence hypothesis, which only Miranda can run). The real
> levers remain §3.1 (encoder estimation accuracy) and §2.1 (decoder).

> **Chip-parity decomposition (ambe3000-clone, 2026-06-04).** Differential
> r34 measurement vs the chip's own bits localizes this deficit precisely.
> The voicing-agreed realized-envelope SHAPE divergence (~3.2 dB) is **uniform
> per-harmonic scatter** — NOT a systematic spectral tilt (linear tilt explains
> only 18%), NOT a fixed pre-emphasis curve (1.2%), NOT concentrated in
> low-amplitude valleys (flat ~3 dB even on harmonics within 6 dB of the frame
> peak). Reconstructing our CONTINUOUS pre-VQ PRBA vector shows the chip's
> chosen quantization cell sits at mean distance-rank 41/512 and 3.18× the cell
> radius away — i.e. a **genuine continuous-estimator divergence**, not a
> quantization cell-crossing artifact (so finer fixed-point precision will not
> close it). It is also uniform across stationary vs transient frames, ruling
> out analysis-window position/alignment. ⇒ the deficit is a fixed
> per-harmonic estimation-RULE difference; the within-frame suspects are the
> energy-INTEGRATION/band-edge rule and DFT/window resolution. For chip parity,
> measure candidate estimator changes by the PRBA distance-RANK (does the chip's
> cell move toward rank 1?), not the post-VQ envelope SHAPE, which overstates
> the true divergence ~3× via VQ granularity. (A modern *better-than-chip*
> estimate is still the §3.1 exceed play; this note is about matching the chip.)
>
> **Refinement (ambe3000-clone, 2026-06-04, forced-pitch isolation).** The
> "uniform stationary vs transient" claim above held only *before* pitch was
> neutralized. Forcing the chip's exact quantized `(ω̂₀, L̂)` into the amplitude
> estimator (so the harmonic grid is placed identically) drops the PRBA cell
> distance-rank only 41→35 and the d-ratio 3.18→2.99 — **not toward 1** — so a
> genuine fixed amplitude-RULE divergence *survives correct pitch* (pitch
> placement was ~40% of the apparent gap; a real ~rank-35 residual remains). And
> with pitch removed a **stationary-vs-transient split emerges that was masked
> before**: stationary (|ΔL|≤1) rank 27.5 vs transient (|ΔL|≥3) rank 49.5 — the
> residual roughly doubles on spectral-transition frames, pointing at
> analysis-**window length/position/response on transients** (partially reversing
> the "rules out window position" line above). A single-bin peak-pick variant
> (vs the spec's band-energy integration) is a clean negative (rank 35→40). So
> two levers: a core fixed-rule divergence on all frames (stationary floor
> rank 27.5, likely the chip's fixed-point spectral-magnitude arithmetic) plus a
> transient-specific window-response excess (the more reachable of the two for
> a *better-than-chip* encoder — attack/transition amplitude tracking, ties to
> §2.4 onset clarity).
>
> **Window-alignment lever found (ambe3000-clone, 2026-06-04).** Chasing that
> transient excess located a concrete, likely-shared bug: the analysis amplitude
> window (`extract_refinement_window`, ±110 Annex C) is centered at the frame
> midpoint in a lookahead buffer that keeps **no past frame**, so its left 30
> samples are zero-filled — an asymmetric, mistimed window. Shifting the window
> center +45 samples (gated `BLIP25_WIN_OFFSET`) improves the chip-bit match on
> **two** vectors (clean +2.5 pp, alert +2.9 pp emitted bit-match) and collapses
> the voiced PRBA amplitude divergence on clean speech (forced-pitch cell rank
> 35→14). For chip *parity* this is a frame-phase alignment to emulate; for
> blip25-mbe *quality* the takeaway is that the windowing currently clips its
> left tail and is mistimed on transients — worth auditing/fixing in the float
> path (retain a past frame so the window centers symmetrically), independent of
> the chip. Likely helps attack/transition timbre (§2.4). Empirical offset, not
> yet a derived constant.
>
> **Resolved (ambe3000-clone, 2026-06-04).** Wired a past-frame retain
> (`BLIP25_WIN_PAST`) to test the "center symmetrically" idea above against the
> chip directly: **un-clipping at the same instant does essentially nothing**
> (forced-pitch PRBA rank 35.1→34.0, bit-match 70.7→70.6). The chip's win is a
> genuine **~+48-sample forward shift** of the amplitude analysis instant, broad
> flat basin +44…+50, now validated across **four** natural-speech vectors
> (clean/alert/mark/dam, +1.9…+3.2 pp). Per-field it is specifically an
> **amplitude-window** correction — gain/PRBA/HOC all improve (gain exact-match
> clean 50→59, mark 21→32; this is the −1.5 dB level gap), while **pitch is
> flat-to-slightly-worse** (it is set by the separate §0.3 LPF tracker, not this
> window). Quality guidance: the asymmetric **left-clip is still a real float
> bug** to fix on its own merits, but do **not** blindly copy +48 — that is a
> chip-parity frame-phase convention (Bucket A), not necessarily a quality win;
> the chip's late amplitude instant may be a compromise. For *exceeding* the
> chip, the cleaner play is a properly-centered, possibly multi-resolution
> amplitude window evaluated by PESQ on transient/attack frames (§2.4), not
> a fixed +48 shift.

> **Window LENGTH/shape resolved — NEGATIVE (ambe3000-clone, 2026-06-04,
> `encode_win_length_2026-06-04`).** Chased the remaining "transient-specific
> window-response excess" from the note above. With the +48 position pinned and
> pitch forced, transient frames still sit ~2.2× the stationary PRBA cell rank
> (clean 21.5 vs 9.8) — but **window length/shape does not close it.** Gated a
> self-consistent `effective_window` (narrowing via Hann half-width, or flattening
> toward rectangular = more freq resolution): **both directions monotonically
> worsen the chip match**, and a shorter window worsens transients *faster*. So
> the chip's amplitude window is essentially Annex-C-shaped at ±110 — it is NOT
> shorter on transients, NOT flatter/higher-resolution. For chip parity the
> window family (position ✓, band-edge ✗, peak-vs-energy ✗, length/shape ✗) is
> exhausted; the residual is the chip's fixed-point spectral magnitude (firmware,
> Bucket C). **Quality takeaway: do NOT chase amplitude-window length/shape for
> blip25-mbe either — Annex C ±110 is at the matching optimum in both directions.**
> The *better-than-chip* play stays a properly-centered window + PESQ-judged
> multi-resolution at onsets (§2.4); a uniform length/taper change is a dead end.
> (The transient excess is also partly content-specific — mark's transient/
> stationary ratio is only 1.2 vs clean's 2.2.)

> **"Extra processing step" suspects RULED OUT (ambe3000-clone, 2026-06-06,
> `encode_amp_extrastep_2026-06-06`).** Final adversarial check before accepting
> the residual as firmware: is the ~3.2 dB amplitude SHAPE caused by an extra
> input-side step *we* run that the chip skips (the pattern that won §3.2 pitch)?
> Tested the three candidates via PRBA distance-rank (forced chip pitch): **all
> negative.** (1) Boll **spectral subtraction is default-ON in the ambe3000-clone
> `AnalysisState` but a complete NO-OP on natural studio speech** — byte-identical
> chip-cell rank on vs off across 1460 frames, because its strict spectral-
> stationarity + low-energy gate only fires on near-silent frames (learns a ≈zero
> noise floor). Note for **blip25-mbe**: the chip runs no NS on real speech
> (§3.4), so input-side spectral subtraction is at best inert here and should not
> be relied on as a quality lever on clean/voiced material. (2) An amplitude EMA
> smoother makes the chip match ~3× worse (cross-frame magnitude smoothing is
> wrong). (3) §0.5 is already raw Eq.43/44 with no extra step (band-edge/peak/
> coherent-projection variants all negative). ⇒ Unlike pitch, the amplitude
> deficit is **not a removable extra step** — it is the fixed per-harmonic
> estimation rule / fixed-point spectral arithmetic. For *exceeding* the chip the
> lever is therefore a genuinely better estimator (multi-resolution / reassigned
> spectrum, §3.1 top), not toggling a preprocessing stage.

> **Denoiser SELF-PRIMING defect pinned (ambe3000-clone, 2026-06-10, defect
> (h)(ii), `encode_b1_gain_2026-06-10/probes3/highfam2`).** The same default-ON
> Boll spectral subtraction that is a no-op on natural speech is **actively
> harmful on sustained constant-level tonal/stationary content**: the noise
> estimator's permissive energy gate (`E_f < 4·η`, with η slow-releasing up to
> the running frame energy in ~29 frames) plus the SCALE-INVARIANT cos-sim
> stationarity test primes the noise model **on the voiced signal itself**.
> Static tones then get a flat ≈−0.5 dB γ̃ suppression, and **alternating
> spectral content at constant level gets a sustained −2.0..−2.5 dB sag**
> (the model holds the stale previous-content PSD; the α-floor takes
> cross-content bins down −17 dB). Chip-pure reads show the chip does none of
> this. Gate `BLIP25_SPECSUB=0` (analysis/mod.rs; the fork DEFAULT is now OFF) removes the sag exactly
> (content offsets then match the chip to ≤0.15 dB); speech output is
> bit-identical on/off (t03, ucarm15a, 480 fr). Quality lever for blip25-mbe:
> the denoiser's eligibility gate needs a voicing/absolute-level guard (or
> default-off for chip parity) — on hum/alarm/tone-like field audio it eats
> real signal.
>
> **Same root cause also pins defect (h)(i) LEVEL OVERSHOOT (2026-06-10,
> highfam2 AM probes).** Because the cos-sim gate is scale-invariant, the
> self-primed noise PSD sits at the signal's *average* power; under amplitude
> modulation the trough frames fall below `β·n_psd` and the Boll gain slams
> to the `√α ≈ −17 dB` floor — γ̃-vs-input-RMS slope expands 1.06 (3 dB AM)
> → 1.28 (2 Hz/8 dB AM) with trough frames squashed up to −12 dB (worst
> single frames), while the chip stays 0.96–0.99 (faithful memoryless level
> reader). `BLIP25_SPECSUB=0` (now the fork default) restores slope to 0.91–0.99 ≈ chip on all 9
> AM/cadence/step/stair/cswitch probes. The binary ~0.6 %/frame pitch-delta
> +0.45 dB jump (defect (h)(iii)) is the same cos≥0.999 threshold crossing
> (denoiser drops out under pitch motion); denoise-off flattens the
> stationarity ladder spread 0.67→0.14 dB (chip 0.19). Perceptual
> implication for blip25-mbe: the denoiser COMPRESSES syllable/AM dynamics
> on tonal content (squashes pulse troughs), an artifact class PESQ on the
> 5-vector set never saw.

> **LEVEL deficit RESOLVED + REALIZED (ambe3000-clone, 2026-06-10,
> `encode_b1_gain_2026-06-10/`).** The persistent encode **−1.4 dB level
> deficit** (the "~1.5–2 dB" in §0) is now decomposed and closed for chip
> parity, and the old "energy-dependent" characterization is **REFUTED**
> (clean-vector contamination; the bias is FLAT in energy on 12/14 vectors).
> Two separable parts: (1) the §0.5 band-edge integration rule — fractional
> boundary-bin coverage weighting (`BLIP25_BIN_EDGE=frac`) is the only rule
> that holds the gain bias L-flat; (2) a remaining FLAT ≈+0.9 dB broadband
> scale (constant normalization difference, exact chip mechanism unpinned).
> With both promoted (ambe3000-clone defaults; `ceil`/`0` recover spec) the
> gain-field median bias goes −1.0 → −0.03 dB pooled, b2 exact 49.6→55.1%,
> and the **end-to-end roundtrip RMS ratio goes 0.874 → 0.993** on clean —
> the level deficit is gone. SHAPE scatter (~3.2 dB per-harmonic) remains
> firmware-adjudicated and is NOT addressed by this. **Quality consequence
> for blip25-mbe**: reproducing decoded speech ~1 dB quiet is a direct
> loudness/level-fidelity loss; port the frac edge rule + flat scale (or
> equivalently fix the Eq.43/44 boundary-bin energy loss) and re-PESQ.
> Bucket A for the edge rule; the +0.9 dB constant is now PINNED as the
> chip's measured flat normalization (same-day live-chip tone ladder:
> chip−ours = +0.900 dB exactly on P=21..70, level-flat over 11 dB; γ law
> confirmed mean(Λ)+½log2 L — Parseval/window/L-cap mechanisms all refuted).
> Safe to port as a constant. The remaining +1.0 dB on the breathy/female
> "HIGH-family" vectors is a separate content-dependent estimator effect
> (the §3.1 SHAPE work-stream), not a level law.
>
> **Chip gain ARTIFACTS — do NOT replicate for quality** (rounds 2-3,
> 2026-06-10, `probes2/`+`probes3/`): the chip's encoder gain has measured
> quirks that are quality DEFECTS, not levers — (1) a content-gated high-L
> deduction (−1.08 dB/octave above a hard L=41 step on dense-harmonic
> tones; absent on real speech); (2) a single-codeword −0.277 dB dip at
> pitch index b0=17; (3) early devoicing of noisy tonal content (from
> −20 dB valley floors). blip25-mbe should keep its level law flat.
> ~~A round-2 claim that the chip drops gain 0.5 dB under pitch
> modulation~~ is **RETRACTED** — see the next note: that was OUR encoder.
>
> **OUR encoder γ (gain) DYNAMICS DEFECTS — HIGH-value quality levers**
> (round-3 reattribution, 2026-06-10, `probes3/` — chip-pure instrument,
> three independent datasets): the chip is a faithful near-memoryless
> level reader (γ̃-vs-RMS slope 0.96–0.99 under AM, level steps, content
> switches, vibrato, ±8% sweeps). OUR encoder is NOT — four pinned
> defects, all reproducible locally with the probes3 inputs, no chip
> needed:
> 1. **Level-overshoot**: our γ tracks input level with slope 1.04–1.30,
>    growing with AM depth (chip ≈1.0) — pumping on modulated speech.
> 2. **Content-context sag**: −0.9..−2.6 dB γ̃ sag on alternating spectral
>    content at constant level — a stateless encoder should not do this;
>    cause unknown (window/pitch-tracker/denoiser interplay suspects).
> 3. **Binary pitch-motion jump**: sustained per-frame pitch delta above
>    ~0.6%/frame (threshold bracketed (0.565,0.677]%/fr) flips our γ̃ up
>    +0.43..+0.49 dB as a mode — inflected/vibrato speech reads louder.
> 4. **Sweep×level coupling**: under deep pitch sweeps near L≈41 our γ̃
>    over-tracks the level envelope (slope 1.13, 1.6 dB V-residual).
> These match the "worse in the wild" symptom shape (γ mis-tracking on
> dynamic, content-varying speech) far better than any front-end story —
> root-causing them is now arguably the top encode quality work item,
> ahead of the SHAPE estimator. Lesson recorded: never read chip behavior
> through a chip−ours delta alone; use chip-absolute (γ̃ − input RMS).
>
> **Round-10 chip probes (ambe3000-clone, 2026-06-11) — three quality-relevant
> facts.** (1) **Silence frames**: the chip parks the whole envelope SHAPE near
> zero on silent analysis windows (window RMS < ~3 dB re 1 LSB) — b̂₃ lands in
> the near-zero PRBA cells {86,87,90,91} ~40% of the time, b̂₄ likewise — while
> gain and V/UV track content normally. Our estimator was quantizing the
> noise-floor spectrum shape into content cells, i.e. spending shape bits
> rendering spurious timbre on silence. Adopting the same "attenuate the pre-VQ
> shape residual to zero below a silence threshold" rule is a cheap, chip-
> validated cleanliness lever for blip25-mbe's encoder too (in the clone:
> +1.12pp b3 exact-match, +15.8pp on the silence-heavy dam80 vector, audio
> guard clean). (2) **Pitch smoothing**: the chip applies a short FIR-like
> (~1-frame memory, NO step bypass) smoother to its pitch estimate — ±1%
> alternating-frame pitch jitter is attenuated ×0.53, while steps settle in 2
> frames and stationary input stays exact. A 2-tap FIR (w≈0.2–0.3) on the
> continuous pitch estimate is a quality-relevant stability lever — BUT only
> after the estimator itself is unbiased: in the clone the FIR is blocked by a
> +0.3-sample estimator bias at P≈28–36 (high-pitch voices), where smoothing a
> biased track adds cell-crossing errors. Fix estimator bias first, then
> smooth. (3) **Classifier-fragility lesson (cautionary)**: a voicing-
> suppression rule conditioned on a spectral statistic (our M(ξ)<0.5 onset
> suppress branch) full-file MUTED sustained flat-envelope voiced content —
> an unbounded failure invisible on every speech battery (speech cost of the
> fix: 0.00pp). The chip itself has NO such conditioning. Any
> classifier-driven suppression in blip25-mbe must be hard-bounded in duration
> (the clone now caps at 2 frames per onset).

### 3.2 (HIGH) Pitch estimation robustness
Pitch doubling/halving is among the most audible MBE failures (octave jumps,
roughness). The encoder uses a pYIN-style tracker; harden against
octave errors and rapid pitch transitions. **Exceed opportunity:** a modern
pitch tracker (CREPE-class, or robust autocorrelation+HMM smoothing) can beat
the chip's tracker — keep a classical fallback for determinism. **Measure:**
gross-pitch-error rate vs a reference tracker on voiced speech; then PESQ.

> **Empirical correction (2026-06-04, ambe3000-clone
> `encode_pitch_win_2026-06-04`).** The "encode pitch corr ~0.55, ≥40% of the
> gap" framing is a *pooling artifact*. Restricting the chip-vs-§0.3 comparison to
> genuinely-voiced frames (chip voiced-fraction ≥ 0.5 via `expand_vuv`): ω₀
> correlation = **0.99** (clean & mark), median error **~1.5 %** (0.022 oct), and
> **zero octave errors** — the §0.3 tracker is *already chip-grade on voiced
> speech*. The apparent 0.55 corr came entirely from unvoiced/silence frames where
> the chip's pitch is a perceptual don't-care (it drives b0 to the table floor,
> ω₀≈0.05/L=56, for noise-shaping resolution; parity-only, Bucket C). So pitch is
> **not** a high-priority quality lever for matching the chip — §0.3 already tracks
> voiced pitch tightly. The residual is a ±1 quantization cloud (bit-parity, not
> audible) and the unvoiced floor-b0 convention (parity-only). The CREPE-class
> *exceed* opportunity still stands for robustness on hard field audio, but
> "octave-error hardening to reach chip parity" is largely already met. Pitch-
> tracker *window timing* is correct (a +48-sample shift, the §0.5 amplitude
> analog, is a clean NEGATIVE for §0.3 — it only hurts).
>
> **§0.4 refinement walks pitch AWAY from the chip (2026-06-06, ambe3000-clone
> `encode_pitch_refine_2026-06-06`).** Quantizing the raw §0.3 estimate `P̂_I`
> directly (no §0.4 E_R refinement) matches the chip's `b0` *better* than the
> refined pitch on all 3 voiced vectors: exact +11pp (clean), +6 (mark), +16
> (dam); |Δb0|≤1 and the |Δb0|≥3 tail both improve; no PRBA/bit-match regression.
> The E_R argmin (±9/8 quarter-sample, P̂_I excluded) is *displaced* from the
> chip's pitch — even with offset 0 admissible it rarely picks it. **Mechanism
> that matters for blip25-mbe:** `refine_pitch`'s E_R DFT reuses the
> amplitude analysis window, so the mis-centred / +48-shifted window (see §3.1
> window note) *also degrades pitch refinement* (clean b0 exact 34.9→27.8 going
> WIN0→WIN48). Two takeaways: (1) for chip parity, emit pitch from the §0.3
> estimate, not the E_R-refined ω̂₀; (2) regardless of the chip, our E_R pitch
> refinement on a left-clipped / mis-aligned window is a **likely real
> blip25-mbe analysis bug** — fixing the window centring should be done *before*
> trusting §0.4 refinement. Gate `BLIP25_PITCH_REFINE=off`.
>
> **Validated on a 4th fresh live-chip vector + decode cross-check (2026-06-06,
> ambe3000-clone `encode_amp_extrastep_2026-06-06`).** `t01.pcm` (a TIA natural-
> speech vector outside the tuning set), fresh cold-session chip r34 encode:
> OFF b0-exact 51.9% vs SPEC 39.3% on 206 voiced frames (+12.6pp), refinement
> hurts 35% / helps 15%. Decode roundtrip (our_enc→our_dec) off vs spec shows NO
> perceptual regression (phase-invariant LSD vs input 14.69 vs 14.64 dB, level
> ratio 0.992). So `BLIP25_PITCH_REFINE=off` is safe to carry as a quality
> default, but the *principled* fix remains correcting the window centring so
> refinement operates on an aligned spectrum.
>
> **High-pitch tracking trap (ambe3000-clone, 2026-06-07, `pitch_probe_2026-06-07`).**
> A synthetic clean-tone probe (known periods 22–120, fed to the chip and our
> encoder) exposed a real §0.3 robustness bug masked by the tuning vectors: for
> short periods **P < 34 (F0 > ~235 Hz)** our tracker traps at a long harmonic
> (period-22 tone → period ~95) while the chip tracks it perfectly (period err
> 0.285 samples). `ep_minima` diagnosis: look-ahead finds the correct short period
> but `decide_initial_pitch`'s `ce_b ≤ 0.48` absolute gate takes the trapped
> look-back (4th-harmonic) pick; and forcing best-CE / look-ahead does NOT escape
> it in steady state, so the trap is also in the E(P) landscape + look-ahead
> failing to escape on sustained high-pitch tones. Natural-speech parity payoff is
> low (speech is mostly P ≥ 40), but this **mistracks high-pitch (female/child)
> voices** — a genuine quality/robustness fix for blip25-mbe: rework the look-back
> continuity window + cold-start P̂=100 trap and the look-ahead octave escape so
> sustained high-F0 input isn't pulled to a sub-harmonic. The chip is the proof it
> is fixable. Also note: on clean tones P ≥ 34 our §0.3 already matches the chip
> ~80% within a ±1 fixed-point cloud — that residual is the chip's unreadable
> fixed-point quantization (chip-parity firmware wall, not a quality lever).
>
> **REALIZED & VALIDATED (2026-06-07, ambe3000-clone
> `encode_pitch_escape_2026-06-07`).** Fixed via a targeted sub-harmonic escape in
> `decide_initial_pitch`: override the look-back pick **only** when `P̂_B / P̂_F` is
> within ±0.12 of an integer `N ≥ 2` (look-back is an octave/harmonic of
> look-ahead) AND `CE_F < CE_B` (look-ahead is genuinely better). On normal voiced
> speech `P̂_B ≈ P̂_F` (ratio ≈ 1) so it never fires. **Results:** synthetic
> clean-tone probe vs the live chip — mean |period err| 7.42 → 0.51 samples
> (chip 0.285), chip==our 64 % → 74 %; decode roundtrip pitch error on P<34 input
> **14.2 → 0.0 samples** (the audible win — a ~360 Hz tone was rendered 2+ octaves
> low, now exact). **Zero regression** on clean/mark/dam/t01 + every DVSI female
> vector (engages on 0 frames there; adult-female F0 stays in-grid), and on the
> one chip-checkable natural frame (t01 #479, ratio 2.02) the chip AGREES with the
> shorter pick. Earlier "decision gates don't fix it" was a stale example binary.
> Default-ON (strict bug fix); `BLIP25_PITCH_DECIDE=noescape` for A/B. General DSP
> (classic octave guard), not P25 IP. The P≥34 ±1 fixed-point cloud is untouched
> (firmware wall, as before).
>
> **REALIZED — sub-sample pitch refinement (2026-06-07, ambe3000-clone
> `encode_pitch_subsample_2026-06-07`).** The §0.3 tracker snapped its final pitch
> to a 0.5-sample grid (`PITCH_GRID_STEP`) with NO sub-sample step, but the chip's
> clean-tone period accuracy (0.285 samples) is finer than that grid. Added a
> **parabolic interpolation of the E(P) minimum** (vertex of the 3 grid neighbours)
> — classic sub-sample autocorrelation-peak refinement, general DSP. Validated vs
> the live chip on 10 r34 vectors: chip-voiced `b0`-exact **+1.4 pp pooled, never
> negative per-vector**, held-out **t02 +8.1 pp / tambf32b +3.4 pp**, with
> neutral/better mean|Δb0| and gross-tail on 9/10. A *dense* sub-grid argmin
> (`fine`) is NET NEUTRAL — it chases spurious E(P) dips; the 3-point curvature fit
> is the robust form. This is a genuine pitch-accuracy quality win (finer ω̂₀ → more
> accurate harmonic placement), not just bit-parity. ambe3000-clone promoted it as
> the half-rate default (`DEFAULT_HALFRATE_SUBSAMPLE = Parabolic`,
> `BLIP25_PITCH_SUBSAMPLE=off` recovers grid); **carry the same parabolic E(P)
> refinement into blip25-mbe's §0.3 for both rates.** NB whole-frame encode parity
> is unmoved (amplitude firmware wall dominates) — this is a pitch-field/quality
> lever, not a route to bit-exact frames.

### 3.3 (HIGH) Voicing (V/UV) decision
Wrong voicing → buzzy (false-voiced) or noisy (false-unvoiced) artifacts,
very audible. Audit `analysis::vuv` band decisions against the chip on borderline
(breathy / fricative) frames. **Measure:** per-band V/UV agreement vs chip on
labelled speech; PESQ on breathy/fricative-heavy clips.

> **OTA corroboration (Miranda 2026-06-21, see §3.1 note).** Independent of
> the chip, the IDAS field comparison shows blip25 *flips voicing too often*:
> V lag-1 autocorr 0.087 vs PG4 0.205, and UV transitions 3.5× more frequent
> than PG4. This is a TEMPORAL (sticky-decision) deficit distinct from the
> high-band UNDER-voicing rule below — our voicing both leans wrong on highs
> *and* chatters frame-to-frame. A sticky/hysteresis V/UV step is a candidate
> lever, PESQ-validate it (smoothing has historically regressed speech here).

> **Chip-parity finding + REALIZED lever (ambe3000-clone, 2026-06-10,
> `encode_b1_gain_2026-06-10/`).** Measured on 14 cached + 2 fresh live-chip
> vectors: the spec encoder systematically **UNDER-VOICES HIGH BANDS** vs the
> chip — 7.4:1 under- vs over-voicing on pitch-agreed voiced frames, rising
> monotonically with band (slot0 2.9% → slot7 30.5% chip-voices-we-not). Not
> pitch-, energy- or transient-driven. The cause is Eq. 37's pitch/band Θ
> rolloff (coefficient 0.3096): the chip behaves as if it is absent. Dropping
> it (`BLIP25_VUV_PITCH_COEF=0.0`, now the ambe3000-clone default; `=spec`
> recovers) lifted chip-voiced-frame b1 agreement +6.7pp pooled and +4.4/+2.3pp
> on fresh held-out live-chip vectors, with zero effect on pitch or spectral
> shape. **Quality consequence for blip25-mbe**: the spec rolloff renders
> high-band harmonics as noise that the chip renders as tone — duller/noisier
> highs. Bucket A (genuine algorithmic deviation, not fixed-point). Try
> coef≈0 here and PESQ on fricative/breathy clips; caveat: adversarial
> cpvbad-class content regressed slightly (−2.7pp, one-sided), and the spec's
> E_P>0.5 force-unvoice rule is doing REAL work (disabling it is a measured
> negative) — keep it.

### 3.4 (HIGH, biggest "exceed" lever) Front-end noise suppression / preprocessing
> **Empirical correction (2026-06-04).** DVSI markets in-chip NS (US8315860),
> but the AMBE-3000R **as driven on the bench does NOT run it** — additive
> broadband noise passes through the chip's encode→decode unchanged (gap noise
> floor 0.94×, i.e. not suppressed; `encode_ns_test_2026-06-04/`). It also does
> no AGC/HPF-leveling at the vocoder. So NS is **purely an EXCEED lever, not a
> parity gap** — the chip sets a *low* bar here. (NS may be a config bit the
> chip exposes but the driver doesn't set; either way the deployed comparison
> point has none.)

A modern denoiser (RNNoise-class spectral gating, or
a learned suppressor with a classical fallback) can **substantially exceed** the
chip on noisy field audio (the chip does nothing there). This is likely the largest single
quality-over-chip opportunity for operational P25 traffic. Keep it a
**pre-analysis** stage so the codec core stays clean. **Measure:** PESQ/POLQA on
clean speech + added babble/vehicle noise at several SNRs, `ours→chip` vs
`chip→chip`.

> **MEASURED — GO (2026-06-07).** First data on this lever, run from the
> ambe3000-clone fork (`conformance/baselines/ns_headroom_2026-06-07/`). Corpus:
> clean + dam, each mixed with {white, babble} at {20, 10} dB SNR (8 conditions);
> scored PESQ-nb vs the clean reference. A *crude, untuned* STFT spectral-
> subtraction front-end (alpha=2.0, static lowest-15%-energy noise estimate),
> bolted ahead of an **unchanged** r34 encode→decode, **beats the chip roundtrip
> on 7/8 conditions, mean +0.217 PESQ-nb**. Per-condition exceed margin
> (ours+dn − chip_rt): white20 +0.03, white10 +0.22, dam_white10 **+0.51**,
> dam_white20 +0.36, dam_babble{10,20} +0.35/+0.23, clean_babble10 +0.08; the
> lone loss is clean_babble20 (−0.03). The front-end also lifts our *own*
> roundtrip on all 8 (+0.006..+0.576). Takeaways: (1) the exceed headroom is
> **real and grows as SNR worsens** — biggest on the hardest (low-SNR, vehicle-
> like) cases that dominate real P25 traffic; (2) **babble (non-stationary) is
> the weak spot** of static spectral subtraction — a learned/adaptive suppressor
> (RNNoise, log-MMSE with noise tracking) should widen it; (3) confirms §3.4 as a
> genuine top exceed lever, *not* a parity lever — it leaves the codec core
> untouched. The earlier "input-side spectral subtraction is at best inert"
> remark (near §0.5) was about the *parity/clean* path; on *noisy* input it is
> decisively positive.

> **DEFECT PINNED — the built-in §0.5 spectral-subtraction's stationary-tone
> priming creates a binary, motion-dependent −0.46 dB encode-gain bias
> (2026-06-10, ambe3000-clone probes3/stationarity (h)(iii)).** The legacy
> input-side denoiser (Boll β=0.1, ON by default in `Vocoder::new`) primes its
> noise estimate on ANY spectrum that stays cos-sim ≥ 0.999 stationary — by
> design (the knox_1 +0.585 PESQ win deliberately treats stationary tones as
> noise). Once primed (a one-way latch within a session) every harmonic bin
> with S≈N gets gain √(1−β) = −0.458 dB. Sustained pitch motion ≳0.6 %/frame
> keeps the cos-sim gate failing so the estimator never primes ⇒ tone probes
> split binarily: still/slow-vibrato arms read −0.46 dB vs fast-vibrato arms
> (measured +0.45..+0.52 dB jump; `primed` flag partitions the arms exactly).
> Real speech rarely primes (13/14 chip-A/B vectors byte-identical with the
> denoiser off), but on noisy speech it engages and HURTS chip parity badly
> (cpvbad: gain bias +1.58→+0.42 dB, b2 exact 32.3→48.9% with it off — the
> chip does not denoise, see the 2026-06-04 correction above). QUALITY
> implications: (1) any tone/vibrato-based gain measurement made with the
> default encoder carries this −0.46 dB / binary-motion artifact — the
> ambe3000-clone tone-ladder gain numerology is partially contaminated; (2) if
> the §3.4 exceed-denoiser is rebuilt, do NOT let it prime on voiced/tonal
> content — gate on true noise, or the codec's own gain field inherits the
> bias. Chip-parity fork DEFAULTS it OFF as of 2026-06-10 (`BLIP25_SPECSUB=1` recovers it)
> (`spectral_subtraction_effective`, analysis/mod.rs).

> **RESOLVED for blip25-mbe (2026-06-14, v0.2.0).** `spectral_subtraction` is
> now **default OFF** here too — flipped in both `AnalysisState::new()`
> (`analysis/mod.rs`) and `VocoderBuilder::new()` (`vocoder.rs`), opt-in via
> `set_spectral_subtraction(true)`. Rationale: on the clean-speech corpus it is
> a documented no-op (13/14 chip-A/B vectors byte-identical with it off) so the
> headline speech PESQ matrix is unchanged; its only measured win is +0.08 on
> one stationary tone (knox_1), against the pinned −0.46 dB self-priming
> encode-gain bias above and the fact that it modified the **full-rate IMBE**
> encode spectrum with no chip oracle and no per-path measurement. Defaulting
> it off keeps the encoder closer to the no-suppression emulation target and
> makes `AnalysisState::new()` fully spec-faithful (consistent with the other
> encode-quality levers in that block, all default-spec). The post-decode
> Classical enhancement chain stays default-ON (separate, measured +0.052 PESQ
> win). If you want the knox_1 tone behavior back, opt in explicitly.

> **IMPLEMENTED + MEASURED (2026-06-13) — `predenoise.rs` log-MMSE front-end,
> a broadband-noise exceed lever.** Built the §3.4 stage for real (not the
> crude fork probe): a **separable pre-PCM** streaming denoiser
> (`analysis/predenoise.rs`, `PreDenoise`) — 256-pt √-Hann WOLA STFT (128 hop)
> → minimum-statistics / MCRA per-bin noise PSD → **log-MMSE (Ephraim-Malah)**
> gain with decision-directed a-priori SNR → ISTFT/OLA. All **general DSP**
> (outside clean-room). Runs on the input frame *before* the codec; the core is
> untouched. Opt-in: `Vocoder::set_denoise(true)` / `--denoise[ --denoise-mode
> logmmse|wiener|specsub]`; **default OFF**. Self-priming (the legacy Boll
> defect) is structurally avoided: a **windowed-median peak guard** never learns
> a spectral peak (tone/harmonic) as noise, a **spectral-flatness bypass**
> (`SFM < 0.02`) protects near-pure tones, and a **global input-SNR gate**
> (speech-peak / noise-floor ≥ 33 dB) keeps clean *input* transparent. Unit
> tests pin clean-tone pass-through, no sustained-tone drift, white-noise
> attenuation, COLA reconstruction, reset.
>
> **Live-chip exceed (ours+denoise→chip vs chip→chip, PESQ-nb vs CLEAN ref,
> `--ref-pcm`; `conformance/scripts/{make_noisy,denoise_sweep,analyze_denoise}`):**
>
> | condition | chip BAR | head→chip | +denoise→chip | dn−head | EXCEED |
> |---|---|---|---|---|---|
> | clean (ctrl) | 2.917 | 3.004 | 2.924 | −0.080 | +0.007 |
> | clean white-10 | 2.329 | 2.262 | 2.330 | +0.068 | +0.001 |
> | clean white-5 | 1.797 | 1.774 | **2.205** | **+0.431** | **+0.408** |
> | clean babble-10 | 2.125 | 2.055 | 2.089 | +0.034 | −0.036 |
> | clean hum-15 | 2.964 | 2.576 | 2.419 | −0.157 | −0.545 |
> | famb white-10 | 1.451 | 1.385 | **1.645** | **+0.260** | **+0.194** |
> | famb babble-10 | 1.904 | 1.750 | 1.815 | +0.065 | −0.089 |
> | famb white-5 | 1.297 | 1.267 | 1.369 | +0.102 | +0.072 |
>
> **Verdict:** a real **broadband-noise** win — on white/hiss/vehicle-class noise
> the denoiser **beats the chip by up to +0.41 PESQ** (mean +0.17 on the four
> white cases) and **helps our own encoder on 6/7 noisy conditions** (+0.115
> mean). **Babble** (non-stationary): helps our encoder modestly (+0.03..+0.07)
> but ~ties the chip — min-statistics under-tracks it; an IMCRA/learned tracker
> is the next step (fork §3.4 note). **Hum** (narrowband): a loss, but that is
> our **encoder's** pre-existing weakness (head is already −0.39 vs the chip on
> hum *without* denoise) — not fixable by the front-end; route hum via a HPF and
> fix the low-band encoder separately. **Clean** input loses −0.08 through the
> chip (it engages on the DVSI vector's residual room tone), irrelevant for an
> opt-in noisy-audio tool. **Shipped opt-in**, recommended for broadband field
> noise; not for narrowband hum. Default-on is deferred pending babble (IMCRA)
> and the low-band/hum encoder fix.

> **FOLLOW-UPS LANDED (2026-06-13b) — hum notch (BIG win) + IMCRA babble upgrade.**
>
> **(A) HUM — root cause found + fixed (`humnotch.rs`).** Chip-probe diagnosis
> (`examples/hum_diag_probe.rs`): the −0.39 hum deficit is the **§0.3 pitch tracker
> locking onto the 60/120 Hz mains line** — on clean+hum-15 dB the voiced f0
> collapses to ~65 Hz on 30 % of voiced frames (the §0.5 amplitude blow-up ×22 and
> §0.7 voicing flips are *downstream* of the wrong pitch). The §0.1 single-pole HPF
> (corner ~13 Hz) passes hum untouched, and a broadband HPF fails (pitch re-locks to
> 120 Hz, a valid male f0). Fix: **two narrow RBJ biquad notches at 60+120 Hz (Q≈10)**
> as a separable, **zero-latency** general-DSP pre-PCM front-end (`HumNotch`,
> `Vocoder::set_hum_notch` / `--hum-notch`, default OFF). Runs *before* the denoiser.
> Live-chip (encoder cell vs clean ref): hum-15 dB **2.576→2.999 (+0.423, beats chip
> +0.035)**, hum-10 dB **2.429→3.021 (+0.592, beats chip +0.212)**; clean control
> +0.022 (safe), white-10 −0.008 (neutral), male `mark`+hum +0.116. Clean damage ≈0
> (the Q=10 ~12 Hz notch spares a male f0 grazing 120 Hz via its untouched harmonic
> comb; validated on `mark.pcm`). The −0.39 deficit becomes a +0.04..+0.21 exceed.
> **A clear, low-risk win.**
>
> **(B) BABBLE — IMCRA two-iteration tracker (`predenoise.rs`).** Replaced the
> single-minimum MCRA noise tracker with Cohen-2003 IMCRA (a second speech-excluded
> smoothing/minimum chain that tracks the floor *through* continuous speech), plus a
> Bmin bias and a slow-EMA engage gate. Anti-self-priming guards (median peak guard,
> SFM bypass, global gate) kept; all 5 unit tests + 479 lib tests green. Live-chip:
> **the bigger win is on WHITE/broadband** — exceed cl_w10 +0.001→**+0.143**,
> cl_w5 +0.408→**+0.549**, fa_w10 +0.194→**+0.468** (the speech-excluded minimum
> estimates the floor far better, so log-MMSE suppresses more accurately). **Babble**
> improved modestly as predicted (the classical ceiling): exceed cl_b10 −0.036→−0.027,
> fa_b10 −0.089→−0.078; dn−head +0.043/+0.076 (helps our encoder). **Caveat:** IMCRA
> engages slightly on the DVSI *clean* control's residual room tone (−0.24 dB) which,
> with the STFT's inherent 128-sample (16 ms) group delay, costs ~−0.26 through the
> *chip* decoder (our decoder shows +0.18). So clean transparency is worse than MCRA
> (−0.26 vs −0.08) — **enable only on noisy audio** (it is opt-in, default OFF). Net:
> IMCRA is strictly better on the broadband/babble targets; the clean caveat is moot
> for a noisy-audio tool. A learned suppressor (RNNoise-class) remains the next step
> for non-stationary babble beyond the classical ceiling; eliminating the group delay
> (zero-latency reconstruction) would restore clean transparency.

> **FOLLOW-UP ATTEMPTS — BOTH REVERTED (2026-06-13c), do not re-try as designed.**
> Chased the two next-steps above; both are measured NET-NEGATIVES on real content
> and were reverted (the committed IMCRA + smoothed-gate denoiser, above, stands as
> the best version). Recorded so they aren't re-attempted:
> - **OM-LSA gain** `G = G_LSA^p · G_MIN^(1−p)` (Cohen's IMCRA companion, p = the
>   IMCRA speech-presence prob). It only ever suppresses *more* (`G ≤ G_LSA`), and the
>   `p_speech` estimate is not reliable enough on this content: mid-`p` voiced tails
>   and mis-classified clean segments get slammed to the −20 dB `G_MIN` floor. Live
>   chip: clean **−0.85** (was −0.26), babble **−0.48 / −0.31** (was −0.03 / −0.08),
>   white unchanged. Reverted. (A safer variant would need a much higher `G_MIN`
>   and/or a far more reliable `p` — likely the learned route, not classical.)
> - **Zero-latency direct-pass bypass** (output the raw input frame, zero delay, when
>   the gate is in BYPASS; OLA only when ENGAGE, with a Schmitt-trigger latch). A
>   `clean_input_bypass_is_lossless` unit test confirms it IS byte-identical on a
>   *consistently*-clean stream — but on REAL content that toggles the gate
>   (clean-with-pauses, babble: the DVSI clean engages ~61% of frames as IMCRA's floor
>   rises in pauses), mixing 0-delay (bypass) and 128-delay (engage) frames chops the
>   audio with 128-sample discontinuities. Live chip: clean **−0.85**, babble
>   **−0.44 / −0.29**, white unchanged. The committed always-OLA path (consistent
>   128-delay, no toggling) is strictly better. Reverted. **Lesson:** any per-frame
>   dual-latency scheme glitches when the gate toggles; the STFT group delay is
>   effectively unavoidable for a bypass-toggling denoiser. The genuine fix for clean
>   transparency is to NOT engage on clean at all (a noise-vs-clean *input* classifier
>   that doesn't rise on speech pauses), or to accept the −0.26 caveat (opt-in tool).
> - **RNNoise-class learned suppressor: DEFERRED** (design-verified, not built). A
>   pre-trained ~85k-param GRU band-gain net is mechanically portable to deterministic
>   pure Rust, BUT it needs an 8 kHz-retrained model + a committed weights artifact +
>   a training recipe + (ideally) a `tract`/`candle`-class dep — none exist in-repo and
>   all violate the repo's pure-Rust/no-ML-dep/determinism discipline. It belongs in a
>   SEPARATE experimental crate behind a non-default feature, never in the codec path.
>   This is the real ceiling-breaker for non-stationary babble; it is a project, not a
>   patch. The shipped classical ceiling is **IMCRA + log-MMSE** (broadband win,
>   modest babble); OM-LSA and zero-latency do not improve it.

### 3.5 (LOW–MED) Encode-side tone detection
Encode-side tone dispatch exists for single/DTMF; verify it matches the chip's
tone handling so alert/Knox/DTMF traffic encodes as tones, not buzzy voice.

---

## 3.6 (DONE 2026-06-13) Encode chip-loop campaign — levers landed + measured stack

First in-repo (blip25-mbe, not the fork) implementation + live-chip A/B of the
chip-validated encode levers above. **Eight levers landed as opt-in flags**
(`AnalysisState` setters + `halfrate-ab-matrix` CLI flags); every one defaults to
the spec/current path so `AnalysisState::new()` is **byte-identical** to before
(verified on clean/dam/fambf22c/tambf22a; 469 lib tests green). Measured in the
`ours→chip` cell on the live AMBE-3000R via `conformance/scripts/lever_sweep.sh`
+ `analyze_levers.py` (chip BAR `chip_enc_chip_dec` was bit-identical across all
configs — oracle is stable).

**Levers (flag · setter · class · per-lever 4-vector Δ vs spec, 600 fr):**

| lever | `halfrate-ab-matrix` flag | `AnalysisState` setter | class | Δmean |
|---|---|---|---|---|
| octave escape | `--pitch-escape` | `set_pitch_decide_escape` | general-DSP | 0 (0 frames on speech; high-F0 guard) |
| parabolic sub-sample | `--pitch-subsample` | `set_pitch_subsample` | general-DSP | −0.037 alone* |
| §0.4 refine off | `--no-pitch-refine` | `set_pitch_refine` | spec-toggle | **+0.056** |
| frac band-edge | `--amp-frac-edges` | `set_amp_frac_band_edges` | general-DSP | +0.031 |
| +0.9 dB level | `--level-scale` | `set_level_scale` | chip-const | −0.013 PESQ / RMS→1.0 |
| silence shape-zero | `--silence-shape-zero` | `set_silence_shape_zero` | spec-toggle | 0 (rarely fires) |
| V/UV Θ rolloff→0 | `--vuv-pitch-coef 0.0` | `set_vuv_pitch_coef` | chip-observed | +0.001 |
| M(ξ) Θ grade | `--vuv-mxi-grade` | `set_vuv_mxi_grade` | chip-observed | +0.001 alone* |

\* sub-sample is **net-negative alone** because §0.4 `refine_pitch` then re-searches
±9/8 around it and overwrites the gain (the documented mis-aligned-window E_R walk);
it only helps **paired with refine-off**, which lets the finer §0.3 pitch reach the
wire. M(ξ)-grade is ~neutral alone but compounds with the pitch stack.

**Winning stack: `no_refine + subsample + vuv_mxi` (+ `pitch_escape` guard).**
7-vector confirmation (800 fr; encoder-cell Δ PESQ-nb vs spec default):

| stack | clean | dam | mark | fambf22c | fambm22a | tambf22a | tambf32b | mean |
|---|---|---|---|---|---|---|---|---|
| no_refine | −0.022 | −0.047 | +0.103 | +0.152 | +0.066 | +0.098 | −0.035 | +0.045 |
| +subsample (nr_ss) | +0.061 | −0.039 | +0.102 | +0.154 | +0.027 | +0.046 | −0.045 | +0.044 |
| **+vuv_mxi (nrss_mxi)** | **+0.078** | −0.050 | **+0.122** | **+0.172** | +0.050 | +0.070 | −0.018 | **+0.060** |

Absolute encoder-cell PESQ-nb: head **2.750** → nrss_mxi **2.810** (BAR
`chip_enc_chip_dec` = 3.018; the stack closes **22%** of the −0.268 encoder gap,
~31% on female content where it was worst). The hard-bounded (multiply-only, ≤2.25×,
**cannot mute**) M(ξ) grade lifts clean/mark/female and never collapsed PESQ on any
vector incl. degraded `dam`. Only cost: `dam` −0.050, `tambf32b` −0.018.

**Interference robustness (the stack tracks the chip better under noise).** head vs
nr_ss `ours→chip` cell, scored vs the *noisy* input (chip does no NS, so matching it
is the target), 400 fr:

| condition | head Δ-to-chip | nr_ss Δ-to-chip |
|---|---|---|
| clean white-10 dB | −0.124 | **−0.008** |
| clean hum-20 dB | −0.160 | −0.024 |
| clean babble-15 dB | +0.020 | **+0.127** |
| fambf22c babble-15 | −0.259 | −0.101 |
| fambf22c white-10 | −0.280 | −0.131 |

The pitch stack makes our encoder track the chip **much** closer under white/babble/hum
(7/8 conditions improved), confirming the win is not clean-speech-overfit.

**Loudness (`--level-scale`).** PESQ normalizes level, so the +0.9 dB lever is
PESQ-neutral (−0.013) but drives `our_enc/chip_enc` RMS **0.905 → 1.001** = chip
loudness parity. Kept opt-in (loudness is a consumer-layer concern per §0; it slightly
overshoots female and costs a hair of gain-quantization-trajectory PESQ).

**Levers neutral on this corpus (kept opt-in):** `vuv_pitch_coef=0`, `silence_shape`,
`specsub_off` are ~0 on clean speech (their value is on fricative / silence-heavy /
tonal-noisy content, per the §3.3/§3.1/§3.4 notes). `specsub` on/off is a literal
no-op on clean studio speech here (±0.000), consistent with the fork's "denoiser only
primes on near-silent frames."

**Root-cause note / follow-up.** The `no_refine` win is a *mitigation* of our §0.4 E_R
search walking pitch on the **left-clipped §0.5 analysis window** (`extract_refinement_window`
zero-fills the left 30 samples — confirmed). The principled fix is the deferred
`retain_past_frame` window-centering lever, but the fork measured un-clipping alone as
~neutral vs the chip, so it is unlikely to recover the `dam`/`tambf32b` regressions
(those are the genuine cost of forgoing quarter-sample resolution where it was correct).
Left as a future investigation; `no_refine` captures the realized win today.

**Spec-faithful note + promotion (2026-06-13).** `no_refine` bypasses spec §0.4 and
`vuv_mxi` relaxes spec Eq.37 Θ on confident-loud frames — both chip-validated. Per the
user decision, the stack is **promoted to the production default in `Vocoder::new()` /
`reset()`** (`Vocoder::production_analysis_state()`, vocoder.rs) — escape + sub-sample +
refine-off + M(ξ)-grade ON — while **`AnalysisState::new()` itself stays spec-faithful**
(the clean-room baseline + 469 lib tests rely on it). Each lever is individually opt-out
via `Vocoder::set_*` (regression test `vocoder_new_enables_production_encode_stack`).
The harness `our_encode_ambe_plus2` non-tone path still constructs the bare spec
`AnalysisState` + explicit CLI flags, so `halfrate-ab-matrix` (no flags) = spec baseline
for future A/Bs. `--level-scale` (loudness parity, RMS→1.0), `--amp-frac-edges`,
`--vuv-pitch-coef 0.0`, `--silence-shape-zero` stay opt-in. Reusable sweep tooling:
`conformance/scripts/{lever_sweep.sh,analyze_levers.py,make_noisy.py}`.

---

## 3.7 (AUDIT 2026-06-14) Fork rounds 9–10 + APX handoff — disposition

Audited everything the sister fork `ambe3000-clone` produced **after** the
2026-06-13 §3.6 port — its campaign rounds 9–10 (commits `4c72a72`, `f2eb99d`,
`8527dda`) and the `handoff/` APX-firmware RE package — against blip25-mbe's
current code (adversarially verified, both repos). **Headline: nothing
high-value remains unported.** The fork's later wins are *chip-bit-parity*
wins on the fork's own goal; its own round-10 half-rate PESQ matrix
(`pesq_hr_matrix_2026-06-11/results_r10.txt`) moved the encode side only
−0.204 → −0.196 MOS across all of rounds 9–10 (flat, ~noise). For blip25-mbe's
PESQ goal the leftovers are cheap alignments, blocked items, or untested
hypotheses — none promises a measured PESQ delta. Disposition:

**SKIP — bit-parity-only or blocked (do not port):**
- **Onset voicing-extent ramp + cascade cap** (r9/r10, fork-PROMOTED, +4.91pp
  b1-idx). Pure chip-bit-parity; the fork's own PESQ shows ~+0.008 MOS (flat).
  The actionable lesson (hard-bound any classifier-driven suppression — the
  fork's unbounded `M(ξ)<0.5` branch full-file-MUTED sustained voiced content)
  is already in §3.6 note (3). The *perceptual* "attack-smoothness extent ramp"
  is a **different, untested** hypothesis from what the fork shipped (a
  parity-restoring devoice/sparsify rule) — see untested list below.
- **MONO65 full-rate gain ladder** (§2.6) — **double-blocked.** (1) Provenance:
  the −0.25·(8−k) low-8 downshift is reverse-engineered from the Motorola APX
  firmware binary (`porcvn_HFWS_DFWA_1.bin`), the forbidden source class, and is
  **not** re-derivable from Annex-E (whose published low-8 deltas are non-uniform
  `[0.148,0.136,0.175,0.162,0.126,0.115,0.145]`, so a uniform −0.25 ramp
  contradicts the spec). (2) No oracle: full-rate IMBE isn't chip-orackable
  (AMBE-3000R is AMBE+2 only). Even the fork that owns the RE never wired MONO65
  into its own decoder (`imbe_wire/dequantize.rs` still uses plain Annex-E). Only
  legitimate route is a spec-author derivation into a TIA spec — a user/spec
  decision, not an implementer port. Leave documented in §2.6, not actionable here.
- **epcorr voicing feature** — round 10 *tested it as a lever and it FAILED*
  (pooled b1-idx z −4.3…−7.8, LOVO picks "none" 14/14, and it **regresses tones**
  — our known weak spot). Not "untested" (the §1 note's wording is stale); it's a
  measured negative. Skip.
- **PITCH_FIR 2-tap pitch smoother** — fork-WIRED-not-promoted, **blocked** on a
  prerequisite (+0.3-sample §0.3 estimator bias at P≈28–36) *and* aimed at a
  non-problem for us (§3.2: our §0.3 is already chip-grade on voiced speech, the
  residual is an inaudible ±1 cloud). Skip until/unless §3.2 P≈28–36 bias is fixed.

**CHEAP ALIGNMENT — clean provenance, ~zero cost, but expected PESQ ≈ 0 (do only
if a PESQ A/B shows a delta; do NOT promote on the fork's parity numbers):**
- **Analysis-window left-clip un-clip** (`retain_past_frame`). `extract_refinement_window`
  (analysis/mod.rs ~924–936) zero-fills the left 30 samples — a genuine
  asymmetric-window float bug (a 221-sample symmetric §0.2 window over a 160-sample
  frame legitimately needs 30 past samples). It feeds the §0.5 amplitude and §0.7
  V/UV legs that `no_refine` does **not** mitigate, so a small attack/transition
  effect is plausible-but-unmeasured. ~10 lines, textbook DSP. **Do NOT** also port
  the fork's `+48` forward shift — that's a half-rate-only, un-derived chip-parity
  frame-phase convention (Bucket A) that hurts pitch refinement and adds level loss.
- **Decode regen scale → 0.4** (keep the `l≤L̃/4` exemption; do **not** add
  `REGEN_ALL`). We currently run the full patent γ=0.44 (effective scale 1.0); the
  chip's measured effective scale is ~0.4× (γ≈0.18). One-line const, chip-observed
  provenance. Fork verdict: "a small parity refinement, not a quality unlock"
  (OFF→full-patent moves the synth residual only ~0.3°). Note this *would* change
  the production AMBE+2 decode default, so gate/measure before promoting.
- **Silence-envelope b3 "residual" mode** — a genuine refinement over our cruder
  `silence_shape_zero`: we shipped the inferior *lam* variant (geometric-mean
  flatten of M̂ pre-prediction, PRBA-only, full-160 RMS gate), which the fork's
  LOVO found strictly worse than the *residual* mode (scale the post-prediction
  log-residual T̃→0, covering **both** PRBA and HOC, gated on the 221-sample
  amplitude-window RMS). Fork ships it default-ON; ours is opt-in/off. Audio delta
  measured inaudible (0.74% energy). Port only if a half-rate PESQ A/B vs our
  existing `silence_shape_zero` on a silence-gap corpus shows a non-trivial delta.

**UNTESTED QUALITY HYPOTHESES — worth a fresh PESQ/STOI A/B (not a port of fork
parity work); nobody has run these:**
- **Onset voicing-EXTENT ramp as attack-smoothness** (distinct from the fork's
  parity devoice rule): a 1–2-frame voiced-band ramp at voiced-run onsets, judged
  by PESQ+STOI on plosive-rich speech (§2.4 tie-in).
- **Onset voiced-phase reset** (`PHASE_UV_RESET`): the chip cold-starts its
  voiced-phase synth at every voiced frame following an unvoiced frame (chip probe:
  onset phase == cold start to 2.3°, history-independent). This is the one phase
  lever that changes phase exactly where the ear is sensitive (onsets/transients),
  unlike steady-voiced φ₀/regen which are inaudible (§Framing). A/B for onset
  crispness. The φ₀ pitch law (`PHASE_CW`, CW=−22.6/C=+2.19) is a weaker companion
  (parity +0.5 dB SNR, but mostly steady-voiced = inaudible).

---

## 4. Suggested first session plan

1. **Stand up the scorer.** Confirm PESQ (wideband) runs in
   `conformance/speech-quality`; assemble a clean-speech corpus.
2. **Run the 4-cell matrix** (chip on `pve`) to get the parity gap and attribute
   it to encoder vs decoder. This decides whether to start at §2 or §3.
3. **Decode:** reproduce the per-harmonic envelope divergence (§1, §2.1) on real
   `tv-rc/r33` speech bits; fix the log-mag prediction / §1.10 stage with the
   largest LSD contribution; re-score `chip→ours`.
4. **Encode:** attack amplitude RMSE (§3.1) and pitch robustness (§3.2); re-score
   `ours→chip`.
5. **Claim the easy "exceed" wins:** verify chip artifacts (§2.2) are off by
   default and A/B on dynamic speech; scope the modern front-end denoiser (§3.4).

## 5. Caveats / clean-room
- The chip is a legitimate **black-box oracle** (feed bits/PCM, observe output).
  Do not read copyrighted TIA PDFs or open-source vocoders; derive from
  `~/blip25-specs` derived specs + general DSP/speech-coding knowledge.
- "Better than the chip" must be judged against a **clean-speech reference via
  PESQ/POLQA**, not against chip output (the chip is not the quality ceiling).
- We can only observe the chip's PCM output, never its internal `M̄_l`/state;
  infer envelope/voicing from output spectra.
</content>
