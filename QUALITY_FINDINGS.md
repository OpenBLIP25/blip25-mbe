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

---

## 0. Measure before optimizing — the attribution method

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

### 3.2 (HIGH) Pitch estimation robustness
Pitch doubling/halving is among the most audible MBE failures (octave jumps,
roughness). The encoder uses a pYIN-style tracker; harden against
octave errors and rapid pitch transitions. **Exceed opportunity:** a modern
pitch tracker (CREPE-class, or robust autocorrelation+HMM smoothing) can beat
the chip's tracker — keep a classical fallback for determinism. **Measure:**
gross-pitch-error rate vs a reference tracker on voiced speech; then PESQ.

### 3.3 (HIGH) Voicing (V/UV) decision
Wrong voicing → buzzy (false-voiced) or noisy (false-unvoiced) artifacts,
very audible. Audit `analysis::vuv` band decisions against the chip on borderline
(breathy / fricative) frames. **Measure:** per-band V/UV agreement vs chip on
labelled speech; PESQ on breathy/fricative-heavy clips.

### 3.4 (HIGH, biggest "exceed" lever) Front-end noise suppression / preprocessing
On *real-world noisy* input the chip runs ~15 dB spectral subtraction
(US8315860) plus HPF/AGC. Without a comparable front end blip25-mbe will sound
worse on field audio — and a modern denoiser (RNNoise-class spectral gating, or
a learned suppressor with a classical fallback) can **substantially exceed** the
chip's 1990s noise suppression. This is likely the largest single
quality-over-chip opportunity for operational P25 traffic. Keep it a
**pre-analysis** stage so the codec core stays clean. **Measure:** PESQ/POLQA on
clean speech + added babble/vehicle noise at several SNRs, `ours→chip` vs
`chip→chip`.

### 3.5 (LOW–MED) Encode-side tone detection
Encode-side tone dispatch exists for single/DTMF; verify it matches the chip's
tone handling so alert/Knox/DTMF traffic encodes as tones, not buzzy voice.

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
