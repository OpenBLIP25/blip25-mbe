# DVSI AMBE-3000R vs TIA-102 — which deviations are quality levers

**Date:** 2026-06-04
**Audience:** a blip25-mbe session raising *perceptual* quality (not bit-exact
parity). Companion to `QUALITY_FINDINGS.md`.
**Source:** distilled from the `ambe3000-clone` half-rate (rate 33/34 AMBE+2)
chip-parity campaign. The full chip-RE catalog (all buckets, with evidence) is
`~/blip25-specs/analysis/ambe3000_chip_oracle_caveats.md` §7.

---

## Why this document exists

The bit-exactness fork measured many ways the chip differs from a naive
spec-float decoder. **Most of those differences are useless — or harmful — to
copy into blip25-mbe**, because they are the chip's *fixed-point precision
limits*, not algorithm choices. blip25-mbe is float and aims to **exceed**
1990s silicon, so emulating its precision would regress quality.

This document keeps only the differences that are **genuine algorithmic
deviations from TIA-102** (things the spec doesn't prescribe) and flags, for
each, whether it's a *match* lever (do what the chip does) or an *exceed* lever
(the chip's choice is a compromise to beat). Everything that is "just
fixed-point" is listed once, under **Do Not Copy**, and then dropped.

> **The one-line rule.** If a chip behavior would also change on a non-P25
> signal *only because of integer arithmetic* (Q-format, rounding, FFT
> wordlength), it is precision — ignore it. If it changes *what the algorithm
> computes* (skips an enhancement, picks a different envelope, classifies
> voicing differently), it is a deviation — consider it.

---

## Do NOT copy — these are fixed-point artifacts, not spec deviations

Listed so nobody mistakes them for quality levers. (Catalog Bucket B.)

- **~Q5 log-magnitude dequant resolution + round-to-nearest.** The chip
  reconstructs the residual envelope at ~5 fractional bits. A float codec runs
  full precision — copying Q5 only coarsens timbre.
- **Synthesis fixed-point arithmetic** (the ~14 % unvoiced per-bin residual, the
  voiced fine structure). Integer realizations of §1.12. Modern float synthesis
  is strictly better.
- **IQ24 coefficient arithmetic.** Substrate detail.

If `QUALITY_FINDINGS.md` ever sprouts a "make it match the chip's Q5/rounding"
task, that task is mis-scoped for this repo.

---

## Genuine deviations that ARE quality levers

### L1 — §1.10 spectral enhancement: the chip effectively skips it (decode)
**Deviation (catalog A3).** Applying spec §1.10 enhancement makes our decode
*match the chip worse* on all four reference vectors (the chip behaves as if it
applies a fixed high-shelf `H(f)`, not the §1.10 ratio-clamp).
**For blip25-mbe:** this is an *exceed* question, not a copy. "Matches the chip
better with §1.10 off" ≠ "sounds better with §1.10 off" — the chip may simply
omit an enhancement that helps intelligibility. **Action:** A/B §1.10 on/off by
PESQ in the `ours→chip` cell, and separately evaluate replacing it with the
chip's fixed-shelf behavior. Feeds `QUALITY_FINDINGS.md` §1 (decode) — see the
existing §1.10 enhancement lever there.

### L2 — Voiced-phase regeneration scale is LOW (decode onsets/continuity)
**Deviation (catalog A4).** The chip's effective voiced-phase regeneration
scale is `γ ≈ 0.18` (`REGEN_SCALE ≈ 0.4`), well below the patent value. This
**overturns** the earlier `REGEN_SCALE ≈ 1.2 + REGEN_ALL` guidance that may
still be implied elsewhere in this repo's notes — correct it.
**For blip25-mbe:** phase is largely inaudible in *steady* voicing (per
`QUALITY_FINDINGS.md` framing), so this matters only at the three places phase
is audible — **onsets, the V/UV boundary, inter-frame continuity**. Use the
corrected lower scale there; do not chase steady-state phase. Caveat: the
genuine reachable headroom from the regenerator is small (most of the
per-frame "operator explains the scatter" effect was overfitting).

### L3 — Spectral-amplitude ESTIMATOR rule differs (encode — the top lever)
**Deviation (catalog C2 — open gap, but the highest-value one).** On encode,
our *continuous* estimated log-envelope diverges from the chip's by ~3× the
PRBA quantization-cell radius — a real estimation-*rule* difference, uniform
per-harmonic, **not** caused by quantization, predictor drift, a spectral tilt,
a fixed pre-emphasis curve, or analysis-window alignment.
**For blip25-mbe:** this confirms `QUALITY_FINDINGS.md` §3.1 (spectral-amplitude
estimation) as the #1 encode lever, and tells you **how to measure** a candidate
estimator: use the **PRBA distance-rank** (does the chip's chosen cell move
toward rank 1?), not the post-VQ envelope SHAPE, which overstates divergence
~3× through VQ granularity. **Exceed opportunity stands:** the chip estimates
with limited windows/precision; a modern multi-resolution / reassigned-spectrum
estimate can place harmonic energy more accurately than the chip *or* the spec.
Within-frame suspects to probe first: the per-harmonic energy-integration /
band-edge rule and the DFT/window resolution.

### L4 — Encode gain is energy-dependent (compressive) (encode)
**Deviation (catalog C4).** The chip's realized level is biased low and
**compressive** — about −2.75 dB on the loudest frames vs −0.68 dB on the
quietest. Not a constant offset; a gain *law* difference.
**For blip25-mbe:** if our encode sounds level-inconsistent vs the chip on loud
passages, this is why. Pin/parameterize the gain law (`mean_lambda +
0.5·log₂(L)` is our form). Lower priority than L3.

### L5 — Beyond-spec activity/stationarity processing (both directions)
**Deviation (catalog A1).** The chip has an amplitude-dependent mute, a ~±10000
internal cap, a `<1%`-jitter stationary-signal smoother, and a
spectral-discontinuity attenuator — none in BABA-A.
**For blip25-mbe:** mostly *compromises to NOT replicate* (the mute and cap are
artifacts of real-time silicon, not quality features). The stationary-smoothing
and discontinuity-attenuation are borderline — they could either reduce
musical-noise artifacts or smear transients. Evaluate only if a specific
artifact (buzzy sustained vowels, or transient smearing) shows up in listening.

---

## Not quality levers (recorded so they're not chased here)

- **V/UV extended codebook indices 17..31 (catalog A2).** A beyond-spec chip
  behavior, but **decode-equivalent** — no audible effect. A bit-exactness
  curiosity only.
- **Encode pitch ±1 cloud (catalog C3).** Relevant to bit-match, but the gross
  pitch behavior (octave/voicing) is the audible part — that's already
  `QUALITY_FINDINGS.md` §3.2, independent of the ±1 cell question.

---

## How this maps onto QUALITY_FINDINGS.md

| Lever here | QUALITY_FINDINGS.md section | Verdict |
|---|---|---|
| L1 §1.10 enhancement | §1 (decode enhancement) | exceed — A/B, don't blind-copy |
| L2 voiced-phase regen scale γ≈0.18 | §1 (decode phase) / onset levers | match (corrected param), narrow scope |
| L3 amplitude estimator rule | §3.1 (HIGH, top encode lever) | exceed — measure by PRBA rank |
| L4 compressive gain law | §3.1 adjacent (gain) | match, lower priority |
| L5 activity/stationarity gates | §3.4 / decode | mostly do-not-replicate |
