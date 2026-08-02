# Project knowledge

Findings that are expensive to re-derive and are **not** recoverable from the
code, the tests, or `git log`.

This is deliberately not a design document (see [`DESIGN.md`](../DESIGN.md)) and
not a changelog. It is the set of things a competent engineer would otherwise
have to rediscover.

> **Scope note.** Reference credentials, host addresses, and the owner's private
> infrastructure are deliberately excluded — this file is public. They live in
> the assistant's local memory only.

---

## 1. What the codec is actually targeting

The correctness target is the **reference codec that ships in a professional
dispatch console**, reverse-engineered into `blip25-codec`. It is *not* the
AMBE-3000R chip and *not* any single reference test-vector set.

That distinction is load-bearing, because the reference vendor ships several
products whose output genuinely disagrees:

| Reference | What it is | Standing |
|---|---|---|
| Console codec | The dispatch console's vocoder | **The target** |
| `tv-std/tv/r33` | the reference SDK P25 vectors, AMBE+2 | Good proxy; decode matches, ear-good |
| `tv-std/tv/p25` | Full-rate IMBE vectors (18 B/frame) | Phase 1 reference |
| `tv-rc` | **Rate-conversion** vectors | **Not** the AMBE+2 reference — easy to grab by mistake |
| AMBE-3000R chip | Third distinct product | Not a reference for either (see §5) |

**They disagree with each other.** An encoder reverse-engineered against the
desktop reference product scores 0% whole-frame against `tv-std/r33` while matching `tv-rc` at
b0 85%. Two faithful reverse engineerings of "the reference encoder" can target
different reference products. When a conformance number looks impossible, check
which reference you are scoring against before you debug the code — scoring
against `tv-rc` by mistake invalidates the whole run.

## 2. The only quality gate is a listening test

Every automatic metric tried on this codec has mispredicted perceived quality
at least once. Treat all percentages as diagnostics that *locate* a divergence,
never as evidence that one build is better than another.

- **PESQ is invalid at 2450 bps AMBE+2.** Usable as a regression tripwire;
  never as a promotion gate.
- **`spec_corr` has a structural ceiling.** A large, content-dependent share of
  spectral correlation is unreachable on real audio, and it falls with spectral
  flatness. Never optimise toward 1.0. Deltas under ~0.04 are inaudible.
- **Use `env_r` (per-frame energy envelope), not sample SNR or Pearson `r`,**
  for IMBE. IMBE decode is content-correct but phase-divergent: `env_r` 0.96
  against `r` 0.03. Judging it by `r` says "broken" about audio that is fine.
- **Field-agreement percentages** (b0/b3 match rates) have repeatedly moved
  opposite to perceived quality.

The criterion for the ear test is **intelligibility over a radio** — digits,
onsets, callsigns — not pleasantness. the reference chose clarity over smoothness, which
is why smoothing changes keep testing net-negative even when they score better.

## 3. Constants that must not be re-tuned

These are correct and cannot be re-derived. Changing any of them will appear to
help on some metric.

| Constant | Value | Note |
|---|---|---|
| `γ_w` | 146.64 | Synthesis gain. Never fit to another decoder's output — see §6. |
| LCG seed | 60584 | Unvoiced noise generator. This is a **phase alignment**, not a tunable parameter. Hold fixed; never sweep. |
| Soft limiter | knee 27000, ceiling 31500 | the reference keeps ~1.1 dB headroom rather than clipping to the i16 rail. |
| Tone amplitude | `round(1.406 · 20·log10(√(p1·p2)) + 0.16)` | Annex-T encode-side `A_D`. |
| Encode gain | `γ = 0.65·log2(frame_energy) − 5.74` | Energy-anchored, fit to the console's reconstructed gain. One gain path by design; there is no reference-exact or clean-room alternative. |

## 4. Do not re-attempt

Each of these is ruled out on evidence. Some name code that no longer exists.

**Encoder**
- Full-reference `encode_wave` amplitude. It follows the AMBE-3000/reference desktop
  convention and comes out unintelligible ("Charlie Brown") through this
  decoder. The reference is not the target.
- PYIN, or any generic pitch estimator, on the shipped path. Its ~25% gross
  downward (subharmonic) tail is the whole audible distortion. The RE'd reference
  chain replaces it — see §9.
- Matched-filter amplitude estimation; the wide analysis window; the
  single-EMA "trusted A_M"; the amplitude-EMA flag.
- 100× encode speedup while staying byte-exact. The byte-exact ceiling is
  ~9–11×; FFT, SIMD, and FMA all change the bits.

**Decoder / synthesis**
- `fresh_phase_step`. It scores as an SNR improvement while breaking voiced
  phase continuity, producing ~50 Hz beating. Eq. 139 accumulation is correct.
- Blaming the two-phasor synthesis for the AMBE+2 tone collapse. The float
  synth sustains fine; the cause is the concealment gate (§7), and an OLA
  bridge does not help.
- Chasing mid-burst tone phase on changing-tone vectors (DTMF sweeps). the reference
  derives it from context that has not been reverse-engineered.
- Wiener-vs-Boll spectral subtraction; the reference spectral-discontinuity clamp.

**Fitting to the wrong thing**
- Fitting `γ_w` to SDRTrunk output. SDRTrunk bakes in ~8–10 dB of post-decode
  gain, so "~9× quiet versus SDRTrunk" is not a defect in this codec.

## 5. Hardware behaviour

- **The AMBE-3000R is AMBE+2 only.** `PKT_RATEP` will happily accept full-rate
  IMBE parameters and route the bits through its AMBE+2 codec. The output
  sounds speech-like and is **not** IMBE. There is no IMBE reference product; this
  project's IMBE decoder is the reference.
- **Never send `PKT_RESET` (0x33).** It puts the reference into a permanently
  unresponsive state. The FTDI bridge still enumerates, but the reference answers
  nothing at any baud rate. USB authorize cycling, `USBDEVFS_RESET`, DTR/RTS
  toggling, and bus rescans do not recover it — only a physical unplug does.
  To clear reference state, re-send `PKT_RATEP` instead.
- **The first encode call of a session can emit garbage.** Re-run it. Decode is
  unaffected.
- **Reference output is state-dependent** and does not reproduce the reference test
  vectors.

## 6. Decisions with non-obvious rationale

- **Post-decode enhancement ships off by default.** Enabling it makes output
  diverge from the reference, which is the definition of a defect here.
- **`AmbeStream` grows ~650 B per frame.** This is PTT-bounded by design, not a
  leak — a transmission is finite and the stream is reset at the boundary.
- **Clippy is deliberately off.** The engine is a transliteration of x86
  fixed-point code; idiomatic-Rust lints fight the port and obscure real
  warnings. The workspace is warning-clean under `rustc` instead.
- **Baseline warnings inside the engine are port artifacts.** Do not "fix"
  them. `BITREV` unused-warnings in particular are false positives caused by a
  dual `include!` site — the constants are live.
- **Scope boundary.** `blip25-mbe` is the codec library only. P25 framing, NID,
  air-interface FEC, modulation, and trunking belong to a separate SDR crate.
- **Encryption sits between the vocoder and the air-interface FEC**, so a real
  P25 radio runs the codec with FEC *off* and does FEC outside crypto. The
  soft-decision decode path therefore serves standalone and interop callers,
  not the encrypted P25 chain.

## 7. Codec behaviour worth not rediscovering

- **IMBE FEC ↔ no-FEC is 100% bit-exact** against the reference.
- **Silence dispatch is a reference-bit-match behaviour only.** It matches the reference's
  bits and tanks perceived quality.
- **Mute should produce comfort noise**, not digital silence (BABA-A §1.11.2).
- **The AMBE+2 tone collapse was the concealment gate.** Annex-T tone frames
  carry `b̂₀` in the erasure range [120,123], so the FEC-erasure gate mistook
  them for erasures and repeated the previous voice frame — whose overlap-add
  rings up and then cancels to silence. Classify tone frames *before* the
  erasure gate.
- **Phase 1 IMBE has no tone-frame opcode.** Knox/DTMF travel out of band.
  Phase 2 AMBE+2 has Annex-T (IDs 144–159 = Knox).
- **r34 byte order is a 3-way interleave**, not sequential. It is the
  AMBE-3000R *serial host* format only — an Icom OTA wire is natural/`AMBE_d`
  order. Bit order is a standing wrong suspect for encoder accuracy problems.

## 8. Reference values worth not re-deriving

**Annex-T tone IDs (AMBE+2 / Phase 2).** The `I_D` field is carried in four
redundant copies for FEC robustness.

| Range | Meaning |
|---|---|
| 5–127 | Single tone |
| 128–143 | DTMF |
| 144–159 | Knox |
| 160–163 | Call progress |
| 255 | Silence |

Full-rate IMBE has no tone-frame mechanism at all (BABA-A §5.4) — Knox and DTMF
travel out of band on Phase 1. A ~1.6 PESQ ceiling on Knox vectors through the
full-rate codec is a structural limit, not a defect.

**P25 default NAC is `$293`** (TIA-102.BAAC-D §2). `$F7E` lets a receiver open
on any NAC; `$F7F` lets a fixed station receive and retransmit any NAC. Every
worked NID/BCH example in TIA-102.BAAB-B Annex B uses `$293`, and TIA-102.CABA
bakes it in as the default interoperability test NAC, with `$300` as the
alternate-rejection NAC for squelch verification. The standard-tone test
pattern is NAC=`$293`, MFID=`$00`, ALGID=`$80`.

## 9. The encode pitch chain

Encode accuracy reduces to **pitch**. Amplitude error is a *victim* of pitch
error, not an independent problem — feeding reference pitch reproduces the amplitude
parameters bit-exactly. Voicing is handled by the RE'd tracker and gain is near
closed-form.

Both encoders therefore run the reverse-engineered reference spectral-DP pitch chain
rather than a generic estimator. Their output bits are frozen — the console A/B
that set them is retired, and `crates/blip25-mbe/tests/golden_output.rs` is what
holds them in place:

- **Pitch** — `enc::b0_audio` derives `b0` from prefiltered PCM alone, with no
  capture-fed side channels, scoring **195/199 (98.0%)** on both `voiced.pcm`
  and `mark.pcm`. It serves **both AMBE+2 and IMBE**, and it is what keeps the
  subharmonic "axe" distortion out.
- **Gain** — energy-anchored: `γ = 0.65·log2(frame_energy) − 5.74`, fit to the
  console's reconstructed gain. A mean-log-amplitude gain floors out on quiet
  frames and produces a "no pure silence" hiss.
- **There is deliberately no way to select a generic-estimator path.** The reference
  has one dependable path, so this has one too.

A generic clean-room pitch estimator still exists in `enc::pitch` and produces
the initial estimate, but the reference chain overrides it on the shipped path. Do
not mistake its presence for it being what ships.

Perf note: `b0` harvesting is a pitch-only chain (`encode_pcm_b0` plus the
shared `b0_sequence` helper, so the two cannot drift). It must not run a full
encode per frame and throw the amplitude away.

---

*Maintenance: this file records conclusions, not history. If a conclusion here
stops holding, correct it in place — a stale "never do X" is worse than no note
at all.*
