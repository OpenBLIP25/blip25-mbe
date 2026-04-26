# blip25-sdr scope plan

**Date:** 2026-04-25.
**Author:** implementer pass for blip25-mbe ↔ p25-decoder integration.
**Status:** draft. Replace with a real RFC if the user wants to formalize.

This document is from blip25-mbe's point of view. It exists because blip25-mbe
ships codec primitives (IMBE / AMBE / AMBE+ / AMBE+2 synth and analysis); the
SDR-side work — RF, DSP, frame sync, FEC, framing, trunking, data — lives
elsewhere. This is the integration map.

## TL;DR

The "SDR" side is `~/p25-decoder/` (workspace named `blip25`). It already
contains 11 crates and ~32k LOC, with `blip25-mbe = { workspace = true }`
wired into `blip25-vocoder` and `ambe-client`. The remaining work is
**filling in the planned crates** the existing CLAUDE.md flags as "(planned)"
and **closing the codec integration boundary** between blip25-mbe and
blip25-vocoder.

## Where things stand

### blip25-mbe (this repo)

Solid:
- IMBE Phase 1 full-rate: encoder + decoder + 5-vector PESQ baseline.
- AMBE+2 half-rate: encoder + decoder + 5-vector PESQ baseline.
- All four AMBE generations have a `synthesize_frame` entry point.
- Carrier-agnostic raw entry points (`decode-raw-halfrate`, `decode-raw-mbe`)
  for non-P25 AMBE+2 carriers.

Open at the codec layer (see memory entries dated 2026-04-25):
- 2-frame onset attack (spec-locked, multi-session if pursued).
- Tonal-content deficit on alert.pcm (spec-locked).
- Two half-rate API gotchas — `MbeParams::silence()` not in the half-rate
  pitch range, `encode_halfrate` errors instead of dispatching silence on
  out-of-range pitch. Worth small upstream fixes.
- AMBE+ (Gen 2) has synth but no dequantize wrapper at the 4-vector
  interface — small wrapper job if a real test case shows up.

### p25-decoder workspace (the SDR side)

Crate inventory as of 2026-04-25:

| Crate              | Files | LOC    | Status (per CLAUDE.md / inspection)  |
|--------------------|------:|-------:|--------------------------------------|
| `blip25-sstp`      |     4 |    682 | Soft-symbol transport protocol — wire format between edges and decoders |
| `blip25-core`      |    28 | 13 326 | P25 protocol logic core              |
| `blip25-dsp`       |    12 |  4 601 | DDC, demod, symbol timing, frame sync |
| `blip25-edge`      |    14 |  8 203 | RF capture + DSP for Pi/Airspy, channelization, NID classification |
| `blip25-vocoder`   |     6 |  1 769 | Voice decode + call lifecycle (uses blip25-mbe) |
| `blip25-trunking`  |     4 |  1 489 | TSBK / ISP decode, planned fuller |
| `blip25-data`      |     3 |    983 | PDU / SNDCP / LRRP / CAD             |
| `blip25-events`    |     3 |  1 215 | Canonical event envelope (`blip25.v1`) |
| `blip25-server`    |     1 |    534 | SSTP multiplexer (distributed mode)  |
| `blip25-scanner`   |     1 |  1 201 | Self-contained binary, watch + follow modes |
| `ambe-client`      |     3 |    624 | TCP client for DVSI AMBE3000 chip server |

Per the CLAUDE.md in p25-decoder, the four "(planned)" labels are
`blip25-trunking`, `blip25-vocoder`, `blip25-data`, and `blip25-scanner`.
Three of those have non-trivial code already; the fourth (data) has a stub.
"Planned" reads as "drafted, not finished."

## The integration boundary

```
          ┌──────────────────────────┐    ┌────────────────────────────┐
RF / IQ   │  blip25-edge / -dsp /     │    │   blip25-mbe (this repo)    │
──────────► -core (DDC, demod,        │    │                            │
          │  C4FM/CQPSK/H-CPM,        │    │  - p25_fullrate {decode,    │
          │  symbol timing, frame     │    │      encode, dequant}        │
          │  sync, soft symbols)      │    │  - p25_halfrate {decode,    │
          └─────────┬─────────────────┘    │      encode, dequant}        │
                    │                       │  - codecs::ambe_plus2,      │
                    │ NID-classified        │      ambe_plus, ambe,        │
                    │ frames + soft         │      mbe_baseline (synth)    │
                    │ payloads               │  - mbe_params::MbeParams    │
                    ▼                       └────────┬───────────────────┘
          ┌──────────────────────────┐               │
          │  blip25-trunking         │               │
          │  (TSBK / MBT / OSP /     │               │
          │   ISP / procedures)      │               │
          └─────────┬────────────────┘               │
                    │                                │
                    │ voice burst frames             │
                    ▼                                ▼
          ┌──────────────────────────┐    ┌────────────────────────────┐
          │  blip25-vocoder          │◄───┤ delegates IMBE/AMBE+2       │
          │  (call lifecycle, LC,    │    │ {decode_frame,              │
          │   TDMA dispatch, IMBE/   │    │  dequantize,                │
          │   AMBE+2 wrapping)       │    │  synthesize_frame}          │
          └─────────┬────────────────┘    └────────────────────────────┘
                    │
                    │ PCM + events
                    ▼
          ┌──────────────────────────┐
          │  blip25-server / -scanner│
          │  (event stream, audio)   │
          └──────────────────────────┘
```

The boundary is clean: `blip25-vocoder` consumes 18-byte FEC frames
(full-rate) or 9-byte FEC frames (half-rate) from upstream and calls
into blip25-mbe. blip25-mbe knows nothing about the air interface,
trunking, sync, or RF.

## What's left (suggested ordering)

### Wave 1 — close the codec integration loop (1–2 sessions each)

1. **Fix blip25-mbe half-rate gotchas surfaced this session.**
   - `MbeParams::silence_halfrate()` constructor (or make
     `MbeParams::silence()` half-rate-compatible).
   - `encode_halfrate` returns `Ok(Silence)` instead of
     `Err(PitchOutOfRange)` on out-of-range pitch.

   These let `blip25-vocoder` use the analysis path symmetrically
   between rates.

2. **AMBE+ (Gen 2) dequantize wrapper.** If the user's NXDN frames turn out
   to be Gen-2 AMBE+ (legacy NXDN/DMR), wire a 4-info-vector dequantize
   that targets the AMBE+ parameter encoding rather than AMBE+2's.
   Synth side (`codecs::ambe_plus`) already exists.

3. **Decoder-side `RawHalfrate` / `RawMbe` entry points in blip25-vocoder.**
   The CLI form already lives in blip25-mbe's
   `conformance-speech-quality`; `blip25-vocoder` should expose the
   same as a library API for `blip25-scanner` consumers who have
   non-P25 AMBE+2 sources.

### Wave 2 — finish the planned p25-decoder crates (multi-session each)

4. **`blip25-trunking` — TSBK and procedures.** The current 1 489 LOC is
   probably TSBK decode; add the trunking procedures (registration,
   affiliation, grants, channel assignment) to drive the scanner's
   `follow` mode from a control channel.

5. **`blip25-data` — PDU / SNDCP / LRRP / CAD.** The 983-LOC stub needs
   PDU assembly (gap reports 0005–0018 in
   `~/blip25-specs/gap_reports/` cover most of the wire-side
   ambiguities the spec-author has worked through). LRRP and CAD-text
   extraction are specific user-visible payloads that justify the
   investment.

6. **`blip25-scanner` — `follow` mode.** The 1 201-LOC stub looks like
   the `watch` mode for now; `follow` requires the trunking state
   machine from #4.

### Wave 3 — polish / not-yet-justified

7. **Phase 2 H-CPM/H-DQPSK demod.** Memory entry
   `project_chip_unblock_session_2026-04-23` mentions Phase 2 work but
   it's unclear whether `blip25-edge` and `blip25-dsp` cover H-CPM
   demod; if not, a Phase 2 receive path is significant DSP work.

8. **Encryption path (TIA-102.AAAD-B).** Block encryption + ESS handling.
   Spec is in derived form already
   (`P25_Block_Encryption_Protocol_Implementation_Spec.md`). Useful
   for monitoring P25 systems that emit encrypted traffic, but the
   keys are typically not available — value is detection + framing,
   not actual decrypt.

9. **TIA-102 link-layer authentication (AACE-A).** Specific to user
   needs; likely deprioritize unless monitoring authenticated systems.

## Open decisions

- **Single repo or split?** Currently p25-decoder is one workspace
  with blip25-mbe as a path dependency. Works fine. Splitting
  blip25-mbe out (it already lives in its own repo) lets it serve
  non-P25 consumers (NXDN, DMR users) independently. **Suggest:**
  keep current arrangement; blip25-mbe is already independently
  publishable.

- **Clean-room team applies to p25-decoder?** Yes — same blip25-vocoder
  team pattern (`spec-author` reads PDFs, `implementer` reads only
  derived works) transfers directly. Memory
  `project_clean_room_team_2026-04-16` notes this. The user is the
  merge gate for spec updates.

- **License consistency?** blip25-mbe is MIT (per `LICENSE`);
  p25-decoder is GPL-3.0 (per `Cargo.toml`). Different licenses are
  fine across a workspace boundary but worth flagging for downstream
  consumers — they take blip25-mbe under MIT and the p25-decoder
  binaries under GPL. No conflict; just visibility.

- **`blip25-edge` Phase 2 readiness.** Need a quick read on whether
  the existing 8 203 LOC covers H-CPM/H-DQPSK or only Phase 1's C4FM
  / CQPSK. This determines whether half-rate end-to-end testing on
  real RF needs DSP work first.

- **Test data for non-RF testing.** blip25-mbe has DVSI vectors;
  p25-decoder has 24 IQ captures (`AirSpy_*.wav`) and a stockpile of
  AMBE dumps (`ambe_dump_ch*_*_ts*.txt`). The dumps look like the raw
  AMBE+2 voice payloads exactly the kind `decode-raw-mbe` consumes —
  worth running through to verify cross-pipeline parity.

## What this plan is NOT

- Not a redesign of either repo's architecture. The architecture is
  fine; the work is filling in the planned crates and tightening the
  integration boundary.
- Not a commit-able artifact for p25-decoder. If the user wants this
  formalized in p25-decoder, copy it there; this lives in blip25-mbe
  because that's where the boundary view is.
- Not a schedule. Sessions are rough sizes ("1–2 sessions each"); the
  user can sequence as priority demands.

## Recommended first concrete cut

If the user wants to start actioning this immediately, the smallest
useful sequence is:

1. **Half-rate gotcha fix in blip25-mbe** (Wave 1 #1) — small commits,
   immediately useful for the AMBE dump cross-pipeline check.
2. **Run the existing AMBE dumps through `decode-raw-mbe`** — validates
   the new entry point against real off-air data.
3. **Pivot to whichever planned p25-decoder crate has the highest user
   value** — likely `blip25-data` (LRRP / CAD text) for monitoring
   value, or `blip25-trunking` procedures if `follow` mode is wanted.

Wave 1 #1 + a real-data run yields a concrete deliverable in one
session and tees up Wave 2 cleanly.
