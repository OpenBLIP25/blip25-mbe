# Fuzz seed corpus

Hand-built starting inputs for the four `cargo fuzz` targets. These are
committed on purpose: CI runs each target for 60 seconds
(`.github/workflows/ci.yml`), and without seeds that entire budget goes into
rediscovering frame structure a fuzzer cannot reach by luck — a valid
Golay/Hamming FEC codeword, an Annex-T tone signature, a soft-decision LLR
frame with correct signs.

**Read the selector-byte layout below before adding or editing a seed.** It is
not the same in all four targets, and a wrong selector byte produces a file
that still "works" while fuzzing an entirely different rate than its name says.

## Input encoding, per target

Each target reinterprets the raw libFuzzer byte string differently.

### `decode_bits`

```
[sel] [frame 0][frame 1] … [frame n-1]
```

* `sel` — **one** byte. `rate = RATES[sel % 4]`.
* Payload is chunked at `rate.fec_frame_bytes()`; a short trailing chunk is
  passed through and rejected as `WrongBitsLength`.
* After the loop the target calls `reset()` and then `decode_bits(payload)` on
  the *whole* payload, so a single-frame seed also exercises the post-reset
  success path.

`RATES` order (`fuzz_targets/decode_bits.rs`):

| `sel % 4` | rate | frame bytes |
|---|---|---|
| 0 | `Imbe7200x4400` | 18 |
| 1 | `Imbe4400x4400` | 11 |
| 2 | `AmbePlus2_3600x2450` | 9 |
| 3 | `AmbePlus2_2450x2450` | 7 |

### `decode_soft`

```
[sel] [llr 0][llr 1] … [llr m-1]
```

* `sel` — **one** byte. `rate = RATES[sel % 4]`.
* Every remaining byte is one signed LLR: the byte is reinterpreted as `i8`
  (`0x64` = +100, `0x9C` = −100, `0x80` = `i8::MIN`, `0x7F` = `i8::MAX`).
* One LLR per FEC channel bit, in raw frame-bit order (`SD0` first) — the same
  MSB-first bit order as the hard `decode_bits` frame. Sign is the hard
  decision, magnitude is confidence.
* Chunked at `rate.soft_frame_bits()`. The two no-FEC rates return `None`
  there, and the target instead makes a single whole-payload call that must
  return `SoftUnsupported`.

**`RATES` order here is NOT the same as the other three — the middle two are
swapped** (`fuzz_targets/decode_soft.rs`):

| `sel % 4` | rate | LLRs per frame |
|---|---|---|
| 0 | `Imbe7200x4400` | 144 |
| 1 | `AmbePlus2_3600x2450` | 72 |
| 2 | `Imbe4400x4400` | — (no FEC) |
| 3 | `AmbePlus2_2450x2450` | — (no FEC) |

### `encode_pcm`

```
[sel] [sample 0 hi][sample 0 lo] [sample 1 hi][sample 1 lo] …
```

* `sel` — **one** byte. `rate = RATES[sel % 4]` (same order as `decode_bits`).
* Every remaining **pair** of bytes is one **big-endian** `i16` PCM sample. An
  odd trailing byte is dropped by `chunks_exact(2)`.
* Samples are chunked at 160 (`frame_samples()`), so one whole frame is 320
  payload bytes. A short trailing chunk is rejected as `WrongPcmLength`.
* The target then calls `encode_pcm(&pcm)` on the whole buffer (a length error
  unless the seed is exactly one frame) and `flush_encode()`.

### `transcode`

```
[sel_from][sel_to] [frame 0][frame 1] …
```

* **Two** selector bytes. `from = RATES[data[0] % 4]`, `to = RATES[data[1] % 4]`
  (same order as `decode_bits`).
* Payload is chunked at `from.fec_frame_bytes()`.
* `Transcoder::new` accepts only these six pairs; anything else returns early
  and the seed does nothing:
  `imbe7200→ambe3600`, `ambe3600→imbe7200`, `imbe7200→imbe4400`,
  `imbe4400→imbe7200`, `ambe3600→ambe2450`, `ambe2450→ambe3600`.

## What the seeds cover

Filenames carry the rate tag (`imbe7200`, `imbe4400`, `ambe3600`, `ambe2450`)
that the selector byte must resolve to, and `transcode` names are
`<from>_to_<to>`.

* `valid_*`, `single_*` — real frames from `Vocoder::encode` on a deterministic
  speech-like signal. The highest-value class: a fuzzer will not synthesise a
  valid FEC codeword by chance.
* `conceal_*` — a run of frames the decoder must conceal, long enough to walk
  repeat → mute. For the AMBE+2 rates that is `b̂₀` in the BABA-1 erasure range
  `[120,123]`; for IMBE it is a pitch index outside Annex-L range (`> 207`),
  which trips the same `consecutive_invalid ≥ 4` hysteresis. Good frames on
  both sides of the run, so recovery is exercised too.
* `fec_errors_*` — one valid frame with 0 / 1 / 3 / 7 flipped channel bits:
  Golay corrects, corrects, then fails.
* `tone_annex_t_*` — valid Annex-T tone frames (`û₀(11..6) == 0x3F` with the
  `û₃(3..0) == 0` trailer). Random bytes will not produce this signature, and
  it must **not** be read as an erasure.
* `zeros_*`, `ones_*` — all-`0x00` and all-`0xFF` payloads.
* `zero_llr_*` — all-zero LLRs, i.e. maximum ambiguity, the Chase-II search's
  worst case. `min_llr_*` / `max_llr_*` saturate at `i8::MIN` / `i8::MAX`.
* `valid_weak_*` — correct signs at magnitude 1; `rescuable_*` /
  `unrescuable_*` — sign errors soft decode can and cannot recover from.
* `nofec_rejected_*` — must return `SoftUnsupported`.
* PCM edge cases: `digital_silence_*`, `rail_full_positive_*`,
  `rail_full_negative_*`, `rail_alternating_*` (±full scale at Nyquist),
  `dc_positive_*` / `dc_negative_*`, `impulse_*`, `ramp_*`,
  `speech_silence_speech_*` (silence-gate transitions), `odd_length_*`
  (one whole frame plus a 90-sample tail), `selector_only_*` (empty payload).

## Regenerating

The seeds were produced by a throwaway generator built **outside** this repo —
a scratch crate with a path dependency on `blip25-mbe`, run once and deleted.
It is not committed: it is a one-shot data producer, not something the build
should carry, and leaving it in the tree would invite someone to "fix" the
corpus by re-running it instead of understanding why a seed changed.

To rebuild it, create a scratch crate outside the repo depending on
`blip25-mbe` and `blip25-codec` by path, and emit, per target:

* valid frames from `Vocoder::new(rate).encode(&pcm)` over a deterministic
  **integer-only** signal (no `libm`, no transcendentals — the input must be
  bit-identical on every target so only the codec is under test);
* erasure frames via `rate33::priority::prioritize` with `b[0] ∈ [120,123]`,
  then `rate33::frame::encode_frame`, packing the dibit stream MSB-first
  (2 bits per dibit) into bytes;
* IMBE invalid-pitch frames via `imbe7200::frame::encode_frame` with
  `û₀ = 0xFFF`, which decodes to `b₀ = 252 > 207`;
* tone frames from `û₀ = (0x3F << 6) | (A_D >> 1)` and
  `û₃ = (I_D << 5) | ((A_D & 1) << 4)`, checked with
  `blip25_codec::tone::classify` before writing;
* soft LLRs by expanding a hard frame MSB-first, bit 1 → `+mag`, bit 0 → `−mag`;
* no-FEC frames by running the FEC-bearing frame through
  `Transcoder::new(fec_rate, nofec_rate)` rather than re-implementing the
  packing.

Then verify by driving the same public API each target drives, asserting the
seed reaches the state it was written to reach — that erasure runs really do
report `FrameDisposition::Repeat` *and* `Mute`, that tone frames report `Use`,
that the selector byte still resolves to the rate in the filename. A seed that
silently stops reaching its target state is worse than no seed.

## Running

```sh
cd fuzz
cargo +nightly fuzz run decode_bits -- -runs=0            # replay seeds only
cargo +nightly fuzz run decode_bits -- -max_total_time=60 # what CI does
```

`cargo fuzz` writes its own discovered inputs back into these directories,
named by the SHA-1 of their contents. Those are gitignored (`.gitignore`
un-ignores only `*.bin`, which is what every committed seed is named), so a
local fuzzing session will not add noise to `git status`. Do not run
`cargo fuzz cmin` against these directories expecting the seeds to survive —
minimization rewrites the corpus and will drop seeds whose coverage is
subsumed, including ones kept here for the state they reach rather than the
edges they light up.
