# Disposition: NOT A BUG — SDRTrunk reference contains post-synth gain

**Status:** Closed, no code change. Routing the loudness parity to the consumer layer.

## Why

The bug report frames the gap against `reference_voice_ts2.wav`, which is
SDRTrunk-synthesized PCM. SDRTrunk applies post-vocoder gain (AGC / fixed
multiplier, depending on build) on top of the AMBE+2 synth output. The
reference file is therefore not a DVSI-equivalent ground truth for the
vocoder boundary — it is SDRTrunk's rendering pipeline end-to-end.

Our calibration target is the **canonical DVSI test-vector PCM**
(`DVSI/Vectors/tv-rc/r33/*.pcm`), which is what the DVSI AMBE-3000R chip
emits directly. Measured on that reference today (2026-04-23):

| Vector | DVSI RMS | Ours RMS | Ratio  |
|--------|---------:|---------:|-------:|
| clean  |   1283.9 |   1285.0 | 1.001× |

This reproduces the 2026-04-19 five-vector audit (alert, clean, cp0, cp1,
cp31 — all within ±6.7% of DVSI). We are at parity with DVSI chip output.
Fitting `γ_w` to SDRTrunk would push it to ≈1200, well past the spec
bound of 146.64 from §1.12.1, and would break BABA-A conformance.

## Half-rate AMBE+2 (P25 Phase 2) — same conclusion (2026-05-13)

A symmetric measurement on **half-rate** (AMBE+2 / P25 Phase 2 TDMA)
against the live AMBE-3000R chip running rate-table index 33
(`PKT_RATET 33`, the half-rate path) extended this finding to the
second codec path. Six field-collected Phase 2 calls were decoded through
both the chip and our codec from the same `.ambe9` byte streams:

| Source | Full-file RMS | Peak | Per-frame chip/ours ratio (median) |
|--------|--------------:|-----:|------------------------------------:|
| Chip   |        506.6  | 5545 |                              0.99× |
| Ours   |        511.1  | 6409 |                              1.00× |
| JMBE   |       3479.4  |31128 |                              6.87× |

Chip ≈ Ours at 0.99× (within −0.08 dB). **JMBE renders the half-rate
path 16.7 dB louder than the actual DVSI chip emits.** That extra gain
comes from JMBE's `DifferentialGain.java` per-bin `mAdjustment` table
(+1.0 .. +1.55 log₂ offset added at gain decode), which the JMBE
maintainer's own comment describes as "adjustment to increase gain
level of generated audio that more closely matches output gain of
hardware generated audio." The actual hardware does *not* emit at JMBE's
levels — JMBE's calibration overshoots.

The same SDRTrunk decision applies: half-rate consumers wanting
JMBE-equivalent loudness should layer a constant gain in the rendering
pipeline (the `BLIP25_VOICE_GAIN_DB=+9` consumer default brings us
partway; +17 dB would match JMBE on average), **not** modify
`compute_m_tilde` or `decode_gain` in `ambe_plus2_wire/dequantize.rs`.

Gap report `~/blip25-specs/gap_reports/0024_*.md` documents the JMBE
adjustment table as a JMBE-side UX choice, not chip-emulation.

### PCM round-trip sanity check on `tv-rc/r33/clean.bit`

A PCM-in / PCM-out round-trip must preserve loudness (within codec
quantization loss). Decoding the DVSI-supplied half-rate test vector
`Research/DVSI Vectors/tv-rc/r33/clean.bit` (3700 chip-encoded
9-byte frames produced from `tv-rc/clean.pcm`) through each path:

| Source                            | RMS    | Peak   | vs input PCM |
|-----------------------------------|-------:|-------:|-------------:|
| Input PCM (`tv-rc/clean.pcm`)     | 1291.6 | 22 266 |    +0.00 dB  |
| DVSI r33 reference output (oracle)| 1283.9 | 19 005 |    −0.05 dB  |
| Live chip decode (AMBE-3000R)     | 1283.9 | 19 005 |    −0.05 dB  |
| **Ours (native, no enhancement)** |**1261.1**|**20 816**|**−0.21 dB** |
| JMBE 1.0.9 decode                 | 5114.7 | 31 128 |   **+11.95 dB** |

Live chip ≡ DVSI oracle (byte-identical, both at −0.05 dB vs input).
Ours sits 0.16 dB quieter than the chip — within codec quantization
loss and exactly consistent with the per-call measurements above.
JMBE outputs 4× the input PCM amplitude, which is structurally
impossible for a conformant decode of a recorded-then-encoded signal.
(The +17 dB measured on the field calls is larger than +11.95 dB
here because JMBE's internal `0.95 × Short.MAX_VALUE` clipper truncates
the loud transients on this longer test file; the field-call analysis
captured the pre-clip ratio.)

Repro:

```
cp '/mnt/share/P25-IQ-Samples/Research/DVSI Vectors/tv-rc/r33/clean.bit' /tmp/tv.bit
cargo run --release --bin conformance-speech-quality -- \
    decode-fec-halfrate --binary --input /tmp/tv.bit --out-wav /tmp/tv_ours.wav
java -cp "$JMBE_CP:conformance/scripts" \
    JMBEDecodeAMBE /tmp/tv.bit /tmp/tv_jmbe.wav
ssh root@192.168.1.6 'dvsi decode-halfrate /tmp/tv.bit /tmp/tv_chip.pcm'
```

## Reproduce

```
cargo run -p blip25-conformance-vectors --release -- \
    decode-pcm-ambe-plus2 clean --write-pcm /tmp/ours.pcm
# compare RMS vs DVSI/Vectors/tv-rc/r33/clean.pcm — expect ≈1.0×
```

## Recommended fix for the consumer

Loudness parity with SDRTrunk / JMBE / other open-source P25 decoders is a
UX property of the rendering pipeline, not a codec constant. Apply a gain
multiplier on the PCM output in the consumer crate (the call site that
feeds playback / WAV writing), **not** in `codecs::ambe_plus2`. The
measured SDRTrunk factor on `expected_ts2_ambe_hex.txt` is ≈9×, but that
number is specific to SDRTrunk's build and not a spec quantity.

A reasonable consumer-side shape:

```rust
// in blip25-edge (or wherever PCM leaves the vocoder boundary):
const SDRTRUNK_PARITY_GAIN: f32 = 9.0; // tune to target playback stack
for s in pcm.iter_mut() {
    *s = (*s as f32 * SDRTRUNK_PARITY_GAIN).clamp(-32768.0, 32767.0) as i16;
}
```

Do not hoist this into the codec crate. If a future consumer wants DVSI
chip-accurate levels (the spec-faithful default), they must not have that
gain baked in.

## Fixture handoff

`expected_ts2_ambe_hex.txt` and `reference_voice_ts2.wav` should **not**
move under `blip25-mbe/conformance/`. Our conformance references are the
DVSI-emitted PCM and the DVSI reference bit files (`*.bit` in
`DVSI/Vectors/tv-rc/r33/`). SDRTrunk-derived fixtures belong with the
consumer, where post-synth gain is applied.

## References

- Memory: `project_sdrtrunk_scale_decoy_2026-04-19.md` — original
  five-vector audit.
- Memory: `project_gamma_w_closed_2026-04-19.md` — γ_w=146.64
  verification, specs/main commit `406e1e0`.
- `INTEGRATION.md` — DVSI vs SDRTrunk loudness calibration section
  (added in commit `50845cd`).
