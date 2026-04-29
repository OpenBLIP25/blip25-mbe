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
