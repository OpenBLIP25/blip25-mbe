# blip25-codec — the blip25 MBE codec engine

`blip25-codec` (lib `blip25_codec`) is the self-contained codec engine that the
public [`blip25-mbe`](https://crates.io/crates/blip25-mbe) `Vocoder` API
dispatches to. **Depend on `blip25-mbe`, not on this crate** — this is an
implementation detail with no API stability promise. It implements
both P25 vocoders, in both directions, with **no external path dependencies**:

| Vocoder | Decode | Encode | Public entry |
|---|---|---|---|
| AMBE+2 (P25 Phase 2, half-rate r33/r34) | ✅ float `synth` + ✅ fixed-point `dec::*` | ✅ | `Decoder`, `Encoder` |
| IMBE (P25 Phase 1, full-rate) | ✅ | ✅ (`imbe::quantize_frame`) | `ImbeDecoder`, `imbe::*` |

## Self-contained

- The only dependency is the in-repo `blip25-codebooks` crate, whose VQ
  codebooks are `static [i16; N]` arrays compiled into the binary.
- The normative spec tables (Annex L/M/N/S + bit prioritization + the encoder
  amplitude/band tables) are frozen into source at `src/tables_generated.rs`
  and `include!`d by `src/tables.rs`. There is no build script and no
  build-time data.
- No fixtures, no `#[cfg(test)]` data blobs, no external includes.
  `grep -rE '/mnt/|\.\./' src/` returns nothing.

## Layout

The pipeline is split into three grouped module trees that mirror one another:

- `dec::*` — decoder: info bits → dequantized `MbeParams` → fixed-point OLA
  synthesis → 8 kHz i16 PCM (excitation, M_l generation, linear amplitude,
  voiced OLA, unvoiced synthesis).
- `enc::*` — encoder: 8 kHz i16 PCM → analysis (pitch, voicing, spectral) →
  quantization → frame bits. The exact inverse of the decode front end.
- `imbe::*` — the IMBE (Phase 1) front end, sharing the same synthesis back end.

## Correctness gate

The engine's output is pinned **byte/sample-identical** through the public
façade by the hermetic acceptance test, for all four rates (AMBE+2 r33/r34,
IMBE full/info):

```
cargo test -p blip25-mbe --test codec_acceptance
```

The hermetic tier runs with no external data; the corpus tier additionally
checks a representative set of real vectors when the reference `tv-rc` tree is
reachable (`BLIP25_TVRC_DIR`). Any change that makes the default `Vocoder`
output diverge from the engine fails this gate.
