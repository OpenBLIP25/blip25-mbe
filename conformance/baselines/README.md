# conformance/baselines

Tracked PESQ / STOI / SNR baselines for regression spotting on
encoder/decoder changes.

## our_enc_our_dec_2026-05-14.csv

5-vector full-rate IMBE `our_enc_our_dec` baseline at HEAD `b369845`
with the **new library defaults**: `spectral_subtraction` ON,
`enhancement = Classical(default)`. Net +0.045 PESQ vs the
2026-04-30 baseline mean. No chip cells (use 2026-04-30 CSV for the
chip side — it hasn't moved). Speech: clean 3.221, dam 3.341,
mark 2.737. Tones: alert 2.578, knox_1 1.647.

## chip_ab_2026-04-30.csv

5-vector full-rate IMBE chip A/B baseline at HEAD `a93c996` with the
**old defaults** (spec-faithful: no enhancement, no spectral
subtraction). Kept as a reference for the chip cells and as a
historical snapshot of the spec-faithful encoder. Wall time ~10 min
via realtime DVSI chip oracle on `pve` (see
`reference_dvsi_chip_access.md` in user memory).

Columns: `vector,method,pesq_nb,stoi,snr_db,samples`. Methods cover
the four encoder/decoder cells (`{ours,chip}_enc_{ours,chip}_dec`).

To re-run and compare:

```bash
mkdir -p /tmp/ab_check
for v in clean dam mark alert knox_1; do
  ./target/release/conformance-speech-quality ab-matrix \
    --pcm DVSI/Vectors/tv-std/tv/$v.pcm \
    --out-dir /tmp/ab_check/$v \
    --chip-host root@192.168.1.6
done
~/.blip25-eval/bin/python3 conformance/scripts/score_ab_matrix.py \
  /tmp/ab_check/{clean,dam,mark,alert,knox_1}
```

A drop in `our_enc_our_dec` PESQ on speech (clean/dam/mark) below the
values in this CSV indicates an encoder regression worth investigating.
Half-rate paths and tone-frame paths are tracked separately in
project memory (`project_chip_ab_baseline_2026-04-30.md`,
`project_detect_single_tone_lgt1_fix_2026-04-30.md`).

## cross_rate_2026-05-15.csv

First DVSI cross-rate transcode comparison baseline. Generated via the
new harness command:

```bash
target/release/conformance-vectors cross-rate-compare \
  --csv conformance/baselines/cross_rate_2026-05-15.csv
```

Drives the 36 P25-only `-rc -rd N -re M` lines in `tv-rc/cmprc.txt` (12
directional pairs × 3 stems: `alltone`, `dam`, `sine0_4k`). Each pair
is run through `Transcoder` directly when a direct hop exists; diagonal
pairs compose 2–3 hops via BFS over Transcoder-supported edges.

Headline results at HEAD:

| Pair                       | Frames-exact | Bytes-exact | Notes |
|----------------------------|-------------:|------------:|-------|
| `imbe_fec ↔ imbe_nofec`    | **100.00%**  | **100.00%** | Annex H FEC strip/add validated against DVSI |
| `r33_fec ↔ r34_nofec`      | 0.00%        | ~6%         | **Mismatch** — our AMBE+2 info-only packing differs from DVSI's r34 layout |
| `imbe ↔ r33` / `imbe ↔ r34`| ~0%          | 1–10%       | Cross-codec parameter-domain conversion; quantizer rounding diverges (expected) |

A drop in either `imbe_fec ↔ imbe_nofec` cell signals a regression in
the IMBE wire layer. The `r33 ↔ r34` and cross-codec cells are tracked
for future investigation, not as pass/fail gates.
