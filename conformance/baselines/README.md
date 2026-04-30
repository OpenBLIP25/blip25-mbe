# conformance/baselines

Tracked PESQ / STOI / SNR baselines for regression spotting on
encoder/decoder changes.

## chip_ab_2026-04-30.csv

5-vector full-rate IMBE chip A/B baseline at HEAD `a93c996`. Wall time
~10 min via realtime DVSI chip oracle on `pve` (see
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
