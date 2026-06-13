#!/usr/bin/env bash
# denoise_sweep.sh — §3.4 exceed-lever measurement: denoised noisy input
# vs the chip roundtrip, scored against the CLEAN reference.
#
# For each "BASE NOISE" condition it runs two arms on the noisy PCM:
#   head    — no denoise (our encoder, chip decode)
#   denoise — +pre-analysis denoiser
# both with --ref-pcm <clean base>, so score_ab_matrix.py scores vs CLEAN.
# Appends to $ROOT/results.csv. Exceed = our_enc_chip_dec − chip_enc_chip_dec.
set -u
ROOT=${ROOT:-/tmp/denoise_sweep}
CHIP_HOST=${CHIP_HOST:-root@192.168.1.6}
FRAMES=${FRAMES:-400}
MODE=${MODE:-logmmse}
BIN=./target/release/conformance-speech-quality
PY=~/.blip25-eval/bin/python3
SCORER=conformance/scripts/score_ab_matrix.py
VECDIR=$HOME/blip25-mbe/DVSI/Vectors/tv-rc
NOISYDIR=${NOISYDIR:-/tmp/noisy}
mkdir -p "$ROOT"
[ -f "$ROOT/results.csv" ] || echo "arm,cond,method,pesq_nb,pesq_wb,stoi,snr_db,samples" > "$ROOT/results.csv"

run() {  # run ARM COND NOISYPCM CLEANREF EXTRA
  local arm=$1 cond=$2 noisy=$3 clean=$4 extra=$5
  local out="$ROOT/$arm/$cond"; mkdir -p "$out"
  echo ">>> [$arm] $cond frames=$FRAMES" >&2
  $BIN halfrate-ab-matrix --pcm "$noisy" --ref-pcm "$clean" --out-dir "$out" \
       --chip-host "$CHIP_HOST" --frames "$FRAMES" $extra >"$out/run.log" 2>&1
  if [ $? -ne 0 ]; then echo "  FAIL $out"; tail -3 "$out/run.log" >&2; return; fi
  $PY "$SCORER" "$out" 2>>"$out/score.log" | tail -n +2 | while IFS=, read -r d m nb wb st snr samp; do
    echo "$arm,$cond,$m,$nb,$wb,$st,$snr,$samp" >> "$ROOT/results.csv"
  done
}

# conditions: "label base noisyfile"
for spec in "$@"; do
  set -- $spec; cond=$1; base=$2; nf=$3
  noisy="$NOISYDIR/$nf.pcm"; clean="$VECDIR/$base.pcm"
  [ -f "$noisy" ] || { echo "MISSING $noisy" >&2; continue; }
  run head    "$cond" "$noisy" "$clean" ""
  run denoise "$cond" "$noisy" "$clean" "--denoise --denoise-mode $MODE"
done
echo "<<< denoise sweep done" >&2
