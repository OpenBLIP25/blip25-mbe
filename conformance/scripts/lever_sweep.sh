#!/usr/bin/env bash
# lever_sweep.sh — encode→chip-decode→PESQ feedback loop for encode-quality levers.
#
# Usage:
#   lever_sweep.sh LABEL FRAMES "VEC1 VEC2 ..." "--extra --harness --flags"
#
# Runs `conformance-speech-quality halfrate-ab-matrix` per vector (driving the
# live AMBE-3000R on $CHIP_HOST), scores all 4 cells with score_ab_matrix.py,
# and appends tagged rows to $ROOT/results.csv. The two cells that matter for
# encoder attribution are:
#   our_enc_chip_dec  — OUR encoder, chip decoder  (encoder isolation)
#   chip_enc_chip_dec — the bar (chip end-to-end)
set -u
ROOT=${ROOT:-/tmp/lever_sweep}
CHIP_HOST=${CHIP_HOST:-root@192.168.1.6}
BIN=./target/release/conformance-speech-quality
PY=~/.blip25-eval/bin/python3
SCORER=conformance/scripts/score_ab_matrix.py
VECDIR=$HOME/blip25-mbe/DVSI/Vectors/tv-rc

LABEL=$1; FRAMES=$2; VECS=$3; EXTRA=${4:-}
mkdir -p "$ROOT"
[ -f "$ROOT/results.csv" ] || echo "label,vector,method,pesq_nb,pesq_wb,stoi,snr_db,samples" > "$ROOT/results.csv"

for v in $VECS; do
  pcm="$VECDIR/$v.pcm"
  if [ ! -f "$pcm" ]; then echo "MISSING $pcm" >&2; continue; fi
  out="$ROOT/$LABEL/$v"; mkdir -p "$out"
  echo ">>> [$LABEL] $v  frames=$FRAMES  flags='$EXTRA'" >&2
  $BIN halfrate-ab-matrix --pcm "$pcm" --out-dir "$out" \
       --chip-host "$CHIP_HOST" --frames "$FRAMES" $EXTRA >"$out/run.log" 2>&1
  if [ $? -ne 0 ]; then echo "  HARNESS FAIL (see $out/run.log)" >&2; tail -3 "$out/run.log" >&2; continue; fi
  # score; tag rows with label+vector
  $PY "$SCORER" "$out" 2>>"$out/score.log" | tail -n +2 | while IFS=, read -r dir method nb wb stoi snr samp; do
    echo "$LABEL,$v,$method,$nb,$wb,$stoi,$snr,$samp" >> "$ROOT/results.csv"
  done
done
echo "<<< [$LABEL] done" >&2
