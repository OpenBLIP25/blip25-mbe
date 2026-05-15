#!/usr/bin/env bash
# Orchestrate the halfrate 2-D b̂₀ sweep against the AMBE-3000R chip on pve.
# Reads cells from <out_dir>/cell_*.ambe9, sends each to chip via ssh,
# fetches per-cell chip_*.pcm. After this finishes, run
# `halfrate_2d_b0_sweep score <out_dir>` to produce results.csv.
#
# Usage: run_halfrate_2d_b0_sweep.sh <out_dir> [--start N] [--end N]
#   --start/--end limit to a cell range (useful for resuming).

set -e

OUT_DIR="$1"; shift || true
START=0
END=999999
while [ $# -gt 0 ]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --end)   END="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
    echo "usage: $0 <out_dir>" >&2; exit 2
fi
if [ -z "${CHIP_PASS:-}" ]; then
    CHIP_PASS='mpqz1010'
fi
CHIP_HOST="${CHIP_HOST:-root@192.168.1.6}"

cells=$(ls "$OUT_DIR" | grep -E '^cell_[0-9]+\.ambe9$' | sort)
total=$(echo "$cells" | wc -l)
i=0
started_at=$(date +%s)
for f in $cells; do
    idx_str="${f#cell_}"; idx_str="${idx_str%.ambe9}"
    idx=$((10#$idx_str))
    if [ $idx -lt $START ]; then i=$((i+1)); continue; fi
    if [ $idx -gt $END ]; then break; fi
    out="$OUT_DIR/chip_${idx_str}.pcm"
    if [ -f "$out" ] && [ -s "$out" ]; then
        i=$((i+1)); continue
    fi
    sshpass -p "$CHIP_PASS" scp -q -o StrictHostKeyChecking=no \
        "$OUT_DIR/$f" "$CHIP_HOST:/tmp/c.ambe9"
    sshpass -p "$CHIP_PASS" ssh -q -o StrictHostKeyChecking=no "$CHIP_HOST" \
        '/usr/local/bin/dvsi decode-halfrate /tmp/c.ambe9 /tmp/c.pcm >/dev/null 2>&1'
    sshpass -p "$CHIP_PASS" scp -q -o StrictHostKeyChecking=no \
        "$CHIP_HOST:/tmp/c.pcm" "$out"
    i=$((i+1))
    if [ $((i % 25)) -eq 0 ]; then
        elapsed=$(( $(date +%s) - started_at ))
        per_cell=$(( elapsed / i ))
        eta=$(( per_cell * (total - i) ))
        echo "  $i/$total done, ${elapsed}s elapsed, ~${per_cell}s/cell, eta ${eta}s"
    fi
done
elapsed=$(( $(date +%s) - started_at ))
echo "done — $i cells in ${elapsed}s"
