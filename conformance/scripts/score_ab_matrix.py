#!/usr/bin/env python3
"""PESQ (narrowband) + STOI scoring over an ab-matrix output directory.

Expects the layout produced by `conformance-speech-quality ab-matrix`:

    <dir>/ref.wav
    <dir>/our_enc_our_dec.wav
    <dir>/our_enc_chip_dec.wav     (optional)
    <dir>/chip_enc_our_dec.wav     (optional)
    <dir>/chip_enc_chip_dec.wav    (optional)
    <dir>/meta.json

Prints a CSV with columns:
    dir,method,pesq_nb,stoi,snr_db,samples

`ref.wav` is the reference channel for PESQ/STOI; each `*.wav` that
exists is scored against it after peak-xcorr alignment and edge-trim
(40 ms on each side).

Usage:
    score_ab_matrix.py <dir> [<dir> ...]

Run inside the blip25-eval venv:
    ~/.blip25-eval/bin/python3 conformance/scripts/score_ab_matrix.py /tmp/ab_clean
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from pesq import pesq
from pystoi import stoi

EDGE_TRIM_MS = 40
METHODS = [
    "our_enc_our_dec",
    "our_enc_chip_dec",
    "chip_enc_our_dec",
    "chip_enc_chip_dec",
]


def load_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data.astype(np.float64)


def align(ref: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ref, cand cropped to a common length after peak-xcorr lag
    correction. Lag is estimated on a centered window of the reference
    so an edge-heavy alignment doesn't dominate the correlation.
    """
    n = min(len(ref), len(cand))
    if n < 4000:
        return ref[:n], cand[:n]
    # Use a 2-second centered window of the reference for the
    # cross-correlation probe.
    mid = n // 2
    win = min(8000 * 2, n // 2)
    r = ref[mid - win // 2 : mid + win // 2]
    c = cand[mid - win // 2 : mid + win // 2]
    xc = correlate(c, r, mode="full")
    lag = int(np.argmax(xc) - (len(r) - 1))
    # Positive lag = cand is delayed relative to ref. Shift cand left.
    if lag > 0:
        cand = cand[lag:]
    elif lag < 0:
        ref = ref[-lag:]
    m = min(len(ref), len(cand))
    return ref[:m], cand[:m]


def trim_edges(a: np.ndarray, sr: int) -> np.ndarray:
    k = int(sr * EDGE_TRIM_MS / 1000)
    if len(a) <= 2 * k:
        return a
    return a[k:-k]


def snr_db(ref: np.ndarray, cand: np.ndarray) -> float:
    err = ref - cand
    num = float(np.sum(ref * ref))
    den = float(np.sum(err * err))
    if den <= 0:
        return float("inf")
    return 10.0 * np.log10(num / den) if num > 0 else float("-inf")


def score_pair(ref: np.ndarray, cand: np.ndarray, sr: int) -> tuple[float, float, float]:
    ref_a, cand_a = align(ref, cand)
    ref_a = trim_edges(ref_a, sr)
    cand_a = trim_edges(cand_a, sr)
    # PESQ expects float32/int16 at 8k/16k.
    try:
        p = float(pesq(sr, ref_a.astype(np.float32), cand_a.astype(np.float32), "nb"))
    except Exception as e:  # noqa: BLE001
        p = float("nan")
        print(f"# PESQ error: {e}", file=sys.stderr)
    try:
        s = float(stoi(ref_a, cand_a, sr, extended=False))
    except Exception as e:  # noqa: BLE001
        s = float("nan")
        print(f"# STOI error: {e}", file=sys.stderr)
    return p, s, snr_db(ref_a, cand_a)


def score_dir(d: Path) -> list[dict]:
    ref_path = d / "ref.wav"
    if not ref_path.exists():
        print(f"# {d}: no ref.wav, skipping", file=sys.stderr)
        return []
    sr, ref = load_mono(ref_path)
    rows = []
    for m in METHODS:
        cand_path = d / f"{m}.wav"
        if not cand_path.exists():
            continue
        sr_c, cand = load_mono(cand_path)
        if sr_c != sr:
            print(f"# {d}/{m}.wav: sample-rate mismatch {sr_c} vs {sr}", file=sys.stderr)
            continue
        p, s, snr = score_pair(ref, cand, sr)
        rows.append(
            {
                "dir": str(d),
                "method": m,
                "pesq_nb": p,
                "stoi": s,
                "snr_db": snr,
                "samples": len(cand),
            }
        )
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print("dir,method,pesq_nb,stoi,snr_db,samples")
    for arg in sys.argv[1:]:
        rows = score_dir(Path(arg))
        for r in rows:
            print(
                f"{r['dir']},{r['method']},{r['pesq_nb']:.3f},{r['stoi']:.3f},"
                f"{r['snr_db']:.2f},{r['samples']}"
            )


if __name__ == "__main__":
    main()
