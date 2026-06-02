#!/usr/bin/env python3
"""PESQ (narrowband + supplementary wideband) + STOI scoring over an
ab-matrix output directory.

Expects the layout produced by `conformance-speech-quality ab-matrix`:

    <dir>/ref.wav
    <dir>/our_enc_our_dec.wav
    <dir>/our_enc_chip_dec.wav     (optional)
    <dir>/chip_enc_our_dec.wav     (optional)
    <dir>/chip_enc_chip_dec.wav    (optional)
    <dir>/meta.json

Prints a CSV with columns:
    dir,method,pesq_nb,pesq_wb,stoi,snr_db,samples

`ref.wav` is the reference channel for PESQ/STOI; each `*.wav` that
exists is scored against it after peak-xcorr alignment and edge-trim
(40 ms on each side).

PESQ band note: the P25 IMBE / AMBE+2 vocoder produces 8 kHz audio
(4 kHz acoustic band), so **narrowband PESQ (ITU-T P.862) is the
apples-to-apples metric** and `pesq_nb` is the attribution basis.
Wideband PESQ (P.862.2) is only defined at 16 kHz, so `pesq_wb` is
computed on audio polyphase-resampled 8 kHz -> 16 kHz. Because the
4-8 kHz band is empty (and therefore matches the reference), `pesq_wb`
is a *softer*, optimistic score; it is reported for completeness only,
not used to attribute the encoder/decoder gap.

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
from scipy.signal import correlate, resample_poly
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


def score_pair(
    ref: np.ndarray, cand: np.ndarray, sr: int
) -> tuple[float, float, float, float]:
    ref_a, cand_a = align(ref, cand)
    ref_a = trim_edges(ref_a, sr)
    cand_a = trim_edges(cand_a, sr)
    # PESQ expects float32/int16 at 8k/16k.
    p_nb = float("nan")
    p_wb = float("nan")
    if sr == 8000:
        try:
            p_nb = float(pesq(8000, ref_a.astype(np.float32), cand_a.astype(np.float32), "nb"))
        except Exception as e:  # noqa: BLE001
            print(f"# PESQ(nb) error: {e}", file=sys.stderr)
        # Supplementary wideband: resample 8k -> 16k and score in wb mode.
        # 4-8 kHz band is empty so this is optimistic; reported for
        # completeness, not attribution. See module docstring.
        try:
            ref16 = resample_poly(ref_a, 2, 1)
            cand16 = resample_poly(cand_a, 2, 1)
            p_wb = float(pesq(16000, ref16.astype(np.float32), cand16.astype(np.float32), "wb"))
        except Exception as e:  # noqa: BLE001
            print(f"# PESQ(wb) error: {e}", file=sys.stderr)
    elif sr == 16000:
        try:
            p_wb = float(pesq(16000, ref_a.astype(np.float32), cand_a.astype(np.float32), "wb"))
        except Exception as e:  # noqa: BLE001
            print(f"# PESQ(wb) error: {e}", file=sys.stderr)
    else:
        print(f"# unsupported sample rate {sr} for PESQ", file=sys.stderr)
    try:
        s = float(stoi(ref_a, cand_a, sr, extended=False))
    except Exception as e:  # noqa: BLE001
        s = float("nan")
        print(f"# STOI error: {e}", file=sys.stderr)
    return p_nb, p_wb, s, snr_db(ref_a, cand_a)


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
        p_nb, p_wb, s, snr = score_pair(ref, cand, sr)
        rows.append(
            {
                "dir": str(d),
                "method": m,
                "pesq_nb": p_nb,
                "pesq_wb": p_wb,
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
    print("dir,method,pesq_nb,pesq_wb,stoi,snr_db,samples")
    for arg in sys.argv[1:]:
        rows = score_dir(Path(arg))
        for r in rows:
            print(
                f"{r['dir']},{r['method']},{r['pesq_nb']:.3f},{r['pesq_wb']:.3f},"
                f"{r['stoi']:.3f},{r['snr_db']:.2f},{r['samples']}"
            )


if __name__ == "__main__":
    main()
