#!/usr/bin/env python3
"""Analyze the lever_sweep results.csv: encoder-cell PESQ delta vs the
`head` baseline, plus RMS-ratio (our_enc_chip_dec vs chip_enc_chip_dec)
for level/loudness levers that PESQ normalizes away.

Usage: analyze_levers.py [results.csv] [sweep_root]
"""
import sys, csv, os, glob
from collections import defaultdict
import numpy as np
from scipy.io import wavfile

ROOT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lever_sweep"
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results.csv")

# pesq[(label,vector,method)] = pesq_nb
pesq = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            pesq[(r["label"], r["vector"], r["method"])] = float(r["pesq_nb"])
        except ValueError:
            pass

labels = sorted({k[0] for k in pesq})
# Auto-detect vectors present (preserve a sensible order), put 'head' first.
_order = ["clean", "dam", "mark", "fambf22c", "fambm22a", "tambf22a", "tambf32b",
          "tambf32e", "tambf22b", "damcar", "cpvbad"]
_present = {k[1] for k in pesq}
vectors = [v for v in _order if v in _present] + sorted(_present - set(_order))
if "head" in labels:
    labels = ["head"] + [l for l in labels if l != "head"]
ENC = "our_enc_chip_dec"   # encoder isolation
BAR = "chip_enc_chip_dec"  # the bar

def rms(path):
    sr, x = wavfile.read(path)
    x = x.astype(float)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))

print("=" * 78)
print("ENCODER CELL (our_enc_chip_dec) PESQ-nb, and Δ vs head baseline")
print("=" * 78)
print(f"{'config':14s} " + " ".join(f"{v[:8]:>8s}" for v in vectors) + f" {'mean':>8s} {'Δmean':>8s}")
base = {v: pesq.get(("head", v, ENC)) for v in vectors}
basebar = {v: pesq.get(("head", v, BAR)) for v in vectors}
for lab in labels:
    cells = [pesq.get((lab, v, ENC)) for v in vectors]
    row = f"{lab:14s} "
    vals = []
    for v, c in zip(vectors, cells):
        if c is None:
            row += f"{'--':>8s} "
        else:
            row += f"{c:8.3f} "
            vals.append(c)
    m = np.mean(vals) if vals else float("nan")
    # delta vs head, averaged over vectors present in both
    dlist = [pesq[(lab, v, ENC)] - base[v] for v in vectors
             if base.get(v) is not None and (lab, v, ENC) in pesq]
    dm = np.mean(dlist) if dlist else float("nan")
    row += f"{m:8.3f} {dm:+8.3f}"
    print(row)

print()
print("Reference: head ENC mean = %.3f ; BAR (chip_enc_chip_dec) mean = %.3f ; gap = %+.3f"
      % (np.mean([base[v] for v in vectors if base[v]]),
         np.mean([basebar[v] for v in vectors if basebar[v]]),
         np.mean([base[v] for v in vectors if base[v]]) - np.mean([basebar[v] for v in vectors if basebar[v]])))

print()
print("=" * 78)
print("PER-VECTOR Δ vs head (encoder cell). + = lever helps; watch clean/dam for regressions")
print("=" * 78)
print(f"{'config':14s} " + " ".join(f"{v[:8]:>9s}" for v in vectors))
for lab in labels:
    if lab == "head":
        continue
    row = f"{lab:14s} "
    for v in vectors:
        if base.get(v) is not None and (lab, v, ENC) in pesq:
            row += f"{pesq[(lab, v, ENC)] - base[v]:+9.3f} "
        else:
            row += f"{'--':>9s} "
    print(row)

# RMS ratio for loudness levers (PESQ-blind). our_enc vs chip_enc, both chip-decoded.
print()
print("=" * 78)
print("RMS RATIO our_enc_chip_dec / chip_enc_chip_dec (target ~1.0 = chip loudness parity)")
print("PESQ normalizes level, so level_scale shows here, not above.")
print("=" * 78)
print(f"{'config':14s} " + " ".join(f"{v[:8]:>9s}" for v in vectors) + f" {'mean':>9s}")
for lab in labels:
    row = f"{lab:14s} "
    ratios = []
    for v in vectors:
        oe = os.path.join(ROOT, lab, v, "our_enc_chip_dec.wav")
        ce = os.path.join(ROOT, lab, v, "chip_enc_chip_dec.wav")
        if os.path.exists(oe) and os.path.exists(ce):
            r = rms(oe) / rms(ce) if rms(ce) > 0 else float("nan")
            row += f"{r:9.3f} "
            if not np.isnan(r):
                ratios.append(r)
        else:
            row += f"{'--':>9s} "
    row += f"{np.mean(ratios):9.3f}" if ratios else f"{'--':>9s}"
    print(row)
