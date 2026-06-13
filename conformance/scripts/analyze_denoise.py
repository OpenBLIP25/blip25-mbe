#!/usr/bin/env python3
"""Analyze the §3.4 denoiser exceed sweep (denoise_sweep.sh output).

EXCEED margin = PESQ(our_enc+denoise → chip_dec, vs clean)
              − PESQ(chip_enc → chip_dec, vs clean)
A positive margin means our denoised encode beats the (NS-free) chip
roundtrip on noisy input. Also reports the denoise-vs-head delta (does the
front-end help our own encoder?) and the clean-input transparency control.

Usage: analyze_denoise.py [results.csv]
"""
import sys, csv
from collections import defaultdict
import statistics as st

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/denoise_sweep/results.csv"
ENC = "our_enc_chip_dec"
BAR = "chip_enc_chip_dec"

# v[(arm,cond,method)] = pesq_nb
v = {}
conds = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try:
            v[(r["arm"], r["cond"], r["method"])] = float(r["pesq_nb"])
        except ValueError:
            pass
        if r["cond"] not in conds:
            conds.append(r["cond"])

print("=" * 86)
print("§3.4 DENOISER EXCEED (PESQ-nb vs CLEAN reference)")
print("=" * 86)
print(f"{'condition':12s} {'chipBAR':>8s} {'head_ENC':>9s} {'dn_ENC':>8s} "
      f"{'dn−head':>8s} {'EXCEED':>8s}")
print("-" * 86)
margins, helps = [], []
for c in conds:
    bar = v.get(("head", c, BAR)) or v.get(("denoise", c, BAR))
    head = v.get(("head", c, ENC))
    dn = v.get(("denoise", c, ENC))
    if bar is None or head is None or dn is None:
        print(f"{c:12s}  (incomplete)")
        continue
    dvh = dn - head
    exceed = dn - bar
    print(f"{c:12s} {bar:8.3f} {head:9.3f} {dn:8.3f} {dvh:+8.3f} {exceed:+8.3f}")
    if c.lower().startswith(("clean", "cleant")) and c.lower() in ("cleant", "clean"):
        pass  # transparency control handled below
    margins.append((c, exceed))
    helps.append((c, dvh))

print("-" * 86)
noisy = [m for (c, m) in margins if c.lower() not in ("cleant", "clean")]
noisy_help = [d for (c, d) in helps if c.lower() not in ("cleant", "clean")]
if noisy:
    print(f"noisy conditions: mean EXCEED {st.mean(noisy):+.3f} "
          f"(min {min(noisy):+.3f}, max {max(noisy):+.3f}); "
          f"wins {sum(1 for m in noisy if m > 0)}/{len(noisy)}")
    print(f"noisy denoise-vs-head: mean {st.mean(noisy_help):+.3f} "
          f"(helps our encoder on {sum(1 for d in noisy_help if d > 0)}/{len(noisy_help)})")
# transparency
for c, dvh in helps:
    if c.lower() in ("cleant", "clean"):
        print(f"CLEAN transparency (cond {c}): denoise−head = {dvh:+.3f} (want ~0; >−0.02 OK)")
