#!/usr/bin/env python3
"""Add interference to a clean PCM at a target SNR, for chip-vs-ours
robustness A/B. Produces 8 kHz mono i16 raw PCM (same format the harness
reads). Noise types: white, babble (sum of time-shifted copies of the
signal = self-babble), tone (1 kHz CW), hum (60/120 Hz).

Usage: make_noisy.py <clean.pcm> <out.pcm> <type> <snr_db>
"""
import sys
import numpy as np

clean_path, out_path, ntype, snr_db = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
x = np.fromfile(clean_path, dtype="<i2").astype(np.float64)
n = len(x)
sig_pow = np.mean(x * x) + 1e-9

# deterministic noise (seeded) so runs are reproducible
rng = np.random.default_rng(1234)
if ntype == "white":
    nz = rng.standard_normal(n)
elif ntype == "babble":
    # self-babble: sum of several random circular shifts of the signal
    nz = np.zeros(n)
    for _ in range(6):
        s = int(rng.integers(1000, n - 1000)) if n > 2000 else 1
        nz += np.roll(x, s)
elif ntype == "tone":
    t = np.arange(n) / 8000.0
    nz = np.sin(2 * np.pi * 1000.0 * t)
elif ntype == "hum":
    t = np.arange(n) / 8000.0
    nz = np.sin(2 * np.pi * 60.0 * t) + 0.5 * np.sin(2 * np.pi * 120.0 * t)
else:
    raise SystemExit(f"unknown noise type {ntype}")

nz_pow = np.mean(nz * nz) + 1e-9
# scale noise to target SNR
target_nz_pow = sig_pow / (10 ** (snr_db / 10.0))
nz *= np.sqrt(target_nz_pow / nz_pow)
y = x + nz
# clip to i16
y = np.clip(np.round(y), -32768, 32767).astype("<i2")
y.tofile(out_path)
ach = 10 * np.log10(sig_pow / (np.mean((y.astype(float) - x) ** 2) + 1e-9))
print(f"{out_path}: {ntype} @ {snr_db} dB (achieved {ach:.1f} dB), {n} samples")
