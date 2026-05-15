#!/usr/bin/env python3
"""Fit candidate-#1 per-bin amplitude lowpass models to the 2-D b̂₀ sweep
chip data.

Inputs: manifest.csv, results.csv, params/params_NNNNN.csv per cell.
Output: per-model goodness-of-fit + fitted parameters.

Models tried:
  M0  (no clamp): predicted_ratio = sqrt(sum(M̄_trig²))/sqrt(sum(M̄_trig²)) = 1
  M1  (index min): per-harmonic index clamp, M̄_clamp[l] = min(M̄_trig[l], K · M̄_low[l])
  M2  (index EMA): per-harmonic index EMA, M̄_clamp[l] = α · M̄_trig[l] + (1-α) · M̄_low[l]
  M3  (freq min): frequency-aligned per-bin clamp on a fixed DFT-like grid
  M4  (freq EMA): frequency-aligned per-bin EMA
  M5  (smooth-down only): like M1 but applied only when M̄_trig > prev
  M6  (smooth-both): smooth both ↑ and ↓ via 1-tap IIR

The "observed_ratio" we're fitting is chip_rms_trigger / chip_rms_pre vs
ours_rms_trigger / ours_rms_pre — i.e., the EXCESS attenuation. We
predict it from M̄ vectors only (no synthesis simulation).
"""
import csv
import sys
from pathlib import Path
import math

import numpy as np
from scipy.optimize import minimize_scalar, minimize


L_MAX = 56
SAMPLES_PER_FRAME = 160
N_BINS = 128  # DFT-like frequency grid bins from 0 to π


def load_results(out_dir):
    rows = {}
    with open(out_dir / "results.csv") as f:
        for r in csv.DictReader(f):
            rows[int(r["cell_idx"])] = {
                "low_b0": int(r["low_b0"]),
                "high_b0": int(r["high_b0"]),
                "b2": int(r["b2"]),
                "low_l": int(r["low_l"]),
                "high_l": int(r["high_l"]),
                "delta_l": int(r["delta_l"]),
                "delta_omega": float(r["delta_omega"]),
                "cos_sim": float(r["cos_sim"]),
                "chip_rms": float(r["chip_rms"]),
                "ours_rms": float(r["ours_rms"]),
                "chip_pre_rms": float(r["chip_pre_rms"]),
                "ours_pre_rms": float(r["ours_pre_rms"]),
            }
    return rows


def load_params_frame(out_dir, cell_idx, frame_num):
    """Return (omega_0, l, M̄[1..56])."""
    with open(out_dir / "params" / f"params_{cell_idx:05d}.csv") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if int(r["frame"]) == frame_num:
                l = int(r["l"])
                amps = np.zeros(L_MAX)
                for k in range(1, L_MAX + 1):
                    amps[k - 1] = float(r[f"M{k}"])
                return float(r["omega_0"]), l, amps
    return None


def freq_bin_envelope(omega_0, l, amps):
    """Place harmonics M̄_1..M̄_l on an N_BINS frequency grid via
    nearest-bin assignment. Returns an N_BINS-length array of squared
    energy per bin."""
    env = np.zeros(N_BINS)
    for l_h in range(1, l + 1):
        freq = l_h * omega_0  # rad/sample
        if freq >= math.pi:
            break
        # Map to bin: bin k spans [k·π/N, (k+1)·π/N]
        k = int(freq * N_BINS / math.pi)
        if 0 <= k < N_BINS:
            env[k] += amps[l_h - 1] ** 2
    return env


def predict_index_min(low_l, low_amps, trig_l, trig_amps, K):
    """M1: per-harmonic-index clamp. predicted_ratio² = sum(min²)/sum(trig²)."""
    n = min(low_l, trig_l)
    clamped_sq = 0.0
    trig_sq = 0.0
    for i in range(trig_l):
        if i < n:
            v = min(trig_amps[i], K * low_amps[i])
        else:
            v = trig_amps[i]  # new harmonic, no clamp
        clamped_sq += v * v
        trig_sq += trig_amps[i] ** 2
    return math.sqrt(clamped_sq / max(trig_sq, 1e-12))


def predict_index_ema(low_l, low_amps, trig_l, trig_amps, alpha):
    """M2: per-harmonic-index EMA blend."""
    n = min(low_l, trig_l)
    blend_sq = 0.0
    trig_sq = 0.0
    for i in range(trig_l):
        if i < n:
            v = alpha * trig_amps[i] + (1 - alpha) * low_amps[i]
        else:
            v = trig_amps[i]
        blend_sq += v * v
        trig_sq += trig_amps[i] ** 2
    return math.sqrt(blend_sq / max(trig_sq, 1e-12))


def predict_freq_min(low_env, trig_env, K):
    """M3: frequency-aligned per-bin clamp on squared energies."""
    clamped = np.minimum(trig_env, K * K * low_env)
    return math.sqrt(clamped.sum() / max(trig_env.sum(), 1e-12))


def predict_freq_ema(low_env, trig_env, alpha):
    """M4: frequency-aligned per-bin EMA on squared energies (sqrt to get amplitude domain)."""
    # Convert sq env to amplitude, blend, square back
    low_amp = np.sqrt(low_env)
    trig_amp = np.sqrt(trig_env)
    blend = alpha * trig_amp + (1 - alpha) * low_amp
    return math.sqrt((blend ** 2).sum() / max(trig_env.sum(), 1e-12))


def fit_model(cells, predict_fn, param_range, label):
    """Fit a 1-parameter model. Minimize sum((predicted_ratio - observed_ratio)²)."""
    def loss(p):
        residuals = []
        for c in cells:
            pred = predict_fn(c, p)
            obs = c["chip_target_ratio"]  # chip's residual ratio = chip_at_trigger / spec-faithful baseline
            residuals.append((pred - obs) ** 2)
        return sum(residuals) / len(residuals)
    res = minimize_scalar(loss, bounds=param_range, method="bounded")
    return res.x, res.fun


def compute_target_ratio(c):
    """Observed ratio we want to predict: chip_rms_trigger / ours_rms_trigger.

    This isolates the chip's beyond-spec attenuation (ours_rms is the
    spec-faithful baseline). Below 1.0 means chip clamps more than ours.
    """
    if c["ours_rms"] <= 0:
        return 1.0
    return c["chip_rms"] / c["ours_rms"]


def main():
    out_dir = Path(sys.argv[1])
    results = load_results(out_dir)
    print(f"Loaded {len(results)} cells")

    # Pre-load frame 49 and frame 50 params for each cell
    print("Loading per-cell params...")
    cells = []
    for ci, r in sorted(results.items()):
        f49 = load_params_frame(out_dir, ci, 49)
        f50 = load_params_frame(out_dir, ci, 50)
        if not f49 or not f50:
            continue
        omega_lo, l_lo, amps_lo = f49
        omega_hi, l_hi, amps_hi = f50
        env_lo = freq_bin_envelope(omega_lo, l_lo, amps_lo)
        env_hi = freq_bin_envelope(omega_hi, l_hi, amps_hi)
        r["amps_lo"] = amps_lo
        r["amps_hi"] = amps_hi
        r["l_lo"] = l_lo
        r["l_hi"] = l_hi
        r["env_lo"] = env_lo
        r["env_hi"] = env_hi
        r["chip_target_ratio"] = compute_target_ratio(r)
        cells.append(r)
    print(f"  {len(cells)} cells with full params")

    off_diag = [c for c in cells if c["low_b0"] != c["high_b0"]]
    print(f"  {len(off_diag)} off-diagonal cells")

    # Baseline: predict ratio = 1.0 (no clamp)
    baseline_loss = sum((1.0 - c["chip_target_ratio"]) ** 2 for c in off_diag) / len(off_diag)
    obs_var = np.var([c["chip_target_ratio"] for c in off_diag])
    print(f"\nBaseline (predict 1.0, off-diag): MSE = {baseline_loss:.5f}, observed_var = {obs_var:.5f}")

    # Model M1: index-min clamp
    def m1(c, K): return predict_index_min(c["l_lo"], c["amps_lo"], c["l_hi"], c["amps_hi"], K)
    k1, l1 = fit_model(off_diag, m1, (0.1, 5.0), "M1 index-min")
    r2_1 = 1 - l1 / obs_var
    print(f"\nM1 (index-min): K={k1:.3f}, MSE={l1:.5f}, R²={r2_1:.3f}")

    # Model M2: index EMA
    def m2(c, a): return predict_index_ema(c["l_lo"], c["amps_lo"], c["l_hi"], c["amps_hi"], a)
    a2, l2 = fit_model(off_diag, m2, (0.0, 1.0), "M2 index-EMA")
    r2_2 = 1 - l2 / obs_var
    print(f"M2 (index-EMA): α={a2:.3f}, MSE={l2:.5f}, R²={r2_2:.3f}")

    # Model M3: freq-min
    def m3(c, K): return predict_freq_min(c["env_lo"], c["env_hi"], K)
    k3, l3 = fit_model(off_diag, m3, (0.1, 5.0), "M3 freq-min")
    r2_3 = 1 - l3 / obs_var
    print(f"M3 (freq-min):  K={k3:.3f}, MSE={l3:.5f}, R²={r2_3:.3f}")

    # Model M4: freq-EMA
    def m4(c, a): return predict_freq_ema(c["env_lo"], c["env_hi"], a)
    a4, l4 = fit_model(off_diag, m4, (0.0, 1.0), "M4 freq-EMA")
    r2_4 = 1 - l4 / obs_var
    print(f"M4 (freq-EMA):  α={a4:.3f}, MSE={l4:.5f}, R²={r2_4:.3f}")

    # Report per-cell residuals for the best model
    best = min([("M1", k1, l1, m1), ("M2", a2, l2, m2),
                ("M3", k3, l3, m3), ("M4", a4, l4, m4)],
               key=lambda x: x[2])
    print(f"\nBest model: {best[0]} with param={best[1]:.3f}, MSE={best[2]:.5f}")

    print(f"\nSample predictions (10 random off-diag cells):")
    print("  cell  low_b0 high_b0 b2  obs    pred  err")
    import random
    rng = random.Random(42)
    sample = rng.sample(off_diag, min(10, len(off_diag)))
    for c in sample:
        pred = best[3](c, best[1])
        obs = c["chip_target_ratio"]
        print(f"  {c['low_b0']:4d}→{c['high_b0']:3d} b2={c['b2']:2d}  obs={obs:.3f}  pred={pred:.3f}  err={pred-obs:+.3f}")


if __name__ == "__main__":
    main()
