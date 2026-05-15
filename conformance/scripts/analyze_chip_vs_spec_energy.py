#!/usr/bin/env python3
"""Diagnostic plots: chip_atten vs spec-faithful M̄ energy ratio.

Tests whether chip behavior is explained by spec-faithful spectral
energy drop (candidate #1's natural macro): if yes, the §1.4 "beyond-spec
clamp" framing softens. If no, the residual locates the chip-specific
behavior.
"""
import csv
from pathlib import Path
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def amps_sum_sq(out_dir, cell, frame):
    with open(out_dir / "params" / f"params_{cell:05d}.csv") as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) == frame:
                l = int(r["l"])
                return sum(float(r[f"M{k}"]) ** 2 for k in range(1, l + 1))
    return 0.0


def main():
    out_dir = Path(sys.argv[1])
    with open(out_dir / "results.csv") as f:
        res = {int(r["cell_idx"]): {k: (float(v) if k != "cell_idx" else int(v))
                                     for k, v in r.items()} for r in csv.DictReader(f)}
    rows = []
    for ci, r in res.items():
        if r["low_b0"] == r["high_b0"]:
            continue
        chip_atten = 1 - r["chip_rms"] / r["chip_pre_rms"]
        ours_atten = 1 - r["ours_rms"] / r["ours_pre_rms"]
        pre_sq = np.mean([amps_sum_sq(out_dir, ci, f) for f in range(45, 50)])
        trig_sq = amps_sum_sq(out_dir, ci, 50)
        if pre_sq <= 0 or trig_sq <= 0:
            continue
        spec_atten = 1 - math.sqrt(trig_sq / pre_sq)
        rows.append({
            "cell": ci, "low_b0": int(r["low_b0"]), "high_b0": int(r["high_b0"]),
            "b2": int(r["b2"]),
            "chip_atten": chip_atten, "ours_atten": ours_atten, "spec_atten": spec_atten,
            "chip_residual": chip_atten - spec_atten,
            "ours_residual": ours_atten - spec_atten,
        })

    chip = np.array([r["chip_atten"] for r in rows])
    ours = np.array([r["ours_atten"] for r in rows])
    spec = np.array([r["spec_atten"] for r in rows])

    # Plot 1: chip_atten vs spec_atten scatter, with y=x diagonal
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y, label in [(axes[0], chip, "chip_atten"), (axes[1], ours, "ours_atten")]:
        ax.scatter(spec, y, s=8, alpha=0.5)
        lo, hi = min(spec.min(), y.min()) - 0.05, max(spec.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x (spec-faithful)")
        ax.set_xlabel("spec_atten = 1 - sqrt(sum(M̄²_trig)) / sqrt(sum(M̄²_pre))")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs spec-faithful M̄-energy attenuation")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "chip_vs_spec_atten.png", dpi=110)
    plt.close(fig)

    # Plot 2: heatmap of chip_residual in (low_b0, high_b0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, b2v in zip(axes, sorted({r["b2"] for r in rows})):
        sub = [r for r in rows if r["b2"] == b2v]
        lows = sorted({r["low_b0"] for r in sub})
        highs = sorted({r["high_b0"] for r in sub})
        grid = np.full((len(lows), len(highs)), np.nan)
        for r in sub:
            i = lows.index(r["low_b0"])
            j = highs.index(r["high_b0"])
            grid[i, j] = r["chip_residual"]
        im = ax.imshow(grid, origin="lower",
                       extent=[highs[0], highs[-1], lows[0], lows[-1]],
                       aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        ax.set_xlabel("high_b̂₀")
        ax.set_ylabel("low_b̂₀")
        ax.set_title(f"chip_residual = chip_atten - spec_atten, b̂₂={b2v}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_chip_residual.png", dpi=110)
    plt.close(fig)

    # Plot 3: ours_residual heatmap — diagnostic for our synthesis vs spec
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, b2v in zip(axes, sorted({r["b2"] for r in rows})):
        sub = [r for r in rows if r["b2"] == b2v]
        lows = sorted({r["low_b0"] for r in sub})
        highs = sorted({r["high_b0"] for r in sub})
        grid = np.full((len(lows), len(highs)), np.nan)
        for r in sub:
            i = lows.index(r["low_b0"])
            j = highs.index(r["high_b0"])
            grid[i, j] = r["ours_residual"]
        im = ax.imshow(grid, origin="lower",
                       extent=[highs[0], highs[-1], lows[0], lows[-1]],
                       aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        ax.set_xlabel("high_b̂₀")
        ax.set_ylabel("low_b̂₀")
        ax.set_title(f"ours_residual = ours_atten - spec_atten, b̂₂={b2v}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_ours_residual.png", dpi=110)
    plt.close(fig)

    print(f"Wrote chip_vs_spec_atten.png, heatmap_chip_residual.png, heatmap_ours_residual.png")
    # Save augmented CSV
    with open(out_dir / "results_with_spec.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote results_with_spec.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
