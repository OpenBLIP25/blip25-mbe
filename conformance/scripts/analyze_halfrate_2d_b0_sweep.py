#!/usr/bin/env python3
"""Analyse halfrate 2-D b̂₀ sweep results — characterise the AMBE-3000R
chip's spectral-discontinuity attenuation (gap 0026 follow-up probe #2).

Inputs: results.csv from `halfrate_2d_b0_sweep score`.
Outputs:
  - results_augmented.csv with chip_atten, ours_atten, excess_atten columns
  - scatter_*.png: excess_atten vs (|ΔL|, Δω̃₀, cos_sim, ΔL/L_lo, etc.)
  - report.txt with Spearman/Pearson correlations and summary statistics
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr


def load(path):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            row = {k: float(v) if k not in {"cell_idx"} else int(v) for k, v in r.items()}
            rows.append(row)
    return rows


def derive(rows):
    out = []
    for r in rows:
        chip_atten = 1.0 - (r["chip_rms"] / r["chip_pre_rms"]) if r["chip_pre_rms"] > 0 else 0.0
        ours_atten = 1.0 - (r["ours_rms"] / r["ours_pre_rms"]) if r["ours_pre_rms"] > 0 else 0.0
        excess = chip_atten - ours_atten
        r["chip_atten"] = chip_atten
        r["ours_atten"] = ours_atten
        r["excess_atten"] = excess
        out.append(r)
    return out


def save_augmented(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def scatter(rows, x_key, y_key, out_path, title, x_label):
    x = np.array([r[x_key] for r in rows])
    y = np.array([r[y_key] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 6))
    # Color by b2 if available
    if "b2" in rows[0]:
        b2 = np.array([r["b2"] for r in rows])
        unique_b2 = sorted(set(b2.tolist()))
        for v in unique_b2:
            m = b2 == v
            ax.scatter(x[m], y[m], s=18, alpha=0.5, label=f"b̂₂={int(v)}")
        ax.legend()
    else:
        ax.scatter(x, y, s=18, alpha=0.5)
    rho, _ = spearmanr(x, y)
    rp, _ = pearsonr(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_key)
    ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}, Pearson r = {rp:.3f}")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return rho, rp


def report(rows, out_dir):
    keys = [
        ("delta_l", "|ΔL|"),
        ("delta_omega", "|Δω̃₀| (rad/sample)"),
        ("cos_sim", "cos_sim(M̄_lo, M̄_hi)"),
    ]
    lines = []
    lines.append(f"# Halfrate 2-D b̂₀ sweep analysis ({len(rows)} cells)\n")
    lines.append(f"chip_atten = 1 - chip_rms / chip_pre_rms\n")
    lines.append(f"ours_atten = 1 - ours_rms / ours_pre_rms\n")
    lines.append(f"excess_atten = chip_atten - ours_atten  (positive ⇒ chip drops more than us at trigger)\n\n")

    lines.append(f"## Aggregate stats\n")
    for k in ["chip_atten", "ours_atten", "excess_atten"]:
        arr = np.array([r[k] for r in rows])
        lines.append(f"  {k}: mean={arr.mean():.3f}, median={np.median(arr):.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}\n")

    lines.append(f"\n## Correlations against excess_atten\n")
    for x_key, x_label in keys:
        x = np.array([r[x_key] for r in rows])
        y = np.array([r["excess_atten"] for r in rows])
        rho, _ = spearmanr(x, y)
        rp, _ = pearsonr(x, y)
        lines.append(f"  vs {x_label}: Spearman ρ = {rho:.3f}, Pearson r = {rp:.3f}\n")

    # Off-diagonal subset (cells with low_b0 != high_b0) — most informative
    off_diag = [r for r in rows if r["low_b0"] != r["high_b0"]]
    lines.append(f"\n## Off-diagonal only ({len(off_diag)} cells, low_b0 != high_b0)\n")
    for x_key, x_label in keys:
        x = np.array([r[x_key] for r in off_diag])
        y = np.array([r["excess_atten"] for r in off_diag])
        rho, _ = spearmanr(x, y)
        rp, _ = pearsonr(x, y)
        lines.append(f"  vs {x_label}: Spearman ρ = {rho:.3f}, Pearson r = {rp:.3f}\n")

    # Top-10 strongest-clamp cells
    lines.append(f"\n## Top-10 strongest clamp (highest excess_atten)\n")
    sorted_rows = sorted(off_diag, key=lambda r: -r["excess_atten"])
    lines.append("  cell  low_b0 high_b0 b2 low_l high_l |ΔL| Δω̃₀     cos_sim  chip_at ours_at excess\n")
    for r in sorted_rows[:10]:
        lines.append(
            f"  {int(r['cell_idx']):4d}  {int(r['low_b0']):6d} {int(r['high_b0']):7d} {int(r['b2']):2d} "
            f"{int(r['low_l']):5d} {int(r['high_l']):6d} {int(r['delta_l']):4d} {r['delta_omega']:.4f}   "
            f"{r['cos_sim']:.4f}  {r['chip_atten']:.3f}  {r['ours_atten']:.3f}  {r['excess_atten']:.3f}\n"
        )

    Path(out_dir).joinpath("report.txt").write_text("".join(lines))
    print("".join(lines))


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_halfrate_2d_b0_sweep.py <out_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    rows = load(out_dir / "results.csv")
    rows = derive(rows)
    save_augmented(rows, out_dir / "results_augmented.csv")

    off_diag = [r for r in rows if r["low_b0"] != r["high_b0"]]
    plots = [
        ("delta_l", "|ΔL|", "vs_delta_l.png"),
        ("delta_omega", "|Δω̃₀| (rad/sample)", "vs_delta_omega.png"),
        ("cos_sim", "cos_sim(M̄_lo, M̄_hi)", "vs_cos_sim.png"),
    ]
    for x_key, x_label, fname in plots:
        scatter(off_diag, x_key, "excess_atten",
                out_dir / fname,
                "Excess attenuation at trigger frame", x_label)
    # Heatmap: excess_atten in (low_b0, high_b0) coords, faceted by b2.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, b2v in zip(axes, sorted({int(r["b2"]) for r in rows})):
        sub = [r for r in rows if int(r["b2"]) == b2v]
        if not sub: continue
        lows = sorted({int(r["low_b0"]) for r in sub})
        highs = sorted({int(r["high_b0"]) for r in sub})
        grid = np.full((len(lows), len(highs)), np.nan)
        for r in sub:
            i = lows.index(int(r["low_b0"]))
            j = highs.index(int(r["high_b0"]))
            grid[i, j] = r["excess_atten"]
        im = ax.imshow(grid, origin="lower",
                       extent=[highs[0], highs[-1], lows[0], lows[-1]],
                       aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        ax.set_xlabel("high_b̂₀")
        ax.set_ylabel("low_b̂₀")
        ax.set_title(f"excess_atten, b̂₂={b2v}")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_excess_atten.png", dpi=110)
    plt.close(fig)

    report(rows, out_dir)


if __name__ == "__main__":
    main()
