"""
Step 7b: Figure 6 (macrophage balance) from the outputs of 8_macrophage_balance.py.

Run from the repository root after step 8. Saves Figures/Figure6 as PNG (300 dpi) and PDF.
Needs: pip install pandas numpy matplotlib
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "Figures")
MR = os.path.join(BASE, "macrophage_results")
W = 7.2
os.makedirs(OUT, exist_ok=True)

LABELS = {
    "C1Q_TAM": "C1Q-TAM score",
    "SPP1_TAM": "SPP1-TAM score",
    "C1Q_minus_SPP1": "C1Q minus SPP1",
    "SPP1": "SPP1 gene",
    "log2_CXCL9_over_SPP1": "log2 CXCL9/SPP1",
}
MODELS = [("unadjusted", "Unadjusted", "#BBBBBB"),
          ("stroma", "Stroma-adjusted", "#56B4E9"),
          ("stroma + total macrophages", "Stroma + total macrophages", "#0072B2")]


def forest_panel(path_png):
    d = pd.read_csv(os.path.join(MR, "tcga_adjusted_models.csv"))
    d = d[d["comparison"] == "EBV_vs_Neither"]
    metrics = [m for m in LABELS if m in set(d["metric"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for i, m in enumerate(metrics):
        for j, (mod, lab, col) in enumerate(MODELS):
            r = d[(d["metric"] == m) & (d["model"] == mod)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (j - 1) * 0.24
            ax.errorbar(r["effect_SD"], y, xerr=[[r["effect_SD"] - r["ci_low"]], [r["ci_high"] - r["effect_SD"]]],
                        fmt="o", color=col, ms=4, capsize=2, lw=1.2, label=lab if i == 0 else None)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([LABELS[m] for m in metrics], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("EBV vs double-negative effect (SD, 95% CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def ratio(p):
    h, w = mpimg.imread(p).shape[:2]
    return h / w


def main():
    tmp = os.path.join(OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    forest_png = os.path.join(tmp, "fig6c.png")
    forest_panel(forest_png)
    rows = [[("A", os.path.join(MR, "atlas_myeloid_umap.png"))],
            [("B", os.path.join(MR, "tcga_macrophage_balance.png"))],
            [("C", forest_png)]]
    widths = [1.0, 1.0, 0.8]
    heights = [W * wf * ratio(r[0][1]) for r, wf in zip(rows, widths)]
    total = sum(heights) + 0.3 * len(rows)
    fig = plt.figure(figsize=(W, total))
    top = 1.0
    for row, h, wf in zip(rows, heights, widths):
        lab, p = row[0]
        ax = fig.add_axes([0.02, top - (h + 0.22) / total, wf - 0.02, h / total])
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
        ax.text(-0.01, 1.0, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
        top -= (h + 0.3) / total
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT, f"Figure6.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Figure6")


if __name__ == "__main__":
    main()
