"""
Step 7: Assemble the five main figures from the outputs of steps 2-6.

Run from the repository root after steps 1-6. Saves Figures/Figure1..5 as PNG (300 dpi) and PDF.
Needs: pip install pandas numpy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT = "Figures"
W = 7.2                                   # full page width, inches
EBV, HP, NEI = "#0072B2", "#D55E00", "#8C8C8C"


def letter(ax, s):
    ax.text(-0.02, 1.0, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right")


def show(ax, path):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def ratio(path):
    h, w = mpimg.imread(path).shape[:2]
    return h / w


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def stack(name, rows):
    """rows: list of lists of (letter, image path); images in a row share the width."""
    heights = []
    for row in rows:
        ws = [1 / ratio(p) for _, p in row]                # widths at unit height
        heights.append(W / sum(ws))
    fig = plt.figure(figsize=(W, sum(heights) + 0.25 * len(rows)))
    top = 1.0
    total = sum(heights) + 0.25 * len(rows)
    for row, h in zip(rows, heights):
        ws = [1 / ratio(p) for _, p in row]
        x = 0.0
        for (lab, p), w in zip(row, ws):
            frac = w / sum(ws)
            ax = fig.add_axes([x + 0.01, top - (h + 0.2) / total, frac - 0.02, h / total])
            show(ax, p)
            letter(ax, lab)
            x += frac
        top -= (h + 0.25) / total
    save(fig, name)


def de_counts_panel(path_png):
    comps = [("EBV_vs_CIN", "EBV vs CIN"), ("EBV_vs_GS", "EBV vs GS"),
             ("EBV_vs_MSI", "EBV vs MSI"), ("EBV_vs_Neither", "EBV vs Neither"),
             ("HP_vs_Neither", "H. pylori vs Neither")]
    up, down, labels = [], [], []
    for f, lab in comps:
        d = pd.read_csv(f"de_results/{f}_all_genes.csv")
        s = d[(d["padj"] < 0.05) & (d["log2FoldChange"].abs() >= 1)]
        up.append((s["log2FoldChange"] > 0).sum())
        down.append((s["log2FoldChange"] < 0).sum())
        labels.append(lab)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.barh(y, up, color="#B2182B", height=0.6, label="Up")
    ax.barh(y, [-d for d in down], color="#2166AC", height=0.6, label="Down")
    for yi, u, d in zip(y, up, down):
        ax.text(u + 80, yi, f"{u:,}", va="center", fontsize=7)
        ax.text(-d - 80, yi, f"{d:,}", va="center", ha="right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="black", lw=0.8)
    lim = max(max(up), max(down)) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xticks([])
    ax.set_xlabel("Differentially expressed genes (padj < 0.05, |log2FC| ≥ 1)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def circuit_panel(path_png):
    c = pd.read_csv("tcga_sc_results/ifng_circuit_correlations.csv")
    genes = ["CXCL9", "CXCL10", "IDO1", "CD274"]
    groups = [("Neither", NEI), ("HP", HP), ("EBV", EBV)]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    x = np.arange(len(genes))
    for i, (g, col) in enumerate(groups):
        r = [c[(c.group == g) & (c.responder_gene == gn)]["spearman_r"].iloc[0] for gn in genes]
        n = int(c[c.group == g]["n"].iloc[0])
        label = f"{'H. pylori' if g == 'HP' else g} (n={n})"
        ax.bar(x + (i - 1) * 0.26, r, width=0.26, color=col, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman r with IFNG/GZMB/PRF1\nproducer score", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def ebv_score_panel(path_png):
    m = pd.read_csv("sc_results/cell_metadata.csv", usecols=["cell_type_detail",
                    "EBV_up_score", "EBV_down_score"])
    g = m.groupby("cell_type_detail")[["EBV_up_score", "EBV_down_score"]].mean()
    g = g.sort_values("EBV_up_score")
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 3.6), sharey=True)
    for ax, col, colr, title in [(axes[0], "EBV_up_score", "#B2182B", "EBV-up genes"),
                                 (axes[1], "EBV_down_score", "#2166AC", "EBV-down genes")]:
        ax.barh(range(len(g)), g[col], color=colr, height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(g)))
    axes[0].set_yticklabels(g.index, fontsize=7)
    fig.supxlabel("Mean module score per cell", fontsize=8)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def cox_forest_panel(path_png):
    c = pd.read_csv("survival_results/cox_models.csv")
    f = c[(c["endpoint"] == "OS") & (c["model"] == "+ age, sex, stage")].copy()
    f["score"] = f["score"].replace({"immune_core": "Combined immune score",
                                     "IFNg response": "IFN-\u03b3 response"})
    f = f.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    y = np.arange(len(f))
    ax.errorbar(f["HR_per_SD"], y, xerr=[f["HR_per_SD"] - f["ci_low"],
                f["ci_high"] - f["HR_per_SD"]], fmt="o", color=EBV, ms=4,
                elinewidth=1, capsize=0)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ticks = [0.7, 0.8, 1.0, 1.25, 1.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(f["score"], fontsize=7)
    ax.set_xlabel("Hazard ratio per 1 SD (95% CI)", fontsize=8)
    ax.set_title("Overall survival, adjusted for age, sex, stage", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{OUT}/_panels"
    os.makedirs(tmp, exist_ok=True)

    de_counts_panel(f"{tmp}/fig1a.png")
    stack("Figure1", [[("A", f"{tmp}/fig1a.png")],
                      [("B", "gsea_results/hallmark_NES_heatmap.png")]])

    stack("Figure2", [[("A", "tme_results/mcp_counter_boxplots.png")],
                      [("B", "tme_results/forest_EBV_vs_Neither.png"),
                       ("C", "tme_results/forest_HP_vs_Neither.png")]])

    ebv_score_panel(f"{tmp}/fig3c.png")
    stack("Figure3", [[("A", "sc_results/umap_overview.png")],
                      [("B", "sc_results/dotplot__key_EBV_genes.png"),
                       ("C", f"{tmp}/fig3c.png")]])

    circuit_panel(f"{tmp}/fig4b.png")
    plt.imsave(f"{tmp}/blank.png", np.ones((10, 6, 3)))
    stack("Figure4", [[("A", "tcga_sc_results/signature_effect_heatmap.png")],
                      [("B", f"{tmp}/fig4b.png"), ("", f"{tmp}/blank.png")]])

    cox_forest_panel(f"{tmp}/fig5d.png")
    stack("Figure5", [[("A", "survival_results/KM_OS_subtype.png"),
                       ("B", "survival_results/KM_PFS_subtype.png")],
                      [("C", "survival_results/KM_OS_IFNg.png"),
                       ("D", f"{tmp}/fig5d.png")]])
    print(f"Figures saved in ./{OUT}/")


if __name__ == "__main__":
    main()
