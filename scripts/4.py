"""
Step 4: Tumour microenvironment composition + stroma-adjusted pathway scores
  Part A: MCP-counter scores for 10 cell populations in every tumour
          (T cells, CD8 T, cytotoxic, NK, B lineage, monocytic, myeloid DC,
           neutrophils, endothelial cells, fibroblasts)
  Part B: Compare HP / EBV / Neither  (Kruskal-Wallis + pairwise Mann-Whitney, BH-FDR)
  Part C: Hallmark pathway scores per tumour, then linear models asking:
          does the group effect survive after adjusting for stromal content
          (fibroblast + endothelial scores)?

Needs:   pip install requests pandas numpy scipy statsmodels gseapy matplotlib
Input:   tcga_stad_hp_ebv_status.csv, star_counts_cache/  (from steps 1-3)
Outputs: tme_results/  (scores, statistics, figures)

Groups are the same as step 2:
  EBV     = STAD_EBV subtype
  HP      = H. pylori "Yes" and CIN / GS / MSI subtype
  Neither = H. pylori "No"  and CIN / GS / MSI subtype

MCP-counter: Becht et al. 2016, Genome Biology 17:218. Marker list is read from
the authors' GitHub repository. Score = mean log2(TPM+1) of a population's markers.
"""

import io
import os
import json
import time
import tarfile
import requests
import numpy as np
import pandas as pd
import gseapy as gp
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GDC = "https://api.gdc.cancer.gov"
CACHE = "star_counts_cache"
OUT = "tme_results"
MCP_URL = ("https://raw.githubusercontent.com/ebecht/MCPcounter/master/"
           "Signatures/genes.txt")
NON_EBV = ["STAD_CIN", "STAD_GS", "STAD_MSI"]
GROUPS = ["Neither", "HP", "EBV"]
COLORS = {"Neither": "#8C8C8C", "HP": "#D55E00", "EBV": "#0072B2"}
PATHWAYS = [
    "Interferon Gamma Response", "Interferon Alpha Response",
    "Allograft Rejection", "Inflammatory Response",
    "IL-6/JAK/STAT3 Signaling", "TNF-alpha Signaling via NF-kB",
    "Epithelial Mesenchymal Transition", "Angiogenesis", "Hypoxia",
    "Coagulation", "Myogenesis", "Wnt-beta Catenin Signaling",
    "E2F Targets", "G2-M Checkpoint", "Myc Targets V1",
]


# ---------------------------------------------------------------
# Data loading (same logic as steps 2-3)
# ---------------------------------------------------------------
def gdc_count_files():
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": f, "value": [v]}} for f, v in [
            ("cases.project.project_id", "TCGA-STAD"),
            ("data_type", "Gene Expression Quantification"),
            ("analysis.workflow_type", "STAR - Counts"),
            ("cases.samples.sample_type", "Primary Tumor")]]}
    r = requests.get(f"{GDC}/files", params={
        "filters": json.dumps(filters),
        "fields": "file_id,cases.submitter_id,cases.samples.submitter_id",
        "format": "JSON", "size": 2000}, timeout=120)
    r.raise_for_status()
    rows = [{"patient": h["cases"][0]["submitter_id"],
             "sample": h["cases"][0]["samples"][0]["submitter_id"],
             "file_id": h["file_id"]} for h in r.json()["data"]["hits"]]
    return (pd.DataFrame(rows).sort_values("sample")
            .drop_duplicates("patient").set_index("patient"))


def download_missing(file_ids, batch_size=10):
    todo = [f for f in file_ids if not os.path.exists(f"{CACHE}/{f}.tsv")]
    if not todo:
        return
    print(f"Downloading {len(todo)} missing count files")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        for attempt in range(1, 6):
            try:
                resp = requests.post(f"{GDC}/data", json={"ids": batch},
                                     timeout=(30, 300))
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException:
                time.sleep(15 * attempt)
        else:
            raise SystemExit("Download kept failing. Run the script again.")
        if len(batch) == 1:
            open(f"{CACHE}/{batch[0]}.tsv", "wb").write(resp.content)
        else:
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if m.name.endswith(".tsv"):
                        open(f"{CACHE}/{m.name.split('/')[0]}.tsv", "wb").write(
                            tar.extractfile(m).read())


def load_groups():
    st = pd.read_csv("tcga_stad_hp_ebv_status.csv")
    st = st[st["has_rna"] == True].copy()
    st["group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "group"] = "EBV"
    st.loc[(st["hp"] == "Yes") & st["subtype"].isin(NON_EBV), "group"] = "HP"
    st.loc[(st["hp"] == "No") & st["subtype"].isin(NON_EBV), "group"] = "Neither"
    return st.dropna(subset=["group"])


def build_log_tpm(patients, files):
    """Genes (symbols) x samples, log2(TPM + 1), protein-coding only."""
    cols = {}
    for p in patients:
        t = pd.read_csv(f"{CACHE}/{files.loc[p, 'file_id']}.tsv",
                        sep="\t", comment="#")
        t = t[t["gene_type"] == "protein_coding"]
        cols[p] = t.groupby("gene_name")["tpm_unstranded"].sum()
    tpm = pd.DataFrame(cols)
    return np.log2(tpm + 1)


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------
def mcp_counter(logtpm):
    mk = pd.read_csv(MCP_URL, sep="\t")
    mk.columns = [c.strip('"') for c in mk.columns]
    scores, used = {}, []
    for pop, genes in mk.groupby("Cell population")["HUGO symbols"]:
        present = [g for g in genes if g in logtpm.index]
        used.append({"population": pop, "markers_total": len(genes),
                     "markers_found": len(present),
                     "markers": ";".join(present)})
        if present:
            scores[pop] = logtpm.loc[present].mean(axis=0)
    return pd.DataFrame(scores), pd.DataFrame(used)


def pathway_scores(logtpm, names):
    """Mean z-score of each Hallmark set's genes (z across all tumours)."""
    lib = gp.get_library("MSigDB_Hallmark_2020", organism="Human")
    expr = logtpm[logtpm.mean(axis=1) > 1]            # drop barely expressed genes
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1) + 1e-9, axis=0)
    out = {}
    for n in names:
        genes = [g for g in lib.get(n, []) if g in z.index]
        if len(genes) >= 10:
            out[n] = z.loc[genes].mean(axis=0)
        else:
            print(f"  skipped '{n}' (only {len(genes)} genes found)")
    return pd.DataFrame(out)


# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
def compare_groups(scores, groups):
    rows = []
    for col in scores.columns:
        vals = {g: scores.loc[groups == g, col].dropna() for g in GROUPS}
        kw_p = kruskal(*vals.values()).pvalue
        for a, b in [("HP", "Neither"), ("EBV", "Neither"), ("EBV", "HP")]:
            p = mannwhitneyu(vals[a], vals[b], alternative="two-sided").pvalue
            rows.append({"feature": col, "comparison": f"{a}_vs_{b}",
                         f"median_A": vals[a].median(), "median_B": vals[b].median(),
                         "difference": vals[a].median() - vals[b].median(),
                         "p_mannwhitney": p, "p_kruskal": kw_p})
    res = pd.DataFrame(rows)
    res["fdr"] = multipletests(res["p_mannwhitney"], method="fdr_bh")[1]
    return res


def stroma_adjusted(path, mcp, meta):
    """Group effect on each pathway score, before and after stromal adjustment.
    Scores are standardised, so coefficients are in SD units."""
    d = pd.concat([path, mcp[["Fibroblasts", "Endothelial cells"]], meta], axis=1)
    d = d.rename(columns={"Fibroblasts": "fibro", "Endothelial cells": "endo"})
    for c in ["fibro", "endo"]:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    rows = []
    for pw in path.columns:
        d["y"] = (d[pw] - d[pw].mean()) / d[pw].std()
        for target, base in [("HP", "+ C(subtype)"), ("EBV", "")]:
            sub = d[d["group"].isin([target, "Neither"])].copy()
            sub["is_target"] = (sub["group"] == target).astype(int)
            for label, extra in [("unadjusted", ""),
                                 ("stroma_adjusted", " + fibro + endo")]:
                fit = smf.ols(f"y ~ is_target {base}{extra}", data=sub).fit()
                ci = fit.conf_int().loc["is_target"]
                rows.append({"pathway": pw, "comparison": f"{target}_vs_Neither",
                             "model": label, "effect_SD": fit.params["is_target"],
                             "ci_low": ci[0], "ci_high": ci[1],
                             "p": fit.pvalues["is_target"], "n": len(sub)})
    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for (comp, model), idx in res.groupby(["comparison", "model"]).groups.items():
        res.loc[idx, "fdr"] = multipletests(res.loc[idx, "p"], method="fdr_bh")[1]
    return res


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
def boxplots(mcp, groups, stats, fname):
    cols = list(mcp.columns)
    nc = 5
    nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.2 * nr))
    rng = np.random.default_rng(1)
    for ax, col in zip(axes.flat, cols):
        data = [mcp.loc[groups == g, col] for g in GROUPS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.35)
            patch.set_edgecolor(COLORS[g])
        for med in bp["medians"]:
            med.set_color("black")
        for i, (g, v) in enumerate(zip(GROUPS, data), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7,
                       color=COLORS[g], alpha=0.7, linewidths=0)
        s = stats[stats["feature"] == col].set_index("comparison")["fdr"]
        ax.set_title(f"{col}\nHP vs N: q={s['HP_vs_Neither']:.2g} | "
                     f"EBV vs N: q={s['EBV_vs_Neither']:.2g}", fontsize=8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(GROUPS, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.supylabel("MCP-counter score (mean log2 TPM+1)", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def forest(adj, comparison, fname):
    d = adj[adj["comparison"] == comparison]
    order = d[d["model"] == "unadjusted"].sort_values("effect_SD")["pathway"].tolist()
    y = {p: i for i, p in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(order) + 1.2))
    for model, off, color in [("unadjusted", 0.15, "#8C8C8C"),
                              ("stroma_adjusted", -0.15, COLORS[comparison.split("_")[0]])]:
        m = d[d["model"] == model]
        ys = [y[p] + off for p in m["pathway"]]
        ax.errorbar(m["effect_SD"], ys,
                    xerr=[m["effect_SD"] - m["ci_low"], m["ci_high"] - m["effect_SD"]],
                    fmt="o", ms=4, color=color, ecolor=color, elinewidth=1,
                    capsize=0, label=model.replace("_", " "))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Group effect on pathway score (SD units, 95% CI)")
    ax.set_title(f"{comparison.replace('_', ' ')}: before vs after stromal adjustment",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


# ---------------------------------------------------------------
def main(files=None):
    os.makedirs(OUT, exist_ok=True)
    st = load_groups()
    if files is None:
        files = gdc_count_files()
    st = st[st["bcr_patient_barcode"].isin(files.index)]
    download_missing(files.loc[st["bcr_patient_barcode"], "file_id"].tolist())
    print("Group sizes:\n", st["group"].value_counts().to_string(), "\n")

    logtpm = build_log_tpm(st["bcr_patient_barcode"].tolist(), files)
    meta = st.set_index("bcr_patient_barcode")[["group", "subtype"]].loc[logtpm.columns]
    groups = meta["group"]
    print(f"Expression matrix: {logtpm.shape[0]} genes x {logtpm.shape[1]} tumours")

    # Part A + B
    mcp, used = mcp_counter(logtpm)
    used.to_csv(f"{OUT}/mcp_markers_used.csv", index=False)
    pd.concat([meta, mcp], axis=1).to_csv(f"{OUT}/mcp_counter_scores.csv")
    mcp_stats = compare_groups(mcp, groups)
    mcp_stats.to_csv(f"{OUT}/mcp_group_comparisons.csv", index=False)
    boxplots(mcp, groups, mcp_stats, f"{OUT}/mcp_counter_boxplots.png")
    print("\nMCP-counter: significant differences (FDR < 0.05)")
    print(mcp_stats[mcp_stats["fdr"] < 0.05][
        ["feature", "comparison", "difference", "fdr"]].round(3).to_string(index=False))

    # Part C
    path = pathway_scores(logtpm, PATHWAYS)
    pd.concat([meta, path], axis=1).to_csv(f"{OUT}/hallmark_pathway_scores.csv")
    adj = stroma_adjusted(path, mcp, meta)
    adj.to_csv(f"{OUT}/stroma_adjusted_pathway_models.csv", index=False)
    forest(adj, "HP_vs_Neither", f"{OUT}/forest_HP_vs_Neither.png")
    forest(adj, "EBV_vs_Neither", f"{OUT}/forest_EBV_vs_Neither.png")

    wide = adj.pivot_table(index=["comparison", "pathway"], columns="model",
                           values=["effect_SD", "fdr"])
    print("\nPathway effect (SD units) before -> after stromal adjustment:")
    print(wide.round(3).to_string())
    print(f"\nAll results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
