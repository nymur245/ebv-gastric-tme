"""
Step 5c: Score TCGA-STAD tumours with the single-cell cell-type signatures (step 5b)

  Part A: 15 atlas signatures + 2 focused programmes (C1Q+ macrophage, IFN-g
          response) scored in every tumour (mean z of log2 TPM+1)
  Part B: EBV / HP / Neither comparison (Kruskal-Wallis + Mann-Whitney, BH-FDR)
  Part C: robustness - EBV vs CIN, EBV vs GS, EBV vs MSI (comparators = not HP+)
  Part D: stroma-adjusted models (adjust for fibroblast + endothelial scores)
  Part E: hypoxia check - Hallmark Hypoxia score + key hypoxia genes
  Part F: IFN-g circuit - does IFNG (from CD8/NK) track CXCL9/10/IDO1/CD274
          (from macrophages) within each group?

Needs:   pip install requests pandas numpy scipy statsmodels gseapy matplotlib
Input:   tcga_stad_hp_ebv_status.csv, star_counts_cache/, sc_results/cell_type_signatures.csv
Outputs: tcga_sc_results/
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import gseapy as gp
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GDC = "https://api.gdc.cancer.gov"
CACHE = "star_counts_cache"
SIGS = "sc_results/cell_type_signatures.csv"
OUT = "tcga_sc_results"
NON_EBV = ["STAD_CIN", "STAD_GS", "STAD_MSI"]
GROUPS = ["Neither", "HP", "EBV"]
COLORS = {"Neither": "#8C8C8C", "HP": "#D55E00", "EBV": "#0072B2"}
EXTRA = {
    "C1Q+ macrophage": ["C1QA", "C1QB", "C1QC", "APOE", "APOC1", "SLC40A1", "MS4A7"],
    "IFNg response":   ["CXCL9", "CXCL10", "CXCL11", "IDO1", "CD274", "GBP5",
                        "STAT1", "IRF1"],
}
HYPOXIA_GENES = ["ADM", "ANGPTL4", "NDRG1", "VEGFA", "CA9", "SLC2A1", "PGK1",
                 "LDHA", "BNIP3", "P4HA1"]
CIRCUIT_X = ["IFNG", "GZMB", "PRF1"]
CIRCUIT_Y = ["CXCL9", "CXCL10", "IDO1", "CD274"]


# ---------------------------------------------------------------
# Data
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


def load_status(files):
    st = pd.read_csv("tcga_stad_hp_ebv_status.csv")
    st = st[(st["has_rna"] == True) &
            st["subtype"].isin(NON_EBV + ["STAD_EBV"]) &
            st["bcr_patient_barcode"].isin(files.index)].copy()
    # main 3-group design (same as steps 2 and 4)
    st["group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "group"] = "EBV"
    st.loc[(st["hp"] == "Yes") & st["subtype"].isin(NON_EBV), "group"] = "HP"
    st.loc[(st["hp"] == "No") & st["subtype"].isin(NON_EBV), "group"] = "Neither"
    # subtype design (same as step 3): EBV vs each subtype, HP+ removed
    st["sub_group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "sub_group"] = "EBV"
    m = st["subtype"].isin(NON_EBV) & (st["hp"] != "Yes")
    st.loc[m, "sub_group"] = st.loc[m, "subtype"].str.replace("STAD_", "")
    have = st["bcr_patient_barcode"].map(
        lambda p: os.path.exists(f"{CACHE}/{files.loc[p, 'file_id']}.tsv"))
    if (~have).any():
        print(f"NOTE: {(~have).sum()} tumours have no cached count file and are "
              "skipped (run step 3 or 4 to download them).")
    return st[have].set_index("bcr_patient_barcode")


def build_log_tpm(patients, files):
    cols = {}
    for p in patients:
        t = pd.read_csv(f"{CACHE}/{files.loc[p, 'file_id']}.tsv",
                        sep="\t", comment="#")
        t = t[t["gene_type"] == "protein_coding"]
        cols[p] = t.groupby("gene_name")["tpm_unstranded"].sum()
    return np.log2(pd.DataFrame(cols) + 1)


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------
def load_signatures(genes_present):
    s = pd.read_csv(SIGS)
    sigs = {r.cell_type: str(r.genes).split(";") for r in s.itertuples()}
    # drop genes shared by 3+ signatures (HLA, CXCR4, BTG1 ...): not cell-type specific
    counts = pd.Series([g for v in sigs.values() for g in set(v)]).value_counts()
    shared = set(counts[counts >= 3].index)
    sigs = {k: [g for g in v if g not in shared] for k, v in sigs.items()}
    sigs.update(EXTRA)
    out, info = {}, []
    for k, v in sigs.items():
        g = [x for x in v if x in genes_present]
        info.append({"signature": k, "n_genes_used": len(g), "genes": ";".join(g)})
        if len(g) >= 5:
            out[k] = g
    print(f"Removed {len(shared)} genes shared by 3+ signatures")
    return out, pd.DataFrame(info)


def score(logtpm, sigs):
    expr = logtpm[logtpm.mean(axis=1) > 0.5]
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1) + 1e-9, axis=0)
    return pd.DataFrame({k: z.loc[[g for g in v if g in z.index]].mean(axis=0)
                         for k, v in sigs.items()})


# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
def compare(scores, labels, pairs):
    rows = []
    for col in scores.columns:
        for a, b in pairs:
            xa = scores.loc[labels == a, col].dropna()
            xb = scores.loc[labels == b, col].dropna()
            if len(xa) < 3 or len(xb) < 3:
                continue
            p = mannwhitneyu(xa, xb, alternative="two-sided").pvalue
            pooled = np.sqrt((xa.var() + xb.var()) / 2) + 1e-9
            rows.append({"feature": col, "comparison": f"{a}_vs_{b}",
                         "n_A": len(xa), "n_B": len(xb),
                         "median_A": xa.median(), "median_B": xb.median(),
                         "effect_SD": (xa.mean() - xb.mean()) / pooled,
                         "p": p})
    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for comp, idx in res.groupby("comparison").groups.items():
        res.loc[idx, "fdr"] = multipletests(res.loc[idx, "p"], method="fdr_bh")[1]
    return res


def stroma_adjusted(scores, meta):
    d = scores.copy()
    d.columns = [f"s{i}" for i in range(d.shape[1])]
    names = dict(zip(d.columns, scores.columns))
    fib = [c for c, n in names.items() if n == "Fibroblast"]
    endo = [c for c, n in names.items() if n == "Endothelial"]
    if not fib or not endo:
        print("Fibroblast / Endothelial signature missing; skipping stromal adjustment")
        return pd.DataFrame()
    d["fibro"] = (d[fib[0]] - d[fib[0]].mean()) / d[fib[0]].std()
    d["endo"] = (d[endo[0]] - d[endo[0]].mean()) / d[endo[0]].std()
    d = d.join(meta[["group", "subtype"]])
    rows = []
    for c, n in names.items():
        if n in ("Fibroblast", "Endothelial", "Pericyte/SMC"):
            continue
        d["y"] = (d[c] - d[c].mean()) / d[c].std()
        for target, base in [("EBV", ""), ("HP", " + C(subtype)")]:
            sub = d[d["group"].isin([target, "Neither"])].copy()
            sub["is_target"] = (sub["group"] == target).astype(int)
            for label, extra in [("unadjusted", ""), ("stroma_adjusted", " + fibro + endo")]:
                fit = smf.ols(f"y ~ is_target{base}{extra}", data=sub).fit()
                ci = fit.conf_int().loc["is_target"]
                rows.append({"signature": n, "comparison": f"{target}_vs_Neither",
                             "model": label, "effect_SD": fit.params["is_target"],
                             "ci_low": ci[0], "ci_high": ci[1],
                             "p": fit.pvalues["is_target"]})
    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for key, idx in res.groupby(["comparison", "model"]).groups.items():
        res.loc[idx, "fdr"] = multipletests(res.loc[idx, "p"], method="fdr_bh")[1]
    return res


def ifng_circuit(logtpm, meta):
    x = logtpm.reindex(CIRCUIT_X).dropna(how="all")
    y = logtpm.reindex(CIRCUIT_Y).dropna(how="all")
    producer = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0).mean()
    rows = []
    for g in GROUPS:
        idx = meta.index[meta["group"] == g]
        for gene in y.index:
            r, p = spearmanr(producer[idx], y.loc[gene, idx])
            rows.append({"group": g, "n": len(idx), "responder_gene": gene,
                         "spearman_r": r, "p": p})
    return pd.DataFrame(rows), producer


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
def boxgrid(scores, labels, order, stats, comp_labels, fname, ncol=6):
    cols = list(scores.columns)
    nr = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nr, ncol, figsize=(2.9 * ncol, 3.0 * nr))
    rng = np.random.default_rng(1)
    palette = {**COLORS, "CIN": "#999999", "GS": "#BBBBBB", "MSI": "#666666"}
    for ax, col in zip(np.ravel(axes), cols):
        data = [scores.loc[labels == g, col].dropna() for g in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(palette.get(g, "#AAAAAA"))
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("black")
        for i, (g, v) in enumerate(zip(order, data), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=5,
                       color=palette.get(g, "#AAAAAA"), alpha=0.6, linewidths=0)
        s = stats[stats["feature"] == col].set_index("comparison")["fdr"]
        sub = " | ".join(f"{lab}: q={s[c]:.2g}" for c, lab in comp_labels if c in s)
        ax.set_title(f"{col}\n{sub}", fontsize=7)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.supylabel("Signature score (mean z, log2 TPM+1)", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def heatmap(stats, comps, fname):
    comps = [c for c in comps if c in set(stats["comparison"])]
    m = stats.pivot(index="feature", columns="comparison", values="effect_SD")[comps]
    q = stats.pivot(index="feature", columns="comparison", values="fdr")[comps]
    m = m.loc[m.mean(axis=1).sort_values(ascending=False).index]
    q = q.loc[m.index]
    lim = np.nanmax(np.abs(m.values))
    fig, ax = plt.subplots(figsize=(1.2 * len(comps) + 3, 0.32 * len(m) + 1.2))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if q.values[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([c.replace("_", " ") for c in comps], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Effect size (SD)")
    ax.set_title("Cell-type signature scores (* FDR < 0.05)", fontsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


# ---------------------------------------------------------------
def main(files=None):
    os.makedirs(OUT, exist_ok=True)
    if files is None:
        files = gdc_count_files()
    meta = load_status(files)
    print("Main groups:", meta["group"].value_counts().to_dict())
    print("Subtype groups:", meta["sub_group"].value_counts().to_dict())

    logtpm = build_log_tpm(meta.index.tolist(), files)
    print(f"Expression: {logtpm.shape[0]} genes x {logtpm.shape[1]} tumours")

    sigs, info = load_signatures(set(logtpm.index))
    info.to_csv(f"{OUT}/signature_genes_used.csv", index=False)
    scores = score(logtpm, sigs)

    # Part E inputs added as extra columns
    try:
        hall = gp.get_library("MSigDB_Hallmark_2020", organism="Human")
        hyp = {"Hallmark Hypoxia": [g for g in hall["Hypoxia"] if g in logtpm.index]}
        scores = scores.join(score(logtpm, hyp))
    except Exception as e:
        print("Could not fetch Hallmark gene sets:", e)
    pd.concat([meta[["hp", "subtype", "group", "sub_group"]], scores], axis=1) \
        .to_csv(f"{OUT}/tcga_signature_scores.csv")

    # Part B
    main_idx = meta.index[meta["group"].notna()]
    stats_main = compare(scores.loc[main_idx], meta.loc[main_idx, "group"],
                         [("EBV", "Neither"), ("HP", "Neither"), ("EBV", "HP")])
    # Part C
    sub_idx = meta.index[meta["sub_group"].notna()]
    stats_sub = compare(scores.loc[sub_idx], meta.loc[sub_idx, "sub_group"],
                        [("EBV", "CIN"), ("EBV", "GS"), ("EBV", "MSI")])
    stats = pd.concat([stats_main, stats_sub])
    stats.to_csv(f"{OUT}/signature_comparisons.csv", index=False)

    comps = ["EBV_vs_Neither", "HP_vs_Neither", "EBV_vs_HP",
             "EBV_vs_CIN", "EBV_vs_GS", "EBV_vs_MSI"]
    heatmap(stats, comps, f"{OUT}/signature_effect_heatmap.png")
    boxgrid(scores.loc[main_idx], meta.loc[main_idx, "group"], GROUPS, stats_main,
            [("EBV_vs_Neither", "EBV"), ("HP_vs_Neither", "HP")],
            f"{OUT}/signature_boxplots_main.png")

    robust = stats_sub.pivot(index="feature", columns="comparison", values=["effect_SD", "fdr"])
    consistent = [f for f in robust.index
                  if robust["fdr"].shape[1] == 3
                  and (robust.loc[f, "fdr"] < 0.05).all()
                  and (np.sign(robust.loc[f, "effect_SD"]).nunique() == 1)]
    print("\nSignatures different in EBV vs ALL three subtypes (same direction, FDR<0.05):")
    for f in consistent:
        print(f"  {f}: effect {robust.loc[f, 'effect_SD'].round(2).to_dict()}")

    # Part D
    adj = stroma_adjusted(scores.loc[main_idx], meta.loc[main_idx])
    if len(adj):
        adj.to_csv(f"{OUT}/stroma_adjusted_models.csv", index=False)

    # Part E - single hypoxia genes
    hg = [g for g in HYPOXIA_GENES if g in logtpm.index]
    hstats = pd.concat([
        compare(logtpm.loc[hg, main_idx].T, meta.loc[main_idx, "group"],
                [("EBV", "Neither"), ("HP", "Neither")]),
        compare(logtpm.loc[hg, sub_idx].T, meta.loc[sub_idx, "sub_group"],
                [("EBV", "CIN"), ("EBV", "GS"), ("EBV", "MSI")])])
    hstats.to_csv(f"{OUT}/hypoxia_gene_comparisons.csv", index=False)

    # Part F
    circ, producer = ifng_circuit(logtpm.loc[:, main_idx], meta.loc[main_idx])
    circ.to_csv(f"{OUT}/ifng_circuit_correlations.csv", index=False)

    # console summary
    show = stats_main[stats_main["comparison"].isin(["EBV_vs_Neither", "HP_vs_Neither"])]
    print("\nMain comparison (effect in SD units, FDR):")
    print(show.pivot(index="feature", columns="comparison", values=["effect_SD", "fdr"])
          .round(3).to_string())
    if len(adj):
        print("\nStroma adjustment, EBV vs Neither (effect before -> after):")
        e = adj[adj["comparison"] == "EBV_vs_Neither"].pivot(
            index="signature", columns="model", values="effect_SD").round(2)
        print(e.to_string())
    print("\nHypoxia genes (effect SD, FDR):")
    print(hstats.pivot(index="feature", columns="comparison", values="effect_SD").round(2).to_string())
    print("\nIFN-g circuit (Spearman r, producer score vs responder gene):")
    print(circ.pivot(index="responder_gene", columns="group", values="spearman_r").round(2).to_string())
    print(f"\nAll results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
