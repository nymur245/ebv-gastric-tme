"""
Step 8: C1Q+ vs SPP1+ macrophage balance (Zhang et al. 2020; Bill et al. 2023)

  Part A  Gastric atlas (GSE183904): are C1Q-high and SPP1-high distinct macrophage
          states in gastric cancer?  Re-cluster the macrophage/monocyte cells, find the
          C1QC-high and SPP1-high subclusters, derive gastric-specific signatures, and keep
          only genes that are macrophage-specific across ALL atlas cell types.
          Also reports which cell types express CXCL9, SPP1 and C1QA/B/C.
  Part B  TCGA-STAD: per tumour
            C1Q-TAM score, SPP1-TAM score       (mean log2 TPM+1 of signature genes)
            C1Q-SPP1 balance                    (difference of the two = log-ratio)
            SPP1 alone, CXCL9 alone, log2(CXCL9+1) - log2(SPP1+1)
          compared across EBV / HP / Neither and EBV vs CIN, GS, MSI,
          then modelled with stromal and total-macrophage adjustment.

Needs:   pip install requests pandas numpy scipy statsmodels scanpy harmonypy==0.0.10 leidenalg igraph matplotlib
Input:   sc_cache/ + geo_inspect/ + sc_results/cell_metadata.csv   (step 5b)
         star_counts_cache/ + tcga_stad_hp_ebv_status.csv        (steps 1-4)
         tcga_sc_results/tcga_signature_scores.csv               (step 5c)
Outputs: macrophage_results/
Runtime: about 15-30 min (re-reads the single-cell files from sc_cache).
"""

import os
import gzip
import json
import requests
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GDC = "https://api.gdc.cancer.gov"
OUT = "macrophage_results"
SC_CACHE = "sc_cache"
TCGA_CACHE = "star_counts_cache"
SEED = 0
MYELOID = ["Macrophage", "Monocyte"]
SEEDS = {"C1Q": "C1QC", "SPP1": "SPP1"}       # seed genes named by Zhang et al. 2020
N_SIG = 25                                     # genes per signature before specificity filter
PAN_MAC = ["CD68", "CD163", "CSF1R", "AIF1", "FCGR3A", "LILRB4"]
# kept out of the TAM signatures so they stay independent of the IFN-g score and of
# total macrophage abundance (no circularity with steps 3-5c)
EXCLUDE = set(PAN_MAC) | {"CXCL9", "CXCL10", "CXCL11", "CD274", "IDO1", "STAT1", "IRF1",
                          "GBP1", "GBP2", "GBP4", "GBP5", "WARS1", "IFNG"}
GROUP_COLORS = {"Neither": "#8C8C8C", "HP": "#D55E00", "EBV": "#0072B2",
                "CIN": "#BBBBBB", "GS": "#999999", "MSI": "#666666"}


# =============================================================== Part A: atlas
def read_sample_cells(path, keep_cells):
    """Read the cells listed in keep_cells (original barcodes) from one sample CSV.
    Handles genes-in-rows and cells-in-rows layouts. Returns AnnData of raw counts."""
    with gzip.open(path, "rt") as fh:
        header = [h.strip('"') for h in fh.readline().rstrip("\n").split(",")]
        first = fh.readline().split(",")[0].strip('"')
    keep = set(keep_cells)
    genes_in_rows = len(header) > 100 and not first.replace("_", "").isdigit()
    if genes_in_rows:
        cols = [0] + [i for i, h in enumerate(header) if i > 0 and h in keep]
        parts, genes = [], []
        for chunk in pd.read_csv(path, usecols=cols, index_col=0, chunksize=2000):
            genes.extend(chunk.index.astype(str).str.strip('"'))
            parts.append(sp.csr_matrix(chunk.values.astype(np.float32)))
        X = sp.vstack(parts).T.tocsr()
        cells = [header[i] for i in cols[1:]]
    else:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str).str.strip('"')
        df = df.loc[[c for c in df.index if c in keep]]
        X = sp.csr_matrix(df.values.astype(np.float32))
        genes = [str(g).strip('"') for g in df.columns]
        cells = list(df.index)
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    return a


def load_atlas_cells():
    meta = pd.read_csv("sc_results/cell_metadata.csv", index_col=0)
    files = pd.read_csv("geo_inspect/GSE183904_samples.csv")
    files["sample"] = files["title"].str.split(":").str[0]
    files = files.set_index("sample")["supplementary_file_1"]
    adatas = []
    for i, (s, m) in enumerate(meta.groupby("sample"), 1):
        path = f"{SC_CACHE}/{os.path.basename(files[s])}"
        barcodes = [c[len(s) + 1:] for c in m.index]
        a = read_sample_cells(path, barcodes)
        a.obs_names = [f"{s}_{c}" for c in a.obs_names]
        adatas.append(a)
        print(f"  {i}/{meta['sample'].nunique()} {s}: {a.n_obs} cells")
    adata = ad.concat(adatas, join="inner")
    adata = adata[adata.obs_names.isin(meta.index)].copy()
    adata.obs = adata.obs.join(meta[["sample", "tissue", "cell_type_detail"]])
    print(f"Atlas reloaded: {adata.n_obs} cells x {adata.n_vars} genes")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def is_myeloid(labels):
    return labels.astype(str).str.startswith(tuple(MYELOID)).values


def recluster_myeloid(adata):
    m = adata[is_myeloid(adata.obs["cell_type_detail"])].copy()
    sc.pp.highly_variable_genes(m, n_top_genes=1500, batch_key="sample")
    h = m[:, m.var["highly_variable"]].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=30, random_state=SEED)
    rep = "X_pca"
    try:
        import harmonypy
        Z = np.asarray(harmonypy.run_harmony(h.obsm["X_pca"], h.obs, "sample",
                                             random_state=SEED).Z_corr)
        h.obsm["X_harmony"] = Z if Z.shape[0] == h.n_obs else Z.T
        rep = "X_harmony"
    except Exception as e:
        print("Harmony skipped:", e)
    sc.pp.neighbors(h, use_rep=rep, n_neighbors=15, random_state=SEED)
    sc.tl.umap(h, random_state=SEED)
    sc.tl.leiden(h, resolution=0.8, key_added="sub", random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    m.obs["sub"] = h.obs["sub"].values
    m.obsm["X_umap"] = h.obsm["X_umap"]
    print(f"Myeloid cells: {m.n_obs}; subclusters: {m.obs['sub'].nunique()}")
    return m


def gene_vec(a, g):
    x = a[:, g].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.ravel(x)


def derive_signatures(adata, m):
    # subcluster summary for seed genes
    rows = []
    for c, idx in m.obs.groupby("sub", observed=True).groups.items():
        sub = m[idx]
        r = {"subcluster": c, "n_cells": sub.n_obs,
             "macrophage_fraction": sub.obs["cell_type_detail"].astype(str).str.startswith("Macrophage").mean()}
        for g in ["C1QA", "C1QB", "C1QC", "SPP1", "CXCL9", "CXCL10", "APOE", "CD14",
                  "FCN1", "VCAN", "CD68", "CD163"]:
            if g in m.var_names:
                v = gene_vec(sub, g)
                r[f"mean_{g}"] = v.mean()
                r[f"pct_{g}"] = 100 * (v > 0).mean()
        rows.append(r)
    summ = pd.DataFrame(rows).set_index("subcluster")
    summ.to_csv(f"{OUT}/atlas_myeloid_subclusters.csv")
    c1q_cl = summ["mean_C1QC"].idxmax()
    spp1_cl = summ["mean_SPP1"].idxmax()
    print(f"C1QC-high subcluster: {c1q_cl}; SPP1-high subcluster: {spp1_cl}")
    if c1q_cl == spp1_cl:
        print("WARNING: C1QC and SPP1 peak in the SAME subcluster - not distinct states here.")

    # single-cell co-expression of the two seeds
    c1qc, spp1 = gene_vec(m, "C1QC"), gene_vec(m, "SPP1")
    r, p = spearmanr(c1qc, spp1)
    co = pd.crosstab(c1qc > 0, spp1 > 0, normalize=True)
    print(f"C1QC vs SPP1 across myeloid cells: Spearman r = {r:.2f}")

    # signatures: C1Q cluster vs SPP1 cluster
    pair = m[m.obs["sub"].isin([c1q_cl, spp1_cl])].copy()
    pair.obs["state"] = np.where(pair.obs["sub"] == c1q_cl, "C1Q", "SPP1")
    sc.tl.rank_genes_groups(pair, "state", method="wilcoxon", pts=True)
    sig = {}
    for st in ["C1Q", "SPP1"]:
        df = sc.get.rank_genes_groups_df(pair, st)
        df = df[(df["logfoldchanges"] > 1) & (df["pvals_adj"] < 0.01)]
        if "pct_nz_group" in df:
            df = df[df["pct_nz_group"] > 0.3]
        df = df[~df["names"].str.match(r"^(MT-|RPL|RPS|HLA-)")]
        other_seed = {"C1Q": {"SPP1"}, "SPP1": {"C1QA", "C1QB", "C1QC"}}[st]
        df = df[~df["names"].isin(EXCLUDE | other_seed)]
        genes = df["names"].head(N_SIG).tolist()
        if SEEDS[st] not in genes:
            genes = [SEEDS[st]] + genes[:N_SIG - 1]
        sig[st] = genes

    # specificity across all atlas cell types
    types = adata.obs["cell_type_detail"]
    spec_rows = []
    allg = sorted(set(sig["C1Q"] + sig["SPP1"] + ["CXCL9", "CXCL10", "SPP1", "C1QA",
                                                   "C1QB", "C1QC"] + PAN_MAC))
    allg = [g for g in allg if g in adata.var_names]
    sub = adata[:, allg]
    X = sub.X.tocsc() if sp.issparse(sub.X) else sub.X
    means = {}
    for t in sorted(types.unique()):
        msk = (types == t).values
        means[t] = np.asarray(X[msk].mean(axis=0)).ravel()
    means = pd.DataFrame(means, index=allg)
    mcols = [c for c in means.columns if str(c).startswith(tuple(MYELOID))]
    myel = means[mcols].max(axis=1)
    other = means.drop(columns=mcols).max(axis=1)
    spec = pd.DataFrame({"myeloid_max": myel, "other_max": other,
                         "other_top_type": means.drop(columns=mcols).idxmax(axis=1),
                         "ratio": (myel + 1e-3) / (other + 1e-3)})
    pd.concat([spec, means.add_prefix("mean_")], axis=1).to_csv(
        f"{OUT}/atlas_gene_cell_type_means.csv")

    final, log = {}, []
    for st, genes in sig.items():
        kept = [g for g in genes if g in spec.index and spec.loc[g, "ratio"] >= 3]
        for g in genes:
            log.append({"state": st, "gene": g, "kept": g in kept,
                        "myeloid_max": spec.loc[g, "myeloid_max"] if g in spec.index else np.nan,
                        "other_max": spec.loc[g, "other_max"] if g in spec.index else np.nan,
                        "other_top_type": spec.loc[g, "other_top_type"] if g in spec.index else ""})
        final[st] = kept
        print(f"{st}-TAM signature: {len(genes)} genes -> {len(kept)} macrophage-specific")
    pd.DataFrame(log).to_csv(f"{OUT}/signature_derivation.csv", index=False)
    json.dump(final, open(f"{OUT}/gastric_TAM_signatures.json", "w"), indent=1)

    print("\nWhere the polarity genes come from (top 3 cell types by mean expression):")
    for g in ["CXCL9", "SPP1", "C1QC"]:
        if g in means.index:
            top = means.loc[g].sort_values(ascending=False).head(3)
            print(f"  {g}: " + ", ".join(f"{k} {v:.2f}" for k, v in top.items()))
    return final, summ, (c1q_cl, spp1_cl), r


def atlas_figure(m, final, clusters):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    um = m.obsm["X_umap"]
    cats = m.obs["sub"].astype(str)
    for c in sorted(cats.unique(), key=lambda x: int(x) if x.isdigit() else x):
        k = (cats == c).values
        lab = c + (" (C1Q)" if c == str(clusters[0]) else "") + (" (SPP1)" if c == str(clusters[1]) else "")
        axes[0].scatter(um[k, 0], um[k, 1], s=2, label=lab)
        axes[0].text(np.median(um[k, 0]), np.median(um[k, 1]), c, fontsize=8, weight="bold")
    axes[0].set_title("Myeloid subclusters")
    axes[0].legend(fontsize=6, markerscale=3, frameon=False, loc="best")
    for ax, g in zip(axes[1:], ["C1QC", "SPP1", "CXCL9"]):
        if g in m.var_names:
            v = gene_vec(m, g)
            o = np.argsort(v)
            s = ax.scatter(um[o, 0], um[o, 1], c=v[o], s=2, cmap="viridis")
            plt.colorbar(s, ax=ax, fraction=0.04)
        ax.set_title(g)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{OUT}/atlas_myeloid_umap.png", dpi=300)
    plt.close()


# =============================================================== Part B: TCGA
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


def tcga_scores(final, files):
    meta = pd.read_csv("tcga_sc_results/tcga_signature_scores.csv", index_col=0)
    meta = meta[meta.index.isin(files.index)]
    genes = set(final["C1Q"] + final["SPP1"] + PAN_MAC + ["CXCL9", "SPP1"])
    cols = {}
    for p in meta.index:
        fpath = f"{TCGA_CACHE}/{files.loc[p, 'file_id']}.tsv"
        if not os.path.exists(fpath):
            continue
        t = pd.read_csv(fpath, sep="\t", comment="#", usecols=["gene_name", "gene_type", "tpm_unstranded"])
        t = t[(t["gene_type"] == "protein_coding") & t["gene_name"].isin(genes)]
        cols[p] = t.groupby("gene_name")["tpm_unstranded"].sum()
    lt = np.log2(pd.DataFrame(cols).fillna(0) + 1)
    meta = meta.loc[lt.columns]
    s = pd.DataFrame(index=lt.columns)
    s["C1Q_TAM"] = lt.reindex(final["C1Q"]).dropna(how="all").mean()
    s["SPP1_TAM"] = lt.reindex(final["SPP1"]).dropna(how="all").mean()
    s["C1Q_minus_SPP1"] = s["C1Q_TAM"] - s["SPP1_TAM"]
    s["SPP1"] = lt.loc["SPP1"]
    s["CXCL9"] = lt.loc["CXCL9"]
    s["log2_CXCL9_over_SPP1"] = lt.loc["CXCL9"] - lt.loc["SPP1"]
    s["pan_macrophage"] = lt.reindex(PAN_MAC).dropna(how="all").mean()
    out = meta[["hp", "subtype", "group", "sub_group", "Fibroblast", "Endothelial",
                "IFNg response"]].join(s)
    out.to_csv(f"{OUT}/tcga_macrophage_scores.csv")
    print(f"\nTCGA tumours scored: {len(out)}")
    return out


METRICS = ["C1Q_TAM", "SPP1_TAM", "C1Q_minus_SPP1", "SPP1", "CXCL9",
           "log2_CXCL9_over_SPP1", "pan_macrophage"]


def compare(d):
    rows = []
    for col in METRICS:
        for lab, a, b in [("group", "EBV", "Neither"), ("group", "HP", "Neither"),
                          ("sub_group", "EBV", "CIN"), ("sub_group", "EBV", "GS"),
                          ("sub_group", "EBV", "MSI")]:
            xa = d.loc[d[lab] == a, col].dropna()
            xb = d.loc[d[lab] == b, col].dropna()
            if len(xa) < 3 or len(xb) < 3:
                continue
            pooled = np.sqrt((xa.var() + xb.var()) / 2) + 1e-9
            rows.append({"metric": col, "comparison": f"{a}_vs_{b}", "n_A": len(xa),
                         "n_B": len(xb), "median_A": xa.median(), "median_B": xb.median(),
                         "effect_SD": (xa.mean() - xb.mean()) / pooled,
                         "p": mannwhitneyu(xa, xb, alternative="two-sided").pvalue})
    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for c, idx in res.groupby("comparison").groups.items():
        res.loc[idx, "fdr"] = multipletests(res.loc[idx, "p"], method="fdr_bh")[1]
    return res


def models(d):
    z = lambda x: (x - x.mean()) / x.std()
    d = d.copy()
    for c in ["Fibroblast", "Endothelial", "pan_macrophage"]:
        d[c + "_z"] = z(d[c])
    rows = []
    for col in ["C1Q_minus_SPP1", "SPP1", "log2_CXCL9_over_SPP1", "C1Q_TAM", "SPP1_TAM"]:
        d["y"] = z(d[col])
        for target, base in [("EBV", ""), ("HP", " + C(subtype)")]:
            sub = d[d["group"].isin([target, "Neither"])].copy()
            sub["is_target"] = (sub["group"] == target).astype(int)
            for lab, extra in [("unadjusted", ""),
                               ("stroma", " + Fibroblast_z + Endothelial_z"),
                               ("stroma + total macrophages",
                                " + Fibroblast_z + Endothelial_z + pan_macrophage_z")]:
                fit = smf.ols(f"y ~ is_target{base}{extra}", data=sub).fit()
                ci = fit.conf_int().loc["is_target"]
                rows.append({"metric": col, "comparison": f"{target}_vs_Neither",
                             "model": lab, "effect_SD": fit.params["is_target"],
                             "ci_low": ci[0], "ci_high": ci[1],
                             "p": fit.pvalues["is_target"], "n": len(sub)})
    return pd.DataFrame(rows)


def tcga_figure(d, stats):
    order = ["Neither", "HP", "EBV", "CIN", "GS", "MSI"]
    lab = d["group"].where(d["group"].notna(), None)
    show = ["C1Q_minus_SPP1", "SPP1", "log2_CXCL9_over_SPP1", "pan_macrophage"]
    titles = {"C1Q_minus_SPP1": "C1Q-TAM minus SPP1-TAM", "SPP1": "SPP1 (log2 TPM+1)",
              "log2_CXCL9_over_SPP1": "log2 CXCL9 / SPP1", "pan_macrophage": "Total macrophages"}
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.6))
    rng = np.random.default_rng(1)
    for ax, col in zip(axes, show):
        data, names = [], []
        for g in order:
            if g in ["Neither", "HP", "EBV"]:
                v = d.loc[d["group"] == g, col].dropna()
            else:
                v = d.loc[d["sub_group"] == g, col].dropna()
            data.append(v); names.append(g)
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], names):
            patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("black")
        for i, (g, v) in enumerate(zip(names, data), 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4,
                       color=GROUP_COLORS[g], alpha=0.6, linewidths=0)
        q = stats[stats["metric"] == col].set_index("comparison")["fdr"]
        txt = " | ".join(f"{k.replace('_vs_', '/')} q={q[k]:.2g}" for k in
                         ["EBV_vs_Neither", "HP_vs_Neither", "EBV_vs_MSI"] if k in q)
        ax.set_title(f"{titles[col]}\n{txt}", fontsize=7)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=7)
        ax.axvline(3.5, color="#CCCCCC", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/tcga_macrophage_balance.png", dpi=300)
    plt.close()


def main(files=None):
    os.makedirs(OUT, exist_ok=True)
    print("== Part A: gastric atlas ==")
    adata = load_atlas_cells()
    m = recluster_myeloid(adata)
    final, summ, clusters, r = derive_signatures(adata, m)
    atlas_figure(m, final, clusters)
    print("Signatures:", json.dumps(final))
    del adata

    print("\n== Part B: TCGA-STAD ==")
    if files is None:
        files = gdc_count_files()
    d = tcga_scores(final, files)
    stats = compare(d)
    stats.to_csv(f"{OUT}/tcga_comparisons.csv", index=False)
    mod = models(d)
    mod.to_csv(f"{OUT}/tcga_adjusted_models.csv", index=False)
    tcga_figure(d, stats)

    pd.set_option("display.width", 200)
    print("\nEffect size (SD), * = FDR < 0.05:")
    t = stats.assign(v=stats["effect_SD"].round(2).astype(str) +
                     np.where(stats["fdr"] < 0.05, "*", ""))
    print(t.pivot(index="metric", columns="comparison", values="v").to_string())
    print("\nEBV and HP effects with adjustment (SD units):")
    print(mod.pivot_table(index=["comparison", "metric"], columns="model",
                          values="effect_SD").round(2).to_string())
    neg = d[d["group"] == "Neither"]
    rr, pp = spearmanr(neg["C1Q_minus_SPP1"], neg["IFNg response"])
    print(f"\nIn double-negative tumours, C1Q-SPP1 balance vs IFN-g response: r = {rr:.2f} (p = {pp:.2g})")
    print(f"\nAll results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
