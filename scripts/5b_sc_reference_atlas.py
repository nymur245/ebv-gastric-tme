"""
Step 5b: Single-cell reference atlas (GSE183904, Kumar et al. 2022) and
         localisation of the core EBV genes to cell types

  Part A: download the per-sample count matrices (stomach tumour + normal only),
          keep up to MAX_CELLS random cells per sample (fits in 16 GB RAM)
  Part B: QC -> normalise -> HVG -> PCA -> Harmony (batch = sample) -> UMAP -> Leiden
  Part C: annotate clusters with marker-gene scores (major lineages)
  Part D: where are the core EBV genes expressed?
          - per gene: mean expression + % expressing in each cell type
          - EBV-up and EBV-down module scores per cell type
  Part E: cell-type signatures (top markers) saved for scoring TCGA in step 5c

Needs:   pip install requests pandas numpy scipy scanpy leidenalg igraph matplotlib
         pip install harmonypy==0.0.10   (newer versions fail to build on Python 3.14)
Input:   geo_inspect/GSE183904_samples.csv (step 5a), de_results/core_EBV_genes.csv (step 3),
         star_counts_cache/ (only used to map Ensembl IDs to symbols if needed)
Outputs: sc_cache/  (downloaded files), sc_results/
Runtime: download ~1-2 GB once; analysis ~20-40 min on a laptop.
"""

import os
import glob
import gzip
import time
import requests
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES = "geo_inspect/GSE183904_samples.csv"
CORE = "de_results/core_EBV_genes.csv"
CACHE = "sc_cache"
OUT = "sc_results"
MAX_CELLS = 2500          # random cells kept per sample
SEED = 0

MARKERS = {
    "Epithelial":        ["EPCAM", "KRT8", "KRT18", "KRT19", "CLDN4", "TFF1",
                          "PGC", "MUC6", "LIPF"],
    "CD8 T":             ["CD8A", "CD8B", "CD3D", "GZMK"],
    "CD4 T":             ["CD4", "IL7R", "CD3D", "CD40LG"],
    "Treg":              ["FOXP3", "IL2RA", "CTLA4", "TNFRSF4"],
    "NK":                ["KLRF1", "NCAM1", "FCGR3A", "GNLY", "KLRD1"],
    "B":                 ["MS4A1", "CD79A", "CD19", "BANK1"],
    "Plasma":            ["MZB1", "JCHAIN", "SDC1", "IGHG1", "XBP1"],
    "Macrophage":        ["CD68", "CD163", "C1QA", "C1QB", "APOE"],
    "Monocyte":          ["CD14", "FCN1", "VCAN", "S100A8", "CSF3R"],
    "Dendritic":         ["CLEC9A", "CD1C", "CLEC10A", "LAMP3", "XCR1"],
    "Mast":              ["TPSAB1", "CPA3", "KIT", "MS4A2"],
    "Fibroblast":        ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Endothelial":       ["PECAM1", "VWF", "CDH5", "PLVAP"],
    "Pericyte/SMC":      ["RGS5", "ACTA2", "MYH11", "PDGFRB"],
    "Cycling":           ["MKI67", "TOP2A", "CENPF", "TYMS"],
}
KEY_GENES = ["CXCL9", "CXCL10", "CXCL11", "CD274", "IDO1", "GBP5", "IFNG",
             "WARS1", "ZNF683", "XCR1", "CX3CL1", "SOX2", "DKK4", "BAMBI",
             "CDX1", "MUC2", "FABP1", "CLDN3", "GUCY2C", "DEFA6", "CLCA1", "PRAME"]


# ---------------------------------------------------------------
# Part A: download + load
# ---------------------------------------------------------------
def download(url, path, tries=5):
    if os.path.exists(path):
        return
    url = url.replace("ftp://", "https://")
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(path + ".part", "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.replace(path + ".part", path)
            return
        except requests.exceptions.RequestException as e:
            print(f"    attempt {attempt} failed ({type(e).__name__}), retrying")
            time.sleep(15 * attempt)
    raise SystemExit(f"Could not download {url}. Run the script again.")


def ensembl_to_symbol():
    f = sorted(glob.glob("star_counts_cache/*.tsv"))
    if not f:
        return {}
    t = pd.read_csv(f[0], sep="\t", comment="#", usecols=["gene_id", "gene_name"])
    t = t.dropna()
    return dict(zip(t["gene_id"].str.split(".").str[0], t["gene_name"]))


def read_sample(path, rng):
    """Reads one CSV (genes x cells or cells x genes), keeps MAX_CELLS cells,
    returns a sparse AnnData (cells x genes) of raw counts."""
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        second = fh.readline().rstrip("\n").split(",")
    first_id = second[0].strip('"')
    genes_in_rows = len(header) > 100 and not first_id.replace("_", "").isdigit()
    if genes_in_rows:                       # rows = genes, columns = cells
        cells = header[1:]
        keep = sorted(rng.choice(len(cells), min(MAX_CELLS, len(cells)),
                                 replace=False))
        usecols = [0] + [k + 1 for k in keep]
        parts, genes = [], []
        for chunk in pd.read_csv(path, usecols=usecols, index_col=0,
                                 chunksize=2000):
            genes.extend(chunk.index.astype(str))
            parts.append(sp.csr_matrix(chunk.values.astype(np.float32)))
        X = sp.vstack(parts).T.tocsr()
        a = ad.AnnData(X=X, obs=pd.DataFrame(index=[cells[k] for k in keep]),
                       var=pd.DataFrame(index=genes))
    else:                                   # rows = cells, columns = genes
        df = pd.read_csv(path, index_col=0)
        if len(df) > MAX_CELLS:
            df = df.iloc[sorted(rng.choice(len(df), MAX_CELLS, replace=False))]
        a = ad.AnnData(X=sp.csr_matrix(df.values.astype(np.float32)),
                       obs=pd.DataFrame(index=df.index.astype(str)),
                       var=pd.DataFrame(index=df.columns.astype(str)))
    a.var_names = a.var_names.str.strip('"')
    a.obs_names = a.obs_names.str.strip('"')
    return a


def load_atlas():
    meta = pd.read_csv(SAMPLES)
    meta = meta[meta["source_name_ch1"].str.strip().str.lower() == "stomach"]
    meta["tissue"] = np.where(meta["title"].str.contains("Tumor"), "Tumour", "Normal")
    meta["sample"] = meta["title"].str.split(":").str[0]
    print(f"Samples used: {len(meta)} "
          f"({(meta.tissue == 'Tumour').sum()} tumour, {(meta.tissue == 'Normal').sum()} normal)")
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sym = None
    adatas = []
    for i, row in enumerate(meta.itertuples(), 1):
        path = f"{CACHE}/{os.path.basename(row.supplementary_file_1)}"
        download(row.supplementary_file_1, path)
        a = read_sample(path, rng)
        if a.var_names[0].startswith("ENSG"):
            sym = sym or ensembl_to_symbol()
            a.var_names = [sym.get(g.split(".")[0], g) for g in a.var_names]
        a.var_names_make_unique()
        a.obs["sample"] = row.sample
        a.obs["tissue"] = row.tissue
        a.obs_names = [f"{row.sample}_{c}" for c in a.obs_names]
        adatas.append(a)
        print(f"  {i}/{len(meta)} {row.sample} ({row.tissue}): {a.n_obs} cells, "
              f"{a.n_vars} genes")
    adata = ad.concat(adatas, join="inner")
    print(f"\nCombined: {adata.n_obs} cells x {adata.n_vars} shared genes")
    return adata


# ---------------------------------------------------------------
# Part B + C: processing and annotation
# ---------------------------------------------------------------
def process(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    vals = adata.X.data[:200000] if sp.issparse(adata.X) else np.ravel(adata.X)[:200000]
    vals = vals[vals > 0]
    raw_counts = bool(np.allclose(vals, np.round(vals)))
    if not raw_counts:
        print("NOTE: values are not integers (already normalised); skipping normalisation.")
    n0 = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    adata = adata[adata.obs["pct_counts_mt"] < 25].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"QC: kept {adata.n_obs}/{n0} cells, {adata.n_vars} genes")

    if raw_counts:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, random_state=SEED)
    try:
        import harmonypy
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, "sample",
                                   random_state=SEED)
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != adata.n_obs:
            Z = Z.T
        adata.obsm["X_pca_harmony"] = Z
        rep = "X_pca_harmony"
        print("Harmony integration done")
    except Exception as e:
        rep = "X_pca"
        print(f"Harmony not available ({e}); using plain PCA")
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=20, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden", random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    print(f"Leiden clusters: {adata.obs['leiden'].nunique()}")
    return adata


def annotate(adata):
    full = adata.raw.to_adata()
    for ct, genes in MARKERS.items():
        g = [x for x in genes if x in full.var_names]
        sc.tl.score_genes(full, g, score_name=ct, random_state=SEED)
    scores = full.obs[list(MARKERS)].copy()
    scores["leiden"] = adata.obs["leiden"].values
    cl = scores.groupby("leiden", observed=True).mean()
    z = (cl - cl.mean()) / cl.std()
    # immune labels only for clusters that express PTPRC (CD45); stops gastric
    # gland cells (which also express LYZ / S100A8) being called monocytes
    if "PTPRC" in full.var_names:
        col = full[:, "PTPRC"].X
        ptprc = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.ravel(col)
    else:
        print("PTPRC not found; skipping the CD45 check")
        ptprc = np.ones(full.n_obs)
    ptprc = pd.Series(ptprc, index=full.obs_names)
    cl_ptprc = ptprc.groupby(adata.obs["leiden"].values, observed=True).mean()
    immune = ["CD8 T", "CD4 T", "Treg", "NK", "B", "Plasma", "Macrophage",
              "Monocyte", "Dendritic", "Mast"]
    zz = z.drop(columns="Cycling").copy()
    low = cl_ptprc.reindex(zz.index) < 0.3 * cl_ptprc.max()
    for c in [c for c in immune if c in zz.columns and c != "Plasma"]:
        zz.loc[low, c] = -np.inf          # plasma cells are often CD45-low
    label = zz.idxmax(axis=1)
    cycling = z["Cycling"] > 1.5
    label[cycling] = label[cycling] + " (cycling)"
    adata.obs["cell_type"] = adata.obs["leiden"].map(label).astype("category")
    epi = adata.obs["cell_type"].astype(str).str.startswith("Epithelial")
    adata.obs["cell_type_detail"] = adata.obs["cell_type"].astype(str)
    adata.obs.loc[epi, "cell_type_detail"] = (
        "Epithelial-" + adata.obs.loc[epi, "tissue"].astype(str))
    out = z.round(2)
    out.insert(0, "assigned", label)
    out.insert(1, "n_cells", adata.obs["leiden"].value_counts().reindex(out.index))
    out.insert(2, "PTPRC_mean", cl_ptprc.reindex(out.index).round(3))
    return out


# ---------------------------------------------------------------
# Part D: localisation of core EBV genes
# ---------------------------------------------------------------
def localise(adata):
    core = pd.read_csv(CORE)
    core = core[core["gene_type"] == "protein_coding"]
    full = adata.raw.to_adata()
    genes = [g for g in core["gene_name"] if g in full.var_names]
    print(f"\nCore EBV protein-coding genes found in atlas: {len(genes)}/{len(core)}")
    groups = adata.obs["cell_type_detail"]
    sub = full[:, genes]
    X = sub.X.tocsc() if sp.issparse(sub.X) else sub.X
    mean, frac = {}, {}
    for ct in sorted(groups.unique()):
        m = (groups == ct).values
        xm = X[m]
        mean[ct] = np.asarray(xm.mean(axis=0)).ravel()
        frac[ct] = np.asarray((xm > 0).mean(axis=0)).ravel()
    mean = pd.DataFrame(mean, index=genes)
    frac = pd.DataFrame(frac, index=genes)
    info = core.set_index("gene_name").loc[genes, ["direction", "mean_log2FC"]]
    top = mean.idxmax(axis=1)
    spec = mean.max(axis=1) / (mean.sum(axis=1) + 1e-9)
    res = info.copy()
    res["top_cell_type"] = top
    res["specificity"] = spec.round(3)
    res["max_mean_expr"] = mean.max(axis=1).round(3)
    res["pct_expressing_top"] = [round(100 * frac.loc[g, top[g]], 1) for g in genes]
    res = res.join(mean.round(3).add_prefix("mean_")).join(
        (100 * frac).round(1).add_prefix("pct_"))
    res = res.sort_values(["direction", "specificity"], ascending=[False, False])
    res.to_csv(f"{OUT}/core_EBV_gene_localisation.csv")

    expressed = res[res["max_mean_expr"] > 0.1]
    summ = pd.crosstab(expressed["top_cell_type"], expressed["direction"])
    summ.to_csv(f"{OUT}/core_EBV_genes_by_cell_type.csv")
    print("\nCore EBV genes (expressed, mean > 0.1) by the cell type that expresses them most:")
    print(summ.to_string())

    # module scores
    for d in ["up", "down"]:
        g = [x for x in core.loc[core.direction == d, "gene_name"] if x in full.var_names]
        sc.tl.score_genes(full, g, score_name=f"EBV_{d}_score", random_state=SEED)
        adata.obs[f"EBV_{d}_score"] = full.obs[f"EBV_{d}_score"].values
    return res


# ---------------------------------------------------------------
# Part E: signatures for TCGA
# ---------------------------------------------------------------
def signatures(adata, n=50):
    sc.tl.rank_genes_groups(adata, "cell_type_detail", method="wilcoxon",
                            use_raw=True, pts=True)
    rows = []
    for ct in adata.obs["cell_type_detail"].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, ct)
        df = df[(df["logfoldchanges"] > 1) & (df["pvals_adj"] < 0.01)]
        if "pct_nz_group" in df and "pct_nz_reference" in df:
            df = df[(df["pct_nz_group"] > 0.25) &
                    (df["pct_nz_group"] - df["pct_nz_reference"] > 0.15)]
        df = df.head(n)
        rows.append({"cell_type": ct, "n_genes": len(df),
                     "genes": ";".join(df["names"])})
    sig = pd.DataFrame(rows)
    sig.to_csv(f"{OUT}/cell_type_signatures.csv", index=False)
    return sig


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
def figures(adata, loc):
    sc.settings.figdir = OUT
    sc.settings.set_figure_params(dpi=100, dpi_save=300, frameon=False)
    sc.pl.umap(adata, color=["cell_type", "tissue", "sample"], ncols=3, wspace=0.5,
               legend_fontsize=7, show=False, save="_overview.png")
    sc.pl.umap(adata, color=["EBV_up_score", "EBV_down_score"], cmap="viridis",
               show=False, save="_EBV_module_scores.png")
    adata.obs["cell_type_detail"] = adata.obs["cell_type_detail"].astype("category")
    sc.pl.violin(adata, ["EBV_up_score", "EBV_down_score"], groupby="cell_type_detail",
                 rotation=90, stripplot=False, show=False, save="_EBV_scores_by_cell_type.png")
    marker_list = {k: [g for g in v if g in adata.raw.var_names][:4]
                   for k, v in MARKERS.items() if k != "Cycling"}
    sc.pl.dotplot(adata, marker_list, groupby="cell_type_detail", use_raw=True,
                  show=False, save="_markers.png")
    key = [g for g in KEY_GENES if g in adata.raw.var_names]
    sc.pl.dotplot(adata, key, groupby="cell_type_detail", use_raw=True,
                  show=False, save="_key_EBV_genes.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    adata = load_atlas()
    adata = process(adata)
    ann = annotate(adata)
    ann.to_csv(f"{OUT}/cluster_annotation_scores.csv")
    print("\nCluster annotation:")
    print(ann[["assigned", "n_cells"]].to_string())
    counts = pd.crosstab(adata.obs["cell_type_detail"], adata.obs["tissue"])
    counts.to_csv(f"{OUT}/cell_type_counts.csv")
    print("\nCells per type:\n", counts.to_string())

    loc = localise(adata)
    sig = signatures(adata)
    print("\nSignatures saved:", ", ".join(f"{r.cell_type} ({r.n_genes})"
                                         for r in sig.itertuples()))
    figures(adata, loc)
    adata.obs.to_csv(f"{OUT}/cell_metadata.csv")
    try:
        adata.write_h5ad(f"{OUT}/atlas_processed.h5ad", compression="gzip")
    except Exception as e:
        print("Could not save h5ad:", e)
    print(f"\nAll results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
