"""
Step 3: Subtype-matched EBV comparisons + pathway-level GSEA
  Part A: EBV vs CIN, EBV vs GS, EBV vs MSI (one subtype at a time)
          -> "core EBV genes" = significant, same direction, in all three
  Part B: GSEA (pre-ranked on the Wald stat) for every comparison:
          HP_vs_Neither, EBV_vs_Neither (from step 2) + the three new ones
          Gene sets: MSigDB Hallmark 2020 and KEGG 2021
  Part C: NES heatmap across comparisons (Hallmark)

Needs:   pip install requests pandas pydeseq2 gseapy matplotlib seaborn
Input:   tcga_stad_hp_ebv_status.csv, star_counts_cache/, de_results/  (from steps 1-2)
Outputs: de_results/EBV_vs_<subtype>_all_genes.csv, de_results/core_EBV_genes.csv,
         gsea_results/*.csv, gsea_results/hallmark_NES_heatmap.png

Comparator groups in Part A: all tumours of that subtype that are NOT H. pylori
positive (hp = No / Not Available / Unknown). Excluding only HP-positive cases
keeps the groups large enough while removing the obvious confounder.
Missing count files are downloaded into the same cache.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

GDC = "https://api.gdc.cancer.gov"
CACHE = "star_counts_cache"
DE = "de_results"
GS_OUT = "gsea_results"
PADJ, LFC = 0.05, 1.0
SUBTYPES = ["STAD_CIN", "STAD_GS", "STAD_MSI"]
GENE_SETS = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"]
SEED = 42


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def gdc_count_files():
    """One primary-tumour STAR count file per patient (prefers -01A)."""
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


def save(path, data):
    """Write to a temp file first so a broken download never leaves a half file."""
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def fetch_batch(batch, tries=5):
    """POST one batch to GDC, retrying with a growing wait on timeouts/errors."""
    for attempt in range(1, tries + 1):
        try:
            resp = requests.post(f"{GDC}/data", json={"ids": batch},
                                 timeout=(30, 300))
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as e:
            wait = 15 * attempt
            print(f"    attempt {attempt}/{tries} failed ({type(e).__name__}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    return None


def download(file_ids, batch_size=10):
    todo = [f for f in file_ids if not os.path.exists(f"{CACHE}/{f}.tsv")]
    print(f"Count files needed: {len(file_ids)}, to download: {len(todo)}")
    failed = []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        content = fetch_batch(batch)
        if content is None:                      # batch kept failing: go one by one
            for fid in batch:
                c = fetch_batch([fid])
                if c is None:
                    failed.append(fid)
                else:
                    save(f"{CACHE}/{fid}.tsv", c)
        elif len(batch) == 1:
            save(f"{CACHE}/{batch[0]}.tsv", content)
        else:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if m.name.endswith(".tsv"):
                        fid = m.name.split("/")[0]
                        save(f"{CACHE}/{fid}.tsv", tar.extractfile(m).read())
        print(f"  downloaded {min(i + batch_size, len(todo))}/{len(todo)}")
    if failed:
        raise SystemExit(f"\n{len(failed)} files could not be downloaded. "
                         "Just run the script again - finished files are kept.")


def build_counts(patients, files):
    counts, annot = {}, None
    for p in patients:
        t = pd.read_csv(f"{CACHE}/{files.loc[p, 'file_id']}.tsv",
                        sep="\t", comment="#")
        t = t[~t["gene_id"].str.startswith("N_")].set_index("gene_id")
        counts[p] = t["unstranded"]
        if annot is None:
            annot = t[["gene_name", "gene_type"]]
    counts = pd.DataFrame(counts).T.astype(int)
    counts = counts.loc[:, (counts >= 10).sum(axis=0) >= 10]
    return counts, annot


def run_deseq(cnt, md, contrast, name, annot):
    try:
        dds = DeseqDataSet(counts=cnt, metadata=md, design="~group")
    except TypeError:
        dds = DeseqDataSet(counts=cnt, metadata=md, design_factors="group")
    dds.deseq2()
    stats = DeseqStats(dds, contrast=contrast)
    stats.summary()
    res = stats.results_df.join(annot)
    res.to_csv(f"{DE}/{name}_all_genes.csv")
    sig = res[(res["padj"] < PADJ) & (res["log2FoldChange"].abs() >= LFC)]
    print(f"{name}: {len(sig)} significant "
          f"({(sig.log2FoldChange > 0).sum()} up, "
          f"{(sig.log2FoldChange < 0).sum()} down)")
    return res, sig


def ranked_list(res):
    """Gene symbol -> Wald stat; one row per symbol (largest |stat| kept)."""
    r = res.dropna(subset=["stat", "gene_name"])
    r = r[r["gene_type"] == "protein_coding"]
    r = r.reindex(r["stat"].abs().sort_values(ascending=False).index)
    r = r.drop_duplicates("gene_name")
    return r.set_index("gene_name")["stat"].sort_values(ascending=False)


def run_gsea(rnk, gene_set, name):
    pre = gp.prerank(rnk=rnk, gene_sets=gene_set, permutation_num=1000,
                     min_size=15, max_size=500, seed=SEED, threads=4,
                     outdir=None, verbose=False)
    df = pre.res2d.copy()
    if "Term" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Term"})
    for c in ["ES", "NES", "NOM p-val", "FDR q-val"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "comparison", name)
    df.insert(1, "library", gene_set)
    return df.sort_values("NES", ascending=False)


def main():
    os.makedirs(GS_OUT, exist_ok=True)

    # -----------------------------------------------------------
    # Part A: EBV vs each non-EBV subtype
    # -----------------------------------------------------------
    st = pd.read_csv("tcga_stad_hp_ebv_status.csv")
    st = st[st["has_rna"] == True].copy()
    st["group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "group"] = "EBV"
    comp = st["subtype"].isin(SUBTYPES) & (st["hp"] != "Yes")
    st.loc[comp, "group"] = st.loc[comp, "subtype"].str.replace("STAD_", "")
    st = st.dropna(subset=["group"])

    files = gdc_count_files()
    st = st[st["bcr_patient_barcode"].isin(files.index)]
    print("Group sizes:\n", st["group"].value_counts(), "\n")

    download(files.loc[st["bcr_patient_barcode"], "file_id"].tolist())
    counts, annot = build_counts(st["bcr_patient_barcode"].tolist(), files)
    meta = st.set_index("bcr_patient_barcode")[["group"]].loc[counts.index]
    print(f"Count matrix: {counts.shape[0]} samples x {counts.shape[1]} genes\n")

    results = {}
    sigs = {}
    for sub in ["CIN", "GS", "MSI"]:
        name = f"EBV_vs_{sub}"
        m = meta[meta["group"].isin(["EBV", sub])].copy()
        m["group"] = m["group"].astype("category")
        cnt = counts.loc[m.index]
        cnt = cnt.loc[:, (cnt >= 10).sum(axis=0) >= 10]
        results[name], sigs[name] = run_deseq(
            cnt, m, ["group", "EBV", sub], name, annot)

    # core EBV genes: significant + same direction in all three comparisons
    common = set.intersection(*[set(s.index) for s in sigs.values()])
    core = []
    for g in common:
        lfcs = [sigs[n].loc[g, "log2FoldChange"] for n in sigs]
        if all(x > 0 for x in lfcs) or all(x < 0 for x in lfcs):
            core.append({"gene_id": g,
                         "gene_name": annot.loc[g, "gene_name"],
                         "gene_type": annot.loc[g, "gene_type"],
                         "direction": "up" if lfcs[0] > 0 else "down",
                         **{f"log2FC_{n}": sigs[n].loc[g, "log2FoldChange"]
                            for n in sigs},
                         **{f"padj_{n}": sigs[n].loc[g, "padj"] for n in sigs}})
    core = pd.DataFrame(core)
    if len(core):
        core["mean_log2FC"] = core.filter(like="log2FC_").mean(axis=1)
        core = core.sort_values("mean_log2FC", ascending=False)
    core.to_csv(f"{DE}/core_EBV_genes.csv", index=False)
    print(f"\nCore EBV genes (all 3 comparisons, same direction): {len(core)}")
    if len(core):
        print(core["direction"].value_counts().to_string())

    # -----------------------------------------------------------
    # Part B: GSEA on every comparison
    # -----------------------------------------------------------
    for prev in ["HP_vs_Neither", "EBV_vs_Neither"]:
        results[prev] = pd.read_csv(f"{DE}/{prev}_all_genes.csv", index_col=0)

    order = ["HP_vs_Neither", "EBV_vs_Neither",
             "EBV_vs_CIN", "EBV_vs_GS", "EBV_vs_MSI"]
    all_gsea = []
    for name in order:
        rnk = ranked_list(results[name])
        for lib in GENE_SETS:
            print(f"GSEA: {name} / {lib} ({len(rnk)} genes)")
            df = run_gsea(rnk, lib, name)
            df.to_csv(f"{GS_OUT}/{name}_{lib}.csv", index=False)
            all_gsea.append(df)
    all_gsea = pd.concat(all_gsea)
    all_gsea.to_csv(f"{GS_OUT}/all_gsea_combined.csv", index=False)

    # -----------------------------------------------------------
    # Part C: Hallmark NES heatmap (* = FDR < 0.05)
    # -----------------------------------------------------------
    h = all_gsea[all_gsea["library"] == "MSigDB_Hallmark_2020"]
    nes = h.pivot(index="Term", columns="comparison", values="NES")[order]
    fdr = h.pivot(index="Term", columns="comparison", values="FDR q-val")[order]
    keep = (fdr < 0.05).any(axis=1)
    nes, fdr = nes[keep], fdr[keep]
    nes = nes.loc[nes.mean(axis=1).sort_values(ascending=False).index]
    fdr = fdr.loc[nes.index]
    nes.to_csv(f"{GS_OUT}/hallmark_NES_matrix.csv")

    star = fdr.map(lambda x: "*" if x < 0.05 else "") if hasattr(fdr, "map") \
        else fdr.applymap(lambda x: "*" if x < 0.05 else "")
    lim = np.nanmax(np.abs(nes.values))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(nes))))
    sns.heatmap(nes, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim,
                annot=star, fmt="", linewidths=0.4, linecolor="white",
                cbar_kws={"label": "NES"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Hallmark GSEA (* FDR < 0.05)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"{GS_OUT}/hallmark_NES_heatmap.png", dpi=300)
    print(f"\nHallmark pathways with FDR < 0.05 in any comparison: {len(nes)}")
    print(f"All GSEA results saved in ./{GS_OUT}/")


if __name__ == "__main__":
    main()