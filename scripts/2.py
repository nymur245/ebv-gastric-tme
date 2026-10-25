"""
Step 2: Differential expression in TCGA-STAD
  Model A: H. pylori-only vs Neither, adjusted for molecular subtype
  Model B: EBV vs Neither
Then splits significant genes into shared / H. pylori-specific / EBV-specific.

Needs:   pip install requests pandas pydeseq2
Input:   tcga_stad_hp_ebv_status.csv  (from the first script, same folder)
Outputs: counts cache folder, DE tables, gene lists (all in ./de_results)
"""

import io
import os
import json
import tarfile
import requests
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

GDC = "https://api.gdc.cancer.gov"
CACHE = "star_counts_cache"
OUT = "de_results"
PADJ, LFC = 0.05, 1.0          # significance thresholds
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

NON_EBV = ["STAD_CIN", "STAD_GS", "STAD_MSI"]   # POLE dropped: too few samples

# ---------------------------------------------------------------
# 1. Define groups
# ---------------------------------------------------------------
st = pd.read_csv("tcga_stad_hp_ebv_status.csv")
st = st[st["has_rna"] == True].copy()

st["group"] = None
st.loc[st["subtype"] == "STAD_EBV", "group"] = "EBV"
st.loc[(st["hp"] == "Yes") & st["subtype"].isin(NON_EBV), "group"] = "HP"
st.loc[(st["hp"] == "No") & st["subtype"].isin(NON_EBV), "group"] = "Neither"
st = st.dropna(subset=["group"])
print("Group sizes:\n", st["group"].value_counts(), "\n")

# ---------------------------------------------------------------
# 2. Find one primary-tumour STAR count file per patient
# ---------------------------------------------------------------
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

rows = []
for h in r.json()["data"]["hits"]:
    case = h["cases"][0]
    rows.append({"patient": case["submitter_id"],
                 "sample": case["samples"][0]["submitter_id"],
                 "file_id": h["file_id"]})
files = (pd.DataFrame(rows)
         .sort_values("sample")              # prefers -01A over -01B
         .drop_duplicates("patient")
         .set_index("patient"))
st = st[st["bcr_patient_barcode"].isin(files.index)]
need = files.loc[st["bcr_patient_barcode"], "file_id"].tolist()

# ---------------------------------------------------------------
# 3. Download (batches of 40, skips files already cached)
# ---------------------------------------------------------------
todo = [f for f in need if not os.path.exists(f"{CACHE}/{f}.tsv")]
print(f"Files needed: {len(need)}, to download: {len(todo)}")
for i in range(0, len(todo), 40):
    batch = todo[i:i + 40]
    resp = requests.post(f"{GDC}/data", json={"ids": batch}, timeout=600)
    resp.raise_for_status()
    if len(batch) == 1:                       # single file comes back raw
        open(f"{CACHE}/{batch[0]}.tsv", "wb").write(resp.content)
    else:                                     # several come back as tar.gz
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".tsv"):
                    fid = m.name.split("/")[0]
                    open(f"{CACHE}/{fid}.tsv", "wb").write(tar.extractfile(m).read())
    print(f"  downloaded {min(i + 40, len(todo))}/{len(todo)}")

# ---------------------------------------------------------------
# 4. Build count matrix (samples x genes, unstranded counts)
# ---------------------------------------------------------------
counts, annot = {}, None
for _, row in st.iterrows():
    fid = files.loc[row["bcr_patient_barcode"], "file_id"]
    t = pd.read_csv(f"{CACHE}/{fid}.tsv", sep="\t", comment="#")
    t = t[~t["gene_id"].str.startswith("N_")].set_index("gene_id")
    counts[row["bcr_patient_barcode"]] = t["unstranded"]
    if annot is None:
        annot = t[["gene_name", "gene_type"]]
counts = pd.DataFrame(counts).T.astype(int)

# light filter: >=10 reads in at least 10 samples
counts = counts.loc[:, (counts >= 10).sum(axis=0) >= 10]
print(f"\nCount matrix: {counts.shape[0]} samples x {counts.shape[1]} genes")

meta = st.set_index("bcr_patient_barcode")[["group", "subtype"]]
meta = meta.loc[counts.index]


# ---------------------------------------------------------------
# 5. DESeq2 helper (works with old and new pydeseq2 versions)
# ---------------------------------------------------------------
def run_deseq(cnt, md, factors, contrast, name):
    try:
        dds = DeseqDataSet(counts=cnt, metadata=md,
                           design="~" + " + ".join(factors))
    except TypeError:
        dds = DeseqDataSet(counts=cnt, metadata=md, design_factors=factors)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=contrast)
    stats.summary()
    res = stats.results_df.join(annot)
    res.to_csv(f"{OUT}/{name}_all_genes.csv")
    sig = res[(res["padj"] < PADJ) & (res["log2FoldChange"].abs() >= LFC)]
    print(f"{name}: {len(sig)} significant genes "
          f"({(sig.log2FoldChange > 0).sum()} up, {(sig.log2FoldChange < 0).sum()} down)")
    return res, sig


# Model A: H. pylori vs Neither, subtype as covariate
mA = meta[meta["group"].isin(["HP", "Neither"])].copy()
for c in ["group", "subtype"]:
    mA[c] = mA[c].astype("category")
_, sigA = run_deseq(counts.loc[mA.index], mA, ["subtype", "group"],
                    ["group", "HP", "Neither"], "HP_vs_Neither")

# Model B: EBV vs Neither
# (subtype can't be a covariate here: EBV is itself a subtype)
mB = meta[meta["group"].isin(["EBV", "Neither"])].copy()
mB["group"] = mB["group"].astype("category")
_, sigB = run_deseq(counts.loc[mB.index], mB, ["group"],
                    ["group", "EBV", "Neither"], "EBV_vs_Neither")

# ---------------------------------------------------------------
# 6. Shared vs pathogen-specific genes
# ---------------------------------------------------------------
both = sigA.index.intersection(sigB.index)
same_dir = [g for g in both
            if (sigA.loc[g, "log2FoldChange"] > 0) == (sigB.loc[g, "log2FoldChange"] > 0)]
opposite = [g for g in both if g not in same_dir]
hp_only = sigA.index.difference(sigB.index)
ebv_only = sigB.index.difference(sigA.index)

summary = pd.concat([
    pd.DataFrame({"gene_id": same_dir, "category": "shared_same_direction"}),
    pd.DataFrame({"gene_id": opposite, "category": "shared_opposite_direction"}),
    pd.DataFrame({"gene_id": hp_only, "category": "HP_specific"}),
    pd.DataFrame({"gene_id": ebv_only, "category": "EBV_specific"}),
]).set_index("gene_id")
summary = summary.join(annot)
summary["log2FC_HP"] = sigA["log2FoldChange"].reindex(summary.index)
summary["log2FC_EBV"] = sigB["log2FoldChange"].reindex(summary.index)
summary.to_csv(f"{OUT}/gene_categories.csv")

print("\nGene categories:")
print(summary["category"].value_counts())
print(f"\nAll results saved in ./{OUT}/")