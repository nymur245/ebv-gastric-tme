"""
Step 9: Checks against two recent papers (revision)

  Part A  Yang et al. 2026 (Transl Res) report a JAK2-CD274 correlation of r = 0.9 that sets
          EBV gastric cancer apart. JAK2, CD274 (PD-L1) and PDCD1LG2 (PD-L2) sit together on
          9p24.1, which is amplified in many EBV tumours. Is a stronger JAK2-CD274 link in EBV
          tumours explained by this shared amplicon, while IFN-g input -> CD274 stays the same?
            - Spearman JAK2~CD274, PDCD1LG2~CD274 and IFN-g producer~CD274 within each group
            - bootstrap 95% CI for the EBV minus double-negative difference
            - the same after removing tumours with 9p24.1 gain/amplification (GISTIC, cBioPortal)
            - partial correlation JAK2~CD274 given the producer score
  Part B  McMiller et al. 2024 (JITC): after normalising to CD45 (PTPRC), EBV-negative tumours
          expressed more myeloid / immunosuppressive genes. Same test in TCGA-STAD.

Run from the repository root after steps 1-5c (uses star_counts_cache/ and
tcga_stad_hp_ebv_status.csv). Outputs: revision_checks/
Needs: pip install requests pandas numpy scipy statsmodels
"""

import json
import os

import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

GDC = "https://api.gdc.cancer.gov"
CBIO = "https://www.cbioportal.org/api"
STUDY = "stad_tcga_pan_can_atlas_2018"
CACHE = "star_counts_cache"
OUT = "revision_checks"
NON_EBV = ["STAD_CIN", "STAD_GS", "STAD_MSI"]
SEED = 0
N_BOOT = 2000

PRODUCER = ["IFNG", "GZMB", "PRF1"]
AMPLICON = {"JAK2": 3717, "CD274": 29126, "PDCD1LG2": 80380}
MCMILLER = ["PTGS2", "PTGER1", "PTGER4", "IL1B", "HAVCR2", "VSIR", "ENTPD1", "NT5E",
            "IL10", "TGFB1", "CD68", "CD163", "ITGAE", "CXCL9", "CXCL10"]
GENES = sorted(set(PRODUCER + list(AMPLICON) + MCMILLER + ["PTPRC"]))


# ---------------------------------------------------------------- data
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


def load_groups(files):
    st = pd.read_csv("tcga_stad_hp_ebv_status.csv")
    st = st[(st["has_rna"] == True) & st["subtype"].isin(NON_EBV + ["STAD_EBV"]) &
            st["bcr_patient_barcode"].isin(files.index)].copy()
    st["group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "group"] = "EBV"
    st.loc[(st["hp"] == "Yes") & st["subtype"].isin(NON_EBV), "group"] = "HP"
    st.loc[(st["hp"] == "No") & st["subtype"].isin(NON_EBV), "group"] = "Neither"
    st["sub_group"] = None
    st.loc[st["subtype"] == "STAD_EBV", "sub_group"] = "EBV"
    m = st["subtype"].isin(NON_EBV) & (st["hp"] != "Yes")
    st.loc[m, "sub_group"] = st.loc[m, "subtype"].str.replace("STAD_", "")
    have = st["bcr_patient_barcode"].map(
        lambda p: os.path.exists(f"{CACHE}/{files.loc[p, 'file_id']}.tsv"))
    return st[have].set_index("bcr_patient_barcode")


def log_tpm(patients, files):
    cols = {}
    for i, p in enumerate(patients, 1):
        t = pd.read_csv(f"{CACHE}/{files.loc[p, 'file_id']}.tsv", sep="\t", comment="#",
                        usecols=["gene_name", "tpm_unstranded"])
        cols[p] = t[t["gene_name"].isin(GENES)].groupby("gene_name")["tpm_unstranded"].sum()
        if i % 100 == 0:
            print(f"  read {i}/{len(patients)} count files")
    return np.log2(pd.DataFrame(cols).T + 1)


def gistic_9p24():
    """GISTIC calls (-2..2) for JAK2, CD274, PDCD1LG2; returns max call per patient."""
    try:
        r = requests.post(f"{CBIO}/molecular-profiles/{STUDY}_gistic/molecular-data/fetch",
                          params={"projection": "SUMMARY"},
                          json={"sampleListId": f"{STUDY}_all",
                                "entrezGeneIds": list(AMPLICON.values())}, timeout=120)
        r.raise_for_status()
        d = pd.DataFrame(r.json())
        d["patient"] = d["patientId"]
        return d.groupby("patient")["value"].max()
    except Exception as e:                                   # keep going without CNA
        print(f"NOTE: could not fetch GISTIC data from cBioPortal ({e}); "
              "the amplicon-excluded analysis is skipped.")
        return None


# ---------------------------------------------------------------- stats
def rho(x, y):
    return spearmanr(x, y).correlation


def partial_rho(x, y, z):
    """Spearman partial correlation of x and y given z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


def boot_diff(a, b, f, rng):
    """Bootstrap CI for f(group a) - f(group b); a, b are DataFrames."""
    out = []
    for _ in range(N_BOOT):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        out.append(f(a.iloc[ia]) - f(b.iloc[ib]))
    return np.nanpercentile(out, [2.5, 97.5])


PAIRS = {
    "JAK2~CD274": lambda d: rho(d["JAK2"], d["CD274"]),
    "PDCD1LG2~CD274": lambda d: rho(d["PDCD1LG2"], d["CD274"]),
    "producer~CD274": lambda d: rho(d["producer"], d["CD274"]),
    "producer~JAK2": lambda d: rho(d["producer"], d["JAK2"]),
    "JAK2~CD274 | producer": lambda d: partial_rho(d["JAK2"], d["CD274"], d["producer"]),
}


def correlation_table(d, label, rng):
    rows = []
    groups = [("group", g) for g in ["Neither", "HP", "EBV"]] + \
             [("sub_group", g) for g in ["CIN", "GS", "MSI"]]
    ref = d[d["group"] == "Neither"]
    ebv = d[d["group"] == "EBV"]
    for name, f in PAIRS.items():
        for col, g in groups:
            sub = d[d[col] == g]
            if len(sub) < 8:
                continue
            row = {"set": label, "pair": name, "group": g, "n": len(sub), "rho": f(sub)}
            if g == "EBV" and len(ref) >= 8:
                lo, hi = boot_diff(ebv, ref, f, rng)
                row.update({"EBV_minus_Neither": f(ebv) - f(ref),
                            "diff_ci_low": lo, "diff_ci_high": hi})
            rows.append(row)
    return rows


def mcmiller_table(d):
    rows = []
    for g in MCMILLER:
        if g not in d:
            continue
        y = d[g] - d["PTPRC"]                       # log2 ratio to CD45, as in McMiller
        for a, b, col in [("EBV", "Neither", "group"), ("EBV", "CIN", "sub_group"),
                          ("EBV", "GS", "sub_group"), ("EBV", "MSI", "sub_group")]:
            xa, xb = y[d[col] == a].dropna(), y[d[col] == b].dropna()
            sd = np.sqrt((xa.var() + xb.var()) / 2) + 1e-9
            rows.append({"gene": g, "comparison": f"{a}_vs_{b}", "n_A": len(xa), "n_B": len(xb),
                         "log2FC_vs_CD45": xa.median() - xb.median(),
                         "effect_SD": (xa.mean() - xb.mean()) / sd,
                         "p": mannwhitneyu(xa, xb, alternative="two-sided").pvalue})
    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for _, idx in res.groupby("comparison").groups.items():
        res.loc[idx, "fdr"] = multipletests(res.loc[idx, "p"], method="fdr_bh")[1]
    return res


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    files = gdc_count_files()
    st = load_groups(files)
    print(f"Tumours with cached counts: {len(st)}")
    expr = log_tpm(st.index, files)
    d = st[["group", "sub_group", "subtype"]].join(expr)
    z = (d[PRODUCER] - d[PRODUCER].mean()) / d[PRODUCER].std()
    d["producer"] = z.mean(axis=1)

    cn = gistic_9p24()
    if cn is not None:
        d["cn_9p24_max"] = cn.reindex(d.index)
        tab = (d.assign(gain=d["cn_9p24_max"] >= 1, amp=d["cn_9p24_max"] >= 2)
               .groupby("group")[["gain", "amp"]].mean().mul(100).round(1))
        tab["n_with_cna"] = d.groupby("group")["cn_9p24_max"].count()
        tab.to_csv(f"{OUT}/9p24_copy_number_by_group.csv")
        print("\n% tumours with 9p24.1 gain (>=1) / amplification (2):\n", tab)

    rows = correlation_table(d, "all tumours", rng)
    if cn is not None:
        rows += correlation_table(d[d["cn_9p24_max"] <= 0], "no 9p24.1 gain or amplification", rng)
    cor = pd.DataFrame(rows)
    cor.to_csv(f"{OUT}/jak2_cd274_correlations.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n", cor.round(3).to_string(index=False))

    mc = mcmiller_table(d)
    mc.to_csv(f"{OUT}/cd45_normalised_myeloid_genes.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n", mc[mc["comparison"] == "EBV_vs_Neither"].round(4).to_string(index=False))
    d.to_csv(f"{OUT}/per_tumour_values.csv")
    print(f"\nSaved results to {OUT}/")


if __name__ == "__main__":
    main()
