"""
Count H. pylori and EBV status in TCGA-STAD.
Needs:  pip install requests pandas
Sources: GDC API (H. pylori, RNA-seq) and cBioPortal API (EBV subtype).
"""

import io
import json
import requests
import pandas as pd

GDC = "https://api.gdc.cancer.gov"


def gdc_files(filters, fields, size=5000):
    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "JSON",
        "size": size,
    }
    r = requests.get(f"{GDC}/files", params=params, timeout=120)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def and_filter(*pairs):
    return {"op": "and", "content": [
        {"op": "in", "content": {"field": f, "value": [v]}} for f, v in pairs
    ]}


# ---------------------------------------------------------------
# 1. H. pylori status (BCR Biotab clinical patient table)
# ---------------------------------------------------------------
hits = gdc_files(
    and_filter(("cases.project.project_id", "TCGA-STAD"),
               ("data_type", "Clinical Supplement"),
               ("data_format", "BCR Biotab")),
    "file_id,file_name",
)
pat_file = [h for h in hits if "clinical_patient_stad" in h["file_name"]][0]
print("Using:", pat_file["file_name"])

txt = requests.get(f"{GDC}/data/{pat_file['file_id']}", timeout=120).text
# rows 1 and 2 under the header are column descriptions, not patients
pat = pd.read_csv(io.StringIO(txt), sep="\t", skiprows=[1, 2], dtype=str)

hp_cols = [c for c in pat.columns if "pylori" in c.lower()]
print("H. pylori column(s):", hp_cols)
hp_col = hp_cols[0]

hp = pat[["bcr_patient_barcode", hp_col]].rename(columns={hp_col: "hp"})
print("\nH. pylori status, all patients:")
print(hp["hp"].value_counts(dropna=False))

# ---------------------------------------------------------------
# 2. EBV status (TCGA molecular subtype, via cBioPortal PanCancer Atlas)
# ---------------------------------------------------------------
url = ("https://www.cbioportal.org/api/studies/stad_tcga_pan_can_atlas_2018/"
       "clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE")
sub = pd.DataFrame(requests.get(url, timeout=120).json())
sub = sub[["patientId", "value"]].rename(
    columns={"patientId": "bcr_patient_barcode", "value": "subtype"})
print("\nMolecular subtypes:")
print(sub["subtype"].value_counts(dropna=False))

# ---------------------------------------------------------------
# 3. Which patients have primary-tumour RNA-seq
# ---------------------------------------------------------------
rna_hits = gdc_files(
    and_filter(("cases.project.project_id", "TCGA-STAD"),
               ("data_type", "Gene Expression Quantification"),
               ("analysis.workflow_type", "STAR - Counts"),
               ("cases.samples.sample_type", "Primary Tumor")),
    "cases.submitter_id",
)
rna_patients = {c["submitter_id"] for h in rna_hits for c in h["cases"]}
print(f"\nPatients with tumour RNA-seq: {len(rna_patients)}")

# ---------------------------------------------------------------
# 4. Merge and cross-tabulate
# ---------------------------------------------------------------
df = hp.merge(sub, on="bcr_patient_barcode", how="outer")
df["has_rna"] = df["bcr_patient_barcode"].isin(rna_patients)

print("\nH. pylori status x RNA-seq available:")
print(pd.crosstab(df["hp"].fillna("missing"), df["has_rna"]))

print("\nH. pylori status x molecular subtype (RNA-seq patients only):")
r = df[df["has_rna"]]
print(pd.crosstab(r["hp"].fillna("missing"), r["subtype"].fillna("missing")))

df.to_csv("tcga_stad_hp_ebv_status.csv", index=False)
print("\nSaved tcga_stad_hp_ebv_status.csv")