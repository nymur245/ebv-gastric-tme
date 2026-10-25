# EBV and *H. pylori* in the gastric cancer tumour microenvironment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23632917.svg)](https://doi.org/10.5281/zenodo.23632917)

Analysis code for the article:

> Rahman N, et al. **Epstein-Barr virus turns up a conserved IFN-γ circuit and expands macrophages without a C1Q/SPP1 polarity shift in gastric cancer: an integrated bulk and single-cell analysis.** *[Journal, year, DOI – to be added]*

The pipeline compares EBV-positive, *H. pylori*-positive and double-negative gastric tumours in TCGA-STAD bulk RNA-seq, and uses a public gastric cancer single-cell atlas (GSE183904) to work out which cells carry each signal. Every input is public and is downloaded by the scripts themselves. No patient data are stored in this repository.

## What the pipeline does

| Step | Script | What it does | Main outputs |
| --- | --- | --- | --- |
| 1 | `scripts/1.py` | Collects *H. pylori* status (GDC BCR Biotab), molecular subtype (cBioPortal) and RNA-seq availability for TCGA-STAD | `tcga_stad_hp_ebv_status.csv` |
| 2 | `scripts/2.py` | Downloads STAR counts; DESeq2 for *H. pylori* vs Neither (subtype-adjusted) and EBV vs Neither | `star_counts_cache/`, `de_results/` |
| 3 | `scripts/3.py` | EBV vs CIN, GS and MSI; core EBV genes; pre-ranked GSEA (Hallmark, KEGG) | `de_results/core_EBV_genes.csv`, `gsea_results/` |
| 4 | `scripts/4.py` | MCP-counter scores; sample-level Hallmark scores; stroma-adjusted models | `tme_results/` |
| 5a | `scripts/5a_inspect_geo.py` | Reads GEO metadata of candidate single-cell datasets (metadata only) | `geo_inspect/` |
| 5b | `scripts/5b_sc_reference_atlas.py` | Builds the GSE183904 atlas (QC, Harmony, Leiden, annotation); maps core EBV genes to cell types; cell-type signatures | `sc_results/` |
| 5c | `scripts/5c_tcga_celltype_scores.py` | Scores TCGA with the atlas signatures; subtype-matched tests; hypoxia genes; IFN-γ circuit correlations | `tcga_sc_results/` |
| 6 | `scripts/6_survival.py` | Kaplan-Meier, log-rank and Cox models (TCGA-CDR endpoints via cBioPortal) | `survival_results/` |
| 7 | `scripts/7_make_figures.py` | Assembles Figures 1-5 | `Figures/` |
| 8 | `scripts/8_macrophage_balance.py` | Re-clusters atlas myeloid cells; derives C1Q- and SPP1-macrophage signatures; scores TCGA; macrophage-adjusted models | `macrophage_results/` |
| 7b | `scripts/7b_figure6.py` | Assembles Figure 6 (run after step 8) | `Figures/Figure6.*` |
| 9 | `scripts/9_revision_checks.py` | JAK2-CD274 correlation with and without 9p24.1 gain; CD45-normalised myeloid genes | `revision_checks/` |

### Where each result in the paper comes from

| Paper item | Script |
| --- | --- |
| Table 1, Figure 1, Tables S2-S3 | 2, 3 |
| Figure 2, Tables S4-S5 | 4 |
| Figure 3, Figures S1-S2, Tables S6-S7 | 5b |
| Figure 4, Figure S3, Tables S8-S10 | 5c |
| Figure 5, Figure S4, Table S11 | 6 |
| Figure 6, Table S12 | 8, 7b |
| Table S13 | 9 |

## Getting started

### 1. Requirements

- Python 3.10 or newer (the analysis was run with Python 3.14 on Windows)
- About 16 GB of RAM (step 5b keeps at most 2,500 random cells per sample so it fits)
- About 5 GB of free disk space for downloads and results
- An internet connection (GDC, cBioPortal, GEO and Enrichr)

### 2. Install

```bash
git clone https://github.com/Nymur245/ebv-gastric-tme.git
cd ebv-gastric-tme
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `python -m pip` from inside the virtual environment, so packages go into the same Python that runs the scripts.

### 3. Run

Run everything from the repository root:

```bash
python run_all.py
```

or one step at a time, in the order of the table above:

```bash
python scripts/1.py
python scripts/2.py
...
```

`python run_all.py 5b 5c` runs only the named steps. Downloads are cached, and interrupted downloads resume, so a failed step can simply be run again. Step 2 downloads the TCGA-STAD count files and step 5b downloads about 1-2 GB of single-cell data from GEO, so these two steps take the longest.

## Data sources

| Data | Source | Used in |
| --- | --- | --- |
| TCGA-STAD STAR gene counts (GENCODE v36) and BCR Biotab clinical tables | NCI Genomic Data Commons API, https://api.gdc.cancer.gov | 1-5c, 8 |
| Molecular subtype and survival endpoints | cBioPortal, study `stad_tcga_pan_can_atlas_2018` | 1, 6 |
| Gastric cancer single-cell RNA-seq | GEO GSE183904 (Kumar et al., Cancer Discov 2022) | 5b, 8 |
| Hallmark 2020 and KEGG 2021 gene sets | Enrichr libraries through gseapy | 3, 4, 5c |
| MCP-counter markers | https://github.com/ebecht/MCPcounter (Signatures/genes.txt) | 4 |

## Reproducibility notes

- **Random seeds** are fixed: 0 for cell sampling, PCA, Harmony, UMAP and Leiden (5b, 8); 42 for GSEA permutations (3).
- **harmonypy is pinned to 0.0.10.** Newer releases need a compiled dependency that does not build on Python 3.14.
- **Remote data can change.** GDC data releases, the cBioPortal study and the Enrichr libraries are updated from time to time, so a rerun may give slightly different numbers. The results in the paper used data downloaded in October 2026 (GDC Data Release 46.0, 10 August 2026). The processed results behind every figure are in the Supplementary Tables of the article.
- **Exact package versions** used for the paper are listed in `requirements-lock.txt`. Main ones: pydeseq2 0.5.4, gseapy 1.3.1, scanpy 1.12.4, anndata 0.13.4, harmonypy 0.0.10, leidenalg 0.12.0, igraph 1.0.0, statsmodels 0.15.0, scipy 1.18.1, numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.1. To reproduce the exact environment on Python 3.14: `python -m pip install -r requirements-lock.txt`.
- **GSE134520** is inspected in step 5a but not used: it is not EBV-annotated.

## Licence

MIT. See `LICENSE`.

## Citation

Please cite the article above and the archived code: Rahman N. *Code for: Epstein-Barr virus turns up a conserved IFN-γ circuit and expands macrophages without a C1Q/SPP1 polarity shift in gastric cancer* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23632917 (see also `CITATION.cff`).

## Contact

Nymur Rahman (ORCID [0009-0005-0328-2618](https://orcid.org/0009-0005-0328-2618)), Department of Microbiology, Khwaja Younus Ali University, Sirajganj, Bangladesh. Please open an issue on GitHub for questions about the code.
