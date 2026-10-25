"""
Step 6: Clinical relevance - does the EBV immune programme relate to survival?

  Part A: survival data for TCGA-STAD (PanCancer Atlas clinical table via cBioPortal;
          OS, PFS and DSS are the curated TCGA-CDR endpoints, Liu et al. 2018 Cell)
  Part B: Kaplan-Meier by molecular subtype (EBV / CIN / GS / MSI) and by HP group
  Part C: Cox models for the core EBV signatures (per 1 SD):
            univariable -> adjusted for age, sex, stage -> + subtype
  Part D: does the IFN-g / immune score account for the EBV survival advantage?
          EBV hazard ratio with and without the score in the model
  Part E: is the score prognostic in EBV-NEGATIVE tumours too?

Needs:   pip install requests pandas numpy statsmodels matplotlib   (no new packages)
Input:   tcga_sc_results/tcga_signature_scores.csv (step 5c)
Outputs: survival_results/
"""

import os
import requests
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = "tcga_sc_results/tcga_signature_scores.csv"
OUT = "survival_results"
STUDY = "stad_tcga_pan_can_atlas_2018"
CORE = ["IFNg response", "C1Q+ macrophage", "Macrophage", "CD8 T", "NK", "Treg"]
ENDPOINTS = {"OS": ("OS_MONTHS", "OS_STATUS"), "PFS": ("PFS_MONTHS", "PFS_STATUS"),
             "DSS": ("DSS_MONTHS", "DSS_STATUS")}
SUB_COLORS = {"EBV": "#0072B2", "MSI": "#009E73", "GS": "#CC79A7", "CIN": "#8C8C8C"}


# ---------------------------------------------------------------
# Part A
# ---------------------------------------------------------------
def fetch_clinical():
    url = (f"https://www.cbioportal.org/api/studies/{STUDY}/clinical-data"
           "?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    wide = d.pivot_table(index="patientId", columns="clinicalAttributeId",
                         values="value", aggfunc="first")
    return wide


def tidy_clinical(c):
    out = pd.DataFrame(index=c.index)
    for ep, (t, s) in ENDPOINTS.items():
        if t in c and s in c:
            out[f"{ep}_time"] = pd.to_numeric(c[t], errors="coerce")
            out[f"{ep}_event"] = c[s].astype(str).str.startswith("1").astype(float)
            out.loc[c[s].isna(), f"{ep}_event"] = np.nan
    age_col = next((a for a in ["AGE", "AGE_AT_DIAGNOSIS"] if a in c), None)
    out["age"] = pd.to_numeric(c[age_col], errors="coerce") if age_col else np.nan
    out["male"] = (c.get("SEX", pd.Series(index=c.index, dtype=str))
                   .astype(str).str.upper().eq("MALE")).astype(float)
    stage_col = next((a for a in ["AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_STAGE"] if a in c), None)
    if stage_col:
        st = c[stage_col].astype(str).str.upper()
        out["stage_late"] = np.where(st.str.contains("STAGE III|STAGE IV"), 1.0,
                             np.where(st.str.contains("STAGE I"), 0.0, np.nan))
    else:
        out["stage_late"] = np.nan
    return out


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def cox(d, time, event, terms):
    d = d.dropna(subset=[time, event] + terms)
    d = d[d[time] > 0]
    if d[event].sum() < 10:
        return None, len(d), int(d[event].sum())
    fit = PHReg(d[time].values, d[terms].astype(float).values,
                status=d[event].astype(int).values, ties="efron").fit()
    ci = np.exp(fit.conf_int())
    res = pd.DataFrame({"term": terms, "HR": np.exp(fit.params),
                        "ci_low": ci[:, 0], "ci_high": ci[:, 1], "p": fit.pvalues})
    return res, len(d), int(d[event].sum())


def km_plot(d, time, event, by, order, colors, title, fname):
    d = d.dropna(subset=[time, event, by])
    d = d[d[time] > 0]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for g in order:
        x = d[d[by] == g]
        if len(x) < 3:
            continue
        sf = SurvfuncRight(x[time].values, x[event].astype(int).values)
        ax.step(np.r_[0, sf.surv_times], np.r_[1, sf.surv_prob], where="post",
                color=colors.get(g, "#444444"), lw=1.6,
                label=f"{g} (n={len(x)}, events={int(x[event].sum())})")
    keep = d[d[by].isin(order)]
    chi, p = survdiff(keep[time].values, keep[event].astype(int).values,
                      keep[by].astype(str).values)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}\nlog-rank p = {p:.3g}", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    return p


def forest(df, fname, title):
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(df) + 1.3))
    y = np.arange(len(df))
    ax.errorbar(df["HR"], y, xerr=[df["HR"] - df["ci_low"], df["ci_high"] - df["HR"]],
                fmt="o", color="#0072B2", ms=4, elinewidth=1, capsize=0)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Hazard ratio (95% CI, log scale)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


# ---------------------------------------------------------------
def main(clinical=None):
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv(SCORES, index_col=0)
    if clinical is None:
        clinical = fetch_clinical()
    clin = tidy_clinical(clinical)
    clin.to_csv(f"{OUT}/clinical_endpoints.csv")
    d = sc.join(clin, how="inner")
    d["subtype_short"] = d["subtype"].str.replace("STAD_", "")
    d["EBV"] = (d["subtype_short"] == "EBV").astype(float)
    for s in ["MSI", "GS"]:
        d[f"sub_{s}"] = (d["subtype_short"] == s).astype(float)      # CIN = reference
    d["age10"] = d["age"] / 10
    core = [c for c in CORE if c in d.columns]
    for c in core:
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    d["immune_core"] = d[[f"z_{c}" for c in core]].mean(axis=1)
    d["immune_core"] = (d["immune_core"] - d["immune_core"].mean()) / d["immune_core"].std()
    d.to_csv(f"{OUT}/analysis_table.csv")
    print(f"Tumours with scores + survival: {len(d)}")
    print(d["subtype_short"].value_counts().to_dict())
    for ep in ENDPOINTS:
        if f"{ep}_event" in d:
            print(f"  {ep}: {int(d[f'{ep}_event'].sum())} events, "
                  f"median follow-up {d[f'{ep}_time'].median():.1f} months")

    # Part B
    km = []
    for ep in ["OS", "PFS"]:
        t, e = f"{ep}_time", f"{ep}_event"
        if t not in d:
            continue
        p1 = km_plot(d, t, e, "subtype_short", ["EBV", "MSI", "GS", "CIN"], SUB_COLORS,
                     f"{ep} by molecular subtype", f"{OUT}/KM_{ep}_subtype.png")
        p2 = km_plot(d, t, e, "group", ["EBV", "HP", "Neither"],
                     {"EBV": "#0072B2", "HP": "#D55E00", "Neither": "#8C8C8C"},
                     f"{ep} by EBV / H. pylori group", f"{OUT}/KM_{ep}_group.png")
        d["IFNg_high"] = np.where(d["z_IFNg response"] >= d["z_IFNg response"].median(),
                                  "IFN-g high", "IFN-g low")
        p3 = km_plot(d, t, e, "IFNg_high", ["IFN-g high", "IFN-g low"],
                     {"IFN-g high": "#C0392B", "IFN-g low": "#7F8C8D"},
                     f"{ep} by IFN-g response score (median split)",
                     f"{OUT}/KM_{ep}_IFNg.png")
        km.append({"endpoint": ep, "logrank_subtype": p1, "logrank_group": p2,
                   "logrank_IFNg_median": p3})
    km = pd.DataFrame(km)
    km.to_csv(f"{OUT}/logrank_tests.csv", index=False)

    # Part C + D + E
    clin_terms = ["age10", "male", "stage_late"]
    rows = []
    for ep in ["OS", "PFS", "DSS"]:
        t, e = f"{ep}_time", f"{ep}_event"
        if t not in d:
            continue
        for score in core + ["immune_core"]:
            var = f"z_{score}" if score != "immune_core" else "immune_core"
            for model, terms, data in [
                    ("univariable", [var], d),
                    ("+ age, sex, stage", [var] + clin_terms, d),
                    ("+ age, sex, stage, subtype", [var] + clin_terms + ["EBV", "sub_MSI", "sub_GS"], d),
                    ("EBV-negative only, + age, sex, stage", [var] + clin_terms,
                     d[d["EBV"] == 0])]:
                res, n, ev = cox(data, t, e, terms)
                if res is None:
                    continue
                r = res.iloc[0]
                rows.append({"endpoint": ep, "score": score, "model": model, "n": n,
                             "events": ev, "HR_per_SD": r.HR, "ci_low": r.ci_low,
                             "ci_high": r.ci_high, "p": r.p})
        # mediation-style: EBV effect with / without immune score
        for label, terms in [
                ("EBV + clinical", ["EBV"] + clin_terms),
                ("EBV + clinical + immune_core", ["EBV"] + clin_terms + ["immune_core"]),
                ("EBV + clinical + IFNg response", ["EBV"] + clin_terms + ["z_IFNg response"])]:
            res, n, ev = cox(d, t, e, terms)
            if res is None:
                continue
            for _, r in res.iterrows():
                rows.append({"endpoint": ep, "score": f"[{label}] {r.term}",
                             "model": "EBV attenuation", "n": n, "events": ev,
                             "HR_per_SD": r.HR, "ci_low": r.ci_low,
                             "ci_high": r.ci_high, "p": r.p})
    cx = pd.DataFrame(rows)
    cx.to_csv(f"{OUT}/cox_models.csv", index=False)

    f = cx[(cx["endpoint"] == "OS") & (cx["model"] == "+ age, sex, stage")].copy()
    if len(f):
        f["label"] = f["score"]
        f = f.rename(columns={"HR_per_SD": "HR"})
        forest(f, f"{OUT}/forest_OS_adjusted.png",
               "OS: hazard ratio per 1 SD (adjusted for age, sex, stage)")

    pd.set_option("display.width", 200)
    print("\nLog-rank tests:")
    print(km.round(4).to_string(index=False))
    print("\nCox, per 1 SD of score (OS):")
    show = cx[(cx["endpoint"] == "OS") & (cx["model"] != "EBV attenuation")]
    print(show.pivot_table(index="score", columns="model", values="HR_per_SD").round(2).to_string())
    print("\n  p-values:")
    print(show.pivot_table(index="score", columns="model", values="p").round(3).to_string())
    print("\nDoes the immune score account for the EBV effect? (OS)")
    att = cx[(cx["endpoint"] == "OS") & (cx["model"] == "EBV attenuation")]
    print(att[["score", "n", "events", "HR_per_SD", "ci_low", "ci_high", "p"]]
          .round(3).to_string(index=False))
    print(f"\nAll results saved in ./{OUT}/")


if __name__ == "__main__":
    main()
