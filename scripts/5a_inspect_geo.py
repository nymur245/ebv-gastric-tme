"""
Step 5a: Look inside the two candidate single-cell datasets before downloading them
  GSE134520  Zhang et al. 2019, Cell Reports  - gastric premalignant lesions + early GC (not EBV-typed; inspected, not used)
  GSE183904  Kumar et al. 2022, Cancer Discov - large gastric cancer atlas (EBV status unclear)

For each dataset it saves, into ./geo_inspect/:
  <GSE>_samples.csv   one row per sample: GSM, title, all characteristics, file links
  <GSE>_files.txt     series-level supplementary files
and prints any sample text that mentions EBV / Epstein.

Needs: pip install requests pandas
Only metadata is downloaded here (a few hundred KB), no expression data.
"""

import os
import re
import requests
import pandas as pd

GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
OUT = "geo_inspect"
DATASETS = ["GSE134520", "GSE183904"]


def geo_text(acc, targ):
    r = requests.get(GEO, params={"acc": acc, "targ": targ,
                                  "form": "text", "view": "full"}, timeout=120)
    r.raise_for_status()
    return r.text


def parse_samples(txt):
    rows, cur = [], None
    for line in txt.splitlines():
        if line.startswith("^SAMPLE"):
            if cur:
                rows.append(cur)
            cur = {"GSM": line.split("=", 1)[1].strip()}
        elif cur is not None and line.startswith("!Sample_"):
            key, _, val = line[8:].partition(" = ")
            key = key.strip()
            if key == "characteristics_ch1" and ":" in val:
                k, v = val.split(":", 1)
                cur[f"char_{k.strip()}"] = v.strip()
            elif key in cur:
                cur[key] = f"{cur[key]} | {val.strip()}"
            else:
                cur[key] = val.strip()
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for acc in DATASETS:
        print(f"\n================ {acc} ================")
        series = geo_text(acc, "self")
        title = re.findall(r"!Series_title = (.*)", series)
        print("Title:", title[0] if title else "?")
        files = re.findall(r"!Series_supplementary_file = (.*)", series)
        with open(f"{OUT}/{acc}_files.txt", "w") as fh:
            fh.write("\n".join(files))
        print(f"Series files ({len(files)}):")
        for f in files:
            print("  ", f)

        df = parse_samples(geo_text(acc, "gsm"))
        keep = ["GSM", "title", "source_name_ch1", "molecule_ch1",
                "library_strategy", "supplementary_file_1"]
        keep += [c for c in df.columns if c.startswith("char_")]
        keep += [c for c in df.columns if c in ("description",
                                                 "supplementary_file_2")]
        df = df[[c for c in dict.fromkeys(keep) if c in df.columns]]
        df.to_csv(f"{OUT}/{acc}_samples.csv", index=False)
        print(f"\nSamples: {len(df)}")
        with pd.option_context("display.max_colwidth", 60, "display.width", 250):
            print(df.drop(columns=[c for c in df.columns
                                   if c.startswith("supplementary")]).to_string(index=False))

        hits = df[df.astype(str).apply(
            lambda r: r.str.contains("EBV|Epstein", case=False).any(), axis=1)]
        print(f"\nSamples mentioning EBV/Epstein: {len(hits)}")
        if len(hits):
            print(hits[["GSM", "title"]].to_string(index=False))
        if "EBV" in series or "Epstein" in series:
            print("(The series description itself mentions EBV.)")

    print(f"\nSaved to ./{OUT}/")


if __name__ == "__main__":
    main()
