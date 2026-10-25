"""
Run the whole pipeline in order from the repository root:

    python run_all.py            # all steps
    python run_all.py 5b 5c      # only the named steps

Each step reads the outputs of the earlier ones, so run them in order the first time.
Downloads are cached, so re-running a step is safe.
"""
import subprocess
import sys
import time

STEPS = [
    ("1", "scripts/1.py"),
    ("2", "scripts/2.py"),
    ("3", "scripts/3.py"),
    ("4", "scripts/4.py"),
    ("5a", "scripts/5a_inspect_geo.py"),
    ("5b", "scripts/5b_sc_reference_atlas.py"),
    ("5c", "scripts/5c_tcga_celltype_scores.py"),
    ("6", "scripts/6_survival.py"),
    ("7", "scripts/7_make_figures.py"),
    ("8", "scripts/8_macrophage_balance.py"),
    ("7b", "scripts/7b_figure6.py"),
    ("9", "scripts/9_revision_checks.py"),
]

wanted = set(sys.argv[1:])
for name, path in STEPS:
    if wanted and name not in wanted:
        continue
    print(f"\n=== Step {name}: {path} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, path])
    if r.returncode != 0:
        sys.exit(f"Step {name} failed (exit code {r.returncode}). Fix it and re-run from this step.")
    print(f"=== Step {name} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
