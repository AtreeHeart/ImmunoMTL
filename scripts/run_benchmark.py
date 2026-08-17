#!/usr/bin/env python3
"""
run_benchmark.py — Benchmark evaluation for ImmunoMTL, ablations, and SOTA tools.

Reads pre-computed prediction CSVs and prints AUROC / AP / PPVn for the
Benchmark dataset. Ablations (STL, ABC, JointSTL, Shuffle) are Benchmark-only.
SOTA tools use rank-based columns (lower = better binding), negated so higher = better.

Usage:
  python run_benchmark.py
  python run_benchmark.py --shuffle_seeds 1 2 3
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--shuffle_seeds", type=int, nargs="+",
                    default=list(range(1, 11)),
                    help="Shuffle training seeds to average (default: 1–10)")
args = parser.parse_args()

PRED = "../pred_results"


def ppvn(yt, yp):
    k = int(yt.sum())
    if k == 0:
        return np.nan
    idx = np.argsort(yp)[::-1][:k]
    return float(yt[idx].sum() / k)


def load_metrics(path, score_col="score", negate=False, label_source=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if score_col not in df.columns:
        print(f"  [WARN] column '{score_col}' not in {os.path.basename(path)}")
        return None
    if label_source is not None:
        ref = pd.read_csv(label_source)[["Peptide", "MHC", "Label"]]
        pep_col = "pep" if "pep" in df.columns else "Peptide"
        mhc_col = "mhc" if "mhc" in df.columns else "MHC"
        df = df.rename(columns={pep_col: "Peptide", mhc_col: "MHC"})
        df = df.merge(ref, on=["Peptide", "MHC"], how="left")
    if "Label" not in df.columns:
        return None
    yt = df["Label"].values.astype(float)
    yp = df[score_col].values.astype(float)
    if negate:
        yp = -yp
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
        return None
    return (roc_auc_score(yt, yp),
            average_precision_score(yt, yp),
            ppvn(yt, yp))


MTL_BENCH = f"{PRED}/MTL/benchmark_predictions.csv"

CONFIGS = [
    # ── Main model ───────────────────────────────────────────────────────────
    ("ImmunoMTL",  MTL_BENCH,                                         "score",                             {}),
    # ── Ablations (Benchmark only) ────────────────────────────────────────────
    ("STL",        f"{PRED}/STL/benchmark_predictions.csv",           "score",                             {}),
    ("ABC",        f"{PRED}/ABC/benchmark_predictions.csv",           "score",                             {}),
    ("JointSTL",   f"{PRED}/JointSTL/benchmark_predictions.csv",      "score",                             {}),
    # ── SOTA ─────────────────────────────────────────────────────────────────
    ("netMHCpan",  f"{PRED}/netMHCpan/BenchmarkSet_netMHCpan.csv",    "EL_Rank",                           {"negate": True}),
    ("MHCflurry",  f"{PRED}/mhcflurry/BenchmarkSet_mhcflurry.csv",    "mhcflurry_presentation_percentile", {"negate": True}),
    ("PRIME2",     f"{PRED}/prime2/BenchmarkSet_prime.csv",            "PRIME_rank",                        {"negate": True}),
    ("BigMHC",     f"{PRED}/bigmhc/BenchmarkSet_bigmhc.csv",           "BigMHC_IM",
     {"label_source": MTL_BENCH}),
]

# ── Header ────────────────────────────────────────────────────────────────────
print()
print(f"  {'Model':<16}  {'AUROC':>7}  {'AP':>7}  {'PPVn':>7}")
print("  " + "-" * 44)


def fmt(r):
    if r is None:
        return f"  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}"
    return f"  {r[0]:7.4f}  {r[1]:7.4f}  {r[2]:7.4f}"


for name, path, score_col, kwargs in CONFIGS:
    r = load_metrics(path, score_col=score_col, **kwargs)
    print(f"  {name:<16}{fmt(r)}")

# ── Shuffle (mean ± std across seeds) ────────────────────────────────────────
print()
shuf_dir = f"{PRED}/immunomtl_shuffle_s1-10"
rows = []
for seed in args.shuffle_seeds:
    path = f"{shuf_dir}/s{seed}_BenchmarkSet.csv"
    r = load_metrics(path, score_col="Predicted Score")
    print(f"  {'Shuffle(s'+str(seed)+')':<16}{fmt(r)}")
    if r is not None:
        rows.append(r)

if len(rows) > 1:
    arr = np.array(rows)
    mu = arr.mean(0)
    sd = arr.std(0)
    print(f"\n  {'Shuffle(mean)':<16}  {mu[0]:7.4f}  {mu[1]:7.4f}  {mu[2]:7.4f}")
    print(f"  {'Shuffle(±std)':<16}  {sd[0]:7.4f}  {sd[1]:7.4f}  {sd[2]:7.4f}")

print()
