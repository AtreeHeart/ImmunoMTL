#!/usr/bin/env python3
"""
eval_ablations.py — Benchmark AUROC/AP/PPV@10%/mPPV for all ablation models (s22).

Reads benchmark_predictions.csv from each model's pred_results dir.
Shuffle reads from pred_results/immunomtl_shuffle_s1-10/ and reports mean ± std.

Usage:
  python eval_ablations.py
  python eval_ablations.py --shuffle_seeds 1 2 3
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--shuffle_seeds", type=int, nargs="+",
                    default=list(range(1, 11)),
                    help="Shuffle training seeds to average (default: 1–10)")
args = parser.parse_args()


def load_metrics(path, score_col="score"):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if score_col not in d.columns:
        return None
    yt = d["Label"].values.astype(float)
    yp = d[score_col].values.astype(float)
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
        return None
    return (roc_auc_score(yt, yp),
            average_precision_score(yt, yp))


CONFIGS = {
    "MTL":      "../pred_results/MTL/benchmark_predictions.csv",
    "STL":      "../pred_results/STL/benchmark_predictions.csv",
    "ABC":      "../pred_results/ABC/benchmark_predictions.csv",
    "JointSTL": "../pred_results/JointSTL/benchmark_predictions.csv",
}

SHUF_DIR = "../pred_results/immunomtl_shuffle_s1-10"

# ── Header ────────────────────────────────────────────────────────────────────
HDR = f"  {'Model':<14}  {'AUROC':>7}  {'AP':>7}"
print()
print(HDR)
print("  " + "-" * (len(HDR) - 2))

def fmt(r):
    if r is None:
        return f"  {'n/a':>7}  {'':>7}"
    return f"  {r[0]:7.4f}  {r[1]:7.4f} "

for name, path in CONFIGS.items():
    r = load_metrics(path)
    print(f"  {name:<14}{fmt(r)}")

# ── Shuffle seeds ─────────────────────────────────────────────────────────────
print()
rows = []
for seed in args.shuffle_seeds:
    path = f"{SHUF_DIR}/s{seed}_BenchmarkSet.csv"
    r = load_metrics(path, score_col="Predicted Score")
    print(f"  {'Shuffle(s'+str(seed)+')':<14}{fmt(r)}")
    if r is not None:
        rows.append(r)

if len(rows) > 1:
    arr = np.array(rows)
    mu  = arr.mean(0)
    sd  = arr.std(0)
    print(f"  {'Shuffle(mean)':<14}  {mu[0]:7.4f}  {mu[1]:7.4f}  ")
    print(f"  {'Shuffle(±std)':<14}  {sd[0]:7.4f}  {sd[1]:7.4f}  ")

print()
