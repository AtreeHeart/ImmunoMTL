"""
eval_subset.py — Per-subgroup bootstrap for radar charts (Fig 3e, 3f).

Groups: peptide Length (8/9/10/11), HLA locus (A/B/C), MMS_Cluster (0-3)
Metrics: AUROC, AP, PPVn
Models: ImmunoMTL, STL, ABC, JointSTL, Shuffle (mean across 10 seeds),
        BigMHC, PRIME2, netMHCpan, MHCflurry

Outputs: analysis/groupwise_bootstrap/BenchmarkSet_bootstrap_{group}_{val}_{metric}.csv

Run from scripts/:
  python eval_subset.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PRED   = "../pred_results"
OUT    = "../analysis/groupwise_bootstrap"
os.makedirs(OUT, exist_ok=True)

N_BOOT = 100
RNG    = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ppvn(yt, yp):
    k = int(yt.sum())
    if k == 0:
        return np.nan
    idx = np.argsort(yp)[::-1][:k]
    return float(yt[idx].sum() / k)

def bootstrap_scores(df, col, metric, n=N_BOOT):
    pos = df[df["Label"] == 1]
    neg = df[df["Label"] == 0]
    out = []
    for _ in range(n):
        ps = pos.sample(n=len(pos), replace=True,
                        random_state=int(RNG.integers(1_000_000)))
        ns = neg.sample(n=len(neg), replace=True,
                        random_state=int(RNG.integers(1_000_000)))
        boot = pd.concat([ps, ns]).sample(frac=1).reset_index(drop=True)
        yt = boot["Label"].values.astype(float)
        yp = boot[col].values.astype(float)
        try:
            if metric == "AUROC":
                out.append(roc_auc_score(yt, yp))
            elif metric == "AP":
                out.append(average_precision_score(yt, yp))
            elif metric == "PPVn":
                out.append(ppvn(yt, yp))
        except Exception:
            out.append(np.nan)
    return out

# ── Build base dataframe ──────────────────────────────────────────────────────

base = pd.read_csv(f"{PRED}/MTL/benchmark_predictions.csv")
base = base.rename(columns={"cluster": "MMS_Cluster", "score": "ImmunoMTL"})
base["MMS_Cluster"] = base["MMS_Cluster"].astype(float)
base["Length"]      = base["Peptide"].str.len()

# Single-model ablations
for name, path, col in [
    ("STL",      f"{PRED}/STL/benchmark_predictions.csv",      "score"),
    ("ABC",      f"{PRED}/ABC/benchmark_predictions.csv",      "score"),
    ("JointSTL", f"{PRED}/JointSTL/benchmark_predictions.csv", "score"),
]:
    if os.path.exists(path):
        base[name] = pd.read_csv(path)[col].values
    else:
        print(f"[WARN] {path} not found")
        base[name] = np.nan

# SOTA tools — rank columns negated so higher = better
for name, path, col, negate in [
    ("BigMHC",    f"{PRED}/bigmhc/BenchmarkSet_bigmhc.csv",                  "BigMHC_IM",                         False),
    ("MUNIS",     f"{PRED}/munis/BenchmarkSet_munis_predictions.csv",         "score",                             False),
    ("PRIME2",    f"{PRED}/prime2/BenchmarkSet_prime.csv",                    "PRIME_rank",                        True),
    ("netMHCpan", f"{PRED}/netMHCpan/BenchmarkSet_netMHCpan.csv",            "EL_Rank",                           True),
    ("MHCflurry", f"{PRED}/mhcflurry/BenchmarkSet_mhcflurry.csv",            "mhcflurry_presentation_percentile", True),
]:
    if os.path.exists(path):
        vals = pd.read_csv(path)[col].values.astype(float)
        base[name] = -vals if negate else vals
    else:
        print(f"[WARN] {path} not found")
        base[name] = np.nan

# BigMHC has no Label; align via Peptide+MHC from base (already in base from MTL)
# (Label already in base from MTL predictions — nothing extra needed)

# Shuffle: element-wise mean across 10 seeds
shuf_stack = []
for s in range(1, 11):
    path = f"{PRED}/immunomtl_shuffle_s1-10/s{s}_BenchmarkSet.csv"
    if os.path.exists(path):
        shuf_stack.append(pd.read_csv(path)["Predicted Score"].values.astype(float))
if shuf_stack:
    base["Shuffle"] = np.mean(shuf_stack, axis=0)
else:
    base["Shuffle"] = np.nan

MODELS  = ["ImmunoMTL", "STL", "ABC", "JointSTL", "Shuffle",
           "BigMHC", "MUNIS", "PRIME2", "netMHCpan", "MHCflurry"]
METRICS = ["AUROC", "AP", "PPVn"]

GROUPS = {
    "Length":      [8, 9, 10, 11],
    "HLA_loci":    ["A", "B", "C"],
    "MMS_Cluster": sorted(base["MMS_Cluster"].dropna().unique()),
}

# ── Bootstrap per group × metric ─────────────────────────────────────────────

for group_name, values in GROUPS.items():
    for val in values:
        if group_name == "HLA_loci":
            subset = base[base["MHC"].str.contains(f"HLA-{val}", na=False)].copy()
        else:
            subset = base[base[group_name] == val].copy()

        n_pos = subset["Label"].sum()
        n_neg = (subset["Label"] == 0).sum()
        if n_pos == 0 or n_neg == 0:
            print(f"[SKIP] {group_name}={val}: no positives or negatives")
            continue

        for metric in METRICS:
            result = {}
            for model in MODELS:
                if model not in subset.columns:
                    continue
                mask = subset["Label"].notna() & subset[model].notna()
                if mask.sum() == 0:
                    continue
                scores = bootstrap_scores(subset[mask].copy(), model, metric)
                result[model] = scores

            if result:
                fname = f"BenchmarkSet_bootstrap_{group_name}_{val}_{metric}.csv"
                pd.DataFrame(result).to_csv(os.path.join(OUT, fname), index=False)
                print(f"[SAVED] {fname}")
