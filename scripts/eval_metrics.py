"""
eval_metrics.py — Collect predictions from all models and compute AUROC / AP / PPVn.

Outputs (in ../results/):
  BenchmarkSet_metrics.csv                        — Figure 3 (ImmunoMTL + ablations + SOTA)
  zero1_metrics.csv                               — Figure 4 (ImmunoMTL + SOTA only)
  zero2_metrics.csv                               — Figure 4 (ImmunoMTL + SOTA only)
  mRNA_metrics.csv                                — Figure 5 (ImmunoMTL + SOTA only)
  BenchmarkSet_bootstrap_natural_{AUROC,AP}.csv   — bootstrap at natural pos:neg ratio
  BenchmarkSet_bootstrap_1_5_{AUROC,AP}.csv       — bootstrap at 1:5 pos:neg ratio
  BenchmarkSet_bootstrap_1_10_{AUROC,AP}.csv      — bootstrap at 1:10 pos:neg ratio

Ablations (STL, ABC, JointSTL, Shuffle) are included for Benchmark only.
PPVn = PPV at the dataset's natural positive rate (top-k where k = n_positives).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PRED   = "../pred_results"
OUT    = "../results"
os.makedirs(OUT, exist_ok=True)

N_BOOT       = 1000
N_BOOT_RATIO = 1000
RATIOS       = [5, 10]
RNG          = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ppvn(yt, yp):
    """PPV at the natural positive rate (top-k, k = number of positives)."""
    k   = int(yt.sum())
    if k == 0:
        return np.nan
    idx = np.argsort(yp)[::-1][:k]
    return float(yt[idx].sum() / k)

def metrics(yt, yp):
    mask = ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    return {
        "AUROC": roc_auc_score(yt, yp),
        "AP":    average_precision_score(yt, yp),
        "PPVn":  ppvn(yt, yp),
    }

def bootstrap(yt, yp, n=N_BOOT):
    aurocs, aps, ppvns = [], [], []
    for _ in range(n):
        idx  = RNG.choice(len(yt), len(yt), replace=True)
        yb, pb = yt[idx], yp[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aurocs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
        ppvns.append(ppvn(yb, pb))
    return np.array(aurocs), np.array(aps), np.array(ppvns)

def bootstrap_ratio(yt, yp, ratio, n=N_BOOT_RATIO, seed=0):
    """Bootstrap with negative subsampling to achieve pos:neg = 1:ratio."""
    rng  = np.random.default_rng(seed)
    pos  = np.where(yt == 1)[0]
    neg  = np.where(yt == 0)[0]
    aurocs, aps = [], []
    for _ in range(n):
        ps  = rng.choice(pos, len(pos), replace=True)
        ns  = rng.choice(neg, len(pos) * ratio, replace=True)
        idx = np.concatenate([ps, ns])
        yb, pb = yt[idx], yp[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aurocs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
    return np.array(aurocs), np.array(aps)

def load(path, score_col, label_col="Label", negate=False,
         pep_col="Peptide", mhc_col="MHC", label_source=None):
    """Load a prediction file and return (yt, yp, df).
    label_source: path to another file that carries the Label column (for
    tools like BigMHC whose output files omit it). Labels are aligned by
    Peptide+MHC merge.
    """
    df = pd.read_csv(path)
    if pep_col != "Peptide" and pep_col in df.columns:
        df = df.rename(columns={pep_col: "Peptide"})
    if mhc_col != "MHC" and mhc_col in df.columns:
        df = df.rename(columns={mhc_col: "MHC"})
    yp = df[score_col].values.astype(float)
    if negate:
        yp = -yp
    if label_source is not None:
        ref = pd.read_csv(label_source)[["Peptide", "MHC", "Label"]]
        df  = df.merge(ref, on=["Peptide", "MHC"], how="left")
    yt = df[label_col].values.astype(float)
    return yt, yp, df

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (label, path, score_col, kwargs)
# kwargs forwarded to load() — use negate=True for rank columns (lower=better)

BENCHMARK_MODELS = [
    # Main model
    ("ImmunoMTL",
     f"{PRED}/MTL/benchmark_predictions.csv",          "score",          {}),
    # Ablations
    ("STL",
     f"{PRED}/STL/benchmark_predictions.csv",      "score",          {}),
    ("ABC",
     f"{PRED}/ABC/benchmark_predictions.csv",      "score",          {}),
    ("JointSTL",
     f"{PRED}/JointSTL/benchmark_predictions.csv", "score",          {}),
    # SOTA
    ("netMHCpan",
     f"{PRED}/netMHCpan/BenchmarkSet_netMHCpan.csv",   "EL_Rank",        {"negate": True}),
    ("MHCflurry",
     f"{PRED}/mhcflurry/BenchmarkSet_mhcflurry.csv",   "mhcflurry_presentation_percentile", {"negate": True}),
    ("PRIME2",
     f"{PRED}/prime2/BenchmarkSet_prime.csv",           "PRIME_rank",     {"negate": True}),
    ("BigMHC",
     f"{PRED}/bigmhc/BenchmarkSet_bigmhc.csv",          "BigMHC_IM",
     {"pep_col": "pep", "mhc_col": "mhc",
      "label_source": f"{PRED}/MTL/benchmark_predictions.csv"}),
    ("MUNIS",
     f"{PRED}/munis/BenchmarkSet_munis_predictions.csv", "score",          {}),
]

ZEROSHOT1_MODELS = [
    ("ImmunoMTL",
     f"{PRED}/MTL/zero1_predictions.csv",              "score",           {}),
    ("netMHCpan",
     f"{PRED}/netMHCpan/zero1_netMHCpan.csv",          "EL_Rank",         {"negate": True}),
    ("MHCflurry",
     f"{PRED}/mhcflurry/zero1_mhcflurry.csv",          "mhcflurry_presentation_percentile", {"negate": True}),
    ("PRIME2",
     f"{PRED}/prime2/zero1_prime.csv",                  "PRIME_rank",      {"negate": True}),
    ("BigMHC",
     f"{PRED}/bigmhc/zero1_bigmhc.csv",                 "BigMHC_IM",
     {"pep_col": "pep", "mhc_col": "mhc",
      "label_source": f"{PRED}/MTL/zero1_predictions.csv"}),
    ("MUNIS",
     f"{PRED}/munis/zero1_munis_predictions.csv",       "score",           {}),
]

ZEROSHOT2_MODELS = [
    ("ImmunoMTL",
     f"{PRED}/MTL/zero2_predictions.csv",              "score",           {}),
    ("netMHCpan",
     f"{PRED}/netMHCpan/zero2_netMHCpan.csv",          "EL_Rank",         {"negate": True}),
    ("MHCflurry",
     f"{PRED}/mhcflurry/zero2_mhcflurry.csv",          "mhcflurry_presentation_percentile", {"negate": True}),
    ("PRIME2",
     f"{PRED}/prime2/zero2_prime.csv",                  "PRIME_rank",      {"negate": True}),
    ("BigMHC",
     f"{PRED}/bigmhc/zero2_bigmhc.csv",                 "BigMHC_IM",
     {"pep_col": "pep", "mhc_col": "mhc",
      "label_source": f"{PRED}/MTL/zero2_predictions.csv"}),
    ("MUNIS",
     f"{PRED}/munis/zero2_munis_predictions.csv",       "score",           {}),
]

MRNA_MODELS = [
    ("ImmunoMTL",
     f"{PRED}/MTL/mRNA_predictions.csv",               "score",           {}),
    ("netMHCpan",
     f"{PRED}/netMHCpan/mRNA_netMHCpan.csv",           "EL_Rank",         {"negate": True}),
    ("MHCflurry",
     f"{PRED}/mhcflurry/mRNA_mhcflurry.csv",           "mhcflurry_presentation_percentile", {"negate": True}),
    ("PRIME2",
     f"{PRED}/prime2/mRNA_prime.csv",                   "PRIME_rank",      {"negate": True}),
    ("BigMHC",
     f"{PRED}/bigmhc/mRNA_bigmhc.csv",                  "BigMHC_IM",
     {"pep_col": "pep", "mhc_col": "mhc",
      "label_source": f"{PRED}/MTL/mRNA_predictions.csv"}),
    ("MUNIS",
     f"{PRED}/munis/mRNA_munis_predictions.csv",        "score",
     {"pep_col": "pep", "mhc_col": "mhc"}),
]

# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(model_list, out_name, do_bootstrap=False):
    rows = []
    boot_auroc, boot_ap, boot_ppvn = {}, {}, {}
    raw_preds = {}   # label → (yt, yp) for ratio bootstrap

    for label, path, score_col, kwargs in model_list:
        if not os.path.exists(path):
            print(f"  [SKIP] {label}: {path} not found")
            continue
        try:
            yt, yp, _ = load(path, score_col, **kwargs)
            m = metrics(yt, yp)
            rows.append({"Model": label, **m})
            print(f"  {label:<14}  AUROC={m['AUROC']:.4f}  AP={m['AP']:.4f}  PPVn={m['PPVn']:.4f}")
            if do_bootstrap:
                a, p, q = bootstrap(yt, yp)
                boot_auroc[label] = a
                boot_ap[label]    = p
                boot_ppvn[label]  = q
                raw_preds[label]  = (yt, yp)
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")

    # Shuffle ablation — mean scores across seeds 1-10, then bootstrap for CI
    if do_bootstrap or out_name == "BenchmarkSet_metrics.csv":
        shuf_dir = f"{PRED}/immunomtl_shuffle_s1-10"
        seed_scores, seed_auroc, seed_ap, seed_ppvn = [], [], [], []
        yt_shuf = None
        for s in range(1, 11):
            p = f"{shuf_dir}/s{s}_BenchmarkSet.csv"
            if not os.path.exists(p):
                continue
            yt_s, yp_s, _ = load(p, "Predicted Score")
            if yt_shuf is None:
                yt_shuf = yt_s
            seed_scores.append(yp_s)
            m = metrics(yt_s, yp_s)
            seed_auroc.append(m["AUROC"])
            seed_ap.append(m["AP"])
            seed_ppvn.append(m["PPVn"])
        if seed_scores:
            yp_mean = np.mean(seed_scores, axis=0)
            m_mean  = metrics(yt_shuf, yp_mean)
            rows.append({
                "Model":     "Shuffle",
                "AUROC":     m_mean["AUROC"],
                "AP":        m_mean["AP"],
                "PPVn":      m_mean["PPVn"],
                "AUROC_std": np.std(seed_auroc),
                "AP_std":    np.std(seed_ap),
                "PPVn_std":  np.std(seed_ppvn),
            })
            print(f"  {'Shuffle':<14}  AUROC={m_mean['AUROC']:.4f}±{np.std(seed_auroc):.4f}"
                  f"  AP={m_mean['AP']:.4f}±{np.std(seed_ap):.4f}"
                  f"  PPVn={m_mean['PPVn']:.4f}±{np.std(seed_ppvn):.4f}")
            if do_bootstrap:
                a, p_boot, q = bootstrap(yt_shuf, yp_mean)
                boot_auroc["Shuffle"] = a
                boot_ap["Shuffle"]    = p_boot
                boot_ppvn["Shuffle"]  = q
                # Shuffle is a mean over 10 seeds — skip for ratio bootstrap

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"  → saved {out_path}\n")

    if do_bootstrap and boot_auroc:
        pd.DataFrame(boot_auroc).to_csv(
            os.path.join(OUT, "BenchmarkSet_bootstrap_natural_AUROC.csv"), index=False)
        pd.DataFrame(boot_ap).to_csv(
            os.path.join(OUT, "BenchmarkSet_bootstrap_natural_AP.csv"), index=False)
        pd.DataFrame(boot_ppvn).to_csv(
            os.path.join(OUT, "BenchmarkSet_bootstrap_natural_PPVn.csv"), index=False)
        print(f"  → saved natural bootstrap CSVs\n")

    if do_bootstrap and raw_preds:
        for ratio in RATIOS:
            rat_auroc, rat_ap = {}, {}
            for label, (yt, yp) in raw_preds.items():
                a, p = bootstrap_ratio(yt, yp, ratio)
                rat_auroc[label] = a
                rat_ap[label]    = p
            safe = f"1_{ratio}"
            pd.DataFrame(rat_auroc).to_csv(
                os.path.join(OUT, f"BenchmarkSet_bootstrap_{safe}_AUROC.csv"), index=False)
            pd.DataFrame(rat_ap).to_csv(
                os.path.join(OUT, f"BenchmarkSet_bootstrap_{safe}_AP.csv"), index=False)
        print(f"  → saved ratio bootstrap CSVs (1:5, 1:10)\n")

# ── Run ───────────────────────────────────────────────────────────────────────

print("\n=== Benchmark (Figure 3) ===")
evaluate_dataset(BENCHMARK_MODELS, "BenchmarkSet_metrics.csv", do_bootstrap=True)

print("=== ZeroShot-1 (Figure 4) ===")
evaluate_dataset(ZEROSHOT1_MODELS, "zero1_metrics.csv")

print("=== ZeroShot-2 (Figure 4) ===")
evaluate_dataset(ZEROSHOT2_MODELS, "zero2_metrics.csv")

print("=== mRNA vaccine (Figure 5) ===")
evaluate_dataset(MRNA_MODELS, "mRNA_metrics.csv")
