import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def calculate_metrics(y_true, y_pred):
    y_pred = np.array(y_pred)
    return {
        "AUROC": roc_auc_score(y_true, y_pred),
        "AP": average_precision_score(y_true, y_pred),
        "F1": f1_score(y_true, (y_pred > 0.5).astype(int))
    }

def bootstrap_metrics_points(df, pred_column, metric="AUROC", n_rounds=100, pos_to_neg_ratio=None, seed=42):
    pos_df = df[df["Label"] == 1]
    neg_df = df[df["Label"] == 0]
    scores = []
    rng = np.random.default_rng(seed)

    for _ in range(n_rounds):
        pos_sample = pos_df.sample(n=len(pos_df), replace=True, random_state=rng.integers(1e6))
        if pos_to_neg_ratio:
            neg_sample = neg_df.sample(n=len(pos_sample) * pos_to_neg_ratio, replace=True, random_state=rng.integers(1e6))
        else:
            neg_sample = neg_df.sample(n=len(neg_df), replace=True, random_state=rng.integers(1e6))

        boot_df = pd.concat([pos_sample, neg_sample]).sample(frac=1).reset_index(drop=True)
        preds = boot_df[pred_column].values
        labels = boot_df["Label"].values

        try:
            if metric == "AUROC":
                score = roc_auc_score(labels, preds)
            elif metric == "AP":
                score = average_precision_score(labels, preds)
            elif metric == "F1":
                score = f1_score(labels, (preds > 0.5).astype(int))
            else:
                score = np.nan
        except:
            score = np.nan
        scores.append(score)
    return scores

# === Load dataset and clustering ===
benchmark_df = pd.read_csv("../pred_results/immunomtl/BenchmarkSet.csv")
benchmark_df["Length"] = benchmark_df["Peptide"].str.len()
hla_clusters = pd.read_csv("../HLA/clustering_res.csv")

# === Attach external scores ===
external_tools = {
    "immunoMTL_shuffle": ("../pred_results/immunomtl_shuffle/BenchmarkSet.csv", "Predicted Score"),
    "immunostl": ("../pred_results/immunostl/BenchmarkSet.csv", "Predicted Score"),
    "BigMHC_IM": ("../pred_results/bigmhc/BenchmarkSet_bigmhc.csv", "BigMHC_IM"),
    "munis": ("../pred_results/munis/BenchmarkSet_munis_predictions.csv", "score"),
    "PRIME_score": ("../pred_results/prime2/BenchmarkSet_prime.csv", "PRIME_score"),
}
for name, (path, col) in external_tools.items():
    if os.path.exists(path):
        benchmark_df[name] = pd.read_csv(path)[col]
    else:
        benchmark_df[name] = np.nan

# === Define groups and metrics ===
group_types = {
    "Length": [8, 9, 10, 11],
    "HLA_loci": ["A", "B", "C"],
    "MMS_Cluster": sorted(benchmark_df["MMS_Cluster"].dropna().unique())
}
models = ["ImmunoMTL_score"] + list(external_tools.keys())
metrics = ["AUROC", "AP", "F1"]

# === Output path ===
output_dir = "../analysis/groupwise_bootstrap"
os.makedirs(output_dir, exist_ok=True)

# === Grouped bootstrapping and saving ===
for group_name, values in group_types.items():
    for val in values:
        if group_name == "HLA_loci":
            subset = benchmark_df[benchmark_df["MHC"].str.contains(f"HLA-{val}", na=False)]
        else:
            subset = benchmark_df[benchmark_df[group_name] == val]

        for metric in metrics:
            result_matrix = {}
            for model in models:
                if model not in subset.columns:
                    continue
                mask = subset["Label"].notna() & subset[model].notna()
                if mask.sum() == 0:
                    continue
                df_group = subset.loc[mask].copy()
                scores = bootstrap_metrics_points(df_group, model, metric=metric, pos_to_neg_ratio=None, seed=42)
                result_matrix[model] = scores

            # Save per-group-per-metric result
            if result_matrix:
                df_result = pd.DataFrame(result_matrix)
                filename = f"BenchmarkSet_bootstrap_{group_name}_{val}_{metric}.csv"
                df_result.to_csv(os.path.join(output_dir, filename), index=False)
                print(f"[SAVED] {filename}")