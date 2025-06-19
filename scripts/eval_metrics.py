import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tabulate import tabulate
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

# === CONFIG ===
folders = {
    "immunomtl": "../pred_results/immunomtl",
    "immunoMTL_shuffle": "../pred_results/immunomtl_shuffle",
    "immunostl": "../pred_results/immunostl",
    "munis": "../pred_results/munis",
    "bigmhc": "../pred_results/bigmhc",
    "prime2": "../pred_results/prime2"
}
output_folder = "../analysis"
os.makedirs(output_folder, exist_ok=True)

dataset_files = sorted([f for f in os.listdir(folders["immunomtl"]) if f.endswith(".csv") and "checkmate153" not in f])
metric_columns = [
    ("ImmunoMTL_score", "ImmunoMTL"),
    ("immunoMTL_shuffle", "immunoMTL_shuffle"),
    ("immunostl", "ImmunoSTL"),
    ("BigMHC_IM", "BigMHC"),
    ("munis", "munis"),
    ("PRIME_score", "PRIME_score"),
]

benchmark_bootstrap = {
    "natural": {"AUROC": {}, "AP": {}, "F1": {}},
    "1:5": {"AUROC": {}, "AP": {}, "F1": {}},
    "1:10": {"AUROC": {}, "AP": {}, "F1": {}}
}

for file in dataset_files:
    dataset_name = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(folders["immunomtl"], file))
    print(df.shape)
    external_tools = {
        "immunoMTL_shuffle": ("immunoMTL_shuffle", "", "Predicted Score"),
        "immunostl": ("immunostl", "", "Predicted Score"),
        "BigMHC_IM": ("bigmhc", "_bigmhc", "BigMHC_IM"),
        "munis": ("munis", "_munis_predictions", "score"),
        "PRIME_score": ("prime2", "_prime", "PRIME_score"),
    }

    for key, (folder_key, suffix, col) in external_tools.items():
        path = os.path.join(folders[folder_key], file.replace(".csv", f"{suffix}.csv"))
        if os.path.exists(path):
            df[key] = pd.read_csv(path)[col]
        else:
            df[key] = np.nan
            print(f"[INFO] {key} not found for {dataset_name}")

    # === Evaluate Standard Metrics ===
    all_metrics = []
    for colname, label in metric_columns:
        if colname in df.columns:
            col_data = df[colname]
            valid_mask = df["Label"].notna() & col_data.notna()
            if valid_mask.sum() == 0:
                print(f"[SKIP] {label} for {dataset_name} has no valid predictions")
                continue
            labels = df.loc[valid_mask, "Label"]
            preds = col_data[valid_mask]

            try:
                result = calculate_metrics(labels, preds)
                result.update({"Model": label})
                all_metrics.append(result)

                # === Benchmark Bootstraps ===
                if dataset_name == "BenchmarkSet":
                    for metric in ["AUROC", "AP", "F1"]:
                        # Natural bootstrap
                        benchmark_bootstrap["natural"][metric][label] = bootstrap_metrics_points(
                            df.loc[valid_mask].copy(), colname, metric=metric, pos_to_neg_ratio=None, seed=42
                        )
                        # Ratio bootstraps
                        for ratio in [5, 10]:
                            benchmark_bootstrap[f"1:{ratio}"][metric][label] = bootstrap_metrics_points(
                                df.loc[valid_mask].copy(), colname, metric=metric, pos_to_neg_ratio=ratio, seed=42
                            )
            except Exception as e:
                print(f"[ERROR] {label} failed on {dataset_name}: {e}")

    # === Save Standard CSV ===
    out_df = pd.DataFrame(all_metrics)
    out_df = out_df[["Model", "AUROC", "AP", "F1"]]
    out_df.to_csv(os.path.join(output_folder, f"{dataset_name}_metrics.csv"), index=False)
    print(f"[SAVED] {dataset_name}_metrics.csv")

# === Save Bootstrap Results for BenchmarkSet ===
for setting in benchmark_bootstrap:
    print(setting)
    for metric in ["AUROC", "AP", "F1"]:
        boot_df = pd.DataFrame(benchmark_bootstrap[setting][metric])
        safe_setting = setting.replace(":", "_")
        boot_df.to_csv(os.path.join(output_folder, f"BenchmarkSet_bootstrap_{safe_setting}_{metric}.csv"), index=False)
        print(f"[SAVED] BenchmarkSet_bootstrap_{setting}_{metric}.csv")

# Collect and print the final metric tables
summary_tables = []

# Go through all metric summary files
for file in os.listdir(output_folder):
    if file.endswith("_metrics.csv") and not file.startswith("BenchmarkSet_bootstrap"):
        df = pd.read_csv(os.path.join(output_folder, file))
        dataset_name = file.replace("_metrics.csv", "")
        df.insert(0, "Dataset", dataset_name)
        summary_tables.append(df)

if summary_tables:
    final_df = pd.concat(summary_tables, ignore_index=True)
    print(tabulate(final_df, headers="keys", tablefmt="grid", floatfmt=".4f"))
else:
    print("[ERROR] No summary metric tables found.")