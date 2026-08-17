"""
1_prepare_split.py

Partition IEDB and CEDAR data into seen/novel sets based on ImmunoMTL training keys.
  seen   = pHLA pairs present in ImmunoMTL training  → IS training pool
  novel  = pHLA pairs NOT in ImmunoMTL training      → held-out test set for both models

Outputs (../data/ relative to this script):
  training_combined.csv   source, allele, peptide, immunogenicity  (IS train/val pool)
  combined_split.json     train/val index split (90/10, torch seed=1)
  iedb_novel.csv          IEDB novel test set (IS full-feature format)
  cedar_novel.csv         CEDAR novel test set (IS full-feature format)

Note: ../data/ is gitignored. Upload training_combined.csv to Mendeley.
"""

import os, json
import torch
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MTL_ROOT   = os.path.join(SCRIPT_DIR, "..", "..", "..")   # /path/to/ImmunoMTL
DATA_DIR   = os.path.join(MTL_ROOT, "data")
IS_DATA    = os.path.join(MTL_ROOT, "data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_SEED = 1

# ── ImmunoMTL training keys ────────────────────────────────────────────────────
train_df   = pd.read_csv(os.path.join(DATA_DIR, "training.csv"))
train_keys = set((train_df["Peptide"].str.upper() + "_" + train_df["MHC"]).tolist())
print(f"ImmunoMTL training keys: {len(train_keys)}")


def is_filter_and_dedup(df, pep_col, allele_col):
    key = df[pep_col].str.upper() + "_" + df[allele_col]
    bad = df[pep_col].str.contains("X", na=False) | key.str.contains("NXVPMVATV", na=False)
    df  = df[~bad].copy()
    df  = df.drop_duplicates(subset=[allele_col, pep_col], keep="first").reset_index(drop=True)
    return df


# ── IEDB ───────────────────────────────────────────────────────────────────────
print("\n── IEDB ──")
iedb = pd.read_csv(os.path.join(IS_DATA, "ImmunoStruct_IEDB_data.csv"))
iedb = is_filter_and_dedup(iedb, "peptide", "allele")
iedb["_key"] = iedb["peptide"].str.upper() + "_" + iedb["allele"]

iedb_seen  = iedb[iedb["_key"].isin(train_keys)].reset_index(drop=True)
iedb_novel = iedb[~iedb["_key"].isin(train_keys)].reset_index(drop=True)
print(f"  Seen: {len(iedb_seen)}  Novel: {len(iedb_novel)}  (pos={int(iedb_novel['immunogenicity'].sum())})")

# ── CEDAR ──────────────────────────────────────────────────────────────────────
print("\n── CEDAR ──")
cedar = pd.read_csv(os.path.join(IS_DATA, "ImmunoStruct_CEDAR_data_cancer.csv"))
cedar = is_filter_and_dedup(cedar, "mut_pep", "allele")
cedar["_key"] = cedar["mut_pep"].str.upper() + "_" + cedar["allele"]

cedar_seen  = cedar[cedar["_key"].isin(train_keys)].reset_index(drop=True)
cedar_novel = cedar[~cedar["_key"].isin(train_keys)].reset_index(drop=True)
print(f"  Seen: {len(cedar_seen)}  Novel: {len(cedar_novel)}  (pos={int(cedar_novel['immunogenicity'].sum())})")

# ── Combined seen → 90/10 train/val ───────────────────────────────────────────
N_combined = len(iedb_seen) + len(cedar_seen)
gen  = torch.Generator().manual_seed(SPLIT_SEED)
perm = torch.randperm(N_combined, generator=gen).tolist()
n_val     = max(1, N_combined // 10)
val_idx   = sorted(perm[:n_val])
train_idx = sorted(perm[n_val:])
print(f"\nCombined split (seed={SPLIT_SEED}): train={len(train_idx)}  val={len(val_idx)}")

# ── Save training_combined.csv (gitignored — upload to Mendeley) ─────────────
iedb_out  = iedb_seen[["allele","peptide","immunogenicity"]].copy()
iedb_out.insert(0, "source", "IEDB")
cedar_out = cedar_seen[["allele","mut_pep","immunogenicity"]].rename(columns={"mut_pep":"peptide"}).copy()
cedar_out.insert(0, "source", "CEDAR")
train_comb = pd.concat([iedb_out, cedar_out], ignore_index=True)
train_comb.to_csv(os.path.join(OUT_DIR, "training_combined.csv"), index=False)
print(f"[SAVED] training_combined.csv  ({len(train_comb)} rows)")

# ── Save full-feature CSVs for IS training ────────────────────────────────────
iedb_seen.drop(columns=["_key"]).to_csv(os.path.join(OUT_DIR, "iedb_seen_IS_features.csv"),   index=False)
cedar_seen.drop(columns=["_key"]).to_csv(os.path.join(OUT_DIR, "cedar_seen_IS_features.csv"), index=False)
iedb_novel.drop(columns=["_key"]).to_csv(os.path.join(OUT_DIR, "iedb_novel.csv"),             index=False)
cedar_novel.drop(columns=["_key"]).to_csv(os.path.join(OUT_DIR, "cedar_novel.csv"),           index=False)
print(f"[SAVED] iedb_seen_IS_features.csv, cedar_seen_IS_features.csv, iedb_novel.csv, cedar_novel.csv")

# ── Save combined split JSON ───────────────────────────────────────────────────
split_info = {
    "train": train_idx,
    "val":   val_idx,
    "metadata": {
        "N_iedb_seen":        len(iedb_seen),
        "N_cedar_seen":       len(cedar_seen),
        "N_combined":         N_combined,
        "N_train":            len(train_idx),
        "N_val":              len(val_idx),
        "N_test_iedb_novel":  len(iedb_novel),
        "N_test_cedar_novel": len(cedar_novel),
        "split_seed":         SPLIT_SEED,
        "split_method":       "torch.randperm with torch.Generator",
        "concat_order":       "ConcatDataset([ImmunoPredDataset(iedb_seen_IS_features.csv), ImmunoPredDataset(cedar_seen_IS_features.csv)])",
        "description":        (
            "Train/val = combined IEDB seen + CEDAR seen (90/10). "
            "Test = IEDB novel + CEDAR novel separately. "
            "Indices 0..N_iedb_seen-1 → IEDB; N_iedb_seen..N_combined-1 → CEDAR."
        ),
    },
}
json_path = os.path.join(OUT_DIR, "combined_split.json")
with open(json_path, "w") as f:
    json.dump(split_info, f, indent=2)
print(f"[SAVED] combined_split.json")
print("\n[DONE] Run 2_train_IS_combined.sh next.")
