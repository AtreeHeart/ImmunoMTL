#!/usr/bin/env python3
"""
predict.py — ImmunoMTL prediction script.

Usage (prediction only):
    python predict.py --input samples.csv --output predictions.csv

Usage (prediction + evaluation):
    python predict.py --input samples.csv --output predictions.csv --label Label

Input CSV must have:
  - Column 1: peptide sequences
  - Column 2: MHC allele names  (e.g. HLA-A*02:01)
  - Optional label column (0/1 integers) for evaluation mode

Output CSV adds an 'ImmunoMTL_score' column (0–1 immunogenicity score).
Rows with unsupported HLA alleles are retained with score=NaN.

Python 3.10+ | torch 2.x | transformers 4.x | scikit-learn 1.x
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── Constants (must match ImmunoMTL_training.py) ──────────────────────────────
ESM_ID  = "facebook/esm2_t12_35M_UR50D"
ESM_DIM = 480
PEP_LEN = 11
MHC_LEN = 34
N_TASKS = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL   = os.path.join(SCRIPT_DIR, "models", "ImmunoMTL_s22.pt")
DEFAULT_HLA_DIR = os.path.join(SCRIPT_DIR, "HLA")


# ── Model architecture ────────────────────────────────────────────────────────
class MTLModel(nn.Module):
    def __init__(self, n_tasks=N_TASKS):
        super().__init__()
        self.pep1 = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.pep2 = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.mhc1 = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.mhc2 = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.LeakyReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(n_tasks)])

    def forward(self, xp, xm):
        p, _ = self.pep1(xp); p, _ = self.pep2(p); p = p[:, -1, :]
        m, _ = self.mhc1(xm); m, _ = self.mhc2(m); m = m[:, -1, :]
        s = self.shared(torch.cat([p, m], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]


# ── HLA utilities ─────────────────────────────────────────────────────────────
def load_hla_resources(hla_dir):
    cluster_map    = pd.read_csv(os.path.join(hla_dir, "clustering_res.csv")).set_index("HLA")["cluster"].to_dict()
    t2_cluster_map = pd.read_csv(os.path.join(hla_dir, "t2_cluster_res_c4.csv")).set_index("HLA")["assigned_cluster"].to_dict()

    pseudo_map = {}
    pseudo_file = os.path.join(hla_dir, "MHC_pseudo.dat")
    if not os.path.exists(pseudo_file):
        raise FileNotFoundError(
            f"[ERROR] {pseudo_file} not found.\n"
            "Please install netMHCpan and copy 'MHC_pseudo.dat' to the HLA/ directory."
        )
    with open(pseudo_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pseudo_map[parts[0]] = parts[1]

    return cluster_map, t2_cluster_map, pseudo_map


def annotate_hla(df, mhc_col, cluster_map, t2_cluster_map, pseudo_map):
    def normalize(name):
        return name.replace("*", "").replace(":", "")

    df = df.copy()
    df["HLA_pseudo"]  = df[mhc_col].apply(lambda x: pseudo_map.get(normalize(str(x))))
    df["_cluster_t1"] = df[mhc_col].map(cluster_map)
    df["_cluster_t2"] = df[mhc_col].map(t2_cluster_map)
    df["cluster"]     = df["_cluster_t1"].fillna(df["_cluster_t2"])
    df["Note"]        = ""
    df.loc[df["_cluster_t1"].isna() & df["_cluster_t2"].notna(), "Note"] = (
        "Assigned from T2 cluster (9-mer MS data only; lower confidence)"
    )
    df.loc[df["cluster"].isna(), "Note"] = "Unsupported HLA allele"
    # cluster known but no pseudo sequence (rare edge case in MHC_pseudo.dat)
    df.loc[df["cluster"].notna() & df["HLA_pseudo"].isna(), "Note"] = (
        "HLA allele not in pseudo-sequence database; score unavailable"
    )
    df = df.drop(columns=["_cluster_t1", "_cluster_t2"])

    # Warn
    unsupported = df[df["cluster"].isna()][mhc_col].value_counts()
    no_pseudo   = df[df["cluster"].notna() & df["HLA_pseudo"].isna()][mhc_col].value_counts()
    t2_only     = df[df["Note"].str.startswith("Assigned")][mhc_col].value_counts()
    if not unsupported.empty:
        print(f"[WARNING] {len(unsupported)} unsupported HLA allele(s) — rows kept with score=NaN:")
        print("  " + ", ".join(unsupported.index.tolist()))
    if not no_pseudo.empty:
        print(f"[WARNING] {len(no_pseudo)} HLA allele(s) have no pseudo sequence — rows kept with score=NaN:")
        print("  " + ", ".join(no_pseudo.index.tolist()))
    if not t2_only.empty:
        print(f"[NOTE] {len(t2_only)} HLA allele(s) mapped via T2 cluster (lower confidence):")
        print("  " + ", ".join(t2_only.index.tolist()))

    return df


# ── ESM2 embedding ────────────────────────────────────────────────────────────
def embed_sequences(seqs, max_len, tok, esm, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(seqs), batch_size), leave=False):
        batch = tok(
            seqs[i:i+batch_size], return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_len, add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            embs.append(esm(**batch).last_hidden_state.cpu().numpy())
    return np.concatenate(embs, axis=0)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(df_pred, pep_emb_map, mhc_emb_map, model, device):
    # Use a Series so original (non-reset) indices are preserved for merge-back.
    scores = pd.Series(np.nan, index=df_pred.index, dtype=float)
    for tid in range(N_TASKS):
        sub = df_pred[df_pred["cluster"] == tid]
        if len(sub) == 0:
            continue
        pe  = torch.tensor(np.stack([pep_emb_map[p] for p in sub["Peptide_seq"]]),  dtype=torch.float32)
        me  = torch.tensor(np.stack([mhc_emb_map[m] for m in sub["HLA_pseudo"]]),   dtype=torch.float32)
        out = []
        for i in range(0, len(pe), 512):
            with torch.no_grad():
                logits = model(pe[i:i+512].to(device), me[i:i+512].to(device))
            out.append(torch.sigmoid(logits[tid]).cpu().numpy())
        scores.loc[sub.index] = np.concatenate(out)
    return scores


# ── Metrics ───────────────────────────────────────────────────────────────────
def ppvn(y, p):
    """Mean precision over top-k predictions, k = 1 .. num_positives."""
    idx      = np.argsort(p)[::-1]
    y_sorted = y[idx]
    cum_tp   = np.cumsum(y_sorted)
    prec     = cum_tp / np.arange(1, len(y_sorted) + 1)
    num_pos  = int(y_sorted.sum())
    if num_pos == 0:
        return np.nan
    return float(np.mean(prec[:num_pos]))


def validate_label_column(df, label_col):
    if label_col not in df.columns:
        sys.exit(f"[ERROR] Label column '{label_col}' not found in input CSV.\n"
                 f"Available columns: {list(df.columns)}")
    col = df[label_col]
    if col.isna().any():
        n = col.isna().sum()
        sys.exit(f"[ERROR] Label column '{label_col}' has {n} missing value(s). "
                 "All rows must have a label for evaluation.")
    bad = ~col.isin([0, 1])
    if bad.any():
        vals = col[bad].unique().tolist()
        sys.exit(f"[ERROR] Label column '{label_col}' contains non-binary values: {vals}. "
                 "Only 0 and 1 are accepted.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ImmunoMTL: predict T-cell immunogenicity of peptide-MHC pairs."
    )
    parser.add_argument("--input",  required=True,
                        help="Input CSV. First column = peptide, second column = MHC allele.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (scores appended as 'ImmunoMTL_score').")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Path to model weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--hla_dir", default=DEFAULT_HLA_DIR,
                        help=f"Directory containing HLA resource files (default: {DEFAULT_HLA_DIR})")
    parser.add_argument("--label", default=None,
                        help="Column name in input CSV containing 0/1 ground-truth labels. "
                             "When provided, AUROC, AP, and PPVn are reported after prediction.")
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    pep_col = df.columns[0]
    mhc_col = df.columns[1]
    print(f"[INFO] Loaded {len(df)} rows  |  peptide='{pep_col}'  mhc='{mhc_col}'")

    if args.label:
        validate_label_column(df, args.label)
        print(f"[INFO] Evaluation mode: label column = '{args.label}'")

    # ── HLA annotation ────────────────────────────────────────────────────────
    cluster_map, t2_cluster_map, pseudo_map = load_hla_resources(args.hla_dir)
    df = annotate_hla(df, mhc_col, cluster_map, t2_cluster_map, pseudo_map)
    df["Peptide_seq"] = df[pep_col]   # alias used internally

    # Rows need both a cluster assignment AND a pseudo sequence to get a score.
    can_score = df["cluster"].notna() & df["HLA_pseudo"].notna()
    df_pred = df[can_score].copy()
    df_pred["cluster"] = df_pred["cluster"].astype(int)
    n_skip = len(df) - len(df_pred)
    print(f"[INFO] {len(df_pred)} rows with supported HLA  |  {n_skip} skipped (unsupported)")

    if len(df_pred) == 0:
        sys.exit("[ERROR] No rows with supported HLA alleles. Check your MHC column format "
                 "(expected e.g. HLA-A*02:01).")

    # ── ESM2 embeddings ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading ESM2 ({ESM_ID}) ...")
    tok = AutoTokenizer.from_pretrained(ESM_ID)
    esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)

    uniq_peps = sorted(df_pred["Peptide_seq"].unique().tolist())
    uniq_mhcs = sorted(df_pred["HLA_pseudo"].unique().tolist())
    print(f"[INFO] Embedding {len(uniq_peps)} unique peptides ...")
    pep_embs = embed_sequences(uniq_peps, PEP_LEN, tok, esm, device)
    print(f"[INFO] Embedding {len(uniq_mhcs)} unique MHC pseudo-sequences ...")
    mhc_embs = embed_sequences(uniq_mhcs, MHC_LEN, tok, esm, device)

    pep_emb_map = dict(zip(uniq_peps, pep_embs))
    mhc_emb_map = dict(zip(uniq_mhcs, mhc_embs))
    del esm; torch.cuda.empty_cache()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading checkpoint: {args.model}")
    model = MTLModel(n_tasks=N_TASKS).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    print("[INFO] Running predictions ...")
    scores = run_inference(df_pred, pep_emb_map, mhc_emb_map, model, device)

    # Assign scores back into the original df — preserves input row order.
    df["ImmunoMTL_score"] = np.nan
    df.loc[scores.index, "ImmunoMTL_score"] = scores.values
    df_out = df.drop(columns=["HLA_pseudo", "cluster", "Peptide_seq"], errors="ignore")

    df_out.to_csv(args.output, index=False)
    print(f"[INFO] Saved {len(df_out)} rows → {args.output}")

    # ── Evaluation (optional) ─────────────────────────────────────────────────
    if args.label:
        from sklearn.metrics import roc_auc_score, average_precision_score

        eval_df = df.loc[scores.index].copy()
        eval_df["ImmunoMTL_score"] = scores.values
        eval_df = eval_df.dropna(subset=["ImmunoMTL_score"])
        y = eval_df[args.label].values.astype(float)
        p = eval_df["ImmunoMTL_score"].values.astype(float)

        if y.sum() == 0 or y.sum() == len(y):
            print("[WARNING] Evaluation skipped: labels are all one class after HLA filtering.")
        else:
            auroc = roc_auc_score(y, p)
            ap    = average_precision_score(y, p)
            pv    = ppvn(y, p)
            n_pos = int(y.sum())
            n_tot = len(y)
            print(f"\n{'─'*42}")
            print(f"  Evaluation on {n_tot} rows ({n_pos} positives)")
            print(f"  AUROC : {auroc:.4f}")
            print(f"  AP    : {ap:.4f}")
            print(f"  PPVn  : {pv:.4f}")
            print(f"{'─'*42}\n")


if __name__ == "__main__":
    main()
