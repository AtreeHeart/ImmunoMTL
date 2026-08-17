#!/usr/bin/env python3
"""
eval_shuffle.py — Inference-only evaluation for MTL-Shuffle models (training seeds 1–10).

Architecture: ESM2-t12-35M (frozen) → dual BiLSTM (pep, MHC separate, 480→128)
              → shared FC(256→64→16) → 4 randomly-assigned cluster heads, sigmoid outside.
Clusters use fixed random HLA assignment (cs=10); training seed varies across seeds 1–10.

Checkpoints:
  revision/models/finetune/Shuffle_s{s}_aA_pw4.0_ep100_el2.0_cs10_wd1e-4_wSparse_hnC0-1.pt

Outputs per seed:
  pred_results/immunomtl_shuffle_s1-10/seed{s}_{BenchmarkSet,mRNA,zero1,zero2}.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def mean_PPVn(y_true, y_pred_score, topk=None):
    idx        = np.argsort(y_pred_score)[::-1]
    y_sorted   = np.array(y_true)[idx]
    cum_tp     = np.cumsum(y_sorted)
    ppvn_curve = cum_tp / np.arange(1, len(y_sorted) + 1)
    num_pos    = int(y_sorted.sum())
    if topk is None or topk >= num_pos:
        return float(np.mean(ppvn_curve[:num_pos]))
    return float(np.mean(ppvn_curve[:num_pos][:topk]))

# ── Config (must match ImmunoMTL_shuffle_multiseed.py) ───────────────────────
ESM_ID      = "facebook/esm2_t12_35M_UR50D"
ESM_DIM     = 480
PEP_LEN     = 11
MHC_LEN     = 34
N_TASKS     = 4
SHUFFLE_SEEDS = list(range(1, 11))   # training seeds 1–10; cluster assignment fixed cs=10
DATA_DIR    = "../data"
HLA_DIR     = "../HLA"
MODEL_DIR   = "../models"
PRED_DIR    = "../pred_results/immunomtl_shuffle_s1-10"
os.makedirs(PRED_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ── MHC pseudo lookup ─────────────────────────────────────────────────────────
mhc_dict = {}
with open(f"{HLA_DIR}/MHC_pseudo.dat") as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2:
            mhc_dict[p[0]] = p[1]

def lookup_pseudo(mhc):
    return mhc_dict.get(mhc.replace("*", "").replace(":", ""), None)

# ── Load datasets (pseudo only; cluster assigned per-seed below) ──────────────
def prep(path):
    df = pd.read_csv(path)
    df["HLA_pseudo"] = df["MHC"].apply(lookup_pseudo)
    df = df.dropna(subset=["HLA_pseudo"]).reset_index(drop=True)
    return df

print("[INFO] Loading datasets ...")
df_bn   = prep(f"{DATA_DIR}/benchmark.csv")
df_mrna = prep(f"{DATA_DIR}/mRNAvaccine_pID.csv")
df_z1   = prep(f"{DATA_DIR}/zeroshot_data.csv")
df_z2   = prep(f"{DATA_DIR}/zeroshot_data2.csv")

DATASETS = [
    ("BenchmarkSet", df_bn),
    ("mRNA",         df_mrna),
    ("zero1",        df_z1),
    ("zero2",        df_z2),
]

# ── ESM2 embeddings (shared across seeds) ────────────────────────────────────
all_peps = sorted(set(p for _, d in DATASETS for p in d["Peptide"]))
all_mhcs = sorted(set(m for _, d in DATASETS for m in d["HLA_pseudo"]))
print(f"[INFO] Embedding {len(all_peps)} peptides + {len(all_mhcs)} MHCs ...")

tok = AutoTokenizer.from_pretrained(ESM_ID)
esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)

def embed_batch(seqs, max_len):
    out = []
    for i in tqdm(range(0, len(seqs), 64), leave=False):
        b = tok(seqs[i:i+64], return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_len, add_special_tokens=False).to(device)
        with torch.no_grad():
            out.append(esm(**b).last_hidden_state.cpu().numpy())
    return np.concatenate(out, axis=0)

PEP_EMB = dict(zip(all_peps, embed_batch(all_peps, PEP_LEN)))
MHC_EMB = dict(zip(all_mhcs, embed_batch(all_mhcs, MHC_LEN)))
del esm; torch.cuda.empty_cache()

# ── Model ─────────────────────────────────────────────────────────────────────
class ShuffleModel(nn.Module):
    def __init__(self, n_tasks=N_TASKS):
        super().__init__()
        self.pep1   = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.pep2   = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.mhc1   = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.mhc2   = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16),  nn.LeakyReLU(),     nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(n_tasks)])
    def forward(self, xp, xm):
        p, _ = self.pep1(xp); p, _ = self.pep2(p); p = p[:, -1, :]
        m, _ = self.mhc1(xm); m, _ = self.mhc2(m); m = m[:, -1, :]
        s = self.shared(torch.cat([p, m], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]

# ── Main loop over seeds ──────────────────────────────────────────────────────
cluster_ids = list(range(N_TASKS))

for shuf_seed in SHUFFLE_SEEDS:
    print(f"\n{'='*50}\n  Shuffle seed {shuf_seed}\n{'='*50}")

    # Fixed cluster assignment (cs=10); training seed is the only variation across seeds
    rand_map = pd.read_csv(f"{HLA_DIR}/random_mhc_cluster_assignment_seed10.csv"
                           ).set_index("HLA")["cluster"].to_dict()

    ckpt = f"{MODEL_DIR}/ImmunoShuffle_s{shuf_seed}_cs10.pt"
    model = ShuffleModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[INFO] Loaded: {ckpt}")

    print(f"  {'Dataset':<12}  {'N':>5}  {'AUROC':>7}  {'AP':>7}  {'PPVn':>7}")
    print("  " + "-" * 46)
    for ds_name, df_base in DATASETS:
        df = df_base.copy()
        df["cluster"] = df["MHC"].map(rand_map)
        df = df.dropna(subset=["cluster"]).reset_index(drop=True)
        df["cluster"] = df["cluster"].astype(int)

        scores = np.full(len(df), np.nan)
        for tid in cluster_ids:
            idx = df.index[df["cluster"] == tid]
            sub = df.loc[idx]
            if len(sub) == 0: continue
            pe = torch.tensor(np.stack([PEP_EMB[p] for p in sub["Peptide"]]),    dtype=torch.float32)
            me = torch.tensor(np.stack([MHC_EMB[m] for m in sub["HLA_pseudo"]]), dtype=torch.float32)
            out = []
            for i in range(0, len(pe), 512):
                with torch.no_grad():
                    o = model(pe[i:i+512].to(device), me[i:i+512].to(device))
                out.append(torch.sigmoid(o[tid]).cpu().numpy())
            scores[idx] = np.concatenate(out)

        df["Predicted Score"] = scores
        df = df.dropna(subset=["Predicted Score"])
        if len(df) == 0 or df["Label"].nunique() < 2:
            print(f"  {ds_name:<12}  SKIPPED (N={len(df)}, no matchable alleles or single class)")
            continue
        auroc = roc_auc_score(df["Label"], df["Predicted Score"])
        ap    = average_precision_score(df["Label"], df["Predicted Score"])
        ppvn  = mean_PPVn(df["Label"].values, df["Predicted Score"].values)
        path  = f"{PRED_DIR}/s{shuf_seed}_{ds_name}.csv"
        df[["Peptide", "MHC", "Label", "Predicted Score"]].to_csv(path, index=False)
        print(f"  {ds_name:<12}  {len(df):>5}  {auroc:.4f}  {ap:.4f}  {ppvn:.4f}  → {path}")

print("\nDone.")
