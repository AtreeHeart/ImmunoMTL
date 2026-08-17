#!/usr/bin/env python3
"""
eval_ABC.py — Inference-only evaluation for ABC ablation model (seed=22).

Architecture: ESM2-t12-35M (frozen) → dual BiLSTM (pep, MHC separate, 480→128)
              → shared FC(256→64→16) → 3 HLA-gene heads (A=0, B=1, C=2).
Inference: route each sample to its HLA-gene head (A→0, B→1, C→2).
Matches train_ABC.py exactly.

Outputs:
  pred_results/finetune/ABC_s22_aA_pw4.0_ep100_el2.0_wd1e-4_wSparse_hnC0-1/{benchmark,mRNA,zero1,zero2}_predictions.csv
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

# ── Config (must match train_ABC.py) ─────────────────────────────────────────
ESM_ID   = "facebook/esm2_t12_35M_UR50D"
ESM_DIM  = 480
PEP_LEN  = 11
MHC_LEN  = 34
N_TASKS  = 3   # HLA-A / HLA-B / HLA-C
CKPT     = "../models/ImmunoABC_s22.pt"
PRED_DIR = "../pred_results/ABC"
DATA_DIR = "../data"
HLA_DIR  = "../HLA"
os.makedirs(PRED_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ── Lookups ───────────────────────────────────────────────────────────────────
cluster_mapping    = pd.read_csv(f"{HLA_DIR}/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
t2_cluster_mapping = pd.read_csv(f"{HLA_DIR}/t2_cluster_res_c4.csv").set_index("HLA")["assigned_cluster"].to_dict()

mhc_dict = {}
with open(f"{HLA_DIR}/MHC_pseudo.dat") as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2:
            mhc_dict[p[0]] = p[1]

def lookup_pseudo(mhc):
    return mhc_dict.get(mhc.replace("*", "").replace(":", ""), None)

# ── Load datasets ─────────────────────────────────────────────────────────────
def prep(path):
    df = pd.read_csv(path)
    df["HLA_pseudo"] = df["MHC"].apply(lookup_pseudo)
    df["cluster"]    = df["MHC"].map(cluster_mapping).fillna(df["MHC"].map(t2_cluster_mapping))
    df = df.dropna(subset=["HLA_pseudo", "cluster"]).reset_index(drop=True)
    return df

print("[INFO] Loading datasets ...")
df_bn   = prep(f"{DATA_DIR}/benchmark.csv")
df_mrna = prep(f"{DATA_DIR}/mRNAvaccine_pID.csv")
df_z1   = prep(f"{DATA_DIR}/zeroshot_data.csv")
df_z2   = prep(f"{DATA_DIR}/zeroshot_data2.csv")

DATASETS = [("benchmark", df_bn), ("mRNA", df_mrna), ("zero1", df_z1), ("zero2", df_z2)]

# ── ESM2 embeddings ───────────────────────────────────────────────────────────
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

# ── Model (attribute names match the saved checkpoint) ────────────────────────
class ABCModel(nn.Module):
    def __init__(self, n_tasks=N_TASKS):
        super().__init__()
        self.pep1   = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.pep2   = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.mhc1   = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.mhc2   = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
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

print(f"[INFO] Loading checkpoint: {CKPT}")
model = ABCModel().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ── Inference: route each sample to its HLA-gene head ────────────────────────
def gene_id(mhc):
    if "HLA-A" in mhc: return 0
    if "HLA-B" in mhc: return 1
    if "HLA-C" in mhc: return 2
    return None

def predict(df):
    df = df.reset_index(drop=True)
    scores = np.full(len(df), np.nan)
    for gid in range(N_TASKS):
        sub_idx = [i for i, m in enumerate(df["MHC"]) if gene_id(m) == gid]
        if not sub_idx: continue
        sub = df.iloc[sub_idx]
        pe = torch.tensor(np.stack([PEP_EMB[p] for p in sub["Peptide"]]),    dtype=torch.float32)
        me = torch.tensor(np.stack([MHC_EMB[m] for m in sub["HLA_pseudo"]]), dtype=torch.float32)
        out = []
        for i in range(0, len(pe), 512):
            with torch.no_grad():
                heads = model(pe[i:i+512].to(device), me[i:i+512].to(device))
                out.append(torch.sigmoid(heads[gid]).cpu().numpy())
        scores[sub_idx] = np.concatenate(out)
    return scores

print("\nPredictions:")
print(f"{'Dataset':<12}  {'N':>5}  {'AUROC':>7}  {'AP':>7}")
print("-" * 38)
for ds_name, df in DATASETS:
    sc = predict(df)
    out = df[["Peptide", "MHC", "HLA_pseudo", "Label"]].copy()
    out["score"] = sc
    auroc = roc_auc_score(out["Label"], sc)
    ap    = average_precision_score(out["Label"], sc)
    path  = f"{PRED_DIR}/{ds_name}_predictions.csv"
    out.to_csv(path, index=False)
    print(f"  {ds_name:<12}  {len(out):>5}  {auroc:.4f}  {ap:.4f}  → {path}")

print("\nDone.")
