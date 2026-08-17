#!/usr/bin/env python3
"""
eval_JointSTL.py — Inference-only evaluation for JointSTL baseline (seed=22).

Architecture: ESM2-t12-35M (frozen) → concatenated [pep_seq ∥ hla_pseudo_seq] (len=45)
              → single BiLSTM × 2 (480→128) → FC(128→64→16→1) — single prediction head.
Matches train_JointSTL.py exactly.

Outputs:
  revision/pred_results/finetune/Simple_s22_aA_pw4.0_ep100_el2.0_wd1e-4_wSparse_hnC0-1/{benchmark,mRNA,zero1,zero2}_predictions.csv
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

# ── Config (must match train_JointSTL.py) ────────────────────────────────────
ESM_ID   = "facebook/esm2_t12_35M_UR50D"
ESM_DIM  = 480
PEP_LEN  = 11
MHC_LEN  = 34
JOINT_LEN = PEP_LEN + MHC_LEN   # 45
CKPT     = "../models/JointSTL_s22.pt"
PRED_DIR  = "../pred_results/JointSTL"
DATA_DIR  = "../data"
HLA_DIR   = "../HLA"
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
DATASETS = [("benchmark", df_bn)]

# ── ESM2 joint embeddings ─────────────────────────────────────────────────────
# Each (peptide, HLA_pseudo) pair is embedded as a single concatenated sequence.
all_pairs = sorted(set(
    (row["Peptide"], row["HLA_pseudo"])
    for _, d in DATASETS
    for _, row in d.iterrows()
))
all_seqs  = [pep + hla for pep, hla in all_pairs]
print(f"[INFO] Embedding {len(all_pairs)} joint pep+HLA sequences (len={JOINT_LEN}) ...")

tok = AutoTokenizer.from_pretrained(ESM_ID)
esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)

embs = []
for i in tqdm(range(0, len(all_seqs), 64), leave=False):
    b = tok(all_seqs[i:i+64], return_tensors="pt", padding="max_length",
            truncation=True, max_length=JOINT_LEN, add_special_tokens=False).to(device)
    with torch.no_grad():
        embs.append(esm(**b).last_hidden_state.cpu().numpy())
emb_array = np.concatenate(embs, axis=0)

JOINT_EMB = dict(zip(all_pairs, emb_array))
del esm; torch.cuda.empty_cache()

# ── Model ─────────────────────────────────────────────────────────────────────
class JointSTLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.LeakyReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(16, 1)
    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        return self.head(self.shared(h[:, -1, :])).squeeze(-1)

print(f"[INFO] Loading checkpoint: {CKPT}")
model = JointSTLModel().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(df):
    xj = torch.tensor(
        np.stack([JOINT_EMB[(row["Peptide"], row["HLA_pseudo"])] for _, row in df.iterrows()]),
        dtype=torch.float32)
    out = []
    for i in range(0, len(xj), 512):
        with torch.no_grad():
            out.append(torch.sigmoid(model(xj[i:i+512].to(device))).cpu().numpy())
    return np.concatenate(out)

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
