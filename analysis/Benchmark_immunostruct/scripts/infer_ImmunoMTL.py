"""
3_infer_ImmunoMTL.py

Run ImmunoMTL inference on the IEDB and CEDAR novel test sets,
then merge with IS Combined predictions to produce ../data/test_predictions.csv.

Model: ImmunoMTL_s22.pt (ESM2-t12-35M, 4 MMS cluster heads)
IS predictions expected in: ../data/ (from 2_train_IS_combined.sh output)
"""

import os, sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MTL_ROOT   = os.path.join(SCRIPT_DIR, "..", "..", "..")
HLA_DIR    = os.path.join(MTL_ROOT, "HLA")
MODEL_PATH = os.path.join(MTL_ROOT, "models", "ImmunoMTL_s22.pt")
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")

# ── Cluster mappings ───────────────────────────────────────────────────────────
cluster_map = pd.read_csv(f"{HLA_DIR}/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
t2_map      = pd.read_csv(f"{HLA_DIR}/t2_cluster_res_c4.csv").set_index("HLA")["assigned_cluster"].to_dict()
N_CLUSTERS  = len(set(cluster_map.values()))

mhc_pseudo = {}
with open(f"{HLA_DIR}/MHC_pseudo.dat") as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2:
            mhc_pseudo[p[0]] = p[1]


def pseudo(mhc):
    return mhc_pseudo.get(mhc.replace("*", "").replace(":", ""), None)


def get_cluster(mhc):
    c = cluster_map.get(mhc)
    if c is None:
        c = t2_map.get(mhc)
    return c


# ── ESM2-t12-35M ──────────────────────────────────────────────────────────────
ESM_ID  = "facebook/esm2_t12_35M_UR50D"
ESM_DIM = 480
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading ESM2-t12-35M...")
tok = AutoTokenizer.from_pretrained(ESM_ID)
esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)


def embed(seqs, max_len):
    out = []
    for i in tqdm(range(0, len(seqs), 64), desc=f"  ESM2 maxlen={max_len}"):
        batch = tok(seqs[i:i+64], return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_len,
                    add_special_tokens=False).to(device)
        with torch.no_grad():
            out.append(esm(**batch).last_hidden_state.cpu())
    return torch.cat(out, dim=0)


# ── ImmunoMTL (s22) ───────────────────────────────────────────────────────────
class MultiHeadModel(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        h, inp = 64, 128
        self.pep1   = nn.LSTM(ESM_DIM, h, batch_first=True, bidirectional=True)
        self.pep2   = nn.LSTM(inp, h, batch_first=True, bidirectional=True)
        self.mhc1   = nn.LSTM(ESM_DIM, h, batch_first=True, bidirectional=True)
        self.mhc2   = nn.LSTM(inp, h, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16),  nn.LeakyReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(n_heads)])

    def forward(self, xp, xm):
        p, _ = self.pep1(xp);  p, _ = self.pep2(p);  pf = p[:, -1, :]
        m, _ = self.mhc1(xm);  m, _ = self.mhc2(m);  mf = m[:, -1, :]
        s = self.shared(torch.cat([pf, mf], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]


print("[INFO] Loading ImmunoMTL_s22...")
mtl = MultiHeadModel(N_CLUSTERS).to(device)
mtl.load_state_dict(torch.load(MODEL_PATH, map_location=device))
mtl.eval()
cluster_ids = sorted(set(cluster_map.values()))


def run_mtl(df, pep_col, mhc_col):
    df = df.copy()
    df["_pseudo"]  = df[mhc_col].apply(pseudo)
    df["_cluster"] = df[mhc_col].apply(get_cluster)
    df = df.dropna(subset=["_pseudo", "_cluster"]).reset_index(drop=True)

    pep_emb = embed(df[pep_col].tolist(), max_len=11)
    mhc_emb = embed(df["_pseudo"].tolist(), max_len=34)

    scores = np.zeros(len(df))
    for c in sorted(df["_cluster"].unique()):
        idx  = df.index[df["_cluster"] == c].tolist()
        xp   = pep_emb[idx].float().to(device)
        xm   = mhc_emb[idx].float().to(device)
        tid  = cluster_ids.index(c)
        with torch.no_grad():
            scores[idx] = torch.sigmoid(mtl(xp, xm)[tid]).cpu().numpy()

    df["ImmunoMTL_pred"] = scores
    return df


# ── IEDB novel ─────────────────────────────────────────────────────────────────
print("\n=== IEDB novel ===")
iedb = pd.read_csv(f"{DATA_DIR}/iedb_novel.csv")
iedb_mtl = run_mtl(iedb, "peptide", "allele")

# ── CEDAR novel ────────────────────────────────────────────────────────────────
print("\n=== CEDAR novel ===")
cedar = pd.read_csv(f"{DATA_DIR}/cedar_novel.csv")
cedar_mtl = run_mtl(cedar, "mut_pep", "allele")

# ── Merge IS Combined predictions ─────────────────────────────────────────────
IS_PRED_IEDB  = f"{DATA_DIR}/predictions_iedb_novel.csv"
IS_PRED_CEDAR = f"{DATA_DIR}/predictions_cedar_novel.csv"

if not os.path.exists(IS_PRED_IEDB) or not os.path.exists(IS_PRED_CEDAR):
    print("[WARN] IS predictions not found. Run 2_train_IS_combined.sh first.")
    print(f"       Expected: {IS_PRED_IEDB}")
    print(f"       Expected: {IS_PRED_CEDAR}")
    sys.exit(1)

is_iedb  = pd.read_csv(IS_PRED_IEDB)[["allele", "peptide", "IS_pred"]]
is_cedar = pd.read_csv(IS_PRED_CEDAR)[["allele", "mut_pep", "IS_pred"]].rename(columns={"mut_pep": "peptide"})

iedb_mtl  = iedb_mtl.rename(columns={"mut_pep": "peptide"}) if "mut_pep" in iedb_mtl.columns else iedb_mtl
cedar_mtl = cedar_mtl.rename(columns={"mut_pep": "peptide"})

iedb_out  = iedb_mtl[["allele","peptide","immunogenicity","ImmunoMTL_pred"]].merge(
    is_iedb, on=["allele","peptide"], how="inner")
iedb_out  = iedb_out.rename(columns={"immunogenicity":"label","IS_pred":"IS_combined_pred"})
iedb_out.insert(0, "source", "IEDB")

cedar_out = cedar_mtl[["allele","peptide","immunogenicity","ImmunoMTL_pred"]].merge(
    is_cedar, on=["allele","peptide"], how="inner")
cedar_out = cedar_out.rename(columns={"immunogenicity":"label","IS_pred":"IS_combined_pred"})
cedar_out.insert(0, "source", "CEDAR")

test_pred = pd.concat([iedb_out, cedar_out], ignore_index=True)
test_pred = test_pred[["source","allele","peptide","label","IS_combined_pred","ImmunoMTL_pred"]]
test_pred.to_csv(f"{DATA_DIR}/test_predictions.csv", index=False)
print(f"\n[SAVED] test_predictions.csv  ({len(test_pred)} rows, pos={int(test_pred['label'].sum())})")
print("Run analysis.ipynb to visualize results.")
