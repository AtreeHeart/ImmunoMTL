#!/usr/bin/env python3
"""
train_ImmunoMTL.py — Train and evaluate the ImmunoMTL model.

Architecture:
  ESM2-t12-35M (frozen) → dual BiLSTM (pep + MHC pseudo, h=64) × 2
  → shared FC(256→64→16) → 4 MMS-cluster task heads

Final configuration (seed=22):
  pw=4.0, epochs=100, EL_rank≤2.0, wd=1e-4, br=2,
  include_sparse=True, hn_clusters=[0,1], task_weights=[1.5,1.5,1.0,1.0]

All outputs are written to .tmp/ to avoid touching committed files.
"""

import copy, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import zip_longest
from scipy.optimize import minimize_scalar
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── Paths (absolute; no os.chdir so relative imports stay predictable) ────────
_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)          # ImmunoMTL_official/
DATA_DIR = os.path.join(_ROOT, "data")
HLA_DIR  = os.path.join(_ROOT, "HLA")
TMP      = os.path.join(_ROOT, ".tmp")
MDL_DIR  = os.path.join(TMP, "models")
PRD_DIR  = os.path.join(TMP, "pred_results", "MTL")
RES_DIR  = os.path.join(TMP, "results")
for d in (MDL_DIR, PRD_DIR, RES_DIR):
    os.makedirs(d, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
SEED          = 22
PW            = 4.0
EPOCHS        = 100
EL_RANK_THRESH= 2.0
WEIGHT_DECAY  = 1e-4
BALANCE_RATIO = 2
INCLUDE_SPARSE= True
HN_CLUSTERS   = [0, 1]
TASK_WEIGHTS  = [1.5, 1.5, 1.0, 1.0]
LR            = 1e-3
D1, D2        = 0.3, 0.2
BATCH_SIZE    = 64
CLIP_NORM     = 1.0
TEST_SIZE     = 0.15
ESM_ID        = "facebook/esm2_t12_35M_UR50D"
ESM_DIM       = 480
PEP_LEN       = 11
MHC_LEN       = 34
SPARSE_ALLELES= ["HLA-B*52:01", "HLA-C*03:02", "HLA-C*15:02"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f"[INFO] Device={device}  seed={SEED}  pw={PW}  epochs={EPOCHS}")

# ── HLA lookups ───────────────────────────────────────────────────────────────
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

def mms_cluster(mhc):
    c = cluster_map.get(mhc)
    if c is None:
        c = t2_map.get(mhc)
    return int(c) if c is not None else None

# ── Load data ─────────────────────────────────────────────────────────────────
print("[INFO] Loading data ...")
df_base = pd.read_csv(f"{DATA_DIR}/training_mhcf.csv").drop(columns=["cluster"], errors="ignore")
df_base["pseudo"]  = df_base["MHC"].apply(pseudo)
df_base["cluster"] = df_base["MHC"].map(cluster_map)
df_base = df_base.dropna(subset=["pseudo", "cluster"]).reset_index(drop=True)
df_base["cluster"] = df_base["cluster"].astype(int)

df_hn = pd.read_csv(f"{DATA_DIR}/HN_training.csv")
df_hn["pseudo"]  = df_hn["MHC"].apply(pseudo)
df_hn["cluster"] = df_hn["MHC"].map(cluster_map)
df_hn = (df_hn.dropna(subset=["pseudo", "cluster"])
               .pipe(lambda d: d[d["MHC"].isin(df_base["MHC"].unique())])
               .reset_index(drop=True))
df_hn["cluster"] = df_hn["cluster"].astype(int)

eval_raw = {
    "benchmark": pd.read_csv(f"{DATA_DIR}/benchmark.csv"),
    "mRNA":      pd.read_csv(f"{DATA_DIR}/mRNAvaccine_pID.csv"),
    "zero1":     pd.read_csv(f"{DATA_DIR}/zeroshot_data.csv"),
    "zero2":     pd.read_csv(f"{DATA_DIR}/zeroshot_data2.csv"),
}
for name, df in eval_raw.items():
    df["pseudo"]  = df["MHC"].apply(pseudo)
    df["cluster"] = df["MHC"].map(cluster_map).fillna(df["MHC"].map(t2_map))
    eval_raw[name] = df.dropna(subset=["pseudo"]).reset_index(drop=True)

# ── ESM2 embeddings ───────────────────────────────────────────────────────────
print("[INFO] Embedding sequences with ESM2 ...")
all_dfs  = [df_base, df_hn] + list(eval_raw.values())
all_peps = sorted(set(p for d in all_dfs for p in d["Peptide"]))
all_mhcs = sorted(set(m for d in all_dfs for m in d["pseudo"]))

tok = AutoTokenizer.from_pretrained(ESM_ID)
esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)

def embed(seqs, L):
    out = []
    for i in tqdm(range(0, len(seqs), 64), desc=f"  L={L}", leave=False):
        b = tok(seqs[i:i+64], return_tensors="pt", padding="max_length",
                truncation=True, max_length=L, add_special_tokens=False).to(device)
        with torch.no_grad():
            out.append(esm(**b).last_hidden_state.cpu().numpy())
    return np.concatenate(out)

PEP_EMB = dict(zip(all_peps, embed(all_peps, PEP_LEN)))
MHC_EMB = dict(zip(all_mhcs, embed(all_mhcs, MHC_LEN)))
del esm; torch.cuda.empty_cache()
print("  Done.\n")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data helpers ──────────────────────────────────────────────────────────────
def filter_base(df):
    df = df.copy() if INCLUDE_SPARSE else df[~df["MHC"].isin(SPARSE_ALLELES)].copy()
    m  = (df["Source"] == "IEDB") & (df["Label"] == 0)
    df = df[~(m & (df["EL_Rank"] > EL_RANK_THRESH))].copy()
    hn = df_hn[df_hn["cluster"].isin(HN_CLUSTERS)]
    if not INCLUDE_SPARSE:
        hn = hn[~hn["MHC"].isin(SPARSE_ALLELES)]
    # filter HN to MHCs present in the full base (matching original train_finetune.py)
    hn = hn[hn["MHC"].isin(df_base["MHC"].unique())].reset_index(drop=True)
    return pd.concat([df, hn], ignore_index=True)

def balanced_per_mhc(df, seed, ratio=2):
    parts = []
    for _, md in df.groupby("MHC"):
        pos = md[md["Label"] == 1]
        neg = md[md["Label"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        parts.append(pd.concat([
            pos.sample(len(pos), random_state=seed),
            neg.sample(min(len(neg), len(pos) * ratio), random_state=seed)]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

class DualDS(Dataset):
    def __init__(self, df):
        self.pep = torch.tensor(np.stack([PEP_EMB[p] for p in df["Peptide"]]), dtype=torch.float32)
        self.mhc = torch.tensor(np.stack([MHC_EMB[m] for m in df["pseudo"]]),  dtype=torch.float32)
        self.lbl = torch.tensor(df["Label"].values,                             dtype=torch.float32)
    def __len__(self):        return len(self.lbl)
    def __getitem__(self, i): return self.pep[i], self.mhc[i], self.lbl[i]

def make_task_loaders(df, seed):
    loaders, val_loaders, val_dfs = [], [], []
    for tid in sorted(df["task"].dropna().astype(int).unique()):
        td = balanced_per_mhc(df[df["task"] == tid], seed, ratio=BALANCE_RATIO)
        if len(td) < 10:
            continue
        td = td.copy()
        td["strata"] = td["MHC"].astype(str) + "_" + td["Label"].astype(str)
        sc   = td["strata"].value_counts()
        sing = td["strata"].map(sc) == 1
        sp   = td[~sing]
        if len(sp) > 0 and sp["strata"].value_counts().min() >= 2:
            tr, va = train_test_split(sp, test_size=TEST_SIZE,
                                      stratify=sp["strata"], random_state=seed)
        else:
            tr, va = train_test_split(td, test_size=TEST_SIZE, random_state=seed)
        tr = pd.concat([tr, td[sing]], ignore_index=True).drop(columns=["strata"])
        va = va.drop(columns=["strata"])
        val_dfs.append((va, tid))
        loaders.append((DataLoader(DualDS(tr), batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True), tid))
        val_loaders.append((DataLoader(DualDS(va), batch_size=BATCH_SIZE), tid))
    return loaders, val_loaders, val_dfs

# ── Model ─────────────────────────────────────────────────────────────────────
class ImmunoMTL(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        h, inp, mlp_in, mlp_h, mlp_out = 64, 128, 256, 64, 16
        self.pep1   = nn.LSTM(ESM_DIM, h,   batch_first=True, bidirectional=True)
        self.pep2   = nn.LSTM(inp,     h,   batch_first=True, bidirectional=True)
        self.mhc1   = nn.LSTM(ESM_DIM, h,   batch_first=True, bidirectional=True)
        self.mhc2   = nn.LSTM(inp,     h,   batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(mlp_in, mlp_h), nn.BatchNorm1d(mlp_h),
            nn.LeakyReLU(), nn.Dropout(D1),
            nn.Linear(mlp_h, mlp_out), nn.LeakyReLU(), nn.Dropout(D2))
        self.heads  = nn.ModuleList([nn.Linear(mlp_out, 1) for _ in range(n_heads)])

    def forward(self, xp, xm):
        p, _ = self.pep1(xp); p, _ = self.pep2(p)
        m, _ = self.mhc1(xm); m, _ = self.mhc2(m)
        s = self.shared(torch.cat([p[:, -1, :], m[:, -1, :]], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]

# ── Training ──────────────────────────────────────────────────────────────────
def train(model, loaders, val_loaders):
    model.to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([PW], device=device))
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val, best_state = 0.0, copy.deepcopy(model.state_dict())

    for ep in range(EPOCHS):
        model.train()
        for batches in zip_longest(*[ldr for ldr, _ in loaders]):
            for batch, (_, tid) in zip(batches, loaders):
                if batch is None:
                    continue
                xp, xm, y = [x.to(device) for x in batch]
                opt.zero_grad()
                (TASK_WEIGHTS[tid] * crit(model(xp, xm)[tid], y)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
        sch.step()

        if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for vl, tid in val_loaders:
                    for xp, xm, y in vl:
                        out = model(xp.to(device), xm.to(device))
                        preds.extend(torch.sigmoid(out[tid]).cpu().numpy())
                        labels.extend(y.numpy())
            labels = np.array(labels)
            v = roc_auc_score(labels, preds) if 0 < labels.sum() < len(labels) else 0.0
            if v > best_val:
                best_val = v
                best_state = copy.deepcopy(model.state_dict())
            print(f"    ep {ep+1:3d}/{EPOCHS}  val_auc={v:.4f}  best={best_val:.4f}", flush=True)

    model.load_state_dict(best_state)
    return model, best_val

# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_routed(model, df, return_logits=False):
    model.eval()
    pe = torch.tensor(np.stack([PEP_EMB[p] for p in df["Peptide"]]), dtype=torch.float32)
    me = torch.tensor(np.stack([MHC_EMB[m] for m in df["pseudo"]]),  dtype=torch.float32)
    n  = len(model.heads)
    hs = [[] for _ in range(n)]
    for i in range(0, len(pe), 512):
        out = model(pe[i:i+512].to(device), me[i:i+512].to(device))
        for ci in range(n):
            hs[ci].append(out[ci].cpu().numpy())
    hs = [np.concatenate(h) for h in hs]
    logits = np.zeros(len(df))
    for idx, mhc in enumerate(df["MHC"]):
        c = mms_cluster(mhc)
        logits[idx] = hs[c][idx] if (c is not None and 0 <= c < n) \
                      else np.mean([hs[ci][idx] for ci in range(n)])
    return logits if return_logits else torch.sigmoid(torch.tensor(logits)).numpy()

# ── Calibration ───────────────────────────────────────────────────────────────
def ece(yt, yp, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    val  = 0.0
    for i in range(n_bins):
        m = (yp >= bins[i]) & (yp < bins[i + 1])
        if m.sum() > 0:
            val += m.sum() * abs(yt[m].mean() - yp[m].mean())
    return val / len(yt)

def fit_temperature(yt, logits):
    def nll(T):
        p = torch.sigmoid(torch.tensor(logits / T)).numpy()
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
    return minimize_scalar(nll, bounds=(0.01, 100.0), method="bounded").x

# ── Run ───────────────────────────────────────────────────────────────────────
print("[INFO] Preparing training data ...")
df_filtered = filter_base(df_base)
df_filtered["task"] = df_filtered["MHC"].map(cluster_map)
df_filtered = df_filtered.dropna(subset=["task"])
df_filtered["task"] = df_filtered["task"].astype(int)

loaders, val_loaders, val_dfs = make_task_loaders(df_filtered, SEED)

print("[INFO] Training ImmunoMTL ...")
model, best_val_auc = train(ImmunoMTL(N_CLUSTERS), loaders, val_loaders)
torch.save(model.state_dict(), f"{MDL_DIR}/ImmunoMTL_s{SEED}.pt")
print(f"  Model saved → {MDL_DIR}/ImmunoMTL_s{SEED}.pt")

# ── Val predictions for temperature scaling ───────────────────────────────────
print("\n[INFO] Collecting val predictions ...")
val_rows = []
for va_df, tid in val_dfs:
    va_df = va_df.reset_index(drop=True)
    logits = predict_routed(model, va_df, return_logits=True)
    for i, row in va_df.iterrows():
        val_rows.append({
            "Peptide": row["Peptide"],
            "MHC":     row["MHC"],
            "Label":   int(row["Label"]),
            "logit":   float(logits[i]),
            "score":   float(torch.sigmoid(torch.tensor(logits[i])).item()),
            "cluster": tid,
        })
df_val = pd.DataFrame(val_rows)
df_val.to_csv(f"{PRD_DIR}/val_predictions.csv", index=False)
print(f"  Val: {len(df_val)} rows  pos={int(df_val['Label'].sum())}")

# ── Test set predictions ──────────────────────────────────────────────────────
print("\n[INFO] Evaluating on test sets ...")
test_results = {}
for ds_name, df_ds in eval_raw.items():
    scores  = predict_routed(model, df_ds)
    logits  = predict_routed(model, df_ds, return_logits=True)
    y       = df_ds["Label"].values.astype(float)
    auroc   = roc_auc_score(y, scores)
    ap      = average_precision_score(y, scores)
    test_results[ds_name] = {"auroc": auroc, "ap": ap, "y": y,
                              "scores": scores, "logits": logits}
    out = df_ds[["Peptide", "MHC"]].copy()
    out["HLA_pseudo"] = df_ds["pseudo"]
    out["Label"]      = df_ds["Label"]
    out["cluster"]    = df_ds["cluster"].astype("Int64")
    out["score"]      = scores
    out.to_csv(f"{PRD_DIR}/{ds_name}_predictions.csv", index=False)
    print(f"  {ds_name:<12}  AUROC={auroc:.4f}  AP={ap:.4f}")

# ── Temperature scaling ───────────────────────────────────────────────────────
print("\n[INFO] Fitting temperature on val set ...")
T_val = fit_temperature(df_val["Label"].values.astype(float), df_val["logit"].values)
print(f"  T (val-fitted) = {T_val:.4f}")

bm = test_results["benchmark"]
yt_bm, logits_bm, scores_bm = bm["y"], bm["logits"], bm["scores"]
ece_raw          = ece(yt_bm, scores_bm)
ece_val_scaled   = ece(yt_bm, torch.sigmoid(torch.tensor(logits_bm / T_val)).numpy())
T_test           = fit_temperature(yt_bm, logits_bm)
ece_test_scaled  = ece(yt_bm, torch.sigmoid(torch.tensor(logits_bm / T_test)).numpy())

# ── Summary report ────────────────────────────────────────────────────────────
lines = [
    f"ImmunoMTL s{SEED} — Training summary",
    "=" * 50,
    f"Best val AUROC (training): {best_val_auc:.4f}",
    f"Val set: {len(df_val)} rows  pos={int(df_val['Label'].sum())}",
    f"Temperature (val-fitted): T = {T_val:.4f}",
    "",
    "Test set performance:",
]
for ds_name, r in test_results.items():
    lines.append(f"  {ds_name:<12}  AUROC={r['auroc']:.4f}  AP={r['ap']:.4f}")
lines += [
    "",
    f"Calibration (benchmark, N={len(yt_bm)}, {yt_bm.mean()*100:.1f}% pos):",
    f"  Raw ECE                        = {ece_raw:.4f}",
    f"  Temp-scaled ECE (val-fitted)   = {ece_val_scaled:.4f}  [T={T_val:.4f}]",
    f"  Temp-scaled ECE (test in-sample) = {ece_test_scaled:.4f}  [T={T_test:.4f}, reference only]",
]
report = "\n".join(lines)
print("\n" + report)
with open(f"{RES_DIR}/ImmunoMTL_s{SEED}_metrics.txt", "w") as fh:
    fh.write(report + "\n")
print(f"\nReport → {RES_DIR}/ImmunoMTL_s{SEED}_metrics.txt")
print("Done.")
