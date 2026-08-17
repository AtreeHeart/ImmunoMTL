#!/usr/bin/env python3
"""
train_ablations.py — Train ABC / STL / Shuffle(×10) / JointSTL ablations.
All outputs to ImmunoMTL_official/.tmp/
Seed placed after ESM2 embedding to match train_finetune.py ordering.
"""
import os, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import zip_longest
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.optimize import minimize_scalar
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)
DATA_DIR = os.path.join(_ROOT, "data")
HLA_DIR  = os.path.join(_ROOT, "HLA")
TMP      = os.path.join(_ROOT, ".tmp")
MDL_DIR  = os.path.join(TMP, "models")
PRD_DIR  = os.path.join(TMP, "pred_results")
RES_DIR  = os.path.join(TMP, "results")
for d in [MDL_DIR, PRD_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters (match train_finetune.py s22 aA pw4.0 ep100) ─────────────
SEED           = 22
EPOCHS         = 100
PW             = 4.0
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
D1, D2         = 0.3, 0.2
BATCH_SIZE     = 64
CLIP_NORM      = 1.0
TEST_SIZE      = 0.15
BALANCE_RATIO  = 2
ESM_ID         = "facebook/esm2_t12_35M_UR50D"
ESM_DIM        = 480
PEP_LEN        = 11
MHC_LEN        = 34
JOINT_LEN      = PEP_LEN + MHC_LEN   # 45
EL_RANK_THRESH = 2.0
HN_CLUSTERS    = [0, 1]
SPARSE_ALLELES = ["HLA-B*52:01", "HLA-C*03:02", "HLA-C*15:02"]
SHUFFLE_CS     = 10
SHUFFLE_SEEDS  = list(range(1, 11))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f"[INFO] Device={device}")

# ── HLA lookups ───────────────────────────────────────────────────────────────
cluster_map = pd.read_csv(f"{HLA_DIR}/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
t2_map      = pd.read_csv(f"{HLA_DIR}/t2_cluster_res_c4.csv").set_index("HLA")["assigned_cluster"].to_dict()
N_CLUSTERS  = len(set(cluster_map.values()))
shuffle_map = (pd.read_csv(f"{HLA_DIR}/random_mhc_cluster_assignment_seed{SHUFFLE_CS}.csv")
                 .set_index("HLA")["cluster"].to_dict())

mhc_pseudo_dict = {}
with open(f"{HLA_DIR}/MHC_pseudo.dat") as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2:
            mhc_pseudo_dict[p[0]] = p[1]

def pseudo(mhc):
    return mhc_pseudo_dict.get(mhc.replace("*", "").replace(":", ""), None)
def mms_cluster(mhc):
    c = cluster_map.get(mhc)
    if c is None: c = t2_map.get(mhc)
    return int(c) if c is not None else None
def gene_task(mhc):
    for i, g in enumerate(["A", "B", "C"]):
        if f"HLA-{g}" in mhc: return i
    return None
def shuffle_route(mhc):
    c = shuffle_map.get(mhc)
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

# ── ESM2 embeddings (computed once, shared across all models) ─────────────────
print("[INFO] Embedding with ESM2 ...")
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

# Joint pep+pseudo embeddings for JointSTL
all_pairs      = sorted(set((r["Peptide"], r["pseudo"])
                             for d in all_dfs for _, r in d.iterrows()
                             if pd.notna(r.get("pseudo"))))
all_joint_seqs = [p + m for p, m in all_pairs]
print(f"  Joint pairs: {len(all_pairs)}")
JOINT_EMB = dict(zip(all_pairs, embed(all_joint_seqs, JOINT_LEN)))

del esm; torch.cuda.empty_cache()
print("  Done.\n")

# Set seeds after embedding (matching train_finetune.py line 665 placement)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Shared utilities ──────────────────────────────────────────────────────────
class DualDS(Dataset):
    def __init__(self, df):
        self.pep = torch.tensor(np.stack([PEP_EMB[p] for p in df["Peptide"]]), dtype=torch.float32)
        self.mhc = torch.tensor(np.stack([MHC_EMB[m] for m in df["pseudo"]]),  dtype=torch.float32)
        self.lbl = torch.tensor(df["Label"].values,                             dtype=torch.float32)
    def __len__(self):        return len(self.lbl)
    def __getitem__(self, i): return self.pep[i], self.mhc[i], self.lbl[i]

def filter_base(include_sparse=True):
    df = df_base.copy()
    if not include_sparse:
        df = df[~df["MHC"].isin(SPARSE_ALLELES)].copy()
    m  = (df["Source"] == "IEDB") & (df["Label"] == 0)
    df = df[~(m & (df["EL_Rank"] > EL_RANK_THRESH))].copy()
    hn = df_hn[df_hn["cluster"].isin(HN_CLUSTERS)].copy()
    if not include_sparse:
        hn = hn[~hn["MHC"].isin(SPARSE_ALLELES)]
    hn = hn[hn["MHC"].isin(df_base["MHC"].unique())].reset_index(drop=True)
    return pd.concat([df, hn], ignore_index=True)

def balanced_per_mhc(df, seed):
    parts = []
    for _, md in df.groupby("MHC"):
        pos = md[md["Label"] == 1]; neg = md[md["Label"] == 0]
        if len(pos) == 0 or len(neg) == 0: continue
        parts.append(pd.concat([
            pos.sample(len(pos), random_state=seed),
            neg.sample(min(len(neg), len(pos) * BALANCE_RATIO), random_state=seed)]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def make_task_loaders(df, task_col, seed):
    loaders, val_loaders, val_dfs = [], [], []
    for tid in sorted(df[task_col].dropna().astype(int).unique()):
        td = balanced_per_mhc(df[df[task_col] == tid], seed)
        if len(td) < 10: continue
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

class DualBiLSTM(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        h, inp = 64, 128
        self.pep1   = nn.LSTM(ESM_DIM, h,   batch_first=True, bidirectional=True)
        self.pep2   = nn.LSTM(inp,     h,   batch_first=True, bidirectional=True)
        self.mhc1   = nn.LSTM(ESM_DIM, h,   batch_first=True, bidirectional=True)
        self.mhc2   = nn.LSTM(inp,     h,   batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(), nn.Dropout(D1),
            nn.Linear(64, 16), nn.LeakyReLU(), nn.Dropout(D2))
        self.heads  = nn.ModuleList([nn.Linear(16, 1) for _ in range(n_heads)])
    def forward(self, xp, xm):
        p, _ = self.pep1(xp); p, _ = self.pep2(p)
        m, _ = self.mhc1(xm); m, _ = self.mhc2(m)
        s = self.shared(torch.cat([p[:, -1, :], m[:, -1, :]], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]

def train_loop(model, loaders, val_loaders, task_weights=None):
    model.to(device)
    tw   = task_weights or [1.0] * len(model.heads)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([PW], device=device))
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val, best_state = 0.0, copy.deepcopy(model.state_dict())
    for ep in range(EPOCHS):
        model.train()
        for batches in zip_longest(*[ldr for ldr, _ in loaders]):
            for batch, (_, tid) in zip(batches, loaders):
                if batch is None: continue
                xp, xm, y = [x.to(device) for x in batch]
                opt.zero_grad()
                (tw[tid] * crit(model(xp, xm)[tid], y)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
        sch.step()
        if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
            model.eval(); preds, labels = [], []
            with torch.no_grad():
                for vl, tid in val_loaders:
                    for xp, xm, y in vl:
                        out = model(xp.to(device), xm.to(device))
                        preds.extend(torch.sigmoid(out[tid]).cpu().numpy())
                        labels.extend(y.numpy())
            labels = np.array(labels)
            v = roc_auc_score(labels, preds) if 0 < labels.sum() < len(labels) else 0.0
            if v > best_val:
                best_val = v; best_state = copy.deepcopy(model.state_dict())
            print(f"    ep {ep+1:3d}/{EPOCHS}  val_auc={v:.4f}  best={best_val:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, best_val

@torch.no_grad()
def predict_dual(model, df, route_fn, return_logits=False):
    model.eval()
    pe = torch.tensor(np.stack([PEP_EMB[p] for p in df["Peptide"]]), dtype=torch.float32)
    me = torch.tensor(np.stack([MHC_EMB[m] for m in df["pseudo"]]),  dtype=torch.float32)
    n  = len(model.heads)
    hs = [[] for _ in range(n)]
    for i in range(0, len(pe), 512):
        out = model(pe[i:i+512].to(device), me[i:i+512].to(device))
        for ci in range(n):
            hs[ci].append((out[ci] if return_logits else torch.sigmoid(out[ci])).cpu().numpy())
    hs = [np.concatenate(h) for h in hs]
    vals = np.zeros(len(df))
    for idx, mhc in enumerate(df["MHC"]):
        c = route_fn(mhc)
        vals[idx] = hs[c][idx] if (c is not None and 0 <= c < n) \
                    else np.mean([hs[ci][idx] for ci in range(n)])
    return vals

def fit_temperature(labels, logits, T_min=0.1, T_max=100.0):
    def nll(T):
        p = torch.sigmoid(torch.tensor(logits / T, dtype=torch.float32)).clamp(1e-7, 1 - 1e-7)
        l = torch.tensor(labels, dtype=torch.float32)
        return (-l * torch.log(p) - (1 - l) * torch.log(1 - p)).mean().item()
    return float(minimize_scalar(nll, bounds=(T_min, T_max), method="bounded").x)

def ece(labels, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total, e = len(labels), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0: continue
        e += m.sum() / total * abs(labels[m].mean() - probs[m].mean())
    return e

def run_calibration(model_name, val_labels, val_logits, bm_y, bm_logits, bm_scores):
    T = fit_temperature(val_labels, val_logits)
    scaled     = torch.sigmoid(torch.tensor(bm_logits / T)).numpy()
    T_test     = fit_temperature(bm_y, bm_logits)
    scaled_tst = torch.sigmoid(torch.tensor(bm_logits / T_test)).numpy()
    e_raw, e_val, e_tst = ece(bm_y, bm_scores), ece(bm_y, scaled), ece(bm_y, scaled_tst)
    print(f"\n[{model_name}] Calibration (benchmark):")
    print(f"  Val-fitted T={T:.4f}  Raw ECE={e_raw:.4f}  "
          f"Val-scaled ECE={e_val:.4f}  In-sample ECE={e_tst:.4f} (ref)")
    return dict(T=T, ece_raw=e_raw, ece_val=e_val, ece_test=e_tst)

def eval_and_save(model, route_fn, model_name, subdir):
    prd_path = os.path.join(PRD_DIR, subdir)
    os.makedirs(prd_path, exist_ok=True)
    res = {}
    print(f"\n[{model_name}] Evaluating test sets:")
    for ds_name, df_ds in eval_raw.items():
        scores = predict_dual(model, df_ds, route_fn)
        logits = predict_dual(model, df_ds, route_fn, return_logits=True)
        y = df_ds["Label"].values.astype(float)
        auroc = roc_auc_score(y, scores)
        ap    = average_precision_score(y, scores)
        res[ds_name] = dict(auroc=auroc, ap=ap, y=y, scores=scores, logits=logits)
        out = df_ds[["Peptide", "MHC"]].copy()
        out["HLA_pseudo"] = df_ds["pseudo"]
        out["Label"]      = df_ds["Label"]
        out["score"]      = scores
        out.to_csv(f"{prd_path}/{ds_name}_predictions.csv", index=False)
        print(f"  {ds_name:<12}  AUROC={auroc:.4f}  AP={ap:.4f}")
    return res

# ═══════════════════════════════════════════════════════════════════════════════
# ABC  (3 heads, gene-based routing A/B/C, uniform task weights)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ABC  (gene routing, 3 heads, seed=22)")
print("=" * 60)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

df_f = filter_base(include_sparse=True)
df_f["task"] = df_f["MHC"].apply(gene_task)
df_f = df_f.dropna(subset=["task"]); df_f["task"] = df_f["task"].astype(int)
loaders_abc, val_loaders_abc, val_dfs_abc = make_task_loaders(df_f, "task", SEED)

abc_model, abc_best_val = train_loop(DualBiLSTM(3), loaders_abc, val_loaders_abc)
torch.save(abc_model.state_dict(), f"{MDL_DIR}/ABC_s22.pt")
print(f"  Saved → {MDL_DIR}/ABC_s22.pt  (best val AUROC={abc_best_val:.4f})")

abc_res = eval_and_save(abc_model, gene_task, "ABC", "ABC")

val_lab_abc  = np.concatenate([va["Label"].values for va, _ in val_dfs_abc])
val_log_abc  = np.concatenate([predict_dual(abc_model, va.reset_index(drop=True),
                                            gene_task, return_logits=True)
                               for va, _ in val_dfs_abc])
abc_cal = run_calibration("ABC", val_lab_abc, val_log_abc,
                          abc_res["benchmark"]["y"],
                          abc_res["benchmark"]["logits"],
                          abc_res["benchmark"]["scores"])
del abc_model; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# STL  (1 head, no routing, global balanced split)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STL  (single head, seed=22)")
print("=" * 60)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

df_f = filter_base(include_sparse=True)
td = balanced_per_mhc(df_f, SEED)
td = td.copy()
td["strata"] = td["MHC"].astype(str) + "_" + td["Label"].astype(str)
sc   = td["strata"].value_counts()
sing = td["strata"].map(sc) == 1
sp   = td[~sing]
if len(sp) > 0 and sp["strata"].value_counts().min() >= 2:
    tr_stl, va_stl = train_test_split(sp, test_size=TEST_SIZE,
                                      stratify=sp["strata"], random_state=SEED)
else:
    tr_stl, va_stl = train_test_split(td, test_size=TEST_SIZE, random_state=SEED)
tr_stl = pd.concat([tr_stl, td[sing]], ignore_index=True).drop(columns=["strata"])
va_stl = va_stl.drop(columns=["strata"])

loader_stl     = DataLoader(DualDS(tr_stl), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader_stl = DataLoader(DualDS(va_stl), batch_size=BATCH_SIZE)

stl_model, stl_best_val = train_loop(DualBiLSTM(1),
                                     [(loader_stl, 0)], [(val_loader_stl, 0)])
torch.save(stl_model.state_dict(), f"{MDL_DIR}/STL_s22.pt")
print(f"  Saved → {MDL_DIR}/STL_s22.pt  (best val AUROC={stl_best_val:.4f})")

stl_res = eval_and_save(stl_model, lambda _: 0, "STL", "STL")

val_log_stl = predict_dual(stl_model, va_stl.reset_index(drop=True),
                           lambda _: 0, return_logits=True)
stl_cal = run_calibration("STL", va_stl["Label"].values.astype(float), val_log_stl,
                          stl_res["benchmark"]["y"],
                          stl_res["benchmark"]["logits"],
                          stl_res["benchmark"]["scores"])
del stl_model; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# Shuffle  (4 heads, random routing cs=10, training seeds 1–10)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SHUFFLE  (random routing cs=10, seeds 1-10)")
print("=" * 60)

df_f_shuf = filter_base(include_sparse=True)  # same base for all shuffle seeds
shuf_summary = {}

for s in SHUFFLE_SEEDS:
    print(f"\n--- Shuffle seed={s} ---")
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

    df_s = df_f_shuf.copy()
    df_s["task"] = df_s["MHC"].map(shuffle_map)
    df_s = df_s.dropna(subset=["task"]); df_s["task"] = df_s["task"].astype(int)
    loaders_s, val_loaders_s, val_dfs_s = make_task_loaders(df_s, "task", s)

    shuf_model, shuf_bv = train_loop(DualBiLSTM(N_CLUSTERS), loaders_s, val_loaders_s)
    pt_path = f"{MDL_DIR}/Shuffle_s{s}_cs{SHUFFLE_CS}.pt"
    torch.save(shuf_model.state_dict(), pt_path)

    prd_path = os.path.join(PRD_DIR, "immunomtl_shuffle_s1-10")
    os.makedirs(prd_path, exist_ok=True)
    s_res = {}
    print(f"  [s={s}] Test sets:")
    for ds_name, df_ds in eval_raw.items():
        scores = predict_dual(shuf_model, df_ds, shuffle_route)
        logits = predict_dual(shuf_model, df_ds, shuffle_route, return_logits=True)
        y = df_ds["Label"].values.astype(float)
        auroc = roc_auc_score(y, scores); ap = average_precision_score(y, scores)
        s_res[ds_name] = dict(auroc=auroc, ap=ap, y=y, scores=scores, logits=logits)
        out = df_ds[["Peptide", "MHC"]].copy()
        out["HLA_pseudo"] = df_ds["pseudo"]; out["Label"] = df_ds["Label"]; out["score"] = scores
        out.to_csv(f"{prd_path}/s{s}_{ds_name}_predictions.csv", index=False)
        print(f"    {ds_name:<12}  AUROC={auroc:.4f}  AP={ap:.4f}")

    val_lab_s = np.concatenate([va["Label"].values for va, _ in val_dfs_s])
    val_log_s = np.concatenate([predict_dual(shuf_model, va.reset_index(drop=True),
                                             shuffle_route, return_logits=True)
                                for va, _ in val_dfs_s])
    s_cal = run_calibration(f"Shuffle_s{s}", val_lab_s, val_log_s,
                            s_res["benchmark"]["y"],
                            s_res["benchmark"]["logits"],
                            s_res["benchmark"]["scores"])
    shuf_summary[s] = dict(res=s_res, cal=s_cal, val_auc=shuf_bv)
    del shuf_model; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# JointSTL  (joint pep+pseudo embedding, 1 head, include_sparse=False)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("JointSTL  (joint embedding, single head, seed=22)")
print("=" * 60)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Data prep matching train_JointSTL.py: global split first, then filter sparse
jstl_base = df_base.copy()
jstl_base["strata"] = jstl_base["MHC"] + "_" + jstl_base["Label"].astype(str)
sc2  = jstl_base["strata"].value_counts()
sing2 = jstl_base["strata"].isin(sc2[sc2 == 1].index)
jstl_tr_split, _ = train_test_split(
    jstl_base[~sing2], test_size=TEST_SIZE,
    stratify=jstl_base[~sing2]["strata"], random_state=SEED)
jstl_tr = pd.concat([jstl_tr_split, jstl_base[sing2]], ignore_index=True)
jstl_tr = jstl_tr.drop(columns=["strata"], errors="ignore")

# Filter: sparse out, EL_Rank, add HN
jstl_tr = jstl_tr[~jstl_tr["MHC"].isin(SPARSE_ALLELES)].copy()
iedb_neg = (jstl_tr["Source"] == "IEDB") & (jstl_tr["Label"] == 0)
jstl_tr  = jstl_tr[~(iedb_neg & (jstl_tr["EL_Rank"] > EL_RANK_THRESH))].copy()
jstl_hn  = (df_hn[df_hn["cluster"].isin(HN_CLUSTERS)]
              .pipe(lambda d: d[~d["MHC"].isin(SPARSE_ALLELES)])
              .pipe(lambda d: d[d["MHC"].isin(jstl_tr["MHC"].unique())])
              .reset_index(drop=True))
jstl_tr  = pd.concat([jstl_tr, jstl_hn], ignore_index=True)

# Balance per MHC
jstl_bal = []
for _, md in jstl_tr.groupby("MHC"):
    pos = md[md["Label"] == 1]; neg = md[md["Label"] == 0]
    if len(pos) == 0 or len(neg) == 0: continue
    jstl_bal.append(pd.concat([
        pos.sample(len(pos), random_state=SEED),
        neg.sample(min(len(neg), len(pos) * 2), random_state=SEED)]))
jstl_bal = pd.concat(jstl_bal).reset_index(drop=True)

jstl_tr2, jstl_va = train_test_split(jstl_bal, test_size=0.15,
                                      stratify=jstl_bal["Label"], random_state=SEED)
print(f"  Train={len(jstl_tr2)}  Val={len(jstl_va)}")

class JointDS(Dataset):
    def __init__(self, df):
        self.x   = torch.tensor(
            np.stack([JOINT_EMB[(r["Peptide"], r["pseudo"])] for _, r in df.iterrows()]),
            dtype=torch.float32)
        self.lbl = torch.tensor(df["Label"].values, dtype=torch.float32)
    def __len__(self):        return len(self.lbl)
    def __getitem__(self, i): return self.x[i], self.lbl[i]

jstl_loader = DataLoader(JointDS(jstl_tr2), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
jstl_val_x  = torch.tensor(
    np.stack([JOINT_EMB[(r["Peptide"], r["pseudo"])] for _, r in jstl_va.iterrows()]),
    dtype=torch.float32)
jstl_val_y  = jstl_va["Label"].values.astype(float)

class JointSTLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1  = nn.LSTM(ESM_DIM, 64, batch_first=True, bidirectional=True)
        self.lstm2  = nn.LSTM(128,     64, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(), nn.Dropout(D1),
            nn.Linear(64, 16), nn.LeakyReLU(), nn.Dropout(D2))
        self.head = nn.Linear(16, 1)
    def forward(self, x):
        h, _ = self.lstm1(x); h, _ = self.lstm2(h)
        return self.head(self.shared(h[:, -1, :])).squeeze(-1)

jstl_model = JointSTLModel().to(device)
jstl_crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([PW], device=device))
jstl_opt   = optim.AdamW(jstl_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
jstl_sch   = optim.lr_scheduler.CosineAnnealingLR(jstl_opt, T_max=EPOCHS)
best_jval, best_jstate = 0.0, copy.deepcopy(jstl_model.state_dict())

for ep in range(EPOCHS):
    jstl_model.train()
    for x, y in jstl_loader:
        x, y = x.to(device), y.to(device)
        jstl_opt.zero_grad()
        jstl_crit(jstl_model(x), y).backward()
        nn.utils.clip_grad_norm_(jstl_model.parameters(), CLIP_NORM)
        jstl_opt.step()
    jstl_sch.step()
    if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
        jstl_model.eval()
        out = []
        for i in range(0, len(jstl_val_x), 512):
            with torch.no_grad():
                out.append(torch.sigmoid(jstl_model(jstl_val_x[i:i+512].to(device))).cpu().numpy())
        v = roc_auc_score(jstl_val_y, np.concatenate(out)) if 0 < jstl_val_y.sum() < len(jstl_val_y) else 0.0
        if v > best_jval:
            best_jval = v; best_jstate = copy.deepcopy(jstl_model.state_dict())
        print(f"    ep {ep+1:3d}/{EPOCHS}  val_auc={v:.4f}  best={best_jval:.4f}", flush=True)

jstl_model.load_state_dict(best_jstate)
torch.save(jstl_model.state_dict(), f"{MDL_DIR}/JointSTL_s22.pt")
print(f"  Saved → {MDL_DIR}/JointSTL_s22.pt  (best val AUROC={best_jval:.4f})")

# Eval JointSTL on all 4 sets
jstl_prd = os.path.join(PRD_DIR, "JointSTL")
os.makedirs(jstl_prd, exist_ok=True)
jstl_res = {}
jstl_model.eval()
print(f"\n[JointSTL] Evaluating test sets:")
for ds_name, df_ds in eval_raw.items():
    bx = torch.tensor(
        np.stack([JOINT_EMB[(r["Peptide"], r["pseudo"])] for _, r in df_ds.iterrows()]),
        dtype=torch.float32)
    sc_out, lg_out = [], []
    for i in range(0, len(bx), 512):
        with torch.no_grad():
            lg = jstl_model(bx[i:i+512].to(device))
            sc_out.append(torch.sigmoid(lg).cpu().numpy())
            lg_out.append(lg.cpu().numpy())
    scores = np.concatenate(sc_out); logits = np.concatenate(lg_out)
    y = df_ds["Label"].values.astype(float)
    auroc = roc_auc_score(y, scores); ap = average_precision_score(y, scores)
    jstl_res[ds_name] = dict(auroc=auroc, ap=ap, y=y, scores=scores, logits=logits)
    out = df_ds[["Peptide", "MHC"]].copy()
    out["HLA_pseudo"] = df_ds["pseudo"]; out["Label"] = df_ds["Label"]; out["score"] = scores
    out.to_csv(f"{jstl_prd}/{ds_name}_predictions.csv", index=False)
    print(f"  {ds_name:<12}  AUROC={auroc:.4f}  AP={ap:.4f}")

jstl_val_logits = []
for i in range(0, len(jstl_val_x), 512):
    with torch.no_grad():
        jstl_val_logits.append(jstl_model(jstl_val_x[i:i+512].to(device)).cpu().numpy())
jstl_val_logits = np.concatenate(jstl_val_logits)
jstl_cal = run_calibration("JointSTL", jstl_val_y, jstl_val_logits,
                           jstl_res["benchmark"]["y"],
                           jstl_res["benchmark"]["logits"],
                           jstl_res["benchmark"]["scores"])

# ═══════════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)

named = {"ABC": (abc_res, abc_cal), "STL": (stl_res, stl_cal), "JointSTL": (jstl_res, jstl_cal)}
hdr = f"{'Model':<12} {'BM_AUROC':>9} {'BM_AP':>7} {'mRNA':>7} {'zero1':>7} {'zero2':>7} {'ECE_raw':>8} {'ECE_val':>8}"
print(hdr); print("-" * len(hdr))
for mn, (res, cal) in named.items():
    print(f"{mn:<12} {res['benchmark']['auroc']:>9.4f} {res['benchmark']['ap']:>7.4f} "
          f"{res['mRNA']['auroc']:>7.4f} {res['zero1']['auroc']:>7.4f} {res['zero2']['auroc']:>7.4f} "
          f"{cal['ece_raw']:>8.4f} {cal['ece_val']:>8.4f}")

# Shuffle aggregate
ds_keys = list(eval_raw.keys())
shuf_aucs = {ds: [shuf_summary[s]["res"][ds]["auroc"] for s in SHUFFLE_SEEDS] for ds in ds_keys}
shuf_aps  = {ds: [shuf_summary[s]["res"][ds]["ap"]    for s in SHUFFLE_SEEDS] for ds in ds_keys}
shuf_eces = {"raw": [shuf_summary[s]["cal"]["ece_raw"] for s in SHUFFLE_SEEDS],
             "val": [shuf_summary[s]["cal"]["ece_val"] for s in SHUFFLE_SEEDS]}
print(f"\nShuffle (cs=10, n=10):")
for ds in ds_keys:
    m, sd = np.mean(shuf_aucs[ds]), np.std(shuf_aucs[ds])
    ma, sa = np.mean(shuf_aps[ds]),  np.std(shuf_aps[ds])
    print(f"  {ds:<12}  AUROC={m:.4f}±{sd:.4f}  AP={ma:.4f}±{sa:.4f}")
print(f"  ECE raw={np.mean(shuf_eces['raw']):.4f}±{np.std(shuf_eces['raw']):.4f}  "
      f"val-scaled={np.mean(shuf_eces['val']):.4f}±{np.std(shuf_eces['val']):.4f}")

# Write report file
lines = ["ABLATION RESULTS", "=" * 70, ""]
for mn, (res, cal) in named.items():
    lines.append(f"{mn}  (val-fitted T={cal['T']:.4f})")
    for ds in ds_keys:
        lines.append(f"  {ds:<12}  AUROC={res[ds]['auroc']:.4f}  AP={res[ds]['ap']:.4f}")
    lines.append(f"  ECE raw={cal['ece_raw']:.4f}  val-scaled={cal['ece_val']:.4f}  in-sample={cal['ece_test']:.4f}")
    lines.append("")
lines.append(f"Shuffle (cs=10, seeds 1-10)  mean T={np.mean([shuf_summary[s]['cal']['T'] for s in SHUFFLE_SEEDS]):.4f}")
for ds in ds_keys:
    lines.append(f"  {ds:<12}  AUROC={np.mean(shuf_aucs[ds]):.4f}±{np.std(shuf_aucs[ds]):.4f}  "
                 f"AP={np.mean(shuf_aps[ds]):.4f}±{np.std(shuf_aps[ds]):.4f}")
lines.append(f"  ECE raw={np.mean(shuf_eces['raw']):.4f}±{np.std(shuf_eces['raw']):.4f}  "
             f"val-scaled={np.mean(shuf_eces['val']):.4f}±{np.std(shuf_eces['val']):.4f}")

rpt_path = os.path.join(RES_DIR, "ablations_metrics.txt")
with open(rpt_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nReport → {rpt_path}")
print("Done.")
