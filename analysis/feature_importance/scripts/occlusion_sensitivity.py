#!/usr/bin/env python3
"""
occlusion_sensitivity.py — Occlusion sensitivity analysis for ImmunoMTL_s22.pt.

Following the DeepImmuno occlusion strategy:
  1. Freeze the trained model (no weight updates).
  2. For each positive sample, compute original score and masked score at each
     real peptide position (embedding zeroed). delta = original - masked.
  3. Bootstrap robustness: 100 iterations × 2,000 randomly-sampled positives.
     Record mean delta and position rank per iteration.
  4. Aggregate: canonical position, peptide length, HLA allele, MMS cluster.
  5. Merge per-HLA positional importance with MMS entropy; compute Spearman r.

Supports two sample pools:
  --mode train   : positive training instances (default, N≈9,938)
  --mode bench   : positive benchmark instances (held-out)
  --mode both    : both combined

Outputs (in pred_results/occlusion_sensitivity/):
  per_sample_delta.csv         – one row per (sample, canonical position)
  bootstrap_stats.csv          – mean delta + rank per (iteration, position)
  per_hla_importance.csv       – per-HLA × position mean delta + MMS entropy
  correlation_results.csv      – Spearman r and p per cluster
  figures/                     – bar, heatmap, scatter, bootstrap rank plots
"""
import os, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train','bench','both'], default='train')
parser.add_argument('--n-bootstrap', type=int, default=100)
parser.add_argument('--bootstrap-n', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ── Config ────────────────────────────────────────────────────────────────────
ESM_ID   = "facebook/esm2_t12_35M_UR50D"
ESM_DIM  = 480
PEP_LEN  = 11
MHC_LEN  = 34
CKPT     = "../../../models/ImmunoMTL_s22.pt"
DATA_DIR = "../../../data"
HLA_DIR  = "../../../HLA"
MMS_DIR  = "../../../MMS_clustering"
OUT_DIR  = "../data"
FIG_DIR  = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}  |  Mode: {args.mode}  |  "
      f"Bootstrap: {args.n_bootstrap} iter × {args.bootstrap_n} samples")

POS_LABELS = ["P1","P2","P3","P4","PΩ-3","PΩ-2","PΩ-1","PΩ"]
CLUSTER_COLORS = {0:"#45566B", 1:"#EDC264", 2:"#F28482", 3:"#84A59D"}

def position_map(L):
    """Absolute index → canonical label for real positions only."""
    m = {0:"P1", 1:"P2", 2:"P3", 3:"P4",
         L-4:"PΩ-3", L-3:"PΩ-2", L-2:"PΩ-1", L-1:"PΩ"}
    return m   # middle positions simply absent from the dict

# MMS column lookup: canonical position × peptide length → column index in MHCflurry_training.csv
MMS_COL = {
    "P1":   {8:0,  9:8,  10:17, 11:27},
    "P2":   {8:1,  9:9,  10:18, 11:28},
    "P3":   {8:2,  9:10, 10:19, 11:29},
    "P4":   {8:3,  9:11, 10:20, 11:30},
    "PΩ-3": {8:4,  9:13, 10:23, 11:34},
    "PΩ-2": {8:5,  9:14, 10:24, 11:35},
    "PΩ-1": {8:6,  9:15, 10:25, 11:36},
    "PΩ":   {8:7,  9:16, 10:26, 11:37},
}

# ── Lookups ───────────────────────────────────────────────────────────────────
mhc_dict = {}
with open(f"{HLA_DIR}/MHC_pseudo.dat") as f:
    for line in f:
        p = line.strip().split()
        if len(p) == 2: mhc_dict[p[0]] = p[1]

def lookup_pseudo(mhc):
    return mhc_dict.get(mhc.replace("*","").replace(":",""), None)

cluster_map  = pd.read_csv(f"{HLA_DIR}/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
t2_cluster_map = pd.read_csv(f"{HLA_DIR}/t2_cluster_res_c4.csv").set_index("HLA")["assigned_cluster"].to_dict()
N_TASKS = len(set(cluster_map.values()))

def lookup_cluster(mhc):
    c = cluster_map.get(mhc)
    if c is None: c = t2_cluster_map.get(mhc)
    return int(c) if c is not None else None

mms_tr = pd.read_csv(f"{MMS_DIR}/data/MHCflurry_training.csv", index_col=0)
mms_tr.columns = mms_tr.columns.astype(int)

def mms_weighted(hla, len_dist):
    """Length-weighted MMS entropy per canonical position for one HLA."""
    row = mms_tr.loc[hla]
    total = sum(len_dist.values())
    if total == 0: return {}
    result = {}
    for pos in POS_LABELS:
        val = 0.0
        for L, cnt in len_dist.items():
            col = MMS_COL[pos].get(L)
            if col is not None:
                val += (cnt / total) * row[col]
        result[pos] = val
    return result

# ── Load data ─────────────────────────────────────────────────────────────────
print("[INFO] Loading data ...")

def load_split(path, label_col="Label"):
    df = pd.read_csv(path)
    df["HLA_pseudo"] = df["MHC"].apply(lookup_pseudo)
    df["cluster"]    = df["MHC"].apply(lookup_cluster)
    df["pep_len"]    = df["Peptide"].str.len()
    df = df.dropna(subset=["HLA_pseudo","cluster"]).reset_index(drop=True)
    df["cluster"] = df["cluster"].astype(int)
    return df

parts = []
if args.mode in ("train","both"):
    tr = load_split(f"{DATA_DIR}/training.csv")
    tr["split"] = "train"
    parts.append(tr)
if args.mode in ("bench","both"):
    bn = load_split(f"{DATA_DIR}/benchmark.csv")
    bn["split"] = "bench"
    parts.append(bn)

df_all = pd.concat(parts, ignore_index=True)

# Restrict to positives for primary analysis
df_pos = df_all[df_all["Label"] == 1].reset_index(drop=True)
print(f"  Total positives: {len(df_pos)}  |  HLAs: {df_pos['MHC'].nunique()}  |  "
      f"lengths: {sorted(df_pos['pep_len'].unique())}")
print(f"  Cluster dist: {df_pos['cluster'].value_counts().sort_index().to_dict()}")

# ── ESM2 embeddings ───────────────────────────────────────────────────────────
all_peps = sorted(df_pos["Peptide"].unique())
all_mhcs = sorted(df_pos["HLA_pseudo"].unique())
print(f"\n[INFO] Embedding {len(all_peps)} unique peptides + {len(all_mhcs)} MHC sequences ...")

tok = AutoTokenizer.from_pretrained(ESM_ID)
esm = AutoModel.from_pretrained(ESM_ID).eval().to(device)

def embed_seqs(seqs, max_len):
    out = []
    for i in tqdm(range(0, len(seqs), 64), desc="  ESM2", leave=False):
        b = tok(seqs[i:i+64], return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_len,
                add_special_tokens=False).to(device)
        with torch.no_grad():
            out.append(esm(**b).last_hidden_state.cpu())
    return torch.cat(out, 0)

PEP_EMB = dict(zip(all_peps, embed_seqs(all_peps, PEP_LEN)))
MHC_EMB = dict(zip(all_mhcs, embed_seqs(all_mhcs, MHC_LEN)))
del esm; torch.cuda.empty_cache()
print(f"  Embeddings done ({time.time()-t0:.0f}s)")

# ── Model (frozen) ────────────────────────────────────────────────────────────
class MTLModel(nn.Module):
    def __init__(self):
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
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(N_TASKS)])

    def forward(self, xp, xm):
        p, _ = self.pep1(xp); p, _ = self.pep2(p); p = p[:, -1, :]
        m, _ = self.mhc1(xm); m, _ = self.mhc2(m); m = m[:, -1, :]
        s = self.shared(torch.cat([p, m], dim=1))
        return [h(s).squeeze(-1) for h in self.heads]

model = MTLModel().eval().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
for p in model.parameters():
    p.requires_grad_(False)          # explicitly freeze
print(f"[INFO] Loaded frozen model: {CKPT}")

def forward_scores(pe_tensor, me_tensor, cluster_ids):
    """Batched inference; routes each sample to its cluster head."""
    scores = torch.zeros(len(cluster_ids))
    with torch.no_grad():
        outs = model(pe_tensor.to(device), me_tensor.to(device))
    for tid in range(N_TASKS):
        mask = cluster_ids == tid
        if mask.any():
            scores[mask] = torch.sigmoid(outs[tid][mask.to(device)]).cpu()
    return scores.numpy()

# ── Pre-stack all positive embeddings ────────────────────────────────────────
print("[INFO] Pre-stacking embeddings ...")
pe_all  = torch.stack([PEP_EMB[p] for p in df_pos["Peptide"]])   # [N, 11, 480]
me_all  = torch.stack([MHC_EMB[m] for m in df_pos["HLA_pseudo"]]) # [N, 34, 480]
cl_all  = torch.tensor(df_pos["cluster"].values)
L_all   = df_pos["pep_len"].values
N_POS   = len(df_pos)

# ── Step 1: original scores ───────────────────────────────────────────────────
print("[INFO] Computing original scores ...")
BATCH = 512
orig_scores = np.zeros(N_POS)
for s in tqdm(range(0, N_POS, BATCH), desc="  original", leave=False):
    e = min(s + BATCH, N_POS)
    orig_scores[s:e] = forward_scores(pe_all[s:e], me_all[s:e], cl_all[s:e])

# ── Step 2: per-position occlusion (all positives) ────────────────────────────
print("[INFO] Computing occlusion deltas (all positives) ...")

# Build flat list of (sample_idx, abs_pos, canonical_label)
pairs = []
for i, L in enumerate(L_all):
    for abs_pos, label in position_map(int(L)).items():
        pairs.append((i, abs_pos, label))

pairs_arr = np.array([(i, ap) for i, ap, _ in pairs])
labels_arr = [lbl for _, _, lbl in pairs]
M = len(pairs_arr)
print(f"  Total masked passes: {M:,}")

occ_delta = np.zeros(M)
for s in tqdm(range(0, M, BATCH), desc="  occlusion", leave=False):
    e = min(s + BATCH, M)
    idxs    = pairs_arr[s:e, 0]
    abs_pos = pairs_arr[s:e, 1]

    pe_b = pe_all[idxs].clone()
    me_b = me_all[idxs]
    cl_b = cl_all[idxs]

    for bi in range(e - s):
        pe_b[bi, abs_pos[bi], :] = 0.0

    occ_delta[s:e] = orig_scores[idxs] - forward_scores(pe_b, me_b, cl_b)

print(f"  Done ({time.time()-t0:.0f}s)")

# ── Build per-sample delta DataFrame ─────────────────────────────────────────
print("[INFO] Building per-sample delta table ...")
rows = []
for k, (i, abs_pos, label) in enumerate(pairs):
    r = df_pos.iloc[i]
    rows.append({
        "sample_idx":  i,
        "Peptide":     r["Peptide"],
        "MHC":         r["MHC"],
        "pep_len":     int(L_all[i]),
        "cluster":     int(r["cluster"]),
        "split":       r.get("split", "train"),
        "abs_pos":     abs_pos,
        "pos_label":   label,
        "orig_score":  orig_scores[i],
        "delta":       occ_delta[k],
    })

delta_df = pd.DataFrame(rows)
delta_df.to_csv(f"{OUT_DIR}/per_sample_delta.csv", index=False)
print(f"  Saved per_sample_delta.csv  ({len(delta_df):,} rows)")

# ── Step 3: bootstrap aggregation ────────────────────────────────────────────
print(f"\n[INFO] Bootstrap: {args.n_bootstrap} iter × {args.bootstrap_n} samples ...")

# Pivot to [N_POS × 8] for fast bootstrap indexing
pivot = (delta_df.pivot_table(index="sample_idx", columns="pos_label",
                               values="delta", aggfunc="mean")
                 .reindex(columns=POS_LABELS))

# Only rows with all 8 positions present (8-mers have all 8; longer peptides drop middles)
pivot_full = pivot.dropna()  # 8-mers and samples where all 8 positions exist
if len(pivot_full) < args.bootstrap_n:
    # Fall back to partial rows (allow NaN, fill with column mean)
    pivot_mat = pivot.fillna(pivot.mean())
else:
    pivot_mat = pivot_full

sample_ids = pivot_mat.index.values
mat = pivot_mat.values  # [N, 8]

boot_rows = []
for it in tqdm(range(args.n_bootstrap), desc="  bootstrap", leave=False):
    n_draw = min(args.bootstrap_n, len(sample_ids))
    idx = rng.choice(len(sample_ids), size=n_draw, replace=True)
    means = mat[idx].mean(axis=0)          # [8] mean delta per position
    ranks = (len(POS_LABELS) + 1
             - pd.Series(means, index=POS_LABELS).rank().values)  # rank 1=most important
    for pi, pos in enumerate(POS_LABELS):
        boot_rows.append({
            "iteration": it,
            "pos_label": pos,
            "mean_delta": means[pi],
            "rank":       ranks[pi],
        })

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(f"{OUT_DIR}/bootstrap_stats.csv", index=False)

# Summary: mean ± std across iterations
boot_summary = (boot_df.groupby("pos_label")
                .agg(mean_delta_mean=("mean_delta","mean"),
                     mean_delta_std =("mean_delta","std"),
                     mean_rank      =("rank","mean"),
                     rank_std       =("rank","std"))
                .reindex(POS_LABELS).reset_index())
print("  Bootstrap summary (mean delta ± std):")
print(boot_summary[["pos_label","mean_delta_mean","mean_delta_std","mean_rank"]].to_string(index=False))

# ── Step 4: per-HLA aggregation + MMS merge ───────────────────────────────────
print("\n[INFO] Per-HLA positional importance + MMS merge ...")
hla_rows = []
for hla, grp in delta_df.groupby("MHC"):
    if hla not in mms_tr.index:
        continue
    cluster = int(grp["cluster"].iloc[0])
    len_dist = grp.groupby("pep_len").size().to_dict()  # {L: count_of_samples}
    mms_w    = mms_weighted(hla, len_dist)
    occ_mean = grp.groupby("pos_label")["delta"].mean().to_dict()
    occ_std  = grp.groupby("pos_label")["delta"].std().to_dict()
    n_samp   = grp["sample_idx"].nunique()
    for pos in POS_LABELS:
        hla_rows.append({
            "HLA":          hla,
            "cluster":      cluster,
            "pos_label":    pos,
            "mean_delta":   occ_mean.get(pos, np.nan),
            "std_delta":    occ_std.get(pos, np.nan),
            "n_samples":    n_samp,
            "mms_entropy":  mms_w.get(pos, np.nan),
        })

hla_df = pd.DataFrame(hla_rows).dropna(subset=["mean_delta","mms_entropy"])
hla_df.to_csv(f"{OUT_DIR}/per_hla_importance.csv", index=False)
print(f"  {len(hla_df)} rows | {hla_df['HLA'].nunique()} HLAs")

# ── Step 5: Spearman correlation ──────────────────────────────────────────────
print("\n[RESULTS] Spearman correlation (mean_delta ~ mms_entropy):")
print(f"{'Cluster':<12} {'HLAs':>5} {'Pairs':>6} {'r':>8} {'p':>12}")
print("-" * 48)
corr_rows = []
for cid in sorted(hla_df["cluster"].unique()):
    sub = hla_df[hla_df["cluster"] == cid]
    r, p = spearmanr(sub["mean_delta"], sub["mms_entropy"])
    n_hla = sub["HLA"].nunique()
    n_pair = len(sub)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
    print(f"  Cluster {cid+1:<3}  {n_hla:>5}  {n_pair:>6}  {r:>+8.3f}  {p:>12.3e}  {sig}")
    corr_rows.append({"cluster": cid+1, "n_hlas": n_hla, "n_pairs": n_pair,
                      "spearman_r": round(r,4), "p_value": p})
r_all, p_all = spearmanr(hla_df["mean_delta"], hla_df["mms_entropy"])
n_all = hla_df["HLA"].nunique()
print(f"  {'Overall':<10}  {n_all:>5}  {len(hla_df):>6}  {r_all:>+8.3f}  {p_all:>12.3e}")
corr_rows.append({"cluster":"all","n_hlas":n_all,"n_pairs":len(hla_df),
                  "spearman_r":round(r_all,4),"p_value":p_all})
pd.DataFrame(corr_rows).to_csv(f"{OUT_DIR}/correlation_results.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────

# Fig 1 — Mean delta per position (bootstrap CI), per cluster
print("\n[PLOT] Fig 1: mean delta per position (bootstrap CI) ...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
fig.suptitle(f"Occlusion Sensitivity: Mean Δscore per Canonical Position\n"
             f"(bootstrap {args.n_bootstrap}×{args.bootstrap_n}, positive samples)",
             fontsize=12)

for ax_i, cid in enumerate(sorted(hla_df["cluster"].unique())):
    ax = axes[ax_i]
    # Per-cluster bootstrap summary
    sub_delta = delta_df[delta_df["cluster"] == cid]
    pivot_c = (sub_delta.pivot_table(index="sample_idx", columns="pos_label",
                                      values="delta", aggfunc="mean")
                        .reindex(columns=POS_LABELS).fillna(0))
    mat_c = pivot_c.values
    ids_c = np.arange(len(mat_c))
    it_means = []
    for _ in range(args.n_bootstrap):
        idx = rng.choice(len(ids_c),
                         size=min(args.bootstrap_n, len(ids_c)), replace=True)
        it_means.append(mat_c[idx].mean(axis=0))
    it_means = np.array(it_means)   # [n_boot, 8]
    mu  = it_means.mean(axis=0)
    ci  = np.percentile(it_means, [2.5, 97.5], axis=0)

    x = np.arange(len(POS_LABELS))
    color = CLUSTER_COLORS[cid]
    ax.bar(x, mu, color=color, alpha=0.82, edgecolor="#555", linewidth=0.6)
    ax.errorbar(x, mu,
                yerr=[mu - ci[0], ci[1] - mu],
                fmt="none", color="#333", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(POS_LABELS, rotation=45, ha="right", fontsize=9)
    r_c = corr_rows[cid]["spearman_r"]
    p_c = corr_rows[cid]["p_value"]
    sig = "**" if p_c < 0.01 else ("*" if p_c < 0.05 else "n.s.")
    ax.set_title(f"Cluster {cid+1}  (ρ={r_c:+.3f} {sig})", fontsize=11)
    ax.set_ylabel("Mean Occlusion Δscore", fontsize=9)
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_mean_delta_bootstrap.pdf", bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig1_mean_delta_bootstrap.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 2 — Bootstrap rank distribution per position (boxplot)
print("[PLOT] Fig 2: bootstrap rank distribution ...")
fig, ax = plt.subplots(figsize=(10, 5))
boot_df["pos_label"] = pd.Categorical(boot_df["pos_label"],
                                       categories=POS_LABELS, ordered=True)
sns.boxplot(data=boot_df, x="pos_label", y="rank", order=POS_LABELS,
            color="#8EA8C3", linewidth=0.8, fliersize=2, ax=ax)
ax.invert_yaxis()   # rank 1 (most important) at top
ax.set_xlabel("Canonical Position", fontsize=11)
ax.set_ylabel("Importance Rank (1 = most important)", fontsize=11)
ax.set_title(f"Bootstrap Rank Stability ({args.n_bootstrap} iterations)", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_bootstrap_rank.pdf", bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig2_bootstrap_rank.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 3 — Scatter: MMS entropy vs mean delta per cluster
print("[PLOT] Fig 3: scatter MMS entropy vs mean delta ...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle("MMS Entropy vs Occlusion Δscore\n"
             "(one point = HLA × canonical position)", fontsize=12, y=1.02)

for ax_i, cid in enumerate(sorted(hla_df["cluster"].unique())):
    ax = axes[ax_i]
    sub = hla_df[hla_df["cluster"] == cid]
    color = CLUSTER_COLORS[cid]
    ax.scatter(sub["mms_entropy"], sub["mean_delta"],
               c=color, alpha=0.72, s=50, edgecolors="#555", linewidths=0.4)
    if len(sub) > 2:
        from numpy.polynomial.polynomial import polyfit as pfp
        cf, mf = pfp(sub["mms_entropy"].values, sub["mean_delta"].values, 1)
        xr = np.linspace(sub["mms_entropy"].min(), sub["mms_entropy"].max(), 50)
        ax.plot(xr, cf + mf*xr, color="#333", lw=1.5, ls="--")
    r  = corr_rows[cid]["spearman_r"]
    p  = corr_rows[cid]["p_value"]
    nh = corr_rows[cid]["n_hlas"]
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
    ax.set_title(f"Cluster {cid+1}  (n={nh} HLAs)\nρ={r:+.3f}  {sig}", fontsize=11)
    ax.set_xlabel("MMS Entropy", fontsize=10)
    if ax_i == 0: ax.set_ylabel("Mean Occlusion Δscore", fontsize=10)
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    for pos in ["P2","PΩ"]:
        pts = sub[sub["pos_label"] == pos]
        if len(pts) == 0: continue
        ax.annotate(pos, (pts["mms_entropy"].mean(), pts["mean_delta"].mean()),
                    fontsize=8, ha="center", va="bottom", color="#333",
                    xytext=(0, 4), textcoords="offset points")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_scatter_mms_vs_delta.pdf", bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig3_scatter_mms_vs_delta.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 4 — Heatmap: per-HLA × position mean delta (normalized)
print("[PLOT] Fig 4: per-HLA heatmap ...")
import matplotlib.patches as mpatches

pivot_hla = hla_df.pivot_table(index="HLA", columns="pos_label",
                                values="mean_delta").reindex(columns=POS_LABELS)
cl_order = hla_df.groupby("HLA")["cluster"].first().sort_values()
pivot_hla = pivot_hla.loc[cl_order.index]
row_colors = cl_order.map(CLUSTER_COLORS)
pivot_norm = pivot_hla.div(pivot_hla.max(axis=1), axis=0)

g = sns.clustermap(pivot_norm, row_cluster=False, col_cluster=False,
                   row_colors=row_colors, cmap="YlOrRd",
                   figsize=(10, max(8, len(pivot_hla)//5)),
                   xticklabels=True, yticklabels=True, linewidths=0,
                   cbar_kws={"label": "Normalized Δscore"})
g.ax_heatmap.set_xlabel("Position", fontsize=11)
g.ax_heatmap.set_ylabel("HLA", fontsize=11)
g.ax_heatmap.set_title("Per-HLA Occlusion Δscore (row-normalized)", fontsize=11)
handles = [mpatches.Patch(color=CLUSTER_COLORS[c], label=f"Cluster {c+1}")
           for c in sorted(CLUSTER_COLORS)]
g.ax_heatmap.legend(handles=handles, loc="upper right",
                    bbox_to_anchor=(1.35, 1.05), fontsize=9, frameon=False)
g.savefig(f"{FIG_DIR}/fig4_heatmap_per_hla.pdf")
g.savefig(f"{FIG_DIR}/fig4_heatmap_per_hla.png", dpi=150)
plt.close()

# Fig 5 — Per-length breakdown: mean delta by length × position
print("[PLOT] Fig 5: per-length position profile ...")
len_pos = (delta_df.groupby(["pep_len","pos_label"])["delta"]
           .mean().reset_index().rename(columns={"delta":"mean_delta"}))
lengths = sorted(len_pos["pep_len"].unique())
fig, axes = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 4), sharey=True)
if len(lengths) == 1: axes = [axes]
fig.suptitle("Mean Occlusion Δscore by Peptide Length", fontsize=12)
for ax, L in zip(axes, lengths):
    sub = len_pos[len_pos["pep_len"] == L].set_index("pos_label").reindex(POS_LABELS)
    ax.bar(range(len(POS_LABELS)), sub["mean_delta"].values,
           color="#8EA8C3", alpha=0.85, edgecolor="#555", linewidth=0.6)
    ax.set_xticks(range(len(POS_LABELS)))
    ax.set_xticklabels(POS_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{L}-mer\n(n={int((delta_df['pep_len']==L).sum()//len(POS_LABELS))})",
                 fontsize=10)
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Mean Occlusion Δscore", fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_per_length.pdf", bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig5_per_length.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[DONE] Total time: {(time.time()-t0)/60:.1f} min")
print(f"Outputs: {OUT_DIR}/")
print(f"  per_sample_delta.csv       {len(delta_df):,} rows")
print(f"  bootstrap_stats.csv        {len(boot_df):,} rows")
print(f"  per_hla_importance.csv     {len(hla_df)} rows | {hla_df['HLA'].nunique()} HLAs")
print(f"  correlation_results.csv    {len(corr_rows)} rows")
print(f"  figures/                   fig1–fig5")
