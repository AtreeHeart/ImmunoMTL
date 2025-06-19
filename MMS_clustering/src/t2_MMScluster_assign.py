# === Development Environment ===
"""
Python version:       3.7.13 
pandas version:       1.3.4
numpy version:        1.21.5
scikit-learn version: 1.0.2
scipy version:        1.7.3
"""

# === Imports ===
import pandas as pd
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from functions import entropy, normalized


# === Load and Preprocess Data ===
# Training data obtained from MHCflurry mass spec (Data_S3.csv)
# Source: https://data.mendeley.com/datasets/zx3kjzc3yx/3
# Note: This dataset is not included in this github repo
mhcflurry_df = pd.read_csv("../data/Data_S3.csv")
mhc_df_filtered = pd.read_csv("../data/MHCflurry_filtered.csv")
training = pd.read_csv("../data/MHCflurry_training.csv", index_col=0)
cluster_res = pd.read_csv("../results/clustering_resC4.csv", index_col = 0)

# === Select tier 2 HLAs (only 9mer data available) ===
mhcflurry_filtered = mhcflurry_df[
    mhcflurry_df["allele"].str.contains("HLA") &
    mhcflurry_df["allele"].str.contains(":") &
    ~mhcflurry_df["allele"].str.contains("D|o|E|G")
]

# Keep only mass spectrometry data and 8–11-mer peptides
mhcflurry_filtered = mhcflurry_filtered.copy()
mhcflurry_filtered["len"] = mhcflurry_filtered["peptide"].str.len()
tier1_hlas = set(mhc_df_filtered["allele"].unique())

# Select MS 9-mer data
mass_spec_9mer = mhcflurry_filtered[
    (mhcflurry_filtered["measurement_kind"] == "mass_spec") & 
    (mhcflurry_filtered["len"] == 9)
]

# Select quantitative affinity binders (IC50 <= 500 nM)
affinity_df = mhcflurry_filtered[mhcflurry_filtered["measurement_kind"] == "affinity"]
strong_binders = affinity_df[
    ((affinity_df["measurement_inequality"] == "=") & (affinity_df["measurement_value"] <= 500)) |
    ((affinity_df["measurement_inequality"] == "<") & (affinity_df["measurement_value"] <= 500))
]
affinity_9mer = strong_binders[strong_binders["peptide"].str.len() == 9]

# Combine MS and affinity data for 9-mers
tier2_pool = pd.concat([mass_spec_9mer, affinity_9mer])

hla_counts = tier2_pool.groupby("allele")["peptide"].count()
tier2_hlas = set(hla_counts[hla_counts > 10].index)

tier2_hlas = tier2_hlas - tier1_hlas
print(f"Tier 2 HLA candidates (n={len(tier2_hlas)}):\n", tier2_hlas)

tier2_mhc_df = mhcflurry_filtered[mhcflurry_filtered["allele"].isin(tier2_hlas)]



def assign_by_similarity(tier1_df, tier2_df, tier1_cluster_df, method="cosine"):
    results = []
    for t2_hla in tier2_df.index:
        t2_vec = tier2_df.loc[t2_hla].values

        best_score = -np.inf
        best_hla = None

        for t1_hla in tier1_df.index:
            t1_vec = tier1_df.loc[t1_hla].values

            if method == "cosine":
                score = 1 - cosine(t1_vec, t2_vec)  # cosine similarity
            elif method == "pearson":
                score, _ = pearsonr(t1_vec, t2_vec)
            elif method == "spearman":
                score, _ = spearmanr(t1_vec, t2_vec)
            else:
                raise ValueError("Unsupported method: choose cosine, pearson, or spearman")

            if score > best_score:
                best_score = score
                best_hla = t1_hla
        if "HLA" in tier1_cluster_df.columns:
            tier1_cluster_df = tier1_cluster_df.set_index("HLA")
        assigned_cluster = tier1_cluster_df.loc[best_hla, "cluster"]
        results.append({"HLA": t2_hla, "assigned_cluster": assigned_cluster, "best_match": best_hla, "similarity": best_score})

    return pd.DataFrame(results)

np_9mer = []
hla_unique_t2 = []

# Subset to 9-mer only
df_9mer = tier2_mhc_df[tier2_mhc_df["len"] == 9]
grouped = df_9mer.groupby("allele")

for allele, group in grouped:
    if len(group) < 10:
        continue  # skip if insufficient peptides

    split_pep = group["peptide"].str.split("", expand=True)
    entropy_res = normalized(entropy(split_pep))

    np_9mer.append(entropy_res)
    hla_unique_t2.append(allele)

# Convert to array and save
np_9mer = np.array(np_9mer)
tier2_entropy_df = pd.DataFrame(np_9mer)
tier2_entropy_df.index = hla_unique_t2
#tier2_entropy_df.to_csv("../HLA/MHCflurry_tier2_entropy_9mer.csv")

out = assign_by_similarity(training.iloc[:, 8:17], tier2_entropy_df, cluster_res, method="cosine")
print(out)
out.to_csv("../results/t2_cluster_res_c4.csv", index = False)