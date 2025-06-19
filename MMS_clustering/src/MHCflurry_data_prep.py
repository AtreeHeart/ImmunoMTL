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
from scipy.spatial.distance import euclidean
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from functions import entropy, normalized


# === Load and Preprocess Data ===
# Training data obtained from MHCflurry mass spec (Data_S3.csv)
# Source: https://data.mendeley.com/datasets/zx3kjzc3yx/3
# Note: This dataset is not included in this github repo
mhcflurry_df = pd.read_csv("../data/Data_S3.csv")

# Filter valid HLA alleles 
mhcflurry_filtered = mhcflurry_df[
    mhcflurry_df["allele"].str.contains("HLA") &
    mhcflurry_df["allele"].str.contains(":") &
    ~mhcflurry_df["allele"].str.contains("D|o|E|G")
]

# Keep only mass spectrometry data and 8–11-mer peptides
mhcflurry_filtered = mhcflurry_filtered.copy()
mhcflurry_filtered["len"] = mhcflurry_filtered["peptide"].str.len()
mhcflurry = mhcflurry_filtered[
    (mhcflurry_filtered["measurement_kind"] == "mass_spec") &
    (mhcflurry_filtered["len"].between(8, 11))
]

# Select HLA types with enough peptides for all 8–11-mer lengths
tmp_df = mhcflurry.groupby(['allele', 'len']).count()
filtered_df = tmp_df[tmp_df["peptide"] > 10]

final_hla = filtered_df.index.get_level_values('allele').unique()
final_hla_lst = [
    hla for hla in final_hla
    if filtered_df.index.get_level_values(0).tolist().count(hla) == 4
]

print(f"Tier 1 HLA candidates (n={len(final_hla_lst)}):\n", final_hla_lst)
mhc_df_filtered = mhcflurry[mhcflurry["allele"].isin(final_hla_lst)]
mhc_df_filtered.to_csv("../data/MHCflurry_filtered.csv")
tier1_hlas = set(final_hla_lst)

# Compute entropy for each length
np_8mer = np.empty(shape=8)
np_9mer = np.empty(shape=9)
np_10mer = np.empty(shape=10)
np_11mer = np.empty(shape=11)

hla_unique = []
for i in range(8,12):
    df = mhc_df_filtered[mhc_df_filtered["len"]==i]
    for y in mhc_df_filtered["allele"].unique():
        hla_groupby = df.groupby(["allele"])
        peplist = hla_groupby.get_group(y)
        split_pep = peplist["peptide"].str.split("", expand = True)
        entropy_res = normalized(entropy(split_pep))
        #print(y + "_" + str(i))
        #print(entropy_res)
        #print(len(peplist))
        if i == 8:
            np_8mer = np.vstack((np_8mer,entropy_res))
            hla_unique.append(y)
        elif i == 9:
            np_9mer = np.vstack((np_9mer,entropy_res))
        elif i == 10:
            np_10mer = np.vstack((np_10mer,entropy_res))
        elif i == 11:
            np_11mer = np.vstack((np_11mer,entropy_res))

np_8mer = np.delete(np_8mer, (0), axis = 0)
np_9mer = np.delete(np_9mer, (0), axis = 0)
np_10mer = np.delete(np_10mer, (0), axis = 0)
np_11mer = np.delete(np_11mer, (0), axis = 0)
np_kmer = np.concatenate((np_8mer, np_9mer, np_10mer, np_11mer), axis=1)

training = pd.DataFrame(np_kmer)
training.index = hla_unique
training.to_csv("../data/MHCflurry_training.csv")

pca = PCA(n_components=2)
pca_df = pca.fit_transform(training)
pca_df_pd = pd.DataFrame(pca_df)
pca_df_pd.to_csv("../data/tier1_hla_pca.csv")
