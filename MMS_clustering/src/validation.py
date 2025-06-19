# === Development Environment ===
"""
Python version:       3.7.13 
pandas version:       1.3.4
numpy version:        1.21.5
scikit-learn version: 1.0.2
scipy version:        1.7.3
tabulate:             0.9.0
"""

# === Imports ===
import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, mutual_info_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from functions import entropy, normalized
from tabulate import tabulate

# Testing data obtained from MHC motif atlas (MHC class I)
# Source: http://mhcmotifatlas.org/class1
# Note: This dataset is not included in this github repo
mhc_motif = pd.read_csv("../data/MHCmotifatlas.txt", sep="\t")

mhc_motif['nMHC'] = mhc_motif['Allele'].apply(lambda x: f"HLA-{x[:1]}*{x[1:3]}:{x[3:]}")
mhc_motif['len'] = mhc_motif['Peptide'].str.len()
mhc_motif = mhc_motif.query("8 <= len <= 11")
mhc_motif['merged'] = mhc_motif['nMHC'] + mhc_motif['Peptide']

training = pd.read_csv("../data/MHCflurry_training.csv", index_col=0)
flurry_df = pd.read_csv("../data/MHCflurry_filtered.csv")
flurry_df['merged'] = flurry_df['allele'] + flurry_df['peptide']

mhc_motif["validHLA"] = mhc_motif["nMHC"].isin(training.index)
mhc_motif["overlap_flurry"] = mhc_motif["merged"].isin(flurry_df["merged"])
mhc_motif = mhc_motif.query("validHLA and not overlap_flurry")

counts = mhc_motif.groupby(["nMHC", "len"]).size().reset_index(name="count")
eligible = counts[counts["count"] > 10]
hla_with_4_lengths = eligible.groupby("nMHC").filter(lambda x: x["len"].nunique() == 4)["nMHC"].unique()
#print(f"Total HLA allele types: {len(hla_with_4_lengths)}")

mhc_filtered = mhc_motif[mhc_motif["nMHC"].isin(hla_with_4_lengths)]
mhc_filtered.to_csv("../data/MHCmotifatlas_filtered.csv", index=False)

# === Compute Entropy Profiles ===
np_8mer = np.empty(shape=8)
np_9mer = np.empty(shape=9)
np_10mer = np.empty(shape=10)
np_11mer = np.empty(shape=11)

hla_unique = []
for i in range(8,12):
    df = mhc_filtered[mhc_filtered["len"]==i]
    for y in mhc_filtered["nMHC"].unique():
        hla_groupby = df.groupby(["nMHC"])
        peplist = hla_groupby.get_group(y)
        split_pep = peplist["Peptide"].str.split("", expand = True)
        entropy_res = normalized(entropy(split_pep))

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

testing = pd.DataFrame(np_kmer)
testing.index = hla_unique

# Combine into single feature matrix
testing = pd.DataFrame(np_kmer, index=hla_unique)
testing.to_csv("../data/MHCmotifatlas_testing.csv")

# === Load Models and run predictions ===
results = []
model_dir = "../models"
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(model_dir, model_file)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Predict clusters
        testing.columns = training.columns
        clusters_test = model.predict(testing)
        #clusters_train = model.labels_
        clusters_train = model.predict(training)

        res_test = pd.DataFrame({"HLA": testing.index, "clusters": clusters_test})
        res_train = pd.DataFrame({"HLA": training.index, "clusters": clusters_train})
        merged = pd.merge(res_test, res_train, on="HLA", suffixes=("_x", "_y"))

        # Compute metrics
        ari = adjusted_rand_score(merged["clusters_y"], merged["clusters_x"])
        mi = mutual_info_score(merged["clusters_y"], merged["clusters_x"])
        acc = accuracy_score(merged["clusters_y"], merged["clusters_x"])

        results.append({
            "Model": model_file,
            "ARI": round(ari, 4),
            "MI": round(mi, 4),
            "Accuracy": round(acc, 4),
            "Misclassified Count": (merged["clusters_y"] != merged["clusters_x"]).sum()
        })
        merged.to_csv(f"../results/{model_file.split('.')[0]}_pred_vs_true.csv", index=False)

# === Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("../results/clustering_eval_summary.csv", index=False)

# === Summary Stats for Figure 2a ===
def get_summary(df, name, mhc='nMHC'):
    num_hlas = df[mhc].nunique()
    num_peptides = df['merged'].nunique()
    return {"Dataset": name, "HLA Count": num_hlas, "pMHC Count": num_peptides}

# MHCflurry summary
mhcflurry_summary = get_summary(flurry_df, "MHCflurry", mhc="allele")

# Motif Atlas (total)
motif_all = mhc_motif.copy()
motif_all['Dataset'] = "Motif Atlas (all)"
motif_all_summary = get_summary(motif_all, "Motif Atlas (all)")

# Motif Atlas (non-overlapping, valid HLA only)
motif_filtered = mhc_filtered.copy()
motif_filtered['Dataset'] = "Motif Atlas (non-overlap)"
motif_filtered_summary = get_summary(motif_filtered, "Motif Atlas (non-overlap)")

# Combine summaries
summary_df = pd.DataFrame([mhcflurry_summary, motif_all_summary, motif_filtered_summary])
#summary_df.to_csv("../results/figure2a_dataset_summary.csv", index=False)

print("\n=== Dataset Summary for Figure 2a ===")
print(tabulate(summary_df, headers="keys", tablefmt="psql"))

print(tabulate(results_df, headers="keys", tablefmt="psql", showindex=False))

