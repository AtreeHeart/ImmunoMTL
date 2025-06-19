import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from bin.TNinputcheck import checkinput

iedb = pd.read_csv("../data/training/BAinfo/iedb_BA.csv")
iedb["pMHC"] = iedb["Peptide"] + "_" + iedb["MHC"]
print("iedb; " + str(iedb.shape))
iedb.insert(loc = 3,
          column = 'Source',
          value = "IEDB")

iedb_processed = pd.read_csv("../data/training/processed/iedb_processed01.csv")
iedb_processed["pMHC"] = iedb_processed["Peptide"] + "_" + iedb_processed["MHC"]

hitide = pd.read_csv("../data/training/BAinfo/hitide_BA.csv")
print("hidite; " + str(hitide.shape))
hitide.insert(loc = 3,
          column = 'Source',
          value = "hitide")

nepdb = pd.read_csv("../data/training/BAinfo/nepdb_BA.csv")
print("nepdb; " + str(nepdb.shape))
nepdb.insert(loc = 3,
          column = 'Source',
          value = "nepdb")

vdjdb = pd.read_csv("../data/validation/BAinfo/vdjdb_BA.csv")
vdjdb.insert(loc = 3,
          column = 'Source',
          value = "VDJdb")
print("vdjdb; " + str(vdjdb.shape))

tsnadb2 = pd.read_csv("../data/training/BAinfo/tsnadb2_BA.csv")
tsnadb2["Label"] = 1
tsnadb2.insert(loc = 3,
          column = 'Source',
          value = "tsnadb2")

checkmate153 = pd.read_csv("../data/validation/other_datasets/CheckMate153.csv")
CM_columns = ["pep", "HLA", "label"]
checkmate153 = checkmate153[CM_columns]
checkmate153.rename(columns = {"pep": "Peptide", "HLA": "MHC", "label": "Label"}, inplace = True)
checkmate153.insert(loc = 3,
          column = 'Source',
          value = "Checkmate153")

#The HN_training file is not included in this repo because of its huge size. 
#You may run the script src/HNpep_generator2.py & src/merge_N.py

HumanNormal = pd.read_csv("../data/training/HN_training.csv")
HumanNormal["Label"] = 0
HumanNormal.insert(loc = 3,
          column = 'Source',
          value = "Human")
HumanNormal["pMHC"] = HumanNormal.Peptide + "_" + HumanNormal.MHC
HumanNormal.drop(columns=["mhcflurry_present_score", "mhcflurry_present_percent", "MHC_cluster"], inplace = True)
print("HumanNormal; " + str(HumanNormal.shape))

#select high confidence negative & strong binder from iedb
orgi_col = iedb.shape[1]-1
iedb_neg = pd.merge(iedb, iedb_processed, how="inner", left_on="pMHC", right_on="pMHC", suffixes=('', '__2'))
iedb_neg = iedb_neg[(iedb_neg["Label"] == 0) & (iedb_neg["Test_count"] >= 10) & (iedb_neg["EL_Rank"] <= 2)]
iedb_neg_HC = iedb_neg.iloc[:,0:orgi_col]
iedb_neg_HC.reset_index(drop=True, inplace = True)
print("High confidence negatives (IEDB); " + str(iedb_neg_HC.shape))

# concat all training dataset
iedb = iedb.iloc[:,0:orgi_col]

merged_training = pd.concat([iedb, nepdb, hitide, tsnadb2])
merged_training.insert(loc=4, column='pMHC', value=merged_training.Peptide + "_" + merged_training.MHC)
merged_training.drop_duplicates(subset="pMHC", inplace=True)
merged_training = merged_training.drop(columns=["stab_Thalf", "stab_Rank"])

# === Step 2: Merge and deduplicate validation data ===
merged_validation = pd.concat([vdjdb, iedb_neg_HC, checkmate153])
merged_validation.insert(loc=4, column='pMHC', value=merged_validation.Peptide + "_" + merged_validation.MHC)
merged_validation.drop_duplicates(subset="pMHC", inplace=True)
merged_validation = merged_validation.drop(columns=["stab_Thalf", "stab_Rank"])

# === Step 3: Remove overlap between training and validation by `pMHC` ===
merged_training = pd.merge(
    merged_training,
    merged_validation,
    how="left",
    on="pMHC",
    indicator=True,
    suffixes=('', '__val')
).query('_merge == "left_only"').drop(columns='_merge')

orgi_col = merged_training.shape[1]  # original column count before cluster addition
merged_training = merged_training.iloc[:, :orgi_col]

# === Step 4: Map clusters (Tier-1) ===
hla_clusters = pd.read_csv("../HLA/clustering_res.csv")
cluster_mapping = hla_clusters.set_index("HLA")["cluster"].to_dict()

merged_training["cluster"] = merged_training["MHC"].map(cluster_mapping)
merged_validation["cluster"] = merged_validation["MHC"].map(cluster_mapping)

training_data = merged_training.dropna(subset=["cluster"])
validation_data = merged_validation.dropna(subset=["cluster"])

# === Step 5: Filter out "sufficient" HLAs and define zero-shot set 1 ===
MIN_PEP_PER_HLA = 10

def count_classes(df):
    return df.groupby("MHC")["Label"].value_counts().unstack(fill_value=0)

class_dist = count_classes(training_data)
sufficient_hlas = class_dist[class_dist[1] >= MIN_PEP_PER_HLA].index

print("Total HLAs with sufficient data:", len(sufficient_hlas))

rare_hla_training = training_data[~training_data["MHC"].isin(sufficient_hlas)].reset_index(drop=True)
training_final = training_data[training_data["MHC"].isin(sufficient_hlas)].reset_index(drop=True)
rare_hla_val = validation_data[~validation_data["MHC"].isin(sufficient_hlas)].reset_index(drop=True)
vali_final = validation_data[validation_data["MHC"].isin(sufficient_hlas)].reset_index(drop=True)
zeroshot_data = pd.concat([rare_hla_training, rare_hla_val])
zeroshot_data = zeroshot_data.loc[:, ~zeroshot_data.columns.str.endswith("__val")]
#zeroshot_data = zeroshot_data[zeroshot_data["EL_Rank"]<2]
zeroshot_data = zeroshot_data[(zeroshot_data["Label"] == 1) | ((zeroshot_data["Label"] == 0) & (zeroshot_data["EL_Rank"] <= 0.5))]
print("Zero-shot set 1:", len(zeroshot_data))
print("Total HLAs in zeroshot set1 :", len(zeroshot_data["MHC"].unique()))


# === Step 6: Build zero-shot set 2 using Tier-2 cluster mapping ===
abandoned = pd.concat([merged_training, merged_validation], ignore_index=True)

t2_hla_clusters = pd.read_csv("../HLA/t2_cluster_res_c4.csv")
t2_cluster_mapping = t2_hla_clusters.set_index("HLA")["assigned_cluster"].to_dict()

abandoned["cluster"] = abandoned["MHC"].map(t2_cluster_mapping)
zeroshot_data2 = abandoned.dropna(subset=["cluster"]).reset_index(drop=True)
zeroshot_data2 = zeroshot_data2.loc[:, ~zeroshot_data2.columns.str.endswith("__val")]
#zeroshot_data2 = zeroshot_data2[zeroshot_data2["EL_Rank"]<2]
zeroshot_data2 = zeroshot_data2[(zeroshot_data2["Label"] == 1) | ((zeroshot_data2["Label"] == 0) & (zeroshot_data2["EL_Rank"] <= 0.5))]
print("Total HLAs in zeroshot set2 :", len(zeroshot_data2["MHC"].unique()))

# === Step 7: Print summary ===
print("Training set:", len(training_final))
print("Validation set:", len(vali_final))
print("Zero-shot set 1:", len(zeroshot_data))
print("Zero-shot set 2:", len(zeroshot_data2))

print("\n##########################################\n")

print("Training data source:")
print(training_final["Source"].value_counts())

print("Training data label distribution:")
print(training_final["Label"].value_counts())

print("Benchmark data source:")
print(vali_final["Source"].value_counts())

print("Benchmark data label distribution:")
print(vali_final["Label"].value_counts())

print("Zero-shot data source:")
print(zeroshot_data["Source"].value_counts())

print("Zero-shot data label distribution:")
print(zeroshot_data["Label"].value_counts())

print("Zero-shot data 2 source:")
print(zeroshot_data2["Source"].value_counts())

print("Zero-shot data 2 label distribution:")
print(zeroshot_data2["Label"].value_counts())

# === Step 8: Export ===
#training_final.to_csv("../data/training/training.csv", index=False)
#vali_final.to_csv("../data/validation/benchmark.csv", index=False)
zeroshot_data.to_csv("../data/validation/zeroshot_data.csv", index=False)
zeroshot_data2.to_csv("../data/validation/zeroshot_data2.csv", index=False)
