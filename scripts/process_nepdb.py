import pandas as pd
import numpy as np
import os

# NOTE: The raw data processed here was obtained from the NEPdb.
# This file, 'NECID_Query.csv', is not included in this repo.
# To use this script, please download the data from the NEPdb website at https://nep.whu.edu.cn/
# After downloading, save the file as 'NECID_Query.csv' in the '../data/training/raw/nepdb/' directory.

nepdb = pd.read_csv("../data/training/raw/nepdb/NECID_Query.csv")

nepdb_select = ["mut_peptide",  "HLA", "response", "Tumor Type", "genesymbol",]
nepdb_select_df = nepdb[nepdb_select]
nepdb_select_df

#iedb_df = pd.read_csv("tcell_full_v3.csv")
nepdb_select_df = nepdb_select_df.dropna()

#8-11mer peptides
len_list = []
for epitope in nepdb_select_df["mut_peptide"]:
    len_list.append(len(epitope))
nepdb_select_df["pep_len"] = len_list
nepdb_select_df = nepdb_select_df[(nepdb_select_df["pep_len"] >= 8) & (nepdb_select_df["pep_len"] <= 11)]

res_list = []
for res in nepdb_select_df["response"]:
    if res == "P":
        res_list.append(1)
    elif res == "N":
        res_list.append(0)

nepdb_select_df["label"] = res_list

#MHC type 
nepdb_select_df = nepdb_select_df[nepdb_select_df['HLA'].str.contains('D')==False]
nepdb_select_df = nepdb_select_df[nepdb_select_df['HLA'].str.contains('E')==False]
nepdb_select_df = nepdb_select_df[nepdb_select_df['HLA'].str.contains('o')==False]
nepdb_select_df = nepdb_select_df[nepdb_select_df['HLA'].str.contains('G')==False]
nepdb_select_df = nepdb_select_df[nepdb_select_df['HLA'].str.contains(':') ]
#nepdb_select_df = nepdb_select_df[nepdb_select_df["HLA"].apply(lambda x: len(x) == 11)]
#iedb_df = iedb_df[iedb_df['nMHC'].isin(HLA_df["HLA"])]

nepdb_select_df["pMHC_merge"] = nepdb_select_df["mut_peptide"] + "_" + nepdb_select_df["HLA"]
nepdb_select_df = nepdb_select_df.drop_duplicates(subset = ["pMHC_merge"])
#iedb_df = iedb_df.dropna(subset = ["Assay - Number of Subjects Tested"])
#iedb_df = iedb_df.dropna(subset = ["Assay - Number of Subjects Positive"])
print(nepdb_select_df.shape)

os.system("mkdir -p ../data/training/processed/")
nepdb_select_df.to_csv("../data/training/processed/nepdb_processed01.csv")

BA_cmd = "python3 ../bin/BABS_predict.py --input ../data/training/processed/nepdb_processed01.csv --hla HLA --pep mut_peptide --l label --out ../data/training/BAinfo/nepdb_BA.csv"
#os.system(BA_cmd)