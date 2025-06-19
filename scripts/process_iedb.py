import pandas as pd
import sys
import os
sys.path.append('../')
from bin.TNinputcheck import checkinput

# NOTE: The raw data processed here was obtained from the Immune Epitope Database (IEDB).
# This file, 'tcell_full_v3.csv', is not included in this repo.
# To use this script, please download the data from the IEDB website at https://www.iedb.org/database_export_v3.php
# After downloading, save the file as 'tcell_full_v3.csv' in the '../data/training/raw/iedb/' directory.

iedb_df = pd.read_csv('../data/training/raw/iedb/tcell_full_v3.csv', header=[0, 1], low_memory=False)
merged_header = [' - '.join(col).strip() for col in iedb_df.columns]
iedb_df.columns = merged_header

#8-11mer peptides
iedb_df = iedb_df[iedb_df["Epitope - Object Type"] == "Linear peptide"]
len_list = []
for epitope in iedb_df["Epitope - Name"]:
    len_list.append(len(epitope))
iedb_df["Epitope - length"] = len_list
iedb_df = iedb_df[(iedb_df["Epitope - length"] >= 8) & (iedb_df["Epitope - length"] <= 11)]

#human as host
iedb_df = iedb_df[iedb_df["Host - Name"].str.contains("Homo")]

#T-cell measurement 
iedb_df = iedb_df[iedb_df["Effector Cell - Name"].str.contains('PBMC|T cell CD8+',regex=True, na=False)]

#MHC type 
iedb_df = iedb_df[iedb_df["MHC Restriction - Class"] == "I"]
iedb_df = iedb_df[iedb_df['MHC Restriction - Name'].str.contains('D')==False]
iedb_df = iedb_df[iedb_df['MHC Restriction - Name'].str.contains('E')==False]
iedb_df = iedb_df[iedb_df['MHC Restriction - Name'].str.contains('o')==False]
iedb_df = iedb_df[iedb_df['MHC Restriction - Name'].str.contains('G')==False]
iedb_df = iedb_df[iedb_df['MHC Restriction - Name'].str.contains(':') ]
iedb_df = iedb_df[iedb_df["MHC Restriction - Name"].apply(lambda x: len(x) == 11)]
iedb_df["nMHC"] = iedb_df["MHC Restriction - Name"].str.replace(r'\*','', regex=True)

iedb_df["pMHC_merge"] = iedb_df["Epitope - Name"] + "_" + iedb_df["MHC Restriction - Name"]
iedb_df = iedb_df.drop_duplicates(subset = ["pMHC_merge"])

def binary_(df):
    eptlist = []
    for i in df["Assay - Qualitative Measurement"]:
        if i == "Negative":
            eptlist.append(0)
        else:
            eptlist.append(1)
    return eptlist

def extract_(df):
    label = binary_(df)
    final_df = pd.DataFrame({"Peptide": df["Epitope - Name"], "MHC" :df["MHC Restriction - Name"], "Len":df["Epitope - length"], "Disease":df["1st in vivo Process - Disease"], 
                            "Label": label, "Qualitative_measurement":df["Assay - Qualitative Measurement"], 
                            "Test_count":df["Assay - Number of Subjects Tested"], "Response_count":df["Assay - Number of Subjects Positive"],
                            "Effector":df["Effector Cell - Name"]})
    return final_df

def input_check(dataset):
    invalidpep_input = checkinput.validate_peptide_sequence(dataset["Peptide"])
    dataset = dataset[~dataset["Peptide"].isin(invalidpep_input)]
    return dataset

extracted_df = extract_(iedb_df)
check_df = input_check(extracted_df)
os.system("mkdir -p ../data/training/processed/")
check_df.to_csv("../data/training/processed/iedb_processed01.csv", index=False)
print("Counts of samples:" + str(check_df.shape[0]))
BA_cmd = "python3 ../bin/BA_predict.py --input ../data/training/processed/iedb_processed01.csv --hla MHC --pep Peptide --l Label --out ../data/training/BAinfo/iedb_BA.csv"
#os.system(BA_cmd)