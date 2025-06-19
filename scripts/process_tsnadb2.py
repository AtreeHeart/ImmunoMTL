import pandas as pd
import os

# NOTE: The raw data processed here was obtained from the tumor-specific neoantigen database (TSNADB 2.0).
# This file, 'validated_tsnadb2_download.txt', is not included in this repo.
# To use this script, please download the data from the TSNADB 2.0 website at https://pgx.zju.edu.cn/tsnadb/
# After downloading, save the file as 'validated_tsnadb2_download.txt' in the '../data/training/raw/tsnadb2/' directory.

tsnadb2 = pd.read_csv("../data/training/raw/tsnadb2/validated_tsnadb2_download.txt", sep="\t")
tsnadb2
#select tier 1 & 2 only
#According to original paper:
#Tier 1: Neoantigens validated for both immunogenicity and presentation on the cell surface.
#Tier 2: Neoantigens validated only for immunogenicity.
#Tier 3: Neoantigens validated only for presentation on the cell surface.

tsnadb2  = tsnadb2[tsnadb2["Level"]!="tier3"]
print(tsnadb2.groupby("Level").count())
tsnadb2_select = ["Tumor Type detaile", "Gene", "HLA", "Mutant Peptide"]
tsnadb2_select_df = tsnadb2.loc[:, tsnadb2_select].copy()
tsnadb2_select_df.dropna(inplace = True)
tsnadb2_select_df.loc[:, "HLA"] = "HLA-" + tsnadb2_select_df["HLA"]

allelename = []
for i in tsnadb2_select_df["HLA"]:
    allelename.append(i[:5] + "*" + i[5:])
    
tsnadb2_select_df.loc[:, "HLA"] = allelename
tsnadb2_select_df

def add_merge_col(input_df, db_name):
    input_df.columns = ["Disease", "Gene/Antigen", "MHC_allele", "peptide"]
    input_df["pMHC_merge"] = input_df["peptide"] + "_" + input_df["MHC_allele"]
    db_str = "s_" + db_name
    #input_df[db_str] = 1
    input_df = input_df[input_df['MHC_allele'].str.contains('D')==False]
    input_df = input_df[input_df['MHC_allele'].str.contains('E')==False]
    input_df = input_df[input_df['MHC_allele'].str.contains('o')==False]
    input_df = input_df[input_df['MHC_allele'].str.contains('G')==False]
    input_df = input_df[input_df['MHC_allele'].str.contains(':') ]
    pep_len = []
    for i in input_df["peptide"]:
        pep_len.append(len(i))
    input_df["len"] = pep_len
    input_df = input_df[(input_df["len"]>7) & (input_df["len"]<12)]

    return input_df

tsnadb2_final = add_merge_col(tsnadb2_select_df, "tsnadb2")
tsnadb2_final["Label"] = 1
tsnadb2_final

os.system("mkdir -p ../data/training/processed/")
tsnadb2_final.to_csv("../data/training/processed/tsnadb2_processed02.csv")
BA_cmd = "python3 ../bin/BABS_predict.py --input ../data/training/processed/tsnadb2_processed02.csv --hla MHC_allele --pep peptide --l Label --out ../data/training/BAinfo/tsnadb2_BA02.csv"
#os.system(BA_cmd)