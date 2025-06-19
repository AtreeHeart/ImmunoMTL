import pandas as pd
import os

# NOTE: The raw data processed here was obtained from the VDJ database (VDJdb).
# This file, 'SearchTable-2024-01-22 15_05_36.758.tsv', is not included in this repo.
# To use this script, please download the data from the VDJdb website at https://vdjdb.cdr3.net/
# After downloading, save the file as 'SearchTable-2024-01-22 15_05_36.758.tsv' in the '../data/validation/raw/' directory.

vdj = pd.read_csv("../data/validation/raw/SearchTable-2024-01-22 15_05_36.758.tsv", sep="\t")

peplen = []
nmhc =[]
mhclen = []
for pep in vdj["Epitope"]:
    peplen.append(len(pep))
    
def cut_hla_string(hla):
    parts = hla.split(':')
    return ':'.join(parts[:2])

for mhc in vdj["MHC A"]:
    if len(mhc) >12:
        mhc = cut_hla_string(mhc)
    nmhc.append(mhc)
    mhclen.append(len(mhc))
vdj["Pep_len"] = peplen
vdj["nMHC"] = nmhc
vdj["MHC_len"] = mhclen
vdj["pMHC"] = vdj["Epitope"] + "_" +vdj["nMHC"]

vdj = vdj[(vdj["Species"] == "HomoSapiens") & (vdj["Pep_len"] <=11) & (vdj["Pep_len"] >=8) & (vdj["MHC_len"] != 8) & (vdj["MHC_len"]!=12)]
vdj = vdj.drop_duplicates(subset = ["pMHC"])
vdjdf = pd.DataFrame({"Epitope": vdj["Epitope"], "MHC": vdj["nMHC"], "Label": 1})
vdjdf.reset_index(drop = True, inplace = True)
print(vdjdf.shape)

os.system("mkdir -p ../data/validation/processed/")
vdjdf.to_csv("../data/validation/processed/vdj_processed01.csv", index = False)
BA_cmd = "python3 ../bin/BABS_predict.py --input ../data/validation/processed/vdj_processed01.csv --hla MHC --pep Epitope --l Label --out ../data/validation/BAinfo/vdjdb_BA.csv"
#os.system(BA_cmd)