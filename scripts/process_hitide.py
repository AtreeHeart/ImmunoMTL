import pandas as pd
import os
# NOTE: The raw data processed here was obtained from the paper "Machine learning methods and harmonized datasets improve immunogenic neoantigen prediction".
# This file, 'HiTIDE.txt', is not included in this repo.
# To use this script, please download the data from the original journal
# After downloading, save the file as 'HiTIDE.txt' in the '../data/training/raw/hitide/' directory.


df = pd.read_csv("../data/training/raw/hitide/HiTIDE.txt", sep = "\t", low_memory=False)

hidite_df = df[(df["dataset"] == "HiTIDE") & (df["response_type"] != "not_tested")]
hidite_df = hidite_df[(hidite_df["seq_len"] >= 8) & (hidite_df["seq_len"] <= 11)]

hidite_df['responseN'] = hidite_df['response_type'].apply(lambda x: '1' if x == "CD8" else '0')

# Define a function to convert HLAs to comprehensive form and select the first one if multiple HLAs are present.
def convert_hla(hla_string):
    # Split the string by space and take the first element (in case there are multiple HLAs)
    first_hla = hla_string.split(",")[0]
    # Insert ":" between the allele group and the protein sequence
    allele_group, protein_sequence = first_hla[1:3], first_hla[3:]
    # Convert to comprehensive form with ":"
    return f'HLA-{first_hla[0]}*{allele_group}:{protein_sequence}'

# Apply the conversion function to the 'HLAs' column
hidite_df['MHC'] = hidite_df['mutant_best_alleles'].apply(convert_hla)

hidite_f_df = pd.DataFrame({"Patient":hidite_df.patient,"OriginalIndex":hidite_df.index,"CancerType":hidite_df.Cancer_Type,"Neoantigen":hidite_df.mutant_seq,"MHC":hidite_df.MHC, "Label":hidite_df.responseN})
print(hidite_f_df.shape)

os.system("mkdir -p ../data/training/processed/")
hidite_f_df.to_csv("../data/training/processed/hitide_processed01.csv", index = False)
BA_cmd = "python3 ../bin/BABS_predict.py --input ../data/training/processed/hitide_processed01.csv --hla MHC --pep Neoantigen --l Label --out ../data/training/BAinfo/hitide_BA.csv"
#os.system(BA_cmd)