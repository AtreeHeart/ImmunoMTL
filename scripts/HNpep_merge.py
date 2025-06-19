import pandas as pd
import glob
from tqdm import tqdm

hla_cluster = pd.read_csv("../HLA/clustering_res.csv")

# Negative data processing
N_df_withC = pd.DataFrame()
file_list = sorted(glob.glob("../data/HN_pepbyHLA/*.csv"))

for file in tqdm(file_list):
    N_df = pd.read_csv(file)
    
    # Skip empty or bad files
    if "Peptide" not in N_df.columns or "HLA" not in N_df.columns:
        print(f"[WARN] Skipping {file} due to missing Peptide or HLA column.")
        continue

    N_df["HLA_merge"] = N_df["Peptide"] + N_df["HLA"]
    N_df = N_df.drop_duplicates("HLA_merge")

    # Format HLA names (A0201 → A*02:01)
    N_df["HLA"] = N_df["HLA"].apply(lambda h: h[:5] + "*" + h[5:])

    # Join with cluster info
    N_df_withC_tmp = pd.merge(N_df, hla_cluster, on="HLA", how="inner")
    N_df_withC = pd.concat([N_df_withC, N_df_withC_tmp], ignore_index=True)

# Final formatted output
Final_N_df = pd.DataFrame({
    "Peptide": N_df_withC["Peptide"],
    "MHC": N_df_withC["HLA"],
    "MHC_cluster": N_df_withC["cluster"],
    "Label": 0,
    "mhcflurry_present_score": N_df_withC["presentation_score"],
    "mhcflurry_present_percent": N_df_withC["presentation_percentile"],
    "EL-score": N_df_withC["EL_Score"],
    "EL_Rank": N_df_withC["EL_Rank"]
})

Final_N_df.to_csv("../data/HN_training.csv", index=False)
