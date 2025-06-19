# === Development Environment ===
"""
Python version:       3.10.15 
pandas version:       2.2.3
numpy version:        1.26.4
scikit-learn version: 1.5.1
torch version: 2.5.1
tqdm version: 4.67.1
transformer version: 4.46.3
"""

# === Imports ===
import os
import torch
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

def esm_embed(seq, model, tokenizer, max_len=11, flatten=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = []

    for i in tqdm(range(0, len(seq), 64)):
        # Add_special_tokens=False ensures no [CLS]/[SEP]
        batch = tokenizer(seq[i:i+64], return_tensors="pt", padding="max_length", truncation=True,
                          max_length=max_len, add_special_tokens=False).to(device)
        with torch.no_grad():
            #pep_tokens = output[:, 1:12, :]
            output = model(**batch).last_hidden_state  # shape: [batch_size, max_len, 320]
            embeddings.append(output.cpu())
    return torch.cat(embeddings, dim=0)  # shape: [N, 11, 320]

def add_mhc_pseudo_sequence(df, mhc_col='HLA', pseudo_file_path="HLA/MHC_pseudo.dat"):
    def normalize_mhc_name(name): return name.replace("*", "").replace(":", "")
    mhc_dict = {}
    if not os.path.exists(pseudo_file_path):
        raise FileNotFoundError(
            "[ERROR] MHC pseudo sequence file not found.\n"
            "Please install netMHCpan and copy 'MHC_pseudo.dat' to 'HLA/'"
        )
    with open(os.path.expanduser(pseudo_file_path), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mhc, seq = parts
                mhc_dict[mhc] = seq

    # Create the new column with mapped pseudo sequences
    def lookup_pseudo(mhc):
        norm = normalize_mhc_name(mhc)
        return mhc_dict.get(norm, None)

    df = df.copy()
    df['HLA_pseudo'] = df[mhc_col].apply(lookup_pseudo)

    missing = df['HLA_pseudo'].isna().sum()
    print(f"[INFO] Added 'HLA_pseudo' column. Missing sequences for {missing} entries.")

    return df

class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.peptide_lstm = nn.LSTM(input_size=320, hidden_size=64, batch_first=True, bidirectional=True)
        self.peptide_lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.mhc_lstm = nn.LSTM(input_size=320, hidden_size=64, batch_first=True, bidirectional=True)
        self.mhc_lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.shared = nn.Sequential(
            nn.Linear(128 + 128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()) for _ in range(num_tasks)])

    def forward(self, x_pep, x_mhc):
        pep_lstm_out, _ = self.peptide_lstm(x_pep)
        pep_lstm_out, _ = self.peptide_lstm2(pep_lstm_out)
        pep_feat = pep_lstm_out[:, -1, :]
        mhc_lstm_out, _ = self.mhc_lstm(x_mhc)
        mhc_lstm_out, _ = self.mhc_lstm2(mhc_lstm_out)
        mhc_feat = mhc_lstm_out[:, -1, :]
        merged = torch.cat([pep_feat, mhc_feat], dim=1)
        shared = self.shared(merged)
        return [head(shared).squeeze(-1) for head in self.heads]

# ==== Main Prediction Function ====
def run_prediction(input_csv, output_csv, model_path, cluster_mapping, t2_cluster_mapping, cluster_ids):

    # === Load trained model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "models/ImmunoMTL_r44.pt"
    print(f"[INFO] Loading model from {model_path}")
    model = MultiTaskModel(num_tasks=len(cluster_ids)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load input
    df = pd.read_csv(input_csv)
    peptide_col = df.columns[0]
    mhc_col = df.columns[1]
    df = add_mhc_pseudo_sequence(df, mhc_col=mhc_col)

    # Add cluster and t2 annotations
    df["cluster"] = df[mhc_col].map(cluster_mapping)
    df["t2_cluster"] = df[mhc_col].map(t2_cluster_mapping)  
    # Assign final cluster
    df["Final_Cluster"] = df["cluster"].fillna(df["t2_cluster"])

    # Add notes
    df["Note"] = ""
    df.loc[df["cluster"].isna() & df["t2_cluster"].notna(), "Note"] = "Assigned from T2 cluster (9-mer mass spectrometry hits only; lower confidence)"
    df.loc[df["Final_Cluster"].isna(), "Note"] = "Unsupported HLA"

    # Warn unsupported and fallback HLAs
    unsupported_hlas = df[df["Final_Cluster"].isna()][mhc_col].value_counts()
    t2_only_hlas = df[(df["cluster"].isna()) & (df["t2_cluster"].notna())][mhc_col].value_counts()

    if not unsupported_hlas.empty:
        print("[WARNING] Unsupported HLA types (not in T1 or T2 clusters):")
        print(unsupported_hlas.to_string())

    if not t2_only_hlas.empty:
        print("[NOTE] HLA types predicted using T2 cluster only (9-mer data):")
        print(t2_only_hlas.to_string())

    # Filter out rows without cluster
    df_pred = df[df["Final_Cluster"].notna()].copy().reset_index(drop=True)

    # Load ESM model
    print("[INFO] Loading ESM2 model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model.to(device)

    # Embed sequences
    print("[INFO] Embedding peptide sequences...")
    pep_emb = esm_embed(df_pred[peptide_col].tolist(), esm_model, tokenizer, max_len=11).float()
    print("[INFO] Embedding MHC pseudo sequences...")
    mhc_emb = esm_embed(df_pred["HLA_pseudo"].tolist(), esm_model, tokenizer, max_len=34).float()

    # Load prediction model
    print("[INFO] Loading trained ImmunoMTL model...")
    model = MultiTaskModel(num_tasks=len(cluster_ids)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run predictions
    print("[INFO] Running predictions...")
    predictions = []
    for cluster in sorted(df_pred["Final_Cluster"].unique()):
        idxs = df_pred[df_pred["Final_Cluster"] == cluster].index
        x_pep = pep_emb[idxs].to(device)
        x_mhc = mhc_emb[idxs].to(device)
        with torch.no_grad():
            outputs = model(x_pep, x_mhc)
            task_id = cluster_ids.index(cluster)
            preds = outputs[task_id].cpu().numpy()
        predictions.extend(zip(idxs, preds))

    # Merge predictions
    pred_df = pd.DataFrame(predictions, columns=["index", "ImmunoMTL_score"]).set_index("index")
    df_pred = df_pred.join(pred_df).sort_index()

    # Recombine full table with missing rows retained
    unsupported_rows = df[df["Final_Cluster"].isna()].copy()
    unsupported_rows["ImmunoMTL_score"] = np.nan  # or "" if preferred
    df_final = pd.concat([df_pred, unsupported_rows], ignore_index=True, sort=False)

    df_final = df_final.drop(columns=["cluster", "t2_cluster", "HLA_pseudo"])
    df_final.to_csv(output_csv, index=False)
    print(f"[INFO] Saved predictions to: {output_csv}")

# ==== CLI Interface ====
def main():
    parser = argparse.ArgumentParser(description="Run ImmunoMTL prediction on input CSV")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to input CSV with the first column as peptide and the second column as MHC. Example: KTFPPTEPK,HLA-A*03:01. The first row should contain the header."
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save prediction CSV")
    parser.add_argument("--model", type=str, default="models/ImmunoMTL_r44.pt", help="Path to trained model weights (.pt file)")    
    args = parser.parse_args()

    cluster_mapping = pd.read_csv("HLA/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
    t2_cluster_mapping = pd.read_csv("HLA/t2_cluster_res_c4.csv").set_index("HLA")["assigned_cluster"].to_dict()
    cluster_ids = joblib.load("models/cluster_ids.pkl")

    run_prediction(args.input, args.output, args.model, cluster_mapping, t2_cluster_mapping, cluster_ids)

if __name__ == "__main__":
    main()