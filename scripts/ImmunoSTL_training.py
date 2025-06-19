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
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc as sklearn_auc

# === Parameters ===
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MIN_PEP_PER_HLA = 10

# Load data
training_data = pd.read_csv("../data/training.csv") # Curated training data, available on Mendeley
training_data = training_data.drop(columns = ["cluster"])
sufficient_hlas = training_data["MHC"].unique()
HN_negative = pd.read_csv("../data/HN_training.csv") # Curated training data, available on Mendeley
filtered_HN = HN_negative[HN_negative["MHC"].isin(sufficient_hlas)].reset_index(drop=True)

training = pd.concat([training_data, filtered_HN], ignore_index=True)

# === ESM Embedding ===
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model_esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()

def esm_embed(seq, model, tokenizer, max_len=11, flatten=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_esm.to(device)
    embeddings = []

    for i in tqdm(range(0, len(seq), 64)):
        batch = tokenizer(seq[i:i+64], return_tensors="pt", padding="max_length", truncation=True,
                          max_length=max_len, add_special_tokens=False).to(device)
        with torch.no_grad():
            output = model_esm(**batch).last_hidden_state  # shape: [batch_size, max_len, 320]
            embeddings.append(output.cpu())
    return torch.cat(embeddings, dim=0)  # shape: [N, 11, 320]

def add_mhc_pseudo_sequence(df, mhc_col='HLA', pseudo_file_path="../HLA/MHC_pseudo.dat"):
    def normalize_mhc_name(name):
        return name.replace("*", "").replace(":", "")

    # Read and parse the pseudo sequence file
    mhc_dict = {}
    with open(os.path.expanduser(pseudo_file_path), 'r') as f: #the pseudo_file_path can be obtained from netMHCpan
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mhc, seq = parts
                mhc_dict[mhc] = seq

    def lookup_pseudo(mhc):
        norm = normalize_mhc_name(mhc)
        return mhc_dict.get(norm, None)

    df = df.copy()
    df['HLA_pseudo'] = df[mhc_col].apply(lookup_pseudo)

    missing = df['HLA_pseudo'].isna().sum()
    print(f"[INFO] Added 'HLA_pseudo' column. Missing sequences for {missing} entries.")

    return df

def embed_unique_mhc(filtered_data, model_esm, tokenizer, max_len=34):
    # Extract and embed unique HLA pseudo sequences
    unique_hlas = filtered_data['HLA_pseudo'].dropna().unique().tolist()
    print(f"[INFO] Embedding {len(unique_hlas)} unique HLA pseudo sequences...")
    hla_embeddings = esm_embed(unique_hlas, model_esm, tokenizer, max_len=max_len, flatten=False).numpy()

    hla_to_embedding = dict(zip(unique_hlas, hla_embeddings))
    filtered_data["MHC_Encoded"] = filtered_data["HLA_pseudo"].map(lambda x: hla_to_embedding.get(x))

    print("[INFO] Finished embedding and mapping back to DataFrame.")
    return filtered_data

#=== pre-processing training data ===
training = add_mhc_pseudo_sequence(training, mhc_col='MHC', pseudo_file_path="../HLA/MHC_pseudo.dat")
training["Peptide_Encoded"] = list(
    esm_embed(training["Peptide"].tolist(), model_esm, tokenizer, max_len=11, flatten=False).numpy()
)
training = embed_unique_mhc(training, model_esm, tokenizer, max_len=34)


class PepMHC_Dataset(Dataset):
    def __init__(self, peptides, mhcs, labels):
        self.peptides = torch.tensor(peptides, dtype=torch.float32)
        self.mhcs = torch.tensor(mhcs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.peptides[idx], self.mhcs[idx], self.labels[idx]

class STLModel(nn.Module):
    def __init__(self):
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
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x_pep, x_mhc):
        pep_out, _ = self.peptide_lstm(x_pep)
        pep_out, _ = self.peptide_lstm2(pep_out)
        pep_feat = pep_out[:, -1, :]
        mhc_out, _ = self.mhc_lstm(x_mhc)
        mhc_out, _ = self.mhc_lstm2(mhc_out)
        mhc_feat = mhc_out[:, -1, :]
        combined = torch.cat([pep_feat, mhc_feat], dim=1)
        return self.shared(combined).squeeze(-1)

def train_stl_model(df, seed=42, save_path="stl_model.pt", epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stratified train/val split
    df["strata"] = df["MHC"] + "_" + df["Label"].astype(str)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["strata"], random_state=seed)

    train_ds = PepMHC_Dataset(np.stack(train_df["Peptide_Encoded"]), np.stack(train_df["MHC_Encoded"]), train_df["Label"].values)
    val_ds = PepMHC_Dataset(np.stack(val_df["Peptide_Encoded"]), np.stack(val_df["MHC_Encoded"]), val_df["Label"].values)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = STLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for x_pep, x_mhc, y in train_loader:
            x_pep, x_mhc, y = x_pep.to(device), x_mhc.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_pep, x_mhc)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_pep, x_mhc, y in val_loader:
                x_pep, x_mhc = x_pep.to(device), x_mhc.to(device)
                preds = model(x_pep, x_mhc).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_loss = criterion(torch.tensor(all_preds), torch.tensor(all_labels)).item()

        try:
            auc = roc_auc_score(all_labels, all_preds)
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            auprc = sklearn_auc(recall, precision)
            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
        except:
            auc, auprc, f1 = np.nan, np.nan, np.nan

        print(f"[Epoch {epoch+1}] Loss: {val_loss:.4f} | AUC: {auc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Best model saved to {save_path}")

    return model


# === Call Training ===
trained_model = train_stl_model(
    training,
    seed=44,
    epochs=50,
    save_path=f"../models/STL_model_r44.pt"
)

