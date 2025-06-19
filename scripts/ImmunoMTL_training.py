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
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score, auc as sklearn_auc


# === Parameters ===
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Load data
training_data = pd.read_csv("../data/training.csv") # Curated training data, available on Mendeley
training_data = training_data.drop(columns = ["cluster"])
sufficient_hlas = training_data["MHC"].unique()
HN_negative = pd.read_csv("../data/HN_training.csv") # Human wild type peptides, available on Mendeley
filtered_HN = HN_negative[HN_negative["MHC"].isin(sufficient_hlas)].reset_index(drop=True)

training = pd.concat([training_data, filtered_HN], ignore_index=True)

hla_clusters = pd.read_csv("../HLA/clustering_res.csv")
cluster_mapping = hla_clusters.set_index('HLA')['cluster'].to_dict()
training['cluster'] = training['MHC'].map(cluster_mapping)

cluster_ids = sorted(training['cluster'].unique())

import joblib
joblib.dump(cluster_ids, "../models/cluster_ids.pkl")

# === ESM Embedding ===
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model_esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()

def esm_embed(seq, model, tokenizer, max_len=11):
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

def embed_unique_mhc(df, model_esm, tokenizer, max_len=34):
    # Extract and embed unique HLA pseudo sequences
    unique_hlas = df['HLA_pseudo'].dropna().unique().tolist()
    print(f"[INFO] Embedding {len(unique_hlas)} unique HLA pseudo sequences...")
    hla_embeddings = esm_embed(unique_hlas, model_esm, tokenizer, max_len=max_len).numpy()

    hla_to_embedding = dict(zip(unique_hlas, hla_embeddings))
    df["MHC_Encoded"] = df["HLA_pseudo"].map(lambda x: hla_to_embedding.get(x))

    print("[INFO] Finished embedding and mapping back to DataFrame.")
    return df

#=== pre-processing training data ===
training = add_mhc_pseudo_sequence(training, mhc_col='MHC', pseudo_file_path="../HLA/MHC_pseudo.dat")
training["Peptide_Encoded"] = list(
    esm_embed(training["Peptide"].tolist(), model_esm, tokenizer, max_len=11).numpy()
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

def train_mtl_model(df, seed=44, save_path="mtl_unified_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_tasks = len(cluster_ids)
    model = MultiTaskModel(num_tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    task_weights = [1.5 if i == 0 else 1.0 for i in range(num_tasks)]
    best_loss = float('inf')
    task_dataloaders = []
    val_loss_log = []

    print("\n[INFO] Sample counts per cluster:")
    for i, cluster in enumerate(cluster_ids):
        cdata = df[df['cluster'] == cluster]
        print(f"Cluster {cluster} (Task {i+1}): {len(cdata)} samples")

        balanced = []
        for mhc in cdata['MHC'].unique():
            mhc_data = cdata[cdata['MHC'] == mhc]
            pos = mhc_data[mhc_data['Label'] == 1]
            neg = mhc_data[mhc_data['Label'] == 0]
            if len(pos) and len(neg) >= len(pos) * 2:
                balanced.append(pd.concat([
                    pos.sample(len(pos), random_state=seed),
                    neg.sample(len(pos) * 2, random_state=seed)
                ]))
            elif len(pos) and len(neg):
                n = min(len(pos), len(neg))
                balanced.append(pd.concat([
                    pos.sample(n, random_state=seed),
                    neg.sample(n, random_state=seed)
                ]))

        if balanced:
            task_df = pd.concat(balanced).reset_index(drop=True)
            task_df['strata'] = task_df['MHC'] + "_" + task_df['Label'].astype(str)
            train_df, val_df = train_test_split(task_df, test_size=0.15, stratify=task_df['strata'], random_state=seed)

            train_ds = PepMHC_Dataset(np.stack(train_df["Peptide_Encoded"]),
                                    np.stack(train_df["MHC_Encoded"]),
                                    train_df["Label"].values)
            val_ds = PepMHC_Dataset(np.stack(val_df["Peptide_Encoded"]),
                                    np.stack(val_df["MHC_Encoded"]),
                                    val_df["Label"].values)

            task_dataloaders.append((
                DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
                DataLoader(val_ds, batch_size=BATCH_SIZE),
                i
            ))

    for epoch in range(EPOCHS):
        model.train()
        for train_loader, _, task_id in task_dataloaders:
            for x_pep, x_mhc, y in train_loader:
                x_pep, x_mhc, y = x_pep.to(device), x_mhc.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x_pep, x_mhc)
                loss = task_weights[task_id] * criterion(outputs[task_id], y)
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        print(f"\n[Epoch {epoch+1}] Validation Metrics by Task")
        with torch.no_grad():
            for _, val_loader, task_id in task_dataloaders:
                all_preds, all_labels = [], []
                for x_pep, x_mhc, y in val_loader:
                    x_pep, x_mhc = x_pep.to(device), x_mhc.to(device)
                    preds = model(x_pep, x_mhc)[task_id].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())

                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                loss = criterion(torch.tensor(all_preds, device=device), torch.tensor(all_labels, device=device)).item()
                val_losses.append(loss)

                try:
                    auc = roc_auc_score(all_labels, all_preds)
                    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
                    auprc = sklearn_auc(recall, precision)
                    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
                except:
                    auc, auprc, f1 = np.nan, np.nan, np.nan

                print(f"  Task {task_id+1} | AUC: {auc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f} | Loss: {loss:.4f}")

                val_loss_log.append({
                    "Epoch": epoch + 1,
                    "Task": task_id + 1,
                    "Val_Loss": loss,
                    "AUC": auc,
                    "AUPRC": auprc,
                    "F1": f1
                })

        avg_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print("[INFO] Best model saved.")

    final_save_path = save_path.replace(".pt", "_last.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"[INFO] Final model saved to {final_save_path}")

    val_loss_df = pd.DataFrame(val_loss_log)
    val_loss_df.to_csv(f"../models/loss_round{seed}.csv", index=False)

    return model

# === Call Training ===
trained_model = train_mtl_model(
training, 
seed=44, 
save_path=f"../models/best_model_round44.pt"
)
