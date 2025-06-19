import os
import torch
import joblib
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn


cluster_mapping = pd.read_csv("../HLA/clustering_res.csv").set_index("HLA")["cluster"].to_dict()
cluster_ids = joblib.load("../models/cluster_ids.pkl")

t2_clusters = pd.read_csv("../HLA/t2_cluster_res_c4.csv")
t2_cluster_mapping = t2_clusters.set_index("HLA")["assigned_cluster"].to_dict()

# Load pretrained ESM model
print("[INFO] Loading ESM model...")
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
            #pep_tokens = output[:, 1:12, :]
            output = model_esm(**batch).last_hidden_state  # shape: [batch_size, max_len, 320]
            embeddings.append(output.cpu())
    return torch.cat(embeddings, dim=0)  # shape: [N, 11, 320]

def add_mhc_pseudo_sequence(df, mhc_col='HLA', pseudo_file_path="../HLA/MHC_pseudo.dat"):
    def normalize_mhc_name(name): return name.replace("*", "").replace(":", "")
    mhc_dict = {}
    with open(os.path.expanduser(pseudo_file_path), 'r') as f:
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_tasks = len(cluster_ids) 
model = MultiTaskModel(num_tasks=num_tasks).to(device)

model.load_state_dict(torch.load("../models/ImmunoMTL_r44.pt", map_location="cuda"))  # or "cuda"

def evaluate_dataset(model, dataset, task_id=0, mhc="MHC", pep="Peptide", label="Label", name="Dataset"):
    print(f"[INFO] input shape: {dataset.shape}")
    model.eval()

    pseudo_file_path = "../HLA/MHC_pseudo.dat"

    dataset = add_mhc_pseudo_sequence(dataset, mhc_col=mhc, pseudo_file_path=pseudo_file_path)
    if name in ["mRNA", "manafest", "ebv", "checkmate153"]:
        dataset["cluster"] = dataset[mhc].map(cluster_mapping)
        dataset["cluster"] = dataset["cluster"].fillna(dataset[mhc].map(t2_cluster_mapping))

    dataset = dataset.dropna(subset=["cluster", "HLA_pseudo"]).reset_index(drop=True)

    print("[INFO] Encoding peptides and MHCs...")
    peptide_emb = esm_embed(dataset[pep].tolist(), model_esm, tokenizer, max_len=11, flatten=False)
    mhc_emb = esm_embed(dataset["HLA_pseudo"].tolist(), model_esm, tokenizer, max_len=34, flatten=False)
    
    results_df = pd.DataFrame(columns=["Peptide", "MHC", "MMS_Cluster", "Label", "ImmunoMTL_score"])
    overall_metrics = {"Cluster": [], "N": [], "AUC": [], "AUPRC": [], "F1": [], "Accuracy": []}
    print(f"[INFO] After removing unsupported HLAs: {dataset.shape}")
    print(f"[INFO] Evaluating {name} by branch...")
    for cluster in sorted(dataset["cluster"].unique()):
        cluster_data = dataset[dataset["cluster"] == cluster]
        indices = cluster_data.index.to_list()
        x_pep = peptide_emb[indices].clone().detach().float()
        x_mhc = mhc_emb[indices].clone().detach().float()
        y_true = cluster_data[label].values

        with torch.no_grad():
            all_outputs = model(x_pep.to(device), x_mhc.to(device))  # returns list of all task outputs
            task_id = cluster_ids.index(cluster)  # map cluster to task index
            preds = all_outputs[task_id].cpu().numpy().flatten()
        cluster_results = pd.DataFrame({
            "Peptide": cluster_data[pep].values,
            "MHC": cluster_data[mhc].values,
            "MMS_Cluster": cluster_data["cluster"].values,
            "Label": y_true,
            "ImmunoMTL_score": preds
        })
        results_df = pd.concat([results_df, cluster_results], ignore_index=True)
    os.system("mkdir -p ../pred_results/immunomtl")
    results_df.to_csv(f"../pred_results/immunomtl/{name}.csv", index=False)
    return overall_metrics


benchmark = pd.read_csv("../data/benchmark.csv")
mRNA = pd.read_csv("../data/mRNAvaccine_pID.csv")
zero1 = pd.read_csv("../data/zeroshot_data.csv")
zero2 = pd.read_csv("../data/zeroshot_data2.csv")

# Evaluate on benchmark and mRNA datasets
benchmark_metrics = evaluate_dataset(
    model=model,
    dataset=benchmark,
    name="BenchmarkSet"  # name used for saving CSV results
)

mRNA_metrics = evaluate_dataset(
    model=model,
    dataset=mRNA,
    name="mRNA"  # name used for saving CSV results
)

zero1_metrics = evaluate_dataset(
    model=model,
    dataset=zero1,
    name="zero1"  # name used for saving CSV results
)

zero2_metrics = evaluate_dataset(
    model=model,
    dataset=zero2,
    name="zero2"  # name used for saving CSV results
)
