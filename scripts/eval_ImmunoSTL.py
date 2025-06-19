import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STLModel().to(device)

model.load_state_dict(torch.load("../models/ImmunoSTL_r44.pt", map_location="cuda"))  # or "cuda"

def evaluate_stl_model(model, dataset, mhc="MHC", pep="Peptide", label="Label", name="Dataset", pseudo_file_path="../HLA/MHC_pseudo.dat"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = dataset.copy()
    dataset = add_mhc_pseudo_sequence(dataset, mhc_col=mhc, pseudo_file_path="../HLA/MHC_pseudo.dat")

    dataset = dataset.dropna(subset=["HLA_pseudo"]).reset_index(drop=True)

    # Embed
    print(f"[INFO] Embedding {len(dataset)} samples...")
    peptide_emb = esm_embed(dataset[pep].tolist(), model_esm, tokenizer, max_len=11, flatten=False)
    mhc_emb = esm_embed(dataset["HLA_pseudo"].tolist(), model_esm, tokenizer, max_len=34, flatten=False)

    x_pep = torch.tensor(peptide_emb, dtype=torch.float32).to(device)
    x_mhc = torch.tensor(mhc_emb, dtype=torch.float32).to(device)
    y_true = dataset[label].values

    with torch.no_grad():
        preds = model(x_pep, x_mhc).cpu().numpy().flatten()

    os.system("mkdir -p ../pred_results/immunostl")
    df["Predicted Score"] = preds   
    df.to_csv(f"../pred_results/immunostl/{name}.csv", index=False)


benchmark = pd.read_csv("../data/benchmark.csv")
mRNA = pd.read_csv("../data/mRNAvaccine_pID.csv")
zero1 = pd.read_csv("../data/zeroshot_data.csv")
zero2 = pd.read_csv("../data/zeroshot_data2.csv")

# Evaluate on benchmark and mRNA datasets
benchmark_metrics = evaluate_stl_model(
    model=model,
    dataset=benchmark,
    name="BenchmarkSet"  # name used for saving CSV results
)

mRNA_metrics = evaluate_stl_model(
    model=model,
    dataset=mRNA,
    name="mRNA"  # name used for saving CSV results
)

zero1_metrics = evaluate_stl_model(
    model=model,
    dataset=zero1,
    name="zero1"  # name used for saving CSV results
)

zero2_metrics = evaluate_stl_model(
    model=model,
    dataset=zero2,
    name="zero2"  # name used for saving CSV results
)
