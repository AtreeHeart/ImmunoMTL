# Scripts

This folder contains all scripts for reproducing ImmunoMTL — from raw data processing through model training and evaluation. The sections below follow the execution order.

---

## 1. Training Data Preparation

These scripts process raw immunogenicity databases into standardised CSVs. Raw files are **not** bundled in this repo; each script's header lists the source URL and the expected local path.

| Script | Source database | Output |
|--------|----------------|--------|
| `process_iedb.py` | [IEDB](https://www.iedb.org/database_export_v3.php) (`tcell_full_v3.csv`) | `data/training/processed/iedb_processed01.csv` |
| `process_hitide.py` | HiTIDE (Markus Müller et al. Immunity, 2023) (`HiTIDE.txt`) | `data/training/processed/hitide_processed01.csv` |
| `process_nepdb.py` | [NEPdb](https://nep.whu.edu.cn/) (`NECID_Query.csv`) | `data/training/processed/nepdb_processed01.csv` |
| `process_tsnadb2.py` | [TSNADB 2.0](https://pgx.zju.edu.cn/tsnadb/) (`validated_tsnadb2_download.txt`) | `data/training/processed/tsnadb2_processed01.csv` |
| `process_vdjdb.py` | [VDJdb](https://vdjdb.cdr3.net/) (`SearchTable-*.tsv`) | `data/validation/processed/vdj_processed01.csv` |

Each script filters for HLA class I, human CD8⁺ T-cell assays, and 8–11-mer peptides, then adds MHCflurry/NetMHCpan binding scores via `bin/BABS_predict.py`.

---

## 2. Human Negative Generation

Human negatives are self-peptides derived from the normal human proteome that bind HLA but are non-immunogenic. They are generated in two steps.

**Step 1 — Screen the human proteome for HLA-binding peptides:**
```
python HNpep_generator.py --hla ../HLA/clustering_res.csv
```
Reads the human proteome FASTA from UniProt. For each HLA allele, all 8–11-mer peptides are extracted and scored by MHCflurry (presentation percentile ≤ 2%) and NetMHCpan 4.1 (EL_Rank ≤ 2). Only peptides passing both thresholds are retained, ensuring candidates are genuine HLA binders. Requires MHCflurry ≥ 2.0 and NetMHCpan 4.1 on `PATH`. Outputs one CSV per allele in `data/HN_pepbyHLA/`.

**Step 2 — Merge into training-ready format:**
```
python HNpep_merge.py
```
Joins all per-HLA files, deduplicates, reformats allele names (`A0201 → A*02:01`), and appends MMS cluster assignments from `HLA/clustering_res.csv`. Output is used as the negative set in `compile_all_datasets.py`.

---
---

## 3. Dataset Compilation

```
python compile_all_datasets.py
```
Merges all processed sources (IEDB, HiTIDE, NEPdb, TSNADB2) with the human negatives and VDJdb validation set into the final training and benchmark splits. Produces the CSVs under `data/` that downstream training scripts read.

---

## 4. Model Training

All training scripts use fixed seed 22 for the man model (or seeds 1–10 for the Shuffle ablation). Hyperparameters match the paper: ESM2-t12-35M backbone (frozen), dual BiLSTM (h=64) × 2, shared FC (256→64→16), positional-weight loss pw=4.0, 100 epochs with CosineAnnealingLR, EL_Rank ≤ 2.0 hard-negative filter.

### Main model
```
python ImmunoMTL_training.py
```
Trains the published ImmunoMTL model (seed=22, 4 MMS-cluster heads). Saves checkpoint to `models/ImmunoMTL_s22.pt`.

### Ablation models
Run in any order — each is independent.

```
python train_STL.py          # Single-task (1 shared head, no cluster routing)
python train_ABC.py          # HLA-gene routing (3 heads: A / B / C)
python train_JointSTL.py     # JointSTL baseline (pep+MHC concatenated, single BiLSTM stream)
```

Checkpoints: `models/ImmunoSTL_s22.pt`, `models/ImmunoABC_s22.pt`, `models/JointSTL_s22.pt`.

### Shuffle ablation (10 seeds)
```
python ImmunoMTL_shuffle_training.py
```
Trains 10 models (seeds 1–10) with randomly permuted HLA-to-cluster assignment (cluster-assignment seed fixed at cs=10). Saves `models/ImmunoShuffle_s{1..10}_cs10.pt`.

---

## 5. Inference & Evaluation

Each `eval_*.py` script loads the corresponding checkpoint, generates predictions on the Benchmark dataset, saves CSVs to `pred_results/`, and prints AUROC/AP.

| Script | Checkpoint | Output directory |
|--------|-----------|-----------------|
| `eval_MTL.py` | `ImmunoMTL_s22.pt` | `pred_results/MTL/` |
| `eval_STL.py` | `ImmunoSTL_s22.pt` | `pred_results/STL_s22/` |
| `eval_ABC.py` | `ImmunoABC_s22.pt` | `pred_results/ABC_s22/` |
| `eval_JointSTL.py` | `JointSTL_s22.pt` | `pred_results/JointSTL_s22/` |
| `eval_shuffle.py` | `ImmunoShuffle_s{1..10}_cs10.pt` | `pred_results/immunomtl_shuffle_s1-10/` |

Run them individually, then evaluate all ablation results at once:
```
python eval_ablations.py
```
Reads saved prediction CSVs from each model's `pred_results/` directory and prints a consolidated AUROC / AP  table. Shuffle results are reported as mean ± std across 10 seeds.

---


## Quick-start (full pipeline)

```bash
# 1. Download raw databases and place under data/training/raw/ (see scripts above)
python process_iedb.py
python process_hitide.py
python process_nepdb.py
python process_tsnadb2.py
python process_vdjdb.py

# 2. Generate and merge hard negatives
python HNpep_generator.py
python HNpep_merge.py

# 3. Compile final dataset
python compile_all_datasets.py

# 4. Train
python ImmunoMTL_training.py        # main model
python train_STL.py                 # ablations (run in parallel if needed)
python train_ABC.py
python train_JointSTL.py
python ImmunoMTL_shuffle_training.py

# 5. Evaluate
python eval_MTL.py
python eval_STL.py
python eval_ABC.py
python eval_JointSTL.py
python eval_shuffle.py
python eval_ablations.py            # consolidated summary table
```

To run inference only (pre-trained checkpoint provided), skip steps 1–4 and use `predict.py` in the repo root:
```bash
python predict.py --input your_data.csv --output predictions.csv
```
