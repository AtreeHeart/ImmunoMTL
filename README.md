# ImmunoMTL

**ImmunoMTL** is a multi-task learning framework for immunogenicity prediction that leverages MHC-specific peptide presentation features.

## Installation
Clone the repository:

```bash
git clone https://github.com/AtreeHeart/ImmunoMTL
cd ImmunoMTL
```

### Environment and Dependencies

The model was trained and evaluated on:

- **GPU**: NVIDIA RTX 3090  
- **OS**: Ubuntu 20.04  
- **Python**: 3.10.15

Essential packages for running the pre-trained MTL model (`predict.py`):

```bash
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.1 \
            torch==2.5.1 tqdm==4.67.1 transformers==4.46.3
```

Dependencies for additional scripts are noted at the top of each script file.

## Usage
### Predict with Pretrained Model

```bash
python predict.py --input path/to/input.csv --output path/to/output.csv
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to input CSV. Must contain columns `Peptide` (e.g., `KTFPPTEPK`) and `MHC` (e.g., `HLA-A*03:01`). |
| `--output` | Yes | Output path for predictions in CSV format. |
| `--model` | No | Path to a `.pt` model weights file. Defaults to `models/ImmunoMTL_s22.pt`. |
| `--hla_dir` | No | Directory containing HLA resource files. Defaults to `HLA/`. |
| `--label` | No | Column name in the input CSV with 0/1 ground-truth labels. When provided, AUROC, AP, and PPVn are reported. |

### Additional Scripts

A set of Python scripts are provided in `scripts/` to support data preparation, training, and evaluation:

| Script | Description |
|--------|-------------|
| `scripts/process_iedb.py`, `scripts/process_vdjdb.py`, etc. | Parse and preprocess datasets from public sources |
| `scripts/HNpep_generator.py`, `scripts/HNpep_merge.py` | Generate and merge high-confidence non-immunogenic peptides from human normal proteome |
| `scripts/compile_all_datasets.py` | Compile training / benchmark / zeroshot datasets |
| `scripts/train_ImmunoMTL.py` | Train the multi-task learning model |
| `scripts/train_ablations.py` | Train ablation variants (STL, shuffled-cluster, joint-STL) |
| `scripts/eval_MTL.py`, `scripts/eval_metrics.py`, etc. | Evaluate predictions and generate metrics |


## Citation

This package has been submitted for peer-reviewed publication.

Citation details will be available upon acceptance.


