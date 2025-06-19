# ImmunoMTL

**ImmunoMTL** is a multi-task learning framework for immunogenicity prediction that leverages MHC-specific peptide presentation features.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/AtreeHeart/ImmunoMTL
cd ImmunoMTL
```

### 📆 Environment and Dependencies

The model was trained and evaluated on:

- **GPU**: NVIDIA RTX 3090  
- **OS**: Ubuntu 20.04  
- **Python**: 3.10.15

Essential packages for running the pre-trained MTL model (`predict.py`):

```text
pandas==2.2.3  
numpy==1.26.4  
scikit-learn==1.5.1  
torch==2.5.1  
tqdm==4.67.1  
transformers==4.46.3  
```

Dependencies for additional scripts are noted at the top of each script file.

---

## 🔍 Usage

### Predict with Pretrained Model

```bash
python predict.py --input path/to/input.csv --output path/to/output.csv
```

**Arguments:**

- `--input` : Path to input CSV. Must include a header with two columns:  
  - `Peptide` (e.g., `KTFPPTEPK`)  
  - `MHC` (e.g., `HLA-A*03:01`)  

- `--output` : Output path for predictions in CSV format.

- `--model` *(optional)*: Path to a `.pt` model file. If not provided, default pretrained model is used.

---

## 💠 Additional Scripts

A set of Python scripts are provided to support data preparation, training, and evaluation:

| Script | Description |
|--------|-------------|
| `process_iedb.py`, `process_vdjdb.py`, etc. | Parse and preprocess datasets from public sources |
| `HNpep_generator.py`, `HNpep_merge.py` | Generate and merge high-confidence non-immunogenic peptides from human normal proteome |
| `compile_all_datasets.py` | Compile training / benchmark / zeroshot datasets |
| `ImmunoMTL_training.py` | Train the multi-task learning model |
| `ImmunoMTL_training_shuffle.py` | Train the multi-task learning model using randomly assigned MHC clusters |
| `ImmunoSTL_training.py` | Train a single-task learning baseline model |
| `eval_ImmunoMTL.py`, `eval_metrics.py`, etc. | Evaluate predictions and generate metrics |

---

## Citation

This package has been submitted for peer-reviewed publication
Citation details will be available upon acceptance.


