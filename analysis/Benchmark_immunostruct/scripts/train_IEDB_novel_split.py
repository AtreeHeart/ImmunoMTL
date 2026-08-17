"""
train_IEDB_novel_split.py
Copy of train_IEDB_wFT.py with one addition: --custom-split-file.

When --custom-split-file is provided, the random 80/10/10 split is replaced
by Subsets whose indices come from a JSON file produced by prepare_novel_split.py.
The JSON must have keys: "train", "val", "test" (lists of int indices).

Without --custom-split-file this script is identical to train_IEDB_wFT.py.
"""

import os
import json
import argparse
import torch
import wandb
from dgl.dataloading import GraphDataLoader

from data_loading import ImmunoPredDataset, collate, SplitDataset, collate_amino_acid
from models.mapping import model_map
from utils import Losses, seed_everything, update_paths
from procedures import inference, train_model, inference_SSL, train_model_SSL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry point.")
    parser.add_argument("--model", default="StructureModel", type=str)
    parser.add_argument("--learning-rate-pretrain", default=1e-3, type=float)
    parser.add_argument("--learning-rate-finetune", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=150, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--full-sequence", action="store_true")
    parser.add_argument("--sequence-loss", action="store_true")
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--model-save-dir", default="$ROOT/results/PropIEDB_ImmunoIEDB_NovelSplit/", type=str)
    parser.add_argument("--graph-dir-IEDB", default="$ROOT/data/graph_pyg_IEDB/", type=str)
    parser.add_argument("--property-path-IEDB", default="$ROOT/data/ImmunoStruct_IEDB_data.csv", type=str)
    parser.add_argument("--hla-path", default="$ROOT/data/HLA_allele_sequences.csv", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--wandb-username", default=None, type=str)
    parser.add_argument("--sequence-pad-count", default=0, type=int)
    parser.add_argument("--structure-pad-count", default=0, type=int)
    parser.add_argument("--self-supervision", action="store_true")
    # ── custom split ────────────────────────────────────────────────────────────
    parser.add_argument("--custom-split-file", default=None, type=str,
                        help="JSON with {train, val, test} index lists. "
                             "When provided, replaces random 80/10/10 split.")
    config = parser.parse_args()

    update_paths(config)

    model_str = f"{config.model}-lr_pt_{config.learning_rate_pretrain}-lr_ft_{config.learning_rate_finetune}" + \
        f"-ep_{config.num_epochs}-bs_{config.batch_size}-fseq_{config.full_sequence}-seql_{config.sequence_loss}" + \
        f"-fs_{config.feature_size}-cs_{config.coord_size}-seed_{config.seed}"
    if config.custom_split_file:
        model_str += "-novel_split"
    config.model_save_path_pretrain = os.path.join(config.model_save_dir, model_str + "_pretrain.pt")
    config.model_save_path_finetune = os.path.join(config.model_save_dir, model_str + "_finetune.pt")
    os.makedirs(config.model_save_dir, exist_ok=True)

    wandb.init(
        project="ImmunoPred-IEDB-MIT",
        entity=config.wandb_username,
        name=f"PropIEDB_ImmunoIEDB:{model_str}",
        config=config,
    )
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    input_dim = 283 * 21 if config.full_sequence else 11 * 21
    model = model_map[config.model](vae_input_dim=input_dim, device=device)
    model.to(device)

    dataset_pt_ft = ImmunoPredDataset(config,
                                      graph_directory=config.graph_dir_IEDB,
                                      property_path=config.property_path_IEDB,
                                      hla_path=config.hla_path)

    # ── Split ──────────────────────────────────────────────────────────────────
    if config.custom_split_file:
        with open(config.custom_split_file) as f:
            split_info = json.load(f)
        train_dataset_pt = torch.utils.data.Subset(dataset_pt_ft, split_info["train"])
        val_dataset_pt   = torch.utils.data.Subset(dataset_pt_ft, split_info["val"])
        test_dataset_pt  = torch.utils.data.Subset(dataset_pt_ft, split_info["test"])
        print(f"Using custom split from {config.custom_split_file}")
    else:
        train_dataset_pt, val_dataset_pt, test_dataset_pt = torch.utils.data.random_split(
            dataset_pt_ft, [0.8, 0.1, 0.1], generator)

    train_dataset_ft, val_dataset_ft, test_dataset_ft = train_dataset_pt, val_dataset_pt, test_dataset_pt
    print("Pretraining train/val/test size:", len(train_dataset_pt), len(val_dataset_pt), len(test_dataset_pt))
    print("Finetuning train/val/test size:", len(train_dataset_ft), len(val_dataset_ft), len(test_dataset_ft))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_pretrain)
    losses = Losses(input_dim, dataset_pt_ft.class_weights, sequence=config.sequence_loss)

    train_split_dataset = SplitDataset(train_dataset_pt, "train", binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset   = SplitDataset(val_dataset_pt,   "val",   binary=False, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True,  num_workers=config.num_workers)
        val_loader   = GraphDataLoader(val_split_dataset,   batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss_SSL)
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True,  num_workers=config.num_workers)
        val_loader   = GraphDataLoader(val_split_dataset,   batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.regression_loss)

    print("DONE PRE-TRAINING")
    del train_split_dataset, val_split_dataset
    del train_loader, val_loader
    del optimizer

    model.load_trained(config.model_save_path_pretrain, new_head=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_finetune, weight_decay=1e-6)

    train_split_dataset = SplitDataset(train_dataset_ft, "train", binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    val_split_dataset   = SplitDataset(val_dataset_ft,   "val",   binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)
    test_split_dataset  = SplitDataset(test_dataset_ft,  "test",  binary=True, full=config.full_sequence, comparative=False, return_amino_acid=config.self_supervision)

    if config.self_supervision:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=True)
        val_loader   = GraphDataLoader(val_split_dataset,   batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        test_loader  = GraphDataLoader(test_split_dataset,  batch_size=config.batch_size, collate_fn=collate_amino_acid, shuffle=False)
        train_losses, val_losses = train_model_SSL(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss_SSL, stage="finetune")
    else:
        train_loader = GraphDataLoader(train_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=True)
        val_loader   = GraphDataLoader(val_split_dataset,   batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        test_loader  = GraphDataLoader(test_split_dataset,  batch_size=config.batch_size, collate_fn=collate, shuffle=False)
        train_losses, val_losses = train_model(config, device, model, train_loader, val_loader, optimizer, losses.BCE_loss, stage="finetune")

    print("DONE FINE TUNING")

    model.load_trained(config.model_save_path_finetune, new_head=False)

    train_stats = inference(config, model, train_loader, device)
    test_stats  = inference(config, model, test_loader,  device, optimal_threshold=train_stats["optimal_threshold"])

    wandb.log({
        "Train ROC AUC": train_stats["roc_auc"], "Train PR AUC": train_stats["pr_auc"],
        "Train Accuracy @0.5": train_stats["accuracy"], "Train Accuracy @op": train_stats["accuracy_op"],
        "Train F1 Score @0.5": train_stats["f1"],       "Train F1 Score @op": train_stats["f1_op"],
        "Train Precision @0.5": train_stats["precision"], "Train Recall @0.5": train_stats["recall"],
        "Train Mean PPVn @0.5": train_stats["ppvn"],    "Train PPVn (n=30) @0.5": train_stats["ppv30"],
    })
    wandb.log({
        "Test ROC AUC": test_stats["roc_auc"],  "Test PR AUC": test_stats["pr_auc"],
        "Test Accuracy @0.5": test_stats["accuracy"],  "Test Accuracy @op": test_stats["accuracy_op"],
        "Test F1 Score @0.5": test_stats["f1"],        "Test F1 Score @op": test_stats["f1_op"],
        "Test Precision @0.5": test_stats["precision"], "Test Recall @0.5": test_stats["recall"],
        "Test Mean PPVn @0.5": test_stats["ppvn"],     "Test PPVn (n=30) @0.5": test_stats["ppv30"],
    })
    print(f"\nFinal test (novel split) — ROC AUC: {test_stats['roc_auc']:.4f}  PR AUC: {test_stats['pr_auc']:.4f}")
    print(f"Checkpoint: {config.model_save_path_finetune}")
