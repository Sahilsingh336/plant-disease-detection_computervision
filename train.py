"""
Main Execution Script for Training and Pretraining.
Authors: Laxman (2024UCA1871), Sahil Singh (2024UCA1815)
Supervisor: Assistant Prof. Ankush Jain

Supports:
- Baseline Bi-Mamba (from scratch)
- Proposed Hybrid ResNet-50 + Bi-Mamba
- Reference Backbones (ResNet-50, ConvNeXt-Tiny, Swin-T)
- PSMamba Progressive Dual-Student SSL Pretraining
- Synthetic Zero-Download Mode for Instant Verification
"""

import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import (
    Classifier,
    HybridMamba,
    PSMambaSSL,
    build_resnet50,
    build_convnext_tiny,
    build_swin_t
)
from src.data import (
    LeafDataset,
    SSLDataset,
    SyntheticPlantDataset,
    get_transforms,
    load_splits
)
from src.training import train_model, pretrain_psmamba_ssl
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="PSMamba & Hybrid Vision Mamba Training Suite")
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["hybrid", "baseline", "resnet50", "convnext", "swin", "ssl"],
                        help="Model architecture to train or pretrain")
    parser.add_argument("--dataset", type=str, default="plantdoc",
                        choices=["plantdoc", "citrus"],
                        help="Dataset benchmark to evaluate on")
    parser.add_argument("--recipe", type=str, default="strong",
                        choices=["standard", "strong"],
                        help="Training recipe (standard AdamW vs strong with Mixup/CutMix/RandAugment/EMA)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                        help="Backbone learning rate for hybrid model")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index for Citrus cross-validation (0 to 4)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Input image resolution (e.g., 224 or 320)")
    parser.add_argument("--splits-dir", type=str, default="./splits",
                        help="Directory containing pre-computed split CSV files")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory for dataset images (optional, re-maps split paths)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save evaluation results CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run using synthetic dummy data (zero dataset downloads required)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run for 2 epochs to verify execution pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    epochs = 2 if args.quick else args.epochs

    print("=" * 65)
    print(f"  EXPERIMENT: {args.model.upper()} on {args.dataset.upper()} ({args.recipe.upper()} recipe)")
    print(f"  Device: {args.device.upper()} | Epochs: {epochs} | Batch: {args.batch_size} | Seed: {args.seed}")
    print("=" * 65)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Datasets & DataLoaders
    if args.synthetic:
        print("[DATA] Operating in SYNTHETIC data mode for zero-dependency testing.")
        n_classes = 28 if args.dataset == "plantdoc" else 5
        train_ds = SyntheticPlantDataset(n_samples=args.batch_size * 2, n_classes=n_classes, resolution=args.resolution)
        val_ds = SyntheticPlantDataset(n_samples=args.batch_size, n_classes=n_classes, resolution=args.resolution)
        test_ds = SyntheticPlantDataset(n_samples=args.batch_size, n_classes=n_classes, resolution=args.resolution)
        train_df, cmap = pd.DataFrame({"label": list(range(n_classes))}), {i: i for i in range(n_classes)}
    else:
        if not os.path.exists(args.splits_dir):
            print(f"[ERROR] Splits directory not found at: {args.splits_dir}. Use --synthetic for mock testing.")
            sys.exit(1)

        print(f"[DATA] Loading pre-stratified splits from {args.splits_dir} (dataset={args.dataset}, fold={args.fold})")
        train_df, val_df, test_df, cmap = load_splits(args.splits_dir, dataset=args.dataset, fold=args.fold)
        n_classes = len(cmap)
        print(f"[DATA] Classes: {n_classes} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        train_tf, eval_tf = get_transforms(resolution=args.resolution, recipe=args.recipe)

        if args.model == "ssl":
            ssl_train_ds = SSLDataset(train_df, resolution=args.resolution, dataset_name=args.dataset, data_dir=args.data_dir)
            train_loader = DataLoader(ssl_train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=2, pin_memory=(args.device == "cuda"))
        else:
            train_ds = LeafDataset(train_df, cmap, transform=train_tf, dataset_name=args.dataset, data_dir=args.data_dir)
            val_ds = LeafDataset(val_df, cmap, transform=eval_tf, dataset_name=args.dataset, data_dir=args.data_dir)
            test_ds = LeafDataset(test_df, cmap, transform=eval_tf, dataset_name=args.dataset, data_dir=args.data_dir)

    if args.model != "ssl":
        num_workers = 0 if args.synthetic else 2
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=(args.device == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=(args.device == "cuda"))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=(args.device == "cuda"))

    # 2. Build Selected Model
    tag = f"{args.model}_{args.recipe}_{args.dataset}_f{args.fold}_s{args.seed}"
    print(f"[MODEL] Initializing {args.model.upper()} architecture...")

    if args.model == "ssl":
        ssl_model = PSMambaSSL(d=192, depth=8, k=1024, patch=16)
        ckpt_path = pretrain_psmamba_ssl(
            ssl_model, train_loader, epochs=epochs, lr=args.lr,
            norm="avg", checkpoint_dir=args.checkpoint_dir, tag=tag, device=args.device
        )
        print(f"[SUCCESS] SSL Pretraining completed. Encoder weights saved to: {ckpt_path}")
        return

    elif args.model == "hybrid":
        model = HybridMamba(n_classes=n_classes, d=192, depth=4, pretrained_backbone=not args.synthetic)
    elif args.model == "baseline":
        model = Classifier(n_classes=n_classes, d=192, depth=8)
    elif args.model == "resnet50":
        model = build_resnet50(n_classes=n_classes, pretrained=not args.synthetic)
    elif args.model == "convnext":
        model = build_convnext_tiny(n_classes=n_classes, pretrained=not args.synthetic)
    elif args.model == "swin":
        model = build_swin_t(n_classes=n_classes, pretrained=not args.synthetic)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # 3. Train and Evaluate
    row, res = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_classes=n_classes,
        epochs=epochs,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        recipe=args.recipe,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        tag=tag,
        train_df=train_df,
        cmap=cmap
    )

    # 4. Save and Report Metrics
    results_csv = os.path.join(args.output_dir, f"results_cli_{args.dataset}.csv")
    row["model"] = args.model
    row["dataset"] = args.dataset
    row["fold"] = args.fold
    row["seed"] = args.seed
    row["resolution"] = args.resolution

    df_row = pd.DataFrame([row])
    df_row.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)

    print("\n" + "=" * 65)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 65)
    for k, v in row.items():
        print(f"  {k:20s}: {v}")
    print("=" * 65)
    print(f"Results appended to: {results_csv}")


if __name__ == "__main__":
    main()
