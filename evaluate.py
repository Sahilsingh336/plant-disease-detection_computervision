"""
Evaluation CLI Script:
Authors: Laxman (2024UCA1871), Sahil Singh (2024UCA1815)
Supervisor: Assistant Prof. Ankush Jain

Loads a saved model checkpoint and evaluates test accuracy,
missing-class-aware macro F1, and minority class F1.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import (
    Classifier,
    HybridMamba,
    build_resnet50,
    build_convnext_tiny,
    build_swin_t
)
from src.data import LeafDataset, get_transforms, load_splits
from src.utils import evaluate, minority_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["hybrid", "baseline", "resnet50", "convnext", "swin"],
                        help="Architecture of the checkpoint")
    parser.add_argument("--dataset", type=str, default="plantdoc",
                        choices=["plantdoc", "citrus"],
                        help="Dataset benchmark to evaluate on")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index for Citrus evaluation")
    parser.add_argument("--splits-dir", type=str, default="./splits",
                        help="Directory containing split CSVs")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory for dataset images")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Evaluation device")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 65)
    print(f"  EVALUATION: {args.model.upper()} on {args.dataset.upper()} (fold={args.fold})")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 65)

    tr_df, va_df, te_df, cmap = load_splits(args.splits_dir, dataset=args.dataset, fold=args.fold)
    n_classes = len(cmap)

    _, eval_tf = get_transforms(resolution=224, recipe="standard")
    test_ds = LeafDataset(te_df, cmap, transform=eval_tf, dataset_name=args.dataset, data_dir=args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "hybrid":
        model = HybridMamba(n_classes=n_classes, d=192, depth=4, pretrained_backbone=False)
    elif args.model == "baseline":
        model = Classifier(n_classes=n_classes, d=192, depth=8)
    elif args.model == "resnet50":
        model = build_resnet50(n_classes=n_classes, pretrained=False)
    elif args.model == "convnext":
        model = build_convnext_tiny(n_classes=n_classes, pretrained=False)
    elif args.model == "swin":
        model = build_swin_t(n_classes=n_classes, pretrained=False)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(args.device)

    res = evaluate(model, test_loader, n_classes=n_classes, device=args.device)
    m_f1 = minority_f1(res, tr_df, cmap)

    print("\n" + "=" * 65)
    print(f"  Test Loss:        {res['loss']:.4f}")
    print(f"  Test Accuracy:    {res['acc']:.2f}%")
    print(f"  Test Macro F1:    {res['macro_f1']:.2f}%")
    print(f"  Minority F1:      {m_f1:.2f}%")
    print("=" * 65)

    # Print Confusion Matrix
    print("\nConfusion Matrix (True Rows x Predicted Columns):")
    inv_cmap = {v: k for k, v in cmap.items()}
    class_names = [inv_cmap[i] for i in range(n_classes)]
    if n_classes <= 5:
        # Detailed table for Citrus (5 classes)
        header = f"{'Class':<16} | " + " | ".join(f"{c[:8]:>8}" for c in class_names)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for i, row in enumerate(res["cm"]):
            row_str = " | ".join(f"{val:>8d}" for val in row)
            print(f"{class_names[i]:<16} | {row_str}")
        print("-" * len(header))
    else:
        # Summary for PlantDoc (28 classes)
        print(f"Computed {n_classes}x{n_classes} confusion matrix. Correct predictions on diagonal: {res['cm'].diagonal().sum()}/{res['cm'].sum()} images.")


if __name__ == "__main__":
    main()
