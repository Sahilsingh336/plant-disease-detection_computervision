"""
Computational Efficiency Profiler CLI:
Authors: Laxman (2024UCA1871), Sahil Singh (2024UCA1815)
Supervisor: Assistant Prof. Ankush Jain

Measures parameters, approximate GMACs, inference throughput (img/s, fp16),
training step throughput (img/s), and peak GPU memory across all models.
"""

import os
import argparse
import pandas as pd
import torch

from src.models import (
    Classifier,
    HybridMamba,
    build_resnet50,
    build_convnext_tiny,
    build_swin_t
)
from src.utils import profile_model


def main():
    parser = argparse.ArgumentParser(description="Profile Model Efficiency")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--output", type=str, default="./results/model_efficiency_profile.csv",
                        help="Path to save efficiency profile CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for profiling")
    args = parser.parse_args()

    print("=" * 70)
    print("  COMPUTATIONAL EFFICIENCY PROFILING SUITE")
    print(f"  Device: {args.device.upper()} | Resolution: {args.resolution}x{args.resolution} | Batch: {args.batch_size}")
    print("=" * 70)

    builders = {
        "Bi-Mamba (Scratch Baseline)": lambda: Classifier(n_classes=28, d=192, depth=8),
        "Hybrid ResNet-BiMamba (Ours)": lambda: HybridMamba(n_classes=28, d=192, depth=4, pretrained_backbone=False),
        "ResNet-50 Reference": lambda: build_resnet50(n_classes=28, pretrained=False),
        "ConvNeXt-Tiny Modern CNN": lambda: build_convnext_tiny(n_classes=28, pretrained=False),
        "Swin-Transformer (Swin-T)": lambda: build_swin_t(n_classes=28, pretrained=False),
    }

    rows = []
    for name, builder in builders.items():
        print(f"[PROFILING] Measuring: {name}...")
        try:
            m = builder()
            row = profile_model(
                name=name,
                model=m,
                n_classes=28,
                res=args.resolution,
                batch_size=args.batch_size,
                device=args.device
            )
            rows.append(row)
            print(f"  -> Params: {row['params_M']}M | GMACs: {row['GMACs_approx']} | "
                  f"Infer: {row['infer_img_per_s']} img/s | Train: {row['train_img_per_s']} img/s")
        except Exception as e:
            print(f"  -> Profiling failed for {name}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("  HARDWARE BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"Saved benchmark summary to: {args.output}")


if __name__ == "__main__":
    main()
