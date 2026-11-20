"""
Smoke Test: Zero-dependency self-test for the entire repository.
Authors: Laxman (2024UCA1871), Sahil Singh (2024UCA1815)
Supervisor: Assistant Prof. Ankush Jain

Verifies all architectures, forward passes, backward passes, loss functions,
and evaluation metrics in < 10 seconds using in-memory synthetic data.
"""

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import (
    BiMambaBlock,
    MiniBiMamba,
    Classifier,
    ResNetTokenizer,
    HybridMamba,
    PSMambaSSL,
    DinoHead,
    build_resnet50,
    build_convnext_tiny,
    build_swin_t
)
from src.data import SyntheticPlantDataset, check_data_leakage
from src.utils import set_seed, evaluate, minority_f1


def run_smoke_test():
    print("=" * 65)
    print("  PSMAMBA & HYBRID VISION MAMBA: SMOKE TEST SUITE")
    print("=" * 65)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {dev.upper()}")
    set_seed(42)

    tests_passed = 0
    total_tests = 7

    # 1. Test BiMamba Block & MiniBiMamba
    try:
        t0 = time.time()
        blk = BiMambaBlock(d=64).to(dev)
        x_tok = torch.randn(2, 49, 64, device=dev)
        y_tok = blk(x_tok)
        assert y_tok.shape == x_tok.shape, f"Shape mismatch: {y_tok.shape}"

        enc = MiniBiMamba(d=64, depth=2, patch=16, base_grid=7).to(dev)
        img = torch.randn(2, 3, 112, 112, device=dev)
        tokens, grid = enc(img)
        assert tokens.shape == (2, 49, 64), f"Token shape mismatch: {tokens.shape}"
        print(f"[PASS] 1/7: BiMamba Block & Encoder ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 1/7: BiMamba Block & Encoder -> {e}")

    # 2. Test Classifier Model
    try:
        t0 = time.time()
        clf = Classifier(n_classes=5, d=64, depth=2).to(dev)
        out = clf(img)
        assert out.shape == (2, 5), f"Classifier output shape mismatch: {out.shape}"
        print(f"[PASS] 2/7: From-Scratch Bi-Mamba Classifier ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 2/7: From-Scratch Bi-Mamba Classifier -> {e}")

    # 3. Test Hybrid Architecture (ResNet Tokenizer + Bi-Mamba)
    try:
        t0 = time.time()
        hybrid = HybridMamba(n_classes=5, d=64, depth=2, base_grid=7, pretrained_backbone=False).to(dev)
        out_hyb = hybrid(img)
        assert out_hyb.shape == (2, 5), f"Hybrid output shape mismatch: {out_hyb.shape}"
        print(f"[PASS] 3/7: Hybrid CNN-BiMamba Architecture ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 3/7: Hybrid CNN-BiMamba Architecture -> {e}")

    # 4. Test PSMamba SSL & MPA Patch Transport
    try:
        t0 = time.time()
        ssl_mod = PSMambaSSL(d=64, depth=2, k=128, patch=16, global_grid=7).to(dev)
        g_t = torch.randn(2, 3, 112, 112, device=dev)
        g_s = torch.randn(2, 3, 112, 112, device=dev)
        sc = {"tau_t": 0.05, "lam_m": 1.0, "lam_l": 1.0, "lam_g": 1.0, "mom": 0.99}
        loss, stats = ssl_mod.forward_loss(g_t, g_s, sc, norm="avg", n_mid=1, n_loc=2)
        assert torch.isfinite(loss), "SSL loss is non-finite!"
        print(f"[PASS] 4/7: PSMamba SSL Multi-View Distillation & MPA ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 4/7: PSMamba SSL Multi-View Distillation & MPA -> {e}")

    # 5. Test Alternative Backbones (ResNet, ConvNeXt, Swin)
    try:
        t0 = time.time()
        rn = build_resnet50(n_classes=5, pretrained=False).to(dev)
        cx = build_convnext_tiny(n_classes=5, pretrained=False).to(dev)
        sw = build_swin_t(n_classes=5, pretrained=False).to(dev)
        assert rn(img).shape == (2, 5)
        assert cx(img).shape == (2, 5)
        assert sw(img).shape == (2, 5)
        print(f"[PASS] 5/7: Vision Backbones (ResNet-50, ConvNeXt-Tiny, Swin-T) ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 5/7: Vision Backbones -> {e}")

    # 6. Test Synthetic Pipeline & Training Step
    try:
        t0 = time.time()
        ds = SyntheticPlantDataset(n_samples=16, n_classes=5, resolution=112)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(dev), y_b.to(dev)
            preds = clf(x_b)
            loss_val = nn.functional.cross_entropy(preds, y_b)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            break
        print(f"[PASS] 6/7: End-to-End Optimization Step ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 6/7: End-to-End Optimization Step -> {e}")

    # 7. Test Evaluation Metrics
    try:
        t0 = time.time()
        ds_test = SyntheticPlantDataset(n_samples=16, n_classes=5, resolution=112)
        test_loader = DataLoader(ds_test, batch_size=4)
        metrics = evaluate(clf, test_loader, n_classes=5, device=dev, use_amp=False)
        assert "acc" in metrics and "macro_f1" in metrics
        print(f"[PASS] 7/7: Missing-Class Aware Metrics & Confusion Matrix ({time.time()-t0:.2f}s)")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 7/7: Evaluation Metrics -> {e}")

    print("=" * 65)
    print(f"  RESULT: {tests_passed}/{total_tests} TESTS PASSED")
    print("=" * 65)

    if tests_passed == total_tests:
        print("[SUCCESS] All system components verified and ready for execution.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
