"""
Supervised Training Module:
Supports:
1. Standard Recipe: AdamW, Cosine Warmup, Label Smoothing
2. Strong Recipe: Mixup, CutMix, Model EMA Weight Averaging, RandAugment, Cosine Annealing
"""

import os
import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.metrics import evaluate, minority_f1


class ModelEMA:
    """
    Exponential Moving Average of model parameters for smooth generalization.
    """
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


def apply_mixup_cutmix(x, y, alpha=0.8, beta=1.0, prob=0.5):
    """
    Applies Mixup or CutMix augmentation randomly with given probability.
    """
    if np.random.rand() > prob:
        return x, y, None

    lam = np.random.beta(alpha, beta) if np.random.rand() < 0.5 else np.random.beta(beta, beta)
    B, C, H, W = x.shape
    index = torch.randperm(B, device=x.device)

    # CutMix box
    if np.random.rand() < 0.5:
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        r_w = int(W * np.sqrt(1 - lam))
        r_h = int(H * np.sqrt(1 - lam))
        x1 = np.clip(r_x - r_w // 2, 0, W)
        x2 = np.clip(r_x + r_w // 2, 0, W)
        y1 = np.clip(r_y - r_h // 2, 0, H)
        y2 = np.clip(r_y + r_h // 2, 0, H)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
        return x_mixed, y, (index, lam_adj)
    else:
        # Standard Mixup
        x_mixed = lam * x + (1 - lam) * x[index]
        return x_mixed, y, (index, lam)


def train_model(model, train_loader, val_loader, test_loader, n_classes,
                epochs=30, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-2,
                recipe="standard", warmup_epochs=3, device="cuda",
                checkpoint_dir="checkpoints", tag="run", train_df=None, cmap=None):

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    use_amp = dev.type == "cuda"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Separate parameters for differential learning rates (e.g. for Hybrid)
    if hasattr(model, "tok") and hasattr(model.tok, "stem"):
        stem_params = list(model.tok.stem.parameters())
        stem_ids = {id(p) for p in stem_params}
        head_params = [p for p in model.parameters() if id(p) not in stem_ids]
        param_groups = [
            {"params": stem_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        ]
    else:
        decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos" not in n]
        no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "pos" not in n)]
        param_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]

    opt = torch.optim.AdamW(param_groups, lr=lr)
    total_steps = epochs * len(train_loader)
    warm_steps = warmup_epochs * len(train_loader)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(1, warm_steps) if s < warm_steps else
        0.5 * (1 + math.cos(math.pi * (s - warm_steps) / max(1, total_steps - warm_steps)))
    )

    scaler = torch.amp.GradScaler(device=dev.type, enabled=use_amp)
    ema = ModelEMA(model, decay=0.999) if recipe == "strong" else None

    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    history = []
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        t0, tl, nb = time.time(), 0.0, 0

        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

            mix_info = None
            if recipe == "strong":
                x, y, mix_info = apply_mixup_cutmix(x, y)

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                if mix_info is not None:
                    idx_b, lam = mix_info
                    loss = lam * F.cross_entropy(out.float(), y, label_smoothing=0.1) + \
                           (1 - lam) * F.cross_entropy(out.float(), y[idx_b], label_smoothing=0.1)
                else:
                    loss = F.cross_entropy(out.float(), y, label_smoothing=0.1)

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                sched.step()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if ema:
                ema.update(model)

            tl += loss.item()
            nb += 1

        # Evaluate validation
        eval_target = ema.module if ema else model
        r_val = evaluate(eval_target, val_loader, n_classes, device=dev.type, use_amp=use_amp)

        if r_val["acc"] > best["val_acc"]:
            best["val_acc"] = r_val["acc"]
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in eval_target.state_dict().items()}

        history.append({
            "epoch": ep + 1,
            "train_loss": tl / max(nb, 1),
            "val_loss": r_val["loss"],
            "val_acc": r_val["acc"]
        })

        print(f"[{tag}] Epoch {ep+1:2d}/{epochs:2d} | Train Loss: {tl/max(nb,1):.4f} | "
              f"Val Acc: {r_val['acc']:5.2f}% | Best Val: {best['val_acc']:5.2f}% (ep {best['epoch']+1}) | {time.time()-t0:.1f}s")

    # Final evaluation on test split using best checkpoint
    model.load_state_dict(best["state"])
    res = evaluate(model, test_loader, n_classes, device=dev.type, use_amp=use_amp)

    row = {
        "tag": tag,
        "recipe": recipe,
        "epochs": epochs,
        "best_epoch": best["epoch"] + 1,
        "val_acc": round(best["val_acc"], 2),
        "test_acc": round(res["acc"], 2),
        "test_macro_f1": round(res["macro_f1"], 2),
        "total_minutes": round((time.time() - t_start) / 60, 2)
    }

    if train_df is not None and cmap is not None:
        row["test_minority_f1"] = round(minority_f1(res, train_df, cmap), 2)

    # Save best checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{tag}_best.pt")
    torch.save({
        "state_dict": best["state"],
        "row": row,
        "history": history
    }, ckpt_path)

    return row, res
