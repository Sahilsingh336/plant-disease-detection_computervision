"""
Evaluation Metrics Module for Plant Pathology:
- Top-1 Classification Accuracy
- Missing-Class-Aware Macro F1 (handles PlantDoc test set class discrepancies)
- Minority-Class F1 (evaluates tail/rare diseases, e.g. bottom 25% rarest classes)
- Confusion Matrix extraction
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, n_classes, device="cuda", use_amp=True):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    dev = torch.device(device)

    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp and dev.type == "cuda"):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        ps.append(out.argmax(1).cpu())
        ys.append(y.cpu())
        n += len(y)

    if n == 0:
        return dict(loss=0.0, acc=0.0, macro_f1=0.0, f1s=[], present=[], cm=np.zeros((n_classes, n_classes)))

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    # Calculate per-class F1
    f1s = f1_score(y, p, average=None, labels=list(range(n_classes)), zero_division=0)
    present = np.unique(y)

    # Macro F1 averaged ONLY over classes present in this split
    macro_f1 = float(f1s[present].mean() * 100) if len(present) > 0 else 0.0
    acc = float((y == p).mean() * 100)
    cm = confusion_matrix(y, p, labels=list(range(n_classes)))

    return dict(
        loss=loss_sum / n,
        acc=acc,
        macro_f1=macro_f1,
        f1s=f1s,
        present=present,
        cm=cm
    )


def minority_f1(res, train_df, cmap):
    """
    Computes mean F1 score for the bottom 25% rarest classes (by training frequency)
    that are present in the test set.
    """
    if "label" not in train_df.columns or len(res["present"]) == 0:
        return float("nan")

    counts = train_df["label"].map(cmap).value_counts()
    threshold = counts.quantile(0.25)
    minority_classes = [c for c in res["present"] if counts.get(c, 0) <= threshold]

    if not minority_classes:
        return float("nan")

    return float(res["f1s"][minority_classes].mean() * 100)
