"""
Computational Efficiency Profiler:
Measures model parameters (M), GMACs, inference throughput (img/s),
training throughput (img/s), and peak GPU memory consumption (GB).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_model(name, model, n_classes=28, res=224, batch_size=32, iters=15, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    row = {
        "model": name,
        "params_M": round(params, 2)
    }

    # FLOPs / GMACs
    gmacs = float("nan")
    if HAS_FLOP_COUNTER:
        try:
            model.eval()
            with FlopCounterMode(display=False) as fc, torch.no_grad():
                dummy = torch.randn(1, 3, res, res, device=dev)
                model(dummy)
            gmacs = round(fc.get_total_flops() / 2 / 1e9, 2)
        except Exception:
            pass
    row["GMACs_approx"] = gmacs

    # Inference Throughput (fp16)
    use_amp = dev.type == "cuda"
    x = torch.randn(batch_size, 3, res, res, device=dev)
    model.eval()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                model(x)
        sync(dev)

        t0 = time.time()
        for _ in range(iters):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                model(x)
        sync(dev)

    elapsed = time.time() - t0
    row["infer_img_per_s"] = round((batch_size * iters) / max(elapsed, 1e-6), 1)
    row["infer_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else 0.0

    # Training Step Throughput (batch 16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_bs = min(16, batch_size)
    xb = torch.randn(train_bs, 3, res, res, device=dev)
    yb = torch.randint(0, n_classes, (train_bs,), device=dev)
    scaler = torch.amp.GradScaler(device=dev.type, enabled=use_amp)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_iters = 6
    for i in range(train_iters):
        if i == 2:
            sync(dev)
            t0 = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(xb)
            loss = F.cross_entropy(out.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    sync(dev)
    measured_steps = train_iters - 2
    elapsed_train = time.time() - t0
    row["train_img_per_s"] = round((train_bs * measured_steps) / max(elapsed_train, 1e-6), 1)
    row["train_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else 0.0

    return row
