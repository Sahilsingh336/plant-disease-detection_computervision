"""
Self-Supervised Learning Training Module:
PSMamba Progressive Dual-Student Distillation pretraining loop.
"""

import os
import time
import torch
from ..models.ssl import get_ssl_schedules


def pretrain_psmamba_ssl(ssl_model, ssl_loader, epochs=50, lr=5e-4, weight_decay=1e-2,
                         norm="avg", use_gg=False, progressive=True,
                         checkpoint_dir="checkpoints", tag="psmamba_ssl", device="cuda"):

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ssl_model = ssl_model.to(dev)
    use_amp = dev.type == "cuda"
    os.makedirs(checkpoint_dir, exist_ok=True)

    opt = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(ssl_loader)
    scaler = torch.amp.GradScaler(device=dev.type, enabled=use_amp)

    print(f"[{tag}] Starting PSMamba SSL Pretraining: {epochs} epochs, norm={norm}, progressive={progressive}")
    t_start = time.time()
    step = 0

    for ep in range(epochs):
        ssl_model.train()
        t0 = time.time()
        loss_accum, ent_accum, proto_ent_accum, nb = 0.0, 0.0, 0.0, 0

        for g_t, g_s in ssl_loader:
            g_t, g_s = g_t.to(dev, non_blocking=True), g_s.to(dev, non_blocking=True)
            sc = get_ssl_schedules(step, total_steps, progressive=progressive)

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss, stats = ssl_model.forward_loss(g_t, g_s, sc, norm=norm, use_gg=use_gg)

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # Update EMA teacher
            ssl_model.update_teacher(sc["mom"])

            loss_accum += stats["loss"]
            ent_accum += stats["entropy"]
            proto_ent_accum += stats["proto_entropy"]
            nb += 1
            step += 1

        elapsed = time.time() - t0
        avg_loss = loss_accum / max(nb, 1)
        avg_ent = ent_accum / max(nb, 1)
        avg_proto_ent = proto_ent_accum / max(nb, 1)

        print(f"[{tag}] SSL Epoch {ep+1:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | "
              f"Token Entropy: {avg_ent:.3f} | Proto Entropy: {avg_proto_ent:.3f} | {elapsed:.1f}s")

    # Save pretrained encoder state dict
    ckpt_path = os.path.join(checkpoint_dir, f"{tag}_encoder.pt")
    torch.save(ssl_model.enc.state_dict(), ckpt_path)
    print(f"[{tag}] Finished SSL pretraining in {(time.time() - t_start)/60:.2f} mins. Saved encoder to {ckpt_path}")
    return ckpt_path
