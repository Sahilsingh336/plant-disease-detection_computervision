"""
Self-Supervised Learning Module: PSMamba Progressive Dual-Student Distillation.
Features:
- Multi-scale hierarchical cropping (Global 224, Mid 160, Local 96)
- Multi-view Patch Alignment (MPA) with deformable coordinate offset Delta
- Anti-collapse DINO-style projection head with prototype cosine similarity & EMA centering
- Masked KL divergence loss normalized by active patch mask area |M|_1
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from .mamba import MiniBiMamba


def sample_boxes(B, n, frac, dev, global_size=224):
    """
    Sample n random square boxes per image in global pixel coordinates:
    (B * n, 3) = [x0, y0, side].
    """
    side = torch.empty(B * n, device=dev).uniform_(*frac) * global_size
    x0 = torch.rand(B * n, device=dev) * (global_size - side)
    y0 = torch.rand(B * n, device=dev) * (global_size - side)
    return torch.stack([x0, y0, side], dim=1)


def get_ssl_schedules(step, total, progressive=True, temp_teacher=(0.04, 0.07), ema_mom=(0.996, 1.0)):
    """
    Dynamic schedules for teacher temperature, distillation weights, and EMA momentum.
    """
    p = step / max(1, total)
    tau_t = temp_teacher[0] + (temp_teacher[1] - temp_teacher[0]) * min(1.0, p / 0.3)
    lam_l, lam_m = (1.5 - p, 0.5 + p) if progressive else (1.0, 1.0)
    mom = ema_mom[1] - (ema_mom[1] - ema_mom[0]) * 0.5 * (1 + math.cos(math.pi * p))
    return dict(tau_t=tau_t, lam_m=lam_m, lam_l=lam_l, lam_g=1.0, mom=mom)


class DinoHead(nn.Module):
    """
    MLP -> L2 normalization -> cosine similarity with K prototypes.
    Logits are strictly bounded in [-1, 1], completely eliminating representation collapse.
    Replaces the base paper's plain linear head.
    """
    def __init__(self, d=192, k=1024, hidden=512, bottleneck=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck)
        )
        self.proto = nn.Parameter(torch.empty(k, bottleneck))
        nn.init.trunc_normal_(self.proto, std=0.02)

    def forward(self, x):
        normed_feat = F.normalize(self.mlp(x), dim=-1)
        normed_proto = F.normalize(self.proto, dim=-1)
        return normed_feat @ normed_proto.T


class PSMambaSSL(nn.Module):
    """
    PSMamba Progressive Self-Supervised Learning Framework.
    """
    def __init__(self, d=192, depth=8, k=1024, patch=16, global_grid=14, use_delta=True):
        super().__init__()
        self.d = d
        self.patch = patch
        self.global_grid = global_grid
        self.use_delta = use_delta

        # Student Encoder (processes Mid and Local crops)
        self.enc = MiniBiMamba(d=d, depth=depth, patch=patch, base_grid=global_grid)

        # EMA Teacher Encoder (sees full Global view)
        self.teacher = copy.deepcopy(self.enc)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Projection heads
        self.head = DinoHead(d=d, k=k)
        self.teacher_head = copy.deepcopy(self.head)
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Scale adaptors for Mid (10x10=100) and Local (6x6=36)
        self.adapt = nn.ParameterDict({
            "mid": nn.Parameter(torch.zeros(1, 100, d)),
            "loc": nn.Parameter(torch.zeros(1, 36, d))
        })
        nn.init.trunc_normal_(self.adapt["mid"], std=0.02)
        nn.init.trunc_normal_(self.adapt["loc"], std=0.02)

        # Learnable spatial coordinate offsets Delta (Paper Eq. 9)
        self.delta = nn.ModuleDict({
            "mid": nn.Linear(d, 2),
            "loc": nn.Linear(d, 2)
        })
        for m in self.delta.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # EMA Center for student sharpening and anti-collapse
        self.register_buffer("center", torch.zeros(1, 1, k))

    def update_teacher(self, momentum):
        with torch.no_grad():
            for ps, pt in zip(self.enc.parameters(), self.teacher.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
            for ps, pt in zip(self.head.parameters(), self.teacher_head.parameters()):
                pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    def transport(self, Z, boxes, grid, key, norm="avg"):
        """
        Multi-view Patch Alignment (MPA):
        Maps student view tokens onto the Global 14x14 grid using bilinear splatting.
        norm="avg": Area-normalized (student improvement to prevent zoom inflation).
        norm="sum": Base paper formulation.
        """
        Kc, n, d = Z.shape
        Z = Z.float() + self.adapt[key]
        idx = torch.arange(n, device=Z.device)
        jx, jy = (idx % grid).float(), (idx // grid).float()
        x0, y0, side = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3]

        # Calculate continuous coordinate in Global token space
        u = (x0 + (jx[None] + 0.5) * side / grid) / self.patch - 0.5
        v = (y0 + (jy[None] + 0.5) * side / grid) / self.patch - 0.5

        if self.use_delta:
            dlt = 0.25 * torch.tanh(self.delta[key](Z.detach()))
            u = u + dlt[..., 0]
            v = v + dlt[..., 1]

        GG = self.global_grid
        u = u.clamp(0, GG - 1 - 1e-4)
        v = v.clamp(0, GG - 1 - 1e-4)

        u0, v0 = u.floor().detach(), v.floor().detach()
        fu, fv = u - u0, v - v0
        u0, v0 = u0.long(), v0.long()

        out = Z.new_zeros(Kc, GG * GG, d)
        mass = Z.new_zeros(Kc, GG * GG)

        for dy in (0, 1):
            for dx in (0, 1):
                w = (fu if dx else 1 - fu) * (fv if dy else 1 - fv)
                cell = (v0 + dy) * GG + (u0 + dx)
                out = out.scatter_add(1, cell.unsqueeze(-1).expand(-1, -1, d), w.unsqueeze(-1) * Z)
                mass = mass.scatter_add(1, cell, w)

        if norm == "avg":
            out = out / mass.clamp_min(1e-6).unsqueeze(-1)
        return out, mass

    def forward_loss(self, g_t, g_s, schedules_dict, norm="avg", use_gg=False,
                     n_mid=2, n_loc=4, mid_frac=(0.4, 0.7), loc_frac=(0.2, 0.4),
                     temp_student=0.1, center_mom=0.9):
        """
        Compute progressive distillation loss between Teacher Global view and Student Multi-crops.
        """
        B = g_t.size(0)
        GG = self.global_grid

        # 1. Teacher forward pass
        with torch.no_grad():
            zt = self.teacher(g_t)[0]
            lt = self.teacher_head(zt.float())  # (B, 196, K)
            q = F.softmax((lt - self.center) / schedules_dict["tau_t"], dim=-1)
            logq = torch.log(q.clamp_min(1e-8))

            # Monitor representation entropy and prototype utilization
            ent = float(-(q * logq).sum(-1).mean())
            pbar = q.mean(dim=(0, 1))
            marg = float(-(pbar * torch.log(pbar.clamp_min(1e-8))).sum())

            # Update center buffer
            self.center.mul_(center_mom).add_(lt.mean(dim=(0, 1), keepdim=True), alpha=1 - center_mom)

        def kl_masked(ls, vis, q_rep, logq_rep):
            logp = F.log_softmax(ls / temp_student, dim=-1)
            kl = (q_rep * (logq_rep - logp)).sum(-1)
            vis = vis.float()
            # Area-normalized token KL loss (Paper Eq. 15)
            return ((kl * vis).sum(1) / vis.sum(1).clamp_min(1.0)).mean()

        losses = {}
        bidx = torch.arange(B, device=g_s.device)

        # 2. Student Multi-Crop branches
        configs = [
            ("mid", n_mid, mid_frac, 160, 10),
            ("loc", n_loc, loc_frac, 96, 6)
        ]
        for key, n, frac, size, grid in configs:
            boxes = sample_boxes(B, n, frac, g_s.device, global_size=224)
            rois = torch.cat([
                bidx.repeat_interleave(n).float().unsqueeze(1),
                boxes[:, 0:1], boxes[:, 1:2],
                boxes[:, 0:1] + boxes[:, 2:3], boxes[:, 1:2] + boxes[:, 2:3]
            ], dim=1)

            crops = roi_align(g_s.float(), rois, output_size=(size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            z = self.enc(crops)[0]
            z_on_global, mass = self.transport(z, boxes, grid, key, norm=norm)
            ls = self.head(z_on_global)

            losses[key] = kl_masked(
                ls, mass > 1e-4,
                q.repeat_interleave(n, 0),
                logq.repeat_interleave(n, 0)
            )

        total_loss = (schedules_dict["lam_m"] * losses["mid"] +
                      schedules_dict["lam_l"] * losses["loc"])

        if use_gg:
            zg = self.enc(g_s)[0]
            lg = self.head(zg.float())
            losses["glob"] = kl_masked(lg, torch.ones(B, GG * GG, dtype=torch.bool, device=g_s.device), q, logq)
            total_loss = total_loss + schedules_dict["lam_g"] * losses["glob"]

        stats = {k: float(v.detach()) for k, v in losses.items()}
        stats["loss"] = float(total_loss.detach())
        stats["entropy"] = ent
        stats["proto_entropy"] = marg
        return total_loss, stats
