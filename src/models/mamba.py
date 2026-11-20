"""
Bidirectional Vision Mamba (Bi-Mamba) Architecture.
Faithful implementation of the Bidirectional SSM encoder from PSMamba (Eq. 7).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mambapy.mamba import MambaBlock, MambaConfig
    HAS_MAMBAPY = True
except ImportError:
    HAS_MAMBAPY = False


class PurePyTorchSSM(nn.Module):
    """
    Pure PyTorch fallback for 1D Selective State Space Model when mambapy is not installed.
    Guarantees that the code runs out-of-the-box in any Python environment.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        # u: (B, L, D)
        B, L, _ = u.shape
        xz = self.in_proj(u)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # 1D Depthwise Conv
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        # Selective SSM parameters
        ssm_proj = self.x_proj(x_act)
        delta, B_ssm, C_ssm = torch.split(ssm_proj, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # Discrete scan (pure PyTorch sequential loop)
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            xt = x_act[:, t, :]  # (B, d_inner)
            dt = delta[:, t, :]  # (B, d_inner)
            Bt = B_ssm[:, t, :]  # (B, d_state)
            Ct = C_ssm[:, t, :]  # (B, d_state)

            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, Bt)
            h = h * dA + dB * xt.unsqueeze(-1)
            y = torch.einsum("bdn,bn->bd", h, Ct) + self.D * xt
            ys.append(y)

        y_seq = torch.stack(ys, dim=1)  # (B, L, d_inner)
        out = y_seq * F.silu(z)
        return self.out_proj(out)


def make_block(d):
    if HAS_MAMBAPY:
        return MambaBlock(MambaConfig(
            d_model=d,
            n_layers=1,
            d_state=16,
            expand_factor=2,
            d_conv=4,
            pscan=True,
            use_cuda=False
        ))
    return PurePyTorchSSM(d_model=d, d_state=16, d_conv=4, expand=2)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block with learnable fusion weights (Paper Eq. 7).
    fwd scan + bwd scan fused via learnable scalar weights wf, wb.
    """
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fwd = make_block(d)
        self.bwd = make_block(d)
        self.wf = nn.Parameter(torch.tensor(0.5))
        self.wb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        h = self.norm(x)
        fwd_out = self.fwd(h)
        bwd_out = self.bwd(h.flip(1)).flip(1)
        return x + self.wf * fwd_out + self.wb * bwd_out


class MiniBiMamba(nn.Module):
    """
    Bidirectional Vision Mamba backbone for 2D images.
    Projects 16x16 non-overlapping patches and passes through depth BiMambaBlocks.
    """
    def __init__(self, d=192, depth=8, patch=16, base_grid=14):
        super().__init__()
        self.patch = patch
        self.d = d
        self.embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, d, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([BiMambaBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, img):
        t = self.embed(img)
        gh, gw = t.shape[-2:]
        pos = F.interpolate(self.pos, size=(gh, gw), mode="bicubic", align_corners=False)
        t = (t + pos).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t), (gh, gw)


class Classifier(nn.Module):
    """
    From-scratch Bi-Mamba Classifier.
    """
    def __init__(self, n_classes, d=192, depth=8):
        super().__init__()
        self.enc = MiniBiMamba(d, depth)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        tokens, _ = self.enc(x)
        return self.head(tokens.mean(1))
