"""
Hybrid Architecture: Pretrained ResNet-50 Inductive Tokenizer + Bi-Mamba Blocks.
Combines 2D CNN local inductive biases with long-range Bidirectional SSM sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .mamba import BiMambaBlock


class ResNetTokenizer(nn.Module):
    """
    ImageNet-pretrained ResNet-50 up to layer3 (stride 16), followed by a 1x1 projection
    to the Mamba latent dimension d.
    Replaces 16x16 patch-embedding conv:
      224 px -> 14x14 tokens
      160 px -> 10x10 tokens
      96 px  -> 6x6 tokens
    """
    def __init__(self, d=192, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool,
            r.layer1, r.layer2, r.layer3
        )
        self.proj = nn.Conv2d(1024, d, kernel_size=1)

    def forward(self, x):
        return self.proj(self.stem(x))  # (B, d, gh, gw)


class HybridMamba(nn.Module):
    """
    Proposed Hybrid Model:
    1. Pretrained ResNet-50 Tokenizer extracts rich inductive 2D local representations.
    2. Continuous 2D bicubic positional embeddings are added.
    3. Bi-Mamba blocks process spatial tokens bidirectionally.
    4. Global average pooling + linear classification head.
    """
    def __init__(self, n_classes, d=192, depth=4, base_grid=14, pretrained_backbone=True):
        super().__init__()
        self.tok = ResNetTokenizer(d=d, pretrained=pretrained_backbone)
        self.pos = nn.Parameter(torch.zeros(1, d, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([BiMambaBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, img):
        t = self.tok(img)
        gh, gw = t.shape[-2:]
        pos = F.interpolate(self.pos, size=(gh, gw), mode="bicubic", align_corners=False)
        t = (t + pos).flatten(2).transpose(1, 2)  # (B, L, d)
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t).mean(1))
