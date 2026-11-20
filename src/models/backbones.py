"""
Standard and Modern Vision Backbones for Reference & Attribution Benchmarking:
- ResNet-50 (He et al., 2016)
- ConvNeXt-Tiny (Liu et al., 2022)
- Swin Transformer Tiny (Liu et al., 2021)
"""

import torch.nn as nn
import torchvision


def build_resnet50(n_classes, pretrained=True):
    weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_convnext_tiny(n_classes, pretrained=True):
    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = torchvision.models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
    return model


def build_swin_t(n_classes, pretrained=True):
    weights = torchvision.models.Swin_T_Weights.DEFAULT if pretrained else None
    model = torchvision.models.swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, n_classes)
    return model
