"""Autonomous Backbone Provider for FabricDefectHub (backbones.py).

Provides lightweight wrappers around torchvision/timm pretrained weights
while allowing full in-house control over forward passes and feature interceptors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def get_backbone(name: str = "resnet18", pretrained: bool = True) -> Tuple[nn.Module, List[str]]:
    """Loads a pretrained backbone and returns default layer names for multi-scale feature extraction.

    Returns:
        Tuple[nn.Module, List[str]]: (backbone_module, default_feature_layer_names)
    """
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError("torchvision is required to load pretrained backbone weights.")

    name = name.lower()

    if name in ("resnet18", "resnet-18"):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        default_layers = ["layer1", "layer2", "layer3", "layer4"]
    elif name in ("resnet50", "resnet-50"):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        default_layers = ["layer1", "layer2", "layer3", "layer4"]
    elif name in ("wideresnet50", "wide_resnet50_2"):
        weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        model = models.wide_resnet50_2(weights=weights)
        default_layers = ["layer2", "layer3"]
    elif name in ("efficientnet_b0", "efficientnet-b0"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        default_layers = ["features.3", "features.5", "features.7"]
    else:
        # Generic ResNet fallback
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        default_layers = ["layer2", "layer3"]

    return model, default_layers
