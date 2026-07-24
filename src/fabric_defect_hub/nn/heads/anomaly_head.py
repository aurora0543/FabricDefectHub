"""Task Heads and Anomaly Heatmap Decoders for FabricDefectHub (anomaly_head.py).

Provides in-house defect segmentation heads and anomaly heatmap decoders.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectSegmentationHead(nn.Module):
    """Multi-Scale Feature Mask Segmentation Head for Defect Mask Generation."""

    def __init__(self, in_channels: int = 256, num_classes: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """Upsamples and merges multi-scale features to target image size."""
        if not features:
            raise ValueError("Empty feature list passed to DefectSegmentationHead.")

        f_fused = torch.zeros_like(features[0])
        for f in features:
            if f.shape[2:] != f_fused.shape[2:]:
                f_resized = F.interpolate(f, size=f_fused.shape[2:], mode="bilinear", align_corners=False)
            else:
                f_resized = f
            f_fused = f_fused + f_resized

        out = self.relu(self.bn1(self.conv1(f_fused)))
        logits = self.conv2(out)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class AnomalyHeatmapDecoder:
    """Decodes memory bank embeddings or feature distance matrices into smoothed anomaly heatmaps."""

    @staticmethod
    def decode(
        feature_maps: List[torch.Tensor],
        target_size: Tuple[int, int],
        sigma: float = 4.0,
    ) -> torch.Tensor:
        """Computes pixel-wise L2 norm anomaly maps across scales and applies Gaussian smoothing."""
        maps = []
        for f in feature_maps:
            # Calculate pixel L2 norm as raw anomaly activation map
            norm_map = torch.norm(f, p=2, dim=1, keepdim=True)
            norm_map = F.interpolate(norm_map, size=target_size, mode="bilinear", align_corners=False)
            maps.append(norm_map)

        fused_map = torch.stack(maps, dim=0).mean(dim=0)
        return fused_map
