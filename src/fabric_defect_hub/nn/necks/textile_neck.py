"""Textile Attention Necks for Fabric Defect Detection (TextileAttentionNeck).

Implements plug-and-play attention necks (SD-Attn, CBAM, Coordinate Attention)
engineered to enhance microscopic fabric defect representation.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabric_defect_hub.nn.necks.base_neck import BaseNeck


class SpaceToDepthModule(nn.Module):
    """Space-to-Depth (SPD) Conv Downsampling for preserving tiny defect pixels."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> split into 4 sub-tensors
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        x_concat = torch.cat([x0, x1, x2, x3], dim=1)
        return self.act(self.bn(self.conv(x_concat)))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention Map
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        c_attn = self.sigmoid(avg_out + max_out)
        x = x * c_attn

        # Spatial Attention Map
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        s_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * s_attn


class TextileAttentionNeck(BaseNeck):
    """Pluggable Textile Attention Neck supporting SD-Attn, CBAM, and Feature Concatenation."""

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        mode: str = "sd_attn",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.out_channels = out_channels

        self.adapt_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])

        if mode == "sd_attn":
            self.attn_blocks = nn.ModuleList([
                SpaceToDepthModule(out_channels, out_channels) for _ in in_channels_list
            ])
        elif mode == "cbam":
            self.attn_blocks = nn.ModuleList([
                CBAMBlock(out_channels) for _ in in_channels_list
            ])
        else:
            self.attn_blocks = nn.ModuleList([nn.Identity() for _ in in_channels_list])

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        enhanced_features = []

        for idx, (layer_name, fmap) in enumerate(feature_maps.items()):
            if idx >= len(self.adapt_convs):
                break
            x = self.adapt_convs[idx](fmap)
            x_enhanced = self.attn_blocks[idx](x)
            enhanced_features.append(x_enhanced)

        return enhanced_features
