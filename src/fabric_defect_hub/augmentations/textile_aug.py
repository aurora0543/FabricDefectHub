"""Textile Periodic Pattern-Aware Augmentation (TPA Module).

Implements specialized data augmentations for textile defect detection that preserve
warp/weft grid periodicity while enhancing defect localization capabilities.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch


class TextilePeriodicAugmenter:
    """Textile Periodic Pattern-Aware Augmentation (TPA).

    Applies frequency-domain phase shifts and grid-aware texture perturbation
    to simulate production-line lighting variations and subtle fabric elastic stretch.
    """

    def __init__(
        self,
        grid_freq: int = 16,
        phase_shift_prob: float = 0.5,
        texture_noise_std: float = 0.03,
    ) -> None:
        self.grid_freq = grid_freq
        self.phase_shift_prob = phase_shift_prob
        self.texture_noise_std = texture_noise_std

    def __call__(self, img_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

            img_tensor: Image tensor of shape [C, H, W] in range [0, 1]
            mask_tensor: Optional binary mask tensor of shape [1, H, W] or [H, W]

        Returns:
            Augmented (image_tensor, mask_tensor)
        """
        augmented = img_tensor.clone()

        if random.random() < self.phase_shift_prob:
            # 1. Frequency-Domain Phase Shift (Simulates periodic texture shift)
            _, h, w = augmented.shape
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 2 * np.pi, h, device=augmented.device),
                torch.linspace(0, 2 * np.pi, w, device=augmented.device),
                indexing="ij",
            )
            sin_pattern = torch.sin(self.grid_freq * grid_x) * torch.cos(self.grid_freq * grid_y)
            sin_pattern = sin_pattern.unsqueeze(0) * self.texture_noise_std

            augmented = torch.clamp(augmented + sin_pattern, 0.0, 1.0)

        # 2. Add subtle Gaussian texture noise on normal background regions
        if self.texture_noise_std > 0:
            noise = torch.randn_like(augmented) * self.texture_noise_std
            if mask_tensor is not None:
                # Mask out defect region from noise perturbation
                inv_mask = (1.0 - mask_tensor.float()).unsqueeze(0) if mask_tensor.dim() == 2 else (1.0 - mask_tensor.float())
                noise = noise * inv_mask
            augmented = torch.clamp(augmented + noise, 0.0, 1.0)

        return augmented, mask_tensor
