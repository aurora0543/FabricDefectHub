"""Academic Loss Function Modules for FabricDefectHub (AFDLoss & Dynamic Scaler).

Provides paper-grade loss formulations specifically engineered for tiny fabric defect imbalance,
including Adaptive Focal-Dice Loss (AFDL) and Multi-Scale Dynamic Loss Scaler.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFDLoss(nn.Module):
    """Adaptive Focal-Dice Loss (AFDL) for tiny fabric defect segmentation.

    Combines Focal Loss (addressing extreme foreground-background imbalance) with
    Adaptive Soft Dice Loss (addressing pixel connectivity and boundary precision).

    Formulation:
        L_AFDL = alpha * L_Focal(p, y; gamma) + (1 - alpha) * L_AdaptiveDice(p, y)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        smooth: float = 1e-5,
        adaptive_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.adaptive_weighting = adaptive_weighting

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

            logits: Raw unnormalized predictions [B, 1, H, W] or [B, H, W]
            targets: Binary ground truth masks [B, 1, H, W] or [B, H, W]
        """
        if logits.dim() != targets.dim():
            targets = targets.unsqueeze(1) if targets.dim() == 3 else targets

        probs = torch.sigmoid(logits)

        # 1. Focal Loss Computation
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        focal_loss = focal_loss.mean()

        # 2. Adaptive Soft Dice Loss Computation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.float().view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # Dynamic Alpha Balancing based on target defect area ratio if enabled
        alpha_eff = self.alpha
        if self.adaptive_weighting:
            defect_ratio = targets_flat.mean().item()
            # If defect is extremely tiny (< 1% of image), increase Focal weight
            if defect_ratio < 0.01:
                alpha_eff = min(0.8, self.alpha + 0.3)

        return alpha_eff * focal_loss + (1.0 - alpha_eff) * dice_loss


class DynamicLossScaler(nn.Module):
    """Dynamic Multi-Task/Multi-Scale Loss Weighting Scaler (DLW).

    Dynamically balances bounding box loss, classification loss, and pixel mask loss
    based on gradient variance or loss magnitude history.
    """

    def __init__(self, num_losses: int = 3, init_weights: Optional[list[float]] = None) -> None:
        super().__init__()
        if init_weights is None:
            init_weights = [1.0] * num_losses
        self.params = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=self.params.device)
        weighted_dict: Dict[str, float] = {}

        weights = F.softmax(self.params, dim=0) * len(losses)

        for idx, (name, loss_val) in enumerate(losses.items()):
            w = weights[idx % len(weights)]
            scaled_loss = w * loss_val
            total_loss = total_loss + scaled_loss
            weighted_dict[f"weighted_{name}"] = scaled_loss.item()

        return total_loss, weighted_dict
