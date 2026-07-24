"""Paper-Driven Recipe for YOLO Series (YOLOv8, YOLOv11).

Implements Small-Defect Attention (SD-Attn / SPD-Conv) integration and
Dynamic Anchor Loss Scaling tuned for microscopic fabric defects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from fabric_defect_hub.augmentations.textile_aug import TextilePeriodicAugmenter
from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe
from fabric_defect_hub.optim.losses import AFDLoss, DynamicLossScaler


@register_recipe("yolov8_sd_attn")
class YOLOv8Recipe(BaseModelRecipe):
    """Optimization Recipe for YOLOv8 & YOLO11 tuned for fabric defect inspection."""

    @property
    def recipe_id(self) -> str:
        return "yolov8_sd_attn"

    @property
    def target_models(self) -> List[str]:
        return ["yolov8n", "yolov8s", "yolov11n", "yolov11s"]

    @property
    def academic_nomenclature(self) -> str:
        return "SD-Attn & DLW (Small-Defect Attention + Dynamic Loss Weighting)"

    @property
    def paper_reference(self) -> str:
        return "Sarr et al., 'Space-to-Depth Downsampling for Micro-Object Detection', IEEE T-CSVT 2023."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "box_loss_weight": 7.5,
            "cls_loss_weight": 0.5,
            "dfl_loss_weight": 1.5,
            "spd_conv_downsample": True,
            "fabric_aug_enabled": True,
        }

    def configure_loss(self, **kwargs) -> Any:
        return DynamicLossScaler(
            num_losses=3,
            init_weights=[7.5, 0.5, 1.5],
        )

    def configure_augmentations(self, img_size: Tuple[int, int] = (640, 640)) -> Any:
        return TextilePeriodicAugmenter(grid_freq=16, phase_shift_prob=0.4, texture_noise_std=0.02)
