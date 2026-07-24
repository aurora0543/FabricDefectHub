"""Config profile for the YOLO series (YOLOv8, YOLO11) on fabric detection.

Just the training settings we run these variants with, in Ultralytics' real
argument vocabulary, anchored to the SPD-Conv paper for the small-object
motivation. No architectural change is claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from fabric_defect_hub.augmentations.textile_aug import TextilePeriodicAugmenter
from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe
from fabric_defect_hub.optim.losses import AFDLoss, DynamicLossScaler


@register_recipe("yolov8")
class YOLOv8Recipe(BaseModelRecipe):
    """Config profile for YOLOv8 & YOLO11 fabric defect detection."""

    @property
    def recipe_id(self) -> str:
        return "yolov8"

    @property
    def target_models(self) -> List[str]:
        return ["yolov8n", "yolov8s", "yolov11n", "yolov11s"]

    @property
    def paper_reference(self) -> str:
        return "Sunkara & Luo, 'No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects', ECML PKDD 2022 (arXiv:2208.03641)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Trainer knobs use Ultralytics' *real* `YOLO.train` argument names
        # (`box`/`cls`/`dfl` are its loss gains, not `*_loss_weight`), so
        # `UltralyticsAdapter.train` folds them in directly via
        # `recipe_trainer_overrides`. The two trailing flags are NOT trainer
        # args — they drive `adapt_architecture` (SPD-Conv) and
        # `configure_augmentations`, and are filtered out of the train kwargs.
        return {
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
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
