"""Paper-Driven Recipe for Memory Bank Anomaly Detectors (PatchCore, PaDiM, SuperSimpleNet).

Implements Domain-Specific Memory Bank Adaptation (DMBA) with adaptive coreset subsampling,
Gaussian smoothing, and WideResNet-50 / DINOv2 self-supervised backbone selection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("patchcore_dmba")
class PatchCoreRecipe(BaseModelRecipe):
    """Optimization Recipe for PatchCore / PaDiM / SuperSimpleNet."""

    @property
    def recipe_id(self) -> str:
        return "patchcore_dmba"

    @property
    def target_models(self) -> List[str]:
        return ["patchcore", "padim", "supersimplenet"]

    @property
    def academic_nomenclature(self) -> str:
        return "DMBA (Domain-Specific Memory Bank Adaptation & Adaptive Coreset Subsampling)"

    @property
    def paper_reference(self) -> str:
        return "Roth et al., 'Towards Total Recall in Industrial Anomaly Detection', CVPR 2022."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "backbone": "wideresnet50",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "n_neighbors": 9,
            "gaussian_sigma": 4.0,
            "anomaly_score_smoothing": True,
            "pretrain_source": "imagenet-22k",
        }

    def configure_loss(self, **kwargs) -> Any:
        # One-class memory bank models don't use standard backprop loss during training
        return None

    def configure_augmentations(self, img_size: Tuple[int, int] = (256, 256)) -> Any:
        return None
