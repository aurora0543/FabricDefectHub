"""Paper-Driven Recipe for Teacher-Student Distillation Models (RD4AD, EfficientAD).

Implements Multi-Scale Feature Alignment Distillation (MSFA-D) with temperature annealing,
dynamic Cosine-L2 feature matching, and multi-resolution anomaly distillation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("rd4ad_msfa_d")
class RD4ADRecipe(BaseModelRecipe):
    """Optimization Recipe for RD4AD and EfficientAD Teacher-Student networks."""

    @property
    def recipe_id(self) -> str:
        return "rd4ad_msfa_d"

    @property
    def target_models(self) -> List[str]:
        return ["rd4ad", "efficientad"]

    @property
    def academic_nomenclature(self) -> str:
        return "MSFA-D (Multi-Scale Feature Alignment Distillation with Cosine-L2 Match)"

    @property
    def paper_reference(self) -> str:
        return "Tien et al., 'Anomaly Detection via Reverse Distillation from One-Class Embedding', CVPR 2022."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lr": 0.005,
            "weight_decay": 0.0001,
            "distillation_temp": 2.0,
            "temp_decay_rate": 0.95,
            "feature_alignment_layers": ["layer1", "layer2", "layer3"],
            "cosine_loss_weight": 0.7,
            "l2_loss_weight": 0.3,
        }

    def configure_loss(self, **kwargs) -> Any:
        return None
