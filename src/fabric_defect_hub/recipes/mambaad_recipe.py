"""Paper-Driven Recipe for Mamba State Space Models (MambaAD).

Implements Selective Scan Texture State Space Tuning (SS-TST) for fabric periodicity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("mambaad_ss_tst")
class MambaADRecipe(BaseModelRecipe):
    """Optimization Recipe for MambaAD (Selective Scan State Space Model)."""

    @property
    def recipe_id(self) -> str:
        return "mambaad_ss_tst"

    @property
    def target_models(self) -> List[str]:
        return ["mambaad"]

    @property
    def academic_nomenclature(self) -> str:
        return "SS-TST (Selective Scan Texture State Space Tuning)"

    @property
    def paper_reference(self) -> str:
        return "He et al., 'MambaAD: Exploring State Space Models for Multi-Class Visual Anomaly Detection', 2024."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lr": 0.0001,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "scan_directions": ["horizontal", "vertical", "diagonal"],
            "texture_suppression_weight": 0.15,
        }

    def configure_loss(self, **kwargs) -> Any:
        return None
