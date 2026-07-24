"""Paper-Driven Recipe for Vision-Language Zero-Shot Models (MoECLIP, WinCLIP).

Implements Text Prompt Optimization & PEFT Adapter Tuning (TPO-PEFT) for cross-domain zero-shot defect transfer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("moeclip_tpo_peft")
class MoECLIPRecipe(BaseModelRecipe):
    """Optimization Recipe for MoECLIP and WinCLIP Zero-Shot Vision-Language models."""

    @property
    def recipe_id(self) -> str:
        return "moeclip_tpo_peft"

    @property
    def target_models(self) -> List[str]:
        return ["moeclip", "winclip"]

    @property
    def academic_nomenclature(self) -> str:
        return "TPO-PEFT (Text Prompt Optimization & LoRA Parameter-Efficient Adapter)"

    @property
    def paper_reference(self) -> str:
        return "Cao et al., 'MoE-CLIP: Mixture-of-Experts for Zero-Shot Anomaly Detection', 2024."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "prompt_templates": [
                "a photo of a {category} fabric with a {defect} defect.",
                "flawless uniform {category} fabric texture.",
                "damaged {category} cloth showing {defect}.",
            ],
            "expert_num": 4,
            "routing_temperature": 0.5,
        }

    def configure_loss(self, **kwargs) -> Any:
        return None
