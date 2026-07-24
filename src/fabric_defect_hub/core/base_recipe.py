"""Base definitions and contract for Model-Specific Optimization Recipes (MORR Engine).

Each recipe encapsulates paper-driven fine-tuning mechanisms, specialized loss functions,
hyperparameter strategies, and architectural adaptations for specific models in FabricDefectHub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseModelRecipe(ABC):
    """Abstract base class for paper-driven model fine-tuning and optimization recipes."""

    @property
    @abstractmethod
    def recipe_id(self) -> str:
        """Unique identifier for this optimization recipe (e.g., 'yolov8_sd_attn')."""
        pass

    @property
    @abstractmethod
    def target_models(self) -> List[str]:
        """List of model names/patterns this recipe applies to (e.g., ['yolov8n', 'yolov8s'])."""
        pass

    @property
    @abstractmethod
    def academic_nomenclature(self) -> str:
        """Academic nomenclature & acronym (e.g., 'SD-Attn (Small-Defect Attention Enhancement)')."""
        pass

    @property
    @abstractmethod
    def paper_reference(self) -> str:
        """Academic paper citation for this recipe's underlying method."""
        pass

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Returns the default hyperparameter search space / optimal settings for this recipe."""
        pass

    def adapt_architecture(self, model: Any) -> Any:
        """Applies model-level architectural modifications (e.g., attaching PEFT adapters,

        replacing conv layers with defect-aware attention modules). Default is identity.
        """
        return model

    def configure_loss(self, **kwargs) -> Any:
        """Configures and returns the specialized loss function for this recipe."""
        return None

    def configure_optimizer(self, model: Any, lr: float = 1e-3, weight_decay: float = 1e-4) -> Any:
        """Configures the optimal optimizer instance for this recipe."""
        return None

    def configure_augmentations(self, img_size: Tuple[int, int] = (256, 256)) -> Any:
        """Configures fabric-specific data augmentation pipeline for this recipe."""
        return None

    def get_recipe_summary(self) -> Dict[str, Any]:
        """Returns a structured summary of this recipe for logging and paper artifact generation."""
        return {
            "recipe_id": self.recipe_id,
            "target_models": self.target_models,
            "academic_nomenclature": self.academic_nomenclature,
            "paper_reference": self.paper_reference,
            "hyperparameters": self.get_default_hyperparameters(),
        }
