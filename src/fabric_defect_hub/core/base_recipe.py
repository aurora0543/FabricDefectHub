"""Base contract for a model *config profile*.

A profile is an honest, named bundle of run settings for one method: the
hyperparameters we run it with, plus optional loss / augmentation / architecture
hooks, all anchored to the method's real upstream paper via `paper_reference`.
It is NOT a novel research contribution and carries no invented acronym — it is
the "these are the settings, and here is the paper they come from" seam that
`load_model(recipe=...)` feeds into training/inference. If and when a profile
grows a genuine, measured modification of its own, that earned change can be
named then — not before.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseModelRecipe(ABC):
    """Abstract base class for a model config profile."""

    @property
    @abstractmethod
    def recipe_id(self) -> str:
        """Unique identifier for this profile (e.g., 'patchcore')."""
        pass

    @property
    @abstractmethod
    def target_models(self) -> List[str]:
        """List of model names/patterns this profile applies to (e.g., ['yolov8n', 'yolov8s'])."""
        pass

    @property
    @abstractmethod
    def paper_reference(self) -> str:
        """Citation for the real upstream method this profile's settings come from."""
        pass

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """The default settings for this profile, expressed in the backend's real vocabulary."""
        pass

    def adapt_architecture(self, model: Any) -> Any:
        """Optional model-level modification hook. Default is identity — override
        only when the profile makes a *real* architectural change (and then
        prove its effect before naming it).
        """
        return model

    def configure_loss(self, **kwargs) -> Any:
        """Optional loss module for this profile. Default None (backend's own loss)."""
        return None

    def configure_optimizer(self, model: Any, lr: float = 1e-3, weight_decay: float = 1e-4) -> Any:
        """Optional optimizer for this profile. Default None (backend's own optimizer)."""
        return None

    def configure_augmentations(self, img_size: Tuple[int, int] = (256, 256)) -> Any:
        """Optional augmentation pipeline for this profile. Default None."""
        return None

    def get_recipe_summary(self) -> Dict[str, Any]:
        """A structured summary of this profile for logging and run provenance."""
        return {
            "recipe_id": self.recipe_id,
            "target_models": self.target_models,
            "paper_reference": self.paper_reference,
            "hyperparameters": self.get_default_hyperparameters(),
        }
