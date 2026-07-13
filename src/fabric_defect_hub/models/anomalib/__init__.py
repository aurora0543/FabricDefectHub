"""Anomalib backend (PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet) — Phase 2.

`adapter.py` implements `ModelAdapter` via the `anomalib` package and
registers itself as `@register_model("anomalib")` on import. `presets.py`
resolves README/paper model names to anomalib classes and supplies
fabric-tailored default hyperparameters per model.
"""

from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.presets import list_supported_models

__all__ = ["AnomalibAdapter", "list_supported_models"]
