"""ModelAdapter: unifies train/predict/export across Ultralytics,
torchvision and Anomalib, while letting each backend keep its native config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Prediction, Sample


@dataclass
class Artifact:
    """Opaque handle to trained weights, produced by `train()`."""

    path: str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportedArtifact:
    """Opaque handle to an exported/converted model, produced by `export()`."""

    path: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Base class every concrete model backend (YOLO, Faster R-CNN, PatchCore, ...) implements."""

    name: str
    backend: str

    def __init__(self, name: str | None = None, **kwargs):
        if name is not None:
            self.name = name
        self.options = kwargs

    @abstractmethod
    def train(self, config: dict[str, Any]) -> Artifact:
        """Fit the model per `config`, return a handle to the resulting weights."""

    @abstractmethod
    def predict(self, samples: list[Sample], artifact: Artifact) -> list[Prediction]:
        """Run inference, returning one `Prediction` per input `Sample`."""

    @abstractmethod
    def export(self, artifact: Artifact, target: str) -> ExportedArtifact:
        """Convert `artifact` to a deployment target (e.g. 'onnx', 'tensorrt')."""
