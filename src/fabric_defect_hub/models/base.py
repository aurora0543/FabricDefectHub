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

    def raw_module(self) -> Any | None:
        """The live, mutable `torch.nn.Module` behind this adapter's
        currently-loaded model, if any -- for callers that need to
        instrument the model itself rather than an exported copy (e.g.
        FLOPs counting via `profiling.flops.compute_model_flops`, which
        needs to attach hooks a frozen `torch.jit.ScriptModule` export
        can't accept). `None` when nothing is loaded yet.

        The default returns the conventional `self._model` attribute every
        concrete adapter already keeps its loaded model under; override
        this when that attribute isn't a raw module (see
        `UltralyticsAdapter`, whose `_model` is a `YOLO` wrapper around the
        real `torch.nn.Module`).
        """

        return getattr(self, "_model", None)
