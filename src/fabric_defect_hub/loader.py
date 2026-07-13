"""The single entry point the rest of the project (CLI, frontend backend,
notebooks) is meant to call: resolve a dataset + model by name, run the
train/predict/evaluate/profile lifecycle, and return a unified
`ExperimentResult`.

This is the "basic core" referenced in the README's Phase 1 goal:
`YOLOv8n + one dataset + PC inference + result JSON`. Concrete dataset and
model backends register themselves with `fabric_defect_hub.core.registry`;
this module only knows the registry, never a specific framework.
"""

from __future__ import annotations

import importlib
from typing import Any

from fabric_defect_hub.core.registry import get_dataset_cls, get_model_cls
from fabric_defect_hub.core.serialization import save_experiment_result, save_predictions
from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import ModelAdapter

# Backend package import paths, used to lazily register model/dataset
# implementations without requiring every optional framework to be
# installed just to import this module.
_MODEL_BACKEND_MODULES = {
    "ultralytics": "fabric_defect_hub.models.ultralytics",
    "torchvision": "fabric_defect_hub.models.torchvision",
    "anomalib": "fabric_defect_hub.models.anomalib",
}


def load_dataset(name: str, root: str, split: str = "test", **kwargs) -> DatasetAdapter:
    """Resolve and instantiate a registered `DatasetAdapter` by name."""

    importlib.import_module("fabric_defect_hub.datasets")  # triggers @register_dataset
    cls = get_dataset_cls(name)
    return cls(root=root, split=split, **kwargs)


def load_model(backend: str, name: str, **kwargs) -> ModelAdapter:
    """Resolve and instantiate a registered `ModelAdapter` by backend + name."""

    module_path = _MODEL_BACKEND_MODULES.get(backend)
    if module_path is not None:
        importlib.import_module(module_path)  # triggers @register_model
    cls = get_model_cls(backend)
    return cls(name=name, **kwargs)


def run_experiment(
    experiment_id: str,
    dataset: DatasetAdapter,
    model: ModelAdapter,
    model_info: ModelInfo,
    runtime: RuntimeInfo,
    train_config: dict[str, Any] | None = None,
    evaluator=None,
    output_dir: str | None = None,
) -> ExperimentResult:
    """Run the minimal end-to-end loop: (train) -> predict -> evaluate.

    `train_config=None` skips training and assumes `model` already carries
    a usable artifact (e.g. a pretrained checkpoint loaded in `__init__`).
    Profiling is deliberately left out here: it targets exported/deployed
    artifacts and is invoked separately via `fabric_defect_hub.profiling`.

    `output_dir`, if given, persists `predictions.json` and `result.json`
    there (matching `schemas/prediction.schema.json` /
    `experiment_result.schema.json` — see `core.serialization`) and records
    their paths in the returned `ExperimentResult.artifacts`. This is what
    a leaderboard/frontend would actually read from disk; without it you
    only get the in-memory `ExperimentResult`.
    """

    samples = dataset.load_samples()

    artifact = model.train(train_config) if train_config is not None else None
    predictions = model.predict(samples, artifact)

    metrics = evaluator.evaluate(samples, predictions) if evaluator is not None else {}

    artifacts: dict[str, str] = {}
    if artifact is not None:
        artifacts["model"] = artifact.path
    if output_dir is not None:
        base = f"{output_dir.rstrip('/')}/{experiment_id}"
        artifacts["predictions"] = str(save_predictions(predictions, f"{base}/predictions.json"))

    result = ExperimentResult(
        experiment_id=experiment_id,
        model=model_info,
        dataset=DatasetInfo(name=dataset.name, split=dataset.split),
        runtime=runtime,
        metrics=metrics,
        artifacts=artifacts,
    )

    if output_dir is not None:
        result_path = save_experiment_result(result, f"{output_dir.rstrip('/')}/{experiment_id}/result.json")
        result.artifacts["result"] = str(result_path)

    return result
