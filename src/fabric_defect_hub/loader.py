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
import inspect
import math
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.registry import get_dataset_cls, get_model_cls
from fabric_defect_hub.core.serialization import save_experiment_result, save_predictions
from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import Artifact, ModelAdapter

# Backend package import paths, used to lazily register model/dataset
# implementations without requiring every optional framework to be
# installed just to import this module.
_MODEL_BACKEND_MODULES = {
    "ultralytics": "fabric_defect_hub.models.ultralytics",
    "torchvision": "fabric_defect_hub.models.torchvision.adapter",
    "anomalib": "fabric_defect_hub.models.anomalib",
    "dinomaly": "fabric_defect_hub.models.dinomaly",
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
    artifact: Artifact | None = None,
    profiler=None,
    profile_config=None,
    export_target: str | None = None,
    export_config: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run the end-to-end loop: (train/load) -> predict -> evaluate -> profile.

    `train_config=None` skips training and assumes `model` already carries
    a usable artifact (e.g. a pretrained checkpoint loaded in `__init__`).
    Pass `artifact` to evaluate an existing checkpoint without training.
    Profiling is opt-in and requires `profiler`, `profile_config`, and
    `export_target`; the active checkpoint is exported, profiled, and its
    runtime metrics are merged into the same `ExperimentResult`.

    `output_dir`, if given, persists `predictions.json` and `result.json`
    there (matching `schemas/prediction.schema.json` /
    `experiment_result.schema.json` — see `core.serialization`) and records
    their paths in the returned `ExperimentResult.artifacts`. This is what
    a leaderboard/frontend would actually read from disk; without it you
    only get the in-memory `ExperimentResult`.
    """

    samples = dataset.load_samples()

    active_artifact = model.train(train_config) if train_config is not None else artifact
    predictions = model.predict(samples, active_artifact)

    evaluated_metrics = evaluator.evaluate(samples, predictions) if evaluator is not None else {}
    metrics = {
        name: float(value)
        for name, value in evaluated_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }

    artifacts: dict[str, str] = {}
    if active_artifact is not None:
        artifacts["model"] = active_artifact.path

    if profiler is not None:
        if active_artifact is None:
            raise ValueError("profiling requires a trained or explicitly supplied model artifact")
        if profile_config is None or export_target is None:
            raise ValueError("profiling requires both profile_config and export_target")
        export_kwargs: dict[str, Any] = {"target": export_target}
        if "config" in inspect.signature(model.export).parameters:
            export_kwargs["config"] = export_config or {}
        exported = model.export(active_artifact, **export_kwargs)
        export_path = Path(exported.path)
        if not export_path.is_file():
            raise FileNotFoundError(f"exported model does not exist: {export_path}")
        profile_metrics = profiler.profile(exported, profile_config)
        profile_metrics["model_size_mb"] = export_path.stat().st_size / (1024 * 1024)
        metrics.update(profile_metrics)
        runtime = profiler.runtime_info(profile_config)
        artifacts[f"model_{export_target}"] = str(export_path)
    if output_dir is not None:
        base = f"{output_dir.rstrip('/')}/{experiment_id}"
        artifacts["predictions"] = str(save_predictions(predictions, f"{base}/predictions.json"))
        power_report = getattr(profiler, "last_power_report", None) if profiler is not None else None
        if power_report is not None:
            from fabric_defect_hub.profiling.power import save_power_report

            artifacts["power_measurement"] = str(save_power_report(power_report, f"{base}/power.json"))

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
