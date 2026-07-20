"""The single `predict` entry point, mirroring `training.run_train`: pick a
model config the same way `fdh train` does (a path, a filename stem under
`configs/models/`, or a model keyword), point it at a previously trained
artifact instead of training a new one, and run inference over either
explicit image paths or a slice of a registered dataset.

Before this module, every backend's `predict()`/`load_trained_model()` was
only reachable through the interactive web UI's `InferenceSessionManager`
(see `inference/session.py`) or by hand-writing a Python script — training
a model was config/CLI-driven via `fdh train`, but running it afterwards
was not. This closes that gap: every model this project can train, it can
also run inference for, from the command line.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.training import (
    DEFAULT_DATASET_ROOTS,
    DEFAULT_MODEL_CONFIG_DIR,
    apply_default_dataset_root,
    apply_model_overrides,
    infer_backend,
    load_raw_config,
    resolve_model_config,
)

# Per backend: (module, class name) of its `ModelAdapter` — kept separate
# from `training._BACKEND_PIPELINE_MODULES` (which points at the
# train/val/export *pipeline* module) because predict only ever needs the
# adapter itself.
_ADAPTER_MODULES: dict[str, tuple[str, str]] = {
    "ultralytics": ("fabric_defect_hub.models.ultralytics.adapter", "UltralyticsAdapter"),
    "torchvision": ("fabric_defect_hub.models.torchvision.adapter", "TorchvisionAdapter"),
    "anomalib": ("fabric_defect_hub.models.anomalib.adapter", "AnomalibAdapter"),
    "dinomaly": ("fabric_defect_hub.models.dinomaly.adapter", "DinomalyAdapter"),
}

# Backends whose `predict()` accepts `output_dir=` to persist pixel-level
# anomaly maps (see `_run_predict`/`--output-dir` in cli.py).
_ANOMALY_MAP_BACKENDS = {"anomalib", "dinomaly"}


@dataclass
class PredictInput:
    """Where to source `Sample`s for prediction — either raw image paths or
    a slice of a registered dataset (mutually exclusive; see `_load_samples`).
    """

    images: list[str] = field(default_factory=list)
    dataset: str | None = None
    dataset_root: str | None = None
    split: str = "test"
    num_samples: int | None = None
    pattern: str | int | None = None
    category: str | None = None
    seed: int = 0


@dataclass
class PredictRunResult:
    backend: str
    variant: str
    predictions: list[Prediction]


def _build_adapter(backend: str, variant: str):
    module_name, cls_name = _ADAPTER_MODULES[backend]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(name=variant)


def _resolve_weights_artifact(backend: str, weights: str) -> Artifact | str:
    """Anomalib's `load_trained_model` refuses a bare path — Lightning
    checkpoints can deserialize arbitrary Python objects, so it requires a
    *trusted* `Artifact` instead (see `AnomalibAdapter.load_trained_model`).
    `register_trained_model` names registry files `<ModelClass>.ckpt`
    (e.g. `Patchcore.ckpt`), so the model class is recovered from the
    filename stem when the caller just passes a path. Ultralytics/
    torchvision checkpoints embed their own architecture/variant info and
    accept a bare path directly.
    """

    if backend != "anomalib":
        return weights
    model_class = Path(weights).stem
    return Artifact(path=weights, backend="anomalib", metadata={"model_class": model_class, "trusted": True})


def _load_samples(source: PredictInput, backend: str) -> list[Sample]:
    if source.images and source.dataset:
        raise ValueError("pass either --image or --dataset, not both")
    if source.images:
        task = "anomaly" if backend in _ANOMALY_MAP_BACKENDS else "detection"
        return [
            Sample(id=Path(image_path).stem, image_path=image_path, task=task, annotations=Annotations())
            for image_path in source.images
        ]
    if not source.dataset:
        raise ValueError("pass --image (one or more) or --dataset to select what to run inference on")

    from fabric_defect_hub.loader import load_dataset

    raw = apply_default_dataset_root({"data": {"dataset": source.dataset, "dataset_root": source.dataset_root}})
    root = raw["data"]["dataset_root"]
    if not root:
        raise ValueError(
            f"no dataset_root for {source.dataset!r}; pass --dataset-root explicitly "
            f"(only {', '.join(sorted(DEFAULT_DATASET_ROOTS))} have a project default)"
        )

    kwargs: dict[str, Any] = {"seed": source.seed}
    if source.num_samples is not None:
        kwargs["num_samples"] = source.num_samples
    if source.pattern is not None:
        kwargs["pattern"] = source.pattern
    if source.category is not None:
        kwargs["category"] = source.category

    dataset = load_dataset(source.dataset, root=root, split=source.split, **kwargs)
    return dataset.load_samples()


def run_predict(
    model: str | Path,
    weights: str,
    source: PredictInput,
    backend: str | None = None,
    variant: str | None = None,
    config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
    output_dir: str | None = None,
) -> PredictRunResult:
    """The unified inference entry point, mirroring `training.run_train`.

    1. Resolve `model` to a config path exactly like `run_train` does (a
       full path, a filename stem under `config_dir`, or a model keyword
       like "yolov8n"/"patchcore"), and resolve its backend.
    2. Layer `variant` onto the model section (same mechanism as
       `training.apply_model_overrides`), so inference can target any
       variant that backend supports, not just whatever the config declares.
    3. Instantiate that backend's adapter and load `weights` — a path to a
       previously trained/registered artifact (see `fdh train`'s
       `registered_artifact.path` output).
    4. Load `source` (explicit image paths or a dataset selection) into
       `Sample`s and run `adapter.predict(...)`.

    `output_dir` is only meaningful for anomalib: it additionally persists
    each sample's pixel-level anomaly map (see `AnomalibAdapter.predict`).
    """

    model_config = resolve_model_config(str(model), config_dir=config_dir)
    raw = load_raw_config(model_config)
    resolved_backend = backend or infer_backend(raw)
    if resolved_backend not in _ADAPTER_MODULES:
        raise ValueError(f"unknown backend '{resolved_backend}'; expected one of {sorted(_ADAPTER_MODULES)}")

    raw = apply_model_overrides(raw, resolved_backend, variant)
    model_key = "name" if resolved_backend in _ANOMALY_MAP_BACKENDS else "variant"
    resolved_variant = raw.get("model", {}).get(model_key)
    if not resolved_variant:
        raise ValueError(f"config has no model.{model_key}; pass --variant explicitly")

    adapter = _build_adapter(resolved_backend, resolved_variant)
    artifact = adapter.load_trained_model(_resolve_weights_artifact(resolved_backend, weights))

    samples = _load_samples(source, resolved_backend)
    if not samples:
        raise ValueError("no samples resolved to run inference on")

    if resolved_backend in _ANOMALY_MAP_BACKENDS:
        predictions = adapter.predict(samples, artifact, output_dir=output_dir)
    else:
        predictions = adapter.predict(samples, artifact)
    return PredictRunResult(backend=resolved_backend, variant=resolved_variant, predictions=predictions)
