"""Config-driven end-to-end runner for the Anomalib backend. Mirrors
`models/ultralytics/pipeline.py`/`models/torchvision/pipeline.py`'s overall
shape: give it an `AnomalibConfig` (typically `AnomalibConfig.from_yaml(
"configs/models/anomalib_*.yaml")`) and it executes the whole declared
lifecycle — resolve data, train, register the trained model, evaluate,
export — driven entirely by the config file.

One real divergence from the other two pipelines (see `config.py`'s module
docstring for the other): there is no backend-native `.validate()` here.
`AnomalibAdapter` only trains and predicts; scoring is `evaluation.anomaly
.AnomalyEvaluator`'s job, same as it would be for any other caller of
`predict()`. So this pipeline's "validation" step is
`adapter.predict(...)` followed by `AnomalyEvaluator(...).evaluate(...)` —
wiring together two things that, for Ultralytics/torchvision, the backend
itself already does in one native call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.config import AnomalibConfig
from fabric_defect_hub.models.base import Artifact, ExportedArtifact


@dataclass
class AnomalibRunResult:
    """Everything a config-driven run produced."""

    config: AnomalibConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(config: AnomalibConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


def run_from_config(config: AnomalibConfig) -> AnomalibRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    adapter = AnomalibAdapter(name=config.model.name)
    result = AnomalibRunResult(config=config)

    test_samples: list[Sample] | None = None
    if config.data.uses_adapter():
        test_samples = _load_split_samples(config, config.data.test_selection)

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = {
            "model_kwargs": config.resolved_model_kwargs(),
            "engine_kwargs": config.resolved_engine_kwargs(),
            "num_workers": config.train.num_workers,
        }
        if config.data.uses_adapter():
            train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)
            train_config["test_samples"] = test_samples
        else:
            train_config["datamodule_kwargs"] = config.data.datamodule_kwargs

        result.trained_artifact = adapter.train(train_config)
        result.registered_artifact = adapter.register_trained_model(
            result.trained_artifact, registry_dir=config.checkpoint.registry_dir
        )

    active_artifact = result.registered_artifact or result.trained_artifact

    # --- Validation (predict + AnomalyEvaluator; see module docstring) --
    # Only runs when `data.dataset` was used (i.e. we have a `Sample` list
    # to evaluate against). `data.datamodule_kwargs` mode points anomalib's
    # `Folder` at an existing on-disk dataset with no corresponding `Sample`
    # objects, so `AnomalyEvaluator` — which is `Sample`/`Prediction`-only,
    # not `Folder`-aware — has nothing to score against; validation is
    # silently skipped rather than guessed at in that mode.
    if config.val.enabled and active_artifact is not None and test_samples is not None:
        predictions = adapter.predict(test_samples, active_artifact, output_dir=config.val.output_dir)
        evaluator = AnomalyEvaluator(
            max_pixels=config.val.max_pixels,
            max_aupro_images=config.val.max_aupro_images,
            seed=config.val.seed,
        )
        result.metrics = evaluator.evaluate(test_samples, predictions)

    # --- Export -----------------------------------------------------------
    if config.export.enabled and config.export.formats and active_artifact is not None:
        for fmt in config.export.formats:
            result.exports.append(adapter.export(active_artifact, fmt))

    return result


def run_from_yaml(path: str) -> AnomalibRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(AnomalibConfig.from_yaml(path))
