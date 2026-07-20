"""Config-driven end-to-end runner for the Dinomaly backend. Mirrors
`models/anomalib/pipeline.py`'s shape: give it a `DinomalyConfig`
(typically `DinomalyConfig.from_yaml("configs/models/dinomaly_*.yaml")`)
and it executes the whole declared lifecycle -- resolve data, train,
register the trained checkpoint, evaluate -- driven entirely by the
config file. Export is skipped with a clear error if enabled (see
`DinomalyAdapter.export`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.dinomaly.adapter import DinomalyAdapter
from fabric_defect_hub.models.dinomaly.config import DinomalyConfig


@dataclass
class DinomalyRunResult:
    """Everything a config-driven run produced."""

    config: DinomalyConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(config: DinomalyConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


def run_from_config(config: DinomalyConfig) -> DinomalyRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    adapter = DinomalyAdapter(name=config.model.name)
    result = DinomalyRunResult(config=config)

    test_samples: list[Sample] | None = None
    if config.data.uses_adapter():
        test_samples = _load_split_samples(config, config.data.test_selection)

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        if config.data.uses_adapter():
            train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)
            train_config["test_samples"] = test_samples
        else:
            train_config["data_root"] = config.data.data_root

        result.trained_artifact = adapter.train(train_config)
        result.registered_artifact = adapter.register_trained_model(
            result.trained_artifact, registry_dir=config.checkpoint.registry_dir
        )

    active_artifact = result.registered_artifact or result.trained_artifact

    # --- Validation (predict + AnomalyEvaluator) -------------------------
    # Only runs in `data.dataset` mode -- `data_root` mode points at an
    # existing on-disk folder with no corresponding `Sample` objects for
    # `AnomalyEvaluator` to score against, same as `AnomalibConfig`'s
    # `datamodule_kwargs` mode (see its pipeline's module docstring).
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


def run_from_yaml(path: str) -> DinomalyRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(DinomalyConfig.from_yaml(path))
