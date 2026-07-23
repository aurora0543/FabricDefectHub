"""Config-driven end-to-end runner for the MambaAD backend. Mirrors
`models/dinomaly/pipeline.py`'s shape: give it a `MambaADConfig`
(typically `MambaADConfig.from_yaml("configs/models/mambaad_example.yaml")`)
and it executes the whole declared lifecycle -- resolve data, train,
register the trained checkpoint, evaluate -- driven entirely by the
config file. Export is skipped with a clear error if enabled (see
`MambaADAdapter.export`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter
from fabric_defect_hub.models.mambaad.config import MambaADConfig


@dataclass
class MambaADRunResult:
    """Everything a config-driven run produced."""

    config: MambaADConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(config: MambaADConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


def run_from_config(config: MambaADConfig) -> MambaADRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    adapter = MambaADAdapter(name=config.model.name)
    result = MambaADRunResult(config=config)

    test_samples = _load_split_samples(config, config.data.test_selection)

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)

        result.trained_artifact = adapter.train(train_config)
        result.registered_artifact = adapter.register_trained_model(
            result.trained_artifact, registry_dir=config.checkpoint.registry_dir
        )

    active_artifact = result.registered_artifact or result.trained_artifact

    # --- Validation (predict + AnomalyEvaluator) -------------------------
    if config.val.enabled and active_artifact is not None and test_samples:
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


def run_from_yaml(path: str) -> MambaADRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(MambaADConfig.from_yaml(path))
