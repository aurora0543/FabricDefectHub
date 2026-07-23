"""Config-driven end-to-end runner for the MoECLIP backend. Mirrors
`models/dinomaly/pipeline.py`'s shape: give it a `MoECLIPConfig`
(typically `MoECLIPConfig.from_yaml("configs/models/moeclip_*.yaml")`)
and it executes the whole declared lifecycle -- resolve data, train,
register the trained checkpoint, evaluate -- driven entirely by the config
file. The one shape difference from the other backends' pipelines is that
training and evaluation read *different* datasets (auxiliary corpus vs.
zero-shot fabric target), which is what makes the metrics zero-shot.
Export is attempted only if enabled and raises a clear error (see
`MoECLIPAdapter.export`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.moeclip.adapter import MoECLIPAdapter
from fabric_defect_hub.models.moeclip.config import MoECLIPConfig


@dataclass
class MoECLIPRunResult:
    """Everything a config-driven run produced."""

    config: MoECLIPConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(
    name: str, root: str, selection: dict[str, Any]
) -> list[Sample]:
    return load_dataset(name, root=root, **selection).load_samples()


def run_from_config(config: MoECLIPConfig) -> MoECLIPRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    adapter = MoECLIPAdapter(name=config.model.name, **config.model.adapter_kwargs())
    result = MoECLIPRunResult(config=config)

    # Evaluation reads a *different* dataset from training whenever
    # `data.test_dataset` is set -- that separation is what makes the
    # reported numbers zero-shot (see `MoECLIPConfig.DataSpec`).
    eval_name, eval_root = config.data.eval_dataset()
    test_samples = _load_split_samples(eval_name, eval_root, config.data.test_selection)

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        train_config["train_samples"] = _load_split_samples(
            config.data.dataset, config.data.dataset_root, config.data.train_selection
        )

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


def run_from_yaml(path: str) -> MoECLIPRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(MoECLIPConfig.from_yaml(path))
