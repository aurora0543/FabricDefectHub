"""Config-driven end-to-end runner for the Ultralytics backend.

This is the payoff of configuration-driven management: give it an
`UltralyticsConfig` (typically
`UltralyticsConfig.from_yaml("configs/models/ultralytics_*.yaml")`) and it
executes the whole declared lifecycle — resolve data, train, validate,
register the trained model, export — with no command-line flags. Each stage
is gated by its `enabled` flag in the config, so the same file can describe
"just train", "train + export to ONNX", "validate an existing checkpoint",
etc.

Returns an `UltralyticsRunResult` bundling the artifacts and metrics so the
upper layer (result contract, leaderboard, frontend) can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.ultralytics.adapter import UltralyticsAdapter
from fabric_defect_hub.models.ultralytics.config import DataSpec, UltralyticsConfig


@dataclass
class UltralyticsRunResult:
    """Everything a config-driven run produced."""

    config: UltralyticsConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(data: DataSpec, selection: dict[str, Any]) -> list[Sample]:
    """Resolve one split's `Sample` list via the configured DatasetAdapter."""

    dataset = load_dataset(data.dataset, root=data.dataset_root, **selection)
    return dataset.load_samples()


def _build_data_kwargs(config: UltralyticsConfig) -> dict[str, Any]:
    """Turn the DataSpec into the `data=`/`samples=` kwargs train/val expect."""

    data = config.data
    if not data.uses_adapter():
        return {"data": data.data_yaml}

    train_samples = _load_split_samples(data, data.train_selection)
    # Fall back to reusing the train selection for val only if none is given.
    if data.val_selection:
        val_samples = _load_split_samples(data, data.val_selection)
    else:
        val_samples = train_samples
    return {
        "samples": {"train": train_samples, "val": val_samples},
        "class_names": data.class_names,
    }


def run_from_config(config: UltralyticsConfig) -> UltralyticsRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    adapter = UltralyticsAdapter(name=config.model.variant)
    result = UltralyticsRunResult(config=config)

    data_kwargs = _build_data_kwargs(config)

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = {}
        train_config.update(config.resolved_train_kwargs())
        train_config.update(data_kwargs)
        train_config["pretrained"] = config.model.pretrained
        train_config["offline"] = config.model.offline
        if config.model.weights:
            train_config["weights"] = config.model.weights
        if config.train.resume:
            train_config["resume"] = True

        result.trained_artifact = adapter.train(train_config)

        # Register the trained model into a stable location.
        result.registered_artifact = adapter.register_trained_model(
            result.trained_artifact, registry_dir=config.checkpoint.registry_dir
        )
    elif config.model.weights:
        # No training: load the specified checkpoint so val/export can run.
        result.trained_artifact = adapter.load_trained_model(config.model.weights)

    active_artifact = result.registered_artifact or result.trained_artifact

    # --- Validation -----------------------------------------------------
    if config.val.enabled and active_artifact is not None:
        val_config: dict[str, Any] = dict(config.val.as_overrides())
        val_config.update(data_kwargs)
        # val() only consumes samples/data + val kwargs, not class training keys.
        result.metrics = adapter.validate(active_artifact, val_config)

    # --- Export ---------------------------------------------------------
    if config.export.enabled and config.export.formats and active_artifact is not None:
        for fmt in config.export.formats:
            result.exports.append(
                adapter.export(active_artifact, fmt, config=config.export.as_overrides(fmt))
            )

    return result


def run_from_yaml(path: str) -> UltralyticsRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(UltralyticsConfig.from_yaml(path))
