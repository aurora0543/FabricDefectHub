"""Config-driven end-to-end runner for the torchvision detection backend.
Mirrors `models/ultralytics/pipeline.py`: give it a `TorchvisionConfig`
(typically `TorchvisionConfig.from_yaml("configs/models/torchvision_*.yaml")`)
and it executes the whole declared lifecycle — resolve data via the
configured `DatasetAdapter`, train, validate, register the trained model,
export — driven entirely by the config file, no command-line flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.torchvision.config import TorchvisionConfig


@dataclass
class TorchvisionRunResult:
    """Everything a config-driven run produced."""

    config: TorchvisionConfig
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)


def _load_split_samples(config: TorchvisionConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


def run_from_config(config: TorchvisionConfig, adapter_factory=None) -> TorchvisionRunResult:
    """Execute the lifecycle declared in `config`."""

    config.validate()
    if adapter_factory is None:
        from fabric_defect_hub.models.torchvision.adapter import TorchvisionAdapter

        adapter_factory = TorchvisionAdapter
    adapter = adapter_factory(name=config.model.variant)
    result = TorchvisionRunResult(config=config)

    train_samples = _load_split_samples(config, config.data.train_selection)
    val_samples = (
        _load_split_samples(config, config.data.val_selection)
        if config.data.val_selection
        else train_samples
    )

    # --- Training -------------------------------------------------------
    if config.train.enabled:
        train_config: dict[str, Any] = dict(config.resolved_train_kwargs())
        train_config["train_samples"] = train_samples
        train_config["val_samples"] = val_samples
        train_config["class_names"] = config.data.class_names
        train_config["pretrained"] = config.model.pretrained
        train_config["offline"] = config.model.offline
        train_config["trainable_backbone_layers"] = config.model.trainable_backbone_layers
        train_config["min_size"] = config.model.min_size
        train_config["max_size"] = config.model.max_size
        train_config["run_dir"] = config.checkpoint.run_dir
        train_config["name"] = config.checkpoint.name
        train_config["save_every_epoch"] = config.checkpoint.save_every_epoch
        if config.model.weights:
            train_config["weights"] = config.model.weights

        result.trained_artifact = adapter.train(train_config)
        result.registered_artifact = adapter.register_trained_model(
            result.trained_artifact, registry_dir=config.checkpoint.registry_dir
        )
    elif config.model.weights:
        result.trained_artifact = adapter.load_trained_model(config.model.weights)

    active_artifact = result.registered_artifact or result.trained_artifact

    # --- Validation -------------------------------------------------------
    if config.val.enabled and active_artifact is not None:
        val_kwargs = {"batch_size": config.val.batch_size, "num_workers": config.val.num_workers}
        val_kwargs = {k: v for k, v in val_kwargs.items() if v is not None}
        result.metrics = adapter.validate(val_samples, active_artifact, val_kwargs)

    # --- Export -------------------------------------------------------
    if config.export.enabled and config.export.formats and active_artifact is not None:
        for fmt in config.export.formats:
            result.exports.append(
                adapter.export(active_artifact, fmt, config={"opset": config.export.opset})
            )

    return result


def run_from_yaml(path: str) -> TorchvisionRunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(TorchvisionConfig.from_yaml(path))
