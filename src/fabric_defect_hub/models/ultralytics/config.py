"""Config-driven experiment description for the Ultralytics backend.

The design goal (per project direction: 配置化管理) is that *what a run
does is declared in a YAML file, not in command-line flags*. A single
`UltralyticsConfig` object captures the entire lifecycle — which model
variant, where the data comes from, training hyperparameters, validation,
prediction, export targets, and checkpoint/registry locations — and is
consumed by `models/ultralytics/pipeline.py`.

Layering of hyperparameters (lowest priority first):
    1. Ultralytics' own DEFAULT_CFG_DICT (implicit, inside ultralytics)
    2. fabric-tailored defaults from `presets.default_train_kwargs()`
    3. whatever the YAML `train:` block specifies
Later layers win. This keeps YAML files short — you only write the knobs
you actually want to change — while still being fully explicit about the
rest via `resolved_train_kwargs()`.

Load with `UltralyticsConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.ultralytics.presets import (
    default_train_kwargs,
    list_supported_variants,
    resolve_variant,
    variant_weights,
)


@dataclass
class ModelSpec:
    """Which model to build and where its initial weights come from.

    variant: one of `presets.list_supported_variants()` (yolov8n/v8s/yolo11n).
    weights: explicit path to a checkpoint to load instead of the variant's
        default. Use this to (a) resume/fine-tune a previously trained
        FabricDefectHub model, or (b) point at custom pretrained weights.
        When set, it takes precedence over `pretrained`.
    pretrained: if True (and `weights` is None), load the variant's
        COCO-pretrained checkpoint (transfer learning). If False, start from
        the architecture spec with random init.
    task: Ultralytics task; 'detect' for this backend.
    """

    variant: str = "yolov8n"
    weights: str | None = None
    pretrained: bool = True
    task: str = "detect"

    def initial_weights(self) -> str:
        """Resolve the file/name Ultralytics should be initialised from."""

        if self.weights:
            return self.weights
        return variant_weights(self.variant, pretrained=self.pretrained)

    def validate(self) -> None:
        resolve_variant(self.variant)  # raises KeyError with a helpful list
        if self.task != "detect":
            raise ValueError(
                f"UltralyticsAdapter only supports task='detect', got {self.task!r}."
            )


@dataclass
class DataSpec:
    """Where training/validation data comes from.

    Two mutually exclusive modes:

    * `data_yaml`: path to an existing Ultralytics `data.yaml`. Use when you
      already have a YOLO-format dataset on disk.
    * `dataset` (+ `dataset_root`, `train_selection`, `val_selection`): a
      registered `DatasetAdapter` name (e.g. 'zju-leaper'). The pipeline
      loads samples for each split via the adapter, then stages them into a
      temporary YOLO dataset on the fly (see `datasets.yolo_bbox`) — no
      converted copy is persisted. `*_selection` dicts are passed straight
      to the adapter constructor (e.g. `{pattern, num_samples, defect_ratio,
      seed}` for ZJU-Leaper).

    `class_names` fixes the class-id order in the generated data.yaml so it
    matches across train/val/predict and any previously trained checkpoint.
    """

    data_yaml: str | None = None
    dataset: str | None = None
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    val_selection: dict[str, Any] = field(default_factory=dict)
    class_names: list[str] | None = None

    def uses_adapter(self) -> bool:
        return self.dataset is not None

    def validate(self) -> None:
        if self.data_yaml and self.dataset:
            raise ValueError(
                "DataSpec: set either 'data_yaml' or 'dataset', not both."
            )
        if not self.data_yaml and not self.dataset:
            raise ValueError(
                "DataSpec: one of 'data_yaml' or 'dataset' is required."
            )
        if self.dataset and not self.dataset_root:
            raise ValueError(
                f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'."
            )


@dataclass
class TrainSpec:
    """Training hyperparameters.

    A curated set of the knobs most worth putting in a config sit as named
    fields (with `None` meaning "fall back to the fabric preset"); anything
    else Ultralytics accepts can go in `extra` and is merged verbatim. See
    `presets.COMMON_FABRIC_TRAIN_DEFAULTS` for the fallbacks.
    """

    enabled: bool = True
    epochs: int | None = None
    imgsz: int | None = None
    batch: int | float | None = None
    patience: int | None = None
    optimizer: str | None = None
    lr0: float | None = None
    lrf: float | None = None
    weight_decay: float | None = None
    cos_lr: bool | None = None
    single_cls: bool | None = None
    freeze: int | list[int] | None = None
    device: str | int | None = None
    workers: int | None = None
    seed: int | None = None
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # Fields that are pipeline-control, not Ultralytics train() kwargs.
    _NON_ULTRALYTICS = {"enabled", "resume", "extra"}

    def as_overrides(self) -> dict[str, Any]:
        """Explicitly-set named fields (non-None) as Ultralytics kwargs."""

        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._NON_ULTRALYTICS or f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out


@dataclass
class ValSpec:
    """Validation / metric-extraction settings."""

    enabled: bool = True
    imgsz: int | None = None
    batch: int | float | None = None
    conf: float | None = None
    iou: float | None = None
    device: str | int | None = None
    split: str = "val"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {"split": self.split}
        for name in ("imgsz", "batch", "conf", "iou", "device"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out


@dataclass
class PredictSpec:
    """Inference defaults applied by `UltralyticsAdapter.predict`."""

    conf: float = 0.25
    iou: float = 0.7
    imgsz: int | None = None
    max_det: int = 300
    device: str | int | None = None
    augment: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "augment": self.augment,
        }
        for name in ("imgsz", "device"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out


@dataclass
class ExportSpec:
    """Post-training export targets, e.g. ONNX / TensorRT.

    formats: list of Ultralytics export format strings ('onnx', 'engine',
        'torchscript', 'openvino', ...). Empty = skip export.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int | None = None
    imgsz: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self, fmt: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": fmt,
            "half": self.half,
            "dynamic": self.dynamic,
            "simplify": self.simplify,
        }
        if self.opset is not None:
            out["opset"] = self.opset
        if self.imgsz is not None:
            out["imgsz"] = self.imgsz
        out.update(self.extra)
        return out


@dataclass
class CheckpointSpec:
    """Where runs, trained models, and exports are written/kept.

    project/name: Ultralytics run directory (`<project>/<name>/`), holding
        weights/best.pt, weights/last.pt, plots, and results.csv.
    registry_dir: a stable location the pipeline copies the final best.pt
        into (named `<variant>_<name>.pt`), so trained models can be loaded
        back later independent of the transient run directory.
    exist_ok: allow reusing an existing run directory (else Ultralytics
        auto-increments the name).
    """

    project: str = "runs/fabric_defect_hub"
    name: str = "yolo_exp"
    registry_dir: str = "artifacts/models"
    exist_ok: bool = False

    def as_overrides(self) -> dict[str, Any]:
        return {"project": self.project, "name": self.name, "exist_ok": self.exist_ok}


@dataclass
class UltralyticsConfig:
    """Top-level, fully-declarative description of a YOLO experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UltralyticsConfig":
        """Build from a plain mapping (e.g. parsed YAML). Unknown top-level
        keys are rejected so typos surface immediately rather than being
        silently ignored.
        """

        section_types = {
            "model": ModelSpec,
            "data": DataSpec,
            "train": TrainSpec,
            "val": ValSpec,
            "predict": PredictSpec,
            "export": ExportSpec,
            "checkpoint": CheckpointSpec,
        }
        unknown = set(data) - set(section_types)
        if unknown:
            raise ValueError(
                f"UltralyticsConfig: unknown top-level keys {sorted(unknown)}; "
                f"expected some of {sorted(section_types)}."
            )
        kwargs: dict[str, Any] = {}
        for key, spec_cls in section_types.items():
            if key in data and data[key] is not None:
                kwargs[key] = _build_section(spec_cls, data[key], key)
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UltralyticsConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Validation & resolution
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_train_kwargs(self) -> dict[str, Any]:
        """The full, explicit Ultralytics `train()` kwargs after layering
        fabric presets < config named fields < config `extra`.
        """

        merged = default_train_kwargs(self.model.variant)
        merged.update(self.train.as_overrides())
        merged.update(self.train.extra)
        merged.update(self.checkpoint.as_overrides())
        return merged


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _build_section(spec_cls, raw: dict[str, Any], section: str):
    """Instantiate a dataclass section, rejecting unknown keys with context."""

    if not isinstance(raw, dict):
        raise ValueError(f"UltralyticsConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"UltralyticsConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def supported_variants() -> list[str]:
    return list_supported_variants()
