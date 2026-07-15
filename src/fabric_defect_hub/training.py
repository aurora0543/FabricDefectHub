"""The single `train` entry point: pick a model config, let the backend be
detected from keywords in that config (or state it explicitly), optionally
point at a different dataset, and optionally override the shot mode
(full-shot / few-shot / an 8-image test-shot smoke run) — then hand the
fully-resolved config to that backend's own `run_from_config`.

This sits one layer above `models/{ultralytics,torchvision,anomalib}
/pipeline.py`: those already execute a fully-declarative single-backend
YAML config end to end. What was missing was one door in front of all
three that (a) tells them apart by keyword instead of requiring the caller
to already know the backend, and (b) lets the train/val sample-selection
dicts embedded in that YAML be overridden from the command line without
hand-editing the file — while still respecting each backend's own rules
about what a "training split" is allowed to contain (see
`apply_dataset_overrides`'s docstring on Anomalib's one-class training).

Every override defaults to `None` ("leave the model config's own value
alone"), so `fdh train configs/models/ultralytics_example.yaml` with no
extra flags behaves exactly like `fdh run` on that same file.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

ShotMode = Literal["full", "few", "test"]

# "test" shot: a tiny end-to-end smoke run (per model/dataset combination)
# meant to verify the pipeline wiring, not to produce a usable checkpoint.
TEST_SHOT_NUM_SAMPLES = 8

# Where `resolve_model_config` looks for configs when given a bare name
# instead of a path (e.g. "ultralytics_example" or "yolov8n").
DEFAULT_MODEL_CONFIG_DIR = Path("configs/models")

# Per backend: (train-split selection key, val/test-split selection key) in
# that backend's `data` config section.
_BACKEND_DATA_SELECTIONS: dict[str, tuple[str, str]] = {
    "anomalib": ("train_selection", "test_selection"),
    "torchvision": ("train_selection", "val_selection"),
    "ultralytics": ("train_selection", "val_selection"),
}

_BACKEND_PIPELINE_MODULES = {
    "ultralytics": "fabric_defect_hub.models.ultralytics.pipeline",
    "torchvision": "fabric_defect_hub.models.torchvision.pipeline",
    "anomalib": "fabric_defect_hub.models.anomalib.pipeline",
}

_BACKEND_CONFIG_CLASSES = {
    "ultralytics": ("fabric_defect_hub.models.ultralytics.config", "UltralyticsConfig"),
    "torchvision": ("fabric_defect_hub.models.torchvision.config", "TorchvisionConfig"),
    "anomalib": ("fabric_defect_hub.models.anomalib.config", "AnomalibConfig"),
}


@dataclass
class DatasetOverrides:
    """CLI-level overrides layered onto a model config's `data` section.

    Every field defaults to `None`, meaning "leave the model config's value
    alone". `mode`/`num_samples` set the train split's (and, unless
    `val_num_samples` is given, the val/test split's) `num_samples`; the
    rest map straight onto the per-split selection dict's matching key
    (`use_defect`, `defect_ratio`, `pattern`, `category`, `seed`).
    """

    dataset: str | None = None
    dataset_root: str | None = None
    mode: ShotMode | None = None
    num_samples: int | None = None
    val_num_samples: int | None = None
    use_defect: bool | None = None
    defect_ratio: float | None = None
    pattern: str | int | None = None
    category: str | None = None
    seed: int | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


def infer_backend(raw: dict[str, Any]) -> str:
    """Keyword-based backend detection over a parsed model-config mapping.

    Checked in order:
    1. An explicit top-level `backend:` key (always wins).
    2. `model.name` present -> `anomalib` (its models are named, e.g.
       'PatchCore'; Ultralytics/torchvision are keyed by `variant` instead).
    3. `model.variant` keyword: `yolo*` -> `ultralytics`,
       `fasterrcnn*`/`maskrcnn*` -> `torchvision`.
    """

    if isinstance(raw, dict) and raw.get("backend"):
        backend = str(raw["backend"]).lower()
        if backend not in _BACKEND_PIPELINE_MODULES:
            raise ValueError(
                f"unknown backend '{backend}'; expected one of {sorted(_BACKEND_PIPELINE_MODULES)}"
            )
        return backend
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(
            "cannot infer backend: config requires a 'model' mapping (or an explicit 'backend' key)"
        )
    model = raw["model"]
    if "name" in model:
        return "anomalib"
    variant = str(model.get("variant", "")).lower()
    if variant.startswith("yolo"):
        return "ultralytics"
    if variant.startswith(("fasterrcnn", "maskrcnn", "cascadercnn", "detr", "unet", "deeplab")):
        return "torchvision"
    raise ValueError("cannot infer backend; pass backend explicitly or add a 'backend' key to the config")


def find_model_configs(config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR) -> list[Path]:
    """Every `*.yaml` model config under `config_dir`, sorted by filename."""

    directory = Path(config_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def resolve_model_config(model: str, config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR) -> Path:
    """Resolve a `train` CLI argument to a model-config YAML path.

    `model` may be, in order of precedence:
    1. An existing file path (used as-is, anywhere on disk) — the original
       `fdh train configs/models/ultralytics_example.yaml` form still works.
    2. A filename stem under `config_dir` (e.g. "ultralytics_example",
       with or without the ".yaml" suffix) — `fdh train ultralytics_example`.
    3. A model keyword matched against every config under `config_dir`'s
       `model.variant` / `model.name` field, case-insensitively — e.g.
       `fdh train yolov8n` or `fdh train patchcore` finds whichever example
       config declares that variant/name, so you don't need to know (or
       type) the file it lives in.
    """

    direct = Path(model)
    if direct.is_file():
        return direct

    directory = Path(config_dir)
    filename = model if model.endswith(".yaml") else f"{model}.yaml"
    by_filename = directory / filename
    if by_filename.is_file():
        return by_filename

    needle = model.strip().lower()
    matches = [path for path in find_model_configs(directory) if needle in _config_keywords(path)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"'{model}' matches multiple configs under {directory}: "
            f"{', '.join(str(path) for path in matches)}; pass one of these paths to disambiguate."
        )

    available = ", ".join(path.stem for path in find_model_configs(directory)) or "<none found>"
    raise FileNotFoundError(
        f"could not resolve model config '{model}' (checked as a path, as a filename under "
        f"'{directory}', and as a model keyword); available under '{directory}': {available}"
    )


def _config_keywords(path: Path) -> set[str]:
    """Case-insensitive keywords a config can be looked up by: its filename
    stem, and its declared `model.variant`/`model.name`.
    """

    keywords = {path.stem.lower()}
    try:
        raw = load_raw_config(path)
    except Exception:
        return keywords
    model_section = raw.get("model") if isinstance(raw, dict) else None
    if isinstance(model_section, dict):
        for key in ("variant", "name"):
            value = model_section.get(key)
            if value:
                keywords.add(str(value).strip().lower())
    return keywords


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Parse a model-config YAML into a plain mapping, with `${VAR}`
    environment-variable expansion (matching each backend's own
    `Config.from_yaml`), but without building the backend dataclass yet —
    that happens after overrides are layered on in `run_train`.
    """

    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"model config {path} must parse to a mapping")
    return _expand_environment_variables(raw)


def apply_dataset_overrides(
    raw: dict[str, Any], backend: str, overrides: DatasetOverrides
) -> dict[str, Any]:
    """Return a copy of `raw` with `overrides` layered onto its `data` section.

    Per-model shot logic: Anomalib's five models all train one-class, so
    its train split always has `use_defect` forced to `False` (and
    `defect_ratio` dropped) regardless of `overrides.use_defect` — that flag
    only ever reaches its *test* split. Ultralytics/torchvision have no such
    constraint; both overrides apply to their train split as given.
    """

    if overrides.is_empty():
        return raw

    data = dict(raw.get("data") or {})
    if overrides.dataset is not None:
        data["dataset"] = overrides.dataset
        data.pop("data_yaml", None)
        data.pop("datamodule_kwargs", None)
    if overrides.dataset_root is not None:
        data["dataset_root"] = overrides.dataset_root

    if not data.get("dataset"):
        raise ValueError(
            "dataset-selection overrides (--mode/--num-samples/--use-defect/...) require "
            "a DatasetAdapter-based 'data.dataset' config; pass --dataset to set one."
        )

    train_key, other_key = _BACKEND_DATA_SELECTIONS[backend]
    train_selection = dict(data.get(train_key) or {})
    other_selection = dict(data.get(other_key) or {})

    _apply_selection_overrides(train_selection, overrides, is_train_split=True)
    _apply_selection_overrides(other_selection, overrides, is_train_split=False)

    if backend == "anomalib":
        # One-class training: the train split is always normal-only, no
        # matter what --use-defect says (that only ever affects the test
        # split's mix, applied above via `other_selection`).
        train_selection["use_defect"] = False
        train_selection.pop("defect_ratio", None)

    data[train_key] = train_selection
    data[other_key] = other_selection
    raw = dict(raw)
    raw["data"] = data
    return raw


def _resolve_num_samples(
    overrides: DatasetOverrides, current: int | None, *, is_train_split: bool
) -> tuple[bool, int | None]:
    """Return (should_set, value) for one split's `num_samples`."""

    if not is_train_split and overrides.val_num_samples is not None:
        return True, overrides.val_num_samples
    if overrides.num_samples is not None:
        return True, overrides.num_samples
    if overrides.mode == "test":
        return True, TEST_SHOT_NUM_SAMPLES
    if overrides.mode == "full":
        return True, None
    if overrides.mode == "few":
        return False, current  # leave the config's own few-shot count as-is
    return False, current


def _apply_selection_overrides(
    selection: dict[str, Any], overrides: DatasetOverrides, *, is_train_split: bool
) -> None:
    should_set, num_samples = _resolve_num_samples(
        overrides, selection.get("num_samples"), is_train_split=is_train_split
    )
    if should_set:
        selection["num_samples"] = num_samples
    if overrides.use_defect is not None:
        selection["use_defect"] = overrides.use_defect
    if overrides.defect_ratio is not None:
        selection["defect_ratio"] = overrides.defect_ratio
    if overrides.pattern is not None:
        selection["pattern"] = overrides.pattern
    if overrides.category is not None:
        selection["category"] = overrides.category
    if overrides.seed is not None:
        selection["seed"] = overrides.seed


def _apply_test_speed_overrides(raw: dict[str, Any], backend: str) -> dict[str, Any]:
    """`mode == "test"` also caps epochs so the 8-image smoke run finishes
    fast — it exists to prove the pipeline wiring works end to end, not to
    produce a usable checkpoint.
    """

    raw = dict(raw)
    train = dict(raw.get("train") or {})
    if backend == "anomalib":
        engine_kwargs = dict(train.get("engine_kwargs") or {})
        engine_kwargs.setdefault("max_epochs", 1)
        train["engine_kwargs"] = engine_kwargs
    else:
        train.setdefault("epochs", 1)
        train.setdefault("patience", 1)
    raw["train"] = train
    return raw


@dataclass
class TrainRunResult:
    """What `run_train` produced, tagged with the backend that ran it."""

    backend: str
    result: Any  # UltralyticsRunResult | TorchvisionRunResult | AnomalibRunResult


def run_train(
    model: str | Path,
    backend: str | None = None,
    overrides: DatasetOverrides | None = None,
    config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
) -> TrainRunResult:
    """The unified training entry point.

    1. Resolve `model` to a config path (see `resolve_model_config` — a
       full path, a filename stem under `config_dir`, or a model keyword
       like "yolov8n"/"patchcore" all work), parse it, and resolve its
       backend (`backend`, if given, skips keyword detection).
    2. Layer `overrides` onto its `data` section (dataset choice, shot mode,
       sample count, defect mix, ...).
    3. Build that backend's own config dataclass and run its full
       train/val/export lifecycle via `models.<backend>.pipeline.run_from_config`.
    """

    model_config = resolve_model_config(str(model), config_dir=config_dir)
    raw = load_raw_config(model_config)
    resolved_backend = backend or infer_backend(raw)
    if resolved_backend not in _BACKEND_PIPELINE_MODULES:
        raise ValueError(
            f"unknown backend '{resolved_backend}'; expected one of {sorted(_BACKEND_PIPELINE_MODULES)}"
        )

    overrides = overrides or DatasetOverrides()
    raw = apply_dataset_overrides(raw, resolved_backend, overrides)
    if overrides.mode == "test":
        raw = _apply_test_speed_overrides(raw, resolved_backend)

    config_module_name, config_cls_name = _BACKEND_CONFIG_CLASSES[resolved_backend]
    config_cls = getattr(importlib.import_module(config_module_name), config_cls_name)
    config = config_cls.from_dict(raw)

    pipeline_module = importlib.import_module(_BACKEND_PIPELINE_MODULES[resolved_backend])
    result = pipeline_module.run_from_config(config)
    return TrainRunResult(backend=resolved_backend, result=result)


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
