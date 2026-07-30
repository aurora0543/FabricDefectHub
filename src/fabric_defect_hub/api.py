"""Public top-level API facade: load_config, load_dataset, load_model, train, predict, evaluate."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.base import Artifact

# Preset module + its published-variant lister, per backend. Every backend's
# preset module exposes `list_supported_variants()` under that one name (see
# each `models/<backend>/presets.py`), so this stays a table lookup rather
# than a per-backend branch.
_PRESET_MODULES: dict[str, str] = {
    "ultralytics": "fabric_defect_hub.models.ultralytics.presets",
    "torchvision": "fabric_defect_hub.models.torchvision.presets",
    "anomalib": "fabric_defect_hub.models.anomalib.presets",
    "dinomaly": "fabric_defect_hub.models.dinomaly.presets",
    "moeclip": "fabric_defect_hub.models.moeclip.presets",
    "mambaad": "fabric_defect_hub.models.mambaad.presets",
}


# --------------------------------------------------------------------------- #
# Layer 1 -- configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    """A resolved experiment configuration: which config file, which model
    inside it, and the override layers to apply on top.

    Deliberately a *description* of a run rather than a live object -- it
    holds exactly the arguments `training.run_train` takes, so `train(cfg)`
    is a call, not a translation. `raw` is the parsed config as it stands
    before the override layers are applied, for inspection.
    """

    model: str
    backend: str
    config_path: Path
    variant: str | None = None
    overrides: Any = None  # training.DatasetOverrides
    set_overrides: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def with_set(self, **dotted: Any) -> "RunConfig":
        """Return a copy with extra dotted-path overrides layered on, e.g.
        `cfg.with_set(**{"train.engine_kwargs.max_epochs": 5})`.
        """

        from dataclasses import replace

        return replace(self, set_overrides={**self.set_overrides, **dotted})


def load_config(
    model: str,
    *,
    dataset: str | None = None,
    dataset_root: str | None = None,
    variant: str | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    config_dir: str | Path | None = None,
    set: dict[str, Any] | None = None,  # noqa: A002 - mirrors `fdh train --set`
) -> RunConfig:
    """Resolve `model` to a config and layer the given overrides onto it.

    `model` accepts everything `fdh train` does: a config path, a config
    filename stem, or a bare model keyword (`"stfpm"`, `"yolov8n"`,
    `"patchcore"`) matched against every config's declared model.

    `dataset`/`dataset_root`/`num_samples`/`seed` become a
    `training.DatasetOverrides` (the same layer `--dataset`/`--num-samples`
    use). `set` is the arbitrary dotted-path escape hatch, highest priority,
    identical to `fdh train --set a.b.c=v`.

    `epochs` is resolved through `training.RUN_LENGTH_KEYS` into whichever
    config key that backend counts its run length in. It raises for the two
    backends that count optimizer iterations rather than epochs (Dinomaly,
    MambaAD) instead of quietly writing an epoch count they would ignore --
    those take `set={"train.total_iters": N}`.
    """

    from fabric_defect_hub.training import (
        DEFAULT_MODEL_CONFIG_DIR,
        RUN_LENGTH_KEYS,
        DatasetOverrides,
        infer_backend,
        load_raw_config,
        resolve_model_config_and_variant,
    )

    config_path, implied_variant = resolve_model_config_and_variant(
        model, config_dir=config_dir or DEFAULT_MODEL_CONFIG_DIR
    )
    raw = load_raw_config(config_path)
    backend = infer_backend(raw)

    set_overrides = dict(set or {})
    if epochs is not None:
        key, unit = RUN_LENGTH_KEYS[backend]
        if unit != "epochs":
            raise ValueError(
                f"the {backend} backend counts its run length in {unit}, not epochs; "
                f"pass set={{{key!r}: N}} instead of epochs={epochs}"
            )
        set_overrides.setdefault(key, epochs)

    return RunConfig(
        model=model,
        backend=backend,
        config_path=config_path,
        variant=variant if variant is not None else implied_variant,
        overrides=DatasetOverrides(
            dataset=dataset,
            dataset_root=dataset_root,
            num_samples=num_samples,
            seed=seed,
        ),
        set_overrides=set_overrides,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Layer 2 -- weights
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PretrainedWeights:
    """A published checkpoint from this project's own model catalog.

    Passes straight into `predict`/`evaluate`'s `weights=` (it implements
    `__fspath__`), while still carrying the provenance the frontend shows --
    which backend and variant produced it, and what `source` string the
    leaderboard labels it with.
    """

    key: str
    path: str
    backend: str
    variant: str
    task: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def as_artifact(self) -> Artifact:
        """The `models.base.Artifact` handle an adapter's `predict`/`export`
        takes directly, for callers going through the adapter layer rather
        than through `api.predict`.
        """

        return Artifact(path=self.path, backend=self.backend, metadata=dict(self.metadata))


def from_pretrained(key: str) -> PretrainedWeights:
    """Look up a published checkpoint by its catalog key (`"PatchCore"`,
    `"yolov8n"`, `"STFPM"`, ... -- see `fdh.list_pretrained()`).

    Raises `FileNotFoundError` when the model is in the catalog but has not
    been trained and published on this machine, which is the common case on
    a fresh checkout: the catalog names what this project *publishes*, and
    the weights themselves are produced by `fdh train`, not shipped in git.
    """

    from fabric_defect_hub.catalog import find_canonical_model_by_key, metadata_for, published_path

    model = find_canonical_model_by_key(key)
    path = published_path(model)
    if not path.is_file():
        raise FileNotFoundError(
            f"no published weights for {key!r} at {path}. Train and publish it first "
            f"(`fdh train {model.config}` with variant {model.variant!r}), or pick another "
            f"model from fdh.list_pretrained()."
        )
    return PretrainedWeights(
        key=model.key,
        path=str(path),
        backend=model.backend,
        variant=model.variant,
        task=model.task,
        source=model.source,
        metadata=metadata_for(model),
    )


def list_pretrained(*, available_only: bool = False) -> list[str]:
    """Catalog keys `from_pretrained` accepts. With `available_only`, only
    those whose weights are actually present on this machine.
    """

    from fabric_defect_hub.catalog import CANONICAL_MODELS, published_path

    return [
        model.key
        for model in CANONICAL_MODELS
        if not available_only or published_path(model).is_file()
    ]


# --------------------------------------------------------------------------- #
# Layer 3 -- discovery
# --------------------------------------------------------------------------- #
def list_models(backend: str | None = None) -> dict[str, list[str]] | list[str]:
    """Every model this project can run: `{backend: [variant, ...]}`, or
    just one backend's list when `backend` is given.

    Reads each backend's own preset module, so it reflects what is actually
    registered rather than a hand-maintained list -- adding an alias to
    `models/anomalib/presets.py::MODEL_ALIASES` shows up here immediately.
    """

    import importlib

    backends = [backend] if backend is not None else list(_PRESET_MODULES)
    listed = {
        name: importlib.import_module(_PRESET_MODULES[name]).list_supported_variants()
        for name in backends
    }
    return listed[backend] if backend is not None else listed


def list_datasets() -> list[str]:
    """Every registered dataset name `load_dataset`/`predict` accept."""

    import importlib

    importlib.import_module("fabric_defect_hub.datasets")  # triggers @register_dataset
    from fabric_defect_hub.core.registry import list_datasets as _list

    return _list()


# --------------------------------------------------------------------------- #
# Verbs
# --------------------------------------------------------------------------- #
def train(
    config: RunConfig | str,
    *,
    publish: bool = True,
    config_dir: str | Path | None = None,
    **kwargs: Any,
):
    """Run training. `config` is a `RunConfig` from `load_config`, or a bare
    model keyword for a run that needs no overrides (`fdh.train("stfpm")`).

    Returns `training.TrainRunResult`. `kwargs` pass through to
    `training.run_train` (e.g. `profile=`).

    `config_dir` is named explicitly rather than left to `**kwargs` because it
    has to reach *both* calls below: resolving a bare keyword reads the config
    to infer its backend, so letting it through to `run_train` alone would
    infer the backend from one directory's config and then train from
    another's.
    """

    from fabric_defect_hub.training import run_train

    if config_dir is not None:
        kwargs["config_dir"] = config_dir
    cfg = (
        config
        if isinstance(config, RunConfig)
        else load_config(config, config_dir=config_dir)
    )
    return run_train(
        cfg.model,
        backend=cfg.backend,
        variant=cfg.variant,
        overrides=cfg.overrides,
        set_overrides=cfg.set_overrides,
        publish=publish,
        **kwargs,
    )


def predict(
    model: str,
    weights: str | os.PathLike[str] | PretrainedWeights,
    source: str | list[str] | None = None,
    *,
    dataset: str | None = None,
    dataset_root: str | None = None,
    split: str = "test",
    num_samples: int | None = None,
    pattern: str | int | None = None,
    category: str | None = None,
    seed: int = 0,
    output_dir: str | None = None,
    **kwargs: Any,
):
    """Run inference. Give either `source` (one image path or a list of
    them) or `dataset` (a registered dataset name, optionally sliced by
    `num_samples`/`pattern`/`category`).

    Returns `predict.PredictRunResult`. `kwargs` pass through to
    `inference.runner.run_predict` (`enable_tiling=`, `enable_tta=`, `tile_size=`,
    `tile_overlap=`, `backend=`, `variant=`, `config_dir=`).
    """

    from fabric_defect_hub.inference.runner import run_predict

    return run_predict(
        model,
        weights=os.fspath(weights),
        source=_as_predict_input(
            source, dataset, dataset_root, split, num_samples, pattern, category, seed
        ),
        output_dir=output_dir,
        **kwargs,
    )


def evaluate(
    model: str,
    weights: str | os.PathLike[str] | PretrainedWeights,
    *,
    dataset: str,
    dataset_root: str | None = None,
    split: str = "test",
    num_samples: int | None = None,
    pattern: str | int | None = None,
    category: str | None = None,
    seed: int = 0,
    task: str | None = None,
    output_dir: str | None = None,
    **kwargs: Any,
):
    """Score a checkpoint against a dataset's ground truth -- the scriptable
    equivalent of the web Benchmark tab's leaderboard row.

    Returns `predict.EvaluateRunResult`. A dataset is required (raw image
    paths carry no ground truth to score against).

    Pass `output_dir` to get pixel-level metrics from an anomaly model: the
    adapters only fill `Prediction.anomaly_map` when given somewhere to write
    it, so without it the result is image-level only. See `run_evaluate`.
    """

    from fabric_defect_hub.inference.runner import run_evaluate

    return run_evaluate(
        model,
        weights=os.fspath(weights),
        source=_as_predict_input(
            None, dataset, dataset_root, split, num_samples, pattern, category, seed
        ),
        task=task,
        output_dir=output_dir,
        **kwargs,
    )


def _as_predict_input(
    source: str | list[str] | None,
    dataset: str | None,
    dataset_root: str | None,
    split: str,
    num_samples: int | None,
    pattern: str | int | None,
    category: str | None,
    seed: int,
):
    """Pack the flat inference arguments into the `PredictInput` that
    `run_predict`/`run_evaluate` already take. Shared by both verbs so the
    two keep identical source-selection semantics.
    """

    from fabric_defect_hub.inference.runner import PredictInput

    images = [source] if isinstance(source, str) else list(source or [])
    return PredictInput(
        images=images,
        dataset=dataset,
        dataset_root=dataset_root,
        split=split,
        num_samples=num_samples,
        pattern=pattern,
        category=category,
        seed=seed,
    )
