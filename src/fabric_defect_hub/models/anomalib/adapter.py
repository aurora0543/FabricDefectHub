"""`ModelAdapter` implementation backed by the `anomalib` package.

Covers the five models the README commits to — PatchCore, PaDiM, RD4AD,
EfficientAD, SuperSimpleNet — via `presets.py`, which resolves README/paper
names to anomalib's actual class names and supplies fabric-tailored default
constructor kwargs (see `presets.py` for why each default was picked).

Anomaly-only: `predict()` always fills `anomaly_score` (image-level) and
can optionally persist pixel-level `anomaly_map`s (see its docstring) for
`evaluation.anomaly.AnomalyEvaluator`'s pixel AUROC/AUPRO.

Requires the `anomalib` extra: `pip install -e ".[anomalib]"`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.models.anomalib.presets import (
    default_model_kwargs,
    resolve_model_class,
    resolve_model_class_name,
)
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter


@register_model("anomalib")
class AnomalibAdapter(ModelAdapter):
    """Wraps an `anomalib.models` class.

    `name` may be a README/paper alias ('PatchCore', 'RD4AD', 'EfficientAD',
    'SuperSimpleNet', 'PaDiM' — case-insensitive) or the literal anomalib
    class name ('Patchcore', 'ReverseDistillation', ...). See
    `presets.list_supported_models()` for the full set.
    """

    backend = "anomalib"

    def __init__(self, name: str = "PatchCore", **kwargs):
        super().__init__(name=name, **kwargs)
        # Fail fast on an unknown model name rather than at train() time.
        self.resolved_class_name = resolve_model_class_name(name)

    def _model_cls(self):
        return resolve_model_class(self.name)

    def train(self, config: dict[str, Any]) -> Artifact:
        """Two ways to point this at data:

        - `config['datamodule_kwargs']`: passed straight through to
          `anomalib.data.Folder(**datamodule_kwargs)` — use this if you
          already have an MVTec-style dataset on disk.
        - `config['train_samples']` + `config['test_samples']`: raw
          `Sample` lists straight out of `DatasetAdapter.load_samples()`
          (`train_samples` all-normal, `test_samples` mixed — e.g.
          `ZJULeaperDataset(..., use_defect=False)` for the former and
          `ZJULeaperDataset(..., use_defect=True, defect_ratio=...)` for
          the latter). These are symlinked into a temporary MVTec-style
          folder for the duration of this call only (see
          `datasets.anomalib_folder`); nothing is left on disk afterwards.

        Other keys: `model_kwargs` (merged over the fabric-tailored preset
        for this model — caller keys win), `engine_kwargs` (passed to
        `Engine`).
        """

        from anomalib.data import Folder
        from anomalib.engine import Engine

        model_kwargs = {**default_model_kwargs(self.name), **config.get("model_kwargs", {})}
        self._validate_model_kwargs(model_kwargs)

        model = self._model_cls()(**model_kwargs)
        engine = Engine(**config.get("engine_kwargs", {}))

        train_samples = config.get("train_samples")
        test_samples = config.get("test_samples")
        if train_samples is not None and test_samples is not None:
            # num_workers=0: the staged directory is symlinks into a
            # tempfile.mkdtemp() dir that lives only for this `with` block;
            # worker subprocesses opening it introduce a shutdown race with
            # no benefit at the sample counts this path is meant for
            # (few-shot / low-shot). Override via config['num_workers'] if
            # you really want parallel loading for a large staged set.
            datamodule_kwargs = {"num_workers": config.get("num_workers", 0)}
            with anomalib_folder_staging_dir(train_samples, test_samples) as layout:
                datamodule = Folder(
                    name=self.resolved_class_name.lower(), **layout.as_kwargs(), **datamodule_kwargs
                )
                engine.fit(model=model, datamodule=datamodule)
        else:
            datamodule = Folder(**config["datamodule_kwargs"])
            engine.fit(model=model, datamodule=datamodule)

        ckpt_path = engine.trainer.checkpoint_callback.best_model_path
        return Artifact(
            path=str(ckpt_path),
            backend=self.backend,
            metadata={
                "model_class": self.resolved_class_name,
                "model_kwargs": model_kwargs,
                "trusted": True,
            },
        )

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        """Catch fabric-specific misconfigurations before they surface as an
        opaque failure deep inside a Lightning training loop.
        """

        if self.resolved_class_name == "EfficientAd":
            imagenet_dir = model_kwargs.get("imagenet_dir")
            if not imagenet_dir or not Path(imagenet_dir).exists():
                raise ValueError(
                    "EfficientAD requires model_kwargs['imagenet_dir'] to point at an "
                    "existing natural-image dataset (used for its regularization loss); "
                    f"got {imagenet_dir!r}. There is no fabric-appropriate default — "
                    "pass a real path, e.g. an Imagenette download."
                )

    def predict(
        self, samples: list[Sample], artifact: Artifact, output_dir: str | None = None
    ) -> list[Prediction]:
        """Always fills `anomaly_score`. Pass `output_dir` to also persist
        each sample's pixel-level `anomaly_map` as a `.npy` file there and
        fill `Prediction.anomaly_map` with its path — needed for
        `evaluation.anomaly.AnomalyEvaluator`'s pixel-level metrics
        (pixel AUROC/AUPRO). Omit it to skip that disk write when you only
        need image-level scores.
        """

        if not artifact.metadata.get("trusted", False):
            raise ValueError(
                "Refusing to load an untrusted Anomalib checkpoint. Use load_trained_model(..., "
                "allow_unsafe_checkpoint=True) only for a checkpoint from a trusted source."
            )

        import numpy as np
        from anomalib.data import PredictDataset
        from anomalib.engine import Engine

        model_cls = resolve_model_class(artifact.metadata.get("model_class", self.name))
        model = _load_checkpoint(model_cls, artifact.path)
        engine = Engine()

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)

        predictions = []
        for sample in samples:
            dataset = PredictDataset(path=sample.image_path)
            batches = engine.predict(model=model, dataset=dataset) or []
            if not batches:
                raise RuntimeError(
                    f"Anomalib produced no prediction output for sample {sample.id!r} "
                    f"({sample.image_path})."
                )
            score = None
            anomaly_map_path = None
            if batches:
                batch = batches[0]
                if batch.pred_score is not None:
                    score = float(batch.pred_score[0])
                raw_map = getattr(batch, "anomaly_map", None)
                if maps_dir is not None and raw_map is not None:
                    arr = raw_map[0]
                    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
                    map_path = maps_dir / f"{sample.id}.npy"
                    np.save(map_path, np.squeeze(arr))
                    anomaly_map_path = str(map_path)
            if score is None:
                raise RuntimeError(
                    f"Anomalib prediction for sample {sample.id!r} has no anomaly score."
                )
            predictions.append(
                Prediction(sample_id=sample.id, anomaly_score=score, anomaly_map=anomaly_map_path)
            )
        return predictions

    def export(self, artifact: Artifact, target: str) -> ExportedArtifact:
        """`target` is an `anomalib.deploy.ExportType` value, e.g. 'onnx', 'openvino'."""

        if not artifact.metadata.get("trusted", False):
            raise ValueError("Refusing to export an untrusted Anomalib checkpoint.")

        from anomalib.engine import Engine

        model_cls = resolve_model_class(artifact.metadata.get("model_class", self.name))
        model = _load_checkpoint(model_cls, artifact.path)
        engine = Engine()
        exported_path = engine.export(model=model, export_type=target)
        return ExportedArtifact(path=str(exported_path), target=target)

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        """Copy a trained checkpoint out of `Engine`'s versioned working
        directory (`<default_root_dir>/<ModelClass>/<name>/v{N}/weights/
        lightning/model.ckpt`) into a stable, named location so it can be
        reloaded later independent of that version path.

        Unlike `UltralyticsAdapter.register_trained_model`, the destination
        filename doesn't need to embed a run-directory name to disambiguate
        runs: `artifact.metadata['model_class']` already uniquely identifies
        which of the five algorithms produced the checkpoint, so the default
        filename is just `<model_class>.ckpt`. Pass `model_name` explicitly
        if you're registering more than one run of the *same* model and want
        to keep both.
        """

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing checkpoint: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        model_class = artifact.metadata.get("model_class", self.resolved_class_name)
        filename = model_name or f"{model_class}.ckpt"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(
        self, artifact_or_path: Artifact | str, allow_unsafe_checkpoint: bool = False
    ) -> Artifact:
        """Load a previously registered/trained checkpoint back into this
        adapter. Unlike `predict()`/`export()`, which resolve the model
        class from `artifact.metadata['model_class']` internally, this just
        validates the checkpoint exists — model-class resolution still
        happens lazily, at the point `predict()`/`export()` actually needs it.
        """

        path = artifact_or_path.path if isinstance(artifact_or_path, Artifact) else artifact_or_path
        if not Path(path).exists():
            raise FileNotFoundError(f"cannot load missing checkpoint: {path}")
        if isinstance(artifact_or_path, Artifact):
            if not artifact_or_path.metadata.get("trusted", False):
                raise ValueError("Anomalib artifact is not marked as trusted.")
            return artifact_or_path
        if not allow_unsafe_checkpoint:
            raise ValueError(
                "Loading a raw Anomalib checkpoint requires allow_unsafe_checkpoint=True because "
                "Lightning checkpoints can deserialize arbitrary Python objects."
            )
        return Artifact(
            path=str(path), backend=self.backend,
            metadata={"model_class": self.resolved_class_name, "trusted": True},
        )


def _load_checkpoint(model_cls, path: str):
    """`weights_only=False`: PyTorch >=2.6 defaults `torch.load` to
    `weights_only=True`, which rejects anomalib's own checkpoint globals
    (e.g. `anomalib.PrecisionType`) unless explicitly allowlisted. Safe
    here because callers are required to pass an `Artifact` marked trusted
    by `train()`/`register_trained_model()`, or to explicitly opt into an
    unsafe raw-checkpoint load in `load_trained_model()`.
    """

    return model_cls.load_from_checkpoint(path, weights_only=False)
