"""`ModelAdapter` implementation backed by `torchvision.models.detection`,
covering the full training lifecycle for Faster R-CNN and Mask R-CNN
(ResNet50-FPN, v1 and v2 heads):

    pretrained-weight loading -> train (real loop, see engine.py)
        -> validate (native torchmetrics mAP) -> predict
        -> register/save the trained model -> load it back
        -> export (TorchScript / ONNX best-effort).

torchvision.models.detection replaces mmdetection as this project's
"comprehensive supervised detection comparison" backend: `mmdetection`'s
`mmcv` dependency has no macOS arm64 wheel and doesn't build on Python 3.14
(confirmed by a real install attempt against this project's environment —
see the git history / conversation, not a hypothetical), while
torchvision is pure PyTorch, actively maintained by the PyTorch team, and
installs everywhere torch does. The tradeoff: Cascade R-CNN, DETR and DINO
(which mmdetection ships and torchvision does not) are out of scope for
this backend; Faster R-CNN and Mask R-CNN cover the "classic two-stage
detector" comparison point on their own.

Detection-only + optional instance segmentation: `predict()` fills `boxes`,
`labels`, `scores` (Faster R-CNN) and additionally `masks` (Mask R-CNN).

Requires the `torchvision` extra: `pip install -e ".[torchvision]"`.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter
from fabric_defect_hub.models.torchvision.dataset import (
    SampleDetectionDataset,
    build_class_map,
    detection_collate_fn,
)
from fabric_defect_hub.models.torchvision.presets import (
    build_model,
    build_transforms,
    resolve_variant,
    uses_masks,
)


@register_model("torchvision")
class TorchvisionAdapter(ModelAdapter):
    """Full-lifecycle wrapper around `torchvision.models.detection.{faster,mask}_rcnn`.

    `name` is a variant ('fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn',
    ...; see `presets.list_supported_variants()`) or a path to a previously
    saved checkpoint (a dict with `state_dict` + `class_map`, as written by
    `register_trained_model`).
    """

    backend = "torchvision"

    def __init__(self, name: str = "fasterrcnn_resnet50_fpn", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model = None
        self._class_map: dict[str, int] | None = None
        self._device = None

    # ------------------------------------------------------------------ #
    # Model / weight loading
    # ------------------------------------------------------------------ #
    def _resolve_device(self, requested: str | None = None):
        import torch

        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_pretrained(
        self,
        class_names: list[str],
        variant: str | None = None,
        trainable_backbone_layers: int | None = None,
        device: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        """Load COCO-pretrained detection weights with the classifier head
        swapped for `class_names` (transfer learning starting point).
        """

        variant = variant or self.name
        self._class_map = build_class_map([], class_names=class_names)
        num_classes = len(class_names) + 1  # +1 for background
        self._device = self._resolve_device(device)
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=True,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
        ).to(self._device)
        return self._model

    def load_scratch(
        self,
        class_names: list[str],
        variant: str | None = None,
        trainable_backbone_layers: int | None = None,
        device: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        """Load an ImageNet-pretrained-backbone-only model with a
        random-init detection head (train from scratch).
        """

        variant = variant or self.name
        self._class_map = build_class_map([], class_names=class_names)
        num_classes = len(class_names) + 1
        self._device = self._resolve_device(device)
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=False,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
        ).to(self._device)
        return self._model

    def load_weights(self, path: str, device: str | None = None):
        """Load a checkpoint previously produced by `register_trained_model`
        (contains `state_dict`, `class_map`, `variant`).

        Builds the bare architecture with `backbone_weights=False` — every
        parameter is about to be overwritten by `checkpoint['state_dict']`
        anyway, so there is no reason to download ImageNet backbone weights
        first (and every reason not to: it makes reloading a trained model
        needlessly dependent on network access).
        """

        import torch

        self._device = self._resolve_device(device)
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._class_map = checkpoint["class_map"]
        variant = checkpoint.get("variant", self.name)
        num_classes = len(self._class_map) + 1
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=False, backbone_weights=False,
        ).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self.name = variant
        return self._model

    @property
    def model(self):
        if self._model is None:
            self.load_pretrained(class_names=["defect"])
        return self._model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, config: dict[str, Any]) -> Artifact:
        """Run a full fine-tuning job and return an `Artifact` pointing at
        the best (by validation mAP) checkpoint.

        Required:
          * `config['train_samples']` — `Sample` list for training.
          * `config['val_samples']`   — `Sample` list for validation (used
            for mAP-based checkpointing and early stopping).

        Optional (else fall back to fabric presets — see `presets.py`):
          * `class_names`, `weights`, `pretrained`, `trainable_backbone_layers`,
            `epochs`, `batch_size`, `optimizer`, `lr`, `momentum`,
            `weight_decay`, `lr_scheduler`, `step_size`, `gamma`,
            `warmup_epochs`, `grad_clip_norm`, `patience`, `num_workers`,
            `device`, `run_dir`, `save_every_epoch`.
        """

        from torch.utils.data import DataLoader

        from fabric_defect_hub.models.torchvision.engine import run_training
        from fabric_defect_hub.models.torchvision.presets import default_train_kwargs

        cfg = {**default_train_kwargs(self.name), **config}
        train_samples: list[Sample] = cfg["train_samples"]
        val_samples: list[Sample] = cfg["val_samples"]
        class_names = cfg.get("class_names") or ["defect"]
        with_masks = uses_masks(self.name)

        weights = cfg.get("weights")
        pretrained = cfg.get("pretrained", True)
        device_str = cfg.get("device")
        tbl = cfg.get("trainable_backbone_layers")
        min_size = cfg.get("min_size")
        max_size = cfg.get("max_size")

        if weights:
            self.load_weights(weights, device=device_str)
        elif pretrained:
            self.load_pretrained(
                class_names, trainable_backbone_layers=tbl, device=device_str,
                min_size=min_size, max_size=max_size,
            )
        else:
            self.load_scratch(
                class_names, trainable_backbone_layers=tbl, device=device_str,
                min_size=min_size, max_size=max_size,
            )

        class_map = self._class_map
        train_ds = SampleDetectionDataset(
            train_samples, class_map=class_map, with_masks=with_masks,
            transforms=build_transforms(
                train=True,
                hflip_prob=cfg.get("hflip_prob", 0.5),
                vflip_prob=cfg.get("vflip_prob", 0.5),
                color_jitter=cfg.get("color_jitter"),
            ),
        )
        val_ds = SampleDetectionDataset(
            val_samples, class_map=class_map, with_masks=with_masks,
            transforms=build_transforms(train=False),
        )

        num_workers = cfg.get("num_workers", 2)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True,
            num_workers=num_workers, collate_fn=detection_collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
            num_workers=num_workers, collate_fn=detection_collate_fn,
        )

        run_dir = Path(cfg.get("run_dir", "runs/fabric_defect_hub_tv")) / cfg.get("name", "torchvision_exp")
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        history_path = run_dir / "history.csv"

        state = {"best_map": -1.0}

        def on_epoch_end(log):
            self._save_checkpoint(last_path, class_names)
            current_map = log.val_metrics.get("map")
            if current_map is not None and current_map > state["best_map"]:
                state["best_map"] = current_map
                self._save_checkpoint(best_path, class_names)
            if cfg.get("save_every_epoch"):
                self._save_checkpoint(run_dir / f"epoch_{log.epoch:03d}.pt", class_names)
            _append_history_row(history_path, log)

        logs = run_training(
            self.model, train_loader, val_loader, self._device,
            epochs=cfg.get("epochs", 30),
            optimizer_name=cfg.get("optimizer", "sgd"),
            lr=cfg.get("lr", 0.002),
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0005),
            lr_scheduler_name=cfg.get("lr_scheduler", "cosine"),
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.1),
            warmup_epochs=cfg.get("warmup_epochs", 1),
            grad_clip_norm=cfg.get("grad_clip_norm", 5.0),
            patience=cfg.get("patience", 8),
            with_masks=with_masks,
            on_epoch_end=on_epoch_end,
        )

        final_path = best_path if best_path.exists() else last_path
        self.load_weights(str(final_path), device=device_str)

        final_metrics = logs[-1].val_metrics if logs else {}
        return Artifact(
            path=str(final_path),
            backend=self.backend,
            metadata={
                "variant": resolve_variant(self.name),
                "run_dir": str(run_dir),
                "last_checkpoint": str(last_path),
                "history_csv": str(history_path),
                "class_names": class_names,
                "epochs_run": len(logs),
                "best_map": state["best_map"],
                "final_val_metrics": final_metrics,
            },
        )

    def _save_checkpoint(self, path: Path, class_names: list[str]) -> None:
        import torch

        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "class_map": self._class_map,
                "class_names": class_names,
                "variant": resolve_variant(self.name),
            },
            path,
        )

    # ------------------------------------------------------------------ #
    # Validation / metrics
    # ------------------------------------------------------------------ #
    def validate(
        self, samples: list[Sample], artifact: Artifact | None = None, config: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Run native torchmetrics mAP evaluation and return a flat metrics
        dict (`map`, `map_50`, `map_75`, `mar_100`, ...; with `segm_*`
        variants too when the loaded model is Mask R-CNN).
        """

        from torch.utils.data import DataLoader

        from fabric_defect_hub.models.torchvision.engine import evaluate as run_evaluate

        if artifact is not None:
            self.load_weights(artifact.path)

        cfg = config or {}
        with_masks = uses_masks(self.name)
        dataset = SampleDetectionDataset(
            samples, class_map=self._class_map, with_masks=with_masks,
            transforms=build_transforms(train=False),
        )
        loader = DataLoader(
            dataset, batch_size=cfg.get("batch_size", 4), shuffle=False,
            num_workers=cfg.get("num_workers", 2), collate_fn=detection_collate_fn,
        )
        return run_evaluate(self.model, loader, self._device, with_masks=with_masks)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        samples: list[Sample],
        artifact: Artifact | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Run inference over `samples`. `config` overrides `score_threshold`
        (default 0.5) and `max_detections` (default 100). If `artifact` is
        given its weights are loaded first.
        """

        import torch

        if artifact is not None:
            self.load_weights(artifact.path)

        cfg = config or {}
        score_threshold = cfg.get("score_threshold", 0.5)
        max_detections = cfg.get("max_detections", 100)
        with_masks = uses_masks(self.name)
        id_to_label = {v: k for k, v in (self._class_map or {}).items()}

        dataset = SampleDetectionDataset(
            samples, class_map=self._class_map, with_masks=with_masks,
            transforms=build_transforms(train=False),
        )

        predictions: list[Prediction] = []
        self.model.eval()
        with torch.no_grad():
            for sample, (image, _target) in zip(samples, dataset):
                output = self.model([image.to(self._device)])[0]
                keep = output["scores"] >= score_threshold
                boxes = output["boxes"][keep][:max_detections].detach().cpu().tolist()
                scores = output["scores"][keep][:max_detections].detach().cpu().tolist()
                labels = [
                    id_to_label.get(int(c), str(int(c)))
                    for c in output["labels"][keep][:max_detections].detach().cpu().tolist()
                ]
                masks = None
                if with_masks and "masks" in output:
                    masks = (
                        (output["masks"][keep][:max_detections] > 0.5)
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                predictions.append(
                    Prediction(sample_id=sample.id, boxes=boxes, labels=labels, scores=scores, masks=masks)
                )
        return predictions

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        """Copy a trained checkpoint out of its transient run directory into
        a stable, named location so it can be reloaded later.
        """

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing weights: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        variant = artifact.metadata.get("variant") or resolve_variant(self.name)
        filename = model_name or f"{variant}_{Path(artifact.metadata.get('run_dir', 'run')).name}.pt"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(self, artifact_or_path: Artifact | str) -> Artifact:
        """Load a previously registered/trained model back into this adapter."""

        path = artifact_or_path.path if isinstance(artifact_or_path, Artifact) else artifact_or_path
        self.load_weights(path)
        if isinstance(artifact_or_path, Artifact):
            return artifact_or_path
        return Artifact(path=str(path), backend=self.backend, metadata={"variant": resolve_variant(self.name)})

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export(self, artifact: Artifact, target: str, config: dict[str, Any] | None = None) -> ExportedArtifact:
        """`target`: 'torchscript' (primary, officially supported by
        torchvision's Faster/Mask R-CNN) or 'onnx' (best-effort — see class
        docstring; a caveat is attached to `metadata['warning']` if the
        export raises and is caught, but you should still smoke-test the
        result).
        """

        import torch

        self.load_weights(artifact.path)
        self.model.eval()

        run_dir = Path(artifact.metadata.get("run_dir", "runs/fabric_defect_hub_tv"))
        run_dir.mkdir(parents=True, exist_ok=True)

        if target == "torchscript":
            scripted = torch.jit.script(self.model)
            out_path = run_dir / "model.torchscript.pt"
            scripted.save(str(out_path))
            return ExportedArtifact(path=str(out_path), target=target, metadata={"source_weights": artifact.path})

        if target == "onnx":
            cfg = config or {}
            opset = cfg.get("opset", 17)
            out_path = run_dir / "model.onnx"
            dummy = [torch.rand(3, 800, 800, device=self._device)]
            metadata: dict[str, Any] = {"source_weights": artifact.path, "opset": opset}
            try:
                torch.onnx.export(
                    self.model, (dummy,), str(out_path),
                    input_names=["images"], output_names=["boxes", "labels", "scores"],
                    opset_version=opset, dynamic_axes={"images": {0: "batch"}},
                )
            except Exception as exc:  # torchvision detection ONNX export is best-effort (see docstring)
                metadata["warning"] = (
                    f"ONNX export raised {type(exc).__name__}: {exc}. Faster/Mask R-CNN's "
                    "internal NMS/RoIAlign ops have inconsistent ONNX opset coverage across "
                    "torch versions; prefer target='torchscript' for a supported export path."
                )
            return ExportedArtifact(path=str(out_path), target=target, metadata=metadata)

        raise ValueError(f"unsupported export target {target!r}; expected 'torchscript' or 'onnx'.")


def _append_history_row(path: Path, log) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["epoch", "train_loss", "lr", *sorted(log.val_metrics)])
        writer.writerow(
            [log.epoch, f"{log.train_loss:.6f}", f"{log.lr:.8f}", *(f"{log.val_metrics[k]:.6f}" for k in sorted(log.val_metrics))]
        )
