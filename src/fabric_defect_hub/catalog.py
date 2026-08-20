"""Canonical model catalog connecting trained artifacts to the frontend."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models"
PUBLISHED_MODEL_ROOT = MODEL_ROOT / "published"
QUANTIZED_MODEL_ROOT = MODEL_ROOT / "quantized"


@dataclass(frozen=True)
class CanonicalModel:
    key: str  # stable model identifier
    backend: str  # backend engine name
    variant: str  # model variant / architecture name
    task: str  # task type (detection, segmentation, anomaly, etc.)
    config: str  # relative path to config YAML
    label: str  # display label in UI
    source: str  # human-facing provenance string


CANONICAL_MODELS: list[CanonicalModel] = [
    # -- Ultralytics: detection -------------------------------------------
    CanonicalModel("yolov8n", "ultralytics", "yolov8n", "detection",
                    "ultralytics_example.yaml", "YOLOv8n · Fabric trained", "local trained artifact"),
    CanonicalModel("yolov8s", "ultralytics", "yolov8s", "detection",
                    "ultralytics_example.yaml", "YOLOv8s · Fabric trained", "local trained artifact"),
    CanonicalModel("yolo11n", "ultralytics", "yolo11n", "detection",
                    "ultralytics_example.yaml", "YOLO11n · Fabric trained", "local trained artifact"),
    # -- torchvision: detection -------------------------------------------
    CanonicalModel("fasterrcnn_resnet50_fpn", "torchvision", "fasterrcnn_resnet50_fpn", "detection",
                    "torchvision_example.yaml", "Faster R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("cascadercnn_resnet50_fpn", "torchvision", "cascadercnn_resnet50_fpn", "detection",
                    "torchvision_example.yaml", "Cascade R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("detr_resnet50", "torchvision", "detr_resnet50", "detection",
                    "torchvision_example.yaml", "DETR · Fabric trained", "local trained artifact"),
    # -- torchvision: segmentation ----------------------------------------
    CanonicalModel("maskrcnn_resnet50_fpn", "torchvision", "maskrcnn_resnet50_fpn", "instance_segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "Mask R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("unetplusplus_resnet34", "torchvision", "unetplusplus_resnet34", "segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "UNet++ · Fabric trained", "local trained artifact"),
    CanonicalModel("deeplabv3plus_resnet50", "torchvision", "deeplabv3plus_resnet50", "segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "DeepLabV3+ · Fabric trained", "local trained artifact"),
    # -- Anomalib: anomaly ------------------------------------------------
    CanonicalModel("PatchCore", "anomalib", "PatchCore", "anomaly",
                    "anomalib_example.yaml", "PatchCore · Normal Lab trained", "Normal Lab"),
    CanonicalModel("PaDiM", "anomalib", "PaDiM", "anomaly",
                    "anomalib_example.yaml", "PaDiM · Normal Lab trained", "Normal Lab"),
    CanonicalModel("RD4AD", "anomalib", "RD4AD", "anomaly",
                    "anomalib_example.yaml", "RD4AD · Normal Lab trained", "Normal Lab"),
    CanonicalModel("EfficientAD", "anomalib", "EfficientAD", "anomaly",
                    "anomalib_efficientad.yaml", "EfficientAD · Normal Lab trained", "Normal Lab"),
    CanonicalModel("SuperSimpleNet", "anomalib", "SuperSimpleNet", "anomaly",
                    "anomalib_example.yaml", "SuperSimpleNet · Normal Lab trained", "Normal Lab"),
    CanonicalModel("STFPM", "anomalib", "STFPM", "anomaly",
                    "anomalib_stfpm.yaml", "STFPM · Normal Lab trained", "Normal Lab"),
    CanonicalModel("GANomaly", "anomalib", "GANomaly", "anomaly",
                    "anomalib_ganomaly.yaml", "GANomaly · Normal Lab trained", "Normal Lab"),
    # -- Zero-shot & Research models --------------------------------------
    CanonicalModel("WinCLIP", "anomalib", "WinClip", "anomaly",
                    "anomalib_winclip.yaml", "WinCLIP · Zero-shot", "Zero-shot CLIP"),
    CanonicalModel("Dinomaly", "dinomaly", "dinov2reg_vit_base_14", "anomaly",
                    "dinomaly_example.yaml", "Dinomaly · Normal Lab trained", "Normal Lab"),
    CanonicalModel("MoECLIP", "moeclip", "ViT-L-14-336", "anomaly",
                    "moeclip_example.yaml", "MoECLIP · Zero-shot", "Zero-shot CLIP (VisA-trained)"),
    CanonicalModel("MambaAD", "mambaad", "resnet34", "anomaly",
                    "mambaad_example.yaml", "MambaAD · Normal Lab trained", "Normal Lab"),
]

_BY_KEY: dict[str, CanonicalModel] = {model.key: model for model in CANONICAL_MODELS}
_EXTENSION = {
    "ultralytics": ".pt", "torchvision": ".pt", "anomalib": ".ckpt",
    "dinomaly": ".pth", "moeclip": ".pth", "mambaad": ".pth",
}


def find_canonical_model(backend: str, variant: str) -> CanonicalModel | None:
    """Find a CanonicalModel by backend and variant name."""

    needle = variant.strip().lower()
    for model in CANONICAL_MODELS:
        if model.backend == backend and model.variant.strip().lower() == needle:
            return model
    return None


def find_canonical_model_by_key(key: str) -> CanonicalModel:
    """Find a CanonicalModel by stable key name."""

    match = _BY_KEY.get(key) or next(
        (model for model in CANONICAL_MODELS if model.key.lower() == key.strip().lower()), None
    )
    if match is None:
        raise KeyError(f"unknown model key {key!r}. Known keys: {sorted(_BY_KEY)}")
    return match


def published_path(model: CanonicalModel) -> Path:
    """Return the destination path for published model weights.

    The name is the canonical key plus the backend's extension — no run
    number, no `best`/`last` suffix, no timestamp. Whatever the training run
    called the file, the published slot always spells it the way the rest of
    the project refers to the model.
    """
    return PUBLISHED_MODEL_ROOT / f"{model.key}{_EXTENSION[model.backend]}"


def published_status(path: Path) -> str:
    """What state a published slot is in: one of `PUBLISHED_STATES`.

    `published/` is deliberately allowed to hold **either** a real file or a
    symlink into `artifacts/models/`. `publish_artifact` writes symlinks (one
    copy of the bytes, and `ls -l` shows which run each model came from), but
    a tree copied down from the training box arrives as real files, and that
    has to keep working without a migration step.

    The state worth naming separately is `broken_link`: a symlink whose
    target did not come along — the normal outcome of copying `published/`
    without `artifacts/models/`. Every reader used to answer this with a bare
    `path.is_file()`, which is False for a dangling link and so reported the
    model as simply "not published", sending the reader off to retrain
    something they already have. It is a different problem with a different
    fix, so it gets a different answer.
    """

    if path.is_symlink():
        # `is_file()` follows the link, so this order matters: a broken link
        # is a symlink that resolves to nothing.
        return "symlink" if path.exists() else "broken_link"
    if path.is_file():
        return "file"
    return "missing"


PUBLISHED_STATES = ("file", "symlink", "broken_link", "missing")


def published_is_usable(path: Path) -> bool:
    """Whether a published slot can actually be loaded — a real file or a
    symlink that still resolves. The check every reader wants."""

    return published_status(path) in {"file", "symlink"}


def describe_published(path: Path) -> str:
    """One line a caller can put in an error message or a UI cell."""

    state = published_status(path)
    if state == "file":
        return f"{path} (regular file)"
    if state == "symlink":
        return f"{path} -> {os.readlink(path)}"
    if state == "broken_link":
        return (
            f"{path} is a symlink to {os.readlink(path)!r}, which does not exist. "
            "If this tree was copied from the training machine, copy "
            "artifacts/models/ across too (the link points inside it), or replace "
            "the link with the weight file itself."
        )
    return f"{path} (nothing published)"


def quantized_path(backend: str, variant: str, level: str) -> Path:
    """Where a quantized export of `(backend, variant)` at `level` lives.

    Quantized artifacts used to have no home at all: `quantize_onnx` wrote
    wherever its caller pointed it, so the result never reached
    `weight_manifest.jsonl` and nothing could later say which fp32 weights,
    which commit, or which calibration set produced it. Giving the layout a
    single owner here is what lets `weight_registry.record_quantized_weight`
    attach the same provenance block the fp32 weights get.

    One file per level rather than per run: a quantized export is a
    derivative of an fp32 checkpoint, so the fp32 record is what carries run
    identity, and keeping every historical INT8 blob would multiply the
    storage problem this layout exists to contain.
    """

    return QUANTIZED_MODEL_ROOT / backend / _slugify(variant) / f"{_slugify(level)}.onnx"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "model"


def metadata_for(model: CanonicalModel) -> dict:
    """Return runtime metadata dictionary for a published model."""

    if model.backend == "anomalib":
        from fabric_defect_hub.models.anomalib.presets import (
            default_model_kwargs,
            resolve_model_class_name,
        )

        model_class = resolve_model_class_name(model.variant)
        metadata = {
            "trusted": True,
            "source": model.source,
            "model_class": model_class,
        }
        # WinCLIP's canonical artifact is a metadata handle, not a serialized
        # Lightning checkpoint: k_shot=0 has no learned state to restore.
        # Carry the reconstruction metadata into the UI session so the
        # adapter can instantiate WinClip instead of calling torch.load().
        if model_class == "WinClip":
            metadata.update({
                "zero_shot": True,
                "model_kwargs": default_model_kwargs(model.variant),
            })
        return metadata
    if model.backend == "dinomaly":
        from fabric_defect_hub.models.dinomaly.presets import DEFAULT_TRAIN_KWARGS, encoder_preset

        return {
            "trusted": True,
            "source": model.source,
            "model_class": "ViTill",
            "encoder_name": model.variant,
            "target_layers": encoder_preset(model.variant)["target_layers"],
            "image_size": DEFAULT_TRAIN_KWARGS["image_size"],
            "crop_size": DEFAULT_TRAIN_KWARGS["crop_size"],
        }
    if model.backend == "moeclip":
        from fabric_defect_hub.models.moeclip.presets import default_arch_kwargs

        return {
            "trusted": True,
            "source": model.source,
            "model_class": "MoECLIP",
            "model_name": model.variant,
            **default_arch_kwargs(),
        }
    if model.backend == "mambaad":
        from fabric_defect_hub.models.mambaad.presets import (
            DEFAULT_TRAIN_KWARGS, D_STATE, DEPTHS_DECODER, DIMS_DECODER,
            DEFAULT_NUM_DIRECTION, DEFAULT_SCAN_TYPE, DROP_PATH_RATE,
        )

        return {
            "trusted": True,
            "source": model.source,
            "model_class": "MambaADNet",
            "encoder_name": model.variant,
            "image_size": DEFAULT_TRAIN_KWARGS["image_size"],
            "dims_decoder": list(DIMS_DECODER),
            "depths_decoder": list(DEPTHS_DECODER),
            "d_state": D_STATE,
            "drop_path_rate": DROP_PATH_RATE,
            "scan_type": DEFAULT_SCAN_TYPE,
            "num_direction": DEFAULT_NUM_DIRECTION,
        }
    return {"trusted": True, "source": model.source}


def publish_artifact(backend: str, variant: str, registered_artifact_path: str) -> Path | None:
    """Point the published slot for `(backend, variant)` at a registered
    checkpoint, via a *relative* symlink.

    This used to be `shutil.copy2`, which meant every published model existed
    twice on disk — 2.1 GB of GANomaly weights in `artifacts/models/` and
    another 2.1 GB under `published/`. A symlink keeps the same two names and
    the same clean published spelling while storing the bytes once, and it
    makes the relationship inspectable: `ls -l published/` now tells you
    exactly which run each published model came from, which a copy could
    never do.

    The link is relative (`../<name>`) so the whole `artifacts/` tree stays
    movable — an absolute link would break the moment the directory is synced
    to a different machine or a different checkout path, which is the normal
    case here (training happens on a cloud box, inspection happens locally).

    The registered artifact is the single source of truth and must live under
    `artifacts/models/`; nothing is copied into `published/` ever again, so
    deleting a registered checkpoint now visibly breaks its published link
    rather than silently leaving a stale duplicate behind. `retention.py`
    treats symlink targets as protected for exactly this reason.
    """

    model = find_canonical_model(backend, variant)
    if model is None:
        return None
    source = Path(registered_artifact_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"cannot publish missing weights: {source}")

    destination = published_path(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _relink(source, destination)
    return destination


def _relink(source: Path, destination: Path) -> None:
    """Replace `destination` with a relative symlink to `source`.

    `os.symlink` refuses to overwrite, and the published slot is overwritten
    on every re-publish, so the old entry has to go first — including when it
    is a leftover *copy* from before this project used symlinks.
    """

    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(target)
