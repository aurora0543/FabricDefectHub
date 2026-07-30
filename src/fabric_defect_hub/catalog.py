"""Canonical model catalog connecting trained artifacts to the frontend."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "published"


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
                    "anomalib_example.yaml", "EfficientAD · Normal Lab trained", "Normal Lab"),
    CanonicalModel("SuperSimpleNet", "anomalib", "SuperSimpleNet", "anomaly",
                    "anomalib_example.yaml", "SuperSimpleNet · Normal Lab trained", "Normal Lab"),
    CanonicalModel("STFPM", "anomalib", "STFPM", "anomaly",
                    "anomalib_stfpm.yaml", "STFPM · Normal Lab trained", "Normal Lab"),
    CanonicalModel("GANomaly", "anomalib", "GANomaly", "anomaly",
                    "anomalib_ganomaly.yaml", "GANomaly · Normal Lab trained", "Normal Lab"),
    # -- Zero-shot & Research models --------------------------------------
    CanonicalModel("WinCLIP", "anomalib", "WinClip", "anomaly",
                    "anomalib_example.yaml", "WinCLIP · Zero-shot", "Zero-shot CLIP"),
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
    """Return the destination path for published model weights."""
    return PUBLISHED_MODEL_ROOT / f"{model.key}{_EXTENSION[model.backend]}"


def metadata_for(model: CanonicalModel) -> dict:
    """Return runtime metadata dictionary for a published model."""

    if model.backend == "anomalib":
        from fabric_defect_hub.models.anomalib.presets import resolve_model_class_name

        return {
            "trusted": True,
            "source": model.source,
            "model_class": resolve_model_class_name(model.variant),
        }
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
    """Copy a newly registered model checkpoint to the published model root."""

    model = find_canonical_model(backend, variant)
    if model is None:
        return None
    destination = published_path(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registered_artifact_path, destination)
    return destination
