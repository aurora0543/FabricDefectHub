"""Single source of truth connecting `fdh train` output to the frontend.

`fdh train` registers each run under a run-specific filename (see
`training.apply_model_overrides` — `<variant>_<checkpoint-name>.pt`), which
is right for experimentation (nothing clobbers a previous run) but wrong
for the frontend: `web/single_image.py`'s `MODEL_CATALOG` needs one *stable*
path per model to load. `publish_artifact` bridges the two: after a
registered run for a canonical model finishes, it is copied to a fixed
"published" location that the frontend always reads from.

CANONICAL_MODELS is intentionally NOT "every variant this backend can run"
(torchvision alone has 15 registered variants, see
`models/torchvision/presets.MODEL_VARIANTS`) — it is the specific set of
models this project trains and exposes to the frontend: 3 Ultralytics
variants, 6 torchvision model families (one representative backbone each —
Faster R-CNN, Mask R-CNN, Cascade R-CNN, DETR, UNet++, DeepLabV3+ — not
every VGG16/ShuffleNet backbone swap of each), all 6 Anomalib models
(including zero-shot WinCLIP), and the vendored Dinomaly research model.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "published"


@dataclass(frozen=True)
class CanonicalModel:
    key: str  # stable id, also the published filename stem
    backend: str  # "ultralytics" | "torchvision" | "anomalib" | "dinomaly"
    variant: str  # model.variant (ultralytics/torchvision) or model.name (anomalib/dinomaly)
    task: str  # "detection" | "instance_segmentation" | "segmentation" | "anomaly"
    config: str  # config filename under configs/models/ used to train + publish this model
    label: str  # frontend dropdown label
    source: str  # human-facing provenance string for the frontend status line


CANONICAL_MODELS: list[CanonicalModel] = [
    # -- Ultralytics: detection -------------------------------------------
    CanonicalModel("yolov8n", "ultralytics", "yolov8n", "detection",
                    "ultralytics_example.yaml", "YOLOv8n · Fabric trained", "local trained artifact"),
    CanonicalModel("yolov8s", "ultralytics", "yolov8s", "detection",
                    "ultralytics_example.yaml", "YOLOv8s · Fabric trained", "local trained artifact"),
    CanonicalModel("yolo11n", "ultralytics", "yolo11n", "detection",
                    "ultralytics_example.yaml", "YOLO11n · Fabric trained", "local trained artifact"),
    # -- torchvision: detection (boxes; needs ZJU-Leaper-style box data) --
    CanonicalModel("fasterrcnn_resnet50_fpn", "torchvision", "fasterrcnn_resnet50_fpn", "detection",
                    "torchvision_example.yaml", "Faster R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("cascadercnn_resnet50_fpn", "torchvision", "cascadercnn_resnet50_fpn", "detection",
                    "torchvision_example.yaml", "Cascade R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("detr_resnet50", "torchvision", "detr_resnet50", "detection",
                    "torchvision_example.yaml", "DETR · Fabric trained", "local trained artifact"),
    # -- torchvision: segmentation (masks; works on all 3 datasets) ------
    CanonicalModel("maskrcnn_resnet50_fpn", "torchvision", "maskrcnn_resnet50_fpn", "instance_segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "Mask R-CNN · Fabric trained", "local trained artifact"),
    CanonicalModel("unetplusplus_resnet34", "torchvision", "unetplusplus_resnet34", "segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "UNet++ · Fabric trained", "local trained artifact"),
    CanonicalModel("deeplabv3plus_resnet50", "torchvision", "deeplabv3plus_resnet50", "segmentation",
                    "torchvision_maskrcnn_segmentation.yaml", "DeepLabV3+ · Fabric trained", "local trained artifact"),
    # -- Anomalib: anomaly (all 5 registered models) ----------------------
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
    # -- WinCLIP: CLIP-based, zero-shot by default (no fabric training data) --
    CanonicalModel("WinCLIP", "anomalib", "WinClip", "anomaly",
                    "anomalib_example.yaml", "WinCLIP · Zero-shot", "Zero-shot CLIP"),
    # -- Dinomaly: anomaly (vendored research model, see components/README.md) --
    CanonicalModel("Dinomaly", "dinomaly", "dinov2reg_vit_base_14", "anomaly",
                    "dinomaly_example.yaml", "Dinomaly · Normal Lab trained", "Normal Lab"),
]

_BY_KEY: dict[str, CanonicalModel] = {model.key: model for model in CANONICAL_MODELS}
_EXTENSION = {"ultralytics": ".pt", "torchvision": ".pt", "anomalib": ".ckpt", "dinomaly": ".pth"}


def find_canonical_model(backend: str, variant: str) -> CanonicalModel | None:
    """Match a resolved (backend, variant/name) pair — case-insensitively
    for anomalib, whose names are conventionally case-varied aliases of the
    same model (PatchCore/Patchcore) — against `CANONICAL_MODELS`. Returns
    `None` for any run that isn't one of this project's published models
    (a one-off variant sweep, an ad-hoc backbone swap, ...); such runs are
    still fully usable via `fdh predict`, they just don't get published to
    the frontend's fixed slot.
    """

    needle = variant.strip().lower()
    for model in CANONICAL_MODELS:
        if model.backend == backend and model.variant.strip().lower() == needle:
            return model
    return None


def published_path(model: CanonicalModel) -> Path:
    return PUBLISHED_MODEL_ROOT / f"{model.key}{_EXTENSION[model.backend]}"


def metadata_for(model: CanonicalModel) -> dict:
    """`Artifact.metadata` for a published model, in the shape every
    backend's `load_trained_model`/`predict` expects (see
    `web/single_image.py:artifact_for_model`). Anomalib needs `trusted` +
    `model_class` (the literal anomalib class name, e.g. "Patchcore" for
    the "PatchCore" alias); Dinomaly needs `encoder_name`/`target_layers`/
    `image_size`/`crop_size` (the architecture it was built with -- see
    `DinomalyAdapter._build_model`) — both resolved here rather than
    hand-typed onto each `CanonicalModel` entry, so a presets.py rename
    can't silently drift out of sync with this catalog.
    """

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
    return {"trusted": True, "source": model.source}


def publish_artifact(backend: str, variant: str, registered_artifact_path: str) -> Path | None:
    """Copy a freshly-registered training artifact to its fixed, frontend-
    facing location (see module docstring). Returns the published path, or
    `None` if (backend, variant) isn't one of `CANONICAL_MODELS` — i.e.
    nothing to publish, not an error.
    """

    model = find_canonical_model(backend, variant)
    if model is None:
        return None
    destination = published_path(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registered_artifact_path, destination)
    return destination
