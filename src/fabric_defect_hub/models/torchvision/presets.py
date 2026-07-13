"""Model-variant registry, fabric-tailored training defaults, and
augmentation presets for the torchvision detection backend
(Faster R-CNN, Mask R-CNN — see the top-level README for why
`torchvision.models.detection` replaces `mmdetection` here: mmdetection's
`mmcv` dependency has no macOS arm64 wheel and no Python 3.14 support,
confirmed by a real install attempt in this environment; torchvision is
pure PyTorch, actively maintained, and installs everywhere torch does).

Mirrors `models/anomalib/presets.py` and `models/ultralytics/presets.py`:
we do not reimplement Faster/Mask R-CNN — `torchvision.models.detection`
already ships correct, maintained implementations (including COCO-pretrained
weights). What lives here is (1) resolution of friendly variant names to
the actual factory functions/weights enums, and (2) fine-tuning defaults
tuned for small fabric-defect datasets rather than COCO-scale training.

Every symbol referenced below was introspected against the installed
torchvision 0.27.1 (see `pip install -e ".[torchvision]"`), not guessed:
the pretrained-weights-plus-custom-num_classes incompatibility, the
`FastRCNNPredictor`/`MaskRCNNPredictor` head-swap recipe, and the
`tv_tensors`-based joint image/box/mask augmentation API were all verified
live before this module was written.
"""

from __future__ import annotations

from typing import Any

# Friendly variant name -> torchvision factory function name + weights enum
# name (both resolved lazily in `build_model()` so importing this module
# never requires torchvision to be installed).
MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "fasterrcnn_resnet50_fpn": {
        "factory": "fasterrcnn_resnet50_fpn",
        "weights_enum": "FasterRCNN_ResNet50_FPN_Weights",
        "task": "detect",
    },
    "fasterrcnn_resnet50_fpn_v2": {
        "factory": "fasterrcnn_resnet50_fpn_v2",
        "weights_enum": "FasterRCNN_ResNet50_FPN_V2_Weights",
        "task": "detect",
    },
    "maskrcnn_resnet50_fpn": {
        "factory": "maskrcnn_resnet50_fpn",
        "weights_enum": "MaskRCNN_ResNet50_FPN_Weights",
        "task": "instance_segmentation",
    },
    "maskrcnn_resnet50_fpn_v2": {
        "factory": "maskrcnn_resnet50_fpn_v2",
        "weights_enum": "MaskRCNN_ResNet50_FPN_V2_Weights",
        "task": "instance_segmentation",
    },
}

VARIANT_ALIASES: dict[str, str] = {
    "fasterrcnn": "fasterrcnn_resnet50_fpn",
    "faster_rcnn": "fasterrcnn_resnet50_fpn",
    "faster-rcnn": "fasterrcnn_resnet50_fpn",
    "fasterrcnn_v2": "fasterrcnn_resnet50_fpn_v2",
    "faster_rcnn_v2": "fasterrcnn_resnet50_fpn_v2",
    "maskrcnn": "maskrcnn_resnet50_fpn",
    "mask_rcnn": "maskrcnn_resnet50_fpn",
    "mask-rcnn": "maskrcnn_resnet50_fpn",
    "maskrcnn_v2": "maskrcnn_resnet50_fpn_v2",
    "mask_rcnn_v2": "maskrcnn_resnet50_fpn_v2",
}

# Fine-tuning defaults for small fabric datasets. torchvision's own
# reference detection training script (references/detection/train.py) uses
# SGD lr=0.005/momentum=0.9/weight_decay=0.0005 with a 3-epoch StepLR decay
# for training Faster/Mask R-CNN *from ImageNet init* on full COCO — those
# numbers are tuned for a much bigger dataset trained for much longer. Since
# we start from full COCO-pretrained detection weights (see
# `adapter.py::_build_model`) and fine-tune on a low-/few-shot fabric
# selection, we use a lower learning rate (less to relearn) and a much
# longer patience-based early stop instead of a fixed decay schedule sized
# for COCO's epoch count.
COMMON_FABRIC_TRAIN_DEFAULTS: dict[str, Any] = {
    "epochs": 30,
    "batch_size": 4,
    "optimizer": "sgd",  # 'sgd' | 'adamw'
    "lr": 0.002,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "cosine",  # 'cosine' | 'step' | 'none'
    "step_size": 10,
    "gamma": 0.1,
    "warmup_epochs": 1,  # linear LR warmup within the first epoch, per torchvision's reference recipe
    "grad_clip_norm": 5.0,
    "patience": 8,  # early stop on no val-mAP improvement
    "num_workers": 2,
    "trainable_backbone_layers": 3,  # of 5; freeze the earliest (most generic) ResNet stages
    # --- augmentation, retuned for small texture defects (see build_transforms) ---
    "hflip_prob": 0.5,
    "vflip_prob": 0.5,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.02},
}

VARIANT_TRAIN_OVERRIDES: dict[str, dict[str, Any]] = {
    "fasterrcnn_resnet50_fpn": {},
    "fasterrcnn_resnet50_fpn_v2": {"lr": 0.0015},  # v2 head is heavier; slightly gentler LR
    "maskrcnn_resnet50_fpn": {"batch_size": 2},  # masks triple memory use per image
    "maskrcnn_resnet50_fpn_v2": {"batch_size": 2, "lr": 0.0015},
}


def resolve_variant(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in MODEL_VARIANTS:
        return key
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    known = sorted(MODEL_VARIANTS)
    raise KeyError(f"unknown torchvision detection variant {name!r}. Known variants: {known}")


def variant_task(name: str) -> str:
    """'detect' (Faster R-CNN) or 'instance_segmentation' (Mask R-CNN)."""

    return MODEL_VARIANTS[resolve_variant(name)]["task"]


def uses_masks(name: str) -> bool:
    return variant_task(name) == "instance_segmentation"


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    backbone_weights: bool = True,
):
    """Construct the torchvision detection model for `name`, sized for
    `num_classes` foreground classes + background (i.e. pass
    `len(class_names) + 1`).

    `pretrained=True` loads full COCO-pretrained detection weights, then
    swaps the classifier (and mask predictor, for Mask R-CNN) head to match
    `num_classes` — the standard torchvision fine-tuning recipe. Passing
    `num_classes` straight into the factory together with COCO weights is
    *not* supported by torchvision (raises `ValueError`: expects 91); the
    head-swap is the only way to combine pretrained features with a custom
    class count.

    `pretrained=False` builds from an ImageNet-pretrained backbone only
    (random-init detection head), for training from scratch — unless
    `backbone_weights=False`, which skips even that download. Use
    `backbone_weights=False` when you're about to overwrite every parameter
    with a full `state_dict` anyway (see `adapter.load_weights`):
    downloading ImageNet weights just to discard them immediately is wasted
    network I/O, and makes checkpoint reloading needlessly dependent on
    internet access.

    `min_size`/`max_size` override the model's internal `GeneralizedRCNNTransform`
    resize range (default 800/1333, tuned for COCO photos). ZJU-Leaper images
    are 512x512, so the default *upscales* them for no benefit; smaller
    values speed up both training and inference with little accuracy cost on
    fabric's uniform, non-photographic content.
    """

    import torchvision.models.detection as tv_detection

    variant = resolve_variant(name)
    spec = MODEL_VARIANTS[variant]
    factory = getattr(tv_detection, spec["factory"])
    size_kwargs: dict[str, Any] = {}
    if min_size is not None:
        size_kwargs["min_size"] = min_size
    if max_size is not None:
        size_kwargs["max_size"] = max_size

    if pretrained:
        weights_enum = getattr(tv_detection, spec["weights_enum"])
        model = factory(
            weights=weights_enum.DEFAULT,
            trainable_backbone_layers=trainable_backbone_layers,
            **size_kwargs,
        )
        _replace_head(model, num_classes, with_mask=uses_masks(variant))
    else:
        no_download_kwargs: dict[str, Any] = {} if backbone_weights else {"weights_backbone": None}
        model = factory(
            weights=None,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            **size_kwargs,
            **no_download_kwargs,
        )
    return model


def _replace_head(model, num_classes: int, with_mask: bool) -> None:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if with_mask:
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)


def default_train_kwargs(name: str) -> dict[str, Any]:
    """Fabric-tailored default training kwargs for `name` (copy, safe to mutate)."""

    variant = resolve_variant(name)
    merged = dict(COMMON_FABRIC_TRAIN_DEFAULTS)
    merged.update(VARIANT_TRAIN_OVERRIDES.get(variant, {}))
    return merged


def build_transforms(train: bool, hflip_prob: float = 0.5, vflip_prob: float = 0.5, color_jitter: dict | None = None):
    """Box/mask-aware augmentation pipeline via `torchvision.transforms.v2`.

    Fabric texture has no preferred orientation and mirror symmetry is
    label-preserving, so both flips are safe free augmentation (unlike
    COCO's natural-object-photo default of horizontal-flip-only). Colour
    jitter is kept mild — some fabric defects (stains, discoloration) are
    themselves colour cues we don't want to wash out. Eval/predict gets no
    augmentation, only the dtype conversion.
    """

    import torch
    from torchvision.transforms import v2

    if not train:
        return v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    color_jitter = color_jitter or {}
    steps = [
        v2.RandomHorizontalFlip(p=hflip_prob),
        v2.RandomVerticalFlip(p=vflip_prob),
        v2.ColorJitter(**color_jitter),
        v2.ToDtype(torch.float32, scale=True),
    ]
    return v2.Compose(steps)


def list_supported_variants() -> list[str]:
    return sorted(MODEL_VARIANTS)
