"""Per-algorithm configuration for the five anomalib models the README
commits to: PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet.

We deliberately do not reimplement these algorithms (see the project's
"don't reimplement models we can depend on" principle in the top-level
README) — `anomalib` already ships correct, maintained implementations.
What *is* our responsibility, and what lives here, is:

1. Alias resolution: the README/paper names ("RD4AD", "EfficientAD",
   "SuperSimpleNet") don't match anomalib's class names
   (`ReverseDistillation`, `EfficientAd`, `Supersimplenet`). Callers should
   be able to use either.
2. Fabric-tailored defaults: anomalib's own defaults are tuned for MVTec-AD
   style single-object images at 256x256. `MODEL_PRESETS` below adjusts the
   handful of knobs that matter for fabric texture images (uniform
   background, defects are small/local, and — per `ZJULeaperDataset` — the
   caller can dial in a low-shot or zero-shot sample count), while still
   letting `train_config['model_kwargs']` override anything.

Verified against anomalib 2.5.0 (`pip install -e ".[anomalib]"`); all five
classes and constructor kwargs below were introspected directly from that
installed version, not guessed.
"""

from __future__ import annotations

from typing import Any

# README/paper name -> anomalib.models class name.
MODEL_ALIASES: dict[str, str] = {
    "patchcore": "Patchcore",
    "padim": "Padim",
    "rd4ad": "ReverseDistillation",
    "reversedistillation": "ReverseDistillation",
    "efficientad": "EfficientAd",
    "supersimplenet": "Supersimplenet",
}

# Fabric-tailored default `model_kwargs` per algorithm, keyed by the
# anomalib class name (post-alias-resolution). Every key here is a real
# constructor parameter on the corresponding anomalib class.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # Memory-bank of nominal patch features + kNN distance. Works well with
    # few normal reference images (our low-shot regime) since it needs no
    # gradient training, only a forward pass + coreset subsampling.
    "Patchcore": {
        "backbone": "wide_resnet50_2",
        "layers": ("layer2", "layer3"),
        "pre_trained": True,
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
    },
    # Per-patch multivariate Gaussian over pretrained features. Cheap and
    # stable with a moderate number of normal images; good zero-shot-style
    # baseline (no defect images needed at all, per `use_defect=False`).
    "Padim": {
        "backbone": "resnet18",
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
        "n_features": None,  # None = anomalib's per-backbone default
    },
    # Teacher/student reverse distillation. Needs an actual training loop
    # (unlike PatchCore/PaDiM), so benefits from more normal samples.
    "ReverseDistillation": {
        "backbone": "wide_resnet50_2",
        "layers": ("layer1", "layer2", "layer3"),
        "pre_trained": True,
        "anomaly_map_mode": "add",
    },
    # Distillation against a pretrained teacher, regularized on a generic
    # natural-image set (`imagenet_dir`). Fabric texture has no object
    # semantics, so `imagenet_dir` MUST be supplied by the caller — there is
    # no fabric-appropriate default, and `AnomalibAdapter.train()` raises
    # early if it is missing rather than failing deep inside Lightning.
    "EfficientAd": {
        "model_size": "small",
        "teacher_out_channels": 384,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "imagenet_dir": None,
    },
    # The one supervised-capable model in this set. ZJU-Leaper ships real
    # defect images + masks (see `ZJULeaperDataset`), so default to
    # `supervised=True` to actually use them instead of synthetic Perlin
    # anomalies; set `use_defect=False` upstream to fall back to the
    # unsupervised synthetic-anomaly mode anomalib was designed for.
    "Supersimplenet": {
        "backbone": "wide_resnet50_2.tv_in1k",
        "layers": ["layer2", "layer3"],
        "supervised": True,
        "perlin_threshold": 0.2,
    },
}


def resolve_model_class_name(name: str) -> str:
    """Map a README alias or literal anomalib class name to the anomalib class name."""

    canonical = MODEL_ALIASES.get(name.strip().lower())
    if canonical is not None:
        return canonical
    if name in MODEL_PRESETS:
        return name
    known = sorted(set(MODEL_ALIASES) | set(MODEL_PRESETS))
    raise KeyError(f"unknown anomalib model {name!r}. Known names: {known}")


def resolve_model_class(name: str):
    """Return the actual `anomalib.models.<Class>` for a README alias or class name."""

    import anomalib.models as anomalib_models

    class_name = resolve_model_class_name(name)
    return getattr(anomalib_models, class_name)


def default_model_kwargs(name: str) -> dict[str, Any]:
    """Fabric-tailored default constructor kwargs for `name` (copy, safe to mutate)."""

    class_name = resolve_model_class_name(name)
    return dict(MODEL_PRESETS.get(class_name, {}))


def list_supported_models() -> list[str]:
    return sorted(MODEL_PRESETS)
