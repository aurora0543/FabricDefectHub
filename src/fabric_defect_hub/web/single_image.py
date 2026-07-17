"""State and inference helpers for the single-image Gradio workspace."""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from typing import Any

from fabric_defect_hub.catalog import CANONICAL_MODELS, metadata_for, published_path
from fabric_defect_hub.core.serialization import sample_from_dict, sample_to_dict
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.loader import load_dataset, load_model
from fabric_defect_hub.models.anomalib.checkpoint import inspect_checkpoint
from fabric_defect_hub.models.base import Artifact


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ANOMALY_MAP_ROOT = PROJECT_ROOT / "artifacts" / "runtime" / "anomaly_maps"
_SSD_VOLUME_PARENT = Path("anomaly-detection-challenges") / "datasets"

# Generated from `catalog.CANONICAL_MODELS` — the same list `fdh train`
# publishes to (see `catalog.publish_artifact`, called from
# `training.run_train`) — so training a canonical model and refreshing this
# page is the entire "get it into the UI" workflow; no manual catalog edits.
# `checkpoint` points at each model's fixed *published* path, not the
# run-specific path `fdh train` registers under (see catalog.py's docstring
# for why those differ).
MODEL_CATALOG = {
    model.label: {
        "backend": model.backend,
        "name": model.variant,
        "checkpoint": published_path(model),
        "task": model.task,
        "metadata": metadata_for(model),
    }
    for model in CANONICAL_MODELS
}

# Each entry names a registered DatasetAdapter plus the UI-facing metadata
# needed to locate it on disk and adapt the sampler controls (texture/category
# filter, default Sample.task) to that dataset's shape. `slice_kwarg` names the
# DatasetAdapter constructor kwarg the "Texture / pattern" dropdown feeds (e.g.
# "pattern" for ZJU-Leaper's textures, "category" for MVTec AD's object
# classes); None means the dataset has no such subdivision. Additional
# datasets can add a catalog entry here without changing the page interaction
# contract, as long as `texture_choices`/`slice_value` below know how to
# enumerate/resolve their slice_kwarg.
DATASET_CATALOG = {
    "ZJU-Leaper": {
        "name": "zju-leaper",
        "dir": "ZJU-Leaper",
        "env": "ZJU_LEAPER_ROOT",
        "slice_kwarg": "pattern",
        "task": "detection",
    },
    "RAW-FABRID": {
        "name": "raw-fabric",
        "dir": "RAW_FABRID",
        "env": "RAW_FABRIC_ROOT",
        "slice_kwarg": None,
        "task": "anomaly",
    },
    "MVTec AD": {
        "name": "mvtec-ad",
        "dir": "MVTec AD",
        "env": "MVTEC_AD_ROOT",
        "slice_kwarg": "category",
        "task": "anomaly",
    },
}
ALL_TEXTURES = "All textures"
ALL_IMAGES = "All images"
DEFECT_ONLY = "Defect only"
NORMAL_ONLY = "Normal only"
SHOT_FULL = "Full-shot"
SHOT_FEW = "Few-shot"
FEW_SHOT_SAMPLE_COUNT = 350
FEW_SHOT_DEFECT_RATIO = 0.3


def _dataset_roots(dataset_label: str) -> list[Path]:
    """Return likely local roots for `dataset_label`, including the
    repository's `data/<dir>` symlink convention and the SSD layout it
    typically points at. Roots are returned as-given (not yet resolved) so
    callers can decide whether to follow symlinks to the underlying storage.
    """

    spec = DATASET_CATALOG[dataset_label]
    roots: list[Path] = []
    configured = os.getenv(spec["env"])
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(PROJECT_ROOT / parent / spec["dir"] for parent in ("data", "Data"))
    volumes = Path("/Volumes")
    if volumes.is_dir():
        roots.extend(volume / _SSD_VOLUME_PARENT / spec["dir"] for volume in volumes.iterdir())
    return roots


def default_dataset_root(dataset_label: str = "ZJU-Leaper") -> str:
    """Resolve the selected dataset without exposing an editable path field.

    `data/<dir>` is expected to be a symlink onto external storage (an SSD,
    a mounted share, ...) rather than the data itself, so every candidate is
    resolved with `Path.resolve()` before being handed to the dataset
    adapter or to Gradio's `allowed_paths` — both need the real on-disk
    location, not the symlink path, to serve/read files reliably.
    """

    if dataset_label not in DATASET_CATALOG:
        raise KeyError(f"Unknown dataset selection {dataset_label!r}.")
    for candidate in _dataset_roots(dataset_label):
        if candidate.is_dir():
            return str(candidate.resolve())
    return ""


def _mvtec_ad_categories(root: str) -> list[str]:
    if not root:
        return []
    root_path = Path(root)
    return sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "train" / "good").is_dir()
    )


def texture_choices(dataset_label: str) -> list[str]:
    """Discover available texture/category slices from the registered
    dataset root. Datasets without a subdivision (`slice_kwarg` is None)
    only ever offer the "All textures" choice.
    """

    choices = [ALL_TEXTURES]
    spec = DATASET_CATALOG.get(dataset_label)
    if spec is None or spec["slice_kwarg"] is None:
        return choices
    root = default_dataset_root(dataset_label)

    if spec["name"] == "zju-leaper":
        patterns = Path(root) / "ImageSets" / "Patterns" if root else None
        if patterns is None or not patterns.is_dir():
            return choices
        pattern_ids = sorted(
            (path.stem.removeprefix("pattern") for path in patterns.glob("pattern*.json")),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        return choices + [f"Pattern {pattern_id}" for pattern_id in pattern_ids]

    if spec["name"] == "mvtec-ad":
        return choices + _mvtec_ad_categories(root)

    return choices


def slice_value(dataset_label: str, texture_label: str) -> str | None:
    """Resolve the "Texture / pattern" dropdown's selection into the value
    passed to the selected dataset's `slice_kwarg` (e.g. "pattern7" for
    ZJU-Leaper, "bottle" for MVTec AD)."""

    if texture_label == ALL_TEXTURES:
        return None
    spec = DATASET_CATALOG[dataset_label]
    if spec["name"] == "zju-leaper":
        if texture_label.lower().startswith("pattern "):
            return f"pattern{texture_label.split()[-1]}"
        raise ValueError(f"Unknown texture selection {texture_label!r}.")
    if spec["name"] == "mvtec-ad":
        return texture_label
    raise ValueError(f"{dataset_label!r} does not support texture/category selection.")


def shot_regime_kwargs(shot_mode: str) -> tuple[int | None, float]:
    """Map the UI's full-shot/few-shot toggle to `(num_samples, defect_ratio)`
    kwargs shared by every `DatasetAdapter` in this project."""

    if shot_mode == SHOT_FULL:
        return None, 0.5
    if shot_mode == SHOT_FEW:
        return FEW_SHOT_SAMPLE_COUNT, FEW_SHOT_DEFECT_RATIO
    raise ValueError(f"Unknown sample regime {shot_mode!r}.")


def empty_gallery_state() -> dict[str, Any]:
    return {"samples": [], "index": 0, "dataset": None}


def model_status(model_label: str) -> str:
    spec = MODEL_CATALOG[model_label]
    package = spec["backend"]
    installed = importlib.util.find_spec(package) is not None
    if not installed:
        return f"🔴 **Unavailable** — install the `{package}` extra before loading this backend."
    path = Path(spec["checkpoint"])
    return (
        f"🟢 **Ready** — {spec['task']} model from `{spec['metadata']['source']}` (`{path.name}`)"
        if path.is_file()
        else f"🟠 **Checkpoint missing** — expected `{path}`."
    )


def checkpoint_diagnostic(model_label: str) -> str:
    """Return on-demand, non-executing provenance data for a selected model."""

    spec = MODEL_CATALOG[model_label]
    if spec["backend"] != "anomalib":
        return "ℹ️ **Native Ultralytics artifact** — readiness is checked from its local `.pt` file."
    diagnostic = inspect_checkpoint(spec["checkpoint"])
    if not diagnostic.exists:
        return f"🟠 **Checkpoint missing** — `{diagnostic.path}` was not found."
    globals_summary = ", ".join(diagnostic.unsafe_globals) or "none"
    return (
        "🟢 **Trusted checkpoint diagnostic**  \n"
        f"SHA-256: `{diagnostic.sha256}`  \n"
        f"Size: `{diagnostic.size_bytes / (1024 * 1024):.1f} MiB`  \n"
        f"Declared checkpoint globals: `{globals_summary}`"
    )


def dataset_status(dataset_label: str) -> str:
    root = default_dataset_root(dataset_label)
    if root:
        return f"🟢 **Ready** — using the registered `{dataset_label}` dataset."
    spec = DATASET_CATALOG[dataset_label]
    return (
        f"🟠 **Dataset unavailable** — connect the storage containing `{dataset_label}` "
        f"(expected at `data/{spec['dir']}`, typically a symlink onto external storage, "
        f"or set `${spec['env']}`), then restart the app."
    )


def build_gallery_state(samples: list[Sample], count: int, seed: int, dataset_label: str) -> dict[str, Any]:
    available = [sample for sample in samples if Path(sample.image_path).is_file()]
    if not available:
        raise ValueError("No readable image files were found in the selected dataset/split.")
    selected_count = min(max(int(count), 1), len(available))
    selected = random.Random(int(seed)).sample(available, selected_count)
    return {"samples": [sample_to_dict(sample) for sample in selected], "index": 0, "dataset": dataset_label}


def load_random_samples(
    dataset_label: str,
    split: str,
    sample_count: int,
    seed: int | None = None,
    texture_label: str = ALL_TEXTURES,
    image_scope: str = ALL_IMAGES,
    shot_mode: str = SHOT_FULL,
) -> tuple[dict[str, Any], str | None, str, str]:
    root = default_dataset_root(dataset_label)
    if not root:
        raise FileNotFoundError(
            f"The `{dataset_label}` root could not be resolved from the local `data/` directory or SSD."
        )
    spec = DATASET_CATALOG[dataset_label]
    actual_seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    num_samples, defect_ratio = shot_regime_kwargs(shot_mode)
    dataset_kwargs: dict[str, Any] = dict(
        root=root,
        split=split,
        task=spec["task"],
        use_defect=image_scope != NORMAL_ONLY,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        dataset_kwargs[spec["slice_kwarg"]] = slice_value(dataset_label, texture_label)
    dataset = load_dataset(spec["name"], **dataset_kwargs)
    samples = dataset.load_samples()
    if image_scope == DEFECT_ONLY:
        samples = [sample for sample in samples if sample.annotations.is_anomalous]
    elif image_scope not in (ALL_IMAGES, NORMAL_ONLY):
        raise ValueError(f"Unknown image selection {image_scope!r}.")
    state = build_gallery_state(samples, sample_count, actual_seed, dataset_label)
    path, position = current_image(state)
    texture = "all textures" if texture_label == ALL_TEXTURES else texture_label
    return state, path, position, (
        f"🟢 Loaded **{len(state['samples'])}** random `{image_scope.lower()}` "
        f"({shot_mode.lower()}) from `{dataset.name}` / `{texture}` / `{split}`."
    )


def current_image(state: dict[str, Any]) -> tuple[str | None, str]:
    samples = state.get("samples", [])
    if not samples:
        return None, "No image loaded yet."
    index = int(state.get("index", 0)) % len(samples)
    state["index"] = index
    sample = sample_from_dict(samples[index])
    return sample.image_path, _sample_caption(sample, index, len(samples))


def move_image(state: dict[str, Any], direction: int) -> tuple[dict[str, Any], str | None, str]:
    if not state.get("samples"):
        return state, None, "Load a dataset before browsing images."
    state = dict(state)
    state["index"] = (int(state.get("index", 0)) + direction) % len(state["samples"])
    path, caption = current_image(state)
    return state, path, caption


def detect_current(state: dict[str, Any], model_label: str) -> tuple[Any, dict[str, Any], str]:
    if not state.get("samples"):
        return None, {}, "🟠 Load a dataset and select an image first."
    spec = MODEL_CATALOG[model_label]
    status = model_status(model_label)
    if status.startswith("🔴") or status.startswith("🟠"):
        return None, {}, status

    sample = sample_from_dict(state["samples"][state["index"]])
    try:
        model = load_model(spec["backend"], spec["name"])
        prediction = _predict_with_model(model, spec, model_label, sample)[0]
        image = render_prediction(sample.image_path, prediction)
    except Exception as exc:
        return None, {}, f"🔴 **Inference failed** — {type(exc).__name__}: {exc}"
    return image, prediction_summary(prediction), "🟢 **Inference complete**"


def load_selected_model(session_manager: Any, model_label: str) -> dict[str, Any]:
    """Load a catalog entry through the UI-independent inference service.

    Checks the checkpoint file exists *before* handing off to the adapter.
    This isn't just a nicer error message: Ultralytics' `YOLO(path)` loader
    treats a missing path whose filename matches a known official release
    asset (e.g. "yolo11n.pt") as a request to auto-download that asset —
    confirmed live, it silently downloaded generic COCO-pretrained weights
    into this catalog's published slot for an unfinished model, which the
    UI would then present as "Ready — Fabric trained". Failing fast here
    keeps every catalog entry either genuinely fabric-trained or clearly
    marked unavailable, never a substitute pretending to be the real thing.
    """

    spec = MODEL_CATALOG[model_label]
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"no trained checkpoint at {checkpoint} — train {model_label!r} "
            f"first (see tools/train_all_models.py), nothing was loaded"
        )
    return session_manager.load(model_label, spec, artifact_for_model(spec))


def unload_selected_model(session_manager: Any) -> dict[str, Any]:
    """Unload the active model through the UI-independent inference service."""

    return session_manager.unload()


def detect_loaded_model(
    session_manager: Any, state: dict[str, Any], model_label: str
) -> tuple[Any, dict[str, Any], str]:
    """Predict through a preloaded backend session without creating an adapter."""

    if not state.get("samples"):
        return None, {}, "🟠 Load a dataset and select an image first."
    spec = MODEL_CATALOG[model_label]
    sample = sample_from_dict(state["samples"][state["index"]])
    try:
        if spec["backend"] == "anomalib":
            maps_dir = RUNTIME_ANOMALY_MAP_ROOT / _model_slug(model_label)
            prediction = session_manager.predict(model_label, [sample], output_dir=str(maps_dir))[0]
        else:
            prediction = session_manager.predict(model_label, [sample])[0]
        image = render_prediction(sample.image_path, prediction)
    except Exception as exc:
        return None, {}, f"🔴 **Inference failed** — {type(exc).__name__}: {exc}"
    return image, prediction_summary(prediction), "🟢 **Inference complete**"


def _predict_with_model(model: Any, spec: dict[str, Any], model_label: str, sample: Sample) -> list[Prediction]:
    if spec["backend"] == "anomalib":
        maps_dir = RUNTIME_ANOMALY_MAP_ROOT / _model_slug(model_label)
        return model.predict([sample], artifact_for_model(spec), output_dir=str(maps_dir))
    return model.predict([sample], artifact_for_model(spec))


def artifact_for_model(spec: dict[str, Any]) -> Artifact:
    return Artifact(path=str(spec["checkpoint"]), backend=spec["backend"], metadata=dict(spec["metadata"]))


def _model_slug(model_label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in model_label).strip("-")


def prediction_summary(prediction: Prediction) -> dict[str, Any]:
    return {
        "sample_id": prediction.sample_id,
        "task": "anomaly" if prediction.anomaly_score is not None else "detection",
        "detections": len(prediction.boxes or []),
        "labels": prediction.labels or [],
        "scores": [round(score, 4) for score in prediction.scores or []],
        "anomaly_score": prediction.anomaly_score,
        "has_masks": bool(prediction.masks),
        "has_anomaly_map": prediction.anomaly_map is not None,
    }


def format_prediction_summary(summary: dict[str, Any]) -> str:
    """Format the unified prediction contract for a human-facing UI panel."""

    if not summary:
        return "### Prediction summary\nNo prediction available yet."
    if summary["task"] == "anomaly":
        label = next(iter(summary["labels"]), "unclassified")
        score = summary["anomaly_score"]
        score_text = "unavailable" if score is None else f"{score:.4f}"
        map_text = "available" if summary["has_anomaly_map"] else "not available"
        return (
            "### Prediction summary\n"
            f"**Result:** {label.title()}  \n"
            f"**Anomaly score:** {score_text}  \n"
            f"**Heatmap:** {map_text}"
        )
    detections = int(summary["detections"])
    if detections == 0:
        return "### Prediction summary\n**Result:** No defect detected."
    details = [
        f"{label} ({score:.2%})"
        for label, score in zip(summary["labels"], summary["scores"])
    ]
    return (
        "### Prediction summary\n"
        "**Result:** Defect detected  \n"
        f"**Detected regions:** {detections}  \n"
        f"**Confidence:** {', '.join(details) or 'unavailable'}"
    )


def render_prediction(image_path: str, prediction: Prediction):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    image = _overlay_masks(image, prediction.masks)
    image = _overlay_anomaly_map(image, prediction.anomaly_map)
    draw = ImageDraw.Draw(image)
    boxes = prediction.boxes or []
    labels = prediction.labels or ["defect"] * len(boxes)
    scores = prediction.scores or [None] * len(boxes)
    for box, label, score in zip(boxes, labels, scores):
        text = label if score is None else f"{label} · {score:.3f}"
        draw.rectangle(box, outline="#f97316", width=4)
        draw.text((box[0], max(0, box[1] - 18)), text, fill="#f97316", stroke_width=1, stroke_fill="white")
    return image.convert("RGB")


def _overlay_masks(image, masks):
    if not masks:
        return image
    import numpy as np
    from PIL import Image

    values = np.asarray(masks, dtype=bool)
    if values.ndim == 3:
        values = values.any(axis=0)
    alpha = Image.fromarray((values * 96).astype("uint8")).resize(image.size)
    overlay = Image.new("RGBA", image.size, "#f97316")
    overlay.putalpha(alpha)
    return Image.alpha_composite(image, overlay)


def _overlay_anomaly_map(image, map_path: str | None):
    if map_path is None:
        return image
    import numpy as np
    from PIL import Image

    values = np.asarray(np.load(map_path), dtype=float)
    values -= values.min()
    if values.max() > 0:
        values /= values.max()
    red = Image.fromarray((values * 255).astype("uint8")).resize(image.size)
    alpha = Image.fromarray((values * 128).astype("uint8")).resize(image.size)
    overlay = Image.merge("RGBA", (red, Image.new("L", image.size), Image.new("L", image.size), alpha))
    return Image.alpha_composite(image, overlay)


def _sample_caption(sample: Sample, index: int, total: int) -> str:
    state = "defect" if sample.annotations.is_anomalous else "normal"
    return f"**{index + 1} / {total}** · `{sample.id}` · {state}"
