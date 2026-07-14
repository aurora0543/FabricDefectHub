"""State and inference helpers for the single-image Gradio workspace."""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.serialization import sample_from_dict, sample_to_dict
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.loader import load_dataset, load_model
from fabric_defect_hub.models.anomalib.checkpoint import inspect_checkpoint
from fabric_defect_hub.models.base import Artifact


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "yolo"
LOCAL_ANOMALIB_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "anomalib"
RUNTIME_ANOMALY_MAP_ROOT = PROJECT_ROOT / "artifacts" / "runtime" / "anomaly_maps"
ZJU_LEAPER_VOLUME_SUFFIX = Path("anomaly-detection-challenges") / "datasets" / "ZJU-Leaper"

# The UI deliberately exposes trained, local artifacts only.  The path remains
# backend-owned so operators never need to paste an absolute checkpoint path.
MODEL_CATALOG = {
    "YOLOv8n · Fabric trained": {
        "backend": "ultralytics",
        "name": "yolov8n",
        "checkpoint": LOCAL_MODEL_ROOT / "YOLOv8n.pt",
        "task": "detection",
        "metadata": {"trusted": True, "source": "local trained artifact"},
    },
    "YOLOv8s · Fabric trained": {
        "backend": "ultralytics",
        "name": "yolov8s",
        "checkpoint": LOCAL_MODEL_ROOT / "YOLOv8s.pt",
        "task": "detection",
        "metadata": {"trusted": True, "source": "local trained artifact"},
    },
    "YOLO11n · Fabric trained": {
        "backend": "ultralytics",
        "name": "yolo11n",
        "checkpoint": LOCAL_MODEL_ROOT / "YOLOv11.pt",
        "task": "detection",
        "metadata": {"trusted": True, "source": "local trained artifact"},
    },
    "PatchCore · Normal Lab trained": {
        "backend": "anomalib",
        "name": "PatchCore",
        "checkpoint": LOCAL_ANOMALIB_MODEL_ROOT / "Patchcore-latest.ckpt",
        "task": "anomaly",
        "metadata": {"trusted": True, "source": "Normal Lab", "model_class": "Patchcore"},
    },
    "RD4AD · Normal Lab trained": {
        "backend": "anomalib",
        "name": "RD4AD",
        "checkpoint": LOCAL_ANOMALIB_MODEL_ROOT / "ReverseDistillation-latest.ckpt",
        "task": "anomaly",
        "metadata": {"trusted": True, "source": "Normal Lab", "model_class": "ReverseDistillation"},
    },
    "EfficientAD · Normal Lab trained": {
        "backend": "anomalib",
        "name": "EfficientAD",
        "checkpoint": LOCAL_ANOMALIB_MODEL_ROOT / "EfficientAd-latest.ckpt",
        "task": "anomaly",
        "metadata": {"trusted": True, "source": "Normal Lab", "model_class": "EfficientAd"},
    },
    "SuperSimpleNet · Normal Lab trained": {
        "backend": "anomalib",
        "name": "SuperSimpleNet",
        "checkpoint": LOCAL_ANOMALIB_MODEL_ROOT / "Supersimplenet-latest.ckpt",
        "task": "anomaly",
        "metadata": {"trusted": True, "source": "Normal Lab", "model_class": "Supersimplenet"},
    },
}

# Each entry names a registered DatasetAdapter.  Additional datasets can add a
# catalog entry here without changing the page interaction contract.
DATASET_CATALOG = {"ZJU-Leaper": {"name": "zju-leaper"}}
ALL_TEXTURES = "All textures"
ALL_IMAGES = "All images"
DEFECT_ONLY = "Defect only"
NORMAL_ONLY = "Normal only"


def _zju_leaper_roots() -> list[Path]:
    """Return likely local roots, including the repository's SSD convention."""

    roots: list[Path] = []
    configured = os.getenv("ZJU_LEAPER_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(PROJECT_ROOT / name / "ZJU-Leaper" for name in ("data", "Data"))
    volumes = Path("/Volumes")
    if volumes.is_dir():
        roots.extend(volume / ZJU_LEAPER_VOLUME_SUFFIX for volume in volumes.iterdir())
    return roots


def default_dataset_root(dataset_label: str = "ZJU-Leaper") -> str:
    """Resolve the selected dataset without exposing an editable path field."""

    if dataset_label != "ZJU-Leaper":
        raise KeyError(f"Unknown dataset selection {dataset_label!r}.")
    for candidate in _zju_leaper_roots():
        if candidate.is_dir():
            return str(candidate.resolve())
    return ""


def texture_choices(dataset_label: str) -> list[str]:
    """Discover available texture slices from the registered dataset root."""

    root = default_dataset_root(dataset_label)
    patterns = Path(root) / "ImageSets" / "Patterns" if root else None
    choices = [ALL_TEXTURES]
    if patterns is None or not patterns.is_dir():
        return choices
    pattern_ids = sorted(
        (path.stem.removeprefix("pattern") for path in patterns.glob("pattern*.json")),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    return choices + [f"Pattern {pattern_id}" for pattern_id in pattern_ids]


def _pattern_value(texture_label: str) -> str | None:
    if texture_label == ALL_TEXTURES:
        return None
    if texture_label.lower().startswith("pattern "):
        return f"pattern{texture_label.split()[-1]}"
    raise ValueError(f"Unknown texture selection {texture_label!r}.")


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
    return "🟠 **Dataset unavailable** — connect the SSD containing ZJU-Leaper, then restart the app."


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
) -> tuple[dict[str, Any], str | None, str, str]:
    root = default_dataset_root(dataset_label)
    if not root:
        raise FileNotFoundError("The ZJU-Leaper root could not be resolved from the local Data directory or SSD.")
    dataset_name = DATASET_CATALOG[dataset_label]["name"]
    actual_seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    dataset = load_dataset(
        dataset_name,
        root=root,
        split=split,
        task="detection",
        use_defect=image_scope != NORMAL_ONLY,
        pattern=_pattern_value(texture_label),
    )
    samples = dataset.load_samples()
    if image_scope == DEFECT_ONLY:
        samples = [sample for sample in samples if sample.annotations.is_anomalous]
    elif image_scope not in (ALL_IMAGES, NORMAL_ONLY):
        raise ValueError(f"Unknown image selection {image_scope!r}.")
    state = build_gallery_state(samples, sample_count, actual_seed, dataset_label)
    path, position = current_image(state)
    texture = "all textures" if texture_label == ALL_TEXTURES else texture_label
    return state, path, position, (
        f"🟢 Loaded **{len(state['samples'])}** random `{image_scope.lower()}` from "
        f"`{dataset.name}` / `{texture}` / `{split}`."
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
    """Load a catalog entry through the UI-independent inference service."""

    spec = MODEL_CATALOG[model_label]
    return session_manager.load(model_label, spec, _artifact_for_inference(spec))


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
        return model.predict([sample], _artifact_for_inference(spec), output_dir=str(maps_dir))
    return model.predict([sample], _artifact_for_inference(spec))


def _artifact_for_inference(spec: dict[str, Any]) -> Artifact:
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
