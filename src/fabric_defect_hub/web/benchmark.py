"""Backend glue for the Benchmark tab: pick a dataset shot regime
(full-shot = every test sample, few-shot = ~350 samples, matching
`single_image.SHOT_FULL`/`SHOT_FEW`) and one or more trained models, then
run each one through `loader.run_experiment` with the task-appropriate
`Evaluator` (`AnomalyEvaluator` for image AUROC/F1, `DetectionEvaluator` for
mAP/precision/recall, `SegmentationEvaluator` for mIoU/Dice/pixel-F1) to
build a leaderboard.

Every selected model is cycled through the same mount -> test -> unmount ->
next-model pipeline (`run_benchmark`'s loop): one model is loaded, evaluated,
then explicitly released (`_release_model`) before the next one is
instantiated, so a full run of all 14 catalog models — several hundred MB to
~1GB of weights each — never holds more than one resident model in memory at
a time. `run_benchmark` is a generator that yields after every model so the
UI can render results as they land instead of blocking until the whole
leaderboard is done.

No heatmaps or bounding boxes are rendered here — the anomalib adapter is
called without `output_dir`, so only image-level metrics are computed and
nothing is written to disk; this tab only ever needs numbers, not images.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Iterator

from fabric_defect_hub.core.types import ModelInfo, RuntimeInfo
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.inference.session import clear_accelerator_cache
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.web.single_image import (
    DATASET_CATALOG,
    MODEL_CATALOG,
    shot_text,
    artifact_for_model,
    default_dataset_root,
    shot_regime_kwargs,
    slice_value,
)


def _dataset_task_for(model_task: str) -> str:
    """A `DatasetAdapter`'s `task` only ever needs to be one of
    detection/segmentation/anomaly (see `core.types.Task`) to decide which
    ground-truth fields to attach — `instance_segmentation` (Mask R-CNN's
    catalog task) is scored the same way as semantic segmentation (both via
    `SegmentationEvaluator`'s unioned binary mask), so it maps onto the
    dataset's `segmentation` bucket too."""

    return "segmentation" if model_task == "instance_segmentation" else model_task


def compatible_models(dataset_label: str) -> list[str]:
    """Models this dataset can supply real ground truth for — i.e. every
    catalog model whose task the dataset's `tasks` set covers (ZJU-Leaper
    has boxes *and* masks, so both detection and segmentation models are
    compatible; RAW-FABRID/MVTec AD have anomaly labels and masks but no
    boxes, so only anomaly and segmentation models are)."""

    tasks = DATASET_CATALOG[dataset_label]["tasks"]
    return [label for label, spec in MODEL_CATALOG.items() if _dataset_task_for(spec["task"]) in tasks]


def _evaluator_for_task(task: str):
    from fabric_defect_hub.evaluation import AnomalyEvaluator, DetectionEvaluator, SegmentationEvaluator

    if task == "anomaly":
        return AnomalyEvaluator()
    if task == "detection":
        return DetectionEvaluator()
    if task == "segmentation":
        return SegmentationEvaluator()
    raise ValueError(f"no evaluator registered for task {task!r}")


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _release_model(model: Any) -> None:
    """Mirrors `InferenceSessionManager._unload_active` (which the Single
    Image tab uses): call the adapter's own `unload()` if it has one (only
    the Ultralytics and Anomalib adapters do), drop our reference, then
    force a GC pass and clear the CUDA/MPS allocator cache so the next
    model's `load_model` isn't fighting the previous one's still-cached
    memory."""

    unload = getattr(model, "unload", None)
    if callable(unload):
        unload()
    del model
    gc.collect()
    clear_accelerator_cache()


def run_benchmark(
    dataset_label: str,
    texture_label: str,
    shot_mode: str,
    model_labels: list[str],
    lang: str = DEFAULT_LANGUAGE,
) -> Iterator[tuple[list[str], list[list[Any]], str]]:
    """Evaluate every model in `model_labels` against the same dataset
    sample (test split only — the benchmark tab never trains), one model at
    a time: mount -> test -> unmount -> next model (`_release_model`).
    Yields `(columns, rows, status)` after every model so the leaderboard
    fills in live instead of appearing all at once; `columns` is the
    superset of metric names produced by any model evaluated so far, so
    every row stays padded to the same shape.
    """

    if not model_labels:
        yield [], [], tr(lang, "bench_select_model")
        return

    root = default_dataset_root(dataset_label)
    if not root:
        yield [], [], tr(lang, "bench_dataset_unavailable", label=dataset_label)
        return

    spec = DATASET_CATALOG[dataset_label]
    dataset_tasks = spec["tasks"]
    num_samples, defect_ratio = shot_regime_kwargs(shot_mode)
    base_dataset_kwargs: dict[str, Any] = dict(
        root=root,
        split="test",
        use_defect=True,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        base_dataset_kwargs[spec["slice_kwarg"]] = slice_value(dataset_label, texture_label)

    device = _detect_device()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    sample_count: int | None = None
    total = len(model_labels)
    yield [], [], tr(lang, "bench_starting", total=total)

    for index, model_label in enumerate(model_labels, start=1):
        model_spec = MODEL_CATALOG[model_label]
        dataset_task = _dataset_task_for(model_spec["task"])
        if dataset_task not in dataset_tasks:
            errors.append(tr(lang, "bench_task_mismatch", model=model_label, dataset=dataset_label, task=model_spec["task"]))
            yield _render(rows, sample_count, shot_mode, errors, lang=lang)
            continue

        model = None
        try:
            dataset = load_dataset(spec["name"], task=dataset_task, **base_dataset_kwargs)
            model = load_model(model_spec["backend"], model_spec["name"])
            evaluator = _evaluator_for_task(dataset_task)
            started = time.perf_counter()
            result = run_experiment(
                experiment_id=f"benchmark-{_slug(model_label)}",
                dataset=dataset,
                model=model,
                model_info=ModelInfo(
                    name=model_spec["name"], backend=model_spec["backend"], task=model_spec["task"]
                ),
                runtime=RuntimeInfo(device=device, engine="python", precision="fp32", input_size=(640, 640)),
                evaluator=evaluator,
                artifact=artifact_for_model(model_spec),
            )
            if sample_count is None:
                sample_count = len(dataset.load_samples())
            rows.append({
                "model": model_label,
                "runtime_s": round(time.perf_counter() - started, 1),
                **result.metrics,
            })
        except Exception as exc:
            errors.append(f"{model_label}: {type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                _release_model(model)

        status = tr(lang, "bench_progress", index=index, total=total, model=model_label)
        yield _render(rows, sample_count, shot_mode, errors, status, lang)

    yield _render(rows, sample_count, shot_mode, errors, lang=lang)


def _render(
    rows: list[dict[str, Any]],
    sample_count: int | None,
    shot_mode: str,
    errors: list[str],
    status: str | None = None,
    lang: str = DEFAULT_LANGUAGE,
) -> tuple[list[str], list[list[Any]], str]:
    if not rows:
        base = tr(lang, "bench_no_results") if not errors else "🔴 " + "; ".join(errors)
        return [], [], base

    metric_columns = sorted({key for row in rows for key in row if key not in ("model", "runtime_s")})
    columns = ["model", "runtime_s", *metric_columns]
    table = [[row.get(column, "") for column in columns] for row in rows]
    if status is None:
        status = tr(
            lang, "bench_done", count=len(rows),
            samples=sample_count if sample_count is not None else "?", shot=shot_text(lang, shot_mode),
        )
    if errors:
        status += " ⚠️ " + "; ".join(errors)
    return columns, table, status


def _slug(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in label).strip("-")
