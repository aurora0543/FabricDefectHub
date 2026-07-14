"""Backend glue for the Benchmark tab: pick a dataset shot regime
(full-shot = every test sample, few-shot = ~350 samples, matching
`single_image.SHOT_FULL`/`SHOT_FEW`) and one or more trained models, then
run each one through `loader.run_experiment` with the task-appropriate
`Evaluator` (`AnomalyEvaluator` for image AUROC/F1, `DetectionEvaluator` for
mAP/precision/recall) to build a leaderboard.

No heatmaps or bounding boxes are rendered here — the anomalib adapter is
called without `output_dir`, so only image-level metrics are computed and
nothing is written to disk; this tab only ever needs numbers, not images.
"""

from __future__ import annotations

import time
from typing import Any

from fabric_defect_hub.core.types import ModelInfo, RuntimeInfo
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.web.single_image import (
    DATASET_CATALOG,
    MODEL_CATALOG,
    artifact_for_model,
    default_dataset_root,
    shot_regime_kwargs,
    slice_value,
)


def compatible_models(dataset_label: str) -> list[str]:
    """Models whose task matches the selected dataset's task (an anomaly
    dataset only ever makes sense scored against anomaly models, etc.)."""

    task = DATASET_CATALOG[dataset_label]["task"]
    return [label for label, spec in MODEL_CATALOG.items() if spec["task"] == task]


def _evaluator_for_task(task: str):
    from fabric_defect_hub.evaluation import AnomalyEvaluator, DetectionEvaluator

    if task == "anomaly":
        return AnomalyEvaluator()
    if task == "detection":
        return DetectionEvaluator()
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


def run_benchmark(
    dataset_label: str,
    texture_label: str,
    shot_mode: str,
    model_labels: list[str],
) -> tuple[list[str], list[list[Any]], str]:
    """Evaluate every model in `model_labels` against the same dataset
    sample (test split only — the benchmark tab never trains). Returns
    `(columns, rows, status)`: `columns` is the superset of metric names
    produced by any model so every row can be padded to the same shape.
    """

    if not model_labels:
        return [], [], "🟠 Select at least one model."

    root = default_dataset_root(dataset_label)
    if not root:
        return [], [], f"🔴 **Dataset unavailable** — connect `{dataset_label}` first."

    spec = DATASET_CATALOG[dataset_label]
    num_samples, defect_ratio = shot_regime_kwargs(shot_mode)
    dataset_kwargs: dict[str, Any] = dict(
        root=root,
        split="test",
        task=spec["task"],
        use_defect=True,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        dataset_kwargs[spec["slice_kwarg"]] = slice_value(dataset_label, texture_label)

    evaluator = _evaluator_for_task(spec["task"])
    device = _detect_device()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    sample_count: int | None = None
    for model_label in model_labels:
        model_spec = MODEL_CATALOG[model_label]
        if model_spec["task"] != spec["task"]:
            errors.append(f"{model_label}: task mismatch with dataset ({spec['task']}).")
            continue
        try:
            dataset = load_dataset(spec["name"], **dataset_kwargs)
            model = load_model(model_spec["backend"], model_spec["name"])
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
            row = {
                "model": model_label,
                "runtime_s": round(time.perf_counter() - started, 1),
                **result.metrics,
            }
            rows.append(row)
        except Exception as exc:
            errors.append(f"{model_label}: {type(exc).__name__}: {exc}")

    if not rows:
        return [], [], "🔴 " + ("; ".join(errors) if errors else "No results produced.")

    metric_columns = sorted({key for row in rows for key in row if key not in ("model", "runtime_s")})
    columns = ["model", "runtime_s", *metric_columns]
    table = [[row.get(column, "") for column in columns] for row in rows]
    status = f"🟢 Evaluated {len(rows)} model(s) on {sample_count if sample_count is not None else '?'} samples ({shot_mode.lower()})."
    if errors:
        status += " ⚠️ " + "; ".join(errors)
    return columns, table, status


def _slug(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in label).strip("-")
