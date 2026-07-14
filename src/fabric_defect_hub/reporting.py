"""Portable aggregate reports for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

from fabric_defect_hub.core.types import ExperimentResult


def save_leaderboard(results: list[ExperimentResult], path: str | Path) -> Path:
    """Write a flat leaderboard as CSV or Markdown based on the suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(results)
    rows = [_flatten(result, columns) for result in results]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    elif suffix in {".md", ".markdown"}:
        path.write_text(_markdown(columns, rows), encoding="utf-8")
    else:
        raise ValueError("leaderboard path must end in .csv, .md, or .markdown")
    return path


def _columns(results: list[ExperimentResult]) -> list[str]:
    metric_names = sorted({name for result in results for name in result.metrics})
    return [
        "experiment_id", "model", "backend", "task", "dataset", "split",
        "device", "engine", "precision", "input_size", *metric_names,
    ]


def _flatten(result: ExperimentResult, columns: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": result.experiment_id,
        "model": result.model.name,
        "backend": result.model.backend,
        "task": result.model.task,
        "dataset": result.dataset.name,
        "split": result.dataset.split,
        "device": result.runtime.device,
        "engine": result.runtime.engine,
        "precision": result.runtime.precision,
        "input_size": "x".join(str(value) for value in result.runtime.input_size),
    }
    row.update(result.metrics)
    return {column: row.get(column, "") for column in columns}


def _markdown(columns: list[str], rows: list[dict[str, object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(row[column]) for column in columns) + " |" for row in rows
    )
    return "\n".join(lines) + "\n"
