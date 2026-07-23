"""Portable aggregate reports for benchmark results, plus a single
append-only run log (`append_run_log`) every backend/task writes the same
shape of row into — see that function's docstring for why this exists.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fabric_defect_hub.core.serialization import experiment_result_to_dict
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


def append_run_log(result: ExperimentResult, path: str | Path) -> Path:
    """Append one JSON line for `result` to a single, ever-growing log file
    shared across every model/backend/task -- so a detection run, a
    segmentation run, and an anomaly run with a completely different
    metric-name set all land as one line each in the *same* file with the
    *same* top-level shape, instead of each program inventing its own
    result format that then has to be reconciled by hand before it can be
    plotted or compared.

    JSON Lines (one `experiment_result_to_dict(result)` per line), not CSV:
    a shared header would force every row to agree on one fixed column set,
    which is exactly what doesn't hold here (an anomaly run's `image_auroc`
    and a detection run's `map50` are both just entries in `metrics`, of
    different sizes and names run to run). JSONL sidesteps that -- every
    line is independently well-formed, appending never requires rewriting
    prior lines, and any later analysis (`pandas.read_json(path, lines=True)`
    or similar) can flatten `metrics` into columns itself.

    Adds a `provenance` block (UTC timestamp, git commit, hostname) that
    isn't part of `experiment_result.schema.json` -- that schema is the
    frontend/leaderboard's read contract and deliberately strict
    (`additionalProperties: false`); provenance is metadata about the *run*,
    not the result, and belongs only in this log, not in `result.json`.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**experiment_result_to_dict(result), "provenance": _provenance()}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def read_run_log(path: str | Path) -> list[dict]:
    """Read back everything `append_run_log` has written to `path` (the
    "Run History" tab's data source). A missing file returns `[]` rather
    than raising -- a fresh deployment or a benchmark that hasn't run yet
    shouldn't be an error, just an empty history.
    """

    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_run_log_rows(rows: list[dict]) -> tuple[list[str], list[list[object]]]:
    """Flatten `read_run_log` output into a `(columns, table)` pair for a
    Gradio `Dataframe` -- pulls the same identifying fields as
    `reporting.py`'s own `_flatten` (model/backend/dataset/device) plus
    `provenance.timestamp_utc`, then every `metrics.*` key as its own
    column (union across all rows, like `web/benchmark.py::_render` does
    for a single run's rows, since different tasks produce different
    metric-name sets).
    """

    metric_columns = sorted({key for row in rows for key in row.get("metrics", {})})
    columns = ["timestamp_utc", "model", "backend", "task", "dataset", "device", *metric_columns]
    table = []
    for row in rows:
        model = row.get("model", {})
        dataset = row.get("dataset", {})
        runtime = row.get("runtime", {})
        provenance = row.get("provenance", {})
        metrics = row.get("metrics", {})
        record = {
            "timestamp_utc": provenance.get("timestamp_utc", ""),
            "model": model.get("name", ""),
            "backend": model.get("backend", ""),
            "task": model.get("task", ""),
            "dataset": dataset.get("name", ""),
            "device": runtime.get("device", ""),
            **metrics,
        }
        table.append([record.get(column, "") for column in columns])
    return columns, table


def latest_run_per_model(rows: list[dict]) -> list[dict]:
    """Collapse `read_run_log` output to one row per model name -- the most
    recent by `provenance.timestamp_utc` -- for a "current standings" chart
    where every historical run for the same model would otherwise clutter
    the picture."""

    latest: dict[str, dict] = {}
    for row in rows:
        name = row.get("model", {}).get("name", "")
        timestamp = row.get("provenance", {}).get("timestamp_utc", "")
        current_timestamp = latest.get(name, {}).get("provenance", {}).get("timestamp_utc", "")
        if name not in latest or timestamp > current_timestamp:
            latest[name] = row
    return list(latest.values())


def _provenance() -> dict[str, str]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "hostname": platform.node(),
    }


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
