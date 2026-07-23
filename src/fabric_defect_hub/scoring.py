"""Combine a leaderboard row's accuracy ("technical") and runtime-cost
("overhead") metrics into one ranked score.

`benchmark.py`'s `leaderboard()` already sorts by a single raw metric name
-- useful when you know exactly which number matters, but it can't answer
"which model is the best overall trade-off" when technical metrics
(AUROC, mAP, mIoU, ...) and overhead metrics (FPS, latency, memory, model
size) pull in different directions. This module is that combination step,
used by the Gradio Benchmark tab (`web/benchmark.py`) so its leaderboard can
rank on a configurable blend of the two instead of forcing a pick of one
column.

Metric names are classified by pattern rather than a fixed enumeration --
new evaluators/profilers add metric names over time (see `evaluation/*.py`,
`profiling/base.py::summarize_latencies`), and a name this module doesn't
recognize should just be excluded from scoring, not require this file to be
updated in lockstep or raise on an unfamiliar key.
"""

from __future__ import annotations

from typing import Any, Literal

MetricDirection = Literal["higher", "lower"]

# Ordered (substring, direction) rules -- the first match wins, so more
# specific patterns (fps_std/fps_cv, jitter measures where *lower* is
# better) must precede the general "fps" rule they'd otherwise also match.
_DIRECTION_RULES: tuple[tuple[str, MetricDirection], ...] = (
    ("fps_std", "lower"),
    ("fps_cv", "lower"),
    ("fps", "higher"),
    ("latency", "lower"),
    ("memory_mb", "lower"),
    ("model_size_mb", "lower"),
    ("runtime_s", "lower"),
    ("alarms_per_unit_length", "lower"),
    ("auroc", "higher"),
    ("aupro", "higher"),
    ("iap", "higher"),
    ("map", "higher"),
    ("mar", "higher"),
    ("miou", "higher"),
    ("dice", "higher"),
    ("f1", "higher"),
    ("precision", "higher"),
    ("recall", "higher"),
)

# Metrics matched by these substrings count toward the "overhead" (runtime
# cost) group; every other classified metric is "technical" (accuracy).
_OVERHEAD_KEYWORDS = ("fps", "latency", "memory_mb", "model_size_mb", "runtime_s")

SCORE_PRESETS: dict[str, tuple[float, float]] = {
    "accuracy_first": (0.85, 0.15),
    "balanced": (0.5, 0.5),
    "efficiency_first": (0.15, 0.85),
}


def metric_direction(name: str) -> MetricDirection | None:
    """Whether a higher or lower value of metric `name` is better, or
    `None` if this module doesn't recognize it (excluded from scoring)."""

    lowered = name.lower()
    for pattern, direction in _DIRECTION_RULES:
        if pattern in lowered:
            return direction
    return None


def metric_group(name: str) -> Literal["technical", "overhead"] | None:
    """'technical' (accuracy) or 'overhead' (runtime cost); `None` if
    `metric_direction` doesn't recognize `name` either."""

    if metric_direction(name) is None:
        return None
    lowered = name.lower()
    return "overhead" if any(keyword in lowered for keyword in _OVERHEAD_KEYWORDS) else "technical"


def score_rows(
    rows: list[dict[str, Any]],
    technical_weight: float = 0.5,
    overhead_weight: float = 0.5,
    metric_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return copies of `rows` with `technical_score`/`overhead_score`/
    `composite_score` (0-100) appended.

    Each recognized metric is min-max normalized against the *other rows in
    this same call* -- these scores rank the given leaderboard against
    itself, they are not an absolute, cross-run scale. A row missing every
    metric in a group gets `None` for that group's score, and the composite
    falls back to whichever group is actually present (renormalizing the
    weights over just those groups); a row missing both groups gets a
    `None` composite.

    `metric_keys`, if given, restricts scoring to that subset of metric
    names (e.g. a "basic parameters" UI toggle that only feeds a few
    metrics into the score) -- metrics outside it are still present in the
    row's own values, just not used for scoring.
    """

    if technical_weight < 0 or overhead_weight < 0:
        raise ValueError("weights must be non-negative")
    if technical_weight + overhead_weight <= 0:
        raise ValueError("at least one of technical_weight/overhead_weight must be positive")

    names_by_group: dict[str, list[str]] = {"technical": [], "overhead": []}
    for name in {key for row in rows for key in row}:
        if metric_keys is not None and name not in metric_keys:
            continue
        group = metric_group(name)
        if group is not None:
            names_by_group[group].append(name)

    # normalized[group][name] maps row index -> normalized value in [0, 1],
    # only for rows that actually have that metric.
    normalized: dict[str, dict[str, dict[int, float]]] = {
        group: {name: _normalize(rows, name) for name in names}
        for group, names in names_by_group.items()
    }

    weights = {"technical": technical_weight, "overhead": overhead_weight}
    scored_rows = []
    for index, row in enumerate(rows):
        group_scores: dict[str, float | None] = {}
        for group in ("technical", "overhead"):
            values = [
                per_metric[index]
                for per_metric in normalized[group].values()
                if index in per_metric
            ]
            group_scores[group] = (sum(values) / len(values) * 100.0) if values else None

        present_weight = sum(
            weights[group] for group, score in group_scores.items() if score is not None
        )
        if present_weight > 0:
            composite = sum(
                weights[group] * score
                for group, score in group_scores.items()
                if score is not None
            ) / present_weight
        else:
            composite = None

        new_row = dict(row)
        new_row["technical_score"] = group_scores["technical"]
        new_row["overhead_score"] = group_scores["overhead"]
        new_row["composite_score"] = composite
        scored_rows.append(new_row)

    return scored_rows


def _normalize(rows: list[dict[str, Any]], name: str) -> dict[int, float]:
    direction = metric_direction(name)
    present = [
        (index, row[name])
        for index, row in enumerate(rows)
        if name in row and isinstance(row[name], (int, float))
    ]
    if not present:
        return {}
    values = [value for _, value in present]
    low, high = min(values), max(values)
    result: dict[int, float] = {}
    for index, value in present:
        if high == low:
            # Every row ties on this metric -- treat it as equally good
            # rather than dividing by zero or arbitrarily favoring one row.
            result[index] = 1.0
            continue
        normalized = (value - low) / (high - low)
        result[index] = 1.0 - normalized if direction == "lower" else normalized
    return result
