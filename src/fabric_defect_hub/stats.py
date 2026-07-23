"""Generic statistical utilities shared by the profiling/evaluation metrics
that need them: scale-free fluctuation (coefficient of variation),
least-squares trend fitting, and a percentile-bootstrap confidence
interval for a between-group comparison. Kept engine/task-agnostic and
dependency-free (no numpy/scipy) so profiling and evaluation code can both
import it without pulling in a heavier stats library for a handful of
formulas.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Sequence


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Population stddev / mean -- the scale-free version of fluctuation,
    so e.g. FPS variability is comparable across runs with very different
    mean throughput. Returns 0.0 for fewer than two points or a zero mean
    (nothing meaningful to divide by).
    """

    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def least_squares_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Ordinary least-squares fit y = beta*x + alpha. Returns (beta, alpha)."""

    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if len(x) < 2:
        raise ValueError("least_squares_slope needs at least two points.")

    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)
    if denominator == 0:
        raise ValueError("least_squares_slope needs at least two distinct x values.")

    beta = numerator / denominator
    alpha = mean_y - beta * mean_x
    return beta, alpha


def bootstrap_group_ci(
    group_a: Sequence,
    group_b: Sequence,
    statistic: Callable[[Sequence], float],
    combine: Callable[[float, float], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Percentile-bootstrap CI for `combine(statistic(group_a), statistic(group_b))`.

    Each group is independently resampled with replacement (same size as
    the original) `n_resamples` times; `statistic` reduces one resampled
    group to a scalar (e.g. a least-squares slope, a mean); `combine`
    reduces the two groups' scalars to the one number the CI is for (e.g.
    a difference, or a relative-degradation ratio). Resamples where
    `combine` raises or returns a non-finite value (e.g. division by a
    resampled-to-zero denominator) are dropped rather than poisoning the
    interval; this needs at least two survivors per side of the requested
    tail to produce a bound.

    One function serves every "between-group delta + CI" metric in this
    project (throughput-resolution slope difference, cross-domain accuracy
    degradation) instead of each reimplementing the same resampling loop.
    """

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("bootstrap_group_ci needs at least two points per group.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    rng = random.Random(seed)
    point_estimate = combine(statistic(group_a), statistic(group_b))

    values: list[float] = []
    for _ in range(n_resamples):
        resampled_a = [group_a[rng.randrange(len(group_a))] for _ in range(len(group_a))]
        resampled_b = [group_b[rng.randrange(len(group_b))] for _ in range(len(group_b))]
        try:
            value = combine(statistic(resampled_a), statistic(resampled_b))
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
        if math.isfinite(value):
            values.append(value)

    tail = (1.0 - confidence) / 2.0
    min_survivors = max(2, int(round(n_resamples * tail)) + 1)
    if len(values) < min_survivors:
        raise ValueError(
            f"Only {len(values)}/{n_resamples} bootstrap resamples produced a finite value; "
            "not enough to bound the requested confidence level."
        )

    values.sort()
    lo_idx = int(tail * len(values))
    hi_idx = min(len(values) - 1, int((1.0 - tail) * len(values)))
    return {
        "estimate": point_estimate,
        "ci_low": values[lo_idx],
        "ci_high": values[hi_idx],
    }
