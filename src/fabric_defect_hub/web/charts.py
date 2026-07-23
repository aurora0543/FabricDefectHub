"""Chart data and rendering for the Benchmark tab's leaderboard.

Two views sit above the leaderboard table, both driven by the *same* scored
rows the table itself renders (`web/benchmark.py::run_benchmark` yields them
alongside the table), so a chart can never disagree with the numbers
directly below it:

- **Bar chart** — one metric at a time across every benchmarked model.
  One metric at a time because the metrics genuinely don't share a unit: an
  AUROC in [0, 1], a latency in milliseconds and a memory figure in MiB on
  one y-axis would be a dual-axis chart in disguise, which invents
  comparisons the data doesn't support. Built as a `pandas.DataFrame` for
  Gradio's native `gr.BarPlot` (pandas is already a hard Gradio dependency,
  so this costs no new install).
- **Radar chart** — several metrics at once for a few models, each metric
  normalized onto a shared 0-1 "higher is better" axis by
  `scoring.normalize_metrics` (the same transform behind `composite_score`,
  not a second one), with `scoring.polygon_area` summarizing "who encloses
  more area".

The radar is hand-written SVG rather than a plotting library: Gradio ships
no polar/radar component, and matplotlib/plotly would each be a new
dependency in a `requirements.txt` deliberately kept lean for the Hugging
Face Spaces deployment. Emitting an HTML/SVG string into a `gr.HTML` panel
is already this project's idiom for custom visuals (see
`web/single_image.py::render_prediction_tags`), and it lets every color live
in `web/app.py`'s stylesheet, so the chart follows the light/dark theme for
free instead of baking in one theme's hexes.
"""

from __future__ import annotations

import html
import math
from typing import Any

from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.scoring import normalize_metrics, polygon_area

# Columns that identify a row rather than measure it -- never chartable.
NON_METRIC_COLUMNS = ("model",)

# A radar needs at least a triangle; two axes enclose no area.
MIN_RADAR_AXES = 3
# Default number of axes drawn before the reader has picked a subset. Past
# roughly this many spokes the polygon turns into a circle and stops saying
# anything.
DEFAULT_RADAR_AXES = 6
# Hard cap on overlapping polygons. Radar polygons overlap arbitrarily, so
# every pair of series is potentially adjacent -- the "all-pairs" case for
# color separation, which only the first three categorical slots clear (see
# this project's palette notes in `web/app.py`'s CSS). More models than this
# is what the bar chart is for: it has a single series, so it can show every
# model at once without any color-separation limit at all.
MAX_RADAR_MODELS = 3

# Smallest radius a spoke can plot at, as a fraction of the full radius.
#
# `normalize_metrics` is min-max *across the rows being compared*, so the
# worst model on a given axis always lands at exactly 0 -- and a model that
# is worst on every axis (entirely possible with only a handful of models)
# would otherwise collapse into a single dot at the centre with zero area,
# which reads as "no data" rather than "last place" and hides the shape that
# is the whole point of the chart. Lifting the floor off zero keeps the
# worst performer visible as a small polygon. The floor is applied to the
# radii *before* the area is computed, so the reported area always describes
# the polygon actually drawn.
RADAR_RADIUS_FLOOR = 0.08

# Chart geometry. The box is wider than it is tall because the spoke labels
# sit *outside* the plot circle: everything, labels included, is kept inside
# the viewBox so the whole chart scales as one unit and never relies on
# overflowing its container (which the surrounding card would just clip).
RADAR_WIDTH = 480.0
RADAR_HEIGHT = 360.0
RADAR_CENTER = (RADAR_WIDTH / 2, 172.0)
RADAR_RADIUS = 100.0
RADAR_LABEL_OFFSET = 14.0


def metric_choices(columns: list[str]) -> list[str]:
    """Leaderboard columns a chart can plot, in the order they appear."""

    return [column for column in columns if column not in NON_METRIC_COLUMNS]


def default_bar_metric(columns: list[str]) -> str | None:
    """Prefer the composite score -- the one column that already blends
    accuracy and cost, so the default view answers "which model is best
    overall" before the reader picks a specific metric."""

    choices = metric_choices(columns)
    if not choices:
        return None
    return "composite_score" if "composite_score" in choices else choices[0]


def bar_y_limits(scored_rows: list[dict[str, Any]], metric: str | None) -> list[float] | None:
    """`[0, max*1.05]` for `gr.BarPlot`'s `y_lim`.

    Bars encode magnitude by *length*, so their axis has to start at zero --
    letting the plot auto-scale to the data's own range (which it does by
    default) crops the baseline and makes a 5% difference between two models
    look like a 5x one. The 5% headroom just keeps the tallest bar off the
    top edge.
    """

    if not scored_rows or not metric:
        return None
    values = [
        row[metric] for row in scored_rows if isinstance(row.get(metric), (int, float))
    ]
    if not values:
        return None
    highest = max(values)
    # An all-zero (or all-negative) metric has no positive range to show;
    # fall back to auto-scaling rather than an inverted or zero-height axis.
    return [0, highest * 1.05] if highest > 0 else None


def bar_frame(scored_rows: list[dict[str, Any]], metric: str | None):
    """A `model`/`metric` `pandas.DataFrame` for `gr.BarPlot`, or `None`
    when nothing plottable is left (no rows, no metric selected, or no row
    carries a numeric value for it -- e.g. an overhead metric when the run
    was done without profiling)."""

    if not scored_rows or not metric:
        return None
    import pandas as pd

    data = [
        {"model": row.get("model", ""), metric: row[metric]}
        for row in scored_rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not data:
        return None
    return pd.DataFrame(data)


def radar_axis_choices(scored_rows: list[dict[str, Any]]) -> list[str]:
    """Metrics that can serve as radar axes: those `scoring` recognizes (so
    it knows whether higher or lower is better and can normalize them), with
    the derived score columns excluded -- plotting `composite_score` as an
    axis beside the very metrics it averages would double-count them."""

    names, _ = normalize_metrics(scored_rows)
    derived = {"composite_score", "technical_score", "overhead_score"}
    return [name for name in names if name not in derived]


def default_radar_axes(scored_rows: list[dict[str, Any]]) -> list[str]:
    return radar_axis_choices(scored_rows)[:DEFAULT_RADAR_AXES]


def default_radar_models(scored_rows: list[dict[str, Any]]) -> list[str]:
    """The top `MAX_RADAR_MODELS` models by composite score. `scored_rows`
    arrives already sorted best-first from `web/benchmark.py::_render`, so
    this is just the head of the leaderboard."""

    return [row.get("model", "") for row in scored_rows[:MAX_RADAR_MODELS]]


def model_choices(scored_rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("model", "") for row in scored_rows]


def render_radar_svg(
    scored_rows: list[dict[str, Any]],
    axis_names: list[str],
    model_names: list[str],
    lang: str = DEFAULT_LANGUAGE,
) -> str:
    """Render the radar as an SVG + HTML legend string for a `gr.HTML` panel.

    Every axis is normalized across **all** `scored_rows`, not just the
    selected `model_names`, so changing which models are drawn never
    rescales the ones that stay -- a reader who learned "this model reaches
    the rim on FPS" should not see that change because a different model was
    ticked off.
    """

    if not scored_rows:
        return _radar_note(tr(lang, "radar_no_results"))
    axes = [name for name in axis_names if name in set(radar_axis_choices(scored_rows))]
    if len(axes) < MIN_RADAR_AXES:
        return _radar_note(tr(lang, "radar_needs_axes", count=MIN_RADAR_AXES))

    _, per_row = normalize_metrics(scored_rows)
    by_model = {
        row.get("model", ""): per_row[index] for index, row in enumerate(scored_rows)
    }
    selected = [name for name in model_names if name in by_model][:MAX_RADAR_MODELS]
    if not selected:
        return _radar_note(tr(lang, "radar_select_model"))

    center, radius = RADAR_CENTER, RADAR_RADIUS
    angles = [(-math.pi / 2) + (2 * math.pi * i / len(axes)) for i in range(len(axes))]

    parts = [
        f'<svg class="fdh-radar" viewBox="0 0 {RADAR_WIDTH:g} {RADAR_HEIGHT:g}" '
        f'role="img" aria-label="{html.escape(tr(lang, "radar_chart_label"))}">'
    ]

    # Every shape carries `fill="none"` as a presentation attribute so that
    # if the stylesheet ever fails to load, the chart degrades to bare
    # outlines instead of stacked solid-black polygons (an SVG `<polygon>`
    # with no fill declared defaults to opaque black, which would hide the
    # chart entirely). CSS declarations outrank presentation attributes, so
    # the themed fills in `web/app.py`'s stylesheet still win normally.
    for ring in (0.25, 0.5, 0.75, 1.0):
        points = _points(center, radius * ring, angles, [1.0] * len(axes))
        parts.append(f'<polygon class="fdh-radar-grid" fill="none" points="{points}" />')
    for angle in angles:
        x, y = _polar(center, radius, angle, 1.0)
        parts.append(
            f'<line class="fdh-radar-grid" fill="none" stroke="currentColor" '
            f'x1="{center[0]:.1f}" y1="{center[1]:.1f}" x2="{x:.1f}" y2="{y:.1f}" />'
        )

    for angle, name in zip(angles, axes):
        x, y = _polar(center, radius + RADAR_LABEL_OFFSET, angle, 1.0)
        parts.append(
            f'<text class="fdh-radar-axis-label" fill="currentColor" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{_anchor(angle)}" dominant-baseline="{_baseline(angle)}">'
            f"{html.escape(_short_axis_name(name))}</text>"
        )

    legend_items = []
    for slot, model_name in enumerate(selected, start=1):
        radii = [
            _floored(by_model[model_name].get(name, 0.0)) for name in axes
        ]
        points = _points(center, radius, angles, radii)
        parts.append(
            f'<polygon class="fdh-radar-series fdh-radar-series-{slot}" '
            f'fill="none" stroke="currentColor" points="{points}" />'
        )
        for angle, value in zip(angles, radii):
            x, y = _polar(center, radius, angle, value)
            parts.append(
                f'<circle class="fdh-radar-dot fdh-radar-dot-{slot}" '
                f'fill="currentColor" cx="{x:.1f}" cy="{y:.1f}" r="4" />'
            )
        area_pct = polygon_area(radii) / polygon_area([1.0] * len(axes)) * 100
        legend_items.append(
            f'<span class="fdh-radar-key">'
            f'<span class="fdh-radar-swatch fdh-radar-swatch-{slot}"></span>'
            f"{html.escape(model_name)} · "
            f'{html.escape(tr(lang, "radar_area"))} {area_pct:.0f}%'
            f"</span>"
        )

    parts.append("</svg>")
    legend = f'<div class="fdh-radar-legend">{"".join(legend_items)}</div>'
    return f'<div class="fdh-radar-wrap">{"".join(parts)}{legend}</div>'


def _radar_note(message: str) -> str:
    return f'<div class="fdh-radar-note">{html.escape(message)}</div>'


def _floored(value: float) -> float:
    """Map a 0-1 normalized metric onto the drawable radius band, so a
    last-place 0 still traces a small visible polygon (see
    `RADAR_RADIUS_FLOOR`)."""

    return RADAR_RADIUS_FLOOR + (1.0 - RADAR_RADIUS_FLOOR) * value


def _polar(
    center: tuple[float, float], radius: float, angle: float, value: float
) -> tuple[float, float]:
    return (
        center[0] + radius * value * math.cos(angle),
        center[1] + radius * value * math.sin(angle),
    )


def _points(
    center: tuple[float, float], radius: float, angles: list[float], values: list[float]
) -> str:
    return " ".join(
        "{:.1f},{:.1f}".format(*_polar(center, radius, angle, value))
        for angle, value in zip(angles, values)
    )


def _anchor(angle: float) -> str:
    x = math.cos(angle)
    if x > 0.1:
        return "start"
    if x < -0.1:
        return "end"
    return "middle"


def _baseline(angle: float) -> str:
    y = math.sin(angle)
    if y > 0.1:
        return "hanging"
    if y < -0.1:
        return "auto"
    return "middle"


def _short_axis_name(name: str) -> str:
    """Keep spoke labels short enough not to collide with their neighbours;
    the leaderboard table below carries every full metric name."""

    # 16 fits every metric name this project currently produces except
    # `alarms_per_unit_length`; notably it keeps `latency_ms_mean` whole.
    return name if len(name) <= 16 else name[:15] + "…"
