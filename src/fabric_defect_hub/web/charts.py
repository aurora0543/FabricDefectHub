"""Chart data and rendering for the Benchmark tab's leaderboard."""

from __future__ import annotations

import html
import math
from typing import Any

from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.scoring import normalize_metrics, polygon_area

NON_METRIC_COLUMNS = ("model",)
MIN_RADAR_AXES = 3
DEFAULT_RADAR_AXES = 6
MAX_RADAR_MODELS = 3

RADAR_PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
    "#14b8a6", "#d946ef", "#a855f7", "#eab308", "#64748b", "#0284c7"
]

# Smallest radius a spoke can plot at, as a fraction of the full radius.
RADAR_RADIUS_FLOOR = 0.08

RADAR_WIDTH = 480.0
RADAR_HEIGHT = 360.0
RADAR_CENTER = (RADAR_WIDTH / 2, 172.0)
RADAR_RADIUS = 100.0
RADAR_LABEL_OFFSET = 14.0


def _short_model_name(label: str) -> str:
    """Strip provenance suffixes like ' · Fabric trained' for clean chart labels."""

    if not label:
        return ""
    return label.split(" · ")[0].strip()


def metric_choices(columns: list[str]) -> list[str]:
    """Leaderboard columns a chart can plot, in the order they appear."""

    return [column for column in columns if column not in NON_METRIC_COLUMNS]


def default_bar_metric(columns: list[str]) -> str | None:
    choices = metric_choices(columns)
    if not choices:
        return None
    return "composite_score" if "composite_score" in choices else choices[0]


def bar_y_limits(scored_rows: list[dict[str, Any]], metric: str | None) -> list[float] | None:
    if not scored_rows or not metric:
        return None
    values = [
        row[metric] for row in scored_rows if isinstance(row.get(metric), (int, float))
    ]
    if not values:
        return None
    highest = max(values)
    return [0, highest * 1.05] if highest > 0 else None


def bar_frame(scored_rows: list[dict[str, Any]], metric: str | None):
    if not scored_rows or not metric:
        return None
    import pandas as pd

    data = [
        {"model": _short_model_name(row.get("model", "")), metric: row[metric]}
        for row in scored_rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not data:
        return None
    return pd.DataFrame(data)


def radar_axis_choices(scored_rows: list[dict[str, Any]]) -> list[str]:
    names, _ = normalize_metrics(scored_rows)
    derived = {"composite_score", "technical_score", "overhead_score"}
    return [name for name in names if name not in derived]


def default_radar_axes(scored_rows: list[dict[str, Any]]) -> list[str]:
    return radar_axis_choices(scored_rows)[:DEFAULT_RADAR_AXES]


def default_radar_models(scored_rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("model", "") for row in scored_rows]


def model_choices(scored_rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("model", "") for row in scored_rows]


def render_radar_svg(
    scored_rows: list[dict[str, Any]],
    axis_names: list[str],
    model_names: list[str],
    lang: str = DEFAULT_LANGUAGE,
) -> str:
    """Render the radar as an SVG + HTML legend string for a `gr.HTML` panel."""

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
        color = RADAR_PALETTE[(slot - 1) % len(RADAR_PALETTE)]
        radii = [
            _floored(by_model[model_name].get(name, 0.0)) for name in axes
        ]
        points = _points(center, radius, angles, radii)
        parts.append(
            f'<polygon class="fdh-radar-series fdh-radar-series-{slot}" '
            f'fill="none" stroke="{color}" stroke-width="2" points="{points}" />'
        )
        for angle, value in zip(angles, radii):
            x, y = _polar(center, radius, angle, value)
            parts.append(
                f'<circle class="fdh-radar-dot fdh-radar-dot-{slot}" '
                f'fill="{color}" cx="{x:.1f}" cy="{y:.1f}" r="3.5" />'
            )
        area_pct = polygon_area(radii) / polygon_area([1.0] * len(axes)) * 100
        legend_items.append(
            f'<span class="fdh-radar-key fdh-radar-key-{slot}" style="display:inline-flex;align-items:center;margin-right:12px;margin-bottom:6px;font-size:12px;">'
            f'<span class="fdh-radar-swatch fdh-radar-swatch-{slot}" style="display:inline-block;width:10px;height:10px;border-radius:2px;background-color:{color};margin-right:5px;"></span>'
            f"{html.escape(model_name)} · "
            f'{html.escape(tr(lang, "radar_area"))} {area_pct:.0f}%'
            f"</span>"
        )

    parts.append("</svg>")
    legend = f'<div class="fdh-radar-legend" style="display:flex;flex-wrap:wrap;margin-top:8px;">{"".join(legend_items)}</div>'
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
