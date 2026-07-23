"""Tests for `web/charts.py`: the Benchmark tab's bar-chart frame and
hand-rolled radar SVG."""

from __future__ import annotations

import pytest

from fabric_defect_hub.web.charts import (
    MAX_RADAR_MODELS,
    MIN_RADAR_AXES,
    RADAR_CENTER,
    RADAR_HEIGHT,
    RADAR_RADIUS,
    RADAR_WIDTH,
    bar_frame,
    bar_y_limits,
    default_bar_metric,
    default_radar_axes,
    default_radar_models,
    metric_choices,
    model_choices,
    radar_axis_choices,
    render_radar_svg,
)


def _scored_rows():
    """Three models, best-first, as `web/benchmark.py::_render` yields them."""

    return [
        {
            "model": "Model A", "composite_score": 90.0, "technical_score": 95.0,
            "overhead_score": 85.0, "image_auroc": 0.95, "image_f1": 0.9,
            "fps": 100.0, "latency_ms_mean": 10.0, "peak_memory_mb": 500.0,
        },
        {
            "model": "Model B", "composite_score": 60.0, "technical_score": 70.0,
            "overhead_score": 50.0, "image_auroc": 0.80, "image_f1": 0.7,
            "fps": 50.0, "latency_ms_mean": 20.0, "peak_memory_mb": 900.0,
        },
        {
            "model": "Model C", "composite_score": 30.0, "technical_score": 20.0,
            "overhead_score": 40.0, "image_auroc": 0.60, "image_f1": 0.5,
            "fps": 20.0, "latency_ms_mean": 50.0, "peak_memory_mb": 1500.0,
        },
    ]


def test_metric_choices_drops_the_identifying_column():
    assert metric_choices(["model", "composite_score", "fps"]) == ["composite_score", "fps"]


def test_default_bar_metric_prefers_composite_score():
    assert default_bar_metric(["model", "fps", "composite_score"]) == "composite_score"


def test_default_bar_metric_falls_back_to_first_metric():
    assert default_bar_metric(["model", "image_auroc"]) == "image_auroc"


def test_default_bar_metric_of_empty_columns_is_none():
    assert default_bar_metric([]) is None
    assert default_bar_metric(["model"]) is None


def test_bar_frame_builds_one_row_per_model():
    frame = bar_frame(_scored_rows(), "image_auroc")

    assert list(frame.columns) == ["model", "image_auroc"]
    assert len(frame) == 3
    assert frame["model"].tolist() == ["Model A", "Model B", "Model C"]


def test_bar_frame_without_rows_or_metric_is_none():
    assert bar_frame([], "image_auroc") is None
    assert bar_frame(_scored_rows(), None) is None


def test_bar_frame_skips_rows_missing_a_numeric_value():
    rows = [{"model": "a", "fps": 10.0}, {"model": "b", "fps": ""}]
    frame = bar_frame(rows, "fps")
    assert frame["model"].tolist() == ["a"]


def test_bar_frame_of_a_metric_no_row_carries_is_none():
    # e.g. an overhead metric when the benchmark ran without profiling.
    assert bar_frame([{"model": "a", "image_auroc": 0.9}], "fps") is None


def test_bar_y_limits_anchor_the_axis_at_zero():
    limits = bar_y_limits(_scored_rows(), "fps")
    assert limits[0] == 0
    assert limits[1] >= 100.0  # the tallest bar, plus headroom


def test_bar_y_limits_without_plottable_values_is_none():
    assert bar_y_limits([], "fps") is None
    assert bar_y_limits(_scored_rows(), None) is None
    assert bar_y_limits([{"model": "a", "fps": 0.0}], "fps") is None


def test_radar_grid_and_series_declare_a_fill_so_css_loss_degrades_gracefully():
    """Without a declared fill an SVG polygon paints opaque black, which
    would bury the chart if the stylesheet ever failed to load."""

    rows = _scored_rows()
    html = render_radar_svg(rows, default_radar_axes(rows), ["Model A"])

    assert '<polygon class="fdh-radar-grid" fill="none"' in html
    assert 'fdh-radar-series-1" fill="none"' in html


def test_radar_axis_choices_excludes_derived_score_columns():
    axes = radar_axis_choices(_scored_rows())

    assert "composite_score" not in axes
    assert "technical_score" not in axes
    assert "overhead_score" not in axes
    assert "image_auroc" in axes
    assert "fps" in axes


def test_default_radar_models_takes_the_leaderboard_head():
    models = default_radar_models(_scored_rows())
    assert models == ["Model A", "Model B", "Model C"]
    assert len(models) <= MAX_RADAR_MODELS


def test_model_choices_lists_every_benchmarked_model():
    assert model_choices(_scored_rows()) == ["Model A", "Model B", "Model C"]


def test_render_radar_svg_draws_one_polygon_per_selected_model():
    rows = _scored_rows()
    axes = default_radar_axes(rows)
    html = render_radar_svg(rows, axes, ["Model A", "Model B"])

    assert "<svg" in html
    # Grid rings + spokes are polygons/lines too, so count the series class.
    assert html.count("fdh-radar-series fdh-radar-series-") == 2
    assert "fdh-radar-series-1" in html
    assert "fdh-radar-series-2" in html


def test_render_radar_svg_legend_names_every_model_with_its_area():
    rows = _scored_rows()
    html = render_radar_svg(rows, default_radar_axes(rows), ["Model A", "Model B"])

    assert "Model A" in html
    assert "Model B" in html
    assert "area" in html
    assert html.count("fdh-radar-swatch fdh-radar-swatch-") == 2


def test_render_radar_svg_caps_the_number_of_overlapping_polygons():
    rows = _scored_rows()
    html = render_radar_svg(rows, default_radar_axes(rows), model_choices(rows) + ["Model A"])
    assert html.count("fdh-radar-series fdh-radar-series-") == MAX_RADAR_MODELS


def test_render_radar_svg_requires_a_minimum_number_of_axes():
    rows = _scored_rows()
    html = render_radar_svg(rows, ["image_auroc", "fps"], ["Model A"])

    assert "<svg" not in html
    assert "fdh-radar-note" in html
    assert str(MIN_RADAR_AXES) in html


def test_render_radar_svg_without_results_returns_a_note():
    html = render_radar_svg([], [], [])
    assert "<svg" not in html
    assert "fdh-radar-note" in html


def test_render_radar_svg_without_a_valid_model_selection_returns_a_note():
    rows = _scored_rows()
    html = render_radar_svg(rows, default_radar_axes(rows), ["Nonexistent Model"])
    assert "<svg" not in html
    assert "fdh-radar-note" in html


def test_render_radar_svg_ignores_axes_that_are_not_chartable():
    rows = _scored_rows()
    axes = ["image_auroc", "fps", "latency_ms_mean", "composite_score"]
    html = render_radar_svg(rows, axes, ["Model A"])

    # composite_score is filtered out as a derived column, leaving 3 real
    # axes -- still enough to draw, and the label must not appear.
    assert "<svg" in html
    assert "composite_score" not in html


def test_render_radar_svg_escapes_model_names():
    rows = [dict(row) for row in _scored_rows()]
    rows[0]["model"] = "<script>alert(1)</script>"
    html = render_radar_svg(rows, default_radar_axes(rows), [rows[0]["model"]])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_radar_svg_normalizes_against_all_rows_not_just_the_selection():
    """Selecting fewer models must not rescale the ones still drawn."""

    rows = _scored_rows()
    axes = default_radar_axes(rows)
    with_all = render_radar_svg(rows, axes, ["Model A", "Model B"])
    only_two = render_radar_svg(rows, axes, ["Model B"])

    # Model B's polygon geometry is identical in both renders -- it is the
    # second series when both are drawn, the first when it is drawn alone.
    assert _series_points_raw(with_all, 2) == _series_points_raw(only_two, 1)


def _series_points(html: str, slot: int) -> list[tuple[float, float]]:
    raw = _series_points_raw(html, slot)
    return [tuple(float(value) for value in pair.split(",")) for pair in raw.split()]


def _series_points_raw(html: str, slot: int) -> str:
    marker = f'class="fdh-radar-series fdh-radar-series-{slot}"'
    return html.split(marker)[1].split('points="')[1].split('"')[0]


def test_worst_model_on_every_axis_still_draws_a_visible_polygon():
    """Min-max normalization puts the worst model at 0 on every axis; without
    a radius floor it would collapse to a dot at the centre and report 0%
    area, which reads as missing data rather than last place."""

    rows = _scored_rows()
    axes = default_radar_axes(rows)
    html = render_radar_svg(rows, axes, ["Model C"])  # worst on every metric

    points = _series_points(html, 1)
    centre = RADAR_CENTER
    assert all(point != pytest.approx(centre) for point in points)
    # The reported area describes the polygon actually drawn, so it is > 0.
    assert "area 0%" not in html


def test_best_model_reaches_the_rim_on_its_winning_axes():
    rows = _scored_rows()
    html = render_radar_svg(rows, ["image_auroc", "image_f1", "fps"], ["Model A"])
    points = _series_points(html, 1)
    centre_x, centre_y = RADAR_CENTER
    radii = [
        ((x - centre_x) ** 2 + (y - centre_y) ** 2) ** 0.5 for x, y in points
    ]
    # Model A tops image_auroc and image_f1, so those spokes hit full radius.
    assert max(radii) == pytest.approx(RADAR_RADIUS, abs=0.5)


def test_polygon_radii_stay_within_the_chart_bounds():
    rows = _scored_rows()
    html = render_radar_svg(rows, default_radar_axes(rows), ["Model A"])
    for x, y in _series_points(html, 1):
        assert 0 <= x <= RADAR_WIDTH
        assert 0 <= y <= RADAR_HEIGHT
