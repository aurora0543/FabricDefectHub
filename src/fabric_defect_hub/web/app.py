"""Gradio application entry point for the FabricDefectHub workspace."""

from __future__ import annotations

from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, LANGUAGES, tr
from fabric_defect_hub.inference.session import InferenceSessionManager, format_session_status
from fabric_defect_hub.reporting import flatten_run_log_rows, latest_run_per_model, read_run_log
from fabric_defect_hub.web.charts import (
    MAX_RADAR_MODELS,
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
from fabric_defect_hub.web.benchmark import (
    DEFAULT_RUN_LOG_PATH,
    compatible_models,
    run_benchmark,
    score_preset_choices,
)
from fabric_defect_hub.web.single_image import (
    DATASET_CATALOG,
    MODEL_CATALOG,
    checkpoint_diagnostic,
    current_image,
    dataset_status as dataset_availability_status,
    default_dataset_root,
    detect_loaded_model,
    empty_gallery_state,
    image_scope_choices,
    load_random_samples,
    load_selected_model,
    model_status,
    move_image,
    render_prediction_tags,
    shot_mode_choices,
    split_choices,
    texture_choices,
    unload_selected_model,
)


CSS = """
html, body, .gradio-container { min-height: 100%; }
body.dark, body.dark .gradio-container { background: #07111f !important; color: #e5edf8 !important; }
body.dark .gradio-container { --body-background-fill: #07111f; --block-background-fill: #111d2e; --block-border-color: #25354c; --block-label-text-color: #a9bad0; --input-background-fill: #0b1626; --input-border-color: #30445f; --input-placeholder-color: #71849d; --body-text-color: #e5edf8; --color-accent: #fb923c; }
body:not(.dark), body:not(.dark) .gradio-container { background: #f4f7fb !important; color: #172033 !important; }
body:not(.dark) .gradio-container { --body-background-fill: #f4f7fb; --block-background-fill: #ffffff; --block-border-color: #d7e0ec; --block-label-text-color: #475569; --input-background-fill: #ffffff; --input-border-color: #cbd5e1; --input-placeholder-color: #94a3b8; --body-text-color: #172033; --color-accent: #ea6e18; }
.fdh-shell { max-width: 1440px; margin: 0 auto; padding: 0 8px 28px; }
.fdh-nav-row { display: flex; align-items: center; padding: 12px 2px 26px; gap: 16px; }
.fdh-nav { display: flex; align-items: center; justify-content: space-between; flex: 1; }
.fdh-brand { display: flex; gap: 12px; align-items: center; font-weight: 800; font-size: 22px; }
.fdh-logo { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; background: #f97316; color: #fff; }
.fdh-links { font-size: 14px; }
.fdh-lang { max-width: 120px; }
.fdh-lang .wrap-inner { min-height: 0 !important; }
.fdh-card, .fdh-control-card, .fdh-dataset-card { border-radius: 14px; padding: 10px; }
.fdh-control-card { min-height: 0; }
.fdh-card label, .fdh-card .wrap, .fdh-card .prose, .fdh-control-card label, .fdh-control-card .wrap, .fdh-dataset-card label { font-weight: 600; }
.fdh-status { padding: 10px 14px; border-radius: 10px; }
.fdh-caption { text-align: center; min-height: 26px; }
.fdh-nav-button button { min-width: 110px; border-radius: 10px !important; }
.fdh-primary button { background: #ea6e18 !important; border-color: #fb923c !important; color: #fff !important; }
.fdh-action-run button { height: 52px !important; min-height: 52px !important; max-height: 52px !important; font-size: 15px; font-weight: 700; width: 100%; margin-top: 6px; }
.fdh-dataset-actions { align-items: end; }
.fdh-placeholder { padding: 80px 30px; text-align: center; }
.fdh-tagpanel-header { font-weight: 700; margin-bottom: 8px; font-size: 15px; }
.fdh-tagpanel-empty { padding: 10px 2px; }
.fdh-tags { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.fdh-tags-column { flex-direction: column; align-items: flex-start; gap: 6px; }
.fdh-tag-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.fdh-tag { display: inline-flex; align-items: center; padding: 5px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; color: #fff; white-space: nowrap; line-height: 1.6; }
.fdh-tag-label { background: var(--color-accent); }
.fdh-tag-normal { background: #16a34a; }
.fdh-tag-anomalous { background: #dc2626; }
.fdh-tag-neutral { background: #64748b; }
body.dark .fdh-brand { color: #f8fafc; }
body.dark .fdh-links { color: #9fb0c8; }
body.dark .fdh-tagpanel-empty { color: #bdcbe0; }
body.dark .fdh-card, body.dark .fdh-control-card, body.dark .fdh-dataset-card { border: 1px solid #293b54; background: #101c2d; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
body.dark .fdh-card label, body.dark .fdh-card .wrap, body.dark .fdh-card .prose, body.dark .fdh-control-card label, body.dark .fdh-control-card .wrap, body.dark .fdh-dataset-card label { color: #dce7f5 !important; }
body.dark .fdh-card input, body.dark .fdh-card textarea, body.dark .fdh-card button.secondary, body.dark .fdh-control-card input, body.dark .fdh-control-card button.secondary, body.dark .fdh-dataset-card input, body.dark .fdh-dataset-card button.secondary { background: #0a1524 !important; color: #e5edf8 !important; border-color: #30445f !important; }
body.dark .fdh-card label.selected, body.dark .fdh-control-card label.selected, body.dark .fdh-dataset-card label.selected { background: #ea6e18 !important; border-color: #fb923c !important; color: #fff !important; }
body.dark .fdh-card button.secondary:hover, body.dark .fdh-control-card button.secondary:hover, body.dark .fdh-dataset-card button.secondary:hover { background: #16283f !important; border-color: #3d5776 !important; }
body.dark .fdh-card button.secondary:active, body.dark .fdh-control-card button.secondary:active, body.dark .fdh-dataset-card button.secondary:active { background: #1d3552 !important; border-color: #fb923c !important; }
body.dark .fdh-status { background: #182a3d; border: 1px solid #36516d; color: #dbeafe; }
body.dark .fdh-caption, body.dark .fdh-placeholder { color: #bdcbe0; }
body:not(.dark) .fdh-brand { color: #172033; }
body:not(.dark) .fdh-links { color: #64748b; }
body:not(.dark) .fdh-tagpanel-empty { color: #52627a; }
body:not(.dark) .fdh-card, body:not(.dark) .fdh-control-card, body:not(.dark) .fdh-dataset-card { border: 1px solid #d7e0ec; background: #ffffff; box-shadow: 0 6px 18px rgba(15,23,42,.07); }
body:not(.dark) .fdh-card label, body:not(.dark) .fdh-card .wrap, body:not(.dark) .fdh-card .prose, body:not(.dark) .fdh-control-card label, body:not(.dark) .fdh-control-card .wrap, body:not(.dark) .fdh-dataset-card label { color: #172033 !important; }
body:not(.dark) .fdh-card input, body:not(.dark) .fdh-card textarea, body:not(.dark) .fdh-card button.secondary, body:not(.dark) .fdh-control-card input, body:not(.dark) .fdh-control-card button.secondary, body:not(.dark) .fdh-dataset-card input, body:not(.dark) .fdh-dataset-card button.secondary { background: #ffffff !important; color: #172033 !important; border-color: #cbd5e1 !important; }
body:not(.dark) .fdh-card label.selected, body:not(.dark) .fdh-control-card label.selected, body:not(.dark) .fdh-dataset-card label.selected { background: #ea6e18 !important; border-color: #ea6e18 !important; color: #fff !important; }
body:not(.dark) .fdh-card button.secondary:hover, body:not(.dark) .fdh-control-card button.secondary:hover, body:not(.dark) .fdh-dataset-card button.secondary:hover { background: #f4f7fb !important; border-color: #94a3b8 !important; }
body:not(.dark) .fdh-card button.secondary:active, body:not(.dark) .fdh-control-card button.secondary:active, body:not(.dark) .fdh-dataset-card button.secondary:active { background: #e9edf3 !important; border-color: #ea6e18 !important; }
body:not(.dark) .fdh-status { background: #f8fafc; border: 1px solid #cbd5e1; color: #24344d; }
body:not(.dark) .fdh-caption, body:not(.dark) .fdh-placeholder { color: #52627a; }

/* -- Radar chart (web/charts.py emits the SVG; every color lives here so
   the chart follows the light/dark theme instead of baking in one theme's
   hexes). The three series slots are the categorical palette's first three
   hues, which are the only ones that clear colorblind separation when every
   pair can overlap -- as radar polygons can. Both modes were validated with
   the dataviz skill's validator against this app's own card surfaces
   (light #ffffff, dark #101c2d), --pairs all: all checks pass; the one
   sub-3:1 contrast warning (light aqua) is relieved by the always-present
   legend labels and the leaderboard table directly below, so identity never
   rests on hue alone. Marks follow the shared specs: 2px strokes, ~12%
   fills, hairline recessive grid, >=8px dots with a 2px surface ring. */
.fdh-radar-wrap { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 6px 0 2px; }
.fdh-radar { width: 100%; max-width: 480px; height: auto; }
.fdh-radar-grid { fill: none; stroke-width: 1; }
.fdh-radar-axis-label { font-size: 10px; font-family: inherit; }
.fdh-radar-series { stroke-width: 2; stroke-linejoin: round; }
.fdh-radar-dot { stroke-width: 2; }
.fdh-radar-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px 14px; font-size: 12px; }
.fdh-radar-key { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.fdh-radar-swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.fdh-radar-note { padding: 40px 16px; text-align: center; font-size: 13px; }
body.dark .fdh-radar-grid { stroke: #2c3d52; }
body.dark .fdh-radar-axis-label { fill: #a9bad0; }
body.dark .fdh-radar-dot { stroke: #101c2d; }
body.dark .fdh-radar-legend { color: #dce7f5; }
body.dark .fdh-radar-note { color: #bdcbe0; }
body.dark .fdh-radar-series-1 { stroke: #d95926; fill: rgba(217, 89, 38, .12); }
body.dark .fdh-radar-series-2 { stroke: #3987e5; fill: rgba(57, 135, 229, .12); }
body.dark .fdh-radar-series-3 { stroke: #199e70; fill: rgba(25, 158, 112, .12); }
body.dark .fdh-radar-dot-1 { fill: #d95926; }
body.dark .fdh-radar-dot-2 { fill: #3987e5; }
body.dark .fdh-radar-dot-3 { fill: #199e70; }
body.dark .fdh-radar-swatch-1 { background: #d95926; }
body.dark .fdh-radar-swatch-2 { background: #3987e5; }
body.dark .fdh-radar-swatch-3 { background: #199e70; }
body:not(.dark) .fdh-radar-grid { stroke: #dde4ee; }
body:not(.dark) .fdh-radar-axis-label { fill: #52627a; }
body:not(.dark) .fdh-radar-dot { stroke: #ffffff; }
body:not(.dark) .fdh-radar-legend { color: #24344d; }
body:not(.dark) .fdh-radar-note { color: #52627a; }
body:not(.dark) .fdh-radar-series-1 { stroke: #eb6834; fill: rgba(235, 104, 52, .12); }
body:not(.dark) .fdh-radar-series-2 { stroke: #2a78d6; fill: rgba(42, 120, 214, .12); }
body:not(.dark) .fdh-radar-series-3 { stroke: #1baf7a; fill: rgba(27, 175, 122, .12); }
body:not(.dark) .fdh-radar-dot-1 { fill: #eb6834; }
body:not(.dark) .fdh-radar-dot-2 { fill: #2a78d6; }
body:not(.dark) .fdh-radar-dot-3 { fill: #1baf7a; }
body:not(.dark) .fdh-radar-swatch-1 { background: #eb6834; }
body:not(.dark) .fdh-radar-swatch-2 { background: #2a78d6; }
body:not(.dark) .fdh-radar-swatch-3 { background: #1baf7a; }
"""


def _nav_html(lang: str) -> str:
    return (
        "<div class='fdh-nav'><div class='fdh-brand'><span class='fdh-logo'>FD</span>"
        f"FabricDefectHub</div><div class='fdh-links'>{tr(lang, 'nav_links')}</div></div>"
    )


def _lang_choices() -> list[tuple[str, str]]:
    return [(display, code) for code, display in LANGUAGES.items()]


def _history_chart_data(rows: list[dict], metric: str | None):
    """A `model -> metric` `pandas.DataFrame` for `gr.BarPlot`, one bar per
    model (its most recent logged run, via `reporting.latest_run_per_model`)
    -- `None` when there's no metric to chart yet."""

    if not metric:
        return None
    import pandas as pd

    data = [
        {"model": row.get("model", {}).get("name", ""), metric: row.get("metrics", {}).get(metric)}
        for row in latest_run_per_model(rows)
    ]
    data = [record for record in data if record[metric] is not None]
    if not data:
        return None
    return pd.DataFrame(data)


def create_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the UI dependencies first: pip install -r requirements.txt") from exc

    sessions = InferenceSessionManager()
    default_model = next(iter(MODEL_CATALOG))
    default_dataset = next(iter(DATASET_CATALOG))
    lang0 = DEFAULT_LANGUAGE

    with gr.Blocks(title="FabricDefectHub") as app:
        lang_state = gr.State(lang0)
        with gr.Column(elem_classes="fdh-shell"):
            with gr.Row(elem_classes="fdh-nav-row"):
                nav_html = gr.HTML(_nav_html(lang0))
                lang_choice = gr.Dropdown(
                    choices=_lang_choices(), value=lang0, show_label=False,
                    container=False, scale=0, min_width=100, elem_classes="fdh-lang",
                )
            with gr.Tabs():
                with gr.Tab(tr(lang0, "tab_single_image"), id="single-image") as tab_single:
                    state = gr.State(empty_gallery_state())

                    # Column 1: model session + Run detection. Column 2: the
                    # sampled dataset image. Column 3: the annotated result.
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            model_header = gr.Markdown(tr(lang0, "model_session_header"))
                            model_choice = gr.Dropdown(
                                list(MODEL_CATALOG), value=default_model, label=tr(lang0, "model_dropdown_label")
                            )
                            model_state = gr.Markdown(model_status(default_model, lang0), elem_classes="fdh-status")
                            with gr.Row():
                                load_model_button = gr.Button(tr(lang0, "btn_load_model"), variant="secondary")
                                unload_model_button = gr.Button(tr(lang0, "btn_unload_model"), variant="secondary")
                            verify_model_button = gr.Button(tr(lang0, "btn_inspect_checkpoint"), variant="secondary")
                            detect_button = gr.Button(
                                tr(lang0, "btn_run_detection"), variant="primary",
                                elem_classes="fdh-primary fdh-action-run",
                            )
                        with gr.Column(scale=4, elem_classes="fdh-card"):
                            source_image = gr.Image(
                                label=tr(lang0, "image_selected_label"), height=360, interactive=False
                            )
                            position = gr.Markdown(tr(lang0, "caption_no_image"), elem_classes="fdh-caption")
                            with gr.Row():
                                previous = gr.Button(tr(lang0, "btn_previous"), elem_classes="fdh-nav-button")
                                next_image = gr.Button(tr(lang0, "btn_next"), elem_classes="fdh-nav-button")
                        with gr.Column(scale=5, elem_classes="fdh-card"):
                            result_image = gr.Image(
                                label=tr(lang0, "image_result_label"), height=360, interactive=False
                            )

                    # Runtime memory sits under the sampled image; the new
                    # colored-tag prediction result sits under the annotated
                    # result image right above it.
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4, elem_classes="fdh-card"):
                            runtime_header = gr.Markdown(tr(lang0, "runtime_memory_header"))
                            runtime_state = gr.Markdown(
                                format_session_status(sessions.status(), lang0), elem_classes="fdh-status"
                            )
                        with gr.Column(scale=5, elem_classes="fdh-card"):
                            result_summary = gr.HTML(render_prediction_tags({}, lang0), elem_classes="fdh-status")
                            inference_status = gr.Markdown(tr(lang0, "inference_hint_start"), elem_classes="fdh-status")

                    with gr.Column(elem_classes="fdh-dataset-card"):
                        dataset_header = gr.Markdown(tr(lang0, "dataset_sampler_header"))
                        with gr.Row():
                            with gr.Column(scale=3):
                                dataset_choice = gr.Dropdown(
                                    list(DATASET_CATALOG), value=default_dataset, label=tr(lang0, "dataset_dropdown_label")
                                )
                            with gr.Column(scale=3):
                                texture_choice = gr.Dropdown(
                                    texture_choices(default_dataset), value="All textures",
                                    label=tr(lang0, "texture_dropdown_label"),
                                )
                            with gr.Column(scale=2):
                                split = gr.Radio(choices=split_choices(lang0), value="test", label=tr(lang0, "split_label"))
                            with gr.Column(scale=2):
                                sample_count = gr.Slider(4, 12, value=8, step=1, label=tr(lang0, "slider_random_images_label"))
                            with gr.Column(scale=2):
                                image_scope = gr.Dropdown(
                                    choices=image_scope_choices(lang0), value="All images",
                                    label=tr(lang0, "image_selection_label"),
                                )
                            with gr.Column(scale=2):
                                shot_mode = gr.Radio(
                                    choices=shot_mode_choices(lang0), value="Full-shot", label=tr(lang0, "sample_regime_label")
                                )
                            with gr.Column(scale=2, elem_classes="fdh-dataset-actions"):
                                load_button = gr.Button(tr(lang0, "btn_load_random_images"))
                        dataset_status = gr.Markdown(
                            dataset_availability_status(default_dataset, lang0), elem_classes="fdh-status"
                        )

                    def load_handler(dataset, texture, selected_split, count, selected_scope, selected_shot_mode, lang):
                        try:
                            new_state, image, caption, status = load_random_samples(
                                dataset,
                                selected_split,
                                count,
                                texture_label=texture,
                                image_scope=selected_scope,
                                shot_mode=selected_shot_mode,
                                lang=lang,
                            )
                            return (
                                new_state, image, caption, status, None,
                                render_prediction_tags({}, lang), tr(lang, "inference_hint_ready"),
                            )
                        except Exception as exc:
                            return (
                                empty_gallery_state(), None, tr(lang, "caption_no_image"),
                                tr(lang, "dataset_load_error", error=exc), None, render_prediction_tags({}, lang), "",
                            )

                    def move_handler(current_state, direction, lang):
                        new_state, image, caption = move_image(current_state, direction, lang)
                        return new_state, image, caption, None, render_prediction_tags({}, lang), tr(lang, "inference_hint_changed")

                    def detect_handler(current_state, model_label, lang):
                        image, summary, status = detect_loaded_model(sessions, current_state, model_label, lang)
                        return image, render_prediction_tags(summary, lang), status, format_session_status(sessions.status(), lang)

                    def load_model_handler(model_label, lang):
                        try:
                            return format_session_status(load_selected_model(sessions, model_label), lang)
                        except Exception as exc:
                            return tr(lang, "model_load_failed", error_type=type(exc).__name__, error=exc)

                    def unload_model_handler(lang):
                        return format_session_status(unload_selected_model(sessions), lang)

                    load_button.click(
                        load_handler,
                        inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode, lang_state],
                        outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                    )
                    previous.click(
                        lambda current_state, lang: move_handler(current_state, -1, lang),
                        inputs=[state, lang_state],
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    next_image.click(
                        lambda current_state, lang: move_handler(current_state, 1, lang),
                        inputs=[state, lang_state],
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    model_choice.change(
                        model_status,
                        inputs=[model_choice, lang_state],
                        outputs=model_state,
                    )
                    verify_model_button.click(
                        checkpoint_diagnostic,
                        inputs=[model_choice, lang_state],
                        outputs=model_state,
                    )
                    load_model_button.click(
                        load_model_handler,
                        inputs=[model_choice, lang_state],
                        outputs=runtime_state,
                    )
                    unload_model_button.click(
                        unload_model_handler,
                        inputs=lang_state,
                        outputs=runtime_state,
                    )
                    dataset_choice.change(
                        lambda dataset: gr.Dropdown(choices=texture_choices(dataset), value="All textures"),
                        inputs=dataset_choice,
                        outputs=texture_choice,
                    )
                    for selection in (texture_choice, split, image_scope, shot_mode):
                        selection.change(
                            load_handler,
                            inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode, lang_state],
                            outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                        )
                    detect_button.click(
                        detect_handler,
                        inputs=[state, model_choice, lang_state],
                        outputs=[result_image, result_summary, inference_status, runtime_state],
                    )

                with gr.Tab(tr(lang0, "tab_benchmark"), id="benchmark") as tab_bench:
                    bench_header = gr.Markdown(tr(lang0, "benchmark_header"))
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_dataset = gr.Dropdown(
                                list(DATASET_CATALOG), value=default_dataset, label=tr(lang0, "benchmark_dataset_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_texture = gr.Dropdown(
                                texture_choices(default_dataset), value="All textures",
                                label=tr(lang0, "benchmark_texture_label"),
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_shot_mode = gr.Radio(
                                choices=shot_mode_choices(lang0), value="Full-shot",
                                label=tr(lang0, "benchmark_shot_label"),
                            )
                    with gr.Row():
                        with gr.Column(scale=7, elem_classes="fdh-control-card"):
                            bench_models = gr.CheckboxGroup(
                                compatible_models(default_dataset), label=tr(lang0, "benchmark_models_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card fdh-action-run"):
                            bench_run_button = gr.Button(
                                tr(lang0, "btn_run_benchmark"), variant="primary", elem_classes="fdh-primary"
                            )
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            bench_profiling = gr.Checkbox(
                                value=False, label=tr(lang0, "benchmark_profiling_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_score_preset = gr.Dropdown(
                                choices=score_preset_choices(lang0), value="balanced",
                                label=tr(lang0, "benchmark_score_preset_label"),
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_custom_weight = gr.Slider(
                                0, 1, value=0.5, step=0.05,
                                label=tr(lang0, "benchmark_custom_weight_label"), visible=False,
                            )
                    bench_status = gr.Markdown(tr(lang0, "benchmark_placeholder"), elem_classes="fdh-status")

                    # Charts sit above the leaderboard table, and their
                    # selectors sit above both in one shared filter row --
                    # never inside an individual chart card, so it stays
                    # obvious what each control scopes. `bench_rows_state`
                    # holds the last run's scored rows so changing a
                    # selector redraws from memory instead of re-running
                    # every model's inference.
                    bench_rows_state = gr.State([])
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            chart_metric = gr.Dropdown(choices=[], label=tr(lang0, "chart_metric_label"))
                        with gr.Column(scale=5, elem_classes="fdh-control-card"):
                            radar_axes = gr.CheckboxGroup(choices=[], label=tr(lang0, "radar_axes_label"))
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            radar_models = gr.CheckboxGroup(
                                choices=[], label=tr(lang0, "radar_models_label", count=MAX_RADAR_MODELS)
                            )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=6, elem_classes="fdh-card"):
                            bench_bar_chart = gr.BarPlot(
                                x="model", label=tr(lang0, "chart_bar_label"), sort="-y", height=320,
                            )
                        with gr.Column(scale=6, elem_classes="fdh-card"):
                            bench_radar = gr.HTML(
                                render_radar_svg([], [], [], lang0), label=tr(lang0, "chart_radar_label")
                            )
                    bench_results = gr.Dataframe(label=tr(lang0, "leaderboard_label"), interactive=False, wrap=True)

                    def bench_dataset_change_handler(dataset_label, lang):
                        return (
                            gr.Dropdown(choices=texture_choices(dataset_label), value="All textures"),
                            gr.CheckboxGroup(choices=compatible_models(dataset_label), value=[]),
                        )

                    def bench_run_handler(
                        dataset_label, texture_label, shot_mode_value, model_labels,
                        include_profiling, score_preset, custom_weight, lang,
                    ):
                        """Stream the leaderboard, and re-derive the charts
                        (and their selectors' choices) on every model that
                        finishes -- the available metrics only become known
                        as results land, so the selectors can't be populated
                        up front."""

                        for columns, rows, status, scored in run_benchmark(
                            dataset_label, texture_label, shot_mode_value, model_labels, lang,
                            include_profiling=include_profiling,
                            score_preset=score_preset,
                            custom_technical_weight=custom_weight,
                        ):
                            table = gr.Dataframe(headers=columns, value=rows) if columns else gr.Dataframe(value=[])
                            metric = default_bar_metric(columns)
                            axes = default_radar_axes(scored)
                            models = default_radar_models(scored)
                            yield (
                                table,
                                status,
                                scored,
                                gr.Dropdown(choices=metric_choices(columns), value=metric),
                                gr.CheckboxGroup(choices=radar_axis_choices(scored), value=axes),
                                gr.CheckboxGroup(choices=model_choices(scored), value=models),
                                gr.BarPlot(
                                    value=bar_frame(scored, metric), x="model",
                                    y=metric or "model", y_lim=bar_y_limits(scored, metric),
                                ),
                                render_radar_svg(scored, axes, models, lang),
                            )

                    def bar_redraw_handler(scored, metric):
                        return gr.BarPlot(
                            value=bar_frame(scored, metric), x="model",
                            y=metric or "model", y_lim=bar_y_limits(scored, metric),
                        )

                    def radar_redraw_handler(scored, axes, models, lang):
                        return render_radar_svg(scored, axes or [], models or [], lang)

                    bench_dataset.change(
                        bench_dataset_change_handler,
                        inputs=[bench_dataset, lang_state],
                        outputs=[bench_texture, bench_models],
                    )
                    bench_score_preset.change(
                        lambda preset: gr.Slider(visible=preset == "custom"),
                        inputs=bench_score_preset,
                        outputs=bench_custom_weight,
                    )
                    bench_run_button.click(
                        bench_run_handler,
                        inputs=[
                            bench_dataset, bench_texture, bench_shot_mode, bench_models,
                            bench_profiling, bench_score_preset, bench_custom_weight, lang_state,
                        ],
                        outputs=[
                            bench_results, bench_status, bench_rows_state,
                            chart_metric, radar_axes, radar_models,
                            bench_bar_chart, bench_radar,
                        ],
                    )
                    chart_metric.change(
                        bar_redraw_handler,
                        inputs=[bench_rows_state, chart_metric],
                        outputs=bench_bar_chart,
                    )
                    for radar_selector in (radar_axes, radar_models):
                        radar_selector.change(
                            radar_redraw_handler,
                            inputs=[bench_rows_state, radar_axes, radar_models, lang_state],
                            outputs=bench_radar,
                        )

                with gr.Tab(tr(lang0, "tab_run_history"), id="run-history") as tab_history:
                    history_header = gr.Markdown(tr(lang0, "history_header"))
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes="fdh-control-card"):
                            history_path = gr.Textbox(
                                value=DEFAULT_RUN_LOG_PATH, label=tr(lang0, "history_path_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            history_metric = gr.Dropdown(choices=[], label=tr(lang0, "history_metric_label"))
                        with gr.Column(scale=2, elem_classes="fdh-control-card fdh-action-run"):
                            history_refresh_button = gr.Button(
                                tr(lang0, "btn_history_refresh"), variant="secondary"
                            )
                    history_status = gr.Markdown(tr(lang0, "history_no_runs"), elem_classes="fdh-status")
                    history_table = gr.Dataframe(
                        label=tr(lang0, "history_table_label"), interactive=False, wrap=True
                    )
                    history_chart = gr.BarPlot(label=tr(lang0, "history_chart_label"))

                    def history_refresh_handler(path, metric, lang):
                        try:
                            rows = read_run_log(path)
                        except (OSError, ValueError) as exc:
                            return (
                                gr.Dataframe(value=[]), gr.Dropdown(choices=[]), None,
                                tr(lang, "history_load_error", error=exc),
                            )
                        if not rows:
                            return gr.Dataframe(value=[]), gr.Dropdown(choices=[]), None, tr(lang, "history_no_runs")

                        columns, table = flatten_run_log_rows(rows)
                        excluded = {"timestamp_utc", "model", "backend", "task", "dataset", "device"}
                        metric_choices = [column for column in columns if column not in excluded]
                        selected_metric = metric if metric in metric_choices else (
                            metric_choices[0] if metric_choices else None
                        )
                        return (
                            gr.Dataframe(headers=columns, value=table),
                            gr.Dropdown(choices=metric_choices, value=selected_metric),
                            _history_chart_data(rows, selected_metric),
                            tr(lang, "history_table_label"),
                        )

                    def history_metric_change_handler(path, metric, lang):
                        rows = read_run_log(path)
                        return _history_chart_data(rows, metric)

                    history_refresh_button.click(
                        history_refresh_handler,
                        inputs=[history_path, history_metric, lang_state],
                        outputs=[history_table, history_metric, history_chart, history_status],
                    )
                    history_metric.change(
                        history_metric_change_handler,
                        inputs=[history_path, history_metric, lang_state],
                        outputs=history_chart,
                    )

        # -- Language toggle: rebuilds every static label/header/button/
        # placeholder in the new language, and recomputes the handful of
        # dynamic panels we have live data for (model/dataset/runtime
        # status, the currently-shown image's caption) instead of just
        # resetting them. Transient result panels (the last inference's
        # tags, the benchmark leaderboard's status line) reset to their
        # placeholder text -- there's no stored "was a result already
        # shown" state to re-render them from, and resetting on a language
        # switch is an acceptable, expected trade-off.
        def apply_language(lang, model_label, dataset_label, gallery_state):
            _, caption = current_image(gallery_state, lang)
            return (
                lang,
                _nav_html(lang),
                gr.Tab(label=tr(lang, "tab_single_image")),
                gr.Tab(label=tr(lang, "tab_benchmark")),
                gr.Tab(label=tr(lang, "tab_run_history")),
                tr(lang, "model_session_header"),
                gr.Dropdown(label=tr(lang, "model_dropdown_label")),
                model_status(model_label, lang),
                tr(lang, "btn_load_model"),
                tr(lang, "btn_unload_model"),
                tr(lang, "btn_inspect_checkpoint"),
                tr(lang, "btn_run_detection"),
                gr.Image(label=tr(lang, "image_selected_label")),
                caption,
                tr(lang, "btn_previous"),
                tr(lang, "btn_next"),
                gr.Image(label=tr(lang, "image_result_label")),
                tr(lang, "runtime_memory_header"),
                format_session_status(sessions.status(), lang),
                render_prediction_tags({}, lang),
                tr(lang, "inference_hint_ready") if gallery_state.get("samples") else tr(lang, "inference_hint_start"),
                tr(lang, "dataset_sampler_header"),
                gr.Dropdown(label=tr(lang, "dataset_dropdown_label")),
                gr.Dropdown(label=tr(lang, "texture_dropdown_label")),
                gr.Radio(choices=split_choices(lang), label=tr(lang, "split_label")),
                gr.Slider(label=tr(lang, "slider_random_images_label")),
                gr.Dropdown(choices=image_scope_choices(lang), label=tr(lang, "image_selection_label")),
                gr.Radio(choices=shot_mode_choices(lang), label=tr(lang, "sample_regime_label")),
                tr(lang, "btn_load_random_images"),
                dataset_availability_status(dataset_label, lang),
                tr(lang, "benchmark_header"),
                gr.Dropdown(label=tr(lang, "benchmark_dataset_label")),
                gr.Dropdown(label=tr(lang, "benchmark_texture_label")),
                gr.Radio(choices=shot_mode_choices(lang), label=tr(lang, "benchmark_shot_label")),
                gr.CheckboxGroup(label=tr(lang, "benchmark_models_label")),
                tr(lang, "btn_run_benchmark"),
                gr.Checkbox(label=tr(lang, "benchmark_profiling_label")),
                gr.Dropdown(choices=score_preset_choices(lang), label=tr(lang, "benchmark_score_preset_label")),
                gr.Slider(label=tr(lang, "benchmark_custom_weight_label")),
                tr(lang, "benchmark_placeholder"),
                gr.Dropdown(label=tr(lang, "chart_metric_label")),
                gr.CheckboxGroup(label=tr(lang, "radar_axes_label")),
                gr.CheckboxGroup(label=tr(lang, "radar_models_label", count=MAX_RADAR_MODELS)),
                gr.BarPlot(label=tr(lang, "chart_bar_label")),
                gr.HTML(label=tr(lang, "chart_radar_label")),
                gr.Dataframe(label=tr(lang, "leaderboard_label")),
                tr(lang, "history_header"),
                gr.Textbox(label=tr(lang, "history_path_label")),
                gr.Dropdown(label=tr(lang, "history_metric_label")),
                tr(lang, "btn_history_refresh"),
                tr(lang, "history_no_runs"),
                gr.Dataframe(label=tr(lang, "history_table_label")),
                gr.BarPlot(label=tr(lang, "history_chart_label")),
            )

        lang_choice.change(
            apply_language,
            inputs=[lang_choice, model_choice, dataset_choice, state],
            outputs=[
                lang_state, nav_html, tab_single, tab_bench, tab_history,
                model_header, model_choice, model_state,
                load_model_button, unload_model_button, verify_model_button, detect_button,
                source_image, position, previous, next_image, result_image,
                runtime_header, runtime_state, result_summary, inference_status,
                dataset_header, dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode,
                load_button, dataset_status,
                bench_header, bench_dataset, bench_texture, bench_shot_mode, bench_models,
                bench_run_button, bench_profiling, bench_score_preset, bench_custom_weight,
                bench_status, chart_metric, radar_axes, radar_models,
                bench_bar_chart, bench_radar, bench_results,
                history_header, history_path, history_metric, history_refresh_button,
                history_status, history_table, history_chart,
            ],
        )
    return app


def launch(**kwargs):
    kwargs.setdefault("css", CSS)
    # The cloud host this project is deployed to only has port 6008 open;
    # standardize on it everywhere so `fdh-ui` works unmodified there.
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", 6008)
    # Gradio 6 only caches files below the workspace or system temp directory
    # unless an external location is explicitly trusted. Every catalog
    # dataset typically lives on external storage reached through a
    # `data/<dir>` symlink, so register each one's *resolved* root (the real
    # on-disk location, not the symlink) for image display.
    configured_paths = list(kwargs.pop("allowed_paths", []) or [])
    for dataset_label in DATASET_CATALOG:
        dataset_root = default_dataset_root(dataset_label)
        if dataset_root and dataset_root not in configured_paths:
            configured_paths.append(dataset_root)
    kwargs["allowed_paths"] = configured_paths
    try:
        import gradio as gr

        kwargs.setdefault("theme", gr.themes.Base(primary_hue="orange", neutral_hue="slate"))
    except ImportError:
        pass
    return create_app().launch(**kwargs)
