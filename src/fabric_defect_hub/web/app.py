"""Gradio application entry point for the FabricDefectHub workspace."""

from __future__ import annotations

from fabric_defect_hub.inference.session import InferenceSessionManager, format_session_status
from fabric_defect_hub.web.single_image import (
    DATASET_CATALOG,
    ALL_IMAGES,
    DEFECT_ONLY,
    NORMAL_ONLY,
    MODEL_CATALOG,
    SHOT_FEW,
    SHOT_FULL,
    checkpoint_diagnostic,
    dataset_status as dataset_availability_status,
    default_dataset_root,
    detect_loaded_model,
    empty_gallery_state,
    format_prediction_summary,
    load_random_samples,
    model_status,
    move_image,
    load_selected_model,
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
.fdh-nav { display: flex; align-items: center; justify-content: space-between; padding: 12px 2px 26px; }
.fdh-brand { display: flex; gap: 12px; align-items: center; font-weight: 800; font-size: 22px; }
.fdh-logo { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; background: #f97316; color: #fff; }
.fdh-links { font-size: 14px; }
.fdh-hero { padding: 28px 32px; border-radius: 18px; margin-bottom: 18px; }
.fdh-hero h1 { margin: 0 0 6px; font-size: 30px; }
.fdh-hero p { margin: 0; }
.fdh-card, .fdh-control-card, .fdh-dataset-card { border-radius: 14px; padding: 10px; }
.fdh-control-card { min-height: 0; }
.fdh-card label, .fdh-card .wrap, .fdh-card .prose, .fdh-control-card label, .fdh-control-card .wrap, .fdh-dataset-card label { font-weight: 600; }
.fdh-status { padding: 10px 14px; border-radius: 10px; }
.fdh-caption { text-align: center; min-height: 26px; }
.fdh-nav-button button { min-width: 110px; border-radius: 10px !important; }
.fdh-primary button { background: #ea6e18 !important; border-color: #fb923c !important; color: #fff !important; }
.fdh-action-run button { height: 60px !important; min-height: 60px !important; max-height: 60px !important; font-size: 16px; font-weight: 700; flex: 0 0 auto !important; }
.fdh-dataset-actions { align-items: end; }
.fdh-placeholder { padding: 80px 30px; text-align: center; }
body.dark .fdh-brand { color: #f8fafc; }
body.dark .fdh-links { color: #9fb0c8; }
body.dark .fdh-hero { border: 1px solid #34465f; background: linear-gradient(118deg, #132842, #17233a 58%, #29233b); color: #f8fafc; box-shadow: 0 18px 42px rgba(0,0,0,.22); }
body.dark .fdh-hero p { color: #c4d0e0; }
body.dark .fdh-card, body.dark .fdh-control-card, body.dark .fdh-dataset-card { border: 1px solid #293b54; background: #101c2d; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
body.dark .fdh-card label, body.dark .fdh-card .wrap, body.dark .fdh-card .prose, body.dark .fdh-control-card label, body.dark .fdh-control-card .wrap, body.dark .fdh-dataset-card label { color: #dce7f5 !important; }
body.dark .fdh-card input, body.dark .fdh-card textarea, body.dark .fdh-card button.secondary, body.dark .fdh-control-card input, body.dark .fdh-control-card button.secondary, body.dark .fdh-dataset-card input, body.dark .fdh-dataset-card button.secondary { background: #0a1524 !important; color: #e5edf8 !important; border-color: #30445f !important; }
body.dark .fdh-status { background: #182a3d; border: 1px solid #36516d; color: #dbeafe; }
body.dark .fdh-caption, body.dark .fdh-placeholder { color: #bdcbe0; }
body:not(.dark) .fdh-brand { color: #172033; }
body:not(.dark) .fdh-links { color: #64748b; }
body:not(.dark) .fdh-hero { border: 1px solid #d7e0ec; background: linear-gradient(118deg, #ffffff, #f4f7fb 60%, #fff1e8); color: #172033; box-shadow: 0 12px 30px rgba(15,23,42,.08); }
body:not(.dark) .fdh-hero p { color: #52627a; }
body:not(.dark) .fdh-card, body:not(.dark) .fdh-control-card, body:not(.dark) .fdh-dataset-card { border: 1px solid #d7e0ec; background: #ffffff; box-shadow: 0 6px 18px rgba(15,23,42,.07); }
body:not(.dark) .fdh-card label, body:not(.dark) .fdh-card .wrap, body:not(.dark) .fdh-card .prose, body:not(.dark) .fdh-control-card label, body:not(.dark) .fdh-control-card .wrap, body:not(.dark) .fdh-dataset-card label { color: #172033 !important; }
body:not(.dark) .fdh-card input, body:not(.dark) .fdh-card textarea, body:not(.dark) .fdh-card button.secondary, body:not(.dark) .fdh-control-card input, body:not(.dark) .fdh-control-card button.secondary, body:not(.dark) .fdh-dataset-card input, body:not(.dark) .fdh-dataset-card button.secondary { background: #ffffff !important; color: #172033 !important; border-color: #cbd5e1 !important; }
body:not(.dark) .fdh-status { background: #f8fafc; border: 1px solid #cbd5e1; color: #24344d; }
body:not(.dark) .fdh-caption, body:not(.dark) .fdh-placeholder { color: #52627a; }
"""


def create_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the UI extra first: pip install -e '.[ui]'.") from exc

    sessions = InferenceSessionManager()
    with gr.Blocks(title="FabricDefectHub") as app:
        with gr.Column(elem_classes="fdh-shell"):
            gr.HTML(
                "<div class='fdh-nav'><div class='fdh-brand'><span class='fdh-logo'>FD</span>"
                "FabricDefectHub</div><div class='fdh-links'>Workspace · Datasets · Models · Results</div></div>"
            )
            with gr.Tabs():
                with gr.Tab("Single Image Detection", id="single-image"):
                    state = gr.State(empty_gallery_state())
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes="fdh-control-card"):
                            gr.Markdown("### Model session")
                            model_choice = gr.Dropdown(list(MODEL_CATALOG), value=next(iter(MODEL_CATALOG)), label="Local trained model")
                            model_state = gr.Markdown(model_status(next(iter(MODEL_CATALOG))), elem_classes="fdh-status")
                            with gr.Row():
                                load_model_button = gr.Button("Load model", variant="secondary")
                                unload_model_button = gr.Button("Unload model", variant="secondary")
                                verify_model_button = gr.Button("Inspect checkpoint", variant="secondary")
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            gr.Markdown("### Runtime memory")
                            runtime_state = gr.Markdown(format_session_status(sessions.status()), elem_classes="fdh-status")
                        with gr.Column(scale=3, elem_classes="fdh-control-card fdh-action-run"):
                            gr.Markdown("### Inference")
                            detect_button = gr.Button("Run detection", variant="primary", elem_classes="fdh-primary")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5, elem_classes="fdh-card"):
                            source_image = gr.Image(label="Selected dataset image", height=390, interactive=False)
                            position = gr.Markdown("No image loaded yet.", elem_classes="fdh-caption")
                            with gr.Row():
                                previous = gr.Button("← Previous", elem_classes="fdh-nav-button")
                                next_image = gr.Button("Next →", elem_classes="fdh-nav-button")
                        with gr.Column(scale=7, elem_classes="fdh-card"):
                            result_image = gr.Image(label="Inference result", height=390, interactive=False)
                            result_summary = gr.Markdown("### Prediction summary\nNo prediction available yet.", elem_classes="fdh-status")
                            inference_status = gr.Markdown("Select an image and a ready model to begin.", elem_classes="fdh-status")

                    with gr.Column(elem_classes="fdh-dataset-card"):
                        gr.Markdown("### Dataset sampler")
                        with gr.Row():
                            with gr.Column(scale=3):
                                dataset_choice = gr.Dropdown(list(DATASET_CATALOG), value=next(iter(DATASET_CATALOG)), label="Dataset")
                            with gr.Column(scale=3):
                                texture_choice = gr.Dropdown(texture_choices(next(iter(DATASET_CATALOG))), value="All textures", label="Texture / pattern")
                            with gr.Column(scale=2):
                                split = gr.Radio(["test", "train"], value="test", label="Split")
                            with gr.Column(scale=2):
                                sample_count = gr.Slider(4, 12, value=8, step=1, label="Random images")
                            with gr.Column(scale=2):
                                image_scope = gr.Dropdown(
                                    [ALL_IMAGES, DEFECT_ONLY, NORMAL_ONLY],
                                    value=ALL_IMAGES,
                                    label="Image selection",
                                )
                            with gr.Column(scale=2):
                                shot_mode = gr.Radio(
                                    [SHOT_FULL, SHOT_FEW],
                                    value=SHOT_FULL,
                                    label="Sample regime",
                                )
                            with gr.Column(scale=2, elem_classes="fdh-dataset-actions"):
                                load_button = gr.Button("Load random images")
                        dataset_status = gr.Markdown(
                            dataset_availability_status(next(iter(DATASET_CATALOG))), elem_classes="fdh-status"
                        )

                    def load_handler(dataset, texture, selected_split, count, selected_scope, selected_shot_mode):
                        try:
                            new_state, image, caption, status = load_random_samples(
                                dataset,
                                selected_split,
                                count,
                                texture_label=texture,
                                image_scope=selected_scope,
                                shot_mode=selected_shot_mode,
                            )
                            return new_state, image, caption, status, None, "### Prediction summary\nNo prediction available yet.", "Image ready. Choose a model to run detection."
                        except Exception as exc:
                            return empty_gallery_state(), None, "No image loaded yet.", f"🔴 **Dataset unavailable** — {exc}", None, "### Prediction summary\nNo prediction available yet.", ""

                    def move_handler(current_state, direction):
                        new_state, image, caption = move_image(current_state, direction)
                        return new_state, image, caption, None, "### Prediction summary\nNo prediction available yet.", "Image changed. Run detection again for this image."

                    def detect_handler(current_state, model_label):
                        image, summary, status = detect_loaded_model(sessions, current_state, model_label)
                        return image, format_prediction_summary(summary), status, format_session_status(sessions.status())

                    def load_model_handler(model_label):
                        try:
                            return format_session_status(load_selected_model(sessions, model_label))
                        except Exception as exc:
                            return f"🔴 **Model load failed** — {type(exc).__name__}: {exc}"

                    def unload_model_handler():
                        return format_session_status(unload_selected_model(sessions))

                    load_button.click(
                        load_handler,
                        inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode],
                        outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                    )
                    previous.click(
                        lambda current_state: move_handler(current_state, -1),
                        inputs=state,
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    next_image.click(
                        lambda current_state: move_handler(current_state, 1),
                        inputs=state,
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    model_choice.change(
                        model_status,
                        inputs=model_choice,
                        outputs=model_state,
                    )
                    verify_model_button.click(
                        checkpoint_diagnostic,
                        inputs=model_choice,
                        outputs=model_state,
                    )
                    load_model_button.click(
                        load_model_handler,
                        inputs=model_choice,
                        outputs=runtime_state,
                    )
                    unload_model_button.click(
                        unload_model_handler,
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
                            inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode],
                            outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                        )
                    detect_button.click(
                        detect_handler,
                        inputs=[state, model_choice],
                        outputs=[result_image, result_summary, inference_status, runtime_state],
                    )

                with gr.Tab("Benchmark", id="benchmark"):
                    gr.HTML(
                        "<div class='fdh-placeholder'><h2>Dataset benchmark workspace</h2>"
                        "<p>The benchmark view will reuse the saved ExperimentResult and leaderboard contracts. "
                        "It is intentionally kept separate while the single-image workflow is finalized.</p></div>"
                    )
    return app


def launch(**kwargs):
    kwargs.setdefault("css", CSS)
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
