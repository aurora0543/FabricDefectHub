"""Fast, framework-free tests for the unified `predict` entry point
(`fabric_defect_hub.predict`): input validation and artifact/sample
resolution helpers, without importing any ML framework or running
inference.
"""

import pytest

import fabric_defect_hub.predict as predict_module
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.predict import (
    PredictInput,
    PredictRunResult,
    _load_samples,
    _resolve_weights_artifact,
    run_evaluate,
)


def test_resolve_weights_artifact_passes_through_bare_path_for_ultralytics():
    assert _resolve_weights_artifact("ultralytics", "artifacts/models/yolov8n_x.pt") == "artifacts/models/yolov8n_x.pt"


def test_resolve_weights_artifact_passes_through_bare_path_for_torchvision():
    assert _resolve_weights_artifact("torchvision", "artifacts/models/fasterrcnn_x.pt") == "artifacts/models/fasterrcnn_x.pt"


def test_resolve_weights_artifact_wraps_anomalib_checkpoint_as_trusted_artifact():
    artifact = _resolve_weights_artifact("anomalib", "artifacts/models/Patchcore.ckpt")
    assert artifact.path == "artifacts/models/Patchcore.ckpt"
    assert artifact.backend == "anomalib"
    assert artifact.metadata["trusted"] is True
    assert artifact.metadata["model_class"] == "Patchcore"


def test_resolve_weights_artifact_infers_model_class_from_stem():
    artifact = _resolve_weights_artifact("anomalib", "/some/dir/Padim.ckpt")
    assert artifact.metadata["model_class"] == "Padim"


def test_load_samples_rejects_both_images_and_dataset():
    source = PredictInput(images=["a.jpg"], dataset="zju-leaper")
    with pytest.raises(ValueError, match="either --image or --dataset"):
        _load_samples(source, "ultralytics")


def test_load_samples_rejects_neither_images_nor_dataset():
    source = PredictInput()
    with pytest.raises(ValueError, match="--image.*--dataset"):
        _load_samples(source, "ultralytics")


def test_load_samples_from_images_builds_one_sample_per_path():
    source = PredictInput(images=["/data/a.jpg", "/data/b.png"])
    samples = _load_samples(source, "ultralytics")
    assert [s.id for s in samples] == ["a", "b"]
    assert [s.image_path for s in samples] == ["/data/a.jpg", "/data/b.png"]
    assert all(s.task == "detection" for s in samples)


def test_load_samples_from_images_uses_anomaly_task_for_anomalib():
    source = PredictInput(images=["/data/a.jpg"])
    samples = _load_samples(source, "anomalib")
    assert samples[0].task == "anomaly"


def test_load_samples_from_dataset_raises_without_root_or_default(monkeypatch):
    source = PredictInput(dataset="some-unregistered-dataset")
    with pytest.raises(ValueError, match="no dataset_root"):
        _load_samples(source, "anomalib")


def _fake_run_predict_result() -> PredictRunResult:
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.1),
        Prediction(sample_id="b", anomaly_score=0.9),
    ]
    return PredictRunResult(backend="anomalib", variant="PatchCore", predictions=predictions, samples=samples)


def test_run_evaluate_scores_predictions_via_task_evaluator(monkeypatch):
    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: _fake_run_predict_result())

    result = run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"))

    assert result.backend == "anomalib"
    assert result.variant == "PatchCore"
    assert result.sample_count == 2
    assert "image_auroc" in result.metrics


def test_run_evaluate_honors_explicit_task_override(monkeypatch):
    captured = {}

    def fake_evaluator_for_task(task):
        captured["task"] = task
        from fabric_defect_hub.evaluation import AnomalyEvaluator

        return AnomalyEvaluator()

    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: _fake_run_predict_result())
    monkeypatch.setattr("fabric_defect_hub.evaluation.evaluator_for_task", fake_evaluator_for_task)

    run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"), task="anomaly")

    assert captured["task"] == "anomaly"


def test_run_evaluate_rejects_image_source_without_dataset():
    with pytest.raises(ValueError, match="--dataset"):
        run_evaluate("some_config", weights="x.ckpt", source=PredictInput(images=["a.jpg"]))


def test_run_evaluate_rejects_empty_sample_resolution(monkeypatch):
    empty_result = PredictRunResult(backend="anomalib", variant="PatchCore", predictions=[], samples=[])
    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: empty_result)

    with pytest.raises(ValueError, match="no samples resolved"):
        run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"))
