"""Fast, framework-free tests for the unified `predict` entry point
(`fabric_defect_hub.predict`): input validation and artifact/sample
resolution helpers, without importing any ML framework or running
inference.
"""

import pytest

from fabric_defect_hub.predict import (
    PredictInput,
    _load_samples,
    _resolve_weights_artifact,
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
