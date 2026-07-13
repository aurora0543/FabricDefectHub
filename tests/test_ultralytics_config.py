"""Fast, framework-free tests for the Ultralytics config/preset layer.

These exercise the config-driven logic (variant resolution, hyperparameter
layering, YAML validation) without importing `ultralytics` or running any
training, so they stay in the default test suite.
"""

import pytest

from fabric_defect_hub.models.ultralytics.config import UltralyticsConfig
from fabric_defect_hub.models.ultralytics.presets import (
    default_train_kwargs,
    list_supported_variants,
    resolve_variant,
    variant_weights,
)


def test_variant_resolution_and_aliases():
    assert set(list_supported_variants()) == {"yolov8n", "yolov8s", "yolo11n"}
    assert resolve_variant("YOLOv8s") == "yolov8s"
    assert resolve_variant("v11n") == "yolo11n"
    assert resolve_variant("yolo11n.pt") == "yolo11n"
    with pytest.raises(KeyError):
        resolve_variant("yolov99x")


def test_variant_weights_pretrained_vs_scratch():
    assert variant_weights("yolov8n", pretrained=True) == "yolov8n.pt"
    assert variant_weights("yolov8n", pretrained=False) == "yolov8n.yaml"


def test_fabric_defaults_are_single_class():
    kwargs = default_train_kwargs("yolov8n")
    assert kwargs["single_cls"] is True
    assert kwargs["epochs"] > 0
    # per-variant override applies for the 's' model
    assert default_train_kwargs("yolov8s")["batch"] == 12


def test_config_from_dict_layers_and_resolves():
    cfg = UltralyticsConfig.from_dict(
        {
            "model": {"variant": "yolo11n", "pretrained": True},
            "data": {"data_yaml": "data.yaml"},
            "train": {"epochs": 5, "imgsz": 512, "extra": {"cache": True}},
            "checkpoint": {"project": "runs/x", "name": "exp1"},
        }
    )
    assert cfg.model.initial_weights() == "yolo11n.pt"
    resolved = cfg.resolved_train_kwargs()
    # config named field overrides the fabric preset
    assert resolved["epochs"] == 5
    assert resolved["imgsz"] == 512
    # fabric preset still fills unspecified knobs
    assert resolved["single_cls"] is True
    # extra is merged verbatim
    assert resolved["cache"] is True
    # checkpoint overrides included
    assert resolved["project"] == "runs/x"
    assert resolved["name"] == "exp1"


def test_config_rejects_unknown_and_conflicting_keys():
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"trian": {}})  # typo in top-level key
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"model": {"variant": "yolov8n", "bogus": 1}, "data": {"data_yaml": "d"}})
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict(
            {"data": {"data_yaml": "d", "dataset": "zju-leaper", "dataset_root": "/r"}}
        )
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"data": {"dataset": "zju-leaper"}})  # missing dataset_root


def test_scratch_init_uses_architecture_yaml():
    cfg = UltralyticsConfig.from_dict(
        {"model": {"variant": "yolov8n", "pretrained": False}, "data": {"data_yaml": "d.yaml"}}
    )
    assert cfg.model.initial_weights() == "yolov8n.yaml"
