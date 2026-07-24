"""Fast, framework-free tests for the Anomalib config/preset layer.

These exercise the config-driven logic (model-name resolution,
hyperparameter layering, YAML validation) without importing `anomalib` or
running any training — `models/anomalib/presets.py` only imports
`anomalib` lazily inside functions that actually need it (`resolve_model_
class`), so this stays in the default test suite. Mirrors
`tests/test_ultralytics_config.py`'s structure.
"""

import pytest

from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.config import AnomalibConfig
from fabric_defect_hub.models.anomalib.presets import (
    default_model_kwargs,
    list_supported_models,
    resolve_model_class_name,
)


def test_model_name_resolution_and_aliases():
    assert set(list_supported_models()) == {
        "Patchcore", "Padim", "ReverseDistillation", "EfficientAd", "Supersimplenet", "WinClip",
    }
    assert resolve_model_class_name("PatchCore") == "Patchcore"
    assert resolve_model_class_name("rd4ad") == "ReverseDistillation"
    assert resolve_model_class_name("EfficientAD") == "EfficientAd"
    assert resolve_model_class_name("WinCLIP") == "WinClip"
    assert resolve_model_class_name("Patchcore") == "Patchcore"  # literal class name passes through
    with pytest.raises(KeyError):
        resolve_model_class_name("NotAModel")


def test_efficientad_requires_imagenet_dir_with_no_fabric_default():
    kwargs = default_model_kwargs("EfficientAD")
    assert kwargs["imagenet_dir"] is None


def test_supersimplenet_defaults_to_supervised():
    # ZJU-Leaper ships real defect masks, so the fabric preset should use
    # them (supervised=True) rather than anomalib's synthetic-anomaly default.
    assert default_model_kwargs("SuperSimpleNet")["supervised"] is True


def test_winclip_defaults_to_zero_shot():
    # WinCLIP is CLIP-based; k_shot=0 keeps it a pure zero-shot baseline that
    # consumes no fabric training images, and class_name gives its prompt
    # ensemble a domain-appropriate noun.
    kwargs = default_model_kwargs("WinCLIP")
    assert kwargs["k_shot"] == 0
    assert kwargs["class_name"] == "fabric"


def test_winclip_zero_shot_skips_data_dependent_fit():
    adapter = AnomalibAdapter(name="WinCLIP")
    assert adapter._is_zero_shot_winclip({"k_shot": 0}) is True
    assert adapter._is_zero_shot_winclip({"k_shot": 5}) is False


def test_config_from_dict_layers_and_resolves():
    cfg = AnomalibConfig.from_dict(
        {
            "model": {"name": "PatchCore"},
            "data": {
                "dataset": "zju-leaper",
                "dataset_root": "/data/zju-leaper",
                "train_selection": {"use_defect": False},
                "test_selection": {"use_defect": True},
            },
            "train": {
                "model_kwargs": {"coreset_sampling_ratio": 0.25},
                "engine_kwargs": {"max_epochs": 3},
            },
            "checkpoint": {"default_root_dir": "results/x", "registry_dir": "artifacts/x"},
        }
    )

    resolved_model = cfg.resolved_model_kwargs()
    # config override wins
    assert resolved_model["coreset_sampling_ratio"] == 0.25
    # fabric preset still fills unspecified knobs
    assert resolved_model["backbone"] == "wide_resnet50_2"
    assert resolved_model["num_neighbors"] == 9

    resolved_engine = cfg.resolved_engine_kwargs()
    assert resolved_engine["max_epochs"] == 3
    # checkpoint.default_root_dir is merged in automatically
    assert resolved_engine["default_root_dir"] == "results/x"


def test_config_rejects_unknown_and_conflicting_keys():
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"modle": {}})  # typo in top-level key
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict(
            {"model": {"name": "PatchCore", "bogus": 1}, "data": {"datamodule_kwargs": {"root": "/x"}}}
        )
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict(
            {"data": {"datamodule_kwargs": {"root": "/x"}, "dataset": "zju-leaper", "dataset_root": "/r"}}
        )
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"data": {"dataset": "zju-leaper"}})  # missing dataset_root
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"data": {}})  # neither datamodule_kwargs nor dataset


def test_config_rejects_unknown_model_name():
    with pytest.raises(KeyError):
        AnomalibConfig.from_dict(
            {"model": {"name": "NotAModel"}, "data": {"datamodule_kwargs": {"root": "/x"}}}
        )


def test_datamodule_kwargs_mode_does_not_use_adapter():
    cfg = AnomalibConfig.from_dict({"data": {"datamodule_kwargs": {"root": "/x"}}})
    assert cfg.data.uses_adapter() is False


def test_dataset_mode_uses_adapter():
    cfg = AnomalibConfig.from_dict(
        {"data": {"dataset": "zju-leaper", "dataset_root": "/r"}}
    )
    assert cfg.data.uses_adapter() is True


def test_default_train_spec_has_zero_workers():
    # See config.py's TrainSpec docstring: staged dirs are transient
    # symlinks, so num_workers=0 avoids a shutdown race, unlike the
    # torchvision backend's default of 2.
    cfg = AnomalibConfig.from_dict({"data": {"datamodule_kwargs": {"root": "/x"}}})
    assert cfg.train.num_workers == 0
