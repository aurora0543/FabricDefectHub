"""Fast, framework-free tests for the unified `train` entry point
(`fabric_defect_hub.training`): backend keyword detection and dataset/shot
overlay logic, without importing any ML framework or running training.
"""

import pytest

from fabric_defect_hub.training import (
    TEST_SHOT_NUM_SAMPLES,
    DatasetOverrides,
    apply_dataset_overrides,
    find_model_configs,
    infer_backend,
    resolve_model_config,
)


@pytest.fixture
def config_dir(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "ultralytics_example.yaml").write_text("model:\n  variant: yolov8n\n")
    (directory / "anomalib_example.yaml").write_text("model:\n  name: PatchCore\n")
    return directory


def test_resolve_model_config_by_existing_path(config_dir):
    path = config_dir / "ultralytics_example.yaml"
    assert resolve_model_config(str(path), config_dir=config_dir) == path


def test_resolve_model_config_by_filename_stem(config_dir):
    assert resolve_model_config("ultralytics_example", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )
    # with the .yaml suffix too
    assert resolve_model_config("ultralytics_example.yaml", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )


def test_resolve_model_config_by_model_keyword(config_dir):
    assert resolve_model_config("yolov8n", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )
    # case-insensitive, matches model.name too
    assert resolve_model_config("patchcore", config_dir=config_dir) == (
        config_dir / "anomalib_example.yaml"
    )


def test_resolve_model_config_missing_raises_with_available_list(config_dir):
    with pytest.raises(FileNotFoundError, match="ultralytics_example"):
        resolve_model_config("nonexistent", config_dir=config_dir)


def test_resolve_model_config_ambiguous_keyword_raises(config_dir):
    (config_dir / "ultralytics_example2.yaml").write_text("model:\n  variant: yolov8n\n")
    with pytest.raises(ValueError, match="multiple configs"):
        resolve_model_config("yolov8n", config_dir=config_dir)


def test_find_model_configs_lists_yaml_files_sorted(config_dir):
    found = find_model_configs(config_dir)
    assert found == sorted(config_dir.glob("*.yaml"))


def test_find_model_configs_empty_dir_returns_empty(tmp_path):
    assert find_model_configs(tmp_path / "does-not-exist") == []


@pytest.mark.parametrize(
    ("raw", "backend"),
    [
        ({"model": {"variant": "yolov8n"}}, "ultralytics"),
        ({"model": {"variant": "fasterrcnn_resnet50_fpn"}}, "torchvision"),
        ({"model": {"name": "PatchCore"}}, "anomalib"),
        ({"backend": "torchvision", "model": {"variant": "yolov8n"}}, "torchvision"),
    ],
)
def test_infer_backend(raw, backend):
    assert infer_backend(raw) == backend


def test_infer_backend_requires_model_or_backend_key():
    with pytest.raises(ValueError):
        infer_backend({})


def test_infer_backend_rejects_unknown_explicit_backend():
    with pytest.raises(ValueError):
        infer_backend({"backend": "tensorflow"})


def test_empty_overrides_are_a_no_op():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"num_samples": 300}}}
    assert apply_dataset_overrides(raw, "ultralytics", DatasetOverrides()) is raw


def test_overrides_require_dataset_adapter_config():
    raw = {"data": {"data_yaml": "data.yaml"}}
    with pytest.raises(ValueError):
        apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="test"))


def test_dataset_and_root_override_clear_conflicting_keys():
    raw = {"data": {"data_yaml": "data.yaml"}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(dataset="zju-leaper", dataset_root="/data/zju")
    )
    assert out["data"]["dataset"] == "zju-leaper"
    assert out["data"]["dataset_root"] == "/data/zju"
    assert "data_yaml" not in out["data"]


def test_test_mode_forces_eight_samples_on_both_splits():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "val_selection": {"num_samples": 100},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="test"))
    assert out["data"]["train_selection"]["num_samples"] == TEST_SHOT_NUM_SAMPLES
    assert out["data"]["val_selection"]["num_samples"] == TEST_SHOT_NUM_SAMPLES


def test_full_mode_clears_num_samples():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "val_selection": {"num_samples": 100},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="full"))
    assert out["data"]["train_selection"]["num_samples"] is None
    assert out["data"]["val_selection"]["num_samples"] is None


def test_few_mode_leaves_configured_count_untouched():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"num_samples": 300}}}
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="few"))
    assert out["data"]["train_selection"]["num_samples"] == 300


def test_explicit_num_samples_wins_over_mode():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(mode="full", num_samples=42)
    )
    assert out["data"]["train_selection"]["num_samples"] == 42
    assert out["data"]["val_selection"]["num_samples"] == 42


def test_val_num_samples_overrides_only_the_second_split():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(num_samples=8, val_num_samples=20)
    )
    assert out["data"]["train_selection"]["num_samples"] == 8
    assert out["data"]["val_selection"]["num_samples"] == 20


def test_anomalib_train_split_is_forced_normal_only():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "test_selection": {"num_samples": 100, "use_defect": True, "defect_ratio": 0.3},
        }
    }
    out = apply_dataset_overrides(
        raw, "anomalib", DatasetOverrides(mode="test", use_defect=True, defect_ratio=0.7)
    )
    train_selection = out["data"]["train_selection"]
    assert train_selection["use_defect"] is False
    assert "defect_ratio" not in train_selection
    assert train_selection["num_samples"] == TEST_SHOT_NUM_SAMPLES
    # The test split still receives the requested defect mix.
    test_selection = out["data"]["test_selection"]
    assert test_selection["use_defect"] is True
    assert test_selection["defect_ratio"] == 0.7
    assert test_selection["num_samples"] == TEST_SHOT_NUM_SAMPLES


def test_pattern_category_and_seed_apply_to_both_splits():
    raw = {"data": {"dataset": "mvtec-ad", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "torchvision", DatasetOverrides(category="bottle", seed=7)
    )
    for key in ("train_selection", "val_selection"):
        assert out["data"][key]["category"] == "bottle"
        assert out["data"][key]["seed"] == 7
