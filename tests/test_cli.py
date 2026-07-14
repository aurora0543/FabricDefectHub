import pytest

from fabric_defect_hub.cli import _infer_backend, build_parser


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ({"variant": "yolov8n"}, "ultralytics"),
        ({"variant": "fasterrcnn_resnet50_fpn"}, "torchvision"),
        ({"name": "PatchCore"}, "anomalib"),
    ],
)
def test_infer_backend(model, backend):
    assert _infer_backend({"model": model}) == backend


def test_cli_parser_accepts_run_and_benchmark():
    assert build_parser().parse_args(["run", "model.yaml"]).command == "run"
    assert build_parser().parse_args(["benchmark", "benchmark.yaml"]).command == "benchmark"


def test_cli_parser_accepts_train():
    args = build_parser().parse_args(["train", "model.yaml"])
    assert args.command == "train"
    assert args.model == "model.yaml"
    assert args.config_dir == "configs/models"
    assert args.list is False
    assert args.backend is None
    assert args.mode is None
    assert args.use_defect is None


def test_cli_parser_train_accepts_bare_name_and_config_dir():
    args = build_parser().parse_args(["train", "yolov8n", "--config-dir", "/other/dir"])
    assert args.model == "yolov8n"
    assert args.config_dir == "/other/dir"


def test_cli_parser_train_list_allows_omitting_model():
    args = build_parser().parse_args(["train", "--list"])
    assert args.model is None
    assert args.list is True


def test_cli_parser_train_accepts_shot_mode_and_dataset_overrides():
    args = build_parser().parse_args(
        [
            "train",
            "model.yaml",
            "--config-dir",
            "configs/models",
            "--backend",
            "anomalib",
            "--dataset",
            "zju-leaper",
            "--dataset-root",
            "/data/zju",
            "--mode",
            "test",
            "--num-samples",
            "8",
            "--val-num-samples",
            "20",
            "--use-defect",
            "--defect-ratio",
            "0.5",
            "--pattern",
            "pattern1",
            "--category",
            "bottle",
            "--seed",
            "3",
        ]
    )
    assert args.backend == "anomalib"
    assert args.dataset == "zju-leaper"
    assert args.dataset_root == "/data/zju"
    assert args.mode == "test"
    assert args.num_samples == 8
    assert args.val_num_samples == 20
    assert args.use_defect is True
    assert args.defect_ratio == 0.5
    assert args.pattern == "pattern1"
    assert args.category == "bottle"
    assert args.seed == 3


def test_cli_parser_train_use_defect_mutually_exclusive():
    args = build_parser().parse_args(["train", "model.yaml", "--no-use-defect"])
    assert args.use_defect is False
