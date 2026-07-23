import pytest

from fabric_defect_hub.cli import _infer_backend, _run_list, build_parser


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


def test_cli_parser_accepts_list():
    assert build_parser().parse_args(["list"]).command == "list"


def test_run_list_reports_every_registry_category():
    payload = _run_list()

    assert set(payload) == {"datasets", "model_backends", "evaluators", "profilers"}
    assert "zju-leaper" in payload["datasets"]
    # Subset, not exact-equality: the registries are shared global state for
    # the whole test session (see test_registry.py), so another test's fake
    # registration can legitimately still be present here.
    assert {"anomaly", "detection", "industrial", "segmentation"} <= set(payload["evaluators"])
    assert {"onnxruntime", "pytorch", "tensorrt"} <= set(payload["profilers"])
    # Not asserted: "available" ⊆ "known". That invariant genuinely holds in
    # real usage (only the 6 real backend modules' @register_model calls
    # ever populate the registry), but other test modules in this same
    # session register their own fake backends (e.g. "fake-backend" in
    # test_loader.py) straight into that same shared global registry, which
    # legitimately breaks the subset relationship here without meaning
    # anything is wrong.
    assert {"ultralytics", "torchvision", "anomalib", "dinomaly", "moeclip", "mambaad"} <= set(
        payload["model_backends"]["known"]
    )


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


def test_cli_parser_train_accepts_variant():
    args = build_parser().parse_args(["train", "model.yaml", "--variant", "yolov8s"])
    assert args.variant == "yolov8s"


def test_cli_parser_train_variant_defaults_to_none():
    args = build_parser().parse_args(["train", "model.yaml"])
    assert args.variant is None


def test_cli_parser_accepts_predict():
    args = build_parser().parse_args(
        ["predict", "model.yaml", "--weights", "artifacts/models/x.pt", "--image", "a.jpg"]
    )
    assert args.command == "predict"
    assert args.model == "model.yaml"
    assert args.weights == "artifacts/models/x.pt"
    assert args.images == ["a.jpg"]


def test_cli_parser_predict_image_is_repeatable():
    args = build_parser().parse_args(
        ["predict", "model.yaml", "--weights", "w.pt", "--image", "a.jpg", "--image", "b.jpg"]
    )
    assert args.images == ["a.jpg", "b.jpg"]


def test_cli_parser_predict_accepts_dataset_selection_and_variant():
    args = build_parser().parse_args(
        [
            "predict", "model.yaml", "--weights", "w.pt",
            "--dataset", "raw-fabric", "--dataset-root", "/data/raw",
            "--split", "train", "--num-samples", "5",
            "--variant", "yolov8s", "--backend", "ultralytics",
            "--output", "preds.json",
        ]
    )
    assert args.dataset == "raw-fabric"
    assert args.dataset_root == "/data/raw"
    assert args.split == "train"
    assert args.num_samples == 5
    assert args.variant == "yolov8s"
    assert args.backend == "ultralytics"
    assert args.output == "preds.json"


def test_cli_parser_predict_requires_weights():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "model.yaml"])
