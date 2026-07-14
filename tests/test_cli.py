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
