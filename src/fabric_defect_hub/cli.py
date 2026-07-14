"""Command-line entry point for config-driven model and benchmark runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdh", description="FabricDefectHub runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a model or benchmark YAML config")
    run_parser.add_argument("config", help="path to YAML config")
    run_parser.add_argument(
        "--backend", choices=("ultralytics", "torchvision", "anomalib"),
        help="model backend; inferred from the config when omitted",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark YAML config")
    benchmark_parser.add_argument("config", help="path to benchmark YAML config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "benchmark":
            payload = _run_benchmark(args.config)
        else:
            payload = _run_config(args.config, args.backend)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"fdh: {exc}") from exc
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_config(path: str, backend: str | None) -> Any:
    import yaml

    with open(path) as file:
        raw = yaml.safe_load(file) or {}
    if isinstance(raw, dict) and "runs" in raw:
        return _run_benchmark(path)
    selected = backend or _infer_backend(raw)
    if selected == "ultralytics":
        from fabric_defect_hub.models.ultralytics.pipeline import run_from_yaml
    elif selected == "torchvision":
        from fabric_defect_hub.models.torchvision.pipeline import run_from_yaml
    else:
        from fabric_defect_hub.models.anomalib.pipeline import run_from_yaml
    result = run_from_yaml(path)
    return {
        "backend": selected,
        "metrics": result.metrics,
        "trained_artifact": _artifact_dict(result.trained_artifact),
        "registered_artifact": _artifact_dict(result.registered_artifact),
        "exports": [asdict(artifact) for artifact in result.exports],
    }


def _run_benchmark(path: str) -> list[dict[str, Any]]:
    from fabric_defect_hub.benchmark import BenchmarkConfig
    from fabric_defect_hub.core.serialization import experiment_result_to_dict

    config = BenchmarkConfig.from_yaml(path)
    return [experiment_result_to_dict(result) for result in config.run()]


def _infer_backend(raw: object) -> str:
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("cannot infer backend: config requires a 'model' mapping")
    model = raw["model"]
    if "name" in model:
        return "anomalib"
    variant = str(model.get("variant", "")).lower()
    if variant.startswith("yolo"):
        return "ultralytics"
    if variant.startswith(("fasterrcnn", "maskrcnn")):
        return "torchvision"
    raise ValueError("cannot infer backend; pass --backend explicitly")


def _artifact_dict(artifact) -> dict[str, Any] | None:
    return asdict(artifact) if artifact is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
