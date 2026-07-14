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

    train_parser = subparsers.add_parser(
        "train",
        help="unified training entry point: pick a model config, optionally override its dataset + shot mode",
    )
    train_parser.add_argument(
        "model", nargs="?",
        help=(
            "a model config: a path (configs/models/ultralytics_example.yaml), a filename stem "
            "under --config-dir (ultralytics_example), or a model keyword matched against every "
            "config's model.variant/model.name there (yolov8n, patchcore, ...). "
            "Omit with --list to see what's available."
        ),
    )
    train_parser.add_argument(
        "--config-dir", default="configs/models",
        help="directory searched when 'model' is a filename stem or keyword (default: configs/models)",
    )
    train_parser.add_argument(
        "--list", action="store_true", help="list resolvable model configs under --config-dir and exit"
    )
    train_parser.add_argument(
        "--backend", choices=("ultralytics", "torchvision", "anomalib"),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    train_parser.add_argument(
        "--dataset", help="registered dataset name (e.g. zju-leaper, raw-fabric, mvtec-ad); overrides data.dataset"
    )
    train_parser.add_argument("--dataset-root", help="dataset root path; overrides data.dataset_root")
    train_parser.add_argument(
        "--mode", choices=("full", "few", "test"), default=None,
        help=(
            "shot mode: full=use every sample (num_samples=null); "
            "few=leave the config's own declared few-shot count untouched; "
            "test=quick 8-image smoke run of the whole pipeline"
        ),
    )
    train_parser.add_argument(
        "--num-samples", type=int,
        help="explicit sample-count override for the train split (and val/test split unless --val-num-samples is given)",
    )
    train_parser.add_argument(
        "--val-num-samples", type=int, help="explicit sample-count override for the val/test split only"
    )
    defect_group = train_parser.add_mutually_exclusive_group()
    defect_group.add_argument("--use-defect", dest="use_defect", action="store_true", default=None)
    defect_group.add_argument("--no-use-defect", dest="use_defect", action="store_false")
    train_parser.add_argument("--defect-ratio", type=float, help="fraction of the loaded split that is defective")
    train_parser.add_argument("--pattern", help="ZJU-Leaper pattern/group filter override")
    train_parser.add_argument("--category", help="MVTec-AD category filter override")
    train_parser.add_argument("--seed", type=int, help="subsampling RNG seed override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "benchmark":
            payload = _run_benchmark(args.config)
        elif args.command == "train":
            payload = _run_train(args)
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


def _run_train(args: argparse.Namespace) -> Any:
    from fabric_defect_hub.training import DatasetOverrides, find_model_configs, run_train

    if args.list:
        return {
            "config_dir": args.config_dir,
            "configs": [str(path) for path in find_model_configs(args.config_dir)],
        }
    if not args.model:
        raise ValueError("'model' is required unless --list is given")

    overrides = DatasetOverrides(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        mode=args.mode,
        num_samples=args.num_samples,
        val_num_samples=args.val_num_samples,
        use_defect=args.use_defect,
        defect_ratio=args.defect_ratio,
        pattern=args.pattern,
        category=args.category,
        seed=args.seed,
    )
    run = run_train(args.model, backend=args.backend, overrides=overrides, config_dir=args.config_dir)
    result = run.result
    return {
        "backend": run.backend,
        "metrics": result.metrics,
        "trained_artifact": _artifact_dict(result.trained_artifact),
        "registered_artifact": _artifact_dict(result.registered_artifact),
        "exports": [asdict(artifact) for artifact in result.exports],
    }


def _infer_backend(raw: object) -> str:
    from fabric_defect_hub.training import infer_backend

    return infer_backend(raw)  # type: ignore[arg-type]


def _artifact_dict(artifact) -> dict[str, Any] | None:
    return asdict(artifact) if artifact is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
