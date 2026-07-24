"""Command-line entry point for config-driven model and benchmark runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any


def _model_backend_choices() -> tuple[str, ...]:
    from fabric_defect_hub.loader import list_model_backends

    return tuple(list_model_backends())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdh", description="FabricDefectHub runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a model or benchmark YAML config")
    run_parser.add_argument("config", help="path to YAML config")
    run_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="model backend; inferred from the config when omitted",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark YAML config")
    benchmark_parser.add_argument("config", help="path to benchmark YAML config")

    subparsers.add_parser(
        "list",
        help="list every registered dataset/model-backend/evaluator/profiler "
        "(self-describing platform catalog, not hardcoded documentation)",
    )

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
        "--backend", choices=_model_backend_choices(),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    train_parser.add_argument(
        "--variant",
        help=(
            "override which model in the backend's family gets trained: written to "
            "model.variant for ultralytics/torchvision (e.g. yolov8n, yolov8s, yolo11n, "
            "fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn) or model.name for anomalib/dinomaly/moeclip/mambaad "
            "(e.g. PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet, dinov2reg_vit_base_14); "
            "lets one config file train any model its backend supports instead of needing one "
            "YAML per model"
        ),
    )
    train_parser.add_argument(
        "--dataset", help="registered dataset name (e.g. zju-leaper, raw-fabric, mvtec-ad); overrides data.dataset"
    )
    train_parser.add_argument("--dataset-root", help="dataset root path; overrides data.dataset_root")
    train_parser.add_argument(
        "--test-dataset",
        help="zero-shot backends (moeclip) only: the dataset to *evaluate* on, when it differs "
        "from the training corpus (overrides data.test_dataset)",
    )
    train_parser.add_argument(
        "--test-dataset-root", help="root path for --test-dataset; falls back to data/<Dataset>"
    )
    train_parser.add_argument(
        "--mode", choices=("full", "medium", "few", "test"), default=None,
        help=(
            "shot mode: full=use every sample, every ZJU-Leaper pattern (num_samples=null); "
            "medium=every ZJU-Leaper pattern but capped per-pattern (150 train / 50 val each); "
            "few=leave the config's own declared few-shot count and pattern subset untouched; "
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
    train_parser.add_argument(
        "--set", dest="set_overrides", action="append", default=[], metavar="path.to.key=value",
        help=(
            "tuning window: override any config value by dotted path, repeatable "
            "(e.g. --set train.model_kwargs.coreset_sampling_ratio=0.05 --set train.model_kwargs.lr=0.0005). "
            "Value is YAML-parsed (numbers/booleans/lists work unquoted); wins over the config file, "
            "the recipe/profile defaults, and every other --* override"
        ),
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help=(
            "unified inference entry point: pick a model config (same resolution as 'train'), "
            "load a previously trained artifact, and run it over images or a dataset selection"
        ),
    )
    predict_parser.add_argument(
        "model",
        help=(
            "a model config: a path, a filename stem under --config-dir, or a model keyword "
            "(yolov8n, patchcore, ...) — resolved exactly like 'fdh train MODEL'"
        ),
    )
    predict_parser.add_argument(
        "--weights", required=True,
        help=(
            "path to a trained/registered artifact to load, e.g. 'fdh train's "
            "registered_artifact.path output (artifacts/models/<name>.pt or .ckpt)"
        ),
    )
    predict_parser.add_argument(
        "--config-dir", default="configs/models",
        help="directory searched when 'model' is a filename stem or keyword (default: configs/models)",
    )
    predict_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    predict_parser.add_argument(
        "--variant",
        help="override which model in the backend's family runs inference (see 'train --variant'); "
        "must match what --weights was actually trained as",
    )
    predict_parser.add_argument(
        "--image", action="append", dest="images", default=[],
        help="path to an image to run inference on; repeatable. Mutually exclusive with --dataset",
    )
    predict_parser.add_argument(
        "--dataset", help="registered dataset name (e.g. zju-leaper, raw-fabric, mvtec-ad) to draw samples from"
    )
    predict_parser.add_argument("--dataset-root", help="dataset root path; falls back to data/<Dataset> if omitted")
    subparsers.add_parser(
        "recipes",
        help="list every model config profile",
    )

    subparsers.add_parser(
        "doctor",
        help=(
            "the availability decision tree: for every model backend, report whether it's "
            "trainable right now on this machine (framework installed + a matching dataset "
            "actually staged under data/<Dataset>), which dataset would be picked, and why — "
            "runnable backends first"
        ),
    )

    export_latex_parser = subparsers.add_parser(
        "export-latex",
        help="export benchmark results to IEEE/CVPR paper-grade LaTeX table code",
    )
    export_latex_parser.add_argument("results_json", help="path to benchmark results JSON file")
    export_latex_parser.add_argument("--output", help="optional output .tex file path")

    predict_parser.add_argument("--split", default="test", choices=("train", "test"), help="dataset split to draw from")
    predict_parser.add_argument("--num-samples", type=int, help="how many dataset samples to run inference on")
    predict_parser.add_argument("--pattern", help="ZJU-Leaper pattern/group filter")
    predict_parser.add_argument("--category", help="MVTec-AD category filter")
    predict_parser.add_argument("--seed", type=int, default=0, help="subsampling RNG seed")
    predict_parser.add_argument(
        "--output", help="write predictions as JSON to this path (in addition to stdout)"
    )
    predict_parser.add_argument(
        "--output-dir",
        help="anomalib/dinomaly/moeclip/mambaad only: also persist each sample's pixel-level anomaly map (.npy) under this directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "benchmark":
            payload = _run_benchmark(args.config)
        elif args.command == "list":
            payload = _run_list()
        elif args.command == "recipes":
            payload = _run_recipes()
        elif args.command == "doctor":
            payload = _run_doctor()
        elif args.command == "export-latex":
            payload = _run_export_latex(args.results_json, args.output)
        elif args.command == "train":
            payload = _run_train(args)
        elif args.command == "predict":
            payload = _run_predict(args)
        else:
            payload = _run_config(args.config, args.backend)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"fdh: {exc}") from exc
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_recipes() -> dict[str, Any]:
    from fabric_defect_hub.core.registry import list_recipes, get_recipe
    import fabric_defect_hub.recipes  # Ensure registration

    summary = {}
    for recipe_id in list_recipes():
        recipe = get_recipe(recipe_id)
        summary[recipe_id] = recipe.get_recipe_summary()
    return summary


def _run_doctor() -> dict[str, Any]:
    """The availability decision tree, surfaced: for every known model
    backend, whether it's trainable right now on *this* machine, which
    dataset would actually be used, and why — runnable backends first, so
    "what can I train given what's staged here" is one command instead of
    reading tracebacks from a training run that got partway through.
    """

    import importlib

    from fabric_defect_hub.core.availability import backend_is_importable
    from fabric_defect_hub.core.decision import decide_dataset
    from fabric_defect_hub.loader import list_model_backends
    from fabric_defect_hub.training import DEFAULT_DATASET_ROOTS, _BACKEND_TRAINABLE_DATASETS

    importlib.import_module("fabric_defect_hub.datasets")  # populate dataset_capabilities-derived roles

    report: dict[str, dict[str, Any]] = {}
    for backend in list_model_backends():
        framework_installed = backend_is_importable(backend)
        entry: dict[str, Any] = {"framework_installed": framework_installed}

        if backend in _BACKEND_TRAINABLE_DATASETS:
            allowed_set, kind = _BACKEND_TRAINABLE_DATASETS[backend]
            decision = decide_dataset(None, allowed_set, root_map=DEFAULT_DATASET_ROOTS)
            entry["dataset_kind"] = kind
            entry["dataset"] = decision.dataset
            entry["trainable_now"] = framework_installed and decision.runnable
            entry["reason"] = decision.reason if framework_installed else "framework not installed"
        else:
            # Detection backends (ultralytics/torchvision) train on
            # whatever detection dataset the caller's config names; there
            # is no fixed allowed-set to pick a default from here.
            entry["trainable_now"] = framework_installed
            entry["reason"] = "framework installed" if framework_installed else "framework not installed"

        report[backend] = entry

    ordered = sorted(report, key=lambda b: (not report[b]["trainable_now"], b))
    return {"backends": {name: report[name] for name in ordered}}


def _run_export_latex(json_path: str, output_path: str | None = None) -> str:
    from fabric_defect_hub.reporting.latex_generator import generate_latex_table

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_list = data if isinstance(data, list) else data.get("leaderboard", [])
    latex_code = generate_latex_table(results_list)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
    return latex_code



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
    elif selected == "dinomaly":
        from fabric_defect_hub.models.dinomaly.pipeline import run_from_yaml
    elif selected == "moeclip":
        from fabric_defect_hub.models.moeclip.pipeline import run_from_yaml
    elif selected == "mambaad":
        from fabric_defect_hub.models.mambaad.pipeline import run_from_yaml
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


def _run_list() -> Any:
    import importlib

    from fabric_defect_hub.core import registry
    from fabric_defect_hub.loader import import_all_model_backends, list_model_backends

    importlib.import_module("fabric_defect_hub.datasets")
    importlib.import_module("fabric_defect_hub.evaluation")
    importlib.import_module("fabric_defect_hub.profiling")
    import_all_model_backends()

    return {
        "datasets": registry.list_datasets(),
        "model_backends": {"known": list_model_backends(), "available": registry.list_models()},
        "evaluators": registry.list_evaluators(),
        "profilers": registry.list_profilers(),
    }


def _parse_set_overrides(raw_items: list[str]) -> dict[str, Any]:
    """Parse repeated `--set path.to.key=value` CLI args into the dotted-path
    dict `training.apply_raw_overrides` expects. `value` is YAML-parsed so
    `--set train.epochs=50` and `--set train.model_kwargs.pre_trained=false`
    both come through as the right Python type, not the literal string.
    """

    import yaml

    overrides: dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"--set expects path.to.key=value, got {item!r}")
        path, _, value = item.partition("=")
        path = path.strip()
        if not path:
            raise ValueError(f"--set expects a non-empty dotted path, got {item!r}")
        overrides[path] = yaml.safe_load(value)
    return overrides


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
        test_dataset=args.test_dataset,
        test_dataset_root=args.test_dataset_root,
        mode=args.mode,
        num_samples=args.num_samples,
        val_num_samples=args.val_num_samples,
        use_defect=args.use_defect,
        defect_ratio=args.defect_ratio,
        pattern=args.pattern,
        category=args.category,
        seed=args.seed,
    )
    set_overrides = _parse_set_overrides(args.set_overrides)
    run = run_train(
        args.model,
        backend=args.backend,
        overrides=overrides,
        config_dir=args.config_dir,
        variant=args.variant,
        set_overrides=set_overrides,
    )
    result = run.result
    return {
        "backend": run.backend,
        "metrics": result.metrics,
        "trained_artifact": _artifact_dict(result.trained_artifact),
        "registered_artifact": _artifact_dict(result.registered_artifact),
        "published_path": run.published_path,
        "exports": [asdict(artifact) for artifact in result.exports],
    }


def _run_predict(args: argparse.Namespace) -> Any:
    from fabric_defect_hub.predict import PredictInput, run_predict

    source = PredictInput(
        images=args.images,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        split=args.split,
        num_samples=args.num_samples,
        pattern=args.pattern,
        category=args.category,
        seed=args.seed,
    )
    run = run_predict(
        args.model,
        weights=args.weights,
        source=source,
        backend=args.backend,
        variant=args.variant,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    predictions = [asdict(prediction) for prediction in run.predictions]
    if args.output:
        from fabric_defect_hub.core.serialization import save_predictions

        save_predictions(run.predictions, args.output)
    return {
        "backend": run.backend,
        "variant": run.variant,
        "num_predictions": len(predictions),
        "predictions": predictions,
    }


def _infer_backend(raw: object) -> str:
    from fabric_defect_hub.training import infer_backend

    return infer_backend(raw)  # type: ignore[arg-type]


def _artifact_dict(artifact) -> dict[str, Any] | None:
    return asdict(artifact) if artifact is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
