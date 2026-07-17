#!/usr/bin/env python3
"""Train every canonical model (see `catalog.CANONICAL_MODELS`) and publish
each to the fixed path the frontend's `MODEL_CATALOG` reads from — the
single command this project's cloud training host is meant to run to
produce a complete, frontend-usable set of weights.

Runs each model's own example config unmodified except for `--variant`
(so each keeps its own tuned hyperparameters/sample counts), continues
past a failing model instead of aborting the whole batch, and prints a
pass/fail summary at the end. Re-run safely: `fdh train`'s own
`checkpoint.exist_ok=False` auto-increments run directories, and
`publish_artifact` simply overwrites that model's one published slot each
time it succeeds — so a partial/failed batch can just be re-run.

Usage:
    python tools/train_all_models.py                  # every canonical model
    python tools/train_all_models.py --only yolov8n PatchCore
    python tools/train_all_models.py --dry-run         # print the plan, train nothing
    python tools/train_all_models.py --mode test       # 8-image smoke run of every model first

Before running on a fresh host, see the "cloud setup" section in this
file's module docstring continuation below (or --help) for the network/
dataset preconditions this script assumes are already met.

--------------------------------------------------------------------------
Cloud setup checklist (do this once per host, not per run):

1. Install every backend extra:
       pip install -e ".[ultralytics,torchvision,anomalib]"
2. Stage all three datasets under data/ as symlinks (this project's
   portability convention — see training.DEFAULT_DATASET_ROOTS):
       data/ZJU-Leaper -> <real ZJU-Leaper location>
       data/RAW_FABRID -> <real RAW_FABRID location>
       data/MVTec AD   -> <real MVTec AD location>
   No environment variables needed if these symlinks exist; `fdh train`
   falls back to them automatically.
3. Anomalib downloads its backbones (wide_resnet50_2, resnet18, ...) from
   huggingface.co — unreachable from some hosts (confirmed: SeetaCloud/
   China network returned "Network is unreachable" against huggingface.co
   directly). Point at a mirror before running anomalib models:
       export HF_ENDPOINT=https://hf-mirror.com
4. Ultralytics downloads yolo*.pt from GitHub releases; torchvision
   downloads from download.pytorch.org. Both are usually reachable, but
   can be slow/flaky — this script does NOT retry failed downloads for
   you; re-running the script re-attempts the model that failed.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.catalog import CANONICAL_MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", nargs="+", metavar="KEY",
        help="train only these canonical model keys (see --list-keys), instead of all of them",
    )
    parser.add_argument("--list-keys", action="store_true", help="print every canonical model key and exit")
    parser.add_argument(
        "--mode", choices=("full", "medium", "few", "test"), default=None,
        help="shot mode override passed to every model (default: each config's own setting)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the training plan without running anything")
    args = parser.parse_args(argv)

    if args.list_keys:
        for model in CANONICAL_MODELS:
            print(f"{model.key:26s} backend={model.backend:12s} task={model.task}")
        return 0

    selected = CANONICAL_MODELS
    if args.only:
        wanted = {key.lower() for key in args.only}
        selected = [model for model in CANONICAL_MODELS if model.key.lower() in wanted]
        missing = wanted - {model.key.lower() for model in selected}
        if missing:
            raise SystemExit(f"unknown model key(s): {', '.join(sorted(missing))} (see --list-keys)")

    print(f"Training plan: {len(selected)} model(s)")
    for model in selected:
        print(f"  {model.key:26s} <- {model.config} --variant {model.variant}")
    if args.dry_run:
        return 0

    from fabric_defect_hub.training import DatasetOverrides, run_train

    results: list[tuple[str, bool, str]] = []
    for model in selected:
        print(f"\n{'=' * 70}\n>>> {model.key} ({model.backend} / {model.variant})\n{'=' * 70}")
        started = time.monotonic()
        try:
            run = run_train(
                model.config,
                variant=model.variant,
                overrides=DatasetOverrides(mode=args.mode) if args.mode else None,
            )
            elapsed = time.monotonic() - started
            published = run.published_path or "NOT PUBLISHED (backend/variant mismatch — check catalog.py)"
            print(f"OK  {model.key} in {elapsed:.0f}s -> {published}")
            results.append((model.key, True, published))
        except Exception as exc:  # noqa: BLE001 - keep training the rest of the batch
            elapsed = time.monotonic() - started
            print(f"FAIL {model.key} after {elapsed:.0f}s: {type(exc).__name__}: {exc}")
            results.append((model.key, False, f"{type(exc).__name__}: {exc}"))

    print(f"\n{'=' * 70}\nSummary ({sum(ok for _, ok, _ in results)}/{len(results)} succeeded)\n{'=' * 70}")
    for key, ok, detail in results:
        print(f"{'OK  ' if ok else 'FAIL'} {key:26s} {detail}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
