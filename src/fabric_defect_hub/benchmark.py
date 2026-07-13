"""Cross-backend benchmark orchestration: run the same (or different)
dataset(s) through several models — any mix of Ultralytics/torchvision/
Anomalib backends — and produce a sorted leaderboard, closing the README's
Phase 2 "完成统一训练、预测、评测和制品管理流程" item.

`loader.run_experiment` already runs one (dataset, model, evaluator) triple
end to end and can persist its `ExperimentResult`. What was still missing
is the layer above that: describing *several* runs declaratively and
getting back a comparable, sorted list — this is what turns individual
experiments into an actual benchmark. `evaluation/*.py` being backend-
agnostic (see their module docstrings) is what makes a fair leaderboard
possible here: two rows produced by different frameworks were still scored
by the exact same metric code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.types import ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment


@dataclass
class BenchmarkRun:
    """One row of a benchmark: which dataset, which model, how to train/
    evaluate it. Either pass a ready-built `dataset`, or `dataset_name` +
    `dataset_root` (+ `dataset_kwargs`) to have it loaded for you.
    """

    experiment_id: str
    model_backend: str
    model_name: str
    model_info: ModelInfo
    runtime: RuntimeInfo
    dataset: DatasetAdapter | None = None
    dataset_name: str | None = None
    dataset_root: str | None = None
    dataset_split: str = "test"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    train_config: dict[str, Any] | None = None
    evaluator: Evaluator | None = None

    def resolve_dataset(self) -> DatasetAdapter:
        if self.dataset is not None:
            return self.dataset
        if self.dataset_name is None or self.dataset_root is None:
            raise ValueError(
                f"BenchmarkRun {self.experiment_id!r}: provide either 'dataset', or "
                "'dataset_name' + 'dataset_root'."
            )
        return load_dataset(
            self.dataset_name, root=self.dataset_root, split=self.dataset_split, **self.dataset_kwargs
        )


def run_benchmark(runs: list[BenchmarkRun], output_dir: str | None = None) -> list[ExperimentResult]:
    """Execute every `BenchmarkRun` in order and return their `ExperimentResult`s.

    A failure in one run is not swallowed — if you need partial results
    from a long benchmark despite one row failing, wrap individual runs
    yourself; silently continuing past a broken run would make the
    leaderboard's comparability claim (identical metric code across rows)
    untrustworthy.
    """

    results: list[ExperimentResult] = []
    for run in runs:
        dataset = run.resolve_dataset()
        model = load_model(run.model_backend, run.model_name, **run.model_kwargs)
        result = run_experiment(
            experiment_id=run.experiment_id,
            dataset=dataset,
            model=model,
            model_info=run.model_info,
            runtime=run.runtime,
            train_config=run.train_config,
            evaluator=run.evaluator,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def leaderboard(
    results: list[ExperimentResult], metric: str, descending: bool = True
) -> list[ExperimentResult]:
    """Sort `results` by `metric` (e.g. 'map50', 'image_auroc'). Results that
    didn't compute `metric` (different task, evaluator not run, ...) are
    dropped rather than sorted arbitrarily — mixing a missing-metric
    sentinel into a ranked list would silently misrank real runs.
    """

    ranked = [r for r in results if metric in r.metrics]
    return sorted(ranked, key=lambda r: r.metrics[metric], reverse=descending)
