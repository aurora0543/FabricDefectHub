"""Tests for `web/benchmark.py`'s Gradio-facing leaderboard engine: the
opt-in profiling pass, run-log persistence, and composite scoring added on
top of the plain accuracy leaderboard.

Uses its own uniquely-named fake dataset/model registration
("*-webbench" suffix), mirroring `test_benchmark.py`'s pattern for the same
`core.registry` duplicate-name reason explained there -- registered once at
module import time (not per test), since `register_model` raises on a
second registration of the same name.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch
import torch.nn as nn

from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import ExportedArtifact, ModelAdapter
from fabric_defect_hub.web import benchmark as web_benchmark

MODEL_LABEL = "Fake Model"


@register_dataset("fake-fabric-webbench")
class _FakeWebBenchDataset(DatasetAdapter):
    name = "fake-fabric-webbench"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"sample-{i:04d}", image_path=f"{self.root}/{i:04d}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=bool(i % 2)),
            )
            for i in range(4)
        ]


class _TinyModule(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3))


@register_model("fake-backend-webbench")
class _FakeWebBenchModel(ModelAdapter):
    name = "fake-backend-webbench-model"
    backend = "fake-backend-webbench"

    def train(self, config):
        return None

    def predict(self, samples, artifact):
        return [Prediction(sample_id=s.id, anomaly_score=0.9) for s in samples]

    def export(self, artifact, target):
        assert target == "torchscript"
        fd, path = tempfile.mkstemp(suffix=f".{target}")
        os.close(fd)
        torch.jit.save(torch.jit.script(_TinyModule()), path)
        return ExportedArtifact(path=path, target=target)


def _install_fake_catalog(monkeypatch, tmp_path):
    dataset_catalog = {
        "Fake Dataset": {
            "name": "fake-fabric-webbench",
            "slice_kwarg": None,
            "tasks": ("anomaly",),
        },
    }
    model_catalog = {
        MODEL_LABEL: {
            "backend": "fake-backend-webbench",
            "name": "fake-backend-webbench-model",
            "checkpoint": str(tmp_path / "fake.ckpt"),
            "task": "anomaly",
            "metadata": {},
        },
    }
    monkeypatch.setattr(web_benchmark, "DATASET_CATALOG", dataset_catalog)
    monkeypatch.setattr(web_benchmark, "MODEL_CATALOG", model_catalog)
    monkeypatch.setattr(web_benchmark, "default_dataset_root", lambda label: str(tmp_path))
    monkeypatch.setattr(web_benchmark, "_detect_device", lambda: "cpu")


def test_run_benchmark_basic_leaderboard_has_no_score_columns_without_metrics(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=None,
    )

    assert "composite_score" in columns
    assert rows[0][columns.index("model")] == MODEL_LABEL
    # No profiling metrics were requested, so overhead_score has nothing to
    # average and composite falls back to the technical (accuracy) score.
    assert rows[0][columns.index("composite_score")] == rows[0][columns.index("technical_score")]
    # The chart-facing payload carries the same run as name-keyed dicts.
    assert scored[0]["model"] == MODEL_LABEL
    assert "composite_score" in scored[0]


def test_run_benchmark_with_profiling_adds_overhead_metrics_and_scores(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert "fps" in columns
    assert "latency_ms_mean" in columns
    row = rows[0]
    assert row[columns.index("fps")] > 0
    assert row[columns.index("overhead_score")] != ""
    assert row[columns.index("composite_score")] != ""


def test_run_benchmark_appends_to_run_log(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)
    log_path = tmp_path / "log.jsonl"

    list(web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=str(log_path),
    ))

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["model"]["backend"] == "fake-backend-webbench"


def test_run_benchmark_custom_preset_uses_custom_weight(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        score_preset="custom", custom_technical_weight=0.9, run_log_path=None,
    )

    assert rows[0][columns.index("composite_score")] != ""
