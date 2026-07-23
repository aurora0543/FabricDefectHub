import csv
import json

import pytest

from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.reporting import append_run_log, save_leaderboard


def _result():
    return ExperimentResult(
        experiment_id="exp-1",
        model=ModelInfo(name="model", backend="backend", task="detection"),
        dataset=DatasetInfo(name="dataset", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="onnxruntime", precision="fp32", input_size=(640, 640)),
        metrics={"map50": 0.8, "fps": 100.0},
    )


def test_save_leaderboard_csv(tmp_path):
    path = save_leaderboard([_result()], tmp_path / "board.csv")
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["experiment_id"] == "exp-1"
    assert rows[0]["map50"] == "0.8"


def test_save_leaderboard_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="must end"):
        save_leaderboard([_result()], tmp_path / "board.json")


def test_append_run_log_writes_one_json_line_with_provenance(tmp_path):
    path = append_run_log(_result(), tmp_path / "runs_log.jsonl")

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["experiment_id"] == "exp-1"
    assert row["metrics"] == {"map50": 0.8, "fps": 100.0}
    assert set(row["provenance"]) == {"timestamp_utc", "git_commit", "hostname"}


def test_append_run_log_appends_across_different_metric_shapes(tmp_path):
    detection_result = _result()
    anomaly_result = ExperimentResult(
        experiment_id="exp-2",
        model=ModelInfo(name="patchcore", backend="anomalib", task="anomaly"),
        dataset=DatasetInfo(name="dataset", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="pytorch", precision="fp32", input_size=(256, 256)),
        metrics={"image_auroc": 0.95, "pixel_aupro": 0.9, "iap": 0.7},
    )

    log_path = tmp_path / "runs_log.jsonl"
    append_run_log(detection_result, log_path)
    append_run_log(anomaly_result, log_path)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["metrics"] == {"map50": 0.8, "fps": 100.0}
    assert rows[1]["metrics"] == {"image_auroc": 0.95, "pixel_aupro": 0.9, "iap": 0.7}
