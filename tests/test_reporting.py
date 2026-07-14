import csv

import pytest

from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.reporting import save_leaderboard


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
