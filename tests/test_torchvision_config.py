"""Framework-free tests for torchvision config and pipeline orchestration."""

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.torchvision.config import TorchvisionConfig
from fabric_defect_hub.models.torchvision.pipeline import run_from_config


def _config():
    return TorchvisionConfig.from_dict(
        {
            "model": {"variant": "fasterrcnn_resnet50_fpn", "pretrained": False, "offline": True},
            "data": {
                "dataset": "zju-leaper",
                "dataset_root": "/dataset",
                "class_names": ["defect"],
                "train_selection": {"split": "train"},
                "val_selection": {"split": "test"},
            },
            "train": {"epochs": 1, "num_workers": 0},
            "export": {"enabled": True, "formats": ["exported_program"]},
        }
    )


class _FakeAdapter:
    def __init__(self, name):
        self.name = name

    def train(self, config):
        assert config["offline"] is True
        assert config["train_samples"]
        return Artifact("trained.pt", "torchvision", {"variant": self.name})

    def register_trained_model(self, artifact, registry_dir):
        return Artifact("registered.pt", "torchvision", artifact.metadata)

    def validate(self, samples, artifact, config):
        return {"map": 0.5}

    def export(self, artifact, target, config=None):
        return ExportedArtifact(f"model.{target}", target)


def test_config_resolves_safe_worker_and_offline_settings():
    config = _config()
    resolved = config.resolved_train_kwargs()
    assert resolved["num_workers"] == 0
    assert config.model.offline is True


def test_pipeline_orchestrates_without_importing_torch(monkeypatch):
    sample = Sample("sample", "image.jpg", "detection", Annotations())
    monkeypatch.setattr(
        "fabric_defect_hub.models.torchvision.pipeline._load_split_samples",
        lambda config, selection: [sample],
    )
    result = run_from_config(_config(), adapter_factory=_FakeAdapter)
    assert result.metrics == {"map": 0.5}
    assert result.registered_artifact.path == "registered.pt"
    assert result.exports[0].target == "exported_program"
