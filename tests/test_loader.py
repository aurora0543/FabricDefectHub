"""Minimal closed-loop smoke test: fake dataset + fake model through
`run_experiment`, proving the core wiring (registry -> loader -> types)
works end-to-end before any real framework backend exists.
"""

from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, ModelInfo, RuntimeInfo, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter
from fabric_defect_hub.core.types import Prediction


@register_dataset("fake-fabric")
class FakeDataset(DatasetAdapter):
    name = "fake-fabric"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id="sample-0001",
                image_path=f"{self.root}/0001.jpg",
                task="detection",
                annotations=Annotations(boxes=[[120, 64, 238, 180]], labels=["broken_end"]),
            )
        ]


@register_model("fake-backend")
class FakeModel(ModelAdapter):
    name = "fake-model"
    backend = "fake-backend"

    def train(self, config):
        return Artifact(path="fake.pt", backend=self.backend)

    def predict(self, samples, artifact):
        return [Prediction(sample_id=s.id, boxes=[[121, 66, 236, 178]], labels=["broken_end"], scores=[0.9]) for s in samples]

    def export(self, artifact, target):
        return ExportedArtifact(path=f"fake.{target}", target=target)


class FakeEvaluator(Evaluator):
    task = "detection"

    def evaluate(self, samples, predictions):
        return {"map50": 1.0}


def test_end_to_end_loop():
    dataset = load_dataset("fake-fabric", root="data/fake")
    model = load_model("fake-backend", name="fake-model")

    result = run_experiment(
        experiment_id="exp-test-001",
        dataset=dataset,
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config={},
        evaluator=FakeEvaluator(),
    )

    assert result.experiment_id == "exp-test-001"
    assert result.metrics == {"map50": 1.0}
    assert result.dataset.name == "fake-fabric"
