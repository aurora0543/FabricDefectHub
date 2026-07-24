from pathlib import Path

import pytest

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.models.dinomaly.adapter import DinomalyAdapter


def _sample(sample_id: str, anomalous: bool, mask_path: str | None = None) -> Sample:
    return Sample(
        id=sample_id,
        image_path="image.png",
        task="anomaly",
        annotations=Annotations(is_anomalous=anomalous, anomaly_mask=mask_path),
    )


def test_dinomaly_preflight_rejects_defects_without_masks():
    with pytest.raises(ValueError, match="missing masks for 1 sample.*'missing'"):
        DinomalyAdapter._validate_test_masks([_sample("missing", anomalous=True)])


def test_dinomaly_preflight_accepts_existing_masks(tmp_path: Path):
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    DinomalyAdapter._validate_test_masks([
        _sample("normal", anomalous=False),
        _sample("defect", anomalous=True, mask_path=str(mask)),
    ])
