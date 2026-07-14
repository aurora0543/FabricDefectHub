from pathlib import Path

import pytest

pytest.importorskip("torch")

from fabric_defect_hub.models.torchvision.adapter import _validate_checkpoint


def _checkpoint():
    return {
        "variant": "fasterrcnn_resnet50_fpn",
        "class_map": {"defect": 1},
        "state_dict": {"backbone.body.conv1.weight": object()},
    }


def test_checkpoint_validation_accepts_adapter_shape():
    class_map, variant, state_dict = _validate_checkpoint(_checkpoint(), Path("model.pt"), "fallback")
    assert class_map == {"defect": 1}
    assert variant == "fasterrcnn_resnet50_fpn"
    assert state_dict


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({}, "missing required keys"),
        ({"state_dict": {}, "class_map": {}}, "invalid non-empty"),
        ({"state_dict": {"x": 1}, "class_map": {"defect": 0}}, "positive integer"),
        ({"state_dict": {"x": 1}, "class_map": {"defect": 1}, "variant": "not-real"}, "unsupported"),
    ],
)
def test_checkpoint_validation_has_actionable_errors(checkpoint, message):
    with pytest.raises(ValueError, match=message):
        _validate_checkpoint(checkpoint, Path("broken.pt"), "fallback")
