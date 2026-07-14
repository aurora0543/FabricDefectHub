import importlib.util
from pathlib import Path


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coco_converter_emits_detection_samples():
    tool = _load_tool("convert_annotations")
    samples = tool.coco_detection_samples(
        {
            "images": [{"id": 7, "file_name": "fabric.jpg", "width": 100, "height": 80}],
            "categories": [{"id": 3, "name": "hole"}],
            "annotations": [{"image_id": 7, "category_id": 3, "bbox": [10, 20, 30, 40]}],
        },
        Path("images"),
    )
    assert samples[0].image_path == "images/fabric.jpg"
    assert samples[0].annotations.boxes == [[10.0, 20.0, 40.0, 60.0]]
    assert samples[0].annotations.labels == ["hole"]
