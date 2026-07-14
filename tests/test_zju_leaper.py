import json

import pytest

from fabric_defect_hub.datasets.zju_leaper import ZJULeaperDataset


def _dataset_root(tmp_path, index):
    image_sets = tmp_path / "ImageSets"
    image_sets.mkdir()
    (image_sets / "total.json").write_text(json.dumps(index))
    (tmp_path / "Images").mkdir()
    (tmp_path / "Annotations" / "xmls").mkdir(parents=True)
    return tmp_path


def test_invalid_imageset_structure_has_contextual_error(tmp_path):
    root = _dataset_root(tmp_path, {"normal": {"test": "not-a-list"}, "defect": {"test": []}})

    with pytest.raises(ValueError, match=r"normal\.test.*list of image-id strings"):
        ZJULeaperDataset(root=str(root)).load_samples()


def test_malformed_bbox_is_skipped_with_warning(tmp_path):
    root = _dataset_root(tmp_path, {"normal": {"test": []}, "defect": {"test": ["bad-box"]}})
    xml = root / "Annotations" / "xmls" / "bad-box.xml"
    xml.write_text(
        "<annotation><bbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>bad</xmax></bbox></annotation>"
    )

    with pytest.warns(RuntimeWarning, match="Skipping malformed bbox"):
        samples = ZJULeaperDataset(root=str(root), task="detection").load_samples()

    assert len(samples) == 1
    assert samples[0].annotations.boxes is None
    assert samples[0].annotations.labels is None
