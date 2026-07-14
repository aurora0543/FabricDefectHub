"""`DatasetAdapter` for the ZJU-Leaper fabric defect benchmark.

ZJU-Leaper ships 19 fabric patterns (textures) and 5 groups, each with a
normal/defect x train/test split defined by JSON id-lists under
`ImageSets/`. Per-image XML annotations carry the pattern, the
`defective` flag, optional bounding boxes and a pixel mask.

This adapter turns any user-chosen slice of that benchmark into a unified
`Sample` list, exposing the four controls the project needs:

1. Sample-count control — `num_samples` (None = use everything, i.e. the
   "few-shot / full data" regime; a small number like 300-400 = the
   "少样本 / low-shot" regime).
2. Defect adaptation — `use_defect` (False = zero-shot / normal-only; True =
   include defective images).
3. Texture/pattern filtering — `pattern` selects one texture pattern, one
   group, or the whole benchmark.
4. Unified config — `num_samples`, `use_defect`, `defect_ratio` together fix
   how many images, whether defects are present, and the defect fraction.

Dataset homepage: http://www.qaas.zju.edu.cn/zju-leaper/
"""

from __future__ import annotations

import csv
import json
import random
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import DatasetAdapter

_DEFECT_LABEL = "defect"  # ZJU-Leaper is single-class (defect vs normal)


@register_dataset("zju-leaper")
class ZJULeaperDataset(DatasetAdapter):
    """Configurable adapter over the ZJU-Leaper fabric dataset.

    Parameters (all via keyword, consumed here rather than left in
    `self.options`):

    root: dataset root containing `Images/`, `Annotations/`, `ImageSets/`.
    split: "train" or "test" (which side of the benchmark split to draw from).
    pattern: texture/group selector. Accepts
        - None or "total"            -> the whole benchmark (ImageSets/total.json)
        - "pattern7" / 7             -> ImageSets/Patterns/pattern7.json
        - a texture name, e.g. "Knot Pattern" (matched against statistic.csv)
        - "group3"                   -> ImageSets/Groups/group3.json
    num_samples: total images to load. None = all images in the selection
        (few-shot / full-data). A small int (e.g. 350) = low-shot regime.
    use_defect: include defective images. False = zero-shot (normal only).
    defect_ratio: fraction of the loaded set that is defective. Only applied
        when `num_samples` is set and `use_defect` is True; ignored when
        `num_samples` is None (then all normal + all defect are used).
    task: "anomaly" (default), "detection", or "segmentation". Drives
        `Sample.task` and which annotation fields are prioritised, but every
        available label (flag / boxes / mask) is still attached.
    seed: RNG seed for reproducible subsampling.
    """

    name = "zju-leaper"

    def __init__(
        self,
        root: str,
        split: str = "test",
        pattern: str | int | None = None,
        num_samples: int | None = None,
        use_defect: bool = True,
        defect_ratio: float = 0.5,
        task: Task = "anomaly",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(root=root, split=split, **kwargs)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0.0 <= defect_ratio <= 1.0:
            raise ValueError(f"defect_ratio must be in [0, 1], got {defect_ratio}")

        self.root_path = Path(root)
        self.pattern = pattern
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def _imageset_file(self) -> Path:
        """Resolve `self.pattern` to the ImageSets JSON that lists its ids."""

        image_sets = self.root_path / "ImageSets"
        p = self.pattern
        if p is None or p == "total":
            return image_sets / "total.json"
        if isinstance(p, int):
            return image_sets / "Patterns" / f"pattern{p}.json"
        text = str(p).strip()
        low = text.lower().replace(" ", "")
        if low.startswith("pattern"):
            return image_sets / "Patterns" / f"{low}.json"
        if low.startswith("group"):
            return image_sets / "Groups" / f"{low}.json"
        # Otherwise treat it as a human texture name from statistic.csv.
        pattern_id = self._pattern_name_to_id(text)
        if pattern_id is None:
            raise ValueError(
                f"unknown pattern {p!r}; expected None/'total', 'patternN'/N, "
                f"'groupN', or a texture name from statistic.csv"
            )
        return image_sets / "Patterns" / f"pattern{pattern_id}.json"

    def _pattern_name_to_id(self, name: str) -> int | None:
        stats = self.root_path / "statistic.csv"
        if not stats.exists():
            return None
        target = name.strip().lower()
        with open(stats, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                if row[0].strip().lower() == target:
                    try:
                        return int(row[1].strip())
                    except ValueError:
                        return None
        return None

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select_ids(self) -> tuple[list[str], list[str]]:
        """Return (normal_ids, defect_ids) chosen per the count/ratio config."""

        index_path = self._imageset_file()
        with open(index_path) as fh:
            index = json.load(fh)

        normal_pool = _validate_index_ids(index, "normal", self.split, index_path)
        defect_pool = (
            _validate_index_ids(index, "defect", self.split, index_path)
            if self.use_defect
            else []
        )

        rng = random.Random(self.seed)
        rng.shuffle(normal_pool)
        rng.shuffle(defect_pool)

        if self.num_samples is None:
            # Few-shot / full-data: take everything in the selection.
            return normal_pool, defect_pool

        # Low-shot: honour total count and defect ratio.
        if not self.use_defect:
            return normal_pool[: self.num_samples], []

        n_defect = min(round(self.num_samples * self.defect_ratio), len(defect_pool))
        n_normal = min(self.num_samples - n_defect, len(normal_pool))
        return normal_pool[:n_normal], defect_pool[:n_defect]

    # ------------------------------------------------------------------ #
    # Sample building
    # ------------------------------------------------------------------ #
    def _build_sample(self, image_id: str, is_defect: bool) -> Sample:
        image_path = self.root_path / "Images" / f"{image_id}.jpg"
        xml_path = self.root_path / "Annotations" / "xmls" / f"{image_id}.xml"

        boxes: list[list[float]] = []
        mask_path: str | None = None
        metadata: dict = {}

        if xml_path.exists():
            root = ET.parse(xml_path).getroot()
            metadata = {
                "pattern_id": _text(root, "pattern_id"),
                "pattern_name": _text(root, "pattern_name"),
                "group_id": _text(root, "group_id"),
                "group_name": _text(root, "group_name"),
                "fabric_type": _text(root, "pattern_name"),
            }
            mask_name = _text(root, "mask_filename")
            if mask_name:
                candidate = self.root_path / "Annotations" / "masks" / mask_name
                mask_path = str(candidate) if candidate.exists() else None
            for bbox in root.findall("bbox"):
                coordinates = [bbox.findtext(tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
                try:
                    parsed = [float(value) for value in coordinates if value is not None]
                except ValueError:
                    parsed = []
                if len(parsed) != 4:
                    warnings.warn(
                        f"Skipping malformed bbox in {xml_path}: expected numeric "
                        "xmin/ymin/xmax/ymax values.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                boxes.append(parsed)

        annotations = Annotations(is_anomalous=is_defect)
        if is_defect and boxes:
            annotations.boxes = boxes
            annotations.labels = [_DEFECT_LABEL] * len(boxes)
        if is_defect and mask_path is not None:
            annotations.anomaly_mask = mask_path
            if self.task == "segmentation":
                annotations.masks = [mask_path]

        return Sample(
            id=image_id,
            image_path=str(image_path),
            task=self.task,
            annotations=annotations,
            metadata=metadata,
        )

    def load_samples(self) -> list[Sample]:
        normal_ids, defect_ids = self._select_ids()
        samples = [self._build_sample(i, is_defect=False) for i in normal_ids]
        samples += [self._build_sample(i, is_defect=True) for i in defect_ids]
        return samples


def _text(element: ET.Element, tag: str) -> str | None:
    value = element.findtext(tag)
    return value.strip() if value is not None else None


def _validate_index_ids(index: object, category: str, split: str, path: Path) -> list[str]:
    if not isinstance(index, dict):
        raise ValueError(f"Invalid ImageSets index {path}: top level must be a JSON object.")
    category_data = index.get(category)
    if not isinstance(category_data, dict):
        raise ValueError(
            f"Invalid ImageSets index {path}: '{category}' must be an object containing split lists."
        )
    ids = category_data.get(split)
    if not isinstance(ids, list) or not all(isinstance(image_id, str) for image_id in ids):
        raise ValueError(
            f"Invalid ImageSets index {path}: '{category}.{split}' must be a list of image-id strings."
        )
    return list(ids)
