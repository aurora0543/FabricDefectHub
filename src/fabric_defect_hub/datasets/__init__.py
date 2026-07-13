"""DatasetAdapter implementations. Each concrete dataset module should
subclass `DatasetAdapter` (see `base.py`) and register itself with
`@fabric_defect_hub.core.registry.register_dataset("<name>")`.

Importing this package registers all bundled datasets.
"""

from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.datasets.yolo_bbox import build_class_map, yolo_staging_dir
from fabric_defect_hub.datasets.zju_leaper import ZJULeaperDataset

__all__ = [
    "ZJULeaperDataset",
    "yolo_staging_dir",
    "build_class_map",
    "anomalib_folder_staging_dir",
]
