"""`DatasetAdapter` for the "Fabric Defects Dataset", a fabric-texture defect
dataset laid out as flat class folders one level below the linked root:
`Fabric Defect Dataset/defect free/` (normal) plus `hole/`, `Vertical/`,
`horizontal/`, `lines/`, `stain/` (defect types). Image-level labels only —
no pixel masks — and no author-provided train/test split, so this reuses
`FlatFolderAnomalyDataset`'s synthesized, leak-free split.

In-domain fabric, so it is a *training-eligible* source (see
`training.ANOMALY_TRAINABLE_DATASETS` and the `fabric-train` composite), and
also usable on its own for image-level anomaly evaluation.
"""

from __future__ import annotations

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.datasets.flat_folder import FlatFolderAnomalyDataset


@register_dataset("fabric-defects")
class FabricDefectsDataset(FlatFolderAnomalyDataset):
    name = "fabric-defects"
    NORMAL_DIRNAME = "defect free"
    # Images sit under "<root>/Fabric Defect Dataset/<class>/", not directly
    # under the linked root.
    ROOT_SUBDIR = "Fabric Defect Dataset"
