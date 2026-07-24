"""In-House Autonomous Neural Network Modules for FabricDefectHub (fdh.nn).

Provides autonomous feature hooks, pluggable attention necks, task heads, and anomaly decoders.
"""

from fabric_defect_hub.nn import heads, necks
from fabric_defect_hub.nn.backbones import get_backbone
from fabric_defect_hub.nn.heads.anomaly_head import AnomalyHeatmapDecoder, DefectSegmentationHead
from fabric_defect_hub.nn.hooks import FeatureHookEngine
from fabric_defect_hub.nn.necks.textile_neck import TextileAttentionNeck

__all__ = [
    "get_backbone",
    "FeatureHookEngine",
    "TextileAttentionNeck",
    "DefectSegmentationHead",
    "AnomalyHeatmapDecoder",
    "necks",
    "heads",
]
