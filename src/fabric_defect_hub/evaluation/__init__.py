"""Task-specific `Evaluator` implementations: `detection.py`, `segmentation.py`,
`anomaly.py`, plus an `industrial.py` for line-level metrics (under/over
detection rate, alarms per unit length). See `base.py` for the shared contract.

Every evaluator here operates purely on `Sample` + `Prediction` — none of
them know or care which `ModelAdapter`/backend produced the predictions,
so the same evaluator scores Ultralytics, torchvision, and Anomalib runs
alike.
"""

from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.evaluation.detection import DetectionEvaluator
from fabric_defect_hub.evaluation.industrial import IndustrialEvaluator
from fabric_defect_hub.evaluation.segmentation import SegmentationEvaluator

__all__ = ["AnomalyEvaluator", "DetectionEvaluator", "SegmentationEvaluator", "IndustrialEvaluator"]
