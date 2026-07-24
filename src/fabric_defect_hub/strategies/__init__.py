"""FabricDefectHub Strategy Package (Loader & Optimization Strategies)."""

from fabric_defect_hub.augmentations.textile_aug import TextilePeriodicAugmenter
from fabric_defect_hub.strategies.loader_strategies import (
    BatchNormCalibrator,
    SlidingWindowTiler,
    SparseSubsampler,
    TTAInferenceWrapper,
)

__all__ = [
    "TextilePeriodicAugmenter",
    "SparseSubsampler",
    "SlidingWindowTiler",
    "TTAInferenceWrapper",
    "BatchNormCalibrator",
]
