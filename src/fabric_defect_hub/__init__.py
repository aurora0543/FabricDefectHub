"""FabricDefectHub (UTAD-Framework): Unified Modular SDK & Benchmark for Fabric Defect Inspection.

Provides per-model config profiles, adaptive loading strategies, metric
evaluators, and reporting tools.

Quick Usage:
    import fabric_defect_hub as fdh

    # 1. Load dataset with 10% sparse ratio and 256x256 tiling strategy
    dataset = fdh.load_dataset("raw-fabric", root="data/RAW_FABRID", sparse_ratio=0.1, tiling=True)

    # 2. Load model with TTA flip-multiscale inference strategy
    model = fdh.load_model("ultralytics", "yolov8n", tta_mode="flip_multiscale")
"""

from fabric_defect_hub import recipes
from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.data_adapter import BatchSpec, DataAdapter, Normalization
from fabric_defect_hub.core.registry import get_recipe, list_recipes
from fabric_defect_hub.core.train_config import TrainConfig
from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
from fabric_defect_hub.evaluation.pro_calculator import compute_pro_score
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.models.base import ModelAdapter, ModelCapabilities
from fabric_defect_hub.reporting.latex_generator import generate_latex_table
from fabric_defect_hub.strategies.loader_strategies import (
    BatchNormCalibrator,
    SlidingWindowTiler,
    SparseSubsampler,
    TTAInferenceWrapper,
)

__version__ = "0.2.0"

__all__ = [
    "load_dataset",
    "load_model",
    "run_experiment",
    "recipes",
    "BaseModelRecipe",
    "get_recipe",
    "list_recipes",
    # The model/data contracts (see docs/EXTENDING.md)
    "ModelAdapter",
    "ModelCapabilities",
    "TrainConfig",
    "DataAdapter",
    "BatchSpec",
    "Normalization",
    "compute_pro_score",
    "calculate_lmei",
    "generate_latex_table",
    "SparseSubsampler",
    "SlidingWindowTiler",
    "TTAInferenceWrapper",
    "BatchNormCalibrator",
]
