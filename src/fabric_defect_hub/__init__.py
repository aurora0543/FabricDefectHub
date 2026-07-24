"""FabricDefectHub (UTAD-Framework): Unified Modular SDK & Benchmark for Fabric Defect Inspection.

Provides paper-driven optimization recipes, adaptive loading strategies, autonomous neural network modules (fdh.nn), losses, metric evaluators, and paper-grade reporting tools.

Quick Usage:
    import fabric_defect_hub as fdh

    # 1. Load dataset with 10% sparse ratio and 256x256 tiling strategy
    dataset = fdh.load_dataset("raw-fabric", root="data/RAW_FABRID", sparse_ratio=0.1, tiling=True)

    # 2. Load model with TTA flip-multiscale inference strategy
    model = fdh.load_model("ultralytics", "yolov8n", tta_mode="flip_multiscale")

    # 3. Assemble custom model using autonomous fdh.nn modules
    backbone, layers = fdh.nn.get_backbone("resnet18")
    hook_engine = fdh.nn.FeatureHookEngine(backbone, layers)
    neck = fdh.nn.TextileAttentionNeck(in_channels_list=[128, 256], mode="sd_attn")
"""

from fabric_defect_hub import nn, recipes
from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import get_recipe, list_recipes
from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
from fabric_defect_hub.evaluation.pro_calculator import compute_pro_score
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.optim.losses import AFDLoss, DynamicLossScaler
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
    "nn",
    "BaseModelRecipe",
    "get_recipe",
    "list_recipes",
    "AFDLoss",
    "DynamicLossScaler",
    "compute_pro_score",
    "calculate_lmei",
    "generate_latex_table",
    "SparseSubsampler",
    "SlidingWindowTiler",
    "TTAInferenceWrapper",
    "BatchNormCalibrator",
]
