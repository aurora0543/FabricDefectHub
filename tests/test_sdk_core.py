"""Unit tests for FabricDefectHub (UTAD-Framework) Modular SDK and config profiles."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

import fabric_defect_hub as fdh
from fabric_defect_hub.augmentations.textile_aug import TextilePeriodicAugmenter
from fabric_defect_hub.core.registry import get_recipe, list_recipes
from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
from fabric_defect_hub.evaluation.pro_calculator import compute_pro_score
from fabric_defect_hub.optim.losses import AFDLoss, DynamicLossScaler
from fabric_defect_hub.reporting.latex_generator import generate_latex_table


def test_recipe_registry():
    """Test that all config profiles are registered and retrievable."""
    recipe_ids = list_recipes()
    assert "yolov8" in recipe_ids
    assert "patchcore" in recipe_ids
    assert "rd4ad" in recipe_ids
    assert "mambaad" in recipe_ids
    assert "moeclip" in recipe_ids
    assert "dinomaly" in recipe_ids

    # Check model name resolution
    recipe_yolo = get_recipe("yolov8n")
    assert recipe_yolo.recipe_id == "yolov8"
    assert "arXiv" in recipe_yolo.paper_reference


def test_afd_loss():
    """Test Adaptive Focal-Dice Loss computation."""
    loss_fn = AFDLoss(alpha=0.5, gamma=2.0)
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.zeros(2, 1, 64, 64)
    targets[:, :, 10:20, 10:20] = 1.0  # Simulated tiny defect

    loss_val = loss_fn(logits, targets)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() > 0.0


def test_dynamic_loss_scaler():
    """Test Multi-Task Dynamic Loss Scaler."""
    scaler = DynamicLossScaler(num_losses=3)
    losses = {
        "box_loss": torch.tensor(1.5),
        "cls_loss": torch.tensor(0.5),
        "dfl_loss": torch.tensor(0.8),
    }
    total_loss, weighted_dict = scaler(losses)
    assert total_loss.item() > 0.0
    assert "weighted_box_loss" in weighted_dict


def test_textile_augmentation():
    """Test Textile Periodic Augmenter (TPA)."""
    aug = TextilePeriodicAugmenter(grid_freq=16, phase_shift_prob=1.0)
    img = torch.rand(3, 64, 64)
    mask = torch.zeros(1, 64, 64)
    aug_img, aug_mask = aug(img, mask)
    assert aug_img.shape == (3, 64, 64)
    assert torch.all(aug_img >= 0.0) and torch.all(aug_img <= 1.0)


def test_pro_calculator():
    """Test Per-Region Overlap (PRO) score calculation."""
    masks_gt = np.zeros((2, 64, 64), dtype=np.uint8)
    masks_gt[0, 10:20, 10:20] = 1

    anomaly_maps = np.zeros((2, 64, 64), dtype=np.float32)
    anomaly_maps[0, 11:19, 11:19] = 0.9

    score = compute_pro_score(masks_gt, anomaly_maps)
    assert 0.0 <= score <= 1.0


def test_lmei_profiler():
    """Test Latency-Memory Efficiency Index calculation."""
    score = calculate_lmei(fps=120.0, vram_mb=512.0, flops_g=8.5, params_m=3.2)
    assert score > 0.0


def test_latex_table_generator():
    """Test IEEE/CVPR LaTeX table code generator."""
    results = [
        {"model": "YOLOv8n", "recipe": "SD-Attn", "I-AUROC": 0.982, "PRO-Score": 0.941, "FPS": 145.0, "LMEI": 1.25},
        {"model": "PatchCore", "recipe": "DMBA", "I-AUROC": 0.965, "PRO-Score": 0.912, "FPS": 85.0, "LMEI": 0.88},
    ]
    latex_code = generate_latex_table(results)
    assert "\\begin{table*}" in latex_code
    assert "\\textbf{0.9820}" in latex_code
