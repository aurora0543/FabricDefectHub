"""Unit tests for Autonomous Neural Network Modules (fdh.nn)."""

from __future__ import annotations

import pytest
import torch

import fabric_defect_hub as fdh


def test_backbone_and_feature_hook_engine():
    """Test backbone loading and non-intrusive FeatureHookEngine feature extraction."""
    backbone, target_layers = fdh.nn.get_backbone("resnet18", pretrained=False)
    hook_engine = fdh.nn.FeatureHookEngine(backbone, target_layers)

    dummy_input = torch.randn(2, 3, 256, 256)
    feature_maps = hook_engine.extract_features(dummy_input)

    assert "layer1" in feature_maps
    assert "layer2" in feature_maps
    assert feature_maps["layer1"].shape == (2, 64, 64, 64)
    assert feature_maps["layer2"].shape == (2, 128, 32, 32)

    hook_engine.remove_hooks()


def test_textile_attention_neck():
    """Test pluggable TextileAttentionNeck (SD-Attn and CBAM modes)."""
    feature_maps = {
        "layer1": torch.randn(2, 64, 64, 64),
        "layer2": torch.randn(2, 128, 32, 32),
    }

    # Test SD-Attn Attention mode
    sd_neck = fdh.nn.TextileAttentionNeck(in_channels_list=[64, 128], out_channels=256, mode="sd_attn")
    enhanced_sd = sd_neck(feature_maps)
    assert len(enhanced_sd) == 2
    assert enhanced_sd[0].shape == (2, 256, 32, 32)  # Space-to-depth 64x64 -> 32x32

    # Test CBAM Attention mode
    cbam_neck = fdh.nn.TextileAttentionNeck(in_channels_list=[64, 128], out_channels=256, mode="cbam")
    enhanced_cbam = cbam_neck(feature_maps)
    assert len(enhanced_cbam) == 2
    assert enhanced_cbam[0].shape == (2, 256, 64, 64)


def test_task_head_and_anomaly_decoder():
    """Test DefectSegmentationHead and AnomalyHeatmapDecoder."""
    enhanced_features = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
    ]

    head = fdh.nn.DefectSegmentationHead(in_channels=256, num_classes=1)
    mask_logits = head(enhanced_features, target_size=(256, 256))
    assert mask_logits.shape == (2, 1, 256, 256)

    heatmap = fdh.nn.AnomalyHeatmapDecoder.decode(enhanced_features, target_size=(256, 256))
    assert heatmap.shape == (2, 1, 256, 256)
