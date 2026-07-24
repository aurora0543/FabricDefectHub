# Native Python SDK & Autonomous Neural Network Modules (`fdh.nn`)

This guide documents the native Python SDK interface (`fabric_defect_hub`) and the autonomous in-house neural network modules (`fdh.nn`).

---

## 1. Modular Python SDK Integration

Rather than relying on command-line glue scripts, `fabric_defect_hub` exposes a native Python SDK for direct importing:

```python
import fabric_defect_hub as fdh

# Load dataset with loading strategies
dataset = fdh.load_dataset(
    name="raw-fabric",
    root="data/RAW_FABRID",
    split="test",
    sparse_ratio=0.1,        # 10% sparse subsampling
    tiling=True,              # 4K sliding-window tiling
    tile_size=(256, 256),
    overlap=0.25,
)

# Load model with inference strategies
model = fdh.load_model(
    backend="ultralytics",
    name="yolov8n",
    weights="artifacts/models/published/yolov8n.pt",
    tta_mode="flip_multiscale",  # Test-Time Augmentation
)

# Predict & Evaluate
samples = dataset.load_samples()
predictions = model.predict(samples)
pro_score = fdh.compute_pro_score(gt_masks, pred_maps)
```

---

## 2. In-House Autonomous Modules (`fdh.nn`)

To eliminate black-box dependencies on external frameworks, `fdh.nn` provides autonomous modules for feature interception, attention necks, and task heads:

```
src/fabric_defect_hub/nn/
├── hooks.py              # FeatureHookEngine (Non-intrusive PyTorch Forward Hooks)
├── backbones.py          # get_backbone (Lightweight ResNet/WideResNet/EfficientNet wrapper)
├── necks/
│   ├── base_neck.py      # BaseNeck Abstract Class
│   └── textile_neck.py   # TextileAttentionNeck (SD-Attn, CBAM, Coordinate Attention)
└── heads/
    └── anomaly_head.py   # DefectSegmentationHead & AnomalyHeatmapDecoder
```

### Modular Assembly Example

```python
import torch
import fabric_defect_hub as fdh

# 1. Obtain backbone and default layer targets
backbone, target_layers = fdh.nn.get_backbone("resnet18", pretrained=True)

# 2. Attach PyTorch Forward Hook Engine
hook_engine = fdh.nn.FeatureHookEngine(backbone, target_layers)

# 3. Assemble Textile Attention Neck (SD-Attn Mode)
sd_neck = fdh.nn.TextileAttentionNeck(
    in_channels_list=[64, 128],
    out_channels=256,
    mode="sd_attn",
)

# 4. Forward execution & feature enhancement
dummy_input = torch.randn(2, 3, 256, 256)
feature_maps = hook_engine.extract_features(dummy_input)
enhanced_features = sd_neck(feature_maps)

# 5. Decode predictions via in-house segmentation head & heatmap decoder
mask_logits = fdh.nn.DefectSegmentationHead(in_channels=256)(enhanced_features, target_size=(256, 256))
heatmap = fdh.nn.AnomalyHeatmapDecoder.decode(enhanced_features, target_size=(256, 256))
```

---

## 3. Module Specifications

### `FeatureHookEngine(backbone: nn.Module, target_layers: List[str])`
- Intercepts PyTorch module outputs during `forward()` using non-intrusive registration hooks.
- Method `extract_features(x: torch.Tensor) -> Dict[str, torch.Tensor]`.
- Method `remove_hooks()` prevents GPU memory leaks.

### `TextileAttentionNeck(in_channels_list, out_channels=256, mode="sd_attn")`
- Supported modes: `"sd_attn"` (Space-to-Depth Downsampling Attention), `"cbam"` (Channel & Spatial Attention), `"identity"`.
- Converts multi-scale feature channels to `out_channels` and applies spatial-channel attention.

### `DefectSegmentationHead(in_channels=256, num_classes=1)`
- Interpolates multi-resolution neck outputs and fuses them into pixel-wise defect logits.
