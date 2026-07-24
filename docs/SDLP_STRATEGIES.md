# Strategy-Driven Loading & Protocol Engine (SDLP) Guide

This guide documents the loading and evaluation strategies implemented in `fabric_defect_hub.strategies`.

---

## 1. Sparse Proportionate Subsampling (`sparse_ratio`)

Enables precise few-shot and data-efficient benchmarking across datasets:

```python
from fabric_defect_hub import load_dataset

# Subsample dataset to exactly 10% sparse ratio
dataset = load_dataset("zju-leaper", root="data/ZJU-Leaper", sparse_ratio=0.1)

# Apply pattern-stratified subsampling across 19 ZJU-Leaper fabric textures
dataset_stratified = load_dataset("zju-leaper", root="data/ZJU-Leaper", stratified_by="pattern", sparse_ratio=0.1)
```

---

## 2. 4K Sliding-Window Tiling & Mask Stitching (`tiling=True`)

Solves the input resolution mismatch between high-resolution industrial cameras (4K/8K images) and deep learning models (256x256 / 640x640 inputs) without modifying model architectures:

```python
dataset = load_dataset(
    "raw-fabric",
    root="data/RAW_FABRID",
    split="test",
    tiling=True,               # Enable sliding window tiling
    tile_size=(256, 256),     # Tile resolution
    overlap=0.25,             # 25% boundary overlap
)

# Pipeline automatically splits 4K images into overlapping tiles during predict(),
# runs model inference, and stitches tile anomaly maps back into full-resolution predictions.
```

---

## 3. Test-Time Augmentation (TTA, `tta_mode`)

Applies horizontal/vertical flipping and multi-scale test augmentation during prediction, averaging anomaly scores and maps to eliminate false positives:

```python
from fabric_defect_hub import load_model

model = load_model("ultralytics", "yolov8n", tta_mode="flip_multiscale")
predictions = model.predict(samples)
```

---

## 4. BatchNorm Feature Calibration (`calibrate_bn=True`)

When deploying a model pretrained on one fabric type to a new fabric material, `BatchNormCalibrator` runs forward passes on a small batch of normal fabric images in `train()` mode to update BatchNorm/LayerNorm `running_mean` and `running_var`:

```python
from fabric_defect_hub.strategies import BatchNormCalibrator

# Calibrates running statistics on normal fabric images before testing
BatchNormCalibrator.calibrate(model.inner_module, normal_samples, num_steps=16)
```
