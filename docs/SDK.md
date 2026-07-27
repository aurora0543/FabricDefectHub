# Native Python SDK (`fabric_defect_hub`)

This guide documents the native Python SDK interface (`fabric_defect_hub`).

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

## 2. Scope: this is a benchmark platform, not a model zoo of our own

The SDK's job is to run *other people's* published methods under one contract
(`ModelAdapter` / `DatasetAdapter` / `Evaluator`) so their numbers are
comparable. It deliberately ships **no in-house network components** — no
backbones, necks, heads, losses, or augmentation modules of our own.

An earlier revision carried an `fdh.nn` package (feature hooks, a
`TextileAttentionNeck`, a segmentation head, an anomaly heatmap decoder) plus
`fdh.optim.losses` and `fdh.augmentations`. None of it was ever called by any
of the 18 registered models, and a config profile that could swap in its own
loss or architecture would have made the benchmark unsound — the row labelled
"YOLOv8" has to be stock YOLOv8. All of it was removed; see
`core/base_recipe.py` for the settings-only rule that replaced it.

If method-level contributions are added later, they belong in a separate line
of work evaluated *against* this benchmark, wired in through the same public
`ModelAdapter` contract as any other method.
