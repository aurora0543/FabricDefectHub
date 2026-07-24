# FabricDefectHub (UTAD-Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Benchmark Protocol: IEEE/CVPR](https://img.shields.io/badge/Protocol-IEEE%2FCVPR-green.svg)](docs/BENCHMARK_AND_LATEX.md)

**FabricDefectHub (UTAD-Framework)** is a unified, modular Python SDK and benchmarking framework for industrial textile anomaly detection and defect segmentation. 

It integrates 18 model architectures behind one interface, per-model config profiles anchored to each method's paper, strategy-driven data loading, in-house neural-network modules (`fdh.nn`), and LaTeX table generation.

---

## 🏛️ System Architecture & Model Matrix

The benchmark consolidates **18 models** across supervised detectors and unsupervised anomaly segmenters. The **Config profile** column is the `recipe_id` that supplies each method's run settings (in the backend's real vocabulary, anchored to the paper) — it is a settings bundle, not a novel contribution:

| # | Model Architecture | Paradigm | Config profile (`recipe_id`) | Backbone / Notes |
| :-: | --- | --- | --- | --- |
| 1 | **YOLOv8n / YOLOv8s** | Supervised (CNN) | `yolov8` | Ultralytics detector; fabric training settings (SPD-Conv motivation) |
| 2 | **YOLO11n** | Supervised (CNN) | `yolov8` | Ultralytics YOLO11 detector |
| 3 | **Faster / Cascade R-CNN** | Supervised (R-CNN) | — *(torchvision baseline)* | FPN multi-scale features |
| 4 | **DETR** | Supervised (ViT) | — *(torchvision baseline)* | Query init & Hungarian matching |
| 5 | **Mask R-CNN / UNet++** | Supervised (FCN) | — *(torchvision baseline)* | Pixel mask supervision |
| 6 | **DeepLabV3+** | Supervised (FCN) | — *(torchvision baseline)* | ASPP dense context |
| 7 | **PatchCore** | Feature Embedding | `patchcore` | WideResNet-50 memory bank + 10% coreset (paper settings) |
| 8 | **PaDiM / SuperSimpleNet** | Feature Embedding | `patchcore` | Feature-embedding baselines |
| 9 | **RD4AD** | Teacher-Student | `rd4ad` | WideResNet-50 reverse distillation (paper settings) |
| 10 | **EfficientAD** | Teacher-Student | `rd4ad` | Teacher-student distillation |
| 11 | **MambaAD** | State Space (SSM) | `mambaad` | State-space decoder; upstream defaults |
| 12 | **Dinomaly** | DINOv2 Enc-Dec | `dinomaly` | DINOv2 encoder-decoder (ViTill); upstream defaults |
| 13 | **MoECLIP / WinCLIP** | Vision-Language | `moeclip` | CLIP + LoRA mixture-of-experts; upstream defaults |

Coverage: the six `recipe_id`s supply run settings for the anomaly/detection methods (rows 1–2, 7–13). The torchvision detectors/segmenters (rows 3–6) run as standard baselines on torchvision's own defaults — they intentionally carry no profile. Every profile's hyperparameters are expressed in its backend's real vocabulary and pinned to the backend's upstream-verified defaults by `tests/test_recipe_reconciliation.py`.

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/aurora0543/FabricDefectHub.git && cd FabricDefectHub
pip install -r requirements.txt
```

`requirements.txt` is intentionally the lean Gradio/inference deployment set.
For local or cloud training with every backend, including WinCLIP, Dinomaly,
and MoECLIP, install `pip install -r requirements-full.txt` instead.

### 2. Python SDK Usage (`import fabric_defect_hub as fdh`)
```python
import fabric_defect_hub as fdh

# Load dataset with 10% sparse ratio and 256x256 4K tiling strategy
dataset = fdh.load_dataset("raw-fabric", root="data/RAW_FABRID", sparse_ratio=0.1, tiling=True)

# Load model with TTA flip-multiscale inference strategy
model = fdh.load_model("ultralytics", "yolov8n", tta_mode="flip_multiscale")

# Run prediction & compute Per-Region Overlap (PRO-Score)
predictions = model.predict(dataset.load_samples())
pro_score = fdh.compute_pro_score(gt_masks, pred_anomaly_maps)
```

### 3. CLI Commands
```bash
# Train using research-grade config
fdh train configs/models/yolov8_textile.yaml

# List all academic recipes
fdh recipes

# Export paper-ready IEEE/CVPR LaTeX table
fdh export-latex results/benchmark.json --output paper_table.tex

# Launch Gradio Web Interface
fdh-ui
```

---

## 📚 Specialized Documentation Index

Detailed technical specifications and user guides are organized under `docs/`:

- 📐 **[SDK & In-House NN Engine Guide](docs/SDK_AND_NN.md)**: Modular SDK usage and autonomous `fdh.nn` (Feature Hooks, Necks, Heads).
- 🔬 **[Config Profiles & Loss Guide](docs/RECIPES_AND_LOSSES.md)**: per-model config profiles, AFDLoss, textile augmenter, and hyperparameters.
- 🚀 **[SDLP Loading & Testing Strategies](docs/SDLP_STRATEGIES.md)**: Sparse ratio sampling (`sparse_ratio`), 4K Sliding-Window Tiling, TTA, and BN Calibration.
- 📊 **[Benchmark Protocols & LaTeX Generator](docs/BENCHMARK_AND_LATEX.md)**: PRO-Score, LMEI Edge Index calculation, and automated LaTeX table rendering.
- 🌳 **[Extending FabricDefectHub](docs/EXTENDING.md)**: the dataset/backend availability decision tree (`fdh doctor`), the `--set` tuning window, and how to add a new dataset, backend, or config profile.

---

## 📜 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
