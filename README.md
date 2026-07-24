# FabricDefectHub (UTAD-Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Benchmark Protocol: IEEE/CVPR](https://img.shields.io/badge/Protocol-IEEE%2FCVPR-green.svg)](docs/BENCHMARK_AND_LATEX.md)

**FabricDefectHub (UTAD-Framework)** is a unified, modular Python SDK and benchmarking framework for industrial textile anomaly detection and defect segmentation. 

It integrates 18 state-of-the-art architectures, paper-driven optimization recipes (MORR Engine), strategy-driven data loading (SDLP Engine), autonomous neural network modules (`fdh.nn`), and paper-ready LaTeX table generation.

---

## 🏛️ System Architecture & Model Matrix

The benchmark consolidates **18 models** categorized into supervised detectors and unsupervised anomaly segmenters, equipped with paper-driven fine-tuning recipes:

| # | Model Architecture | Paradigm | Academic Recipe (`recipe_id`) | Core Mechanism / Nomenclature |
| :-: | --- | --- | --- | --- |
| 1 | **YOLOv8n / YOLOv8s** | Supervised (CNN) | `yolov8_sd_attn` | **SD-Attn (SPD-Conv)** + AFDLoss (Adaptive Focal-Dice) |
| 2 | **YOLO11n** | Supervised (CNN) | `yolov8_sd_attn` | Space-to-Depth Downsampling + Dynamic Loss Scaler |
| 3 | **Faster / Cascade R-CNN** | Supervised (R-CNN) | Standard R-CNN | FPN Multi-Scale Feature Weighting |
| 4 | **DETR** | Supervised (ViT) | Deformable Matching | DINO-style Query Initialization & Hungarian Matching |
| 5 | **Mask R-CNN / UNet++** | Supervised (FCN) | Segmentation | Multi-level Pixel Mask Supervision |
| 6 | **DeepLabV3+** | Supervised (FCN) | Atrous Conv | ASPP Dense Context Aggregation |
| 7 | **PatchCore** | Feature Embedding | `patchcore_dmba` | **DMBA** (Domain Memory Bank Adaptation & Coreset Subsampling) |
| 8 | **PaDiM / SuperSimpleNet** | Feature Embedding | `patchcore_dmba` | Gaussian Smooth Feature Subspace Estimation |
| 9 | **RD4AD** | Teacher-Student | `rd4ad_msfa_d` | **MSFA-D** (Multi-Scale Feature Alignment Distillation & Temp Anneal) |
| 10 | **EfficientAD** | Teacher-Student | `rd4ad_msfa_d` | Logical & Structural Student Anomaly Distillation |
| 11 | **MambaAD** | State Space (SSM) | `mambaad_ss_tst` | **SS-TST** (Selective Scan Texture State Space Tuning) |
| 12 | **Dinomaly** | DINOv2 Enc-Dec | Zero-shot Anomaly | DINOv2 Feature Alignment Bottleneck |
| 13 | **MoECLIP / WinCLIP** | Vision-Language | `moeclip_tpo_peft` | **TPO-PEFT** (Text Prompt Optimization & LoRA Adapter) |

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/aurora0543/FabricDefectHub.git && cd FabricDefectHub
pip install -r requirements.txt
```

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
fdh train configs/models/yolov8_sd_attn_textile.yaml

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
- 🔬 **[Academic Recipes & Loss Tuning Guide](docs/RECIPES_AND_LOSSES.md)**: MORR Recipes, AFDLoss, TPA Textile Augmenter, and Hyperparameter Optimization.
- 🚀 **[SDLP Loading & Testing Strategies](docs/SDLP_STRATEGIES.md)**: Sparse ratio sampling (`sparse_ratio`), 4K Sliding-Window Tiling, TTA, and BN Calibration.
- 📊 **[Benchmark Protocols & LaTeX Generator](docs/BENCHMARK_AND_LATEX.md)**: PRO-Score, LMEI Edge Index calculation, and automated LaTeX table rendering.

---

## 📜 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
