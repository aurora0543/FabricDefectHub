# FabricDefectHub (UTAD-Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Benchmark Protocol: IEEE/CVPR](https://img.shields.io/badge/Protocol-IEEE%2FCVPR-green.svg)](docs/BENCHMARK_AND_LATEX.md)

**FabricDefectHub (UTAD-Framework)** is a unified, modular Python SDK and benchmarking framework for industrial textile anomaly detection and defect segmentation. 

It integrates **20 canonical model entries across 6 backend adapters** behind one interface: 3 Ultralytics detectors, 6 torchvision detection/segmentation models, 8 Anomalib anomaly models, and Dinomaly, MoECLIP, and MambaAD. Each model has a stable catalog key and published-weight slot; the catalog is the source of truth for the Web UI and batch benchmark. On the current workspace, **16 canonical slots already contain published artifacts**; the cloud completion scope is the remaining four (`EfficientAD`, `WinCLIP`, `MoECLIP`, `MambaAD`). The project also registers **9 dataset adapters** (8 source datasets plus the `fabric-train` aggregate) for strategy-driven data loading and LaTeX table generation.

---

## 🏛️ System Architecture & Model Matrix

The benchmark consolidates **20 canonical models** across supervised detection, segmentation, and anomaly detection. This is the published catalog used by the Web UI and `fdh train-all`, not the larger set of experimental variants accepted by individual backends. The **Config profile** column is the `recipe_id` that supplies each method's run settings (in the backend's real vocabulary, anchored to the paper) — it is a settings bundle, not a novel contribution:

| # | Model Architecture | Backend | Paradigm | Config profile (`recipe_id`) | Current status |
| :-: | --- | --- | --- | --- | --- |
| 1 | **YOLOv8n** | Ultralytics | Supervised detection | `yolov8` | ✅ covered; ZJU run path and published weight |
| 2 | **YOLOv8s** | Ultralytics | Supervised detection | `yolov8` | ✅ covered; ZJU run path and published weight |
| 3 | **YOLO11n** | Ultralytics | Supervised detection | `yolov8` | ✅ covered; ZJU run path and published weight |
| 4 | **Faster R-CNN** | torchvision | Supervised detection | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 5 | **Cascade R-CNN** | torchvision | Supervised detection | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 6 | **DETR** | torchvision | Supervised detection | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 7 | **Mask R-CNN** | torchvision | Instance segmentation | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 8 | **UNet++** | torchvision | Semantic segmentation | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 9 | **DeepLabV3+** | torchvision | Semantic segmentation | — *(baseline)* | ✅ covered; ZJU run path and published weight |
| 10 | **PatchCore** | Anomalib | Feature embedding | `patchcore` | ✅ covered; ZJU normal-only path and published weight |
| 11 | **PaDiM** | Anomalib | Feature embedding | `patchcore` | ✅ covered; ZJU normal-only path and published weight |
| 12 | **RD4AD** | Anomalib | Teacher-student | `rd4ad` | ✅ covered; ZJU normal-only path and published weight |
| 13 | **EfficientAD** | Anomalib | Teacher-student | `rd4ad` | 🟡 configured, but needs an external natural-image `imagenet_dir`; not a completed ZJU run |
| 14 | **SuperSimpleNet** | Anomalib | Feature embedding | `patchcore` | ✅ covered; ZJU path and published weight |
| 15 | **STFPM** | Anomalib | Teacher-student | — *(constructor defaults)* | ✅ covered; ZJU normal-only path and published weight |
| 16 | **GANomaly** | Anomalib | Adversarial (GAN) | — *(constructor defaults)* | ✅ covered; image-level anomaly output and published weight |
| 17 | **WinCLIP** | Anomalib | Vision-language / zero-shot | `moeclip` | 🟡 configured; training-free, requires cached/downloadable OpenCLIP weights |
| 18 | **Dinomaly** | Dinomaly | DINOv2 encoder-decoder | `dinomaly` | ✅ covered; ZJU normal-only path and published weight |
| 19 | **MoECLIP** | MoECLIP | Vision-language / zero-shot | `moeclip` | 🟡 configured; requires `ViT-L-14-336px.pt` and MVTec AD auxiliary training |
| 20 | **MambaAD** | MambaAD | State-space (SSM) | `mambaad` | 🟡 implemented, but high-resource and not yet an accepted completed baseline; export is unsupported |

Status meaning: **✅ covered** means the adapter, configuration-driven lifecycle,
and a project artifact/run path are present. **🟡 configured** means the model
is represented by the same contracts and can be attempted through `fdh`, but a
cloud prerequisite or an explicit resource limitation remains before claiming a
completed baseline. In particular, WinCLIP is training-free (its artifact is a
reconstructable zero-shot handle, not a learned checkpoint); MoECLIP trains its
adapter on MVTec AD and evaluates ZJU as an unseen domain; EfficientAD requires
an external natural-image regularization directory; and MambaAD is retained as
an implemented high-resource model whose export path is intentionally absent.

Coverage: the six `recipe_id`s supply run settings for the YOLO, PatchCore/PaDiM/SuperSimpleNet, RD4AD/EfficientAD, MambaAD, Dinomaly, and CLIP-family entries. The six torchvision detectors/segmenters (rows 4–9) run as standard baselines on torchvision's own defaults — they intentionally carry no profile. Every profile's hyperparameters are expressed in its backend's real vocabulary and pinned to the backend's upstream-verified defaults by `tests/test_recipe_reconciliation.py`.

Rows 15–16 (STFPM, GANomaly) carry no profile either, and that is a statement of fact rather than a gap to fill later: a `recipe_id` here means "these settings are anchored to that method's paper", and neither has been reproduced against its paper in this project yet. Both run on the constructor defaults declared in `models/anomalib/presets.py`, with each preset's comment saying explicitly which values are upstream's own and which (if any) were adjusted for fabric. A profile gets added when a reproduction earns it — see [`docs/MODEL_CONFIGURATION.md`](docs/MODEL_CONFIGURATION.md) for details.

Beyond the published catalog above, the anomalib backend also reaches
**DRAEM**, **DSR**, **GLASS**, **FastFlow**, **UniNet** and **AnomalyDINO**
through `models/anomalib/presets.py` — reconstruction, normalizing-flow and
zero-shot-DINOv2 families that are runnable today (`fdh train draem`) but not
yet trained and published here, so they are deliberately kept out of the
catalog and the UI dropdown. `fdh.list_models("anomalib")` prints the full set.

---

## 🗂️ Dataset Catalog

FabricDefectHub currently registers **9 dataset adapters**: **8 source datasets** and the **`fabric-train` aggregate**. They are converted by `DatasetAdapter` into the shared `Sample` contract; each backend then consumes only the annotations it needs (boxes for detection, masks for segmentation, or normal/abnormal labels and maps for anomaly detection).

| Dataset key | Dataset | Scope | Supported tasks | Intended role |
| --- | --- | --- | --- | --- |
| `zju-leaper` | ZJU-Leaper | Fabric, 19 patterns | Detection, segmentation, anomaly | Primary in-domain training and evaluation source; supports normal-only and defect-inclusive selections |
| `raw-fabric` | RAW_FABRID | Fabric | Segmentation, anomaly | In-domain anomaly training source with MVTec-style normal/defect/mask folders |
| `tilda-400` | TILDA-400 | Fabric | Anomaly | In-domain anomaly training and held-out evaluation |
| `fabric-defects` | Fabric Defects Dataset | Fabric | Segmentation, anomaly | In-domain anomaly training and mask-bearing evaluation |
| `tianchi` | Tianchi Guangdong Fabric Defect Challenge | Fabric | Detection, anomaly | Detection training plus normal-image anomaly training |
| `fabric-train` | Fabric training aggregate | Composite | Detection, segmentation, anomaly | Union of the in-domain fabric sources for one-class anomaly training; not an independent raw dataset |
| `mvtec-ad` | MVTec AD | Cross-domain objects/textures | Segmentation, anomaly | Auxiliary zero-shot training source and external evaluation |
| `mvtec-loco` | MVTec LOCO AD | Cross-domain logical anomalies | Segmentation, anomaly | Auxiliary zero-shot training source and external evaluation |
| `visa` | VisA | Cross-domain industrial objects | Segmentation, anomaly | MoECLIP auxiliary training source and external evaluation |

The five in-domain fabric sources are `zju-leaper`, `raw-fabric`, `tilda-400`, `fabric-defects`, and `tianchi`. `fabric-train` combines their compatible samples. `mvtec-ad`, `mvtec-loco`, and `visa` are deliberately kept separate from ordinary fabric anomaly training; they provide cross-domain or zero-shot protocols instead. Dataset bytes are not committed to the repository: stage each source under its `data/<Dataset>` path (or a symlink) and use `fdh doctor` to verify what is available on the current machine.

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

### What a fresh clone gives you — and what it doesn't

The repository ships **code, configs, and contracts. It does not ship weights
or datasets**, and both are gitignored (`/artifacts/`, `/data/*`). So on a
fresh clone:

| | State | How to get it |
| --- | --- | --- |
| CLI, SDK, web UI | ✅ work immediately | — |
| `fdh models` / `fdh doctor` / `fdh recipes` | ✅ work immediately | — |
| Model checkpoints | 🟠 every model reads **"Checkpoint missing"** | `fdh train <model>` publishes to `artifacts/models/published/`; the UI picks it up on refresh |
| Datasets | 🟠 sampler finds nothing | stage under `data/<Dataset>` (usually a symlink onto external storage) |

`fdh doctor` is the one command that reports exactly which backends are
installed and which datasets are staged **on this machine**, and why anything
isn't runnable — start there rather than guessing. Full path from a bare
machine to trained weights: [`docs/cloud_training_runbook.md`](docs/cloud_training_runbook.md).

### 2. Python SDK Usage (`import fabric_defect_hub as fdh`)
```python
import fabric_defect_hub as fdh

# What can I run?
fdh.list_models("anomalib")     # every anomalib model this backend accepts
fdh.list_datasets()             # every registered dataset

# Configure -> train -> infer, all config-driven (same path as `fdh train`)
cfg = fdh.load_config("stfpm", dataset="zju-leaper", num_samples=300, epochs=50)
run = fdh.train(cfg)
out = fdh.predict("stfpm", weights=run.result.registered_artifact.path, source="sample.jpg")

# Or score a previously published checkpoint against another dataset
weights = fdh.from_pretrained("PatchCore")
metrics = fdh.evaluate("PatchCore", weights=weights, dataset="tilda-400").metrics
```

Prefer to assemble the pieces yourself? The loading strategies and the raw
adapters are still directly available:

```python
dataset = fdh.load_dataset("raw-fabric", root="data/RAW_FABRID", sparse_ratio=0.1, tiling=True)
model = fdh.load_model("ultralytics", "yolov8n", tta_mode="flip_multiscale")
predictions = model.predict(dataset.load_samples())
pro_score = fdh.compute_pro_score(gt_masks, pred_anomaly_maps)
```

### 3. CLI Commands
```bash
# Train a YOLO variant with the shared fabric configuration
fdh train yolov8n

# Score a trained checkpoint against a validation dataset (e.g. the smaller
# TILDA-400 / Fabric Defects sets) without opening the web UI
fdh evaluate patchcore_textile --weights artifacts/models/published/PatchCore.ckpt --dataset tilda-400

# ...add --output-dir to also get pixel-level metrics (pixel_auroc / pixel_aupro / iap);
# without it an anomaly model is scored on image-level metrics alone
fdh evaluate patchcore_textile --weights artifacts/models/published/PatchCore.ckpt \
    --dataset tilda-400 --output-dir artifacts/anomaly_maps/eval

# List every model this project can run, grouped by backend
fdh models
fdh models --backend anomalib

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

- � **[Interface Contract Spec](docs/interface.md)**: 系统契约、后端适配器接口、面向用户和 Web 层的分层规则。
- ⚙️ **[Model Configuration](docs/config.md)**: config priority order, per-backend YAML fields, the config-profile (`recipe_id`) table, and validating on a held-out dataset.
- 🌱 **[Add New Model / Dataset](docs/add_new.md)**: how to implement a new backend or dataset adapter, plus the `pytest -m architecture` review guard.
- 🧾 **[Report & Result Schema](docs/report.md)**: run logging, benchmark reporting, and expected JSON schema outputs.
- 🎥 **[Video Examples](docs/videos/)**: recorded walkthroughs and example runs for the UI and model pipelines.
- 📓 **[Training notebook](notebooks/train_on_gpu.ipynb)**: the same `fdh.load_config`/`train`/`evaluate` path, one stage per cell, for interactive single-model work on a GPU box.

---

## 📜 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
