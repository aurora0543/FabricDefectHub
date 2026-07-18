# FabricDefectHub

FabricDefectHub is a unified benchmarking hub for classical fabric defect detection models, developed for real-world industrial textile quality inspection. It consolidates SOTA fabric defect datasets within a full-stack system architecture designed to streamline benchmarking and facilitate practical deployment.

For data collection, two SOTA fabric defect datasets were assembled, supplemented by an additional dataset targeting general-purpose defect inspection (see [Datasets](#datasets)). With respect to algorithms, a total of 14 models spanning anomaly detection and defect detection are integrated and systematically categorized (see [Models](#models)).

The system architecture comprises a frontend and a backend. The frontend provides a web-based interface supporting single-image inference across datasets and models, together with a dedicated Benchmark module that evaluates all models concurrently and automatically produces a leaderboard. The backend implements the underlying logic required by the frontend and additionally furnishes automation scripts for model training, inference, benchmark scoring, and performance validation.

## Models

14 models across two branches: 5 unsupervised, zero-shot anomaly detectors (**Anomalib**), and 9 supervised defect detectors/segmenters (**Ultralytics**, **torchvision**).

| # | Model | Architecture | Setting |
| - | --- | --- | --- |
| 1 | YOLOv8n | CNN (YOLO) | Few-/full-shot (supervised) |
| 2 | YOLOv8s | CNN (YOLO) | Few-/full-shot (supervised) |
| 3 | YOLO11n | CNN (YOLO) | Few-/full-shot (supervised) |
| 4 | Faster R-CNN | R-CNN | Few-/full-shot (supervised) |
| 5 | Cascade R-CNN | R-CNN | Few-/full-shot (supervised) |
| 6 | DETR | Vision Transformer | Few-/full-shot (supervised) |
| 7 | Mask R-CNN | R-CNN | Few-/full-shot (supervised) |
| 8 | UNet++ | FCN | Few-/full-shot (supervised) |
| 9 | DeepLabV3+ | FCN | Few-/full-shot (supervised) |
| 10 | PatchCore | Feature embedding | Zero-shot (unsupervised) |
| 11 | PaDiM | Feature embedding | Zero-shot (unsupervised) |
| 12 | RD4AD | Teacher-Student | Zero-shot (unsupervised) |
| 13 | EfficientAD | Teacher-Student | Zero-shot (unsupervised) |
| 14 | SuperSimpleNet | Feature embedding | Zero-shot (unsupervised) |

## Demo Videos

Two short recordings of the Gradio UI in action.

**Single Image Detection** — detection/segmentation/anomaly inference on one image:

[<video src="https://github.com/aurora0543/FabricDefectHub/raw/main/docs/videos/detection.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/5ce67795-705e-4888-9459-d324ca6f4315)

**Benchmark** — scoring multiple models into a leaderboard:

[<video src="https://github.com/aurora0543/FabricDefectHub/raw/main/docs/videos/benchmark.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/f11ec7c1-0bd7-4a97-8ccd-37bb29e7d8e8)

## Datasets

- **ZJU-Leaper** — 94,833 images (71,127 normal / 23,706 defective), 19 fabric patterns. [Homepage](http://www.qaas.zju.edu.cn/zju-leaper/).
- **RAW-Fabric (RAW_FABRID)** — 709 high-res grayscale images + 204 masks, plus an MVTec-AD-style 256×256 patch set (14,196/4,969/687/687).
- **MVTec AD** — 5,354 images (3,629/1,725), 15 non-fabric categories; used for cross-domain zero-shot evaluation, not training.

## Quick Start

Pretrained checkpoints/weights aren't tracked in this repo — first download them from [this Google Drive folder](https://drive.google.com/drive/folders/1sIe5oP42GyOfaz-ON9FRkQzlCb2NlnNj?usp=drive_link) and place them under `artifacts/models/published/`. Then:

```bash
git clone https://github.com/aurora0543/FabricDefectHub.git && cd FabricDefectHub
pip install -r requirements.txt
fdh-ui
```

## CLI

The same backend is also driven headlessly by three `fdh` subcommands, plus `fdh-ui` for the Gradio UI (a thin client over the same `load`/`predict`/`unload` calls — see the [Gradio Workspace wiki page](https://github.com/aurora0543/FabricDefectHub/wiki/Gradio-Workspace) for session details):

```bash
pip install -r requirements-full.txt
fdh train configs/models/ultralytics_example.yaml   # unified entry point: train/val/export one model, config-driven
fdh run configs/models/ultralytics_example.yaml      # what fdh train reduces to with no extra flags
fdh benchmark configs/benchmark_example.yaml         # cross-backend leaderboard
```

`fdh train` also resolves a model by filename or keyword (`fdh train yolov8n`), and can override dataset/shot-mode without touching the YAML (`--mode test` for an 8-image pipeline smoke check, `--dataset`, `--num-samples`, ...). Post-training quantization (fp16 / INT8) and TensorRT engine building live in `tools/export_model.py` for edge deployment. Full flag reference: **[CLI Usage](https://github.com/aurora0543/FabricDefectHub/wiki/CLI-Usage)** and **[Edge Deployment](https://github.com/aurora0543/FabricDefectHub/wiki/Edge-Deployment)** on the wiki.

## Development

**(a) Install the full requirements** (covers training/benchmarking across all three backends):

```bash
pip install -r requirements-full.txt
```

**(b) Add a new model.**
- To add a variant to an existing backend, register it in that backend's presets module: `models/torchvision/presets.py`'s `MODEL_VARIANTS` maps a friendly name to its factory function and weights enum, and `models/anomalib/presets.py`'s `MODEL_ALIASES` / `MODEL_PRESETS` do the same for Anomalib.
- To add an entirely new backend, implement the `ModelAdapter` abstract base class (`models/base.py`) — `train()`, `predict()`, `export()` — and register it with `@register_model("<backend-name>")` (`core/registry.py`).

**(c) Add a new dataset.** Implement the `DatasetAdapter` abstract base class (`datasets/base.py`) — a single `load_samples()` method returning a unified list of `Sample` objects — and register it with `@register_dataset("<dataset-name>")` (`core/registry.py`). Once registered, it is resolvable by name via `fdh train --dataset <dataset-name>`.

## Learn More

| Wiki page | Covers |
| --- | --- |
| [Architecture](https://github.com/aurora0543/FabricDefectHub/wiki/Architecture) | Project vision, `DatasetAdapter`/`ModelAdapter`/`Evaluator`/`BackendProfiler` design, unified JSON contracts, directory layout |
| [CLI Usage](https://github.com/aurora0543/FabricDefectHub/wiki/CLI-Usage) | `fdh run`/`train`/`benchmark`, all flags and examples |
| [Gradio Workspace](https://github.com/aurora0543/FabricDefectHub/wiki/Gradio-Workspace) | UI pages and the inference-session mechanism |
| [Edge Deployment](https://github.com/aurora0543/FabricDefectHub/wiki/Edge-Deployment) | Quantization (fp16/INT8) and cross-platform power measurement |
| [Roadmap & Fair Benchmarking](https://github.com/aurora0543/FabricDefectHub/wiki/Roadmap-and-Benchmarking) | Phased roadmap and what a published benchmark result must report |

## License

This project is licensed under the [MIT License](LICENSE). Third-party frameworks, model weights, and datasets remain subject to their own licenses and terms of use.
