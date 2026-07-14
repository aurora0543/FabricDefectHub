# FabricDefectHub

A unified defect-detection benchmark platform for real-world fabric quality inspection.

> [!IMPORTANT]
> This project is currently in the design and early-development stage. The README describes the target architecture and implementation roadmap; interfaces and directory layout will stabilize progressively as the first working end-to-end loop lands.

## Project Vision

FabricDefectHub does not reimplement models — instead, it integrates three defect-detection paradigms on top of a unified dataset, model, evaluation, and on-device performance interface:

| Backend | Representative models | Use case |
| --- | --- | --- |
| `ultralytics` | YOLOv8n/v8s, YOLO11n | Defect annotations available, targeting real-time on-device detection |
| `torchvision` | Faster R-CNN, Mask R-CNN (ResNet50-FPN v1/v2) | Two-stage detection/instance-segmentation comparison under labeled conditions |
| `anomalib` | PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet | Few or no defect samples — normal-only or low-shot anomaly data |

The project's main work is focused on:

- Uniformly adapting different datasets, tasks, and algorithm frameworks;
- Standardizing accuracy and industrial-metric evaluation;
- Fair benchmarking across inference backends such as PyTorch, ONNX Runtime, and TensorRT;
- Supporting device, working-condition, and under-/over-detection analysis for real fabric production lines;
- Connecting the experiment backend and visualization frontend through a unified result contract.

The current formal benchmark dataset scope is focused on ZJU-Leaper; other public datasets are not being expanded for now. Enterprise or custom data can be onboarded through the unified `Sample` JSON contract or the COCO-conversion entry point in `tools/convert_annotations.py`.

## Development Principles

The project follows a "frontend prototype first, but close the real loop as fast as possible" approach:

1. Build a clickable frontend prototype using mock data while nailing down the data contract;
2. Close the minimal end-to-end loop: `YOLOv8n + one dataset + PC inference + result JSON`;
3. Have the frontend read real experiment JSON to validate whether the contract holds up;
4. Then bring in Anomalib, MMDetection, ONNX/TensorRT, and on-device testing in sequence.

The frontend serves the unified benchmark design — it is not developed long-term as a standalone project detached from the real training/inference/evaluation pipeline.

## Frontend Prototype

Phase 0 implements the following pages using mock data:

- **Dataset page**: dataset name, task type, sample count, classes, annotation format, and working-condition info;
- **Model page**: YOLO, MMDetection, and Anomalib model cards, showing task capability, parameter count, and on-device export support;
- **Experiment config page**: select dataset, model, device, input size, and precision mode;
- **Results page**: leaderboard by mAP/AUROC, P50 latency, FPS, memory, power, and model size;
- **Experiment detail page**: confusion matrix, PR curve, defect visualization, and per-sample predictions.

Mock data must follow the same interface as the real backend, to avoid rewriting pages later when real models are wired in.

## Core Architecture

```text
DatasetAdapter
  Various public/enterprise datasets -> unified internal sample description

ModelAdapter
  Ultralytics / MMDetection / Anomalib -> unified train/predict/export interface

Evaluator
  Task + model capability -> mAP, F1, AUROC, AUPRO, mIoU, etc.

BackendProfiler
  PyTorch / ONNX Runtime / TensorRT -> latency, FPS, memory, power, model size
```

### DatasetAdapter

Different datasets share a unified `Sample` metadata shape, but keep each task's native label semantics rather than forcing every dataset into one annotation format:

| Task | Label fields |
| --- | --- |
| Object detection | `boxes`, `labels` |
| Instance/semantic segmentation | `masks`, `labels` |
| Anomaly detection | `is_anomalous`, optional `anomaly_mask` |

Conceptual structure:

```json
{
  "id": "sample-0001",
  "image_path": "data/images/0001.jpg",
  "task": "detection",
  "annotations": {
    "boxes": [[120, 64, 238, 180]],
    "labels": ["broken_end"]
  },
  "metadata": {
    "fabric_type": "cotton",
    "lighting": "line_scan"
  }
}
```

### ModelAdapter

Each model backend is onboarded through a unified lifecycle, while still letting each backend keep its own config:

```text
train(config) -> Artifact
predict(samples, artifact) -> list[Prediction]
export(artifact, target) -> ExportedArtifact
```

The unified `Prediction` is made up of optional fields:

```json
{
  "sample_id": "sample-0001",
  "boxes": [[121, 66, 236, 178]],
  "labels": ["broken_end"],
  "scores": [0.93],
  "masks": null,
  "anomaly_score": null,
  "anomaly_map": null
}
```

YOLO and Faster R-CNN fill `boxes`/`labels`/`scores`; Mask R-CNN additionally fills `masks`; PatchCore, PaDiM, and similar models fill `anomaly_score` and `anomaly_map`.

### Evaluator

The evaluator picks metrics based on the dataset's task, label availability, and model capability, rather than a single `accuracy` figure:

| Type | Suggested metrics |
| --- | --- |
| Object detection | mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 |
| Segmentation | mIoU, Dice, pixel-level F1 |
| Anomaly detection | image-/pixel-level AUROC, AUPRO, F1 |
| Industrial inspection | under-detection rate, over-detection rate, alarms per unit fabric length |

### BackendProfiler

Performance testing records the run environment and measurement methodology, so results across models stay comparable:

- P50/P95 latency, throughput, and FPS;
- Peak RAM/VRAM, power draw, and model file size;
- Device, runtime, precision mode, input size, batch size, and warmup count;
- Runtime info for PyTorch, ONNX Runtime, TensorRT, etc.

## Unified Experiment Result Contract

Training, inference, evaluation, and performance testing all roll up into a unified `ExperimentResult`. The following is a minimal example; the formal schema will be finalized alongside the first end-to-end loop:

```json
{
  "experiment_id": "exp-2026-001",
  "model": {
    "name": "yolov8n",
    "backend": "ultralytics",
    "task": "detection"
  },
  "dataset": {
    "name": "fabric-demo",
    "split": "test"
  },
  "runtime": {
    "device": "Jetson Orin Nano",
    "engine": "TensorRT",
    "precision": "fp16",
    "input_size": [640, 640]
  },
  "metrics": {
    "map50": 0.81,
    "latency_ms_p50": 12.4,
    "fps": 80.6
  },
  "artifacts": {
    "predictions": "artifacts/exp-2026-001/predictions.json",
    "model": "artifacts/exp-2026-001/model.engine"
  }
}
```

When a new backend is added, as long as it can produce predictions and experiment results matching the contract, the frontend and leaderboard need no framework-specific rewrite.

## Planned Directory Structure

The following directories are created progressively as their corresponding phase lands; not all of them exist yet:

```text
FabricDefectHub/
├── frontend/              # Dataset, model, experiment, and result visualization
├── configs/               # Dataset, model, runtime, and experiment configs
├── data/                  # Local data (excluded from version control by default)
├── schemas/               # Sample, Prediction, ExperimentResult schemas
├── src/fabric_defect_hub/
│   ├── datasets/          # DatasetAdapter and dataset implementations
│   ├── models/            # ModelAdapter and the three framework backends
│   ├── evaluation/        # Task metrics and industrial metrics
│   └── profiling/         # Multi-runtime, on-device performance testing
├── tools/                 # Data conversion, export, and visualization tools
└── tests/                 # Unit tests and minimal end-to-end integration tests
```

## Roadmap

### Phase 0: Contract and clickable prototype

- [ ] Define `Sample`, `Prediction`, `ExperimentResult` JSON schemas;
- [ ] Build the five core pages against one shared mock contract;
- [ ] Nail down capability and metric mapping for detection, segmentation, and anomaly detection.

### Phase 1: Minimal real end-to-end loop

- [ ] Onboard one public or de-identified fabric dataset;
- [ ] Get Ultralytics `YOLOv8n` running PC inference;
- [ ] Emit real predictions and experiment-result JSON;
- [ ] Show a real leaderboard and experiment detail in the frontend.

### Phase 2: Unified algorithm benchmark

- [ ] Onboard representative Anomalib models and anomaly-detection metrics;
- [ ] Onboard representative MMDetection models and detection/segmentation metrics;
- [ ] Complete the unified train/predict/evaluate/artifact-management pipeline.

### Phase 3: Deployment and industrial evaluation

- [ ] Support ONNX Runtime, TensorRT, and precision-mode switching;
- [ ] Run reproducible performance tests on PC and target devices like Jetson;
- [ ] Add power draw, model size, under-detection rate, and over-detection rate;
- [ ] Support real working-condition metadata and enterprise-data adaptation.

## Fair Benchmarking Requirements

When submitting or publishing benchmark results, record at least:

- Dataset version, split method, and preprocessing pipeline;
- Model version, weight source, training config, and random seed;
- Hardware, software stack, inference engine, and precision mode;
- Input size, batch size, warmup count, and sample count used for statistics;
- Metric implementation, thresholds, and post-processing parameters.

A single mAP, AUROC, or FPS number without this context should not be used to rank models directly.

## Command-Line Usage

Once the project is installed, you can launch a single-backend run or a cross-backend benchmark directly from YAML:

```bash
pip install -e ".[dev]"
fdh run configs/models/ultralytics_example.yaml
fdh benchmark configs/benchmark_example.yaml
```

Benchmark configs support dataset/split selection, existing checkpoints, a unified evaluator, optional PyTorch/ONNX Runtime/TensorRT profiling, leaderboard sorting, and CSV/Markdown reports. Paths can use environment variables — e.g. set `ZJU_LEAPER_ROOT` before running the example config.

### Unified training entry point: `fdh train`

`fdh train <model>` is the unified training entry point: it reads a model config file (`configs/models/*.yaml`), auto-detects the backend from keywords in it (`model.name` -> Anomalib; `model.variant` starting with `yolo`/`fasterrcnn`/`maskrcnn` -> Ultralytics/torchvision; or specify it explicitly via `--backend` or a top-level `backend:` key in the config), then runs that backend's full train/val/export pipeline — with no extra flags, this is exactly equivalent to `fdh run`.

`<model>` doesn't need to be a full path — three forms work (in priority order):

```bash
fdh train configs/models/ultralytics_example.yaml   # full/custom path
fdh train ultralytics_example                        # a filename under configs/models/ (.yaml is optional)
fdh train yolov8n                                    # keyword: matched against every config's
                                                      # model.variant / model.name under configs/models/ (case-insensitive)
```

Keyword/filename lookup searches `configs/models/` by default; change the directory with `--config-dir`. If you're not sure what configs are available, run `fdh train --list` to list every resolvable config under that directory. A keyword matching multiple files, or matching none, raises an error listing the candidates.

On top of that, you can optionally override the dataset and the "shot" mode without hand-editing the YAML:

```bash
# Swap in a different registered dataset/path (any <model> form works)
fdh train ultralytics_example \
  --dataset raw-fabric --dataset-root /data/RAW_FABRID

# full-shot: use every image in that split (equivalent to num_samples: null)
fdh train patchcore --mode full

# few-shot: leave the sample count already declared in the config untouched
# (this is also the default behavior — --mode few is equivalent to omitting --mode)
fdh train patchcore --mode few

# test-shot: run the full pipeline once with just 8 images, to verify the
# model+dataset combination actually works end to end (this also caps epochs
# to 1 — it's meant to verify the pipeline, not produce a usable model)
fdh train patchcore --mode test

# List every config currently resolvable under configs/models/
fdh train --list
```

The remaining flags — `--num-samples` / `--val-num-samples` / `--use-defect` / `--no-use-defect` / `--defect-ratio` / `--pattern` / `--category` / `--seed` — map to the identically-named fields in each dataset's `DatasetAdapter` constructor, and only override what's explicitly passed. Anomalib's five models all train one-class, so their training split always has `use_defect: false` forced on it (`--use-defect` only affects the test split's defect mix) — this is the training logic automatically deciding based on the backend, and needs no special handling on the command line.

Setting `offline: true` in a model config forbids implicit downloads, checking explicit weights, `FDH_MODEL_CACHE`, the project's model directory, and common framework caches before entering the framework's own training code — a clear caching suggestion is raised immediately if nothing is found.

On Python 3.14+, torchvision prefers `exported_program` (`.pt2`/`torch.export`); `torchscript` remains only as a compatibility target for older environments and operators not yet supported by `torch.export`.

See [VALIDATION.md](VALIDATION.md) for how to validate each backend locally, in the cloud, and across power-measurement platforms.

## Gradio Workspace

Install the `ui` extra and run `fdh-ui`, or use the root-level `app.py` on Hugging Face Spaces. The first tab, **Single Image Detection**, supports random sampling from ZJU-Leaper, paging between images left/right, choosing a model/checkpoint, and displaying bbox, mask, or anomaly-map inference results. See [frontend/README.md](frontend/README.md) for details.

### Cross-Platform Power Measurement

Profiling auto-detects the available power sensor by default, writing `power_w_mean`, `power_w_peak`, and `energy_j` into the result, and records the measurement source, scope, and reason for unavailability (if any) into `power.json`.

- Cloud NVIDIA: NVML, measures GPU power; install `.[profiling-power-nvidia]`.
- Jetson: `tegrastats`, measures board-level `VDD_IN` input power.
- macOS: `powermetrics` SMC sampling, measures package power; requires non-interactive `sudo` authorization.
- Raspberry Pi: requires an external sensor (INA219/INA226, etc.), pointed at via `power_sensor_path` to the sysfs power file.

Setting `power_mode: required` makes profiling fail explicitly when the sensor or permissions aren't available; `auto` (default) keeps the latency metrics and records the reason power couldn't be measured in `power.json`.

## License

This project is licensed under the [MIT License](LICENSE). Third-party frameworks, model weights, and datasets remain subject to their own licenses and terms of use.

## Contact

This project was initiated by, and is under active development at, a research group at Beijing Institute of Technology. For collaboration, dataset adaptation, or benchmark submission suggestions, please reach out to the maintainers via an Issue.
