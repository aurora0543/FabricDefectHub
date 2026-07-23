# FabricDefectHub

FabricDefectHub is a unified benchmarking hub for classical fabric defect detection models, developed for real-world industrial textile quality inspection. It consolidates SOTA fabric defect datasets within a full-stack system architecture designed to streamline benchmarking and facilitate practical deployment.

For data collection, two SOTA fabric defect datasets were assembled, supplemented by an additional dataset targeting general-purpose defect inspection (see [Datasets](#datasets)). With respect to algorithms, a total of 18 models spanning anomaly detection and defect detection are integrated and systematically categorized (see [Models](#models)).

The system architecture comprises a frontend and a backend. The frontend provides a web-based interface supporting single-image inference across datasets and models, together with a dedicated Benchmark module that evaluates all models concurrently and automatically produces a leaderboard. The backend implements the underlying logic required by the frontend and additionally furnishes automation scripts for model training, inference, benchmark scoring, and performance validation.

## Models

18 models across two branches: 9 anomaly detectors (**Anomalib**, the vendored research models under `components/`, and one clean-room reimplementation, see [components/README.md](components/README.md)), and 9 supervised defect detectors/segmenters (**Ultralytics**, **torchvision**).

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
| 15 | WinCLIP | CLIP (vision-language) | Zero-shot (no fabric training) |
| 16 | Dinomaly | DINOv2 encoder-decoder | Zero-shot (unsupervised) |
| 17 | MoECLIP | CLIP + LoRA mixture-of-experts | Zero-shot (trained on labelled defects) |
| 18 | MambaAD | CNN encoder + Mamba (SSM) decoder | Zero-shot (unsupervised) |

Dinomaly and MoECLIP (16, 17) are vendored research code rather than
pip-installable libraries — they live as git submodules under
`components/` and each has an adapter under
`src/fabric_defect_hub/models/<name>/` that translates between this
project's `Sample`/`Prediction`/`Artifact` contracts and whatever the
upstream repo natively uses. Two extra setup steps apply:
`git submodule update --init --recursive` to populate them, and, for
MoECLIP, dropping the OpenCLIP ViT-L-14-336px checkpoint into
`components/moeclip/model/` (nothing downloads it — the path and link are
in the error the adapter raises).

MambaAD (`models/mambaad/`) is the one entry that is a **clean-room
reimplementation**, not a vendored submodule: the official repo
(`lewandofskee/MambaAD`) is a plugin that only runs inside an ADer
checkout (its own `model.py`/`trainer.py` import ADer's `util`/`data`/
`optim`/`loss` packages, absent from MambaAD's own repo) and its
selective-scan core needs `mamba_ssm`, a CUDA-only compiled kernel that
does not install on this project's dev machine. The published architecture
and recipe are ported directly instead. No extra install step: unlike
Dinomaly/MoECLIP there is no submodule to `git submodule update` and no
manually-placed checkpoint.

Two things about it are worth knowing:

- **It is the only *multi-class unified* model here.** That is the paper's
  actual claim: ONE model trained across every category at once, rather
  than anomalib's one-model-per-category. Train it across the whole corpus
  (`--dataset fabric-train --mode full`, or all 19 ZJU-Leaper patterns) to
  use it as intended.
- **On a CUDA box it uses upstream's real kernel.** `models/mambaad/ssm.py`
  detects `mamba_ssm` at run time and dispatches to its fused
  `selective_scan_fn` when the tensors are on CUDA — exact upstream
  semantics *and* speed, no configuration. Install it with the optional
  `mambaad-cuda` extra on the training host. Without it (CPU/MPS, or CUDA
  without the package) the backend still runs, on portable implementations
  of the same recurrence that are tested against each other for numerical
  equality — a good deal slower, which is fine for development, CI and
  inference but not for a full reproduction run.

MoECLIP is the one model here trained under a **zero-shot (ZSAD)**
protocol, and its data rules are the mirror image of every other anomaly
model's:

- It trains on an **auxiliary cross-domain corpus** (VisA, MVTec AD or
  MVTec LOCO — the sets that are *eval-only* for everything else) and is
  then applied to fabric it has never seen. `data.test_dataset` names that
  zero-shot evaluation target separately from `data.dataset`, and
  `fdh train` **rejects a fabric training corpus** for this backend: it
  would make its fabric scores in-domain and void the transfer claim the
  benchmark measures. Override the target with
  `--test-dataset raw-fabric` / `--test-dataset-root ...`.
- It learns from *labelled* defects rather than one-class, so its train
  split needs `use_defect: true` and a mask-bearing task; defective
  training samples without a pixel mask are dropped and counted on the
  resulting artifact.

Its leaderboard provenance therefore reads `Zero-shot CLIP (VisA-trained)`
rather than `local trained artifact` — a fabric number from MoECLIP is a
transfer result. One caveat when running the Benchmark tab: evaluating it
against the same dataset it was trained on (VisA by default) is not a
zero-shot measurement; pick a fabric dataset there.

## Web UI

We built a Gradio-based UI hub for the benchmark, providing an intuitive interface for the following functions. These capabilities are demonstrated in the two recordings below.

**Single Image Detection** — detection/segmentation/anomaly inference on one image:

[<video src="https://github.com/aurora0543/FabricDefectHub/raw/main/docs/videos/detection.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/5ce67795-705e-4888-9459-d324ca6f4315)

**Benchmark** — scoring multiple models into a leaderboard:

[<video src="https://github.com/aurora0543/FabricDefectHub/raw/main/docs/videos/benchmark.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/f11ec7c1-0bd7-4a97-8ccd-37bb29e7d8e8)

## Datasets

Datasets fall into three roles. Anomaly (one-class) training is restricted
to the **in-domain fabric** sources; the cross-domain object benchmarks are
**eval-only** for those models (training a fabric model on them is rejected
— see `training.ANOMALY_TRAINABLE_DATASETS`); YOLO-labelled sets belong to
the **detection** backends. The `fabric-train` composite unions every
fabric source into one training corpus so no model is trained on a single
dataset.

The zero-shot backend inverts the first two roles: MoECLIP may *only* be
trained on the cross-domain corpora and *not* on fabric (see
`training.ZERO_SHOT_TRAINABLE_DATASETS` and [Models](#models)).

**In-domain fabric (train + eval):**

- **ZJU-Leaper** — 94,833 images (71,127 normal / 23,706 defective), 19 fabric patterns. [Homepage](http://www.qaas.zju.edu.cn/zju-leaper/).
- **RAW-Fabric (RAW_FABRID)** — 709 high-res grayscale images + 204 masks, plus an MVTec-AD-style 256×256 patch set (14,196/4,969/687/687).
- **TILDA-400** — fabric texture patches, `good/` + 4 defect types (hole / oil spot / thread error / objects); image-level labels, no pixel masks.
- **Fabric Defects Dataset** — fabric, `defect free/` + 5 defect classes (hole / stain / lines / vertical / horizontal); hole/vertical/horizontal ship binary pixel masks (the dataset's `_processed` files), lines/stain are image-level only.
- **`fabric-train`** — a composite (not a folder on disk) that unions the four fabric sources above for one-class training.

**Cross-domain benchmarks (eval-only — except as MoECLIP's zero-shot
training corpus, see [Models](#models)):**

- **MVTec AD** — 5,354 images (3,629/1,725), 15 non-fabric categories; cross-domain zero-shot evaluation, not training.
- **MVTec LOCO AD** — 5 categories with logical + structural anomalies; per-image ground-truth mask directories.
- **VisA** — 12 object categories (Normal/Anomaly + pixel masks); cross-domain zero-shot evaluation.

**Detection track (YOLO labels, not anomaly):**

- **SDUST-FDD** — fabric, YOLO bounding-box labels (6 defect classes); feeds the Ultralytics/torchvision detectors, not the one-class anomaly models.

## Quick Start

1. Obtain the pretrained model weights from [ Google Drive folder](https://drive.google.com/drive/folders/1sIe5oP42GyOfaz-ON9FRkQzlCb2NlnNj?usp=drive_link) and place them under `artifacts/models/published/`.

2. Get and place the datasets under `data/`. The datasets can be downloaded from the following links:

- [ZJU-Leaper](http://www.qaas.zju.edu.cn/zju-leaper/)
- [RAW-Fabric](https://data.mendeley.com/datasets/db6g85xsyg/1)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

1. Install the dependencies and launch the Gradio UI:

```bash
conda create -n fdh_env python=3.14 -y
conda activate fdh_env

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

**(b) Prepare datasets and model artifacts.**

- Download the datasets you need and place them below `data/`: [ZJU-Leaper](http://www.qaas.zju.edu.cn/zju-leaper/), [RAW-Fabric](https://data.mendeley.com/datasets/db6g85xsyg/1), or [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Use only ZJU-Leaper and RAW-Fabric for fabric-model training; MVTec AD is reserved for cross-domain zero-shot evaluation.
- For pretrained inference, download the published weights from the [Google Drive folder](https://drive.google.com/drive/folders/1sIe5oP42GyOfaz-ON9FRkQzlCb2NlnNj?usp=drive_link) and place them in `artifacts/models/published/`.
- Select a registered dataset from the command line while developing a model. Start with the small smoke-test mode, then run the full configuration once its data loading succeeds:

```bash
fdh train yolov8n --dataset <dataset-name> --mode test
fdh train configs/models/ultralytics_example.yaml --dataset <dataset-name>
```

Replace `<dataset-name>` with the dataset registry name. The first command exercises the dataset adapter with an 8-image pipeline; the second runs the normal config-driven training, validation, and export workflow. Use `fdh benchmark configs/benchmark_example.yaml` after preparing the required datasets and model artifacts to generate a cross-backend leaderboard.

### Extension Interfaces

FabricDefectHub keeps data loading and model execution behind two adapters. Extensions convert backend-specific inputs and outputs at these boundaries; the CLI, evaluator, UI, and benchmark runner then use the same registry names and unified contracts. All source paths in this section are relative to `src/fabric_defect_hub/`.

| Extension | Interface to implement | What the interface provides | Registry entry |
| --- | --- | --- | --- |
| Dataset | `datasets/base.py` → `DatasetAdapter` | `load_samples()` returns `list[Sample]` in the unified dataset contract | `@register_dataset("<dataset-name>")` |
| Model/backend | `models/base.py` → `ModelAdapter` | `train()`, `predict()`, and `export()` bridge a backend to the unified prediction and artifact workflow | `@register_model("<backend-name>")` |

`Sample` and `Prediction` are defined in `core/types.py`. `Sample` contains `id`, `image_path`, `task`, and `annotations`; `Prediction` contains `sample_id` and only the task-relevant outputs. Detection fills `boxes`, `labels`, and `scores`; segmentation additionally fills `masks`; anomaly detection fills `anomaly_score` and optionally `anomaly_map`. Do not force those tasks into one annotation format.

#### Add a dataset

1. Place the original files under `data/` and create `datasets/my_fabric.py`. The adapter owns all source-format parsing; callers should only see `Sample` objects.
2. Subclass `DatasetAdapter`, set `name`, and return one `Sample` per image. For a detection dataset, the minimal shape is:

```python
# src/fabric_defect_hub/datasets/my_fabric.py
from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter


@register_dataset("my-fabric")
class MyFabricDataset(DatasetAdapter):
    name = "my-fabric"

    def load_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for image_path, boxes, labels in self._read_split(self.root, self.split):
            samples.append(
                Sample(
                    id=image_path.stem,
                    image_path=str(image_path),
                    task="detection",
                    annotations=Annotations(boxes=boxes, labels=labels),
                )
            )
        return samples
```

For segmentation, also fill `Annotations.masks`; for anomaly detection, fill `Annotations.is_anomalous` and, when available, `Annotations.anomaly_mask`.

3. Import the new adapter in `datasets/__init__.py`. `load_dataset()` imports this package to trigger decorators, so omitting this import leaves the registry unaware of the dataset.
4. Add `<dataset-name>: "data/<directory>"` to `DEFAULT_DATASET_ROOTS` in `training.py` if the dataset should work with the default local layout. Then use the registry name in a model YAML's `data.dataset` field or pass it with `--dataset`.
5. Validate loading before a full run:

```bash
fdh train <model-name> --dataset <dataset-name> --mode test
```

Once this succeeds, the dataset is available to normal training and benchmark configurations through its registry name.

#### Add a model

There are two integration paths. Add a **variant** when the model belongs to an existing Ultralytics, torchvision, or Anomalib backend; add a **backend** only when its native train/predict/export lifecycle needs a new adapter.

##### 1. Add a variant to an existing backend

Edit the backend's preset module; no new `ModelAdapter` is needed.

- **Ultralytics:** add a canonical name to `models/ultralytics/presets.py` → `MODEL_VARIANTS`, then add user-friendly spellings to `VARIANT_ALIASES` and optional training overrides to `VARIANT_TRAIN_OVERRIDES`.

```python
# models/ultralytics/presets.py
MODEL_VARIANTS["yolov12n"] = {
    "checkpoint": "yolo12n.pt",
    "architecture": "yolo12n.yaml",
}
VARIANT_ALIASES["yolo12n"] = "yolov12n"
VARIANT_TRAIN_OVERRIDES["yolov12n"] = {"batch": 16}
```

- **torchvision:** add a `MODEL_VARIANTS` record in `models/torchvision/presets.py`. Its `factory`, `weights_enum`, and `task` values are consumed by the existing adapter; use `task: "detect"` or `task: "instance_segmentation"`.

```python
# models/torchvision/presets.py
MODEL_VARIANTS["my_detector"] = {
    "factory": "my_detector",
    "weights_enum": "MyDetector_Weights",
    "task": "detect",
}
VARIANT_ALIASES["my-detector"] = "my_detector"
```

- **Anomalib:** add the public name to `models/anomalib/presets.py` → `MODEL_ALIASES`, and add constructor defaults keyed by the Anomalib class name in `MODEL_PRESETS`.

```python
# models/anomalib/presets.py
MODEL_ALIASES["myanomaly"] = "MyAnomaly"
MODEL_PRESETS["MyAnomaly"] = {"backbone": "resnet18"}
```

Then add or update a YAML under `configs/models/` with the appropriate `backend` and `model.variant` (Ultralytics/torchvision) or `model.name` (Anomalib). `fdh train <variant-name>` resolves against these configuration files, so a preset alone does not create a CLI training entry.

##### 2. Add a new backend

Create `models/<backend-name>/adapter.py`. The three abstract methods are mandatory: `train()` returns an `Artifact`, `predict()` returns exactly one `Prediction` for each input `Sample`, and `export()` returns an `ExportedArtifact`.

```python
# src/fabric_defect_hub/models/my_backend/adapter.py
from typing import Any

from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter


@register_model("my-backend")
class MyBackendAdapter(ModelAdapter):
    backend = "my-backend"

    def __init__(self, name: str = "my-model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.native_model = None

    def train(self, config: dict[str, Any]) -> Artifact:
        weights_path = self._train_with_native_library(config)
        return Artifact(
            path=str(weights_path),
            backend=self.backend,
            metadata={"model_name": self.name},
        )

    def predict(self, samples: list[Sample], artifact: Artifact) -> list[Prediction]:
        self._load_native_weights(artifact.path)
        return [
            Prediction(
                sample_id=sample.id,
                boxes=[[10.0, 20.0, 30.0, 40.0]],
                labels=["defect"],
                scores=[0.95],
            )
            for sample in samples
        ]

    def export(self, artifact: Artifact, target: str) -> ExportedArtifact:
        exported_path = self._export_with_native_library(artifact.path, target)
        return ExportedArtifact(path=str(exported_path), target=target)
```

The literal detection values above are placeholders: replace them with the native model outputs, converted to image-coordinate boxes and aligned to the input `sample.id`. A segmentation backend instead fills `masks`; an anomaly backend fills `anomaly_score` and optionally `anomaly_map`.

Registering the decorator is necessary but not sufficient. Make these additional changes so the backend can be loaded and trained through the project entry points:

1. Add `"my-backend": "fabric_defect_hub.models.my_backend.adapter"` to `_MODEL_BACKEND_MODULES` in `loader.py`. This makes `load_model("my-backend", "my-model")` import the module and trigger `@register_model`.
2. Create the backend's `config.py` and `pipeline.py`, modelled on an existing backend, plus a YAML under `configs/models/`.
3. Add the backend to `_BACKEND_PIPELINE_MODULES`, `_BACKEND_CONFIG_CLASSES`, `_BACKEND_DATA_SELECTIONS`, and `_BACKEND_MODEL_KEY` in `training.py`. These maps tell `fdh train` which pipeline to run, how to parse its YAML, which dataset-selection fields to override, and whether its model selector is `model.name` or `model.variant`.
4. Add the backend name to the `--backend` choices in `cli.py` for both `fdh run` and `fdh train`.
5. If the Web UI should expose the model, add its catalog entry and checkpoint metadata there as a separate UI integration step.

Put any published checkpoint needed for inference in `artifacts/models/published/`, then verify the model with a registered dataset before using it in a benchmark:

```bash
fdh train <model-name> --dataset <dataset-name> --mode test
fdh benchmark configs/benchmark_example.yaml
```

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
