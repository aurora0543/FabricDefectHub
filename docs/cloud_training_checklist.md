# Cloud Completion Checklist

This document is only for the four canonical models that do **not** have a
published project weight on the current local machine. The other 16 canonical
models already have published artifacts and must not be uploaded to, or
retrained on, the cloud merely to recreate them.

## Scope

| Model | Why it is missing locally | Cloud data | Base weight behavior |
| --- | --- | --- | --- |
| EfficientAD | Needs a separate natural-image regularization set | ZJU-Leaper + Imagenette | Anomalib/timm downloads or reuses its pretrained teacher |
| WinCLIP | Training-free, no learned ZJU checkpoint | ZJU-Leaper test set | OpenCLIP downloads or reuses its cache at first inference |
| MoECLIP | Needs an auxiliary zero-shot training corpus | MVTec AD + ZJU-Leaper | **Manual** `ViT-L-14-336px.pt` placement required |
| MambaAD | High-resource model; no accepted published baseline yet | ZJU-Leaper | timm downloads or reuses its pretrained ResNet teacher |

`MambaAD` remains a required attempted run. A CUDA out-of-memory or an
implementation/runtime failure is recorded as a failed run with its log; it
is not silently omitted from the batch.

## 1. What to upload

Upload or clone only the source repository. The repository's `.gitignore`
already excludes `data/`, `artifacts/`, `runs/`, `results/`, and all common
weight/export formats (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.engine`). Do
not upload the 16 local published weights or your local `artifacts/` tree.

On the cloud machine, manually stage these paths (copies, mounts, or symlinks
are all valid):

```text
data/ZJU-Leaper/                         # ZJU images + annotations + masks
data/MVTec AD/                           # MVTec AD train/test/ground_truth
data/imagenette/                         # Imagenette train/val directories
components/moeclip/model/ViT-L-14-336px.pt
```

The MoECLIP checkpoint is the only required manually placed model weight.
WinCLIP and MambaAD may download their pretrained backbones at first run, so
give the cloud machine network access or pre-populate its model cache.

## 2. Prepare and validate prerequisites

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-full.txt

export ZJU_LEAPER_ROOT="$PWD/data/ZJU-Leaper"
export MVTEC_AD_ROOT="$PWD/data/MVTec AD"
export IMAGENET_DIR="$PWD/data/imagenette"

python tools/preflight_cloud_models.py
```

The preflight command must report four `ready` models before a real run. It
checks the selected configs, required datasets, installed backend packages,
and MoECLIP's manually supplied backbone. It does not download weights or
start training.

## 3. Run every missing model

Use separate commands so every model has a clear log and an independent
success/failure record. The ZJU selections in the configs are pattern1-4,
with 1,200 train and 400 test samples total (300/100 per pattern requested).

```bash
mkdir -p artifacts/training_runs/cloud_missing_models/logs
set -o pipefail

fdh train configs/models/anomalib_efficientad.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  --set train.model_kwargs.imagenet_dir="$IMAGENET_DIR" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/EfficientAD.log

fdh train configs/models/anomalib_winclip.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/WinCLIP.log

fdh train configs/models/moeclip_example.yaml \
  --dataset-root "$MVTEC_AD_ROOT" \
  --test-dataset-root "$ZJU_LEAPER_ROOT" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/MoECLIP.log

fdh train configs/models/mambaad_example.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  --num-samples 1200 --val-num-samples 400 --use-defect --defect-ratio 0.5 \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/MambaAD.log
```

The commands intentionally do not use `fdh train-all`: that command would
re-run all 20 catalog entries, including the 16 models already completed
locally. Run MambaAD even if it fails; preserve its log for the project record.

## 4. What is created and what to download

Each successful `fdh train` writes a registered weight under
`artifacts/models/`, publishes the four missing canonical slots when applicable,
writes `artifacts/models/weight_manifest.jsonl`, and writes model-specific run
files, metrics, and anomaly maps under `runs/` / `artifacts/anomaly_maps/`.
The `tee` commands also guarantee one raw cloud log per model.

After all four attempts, package only the newly generated cloud artifacts:

```bash
python tools/collect_cloud_artifacts.py \
  --run-id cloud_missing_models --archive
```

Download the single output file:

```text
artifacts/cloud_runs/cloud_missing_models.tar.gz
```

It contains project-generated weights, manifests, configs, logs, metrics,
curves, anomaly maps, environment information, and SHA-256 checksums. It does
not include your datasets or the manually staged MoECLIP backbone, by design.
