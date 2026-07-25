# Cloud training runbook

How to train every model this project supports (see
`src/fabric_defect_hub/catalog.py`) on a cloud GPU host, and get the
weights into the fixed location the frontend reads from
(`artifacts/models/published/`).

This covers *training only*. The frontend UI (`fdh-ui`) is a separate,
later step — see [README.md](../README.md) for that.

## Prerequisites

- A cloud host with a CUDA GPU and this repository cloned onto it.
- ZJU-Leaper (and, if you also want to train on the other two datasets,
  RAW_FABRID / MVTec AD) staged somewhere on the host's disk.
- An Imagenette/ImageNet-style natural-image directory for EfficientAD's
  regularization loss, plus VisA (or MVTec AD/LOCO) with pixel masks if you
  want the MoECLIP training result. These are model-specific prerequisites,
  not substitutes for the fabric benchmark datasets.

## 1. Pull the code

```bash
cd /path/to/FabricDefectHub
git fetch origin
git checkout feat/raw-fabric-dataset   # or main, once merged
git pull
```

## 2. Activate the environment and install dependencies

```bash
conda activate fdh_env
pip install -r requirements-full.txt
```

Only needed again later if `requirements-full.txt`/`pyproject.toml` change
— it ends with an editable install of this project (`-e .`), which picks up
code changes automatically without reinstalling.

The full set includes the optional VLM package (`open-clip-torch`) required by
WinCLIP and the vendored-model imports required by Dinomaly and MoECLIP. It
does not remove model-specific runtime prerequisites: EfficientAD still needs
an explicit natural-image `imagenet_dir`, and MambaAD may require a smaller
batch/image size or an otherwise idle GPU when CUDA memory is exhausted.

## 3. Stage datasets under `data/` as symlinks

`fdh train` resolves each dataset's root automatically from
`data/<Dataset>` (see `training.DEFAULT_DATASET_ROOTS`) — **no environment
variable needed** for this. Symlink whatever real storage the data lives
on into place:

```bash
ln -s /actual/path/to/ZJU-Leaper data/ZJU-Leaper
# optional, only if training against these too:
ln -s /actual/path/to/RAW_FABRID data/RAW_FABRID
ln -s "/actual/path/to/MVTec AD" "data/MVTec AD"
```

Verify:

```bash
ls -la data/ZJU-Leaper   # should show Images/ Annotations/ ImageSets/
```

## 4. Set the HuggingFace mirror (Anomalib only)

Anomalib downloads its backbones (`wide_resnet50_2`, `resnet18`, ...) from
`huggingface.co` — confirmed unreachable from at least one China-based
cloud host used for this project (`Network is unreachable`). Point at a
mirror before training any Anomalib model:

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

(One-time, persists across sessions on this host. If you're on a host
that reaches `huggingface.co` directly, this isn't necessary.)

Ultralytics (GitHub releases) and torchvision (`download.pytorch.org`)
don't have a known blocker on this host, but haven't been confirmed
reachable either — if a training run hangs or fails on a download, that's
the first thing to check.

## 5. Smoke test everything first

Before committing to a full training pass, run every model for 1
epoch / 8 images to confirm the whole pipeline — data loading, model
construction, checkpointing, publishing — works end to end:

```bash
python tools/train_all_models.py --mode test
```

Each model is launched in its own Python process, so CUDA memory is released
before the next model begins. Check the summary at the end; only models whose
external prerequisites have been staged should be expected to say `OK`.
If one fails, re-run just that model after fixing the issue:

```bash
python tools/train_all_models.py --only <model-key> --mode test
# see every key: python tools/train_all_models.py --list-keys
```

For the remaining model-specific setup, set an actual path in the
EfficientAD config's `train.model_kwargs.imagenet_dir`, and stage VisA under
`data/VisA` before running MoECLIP. Dinomaly requires every selected defect
test image to expose an `anomaly_mask`; the preflight error prints the sample
IDs that need correction.

## 6. Run the real training pass

```bash
python tools/train_all_models.py
```

No `--mode` flag: each model trains with its own config's declared setting
("few" — 300/100 train/val images, drawn evenly from its configured
ZJU-Leaper pattern subset; the YOLO textile recipes use patterns 1-4).
The mode changes only the sample budget — it does not overwrite an explicit
`pattern:` list in the config:

```bash
python tools/train_all_models.py --mode medium   # 150/50 images per configured pattern (600/200 for patterns 1-4)
python tools/train_all_models.py --mode full     # every image in the configured pattern subset
```

"medium" is the practical choice for the four-pattern fabric recipes; it
keeps their intended production-transfer scope while giving each selected
texture 150/50 train/validation images. "full" is there for a final run when
training time is not the constraint. Configs that omit `pattern:` retain the
whole 19-pattern ZJU-Leaper selection (2,850/950 images in medium mode).

This can be significantly slower, especially for Anomalib's PatchCore
(coreset selection scales with total image count, not just epochs).

The script continues past a failing model instead of aborting the whole
batch, and prints a final `OK`/`FAIL` summary — safe to re-run to retry
only what failed.

## 7. Interrupt-safe runs and curves

Give a long run an explicit ID. The runner writes one log per model and
atomically updates its state after a model starts or ends, so a shutdown does
not erase the completed-model record:

```bash
python tools/train_all_models.py --run-id zju-full --mode full
```

After a restart, continue the same queue. Models already marked successful in
`artifacts/training_runs/zju-full/state.json` are skipped; an interrupted
Torchvision model also receives `train.resume=true` and continues from its
last completed epoch checkpoint when available:

```bash
python tools/train_all_models.py --run-id zju-full --resume --mode full
```

Read an individual model's live/previous output at
`artifacts/training_runs/zju-full/logs/<model-key>.log`. Dinomaly and
MoECLIP also preserve a per-iteration `history.csv` alongside their stable
registered artifact. They do not yet preserve optimizer state mid-run, so an
interrupted Dinomaly/MoECLIP/MambaAD attempt restarts that one model while
the queue still skips every completed model.

YOLO writes `results.csv` and Torchvision writes `history.csv` at each
completed epoch. Render all discovered curves as SVG without a separate
plotting dependency:

```bash
python tools/plot_training_curves.py runs artifacts/models artifacts/training_runs/zju-full \
  --output-dir artifacts/training_curves/zju-full
```

## 8. Weight provenance manifest

Every successful `fdh train` run appends one immutable JSON record to
`artifacts/models/weight_manifest.jsonl`. The record distinguishes the
run-specific registered artifact (`artifacts/models/...`) from the optional
frontend-facing published copy (`artifacts/models/published/...`), and stores
the resolved config snapshot, source config path, parameter count, metrics,
file size, model/backend/variant, and batch run ID. The adjacent
`artifacts/models/records/*.config.json` files are the exact resolved
configuration snapshots; do not delete them while retaining their manifest
records.

For example, inspect the most recently appended record:

```bash
tail -n 1 artifacts/models/weight_manifest.jsonl | python -m json.tool
```

## 9. Confirm weights landed in the fixed location

```bash
ls -la artifacts/models/published/
```

Expect 18 files: `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`,
`fasterrcnn_resnet50_fpn.pt`, `cascadercnn_resnet50_fpn.pt`,
`detr_resnet50.pt`, `maskrcnn_resnet50_fpn.pt`, `unetplusplus_resnet34.pt`,
`deeplabv3plus_resnet50.pt`, `PatchCore.ckpt`, `PaDiM.ckpt`, `RD4AD.ckpt`,
`EfficientAD.ckpt`, `SuperSimpleNet.ckpt`, `WinCLIP.ckpt`, `Dinomaly.pth`,
`MoECLIP.pth`, `MambaAD.pth`.

This is the stable location the frontend's `MODEL_CATALOG`
(`web/single_image.py`) reads from — re-running training for any one
model overwrites just that model's file here, nothing else.

## Known gaps at time of writing

- Cascade R-CNN, DETR, UNet++, and DeepLabV3+ have not been run end to
  end on a real GPU host yet — they were wired up and verified by tracing
  the adapter's own dataset-class-selection logic, not by a live run.
  Watch these four in the `--mode test` smoke pass.
