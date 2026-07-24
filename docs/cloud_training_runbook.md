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

Check the summary at the end; every one of the 14 models should say `OK`.
If one fails, re-run just that model after fixing the issue:

```bash
python tools/train_all_models.py --only <model-key> --mode test
# see every key: python tools/train_all_models.py --list-keys
```

## 6. Run the real training pass

```bash
python tools/train_all_models.py
```

No `--mode` flag: each model trains with its own config's declared setting
("few" — 300/100 train/val images, drawn evenly from ZJU-Leaper patterns
1-4 only). Two other shot modes are available, both widening that to every
one of the benchmark's 19 patterns:

```bash
python tools/train_all_models.py --mode medium   # every pattern, capped at 150/50 images each (2850/950 total)
python tools/train_all_models.py --mode full     # every pattern, every image (tens of thousands — slow)
```

"medium" is the practical choice for real cross-texture generalization
without "full"'s runtime; "full" is there for a from-scratch or final
run where training time isn't the constraint.

This can be significantly slower, especially for Anomalib's PatchCore
(coreset selection scales with total image count, not just epochs).

The script continues past a failing model instead of aborting the whole
batch, and prints a final `OK`/`FAIL` summary — safe to re-run to retry
only what failed.

## 7. Confirm weights landed in the fixed location

```bash
ls -la artifacts/models/published/
```

Expect 14 files: `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`,
`fasterrcnn_resnet50_fpn.pt`, `cascadercnn_resnet50_fpn.pt`,
`detr_resnet50.pt`, `maskrcnn_resnet50_fpn.pt`, `unetplusplus_resnet34.pt`,
`deeplabv3plus_resnet50.pt`, `PatchCore.ckpt`, `PaDiM.ckpt`, `RD4AD.ckpt`,
`EfficientAD.ckpt`, `SuperSimpleNet.ckpt`.

This is the stable location the frontend's `MODEL_CATALOG`
(`web/single_image.py`) reads from — re-running training for any one
model overwrites just that model's file here, nothing else.

## Known gaps at time of writing

- Cascade R-CNN, DETR, UNet++, and DeepLabV3+ have not been run end to
  end on a real GPU host yet — they were wired up and verified by tracing
  the adapter's own dataset-class-selection logic, not by a live run.
  Watch these four in the `--mode test` smoke pass.
