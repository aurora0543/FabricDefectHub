# AnomalyDiffusion: vendoring status and integration plan

`components/anomalydiffusion` vendors [sjtuplayer/anomalydiffusion](
https://github.com/sjtuplayer/anomalydiffusion) (AAAI 2024,
arXiv:2312.05767) through our fork `aurora0543/anomalydiffusion`, following
the `components/README.md` convention. This page records what the model is
to this project, why it deliberately has **no backend under
`src/fabric_defect_hub/models/`**, and the concrete steps left to make it
useful.

## What it is (and is not)

AnomalyDiffusion is a few-shot **anomaly image generator**: given a handful
of real defect crops plus their pixel masks, it learns a disentangled
(appearance, location) embedding over a pretrained text-to-image latent
diffusion model, then synthesizes new defect image–mask pairs. Upstream
uses those pairs to train a downstream U-Net detector
(`train-localization.py`).

It is therefore **not a detector** and does not fit the frozen
`ModelAdapter` train/predict contract — its output is a dataset, not a
prediction. Wiring it in as a fake anomaly backend would put a generator on
the detection leaderboard, which is exactly the kind of category error the
catalog rules exist to prevent.

## Why no adapter yet (two hard constraints)

1. **Role mismatch (above).** Its natural place in this project is *in
   front of* the existing training pipelines: generate defect image–mask
   pairs for defect-scarce fabric classes, stage them as a dataset the
   existing `DatasetAdapter`s can load, and let every real detector train
   on the enriched pool. That is a data-augmentation producer, a role the
   frozen interfaces don't model yet (C-line follow-up if we adopt it).
2. **Environment pin conflict.** Upstream pins `torch 1.12 /
   pytorch-lightning 1.5.0 / torchmetrics 0.6` (its `environment.yaml`) —
   irreconcilable with this project's anomalib 2.5 / lightning 2.x env. It
   must run as a subprocess in its own conda env (upstream's
   `Anomalydiffusion` env), never be imported in-process. This is the same
   reason a `vendor.py` import boundary would not help here: there is
   nothing importable from our env.

## Cloud execution plan (GPU box, in order)

All steps run on the training host, inside upstream's own env:

```bash
cd components/anomalydiffusion
conda env create -f environment.yaml && conda activate Anomalydiffusion
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

1. **Stage fabric data in MVTec layout.** Upstream's entry points take
   `--mvtec_path` and expect `<class>/train|test/<defect_type>/` with
   `ground_truth` masks. ZJU-Leaper has pixel masks, so
   `datasets/zju_leaper.py` samples can be exported into that layout
   (small `tools/` script, to be written when this step is reached —
   against the *real* directory listing, not assumed here).
2. **Train the generator** (textual inversion + spatial encoder), upstream
   command verbatim:
   ```bash
   python main.py --spatial_encoder_embedding --data_enhance \
     --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
     --actual_resume models/ldm/text2img-large/model.ckpt -n fabric --gpus 0, \
     --init_word anomaly --mvtec_path=<staged_fabric_root>
   ```
3. **Train the mask generator and sample pairs**: `python run-mvtec.py
   --data_path=<staged_fabric_root>`.
4. **Fold the generated pairs back in** as a flat-folder dataset
   (`datasets/flat_folder.py` handles image+mask trees) and rerun the
   affected detectors' training with the enriched pool; compare
   leaderboard deltas against the un-augmented baselines. Only the *delta
   measurement* makes this component worth keeping — if augmentation does
   not move fabric AUROC/mAP, the honest outcome is recording that.

## Status

- [x] Fork + submodule pinned (`b85297f`, upstream master as of 2026-07).
- [ ] Fabric-to-MVTec staging exporter (`tools/`), blocked on GPU host time.
- [ ] Generator/mask training on cloud (needs ~24GB GPU; SeetaCloud box).
- [ ] Augmentation-delta benchmark vs. un-augmented baselines.
