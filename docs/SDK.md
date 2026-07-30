# Native Python SDK (`fabric_defect_hub`)

Using this project from another codebase. Two paths, in the order you should
reach for them:

1. **The front door** (§1) — `fdh.load_config` / `train` / `predict` /
   `evaluate` / `from_pretrained`. Config-driven, one call per step. This is
   what `fdh train` itself runs, so anything reproducible from the CLI is
   reproducible from Python.
2. **Composing a run by hand** (§3) — `load_dataset` + `load_model` +
   `run_experiment`, for when you need to assemble the pieces yourself.

Both sit on the same frozen contracts (§2). See
[`INTERFACE_SPEC.md`](INTERFACE_SPEC.md) for those.

---

## 1. The front door

```python
import fabric_defect_hub as fdh
```

### Discover

```python
fdh.list_models()                 # {backend: [variant, ...]} across all 6 backends
fdh.list_models("anomalib")       # ['AnomalyDINO', 'Draem', 'Dsr', 'EfficientAd',
                                  #  'Fastflow', 'Ganomaly', 'Glass', 'Padim',
                                  #  'Patchcore', 'ReverseDistillation', 'Stfpm',
                                  #  'Supersimplenet', 'UniNet', 'WinClip']
fdh.list_datasets()               # ['fabric-defects', 'fabric-train', 'mvtec-ad', ...]
fdh.list_pretrained()             # catalog keys `from_pretrained` accepts
fdh.list_pretrained(available_only=True)   # ...of those, the ones trained on this machine
```

These read each backend's own preset module and the dataset registry, so they
reflect what is actually registered — a new alias in
`models/anomalib/presets.py::MODEL_ALIASES` shows up here with no other edit.

### Configure

```python
cfg = fdh.load_config(
    "stfpm",                      # a model keyword, config stem, or config path
    dataset="zju-leaper",
    num_samples=300,
    epochs=50,
)
```

`load_config` resolves the model the same way `fdh train` does — a bare name
lands on that backend's general-purpose config, a purpose-built config is
addressed by its filename, and `cfg.config_path` / `cfg.variant` tell you what
it picked (full rules: [`MODEL_CONFIGURATION.md`](MODEL_CONFIGURATION.md)). It
returns a `RunConfig` — a description of the run, not a live object. Layer
anything else on with dotted paths, identical to `fdh train --set`:

```python
cfg = cfg.with_set(**{
    "train.engine_kwargs.accelerator": "gpu",
    "val.output_dir": "artifacts/anomaly_maps/stfpm",
})
```

`epochs=` resolves through `training.RUN_LENGTH_KEYS` to whichever key that
backend counts its run length in. The two distillation backends (Dinomaly,
MambaAD) count optimizer *iterations*, not epochs — `load_config` raises and
tells you to use `set={"train.total_iters": N}` rather than silently writing an
epoch count they ignore.

### Train, predict, evaluate

```python
run = fdh.train(cfg)                            # -> TrainRunResult
weights = run.result.registered_artifact.path

out = fdh.predict("stfpm", weights=weights, source="sample.jpg")
out.predictions[0].anomaly_score

metrics = fdh.evaluate("stfpm", weights=weights, dataset="tilda-400").metrics
```

`predict` takes either `source=` (one image path or a list) or `dataset=`
(sliced by `num_samples` / `pattern` / `category`). `evaluate` requires a
dataset — raw image paths carry no ground truth to score against.

For an anomaly model, pass `output_dir=` to `predict`/`evaluate` to get
pixel-level metrics:

```python
fdh.evaluate("stfpm", weights=weights, dataset="tilda-400",
             output_dir="artifacts/anomaly_maps/eval").metrics
# with output_dir:    image_auroc, image_f1, ..., pixel_auroc, pixel_aupro, pixel_f1, iap
# without output_dir: image_auroc, image_f1, ... only
```

The adapters fill `Prediction.anomaly_map` only when given somewhere to write
it (one `.npy` per sample), and `AnomalyEvaluator` scores pixels from that
field — so the maps are opt-in, and the metrics follow. It has no effect on a
model whose `capabilities()` doesn't fill `anomaly_map` (GANomaly) or on the
detection/segmentation backends, which score from boxes and masks.

### Published weights

```python
weights = fdh.from_pretrained("PatchCore")
weights.path, weights.backend, weights.task, weights.source

fdh.evaluate("PatchCore", weights=weights, dataset="tilda-400")
```

`from_pretrained` looks up this project's own model catalog
(`catalog.CANONICAL_MODELS` — the same list that drives the web UI's dropdown).
It raises `FileNotFoundError` when a catalogued model has not been trained and
published *on this machine*, which is the normal state of a fresh checkout: the
catalog names what this project publishes, and the checkpoints come from
`fdh train`, not from git.

### What the front door is not

`api.py` holds no logic of its own. Every function there resolves arguments
into the shape an existing, contract-tested entry point already accepts, and
delegates. `tests/test_api_facade.py` enforces this at the AST level: no
backend name may appear inside any facade function, and every public function
must delegate into the contract layer.

That constraint is the point. A friendly flat surface is exactly where "just
this one special case for anomalib" accumulates, and once it does the project
has two pipelines — the contract-tested one, and an untested shadow of it that
`test_pipeline_contract.py` cannot see. Where a uniform surface genuinely needs
per-backend knowledge, that knowledge lives in the layer that owns it
(`training.RUN_LENGTH_KEYS`) and is looked up.

---

## 2. The three abstractions underneath

Everything above is a thin layer over three contracts, frozen and guarded by
tests (full spec: [`INTERFACE_SPEC.md`](INTERFACE_SPEC.md)):

| Contract | What it is | Guarded by |
|---|---|---|
| `ModelAdapter` + `ModelCapabilities` | a model backend: `capabilities` / `train` / `predict` / `export`, identical signatures across all six backends | `test_adapter_contract.py` |
| `DataAdapter` + `BatchSpec` | `Sample` list → that backend's training batch | `test_data_adapter_contract.py` |
| `TrainConfig` | one vocabulary for the shared hyperparameters, translated per backend via `TRAIN_CONFIG_KEYS` | `test_train_config.py` |

All three are exported from the package root, so a caller who wants to
implement a backend or inspect one does not need to know where they live.

`ModelCapabilities` is worth knowing about even as a user: it declares what a
model can actually produce, so a metric's applicability is answerable without
running anything. GANomaly, for instance, returns only an image-level score —
its `capabilities().prediction_fields` omits `anomaly_map`, and pixel
AUROC/AUPRO is therefore not computable for it. That is declared, not
discovered halfway through an evaluation.

---

## 3. Composing a run by hand

When you want the pieces rather than the pipeline:

```python
import fabric_defect_hub as fdh

dataset = fdh.load_dataset(
    name="raw-fabric",
    root="data/RAW_FABRID",
    split="test",
    sparse_ratio=0.1,         # 10% sparse subsampling
    tiling=True,              # 4K sliding-window tiling
    tile_size=(256, 256),
    overlap=0.25,
)

model = fdh.load_model(
    backend="ultralytics",
    name="yolov8n",
    tta_mode="flip_multiscale",   # Test-Time Augmentation
)

samples = dataset.load_samples()
predictions = model.predict(samples)
pro_score = fdh.compute_pro_score(gt_masks, pred_maps)
```

`fdh.run_experiment(...)` ties the two together with an evaluator, profiler and
export target when you want the full lifecycle without going through a config
file. It takes `ModelInfo`/`RuntimeInfo` explicitly — which is precisely why §1
exists for the common case.

The loading strategies (`sparse_ratio`, `tiling`, `tta_mode`, BatchNorm
calibration) are documented in [`MODEL_CONFIGURATION.md`](MODEL_CONFIGURATION.md).

---

## 4. Scope: this is a benchmark platform, not a model zoo of our own

The SDK's job is to run *other people's* published methods under one contract
(`ModelAdapter` / `DatasetAdapter` / `Evaluator`) so their numbers are
comparable. It deliberately ships **no in-house network components** — no
backbones, necks, heads, losses, or augmentation modules of our own.

An earlier revision carried an `fdh.nn` package (feature hooks, a
`TextileAttentionNeck`, a segmentation head, an anomaly heatmap decoder) plus
`fdh.optim.losses` and `fdh.augmentations`. None of it was ever called by any
registered model, and a config profile that could swap in its own loss or
architecture would have made the benchmark unsound — the row labelled "YOLOv8"
has to be stock YOLOv8. All of it was removed; see `core/base_recipe.py` for
the settings-only rule that replaced it.

This also answers the natural next question — "can I compose an encoder /
decoder / GAN out of the parts here?". There is no shared block library, on
purpose. Adding a method follows one of three routes instead:

| Route | When | Precedent |
|---|---|---|
| alias + preset | upstream is already an `anomalib.models` class | every entry in `models/anomalib/presets.py` |
| vendored submodule + `vendor.py` | upstream is a self-contained runnable repo | `components/dinomaly`, `components/moeclip` |
| clean-room reimplementation | upstream has no runnable code | `models/mambaad/` |

Only the third needs building blocks, and it currently has one instance, so its
parts stay private to that backend. Should a second one need them, they get
lifted into a shared module *then* — under the rule that it may hold only
structures published upstream, each citing the file it reproduces, and that
nothing in it may be reachable from a config profile. If method-level
contributions of our own are added later, they belong in a separate line of
work evaluated *against* this benchmark, wired in through the same public
`ModelAdapter` contract as any other method.
