# Extending FabricDefectHub

How to add a new dataset, model backend, or config profile, and how the
project stays trainable when a given machine (a cloud box especially) doesn't
have every dataset or framework installed.

---

## 1. Dataset & backend availability — the decision tree

Not every machine has every dataset staged under `data/<Dataset>` (see
`training.DEFAULT_DATASET_ROOTS`'s symlink convention) or every optional ML
framework (`anomalib`, `ultralytics`, ...) installed. Rather than fail deep
inside a backend's training loop the moment a config names something this
machine doesn't have, three modules answer "what's actually usable *here*,
right now" and degrade gracefully:

- **`core.availability`** — is a given dataset root actually staged on disk
  (non-empty directory, not an unresolved `${ENV_VAR}` or a dangling
  symlink)? Is a given backend's framework actually importable here?
- **`core.decision`** — given a backend's allowed dataset set and what's
  staged, decide what to train on: the requested dataset if it's staged;
  otherwise the alphabetically-first staged alternative in the same allowed
  set (deterministic, not a hand-tuned "best dataset" ranking — this project
  has no benchmark evidence that one in-domain fabric source trains a better
  model than another, so it doesn't pretend to); otherwise, not runnable,
  with a reason naming exactly what to stage.
- **`training.apply_available_dataset`** — wires the decision into
  `run_train`: substitutes a staged alternative (with a `warnings.warn`, not
  silently) or raises a `FileNotFoundError` naming what's missing, instead of
  whatever cryptic error a dataloader would raise three layers down.

Run **`fdh doctor`** to see the decision tree's output directly: every known
model backend, whether it's trainable right now, which dataset it would
actually use, and why — runnable backends first. This is the answer to
"what can I train given what's staged on this machine" without starting a
run and reading a traceback.

### Adding a new dataset

1. Implement a `DatasetAdapter` and `@register_dataset("my-dataset")` it
   (see any file under `datasets/`).
2. Declare its capabilities in `core/dataset_capabilities.py`:
   ```python
   register_capabilities(
       "my-dataset",
       default_root="data/MyDataset",   # the project's data/<Dataset> convention
       roles={"anomaly_train"},          # or detection_train / zero_shot_train / fabric_train_member
       tasks=("anomaly",),
   )
   ```
   That single declaration is enough to make it training-eligible, show up
   in `ANOMALY_TRAINABLE_DATASETS`/`ZERO_SHOT_TRAINABLE_DATASETS`
   (`training.py` derives its sets from this registry), and be considered by
   the decision tree the moment it's staged under its `default_root` on a
   given machine — no other file needs to change.

### Adding a new model backend — three steps

Adding a model should mean filling in a table, not changing the framework. If
you find yourself editing an existing file other than the two registration
lines below, the contract has a hole — say so rather than working around it.

**1. Implement the `ModelAdapter` contract** (`models/base.py`) in a new
`models/<backend>/` package, and `@register_model("my-backend")` it. Four
methods, with signatures identical for every backend:

```python
@register_model("my-backend")
class MyBackendAdapter(ModelAdapter):
    backend = "my-backend"

    # Canonical TrainConfig field -> this backend's real argument name.
    TRAIN_CONFIG_KEYS = {"epochs": "epochs", "lr": "learning_rate", "batch_size": "bs"}

    def capabilities(self) -> ModelCapabilities:
        # What this backend can do, declared rather than discovered: the tasks
        # it serves, which `Prediction` fields `predict()` fills, what labels
        # training needs, what it can export, whether AMP is real here.
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score", "anomaly_map"),
            required_annotations=(),
            export_targets=("onnx",),
            supports_amp=False,
        )

    def train(self, config): ...
    def predict(self, samples, artifact=None, output_dir=None, config=None): ...
    def export(self, artifact, target, config=None): ...
```

Take the union signature as written, including the parameter your backend
ignores. It is what lets `loader.run_experiment` call any backend without
inspecting it first.

Start `train()` with
`config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)` so a caller
can hand you either a `TrainConfig` or your own dict.

**2. Implement the `DataAdapter` contract** (`core/data_adapter.py`) for the
`Sample` -> batch conversion, in your own `models/<backend>/data.py`. Write
the conversion your backend actually needs — do *not* try to reuse another
backend's; the preprocessing genuinely differs. What is shared is the seam:
`(samples, **options)` construction, `__len__`/`__getitem__`, a `batch_spec()`
declaring layout/normalization/mask semantics, and a `collate_fn` when the
default cannot stack your targets. `build_dataloader()` then works for you
for free.

**3. Register it in two places** so it is discoverable without importing every
optional framework:

- `loader._MODEL_BACKEND_MODULES` — its import path, for
  `load_model` / `import_all_model_backends` / `fdh doctor`.
- `training._BACKEND_TRAINABLE_DATASETS` — only if it is a one-class or
  zero-shot anomaly backend, so `_enforce_trainable_dataset` and the dataset
  decision tree treat it like the other anomaly backends. Detection backends
  need no entry.

Then run `pytest tests/test_adapter_contract.py tests/test_data_adapter_contract.py`.
Those two files are parametrised over every registered backend, so a missing
method, a drifted signature, an undeclared capability or a typo'd vocabulary
term fails there rather than in someone's training run.

---

## 2. Config profiles (`fabric_defect_hub.recipes`)

A config profile is a named, paper-anchored bundle of run settings for one
method — hyperparameters in the *backend's real vocabulary* — fed into a run
via `load_model(..., recipe="patchcore")`. See `docs/MODEL_CONFIGURATION.md`
for the current six profiles and `core/base_recipe.py` for the contract.

**Settings only.** A profile may not supply a loss, an architecture change, or
an augmentation pipeline; those hooks used to exist and were removed. A
benchmark row labelled "YOLOv8" has to be stock YOLOv8, or no cross-model
comparison in the results table means anything.

**A profile is not a novel contribution and carries no invented acronym.**
If you make a genuine, measured architectural change (not just a settings
bundle), that earned change can be named once it's implemented and its
effect is measured against a real baseline — not before. To add a profile:

1. Subclass `BaseModelRecipe`, `@register_recipe("my-method")` it.
2. `paper_reference` must cite a real, verifiable paper (or say "no paper —
   in-house baseline" if there isn't one).
3. `get_default_hyperparameters()` must use the *target backend's real
   constructor/trainer argument names* — check `models/<backend>/presets.py`
   for what's actually accepted, and see `recipes/apply.py`'s
   `recipe_trainer_overrides`/`recipe_model_kwargs` for how a backend safely
   filters a profile's hyperparameters down to only the ones it understands.
4. Add a `tests/test_recipe_reconciliation.py`-style drift guard: assert the
   profile's keys are a subset of the backend's real accepted arguments, so
   it can't silently drift back into invented knobs.

---

## 3. The tuning window (`--set`)

`fdh train <config> --set path.to.key=value` (repeatable) overrides any
config value by dotted path — the highest-priority layer of all, above the
config file, above a profile's defaults, above every other `--*` flag. Values
are YAML-parsed, so numbers/booleans/lists come through as the right Python
type without quoting tricks:

```bash
fdh train configs/models/patchcore_textile.yaml \
  --set train.model_kwargs.coreset_sampling_ratio=0.05 \
  --set train.model_kwargs.lr=0.0005
```

See `training.apply_raw_overrides` for the implementation — it's a thin,
generic dict-patcher with no knowledge of any particular backend, so it works
for every config section, not just the ones with a dedicated CLI flag
already.
