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

### Adding a new model backend

1. Implement a `ModelAdapter` (`train`/`predict`/`export`) and
   `@register_model("my-backend")` it.
2. Add it to `loader._MODEL_BACKEND_MODULES` (its import path) so
   `load_model`/`import_all_model_backends`/`fdh doctor` can find it without
   requiring every optional framework to be installed to import the module.
3. If it's a one-class or zero-shot anomaly backend, add it to
   `training._BACKEND_TRAINABLE_DATASETS` so `_enforce_trainable_dataset`
   and the decision tree apply to it the same way they do to the existing
   anomaly backends. Detection backends need no entry there.

---

## 2. Config profiles (`fabric_defect_hub.recipes`)

A config profile is a named, paper-anchored bundle of run settings for one
method — hyperparameters in the *backend's real vocabulary*, plus optional
loss/augmentation/architecture hooks — fed into a run via
`load_model(..., recipe="patchcore")`. See `docs/RECIPES_AND_LOSSES.md` for
the current six profiles and `core/base_recipe.py` for the contract.

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
