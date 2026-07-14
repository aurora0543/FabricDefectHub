# FabricDefectHub — Outstanding Work

Generated from a two-agent read-only audit of `src/fabric_defect_hub/` (frontend excluded — that's tracked separately). Each item below includes enough context for an agent picking it up cold to act without re-deriving the finding. Priority: **P0** = crashes/produces silently wrong results, **P1** = real functional gap, **P2** = polish/consistency.

Verification note: none of the fixes below have been executed on this machine (Apple Silicon, no CUDA). Anything touching `torch`/`torchvision`/`anomalib`/`onnxruntime` needs to be run for real after the change, not just reviewed.

---

## P0 — Crash risks / silently-wrong results

- [ ] **XML bbox parsing crashes on malformed annotations.** `datasets/zju_leaper.py` (`_build_sample`, around the `bbox.findtext("xmin")` calls) does `float(bbox.findtext("xmin"))` etc. with no null check — `findtext` returns `None` for a missing tag, and `float(None)` raises `TypeError`, killing `load_samples()` for the *entire* dataset over one bad annotation. Add a guard (skip the box, log a warning, or raise a clear error naming the offending file) instead of letting a raw `TypeError` propagate.

- [ ] **`ImageSets/*.json` structure assumed without validation.** `datasets/zju_leaper.py::_select_ids` does `index["normal"][self.split]` / `index["defect"][self.split]` with no shape check — a differently-structured JSON raises an unhelpful `KeyError`/`TypeError` deep in the call stack. Validate structure up front with a clear error message.

- [ ] **`TorchvisionAdapter.export(target="onnx")` returns a "successful" artifact even when export failed.** `models/torchvision/adapter.py` (`export`, the `except Exception` block around `torch.onnx.export`) swallows the exception, stuffs a message into `metadata["warning"]`, but still returns an `ExportedArtifact` pointing at a file that may not exist or be truncated. Callers that don't check `metadata["warning"]` get a fake success; the real failure only surfaces later when something tries to load the broken file (e.g. `ONNXRuntimeProfiler`). Either raise by default with an opt-in `best_effort=True` flag, or make the broken state impossible to miss (e.g. don't return a path that doesn't exist).

- [ ] **NaN metrics silently corrupt leaderboard ordering.** `evaluation/anomaly.py` writes `float('nan')` into `image_auroc`/`pixel_auroc`/`pixel_aupro` when only one class is present in the eval set (a real scenario — this project's own `use_defect`/`defect_ratio` dataset controls can produce all-defect or all-normal slices). `benchmark.py::leaderboard` only drops rows where the metric key is *absent*, not where it's NaN, so `sorted(...)` on a NaN key produces an arbitrary, unstable order. Fix `leaderboard()` to also drop/flag NaN-valued rows.

- [ ] **`UltralyticsAdapter.validate()` can silently return `{}`.** `models/ultralytics/adapter.py::_normalise_metrics` falls back to `{}` if `results.results_dict` doesn't have the expected keys, and swallows `TypeError`/`ZeroDivisionError` in the per-class block. Combined with the leaderboard's "drop rows missing the metric" behavior, a model that failed to validate just quietly vanishes from a benchmark run instead of raising. Should raise (or at minimum log loudly) when the expected metrics can't be extracted.

- [ ] **`AnomalibAdapter.predict()` silently produces `None` scores on empty engine output.** `models/anomalib/adapter.py::predict` does `engine.predict(...) or []`; if empty, `Prediction.anomaly_score` stays `None`, and `AnomalyEvaluator.evaluate()` silently skips that sample. A partially-broken predict run looks like a smaller-but-valid eval set instead of an error. Should surface a warning/count of skipped samples at minimum.

---

## P1 — Real functional gaps

- [ ] **No CLI entry point.** `pyproject.toml` has no `[project.scripts]`, no `__main__.py`, no argparse/click anywhere in `src/`. README repeatedly frames the project as "配置化，而非命令行" but there isn't even a thin wrapper like `fdh run configs/models/ultralytics_example.yaml`. Needed for the project to be usable outside a Python REPL/notebook.

- [ ] **No top-level "one config → full benchmark" path.** `benchmark.py`'s `BenchmarkRun` objects (with `ModelInfo`, `RuntimeInfo`, an `Evaluator` instance) must currently be constructed in Python. There's no YAML schema/loader that turns a single benchmark config into a `list[BenchmarkRun]`. Compare to how `UltralyticsConfig.from_yaml` / `TorchvisionConfig.from_yaml` work for single-backend runs (`models/ultralytics/config.py`, `models/torchvision/config.py`) — a `BenchmarkConfig.from_yaml` in the same spirit, spanning multiple backends/datasets, is the missing piece.

- [x] ~~**Anomalib backend has no `config.py`/`pipeline.py`.**~~ **Done.** Added `models/anomalib/config.py` (`AnomalibConfig`, mirrors `UltralyticsConfig`/`TorchvisionConfig`'s `from_yaml`/`from_dict` shape) and `models/anomalib/pipeline.py` (`run_from_config`/`run_from_yaml`), plus `configs/models/anomalib_example.yaml` and `models/anomalib/adapter.py::register_trained_model`/`load_trained_model` (previously missing — needed for `checkpoint.registry_dir` parity with the other two backends). Two deliberate, documented divergences from the other two configs (not oversights): `DataSpec` splits into `train_selection`/`test_selection` (anomalib's own `Folder` vocabulary has no "val" split), and there's no native `.validate()` — `ValSpec` instead wires `predict()` output into `evaluation.anomaly.AnomalyEvaluator`. `TrainSpec.model_kwargs` stays a free-form dict (unlike the other backends' fixed hyperparameter fields) because the five anomalib models' constructors don't share a common parameter set. Verified end-to-end on real ZJU-Leaper data: train (PatchCore) → register → validate (real AUROC/AUPRO via the wired-up evaluator) → export (`torch` format) → reload from registry → predict. New tests in `tests/test_anomalib_config.py` (9 tests, no `anomalib` install required — mirrors `test_ultralytics_config.py`'s framework-free style).

- [ ] **Accuracy and performance profiling are on disjoint code paths.** `loader.py::run_experiment`'s docstring says "Profiling is deliberately left out here"; `benchmark.py` never touches `profiling/` either. Result: no single call produces an `ExperimentResult` with both `metrics` (mAP/AUROC/etc.) *and* `runtime`/latency/fps populated — exactly the shape the README's own `ExperimentResult` example shows. Needs either an opt-in profiling step in `run_experiment`/`run_benchmark`, or a documented two-step workflow (run experiment, run profiler, merge) with a helper function for the merge.

- [ ] **No report/aggregate output.** `benchmark.py::leaderboard()` only sorts an in-memory `list[ExperimentResult]`. Nothing renders a CSV/markdown table/HTML summary or writes an aggregate artifact to disk. `core/serialization.py` already has `save_experiment_result` per-run — a `save_leaderboard(results, path)` (CSV or markdown table) would close this gap cheaply.

- [ ] **`tools/` is empty.** Only `tools/README.md` exists, listing three planned scripts that don't exist yet:
  - `convert_annotations.py` — third-party annotation format → `Sample` JSON
  - `export_model.py` — CLI wrapper around `ModelAdapter.export`
  - `visualize_predictions.py` — draw `Prediction` boxes/masks/anomaly maps on images
  Useful reference for the expected interface: `core/serialization.py` (`save_samples`/`load_samples`) for `convert_annotations.py`; each adapter's `export()` method for `export_model.py`.

- [ ] **No power / model-size profiling metrics.** README's `BackendProfiler` section and Phase 3 roadmap explicitly promise 功耗 (power draw) and 模型文件大小 (model file size). `profiling/base.py::summarize_latencies` only returns latency/fps/peak_memory — neither is implemented anywhere. Model size is cheap (file size of the exported artifact); power draw needs a platform-specific approach (e.g. `nvidia-smi`/NVML on the TensorRT/CUDA path — tie into `profiling/tensorrt.py` once that's actually verified on real hardware) and should be explicitly out-of-scope-documented for CPU/MPS if not implemented there.

- [ ] **Only one dataset adapter exists.** `datasets/zju_leaper.py` is the only real `DatasetAdapter`; `yolo_bbox.py`/`anomalib_folder.py` are format-conversion staging helpers, not datasets, despite living in the same directory. If the roadmap still intends multiple datasets (README's backend table implies breadth), scope and prioritize which to add next; if ZJU-Leaper-only is now the intended scope, update the README to stop implying otherwise.

---

## P2 — Robustness / consistency polish

- [ ] **Global registry has no reset mechanism.** `core/registry.py`'s `_DATASET_REGISTRY`/`_MODEL_REGISTRY` are module-level dicts; `register_dataset`/`register_model` raise `ValueError` on any duplicate name, with no `unregister`/`clear`. Fine for normal `import`-once usage, but breaks under `importlib.reload`, notebook cell re-runs, or any test that manipulates `sys.modules`. A `clear_registries()` (test-only) or idempotent-registration option would remove this trap. Already worked around once in `tests/test_benchmark.py` via unique naming — worth fixing at the source instead of relying on naming discipline everywhere.

- [ ] **`torch.jit.script`/`torch.jit.load`/`torch.onnx.export` all emit "not supported in Python 3.14+ and may break" DeprecationWarnings** on this project's dev environment (Python 3.14). Affected: `models/torchvision/adapter.py::export` (torchscript path — the "officially supported" one, per its own docstring, and unguarded; onnx path already has a try/except but see the P0 item above about what that except block does wrong), `profiling/pytorch.py::profile` (`torch.jit.load` — this is the baseline every other runtime's profiling is compared against). Track upstream PyTorch's replacement path (`torch.export`) and plan a migration before this actually breaks, not after.

- [ ] **Hardcoded `/Volumes/SSD/...` dataset paths in example configs.** `configs/models/ultralytics_example.yaml` and `configs/models/torchvision_example.yaml` both point `dataset_root` at a specific external SSD mount. Fine as *examples*, but should be called out as placeholder/machine-specific — maybe a comment or an env-var indirection (`${ZJU_LEAPER_ROOT}`) so copying the config doesn't silently `FileNotFoundError` on another machine.

- [ ] **No pre-flight check before offline training with `pretrained=True`.** `TorchvisionAdapter`/`UltralyticsAdapter` trigger a weights download on first use when `pretrained=True` (the default), with no connectivity check beforehand — fails deep in a torch-hub/ultralytics call rather than with a clear "no internet, and no cached weights" message.

- [ ] **Unvalidated checkpoint structure on load.** `TorchvisionAdapter.load_weights` does `checkpoint["class_map"]` / `load_state_dict(checkpoint["state_dict"])` with no key-existence check — missing keys raise `KeyError`, architecture mismatches raise `RuntimeError` from `load_state_dict`, neither with context about what checkpoint/adapter mismatch caused it. Also uses `weights_only=False` (both here and in `models/anomalib/adapter.py`), which executes arbitrary pickle content — safe only under the documented "we produced this ourselves" assumption, which nothing in the code actually enforces (e.g. no signature/provenance check). Low risk currently (single-user local project) but worth a comment-level acknowledgment at minimum if not an actual guard.

- [ ] **`num_workers` defaults inconsistent across backends re: macOS spawn-worker risk.** `models/anomalib/adapter.py` defaults `num_workers=0` specifically because of a documented macOS spawn-worker shutdown race with its symlink-staged temp dirs (see the comment there). `models/torchvision/adapter.py`/`presets.py` default `num_workers=2` with no equivalent protection, despite using a similar on-the-fly `Sample`-backed dataset. Either justify the difference explicitly or align the defaults.

- [ ] **Segmentation "both empty = perfect score" can flatter a broken model.** `evaluation/segmentation.py`'s `_iou`/`_dice`/`_pixel_f1` return `1.0` when both ground truth and prediction are empty. A model that predicts nothing at all on an all-normal image set would score a perfect mIoU. Consider whether this should instead be excluded from the average (like the "no matching predictions" case already is) rather than counted as a perfect match.

- [ ] **Stale docstring reference to MMDetection.** `models/base.py`'s module docstring still says the adapter "unifies … MMDetection …" — leftover from before the torchvision swap (already fixed elsewhere, e.g. `README.md`, `core/registry.py`). One-line fix.

- [ ] **`Task` literal type doesn't include `"industrial"`.** `core/types.py` defines `Task = Literal["detection", "segmentation", "anomaly"]`, but `IndustrialEvaluator.task = "industrial"` in `evaluation/industrial.py` falls outside that literal. Not runtime-enforced (Python doesn't check `Literal` at runtime) but a type-checker would flag it, and it's a real inconsistency in the task taxonomy.

- [ ] **`np.trapezoid` is NumPy-2.0-only naming.** `evaluation/anomaly.py::_compute_aupro` uses `np.trapezoid`, which is `np.trapz` on NumPy <2.0. If this project ever needs to support an older NumPy pin, this call needs a version-conditional fallback. Currently fine given the pinned environment, but worth a version-floor comment/dependency constraint in `pyproject.toml` if not already present.

---

## Test coverage gaps (tracked separately from the functional gaps above)

- [ ] No test exercises a real model backend's actual `train()`/`predict()` (Ultralytics/torchvision/Anomalib) — all verification of those paths so far has been manual/interactive, not committed as pytest. Understandable given the weight/compute cost, but at minimum a `pytest.mark.slow`-gated smoke test (skipped by default, runnable on a real GPU box) would give future changes a real regression check.
- [ ] No test for `datasets/zju_leaper.py` (needs either a tiny bundled fixture dataset or a skip-if-real-dataset-root-missing test).
- [ ] No test for `models/torchvision/config.py`/`pipeline.py` (the `models/ultralytics/` equivalents have `tests/test_ultralytics_config.py`; torchvision's config layer has no equivalent).
- [ ] No integration test end-to-end through `benchmark.py` + a real (not fake) backend.
