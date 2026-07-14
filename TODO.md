# FabricDefectHub — Outstanding Work

Generated from a two-agent read-only audit of `src/fabric_defect_hub/` (frontend excluded — that's tracked separately). Each item below includes enough context for an agent picking it up cold to act without re-deriving the finding. Priority: **P0** = crashes/produces silently wrong results, **P1** = real functional gap, **P2** = polish/consistency.

Verification note: framework-free paths are compiled and smoke-tested locally. Real framework, CUDA, Jetson, Raspberry Pi, and privileged macOS power checks are represented by opt-in validation code and remain environment validation rather than code-completion blockers.

---

## P0 — Crash risks / silently-wrong results

- [x] **XML bbox parsing crashes on malformed annotations.** Malformed boxes are now skipped with a warning naming the XML file.

- [x] **`ImageSets/*.json` structure assumed without validation.** Category, split, and image-id list types are validated with contextual errors.

- [x] **`TorchvisionAdapter.export(target="onnx")` returns a "successful" artifact even when export failed.** Export now removes partial files and raises a contextual `RuntimeError`; zero-byte/missing output is also rejected.

- [x] **NaN metrics silently corrupt leaderboard ordering.** Leaderboards now exclude all non-finite metric values.

- [x] **`UltralyticsAdapter.validate()` can silently return `{}`.** Metric normalization raises and reports available keys when nothing recognized can be extracted.

- [x] **`AnomalibAdapter.predict()` silently produces `None` scores on empty engine output.** Empty batches and missing anomaly scores now raise with the sample id and image path.

---

## P1 — Real functional gaps

- [x] **No CLI entry point.** Added `fdh run`, `fdh benchmark`, and `python -m fabric_defect_hub` entry points.

- [x] **No top-level "one config → full benchmark" path.** Added validated `BenchmarkConfig.from_yaml`, including datasets, training selections, evaluators, checkpoints, profiling, sorting, and reports.

- [x] ~~**Anomalib backend has no `config.py`/`pipeline.py`.**~~ **Done.** Added `models/anomalib/config.py` (`AnomalibConfig`, mirrors `UltralyticsConfig`/`TorchvisionConfig`'s `from_yaml`/`from_dict` shape) and `models/anomalib/pipeline.py` (`run_from_config`/`run_from_yaml`), plus `configs/models/anomalib_example.yaml` and `models/anomalib/adapter.py::register_trained_model`/`load_trained_model` (previously missing — needed for `checkpoint.registry_dir` parity with the other two backends). Two deliberate, documented divergences from the other two configs (not oversights): `DataSpec` splits into `train_selection`/`test_selection` (anomalib's own `Folder` vocabulary has no "val" split), and there's no native `.validate()` — `ValSpec` instead wires `predict()` output into `evaluation.anomaly.AnomalyEvaluator`. `TrainSpec.model_kwargs` stays a free-form dict (unlike the other backends' fixed hyperparameter fields) because the five anomalib models' constructors don't share a common parameter set. Verified end-to-end on real ZJU-Leaper data: train (PatchCore) → register → validate (real AUROC/AUPRO via the wired-up evaluator) → export (`torch` format) → reload from registry → predict. New tests in `tests/test_anomalib_config.py` (9 tests, no `anomalib` install required — mirrors `test_ultralytics_config.py`'s framework-free style).

- [x] **Accuracy and performance profiling are on disjoint code paths.** `run_experiment` and YAML benchmarks can now export/profile and merge accuracy, latency, FPS, memory, model size, runtime, and artifact paths into one result.

- [x] **No report/aggregate output.** Added `save_leaderboard()` with CSV and Markdown output and automatic metric columns.

- [x] **`tools/` is empty.** Added COCO-to-`Sample` conversion, model export/TensorRT build, and prediction visualization scripts.

- [x] **TensorRT engine build path.** Added ONNX-to-engine construction with fp32/fp16 validation and dynamic-shape optimization profiles. INT8 requires an explicit dataset calibrator and is intentionally rejected.

- [x] **Cross-platform power profiling.** Added capability assessment plus NVML (NVIDIA GPU), `tegrastats` (Jetson board input), `powermetrics` (macOS package), and sysfs sensor (Raspberry Pi INA219/INA226) sampling. Unavailable sensors/permissions are reported explicitly; `power_mode: required` makes them a failure.

- [x] **Dataset scope decision.** ZJU-Leaper is the current official benchmark adapter; additional public datasets are intentionally deferred. Generic COCO conversion and the unified `Sample` contract remain available for enterprise/custom data.

---

## P2 — Robustness / consistency polish

- [x] **Global registry has no reset mechanism.** Added `clear_registries()` for isolated tests and interactive-session resets.

- [x] **PyTorch 3.14 export migration path.** Added `torch.export`/`.pt2` export and profiling support; TorchScript remains a legacy compatibility target. Actual torchvision detection operator coverage is exercised by the opt-in real-backend tests.

- [x] **Hardcoded `/Volumes/SSD/...` dataset paths in example configs.** Examples now use `${ZJU_LEAPER_ROOT}` and all model YAML loaders expand environment variables.

- [x] **No pre-flight check before offline training with `pretrained=True`.** Added `offline` model config, shared cache resolution, `FDH_MODEL_CACHE`, and actionable errors before framework download code runs.

- [x] **Unvalidated checkpoint structure on load.** Torchvision checkpoints now use safe weights-only loading by default and validate fields, class-map ids, variant, and state dict; Anomalib raw Lightning checkpoints require explicit unsafe opt-in and trusted artifacts.

- [x] **`num_workers` defaults inconsistent across backends re: macOS spawn-worker risk.** Torchvision now defaults to `num_workers=0`, matching Anomalib; prepared Linux/CUDA hosts can override it in config.

- [x] **Segmentation "both empty = perfect score" can flatter a broken model.** Both-empty masks are now excluded from the aggregate and counted as `num_skipped_empty`.

- [x] **Stale docstring reference to MMDetection.** Updated to torchvision.

- [x] **`Task` literal type doesn't include `"industrial"`.** Added `"industrial"` to the shared task type.

- [x] **`np.trapezoid` is NumPy-2.0-only naming.** Added a runtime fallback to `np.trapz` for older NumPy releases.

---

## Test coverage gaps (tracked separately from the functional gaps above)

- [x] Added opt-in `pytest.mark.slow` real-backend lifecycle tests, activated by prepared backend config paths on cloud/GPU hosts.
- [x] Added fixture-style tests for `datasets/zju_leaper.py`, covering malformed ImageSets structure and malformed XML bounding boxes.
- [x] Added framework-free tests for `models/torchvision/config.py`/`pipeline.py`, enabled by lazy adapter loading and dependency injection.
- [x] Added an opt-in real-backend benchmark integration test driven by `FDH_BENCHMARK_INTEGRATION_CONFIG`.

## Additional engineering hardening

- [x] Added a complete `dev` dependency set for the committed CPU/default tests and documented the permanent local/cloud/platform validation matrix in `VALIDATION.md`.
- [x] Made the experiment-result backend contract extensible, synchronized the `industrial` task across schemas, rejected non-standard NaN JSON, and filtered non-finite metrics at the unified experiment boundary.

## Environment validation — currently unavailable, not counted as code completion

- [ ] Run the prepared validation suite on a real NVIDIA cloud GPU, Jetson, privileged macOS `powermetrics`, and Raspberry Pi with INA219/INA226; record source/scope-specific baselines.
- [ ] Confirm the new torchvision `.pt2` export against the installed cloud PyTorch/torchvision operator set; retain TorchScript only where `torch.export` reports unsupported detection operators.
