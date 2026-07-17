# Validation Matrix

FabricDefectHub separates code completion from environment validation. A
feature is code-complete when its implementation, configuration contract,
failure paths, and validation code exist. Hardware-specific execution is
recorded separately and does not become a false failure on unsupported hosts.

## Local validation

Install the complete CPU test stack and run the default suite:

```bash
python -m pip install -r requirements-full.txt
python -m pytest -q
```

The default suite covers contracts, configuration, orchestration, evaluation,
reports, tools, offline pre-flight checks, checkpoint validation, power-source
assessment, and CPU-capable profiler paths. Slow real-backend tests are skipped.

## Gradio workspace validation

Install the UI dependency before rendering the application:

```bash
python -m pip install -r requirements.txt
fdh-ui
```

The Gradio-specific build test is intentionally skipped until `gradio` is
installed. Once available, verify the Single Image Detection tab by setting
`ZJU_LEAPER_ROOT`, loading random images, browsing with both navigation
buttons, and running a configured checkpoint or pretrained model, then
verify the Benchmark tab by selecting one or more compatible models and
running a leaderboard pass.

## Real backend lifecycle

The validation code is already present in `tests/test_backend_integration.py`.
On a prepared training host, enable one or more real configuration files:

```bash
export FDH_RUN_BACKEND_INTEGRATION=1
export FDH_ULTRALYTICS_INTEGRATION_CONFIG=/path/to/ultralytics.yaml
export FDH_TORCHVISION_INTEGRATION_CONFIG=/path/to/torchvision.yaml
export FDH_ANOMALIB_INTEGRATION_CONFIG=/path/to/anomalib.yaml
export FDH_BENCHMARK_INTEGRATION_CONFIG=/path/to/benchmark.yaml
python -m pytest -m slow tests/test_backend_integration.py -q
```

Each backend config owns its dataset paths, train/validation switches,
checkpoint directory, and export targets. This keeps cloud training separate
from local deployment validation without maintaining temporary test scripts.

## Platform power and deployment validation

| Platform | Prepared measurement path | Scope | Current-host status |
| --- | --- | --- | --- |
| NVIDIA cloud GPU | NVML / `nvidia-ml-py` | GPU | Cannot execute without NVIDIA GPU |
| Jetson | `tegrastats` `VDD_IN` | Board input | Cannot execute without Jetson |
| macOS | privileged `powermetrics` SMC | Package | Code available; requires sudo policy |
| Raspberry Pi | INA219/INA226 sysfs | Sensor rail | Cannot execute without sensor hardware |
| TensorRT | ONNX builder + runtime profiler | Engine/device | Cannot execute without TensorRT/CUDA |
| TensorRT INT8 calibration | `quantization/tensorrt_calibration.py` | Engine build | Cannot execute without TensorRT/CUDA |

Unavailable platform runs are expected. They do not reduce code completion.
When a platform is available, use `power_mode: required` so missing permissions,
drivers, or sensors fail explicitly instead of producing an incomplete result.

## Quantization validation

`quantization/onnx_quant.py` (fp16, INT8 dynamic, INT8 static) is CPU-only
and fully executable on any host: `tests/test_quantization.py` runs all
three levels end to end against a real (small, hand-built) ONNX model with
`onnx`/`onnxruntime`/`onnxconverter-common` — part of `requirements-full.txt`,
no hardware dependency. TensorRT INT8 calibration
(`quantization/tensorrt_calibration.py`) shares the same status as the rest
of `profiling/tensorrt.py`: written against the documented API, not
executable without a CUDA/TensorRT host — see that module's docstring.

## Expected artifacts

A complete benchmark validation produces:

- per-run `predictions.json` and `result.json`;
- exported `.onnx`, `.pt2`, `.torchscript.pt`, or `.engine` artifacts;
- `power.json` with measurement source, scope, status, and reason;
- CSV or Markdown leaderboard output;
- registered checkpoints with backend metadata and trusted-source handling.
