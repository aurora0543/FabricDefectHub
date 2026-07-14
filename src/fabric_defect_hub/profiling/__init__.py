"""Runtime-specific `BackendProfiler` implementations: `pytorch.py`
(TorchScript, verified), `onnxruntime.py` (verified), `tensorrt.py`
(written against the documented API, **not verified** — no CUDA-capable
machine in this project's dev environment; see its module docstring).
See `base.py` for the shared `ProfileConfig`/`summarize_latencies` contract.

Every concrete profiler class only imports its runtime (torch/onnxruntime/
tensorrt) lazily inside `profile()`, so importing this package — and the
classes below — never requires any of them to be installed.
"""

from fabric_defect_hub.profiling.onnxruntime import ONNXRuntimeProfiler
from fabric_defect_hub.profiling.pytorch import PyTorchProfiler
from fabric_defect_hub.profiling.tensorrt import TensorRTProfiler

__all__ = ["PyTorchProfiler", "ONNXRuntimeProfiler", "TensorRTProfiler"]
