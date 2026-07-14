"""TensorRT `BackendProfiler` — Jetson/edge deployment profiling.

**Not verified on this project's dev machine.** Unlike `pytorch.py` and
`onnxruntime.py`, which were built by loading a real exported model and
timing real inference runs, TensorRT is NVIDIA-proprietary and has no
CUDA-capable install target on Apple Silicon (this repo's dev environment)
— `pip install tensorrt` simply has nothing to install here. This class is
written directly against TensorRT's documented Python API method names and
the CUDA-event benchmarking pattern TensorRT's own docs recommend, not
guessed, but it has never actually been executed. Treat it the same way
the project treated the mmdetection→torchvision swap: don't trust the
numbers until it's run for real on a Jetson/CUDA box.

Concretely unverified/undisclaimed here:

* **API generation**: targets the *named-tensor* execution API current as
  of TensorRT 8.6/10.x (`ICudaEngine.get_tensor_name`,
  `IExecutionContext.set_input_shape`/`set_tensor_address`,
  `execute_async_v3`). TensorRT <= 8.4's integer-`binding_index` API
  (`get_binding_index`, `execute_v2(bindings=[...])`) was deprecated in 8.5
  and removed in 10.x — this class will not work against an older
  TensorRT install or an engine built for one without rewriting the I/O
  binding calls.
* **Memory metric is an approximation, not a measurement**: there is no
  NVML/`nvidia-smi` dependency here, so `peak_memory_mb` is the sum of the
  allocated I/O device buffer sizes — a lower bound on the engine's actual
  working set, not the validated `torch.cuda.max_memory_allocated()`-style
  peak `pytorch.py` reports. Don't treat the two as comparable numbers.
* **No engine-building path**: this profiler only loads and runs a
  pre-built `.engine`/`.plan` file. No `ModelAdapter.export()` in this
  codebase currently produces `target == "tensorrt"` — building one (via
  `trtexec` or the ONNX->TensorRT builder API) is out of scope here and has
  to happen outside this project first.
* Not handled: multiple model outputs, INT8 calibration, dynamic-batch
  optimization-profile selection (`set_optimization_profile_async`), DLA
  cores, custom plugins.

Requires the `profiling-tensorrt` extra (`pip install -e ".[profiling-tensorrt]"`)
— `tensorrt` + `pycuda` — on a CUDA-capable machine.
"""

from __future__ import annotations

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies


class TensorRTProfiler(BackendProfiler):
    """Profiles a pre-built TensorRT engine via `execute_async_v3` + CUDA-event timing."""

    engine = "tensorrt"

    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        # Validate before importing tensorrt/pycuda: this branch stays
        # testable (and correctly ordered — see the module docstring) even
        # on a machine with neither package installed.
        if artifact.target != "tensorrt":
            raise ValueError(
                f"TensorRTProfiler expects a 'tensorrt' export, got target={artifact.target!r}."
            )
        if not config.device.startswith("cuda"):
            raise ValueError(
                f"TensorRTProfiler requires a CUDA device, got config.device={config.device!r}."
            )

        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context as a side effect)
        import pycuda.driver as cuda
        import tensorrt as trt

        engine = _load_engine(artifact.path)
        context = engine.create_execution_context()
        io = _allocate_io_tensors(engine, context, config)
        stream = cuda.Stream()

        for _ in range(config.warmup_runs):
            _execute(context, stream)

        start_event = cuda.Event()
        end_event = cuda.Event()
        latencies_ms: list[float] = []
        for _ in range(config.measured_runs):
            start_event.record(stream)
            _execute(context, stream)
            end_event.record(stream)
            end_event.synchronize()
            latencies_ms.append(start_event.time_till(end_event))

        peak_memory_bytes = sum(buf.nbytes for buf in io.device_buffers)
        return summarize_latencies(latencies_ms, config.batch_size, peak_memory_bytes)


class _IOTensors:
    def __init__(self, device_buffers: list):
        self.device_buffers = device_buffers


def _load_engine(path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def _allocate_io_tensors(engine, context, config: ProfileConfig) -> "_IOTensors":
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    h, w = config.input_size
    device_buffers = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = (config.batch_size, 3, h, w)
            context.set_input_shape(name, shape)
        else:
            shape = tuple(context.get_tensor_shape(name))

        dtype = _trt_dtype_to_numpy(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        device_buffer = cuda.mem_alloc(nbytes)
        device_buffer.nbytes = nbytes  # for the peak-memory approximation in `profile()`
        context.set_tensor_address(name, int(device_buffer))
        device_buffers.append(device_buffer)

    return _IOTensors(device_buffers=device_buffers)


def _execute(context, stream) -> None:
    context.execute_async_v3(stream.handle)


def _trt_dtype_to_numpy(dtype) -> str:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: "float32",
        trt.DataType.HALF: "float16",
        trt.DataType.INT8: "int8",
        trt.DataType.INT32: "int32",
        trt.DataType.BOOL: "bool",
    }
    return mapping.get(dtype, "float32")
