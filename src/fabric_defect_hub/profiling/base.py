"""BackendProfiler: measures inference performance (latency/FPS/memory/power)
for a given runtime (PyTorch, ONNX Runtime, TensorRT, ...) under a fixed,
recorded set of run conditions so results across models stay comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fabric_defect_hub.core.types import RuntimeInfo
from fabric_defect_hub.models.base import ExportedArtifact


@dataclass
class ProfileConfig:
    device: str
    engine: str
    precision: str = "fp32"
    input_size: tuple[int, int] = (640, 640)
    batch_size: int = 1
    warmup_runs: int = 10
    measured_runs: int = 100
    # 'batched': a single [batch, 3, H, W] tensor input — CNN-style models
    #   (Ultralytics YOLO, Anomalib backbones) exported to TorchScript/ONNX.
    # 'list': a List[Tensor] of `batch_size` [3, H, W] tensors — torchvision
    #   detection models' native TorchScript calling convention (see
    #   `models/torchvision/adapter.py::export`). Only relevant to
    #   `profiling/pytorch.py`; ONNX graphs are always a fixed tensor shape
    #   regardless of the pre-export Python calling convention.
    input_style: str = "batched"


class BackendProfiler(ABC):
    """Base class for a runtime-specific performance benchmark."""

    engine: str

    @abstractmethod
    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        """Run `measured_runs` inferences after `warmup_runs`, return latency/FPS/memory/etc."""

    def runtime_info(self, config: ProfileConfig) -> RuntimeInfo:
        return RuntimeInfo(
            device=config.device,
            engine=config.engine,
            precision=config.precision,
            input_size=config.input_size,
        )


def summarize_latencies(latencies_ms: list[float], batch_size: int, peak_memory_bytes: int) -> dict[str, float]:
    """Shared latency/FPS/memory summary every concrete `BackendProfiler`
    (`pytorch.py`, `onnxruntime.py`, `tensorrt.py`) reduces its measured
    runs down to, so the resulting metric names/definitions are identical
    across runtimes — required for the cross-engine comparison this whole
    module exists for.
    """

    import statistics

    sorted_latencies = sorted(latencies_ms)
    mean_latency = statistics.fmean(sorted_latencies) if sorted_latencies else 0.0
    return {
        "latency_ms_mean": mean_latency,
        "latency_ms_p50": _percentile(sorted_latencies, 50),
        "latency_ms_p95": _percentile(sorted_latencies, 95),
        "latency_ms_p99": _percentile(sorted_latencies, 99),
        "fps": (1000.0 / mean_latency) * batch_size if mean_latency > 0 else 0.0,
        "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[idx]
