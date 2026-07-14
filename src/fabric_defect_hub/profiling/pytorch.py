"""PyTorch (eager or TorchScript) `BackendProfiler` — the PC baseline every
other runtime (ONNX Runtime, TensorRT) gets compared against.

Loads a `torch.export` ExportedProgram or legacy TorchScript artifact and
times repeated forward passes with proper device synchronization — a naive
`time.perf_counter()` around a CUDA/MPS call measures kernel-launch time,
not execution time, unless you synchronize first; this is a genuinely easy
mistake, not a hypothetical one, so it's called out explicitly below.
"""

from __future__ import annotations

import time

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies


class PyTorchProfiler(BackendProfiler):
    """Profiles a TorchScript-exported model under real (or CPU-simulated)
    latency conditions.
    """

    engine = "pytorch"

    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        if artifact.target not in {"torchscript", "exported_program"}:
            raise ValueError(
                "PyTorchProfiler expects an 'exported_program' or 'torchscript' artifact, "
                f"got target={artifact.target!r}."
            )

        import torch

        device = torch.device(config.device)
        if artifact.target == "exported_program":
            model = torch.export.load(artifact.path).module().to(device)
        else:
            model = torch.jit.load(artifact.path, map_location=device)
        model.eval()

        dummy_input = _build_dummy_input(config, device)

        with torch.no_grad():
            for _ in range(config.warmup_runs):
                model(dummy_input)

            latencies_ms: list[float] = []
            peak_memory_bytes = 0
            monitor = self.start_power_monitor(config)
            try:
                for _ in range(config.measured_runs):
                    _reset_peak_memory(device)
                    start = time.perf_counter()
                    model(dummy_input)
                    _synchronize(device)
                    latencies_ms.append((time.perf_counter() - start) * 1000.0)
                    peak_memory_bytes = max(peak_memory_bytes, _peak_memory_bytes(device))
                    monitor.sample()
            finally:
                metrics = summarize_latencies(latencies_ms, config.batch_size, peak_memory_bytes)
                self.finish_power_monitor(monitor, metrics)

        return metrics


def _build_dummy_input(config: ProfileConfig, device):
    import torch

    h, w = config.input_size
    if config.input_style == "list":
        return [torch.rand(3, h, w, device=device) for _ in range(config.batch_size)]
    if config.input_style == "batched":
        return torch.rand(config.batch_size, 3, h, w, device=device)
    raise ValueError(f"unknown input_style {config.input_style!r}; expected 'batched' or 'list'.")


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # cpu: nothing to synchronize, execution is already blocking


def _reset_peak_memory(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.reset_peak_memory_stats()


def _peak_memory_bytes(device) -> int:
    import torch

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    import psutil

    return int(psutil.Process().memory_info().rss)
