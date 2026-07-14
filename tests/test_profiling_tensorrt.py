"""Tests for `profiling/tensorrt.py`.

TensorRT itself cannot be installed on this dev machine (Apple Silicon, no
CUDA GPU — see the module's docstring for the full disclosure). The only
things testable without either a real TensorRT install or mocking its
internals (which this project deliberately avoids — see
`profiling/tensorrt.py`'s docstring) are the validation branches that run
*before* `import tensorrt`/`import pycuda` — and that those imports are
correctly deferred until after validation, so a missing-target error never
gets masked by a missing-dependency error.
"""

import pytest

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import ProfileConfig
from fabric_defect_hub.profiling.tensorrt import TensorRTProfiler


def test_rejects_wrong_target_without_needing_tensorrt_installed():
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cuda:0", engine="tensorrt")
    with pytest.raises(ValueError, match="tensorrt"):
        profiler.profile(ExportedArtifact(path="x", target="onnx"), config)


def test_rejects_non_cuda_device_without_needing_tensorrt_installed():
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cpu", engine="tensorrt")
    with pytest.raises(ValueError, match="CUDA"):
        profiler.profile(ExportedArtifact(path="x", target="tensorrt"), config)


def test_correct_target_and_device_reaches_the_tensorrt_import():
    # Proves validation happens before the import (not after) — a correctly
    # formed request fails on the *missing dependency*, not a validation
    # error, which is the one meaningful check possible without a real
    # TensorRT/pycuda install.
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cuda:0", engine="tensorrt")
    with pytest.raises(ModuleNotFoundError):
        profiler.profile(ExportedArtifact(path="x", target="tensorrt"), config)


def test_runtime_info_maps_config_fields():
    config = ProfileConfig(device="cuda:0", engine="tensorrt", precision="fp16", input_size=(512, 512))
    info = TensorRTProfiler().runtime_info(config)
    assert info.device == "cuda:0"
    assert info.engine == "tensorrt"
    assert info.precision == "fp16"
    assert info.input_size == (512, 512)
