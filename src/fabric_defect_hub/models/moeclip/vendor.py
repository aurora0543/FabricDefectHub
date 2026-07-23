"""Bootstraps imports from the vendored `components/moeclip` checkout.

Like Dinomaly (see `models/dinomaly/vendor.py`), MoECLIP ships as flat
top-level modules meant to be run from its own repo root -- `utils`,
`dataset`, `model`, `forward_utils` -- with no package prefix and no pip
release. Unlike Dinomaly, we cannot simply leave those names in
`sys.modules`: MoECLIP and Dinomaly *both* define a top-level `utils` and
`dataset`, and the Benchmark tab runs every canonical model back to back
in one process (see `web/benchmark.py`). Whichever backend imported first
would win, and the second would silently get the other repo's modules.

So `import_vendor()` imports MoECLIP's modules inside a window where

1. `components/moeclip` is `sys.path[0]`, and
2. any already-imported module owning one of those names (i.e. Dinomaly's)
   is temporarily removed from `sys.modules`,

then *takes MoECLIP's modules back out* of `sys.modules` and keeps them in
a private cache, restoring whatever was there before. The imported module
objects keep working afterwards -- their cross-references were resolved
into module globals at import time -- but they no longer occupy the shared
names, so Dinomaly's `import utils` still finds Dinomaly's. The one rule
this relies on is that the vendored code does not `import utils`/`import
dataset...` lazily from inside a function body; it doesn't (checked
against the pinned commit).

MoECLIP also hardcodes `torch.device("cuda:0")` when allocating its LoRA
expert weights (`model/moe_adapter.py::SimpleLoraExpert.__init__`), which
makes the model unconstructable on any CPU/MPS machine. Until that is
patched on the fork (the proper fix -- see `components/README.md`),
`cuda_free_module_init()` supplies a scoped compatibility shim.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "components" / "moeclip"

# Top-level names the vendored checkout owns. `model` and `dataset` are
# packages, so their submodules (`model.clip`, `dataset.constants`, ...)
# have to be moved in and out along with them.
_OWNED_ROOTS = ("utils", "dataset", "model", "forward_utils")

# Modules `import_vendor()` resolves, in dependency order.
_ENTRY_MODULES = (
    "utils",
    "dataset.constants",
    "model.clip",
    "model.moe_adapter",
    "model.tokenizer",
    "forward_utils",
)

_cache: dict[str, ModuleType] = {}


def vendor_root() -> Path:
    return _VENDOR_ROOT


def _owned(name: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in _OWNED_ROOTS)


def import_vendor() -> dict[str, ModuleType]:
    """Import (once) and return MoECLIP's vendored modules by name.

    Keys are the module paths in `_ENTRY_MODULES`, e.g.
    `import_vendor()["model.moe_adapter"].MoECLIP`.
    """

    if _cache:
        return _cache

    if not _VENDOR_ROOT.is_dir():
        raise FileNotFoundError(
            f"MoECLIP vendor checkout not found at {_VENDOR_ROOT}. Expected the "
            "submodule under components/moeclip -- run "
            "'git submodule update --init --recursive' (see components/README.md)."
        )

    displaced = {name: module for name, module in sys.modules.items() if _owned(name)}
    for name in displaced:
        del sys.modules[name]

    path_str = str(_VENDOR_ROOT)
    sys.path.insert(0, path_str)
    try:
        imported = {name: importlib.import_module(name) for name in _ENTRY_MODULES}
    finally:
        # Reclaim MoECLIP's modules out of the shared namespace, then put
        # back whatever (Dinomaly, or a genuine third-party `dataset`) was
        # there before. Runs even on failure so a half-finished import
        # can't leave the process in a mixed state.
        for name in [name for name in sys.modules if _owned(name)]:
            del sys.modules[name]
        sys.modules.update(displaced)
        if path_str in sys.path:
            sys.path.remove(path_str)

    _cache.update(imported)
    return _cache


@contextmanager
def cuda_free_module_init(device) -> Iterator[None]:
    """Make `nn.Linear(..., device='cuda:0')` fall back to `device`.

    `SimpleLoraExpert.__init__` allocates its LoRA `A`/`B` matrices with a
    hardcoded `torch.device("cuda:0")`, so building `MoECLIP` raises on any
    machine without CUDA -- including this project's dev laptop and CI.
    Inside this context (and only when CUDA really is unavailable), any
    `nn.Linear` asked for a CUDA device is built on `device` instead;
    everything else is untouched, and the patch is reverted on exit.

    The subclass is transparent to `isinstance`, `state_dict()` keys and
    checkpoint round-trips, so a model built here loads a checkpoint
    trained on a CUDA box and vice versa.
    """

    import torch

    if torch.cuda.is_available():
        yield
        return

    original = torch.nn.Linear
    device_str = str(device)

    class _DeviceFallbackLinear(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, device=None, **kwargs):
            if device is not None and torch.device(device).type == "cuda":
                device = torch.device(device_str)
            super().__init__(*args, device=device, **kwargs)

    torch.nn.Linear = _DeviceFallbackLinear
    try:
        yield
    finally:
        torch.nn.Linear = original
