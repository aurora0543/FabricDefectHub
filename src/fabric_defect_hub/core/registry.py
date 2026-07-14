"""Name-based registries so the loader can resolve datasets and models by
string name/backend instead of importing every framework eagerly.

Framework-specific packages (`models/ultralytics`, `models/torchvision`,
`models/anomalib`, individual `datasets/*` modules) register themselves via
the decorators below when imported. Optional backends are only imported on
demand inside `loader.py`, so a machine without, say, Anomalib installed
can still use the Ultralytics or torchvision paths.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_DATASET_REGISTRY: dict[str, type] = {}
_MODEL_REGISTRY: dict[str, type] = {}


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in _DATASET_REGISTRY:
            raise ValueError(f"dataset '{name}' is already registered")
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def register_model(backend: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if backend in _MODEL_REGISTRY:
            raise ValueError(f"model backend '{backend}' is already registered")
        _MODEL_REGISTRY[backend] = cls
        return cls

    return decorator


def get_dataset_cls(name: str) -> type:
    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown dataset '{name}'. Known datasets: {known}") from exc


def get_model_cls(backend: str) -> type:
    try:
        return _MODEL_REGISTRY[backend]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown model backend '{backend}'. Known backends: {known}") from exc


def list_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)
