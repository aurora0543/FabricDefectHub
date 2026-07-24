"""Autonomous Feature Hook Engine (FeatureHookEngine).

Intercepts intermediate multi-scale feature maps from PyTorch backbones
without modifying the backbone source code or relying on external frameworks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class FeatureHookEngine:
    """Non-intrusive Feature Interception Engine using PyTorch Forward Hooks."""

    def __init__(self, backbone: nn.Module, target_layers: List[str]) -> None:
        self.backbone = backbone
        self.target_layers = target_layers
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Finds named modules matching target_layers and registers forward hooks."""
        named_modules = dict(self.backbone.named_modules())

        for layer_name in self.target_layers:
            if layer_name not in named_modules:
                raise KeyError(
                    f"Layer '{layer_name}' not found in backbone modules. "
                    f"Available layers: {list(named_modules.keys())[:10]}..."
                )

            target_module = named_modules[layer_name]

            def hook_fn(module: nn.Module, input: Any, output: torch.Tensor, name: str = layer_name):
                self._feature_maps[name] = output

            hook_handle = target_module.register_forward_hook(hook_fn)
            self._hooks.append(hook_handle)

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs forward pass on backbone and returns intermediate multi-scale feature maps."""
        self._feature_maps.clear()
        self.backbone(x)
        return {layer_name: self._feature_maps[layer_name] for layer_name in self.target_layers}

    def remove_hooks(self) -> None:
        """Removes all registered hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._feature_maps.clear()

    def __del__(self) -> None:
        self.remove_hooks()
