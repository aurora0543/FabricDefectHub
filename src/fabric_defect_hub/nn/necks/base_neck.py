"""Base Abstract Class for Pluggable Feature Necks (BaseNeck)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn as nn


class BaseNeck(nn.Module, ABC):
    """Abstract base class for all pluggable feature necks and attention modules."""

    @abstractmethod
    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Processes multi-scale feature maps and returns enhanced feature representations."""
        pass
