"""Pluggable Feature Necks for FabricDefectHub (SD-Attn, CBAM)."""

from fabric_defect_hub.nn.necks.base_neck import BaseNeck
from fabric_defect_hub.nn.necks.textile_neck import TextileAttentionNeck

__all__ = ["BaseNeck", "TextileAttentionNeck"]
