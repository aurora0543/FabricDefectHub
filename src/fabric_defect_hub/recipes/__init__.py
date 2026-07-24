"""Model-Specific Optimization Recipe Registry (MORR Engine)."""

from fabric_defect_hub.recipes.mambaad_recipe import MambaADRecipe
from fabric_defect_hub.recipes.moeclip_recipe import MoECLIPRecipe
from fabric_defect_hub.recipes.patchcore_recipe import PatchCoreRecipe
from fabric_defect_hub.recipes.rd4ad_recipe import RD4ADRecipe
from fabric_defect_hub.recipes.yolov8_recipe import YOLOv8Recipe

__all__ = [
    "YOLOv8Recipe",
    "PatchCoreRecipe",
    "RD4ADRecipe",
    "MambaADRecipe",
    "MoECLIPRecipe",
]
