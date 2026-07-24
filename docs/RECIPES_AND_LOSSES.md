# Academic Recipes (MORR) & Loss Functions Guide

This guide documents the **Model-Specific Optimization Recipe Registry (MORR Engine)**, adaptive loss functions, and hyperparameter tuning configurations.

---

## 1. Model-Specific Optimization Recipes (MORR)

Every model architecture in FabricDefectHub is assigned a paper-driven optimization recipe:

| Recipe ID | Target Models | Academic Nomenclature | Paper Reference |
| :--- | :--- | :--- | :--- |
| `yolov8_sd_attn` | YOLOv8n, YOLOv8s, YOLO11n | **SD-Attn & DLW** (Small-Defect Attention + Dynamic Loss Weighting) | Sarr et al., IEEE T-CSVT 2023 |
| `patchcore_dmba` | PatchCore, PaDiM, SuperSimpleNet | **DMBA** (Domain-Specific Memory Bank Adaptation) | Roth et al., CVPR 2022 |
| `rd4ad_msfa_d` | RD4AD, EfficientAD | **MSFA-D** (Multi-Scale Feature Alignment Distillation) | Tien et al., CVPR 2022 |
| `mambaad_ss_tst` | MambaAD | **SS-TST** (Selective Scan Texture State Space Tuning) | He et al., 2024 |
| `moeclip_tpo_peft` | MoECLIP, WinCLIP | **TPO-PEFT** (Text Prompt Optimization & LoRA Adapter) | Cao et al., 2024 |

---

## 2. Specialized Loss Modules

### Adaptive Focal-Dice Loss (`AFDLoss`)

Designed to address severe foreground-background pixel imbalance (micro-defects occupying $< 0.1\%$ of the fabric image):

$$\mathcal{L}_{\text{AFDL}} = \alpha \cdot \mathcal{L}_{\text{Focal}}(p, y; \gamma) + (1 - \alpha) \cdot \mathcal{L}_{\text{SoftDice}}(p, y)$$

```python
from fabric_defect_hub.optim import AFDLoss

loss_fn = AFDLoss(
    alpha=0.5,                  # Balance factor between Focal and Dice
    gamma=2.0,                  # Focusing parameter for hard examples
    adaptive_weighting=True,    # Dynamically boosts Focal weight if defect ratio < 1%
)

loss_value = loss_fn(logits, masks)
```

### Dynamic Multi-Task Loss Scaler (`DynamicLossScaler`)

Dynamically balances bounding box loss, classification loss, and distribution focal loss based on softmax gradient variance:

```python
from fabric_defect_hub.optim import DynamicLossScaler

scaler = DynamicLossScaler(num_losses=3, init_weights=[7.5, 0.5, 1.5])
total_loss, weighted_dict = scaler({"box_loss": l_box, "cls_loss": l_cls, "dfl_loss": l_dfl})
```

---

## 3. YAML Configuration Reference

Include `recipe` and `loss_fn` in your model YAML configuration:

```yaml
model:
  variant: yolov8n
  pretrained: true
  task: detect
  recipe: yolov8_sd_attn             # Academic Recipe
  loss_fn: AFDLoss                    # Adaptive Focal-Dice Loss

training:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  cos_lr: true
  loss_alpha: 0.5
  loss_gamma: 2.0

data:
  dataset: zju-leaper
  grid_freq: 16                       # Textile Periodic Augmentation (TPA)
  phase_shift_prob: 0.4
```
