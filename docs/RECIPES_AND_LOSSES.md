# Config Profiles & Loss Functions Guide

This guide documents the per-model **config profiles** (`recipe_id`s), the loss modules, and hyperparameter settings.

A config profile is an honest, named bundle of run settings for one method — the hyperparameters we run it with (in the backend's real vocabulary), anchored to the method's real paper. It is **not** a novel contribution and carries no invented acronym. If a profile ever grows a genuine, measured modification of its own, that earned change can be named then — not before.

---

## 1. Model Config Profiles

| `recipe_id` | Target Models | What the profile sets | Anchored to (paper) |
| :--- | :--- | :--- | :--- |
| `yolov8` | YOLOv8n, YOLOv8s, YOLO11n | Ultralytics trainer settings for fabric detection | Sunkara & Luo, "No More Strided Convolutions or Pooling" (SPD-Conv), ECML PKDD 2022 (arXiv:2208.03641) |
| `patchcore` | PatchCore, PaDiM, SuperSimpleNet | WideResNet-50 features, 10% coreset, k=9 | Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), CVPR 2022 |
| `rd4ad` | RD4AD, EfficientAD | WideResNet-50 teacher, layer1+2+3, additive map | Deng & Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding", CVPR 2022 (arXiv:2201.10703) |
| `mambaad` | MambaAD | resnet34 encoder + upstream training schedule | He et al., "MambaAD", NeurIPS 2024 (arXiv:2404.06564) |
| `moeclip` | MoECLIP, WinCLIP | LoRA rank / experts (upstream defaults) | Park et al., "MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection", CVPR 2026 (arXiv:2603.03101) |
| `dinomaly` | Dinomaly | ViT-Base DINOv2 encoder + upstream schedule | Guo et al., "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection", CVPR 2025 (arXiv:2405.14325) |

> Coverage: these six profiles supply run settings for the methods among the
> project's 18 models. The torchvision detectors/segmenters (Faster/Cascade
> R-CNN, DETR, Mask R-CNN, UNet++, DeepLabV3+) run as standard baselines on
> torchvision's own defaults and intentionally carry **no** profile — see the
> README model matrix.

### How a recipe takes effect

A recipe is not just metadata for `fdh list-recipes` — it is wired into the run:

```python
model = fdh.load_model("ultralytics", "yolov8n", recipe="yolov8")
# During run_experiment(...), just before training, the recipe's hooks fire:
#   * get_default_hyperparameters() -> attached as model._recipe_hparams
#   * configure_loss()              -> attached as model._recipe_loss
#   * adapt_architecture(module)    -> applied if a module is already loaded
# The trained artifact is stamped with metadata["recipe"] = "yolov8".
```

Each backend then folds in only the **trainer-safe** subset of the recipe's
hyperparameters via `recipes.recipe_trainer_overrides(...)` — real trainer
knobs like `lr0`/`momentum` reach the trainer, while architecture/augmentation
flags (`spd_conv_downsample`) and differently-named loss gains
(`box_loss_weight`, which Ultralytics calls `box`) never leak into
`train(**kwargs)`. Anything the caller sets explicitly overrides the recipe.

> **Status:** all six profiles now express their hyperparameters in their
> backend's *real* vocabulary (verified against the backend presets / trainer
> args): `patchcore` & `rd4ad` → anomalib constructor kwargs,
> consumed via `AnomalibConfig.resolved_model_kwargs`; `yolov8` → YOLO
> trainer args incl. the `box`/`cls`/`dfl` loss gains, consumed via
> `UltralyticsAdapter.train`; `moeclip` & `mambaad` → the real
> knobs of those single-architecture clean-room backends (whose presets already
> encode the published recipe). `tests/test_recipe_reconciliation.py` pins each
> recipe to its backend's verified defaults so they can't drift back into
> invented knobs.
>
> **Remaining:** actually *running* the paper-dataset reproductions and filling
> the result tables (see `docs/REPRODUCTION_PATCHCORE.md`) — the settings are
> correct, the GPU runs are pending.

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
  recipe: yolov8             # config profile
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
