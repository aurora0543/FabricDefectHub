# 模型配置
当前 `catalog.CANONICAL_MODELS` 中共有 20 个已发布模型

## 1. 当前支持的模型

### Ultralytics / YOLO
- `yolov8n`
- `yolov8s`
- `yolo11n`

### Torchvision
- `fasterrcnn_resnet50_fpn`
- `cascadercnn_resnet50_fpn`
- `detr_resnet50`
- `maskrcnn_resnet50_fpn`
- `unetplusplus_resnet34`
- `deeplabv3plus_resnet50`

### Anomalib
- `PatchCore`
- `PaDiM`
- `RD4AD`
- `EfficientAD`
- `SuperSimpleNet`
- `STFPM`
- `GANomaly`
- `WinCLIP`

### Dinomaly
- `dinov2reg_vit_base_14`

### MoECLIP
- `ViT-L-14-336`

### MambaAD
- `resnet34`

## 2. 配置查找规则

`fdh train` / `fdh predict` / `fdh.load_config()` 会按以下顺序解析配置：

1. 直接传入的 YAML 文件路径
2. `configs/models/` 下的文件名
3. 配置里声明的 `model.name` / `model.variant`
4. 已发布模型目录 `catalog.CANONICAL_MODELS` 中对应的配置
5. 该后端支持的变体名，映射到对应通用配置

如果你想避免覆盖当前前端权重，训练已发布模型时请加 `--no-publish`。

## 3. 配置优先级

配置从低到高的优先级为：

- 后端默认值
- 模型 YAML
- `configs/training_profile.yaml`（仅调整 ZJU 数据选择策略）
- 命令行的数据选择参数
- `--set path.to.key=value`

长期复现应把稳定设置写入 YAML，临时实验可用 `--set`。

## 4. 验证模式

大多数后端支持通过 `model.weights` + `train.enabled: false` 加载已有 checkpoint 进行验证或导出。

示例：

```yaml
model:
  name: dinov2reg_vit_base_14
  weights: artifacts/models/published/Dinomaly.pth
train:
  enabled: false
val:
  enabled: true
```

Anomalib 的 Lightning checkpoint 需要额外设置 `model.allow_unsafe_checkpoint: true`。

## 5. 其他说明

- 不要把 YOLO 的 `tta`、`tiling`、`mosaic` 等字段直接复制到其他后端。
- `artifacts/models/published/` 是前端可直接读取的已发布权重位置。
- 配置文件应聚焦模型选择、训练/验证/导出设置，不必写模型出处或论文信息。
