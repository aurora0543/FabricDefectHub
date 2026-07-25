# 模型配置

训练由模型 YAML 驱动；Web UI 只用于加载已发布权重并推理，不会修改训练超参数。一次训练实际使用的最终配置会被写入 `artifacts/models/records/`，并由 `artifacts/models/weight_manifest.jsonl` 关联到对应权重。

## 配置优先级

从低到高依次为：后端预设、模型 YAML、`configs/training_profile.yaml` 的 ZJU 数据策略、命令行的数据选择参数、`--set path.to.key=value`。`--set` 适合一次性实验；需要长期复现的设置应写回独立 YAML。

`training_profile.yaml` 只控制 ZJU 样本规模、pattern 和正常/异常比例。模型结构、训练超参数、增强、推理策略必须放在各自的模型 YAML 中。

## YOLO

`configs/models/ultralytics_example.yaml` 是 YOLOv8n、YOLOv8s 与
YOLO11n 的共用织物检测配置。公共策略写在顶层；每个版本的 `epochs`、
`imgsz`、`batch`、`lr0`、输出目录名写在 `variants.<variant>`。运行时先
按 `model.variant` 选择对应 profile，再应用命令行覆盖，因此编辑 YAML
即可独立调优各版本，不需要修改 Python preset。常规训练使用模型名即可：
`fdh train yolov8n`、`fdh train yolov8s` 或 `fdh train yolo11n`。

YOLO 的训练期切片在 `data.tiling`、`data.tile_size`、`data.overlap`；推理期切片独立位于 `predict.tiling`、`predict.tile_size`、`predict.tile_overlap`。两者默认关闭，互不隐式开启。

`predict.tta_mode: flip_multiscale` 会启用 Ultralytics 原生 TTA；`predict.augment` 可直接使用原生增强推理。TTA 和切片可以单独或同时开启。

常用布匹增强写在 `train.augmentation`：`hsv_h`、`hsv_s`、`hsv_v`、`degrees`、`translate`、`scale`、`flipud`、`fliplr`、`mosaic`、`close_mosaic`、`mixup`、`cutmix`、`erasing`。这些字段会转换成真实的 `YOLO.train()` 参数；概率字段必须为 0 到 1。需要尚未列出的上游参数时，使用 `train.extra`，但不能与 `train.augmentation` 重复。

`AFDLoss` 和 TPA 并不是 Ultralytics 的可直接配置参数，项目目前没有实现它们，因此 YAML 不再接受这些字段。只有实现损失函数/数据增强并接入训练循环后，才能重新添加对应字段。

示例：

```yaml
train:
  batch: 32
  epochs: 150
  augmentation:
    mosaic: 0.3
    mixup: 0.0
    fliplr: 0.5
predict:
  conf: 0.35
  iou: 0.6
  tta_mode: flip_multiscale
  tiling: true
  tile_size: [640, 640]
  tile_overlap: 0.25
```

## 其他后端

Torchvision、Anomalib、Dinomaly、MoECLIP 和 MambaAD 都使用各自 YAML 的 `model`、`train`、`val`、`predict` 段。不要把 YOLO 的 TTA、tile 或 mosaic 字段复制到这些模型；配置解析会拒绝未知字段，避免出现“写了但未生效”的实验记录。
