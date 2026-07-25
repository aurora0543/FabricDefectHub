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

## 用较小数据集做验证

`tilda-400`、`fabric-defects` 已注册（见 `core/dataset_capabilities.py`）、有默认路径（`data/TILDA_400`、`data/Fabric Defects Dataset`），也已经出现在 Web UI 的 Benchmark 页面数据集下拉框里——训练仍然用 ZJU-Leaper（或 `fabric-train` 联合数据集），这两个数据集专门用来在训练域之外验证已训练好的权重。

除了 Web UI，命令行也能跑同样的验证，不需要开浏览器：

```bash
fdh evaluate patchcore_textile \
  --weights artifacts/models/published/PatchCore.ckpt \
  --dataset tilda-400
```

`fdh evaluate` 是 `fdh predict` 的姊妹命令：跑推理的方式完全一样（同样支持 `--enable-tiling`/`--enable-tta`），区别是它额外用数据集自带的真值标签算出指标（`evaluation.evaluator_for_task` 按任务选 `AnomalyEvaluator`/`DetectionEvaluator`/`SegmentationEvaluator`），而不是只输出预测结果。

想量化"训练域 vs 验证域"退化了多少：Benchmark 页面的"跨域退化率目标数据集"下拉框选 `TILDA-400` 或 `Fabric Defects`，结果表会多出 `cross_domain_delta_acc_pct` 一列（`evaluation/cross_domain.py`），按每个任务的主指标（异常检测用 `image_auroc`、检测用 `map`、分割用 `miou`）算出百分比退化。
