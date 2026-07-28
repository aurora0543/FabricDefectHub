# 模型配置

训练由模型 YAML 驱动；Web UI 只用于加载已发布权重并推理，不会修改训练超参数。一次训练实际使用的最终配置会被写入 `artifacts/models/records/`，并由 `artifacts/models/weight_manifest.jsonl` 关联到对应权重。

## `fdh train <名字>` 是怎么找到配置的

一句话规则：**裸模型名解析到该后端的通用配置；专用配置（复现用、textile 调优版）按文件名寻址。**

`fdh train` / `fdh predict` / `fdh.load_config()` 共用同一套解析（`training.resolve_model_config_and_variant`），按以下顺序：

| # | 依据 | 例子 |
|---|---|---|
| 1 | 磁盘上的文件路径 | `fdh train configs/models/anomalib_example.yaml` |
| 2 | `configs/models/` 下的文件名（可省 `.yaml`） | `fdh train patchcore_mvtec_repro` |
| 3 | **某个配置自己声明的** `model.name` / `model.variant` / `variants.<名>` | `fdh train yolov8n`、`fdh train stfpm` |
| 4 | **目录**：`catalog.CANONICAL_MODELS` 里该已发布模型自己声明的 `config` | `fdh train padim`、`fdh train unetplusplus_resnet34` |
| 5 | **后端支持的变体**：某后端 presets 接受的名字 → 该后端的 `<backend>_example.yaml` | `fdh train fastflow`、`fdh train glass` |

几条要点：

- 第 3 步**只在恰好命中一个配置时**采用。三个配置都声明了 PatchCore（`anomalib_example` / `patchcore_textile` / `patchcore_mvtec_repro`），此时不报错，而是交给第 4 步——目录说通用的那个是 `anomalib_example.yaml`。想要另外两个，按文件名（第 2 步）寻址。
- 第 4 步不是启发式。`CanonicalModel.config` 本来就记录着"这个已发布模型是用哪个配置训的"，因为这个答案推不出来。torchvision 上尤其明显：`unetplusplus_resnet34` 的任务是 `segmentation`，而两个 torchvision 配置声明的模型分别是 `detect` 和 `instance_segmentation`——任何按 task 猜的规则都会挑错文件，目录直接给出答案。
- 第 3–5 步都会带出 `variant`（`--variant` 显式指定时仍然优先），所以真正训练的是你要的那个模型，而不是配置自己的默认值。第 4 步给的是目录里的规范拼写（`PaDiM` 而不是 `padim`）。
- 第 4、5 步会**替你选一个你没点名的配置**。这是有意的——否则目录里一半模型无法按名字训练——但代价是选择必须可见：`fdh train` 的输出里有 `resolved_config` 和 `resolved_variant` 两个字段，先确认它们再看指标。
- `tests/test_catalog.py::test_every_published_model_is_reachable_by_its_own_name` 对目录全量参数化，往目录里加模型却没配好配置，会在这里红，而不是在别人的训练里。

**注意**：`fdh train <名字>` 训练一个在 `CANONICAL_MODELS` 里的模型时，**默认会把结果发布到 `artifacts/models/published/`**，覆盖 Web UI 正在用的那份权重。冒烟和实验请加 `--no-publish`。

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

自定义损失函数和自研数据增强不是 Ultralytics 的可直接配置参数，YAML 不接受这类字段。本项目是基准平台，不引入自研的损失/增强模块——理由见 `docs/SDK.md` 第 2 节。

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

## Config Profile（`recipe_id`）

Config profile 是某个方法的一份诚实的运行设置集合——用该后端真实的参数名字，锚定到那篇论文——通过 `load_model(..., recipe="patchcore")` 生效。它**不是**新方法，也不带发明出来的缩写；如果某个 profile 真的长出了可测量的架构改动，等改完、测过之后再考虑起名，不是现在。

| `recipe_id` | 适用模型 | Profile 设置了什么 | 锚定论文 |
| :--- | :--- | :--- | :--- |
| `yolov8` | YOLOv8n, YOLOv8s, YOLO11n | Ultralytics 自身的训练器默认参数 | Jocher et al., Ultralytics YOLO (v8/v11), `ultralytics/cfg/default.yaml` |
| `patchcore` | PatchCore, PaDiM, SuperSimpleNet | WideResNet-50 特征、10% coreset、k=9 | Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), CVPR 2022 |
| `rd4ad` | RD4AD, EfficientAD | WideResNet-50 teacher，layer1+2+3，加性 map | Deng & Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding", CVPR 2022 (arXiv:2201.10703) |
| `mambaad` | MambaAD | resnet34 encoder + 上游训练计划 | He et al., "MambaAD", NeurIPS 2024 (arXiv:2404.06564) |
| `moeclip` | MoECLIP, WinCLIP | LoRA rank / experts（上游默认值） | Park et al., "MoECLIP", CVPR 2026 (arXiv:2603.03101) |
| `dinomaly` | Dinomaly | ViT-Base DINOv2 encoder + 上游训练计划 | Guo et al., "Dinomaly", CVPR 2025 (arXiv:2405.14325) |

覆盖范围：以上六个 profile 覆盖已发布目录（`catalog.CANONICAL_MODELS`，20 个模型）里有独立方法可锚定的部分。两类模型**故意不带 profile**，都是事实陈述而非待补的坑：

- torchvision 的检测器/分割器（Faster/Cascade R-CNN、DETR、Mask R-CNN、UNet++、DeepLabV3+）按 torchvision 自身默认值跑，作为标准 baseline。
- STFPM、GANomaly，以及只在 presets 里可用、尚未进目录的 DRAEM / DSR / GLASS / FastFlow / UniNet / AnomalyDINO——它们跑的是各自上游的构造器默认值（`models/anomalib/presets.py` 里每条注释都写明了哪些是上游原值、哪些为织物做了调整）。profile 意味着"设置锚定到那篇论文"，在这个项目里真的复现过之前不发这个名分。执行步骤见 `docs/cloud_training_runbook.md` §10b。

Profile 不只是给 `fdh list-recipes` 看的元数据，是真的接入了训练流程：

```python
model = fdh.load_model("ultralytics", "yolov8n", recipe="yolov8")
# run_experiment(...) 训练前，profile 的设置被解析并挂上：
#   * get_default_hyperparameters() -> 挂到 model._recipe_hparams
# 训练产出的 artifact 会打上 metadata["recipe"] = "yolov8"
```

Profile **只能提供设置**，不能改 loss、不能改结构、不能改增强流程（约束写在
`core/base_recipe.py`，由 `tests/test_recipe_application.py` 钉死）。否则结果表里
标着 "YOLOv8" 的那一行就不是原版 YOLOv8，横向对比不成立。

每个后端只会吸收 profile 里"训练器安全"的那部分参数（`recipes.recipe_trainer_overrides(...)`）：`lr0`/`momentum` 这类真实训练器参数会传进去，而模型构造参数（如 `backbone`）和命名不同的 loss 权重（`box_loss_weight`，Ultralytics 里叫 `box`）不会泄漏进 `train(**kwargs)`。调用方显式传的值总是优先于 profile。六个 profile 目前都已核对为各自后端的真实参数名（`tests/test_recipe_reconciliation.py` 会钉死每个 profile 对应后端的已验证默认值，防止再退化回发明出来的参数）；尚未做完的是真正跑一遍论文数据集复现、填上结果表（见 `docs/REPRODUCTION_PATCHCORE.md`）——设置是对的，GPU 跑分还没跑完。

## 只验证、不训练（`model.weights`）

六个后端的 `model` 段现在都有 `weights` 字段。配合 `train.enabled: false`，一份配置就能表达"拿昨天训好的 checkpoint 再跑一遍验证/导出"：

```yaml
model:
  name: dinov2reg_vit_base_14
  weights: artifacts/models/published/Dinomaly.pt
train:
  enabled: false
val:
  enabled: true
```

anomalib 多一个必填的显式开关 `model.allow_unsafe_checkpoint: true`——它的 checkpoint 是 Lightning pickle，加载会执行任意 Python 代码，所以适配器一直要求显式确认，配置只是把这个确认摆到明面上。只对自己产出的 checkpoint 打开它。

checkpoint 必须与 `model` 段其余字段匹配（Dinomaly/MambaAD 的 encoder、MoECLIP 的架构参数），否则 `load_state_dict` 会因形状不符而失败。

**`train.enabled: false` 又没给 `weights`，但开了 `val`/`export` 的配置会直接报错**（`core/pipeline.py`）。这个组合以前是静默返回空结果——四个异常检测后端此前甚至连 `weights` 字段都没有，也就是说这类配置在任何写法下都跑不出东西，而且不报错。

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
