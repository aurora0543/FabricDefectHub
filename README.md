# FabricDefectHub

面向真实布匹质检场景的统一缺陷检测 Benchmark 平台。

> [!IMPORTANT]
> 项目目前处于设计与早期开发阶段。README 描述的是目标架构和实施路线，接口与目录会随着首个可运行闭环逐步稳定。

## 项目愿景

FabricDefectHub 不重新实现所有模型，而是在统一的数据、模型、评测和端侧性能接口之上，整合三类缺陷检测范式：

| 后端 | 代表模型 | 适用场景 |
| --- | --- | --- |
| `ultralytics` | YOLOv8n/v8s、YOLO11n | 有缺陷标注，追求端侧实时检测 |
| `torchvision` | Faster R-CNN、Mask R-CNN（ResNet50-FPN v1/v2） | 有标注条件下的两阶段检测/实例分割对比 |
| `anomalib` | PatchCore、PaDiM、RD4AD、EfficientAD、SuperSimpleNet | 缺陷样本少，只有正常样本或少量异常样本 |

项目的主要工作集中在：

- 统一适配不同数据集、任务和算法框架；
- 标准化准确率与工业指标评测；
- 对 PyTorch、ONNX Runtime、TensorRT 等推理后端进行公平测试；
- 支持真实布匹产线中的设备、工况和欠检/过检分析；
- 用统一结果契约连接实验后端与可视化前端。

## 开发原则

项目采用“前端原型先行，但尽快打通真实闭环”的方式推进：

1. 使用 mock 数据完成可点击的前端原型，同时确定数据契约；
2. 打通 `YOLOv8n + 一个数据集 + PC 推理 + 结果 JSON` 的最小闭环；
3. 让前端读取真实实验 JSON，验证契约是否合理；
4. 再依次接入 Anomalib、MMDetection、ONNX/TensorRT 和端侧设备测试。

前端服务于统一 Benchmark 设计，不作为一个脱离真实训练、推理和评测流程的独立项目长期开发。

## 前端原型

第一阶段使用 mock 数据实现以下页面：

- **数据集页**：数据集名称、任务类型、样本数、类别、标注格式和工况信息；
- **模型页**：YOLO、MMDetection、Anomalib 模型卡片，展示任务能力、参数量和端侧导出支持；
- **实验配置页**：选择数据集、模型、设备、输入尺寸和精度模式；
- **结果页**：按 mAP/AUROC、P50 延迟、FPS、内存、功耗和模型大小生成排行榜；
- **实验详情页**：展示混淆矩阵、PR 曲线、缺陷可视化和逐样本预测结果。

Mock 数据必须遵循与真实后端相同的接口，避免后续接入模型时重写页面。

## 核心架构

```text
DatasetAdapter
  不同公开数据集、企业数据 -> 统一内部样本描述

ModelAdapter
  Ultralytics / MMDetection / Anomalib -> 统一 train、predict、export 接口

Evaluator
  根据任务与模型能力 -> mAP、F1、AUROC、AUPRO、mIoU 等指标

BackendProfiler
  PyTorch / ONNX Runtime / TensorRT -> 延迟、FPS、内存、功耗、模型大小
```

### DatasetAdapter

不同数据集共享统一的 `Sample` 元信息，但保留各任务原生所需的标签语义，不把所有数据强制转换成同一种标注格式：

| 任务 | 标签字段 |
| --- | --- |
| 目标检测 | `boxes`、`labels` |
| 实例/语义分割 | `masks`、`labels` |
| 异常检测 | `is_anomalous`、可选 `anomaly_mask` |

概念结构如下：

```json
{
  "id": "sample-0001",
  "image_path": "data/images/0001.jpg",
  "task": "detection",
  "annotations": {
    "boxes": [[120, 64, 238, 180]],
    "labels": ["broken_end"]
  },
  "metadata": {
    "fabric_type": "cotton",
    "lighting": "line_scan"
  }
}
```

### ModelAdapter

各模型后端通过统一生命周期接入，但允许后端保留自己的配置：

```text
train(config) -> Artifact
predict(samples, artifact) -> list[Prediction]
export(artifact, target) -> ExportedArtifact
```

统一 `Prediction` 由可选字段组成：

```json
{
  "sample_id": "sample-0001",
  "boxes": [[121, 66, 236, 178]],
  "labels": ["broken_end"],
  "scores": [0.93],
  "masks": null,
  "anomaly_score": null,
  "anomaly_map": null
}
```

YOLO 和 Faster R-CNN 填充 `boxes`、`labels`、`scores`；Mask R-CNN 额外填充 `masks`；PatchCore、PaDiM 等填充 `anomaly_score` 和 `anomaly_map`。

### Evaluator

评测器根据数据任务、标签可用性和模型能力选择指标，而不是使用单一的 `accuracy`：

| 类型 | 建议指标 |
| --- | --- |
| 目标检测 | mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1 |
| 分割 | mIoU、Dice、像素级 F1 |
| 异常检测 | 图像级/像素级 AUROC、AUPRO、F1 |
| 工业质检 | 欠检率、过检率、单位布长告警数 |

### BackendProfiler

性能测试记录运行环境和统计口径，保证不同模型结果可比较：

- P50/P95 延迟、吞吐量和 FPS；
- 峰值 RAM/VRAM、功耗和模型文件大小；
- 设备、运行时、精度模式、输入尺寸、批大小和预热次数；
- PyTorch、ONNX Runtime、TensorRT 等运行时信息。

## 统一实验结果契约

训练、推理、评测和性能测试最终汇总为统一的 `ExperimentResult`。以下为最小示例，正式 Schema 将随首个闭环固化：

```json
{
  "experiment_id": "exp-2026-001",
  "model": {
    "name": "yolov8n",
    "backend": "ultralytics",
    "task": "detection"
  },
  "dataset": {
    "name": "fabric-demo",
    "split": "test"
  },
  "runtime": {
    "device": "Jetson Orin Nano",
    "engine": "TensorRT",
    "precision": "fp16",
    "input_size": [640, 640]
  },
  "metrics": {
    "map50": 0.81,
    "latency_ms_p50": 12.4,
    "fps": 80.6
  },
  "artifacts": {
    "predictions": "artifacts/exp-2026-001/predictions.json",
    "model": "artifacts/exp-2026-001/model.engine"
  }
}
```

新增后端时，只要能够生成符合契约的预测与实验结果，前端和排行榜就无需针对框架重写。

## 计划中的目录结构

以下目录会随对应阶段逐步创建，目前并不代表均已实现：

```text
FabricDefectHub/
├── frontend/              # 数据集、模型、实验和结果可视化
├── configs/               # 数据集、模型、运行时和实验配置
├── data/                  # 本地数据（默认不纳入版本控制）
├── schemas/               # Sample、Prediction、ExperimentResult Schema
├── src/fabric_defect_hub/
│   ├── datasets/          # DatasetAdapter 与数据集实现
│   ├── models/            # ModelAdapter 与三类框架后端
│   ├── evaluation/        # 任务指标与工业指标
│   └── profiling/         # 多运行时、端侧性能测试
├── tools/                 # 数据转换、导出和可视化工具
└── tests/                 # 单元测试与最小闭环集成测试
```

## 路线图

### Phase 0：契约与可点击原型

- [ ] 定义 `Sample`、`Prediction`、`ExperimentResult` JSON Schema；
- [ ] 使用同一套 mock 契约完成五个核心页面；
- [ ] 明确检测、分割、异常检测的能力与指标映射。

### Phase 1：最小真实闭环

- [ ] 接入一个公开或脱敏布匹数据集；
- [ ] 完成 Ultralytics `YOLOv8n` 的 PC 推理；
- [ ] 输出真实预测与实验结果 JSON；
- [ ] 在前端展示真实排行榜和实验详情。

### Phase 2：统一算法 Benchmark

- [ ] 接入 Anomalib 代表模型与异常检测指标；
- [ ] 接入 MMDetection 代表模型与检测/分割指标；
- [ ] 完成统一训练、预测、评测和制品管理流程。

### Phase 3：部署与工业评测

- [ ] 支持 ONNX Runtime、TensorRT 和精度模式切换；
- [ ] 在 PC 与 Jetson 等目标设备上进行可复现性能测试；
- [ ] 加入功耗、模型大小、欠检率和过检率；
- [ ] 支持真实工况元信息和企业数据适配。

## 公平评测要求

提交或发布 Benchmark 结果时，至少同时记录：

- 数据集版本、划分方式和预处理流程；
- 模型版本、权重来源、训练配置和随机种子；
- 硬件、软件栈、推理引擎和精度模式；
- 输入尺寸、批大小、预热次数和统计样本数；
- 指标实现、阈值和后处理参数。

缺少上述上下文的单一 mAP、AUROC 或 FPS 数值不应直接用于模型排名。

## License

本项目采用 [MIT License](LICENSE)。第三方框架、模型权重和数据集仍遵循各自的许可证与使用条款。

## 联系方式

项目由北京理工大学相关研究团队发起并处于持续开发中。合作、数据集适配或 Benchmark 提交建议请通过 Issue 与维护者联系。
