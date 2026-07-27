# 统一 Interface — 任务清单

> 目标：把"每接一个模型就搬一个仓库"变成"每接一个模型就填一张表"。
> 前置约束：**Track A 必须在补 MambaAD / FabricMamba / DRAEM / AnomalyDiffusion 之前完成**，否则接口要改 4 遍。

## 0. 先厘清三层边界

| 层 | 例子 | 该怎么办 |
|---|---|---|
| 上游**算法本体 / 网络定义 / 预训练权重** | DINOv2、CLIP、PatchCore backbone | **保持 vendored，不重写**。重写只带来复现风险，学术上零收益 |
| 上游**训练循环 / 数据管道 / 评测 / 日志** | `dinomaly_mvtec_sep.py::train()` | **必须换成我们的**（Dinomaly 已经这么做了，见 `models/dinomaly/adapter.py:124`） |
| **自研网络组件** | TextileAttentionNeck、AFDLoss、TPA | ✅ **已删除**——基准平台不该有自己的模型零件，见下 |

### 已完成：清理与基准平台无关的自研模块

`fabric_defect_hub.nn`（backbones + hooks + necks + heads，434 行）、`fabric_defect_hub.optim.losses`（103 行）、`fabric_defect_hub.augmentations`（66 行）在 `src/` `tests/` `tools/` 里全部**零调用**，只有文档在描述它们。同时 `BaseModelRecipe` 的
`configure_loss` / `adapt_architecture` / `configure_augmentations` / `configure_optimizer`
四个 hook 让 profile 有能力改模型本身——一旦生效，结果表里标着 "YOLOv8" 的那行就不是原版 YOLOv8。

以上全部删除，profile 收敛为**纯设置**，并由 `tests/test_recipe_application.py::test_base_recipe_exposes_no_model_modification_hooks` 钉死。
`yolov8` profile 的 `paper_reference` 同步改为 Ultralytics 自身的训练默认值（原先锚定 SPD-Conv 论文，但从未真的改过结构）。
`docs/SDK_AND_NN.md` → `docs/SDK.md`，第 2 节记录了这条边界。

---

## Track A — 冻结 IO 契约（P0）

- [x] **A1 统一 `train` / `predict` / `export` 签名**
  基类声明完整签名，6 个 adapter 全部对齐（anomaly 族补 `config`，detection 族补 `output_dir`）。
  `loader.py` 的 `inspect.signature(model.export)` 运行时嗅探已删除，`import inspect` 一并移除。

- [x] **A2 `TrainConfig` 取代 `dict[str, Any]`**
  `core/train_config.py`：公共字段固定 + `backend_specific` 兜底 + 构造期校验（未知精度、`epochs`/`max_iters` 同时设置、非正值、`backend_specific` 影子键全部报错）。
  翻译表 `TRAIN_CONFIG_KEYS` 声明在各 adapter 上（`lr` → Ultralytics 的 `lr0`、Dinomaly 的 `lr`），`core` 不认识任何具体框架。
  `train()` 同时接受 `TrainConfig` 与原有 dict，旧调用方零改动。

- [x] **A3 `ModelAdapter.capabilities()`**
  `ModelCapabilities`：tasks / prediction_fields / required_annotations / export_targets / supports_amp，词表校验（拼错即报错）。
  6 个 backend 全部声明。过程中发现并纠正一处：torchvision 实际支持 AMP（`engine.run_training` 用了 `torch.autocast` + `GradScaler`），先前按"未验证"记为 False 是错的。

- [x] **A4 统一 pipeline 层**
  `core/pipeline.py`：`BasePipeline`（run 顺序写一次：validate → build_adapter → prepare → train+register → evaluate → export）+ 单一 `RunResult` + `AnomalyPipeline`（predict + `AnomalyEvaluator`，四个异常检测后端此前各抄一遍）。
  六个后端只保留差异钩子（`build_adapter` / `build_train_config` / `evaluate` / 可选 `prepare`、`load_existing_artifact`、`export_config`）。
  `run_from_config` / `run_from_yaml` / `XxxRunResult` 三个公共名字全部保留，调用方零改动。
  净减少 128 行；`tests/test_pipeline_contract.py` 钉死 run 顺序、分支条件，以及"没有后端可以自己重写 `run()`"。

- [x] **A5 契约一致性测试**
  `tests/test_adapter_contract.py`（签名、capabilities、export 诚实性、TrainConfig 翻译）
  \+ `tests/test_data_adapter_contract.py`（DataAdapter 契约）
  \+ `tests/test_train_config.py`（校验与翻译本身）。
  参数化覆盖全部 6 个已注册 backend。

## Track B — 数据侧接口（P0）

> B1 / B2 / B5 / B6（组件注册表、装配层 `ComposedModel`、用装配层落地模型）已移出本阶段：
> 那些是"造模型"的能力，不属于基准平台。相关代码已按上一节删除。

- [x] **B3 `DataAdapter` 接口（`Sample` → backend batch）**
  `core/data_adapter.py`：统一构造签名、`__len__`/`__getitem__`、`batch_spec()`（item 形态 / 张量布局 / 归一化统计 / mask 语义 / 图像尺寸，全部词表校验）、`collate_fn`、`build_dataloader()`。
  `models/mambaad/data.py`、`models/moeclip/data.py`、`models/torchvision/dataset.py` 四个转换类全部实现它，**转换实现保持各自独立**。
  副产物：MambaAD 用 ImageNet 统计、MoECLIP 用 CLIP 统计，这个区别以前只存在于源码里，现在是可查询、可写进 run log 的声明。

- [x] **B4 训练设施可控可记录**
  `core/provenance.py`：`describe_training`（optimizer/scheduler/精度从 live 对象读取——Ultralytics 的 `"auto"` 记运行时解析出的真实类，Lightning 后端读 trainer，PatchCore 无优化器如实记 `none`）+ `collect_provenance`（仓库 commit + 每个 submodule 的 pinned commit 及 clean/modified 状态）。
  两个账本同一个块：训练侧 `weight_registry.record_weight`（`Artifact.metadata["training"]` + `"batch_spec"` 随 `artifact_metadata` 落盘）、评测侧 `reporting.append_run_log`。C3 的遗留（hash 进 run log）一并关闭。
  守护：`tests/test_provenance.py`。

## Track C — vendor 边界纪律

- [x] **C1 消除 `sys.path` 污染**
  `core/vendor.py::VendoredRepo`：把 MoECLIP 已验证的隔离导入机制提取出来，两个 checkout 共用。
  导入在一个窗口内完成（checkout 置于 `sys.path[0]`、同名模块临时移出 `sys.modules`），结束后把自己的模块取回私有缓存、还原原状、移除 `sys.path` 条目。
  Dinomaly 原来那个永久 `sys.path` bootstrap 已删除——它此前只是"碰巧先加载"才没出事。
  新增一个 vendored 仓库现在是三行声明（`owned_roots` / `entry_modules` / 缺失提示）。

- [x] **C2 上游模块名只允许出现在 `vendor.py`**
  `dinomaly/adapter.py` 三处 `from dataset import ...` / `from models.uad import ...` 全部改为 `import_vendor()[...]`。
  `tests/test_vendor_boundary.py` 用 `ast` 解析 `src/` 下每个文件（不是 grep：字符串里的同名词不会误报，函数体内的延迟导入不会漏报），并附行为断言——导入后端后 `sys.modules` 不留痕、checkout 不留在 `sys.path`，以及两个仓库的 `utils` 确实是两个不同对象。

- [x] **C3 `components/` 记录上游 commit hash + 是否被修改**
  已经由 git submodule 机制提供：`git submodule status` 同时给出 pinned commit 和 `+`（脏/偏离）标记。
  hash 写进 run log / 结果表已随 B4 落地（`core/provenance.py::vendored_components`）。

- [ ] **C4 clean-room 实现要有对照验证**
  MambaAD 是自己重写的（`models/mambaad/adapter.py:2`），需要一个与论文/上游数值对齐的验证记录，否则重写反而是风险点。

## Track D — 交付物（P1）

- [x] **D1 一页 interface spec** → `docs/INTERFACE_SPEC.md`（五个抽象 + 数据契约 + 复现契约 + "新增模型只需实现 3 个方法"）
- [x] **D2 更新 `docs/EXTENDING.md` 为"新增模型三步走"**
- [ ] **D3 接口冻结打 tag**，之后改接口需要走一次评审

---

## 剩余工作

| 优先级 | 项 | 说明 |
|---|---|---|
| P1 | C4 | MambaAD clean-room 数值对照（需 GPU 上与论文/上游数值对齐，本地做不了） |
| P1 | D3 | 接口冻结打 tag，之后改契约走评审 |

华纺数据集接入走 `DatasetAdapter`，不受本清单影响，可并行推进。

## 已冻结的契约（改动需评审）

| 契约 | 位置 | 由谁守住 |
|---|---|---|
| `Sample` / `Prediction` / `ExperimentResult` | `core/types.py` | `schemas/*.schema.json` |
| `ModelAdapter`（capabilities / train / predict / export） | `models/base.py` | `tests/test_adapter_contract.py` |
| `ModelCapabilities` | `models/base.py` | 同上（含词表校验） |
| `TrainConfig` + `TRAIN_CONFIG_KEYS` | `core/train_config.py` + 各 adapter | `tests/test_train_config.py` |
| `DataAdapter` / `BatchSpec` | `core/data_adapter.py` | `tests/test_data_adapter_contract.py` |
| `BasePipeline` / `RunResult` | `core/pipeline.py` | `tests/test_pipeline_contract.py` |
| vendor 边界（上游模块名只在 `vendor.py`） | `core/vendor.py` | `tests/test_vendor_boundary.py` |
| provenance 块 + 训练设施记录 | `core/provenance.py` | `tests/test_provenance.py` |
| `DatasetAdapter` | `datasets/base.py` | — |
| `Evaluator` | `evaluation/base.py` | — |
| 配置档案只提供设置 | `core/base_recipe.py` | `tests/test_recipe_application.py` |
