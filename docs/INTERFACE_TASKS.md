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

## Track A — 冻结 IO 契约（P0，约 3 天）

- [ ] **A1 统一 `train` / `predict` / `export` 签名**
  现状：`predict` 已分裂成两族——anomaly 族 `(samples, artifact, output_dir=None)`，detection 族 `(samples, artifact=None, config=None)`；`export` 有的多一个 `config`。
  验收：删掉 `loader.py:211` 的 `inspect.signature(model.export)` 运行时嗅探，6 个 adapter 签名与基类完全一致。

- [ ] **A2 `TrainConfig` 取代 `dict[str, Any]`**
  公共字段固定（`epochs / lr / batch_size / img_size / device / seed / precision / work_dir`）+ `backend_specific: dict` 兜底 + 构造期校验（未知键报错而不是静默忽略）。
  验收：6 个 backend 的 `config.py` 里公共超参只有一套名字。

- [ ] **A3 `ModelAdapter.capabilities()`**
  返回：支持的 task、需要的标注级别、支持的 export target、是否支持 AMP/混合精度。
  验收：evaluator 依据 capability 决定算不算 pixel-AUROC；周报"退化率指标不适用"的判断由代码给出，不再靠人记。与 `core/dataset_capabilities.py` 形成对称。

- [ ] **A4 统一 pipeline 层**
  现状：5 份平行的 `run_from_config` / `run_from_yaml` + 各自的 `XxxRunResult` dataclass，无共同基类。
  验收：一个 `BasePipeline` + 一个 `RunResult`，各 backend 只写差异部分。

- [ ] **A5 契约一致性测试 `tests/test_adapter_contract.py`**
  参数化跑遍所有已注册 backend：签名、capabilities、`Prediction` 字段填充是否与声明一致。
  验收：新加一个 backend 但没实现完接口 → 测试直接红。**这是整套接口能不能守住的关键。**

## Track B — 数据侧接口（P0）

> B1 / B2 / B5 / B6（组件注册表、装配层 `ComposedModel`、用装配层落地模型）已移出本阶段：
> 那些是"造模型"的能力，不属于基准平台。相关代码已按上一节删除。

- [ ] **B3 `DataAdapter` 接口（`Sample` → backend batch）**
  周报第 4 条"各模型独立实现 DataLoader"的**决定是对的，不要回退**；要做的是把"转换"本身变成接口：统一签名、张量布局、归一化元信息、mask 语义、collate 契约，各模型实现各自的类，由 registry 解析。
  现状散落于 `models/moeclip/data.py`、`models/mambaad/data.py`、`models/torchvision/dataset.py`。
  对导师的说法：**"实现是独立的，接口是统一的"**。

- [ ] **B4 训练设施可控可记录**（P1）
  现状 Dinomaly 直接用上游 `StableAdamW` + `WarmCosineScheduler`。基准平台不需要自研优化器，只需要**每次跑用了什么 optimizer/scheduler/精度被完整记录进 run log**，否则结果不可复现。

## Track C — vendor 边界纪律（P1，约 1 天）

- [ ] **C1 消除 `sys.path` 污染**
  `models/dinomaly/vendor.py` 把上游的 `utils` / `dataset` / `models` 这些通用名塞进全局 `sys.modules`，再多接两个仓库必然撞名。改为 importlib 独立命名空间加载。

- [ ] **C2 上游模块名只允许出现在 `vendor.py`**
  adapter 里不得直接 `from dataset import MVTecDataset`（现状 `dinomaly/adapter.py:151` 就是这样）。
  验收：一条 grep 规则进 CI。

- [ ] **C3 `components/` 记录上游 commit hash + 是否被修改**（复现性，审稿人会问）

- [ ] **C4 clean-room 实现要有对照验证**
  MambaAD 是自己重写的（`models/mambaad/adapter.py:2`），需要一个与论文/上游数值对齐的验证记录，否则重写反而是风险点。

## Track D — 交付物（P1）

- [ ] **D1 一页 interface spec**（给导师：五个抽象 + 数据契约 + "新增模型只需实现 3 个方法"）
- [ ] **D2 更新 `docs/EXTENDING.md` 为"新增模型三步走"**
- [ ] **D3 接口冻结打 tag**，之后改接口需要走一次评审

---

## 建议排期

| 周 | 内容 |
|---|---|
| 第 1 周 | A1–A5 + C1–C2 → **接口冻结** |
| 第 2 周 | B1–B4 |
| 第 3 周 | B5–B6，并用新接口补 DRAEM / FabricMamba，验证"填表不改框架" |

华纺数据集接入走 `DatasetAdapter`，不受本清单影响，可并行推进。
