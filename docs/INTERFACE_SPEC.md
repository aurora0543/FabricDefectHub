# FabricDefectHub 统一接口规格（一页版）

> 目的：让"接入一个新模型"从"搬一个仓库"变成"实现 3 个方法"。
> 状态：**已冻结**。改动任何一份契约需先改对应测试并走评审（见 `docs/INTERFACE_TASKS.md` 末表）。

## 五个抽象

| 抽象 | 职责 | 位置 | 守护测试 |
|---|---|---|---|
| `ModelAdapter` | 一个模型后端 = `capabilities` / `train` / `predict` / `export` 四个方法，六个后端签名完全一致 | `models/base.py` | `test_adapter_contract.py` |
| `DataAdapter` | `Sample` 列表 → 该后端的训练 batch。转换实现各自独立，接口统一 | `core/data_adapter.py` | `test_data_adapter_contract.py` |
| `DatasetAdapter` | 磁盘上的数据集 → 统一的 `Sample` 列表（9 个数据集已实现） | `datasets/base.py` | — |
| `Evaluator` | `(Sample, Prediction)` 对 → 指标 dict，按 task 注册 | `evaluation/base.py` | — |
| `BasePipeline` | 一次 config 驱动的运行：validate → build → prepare → train → evaluate → export，顺序只写一次，后端只填差异钩子 | `core/pipeline.py` | `test_pipeline_contract.py` |

## 数据契约（跨层的公共类型）

- **`Sample` / `Prediction`**（`core/types.py`）：所有数据集产出 `Sample`，所有模型产出 `Prediction`。由 `schemas/*.schema.json` 钉死。
- **`TrainConfig`**（`core/train_config.py`）：公共超参一套名字（`epochs / lr / batch_size / image_size / device / seed / precision / work_dir`），构造期校验，未知键报错；各后端用 `TRAIN_CONFIG_KEYS` 翻译成自家词汇（如 `lr` → Ultralytics 的 `lr0`）。
- **`ModelCapabilities`**：后端声明支持的 task、会填哪些 `Prediction` 字段、需要什么标注、能导出什么格式、是否支持 AMP。词表校验，拼错即 import 报错。评测器据此判断某个指标是否可算，不再靠人记。
- **`BatchSpec`**：`DataAdapter` 声明 item 形态 / 张量布局 / 归一化统计 / mask 语义。MambaAD 用 ImageNet 统计、MoECLIP 用 CLIP 统计这类差异从此可查询。
- **`RunResult` / `Artifact`**：一次运行 / 一份权重的统一句柄，六个后端共用。

## 每次运行记录什么（复现契约）

两个 append-only 账本共享同一个 provenance 块（`core/provenance.py`）：

- **训练侧** `artifacts/models/weight_manifest.jsonl`（`weight_registry.record_weight`）
- **评测侧** run log JSONL（`reporting.append_run_log`）

每行记录：UTC 时间戳、仓库 commit、主机名、**每个 vendored 上游 checkout 的 pinned commit 及是否被改动**（`git submodule status` 语义），训练侧另含实际使用的 **optimizer / scheduler / 精度**（从 live 对象读取，不是手抄声明）和 **`BatchSpec`**。任何一行结果都能回答"是哪份代码、什么设施跑出来的"。

## vendor 边界

上游研究仓库以 git submodule 形式 vendored（`components/`），算法本体不重写；上游模块名（`utils`、`dataset`…）**只允许出现在该后端的 `vendor.py`**，经 `core/vendor.py::VendoredRepo` 隔离导入，不污染 `sys.path` / `sys.modules`。由 `test_vendor_boundary.py`（AST 级检查）钉死。

## 新增一个模型 = 实现 3 个方法

```python
@register_model("newmodel")
class NewModelAdapter(ModelAdapter):
    TRAIN_CONFIG_KEYS = {"epochs": "epochs", "lr": "lr", ...}

    def capabilities(self) -> ModelCapabilities: ...   # 声明能做什么
    def train(self, config) -> Artifact: ...           # Sample 进，权重出
    def predict(self, samples, artifact=None,
                output_dir=None, config=None) -> list[Prediction]: ...
```

`export` 不支持时抛 `NotImplementedError` 即可（capabilities 里 `export_targets=()` 已声明）。接完跑 `pytest tests/test_adapter_contract.py`——没实现完，测试直接红。操作步骤见 `docs/EXTENDING.md`"新增模型三步走"。
