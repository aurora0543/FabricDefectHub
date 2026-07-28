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

## 门面层（`api.py`）

上面五个抽象是**契约**，面向的是"接一个新模型的人"。另有一层**门面**，面向的是"用这个项目的人"——`fabric_defect_hub.api`，从包根导出：

```python
fdh.list_models() / list_datasets() / list_pretrained()   # 有什么
fdh.load_config(model, dataset=..., epochs=...)           # → RunConfig
fdh.train(cfg) / fdh.predict(...) / fdh.evaluate(...)     # 三个动词
fdh.from_pretrained(key)                                  # 已发布的权重
```

门面**不是第六个抽象**，它不新增任何概念，只是把 `training.run_train` / `predict.run_predict` / `catalog` 这些已有入口用一致的名字暴露出来（此前它们没有从包根导出，`import fabric_defect_hub` 够不到 config 驱动的那条路径，而那恰恰是 `fdh train` 实际走的路径）。

**约束（由 `tests/test_api_facade.py` 在 AST 层钉死）**：

- 门面函数体内**不允许出现任何后端名字**——不管是 `if backend == "anomalib"` 还是以后端名为键的 dict 查找，两者耦合度相同。
- 每个公开函数**必须**委派进契约层模块（`training` / `predict` / `loader` / `catalog` / `core.registry`），委派不到任何东西的函数说明它在自己干活。
- `api.py` 模块级不得 import 任何深度学习框架——`import fabric_defect_hub` 在没装任何后端的机器上也要能做发现和配置。

理由和五个抽象的理由是同一个：一个友好的扁平接口正是"就加这一个 anomalib 特例"最容易堆积的地方，堆起来之后项目就有了两条流水线——被 `test_pipeline_contract.py` 守着的那条，和它看不见的影子那条。确实需要按后端区分的知识（如"run length 写在哪个 key"）放在拥有它的那一层（`training.RUN_LENGTH_KEYS`），门面只做查表。

改动门面的公开签名与改动上面五个契约同级，需先改测试并走评审。
