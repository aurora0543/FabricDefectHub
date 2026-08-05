---

# Extending FabricDefectHub

本文档说明如何扩展数据集、模型 Backend 及配置 Recipe，以及如何在缺失部分数据集或依赖库时保持系统优雅降级。

---

## 1. 数据集与 Backend 可用性决策树

通过三大核心模块评估当前机器的运行环境，实现优雅降级：

* **`core.availability`**：检测数据集目录（非空且非悬空软链接）与 Backend 框架是否可用。
* **`core.decision`**：选择可用的数据集（若指定数据集不可用，按字母顺序自动选择同一许可集合中的备用数据集）。
* **`training.apply_available_dataset`**：应用选择结果并抛出警告；若无可用数据集则明确报错。

运行以下命令直观查看诊断结果：

```bash
fdh doctor

```

### 添加新数据集

1. 继承 `DatasetAdapter` 并使用 `@register_dataset("my-dataset")` 注册（参考 `datasets/`）。
2. 在 `core/dataset_capabilities.py` 中声明能力：
```python
register_capabilities(
    "my-dataset",
    default_root="data/MyDataset",
    roles={"anomaly_train"},
    tasks=("anomaly",),
)

```



---

### 添加新模型 — 路径选择

| 路径 | 适用场景 | 成本/要求 | 现有参考 |
| --- | --- | --- | --- |
| **A. Alias + Preset** | 已是 `anomalib.models` 中的类 | 修改 `MODEL_ALIASES` 与 `MODEL_PRESETS` | `models/anomalib/presets.py` |
| **B. Vendored Submodule** | 上游提供独立可运行仓库 | `components/` Git 子模块 + `vendor.py` + Adapter | `components/dinomaly` |
| **C. Clean-room** | 上游无独立运行代码/依赖复杂 | 从零重构架构，仅复用论文参数 | `models/mambaad/` |

#### 路径 A 步骤：

1. `MODEL_ALIASES` 中添加论文别名（小写 key $\rightarrow$ anomalib 类名）。
2. `MODEL_PRESETS` 中添加默认参数（即使与上游一致也需声明）。
3. 无像素级输出的模型需记入 `IMAGE_LEVEL_ONLY`。
4. 若需外部数据集，在 `AnomalibAdapter._validate_model_kwargs` 添加前置校验。
5. 运行测试：`pytest tests/test_anomalib_config.py`。

#### 路径 C 规范：

* 基础模块写在独立的私有包内（如 `models/mambaad/`），不预先抽离共享库。

---

### 添加新模型 Backend（三步法）

1. **实现 `ModelAdapter` 接口**（`models/base.py`）并使用 `@register_model("my-backend")` 注册：
* 必须实现 `capabilities()`、`train()`、`predict()`、`export()` 四个标准方法。
* 在 `train()` 中首先调用 `resolve_train_config(config, self.TRAIN_CONFIG_KEYS)`。


2. **实现 `DataAdapter` 接口**（`core/data_adapter.py`）：
* 在 `models/<backend>/data.py` 中实现 `Sample` $\rightarrow$ Batch 的转换逻辑。


3. **完成双向注册**：
* 在 `loader._MODEL_BACKEND_MODULES` 中注册导入路径。
*（单类/零样本异常检测）在 `training._BACKEND_TRAINABLE_DATASETS` 中注册。
* 模块需提供 `list_supported_variants()` 函数。


4. 运行契约测试：`pytest tests/test_adapter_contract.py tests/test_data_adapter_contract.py`。

---

### 发布模型到前端

在 `catalog.CANONICAL_MODELS` 中配置：

* `config` 需绑定 `configs/models/` 下的真实配置文件。
* 只有训练并生成 Checkpoint 后才可添加配置（未发布的模型可通过 Preset 访问）。
* 执行 `fdh train <model>` 默认会覆盖现有的 Checkpoint（测试运行可使用 `--no-publish`）。

---

## 2. 配置 Recipe (`fabric_defect_hub.recipes`)

Recipe 是针对特定论文预设的运行参数集合（仅限超参数，不可修改损失函数或网络结构）：

1. 继承 `BaseModelRecipe` 并使用 `@register_recipe("my-method")` 注册。
2. `paper_reference` 必须引用真实论文。
3. `get_default_hyperparameters()` 需使用目标 Backend 的原生参数名。
4. 编写防漂移校验测试（参考 `tests/test_recipe_reconciliation.py`）。

---

## 3. 动态参数覆盖 (`--set`)

通过 `--set path.to.key=value` 可以按点号路径直接覆盖配置文件中的任意字段（优先级最高，自动解析 YAML 数据类型）：

```bash
fdh train configs/models/patchcore_textile.yaml \
  --set train.model_kwargs.coreset_sampling_ratio=0.05 \
  --set train.model_kwargs.lr=0.0005

```