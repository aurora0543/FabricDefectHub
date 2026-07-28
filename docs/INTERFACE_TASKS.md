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

- [ ] **C4 clean-room 实现要有对照验证**（阻塞：需 GPU + 数据集，本地 Mac 做不了）
  MambaAD 是自己重写的（`models/mambaad/adapter.py:2`），需要一个与论文/上游数值对齐的验证记录，否则重写反而是风险点。
  待云端执行的具体动作：在有 MVTec AD 的 GPU 机器上跑 `fdh` 的 mambaad 后端（ResNet34 teacher，上游默认超参已内置于 `presets.DEFAULT_TRAIN_KWARGS`），对照 MambaAD 论文 Table 1 的 MVTec image-AUROC（多类统一训练，论文报 97.8）；差距 >1 个点即视为未对齐，需逐层排查 scan/SSM。结果（含 run log 行）存入 `docs/`，作为 clean-room 的验证记录。

## Track D — 交付物（P1）

- [x] **D1 一页 interface spec** → `docs/INTERFACE_SPEC.md`（五个抽象 + 数据契约 + 复现契约 + "新增模型只需实现 3 个方法"）
- [x] **D2 更新 `docs/EXTENDING.md` 为"新增模型三步走"**
- [x] **D3 接口冻结打 tag** → `interface-freeze-v1.0`（含 A1–A5 / B3–B4 / C1–C3 / D1–D2 全部落地），之后改"已冻结的契约"表中任何一项需先改守护测试并走评审

## Track E — 门面层与后端广度（冻结之后）

Track A–D 面向"接一个新模型的人"，这一条面向"用这个项目的人"。两者不冲突：门面建在已冻结的契约之上，不新增抽象。

- [x] **E1 anomalib 后端从 6 个模型扩到 14 个**
  `MODEL_ALIASES` / `MODEL_PRESETS` 新增 STFPM、UniNet（师生）、DRAEM、DSR（重构）、GANomaly（对抗）、FastFlow（流）、GLASS（合成异常 + 判别器）、AnomalyDINO（零样本 DINOv2）。全部是配置工作，零 adapter 改动——这条本身就是"新增模型 = 填表"的检验。
  三处**在实现中发现、而非事先假设**的事实：
  1. GANomaly 的推理输出只有 image-level score，没有 anomaly map（对照 `GanomalyModel.forward` 确认）。`capabilities()` 因此改为 per-model：`presets.IMAGE_LEVEL_ONLY` 里的模型不再声明 `anomaly_map`，否则评测器会以为像素级 AUROC/AUPRO 可算。
  2. DRAEM 在 `dtd_dir` 缺失时**不报错**，而是 `download_and_extract` 拉 ~600MB。`_validate_model_kwargs` 改为提前拒绝，下载需显式 `allow_dtd_download: true`。
  3. GANomaly 的 `batch_size` 是构造器参数、`TrainConfig.batch_size` 到不了它——但在 anomalib 2.5.0 里它只用于两个没人读的 label buffer，所以只记录、不加守卫（加了就是断言一个此版本并不存在的耦合）。
  `tests/test_anomalib_config.py` 新增两条反射测试：每个 preset 的 key 必须是**已安装的** anomalib 的真实构造器参数；`IMAGE_LEVEL_ONLY` 必须与各模型 `torch_model.forward` 的实际输出一致。二者都在升级打破它的那次失败，而不是几个月后。
  STFPM 与 GANomaly 进 `catalog.CANONICAL_MODELS`（无外部数据依赖，任何机器 `fdh train` 即可发布）；其余六个留在 presets 可用但不进目录，避免 UI 下拉框出现点了没权重的条目。

- [x] **E2 门面层 `api.py`**
  `load_config` / `train` / `predict` / `evaluate` / `from_pretrained` / `list_models` / `list_datasets` / `list_pretrained`，从包根导出；`fdh models` 子命令是同一查表的 CLI 视图。
  背景：此前 `import fabric_defect_hub` 只够得到可组合的低层入口（`load_dataset` + `load_model` + `run_experiment`，要自己拼 `ModelInfo`/`RuntimeInfo`），而 config 驱动的 `run_train`/`run_predict`——`fdh train` 实际走的那条——根本没导出。
  约束由 `tests/test_api_facade.py` 在 AST 层钉死：门面函数体内不得出现任何后端名字（if 链和以后端名为键的 dict 查找同罪），每个公开函数必须委派进契约层，`api.py` 模块级不得 import 任何深度学习框架。
  确实需要按后端区分的知识提到契约层：`training.RUN_LENGTH_KEYS`（各后端的 run length 写在哪个 key、单位是 epoch 还是 iteration），由 `test_train_config.py::test_run_length_keys_match_test_speed_overrides` 与 `_apply_test_speed_overrides` 互钉。
  顺带统一：六个后端的 presets 模块现在都有 `list_supported_variants()`，此前是 `list_supported_models` / `list_supported_variants` / 什么都没有三种写法。

- [x] **E4 模型关键词解析：从"只认配置里字面写了的名字"改成三级解析**
  改之前，已发布目录 20 个模型里只有 10 个能用 `fdh train <名字>` 训练：PaDiM / RD4AD / EfficientAD / SuperSimpleNet / WinCLIP / cascadercnn / detr_resnet50 / unetplusplus / deeplabv3plus 报"找不到"（没有任何配置声明它们），PatchCore 报歧义（三个配置都声明了）。ultralytics 之所以正常，只是因为它碰巧有 `variants:` 块把三个变体都登记了。
  `resolve_model_config_and_variant` 现在是：路径 → 文件名 → **声明**（恰好命中一个才用，多个则下沉而非报错）→ **目录**（读 `CanonicalModel.config`，不是启发式：这个字段本来就记录着"该已发布模型用哪个配置训"，torchvision 上尤其关键——`unetplusplus_resnet34` 是 `segmentation`，两个 torchvision 配置声明的却是 `detect` 和 `instance_segmentation`，按 task 猜必错）→ **后端支持的变体**（落到 `<backend>_example.yaml`，覆盖 FastFlow/GLASS 这些可运行但未发布的）。规则一句话：**裸模型名 → 该后端的通用配置；专用配置按文件名寻址。**
  目录条目 STFPM / GANomaly 的 `config` 字段同步改成它们自己的配置文件（此前写的是 `anomalib_example.yaml`，与实际解析结果不符）。
  后两级会替调用者选一个他没点名的配置，因此 `fdh train` 输出新增 `resolved_config` / `resolved_variant` 两个字段——静默选择加事后不可见才是真风险。
  `tests/test_catalog.py::test_every_published_model_is_reachable_by_its_own_name` 对目录全量参数化，往目录加模型却没配好配置会在这里红。现在 20/20 可达。
  连带发现并修掉一个更大的脚坑：`fdh train` 训练目录内模型时**默认发布**，会覆盖 Web UI 正在用的权重，而 CLI 此前没有关闭开关（我自己就用一次 8 张图 1 epoch 的冒烟跑覆盖过一次 `published/PaDiM.ckpt`，已删除）。新增 `--no-publish`。

- [x] **E5 收尾排查：门面/UI/评测三处一致性缺口**
  E1–E4 落地后按"整条链路"而不是"改过的地方"重新走了一遍，查出并修掉三处：
  1. **`fdh evaluate` 结构上产不出像素级指标。** `run_evaluate` 没有 `output_dir` 参数，而各 adapter 只在拿到写入目录时才填 `Prediction.anomaly_map`，`AnomalyEvaluator` 又只从这个字段算像素指标——于是无论模型多有像素能力，`fdh evaluate` 永远只返回图像级指标。已贯通 `run_evaluate` → `run_predict` → CLI `--output-dir` → `api.evaluate(output_dir=)`。用真的 FastFlow checkpoint 验证：不给目录 5 个指标，给了之后多出 `pixel_auroc` / `pixel_aupro` / `pixel_f1` / `iap`。
  2. **UI 分不清"这个模型没有热力图"和"这次没生成热力图"。** GANomaly 属于前者且永远如此。`prediction_summary` 现在可接收 `ModelCapabilities`（`InferenceSessionManager.capabilities()` 新增，直接读常驻 adapter，不重复构造），标签区分 `tag_heatmap_unsupported`（"仅图像级分数"）与原有的 `tag_heatmap_unavailable`。不传 capabilities 时保持原措辞，不让不知情的调用方替模型下更强的结论。
  3. **`fdh.train("stfpm", config_dir=...)` 会错配。** `config_dir` 落在 `**kwargs` 里只传给了 `run_train`，而解析关键词时用的是默认目录——即"用 A 目录的配置推断后端、用 B 目录的配置训练"。已提为显式参数并两处都传。
  另外把"解析到了哪个配置"从 `train` 扩到 `predict` / `evaluate`：三个 RunResult 都带 `config_path`，CLI 三个子命令都报 `resolved_config`，不再在 CLI 里重复解析。

- [x] **E6 移除冗余代码 + UI 改为只调用契约层**
  按"同一个事实被写了几遍"逐条清，而不是按文件清。查出并修掉：
  1. **`_ANOMALY_MAP_BACKENDS`**：同一个四后端名字集合在 `predict.py` 和 `web/single_image.py` 各存一份，且 `predict.py` 用它回答**三个不相干的问题**（裸图片路径隐含什么 task、config 里哪个键命名模型、要不要传 `output_dir`）——三者对这四个后端答案碰巧一致，不是定义上一致。各自归位到 `ModelCapabilities.supports_task` / `training.BACKEND_MODEL_KEY`（提为 public）/ `ModelCapabilities.fills("anomaly_map")`，两份副本全删。顺带修好一个真错误：GANomaly 是没有像素图的 anomalib 模型，此前按后端名判断会白给它一个输出目录。
  2. **`_BACKEND_PROBE_MODULE`**：UI 自己维护"每个后端探哪个代理 pip 包"，重复实现了 `core.availability.backend_is_importable`，而且问题更弱——代理包在不等于后端能 import。
  3. **`models/anomalib/checkpoint.py` 被 UI 直接 import**，且 `if backend != "anomalib"` 让另外五个后端看到"这是原生 Ultralytics 权重"的文案——对其中四个是错的。该模块本身没有任何 anomalib 特有的东西（哈希 + `torch.serialization` 读取声明的 globals），已移到 `core/checkpoint.py`，UI 对**所有**后端提供诊断（实测 YOLO `.pt` 也能完整读出 SHA/大小/globals）。
  4. **`_input_style_for`**：`backend == "torchvision" and task in (...)` 决定导出模块吃 batched 还是 list 输入——这是 torchvision forward 签名的事实，UI 无从知道。改为契约字段 `ModelCapabilities.export_input_style`（词表校验，六个后端各自声明）。
  5. **`DATASET_CATALOG` 重述契约层**：`tasks` 和 `dir` 在 `core/dataset_capabilities.py` 已有唯一声明。两份今天恰好一致——这是重复事实最危险的状态。UI 目录收敛为纯呈现信息。
  6. **`fdh.predict` 被同名模块遮蔽**：`fabric_defect_hub/predict.py` 一旦被 import，就把包上的 `predict` 属性从门面函数改写成模块——而**第一次** `fdh.predict(...)` 调用自己就会触发这次改写，于是同一进程里第二次调用报 `TypeError: 'module' object is not callable`。模块迁到 `inference/runner.py`（与 `inference/session.py` 同族），并加测试遍历所有子模块后断言 `__all__` 里没有任何名字解析成 module。
  7. 死代码清理：`clear_registries`、`supported_models`、`visa._IMAGE_SUFFIXES`（定义了但调用点内联了自己的元组）、`_ANOMALY_BACKENDS`、`BASE_SCAN_SIZE`、`list_supported_models`（与 `list_supported_variants` 重复）、`_evaluator_for_task`（纯转发包装）、若干未使用 import；`ShotMode` 词表和图片后缀元组各自收敛为单一声明。
  规则由 `tests/test_web_layering.py` 钉死：`web/*.py` 只许 import `models.base`，代码里不许出现后端名字（注释豁免），`DATASET_CATALOG` 只许放呈现字段。已验证这两条规则确实能抓到人为注入的违规。

- [ ] **E3 新增族系的复现验证**（阻塞：与 C4 同因，需 GPU + 完整训练）
  E1 的八个模型目前只验证到"能构造、能端到端跑通"（STFPM/GANomaly 已在 ZJU-Leaper 上跑通 train → predict → evaluate 全链路，8 张图 1 epoch 的冒烟规模）。**它们都还没有 `recipe_id`，这是事实陈述而非待补的坑**——profile 意味着"设置锚定到该方法的论文"，在复现兑现之前不发这个名分（见 `project_recipe_citation_integrity` 的规则：先锚上游 → 再复现 → 才配拥有名字）。
  待云端执行：按各自论文设置跑满，与论文报告的 MVTec 数值对照，达标后再补 profile 与目录条目。

---

## 剩余工作

| 优先级 | 项 | 说明 |
|---|---|---|
| P1 | C4 | MambaAD clean-room 数值对照，阻塞在 GPU/数据集，执行步骤见上方 C4 条目 |
| P2 | E3 | E1 新增八个族系的论文对照复现；在此之前它们不配 `recipe_id`，见上方 E3 条目 |

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
| 门面层只做委派（无后端分支） | `api.py` | `tests/test_api_facade.py` |
