# 交付状态（一页版）

> 用途：让评审者在五分钟内知道**哪些是做完并验证过的、哪些是明确未做的**，不用去翻代码猜。
> 最后更新：2026-07-29。数字全部现场跑出来，不是手抄。

## 一句话

代码、契约、接口、文档、测试都已完成并可运行；**尚未完成的只有一类：需要 GPU 的论文数值复现**（`INTERFACE_TASKS.md` 的 C4 / E3）。仓库不含权重和数据集——两者都由使用者在自己机器上产出/挂载，见 README 的"新克隆会看到什么"。

## 规模

| | 数量 | 备注 |
|---|---|---|
| 模型后端 | 6 | ultralytics / torchvision / anomalib / dinomaly / moeclip / mambaad |
| 可运行模型变体 | 38 | `fdh models` |
| 已进目录（前端可选、可发布） | 20 | `catalog.CANONICAL_MODELS` |
| 数据集 | 9 | `fdh list` |
| 配置档案（`recipe_id`） | 6 | 只覆盖已复现过的方法，见下 |
| 评测器 / profiler | 4 / 3 | 按 task、按 engine 注册 |
| 测试 | 1007 收集 / 997 通过 / 10 跳过 | `pytest tests/`；跳过的是需要未安装可选依赖的用例 |
| 源文件 / 测试文件 | 127 / 71 | |

## 已完成且验证过

**契约层（已冻结，`interface-freeze-v1.0`）**
五个抽象 + 数据契约 + 复现契约，见 `docs/INTERFACE_SPEC.md`。每一条都有守护测试，改动需先改测试。

**门面层（`api.py`）**
`fdh.load_config / train / predict / evaluate / from_pretrained / list_*`，从包根导出。AST 级测试钉死"只做委派、不含后端分支、模块级不 import 深度学习框架"。

**UI 层（`web/`）**
只 import `models.base`，代码里零个后端名字；数据集能力、权重诊断、后端可用性全部读契约层。规则由 `tests/test_web_layering.py` 钉死，并验证过它能抓到人为注入的违规。

**端到端真跑过（ZJU-Leaper，本机 MPS）**
- `fdh train stfpm` / `ganomaly` / `padim` / `rd4ad` / `fastflow` → 训练→评测全链路通
- `fdh predict` / `fdh evaluate` → 含像素级指标（给 `--output-dir` 时）
- Gradio 前端 → `create_app()` 构建成功，`launch()` 起服务返回 HTTP 200

**可复现性**
每次训练/评测都往两个 append-only 账本写同一个 provenance 块：仓库 commit、主机名、每个 vendored 上游 checkout 的 pinned commit 及是否被改动、实际使用的 optimizer/scheduler/精度（从 live 对象读，不是手抄）。

## 明确未完成（不是遗漏，是阻塞）

| 项 | 内容 | 阻塞原因 |
|---|---|---|
| **C4** | MambaAD clean-room 实现 vs 论文 Table 1 的 MVTec image-AUROC 对照（论文报 97.8，差 >1 点算未对齐） | 需 GPU + MVTec AD |
| **E3** | E1 新增的八个 anomalib 族系（STFPM / GANomaly / DRAEM / DSR / GLASS / FastFlow / UniNet / AnomalyDINO）按各自论文设置复现 | 同上 |

两者的执行步骤已写成可直接照做的命令，见 `docs/cloud_training_runbook.md` §10。

**这两项未完成的直接后果**：这八个模型**没有 `recipe_id`**，跑的是各自上游的构造器默认值。这是事实陈述，不是待补的坑——在这个项目里，profile 意味着"设置锚定到那篇论文"，复现兑现之前不发这个名分（规则见 `docs/MODEL_CONFIGURATION.md`）。

## 已知的诚实边界

- **仓库不含权重**：`/artifacts/` gitignored。前端在新克隆上会把 20 个模型全部显示为"权重缺失"，这是预期行为。
- **仓库不含数据集**：`/data/*` gitignored，按 `data/<Dataset>` 符号链接约定挂载。
- **`recipe_id` 只有 6 个**：torchvision 的检测/分割器按上游默认值跑作为 baseline（故意不带 profile）；上面 E3 的八个待复现。
- **没有 CI**：测试本地跑（`~/miniconda3/envs/anomalib_env/bin/python -m pytest`）。跑全套需要完整训练栈（torch + anomalib + ultralytics，数 GB），未配置 GitHub Actions 是权衡后的选择，不是疏忽。

## 怎么验收

```bash
pip install -r requirements-full.txt
pytest tests/ -q          # 本机：997 passed, 10 skipped（缺可选后端时跳过更多）
fdh doctor                # 这台机器上什么能跑、为什么
fdh models                # 六个后端各自支持哪些模型
fdh-ui                    # 前端
```
