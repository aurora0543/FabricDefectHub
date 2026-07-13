# frontend

数据集、模型、实验配置、结果排行榜和实验详情的可视化前端。第一阶段使用
`src/mock/` 下的 mock 数据实现可点击原型，字段需与 `schemas/` 中的
`Sample` / `Prediction` / `ExperimentResult` 契约保持一致，避免后续接入
真实后端时重写页面。

具体技术栈（React/Vue + 构建工具）尚未选定，将在 Phase 0 结束前确定并在
此补充启动方式。

## 计划中的目录结构

```text
frontend/
├── src/
│   ├── pages/
│   │   ├── datasets/           # 数据集页
│   │   ├── models/             # 模型页
│   │   ├── experiments/        # 实验配置页
│   │   ├── results/            # 结果排行榜页
│   │   └── experiment-detail/  # 实验详情页
│   └── mock/                   # 与 schemas/ 契约一致的 mock 数据
└── README.md
```
