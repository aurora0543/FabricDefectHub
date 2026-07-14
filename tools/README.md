# tools

数据转换、导出和可视化脚本（非包代码，独立运行）：

- `convert_annotations.py` — COCO detection 标注 -> `Sample` JSON。
- `export_model.py` — 调用 `ModelAdapter.export`，也可将 ONNX 构建为 TensorRT engine。
- `visualize_predictions.py` — 在图像上绘制 `Prediction`（框/掩码/异常热力图）。

三个脚本都可通过 `python tools/<script>.py --help` 查看参数。
