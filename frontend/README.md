# frontend

FabricDefectHub 使用 Gradio 提供可部署的模型工作区，而不是另建一套与后端
脱节的 mock 前端。页面直接消费统一的 `Sample`、`Prediction` 和
`ExperimentResult` 契约。

## 启动

```bash
pip install -e ".[ui]"
fdh-ui
```

Hugging Face Spaces 可以直接使用仓库根目录的 `app.py`。设置
`ZJU_LEAPER_ROOT` 后，Single Image Detection 页面会从 ZJU-Leaper 的选定
切分中随机抽取一组本地图片；使用左右按钮浏览，并按模型/权重配置执行真实推理。

## 推理会话

单图页不会在每次点击推理时重新创建模型。先在本地模型下拉框选择制品，再点击
**Load model**；该动作通过后端 `InferenceSessionManager` 将模型驻留到自动选择的
CUDA、Apple MPS 或 CPU 设备。页面显示模型参数/缓冲区占用、当前进程 RSS，以及
CUDA/MPS 分配内存（平台支持时）。**Unload model** 会释放活动模型和加速器缓存。

UI 仅调用后端的 `load`、`predict`、`unload` 接口，因此同一个会话机制可被 Gradio、
CLI、服务端 API 或其它平台 UI 复用；UI 不直接持有框架模型对象。

## 当前页面

- **Single Image Detection**：数据集随机抽样、图片浏览、模型状态、checkpoint/
  pretrained 选择、bbox/mask/anomaly-map 结果展示。
- **Benchmark**：保留独立占位，后续复用后端保存的 leaderboard 与
  `ExperimentResult` 制品实现。
