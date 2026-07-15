# FabricDefectHub — Outstanding Work

Generated from a two-agent read-only audit of `src/fabric_defect_hub/` (frontend excluded — that's tracked separately). Each item below includes enough context for an agent picking it up cold to act without re-deriving the finding. Priority: **P0** = crashes/produces silently wrong results, **P1** = real functional gap, **P2** = polish/consistency.

Verification note: framework-free paths are compiled and smoke-tested locally. Real framework, CUDA, Jetson, Raspberry Pi, and privileged macOS power checks are represented by opt-in validation code and remain environment validation rather than code-completion blockers.

---

## Environment validation — currently unavailable, not counted as code completion

- [ ] Run the prepared validation suite on a real NVIDIA cloud GPU, Jetson, privileged macOS `powermetrics`, and Raspberry Pi with INA219/INA226; record source/scope-specific baselines.
- [ ] Confirm the new torchvision `.pt2` export against the installed cloud PyTorch/torchvision operator set; retain TorchScript only where `torch.export` reports unsupported detection operators.

---

## Model Integration Roadmap (Supervised Defect Detection)

We aim to expand the project with support for a wider variety of supervised algorithms (One-Stage, Multi-Stage, Transformer, and Semantic Segmentation).

### 1. Object Detection (定位框级别检测)
- [x] **完整的 Faster R-CNN / 骨干网络拓展**：支持在 `presets.py` 中通过自定义 Backbone/FPN 动态组装各种骨干网络（如 VGG-16、ShuffleNet V2）的 Faster R-CNN 变体。
- [x] **级联 R-CNN (Cascade R-CNN)**：实现或集成多级回归检测头，提高缺陷检测的边界框定位精度。
- [x] **DETR 系列 (Transformer-based Detection)**：集成基于注意力机制的端到端目标检测模型（如标准 DETR、Deformable DETR 或实时的 RT-DETR）。

### 2. Semantic Segmentation (像素级缺陷分割)
- [x] **U-Net++**：实现高分辨率密集跳跃连接的语义分割网络，精准提取微小细长裂缝等缺陷的像素面积与轮廓。
- [x] **DeepLab V3+**：集成基于空洞卷积与多尺度特征提取的语义分割算法，适配尺寸跨度巨大的复杂缺陷。
