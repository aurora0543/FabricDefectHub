# FabricDefectHub

An open-source benchmarking platform for fabric defect detection, designed for real industry evaluation.

---

## 📢 Project Status: Under Development 🚀

> **Note:** This project is currently in its active initial planning and development phase. The core architecture, unified interfaces, and data pipelines are being actively built by the research team at Beijing Institute of Technology (BIT). Features and APIs are subject to change.

---

## 📌 Overview

**FabricDefectHub** aims to be a unified, fair, and comprehensive benchmarking platform for automated textile surface inspection. In real-world manufacturing, defect detection methods vary drastically—ranging from standard supervised object detection to unsupervised anomaly localization approaches. 

This platform bridges these architectural divides by providing a centralized top-level abstraction layer over three state-of-the-art computer vision ecosystems, allowing industrial engineers and researchers to benchmark models under identical hardware constraints and standardized data pipelines.

---

## ⚙️ Core Architecture & Backends

FabricDefectHub integrates three distinct deep learning frameworks into a unified execution engine, catering to different industrial production scenarios:

1. **YOLO Backend (`ultralytics`)**
   * Targeted for high-throughput, edge-device deployments requiring single-stage real-time object detection (e.g., YOLOv8, YOLOv10, YOLOv11).
2. **OpenMMLab Backend (`mmdetection`)**
   * Reserved for comprehensive supervised learning evaluations, covering classic two-stage detectors, anchor-free models, and modern Transformer-based architectures (e.g., Faster R-CNN, Cascade R-CNN, DETR, DINO).
3. **Anomaly Lab Backend (`anomalib`)**
   * Dedicated to unsupervised and one-class anomaly detection methods, optimized for scenarios where negative (defective) training samples are extremely scarce.

---

## 🗺️ Repository Directory Structure

The project is organized into distinct physical folders separating configurations, raw data processing, core logic, and deployment utilities:

*   **`configs/`** — Houses all top-level YAML configuration files for managing global environments, datasets, and specific model training/evaluation hyperparameters.
*   **`data/`** — Local storage for datasets (Git-ignored). Includes separate folders for raw downloads (e.g., AITEX, TILDA) and standardized, processed formats (COCO and MVTec structures).
*   **`src/`** — The core source code directory. It contains the unified execution entry point, data format converters, and the abstract engine wrappers that manage the three underlying frameworks.
*   **`tools/`** — Auxiliary scripts for automation, data visualization, and model export (e.g., converting checkpoints to ONNX or TensorRT).
*   **`tests/`** — Unit and integration tests to ensure framework stability and metric calculation accuracy.

---

## 📊 Industrial Evaluation Metrics

Unlike purely academic benchmarks that solely focus on mAP, FabricDefectHub emphasizes **"Real Industry Evaluation"** by reporting a holistic metric board directly measured at the unified runtime:

*   **Accuracy Metrics:** mAP@0.5, mAP@0.5:0.95, Dice Score, F1-Score, and Under-kill/Over-kill rates.
*   **Efficiency Metrics:** Inference Latency (ms), Frames Per Second (FPS), and FLOPs.
*   **Hardware Profiles:** Peak RAM utilization, VRAM footprint, and model parameter scale across target devices.

---

## 🛠️ Upcoming Roadmap

- [ ] Core Abstract Engine implementation.
- [ ] Automated dataset alignment scripts (Raw data $\rightarrow$ unified COCO & MVTec structures).
- [ ] Integration of the `ultralytics` YOLO inference pipeline.
- [ ] Integration of the `mmdetection` model zoo.
- [ ] Integration of the `anomalib` unsupervised pipelines.
- [ ] Standardized logging and export module (supporting TensorBoard, WandB, and Excel exports).

---

## 📄 License

This project is licensed under the **Apache License 2.0**. It provides a robust, legally secure framework that allows industrial commercialization and research experimentation while protecting intellectual property and patent rights. 

*(Please note: Third-party datasets benchmarks hosted or referenced by this repository are subject to their respective original licenses).*

---

## 👥 Contact & Affiliation

Developed with pride by the Joint Research Team at **Beijing Institute of Technology (BIT)**. For inquiries regarding collaboration, corporate benchmark submissions, or academic extensions, please open an Issue or contact the repository maintainers.
