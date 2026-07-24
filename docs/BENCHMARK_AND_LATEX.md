# Multi-Granularity Metrics (MGM) & LaTeX Generator Guide

This guide documents the multi-granularity evaluation metrics (PRO-Score, LMEI Index) and automated paper-grade LaTeX table rendering.

---

## 1. Multi-Granularity Metric Suite (MGM)

### Per-Region Overlap Metric (`PRO-Score`)
Unlike global pixel-level AUROC, `PRO-Score` evaluates connected defect components independently, preventing large defect regions from masking missed microscopic defects:

```python
from fabric_defect_hub import compute_pro_score

# Compute region-wise overlap AUC score up to FPR = 0.3
pro_auc = compute_pro_score(masks_gt=gt_masks, anomaly_maps=pred_maps, max_fpr=0.3)
```

### Latency-Memory Efficiency Index (`LMEI`)
Evaluates real-time edge hardware trade-off across throughput (FPS), peak VRAM, and FLOPs:

$$\text{LMEI} = \frac{\text{FPS} / 100}{\log_{10}(\text{FLOPs}_{\text{G}} + 1) \cdot \log_{10}(\text{VRAM}_{\text{MB}} / 100 + 1) + 1e-5}$$

```python
from fabric_defect_hub import calculate_lmei

lmei_score = calculate_lmei(fps=145.0, vram_mb=512.0, flops_g=8.5, params_m=3.2)
```

---

## 2. Automated Paper-Grade LaTeX Table Generator

Generates IEEE / CVPR / ICCV formatted LaTeX code directly from benchmark result JSON files, automatically bolding (`\textbf{}`) the best score and underlining (`\underline{}`) the second-best score.

### CLI Command
```bash
fdh export-latex results/benchmark.json --output paper_table.tex
```

### Python SDK Usage
```python
import fabric_defect_hub as fdh

results = [
    {"model": "YOLOv8n", "recipe": "yolov8", "I-AUROC": 0.9820, "PRO-Score": 0.9410, "FPS": 145.0, "LMEI": 1.25},
    {"model": "PatchCore", "recipe": "patchcore", "I-AUROC": 0.9650, "PRO-Score": 0.9120, "FPS": 85.0, "LMEI": 0.88},
]

latex_code = fdh.generate_latex_table(
    results,
    metrics=["I-AUROC", "PRO-Score", "FPS", "LMEI"],
    caption="Quantitative Evaluation of Fabric Defect Inspection Models under UTAD-Framework.",
    label="tab:fabric_benchmark",
)

print(latex_code)
```

### Rendered LaTeX Output
```latex
\begin{table*}[t]
\centering
\caption{Quantitative Evaluation of Fabric Defect Inspection Models under UTAD-Framework.}
\label{tab:fabric_benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c c c}
\toprule
Model & Recipe / Acronym & I-AUROC & PRO-Score & FPS & LMEI \\
\midrule
YOLOv8n & yolov8 & \textbf{0.9820} & \textbf{0.9410} & \textbf{145.0000} & \textbf{1.2500} \\
PatchCore & patchcore & \underline{0.9650} & \underline{0.9120} & 85.0000 & 0.8800 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```
