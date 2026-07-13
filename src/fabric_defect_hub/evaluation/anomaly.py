"""Anomaly `Evaluator`: image-level AUROC/F1/precision/recall (at an
F1-optimal threshold, not a hardcoded 0.5), plus pixel-level AUROC/AUPRO
when `Prediction.anomaly_map` files are available (see
`AnomalibAdapter.predict(..., output_dir=...)`).

Uses `scikit-learn` for the standard curve/threshold math and
`scikit-image` for AUPRO's per-region connected-component labeling —
correct, maintained implementations, in keeping with the project's
"don't reimplement what a library already gets right" principle (see
`models/anomalib/presets.py`). What's ours to own is wiring `Sample` +
`Prediction` into those computations, and the memory-safety pixel
subsampling below (a full-resolution ZJU-Leaper-scale test set can easily
exceed available RAM if every pixel is compared).
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.evaluation.base import Evaluator

# Cap on pixels fed to pixel AUROC/F1 (subsampled uniformly at random) and
# on images fed to AUPRO (subsampled whole, so connected components stay
# intact) — keeps evaluation memory bounded regardless of test-set size.
DEFAULT_MAX_PIXELS = 1_000_000
DEFAULT_MAX_AUPRO_IMAGES = 50


class AnomalyEvaluator(Evaluator):
    """Image-level (always) + pixel-level (when maps are present) anomaly metrics."""

    task = "anomaly"

    def __init__(
        self,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_aupro_images: int = DEFAULT_MAX_AUPRO_IMAGES,
        seed: int = 0,
    ):
        self.max_pixels = max_pixels
        self.max_aupro_images = max_aupro_images
        self.seed = seed

    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        import numpy as np

        pred_by_id = {p.sample_id: p for p in predictions}

        y_true: list[int] = []
        y_score: list[float] = []
        pixel_pairs: list[tuple[Any, Any]] = []  # (gt_mask_2d, pred_map_2d) per sample

        for sample in samples:
            pred = pred_by_id.get(sample.id)
            if pred is None or pred.anomaly_score is None:
                continue
            y_true.append(1 if sample.annotations.is_anomalous else 0)
            y_score.append(pred.anomaly_score)

            if pred.anomaly_map is not None:
                pred_map = np.load(pred.anomaly_map)
                gt_mask = _load_ground_truth_mask(sample, pred_map.shape)
                pixel_pairs.append((gt_mask, pred_map))

        if not y_true:
            return {}

        metrics = _image_level_metrics(np.asarray(y_true), np.asarray(y_score, dtype=float))

        if pixel_pairs:
            metrics.update(
                _pixel_level_metrics(pixel_pairs, self.max_pixels, self.max_aupro_images, self.seed)
            )

        return metrics


def _load_ground_truth_mask(sample: Sample, target_shape: tuple[int, ...]):
    """Binary pixel ground truth for `sample`, resized to `target_shape`
    (the predicted anomaly map's resolution, which generally differs from
    the raw image/mask resolution since models resize internally).
    """

    import numpy as np

    mask_path = sample.annotations.anomaly_mask
    if mask_path is None:
        return np.zeros(target_shape, dtype=np.uint8)

    from PIL import Image

    with Image.open(mask_path) as img:
        resized = img.convert("L").resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return (np.asarray(resized) > 0).astype(np.uint8)


def _best_f1_threshold(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if len(set(y_true.tolist())) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(f1.argmax())])


def _image_level_metrics(y_true, y_score) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics: dict[str, float] = {}
    metrics["image_auroc"] = (
        float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) >= 2 else float("nan")
    )

    threshold = _best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    metrics["image_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["image_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["image_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["image_threshold"] = threshold
    return metrics


def _pixel_level_metrics(
    pixel_pairs: list, max_pixels: int, max_aupro_images: int, seed: int
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    flat_true = np.concatenate([m.reshape(-1) for m, _ in pixel_pairs])
    flat_score = np.concatenate([s.reshape(-1) for _, s in pixel_pairs])

    rng = np.random.default_rng(seed)
    if len(flat_true) > max_pixels:
        idx = rng.choice(len(flat_true), max_pixels, replace=False)
        flat_true, flat_score = flat_true[idx], flat_score[idx]

    metrics: dict[str, float] = {}
    metrics["pixel_auroc"] = (
        float(roc_auc_score(flat_true, flat_score)) if len(set(flat_true.tolist())) >= 2 else float("nan")
    )
    px_threshold = _best_f1_threshold(flat_true, flat_score)
    metrics["pixel_f1"] = float(
        f1_score(flat_true, (flat_score >= px_threshold).astype(int), zero_division=0)
    )
    metrics["pixel_aupro"] = _compute_aupro(pixel_pairs, max_aupro_images, seed)
    return metrics


def _compute_aupro(pixel_pairs: list, max_images: int, seed: int, num_thresholds: int = 100) -> float:
    """Area under the per-region overlap (PRO) curve, integrated over FPR.

    Each ground-truth defect region (connected component of the mask)
    contributes its own recall at each threshold; PRO is the mean recall
    across regions, plotted against the false-positive rate on normal
    pixels — this rewards detecting every defect region at least partially,
    rather than letting one large region dominate a plain pixel AUROC.
    """

    import numpy as np
    from skimage.measure import label

    if len(pixel_pairs) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pixel_pairs), max_images, replace=False)
        pixel_pairs = [pixel_pairs[i] for i in idx]

    region_preds = []
    neg_chunks = []
    for mask, score_map in pixel_pairs:
        labeled, num_regions = label(mask.astype(np.uint8), return_num=True)
        for region_id in range(1, num_regions + 1):
            region_preds.append(score_map[labeled == region_id])
        neg_chunks.append(score_map[mask == 0])

    if not region_preds:
        return float("nan")

    neg_preds = np.concatenate(neg_chunks) if neg_chunks else np.array([])
    if neg_preds.size == 0:
        return float("nan")

    all_scores = np.concatenate([*region_preds, neg_preds])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    pro_scores, fprs = [], []
    for th in thresholds:
        pro_scores.append(float(np.mean([(preds >= th).sum() / preds.size for preds in region_preds])))
        fprs.append(float((neg_preds >= th).sum() / neg_preds.size))

    order = np.argsort(fprs)
    fprs_sorted = np.asarray(fprs)[order]
    pro_sorted = np.asarray(pro_scores)[order]
    return float(np.trapezoid(pro_sorted, fprs_sorted))
