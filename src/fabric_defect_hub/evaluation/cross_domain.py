"""Cross-domain accuracy degradation: how much accuracy drops when a
model trained/calibrated on one dataset (source domain) is evaluated on
another (target domain) without retraining -- the generalization-gap
number this project's per-dataset `Evaluator`s don't produce on their own,
since each only ever sees one dataset per `evaluate()` call. Deliberately
a post-processing step over two already-computed accuracy numbers (or two
per-sample correctness arrays), not a new `Evaluator`, so it works
uniformly across anomaly/detection/industrial without caring which one
produced the underlying accuracy.
"""

from __future__ import annotations

from typing import Sequence

import statistics

from fabric_defect_hub.stats import bootstrap_group_ci


def cross_domain_degradation(acc_src: float, acc_tgt: float) -> float:
    """DeltaAcc = (Acc_src - Acc_tgt) / Acc_src * 100, as a percentage.
    Positive means the target domain scores worse than the source.
    """

    if acc_src == 0:
        raise ValueError("acc_src must be nonzero to compute a relative degradation.")
    return (acc_src - acc_tgt) / acc_src * 100.0


def cross_domain_degradation_ci(
    src_correct: Sequence[bool],
    tgt_correct: Sequence[bool],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Bootstrap CI for `cross_domain_degradation`, computed from each
    domain's per-sample correctness (one bool per prediction: predicted
    class == ground truth, or predicted-anomalous == actually-anomalous).
    Resampling the raw per-sample outcomes -- rather than bootstrapping
    around the two fixed accuracy numbers -- lets sample-count and
    class-imbalance differences between the two domains propagate into
    the interval width, which a CI built from two point accuracies alone
    couldn't reflect.
    """

    def _combine(acc_src: float, acc_tgt: float) -> float:
        return cross_domain_degradation(acc_src, acc_tgt)

    result = bootstrap_group_ci(
        list(src_correct), list(tgt_correct), statistics.fmean, _combine, n_resamples, confidence, seed
    )
    return {"delta_acc_pct": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}
