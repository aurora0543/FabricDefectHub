"""Cross-domain accuracy degradation metrics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Sequence

import statistics

from fabric_defect_hub.stats import bootstrap_group_ci


def cross_domain_degradation(acc_src: float, acc_tgt: float) -> float:
    """Compute relative accuracy degradation percentage: (acc_src - acc_tgt) / acc_src * 100."""

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
    """Compute bootstrap confidence interval for cross-domain accuracy degradation."""

    def _combine(acc_src: float, acc_tgt: float) -> float:
        return cross_domain_degradation(acc_src, acc_tgt)

    result = bootstrap_group_ci(
        list(src_correct), list(tgt_correct), statistics.fmean, _combine, n_resamples, confidence, seed
    )
    return {"delta_acc_pct": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}
