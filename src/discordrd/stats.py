"""Paired tests over the cells of a setting (Appendix C, "Runs and tests").

Paired comparisons use the two-sided Wilcoxon signed-rank test over the cells
of a setting, with Holm correction within each metric and setting.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a, b) -> tuple[float, float]:
    """(median paired difference a - b, two-sided p).  p is NaN when all differences are zero."""
    d = np.asarray(a, float) - np.asarray(b, float)
    med = float(np.median(d))
    if np.allclose(d, 0):
        return med, float("nan")
    try:
        return med, float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return med, float("nan")


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values, keyed as the input (NaN stays NaN)."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    n = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (n - i) * p)
        adj[k] = float(min(1.0, running))
    for k in pvals:
        adj.setdefault(k, float("nan"))
    return adj


def relative_gain(method_cells, reference_cells) -> float:
    """100 * (1 - mean(method) / mean(reference)) over paired cells, e.g. for MAE."""
    m = np.asarray(method_cells, float)
    r = np.asarray(reference_cells, float)
    return float(100 * (1 - m.mean() / r.mean()))
