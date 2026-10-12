"""The 31 metrics of the paper, in seven groups (Appendix C, Table of metrics).

Each metric is computed on the candidates of one cell (one benchmark in one
run) and then averaged over the cells of a setting.  Conventions:

* estimates are clipped to [0, 1] before every metric;
* ties among the estimates are broken uniformly at random, in expectation
  (no metric depends on the order of the rows);
* a metric over the top k is NaN when a cell has fewer than k candidates;
* MAE and RMSE are on the score scale [0, 1] (tables multiply by 100);
  regret@k is in percentage points.

The main text reports MAE, Recall@10, regret@1 and Spearman@20.
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

MAIN_METRICS = ("MAE", "Recall@10", "regret@1", "Spearman@20")
METRIC_GROUPS = {
    "Estimation error": ("MAE", "RMSE"),
    "Whole ranking": ("Spearman", "Kendall"),
    "Order within the top k": tuple(f"{m}@{k}" for m in ("Spearman", "Kendall")
                                    for k in (5, 10, 20, 50)),
    "Top-k recovery": tuple(f"Recall@{k}" for k in (1, 3, 5, 10, 20, 50)),
    "Position-weighted ranking": tuple(f"NDCG@{k}" for k in (5, 10, 20)),
    "Finding the best model": tuple(f"Hit@{k}" for k in (1, 3, 5, 10)) + ("MRR",),
    "Cost of the pick": tuple(f"regret@{k}" for k in (1, 3, 5, 10, 20)),
}
ALL_METRICS = tuple(m for group in METRIC_GROUPS.values() for m in group)
HIGHER_IS_BETTER = {m for m in ALL_METRICS if not m.startswith(("MAE", "RMSE", "regret"))}


def expected_recall_at(pred, truth, k: int) -> float:
    """Recall@k under uniform random tie-breaking at the k-th estimate."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    if len(t) < k:
        return float("nan")
    target = set(np.argsort(-t, kind="stable")[:k].tolist())
    thr = np.partition(p, -k)[-k]
    above = np.flatnonzero(p > thr)
    at = np.flatnonzero(p == thr)
    take = k - len(above)
    hit = len(target & set(above.tolist()))
    if take > 0 and len(at):
        hit += take / len(at) * len(target & set(at.tolist()))
    return float(hit / k)


def expected_regret_at(pred, truth, k: int) -> float:
    """regret@k in points: true score of the champion minus the best true score
    among the k highest estimates, under uniform random tie-breaking."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    n = len(t)
    if k > n:
        return float("nan")
    if k == n:
        return 0.0
    thr = np.partition(p, -k)[-k]
    above = np.flatnonzero(p > thr)
    tied = np.flatnonzero(p == thr)
    take = k - len(above)
    best_above = float(t[above].max()) if len(above) else -np.inf
    if take <= 0:
        return float(t.max() - best_above) * 100.0
    if take >= len(tied):
        return float(t.max() - max(best_above, float(t[tied].max()))) * 100.0
    # The best of the `take` sampled tied models is the j-th largest truth with
    # probability C(m-j, take-1) / C(m, take); average max(best_above, that).
    tied_truth = np.sort(t[tied])[::-1]
    total = 0.0
    for j in range(1, len(tied_truth) - take + 2):
        weight = comb(len(tied_truth) - j, take - 1) / comb(len(tied_truth), take)
        total += weight * max(best_above, float(tied_truth[j - 1]))
    return float(t.max() - total) * 100.0


def expected_regret(pred, truth) -> float:
    """regret@1 in points, averaged over the models tied at the top estimate."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    return float(t.max() - t[p == p.max()].mean()) * 100.0


def desc_rank(pred) -> np.ndarray:
    """Rank in descending order of the estimates (1 = highest); ties get their average position."""
    ascending = rankdata(np.asarray(pred, float), method="average")
    return len(ascending) + 1.0 - ascending


def mrr(pred, truth) -> float:
    """Reciprocal of the champion's estimated rank (averaged over tied champions)."""
    t = np.asarray(truth, float)
    champions = np.flatnonzero(t == t.max())
    return float(np.mean(1.0 / desc_rank(pred)[champions]))


def hit_at(pred, truth, k: int) -> float:
    """Whether the champion is among the k highest estimates."""
    t = np.asarray(truth, float)
    if len(t) < k:
        return float("nan")
    champions = np.flatnonzero(t == t.max())
    return float(np.mean((desc_rank(pred)[champions] <= k).astype(float)))


def ndcg_at(pred, truth, k: int) -> float:
    """NDCG@k with the true scores rescaled to [0, 1] within the cell as relevance."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    if len(t) < k:
        return float("nan")
    low, high = float(t.min()), float(t.max())
    if high <= low:
        return float("nan")
    relevance = (t - low) / (high - low)
    gains = 2.0 ** relevance - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    order = np.argsort(desc_rank(p), kind="stable")
    dcg = float(np.sum(gains[order[:k]] * discounts))
    ideal = float(np.sum(np.sort(gains)[::-1][:k] * discounts))
    if ideal <= 0:
        return float("nan")
    return dcg / ideal


def evaluate(pred, truth, *, clip: bool = True) -> dict[str, float]:
    """All 31 metrics of one cell.

    Parameters
    ----------
    pred : estimated scores of the candidates
    truth : true full-benchmark scores of the same candidates
    clip : clip the estimates to [0, 1] first (as everywhere in the paper)
    """
    p = np.clip(np.asarray(pred, float), 0.0, 1.0) if clip else np.asarray(pred, float)
    t = np.asarray(truth, float)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {t.shape}")
    order_t = np.argsort(-t, kind="stable")
    out = {
        "MAE": float(np.mean(np.abs(p - t))),
        "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
        "Spearman": float(spearmanr(t, p).statistic),
        "Kendall": float(kendalltau(t, p).statistic),
    }
    # Order within the top k: the top k by TRUE score.
    for name, stat in (("Spearman", spearmanr), ("Kendall", kendalltau)):
        for k in (5, 10, 20, 50):
            top = order_t[:k]
            out[f"{name}@{k}"] = (float(stat(t[top], p[top]).statistic)
                                  if len(top) >= 2 and len(t) >= k else float("nan"))
    for k in (1, 3, 5, 10, 20, 50):
        out[f"Recall@{k}"] = expected_recall_at(p, t, k)
    for k in (5, 10, 20):
        out[f"NDCG@{k}"] = ndcg_at(p, t, k)
    for k in (1, 3, 5, 10):
        out[f"Hit@{k}"] = hit_at(p, t, k)
    out["MRR"] = mrr(p, t)
    out["regret@1"] = expected_regret(p, t)
    for k in (3, 5, 10, 20):
        out[f"regret@{k}"] = expected_regret_at(p, t, k)
    return {name: out[name] for name in ALL_METRICS}
