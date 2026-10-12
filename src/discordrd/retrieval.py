"""Consensus and discordance retrieval (Section 3 and Appendix A).

Notation follows the paper.  For a model c, g_c is its anchor mean (its mean
answer on the anchor questions) and N_k(c) its k nearest past models, with
inverse-distance weights w_j(c) that sum to one.  For a past model j, y_j is
its full-benchmark score and delta_j = y_j - g_j its local correction.

    consensus     y_con(c) = sum_j w_j(c) * y_j
    discordance   y_dis(c) = g_c + sum_j w_j(c) * delta_j = y_con(c) + Delta_c

with the discordance Delta_c = g_c - sum_j w_j(c) * g_j.  Consensus retrieval
discards Delta_c; discordance retrieval keeps it.  Each retrieval predictor
chooses k from K_GRID by its mean absolute error on the validation set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_GRID = (1, 5, 10, 25, 50)


def knn_predict(distances, memory_values, k: int) -> np.ndarray:
    """Inverse-distance weighted mean of ``memory_values`` over the k nearest past models.

    ``distances`` has shape (n_query, n_memory).  The weights are proportional
    to 1 / distance (distances below 1e-8 are treated as 1e-8) and normalised
    to sum to one.
    """
    distances = np.asarray(distances, np.float64)
    y = np.asarray(memory_values, np.float64)
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(distances, nearest, axis=1)
    weights = 1.0 / np.maximum(selected, 1e-8)
    values = y[nearest]
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


@dataclass(frozen=True)
class Retrieval:
    """Predictions of one retrieval predictor with its chosen k."""

    valid: np.ndarray
    query: np.ndarray
    k: int
    valid_mae: float


def _choose_k(valid_distances, query_distances, memory_values, valid_truth, *,
              valid_offset=None, query_offset=None, k_grid=K_GRID) -> Retrieval:
    best = None
    n_memory = np.asarray(valid_distances).shape[1]
    for k in k_grid:
        if k > n_memory:
            continue
        valid = knn_predict(valid_distances, memory_values, k)
        query = knn_predict(query_distances, memory_values, k)
        if valid_offset is not None:
            valid = valid_offset + valid
            query = query_offset + query
        loss = float(np.mean(np.abs(valid - valid_truth)))
        if best is None or loss < best.valid_mae:
            best = Retrieval(valid=valid, query=query, k=int(k), valid_mae=loss)
    if best is None:
        raise ValueError(f"no k in {k_grid} fits a memory of {n_memory} past models")
    return best


def consensus_retrieval(valid_distances, query_distances, memory_scores,
                        valid_truth, *, k_grid=K_GRID) -> Retrieval:
    """Weighted mean of the neighbours' full-benchmark scores.

    Parameters
    ----------
    valid_distances, query_distances : arrays of shape (n_valid, n_memory), (n_query, n_memory)
    memory_scores : full-benchmark scores of the past models, shape (n_memory,)
    valid_truth : full-benchmark scores of the validation models (used only to choose k)
    """
    return _choose_k(valid_distances, query_distances,
                     np.asarray(memory_scores, float), np.asarray(valid_truth, float),
                     k_grid=k_grid)


def discordance_retrieval(valid_distances, query_distances, memory_scores,
                          memory_anchor_means, valid_anchor_means, query_anchor_means,
                          valid_truth, *, k_grid=K_GRID) -> Retrieval:
    """The model's own anchor mean plus the neighbours' weighted local corrections.

    The local correction delta_j = y_j - g_j of a past model is how far its
    full score lies from its anchor mean.
    """
    local_corrections = np.asarray(memory_scores, float) - np.asarray(memory_anchor_means, float)
    return _choose_k(valid_distances, query_distances, local_corrections,
                     np.asarray(valid_truth, float),
                     valid_offset=np.asarray(valid_anchor_means, float),
                     query_offset=np.asarray(query_anchor_means, float),
                     k_grid=k_grid)
