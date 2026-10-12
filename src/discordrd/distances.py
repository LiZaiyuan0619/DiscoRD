"""The two distances between models (Appendix A, "Distances and weights").

Both functions return an array of shape ``(n_query, n_memory)``: one row per
model whose score is being estimated (a candidate or a validation model) and
one column per past model in the evaluation memory.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def score_distances(memory_scores, query_scores) -> np.ndarray:
    """Euclidean distance between score profiles across the benchmarks of a suite.

    Each benchmark is standardised with the mean and standard deviation of the
    memory.  The profile of a past model holds its full-benchmark scores; the
    profile of a candidate holds its SparseEval-MLP estimates (or, when another
    method supplies the consensus estimate, that method's estimates).

    Parameters
    ----------
    memory_scores : array of shape (n_memory, n_benchmarks) or (n_memory,)
    query_scores : array of shape (n_query, n_benchmarks) or (n_query,)
    """
    pool = np.asarray(memory_scores, np.float64)
    query = np.asarray(query_scores, np.float64)
    if pool.ndim == 1:
        pool = pool[:, None]
    if query.ndim == 1:
        query = query[:, None]
    mean = pool.mean(axis=0)
    scale = pool.std(axis=0)
    scale[scale == 0] = 1.0
    pool_z = (pool - mean) / scale
    query_z = (query - mean) / scale
    squared = (
        np.square(query_z).sum(axis=1)[:, None]
        + np.square(pool_z).sum(axis=1)[None, :]
        - 2.0 * (query_z @ pool_z.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def response_distances(memory_answers, query_answers) -> np.ndarray:
    """Distance between the answers of two models on the anchor questions.

    For binary answers this is the square root of the number of anchor
    questions on which the two models answer differently.  When a suite has
    several benchmarks, the anchor answers of all benchmarks are concatenated
    before the call.  Continuous scores (AlpacaEval) use the same Euclidean
    formula.

    Parameters
    ----------
    memory_answers : array of shape (n_memory, n_anchor_questions)
    query_answers : array of shape (n_query, n_anchor_questions)
    """
    return cdist(np.asarray(query_answers, np.float64),
                 np.asarray(memory_answers, np.float64))
