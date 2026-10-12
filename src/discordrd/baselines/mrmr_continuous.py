"""mRMR (MIQ) question selection for continuous scores (used on AlpacaEval).

Adapted from the released mRMR benchmark-prediction code (MIT licence,
Copyright (c) 2024 Guanhua Zhang and Moritz Hardt; see THIRD_PARTY.md).  For
continuous scores, relevance and redundancy are both estimated with the
PCA-corrected KSG estimator of Gao, Ver Steeg and Galstyan (2015); the
relevance target is each model's mean score, and the MIQ criterion divides
relevance by mean redundancy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import psi


def mutual_information_lnc(x, y, k: int = 5, alpha: float = 0.25) -> float:
    """MI between two continuous variables: KSG with local nonuniformity correction."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < k + 1:
        return 0.0
    intens = 1e-10
    x = x + intens * np.random.default_rng(0).standard_normal(n)
    y = y + intens * np.random.default_rng(1).standard_normal(n)
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    dd, ii = tree.query(xy, k=k + 1, p=np.inf)
    eps = dd[:, -1]
    all_nbr = xy[ii]
    centred = all_nbr - all_nbr[:, 0:1, :]
    dvec_x = np.maximum(np.max(np.abs(centred[:, :, 0]), axis=1), 1e-15)
    dvec_y = np.maximum(np.max(np.abs(centred[:, :, 1]), axis=1), 1e-15)
    r = np.maximum(eps - 1e-15, 0.0)
    nx = cKDTree(x.reshape(-1, 1)).query_ball_point(x.reshape(-1, 1), r, p=np.inf, return_length=True) - 1
    ny = cKDTree(y.reshape(-1, 1)).query_ball_point(y.reshape(-1, 1), r, p=np.inf, return_length=True) - 1
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    mi_ksg = float(psi(k) + psi(n) - np.mean(psi(nx + 1) + psi(ny + 1)))
    nbr_only = centred[:, 1:, :]
    cov = np.einsum("nki,nkj->nij", nbr_only, nbr_only) / k
    _, eigvecs = np.linalg.eigh(cov)
    projected = np.einsum("nki,nij->nkj", centred, eigvecs)
    max_proj = np.maximum(np.max(np.abs(projected), axis=1), 1e-30)
    log_v_pca = np.sum(np.log(max_proj), axis=1)
    log_v_rect = np.log(dvec_x) + np.log(dvec_y)
    lnc_mask = log_v_pca < log_v_rect + np.log(max(alpha, 1e-30))
    correction = np.sum(log_v_rect[lnc_mask] - log_v_pca[lnc_mask]) / n
    return max(0.0, mi_ksg + correction)


def mrmr_miq_order(scores, coreset_size: int, *, k: int = 5, seed: int = 42) -> np.ndarray:
    """Greedy mRMR-MIQ order of ``coreset_size`` questions from a models x questions matrix."""
    scores = np.asarray(scores, float)
    num_model, num_data = scores.shape
    if num_model <= 1 or num_data <= coreset_size or coreset_size <= 1:
        raise ValueError(f"invalid mRMR input {scores.shape} for coreset_size={coreset_size}")
    np.random.seed(seed)
    target = scores.mean(-1)
    relevance = np.array([mutual_information_lnc(scores[:, j], target, k) for j in range(num_data)])
    selected = [int(np.argmax(relevance))]
    remaining = set(range(num_data)) - set(selected)
    redundancy_sum = np.zeros(num_data)

    def add_redundancy(new):
        rem = np.array(sorted(remaining))
        if len(rem):
            redundancy_sum[rem] += [mutual_information_lnc(scores[:, j], scores[:, new], k) for j in rem]

    add_redundancy(selected[0])
    while len(selected) < coreset_size:
        rem = np.array(sorted(remaining))
        mean_red = redundancy_sum[rem] / len(selected)
        winner = int(rem[int(np.argmax(relevance[rem] / (mean_red + 1e-10)))])
        selected.append(winner)
        remaining.discard(winner)
        add_redundancy(winner)
    return np.asarray(selected, int)
