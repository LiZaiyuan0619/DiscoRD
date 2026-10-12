"""Random+KRR and mRMR+KRR (Bowyer et al., 2026).

Both regress the full-benchmark score on the answers to A questions with
degree-2 polynomial kernel ridge regression.  Random+KRR uses the shared
anchor questions; mRMR+KRR chooses its own questions on the memory by
minimum-redundancy maximum-relevance selection (MIQ form).

This follows the authors' released code (MIT licence) and changes only the
computational path, so that it scales to thousands of past models:

* MIQ mRMR with the Ross binary/continuous mutual-information estimator (k = 5),
  and empirical binary/binary mutual information for the redundancy;
* the released degree-2 polynomial kernel ``(x @ y / p + 1)^2`` (sklearn's
  default gamma = 1/p), written as an explicit feature map;
* the same five penalties, chosen by exact leave-one-out error from one SVD.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.special import psi


MI_NEIGHBOURS = 5
KRR_ALPHAS = (0.1, 10 ** -0.5, 1.0, 10 ** 0.5, 10.0)
# Penalties used outside the Open LLM Leaderboard (WILD, DataDecide, AlpacaEval).
KRR_ALPHAS_WIDE = tuple(np.logspace(-2, 2, 9))


@dataclass
class SelectionResult:
    anchors: np.ndarray
    relevance: np.ndarray
    selected_relevance: np.ndarray
    selected_redundancy: np.ndarray
    wall_seconds: float


def _ross_mi_binary_continuous(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    tree_all: cKDTree | None = None,
) -> float:
    """Mutual information between a binary and a continuous variable (Ross, 2014)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return 0.0
    y0 = y[x == 0].reshape(-1, 1)
    y1 = y[x == 1].reshape(-1, 1)
    nx0, nx1 = len(y0), len(y1)
    if nx0 < k + 1 or nx1 < k + 1:
        return 0.0

    tree0 = cKDTree(y0)
    tree1 = cKDTree(y1)
    if tree_all is None:
        tree_all = cKDTree(y.reshape(-1, 1))

    d0 = np.asarray(tree0.query(y0, k=k + 1)[0])[:, -1]
    d1 = np.asarray(tree1.query(y1, k=k + 1)[0])[:, -1]
    m0 = np.asarray(
        tree_all.query_ball_point(y0, d0 - 1e-15, return_length=True)
    )
    m1 = np.asarray(
        tree_all.query_ball_point(y1, d1 - 1e-15, return_length=True)
    )
    if np.any(m0 <= 0) or np.any(m1 <= 0):
        raise FloatingPointError("Ross MI produced a non-positive neighbour count")
    avg_psi_nx = (nx0 * psi(nx0) + nx1 * psi(nx1)) / n
    avg_psi_m = (psi(m0).sum() + psi(m1).sum()) / n
    return float(max(0.0, psi(n) - avg_psi_nx + psi(k) - avg_psi_m))


def _binary_mi_batch(X: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Empirical MI between every binary column of X and binary x2."""
    n = X.shape[0]
    if n == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    # Keep the input dtype (float32) in this sum, as the released code does;
    # promoting it changes MIQ tie-breaking.
    s1 = X.sum(axis=0)
    s2 = x2.sum()
    s11 = np.asarray(X.T @ x2, dtype=np.float64)
    p1_1 = s1 / n
    p1_0 = 1.0 - p1_1
    p2_1 = s2 / n
    p2_0 = 1.0 - p2_1
    p_11 = s11 / n
    p_10 = p1_1 - p_11
    p_01 = p2_1 - p_11
    p_00 = 1.0 - p1_1 - p2_1 + p_11
    out = np.zeros(X.shape[1], dtype=np.float64)
    for joint, marginal in (
        (p_00, p1_0 * p2_0),
        (p_01, p1_0 * p2_1),
        (p_10, p1_1 * p2_0),
        (p_11, p1_1 * p2_1),
    ):
        mask = joint > 0
        out[mask] += joint[mask] * np.log(joint[mask] / (marginal[mask] + 1e-15))
    return np.maximum(out, 0.0)


def select_mrmr_miq(
    source_scores: np.ndarray,
    max_anchors: int = 100,
    *,
    k: int = MI_NEIGHBOURS,
    n_jobs: int = 1,
) -> SelectionResult:
    """Select a nested max-size mRMR-MIQ coreset from models x items scores."""
    started = time.perf_counter()
    X = np.asarray(source_scores)
    if X.ndim != 2 or X.shape[0] <= 1 or X.shape[1] <= max_anchors:
        raise ValueError(f"invalid source score shape {X.shape} for {max_anchors=}")
    finite = X[np.isfinite(X)]
    if finite.size == 0 or not np.isin(np.unique(finite), [0, 1]).all():
        raise ValueError("mRMR-MIQ selection here requires a binary score matrix")
    if not np.isfinite(X).all():
        raise ValueError("mRMR-MIQ selection does not impute missing scores")

    # Relevance target: each model's mean score, as in the released code.
    target = X.mean(axis=1)
    tree_all = cKDTree(target.reshape(-1, 1))
    worker = delayed(_ross_mi_binary_continuous)
    relevance = np.asarray(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            worker(X[:, j], target, k=k, tree_all=tree_all)
            for j in range(X.shape[1])
        ),
        dtype=np.float64,
    )
    if not np.isfinite(relevance).all():
        raise FloatingPointError("non-finite mRMR relevance")

    selected: list[int] = [int(np.argmax(relevance))]
    selected_rel = [float(relevance[selected[0]])]
    selected_red = [0.0]
    remaining = np.ones(X.shape[1], dtype=bool)
    remaining[selected[0]] = False
    red_sum = np.zeros(X.shape[1], dtype=np.float64)
    idx = np.flatnonzero(remaining)
    red_sum[idx] += _binary_mi_batch(X[:, idx], X[:, selected[0]])

    while len(selected) < max_anchors:
        idx = np.flatnonzero(remaining)
        mean_red = red_sum[idx] / len(selected)
        score = relevance[idx] / (mean_red + 1e-10)
        pos = int(np.argmax(score))
        winner = int(idx[pos])
        selected.append(winner)
        selected_rel.append(float(relevance[winner]))
        selected_red.append(float(mean_red[pos]))
        remaining[winner] = False
        idx = np.flatnonzero(remaining)
        if len(idx):
            red_sum[idx] += _binary_mi_batch(X[:, idx], X[:, winner])

    return SelectionResult(
        anchors=np.asarray(selected, dtype=np.int64),
        relevance=relevance,
        selected_relevance=np.asarray(selected_rel),
        selected_redundancy=np.asarray(selected_red),
        wall_seconds=time.perf_counter() - started,
    )


def polynomial2_features(X: np.ndarray) -> np.ndarray:
    """Map matching sklearn ``poly`` default: (x@y / n_features + 1)^2."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    n, p = X.shape
    if p == 0:
        raise ValueError("X must contain at least one feature")
    gamma = 1.0 / p
    d = 1 + 2 * p + p * (p - 1) // 2
    out = np.empty((n, d), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1 : 1 + p] = math.sqrt(2.0 * gamma) * X
    out[:, 1 + p : 1 + 2 * p] = gamma * X * X
    col = 1 + 2 * p
    cross_scale = math.sqrt(2.0) * gamma
    for i in range(p):
        width = p - i - 1
        if width:
            out[:, col : col + width] = cross_scale * X[:, i, None] * X[:, i + 1 :]
            col += width
    assert col == d
    return out


@dataclass
class PolynomialKRR:
    coefficients: np.ndarray
    alpha: float
    n_features: int
    wall_seconds: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(polynomial2_features(X) @ self.coefficients).ravel()


def fit_polynomial2_krr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alphas: tuple[float, ...] = KRR_ALPHAS,
) -> PolynomialKRR:
    """Fit degree-2 KRR with exact SVD leave-one-out alpha selection.

    For a ridge hat matrix H, the leave-one-out residual is
    ``(y - H@y) / (1 - diag(H))``.  A single thin SVD therefore evaluates all
    five released penalties exactly, without refitting thousands of folds.
    """
    started = time.perf_counter()
    phi = polynomial2_features(X)
    target = np.asarray(y, dtype=np.float64).ravel()
    if phi.shape[0] != len(target):
        raise ValueError("X and y row counts differ")
    U, singular, Vt = np.linalg.svd(phi, full_matrices=False)
    projected = U.T @ target
    singular2 = singular * singular
    losses = []
    for alpha in alphas:
        shrink = singular2 / (singular2 + float(alpha))
        fitted = U @ (shrink * projected)
        leverage = (U * U) @ shrink
        denom = 1.0 - leverage
        if np.any(denom <= 1e-12):
            losses.append(np.inf)
        else:
            loo_residual = (target - fitted) / denom
            losses.append(float(np.mean(loo_residual * loo_residual)))
    best = int(np.argmin(losses))
    alpha = float(alphas[best])
    coefficients = Vt.T @ ((singular / (singular2 + alpha)) * projected)
    return PolynomialKRR(
        coefficients=coefficients,
        alpha=alpha,
        n_features=phi.shape[1],
        wall_seconds=time.perf_counter() - started,
    )
