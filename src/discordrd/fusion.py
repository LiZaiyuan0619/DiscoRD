"""Fusion of the predictors (Section 3; Appendix A, "Fusion").

For each benchmark the fusion weights minimise the mean absolute error on the
validation set over the simplex.  They are then smoothed towards equal weights,

    alpha' = (1 - lam) * alpha + lam / K,    lam = K / (K + |V|),

where K is the number of fused predictors and |V| the size of the validation
set.  A small validation set drives lam towards one (equal weights); a large
one keeps the fitted weights.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

WEIGHT_RULES = ("smoothed", "fitted", "equal", "best_single")


def simplex_weights(predictions, truth) -> np.ndarray:
    """Weights on the simplex that minimise the mean absolute error.

    SLSQP is started from equal weights and from every vertex; the best
    successful solution is kept.

    Parameters
    ----------
    predictions : array of shape (n_models, K), one column per predictor
    truth : array of shape (n_models,)
    """
    X = np.asarray(predictions, float)
    y = np.asarray(truth, float)
    n_arms = X.shape[1]
    best = None
    for start in (np.ones(n_arms) / n_arms, *np.eye(n_arms)):
        result = minimize(
            lambda weights: np.mean(np.abs(X @ weights - y)), start,
            method="SLSQP", bounds=[(0, 1)] * n_arms,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("simplex optimization failed")
    return best.x


def smoothed_weights(weights, n_valid: int) -> np.ndarray:
    """Smooth fitted weights towards equal weights with lam = K / (K + n_valid)."""
    weights = np.asarray(weights, float)
    k = len(weights)
    lam = k / (k + n_valid)
    return (1 - lam) * weights + lam * np.full(k, 1 / k)


def fusion_weights(valid_predictions, valid_truth, rule: str = "smoothed", *,
                   equal_if_unsolvable: bool = False) -> np.ndarray:
    """Weights of the fused predictors under one rule.

    rule
        ``smoothed``     simplex weights smoothed towards equal weights (DiscoRD)
        ``fitted``       simplex weights without smoothing
        ``equal``        equal weights
        ``best_single``  all weight on the predictor with the lowest validation error
    equal_if_unsolvable
        fall back to equal weights when the simplex problem has no solution,
        which can happen with two or three validation models.
    """
    V = np.asarray(valid_predictions, float)
    vy = np.asarray(valid_truth, float)
    k = V.shape[1]
    if rule == "equal":
        return np.full(k, 1 / k)
    if rule == "best_single":
        best = np.zeros(k)
        best[int(np.abs(V - vy[:, None]).mean(0).argmin())] = 1.0
        return best
    if rule not in ("smoothed", "fitted"):
        raise ValueError(f"unknown weight rule {rule!r}; choose from {WEIGHT_RULES}")
    try:
        fitted = simplex_weights(V, vy)
    except RuntimeError:
        if not equal_if_unsolvable:
            raise
        fitted = np.full(k, 1 / k)
    return fitted if rule == "fitted" else smoothed_weights(fitted, len(vy))
