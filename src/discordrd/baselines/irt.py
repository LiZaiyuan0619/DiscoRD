"""Item response models shared by the IRT-based baselines.

* a one-parameter (1PL) model, p = sigmoid(theta + z_i), with item easiness
  z_i = logit of the item's mean score in the memory;
* Nested MIRT (one general ability plus K specific factors), fitted on the
  memory, with MAP scoring under the empirical Gaussian prior and V-optimal
  question selection.  This is a re-implementation from the equations of
  Krumdick et al. (2026), not the authors' code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit


# ---------------------------------------------------------------------------
# 1PL
# ---------------------------------------------------------------------------
def memory_easiness(memory) -> np.ndarray:
    """1PL item easiness z with p = sigmoid(theta + z), from the memory's mean scores."""
    p = np.clip(np.asarray(memory, float).mean(axis=0), 1e-3, 1 - 1e-3)
    return logit(p)


def theta_map_1pl(responses, z) -> float:
    """MAP ability over the given items under a N(0, 1) prior."""
    responses = np.asarray(responses, float)
    z = np.asarray(z, float)

    def objective(theta: float) -> float:
        p = expit(theta + z)
        nll = -float(np.sum(responses * np.log(p + 1e-10)
                            + (1 - responses) * np.log(1 - p + 1e-10)))
        return nll + 0.5 * theta * theta
    return float(minimize_scalar(objective, bounds=(-6, 6), method="bounded").x)


def theta_mle_1pl(y, z, bounds=(-20.0, 20.0)) -> float:
    """Bounded maximum-likelihood ability: the root of sum(y - sigmoid(theta + z)).

    Accepts scores in [0, 1] (soft labels).  All-correct or all-wrong answers
    return the bound.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    lo, hi = map(float, bounds)

    def score(theta):
        return float(np.sum(y - expit(theta + z)))

    if score(lo) <= 0.0:
        return lo
    if score(hi) >= 0.0:
        return hi
    return float(brentq(score, lo, hi, xtol=1e-10, rtol=1e-12))


# ---------------------------------------------------------------------------
# Nested MIRT
# ---------------------------------------------------------------------------
class NestedMIRT(nn.Module):
    def __init__(self, n_models: int, n_items: int, k: int = 3):
        super().__init__()
        self.k = int(k)
        self.dim = self.k + 1
        self.theta = nn.Parameter(torch.zeros(n_models, self.dim))
        self.item_loading = nn.Parameter(torch.randn(n_items, self.k) * 0.05)
        self.raw_alpha = nn.Parameter(torch.zeros(n_items))
        self.difficulty = nn.Parameter(torch.zeros(n_items))

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha) + 0.05

    def logits(self, model_idx, item_idx, *, general_only=False):
        theta = self.theta[model_idx]
        general = theta[:, 0]
        specific = (0.0 if general_only else
                    (theta[:, 1:] * self.item_loading[item_idx]).sum(dim=1))
        return self.alpha[item_idx] * (general + specific - self.difficulty[item_idx])


@dataclass
class MIRTParameters:
    alpha: np.ndarray
    difficulty: np.ndarray
    loading: np.ndarray
    theta_mean: np.ndarray
    theta_cov: np.ndarray


def fit_nested_mirt(responses, *, k: int = 3, nested_lambda: float = 1.0, epochs: int = 300,
                    lr: float = 0.03, batch_size: int = 65536, seed: int = 42,
                    device: str = "cpu") -> MIRTParameters:
    """Fit Nested MIRT on a binary models x items matrix (the memory)."""
    Y = np.asarray(responses, np.float32)
    if Y.ndim != 2 or not np.isin(Y[np.isfinite(Y)], [0, 1]).all():
        raise ValueError("Nested MIRT requires a models x items binary matrix")
    observed_models, observed_items = np.where(np.isfinite(Y))
    observed_values = Y[observed_models, observed_items]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NestedMIRT(Y.shape[0], Y.shape[1], k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = rng.permutation(len(observed_values))
        for start in range(0, len(order), batch_size):
            take = order[start:start + batch_size]
            mi = torch.as_tensor(observed_models[take], dtype=torch.long, device=device)
            ii = torch.as_tensor(observed_items[take], dtype=torch.long, device=device)
            y = torch.as_tensor(observed_values[take], dtype=torch.float32, device=device)
            full = F.binary_cross_entropy_with_logits(model.logits(mi, ii), y)
            general = F.binary_cross_entropy_with_logits(model.logits(mi, ii, general_only=True), y)
            prior = 0.5 * model.theta.square().mean()
            loss = full + nested_lambda * general + 1e-2 * prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        theta = model.theta.detach().cpu().numpy()
        cov = np.cov(theta, rowvar=False) + np.eye(model.dim) * 1e-4
        return MIRTParameters(model.alpha.detach().cpu().numpy(),
                              model.difficulty.detach().cpu().numpy(),
                              model.item_loading.detach().cpu().numpy(),
                              theta.mean(axis=0), cov)


def fit_soft_mirt(y, seed: int, k: int = 3, nested_lambda: float = 1.0,
                  epochs: int = 300) -> MIRTParameters:
    """Nested MIRT fitted with scores in [0, 1] read as soft labels.

    Used outside the Open LLM Leaderboard (for binary scores it is the same
    model).  ``nested_lambda=0`` gives a plain (non-nested) MIRT, which the
    tinyBenchmarks baselines use with ``k=14``.
    """
    y = np.asarray(y, np.float32)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("soft MIRT requires finite [0,1] responses")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NestedMIRT(*y.shape, k=k)
    opt = torch.optim.Adam(model.parameters(), lr=.03)
    mi, ii = np.indices(y.shape)
    mi, ii, values = mi.ravel(), ii.ravel(), y.ravel()
    for _ in range(epochs):
        order = rng.permutation(len(values))
        for start in range(0, len(order), 65536):
            take = order[start:start + 65536]
            a, b = torch.as_tensor(mi[take]), torch.as_tensor(ii[take])
            target = torch.as_tensor(values[take])
            loss = F.binary_cross_entropy_with_logits(model.logits(a, b), target)
            if nested_lambda:
                loss += nested_lambda * F.binary_cross_entropy_with_logits(
                    model.logits(a, b, general_only=True), target)
            loss += .005 * model.theta.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        theta = model.theta.numpy()
        return MIRTParameters(model.alpha.numpy(), model.difficulty.numpy(),
                              model.item_loading.numpy(), theta.mean(axis=0),
                              np.cov(theta, rowvar=False) + np.eye(k + 1) * 1e-4)


def _numpy_logits(theta, params: MIRTParameters, items):
    return params.alpha[items] * (theta[0] + params.loading[items] @ theta[1:]
                                  - params.difficulty[items])


def map_theta(observed_items, observed_responses, params: MIRTParameters, *,
              max_iter: int = 80) -> np.ndarray:
    """MAP ability vector from the observed items under the empirical Gaussian prior."""
    items = np.asarray(observed_items, int)
    y = np.asarray(observed_responses, np.float64)
    mean = torch.tensor(params.theta_mean, dtype=torch.float64)
    precision = torch.tensor(np.linalg.inv(params.theta_cov), dtype=torch.float64)
    alpha = torch.tensor(params.alpha[items], dtype=torch.float64)
    loading = torch.tensor(params.loading[items], dtype=torch.float64)
    difficulty = torch.tensor(params.difficulty[items], dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    theta = mean.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = alpha * (theta[0] + loading @ theta[1:] - difficulty)
        delta = theta - mean
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        loss = loss + 0.5 * delta @ precision @ delta
        loss.backward()
        return loss
    optimizer.step(closure)
    return theta.detach().numpy()


def predict_probabilities(theta, params: MIRTParameters) -> np.ndarray:
    items = np.arange(len(params.alpha))
    return 1.0 / (1.0 + np.exp(-np.clip(_numpy_logits(theta, params, items), -30, 30)))


def v_optimal_order(params: MIRTParameters, budget: int) -> np.ndarray:
    """Sequential V-optimal design: the ``budget`` items that most reduce the ability variance."""
    if budget <= 0 or budget > len(params.alpha):
        raise ValueError("invalid V-optimal budget")
    vectors = params.alpha[:, None] * np.column_stack([np.ones(len(params.alpha)), params.loading])
    covariance = params.theta_cov.copy()
    remaining = np.ones(len(params.alpha), bool)
    selected = []
    theta = params.theta_mean
    for _ in range(budget):
        p = 1.0 / (1.0 + np.exp(-np.clip(_numpy_logits(theta, params, np.arange(len(vectors))), -30, 30)))
        fisher_weight = p * (1 - p)
        Cv = vectors @ covariance
        denom = 1.0 + fisher_weight * np.sum(Cv * vectors, axis=1)
        score = fisher_weight * np.sum(Cv * Cv, axis=1) / denom
        score[~remaining] = -np.inf
        winner = int(np.argmax(score))
        selected.append(winner)
        remaining[winner] = False
        v = vectors[winner][:, None]
        w = fisher_weight[winner]
        covariance = covariance - (w * covariance @ v @ v.T @ covariance) / (
            1.0 + w * float((v.T @ covariance @ v).item()))
    return np.asarray(selected, int)
