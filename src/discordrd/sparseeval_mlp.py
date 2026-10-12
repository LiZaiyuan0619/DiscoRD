"""SparseEval-MLP: an MLP that regresses the full-benchmark score on the anchor answers.

Re-implementation of the MLP estimator of SparseEval (Zhang et al., 2026,
"SparseEval: Efficient Evaluation of Large Language Models by Sparse
Optimization", ICLR 2026), following the network, the k-means initialisation
of the anchors and the gradient-guided anchor refinement of the paper and its
official code.  Two choices differ from the official script and are used for
every setting of our experiments:

* the MLP and the anchor refinement are trained on the memory only, and the
  checkpoint is chosen on the validation set (never on the candidates);
* the regression target is the weighted full-benchmark score, so benchmarks
  that macro-average their subtasks are handled with their item weights.

``fit_predict`` either selects the anchors itself (``fixed_anchors=None``, as
on the Open LLM Leaderboard) or uses anchors drawn beforehand (uniformly at
random, as on WILD, DataDecide and AlpacaEval; there ``refinement_steps`` is
irrelevant).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def score_tensor(Y: torch.Tensor, item_weights: torch.Tensor) -> torch.Tensor:
    return Y @ item_weights


def train_item_vectors(Y_train: np.ndarray) -> np.ndarray:
    """Normalised item vectors from the memory only; no evaluated model selects anchors."""
    item_matrix = (2.0 * np.asarray(Y_train, np.float32) - 1.0).T
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return item_matrix / norms


def _mlp(n_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(32, 1), nn.Sigmoid(),
    )


def generate_anchors(Y_train: torch.Tensor, item_weights: torch.Tensor,
                     num_anchors: int, seed: int, device: torch.device,
                     *, refinement_steps: int = 10, proxy_epochs: int = 2000,
                     learning_rate: float = 5e-4) -> list[int]:
    """SparseEval's anchor selection: k-means initialisation, then refinement.

    Each refinement step trains a proxy MLP on the current anchors, drops the
    anchor with the smallest input-gradient importance and adds the question
    whose answers correlate most with the proxy's residuals.
    """
    if num_anchors <= 0 or num_anchors > Y_train.shape[1]:
        raise ValueError("invalid anchor count")
    rng = random.Random(seed)
    vectors = train_item_vectors(Y_train.detach().cpu().numpy())
    # k-means initialisation, as in the published default.
    kmeans = KMeans(n_clusters=num_anchors, random_state=seed, n_init="auto")
    kmeans.fit(vectors)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
    anchors = list(np.unique(closest).astype(int))
    while len(anchors) < num_anchors:
        candidate = rng.randrange(vectors.shape[0])
        if candidate not in anchors:
            anchors.append(candidate)
    y_train = score_tensor(Y_train, item_weights)

    for _ in range(refinement_steps):
        model = _mlp(len(anchors)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        X = Y_train[:, anchors]
        model.train()
        for _ in range(proxy_epochs):
            optimizer.zero_grad()
            loss = criterion(model(X).squeeze(-1), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        X_grad = Y_train[:, anchors].detach().clone().requires_grad_(True)
        model.zero_grad()
        loss = criterion(model(X_grad).squeeze(-1), y_train)
        loss.backward()
        importance = torch.abs(X_grad.grad).mean(dim=0)
        weakest = int(torch.argmin(importance).item())
        with torch.no_grad():
            error = model(Y_train[:, anchors]).squeeze(-1) - y_train
            potential = torch.abs((Y_train - 0.5).T @ error)
            potential[anchors] = -1.0
            challenger = int(torch.argmax(potential).item())
        if challenger not in anchors:
            anchors[weakest] = challenger
    if len(set(anchors)) != num_anchors:
        raise AssertionError("anchor generator returned duplicates")
    return anchors


def fit_predict(Y_train: np.ndarray, Y_valid: np.ndarray,
                prediction_cohorts: dict[str, np.ndarray], item_weights: np.ndarray,
                num_anchors: int, seed: int, *, device: str = "cpu",
                refinement_steps: int = 10, proxy_epochs: int = 2000,
                num_epochs: int = 20000, patience: int = 30,
                learning_rate: float = 5e-4, eval_every: int = 500,
                fixed_anchors: np.ndarray | None = None) -> dict:
    """Fit SparseEval-MLP on the memory and predict the full score of other models.

    Parameters
    ----------
    Y_train : (n_memory, n_questions) answers of the past models
    Y_valid : (n_valid, n_questions) answers of the validation models (checkpoint choice)
    prediction_cohorts : {name: answers} of the models to predict; only the
        anchor columns are read
    item_weights : (n_questions,) weights of the questions, summing to one
    num_anchors : number of anchor questions A
    seed : random seed of the fit
    fixed_anchors : anchors drawn beforehand, or None to let SparseEval choose

    Returns
    -------
    dict with ``anchors``, ``predictions`` (per cohort), ``truths`` (per cohort),
    and the best validation loss, Kendall tau and epoch.
    """
    torch_device = torch.device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train = torch.as_tensor(Y_train, dtype=torch.float32, device=torch_device)
    valid = torch.as_tensor(Y_valid, dtype=torch.float32, device=torch_device)
    weights = torch.as_tensor(item_weights, dtype=torch.float32, device=torch_device)
    if weights.ndim != 1 or len(weights) != train.shape[1] or not torch.isclose(weights.sum(), torch.tensor(1.0, device=torch_device), atol=1e-6):
        raise ValueError("item_weights must be a length-n_items probability vector")
    cohorts = {name: torch.as_tensor(value, dtype=torch.float32, device=torch_device)
               for name, value in prediction_cohorts.items()}
    if fixed_anchors is None:
        anchors = generate_anchors(train, weights, num_anchors, seed, torch_device,
                                   refinement_steps=refinement_steps,
                                   proxy_epochs=proxy_epochs,
                                   learning_rate=learning_rate)
    else:
        anchors = np.asarray(fixed_anchors)
        if (anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer)
                or len(anchors) != num_anchors or len(np.unique(anchors)) != num_anchors
                or np.any(anchors < 0) or np.any(anchors >= train.shape[1])):
            raise ValueError('invalid fixed anchors')
        anchors = anchors.astype(int).tolist()
    model = _mlp(len(anchors)).to(torch_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    y_train = score_tensor(train, weights)
    y_valid = score_tensor(valid, weights)
    best_loss = float("inf"); best_tau = -2.0; best_epoch = -1; stale = 0
    best_predictions = None
    for epoch in range(num_epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(train[:, anchors]).squeeze(-1), y_train)
        loss.backward(); optimizer.step()
        if epoch % eval_every:
            continue
        model.eval()
        with torch.no_grad():
            pred_valid = model(valid[:, anchors]).squeeze(-1)
            valid_loss = float(criterion(pred_valid, y_valid).item())
            tau = float(kendalltau(y_valid.cpu().numpy(), pred_valid.cpu().numpy()).statistic)
            if not np.isfinite(tau):
                tau = -2.0
            if valid_loss <= best_loss and tau >= best_tau:
                best_loss, best_tau, best_epoch, stale = valid_loss, tau, epoch, 0
                best_predictions = {
                    name: model(value[:, anchors]).squeeze(-1).detach().cpu().numpy().astype(np.float64)
                    for name, value in cohorts.items()
                }
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_predictions is None:
        raise RuntimeError("SparseEval-MLP found no valid checkpoint")
    truths = {name: score_tensor(value, weights).detach().cpu().numpy().astype(np.float64)
              for name, value in cohorts.items()}
    return {"anchors": np.asarray(anchors, dtype=int), "predictions": best_predictions,
            "truths": truths, "best_valid_loss": best_loss,
            "best_valid_tau": best_tau, "best_epoch": best_epoch}
