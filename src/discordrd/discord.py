"""DiscoRD: fuse consensus and discordance predictors (Section 3; Appendix A).

The main instance fuses six predictors for each target benchmark:

    consensus     SparseEval-MLP, Ridge transfer, Score-kNN, Response-kNN
    discordance   Discordance Response-kNN, Discordance Score-kNN

SparseEval-MLP and Ridge transfer are computed outside this module (see
``sparseeval_mlp.py`` and ``predictors.py``) and passed in as estimates.  The
four retrieval predictors are computed here from the two distances.  The
fusion weights are fitted on the validation set and smoothed towards equal
weights (``fusion.py``).

When a published method supplies the consensus estimate instead, pass its
estimates under its own name and fuse it with the two discordance predictors
(K = 3); the score distance should then be built from that method's estimates
and the anchor means from its own anchor questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fusion import fusion_weights
from .retrieval import K_GRID, consensus_retrieval, discordance_retrieval

PREDICTORS = ("SparseEval-MLP", "Ridge transfer", "Score-kNN", "Response-kNN",
              "Discordance Response-kNN", "Discordance Score-kNN")
CONSENSUS_PREDICTORS = PREDICTORS[:4]
DISCORDANCE_PREDICTORS = PREDICTORS[4:]
RETRIEVAL_PREDICTORS = PREDICTORS[2:]


@dataclass
class TargetData:
    """Inputs of DiscoRD for one target benchmark in one run.

    Attributes
    ----------
    memory_scores : (n_memory,) full-benchmark scores y_j of the past models
    memory_anchor_means : (n_memory,) anchor means g_j of the past models
    valid_truth : (n_valid,) full scores of the validation models; used only to
        choose k and the fusion weights
    valid_anchor_means, query_anchor_means : anchor means of the validation
        models and of the candidates
    score_distances : (valid, query) distance matrices under the score distance
    response_distances : (valid, query) distance matrices under the response distance
    estimates : {name: (valid, query)} predictions of the consensus predictors
        that are not retrieval predictors, e.g. SparseEval-MLP and Ridge transfer
    """

    memory_scores: np.ndarray
    memory_anchor_means: np.ndarray
    valid_truth: np.ndarray
    valid_anchor_means: np.ndarray
    query_anchor_means: np.ndarray
    score_distances: tuple[np.ndarray, np.ndarray]
    response_distances: tuple[np.ndarray, np.ndarray]
    estimates: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class DiscoRDOutput:
    """Fused predictions together with the parts they were built from."""

    valid: np.ndarray
    query: np.ndarray
    weights: dict[str, float]
    k: dict[str, int]
    components_valid: dict[str, np.ndarray]
    components_query: dict[str, np.ndarray]
    valid_truth: np.ndarray | None = None


def retrieval_predictors(data: TargetData, *, k_grid=K_GRID):
    """The four retrieval predictors: consensus and discordance under two distances.

    Returns
    -------
    (valid, query, k) : dicts keyed by predictor name
    """
    score_v, score_q = data.score_distances
    resp_v, resp_q = data.response_distances
    py = np.asarray(data.memory_scores, float)
    vy = np.asarray(data.valid_truth, float)
    valid, query, ks = {}, {}, {}
    for name, (dv, dq) in (("Score-kNN", (score_v, score_q)),
                           ("Response-kNN", (resp_v, resp_q))):
        result = consensus_retrieval(dv, dq, py, vy, k_grid=k_grid)
        valid[name], query[name], ks[name] = result.valid, result.query, result.k
    for name, (dv, dq) in (("Discordance Response-kNN", (resp_v, resp_q)),
                           ("Discordance Score-kNN", (score_v, score_q))):
        result = discordance_retrieval(dv, dq, py, data.memory_anchor_means,
                                       data.valid_anchor_means, data.query_anchor_means,
                                       vy, k_grid=k_grid)
        valid[name], query[name], ks[name] = result.valid, result.query, result.k
    return valid, query, ks


class DiscoRD:
    """Discordance retrieval with fusion, for one target benchmark at a time.

    Parameters
    ----------
    predictors : names of the fused predictors, in a fixed order.  The default
        is the main instance; ``CONSENSUS_PREDICTORS`` gives the ablation
        without discordance predictors.
    rule : weight rule, one of ``smoothed`` (DiscoRD), ``fitted`` (no
        smoothing), ``equal`` or ``best_single``.
    k_grid : candidate numbers of neighbours.
    equal_if_unsolvable : use equal weights when the simplex problem has no
        solution (very small validation sets).

    Example
    -------
    >>> model = DiscoRD()
    >>> out = model.fit_predict(data)      # data: TargetData
    >>> out.query                          # estimates for the candidates
    """

    def __init__(self, predictors=PREDICTORS, rule: str = "smoothed", *,
                 k_grid=K_GRID, equal_if_unsolvable: bool = False):
        self.predictors = tuple(predictors)
        self.rule = rule
        self.k_grid = tuple(k_grid)
        self.equal_if_unsolvable = equal_if_unsolvable

    def fit_predict(self, data: TargetData) -> DiscoRDOutput:
        valid = {name: np.asarray(v, float) for name, (v, _) in data.estimates.items()}
        query = {name: np.asarray(q, float) for name, (_, q) in data.estimates.items()}
        ks: dict[str, int] = {}
        if any(name in RETRIEVAL_PREDICTORS for name in self.predictors):
            rv, rq, ks = retrieval_predictors(data, k_grid=self.k_grid)
            valid.update(rv)
            query.update(rq)
        missing = [name for name in self.predictors if name not in valid]
        if missing:
            raise KeyError(f"no predictions for {missing}; pass them in TargetData.estimates")
        V = np.column_stack([valid[name] for name in self.predictors])
        Q = np.column_stack([query[name] for name in self.predictors])
        w = fusion_weights(V, data.valid_truth, self.rule,
                           equal_if_unsolvable=self.equal_if_unsolvable)
        return DiscoRDOutput(valid=V @ w, query=Q @ w,
                             weights=dict(zip(self.predictors, w.tolist())),
                             k={name: ks[name] for name in self.predictors if name in ks},
                             components_valid=valid, components_query=query,
                             valid_truth=np.asarray(data.valid_truth, float))


# Rows of the ablation table: (fused predictors, weight rule).
ABLATIONS = {
    "- weight smoothing": (PREDICTORS, "fitted"),
    "- discordance predictors": (CONSENSUS_PREDICTORS, "smoothed"),
    "- weight smoothing, - discordance predictors": (CONSENSUS_PREDICTORS, "fitted"),
    "Equal fusion weights": (PREDICTORS, "equal"),
    "Best single predictor": (PREDICTORS, "best_single"),
}


def ablation_estimates(output: DiscoRDOutput, ablations=None, *,
                       equal_if_unsolvable: bool = False) -> dict[str, np.ndarray]:
    """Fuse the same predictions again under the ablated rules (no predictor is refitted)."""
    out = {}
    for name, (predictors, rule) in (ablations or ABLATIONS).items():
        V = np.column_stack([output.components_valid[p] for p in predictors])
        Q = np.column_stack([output.components_query[p] for p in predictors])
        w = fusion_weights(V, output.valid_truth, rule, equal_if_unsolvable=equal_if_unsolvable)
        out[name] = Q @ w
    return out
