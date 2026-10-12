"""Shared pieces of the per-collection pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..discord import PREDICTORS, DiscoRD, DiscoRDOutput, TargetData, ablation_estimates
from ..distances import response_distances, score_distances
from ..ladder import ladder_estimates
from ..metrics import evaluate
from ..predictors import anchor_mean, ridge_transfer
from ..sparseeval_mlp import fit_predict

DEFAULT_EPOCHS = 20000


@dataclass
class CellResult:
    """Estimates for the candidates of one cell (one benchmark in one run)."""

    collection: str
    setting: str
    benchmark: str
    budget: int
    seed: int
    repeat: int
    candidates: np.ndarray
    truth: np.ndarray
    predictions: dict[str, np.ndarray]
    weights: dict[str, float] = field(default_factory=dict)
    k: dict[str, int] = field(default_factory=dict)

    def keys(self) -> dict:
        return dict(collection=self.collection, setting=self.setting, benchmark=self.benchmark,
                    budget=self.budget, seed=self.seed, repeat=self.repeat)

    def metric_rows(self) -> list[dict]:
        """One row of the 31 metrics per method."""
        return [{**self.keys(), "method": name, "n_candidates": len(self.truth),
                 **evaluate(pred, self.truth)}
                for name, pred in self.predictions.items()]

    def prediction_frame(self) -> pd.DataFrame:
        """Long table: one row per (method, candidate)."""
        return pd.concat([pd.DataFrame({**self.keys(), "method": name,
                                        "model_name": self.candidates,
                                        "truth": self.truth, "estimate": pred})
                          for name, pred in self.predictions.items()], ignore_index=True)


def single_benchmark_discord(memory, valid, query, anchors, *, fit_seed: int,
                             item_weights=None, horvitz_thompson: bool = False,
                             sparseeval_options: dict | None = None,
                             predictors=PREDICTORS, rule: str = "smoothed"):
    """DiscoRD on a benchmark that is evaluated on its own (AlpacaEval, DataDecide).

    Ridge transfer is fitted on the anchor answers of the memory, and both
    distances use this benchmark alone.  The candidates' answers outside the
    anchors are never read.

    Parameters
    ----------
    memory, valid, query : per-question scores of the past models, the
        validation models and the candidates
    anchors : indices of the anchor questions
    fit_seed : seed of the SparseEval-MLP fit
    item_weights : question weights (uniform plain means when None)
    horvitz_thompson : anchor means in Horvitz-Thompson form (DataDecide)
    sparseeval_options : extra keyword arguments of ``sparseeval_mlp.fit_predict``

    Returns
    -------
    (DiscoRDOutput, TargetData)
    """
    memory, valid, query = (np.asarray(x, float) for x in (memory, valid, query))
    anchors = np.asarray(anchors, int)
    n_questions = memory.shape[1]
    if item_weights is None:
        weights = np.full(n_questions, 1 / n_questions)
        y_memory, y_valid = memory.mean(axis=1), valid.mean(axis=1)

        def g(x):
            return anchor_mean(x, anchors)
    else:
        weights = np.asarray(item_weights, float)
        y_memory, y_valid = memory @ weights, valid @ weights

        def g(x):
            return anchor_mean(x, anchors, weights, horvitz_thompson=horvitz_thompson)

    # The candidates reveal their anchor answers only.
    observed = np.zeros_like(query)
    observed[:, anchors] = query[:, anchors]

    torch.set_num_threads(1)
    options = {"device": "cpu", **(sparseeval_options or {})}
    fit = fit_predict(memory, valid, {"valid": valid, "query": observed}, weights,
                      len(anchors), fit_seed, fixed_anchors=anchors, **options)
    mlp_valid, mlp_query = fit["predictions"]["valid"], fit["predictions"]["query"]
    ridge_valid, ridge_query = ridge_transfer(memory[:, anchors], y_memory,
                                              valid[:, anchors], observed[:, anchors])
    data = TargetData(
        memory_scores=y_memory,
        memory_anchor_means=g(memory),
        valid_truth=y_valid,
        valid_anchor_means=g(valid),
        query_anchor_means=g(observed),
        score_distances=(score_distances(y_memory, mlp_valid),
                         score_distances(y_memory, mlp_query)),
        response_distances=(response_distances(memory[:, anchors], valid[:, anchors]),
                            response_distances(memory[:, anchors], observed[:, anchors])),
        estimates={"SparseEval-MLP": (mlp_valid, mlp_query),
                   "Ridge transfer": (ridge_valid, ridge_query)},
    )
    return DiscoRD(predictors, rule).fit_predict(data), data


def candidate_estimates(output: DiscoRDOutput, data: TargetData | None = None, *,
                        extras: bool = False, equal_if_unsolvable: bool = False) -> dict[str, np.ndarray]:
    """Estimates for the candidates: every predictor and DiscoRD.

    With ``extras``, also the rows of the ablation table: the fusions without
    smoothing, without discordance predictors, with equal weights or with the
    single best predictor, and the corrections built up from no memory.
    """
    out = {**output.components_query, "DiscoRD": output.query}
    if extras:
        out.update(ablation_estimates(output, equal_if_unsolvable=equal_if_unsolvable))
        if data is not None:
            out.update(ladder_estimates(data))
    return out
