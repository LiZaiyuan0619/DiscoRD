"""Open LLM Leaderboard (OLLB): six benchmarks per leaderboard version.

Every model of a setting has one role per split seed: ``memory`` (past models
whose answers are all known), ``transfer`` (a further set of past models that
fits Ridge transfer), ``validation`` or ``candidate``.  On OLLB the anchor
questions are chosen as in SparseEval, so the pipeline has two stages:

1. ``fit_sparseeval``: for each benchmark, SparseEval-MLP selects A anchor
   questions on the memory and estimates every other model's score;
2. ``run_cell``: the six predictors of DiscoRD are built across the six
   benchmarks and fused for each target benchmark.

Ridge transfer predicts the target benchmark from the SparseEval-MLP estimates
on the other five benchmarks and is fitted on the transfer models.  The
response distance concatenates the anchor answers of the six benchmarks, and
the score distance compares a past model's six true scores with a candidate's
six SparseEval-MLP estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data import ResponseMatrix
from ..discord import PREDICTORS, DiscoRD, TargetData
from ..distances import response_distances, score_distances
from ..predictors import anchor_mean, ridge_transfer
from ..protocol import ollb_fit_seed
from ..sparseeval_mlp import fit_predict
from .common import CellResult, candidate_estimates

COLLECTION = "ollb"
V2_BENCHMARKS = ("ifeval", "bbh", "math_lvl5", "gpqa", "musr", "mmlu_pro")
V1_BENCHMARKS = ("arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande")
BUDGETS = (20, 100)
ROLES = ("memory", "transfer", "validation", "candidate")


def roles_for_seed(splits: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Model names of each role for one split seed (``splits`` has columns model_name, seed<k>)."""
    column = f"seed{seed}"
    return {role: splits.loc[splits[column] == role, "model_name"].to_numpy(str) for role in ROLES}


def fit_sparseeval(matrix: ResponseMatrix, roles: dict[str, np.ndarray], budget: int,
                   seed: int, repeat: int, *, device: str = "cpu", **options) -> dict:
    """Stage 1 on one benchmark: SparseEval's anchors and its estimates for every non-memory model."""
    position = {m: i for i, m in enumerate(matrix.model_names)}
    rows = {role: np.asarray([position[m] for m in names], int) for role, names in roles.items()}
    y = matrix.responses.astype(np.float32)
    if device == "cpu":
        torch.set_num_threads(1)
    fit = fit_predict(y[rows["memory"]], y[rows["validation"]],
                      {role: y[rows[role]] for role in ("transfer", "validation", "candidate")},
                      matrix.item_weights, budget, seed=ollb_fit_seed(seed, repeat),
                      device=device, **options)
    return {"anchors": fit["anchors"],
            "estimates": {role: pd.Series(fit["predictions"][role], index=roles[role])
                          for role in ("transfer", "validation", "candidate")}}


def run_cell(matrices: dict[str, ResponseMatrix], roles: dict[str, np.ndarray],
             stage1: dict[str, dict], setting: str, budget: int, seed: int, repeat: int, *,
             predictors=PREDICTORS, rule: str = "smoothed", extras: bool = False) -> list[CellResult]:
    """Stage 2: DiscoRD for each of the six benchmarks of one run.

    ``stage1`` maps benchmark -> output of ``fit_sparseeval`` for this run.
    """
    benches = list(matrices)

    def rows_of(bench, role):
        position = {m: i for i, m in enumerate(matrices[bench].model_names)}
        return np.asarray([position[m] for m in roles[role]], int)

    def scores(bench, role):
        return matrices[bench].scores(rows_of(bench, role))

    def answers(role):
        return np.hstack([matrices[b].responses[rows_of(b, role)][:, stage1[b]["anchors"]]
                          for b in benches])

    def estimates(role, names):
        return np.column_stack([stage1[b]["estimates"][role].loc[roles[role]].to_numpy(float)
                                for b in names])

    memory_profiles = np.column_stack([scores(b, "memory") for b in benches])
    response = (response_distances(answers("memory"), answers("validation")),
                response_distances(answers("memory"), answers("candidate")))
    score = (score_distances(memory_profiles, estimates("validation", benches)),
             score_distances(memory_profiles, estimates("candidate", benches)))
    results = []
    for target in benches:
        others = [b for b in benches if b != target]
        ridge_valid, ridge_query = ridge_transfer(
            estimates("transfer", others), scores(target, "transfer"),
            estimates("validation", others), estimates("candidate", others))
        matrix, anchors = matrices[target], stage1[target]["anchors"]

        def g(role):
            return anchor_mean(matrix.responses[rows_of(target, role)], anchors, matrix.item_weights)

        mlp = stage1[target]["estimates"]
        data = TargetData(
            memory_scores=scores(target, "memory"),
            memory_anchor_means=g("memory"),
            valid_truth=scores(target, "validation"),
            valid_anchor_means=g("validation"),
            query_anchor_means=g("candidate"),
            score_distances=score,
            response_distances=response,
            estimates={"SparseEval-MLP": (mlp["validation"].loc[roles["validation"]].to_numpy(float),
                                          mlp["candidate"].loc[roles["candidate"]].to_numpy(float)),
                       "Ridge transfer": (ridge_valid, ridge_query)},
        )
        out = DiscoRD(predictors, rule).fit_predict(data)
        results.append(CellResult(COLLECTION, f"{setting} A={budget}", target, budget, seed, repeat,
                                  candidates=roles["candidate"], truth=scores(target, "candidate"),
                                  predictions=candidate_estimates(out, data, extras=extras),
                                  weights=out.weights, k=out.k))
    return results
