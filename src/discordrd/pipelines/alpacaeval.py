"""AlpacaEval: one benchmark of 790 instructions with continuous scores in [0, 1].

Models are split at random into memory (60%), validation set (20%) and
candidates (20%); the split seed is the run seed.  Anchor questions are drawn
uniformly at random.
"""
from __future__ import annotations

from ..data import ResponseMatrix
from ..discord import PREDICTORS
from ..protocol import anchor_seed, random_split, uniform_anchors
from .common import CellResult, candidate_estimates, single_benchmark_discord

COLLECTION = "alpacaeval"
BUDGETS = (20, 100)


def run_cell(matrix: ResponseMatrix, seed: int, repeat: int, budget: int, *,
             sparseeval_options: dict | None = None, predictors=PREDICTORS,
             rule: str = "smoothed", extras: bool = False) -> CellResult:
    y = matrix.responses
    memory, valid, candidates = random_split(len(y), seed)
    draw = anchor_seed(COLLECTION, seed, repeat)
    anchors = uniform_anchors(y.shape[1], budget, draw)
    output, data = single_benchmark_discord(y[memory], y[valid], y[candidates], anchors,
                                      fit_seed=draw, sparseeval_options=sparseeval_options,
                                      predictors=predictors, rule=rule)
    return CellResult(COLLECTION, f"AlpacaEval A={budget}", matrix.name, budget, seed, repeat,
                      candidates=matrix.model_names[candidates],
                      truth=y[candidates].mean(axis=1),
                      predictions=candidate_estimates(output, data, extras=extras),
                      weights=output.weights, k=output.k)
