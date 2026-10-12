"""WILD: 65 production models on the 24 benchmarks with at least 500 questions.

The outer folds hold out one organisation at a time (its models are the
candidates); inside each fold the other organisations are split into memory
and validation set.  Anchor questions are drawn uniformly at random on every
benchmark.  DiscoRD works across the suite: the response distance concatenates
the anchor answers of all benchmarks, the score distance compares profiles
over all benchmarks, and Ridge transfer predicts the target benchmark from the
anchor means on the other benchmarks (fitted on the memory).

Metrics: MAE and RMSE are computed within each fold and averaged over the
folds of a cell; the ranking and selection metrics are computed on the
candidates pooled over the folds (a single fold holds 1-17 candidates).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import ResponseMatrix
from ..discord import PREDICTORS, DiscoRD, TargetData
from ..distances import response_distances, score_distances
from ..metrics import evaluate
from ..predictors import anchor_mean, ridge_transfer
from ..protocol import anchor_seed, organisation_folds, uniform_anchors
from ..sparseeval_mlp import fit_predict
from .common import DEFAULT_EPOCHS, CellResult, candidate_estimates

COLLECTION = "wild"
MIN_QUESTIONS = 500
BUDGETS = (10, 20, 50)


def organisation(model_id: str) -> str:
    """Organisation that released a model, from its identifier."""
    name = str(model_id).lower()
    if "/" in name:
        return name.split("/", 1)[0]
    prefixes = (("gpt-", "openai"), ("claude-", "anthropic"),
                ("nova-", "amazon"), ("llama-", "meta"))
    for prefix, org in prefixes:
        if name.startswith(prefix):
            return org
    raise ValueError(f"model needs an explicit organisation mapping: {model_id}")


class Suite:
    """All benchmarks of WILD restricted to one shared, identically ordered model set."""

    def __init__(self, matrices: dict[str, ResponseMatrix], min_questions: int = MIN_QUESTIONS):
        kept = sorted((m for m in matrices.values() if m.n_questions >= min_questions),
                      key=lambda m: m.name)
        common = sorted(set.intersection(*[set(m.model_names) for m in kept]))
        self.models = np.asarray(common)
        self.tasks = {m.name: m.subset_models(common) for m in kept}
        self.names = list(self.tasks)
        self.groups = np.asarray([organisation(m) for m in self.models])

    def responses(self, task: str) -> np.ndarray:
        return self.tasks[task].responses

    def weights(self, task: str) -> np.ndarray:
        return self.tasks[task].item_weights


@dataclass
class FoldedCellResult(CellResult):
    """A cell whose candidates come from several folds."""

    folds: np.ndarray = None

    def metric_rows(self) -> list[dict]:
        rows = []
        for name, pred in self.predictions.items():
            metrics = evaluate(pred, self.truth)
            p = np.clip(np.asarray(pred, float), 0, 1)
            errors = [p[self.folds == f] - self.truth[self.folds == f] for f in np.unique(self.folds)]
            metrics["MAE"] = float(np.mean([np.mean(np.abs(e)) for e in errors]))
            metrics["RMSE"] = float(np.mean([np.sqrt(np.mean(e ** 2)) for e in errors]))
            rows.append({**self.keys(), "method": name, "n_candidates": len(self.truth), **metrics})
        return rows


def _sparseeval_fold(suite: Suite, task: str, anchors, split, fit_seed: int, epochs: int):
    """SparseEval-MLP on one benchmark of one fold: estimates for validation models and candidates."""
    memory, valid, query = split
    y = suite.responses(task)
    observed = np.zeros_like(y[query])
    observed[:, anchors] = y[query][:, anchors]
    torch.set_num_threads(1)
    fit = fit_predict(y[memory], y[valid], {"valid": y[valid], "query": observed},
                      suite.weights(task), len(anchors), fit_seed, fixed_anchors=anchors,
                      device="cpu", refinement_steps=0, num_epochs=epochs)
    return fit["predictions"]


def target_data(suite: Suite, target: str, anchors: dict, split, mlp: dict) -> TargetData:
    """Inputs of DiscoRD for one target benchmark of one fold."""
    memory, valid, query = split
    names = suite.names
    w = suite.weights(target)
    y = suite.responses(target)

    def g(task, rows):
        return anchor_mean(suite.responses(task)[rows], anchors[task], suite.weights(task),
                           horvitz_thompson=True)

    def answers(rows):
        return np.hstack([suite.responses(n)[np.ix_(rows, anchors[n])] for n in names])

    memory_profiles = np.column_stack([suite.responses(n)[memory] @ suite.weights(n) for n in names])
    others = [n for n in names if n != target]

    def features(rows):
        return np.column_stack([g(n, rows) for n in others])

    ridge_valid, ridge_query = ridge_transfer(features(memory), y[memory] @ w,
                                              features(valid), features(query))
    return TargetData(
        memory_scores=y[memory] @ w,
        memory_anchor_means=g(target, memory),
        valid_truth=y[valid] @ w,
        valid_anchor_means=g(target, valid),
        query_anchor_means=g(target, query),
        score_distances=tuple(score_distances(memory_profiles,
                                              np.column_stack([mlp[n][side] for n in names]))
                              for side in ("valid", "query")),
        response_distances=(response_distances(answers(memory), answers(valid)),
                            response_distances(answers(memory), answers(query))),
        estimates={"SparseEval-MLP": (mlp[target]["valid"], mlp[target]["query"]),
                   "Ridge transfer": (ridge_valid, ridge_query)},
    )


def run_cell(suite: Suite, seed: int, repeat: int, budget: int, *,
             epochs: int = DEFAULT_EPOCHS, predictors=PREDICTORS,
             rule: str = "smoothed", extras: bool = False) -> list[FoldedCellResult]:
    """One run: every benchmark of the suite as target, candidates pooled over the folds."""
    collected = {t: {"names": [], "truth": [], "fold": [], "pred": {}} for t in suite.names}
    for fold, split in enumerate(organisation_folds(suite.groups, seed)):
        memory, valid, query = split
        anchors = {n: uniform_anchors(suite.tasks[n].n_questions, budget,
                                      anchor_seed(COLLECTION, seed, repeat, fold))
                   for n in suite.names}
        mlp = {n: _sparseeval_fold(suite, n, anchors[n], split, seed + repeat + fold, epochs)
               for n in suite.names}
        for target in suite.names:
            data = target_data(suite, target, anchors, split, mlp)
            out = DiscoRD(predictors, rule, equal_if_unsolvable=True).fit_predict(data)
            box = collected[target]
            box["names"].append(suite.models[query])
            box["truth"].append(suite.responses(target)[query] @ suite.weights(target))
            box["fold"].append(np.full(len(query), fold))
            estimates = candidate_estimates(out, data, extras=extras, equal_if_unsolvable=True)
            for name, values in estimates.items():
                box["pred"].setdefault(name, []).append(values)
    results = []
    for target, box in collected.items():
        results.append(FoldedCellResult(
            COLLECTION, f"WILD A={budget}", target, budget, seed, repeat,
            candidates=np.concatenate(box["names"]), truth=np.concatenate(box["truth"]),
            predictions={k: np.concatenate(v) for k, v in box["pred"].items()},
            folds=np.concatenate(box["fold"])))
    return results
