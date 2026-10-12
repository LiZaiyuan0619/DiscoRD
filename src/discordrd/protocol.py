"""Evaluation protocol: splits, anchor draws and the grid of runs (Appendix C).

Every setting has 30 runs: three random seeds, each with ten draws of anchor
questions.  The seeds also fix the split of models on the Open LLM Leaderboard,
WILD and AlpacaEval, while DataDecide keeps one split.  The validation set only
chooses the number of neighbours and the fusion weights; every metric is
computed on the candidates alone.
"""
from __future__ import annotations

import numpy as np

SEEDS = (42, 0, 1)
REPEATS = tuple(range(10))

# Fixed base seeds of the experiments, kept so that the splits and anchor
# draws of the paper can be regenerated exactly.
ALPACAEVAL_ANCHOR_BASE = 20260910
DATADECIDE_SPLIT_SEED = 20260911


def runs(seeds=SEEDS, repeats=REPEATS):
    """The (seed, repeat) pairs of one setting."""
    return [(seed, repeat) for seed in seeds for repeat in repeats]


def uniform_anchors(n_questions: int, budget: int, seed: int) -> np.ndarray:
    """``budget`` anchor questions drawn uniformly without replacement."""
    if not 0 < budget <= n_questions:
        raise ValueError(f"cannot draw {budget} anchors from {n_questions} questions")
    return np.random.default_rng(seed).permutation(n_questions)[:budget]


def anchor_seed(collection: str, seed: int, repeat: int, fold: int = 0) -> int:
    """Seed of the anchor draw of one run, as used in the experiments."""
    if collection == "alpacaeval":
        return ALPACAEVAL_ANCHOR_BASE + 1000 * seed + repeat
    if collection == "wild":
        return seed + 7919 * repeat + fold
    if collection == "datadecide":
        return seed + 7919 * repeat
    raise ValueError(f"anchors of {collection!r} are not drawn uniformly")


def ollb_fit_seed(split_seed: int, repeat: int) -> int:
    """Seed of the SparseEval-MLP fit (and its anchor selection) on the Open LLM Leaderboard."""
    return split_seed * 100 + repeat


def random_split(n_models: int, seed: int, *, memory_share: float = 0.6,
                 validation_end: float = 0.8):
    """Random split into memory (60%), validation set (20%) and candidates (20%).

    Used on AlpacaEval.  Returns three index arrays.
    """
    order = np.random.default_rng(seed).permutation(n_models)
    n, m = int(memory_share * n_models), int(validation_end * n_models)
    return order[:n], order[n:m], order[m:]


def grouped_kfold(groups, k: int, seed: int):
    """k folds that never split a group (an organisation or a data mix)."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if k < 2 or k > len(unique):
        raise ValueError(f"k={k} is invalid for {len(unique)} groups")
    rng = np.random.default_rng(seed)
    order = unique[rng.permutation(len(unique))]
    folds = np.array_split(order, k)
    result = []
    for held in folds:
        test = np.isin(groups, held)
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    if set(np.concatenate([q for _, q in result])) != set(range(len(groups))):
        raise AssertionError("grouped folds do not cover every model")
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("group leakage across folds")
    return result


def organisation_folds(groups, seed: int, outer_k: int | None = None):
    """Nested folds that hold out whole organisations (WILD).

    The outer folds hold out one organisation at a time and give the
    candidates.  Inside each outer fold, the first inner fold splits the other
    organisations into the memory and the validation set.

    Returns a list of (memory, validation, candidates) index arrays.
    """
    groups = np.asarray(groups).astype(str)
    k = outer_k or len(np.unique(groups))
    outer = grouped_kfold(groups, k, seed)
    result = []
    for train, query in outer:
        inner_groups = groups[train]
        inner = grouped_kfold(inner_groups, min(5, len(np.unique(inner_groups))), seed + 100003)
        pool_local, valid_local = inner[0][0], inner[0][1]
        result.append((train[pool_local], train[valid_local], query))
    return result


def data_mix_split(mix_ids, *, seed: int = DATADECIDE_SPLIT_SEED, fractions=(.6, .2, .2)) -> dict:
    """Split training runs by pretraining data mix into memory, validation set and candidates (DataDecide)."""
    mix = np.asarray(mix_ids).astype(str)
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must be three values summing to one")
    unique = np.unique(mix)
    order = unique[np.random.default_rng(seed).permutation(len(unique))]
    cuts = [round(fractions[0] * len(order)), round((fractions[0] + fractions[1]) * len(order))]
    labels = np.full(len(mix), "query", dtype=object)
    labels[np.isin(mix, order[:cuts[0]])] = "pool"
    labels[np.isin(mix, order[cuts[0]:cuts[1]])] = "valid"
    return {"memory": np.flatnonzero(labels == "pool"),
            "validation": np.flatnonzero(labels == "valid"),
            "candidates": np.flatnonzero(labels == "query")}
