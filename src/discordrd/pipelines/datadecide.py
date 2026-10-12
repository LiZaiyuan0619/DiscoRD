"""DataDecide: checkpoints of small pretraining runs on nine benchmarks.

Training runs are split by pretraining data mix into memory, validation set
and candidates (one fixed split).  The validation models and the candidates
are the final checkpoints of their runs.  The memory holds, for the runs of
the memory mixes, either the final checkpoint of each run, four checkpoints
sampled along each run, or every checkpoint; the paper averages over the
three.  Each benchmark is evaluated on its own, so Ridge transfer and both
distances use that benchmark alone.  Anchor questions are drawn uniformly at
random.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import ResponseMatrix
from ..discord import PREDICTORS
from ..protocol import anchor_seed, data_mix_split, uniform_anchors
from .common import DEFAULT_EPOCHS, CellResult, candidate_estimates, single_benchmark_discord

COLLECTION = "datadecide"
BENCHMARKS = ("arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
              "openbookqa", "piqa", "socialiqa", "winogrande")
MEMORIES = ("trajectory_final", "trajectory_sampled", "all_checkpoints")
BUDGETS = (20, 100)
SAMPLED_CHECKPOINTS = 4


def checkpoint_index(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add the run identity (data mix and model size) and a numeric step.

    ``metadata`` has one row per checkpoint, aligned with the rows of the
    score matrices, with columns model_data_mix, model_size and model_step.
    """
    index = metadata[["model_data_mix", "model_size", "model_step"]].copy().reset_index(drop=True)
    index["model_step_numeric"] = pd.to_numeric(index.model_step, errors="raise")
    index["trajectory_id"] = index.model_data_mix.astype(str) + "::" + index.model_size.astype(str)
    return index


def memory_variants(index: pd.DataFrame, sampled_k: int = SAMPLED_CHECKPOINTS) -> dict[str, list[int]]:
    """Rows of the three memories: final checkpoints, sampled checkpoints, all checkpoints."""
    final, sampled = [], []
    for trajectory, frame in index.groupby("trajectory_id", sort=True):
        frame = frame.sort_values("model_step_numeric", kind="stable")
        final.append(int(frame.index[-1]))
        if len(frame) < sampled_k:
            raise ValueError(f"run {trajectory} has fewer than {sampled_k} checkpoints")
        sampled.extend(frame.iloc[np.linspace(0, len(frame) - 1, sampled_k, dtype=int)].index.tolist())
    return {"trajectory_final": final, "trajectory_sampled": sampled,
            "all_checkpoints": index.index.tolist()}


def final_rows(index: pd.DataFrame, positions) -> np.ndarray:
    """The final checkpoint of every run among ``positions``."""
    frame = index.iloc[np.asarray(positions)].sort_values(["trajectory_id", "model_step_numeric"])
    return np.asarray([g.index[-1] for _, g in frame.groupby("trajectory_id", sort=False)], int)


def split(index: pd.DataFrame, memory: str):
    """(memory, validation, candidates) row indices for one memory variant."""
    parts = data_mix_split(index.model_data_mix.to_numpy())
    pool = set(parts["memory"].tolist())
    memory_rows = np.asarray([i for i in memory_variants(index)[memory] if i in pool], int)
    return memory_rows, final_rows(index, parts["validation"]), final_rows(index, parts["candidates"])


def run_cell(matrix: ResponseMatrix, index: pd.DataFrame, memory: str, seed: int, repeat: int,
             budget: int, *, epochs: int = DEFAULT_EPOCHS, predictors=PREDICTORS,
             rule: str = "smoothed", extras: bool = False) -> CellResult:
    memory_rows, valid_rows, candidate_rows = split(index, memory)
    y = matrix.responses
    anchors = uniform_anchors(y.shape[1], budget, anchor_seed(COLLECTION, seed, repeat))
    output, data = single_benchmark_discord(
        y[memory_rows], y[valid_rows], y[candidate_rows], anchors,
        fit_seed=seed + repeat, item_weights=matrix.item_weights, horvitz_thompson=True,
        sparseeval_options={"num_epochs": epochs, "refinement_steps": 0},
        predictors=predictors, rule=rule)
    return CellResult(COLLECTION, f"DataDecide A={budget}", f"{matrix.name}|{memory}", budget,
                      seed, repeat, candidates=matrix.model_names[candidate_rows],
                      truth=y[candidate_rows] @ matrix.item_weights,
                      predictions=candidate_estimates(output, data, extras=extras),
                      weights=output.weights, k=output.k)
