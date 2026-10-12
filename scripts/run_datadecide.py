"""Run DiscoRD on DataDecide (nine benchmarks, three memories, A = 20 and 100).

Prepare the data first:

    python data/prepare_datadecide.py

then

    python scripts/run_datadecide.py --workers 16
    python scripts/summarize.py outputs/datadecide

The paper averages each setting over the nine benchmarks, the three memories
(final checkpoints, sampled checkpoints, all checkpoints) and the 30 runs.
"""
from __future__ import annotations

from functools import partial

import pandas as pd

import _cli
from discordrd.baselines import predict_baselines
from discordrd.data import load_matrix
from discordrd.pipelines import datadecide as P
from discordrd.protocol import anchor_seed, uniform_anchors


def one(task, *, data, extras, baselines, epochs):
    benchmark, memory, seed, repeat, budget = task
    matrix = load_matrix(data / f"{benchmark}.npz")
    index = P.checkpoint_index(pd.read_csv(data / "checkpoints.csv"))
    result = P.run_cell(matrix, index, memory, seed, repeat, budget, epochs=epochs, extras=extras)
    if baselines:
        memory_rows, valid_rows, candidate_rows = P.split(index, memory)
        y = matrix.responses
        anchors = uniform_anchors(y.shape[1], budget, anchor_seed(P.COLLECTION, seed, repeat))
        _cli.add_baselines(result, predict_baselines(
            y[memory_rows], y[valid_rows], y[candidate_rows], anchors, matrix.item_weights,
            seed=seed, form="own"))
    return result


def main() -> None:
    p = _cli.parser(__doc__, data="data/processed/datadecide", out="outputs/datadecide",
                    budgets=P.BUDGETS)
    p.add_argument("--benchmarks", nargs="+", default=list(P.BENCHMARKS))
    p.add_argument("--memories", nargs="+", default=list(P.MEMORIES), choices=P.MEMORIES)
    p.add_argument("--epochs", type=int, default=P.DEFAULT_EPOCHS,
                   help="maximum SparseEval-MLP epochs")
    args = _cli.parse(p)
    tasks = [(t, m, s, r, b) for b in args.budgets for t in args.benchmarks for m in args.memories
             for s in args.seeds for r in args.repeats]
    results = _cli.run_tasks(partial(one, data=args.data, extras=args.extras,
                                     baselines=args.baselines, epochs=args.epochs),
                             tasks, args.workers)
    _cli.write_outputs(results, args.out, save_predictions=args.save_predictions, config=vars(args))


if __name__ == "__main__":
    main()
