"""Run DiscoRD on AlpacaEval (A = 20 and 100, 3 seeds x 10 anchor draws).

Prepare the data first:

    python data/prepare_alpacaeval.py

then

    python scripts/run_alpacaeval.py --workers 8
    python scripts/summarize.py outputs/alpacaeval

Each cell fits SparseEval-MLP once on the memory (CPU).  ``--baselines``
adds the published estimators, with continuous scores read as soft labels.
"""
from __future__ import annotations

from functools import partial

import numpy as np

import _cli
from discordrd.baselines import predict_baselines
from discordrd.data import load_matrix
from discordrd.pipelines import alpacaeval as P
from discordrd.protocol import anchor_seed, random_split, uniform_anchors


def one(task, *, data, extras, baselines, epochs):
    seed, repeat, budget = task
    matrix = load_matrix(data / "alpacaeval.npz")
    options = None if epochs is None else {"num_epochs": epochs}
    result = P.run_cell(matrix, seed, repeat, budget, sparseeval_options=options, extras=extras)
    if baselines:
        y = matrix.responses
        memory, valid, candidates = random_split(len(y), seed)
        anchors = uniform_anchors(y.shape[1], budget, anchor_seed(P.COLLECTION, seed, repeat))
        _cli.add_baselines(result, predict_baselines(
            y[memory], y[valid], y[candidates], anchors, np.full(y.shape[1], 1 / y.shape[1]),
            seed=seed, form="own"))
    return result


def main() -> None:
    p = _cli.parser(__doc__, data="data/processed/alpacaeval", out="outputs/alpacaeval",
                    budgets=P.BUDGETS)
    p.add_argument("--epochs", type=int, default=None,
                   help="maximum SparseEval-MLP epochs (default 20000 as in the paper)")
    args = _cli.parse(p)
    tasks = [(s, r, b) for b in args.budgets for s in args.seeds for r in args.repeats]
    results = _cli.run_tasks(partial(one, data=args.data, extras=args.extras,
                                     baselines=args.baselines, epochs=args.epochs),
                             tasks, args.workers)
    _cli.write_outputs(results, args.out, save_predictions=args.save_predictions, config=vars(args))


if __name__ == "__main__":
    main()
