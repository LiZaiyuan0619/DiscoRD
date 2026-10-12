"""Run DiscoRD on WILD (24 benchmarks, organisation folds, A = 10, 20 and 50).

Prepare the data first:

    python data/prepare_wild.py

then

    python scripts/run_wild.py --workers 16
    python scripts/summarize.py outputs/wild

One run fits SparseEval-MLP on every benchmark of every fold, then estimates
every benchmark with DiscoRD across the suite.
"""
from __future__ import annotations

from functools import partial

import numpy as np

import _cli
from discordrd.baselines import predict_baselines
from discordrd.data import load_collection
from discordrd.pipelines import wild as P
from discordrd.protocol import anchor_seed, organisation_folds, uniform_anchors


def baselines_for(suite, target, seed, repeat, budget):
    """Published estimators on one benchmark, fold by fold, in the order of the pooled candidates."""
    y, w = suite.responses(target), suite.weights(target)
    per_fold = []
    for fold, (memory, valid, query) in enumerate(organisation_folds(suite.groups, seed)):
        anchors = uniform_anchors(y.shape[1], budget, anchor_seed(P.COLLECTION, seed, repeat, fold))
        per_fold.append(predict_baselines(y[memory], y[valid], y[query], anchors, w,
                                          seed=seed + fold, form="own"))
    names = per_fold[0][1].keys()
    query = {n: np.concatenate([q[n] for _, q in per_fold]) for n in names}
    return {}, query


def one(task, *, data, extras, baselines, epochs):
    seed, repeat, budget = task
    suite = P.Suite(load_collection(data))
    results = P.run_cell(suite, seed, repeat, budget, epochs=epochs, extras=extras)
    if baselines:
        for result in results:
            _cli.add_baselines(result, baselines_for(suite, result.benchmark, seed, repeat, budget))
    return results


def main() -> None:
    p = _cli.parser(__doc__, data="data/processed/wild", out="outputs/wild", budgets=P.BUDGETS)
    p.add_argument("--epochs", type=int, default=P.DEFAULT_EPOCHS,
                   help="maximum SparseEval-MLP epochs")
    args = _cli.parse(p)
    tasks = [(s, r, b) for b in args.budgets for s in args.seeds for r in args.repeats]
    nested = _cli.run_tasks(partial(one, data=args.data, extras=args.extras,
                                    baselines=args.baselines, epochs=args.epochs),
                            tasks, args.workers)
    _cli.write_outputs([r for group in nested for r in group], args.out,
                       save_predictions=args.save_predictions, config=vars(args))


if __name__ == "__main__":
    main()
