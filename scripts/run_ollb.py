"""Run DiscoRD on the Open LLM Leaderboard (OLLB-150, OLLB-708, OLLB-temporal; A = 20 and 100).

Prepare the data first (see data/README.md), which gives one directory per
setting with six matrices and ``splits.csv``:

    python data/prepare_ollb.py manifest && python data/prepare_ollb.py download && python data/prepare_ollb.py build

then

    python scripts/run_ollb.py --setting ollb_150 --workers 16
    python scripts/summarize.py outputs/ollb_150

Stage 1 fits SparseEval-MLP per benchmark and run (SparseEval chooses the
anchor questions on the memory) and caches the result under
``<out>/sparseeval``; stage 2 builds and fuses the six predictors.  A GPU
speeds up stage 1 (``--device cuda``) but is not required.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import _cli
from discordrd.baselines import predict_baselines
from discordrd.data import load_collection
from discordrd.pipelines import ollb as P


def stage1(matrices, roles, bench, budget, seed, repeat, cache: Path, device: str) -> dict:
    path = cache / f"{bench}_A{budget}_seed{seed}_repeat{repeat}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return {"anchors": np.asarray(saved["anchors"], int),
                "estimates": {role: pd.Series(v) for role, v in saved["estimates"].items()}}
    fit = P.fit_sparseeval(matrices[bench], roles, budget, seed, repeat, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"anchors": fit["anchors"].tolist(),
                                "estimates": {r: s.to_dict() for r, s in fit["estimates"].items()}}))
    return fit


def one(task, *, data, setting, cache, device, extras, baselines):
    seed, repeat, budget = task
    matrices = load_collection(data)
    benches = [b for b in (P.V2_BENCHMARKS + P.V1_BENCHMARKS) if b in matrices]
    matrices = {b: matrices[b] for b in benches}
    roles = P.roles_for_seed(pd.read_csv(data / "splits.csv"), seed)
    fits = {b: stage1(matrices, roles, b, budget, seed, repeat, cache, device) for b in benches}
    results = P.run_cell(matrices, roles, fits, setting, budget, seed, repeat, extras=extras)
    if baselines:
        for result in results:
            m = matrices[result.benchmark]
            rows = {r: m.subset_models(roles[r]).responses for r in ("memory", "validation", "candidate")}
            _cli.add_baselines(result, predict_baselines(
                rows["memory"], rows["validation"], rows["candidate"], fits[result.benchmark]["anchors"],
                m.item_weights, seed=seed, form="shared", item_ids=m.item_ids,
                benchmark=result.benchmark))
    return results


def main() -> None:
    p = _cli.parser(__doc__, data="data/processed/ollb", out="outputs", budgets=P.BUDGETS)
    p.add_argument("--setting", default=None, help="ollb_150, ollb_708 or ollb_temporal")
    p.add_argument("--device", default="cpu", help="device of the SparseEval-MLP fits (cpu or cuda)")
    args = _cli.parse(p)
    if args.setting is None:
        p.error("--setting is required (or set it in the --config file)")
    data = args.data / args.setting
    out = args.out / args.setting
    tasks = [(s, r, b) for b in args.budgets for s in args.seeds for r in args.repeats]
    nested = _cli.run_tasks(partial(one, data=data, setting=args.setting, cache=out / "sparseeval",
                                    device=args.device, extras=args.extras, baselines=args.baselines),
                            tasks, args.workers)
    _cli.write_outputs([r for group in nested for r in group], out,
                       save_predictions=args.save_predictions, config=vars(args))


if __name__ == "__main__":
    main()
