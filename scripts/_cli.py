"""Command-line helpers shared by the run scripts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discordrd.protocol import REPEATS, SEEDS  # noqa: E402


def parser(description: str, *, data: str, out: str, budgets) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / data, help="directory written by the script in data/")
    p.add_argument("--out", type=Path, default=ROOT / out, help="output directory")
    p.add_argument("--budgets", type=int, nargs="+", default=list(budgets), help="anchor questions per benchmark")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--repeats", type=int, nargs="+", default=list(REPEATS),
                   help="draws of anchor questions per seed (0-9 in the paper)")
    p.add_argument("--workers", type=int, default=1, help="parallel processes")
    p.add_argument("--extras", action="store_true",
                   help="also report the ablations and the corrections built up from no memory")
    p.add_argument("--baselines", action="store_true", help="also run the published estimators")
    p.add_argument("--save-predictions", action="store_true", help="write every estimate, not only the metrics")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML file whose keys set the defaults of these options (see configs/)")
    return p


def parse(p: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line; values in ``--config`` act as defaults, flags given explicitly win."""
    first, _ = p.parse_known_args()
    if first.config is not None:
        import yaml
        values = yaml.safe_load(first.config.read_text()) or {}
        known = {a.dest for a in p._actions}
        unknown = set(values) - known
        if unknown:
            raise SystemExit(f"unknown keys in {first.config}: {sorted(unknown)}")
        for key in ("data", "out"):
            if key in values:
                values[key] = ROOT / values[key]
        p.set_defaults(**values)
    return p.parse_args()


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import torch
    torch.set_num_threads(1)


def run_tasks(function, tasks, workers: int):
    """Map ``function`` over ``tasks`` with one BLAS thread per process; print progress."""
    started = time.time()
    results = []
    if workers <= 1:
        _init_worker()
        iterator = map(function, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
        iterator = pool.map(function, tasks, chunksize=1)
    try:
        for i, result in enumerate(iterator, 1):
            results.append(result)
            print(f"[{i}/{len(tasks)}] {time.time() - started:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return results


def add_baselines(result, valid_and_query: tuple[dict, dict]):
    """Attach the candidates' estimates of the published estimators to a cell result."""
    _, query = valid_and_query
    result.predictions.update(query)
    return result


def write_outputs(results, out: Path, *, save_predictions: bool, config: dict) -> pd.DataFrame:
    """Write ``metrics.csv`` (31 metrics per cell and method) and optionally the estimates."""
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([row for r in results for row in r.metric_rows()])
    metrics.to_csv(out / "metrics.csv", index=False)
    weights = pd.DataFrame([{**r.keys(), **{f"weight[{k}]": v for k, v in r.weights.items()},
                             **{f"k[{k}]": v for k, v in r.k.items()}} for r in results])
    weights.to_csv(out / "fusion_weights.csv", index=False)
    if save_predictions:
        pd.concat([r.prediction_frame() for r in results], ignore_index=True).to_csv(
            out / "predictions.csv.gz", index=False, compression="gzip")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print(f"wrote {len(metrics)} rows to {out / 'metrics.csv'}")
    return metrics
