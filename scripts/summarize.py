"""Average cell metrics into setting-level numbers and compare methods with DiscoRD.

    python scripts/summarize.py outputs/alpacaeval
    python scripts/summarize.py outputs/ollb_150 outputs/wild --metrics MAE Recall@10 --tests

Each metric is averaged over the cells of a setting (arithmetic mean over
benchmarks, runs and, on DataDecide, the three memories).  MAE, RMSE and
Recall/NDCG/Hit/MRR are shown in percent.  ``--tests`` adds, for every
method, the two-sided Wilcoxon signed-rank test against DiscoRD over the
cells of the setting, Holm-corrected within each metric and setting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _cli  # noqa: F401  (puts src/ on the path)
from discordrd.metrics import ALL_METRICS, HIGHER_IS_BETTER, MAIN_METRICS
from discordrd.stats import holm, paired_wilcoxon

PERCENT = {m for m in ALL_METRICS if not m.startswith("regret") and not m.startswith(("Spearman", "Kendall"))}
CELL = ["benchmark", "seed", "repeat"]


def setting_table(metrics: pd.DataFrame, names) -> pd.DataFrame:
    table = metrics.groupby(["setting", "method"])[list(names)].mean()
    for m in names:
        if m in PERCENT:
            table[m] = 100 * table[m]
    return table


def tests(metrics: pd.DataFrame, names, reference: str = "DiscoRD") -> pd.DataFrame:
    rows = []
    for setting, frame in metrics.groupby("setting"):
        for metric in names:
            wide = frame.pivot_table(index=CELL, columns="method", values=metric)
            if reference not in wide:
                continue
            raw = {}
            for method in wide.columns.drop(reference):
                pair = wide[[method, reference]].dropna()
                if len(pair):
                    raw[method] = paired_wilcoxon(pair[method], pair[reference])
            adjusted = holm({m: p for m, (_, p) in raw.items()})
            for method, (median, p) in raw.items():
                better = median > 0 if metric in HIGHER_IS_BETTER else median < 0
                rows.append({"setting": setting, "metric": metric, "method": method,
                             "median_difference": median, "p": p, "p_holm": adjusted[method],
                             "significantly_better_than_DiscoRD": bool(better and adjusted[method] < 0.05)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="output directories with metrics.csv")
    parser.add_argument("--metrics", nargs="+", default=list(MAIN_METRICS))
    parser.add_argument("--all", action="store_true", help="all 31 metrics")
    parser.add_argument("--tests", action="store_true", help="paired Wilcoxon tests against DiscoRD")
    parser.add_argument("--out", type=Path, default=None, help="write the tables here as CSV")
    args = parser.parse_args()
    names = list(ALL_METRICS) if args.all else args.metrics
    metrics = pd.concat([pd.read_csv(d / "metrics.csv") for d in args.runs], ignore_index=True)
    table = setting_table(metrics, names)
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        for setting, block in table.groupby(level=0):
            print(f"\n{setting}  ({metrics[metrics.setting == setting].groupby(CELL).ngroups} cells)")
            print(block.droplevel(0).round(2).to_string())
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "summary.csv")
    if args.tests:
        result = tests(metrics, names)
        print("\nMethods significantly better than DiscoRD (Holm-corrected p < 0.05):")
        hits = result[result.significantly_better_than_DiscoRD]
        print(hits.to_string(index=False) if len(hits) else "none")
        if args.out:
            result.to_csv(args.out / "tests.csv", index=False)


if __name__ == "__main__":
    main()
